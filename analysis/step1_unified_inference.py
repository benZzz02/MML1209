"""Step 1: Unified inference — save diagnostic data for all downstream analyses.

For each checkpoint, runs inference on val set and saves:
  - raw_logits (before LC), logits_lc, probs, y_dense, y_sp, video_ids, A_star
  - Prior variants: CLIP raw cos, random, shuffled → each with logits + probs

Usage:
    python analysis/step1_unified_inference.py -c configs/scpnet_hill_Ignore_sp.yaml [override flags...]

Override flags (must come AFTER -c):
    --ckpt PATH            Checkpoint path (overrides config's ckpt)
    --output-dir PATH      Output directory for diagnostic data
    --external-matrix PATH External matrix .npy path (overrides config)
    --disable-lc           Disable logit compensation
    --disable-ignore       Disable ignore mask
    --no-external          Disable external matrix fusion
    --lc-alpha VAL         Override SCP_COMP_ALPHA
    --lc-temp VAL          Override SCP_COMP_TEMP
    --batch-size N         Inference batch size
    --skip-priors          Skip prior variant inference
"""
import os, sys, json, warnings, random, copy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# Must import project config BEFORE anything else
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
warnings.filterwarnings("ignore")

# Parse --ckpt / --output-dir / custom flags BEFORE project args.py kicks in
_extra_args = {}
_extra_keys = ["--ckpt", "--output-dir", "--external-matrix", "--disable-lc",
               "--disable-ignore", "--no-external", "--lc-alpha", "--lc-temp",
               "--batch-size", "--skip-priors"]
_i = 1
while _i < len(sys.argv):
    if sys.argv[_i] in _extra_keys:
        key = sys.argv[_i].lstrip("-").replace("-", "_")
        if sys.argv[_i] in ("--disable-lc", "--disable-ignore", "--no-external", "--skip-priors"):
            _extra_args[key] = True
            sys.argv.pop(_i)
        else:
            _extra_args[key] = sys.argv[_i + 1]
            sys.argv.pop(_i)
            sys.argv.pop(_i)
    else:
        _i += 1

# Now import project modules (triggers project argparse)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from omegaconf import OmegaConf
from config import cfg as global_cfg
from config import base_cfg
from dataset import build_dataset
from dataloader import Cholec_Test
from model import (
    load_clip_model, MMLSurgAdaptSCPNet, TextEncoder,
    compensate_logits_by_pred_prob, get_topk_related_labels
)
from torchvision import transforms
from torchvision.transforms import InterpolationMode

# ─────────────────────────────────────────────
# Seed helpers
# ─────────────────────────────────────────────
SEED = 42

def set_seed(s):
    torch.manual_seed(s)
    torch.cuda.manual_seed(s)
    random.seed(s)
    np.random.seed(s)

# ─────────────────────────────────────────────
# SP label generation
# ─────────────────────────────────────────────
def random_pick_one(label_2d):
    out = torch.zeros_like(label_2d)
    N = label_2d.shape[0]
    for i in range(N):
        ones = torch.nonzero(label_2d[i], as_tuple=False)
        if ones.shape[0] > 0:
            r = torch.randint(ones.shape[0], (1,)).item()
            out[i, ones[r]] = 1.0
    return out

# ─────────────────────────────────────────────
# Prior variant generators
# ─────────────────────────────────────────────
def _build_structure_mask(n_cls, enable_intra_mutex=True, enable_inter_cooccur=True):
    pr, vr, ar = slice(0, 7), slice(7, 10), slice(10, n_cls)
    mask = torch.ones((n_cls, n_cls), dtype=torch.bool)
    if enable_intra_mutex:
        mask[pr, pr] = False
        mask[vr, vr] = False
        mask[ar, ar] = False
    if not enable_inter_cooccur:
        mask[pr, vr] = False; mask[vr, pr] = False
        mask[pr, ar] = False; mask[ar, pr] = False
        mask[vr, ar] = False; mask[ar, vr] = False
    mask.fill_diagonal_(True)
    return mask

def generate_random_prior(A_star, enable_intra_mutex=True, enable_inter_cooccur=True):
    n = A_star.shape[0]
    mask = _build_structure_mask(n, enable_intra_mutex, enable_inter_cooccur).numpy()
    rand = np.random.rand(n, n).astype(np.float32) * 0.5
    rand = rand * mask.astype(np.float32)
    diag = np.eye(n, dtype=bool)
    rand[diag] = 0.0
    orig_off_diag = A_star[~diag]
    density = (orig_off_diag > 0).mean()
    if density < 1:
        threshold = np.percentile(rand[rand > 0], (1 - density) * 100)
        rand[rand < threshold] = 0.0
    row_sum = rand.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1e-12
    rand = rand / row_sum
    rand[diag] = A_star[diag]
    return rand

def generate_shuffled_prior(A_star):
    n = A_star.shape[0]
    ranges = [(0, 7), (7, 10), (10, n)]
    shuffled = A_star.copy()
    diag = np.eye(n, dtype=bool)
    for r1_start, r1_end in ranges:
        for r2_start, r2_end in ranges:
            blk = shuffled[r1_start:r1_end, r2_start:r2_end].copy()
            flat = blk.ravel()
            np.random.shuffle(flat)
            shuffled[r1_start:r1_end, r2_start:r2_end] = flat.reshape(blk.shape)
    shuffled[diag] = A_star[diag]
    return shuffled

# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    set_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Step1] Device: {device}")

    # ── Apply extra args to global_cfg ──
    ckpt_path = _extra_args.get("ckpt")
    output_dir = _extra_args.get("output_dir")
    ext_matrix = _extra_args.get("external_matrix", None)
    disable_lc = _extra_args.get("disable_lc", False)
    disable_ignore = _extra_args.get("disable_ignore", False)
    no_external = _extra_args.get("no_external", False)
    lc_alpha = _extra_args.get("lc_alpha", None)
    lc_temp = _extra_args.get("lc_temp", None)
    batch_size = int(_extra_args.get("batch_size", 32))
    skip_priors = _extra_args.get("skip_priors", False)

    if not ckpt_path:
        ckpt_path = getattr(global_cfg, "ckpt", None)
        if not ckpt_path or not os.path.exists(ckpt_path):
            print("[ERROR] No --ckpt provided and no valid ckpt in config")
            sys.exit(1)
    if not output_dir:
        print("[ERROR] --output-dir is required")
        sys.exit(1)

    print(f"[Step1] Checkpoint: {ckpt_path}")
    print(f"[Step1] Output: {output_dir}")

    use_comp = not disable_lc
    use_ignore = not disable_ignore
    use_external = not no_external

    # ── Override config parameters ──
    global_cfg.SCP_ENABLE_LOGIT_COMP = use_comp
    global_cfg.SCP_ENABLE_IGNORE_MASK = use_ignore
    if lc_alpha is not None:
        global_cfg.SCP_COMP_ALPHA = float(lc_alpha)
    if lc_temp is not None:
        global_cfg.SCP_COMP_TEMP = float(lc_temp)
    if ext_matrix is not None:
        global_cfg.SCP_EXTERNAL_MATRIX_PATH = ext_matrix
    if no_external:
        global_cfg.SCP_EXTERNAL_MATRIX_PATH = ""
    global_cfg.perform_init = False

    # ── Build model ──
    print("[Step1] Loading CLIP model...")
    clip_model, _ = load_clip_model()
    clip_model.float()

    with open("cholec/cholec_labels.txt") as f:
        classnames = f.read().split("\n")

    print("[Step1] Building SCPNet...")
    model = MMLSurgAdaptSCPNet(classnames, clip_model)
    model = model.to(device)
    model.eval()

    state = torch.load(ckpt_path, map_location=device)
    new_state = {}
    for k, v in state.items():
        k = k.replace("module.", "")
        if "child_prompt_learner" in k:
            k = k.replace("child_prompt_learner", "prompt_learner")
        new_state[k] = v
    missing, unexp = model.load_state_dict(new_state, strict=False)
    if missing:
        print(f"[WARN] Missing keys: {[k for k in missing if 'prompt' not in k and 'token' not in k][:10]}")
    print(f"[Step1] Model loaded. Unexpected keys: {len(unexp)}")

    # ── Handle external matrix ──
    spp = model.spp
    if not use_external:
        print("[Step1] External matrix: DISABLED")
        spp.has_external = False
        spp.raw_cross_gate = None
    elif ext_matrix and os.path.isfile(ext_matrix):
        print(f"[Step1] External matrix: {ext_matrix}")
        ext_np = np.load(ext_matrix, allow_pickle=False)
        ext = torch.as_tensor(ext_np, dtype=spp.A_clip_cos.dtype, device=device)
        ext = 0.5 * (ext + ext.t())
        ext = ext.clamp(-1.0, 1.0)
        if hasattr(spp, "A_ext_cos"):
            spp.A_ext_cos = ext
        else:
            spp.register_buffer("A_ext_cos", ext)
        spp.has_external = True
    else:
        print("[Step1] External matrix: not provided, using config default")

    # ── Build dataloader ──
    imsize = getattr(global_cfg, "image_size", 224)
    val_preprocess = transforms.Compose([
        transforms.Resize((imsize, imsize), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                             (0.26862954, 0.26130258, 0.27577711)),
    ])

    dataset = Cholec_Test(transform=val_preprocess, class_num=global_cfg.num_classes)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=False
    )
    print(f"[Step1] Val samples: {len(dataset)}")

    # ── Run inference ──
    print("[Step1] Running inference...")

    raw_logits_list, logits_lc_list, probs_list = [], [], []
    y_dense_list, video_ids_list = [], []
    A_star_np = None
    A_clip_raw_np = None
    Z_all, img_feat_all = [], []

    for images, labels, vids in loader:
        images = images.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            child_prompts = model.prompt_learner()
            Z = model.text_encoder(child_prompts, model.tokenized_prompts)

            img_feat = model.encode_image(images)
            img_feat = F.normalize(img_feat, p=2, dim=-1)

            A_star = model.spp()

            if A_star_np is None:
                A_star_np = A_star.detach().cpu().numpy()
                if hasattr(spp, "A_clip_cos"):
                    A_clip_raw_np = spp.A_clip_cos.detach().cpu().numpy()

            row_sum = A_star.sum(dim=1, keepdim=True)
            A_gcn = A_star / (row_sum + 1e-12)
            text_refined = model.sam(Z, A_gcn)
            text_refined = F.normalize(text_refined, p=2, dim=-1)

            raw = 10.0 * img_feat @ text_refined.t()

            if use_comp or use_ignore:
                pred_idx = raw.argmax(dim=-1)
                topk_related = [
                    get_topk_related_labels(int(pred_idx[b].item()), raw[b], A_star)
                    for b in range(pred_idx.shape[0])
                ]
                if use_comp:
                    logits_lc = compensate_logits_by_pred_prob(raw, pred_idx, topk_related)
                else:
                    logits_lc = raw.clone()
            else:
                logits_lc = raw.clone()

            probs = torch.sigmoid(logits_lc)

        raw_logits_list.append(raw.detach().cpu())
        logits_lc_list.append(logits_lc.detach().cpu())
        probs_list.append(probs.detach().cpu())
        y_dense_list.append(labels.detach().cpu())
        video_ids_list.extend(vids)
        Z_all.append(Z.detach().cpu())
        img_feat_all.append(img_feat.detach().cpu())

    # ── Concatenate and save main ──
    raw_logits = torch.cat(raw_logits_list, dim=0).numpy().astype(np.float32)
    logits_lc = torch.cat(logits_lc_list, dim=0).numpy().astype(np.float32)
    probs = torch.cat(probs_list, dim=0).numpy().astype(np.float32)
    y_dense = torch.cat(y_dense_list, dim=0).numpy().astype(np.float32)
    video_ids = np.array(video_ids_list)

    set_seed(SEED)
    y_sp = random_pick_one(torch.from_numpy(y_dense)).numpy().astype(np.float32)
    pred_idx = np.argmax(probs, axis=1).astype(np.int32)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "raw_logits.npy", raw_logits)
    np.save(out_dir / "logits_lc.npy", logits_lc)
    np.save(out_dir / "probs.npy", probs)
    np.save(out_dir / "y_dense.npy", y_dense)
    np.save(out_dir / "y_sp.npy", y_sp)
    np.save(out_dir / "video_ids.npy", video_ids)
    np.save(out_dir / "pred_idx.npy", pred_idx)
    np.save(out_dir / "A_star.npy", A_star_np)
    if A_clip_raw_np is not None:
        np.save(out_dir / "A_clip_raw.npy", A_clip_raw_np)

    meta = {
        "ckpt": str(ckpt_path),
        "config": sys.argv[sys.argv.index("-c") + 1] if "-c" in sys.argv else "",
        "use_comp": use_comp,
        "use_ignore": use_ignore,
        "use_external": use_external,
        "external_matrix": ext_matrix if use_external else None,
        "comp_alpha": float(getattr(global_cfg, "SCP_COMP_ALPHA", 2.0)),
        "comp_temp": float(getattr(global_cfg, "SCP_COMP_TEMP", 1.0)),
        "num_samples": len(probs),
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[Step1] Saved main output to {out_dir}")

    # ────────────────────────────────────────────
    # Prior variants
    # ────────────────────────────────────────────
    if not skip_priors and A_clip_raw_np is not None:
        print("[Step1] Running prior variants...")
        priors_dir = out_dir / "prior_variants"

        prior_configs = {
            "clip_raw": {
                "matrix": A_clip_raw_np,
            },
            "random": {
                "matrix": generate_random_prior(A_star_np),
            },
            "shuffled": {
                "matrix": generate_shuffled_prior(A_star_np),
            },
        }

        for prior_name, pc in prior_configs.items():
            print(f"  Prior: {prior_name}")
            p_out = priors_dir / prior_name
            p_out.mkdir(parents=True, exist_ok=True)
            np.save(p_out / "A_star.npy", pc["matrix"])

            logits_prior_list, probs_prior_list = [], []

            for images, labels, vids in loader:
                images = images.to(device)
                with torch.no_grad():
                    child_prompts = model.prompt_learner()
                    Z = model.text_encoder(child_prompts, model.tokenized_prompts)
                    img_feat = model.encode_image(images)
                    img_feat = F.normalize(img_feat, p=2, dim=-1)

                    A_custom_t = torch.as_tensor(pc["matrix"], dtype=Z.dtype, device=device)
                    row_sum = A_custom_t.sum(dim=1, keepdim=True)
                    A_gcn = A_custom_t / (row_sum + 1e-12)
                    text_refined = model.sam(Z, A_gcn)
                    text_refined = F.normalize(text_refined, p=2, dim=-1)

                    logits = 10.0 * img_feat @ text_refined.t()

                    if use_comp or use_ignore:
                        pred_idx = logits.argmax(dim=-1)
                        topk_related = [
                            get_topk_related_labels(int(pred_idx[b].item()), logits[b], A_custom_t)
                            for b in range(pred_idx.shape[0])
                        ]
                        if use_comp:
                            logits = compensate_logits_by_pred_prob(logits, pred_idx, topk_related)

                    probs = torch.sigmoid(logits)

                logits_prior_list.append(logits.detach().cpu())
                probs_prior_list.append(probs.detach().cpu())

            logits_prior = torch.cat(logits_prior_list, dim=0).numpy().astype(np.float32)
            probs_prior = torch.cat(probs_prior_list, dim=0).numpy().astype(np.float32)
            np.save(p_out / "logits.npy", logits_prior)
            np.save(p_out / "probs.npy", probs_prior)

    print("[Step1] Done.")


if __name__ == "__main__":
    main()
