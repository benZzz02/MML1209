"""Experiment 4: Extended prior variants — block-shuffled, CLIP-only, LLM-only, no prior.

Needs GPU to run model forward with custom priors. Should be run AFTER step1.

Usage:
    CUDA_VISIBLE_DEVICES=0 python analysis/experiment_prior_variants_extended.py
        --ckpt "checkpoints/.../epoch_3.ckpt"
        --output analysis/output/full_epoch3
        --external-matrix surgical_matrix_clip_boosted_gpt-5.2.npy
"""
import sys, os, json
from pathlib import Path

# Intercept custom args BEFORE any project import
_custom_keys = ["--ckpt", "--output", "--external-matrix", "--config", "-c"]
_custom_args = {}
_i = 1
while _i < len(sys.argv):
    if sys.argv[_i] in _custom_keys:
        key = sys.argv[_i].lstrip("-").replace("-", "_")
        val = sys.argv[_i + 1] if _i + 1 < len(sys.argv) else ""
        _custom_args[key] = val
        sys.argv.pop(_i); sys.argv.pop(_i) if val else None
    else:
        _i += 1

# Ensure -c and -t are in args for project import (test mode)
config_file = _custom_args.get("config", _custom_args.get("c", "configs/scpnet_hill_Ignore_sp.yaml"))
sys.argv.extend(["-c", config_file, "-t"])

import numpy as np
import torch
import torch.nn.functional as F

from omegaconf import OmegaConf
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# This triggers config.py which needs -c in sys.argv
from config import cfg as global_cfg
from dataloader import Cholec_Test
from model import load_clip_model, MMLSurgAdaptSCPNet, compensate_logits_by_pred_prob, get_topk_related_labels
from torchvision import transforms
from torchvision.transforms import InterpolationMode

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


# ── Prior matrix generators ──
def _block_indices(n):
    return [(0, 7), (7, 10), (10, n)]

def generate_block_shuffled(A_star):
    n = A_star.shape[0]
    blocks = _block_indices(n)
    shuffled = A_star.copy()
    diag = np.eye(n, dtype=bool)
    for r1s, r1e in blocks:
        for r2s, r2e in blocks:
            blk = shuffled[r1s:r1e, r2s:r2e].copy()
            flat = blk.ravel()
            np.random.shuffle(flat)
            shuffled[r1s:r1e, r2s:r2e] = flat.reshape(blk.shape)
    shuffled[diag] = A_star[diag]
    return shuffled

def generate_clip_only(A_clip_cos, sim_threshold=0.05, s_reweight=0.2, block_gamma=1.0):
    """Replicate _postprocess_cos without external fusion."""
    n = A_clip_cos.shape[0]
    pr, vr, ar = slice(0, 7), slice(7, 10), slice(10, n)
    ranges = [(0, 7), (7, 10), (10, n)]
    
    structure_mask = torch.ones_like(A_clip_cos, dtype=torch.bool)
    structure_mask[pr, pr] = False
    structure_mask[vr, vr] = False
    structure_mask[ar, ar] = False
    structure_mask[pr, vr] = False; structure_mask[vr, pr] = False
    structure_mask[pr, ar] = False; structure_mask[ar, pr] = False
    structure_mask[vr, ar] = False; structure_mask[ar, vr] = False
    structure_mask.fill_diagonal_(True)
    
    A_masked = A_clip_cos * structure_mask.float()
    A_norm = torch.zeros_like(A_masked)
    for ss, se in ranges:
        for ts, te in ranges:
            block = A_masked[ss:se, ts:te]
            if block.sum() == 0:
                continue
            block = torch.where(block > sim_threshold, block, torch.zeros_like(block))
            block_max = block.max(dim=1, keepdim=True)[0]
            r = block / (block_max + 1e-12)
            r = r.pow(block_gamma)
            A_norm[ss:se, ts:te] = r
    
    diag = torch.eye(n, dtype=torch.bool, device=A_clip_cos.device)
    A_norm_off = A_norm * (1.0 - s_reweight)
    A_norm_off[diag] = s_reweight
    return A_norm_off

def generate_llm_only(A_ext_raw, sim_threshold=0.05, s_reweight=0.2, block_gamma=1.0):
    """Like clip_only but using external matrix instead of CLIP cos."""
    A_ext = 0.5 * (A_ext_raw + A_ext_raw.T)
    A_ext = np.clip(A_ext, -1.0, 1.0)
    A_ext_t = torch.as_tensor(A_ext, dtype=torch.float32)
    return generate_clip_only(A_ext_t, sim_threshold, s_reweight, block_gamma)

def generate_no_prior(n_cls=110):
    """Identity matrix: no cross-class information."""
    return torch.eye(n_cls, dtype=torch.float32)


# ── Main ──
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = _custom_args.get("ckpt")
    output = _custom_args.get("output")
    ext_matrix = _custom_args.get("external_matrix", "")
    config = _custom_args.get("config", "configs/scpnet_hill_Ignore_sp.yaml")
    
    if not ckpt or not output:
        print("Usage: --ckpt PATH --output DIR [--external-matrix PATH] [--config PATH]")
        sys.exit(1)
    
    out_dir = Path(output)
    priors_dir = out_dir / "prior_variants"
    
    print(f"[PriorExt] Device: {device}, Checkpoint: {ckpt}")
    
    # Load model
    cfg = OmegaConf.load(config)
    OmegaConf.update(global_cfg, "num_classes", cfg.num_classes)
    OmegaConf.update(global_cfg, "image_size", cfg.image_size)
    for k, v in cfg.items():
        setattr(global_cfg, k, v)
    global_cfg.perform_init = False
    global_cfg.data = "cholec"
    global_cfg.use_lfile = False
    global_cfg.getitem = False
    global_cfg.partial = cfg.get("partial", False)
    global_cfg.val_sp = cfg.get("val_sp", True)
    
    clip_model, _ = load_clip_model()
    clip_model.float()
    
    with open("cholec/cholec_labels.txt") as f:
        classnames = f.read().split("\n")
    
    model = MMLSurgAdaptSCPNet(classnames, clip_model).to(device)
    model.eval()
    
    state = torch.load(ckpt, map_location=device)
    new_state = {}
    for k, v in state.items():
        k = k.replace("module.", "")
        if "child_prompt_learner" in k:
            k = k.replace("child_prompt_learner", "prompt_learner")
        new_state[k] = v
    model.load_state_dict(new_state, strict=False)
    print("[PriorExt] Model loaded")
    
    # Load external matrix if needed
    spp = model.spp
    A_clip_cos = spp.A_clip_cos.detach().cpu()
    A_ext_raw = None
    if ext_matrix and os.path.isfile(ext_matrix):
        A_ext_raw = np.load(ext_matrix, allow_pickle=False)
        print(f"[PriorExt] External matrix loaded: {ext_matrix}")
    else:
        # Check if already loaded in model
        if hasattr(spp, "A_ext_cos"):
            A_ext_raw = spp.A_ext_cos.detach().cpu().numpy()
    
    # Build dataloader
    imsize = getattr(global_cfg, "image_size", 224)
    transform = transforms.Compose([
        transforms.Resize((imsize, imsize), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                             (0.26862954, 0.26130258, 0.27577711)),
    ])
    dataset = Cholec_Test(transform=transform, class_num=global_cfg.num_classes)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=False
    )
    print(f"[PriorExt] Dataset: {len(dataset)} samples")
    
    use_comp = cfg.get("SCP_ENABLE_LOGIT_COMP", True)
    
    # Generate prior matrices
    sim_th = float(cfg.get("sim_threshold", 0.05))
    s_rw = float(cfg.get("reweight_p", 0.2))
    b_gamma = float(cfg.get("SCP_BLOCK_GAMMA", 1.0))
    
    n_cls = model.spp.A_clip_cos.shape[0]
    A_original = np.load(out_dir / "A_star.npy")
    
    prior_configs = {
        "block_shuffled": {
            "matrix": generate_block_shuffled(A_original),
        },
        "no_prior": {
            "matrix": generate_no_prior(n_cls).numpy(),
        },
    }
    
    # CLIP-only and LLM-only need careful generation
    A_clip_cos_np = A_clip_cos.numpy()
    clip_only_matrix = generate_clip_only(A_clip_cos, sim_th, s_rw, b_gamma).numpy()
    prior_configs["clip_only"] = {"matrix": clip_only_matrix}
    
    if A_ext_raw is not None:
        llm_only_matrix = generate_llm_only(
            torch.as_tensor(A_ext_raw, dtype=torch.float32) if isinstance(A_ext_raw, np.ndarray) else A_ext_raw,
            sim_th, s_rw, b_gamma
        ).numpy()
        prior_configs["llm_only"] = {"matrix": llm_only_matrix}
    
    for prior_name, pc in prior_configs.items():
        p_out = priors_dir / prior_name
        if p_out.exists() and (p_out / "probs.npy").exists():
            print(f"  {prior_name} already exists, skip.")
            continue
        
        print(f"  Prior: {prior_name}...")
        p_out.mkdir(parents=True, exist_ok=True)
        np.save(p_out / "A_star.npy", pc["matrix"])
        
        logits_list, probs_list = [], []
        n_batches = len(loader)
        
        for batch_idx, (images, labels, vids) in enumerate(loader):
            images = images.to(device)
            with torch.no_grad():
                child_prompts = model.prompt_learner()
                Z = model.text_encoder(child_prompts, model.tokenized_prompts)
                img_feat = model.encode_image(images)
                img_feat = F.normalize(img_feat, p=2, dim=-1)
                
                A_custom = torch.as_tensor(pc["matrix"], dtype=Z.dtype, device=device)
                row_sum = A_custom.sum(dim=1, keepdim=True)
                A_gcn = A_custom / (row_sum + 1e-12)
                text_refined = model.sam(Z, A_gcn)
                text_refined = F.normalize(text_refined, p=2, dim=-1)
                
                logits = 10.0 * img_feat @ text_refined.t()
                
                if use_comp:
                    pred_idx = logits.argmax(dim=-1)
                    topk_related = [
                        get_topk_related_labels(int(pred_idx[b].item()), logits[b], A_custom)
                        for b in range(pred_idx.shape[0])
                    ]
                    logits = compensate_logits_by_pred_prob(logits, pred_idx, topk_related)
                
                probs = torch.sigmoid(logits)
            
            logits_list.append(logits.detach().cpu())
            probs_list.append(probs.detach().cpu())
            if (batch_idx + 1) % 100 == 0 or batch_idx == n_batches - 1:
                print(f"    batch {batch_idx+1}/{n_batches}")
        
        logits_all = torch.cat(logits_list, dim=0).numpy().astype(np.float32)
        probs_all = torch.cat(probs_list, dim=0).numpy().astype(np.float32)
        np.save(p_out / "logits.npy", logits_all)
        np.save(p_out / "probs.npy", probs_all)
        print(f"    Saved {len(probs_all)} samples")
    
    print("[PriorExt] Done.")


if __name__ == "__main__":
    main()
