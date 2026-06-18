"""Test all multiseed checkpoints: run inference + print metrics for each epoch.

Usage:
    CUDA_VISIBLE_DEVICES=0 python analysis/test_multiseed.py
"""
import sys, os, json
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import f1_score, average_precision_score
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from omegaconf import OmegaConf
import ivtmetrics

# ── Find all checkpoint dirs ──
CKPT_BASE = Path("checkpoints/cholec_vals/multiseed")
OUT_BASE = Path("analysis/output/multiseed")

def find_ckpts():
    """Return list of (seed_name, epoch_label, ckpt_path) sorted by mAP."""
    entries = []
    for seed_dir in sorted(CKPT_BASE.iterdir()):
        if not seed_dir.is_dir():
            continue
        for round_dir in sorted(seed_dir.iterdir()):
            trial_dir = round_dir / seed_dir.name
            if not trial_dir.exists():
                continue
            for ckpt in sorted(trial_dir.glob("*.ckpt")):
                # Parse mAP from filename
                parts = ckpt.stem.split("_")
                try:
                    idx = parts.index("mAP")
                    map_val = float(parts[idx + 1])
                except (ValueError, IndexError):
                    map_val = 0
                entries.append((seed_dir.name, ckpt.stem, str(ckpt), map_val))
    # Group by seed, pick best
    best_per_seed = {}
    for name, epoch, path, map_val in entries:
        if name not in best_per_seed or map_val > best_per_seed[name][2]:
            best_per_seed[name] = (epoch, path, map_val)
    return best_per_seed


# ── Official metrics (lightweight, no analysis/output dependency) ──
def cholect50_components(y_true, y_pred, video_ids):
    unique_vids = np.unique(video_ids)
    ap_i, ap_v, ap_t, ap_iv, ap_it, ap_ivt = [], [], [], [], [], []
    for vid in unique_vids:
        idx = np.where(video_ids == vid)[0]
        labels, preds = y_true[idx], y_pred[idx]
        f = ivtmetrics.Disentangle()
        ap_ivt.append(average_precision_score(labels, preds, average=None).reshape(1, -1))
        ap_i.append(f.extract(inputs=labels, component="i"), f.extract(inputs=preds, component="i"))
        # ... need to do properly
    # Actually let me just do it the same way as test.py
    pass


def compute_metrics(probs, labels, video_ids):
    vid_arr = np.array(video_ids)
    results = {}
    
    # Cholec80 F1
    mask = np.array(["cholec80" in v for v in vid_arr])
    if mask.any():
        yt = np.argmax(labels[mask][:, :7], axis=1)
        yp = np.argmax(probs[mask][:, :7], axis=1)
        results["cholec80_f1"] = round(f1_score(yt, yp, average="macro", labels=np.unique(yt)) * 100, 2)
    
    # Endoscapes mAP + per-criterion
    mask = np.array(["endoscapes" in v for v in vid_arr])
    if mask.any():
        y, s = labels[mask][:, 7:10], probs[mask][:, 7:10]
        results["endoscapes_map"] = round(float(np.mean([average_precision_score(y[:,c], s[:,c])*100 for c in range(3)])), 2)
        for c in range(3):
            results[f"cvs_c{c+1}"] = round(average_precision_score(y[:,c], s[:,c])*100, 2)
    
    # CholecT50 components (official)
    mask = np.array(["cholect50" in v for v in vid_arr])
    if mask.any():
        unique_vids = np.unique(vid_arr[mask])
        ap_lists = {"i": [], "v": [], "t": [], "iv": [], "it": []}
        ap_ivt_list = []
        filter_obj = ivtmetrics.Disentangle()
        
        for vid in unique_vids:
            idx = np.where(vid_arr == vid)[0]
            y_sub, s_sub = labels[idx][:, 10:], probs[idx][:, 10:]
            ap_ivt_list.append(average_precision_score(y_sub, s_sub, average=None).reshape(1, -1))
            for comp in ["i", "v", "t", "iv", "it"]:
                lc = filter_obj.extract(inputs=y_sub, component=comp)
                pc = filter_obj.extract(inputs=s_sub, component=comp)
                ap_lists[comp].append(average_precision_score(lc, pc, average=None).reshape(1, -1))
        
        for comp in ["i", "v", "t", "iv", "it"]:
            if ap_lists[comp]:
                arr = np.concatenate(ap_lists[comp], axis=0)
                results[f"cholect50_ap_{comp}"] = round(float(np.nanmean(np.nanmean(arr, axis=0))), 2)
        
        if ap_ivt_list:
            arr = np.concatenate(ap_ivt_list, axis=0)
            results["cholect50_ap_ivt"] = round(float(np.nanmean(np.nanmean(arr, axis=0))), 2)
    
    return results


# ── Model + inference ──
from model import load_clip_model, MMLSurgAdaptSCPNet
from dataloader import Cholec_Test


def build_model(ckpt_path, config_path, device="cuda"):
    cfg = OmegaConf.load(config_path)
    OmegaConf.merge(OmegaConf.load("configs/base.yaml"), cfg)
    
    clip_model, _ = load_clip_model()
    clip_model.float()
    
    with open("cholec/cholec_labels.txt") as f:
        classnames = f.read().split("\n")
    
    model = MMLSurgAdaptSCPNet(classnames, clip_model).to(device)
    model.eval()
    
    state = torch.load(ckpt_path, map_location=device)
    new_state = {}
    for k, v in state.items():
        k = k.replace("module.", "").replace("child_prompt_learner", "prompt_learner")
        new_state[k] = v
    model.load_state_dict(new_state, strict=False)
    return model


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print()
    
    best_per_seed = find_ckpts()
    
    all_results = {}
    for seed_name, (epoch, ckpt_path, train_map) in sorted(best_per_seed.items()):
        print(f"{'='*60}")
        print(f"  {seed_name} — best: {epoch} (train mAP={train_map})")
        print(f"{'='*60}")
        
        # Determine config
        if "mml_base" in seed_name:
            config_path = f"configs/multiseed/{seed_name.replace('s0', 's0').replace('s1', 's1').replace('s2', 's2')}.yaml"
        elif "ours_full" in seed_name:
            config_path = f"configs/multiseed/{seed_name}.yaml"
        else:
            config_path = "configs/multiseed/ours_full_seed0.yaml"
        
        # Map actual filename
        config_map = {
            "ours_full_s1": "configs/multiseed/ours_full_seed1.yaml",
            "ours_full_s2": "configs/multiseed/ours_full_seed2.yaml",
        }
        config_path = config_map.get(seed_name, "configs/multiseed/ours_full_seed1.yaml")
        
        if not os.path.exists(ckpt_path):
            print(f"  Checkpoint not found: {ckpt_path}")
            continue
        
        try:
            model = build_model(ckpt_path, config_path, device)
        except Exception as e:
            print(f"  Model load failed: {e}")
            continue
        
        # Build dataloader
        imsize = 224
        transform = transforms.Compose([
            transforms.Resize((imsize, imsize), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                 (0.26862954, 0.26130258, 0.27577711)),
        ])
        dataset = Cholec_Test(transform=transform, class_num=110)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=False
        )
        
        # Inference
        probs_list, labels_list, vids_list = [], [], []
        with torch.no_grad():
            for images, labels, vids in loader:
                images = images.to(device)
                logits, _ = model(images)
                probs = torch.sigmoid(logits).detach().cpu()
                probs_list.append(probs)
                labels_list.append(labels)
                vids_list.extend(vids)
        
        probs = torch.cat(probs_list, dim=0).numpy().astype(np.float32)
        labels = torch.cat(labels_list, dim=0).numpy().astype(np.float32)
        video_ids = np.array(vids_list)
        
        metrics = compute_metrics(probs, labels, video_ids)
        all_results[seed_name] = {"epoch": epoch, "train_mAP": train_map, "test_metrics": metrics}
        
        print(f"  Cholec80 F1: {metrics.get('cholec80_f1', 'N/A')}")
        print(f"  Endoscapes mAP: {metrics.get('endoscapes_map', 'N/A')}")
        print(f"  CVS C1/C2/C3: {metrics.get('cvs_c1','N/A')}/{metrics.get('cvs_c2','N/A')}/{metrics.get('cvs_c3','N/A')}")
        print(f"  CholecT50 AP_ivt: {metrics.get('cholect50_ap_ivt', 'N/A')}")
        print()
    
    # Print comparison table
    print(f"\n{'='*70}")
    print("COMPARISON TABLE: All Seeded Runs")
    print(f"{'='*70}")
    header = f"{'Seed':<18} {'C80F1':>7} {'Endo':>7} {'C1':>7} {'C2':>7} {'C3':>7} {'AP_ivt':>8} {'train_mAP':>10}"
    print(header)
    print("-" * 70)
    
    # Also include seed0 from original
    print(f"{'ORIGINAL seed0':<18} {'75.53':>7} {'61.13':>7} {'65.45':>7} {'52.48':>7} {'65.47':>7} {'11.99':>8} {'6.03':>10}")
    
    for seed_name in sorted(all_results.keys()):
        r = all_results[seed_name]
        m = r["test_metrics"]
        print(f"{seed_name:<18} {m.get('cholec80_f1','N/A'):>7} {m.get('endoscapes_map','N/A'):>7} "
              f"{m.get('cvs_c1','N/A'):>7} {m.get('cvs_c2','N/A'):>7} {m.get('cvs_c3','N/A'):>7} "
              f"{m.get('cholect50_ap_ivt','N/A'):>8} {r['train_mAP']:>10.2f}")
    
    # Save
    out_path = OUT_BASE / "multiseed_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
