"""Step 6: Frequency-stratified analysis (Experiment 5).

Compares model performance across low/high frequency classes.
Uses training label distribution to stratify 110 classes into groups,
then compares Full vs 裸 model AP/F1 per group.

Usage:
    python analysis/step6_freq_analysis.py \
        --ref analysis/output/full --baseline analysis/output/noboth \
        --save results.json
"""
import sys, os, json, argparse
from pathlib import Path
from collections import defaultdict

# Intercept custom args before project import
_custom_keys_6 = ["--ref", "--baseline", "--save", "--label-ref", "--label-base"]
_custom_args_6 = {}
_i = 1
while _i < len(sys.argv):
    if sys.argv[_i] in _custom_keys_6:
        key = sys.argv[_i].lstrip("-").replace("-", "_")
        _custom_args_6[key] = sys.argv[_i + 1]
        sys.argv.pop(_i)
        sys.argv.pop(_i)
    else:
        _i += 1

import numpy as np
from sklearn.metrics import f1_score, average_precision_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ─────────────────────────────────────────────
# Frequency estimation
# ─────────────────────────────────────────────
def estimate_freq_from_traindata():
    """Load training data directly and compute per-class positive frequency."""
    import pickle, json, os
    pos_counts = np.zeros(110, dtype=np.int64)
    n_train = 0
    
    data_path = "/data/cholecdata"
    
    # Cholec80 phases
    path = os.path.join(data_path, "cholec80/labels/train/frame_phase_train.pkl")
    data = pickle.load(open(path, "rb"))
    for video in data.values():
        for frame in video:
            pos_counts[frame["Phase_gt"]] += 1
            n_train += 1
    
    # Endoscapes CVS
    path = os.path.join(data_path, "endoscapes/train/annotation_ds_coco.json")
    with open(path) as f:
        endo = json.load(f)
    for img in endo["images"]:
        for i, v in enumerate(img["ds"]):
            if round(v) == 1:
                pos_counts[7 + i] += 1
        n_train += 1
    
    # CholecT50 triplets
    train_split = [1,2,4,5,13,15,18,22,23,25,26,27,31,35,36,40,43,47,48,49,52,56,57,60,62,65,66,68,70,75,79,92,96,103,110]
    labels_dir = os.path.join(data_path, "cholect50/labels")
    for i in train_split:
        fname = f"VID{i:02d}.json" if i < 100 else f"VID{i:03d}.json"
        with open(os.path.join(labels_dir, fname)) as f:
            ann = json.load(f)
        for gts in ann["annotations"].values():
            for gt in gts:
                idx = gt[0]
                if idx != -1:
                    pos_counts[10 + idx] += 1
            n_train += 1
    
    return pos_counts, n_train


# ─────────────────────────────────────────────
# Group metrics
# ─────────────────────────────────────────────
def compute_ap_per_class(y_true, y_pred):
    """Per-class average precision. Returns [C] array."""
    C = y_true.shape[1]
    ap = np.zeros(C)
    for c in range(C):
        if y_true[:, c].sum() > 0:
            ap[c] = average_precision_score(y_true[:, c], y_pred[:, c]) * 100
    return ap


def group_metrics(probs_ref, probs_base, y_dense, video_ids, groups):
    """Compute AP/F1 per group for ref and base."""
    vid_arr = np.array(video_ids)
    
    def compute_for_slice(col_idx, name):
        ref = probs_ref[:, col_idx]
        base = probs_base[:, col_idx]
        labels = y_dense[:, col_idx]
        
        ap_ref = compute_ap_per_class(labels, ref)
        ap_base = compute_ap_per_class(labels, base)
        
        return {
            "name": name,
            "n_classes": len(col_idx),
            "ap_ref": [round(float(v), 2) for v in ap_ref],
            "ap_base": [round(float(v), 2) for v in ap_base],
            "ap_ref_mean": round(float(np.mean(ap_ref)), 2),
            "ap_base_mean": round(float(np.mean(ap_base)), 2),
            "ap_delta_mean": round(float(np.mean(ap_base - ap_ref)), 2),
        }
    
    results = []
    for col_idx, name in groups:
        results.append(compute_for_slice(col_idx, name))
    
    return results


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    ref = _custom_args_6.get("ref")
    baseline = _custom_args_6.get("baseline")
    save = _custom_args_6.get("save", None)
    label_ref = _custom_args_6.get("label_ref", "Full")
    label_base = _custom_args_6.get("label_base", "裸模型")
    
    if not ref or not baseline:
        print("Usage: python step6_freq_analysis.py --ref DIR --baseline DIR [--save PATH]")
        sys.exit(1)
    
    ref_dir = Path(ref)
    base_dir = Path(baseline)
    
    probs_ref = np.load(ref_dir / "probs.npy")
    probs_base = np.load(base_dir / "probs.npy")
    y_dense = np.load(ref_dir / "y_dense.npy")
    video_ids = np.load(ref_dir / "video_ids.npy", allow_pickle=True)
    
    print(f"Samples: {len(probs_ref)}")
    
    # Frequency
    try:
        pos_counts, n_train = estimate_freq_from_traindata()
        print(f"Training samples: {n_train}, pos_counts range [{pos_counts.min()}, {pos_counts.max()}]")
    except Exception as e:
        print(f"Could not load training data: {e}")
        print("Using uniform frequency fallback.")
        pos_counts = np.ones(110, dtype=np.int64)
    
    # Group classes by frequency
    phase_idx = np.arange(0, 7)
    cvs_idx = np.arange(7, 10)
    triplet_idx = np.arange(10, 110)
    
    # Split triplets into low/high by frequency
    triplet_counts = pos_counts[10:]
    median = np.median(triplet_counts)
    triplet_low = 10 + np.where(triplet_counts <= median)[0]
    triplet_high = 10 + np.where(triplet_counts > median)[0]
    
    groups = [
        (phase_idx, "Phase (all)"),
        (cvs_idx, "CVS (all)"),
        (triplet_low, "Triplet Low-freq"),
        (triplet_high, "Triplet High-freq"),
        (np.concatenate([phase_idx, cvs_idx, triplet_low]), "Low-freq overall"),
        (triplet_high, "High-freq overall"),
    ]
    
    results = group_metrics(probs_ref, probs_base, y_dense, video_ids, groups)
    
    # Print table
    print(f"\n{'=' * 80}")
    print(f"Frequency-stratified: {label_ref} vs {label_base}")
    print(f"{'=' * 80}")
    print(f"{'Group':<20} {'#cls':>5} {'AP_ref':>8} {'AP_base':>8} {'Δ':>8}")
    print("-" * 80)
    for r in results:
        print(f"{r['name']:<20} {r['n_classes']:>5d} {r['ap_ref_mean']:>8.2f} {r['ap_base_mean']:>8.2f} {r['ap_delta_mean']:>+8.2f}")
    print("-" * 80)
    
    if save:
        out = {
            "ref": label_ref,
            "baseline": label_base,
            "pos_counts": pos_counts.tolist(),
            "n_train": int(n_train) if 'n_train' in dir() else None,
            "groups": results,
        }
        with open(save, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Saved to {save}")


if __name__ == "__main__":
    main()
