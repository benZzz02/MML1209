"""Step 4: Offline LC lambda sweep (Experiment 3).

Reads Step 1 output (from 裸模型 = no LC, no Ignore) and:
  1. Applies compensate_logits_by_pred_prob in numpy with sweep params
  2. Computes Cholec80 F1, Endoscapes mAP, CholecT50 AP for each (alpha, T)
  3. Produces comparison table: w/o LC vs LC(α=..., T=...)

Usage:
    python analysis/step4_lc_sweep.py analysis/output/noboth [--save results.json]
"""
import sys, os, json, argparse
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.metrics import cholec80_f1, endoscapes_map, cholect50_all_components


# ─────────────────────────────────────────────
# Numpy port of compensate_logits_by_pred_prob + helpers
# ─────────────────────────────────────────────
def softmax(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def get_topk_related_labels_numpy(pred_idx, logits_row, A_star, k=1, use_sigmoid=True):
    """Numpy version of model.get_topk_related_labels."""
    n_cls = A_star.shape[0]
    phase = (0, 7)
    view = (7, 10)
    action = (10, n_cls)
    
    def in_range(i, r):
        return r[0] <= i < r[1]
    
    if in_range(pred_idx, phase):
        targets = [view, action]
    elif in_range(pred_idx, view):
        targets = [phase, action]
    elif in_range(pred_idx, action):
        targets = [phase, view]
    else:
        raise ValueError(f"pred_idx {pred_idx} out of range")
    
    conf = 1.0 / (1.0 + np.exp(-logits_row)) if use_sigmoid else softmax(logits_row, axis=-1)
    prior = A_star[pred_idx]
    joint = conf * prior
    
    results = []
    for r_start, r_end in targets:
        scores = joint[r_start:r_end]
        kk = min(k, scores.size)
        rel_idx = np.argsort(-scores)[:kk]
        results.append((rel_idx + r_start).tolist())
    return results


def compensate_logits_numpy(logits, A_star, alpha=2.0, temp=1.0, k=1, use_sigmoid=True):
    """Numpy version of model.compensate_logits_by_pred_prob."""
    N, C = logits.shape
    logits_out = logits.copy()
    probs = softmax(logits_out / temp, axis=-1)
    
    for b in range(N):
        pred_idx = int(np.argmax(logits_out[b]))
        p = probs[b, pred_idx]
        delta = alpha * p
        related = get_topk_related_labels_numpy(pred_idx, logits_out[b], A_star, k, use_sigmoid)
        for group in related:
            for idx in group:
                logits_out[b, idx] += delta
    return logits_out


# ─────────────────────────────────────────────
# Metrics (official)
# ─────────────────────────────────────────────
def compute_all_metrics(probs, y_dense, video_ids):
    """Compute all task metrics (uses official metrics.py)."""
    vid_arr = np.array(video_ids)
    results = {}
    
    mask = np.array(["cholec80" in v for v in vid_arr])
    if mask.any():
        results["cholec80_f1"] = round(cholec80_f1(y_dense[mask][:, :7], probs[mask][:, :7]), 2)
    
    mask = np.array(["endoscapes" in v for v in vid_arr])
    if mask.any():
        results["endoscapes_map"] = round(endoscapes_map(y_dense[mask][:, 7:10], probs[mask][:, 7:10]), 2)
    
    mask = np.array(["cholect50" in v for v in vid_arr])
    if mask.any():
        c50 = cholect50_all_components(y_dense[mask][:, 10:], probs[mask][:, 10:], vid_arr[mask])
        results["cholect50_ap_ivt"] = c50["AP_ivt"]
        for k in ["AP_i", "AP_v", "AP_t", "AP_iv", "AP_it"]:
            results[f"cholect50_{k.lower()}"] = c50[k]
    
    return results


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="LC lambda sweep")
    parser.add_argument("exp_dir", help="Step 1 output directory (e.g., analysis/output/noboth)")
    parser.add_argument("--save", default=None, help="Save results JSON")
    parser.add_argument("--alpha-sweep", type=float, nargs="+", default=[0, 0.5, 1.0, 2.0, 4.0])
    parser.add_argument("--temp-sweep", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    args = parser.parse_args()
    
    exp_dir = Path(args.exp_dir)
    
    # Load data (use raw_logits, before LC)
    raw_logits = np.load(exp_dir / "raw_logits.npy")  # [N, 110]
    y_dense = np.load(exp_dir / "y_dense.npy")
    A_star = np.load(exp_dir / "A_star.npy")
    video_ids = np.load(exp_dir / "video_ids.npy", allow_pickle=True)
    
    with open(exp_dir / "meta.json") as f:
        meta = json.load(f)
    
    print(f"Loaded: {len(raw_logits)} samples, A_star shape {A_star.shape}")
    
    # Baseline: no LC (alpha=0)
    results_rows = []
    
    print(f"\n{'α (alpha)':>10} {'T (temp)':>10} {'Cholec80 F1':>12} {'Endoscapes mAP':>14} {'CholecT50 AP_ivt':>15}")
    print("-" * 63)
    
    for alpha in args.alpha_sweep:
        for temp in args.temp_sweep:
            if alpha == 0:
                logits_adj = raw_logits.copy()
            else:
                logits_adj = compensate_logits_numpy(raw_logits, A_star, alpha=alpha, temp=temp)
            
            probs = 1.0 / (1.0 + np.exp(-logits_adj))
            metrics = compute_all_metrics(probs, y_dense, video_ids)
            
            f1 = metrics.get("cholec80_f1", float("nan"))
            m_ap = metrics.get("endoscapes_map", float("nan"))
            ivt = metrics.get("cholect50_ap_ivt", float("nan"))
            
            print(f"{alpha:>10.1f} {temp:>10.1f} {f1:>12.2f} {m_ap:>14.2f} {ivt:>12.2f}")
            
            results_rows.append({
                "alpha": alpha,
                "temp": temp,
                "cholec80_f1": round(f1, 2),
                "endoscapes_map": round(m_ap, 2),
                "cholect50_ap_ivt": round(ivt, 2),
            })
    
    print("-" * 60)
    
    if args.save:
        out = {
            "source_exp": str(exp_dir),
            "meta": meta,
            "sweep_results": results_rows,
        }
        with open(args.save, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Saved to {args.save}")


if __name__ == "__main__":
    main()
