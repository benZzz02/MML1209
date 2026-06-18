"""Experiment 3: Full vs w/o External precision / over-compensation diagnostic.

Compares candidate precision and false-positive rate between two models.

Usage:
    python analysis/experiment_precision_diag.py --save analysis/output/exp3_precision.json
"""
import sys, json
from pathlib import Path

_custom_keys = ["--save"]
_custom_args = {}
_i = 1
while _i < len(sys.argv):
    if sys.argv[_i] in _custom_keys:
        _custom_args[sys.argv[_i].lstrip("-")] = sys.argv[_i + 1]
        sys.argv.pop(_i); sys.argv.pop(_i)
    else:
        _i += 1

import numpy as np


EXPS = [
    ("Full",    "analysis/output/full_epoch3"),
    ("w/o Ext", "analysis/output/noext"),
]


def candidate_analysis(y_sp, y_dense, probs, top_k=1):
    """
    Hidden positives exist only WITHIN the same group as SP
    (because Cholec80/Endoscapes/CholecT50 are separate datasets).
    
    This diagnostic checks: when the model's within-group top-1 prediction
    is NOT the SP class, is it a true hidden positive?
    
    This reveals whether a model's alternative predictions within the same
    dataset are correct (precision) or incorrect (FP rate).
    """
    N, C = probs.shape
    sp_idx = np.argmax(y_sp, axis=1)
    
    def group_of(idx):
        if idx < 7: return 0
        if idx < 10: return 1
        return 2
    
    group_ranges = [(0, 7), (7, 10), (10, C)]
    
    has_hidden = (y_dense.sum(axis=1) - y_sp.sum(axis=1)) > 0
    
    n_alt_pred = 0
    n_alt_correct = 0
    n_with_alt = 0
    
    for i in range(N):
        if not has_hidden[i]:
            continue
        
        gs, ge = group_ranges[group_of(sp_idx[i])]
        sp_local = sp_idx[i] - gs
        
        # Within-group predictions sorted by confidence
        group_probs = probs[i, gs:ge]
        local_ranks = np.argsort(-group_probs)
        
        # Top-1 within-group prediction that is NOT the SP class
        alt_preds = [r for r in local_ranks if r != sp_local]
        if not alt_preds:
            continue
        
        best_alt_local = alt_preds[0]
        best_alt_global = gs + best_alt_local
        
        n_with_alt += 1
        
        if y_dense[i, best_alt_global] == 1:
            n_alt_correct += 1
            n_alt_pred += 1
        else:
            n_alt_pred += 1
        
        # Also check if ANY non-SP prediction in the group is correct
        # (more lenient: "does the model know about the hidden positive?")
    
    total_hidden = has_hidden.sum()
    precision = n_alt_correct / max(n_alt_pred, 1) * 100
    fp_rate = (n_alt_pred - n_alt_correct) / max(n_alt_pred, 1) * 100
    
    return {
        "within_group_top1_precision": round(precision, 2),
        "within_group_fp_rate": round(fp_rate, 2),
        "samples_with_hidden": int(total_hidden),
        "samples_with_alt_pred": int(n_with_alt),
        "alt_pred_ratio": round(n_with_alt / max(total_hidden, 1) * 100, 2),
    }


def main():
    save_path = _custom_args.get("save", "analysis/output/exp3_precision.json")
    results = {}
    
    for label, exp_dir in EXPS:
        exp_dir = Path(exp_dir)
        y_dense = np.load(exp_dir / "y_dense.npy")
        y_sp = np.load(exp_dir / "y_sp.npy")
        probs = np.load(exp_dir / "probs.npy")
        
        r = candidate_analysis(y_sp, y_dense, probs, top_k=1)
        results[label] = r
        print(f"\n  {label}:")
        for k, v in r.items():
            print(f"    {k}: {v}")
    
    print(f"\n{'=' * 60}")
    print("Within-Group Precision Diagnostic: Full vs w/o External")
    print(f"{'=' * 60}")
    print(f"{'Metric':<35} {'Full':>10} {'w/o Ext':>10}")
    print("-" * 60)
    for metric in ["within_group_top1_precision", "within_group_fp_rate", "alt_pred_ratio"]:
        v1 = results.get("Full", {}).get(metric, "N/A")
        v2 = results.get("w/o Ext", {}).get(metric, "N/A")
        if isinstance(v1, float):
            print(f"{metric:<35} {v1:>10.2f} {v2:>10.2f}")
        else:
            print(f"{metric:<35} {str(v1):>10} {str(v2):>10}")
    
    if save_path:
        with open(save_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to {save_path}")


if __name__ == "__main__":
    main()
