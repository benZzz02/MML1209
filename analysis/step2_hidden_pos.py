"""Step 2: Hidden-positive recovery + Prior diagnostic (Experiments 1 & 2).

Reads Step 1 output dir(s) and computes:
  1. Hit@k / Recall@k for hidden-positive recovery
  2. Compares main model vs prior variants (CLIP raw / random / shuffled)

Usage:
    python analysis/step2_hidden_pos.py analysis/output/exp_name ...
"""
import sys, os, json, argparse
from pathlib import Path

import numpy as np


# ─────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────
def hidden_pos_metrics(y_sp: np.ndarray, y_dense: np.ndarray, probs: np.ndarray, ks=(1, 2)):
    """Compute Hit@k and Recall@k for hidden positive recovery.
    
    y_sp:    [N, C] — Single-positive labels (only 1 positive per sample)
    y_dense: [N, C] — Full ground truth (>= 1 positive per sample)
    probs:   [N, C] — Sigmoid predictions
    
    Returns dict with Hit@k and Recall@k for each k.
    """
    N, C = y_sp.shape
    
    # Identify which samples have hidden positives
    sp_idx = np.argmax(y_sp, axis=1)  # [N] which class is the single positive
    
    hidden = y_dense.copy().astype(bool)
    hidden[np.arange(N), sp_idx] = False  # remove the SP-visible positive
    has_hidden = hidden.sum(axis=1) > 0
    num_hidden_per_sample = hidden.sum(axis=1)  # [N]
    
    if has_hidden.sum() == 0:
        return {"hit@1": float("nan"), "hit@2": float("nan"),
                "recall@1": float("nan"), "recall@2": float("nan"),
                "num_hidden_samples": 0}
    
    # Rank predictions (descending)
    ranks = np.argsort(-probs, axis=1)  # [N, C]
    
    results = {}
    for k in ks:
        topk = ranks[:, :k]  # [N, k]
        
        hit = np.zeros(N, dtype=bool)
        recall_sum = 0.0
        
        for i in range(N):
            if has_hidden[i]:
                hidden_cls = np.where(hidden[i])[0]
                # Hit: any hidden class in top-k?
                hit[i] = np.any(np.isin(topk[i], hidden_cls))
                # Recall: what fraction of hidden classes are in top-k?
                recall_sum += np.isin(topk[i], hidden_cls).sum() / num_hidden_per_sample[i]
        
        hit_rate = hit[has_hidden].mean() * 100
        recall = recall_sum / has_hidden.sum() * 100
        
        results[f"hit@{k}"] = float(hit_rate)
        results[f"recall@{k}"] = float(recall)
    
    results["num_hidden_samples"] = int(has_hidden.sum())
    return results


# ─────────────────────────────────────────────
# Load and run
# ─────────────────────────────────────────────
def analyze_exp(exp_dir: str, label: str):
    """Run hidden-positive analysis on one experiment output."""
    exp_dir = Path(exp_dir)
    
    y_dense = np.load(exp_dir / "y_dense.npy")
    y_sp = np.load(exp_dir / "y_sp.npy")
    probs = np.load(exp_dir / "probs.npy")
    
    with open(exp_dir / "meta.json") as f:
        meta = json.load(f)
    
    # Main model
    main_metrics = hidden_pos_metrics(y_sp, y_dense, probs)
    
    # Prior variants (if exist)
    prior_results = {}
    priors_dir = exp_dir / "prior_variants"
    if priors_dir.exists():
        for pname in sorted(os.listdir(str(priors_dir))):
            pdir = priors_dir / pname
            if not pdir.is_dir():
                continue
            p_probs = np.load(pdir / "probs.npy")
            pm = hidden_pos_metrics(y_sp, y_dense, p_probs)
            prior_results[pname] = pm
    
    return {
        "label": label,
        "meta": meta,
        "main": main_metrics,
        "prior_variants": prior_results,
    }


def print_table(results_list):
    """Pretty-print comparison table."""
    rows = []
    for r in results_list:
        label = r["label"]
        m = r["main"]
        rows.append({
            "exp": label,
            "type": "main",
            "hit@1": f"{m.get('hit@1', 'N/A'):.1f}" if not np.isnan(m.get('hit@1', np.nan)) else "N/A",
            "hit@2": f"{m.get('hit@2', 'N/A'):.1f}" if not np.isnan(m.get('hit@2', np.nan)) else "N/A",
            "recall@1": f"{m.get('recall@1', 'N/A'):.1f}" if not np.isnan(m.get('recall@1', np.nan)) else "N/A",
            "recall@2": f"{m.get('recall@2', 'N/A'):.1f}" if not np.isnan(m.get('recall@2', np.nan)) else "N/A",
            "n_hidden": m.get("num_hidden_samples", "?"),
        })
        
        for pname, pm in r.get("prior_variants", {}).items():
            rows.append({
                "exp": f"  ├─ prior={pname}",
                "type": "prior",
                "hit@1": f"{pm.get('hit@1', 'N/A'):.1f}" if not np.isnan(pm.get('hit@1', np.nan)) else "N/A",
                "hit@2": f"{pm.get('hit@2', 'N/A'):.1f}" if not np.isnan(pm.get('hit@2', np.nan)) else "N/A",
                "recall@1": f"{pm.get('recall@1', 'N/A'):.1f}" if not np.isnan(pm.get('recall@1', np.nan)) else "N/A",
                "recall@2": f"{pm.get('recall@2', 'N/A'):.1f}" if not np.isnan(pm.get('recall@2', np.nan)) else "N/A",
                "n_hidden": pm.get("num_hidden_samples", "?"),
            })
    
    # Print
    header = f"{'Experiment':<30} {'Hit@1':>8} {'Hit@2':>8} {'Recall@1':>10} {'Recall@2':>10} {'#hidden':>8}"
    sep = "-" * len(header)
    print(f"\n{'=' * len(header)}")
    print("Hidden-Positive Recovery Results")
    print(f"{'=' * len(header)}")
    print(header)
    print(sep)
    for row in rows:
        print(f"{row['exp']:<30} {row['hit@1']:>8} {row['hit@2']:>8} {row['recall@1']:>10} {row['recall@2']:>10} {row['n_hidden']:>8}")
    print(sep)
    print()


def save_results(results_list, out_path):
    """Save results as JSON."""
    with open(out_path, "w") as f:
        json.dump(results_list, f, indent=2)
    print(f"Saved results to {out_path}")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Hidden-positive recovery + prior diagnostic")
    parser.add_argument("exp_dirs", nargs="+", help="Step 1 output directories")
    parser.add_argument("--labels", nargs="+", default=None, help="Experiment labels (same order as exp_dirs)")
    parser.add_argument("--save", default=None, help="Save results JSON to path")
    args = parser.parse_args()
    
    labels = args.labels if args.labels else [f"exp{i}" for i in range(len(args.exp_dirs))]
    
    results = []
    for exp_dir, label in zip(args.exp_dirs, labels):
        print(f"Processing: {label} ({exp_dir})")
        r = analyze_exp(exp_dir, label)
        results.append(r)
    
    print_table(results)
    
    if args.save:
        save_results(results, args.save)


if __name__ == "__main__":
    main()
