"""Experiments 1+2+4: CVS per-criterion AP, task metrics, HP recovery by task group.

Usage:
    python analysis/experiment_task_metrics.py --save analysis/output/exp124_results.json
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis.metrics import all_metrics, cholec80_f1, endoscapes_map, endoscapes_per_criterion, cholect50_all_components


# ── Config: all experiment dirs ──
EXPS = [
    ("Full",     "analysis/output/full_epoch3"),
    ("裸模型",    "analysis/output/noboth"),
    ("w/o Ext",  "analysis/output/noext"),
    ("PP",       "analysis/output/pp"),
]


# ── Task metrics (using official metrics.py) ──
# cholec80_f1, endoscapes_map, endoscapes_per_criterion, cholect50_all_components
# are imported from analysis.metrics


# ─────────────────────────────────────────────
# HP recovery by task group
# ─────────────────────────────────────────────
def hidden_pos_by_group(y_sp, y_dense, probs, col_slice, ks=(1, 2)):
    N, _ = y_sp.shape
    y_sp_b = y_sp[:, col_slice].copy().astype(bool)
    y_gt_b = y_dense[:, col_slice].copy().astype(bool)
    
    # SP label within this group
    sp_onehot = y_sp_b.astype(int)
    sp_idx = np.argmax(sp_onehot, axis=1) if sp_onehot.sum() > 0 else np.zeros(N, dtype=int)
    
    hidden = y_gt_b.copy()
    hidden[np.arange(N), sp_idx] = False
    has_hidden = hidden.sum(axis=1) > 0
    
    if not has_hidden.any():
        return {}
    
    ranks = np.argsort(-probs[:, col_slice], axis=1)
    
    results = {}
    for k in ks:
        topk = ranks[:, :k]
        hit = np.zeros(N, dtype=bool)
        recall_sum = 0.0
        for i in range(N):
            if not has_hidden[i]:
                continue
            hc = np.where(hidden[i])[0]
            hit[i] = np.any(np.isin(topk[i], hc))
            recall_sum += np.isin(topk[i], hc).sum() / len(hc)
        
        nh = has_hidden.sum()
        results[f"hit@{k}"] = round(float(hit[has_hidden].mean() * 100), 2) if nh > 0 else 0
        results[f"recall@{k}"] = round(float(recall_sum / nh * 100), 2) if nh > 0 else 0
        results["n_hidden"] = int(nh)
    return results


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    save_path = _custom_args.get("save", "analysis/output/exp124_results.json")
    out = {}
    
    for label, exp_dir in EXPS:
        print(f"\n{'=' * 50}")
        print(f"  {label}")
        print(f"{'=' * 50}")
        exp_dir = Path(exp_dir)
        
        y_dense = np.load(exp_dir / "y_dense.npy")
        y_sp = np.load(exp_dir / "y_sp.npy")
        probs = np.load(exp_dir / "probs.npy")
        video_ids = np.load(exp_dir / "video_ids.npy", allow_pickle=True)
        
        with open(exp_dir / "meta.json") as f:
            meta = json.load(f)
        
        exp_out = {"meta": meta}
        
        # ── Task metrics (uses official metrics.py) ──
        task = {}
        mask_c80 = np.array(["cholec80" in v for v in video_ids])
        mask_endo = np.array(["endoscapes" in v for v in video_ids])
        mask_c50 = np.array(["cholect50" in v for v in video_ids])
        
        if mask_c80.any():
            task["cholec80_f1"] = round(cholec80_f1(y_dense[mask_c80][:, :7], probs[mask_c80][:, :7]), 2)
        if mask_endo.any():
            task["endoscapes_map"] = round(endoscapes_map(y_dense[mask_endo][:, 7:10], probs[mask_endo][:, 7:10]), 2)
            task["endoscapes_per_criterion"] = endoscapes_per_criterion(y_dense[mask_endo][:, 7:10], probs[mask_endo][:, 7:10])
        if mask_c50.any():
            c50 = cholect50_all_components(y_dense[mask_c50][:, 10:], probs[mask_c50][:, 10:], video_ids[mask_c50])
            for comp_key, val in c50.items():
                if not comp_key.endswith("_per_video"):
                    task[f"cholect50_{comp_key.lower()}"] = round(val, 2)
        
        exp_out["task_metrics"] = task
        print(f"  Task metrics: {task}")
        
        # ── HP recovery by task group ──
        groups = {
            "Phase": slice(0, 7),
            "CVS": slice(7, 10),
            "Triplet": slice(10, 110),
        }
        hp_groups = {}
        for gname, gslice in groups.items():
            hp = hidden_pos_by_group(y_sp, y_dense, probs, gslice)
            if hp:
                hp_groups[gname] = hp
                print(f"  HP recovery [{gname}]: Hit@1={hp.get('hit@1','N/A')} Hit@2={hp.get('hit@2','N/A')}")
        exp_out["hp_recovery_by_group"] = hp_groups
        
        out[label] = exp_out
    
    # ── Print tables ──
    print(f"\n\n{'=' * 70}")
    print("TABLE 1: CVS Per-Criterion AP")
    print(f"{'=' * 70}")
    print(f"{'Model':<12} {'C1':>8} {'C2':>8} {'C3':>8} {'mAP':>8}")
    for label, _ in EXPS:
        tc = out[label].get("task_metrics", {})
        pc = tc.get("endoscapes_per_criterion", {})
        mp = tc.get("endoscapes_map", 0)
        print(f"{label:<12} {pc.get('C1',0):>8.2f} {pc.get('C2',0):>8.2f} {pc.get('C3',0):>8.2f} {mp:>8.2f}")
    
    print(f"\n\n{'=' * 70}")
    print("TABLE 2: Full Task Metrics")
    print(f"{'=' * 70}")
    header = f"{'Model':<10} {'F1':>7} {'Endo':>7} {'AP_i':>7} {'AP_v':>7} {'AP_t':>7} {'AP_iv':>7} {'AP_it':>7} {'AP_ivt':>8}"
    print(header)
    print("-" * 70)
    for label, _ in EXPS:
        tc = out[label].get("task_metrics", {})
        print(f"{label:<10} {tc.get('cholec80_f1',0):>7.2f} {tc.get('endoscapes_map',0):>7.2f} "
              f"{tc.get('cholect50_ap_i',0):>7.2f} {tc.get('cholect50_ap_v',0):>7.2f} "
              f"{tc.get('cholect50_ap_t',0):>7.2f} {tc.get('cholect50_ap_iv',0):>7.2f} "
              f"{tc.get('cholect50_ap_it',0):>7.2f} {tc.get('cholect50_ap_ivt',0):>8.2f}")
    
    print(f"\n\n{'=' * 70}")
    print("TABLE 3: HP Recovery by Task Group")
    print(f"{'=' * 70}")
    header = f"{'Model':<12} {'Group':<10} {'Hit@1':>8} {'Hit@2':>8} {'Recall@1':>10} {'Recall@2':>10}"
    print(header)
    print("-" * 70)
    for label, _ in EXPS:
        hp = out[label].get("hp_recovery_by_group", {})
        for gname in ["Phase", "CVS", "Triplet"]:
            h = hp.get(gname, {})
            if h:
                print(f"{label:<12} {gname:<10} {h.get('hit@1',0):>8.2f} {h.get('hit@2',0):>8.2f} {h.get('recall@1',0):>10.2f} {h.get('recall@2',0):>10.2f}")
            else:
                print(f"{label:<12} {gname:<10} {'N/A':>8} {'N/A':>8} {'N/A':>10} {'N/A':>10}")
    
    if save_path:
        with open(save_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nSaved to {save_path}")


if __name__ == "__main__":
    main()
