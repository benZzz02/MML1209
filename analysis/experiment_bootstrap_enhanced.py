"""Experiments 1+2: Paired bootstrap — Ours vs MML-SurgAdapt + CVS per-criterion.

Usage:
    python analysis/experiment_bootstrap_enhanced.py --save analysis/output/bootstrap_enhanced.json
"""
import sys, json, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_custom_keys = ["--save"]
_custom_args = {}
_i = 1
while _i < len(sys.argv):
    if sys.argv[_i] in _custom_keys:
        _custom_args[sys.argv[_i].lstrip("-")] = sys.argv[_i + 1]
        sys.argv.pop(_i); sys.argv.pop(_i)
    else:
        _i += 1


OURS_DIR = "analysis/output/full_epoch3"
MML_DIR = "analysis/output/noboth"
B = 1000


def load():
    p_r = np.load(Path(OURS_DIR) / "probs.npy")
    p_b = np.load(Path(MML_DIR) / "probs.npy")
    lab = np.load(Path(OURS_DIR) / "y_dense.npy")
    vid = np.load(Path(OURS_DIR) / "video_ids.npy", allow_pickle=True)
    uv = np.unique(vid)
    v2i = {v: np.where(vid == v)[0] for v in uv}
    v2l = {v: len(idx) for v, idx in v2i.items()}
    print(f"{len(p_r)} samples, {len(uv)} videos")
    return p_r, p_b, lab, vid, uv, v2i, v2l


def full_metrics(p, lab, vid):
    """Return dict of all metrics (uses official CholecT50 AP components)."""
    return all_metrics(p, lab, vid)


def fast_metrics(p, l, v):
    """Fast version: skip CholecT50 components for bootstrap speed."""
    r = {}
    vid_arr = np.array(v)
    from sklearn.metrics import f1_score, average_precision_score
    
    mask = np.array(["cholec80" in x for x in vid_arr])
    if mask.any():
        yt = np.argmax(l[mask][:, :7], axis=1); yp = np.argmax(p[mask][:, :7], axis=1)
        r["cholec80_f1"] = f1_score(yt, yp, average="macro", labels=np.unique(yt)) * 100
    
    mask = np.array(["endoscapes" in x for x in vid_arr])
    if mask.any():
        r["endoscapes_map"] = round(float(np.mean(
            [average_precision_score(l[mask][:, 7+c], p[mask][:, 7+c]) * 100 for c in range(3)]
        )), 2)
        for c in range(3):
            r[f"cvs_c{c+1}"] = round(average_precision_score(l[mask][:, 7+c], p[mask][:, 7+c]) * 100, 2)
    return r


def bootstrap(p_r, p_b, lab, vid, uv, v2i, v2l, B=1000, seed=42):
    rng = np.random.RandomState(seed)
    rr = fast_metrics(p_r, lab, vid)
    bb = fast_metrics(p_b, lab, vid)
    print(f"Ours: {rr}")
    print(f"MML:  {bb}")
    
    keys = [k for k in rr if isinstance(rr[k], (int, float))]
    gains = {k: [] for k in keys}
    t0 = time.time()
    
    for b in range(B):
        sv = rng.choice(uv, len(uv), replace=True)
        idx = np.concatenate([v2i[v] for v in sv])
        vr = fast_metrics(p_r[idx], lab[idx], vid[idx])
        vb = fast_metrics(p_b[idx], lab[idx], vid[idx])
        for k in keys:
            gains[k].append(vr.get(k, 0) - vb.get(k, 0))
        if (b + 1) % 200 == 0:
            print(f"  {b+1}/{B} ({time.time()-t0:.0f}s)")
    
    results = {}
    for k in keys:
        g = np.array(gains[k])
        results[k] = {
            "ours": round(rr.get(k, 0), 2),
            "mml": round(bb.get(k, 0), 2),
            "gain_mean": round(float(np.mean(g)), 2),
            "gain_std": round(float(np.std(g)), 2),
            "ci_low": round(float(np.percentile(g, 2.5)), 2),
            "ci_high": round(float(np.percentile(g, 97.5)), 2),
            "pos_ratio": round(float((g > 0).mean()), 4),
        }
    return results


def print_tables(results):
    # Table 1: Overall
    print(f"\n{'='*70}")
    print("Table 1: Ours vs MML-SurgAdapt — Paired Bootstrap B=1000")
    print(f"{'='*70}")
    print(f"{'Metric':<25} {'Ours':>8} {'MML':>8} {'Gain μ':>8} {'95% CI':>16} {'P(gain>0)':>10}")
    print("-" * 75)
    for k in ["cholec80_f1", "endoscapes_map"]:
        if k in results:
            r = results[k]
            ci = f"[{r['ci_low']:.2f}, {r['ci_high']:.2f}]"
            print(f"{k:<25} {r['ours']:>8.2f} {r['mml']:>8.2f} {r['gain_mean']:>8.2f} {ci:>16} {r['pos_ratio']:>10.2%}")
    
    
    # Table 2: CVS per-criterion
    print(f"\n{'='*70}")
    print("Table 2: CVS Per-Criterion Bootstrap")
    print(f"{'='*70}")
    print(f"{'Criterion':<12} {'Ours':>8} {'MML':>8} {'Gain μ':>8} {'95% CI':>16} {'P(gain>0)':>10}")
    print("-" * 70)
    for k in ["cvs_c1", "cvs_c2", "cvs_c3", "endoscapes_map"]:
        if k in results:
            r = results[k]
            ci = f"[{r['ci_low']:.2f}, {r['ci_high']:.2f}]"
            print(f"{k:<12} {r['ours']:>8.2f} {r['mml']:>8.2f} {r['gain_mean']:>8.2f} {ci:>16} {r['pos_ratio']:>10.2%}")
    print("-" * 70)


def main():
    save_path = _custom_args.get("save", "analysis/output/bootstrap_enhanced.json")
    p_r, p_b, lab, vid, uv, v2i, v2l = load()
    results = bootstrap(p_r, p_b, lab, vid, uv, v2i, v2l, B)
    print_tables(results)
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {save_path}")


if __name__ == "__main__":
    main()
