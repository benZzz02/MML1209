"""Step 5: Video-level paired bootstrap (Experiment 4).

Usage:
    python analysis/step5_bootstrap.py \
        --ref analysis/output/full_epoch3 --baseline analysis/output/noboth \
        --save analysis/output/step5_bootstrap.json
"""
import sys, os, json, time
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score, average_precision_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def micro_metrics(p, l, v):
    """Micro metrics for faster bootstrap."""
    r = {}
    for ds, sl in [('cholec80', slice(0, 7)), ('endoscapes', slice(7, 10)), ('cholect50', slice(10, 110))]:
        m = np.array([ds in x for x in v])
        if not m.any():
            continue
        y = l[m][:, sl]
        s = p[m][:, sl]
        if sl == slice(0, 7):
            r[ds + '_f1'] = f1_score(np.argmax(y, 1), np.argmax(s, 1),
                                       average='macro', labels=np.unique(np.argmax(y, 1))) * 100
        else:
            if y.sum() > 0:
                r[ds + '_map'] = float(np.nanmean(average_precision_score(y, s, average='micro') * 100))
            else:
                r[ds + '_map'] = 0.0
    return r


def main():
    ref = sys.argv[sys.argv.index("--ref") + 1] if "--ref" in sys.argv else "analysis/output/full_epoch3"
    base = sys.argv[sys.argv.index("--baseline") + 1] if "--baseline" in sys.argv else "analysis/output/noboth"
    save_path = sys.argv[sys.argv.index("--save") + 1] if "--save" in sys.argv else "analysis/output/step5_bootstrap.json"

    p_r = np.load(Path(ref) / "probs.npy")
    p_b = np.load(Path(base) / "probs.npy")
    lab = np.load(Path(ref) / "y_dense.npy")
    vid = np.load(Path(ref) / "video_ids.npy", allow_pickle=True)

    uv = np.unique(vid)
    nv = len(uv)
    v2i = {v: np.where(vid == v)[0] for v in uv}
    print(f"{len(p_r)} samples, {nv} videos")

    rr = micro_metrics(p_r, lab, vid)
    bb = micro_metrics(p_b, lab, vid)
    print(f"Ref: {rr}")
    print(f"Base: {bb}")

    B = 100
    rng = np.random.RandomState(42)
    g = {k: [] for k in rr}
    t0 = time.time()

    for b in range(B):
        sv = rng.choice(uv, nv, replace=True)
        idx = np.concatenate([v2i[v] for v in sv])
        r = micro_metrics(p_r[idx], lab[idx], vid[idx])
        bm = micro_metrics(p_b[idx], lab[idx], vid[idx])
        for k in rr:
            g[k].append(r[k] - bm[k])
        if (b + 1) % 25 == 0:
            print(f"  {b+1}/{B} ({time.time()-t0:.0f}s)")

    print(f"\n{'=' * 60}")
    print(f"Bootstrap: Full(epoch3) vs 裸模型(epoch3)  B={B}")
    print(f"{'=' * 60}")
    print(f"{'Dataset':<15} {'Ref':>8} {'Base':>8} {'Gain μ':>8} {'95% CI':>16} {'P(gain>0)':>10}")
    for k in rr:
        gk = np.array(g[k])
        ci = f"[{np.percentile(gk, 2.5):.2f}, {np.percentile(gk, 97.5):.2f}]"
        print(f"{k:<15} {rr[k]:>8.2f} {bb[k]:>8.2f} {np.mean(gk):>8.2f} {ci:>16} {(gk > 0).mean():>10.2%}")

    out = {"ref": "Full(epoch3)", "baseline": "裸模型(epoch3)", "n_bootstrap": B, "results": {}}
    for k in rr:
        gk = np.array(g[k])
        out["results"][k] = {
            "ref": round(rr[k], 2), "base": round(bb[k], 2),
            "gain_mean": round(float(np.mean(gk)), 2),
            "gain_std": round(float(np.std(gk)), 2),
            "ci_low": round(float(np.percentile(gk, 2.5)), 2),
            "ci_high": round(float(np.percentile(gk, 97.5)), 2),
            "pos_ratio": round(float((gk > 0).mean()), 4),
        }
    with open(save_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved to {save_path}")


if __name__ == "__main__":
    main()
