"""CholecT50 bootstrap: Ours vs MML, B=100, official AP_ivt."""
import sys, os, json, time
import numpy as np
import ivtmetrics
from sklearn.metrics import average_precision_score

OURS = "analysis/output/full_epoch3"
MML = "analysis/output/noboth"
OUT = "analysis/output/bootstrap_enhanced.json"

def resolve_nan(cw):
    cw[cw == -0.0] = np.nan
    return cw

def ap_ivt(p, l, vids):
    mask = np.array(["cholect50" in v for v in vids])
    if not mask.any():
        return 0.0
    y = l[mask][:, 10:]
    s = p[mask][:, 10:]
    uv = np.unique(vids[mask])
    vals = []
    for v in uv:
        idx = np.where(vids[mask] == v)[0]
        ap = resolve_nan(average_precision_score(y[idx], s[idx], average=None) * 100)
        vals.append(np.nanmean(ap))
    return float(np.nanmean(vals))

p_r = np.load(f"{OURS}/probs.npy")
p_b = np.load(f"{MML}/probs.npy")
l = np.load(f"{OURS}/y_dense.npy")
v = np.load(f"{OURS}/video_ids.npy", allow_pickle=True)

mask = np.array(["cholect50" in v_ for v_ in v])
uv = np.unique(v[mask])
v2i = {u: np.where(v == u)[0] for u in uv}
print(f"CholecT50: {len(uv)} videos")

rr = ap_ivt(p_r, l, v)
bb = ap_ivt(p_b, l, v)
print(f"Full: Ours={rr:.2f}, MML={bb:.2f}")

B = 100
rng = np.random.RandomState(42)
gains = []
t0 = time.time()

for b in range(B):
    sv = rng.choice(uv, len(uv), replace=True)
    idx = np.concatenate([v2i[u] for u in sv])
    gains.append(ap_ivt(p_r[idx], l[idx], v[idx]) - ap_ivt(p_b[idx], l[idx], v[idx]))
    if (b + 1) % 25 == 0:
        print(f"  {b+1}/{B} ({time.time()-t0:.0f}s)")

g = np.array(gains)
print(f"AP_ivt: gain={np.mean(g):+.2f} CI=[{np.percentile(g,2.5):.2f},{np.percentile(g,97.5):.2f}] P={(g>0).mean()*100:.1f}%")

with open(OUT) as f:
    out = json.load(f)
out["cholect50_ap_ivt"] = {
    "ours": round(rr, 2), "mml": round(bb, 2),
    "gain_mean": round(float(np.mean(g)), 2),
    "gain_std": round(float(np.std(g)), 2),
    "ci_low": round(float(np.percentile(g, 2.5)), 2),
    "ci_high": round(float(np.percentile(g, 97.5)), 2),
    "pos_ratio": round(float((g > 0).mean()), 4),
}
with open(OUT, "w") as f:
    json.dump(out, f, indent=2)
print(f"Saved to {OUT}")
