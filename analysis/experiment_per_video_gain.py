"""Experiment 3: Per-video gain distribution — Ours vs MML-SurgAdapt.

Usage:
    python analysis/experiment_per_video_gain.py --save analysis/output/per_video_gain.json
"""
import sys, json
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis.metrics import cholect50_all_components
from sklearn.metrics import f1_score, average_precision_score

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


def per_video_f1(l, p):
    yt = np.argmax(l[:, :7], axis=1)
    yp = np.argmax(p[:, :7], axis=1)
    return f1_score(yt, yp, average="macro", labels=np.unique(yt)) * 100


def per_video_endoscapes(l, p):
    aps = [average_precision_score(l[:, c], p[:, c]) * 100 for c in range(3) if l[:, c].sum() > 0]
    return float(np.mean(aps)) if aps else 0


def main():
    save_path = _custom_args.get("save", "analysis/output/per_video_gain.json")
    
    p_r = np.load(Path(OURS_DIR) / "probs.npy")
    p_b = np.load(Path(MML_DIR) / "probs.npy")
    lab = np.load(Path(OURS_DIR) / "y_dense.npy")
    vid = np.load(Path(OURS_DIR) / "video_ids.npy", allow_pickle=True)
    
    vid_arr = np.array(vid)
    unique_vids = np.unique(vid_arr)
    
    # Compute per-video metrics for both models
    ours_pv = {}
    mml_pv = {}
    
    for vid_name in unique_vids:
        vname = str(vid_name)
        idx = np.where(vid_arr == vid_name)[0]
        p_r_v, p_b_v = p_r[idx], p_b[idx]
        l_v = lab[idx]
        
        ours_pv[vname] = {"video": vname, "n_frames": len(idx)}
        mml_pv[vname] = {"video": vname, "n_frames": len(idx)}
        
        # Cholec80 F1
        if "cholec80" in vid_name:
            ours_pv[vid_name]["cholec80_f1"] = per_video_f1(l_v, p_r_v)
            mml_pv[vid_name]["cholec80_f1"] = per_video_f1(l_v, p_b_v)
        
        # Endoscapes mAP
        if "endoscapes" in vid_name:
            ours_pv[vid_name]["endoscapes_map"] = per_video_endoscapes(l_v[:, 7:10], p_r_v[:, 7:10])
            mml_pv[vid_name]["endoscapes_map"] = per_video_endoscapes(l_v[:, 7:10], p_b_v[:, 7:10])
    
    # CholecT50: official per-video AP components
    mask_c50 = np.array(["cholect50" in v for v in vid_arr])
    if mask_c50.any():
        c50_r = cholect50_all_components(lab[mask_c50][:, 10:], p_r[mask_c50][:, 10:], vid_arr[mask_c50], return_per_video=True)
        c50_b = cholect50_all_components(lab[mask_c50][:, 10:], p_b[mask_c50][:, 10:], vid_arr[mask_c50], return_per_video=True)
        
        pv_r = c50_r.get("_per_video_list", {})
        pv_b = c50_b.get("_per_video_list", {})
        pv_vids = c50_r.get("_per_video_vids", [])
        
        for comp in ["AP_i", "AP_v", "AP_t", "AP_iv", "AP_it", "AP_ivt"]:
            if comp in pv_r and comp in pv_b:
                for i, vname in enumerate(pv_vids):
                    if i < len(pv_r[comp]) and i < len(pv_b[comp]):
                        ours_pv[vname][f"cholect50_{comp.lower()}"] = pv_r[comp][i]
                        mml_pv[vname][f"cholect50_{comp.lower()}"] = pv_b[comp][i]
    
    # Merge and compute gains
    summary = {"cholec80": [], "endoscapes": [], "cholect50_ap_ivt": []}
    per_video_out = {}
    
    for vid_name in ours_pv:
        o = ours_pv[vid_name]
        m = mml_pv.get(vid_name, {})
        entry = {"video": vid_name, "n_frames": o["n_frames"]}
        
        if "cholec80_f1" in o:
            delta = o["cholec80_f1"] - m.get("cholec80_f1", 0)
            entry["ours_f1"] = round(o["cholec80_f1"], 2)
            entry["mml_f1"] = round(m.get("cholec80_f1", 0), 2)
            entry["delta_f1"] = round(delta, 2)
            summary["cholec80"].append(delta)
        
        if "endoscapes_map" in o:
            delta = o["endoscapes_map"] - m.get("endoscapes_map", 0)
            entry["ours_map"] = round(o["endoscapes_map"], 2)
            entry["mml_map"] = round(m.get("endoscapes_map", 0), 2)
            entry["delta_map"] = round(delta, 2)
            summary["endoscapes"].append(delta)
        
        for ccomp in ["ap_i", "ap_v", "ap_t", "ap_iv", "ap_it", "ap_ivt"]:
            key = f"cholect50_{ccomp}"
            if key in o:
                delta = o[key] - m.get(key, 0)
                entry[f"ours_{key}"] = round(o[key], 2)
                entry[f"mml_{key}"] = round(m.get(key, 0), 2)
                entry[f"delta_{key}"] = round(delta, 2)
                if ccomp == "ap_ivt":
                    summary["cholect50_ap_ivt"].append(delta)
        
        per_video_out[vid_name] = entry
    
    # Print summary
    for ds_name, key in [("Cholec80 F1", "cholec80"), ("Endoscapes mAP", "endoscapes")]:
        gains = np.array(summary[key])
        if len(gains) == 0:
            continue
        print(f"\n{ds_name} ({len(gains)} videos):")
        print(f"  Mean Δ: {np.mean(gains):+.2f}   Std: {np.std(gains):.2f}")
        print(f"  Median Δ: {np.median(gains):+.2f}   IQR: [{np.percentile(gains,25):.2f}, {np.percentile(gains,75):.2f}]")
        print(f"  Min Δ: {np.min(gains):+.2f}   Max Δ: {np.max(gains):+.2f}")
        print(f"  Improved: {(gains > 0).sum()}/{len(gains)} ({(gains>0).mean()*100:.0f}%)")
        print(f"  Worsened: {(gains < 0).sum()}/{len(gains)} ({(gains<0).mean()*100:.0f}%)")
        if len(gains) >= 5:
            print(f"  Top-3 improved: {np.sort(gains)[-3:][::-1].round(2)}")
            print(f"  Top-3 worsened: {np.sort(gains)[:3].round(2)}")
    
    # CholecT50 components
    for ccomp in ["AP_i", "AP_v", "AP_t", "AP_iv", "AP_it", "AP_ivt"]:
        key = f"cholect50_{ccomp.lower()}"
        if key not in summary:
            continue
        gains = np.array(summary[key])
        if len(gains) == 0:
            continue
        print(f"\nCholecT50 {ccomp} ({len(gains)} videos):")
        print(f"  Mean Δ: {np.mean(gains):+.2f}   Std: {np.std(gains):.2f}")
        print(f"  Improved: {(gains > 0).sum()}/{len(gains)} ({(gains>0).mean()*100:.0f}%)")
    
    # Save
    out = {"summary": {}, "per_video": per_video_out}
    for ds_name, key in [("Cholec80 F1", "cholec80"), ("Endoscapes mAP", "endoscapes"), ("CholecT50 AP_ivt", "cholect50_ap_ivt")]:
        g = np.array(summary[key])
        if len(g) > 0:
            out["summary"][key] = {
                "n_videos": len(g),
                "mean_delta": round(float(np.mean(g)), 2),
                "std_delta": round(float(np.std(g)), 2),
                "median_delta": round(float(np.median(g)), 2),
                "q25": round(float(np.percentile(g, 25)), 2),
                "q75": round(float(np.percentile(g, 75)), 2),
                "min_delta": round(float(np.min(g)), 2),
                "max_delta": round(float(np.max(g)), 2),
                "n_improved": int((g > 0).sum()),
                "n_worsened": int((g < 0).sum()),
                "pct_improved": round(float((g > 0).mean()), 4),
            }
    
    with open(save_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {save_path}")


if __name__ == "__main__":
    main()
