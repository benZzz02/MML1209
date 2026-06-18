"""Shared metrics: official CholecT50 component AP, Cholec80 F1, Endoscapes mAP."""
import numpy as np
from sklearn.metrics import f1_score, average_precision_score
import ivtmetrics


def resolve_nan(classwise):
    """Match test.py's resolve_nan: convert -0.0 to NaN."""
    classwise[classwise == -0.0] = np.nan
    return classwise

def cholect50_all_components(y_true, y_pred, video_ids, return_per_video=False):
    """Official CholecT50 metric: matches test.py process_cholect50 exactly.
    
    For each video:
      1. Decompose 100-dim triplet into i/v/t/iv/it components using ivtmetrics
      2. Compute per-class AP for each component (average=None)
      3. resolve_nan to handle -0.0
      4. Collect per-video × per-class AP arrays
    Then:
      5. np.nanmean over videos (axis=0) → per-class AP
      6. np.nanmean over classes → component mAP
    
    AP_ivt is computed on the raw 100-dim vectors (no decomposition).
    """
    unique_vids = np.unique(video_ids)
    components = ["i", "v", "t", "iv", "it"]
    
    ap_lists = {c: [] for c in components}
    ap_ivt_list = []
    
    for vid in unique_vids:
        indices = np.where(video_ids == vid)[0]
        ivt_labels = y_true[indices]
        ivt_preds = y_pred[indices]
        
        filter = ivtmetrics.Disentangle()
        
        # AP_ivt on raw vectors (official)
        ap_ivt = resolve_nan(average_precision_score(ivt_labels, ivt_preds, average=None) * 100)
        ap_ivt_list.append(ap_ivt.reshape([1, -1]))
        
        # Component APs
        for comp in components:
            labels_comp = filter.extract(inputs=ivt_labels, component=comp)
            preds_comp = filter.extract(inputs=ivt_preds, component=comp)
            ap = resolve_nan(average_precision_score(labels_comp, preds_comp, average=None) * 100)
            ap_lists[comp].append(ap.reshape([1, -1]))
    
    results = {}
    per_video = {}
    
    # Components: nanmean over videos (axis=0) then nanmean over classes
    for comp in components:
        if len(ap_lists[comp]) == 0:
            results[f"AP_{comp}"] = 0.0
            continue
        arr = np.concatenate(ap_lists[comp], axis=0)     # [n_videos, n_classes]
        per_class = np.nanmean(arr, axis=0)               # [n_classes]  mean over videos
        results[f"AP_{comp}"] = round(float(np.nanmean(per_class)), 2)
        if return_per_video:
            per_video[f"AP_{comp}"] = [round(float(np.nanmean(arr[i])), 2) for i in range(arr.shape[0])]
    
    # AP_ivt (official)
    if len(ap_ivt_list) > 0:
        arr_ivt = np.concatenate(ap_ivt_list, axis=0)    # [n_videos, 100]
        per_class_ivt = np.nanmean(arr_ivt, axis=0)       # [100] mean over videos
        results["AP_ivt"] = round(float(np.nanmean(per_class_ivt)), 2)
        if return_per_video:
            per_video["AP_ivt"] = [round(float(np.nanmean(arr_ivt[i])), 2) for i in range(arr_ivt.shape[0])]
    else:
        results["AP_ivt"] = 0.0
    
    if return_per_video:
        results["_per_video_list"] = per_video
        results["_per_video_vids"] = [str(v) for v in unique_vids]
    
    return results


def cholec80_f1(y_true, y_pred):
    """Per-video macro F1 for Cholec80.
    y_true: [N, 7], y_pred: [N, 7] (sigmoid)
    """
    yt = np.argmax(y_true, axis=1)
    yp = np.argmax(y_pred, axis=1)
    return f1_score(yt, yp, average="macro", labels=np.unique(yt)) * 100


def endoscapes_map(y_true, y_pred):
    """Per-class mAP for Endoscapes.
    y_true: [N, 3], y_pred: [N, 3] (sigmoid)
    """
    aps = []
    for c in range(3):
        if y_true[:, c].sum() > 0:
            aps.append(average_precision_score(y_true[:, c], y_pred[:, c]) * 100)
    return float(np.mean(aps)) if aps else 0.0


def endoscapes_per_criterion(y_true, y_pred):
    """Per-criterion AP for Endoscapes. Returns {C1: x, C2: x, C3: x}."""
    result = {}
    for c in range(3):
        result[f"C{c+1}"] = round(
            average_precision_score(y_true[:, c], y_pred[:, c]) * 100, 2
        )
    return result


def all_metrics(p, l, v):
    """Convenience: returns all metrics for a dataset split."""
    r = {}
    vid_arr = np.array(v)
    
    mask = np.array(["cholec80" in x for x in vid_arr])
    if mask.any():
        r["cholec80_f1"] = cholec80_f1(l[mask][:, :7], p[mask][:, :7])
    
    mask = np.array(["endoscapes" in x for x in vid_arr])
    if mask.any():
        r["endoscapes_map"] = endoscapes_map(l[mask][:, 7:10], p[mask][:, 7:10])
        r["endoscapes_per_criterion"] = endoscapes_per_criterion(l[mask][:, 7:10], p[mask][:, 7:10])
        for c in range(3):
            r[f"cvs_c{c+1}"] = r["endoscapes_per_criterion"][f"C{c+1}"]
    
    mask = np.array(["cholect50" in x for x in vid_arr])
    if mask.any():
        cholect50 = cholect50_all_components(l[mask][:, 10:], p[mask][:, 10:], vid_arr[mask])
        for k, v in cholect50.items():
            if not k.endswith("_per_video"):
                r[f"cholect50_{k.lower()}"] = round(v, 2)
    
    return r
