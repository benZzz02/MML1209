"""Run hidden-positive recovery audits on frame-aligned overlap CSV files.

This script is intentionally standalone: it does not modify the training or
test dataloaders. It reads overlap CSVs, runs model inference on the listed
image paths, and evaluates whether positive labels from non-visible tasks are
recovered by probability ranking or by the structured candidate mechanism.

Example:
    python analysis/audit_hidden_positive_recovery.py \
        -c configs/scpnet_hill_Ignore_sp.yaml \
        --ckpt /path/to/epoch_xxx.ckpt \
        --audit-csv /data/MML1209/analysis/joint_audit_alignment_triple.csv
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode


def preparse_args() -> argparse.Namespace:
    """Parse audit-specific args before project args.py sees sys.argv."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--audit-csv",
        "--csv",
        nargs="+",
        default=["joint_audit_alignment_triple.csv"],
        help="One or more frame-aligned audit CSV files.",
    )
    parser.add_argument("--ckpt", required=True, help="Checkpoint path.")
    parser.add_argument("--output-dir", default="analysis/audit_recovery_out")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--image-column",
        default="visible-task",
        help="CSV image path column, auto, or visible-task.",
    )
    parser.add_argument(
        "--tasks",
        default="auto",
        help="Comma-separated task set to evaluate, or auto from CSV name.",
    )
    parser.add_argument("--candidate-topk", type=int, default=None)
    parser.add_argument("--thresholds", default="0.3,0.5")
    lc_group = parser.add_mutually_exclusive_group()
    lc_group.add_argument(
        "--enable-lc",
        dest="lc_override",
        action="store_true",
        default=None,
        help="Override YAML and enable logit compensation.",
    )
    lc_group.add_argument(
        "--disable-lc",
        dest="lc_override",
        action="store_false",
        help="Override YAML and disable logit compensation.",
    )
    ignore_group = parser.add_mutually_exclusive_group()
    ignore_group.add_argument(
        "--enable-ignore",
        dest="ignore_override",
        action="store_true",
        default=None,
        help="Override YAML and enable ignore-mask candidate logic.",
    )
    ignore_group.add_argument(
        "--disable-ignore",
        dest="ignore_override",
        action="store_false",
        help="Override YAML and disable ignore-mask candidate logic.",
    )
    parser.add_argument("--external-matrix", default=None)
    parser.add_argument("--no-external", action="store_true")
    parser.add_argument("--skip-missing", action="store_true")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--save-npy", action="store_true")

    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return args


AUDIT_ARGS = preparse_args()

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
warnings.filterwarnings("ignore")

# Project imports happen after audit args are removed from sys.argv.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from config import cfg as global_cfg  # noqa: E402
from model import (  # noqa: E402
    MMLSurgAdaptSCPNet,
    compensate_logits_by_pred_prob,
    get_topk_related_labels,
    load_clip_model,
)


TASK_RANGES = {
    "phase": range(0, 7),
    "cvs": range(7, 10),
    "triplet": range(10, 110),
}

DEFAULT_TASK_TOPKS = {
    "phase": [1],
    "cvs": [1, 2, 3],
    "triplet": [1, 5, 10],
}


@dataclass
class AuditRow:
    csv_name: str
    csv_path: str
    row_index: int
    row: Dict[str, str]
    image_path: str
    tasks: List[str]
    positives: Dict[str, Set[int]]


@dataclass
class AuditCase:
    row_ref: AuditRow
    visible_task: str
    hidden_task: str
    image_path: str
    visible_labels: Set[int]
    hidden_labels: Set[int]


class AuditImageDataset(Dataset):
    def __init__(self, image_paths: Sequence[str], transform=None):
        self.image_paths = list(image_paths)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        path = self.image_paths[index]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, index


def parse_literal(value: object):
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return ast.literal_eval(text)
    except Exception:
        return None


def parse_int(value: object) -> Optional[int]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(float(text))
    except Exception:
        return None


def first_int(row: Dict[str, str], keys: Sequence[str]) -> Optional[int]:
    for key in keys:
        value = parse_int(row.get(key))
        if value is not None:
            return value
    return None


def resolve_csv_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.exists():
        return path.resolve()
    candidates = [
        Path.cwd() / path,
        PROJECT_ROOT / path,
        PROJECT_ROOT / "analysis" / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return path.resolve()


def infer_tasks(csv_path: Path, override: str) -> List[str]:
    if override != "auto":
        tasks = [item.strip() for item in override.split(",") if item.strip()]
        bad = [task for task in tasks if task not in TASK_RANGES]
        if bad:
            raise ValueError(f"Unknown task(s) in --tasks: {bad}")
        return tasks

    name = csv_path.name.lower()
    if "phase_triplet" in name:
        return ["phase", "triplet"]
    if "phase_cvs" in name:
        return ["phase", "cvs"]
    if "triplet_cvs" in name or "endo_c50" in name:
        return ["triplet", "cvs"]
    if "triple" in name:
        return ["phase", "cvs", "triplet"]
    return ["phase", "cvs", "triplet"]


def select_image_path(row: Dict[str, str], image_column: str) -> str:
    if image_column not in ("auto", "visible-task"):
        return str(row.get(image_column, "")).strip()

    for key in (
        "image_path",
        "endoscapes_image_path",
        "cholect50_image_path",
        "cholec80_image_path",
    ):
        value = str(row.get(key, "")).strip()
        if value:
            return value
    return ""


def select_case_image_path(row: AuditRow, visible_task: str, image_column: str) -> str:
    if image_column not in ("auto", "visible-task"):
        return str(row.row.get(image_column, "")).strip()

    if image_column == "visible-task":
        task_column = {
            "phase": "cholec80_image_path",
            "triplet": "cholect50_image_path",
            "cvs": "endoscapes_image_path",
        }[visible_task]
        value = str(row.row.get(task_column, "")).strip()
        if value:
            return value

    return row.image_path


def row_positives(row: Dict[str, str]) -> Dict[str, Set[int]]:
    positives = {task: set() for task in TASK_RANGES}

    phase_id = first_int(row, ("phase_id", "cholec80_phase_id", "cholect50_phase_id"))
    if phase_id is not None and 0 <= phase_id <= 6:
        positives["phase"].add(phase_id)

    cvs_added = False
    for i, key in enumerate(("cvs_c1", "cvs_c2", "cvs_c3")):
        value = parse_int(row.get(key))
        if value == 1:
            positives["cvs"].add(7 + i)
            cvs_added = True

    if not cvs_added:
        cvs = parse_literal(row.get("endoscapes_gt_cvs"))
        if isinstance(cvs, list):
            for i, value in enumerate(cvs[:3]):
                try:
                    if int(round(float(value))) == 1:
                        positives["cvs"].add(7 + i)
                except Exception:
                    pass

    triplet_global = parse_literal(row.get("triplet_global_ids"))
    if isinstance(triplet_global, list):
        for value in triplet_global:
            label = parse_int(value)
            if label is not None and 10 <= label < 110:
                positives["triplet"].add(label)

    if not positives["triplet"]:
        triplet_local = parse_literal(row.get("triplet_ids"))
        if isinstance(triplet_local, list):
            for value in triplet_local:
                label = parse_int(value)
                if label is not None and 0 <= label < 100:
                    positives["triplet"].add(10 + label)

    return positives


def load_audit_rows(args: argparse.Namespace) -> List[AuditRow]:
    loaded: List[AuditRow] = []
    for csv_arg in args.audit_csv:
        csv_path = resolve_csv_path(csv_arg)
        tasks = infer_tasks(csv_path, args.tasks)
        with csv_path.open(newline="") as f:
            reader = csv.DictReader(f)
            for row_index, row in enumerate(reader):
                image_path = select_image_path(row, args.image_column)
                positives = row_positives(row)
                loaded.append(
                    AuditRow(
                        csv_name=csv_path.stem,
                        csv_path=str(csv_path),
                        row_index=row_index,
                        row=row,
                        image_path=image_path,
                        tasks=tasks,
                        positives=positives,
                    )
                )

    if args.max_samples and args.max_samples > 0:
        loaded = loaded[: args.max_samples]
    return loaded


def validate_image_paths(rows: List[AuditRow], skip_missing: bool) -> List[AuditRow]:
    missing = [row for row in rows if not row.image_path or not Path(row.image_path).exists()]
    if missing and not skip_missing:
        examples = "\n".join(
            f"  {item.csv_name} row={item.row_index} path={item.image_path}"
            for item in missing[:10]
        )
        raise FileNotFoundError(
            f"{len(missing)} image paths are missing. Use --skip-missing to ignore them.\n{examples}"
        )
    if missing:
        missing_ids = {(item.csv_name, item.row_index) for item in missing}
        rows = [row for row in rows if (row.csv_name, row.row_index) not in missing_ids]
        print(f"[Audit] Skipped missing images: {len(missing)}")
    return rows


def make_cases(rows: Sequence[AuditRow], image_column: str) -> List[AuditCase]:
    cases: List[AuditCase] = []
    for row in rows:
        for visible_task in row.tasks:
            visible_labels = set(row.positives.get(visible_task, set()))
            if not visible_labels:
                continue
            image_path = select_case_image_path(row, visible_task, image_column)
            for hidden_task in row.tasks:
                if hidden_task == visible_task:
                    continue
                hidden_labels = set(row.positives.get(hidden_task, set()))
                if not hidden_labels:
                    continue
                cases.append(
                    AuditCase(
                        row_ref=row,
                        visible_task=visible_task,
                        hidden_task=hidden_task,
                        image_path=image_path,
                        visible_labels=visible_labels,
                        hidden_labels=hidden_labels,
                    )
                )
    return cases


def validate_case_image_paths(cases: List[AuditCase], skip_missing: bool) -> List[AuditCase]:
    missing = [case for case in cases if not case.image_path or not Path(case.image_path).exists()]
    if missing and not skip_missing:
        examples = "\n".join(
            f"  {case.row_ref.csv_name} row={case.row_ref.row_index} "
            f"{case.visible_task}->{case.hidden_task} path={case.image_path}"
            for case in missing[:10]
        )
        raise FileNotFoundError(
            f"{len(missing)} case image paths are missing. Use --skip-missing to ignore them.\n{examples}"
        )
    if missing:
        missing_ids = {
            (case.row_ref.csv_name, case.row_ref.row_index, case.visible_task, case.hidden_task)
            for case in missing
        }
        cases = [
            case
            for case in cases
            if (case.row_ref.csv_name, case.row_ref.row_index, case.visible_task, case.hidden_task)
            not in missing_ids
        ]
        print(f"[Audit] Skipped missing case images: {len(missing)}")
    return cases


def configure_runtime(args: argparse.Namespace) -> None:
    global_cfg.perform_init = False
    if args.lc_override is not None:
        global_cfg.SCP_ENABLE_LOGIT_COMP = bool(args.lc_override)
    if args.ignore_override is not None:
        global_cfg.SCP_ENABLE_IGNORE_MASK = bool(args.ignore_override)
    if args.candidate_topk is not None:
        global_cfg.SCP_TOPK_K = int(args.candidate_topk)
    if args.external_matrix is not None:
        global_cfg.SCP_EXTERNAL_MATRIX_PATH = args.external_matrix
    if args.no_external:
        global_cfg.SCP_EXTERNAL_MATRIX_PATH = ""


def load_model(ckpt_path: str, device: torch.device) -> MMLSurgAdaptSCPNet:
    print("[Audit] Loading CLIP/model...")
    clip_model, _ = load_clip_model()
    clip_model.float()

    with open(PROJECT_ROOT / "cholec" / "cholec_labels.txt") as f:
        classnames = [line.strip() for line in f.read().splitlines() if line.strip()]

    model = MMLSurgAdaptSCPNet(classnames, clip_model).to(device)
    model.eval()

    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    new_state = {}
    for key, value in state.items():
        key = key.replace("module.", "")
        if "child_prompt_learner" in key:
            key = key.replace("child_prompt_learner", "prompt_learner")
        new_state[key] = value

    missing, unexpected = model.load_state_dict(new_state, strict=False)
    missing_core = [k for k in missing if "token" not in k and "prompt" not in k]
    if missing_core:
        print(f"[WARN] Missing non-prompt keys: {missing_core[:10]}")
    if unexpected:
        print(f"[WARN] Unexpected keys: {len(unexpected)}")

    spp = model.spp
    ext_matrix = getattr(global_cfg, "SCP_EXTERNAL_MATRIX_PATH", "")
    if ext_matrix and os.path.isfile(str(ext_matrix)):
        ext_np = np.load(str(ext_matrix), allow_pickle=False)
        ext = torch.as_tensor(ext_np, dtype=spp.A_clip_cos.dtype, device=device)
        ext = 0.5 * (ext + ext.t())
        ext = ext.clamp(-1.0, 1.0)
        if hasattr(spp, "A_ext_cos"):
            spp.A_ext_cos = ext
        else:
            spp.register_buffer("A_ext_cos", ext)
        spp.has_external = True
        print(f"[Audit] External matrix: {ext_matrix}")
    elif AUDIT_ARGS.no_external:
        spp.has_external = False
        print("[Audit] External matrix: disabled")

    return model


def run_inference(
    model: MMLSurgAdaptSCPNet,
    image_paths: Sequence[str],
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    imsize = int(getattr(global_cfg, "image_size", 224))
    preprocess = transforms.Compose(
        [
            transforms.Resize((imsize, imsize), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )
    dataset = AuditImageDataset(image_paths, transform=preprocess)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    raw_logits_list: List[torch.Tensor] = []
    logits_lc_list: List[torch.Tensor] = []
    probs_list: List[torch.Tensor] = []
    order_list: List[torch.Tensor] = []
    A_star_np = None

    use_comp = bool(getattr(global_cfg, "SCP_ENABLE_LOGIT_COMP", True))
    use_ignore = bool(getattr(global_cfg, "SCP_ENABLE_IGNORE_MASK", True))

    print(f"[Audit] Running inference on {len(dataset)} unique case images...")
    with torch.no_grad():
        for images, indices in loader:
            images = images.to(device)
            child_prompts = model.prompt_learner()
            Z = model.text_encoder(child_prompts, model.tokenized_prompts)

            img_feat = model.encode_image(images)
            img_feat = F.normalize(img_feat, p=2, dim=-1)

            A_star = model.spp()
            if A_star_np is None:
                A_star_np = A_star.detach().cpu().numpy().astype(np.float32)

            row_sum = A_star.sum(dim=1, keepdim=True)
            A_gcn = A_star / (row_sum + 1e-12)
            text_refined = model.sam(Z, A_gcn)
            text_refined = F.normalize(text_refined, p=2, dim=-1)

            raw = 10.0 * img_feat @ text_refined.t()
            logits_lc = raw.clone()

            if use_comp or use_ignore:
                pred_idx = raw.argmax(dim=-1)
                topk_related = [
                    get_topk_related_labels(int(pred_idx[b].item()), raw[b], A_star)
                    for b in range(pred_idx.shape[0])
                ]
                if use_comp:
                    logits_lc = compensate_logits_by_pred_prob(raw, pred_idx, topk_related)

            probs = torch.sigmoid(logits_lc)

            raw_logits_list.append(raw.detach().cpu())
            logits_lc_list.append(logits_lc.detach().cpu())
            probs_list.append(probs.detach().cpu())
            order_list.append(indices.detach().cpu())

    order = torch.cat(order_list, dim=0).numpy()
    raw_logits = torch.cat(raw_logits_list, dim=0).numpy().astype(np.float32)
    logits_lc = torch.cat(logits_lc_list, dim=0).numpy().astype(np.float32)
    probs = torch.cat(probs_list, dim=0).numpy().astype(np.float32)

    # DataLoader with shuffle=False should already be ordered; this keeps us safe.
    inv = np.argsort(order)
    return raw_logits[inv], logits_lc[inv], probs[inv], A_star_np


def task_topk(probs_row: np.ndarray, task: str, k: int) -> Set[int]:
    labels = list(TASK_RANGES[task])
    scores = probs_row[labels]
    k = min(k, len(labels))
    local = np.argsort(-scores)[:k]
    return {labels[int(i)] for i in local}


def flatten_candidate_output(nested: Sequence[Sequence[int]]) -> Set[int]:
    out: Set[int] = set()
    for group in nested:
        out.update(int(item) for item in group)
    return out


def candidate_set_for_anchors(
    anchors: Iterable[int],
    raw_row: np.ndarray,
    A_star: np.ndarray,
) -> Set[int]:
    raw_t = torch.as_tensor(raw_row, dtype=torch.float32)
    A_t = torch.as_tensor(A_star, dtype=torch.float32)
    out: Set[int] = set()
    for anchor in anchors:
        if 0 <= int(anchor) < A_star.shape[0]:
            out.update(flatten_candidate_output(get_topk_related_labels(int(anchor), raw_t, A_t)))
    return out


def label_name(label_id: int, names: Sequence[str]) -> str:
    if 0 <= label_id < len(names):
        return names[label_id]
    return str(label_id)


def summarize_metrics(records: Sequence[Dict[str, object]], thresholds: Sequence[float]) -> Dict[str, object]:
    if not records:
        return {"cases": 0}

    hidden_label_count = sum(int(r["num_hidden"]) for r in records)
    summary: Dict[str, object] = {
        "cases": len(records),
        "hidden_label_count": hidden_label_count,
        "visible_candidate_hit": float(np.mean([r["visible_candidate_hit"] for r in records]) * 100.0),
        "visible_candidate_recall": float(
            sum(float(r["visible_candidate_recovered"]) for r in records) / hidden_label_count * 100.0
        ),
        "pred_candidate_hit": float(np.mean([r["pred_candidate_hit"] for r in records]) * 100.0),
        "pred_candidate_recall": float(
            sum(float(r["pred_candidate_recovered"]) for r in records) / hidden_label_count * 100.0
        ),
    }

    topk_keys = sorted({key for r in records for key in r if key.startswith("prob_recovered@")})
    for key in topk_keys:
        k = key.split("@", 1)[1]
        hit_key = f"prob_hit@{k}"
        summary[hit_key] = float(np.mean([r[f"prob_hit@{k}"] for r in records]) * 100.0)
        summary[f"prob_recall@{k}"] = float(
            sum(float(r[key]) for r in records) / hidden_label_count * 100.0
        )

    for thr in thresholds:
        key = f"thr_recovered@{thr:g}"
        summary[f"thr_hit@{thr:g}"] = float(
            np.mean([r[f"thr_hit@{thr:g}"] for r in records]) * 100.0
        )
        summary[f"thr_recall@{thr:g}"] = float(
            sum(float(r[key]) for r in records) / hidden_label_count * 100.0
        )

    return summary


def evaluate(
    cases: Sequence[AuditCase],
    path_to_infer_index: Dict[str, int],
    raw_logits: np.ndarray,
    probs: np.ndarray,
    A_star: np.ndarray,
    thresholds: Sequence[float],
    classnames: Sequence[str],
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    records: List[Dict[str, object]] = []

    pred_idx_all = np.argmax(raw_logits, axis=1)

    for case in cases:
        row = case.row_ref
        idx = path_to_infer_index[case.image_path]
        hidden = set(case.hidden_labels)

        visible_candidates = candidate_set_for_anchors(case.visible_labels, raw_logits[idx], A_star)
        pred_candidates = candidate_set_for_anchors([int(pred_idx_all[idx])], raw_logits[idx], A_star)

        rec: Dict[str, object] = {
            "csv_name": row.csv_name,
            "row_index": row.row_index,
            "private_video_id": row.row.get("private_video_id", ""),
            "image_path": case.image_path,
            "visible_task": case.visible_task,
            "hidden_task": case.hidden_task,
            "visible_labels": sorted(case.visible_labels),
            "visible_label_names": [label_name(x, classnames) for x in sorted(case.visible_labels)],
            "hidden_labels": sorted(hidden),
            "hidden_label_names": [label_name(x, classnames) for x in sorted(hidden)],
            "num_hidden": len(hidden),
            "raw_pred_idx": int(pred_idx_all[idx]),
            "raw_pred_name": label_name(int(pred_idx_all[idx]), classnames),
            "visible_candidates": sorted(visible_candidates),
            "visible_candidate_names": [label_name(x, classnames) for x in sorted(visible_candidates)],
            "pred_candidates": sorted(pred_candidates),
            "pred_candidate_names": [label_name(x, classnames) for x in sorted(pred_candidates)],
        }

        recovered_visible = hidden & visible_candidates
        recovered_pred = hidden & pred_candidates
        rec["visible_candidate_recovered"] = len(recovered_visible)
        rec["visible_candidate_hit"] = int(bool(recovered_visible))
        rec["pred_candidate_recovered"] = len(recovered_pred)
        rec["pred_candidate_hit"] = int(bool(recovered_pred))

        for k in DEFAULT_TASK_TOPKS[case.hidden_task]:
            top = task_topk(probs[idx], case.hidden_task, k)
            recovered = hidden & top
            rec[f"prob_top{k}"] = sorted(top)
            rec[f"prob_recovered@{k}"] = len(recovered)
            rec[f"prob_hit@{k}"] = int(bool(recovered))

        for thr in thresholds:
            recovered_thr = {label for label in hidden if float(probs[idx, label]) >= thr}
            rec[f"thr_recovered@{thr:g}"] = len(recovered_thr)
            rec[f"thr_hit@{thr:g}"] = int(bool(recovered_thr))

        records.append(rec)

    grouped: Dict[str, List[Dict[str, object]]] = {}
    for rec in records:
        key = f"{rec['csv_name']}::{rec['visible_task']}->{rec['hidden_task']}"
        grouped.setdefault(key, []).append(rec)

    summary = {
        "num_csv_rows": len({(case.row_ref.csv_name, case.row_ref.row_index) for case in cases}),
        "num_unique_case_images": len(path_to_infer_index),
        "num_cases": len(cases),
        "candidate_topk": int(getattr(global_cfg, "SCP_TOPK_K", 1)),
        "thresholds": list(thresholds),
        "groups": {
            key: summarize_metrics(value, thresholds)
            for key, value in sorted(grouped.items())
        },
    }
    return records, summary


def write_case_csv(path: Path, records: Sequence[Dict[str, object]]) -> None:
    if not records:
        return
    fieldnames: List[str] = []
    seen = set()
    for rec in records:
        for key in rec.keys():
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for rec in records:
            row = {key: "" for key in fieldnames}
            for key, value in rec.items():
                if isinstance(value, (list, dict, set)):
                    row[key] = json.dumps(value, ensure_ascii=False)
                else:
                    row[key] = value
            writer.writerow(row)


def print_summary(summary: Dict[str, object]) -> None:
    print("\nHidden-positive audit recovery")
    print(
        f"CSV rows: {summary['num_csv_rows']} | "
        f"unique case images: {summary['num_unique_case_images']} | "
        f"directed cases: {summary['num_cases']}"
    )
    print(f"Candidate top-k per target group: {summary['candidate_topk']}")
    print(
        f"{'group':<46} {'cases':>7} {'labels':>8} "
        f"{'visCandR':>9} {'predCandR':>10} {'probR':>8}"
    )
    print("-" * 96)
    for group, metrics in summary["groups"].items():
        if not metrics.get("cases"):
            continue
        hidden_task = group.rsplit("->", 1)[-1]
        primary_k = {"phase": 1, "cvs": 1, "triplet": 5}.get(hidden_task, 1)
        prob_key = f"prob_recall@{primary_k}"
        if prob_key not in metrics:
            prob_keys = [k for k in metrics if k.startswith("prob_recall@")]
            prob_key = sorted(prob_keys, key=lambda x: int(x.split("@")[1]))[-1] if prob_keys else None
        prob_value = metrics.get(prob_key, float("nan")) if prob_key else float("nan")
        prob_label = f"{prob_value:.1f}"
        if prob_key:
            prob_label += f"@{prob_key.split('@')[1]}"
        print(
            f"{group:<46} {int(metrics['cases']):>7} {int(metrics['hidden_label_count']):>8} "
            f"{float(metrics['visible_candidate_recall']):>8.1f}% "
            f"{float(metrics['pred_candidate_recall']):>9.1f}% "
            f"{prob_label:>8}"
        )


def main() -> None:
    args = AUDIT_ARGS
    configure_runtime(args)

    rows = load_audit_rows(args)
    rows = validate_image_paths(rows, args.skip_missing)
    cases = make_cases(rows, args.image_column)
    cases = validate_case_image_paths(cases, args.skip_missing)
    if not rows:
        raise RuntimeError("No audit rows loaded.")
    if not cases:
        raise RuntimeError("No effective hidden-positive cases. Check CSV/task selection.")

    print(f"[Audit] Loaded rows: {len(rows)}")
    print(f"[Audit] Effective directed cases: {len(cases)}")
    unique_image_paths = sorted({case.image_path for case in cases})
    path_to_infer_index = {path: idx for idx, path in enumerate(unique_image_paths)}
    print(f"[Audit] Unique case images: {len(unique_image_paths)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.ckpt, device)
    raw_logits, logits_lc, probs, A_star = run_inference(model, unique_image_paths, args, device)

    with open(PROJECT_ROOT / "cholec" / "cholec_labels.txt") as f:
        classnames = [line.strip() for line in f.read().splitlines() if line.strip()]

    thresholds = [float(x) for x in args.thresholds.split(",") if x.strip()]
    records, summary = evaluate(
        cases,
        path_to_infer_index,
        raw_logits,
        probs,
        A_star,
        thresholds,
        classnames,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_case_csv(output_dir / "audit_recovery_cases.csv", records)
    with (output_dir / "audit_recovery_summary.json").open("w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    meta = {
        "audit_csv": [str(resolve_csv_path(x)) for x in args.audit_csv],
        "ckpt": str(args.ckpt),
        "config": sys.argv[sys.argv.index("-c") + 1] if "-c" in sys.argv else "",
        "image_column": args.image_column,
        "tasks": args.tasks,
        "use_logit_comp": bool(getattr(global_cfg, "SCP_ENABLE_LOGIT_COMP", True)),
        "use_ignore": bool(getattr(global_cfg, "SCP_ENABLE_IGNORE_MASK", True)),
        "candidate_topk": int(getattr(global_cfg, "SCP_TOPK_K", 1)),
    }
    with (output_dir / "audit_recovery_meta.json").open("w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    if args.save_npy:
        np.save(output_dir / "raw_logits.npy", raw_logits)
        np.save(output_dir / "logits_lc.npy", logits_lc)
        np.save(output_dir / "probs.npy", probs)
        np.save(output_dir / "A_star.npy", A_star)

    print_summary(summary)
    print(f"\n[Audit] Wrote: {output_dir / 'audit_recovery_summary.json'}")
    print(f"[Audit] Wrote: {output_dir / 'audit_recovery_cases.csv'}")


if __name__ == "__main__":
    main()
