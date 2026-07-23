"""Build a frame-level joint audit table from overlapping CAMMA datasets.

This script aligns Endoscapes CVS labels, Cholec80 phase labels, and
CholecT50 triplet labels on the three videos shared by all three datasets.
CholecT50 also carries a phase label in each annotation tuple in this
codebase, so the output includes both Cholec80 and CholecT50 phase columns.

Examples:
    # Print the overlap videos without requiring dataset files.
    python3 analysis/build_joint_audit_alignment.py --dry-run

    # Build the triple-overlap audit CSV using the data layout expected by
    # this repository. By default, rows missing the Cholec80 phase match are
    # skipped, so each output row has ground truth from all three datasets.
    python3 analysis/build_joint_audit_alignment.py \
        --data-root /data/cholecdata \
        --output analysis/joint_audit_alignment_triple.csv

    # If Endoscapes filenames contain original 25-fps frame ids while
    # CholecT50 uses 1-fps ids, use a scale factor.
    python3 analysis/build_joint_audit_alignment.py \
        --data-root /data/cholecdata \
        --endoscapes-frame-scale 25 \
        --frame-tolerance 1
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pickle
import re
from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


PHASE_SLICE = slice(0, 7)
CVS_SLICE = slice(7, 10)
TRIPLET_OFFSET = 10
DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OVERLAP_ROOT = Path(__file__).resolve().parents[2] / "camma_dataset_overlaps"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "joint_audit_alignment_triple.csv"


@dataclass(frozen=True)
class OverlapVideo:
    private_video_id: int
    cholec80_split: Optional[str] = None
    cholec80_id: Optional[int] = None
    cholect50_split: Optional[str] = None
    cholect50_id: Optional[int] = None
    endoscapes_split: Optional[str] = None
    endoscapes_id: Optional[int] = None


@dataclass
class EndoscapesFrame:
    split: str
    video_id: int
    frame_id: int
    file_name: str
    image_path: str
    cvs: List[int]


@dataclass
class CholecT50Frame:
    split: str
    video_id: int
    frame_id: int
    image_path: str
    triplet_ids: List[int]
    phase_id: Optional[int]
    raw_labels: Any


@dataclass
class Cholec80Frame:
    split: str
    video_id: int
    frame_id: int
    raw_frame_id: str
    image_path: str
    phase_id: int


def parse_frame_number(value: Any) -> int:
    """Parse numeric frame id from ints or strings such as 'video66_000001'."""
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)

    text = str(value)
    if text.isdigit():
        return int(text)

    numbers = re.findall(r"\d+", text)
    if not numbers:
        raise ValueError(f"Cannot parse frame id from {value!r}")
    return int(numbers[-1])


def frame_file_name(raw_frame_id: Any) -> str:
    text = str(raw_frame_id)
    if os.path.splitext(text)[1]:
        return text
    return f"{text}.png"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_label_names(repo_root: Path) -> Tuple[List[str], List[str], List[str]]:
    labels_path = repo_root / "cholec" / "cholec_labels.txt"
    if not labels_path.exists():
        return [], [], []

    labels = labels_path.read_text(encoding="utf-8").splitlines()
    phase_names = labels[PHASE_SLICE]
    cvs_names = labels[CVS_SLICE]
    triplet_names = labels[TRIPLET_OFFSET : TRIPLET_OFFSET + 100]
    return phase_names, cvs_names, triplet_names


def split_lookup_from_public_ids(
    splits: Dict[str, Sequence[int]],
    public_to_private: Dict[int, int],
) -> Dict[int, Tuple[str, int]]:
    out: Dict[int, Tuple[str, int]] = {}
    for split, public_ids in splits.items():
        for public_id in public_ids:
            private_id = public_to_private[int(public_id)]
            out[private_id] = (split, int(public_id))
    return out


def load_overlap_videos(overlap_root: Path, mode: str) -> List[OverlapVideo]:
    resources = overlap_root / "resources"
    cholec80_splits = load_json(resources / "Cholec80_splits.json")
    cholect50_splits = load_json(resources / "CholecT50_splits.json")
    endoscapes_splits = load_json(resources / "Endoscapes_splits.json")
    mapping = {int(k): int(v) for k, v in load_json(resources / "mapping_to_endoscapes.json").items()}

    private_to_endoscapes_public: Dict[int, int] = {}
    with (resources / "endoscapes_vid_id_map.csv").open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            private_to_endoscapes_public[int(row["orig_vid_id"])] = int(row["public_vid_id"])

    cholec80_by_private = split_lookup_from_public_ids(cholec80_splits, mapping)
    cholect50_by_private = split_lookup_from_public_ids(cholect50_splits, mapping)
    endoscapes_by_private = {
        int(private_id): (split, private_to_endoscapes_public[int(private_id)])
        for split, private_ids in endoscapes_splits.items()
        for private_id in private_ids
    }

    if mode == "endoscapes-cholect50":
        private_ids = sorted(set(endoscapes_by_private) & set(cholect50_by_private))
    elif mode == "triple":
        private_ids = sorted(set(endoscapes_by_private) & set(cholect50_by_private) & set(cholec80_by_private))
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    rows: List[OverlapVideo] = []
    for private_id in private_ids:
        c80 = cholec80_by_private.get(private_id)
        c50 = cholect50_by_private.get(private_id)
        endo = endoscapes_by_private.get(private_id)
        rows.append(
            OverlapVideo(
                private_video_id=private_id,
                cholec80_split=c80[0] if c80 else None,
                cholec80_id=c80[1] if c80 else None,
                cholect50_split=c50[0] if c50 else None,
                cholect50_id=c50[1] if c50 else None,
                endoscapes_split=endo[0] if endo else None,
                endoscapes_id=endo[1] if endo else None,
            )
        )
    return rows


def extract_frame_id(
    image_record: Dict[str, Any],
    file_name: str,
    video_id: int,
    frame_key: Optional[str],
    frame_regex: Optional[str],
) -> Optional[int]:
    if frame_key and frame_key in image_record:
        return int(image_record[frame_key])

    if frame_regex:
        match = re.search(frame_regex, file_name)
        if not match:
            return None
        return int(match.group(1))

    base = os.path.basename(file_name)
    numbers = [int(x) for x in re.findall(r"\d+", base)]
    if not numbers:
        return None

    # File names often contain video id followed by frame id. The frame id is
    # usually the final numeric token, so prefer it by default.
    if len(numbers) > 1 and numbers[-1] == video_id:
        return numbers[-2]
    return numbers[-1]


def normalize_cvs(values: Sequence[Any]) -> List[int]:
    cvs = [int(round(float(v))) for v in values[:3]]
    if len(cvs) < 3:
        cvs.extend([0] * (3 - len(cvs)))
    return cvs


def load_endoscapes_frames(
    data_root: Path,
    wanted_public_ids: Iterable[int],
    frame_key: Optional[str],
    frame_regex: Optional[str],
) -> Dict[int, List[EndoscapesFrame]]:
    wanted = set(int(x) for x in wanted_public_ids)
    by_video: Dict[int, List[EndoscapesFrame]] = {video_id: [] for video_id in wanted}

    for split in ("train", "val", "test"):
        split_root = data_root / "endoscapes" / split
        ann_path = split_root / "annotation_ds_coco.json"
        if not ann_path.exists():
            continue

        data = load_json(ann_path)
        for image_record in data.get("images", []):
            if "video_id" not in image_record or "file_name" not in image_record:
                continue
            public_id = int(image_record["video_id"])
            if public_id not in wanted:
                continue

            file_name = str(image_record["file_name"])
            frame_id = extract_frame_id(image_record, file_name, public_id, frame_key, frame_regex)
            if frame_id is None:
                continue

            ds = image_record.get("ds", [0, 0, 0])
            by_video.setdefault(public_id, []).append(
                EndoscapesFrame(
                    split=split,
                    video_id=public_id,
                    frame_id=frame_id,
                    file_name=file_name,
                    image_path=str(split_root / file_name),
                    cvs=normalize_cvs(ds),
                )
            )

    for frames in by_video.values():
        frames.sort(key=lambda item: item.frame_id)
    return by_video


def label_int(label: Any, keys: Sequence[str], index: Optional[int]) -> Optional[int]:
    if isinstance(label, dict):
        for key in keys:
            if key in label and label[key] is not None:
                return int(label[key])
        return None
    if index is None:
        return None
    if isinstance(label, (list, tuple)) and len(label) > index:
        value = label[index]
        if value is None:
            return None
        return int(value)
    return None


def phase_from_json_sidecar(data: Dict[str, Any], frame_id: int) -> Optional[int]:
    for key in ("phase_annotations", "phase", "phases"):
        item = data.get(key)
        if isinstance(item, dict):
            value = item.get(str(frame_id), item.get(frame_id))
            if isinstance(value, (list, tuple)):
                value = value[0] if value else None
            if value is not None:
                return int(value)
    return None


def load_cholect50_frames(
    data_root: Path,
    wanted_public_ids: Iterable[int],
    phase_index: int,
) -> Dict[int, Dict[int, CholecT50Frame]]:
    by_video: Dict[int, Dict[int, CholecT50Frame]] = {}
    wanted = set(int(x) for x in wanted_public_ids)

    for public_id in sorted(wanted):
        vid_name = f"VID{public_id:02d}" if public_id < 100 else f"VID{public_id:03d}"
        label_path = data_root / "cholect50" / "labels" / f"{vid_name}.json"
        video_root = data_root / "cholect50" / "videos" / vid_name
        if not label_path.exists():
            continue

        data = load_json(label_path)
        annotations = data.get("annotations", {})
        frames: Dict[int, CholecT50Frame] = {}

        for raw_frame_id, labels in annotations.items():
            frame_id = int(raw_frame_id)
            triplet_ids: List[int] = []
            phase_id = phase_from_json_sidecar(data, frame_id)

            for label in labels:
                triplet_id = label_int(label, ("triplet", "triplet_id", "ivt", "ivt_id"), 0)
                if triplet_id is not None and triplet_id >= 0:
                    triplet_ids.append(triplet_id)

                if phase_id is None:
                    phase = label_int(label, ("phase", "phase_id"), phase_index)
                    if phase is not None and phase >= 0:
                        phase_id = phase

            frames[frame_id] = CholecT50Frame(
                split="",
                video_id=public_id,
                frame_id=frame_id,
                image_path=str(video_root / f"{frame_id:06d}.png"),
                triplet_ids=sorted(set(triplet_ids)),
                phase_id=phase_id,
                raw_labels=labels,
            )
        by_video[public_id] = frames
    return by_video


def load_cholec80_phase_frames(
    data_root: Path,
    overlaps: Sequence[OverlapVideo],
) -> Dict[int, Dict[int, Cholec80Frame]]:
    """Load optional Cholec80 phase labels keyed by Cholec80 public id/frame id."""
    wanted = {
        (item.cholec80_split, item.cholec80_id)
        for item in overlaps
        if item.cholec80_split is not None and item.cholec80_id is not None
    }
    by_video: Dict[int, Dict[int, Cholec80Frame]] = {}

    for split, public_id in sorted(wanted):
        label_path = data_root / "cholec80" / "labels" / split / f"frame_phase_{split}.pkl"
        if not label_path.exists():
            alt_path = data_root / "cholec80" / "labels" / split / "1fps.pickle"
            label_path = alt_path if alt_path.exists() else label_path
        if not label_path.exists():
            continue

        with label_path.open("rb") as f:
            data = pickle.load(f)

        video_key = f"video{public_id:02d}" if public_id < 100 else f"video{public_id:03d}"
        if video_key not in data:
            video_key = f"video{public_id}"
        if video_key not in data:
            continue

        frames: Dict[int, Cholec80Frame] = {}
        video_root = data_root / "cholec80" / "frames" / split / video_key
        for record in data[video_key]:
            raw_frame_id = record["Frame_id"]
            frame_id = parse_frame_number(raw_frame_id)
            frames[frame_id] = Cholec80Frame(
                split=split,
                video_id=int(public_id),
                frame_id=frame_id,
                raw_frame_id=str(raw_frame_id),
                image_path=str(video_root / frame_file_name(raw_frame_id)),
                phase_id=int(record["Phase_gt"]),
            )
        by_video[int(public_id)] = frames
    return by_video


def scaled_frame_id(frame_id: int, scale: float, offset: float) -> int:
    return int(round((float(frame_id) + offset) / scale))


def nearest_key(keys: Sequence[int], target: int, tolerance: int) -> Optional[int]:
    if not keys:
        return None
    pos = bisect_left(keys, target)
    candidates = []
    if pos < len(keys):
        candidates.append(keys[pos])
    if pos > 0:
        candidates.append(keys[pos - 1])
    if not candidates:
        return None
    best = min(candidates, key=lambda value: abs(value - target))
    if abs(best - target) <= tolerance:
        return best
    return None


def join_names(ids: Sequence[int], names: Sequence[str], offset: int = 0) -> str:
    out = []
    for item_id in ids:
        local = item_id - offset
        if 0 <= local < len(names):
            out.append(names[local])
        else:
            out.append(str(item_id))
    return " | ".join(out)


def build_rows(args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    repo_root = Path(args.repo_root).resolve()
    data_root = Path(args.data_root).resolve()
    overlap_root = Path(args.overlap_root).resolve()

    overlaps = load_overlap_videos(overlap_root, args.mode)
    phase_names, cvs_names, triplet_names = load_label_names(repo_root)

    endo_ids = [item.endoscapes_id for item in overlaps if item.endoscapes_id is not None]
    c50_ids = [item.cholect50_id for item in overlaps if item.cholect50_id is not None]

    endo_frames = load_endoscapes_frames(data_root, endo_ids, args.endoscapes_frame_key, args.endoscapes_frame_regex)
    c50_frames = load_cholect50_frames(data_root, c50_ids, args.cholect50_phase_index)
    c80_phase = load_cholec80_phase_frames(data_root, overlaps) if args.mode == "triple" else {}

    rows: List[Dict[str, Any]] = []
    summary = {
        "overlap_videos": len(overlaps),
        "endoscapes_frames_seen": 0,
        "matched_rows": 0,
        "unmatched_endoscapes_frames": 0,
        "skipped_missing_cholec80_phase": 0,
        "rows_without_phase": 0,
        "rows_without_triplet": 0,
    }

    for item in overlaps:
        if item.endoscapes_id is None or item.cholect50_id is None:
            continue

        endo_list = endo_frames.get(item.endoscapes_id, [])
        c50_by_frame = c50_frames.get(item.cholect50_id, {})
        c50_keys = sorted(c50_by_frame)
        summary["endoscapes_frames_seen"] += len(endo_list)

        for endo in endo_list:
            target_frame = scaled_frame_id(endo.frame_id, args.endoscapes_frame_scale, args.endoscapes_frame_offset)
            matched_frame = nearest_key(c50_keys, target_frame, args.frame_tolerance)
            if matched_frame is None:
                summary["unmatched_endoscapes_frames"] += 1
                continue

            c50 = c50_by_frame[matched_frame]
            cholect50_phase_id = c50.phase_id
            cholec80_phase_id = None
            cholec80_phase_frame = ""
            cholec80_raw_frame = ""
            cholec80_image_path = ""
            if item.cholec80_id is not None:
                c80_by_frame = c80_phase.get(item.cholec80_id, {})
                c80_keys = sorted(c80_by_frame)
                c80_frame = nearest_key(c80_keys, target_frame, args.frame_tolerance)
                if c80_frame is not None:
                    c80_match = c80_by_frame[c80_frame]
                    cholec80_phase_frame = c80_match.frame_id
                    cholec80_raw_frame = c80_match.raw_frame_id
                    cholec80_image_path = c80_match.image_path
                    cholec80_phase_id = c80_match.phase_id

            if args.phase_source == "cholec80":
                phase_id = cholec80_phase_id if cholec80_phase_id is not None else cholect50_phase_id
                phase_source = "cholec80" if cholec80_phase_id is not None else "cholect50_fallback"
            else:
                phase_id = cholect50_phase_id if cholect50_phase_id is not None else cholec80_phase_id
                phase_source = "cholect50" if cholect50_phase_id is not None else "cholec80_fallback"

            if args.mode == "triple" and not args.keep_incomplete and cholec80_phase_id is None:
                summary["skipped_missing_cholec80_phase"] += 1
                continue

            if phase_id is None:
                summary["rows_without_phase"] += 1
            if not c50.triplet_ids:
                summary["rows_without_triplet"] += 1

            cvs_positive_global = [7 + i for i, value in enumerate(endo.cvs) if value == 1]
            row = {
                "private_video_id": item.private_video_id,
                "image_path": endo.image_path,
                "image_source": "endoscapes",
                "cholec80_split": item.cholec80_split or "",
                "cholec80_id": item.cholec80_id or "",
                "cholect50_split": item.cholect50_split or "",
                "cholect50_id": item.cholect50_id,
                "endoscapes_split": item.endoscapes_split or endo.split,
                "endoscapes_id": item.endoscapes_id,
                "endoscapes_file": endo.file_name,
                "endoscapes_image_path": endo.image_path,
                "endoscapes_frame_id": endo.frame_id,
                "target_cholect50_frame_id": target_frame,
                "matched_cholect50_frame_id": matched_frame,
                "frame_delta": matched_frame - target_frame,
                "cholec80_image_path": cholec80_image_path,
                "cholect50_image_path": c50.image_path,
                "endoscapes_gt_cvs": json.dumps(endo.cvs),
                "phase_source": phase_source,
                "phase_id": "" if phase_id is None else phase_id,
                "phase_name": "" if phase_id is None else join_names([phase_id], phase_names),
                "cholec80_phase_frame_id": cholec80_phase_frame,
                "cholec80_raw_frame_id": cholec80_raw_frame,
                "cholec80_phase_id": "" if cholec80_phase_id is None else cholec80_phase_id,
                "cholec80_phase_name": "" if cholec80_phase_id is None else join_names([cholec80_phase_id], phase_names),
                "cholect50_phase_id": "" if cholect50_phase_id is None else cholect50_phase_id,
                "cholect50_phase_name": "" if cholect50_phase_id is None else join_names([cholect50_phase_id], phase_names),
                "cvs_c1": endo.cvs[0],
                "cvs_c2": endo.cvs[1],
                "cvs_c3": endo.cvs[2],
                "cvs_positive_global_ids": json.dumps(cvs_positive_global),
                "cvs_positive_names": join_names(cvs_positive_global, cvs_names, offset=7),
                "has_triplet_positive": int(bool(c50.triplet_ids)),
                "triplet_ids": json.dumps(c50.triplet_ids),
                "triplet_global_ids": json.dumps([TRIPLET_OFFSET + tid for tid in c50.triplet_ids]),
                "triplet_names": join_names(c50.triplet_ids, triplet_names),
                "cholect50_raw_labels": json.dumps(c50.raw_labels, ensure_ascii=True, separators=(",", ":")),
            }
            rows.append(row)
            summary["matched_rows"] += 1

    return rows, summary


def print_overlaps(overlaps: Sequence[OverlapVideo]) -> None:
    print(f"Overlap videos: {len(overlaps)}")
    for item in overlaps:
        print(
            "  "
            f"private={item.private_video_id} | "
            f"Endoscapes-{item.endoscapes_split}:{item.endoscapes_id} | "
            f"CholecT50-{item.cholect50_split}:{item.cholect50_id} | "
            f"Cholec80-{item.cholec80_split}:{item.cholec80_id}"
        )


def range_text(values: Sequence[int]) -> str:
    if not values:
        return "empty"
    return f"{min(values)}..{max(values)}"


def count_frame_matches(endo_ids: Sequence[int], target_ids: Sequence[int], scale: float, tolerance: int) -> int:
    target_sorted = sorted(target_ids)
    count = 0
    for frame_id in endo_ids:
        scaled = scaled_frame_id(frame_id, scale, 0.0)
        if nearest_key(target_sorted, scaled, tolerance) is not None:
            count += 1
    return count


def inspect_frame_ids(args: argparse.Namespace) -> None:
    data_root = Path(args.data_root).resolve()
    overlap_root = Path(args.overlap_root).resolve()
    overlaps = load_overlap_videos(overlap_root, args.mode)

    endo_ids = [item.endoscapes_id for item in overlaps if item.endoscapes_id is not None]
    c50_ids = [item.cholect50_id for item in overlaps if item.cholect50_id is not None]
    endo_frames = load_endoscapes_frames(data_root, endo_ids, args.endoscapes_frame_key, args.endoscapes_frame_regex)
    c50_frames = load_cholect50_frames(data_root, c50_ids, args.cholect50_phase_index)
    c80_phase = load_cholec80_phase_frames(data_root, overlaps) if args.mode == "triple" else {}

    print("\nFrame-id inspection:")
    for item in overlaps:
        endo_list = endo_frames.get(item.endoscapes_id or -1, [])
        c50_by_frame = c50_frames.get(item.cholect50_id or -1, {})
        c80_by_frame = c80_phase.get(item.cholec80_id or -1, {})

        endo_frame_ids = [frame.frame_id for frame in endo_list]
        c50_frame_ids = sorted(c50_by_frame)
        c80_frame_ids = sorted(c80_by_frame)

        print(
            "\n"
            f"private={item.private_video_id} | "
            f"Endoscapes-{item.endoscapes_split}:{item.endoscapes_id} | "
            f"CholecT50-{item.cholect50_split}:{item.cholect50_id} | "
            f"Cholec80-{item.cholec80_split}:{item.cholec80_id}"
        )
        print(f"  Endoscapes frames: n={len(endo_frame_ids)}, range={range_text(endo_frame_ids)}")
        for frame in endo_list[:5]:
            print(f"    endo sample: frame_id={frame.frame_id}, file={frame.file_name}")

        print(f"  CholecT50 frames: n={len(c50_frame_ids)}, range={range_text(c50_frame_ids)}")
        print(f"    c50 sample ids: {c50_frame_ids[:10]}")

        print(f"  Cholec80 phase frames: n={len(c80_frame_ids)}, range={range_text(c80_frame_ids)}")
        c80_samples = [c80_by_frame[key] for key in c80_frame_ids[:5]]
        for frame in c80_samples:
            print(f"    c80 sample: parsed={frame.frame_id}, raw={frame.raw_frame_id}")

        if endo_frame_ids and c50_frame_ids:
            print("  Endoscapes -> CholecT50 direct/scale match counts:")
            for scale in (1.0, 5.0, 10.0, 25.0, 30.0):
                exact = count_frame_matches(endo_frame_ids, c50_frame_ids, scale, 0)
                loose = count_frame_matches(endo_frame_ids, c50_frame_ids, scale, max(args.frame_tolerance, 2))
                print(f"    scale={scale:g}: exact={exact}, tol={max(args.frame_tolerance, 2)} -> {loose}")


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Align overlapping Endoscapes and CholecT50 frames into a joint CVS+phase+triplet audit CSV."
    )
    parser.add_argument("--data-root", default="/data/cholecdata", help="Root containing cholec80/endoscapes/cholect50.")
    parser.add_argument("--repo-root", default=str(DEFAULT_REPO_ROOT), help="MML1209 repository root, used to read cholec/cholec_labels.txt.")
    parser.add_argument(
        "--overlap-root",
        default=str(DEFAULT_OVERLAP_ROOT),
        help="Path to CAMMA-public/camma_dataset_overlaps checkout.",
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output CSV path.")
    parser.add_argument(
        "--mode",
        choices=("endoscapes-cholect50", "triple"),
        default="triple",
        help="Use all Endoscapes/CholecT50 overlaps, or restrict to videos also present in Cholec80.",
    )
    parser.add_argument(
        "--phase-source",
        choices=("cholect50", "cholec80"),
        default="cholec80",
        help="Primary phase source for phase_id/phase_name. Both Cholec80 and CholecT50 phase columns are exported when available.",
    )
    parser.add_argument(
        "--cholect50-phase-index",
        type=int,
        default=14,
        help="Index of phase id inside each CholecT50 annotation tuple.",
    )
    parser.add_argument(
        "--endoscapes-frame-key",
        default=None,
        help="Optional explicit JSON key containing Endoscapes frame id.",
    )
    parser.add_argument(
        "--endoscapes-frame-regex",
        default=None,
        help="Optional regex with one capturing group to extract frame id from Endoscapes file_name.",
    )
    parser.add_argument(
        "--endoscapes-frame-scale",
        type=float,
        default=1.0,
        help="Divide extracted Endoscapes frame id by this value before matching CholecT50 frame id.",
    )
    parser.add_argument(
        "--endoscapes-frame-offset",
        type=float,
        default=0.0,
        help="Add this offset to the extracted Endoscapes frame id before scaling.",
    )
    parser.add_argument(
        "--frame-tolerance",
        type=int,
        default=0,
        help="Maximum absolute frame-id difference allowed after scaling.",
    )
    parser.add_argument(
        "--keep-incomplete",
        action="store_true",
        help="Keep matched Endoscapes/CholecT50 rows even if Cholec80 phase cannot be matched.",
    )
    parser.add_argument(
        "--inspect-frame-ids",
        action="store_true",
        help="Print filename/frame-id samples and scale-match counts, then exit.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only print overlap videos; do not read dataset annotations.")
    args = parser.parse_args()

    if not math.isfinite(args.endoscapes_frame_scale) or args.endoscapes_frame_scale == 0:
        parser.error("--endoscapes-frame-scale must be finite and non-zero.")
    if args.frame_tolerance < 0:
        parser.error("--frame-tolerance must be >= 0.")
    return args


def main() -> None:
    args = parse_args()
    overlap_root = Path(args.overlap_root).resolve()
    overlaps = load_overlap_videos(overlap_root, args.mode)
    print_overlaps(overlaps)

    if args.dry_run:
        return
    if args.inspect_frame_ids:
        inspect_frame_ids(args)
        return

    rows, summary = build_rows(args)
    output_path = Path(args.output).resolve()
    write_csv(output_path, rows)

    print("\nAlignment summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    print(f"\nWrote: {output_path}")

    if summary["matched_rows"] == 0:
        print(
            "\nNo rows were matched. Check whether --data-root points to the real datasets "
            "and whether Endoscapes frame ids need --endoscapes-frame-scale or --endoscapes-frame-regex."
        )


if __name__ == "__main__":
    main()
