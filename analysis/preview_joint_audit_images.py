"""Create an HTML gallery for the triple-overlap joint audit frames.

The gallery is intended for debugging frame alignment. Each row starts from
one Endoscapes image and shows the nearest CholecT50 and Cholec80 frames after
the same frame-id scaling used by build_joint_audit_alignment.py.

Example:
    cd /data/MML1209/analysis
    python preview_joint_audit_images.py --data-root /data/cholecdata

Then open:
    /data/MML1209/analysis/joint_audit_preview/index.html
"""

from __future__ import annotations

import argparse
import html
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

try:
    from PIL import Image
except Exception:  # pragma: no cover - PIL is optional for this helper.
    Image = None

from build_joint_audit_alignment import (
    DEFAULT_OVERLAP_ROOT,
    DEFAULT_REPO_ROOT,
    TRIPLET_OFFSET,
    join_names,
    load_cholec80_phase_frames,
    load_cholect50_frames,
    load_endoscapes_frames,
    load_label_names,
    load_overlap_videos,
    scaled_frame_id,
)


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "joint_audit_preview"


def nearest_any(keys: Sequence[int], target: int) -> Tuple[Optional[int], Optional[int]]:
    if not keys:
        return None, None
    best = min(keys, key=lambda value: abs(value - target))
    return best, best - target


def copy_or_thumbnail(src: str, dst: Path, max_width: int) -> Optional[str]:
    src_path = Path(src)
    if not src_path.exists():
        return None

    dst.parent.mkdir(parents=True, exist_ok=True)
    if Image is None:
        shutil.copy2(src_path, dst)
        return str(dst)

    with Image.open(src_path) as img:
        img = img.convert("RGB")
        width, height = img.size
        if width > max_width:
            new_height = max(1, int(height * max_width / width))
            img = img.resize((max_width, new_height))
        img.save(dst, quality=88)
    return str(dst)


def img_cell(path: str, label: str, out_dir: Path, thumbs_dir: Path, image_id: str, max_width: int) -> str:
    ext = Path(path).suffix.lower()
    if ext not in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
        ext = ".jpg"
    thumb_path = thumbs_dir / f"{image_id}{ext}"
    copied = copy_or_thumbnail(path, thumb_path, max_width)
    title = html.escape(path)
    caption = html.escape(label)
    if copied is None:
        return f'<div class="missing">missing<br><code>{title}</code></div><div class="cap">{caption}</div>'

    rel = os.path.relpath(copied, out_dir)
    return f'<img src="{html.escape(rel)}" title="{title}"><div class="cap">{caption}</div><code>{title}</code>'


def jsonish(values: Any) -> str:
    return html.escape(str(values))


def build_gallery(args: argparse.Namespace) -> Path:
    data_root = Path(args.data_root).resolve()
    repo_root = Path(args.repo_root).resolve()
    overlap_root = Path(args.overlap_root).resolve()
    out_dir = Path(args.output_dir).resolve()
    thumbs_dir = out_dir / "thumbs"
    out_dir.mkdir(parents=True, exist_ok=True)

    overlaps = load_overlap_videos(overlap_root, "triple")
    phase_names, cvs_names, triplet_names = load_label_names(repo_root)

    endo_ids = [item.endoscapes_id for item in overlaps if item.endoscapes_id is not None]
    c50_ids = [item.cholect50_id for item in overlaps if item.cholect50_id is not None]

    endo_frames = load_endoscapes_frames(data_root, endo_ids, args.endoscapes_frame_key, args.endoscapes_frame_regex)
    c50_frames = load_cholect50_frames(data_root, c50_ids, args.cholect50_phase_index)
    c80_frames = load_cholec80_phase_frames(data_root, overlaps)

    blocks = []
    total_rows = 0

    for item in overlaps:
        endo_list = endo_frames.get(item.endoscapes_id or -1, [])
        if args.max_frames_per_video > 0:
            endo_list = endo_list[: args.max_frames_per_video]

        c50_by_frame = c50_frames.get(item.cholect50_id or -1, {})
        c80_by_frame = c80_frames.get(item.cholec80_id or -1, {})
        c50_keys = sorted(c50_by_frame)
        c80_keys = sorted(c80_by_frame)

        rows_html = []
        for idx, endo in enumerate(endo_list):
            target = scaled_frame_id(endo.frame_id, args.endoscapes_frame_scale, args.endoscapes_frame_offset)
            c50_key, c50_delta = nearest_any(c50_keys, target)
            c80_key, c80_delta = nearest_any(c80_keys, target)

            c50 = c50_by_frame.get(c50_key) if c50_key is not None else None
            c80 = c80_by_frame.get(c80_key) if c80_key is not None else None

            c50_status = "missing" if c50 is None else f"target={target}, matched={c50_key}, delta={c50_delta}"
            c80_status = "missing" if c80 is None else f"target={target}, matched={c80_key}, delta={c80_delta}"

            cvs_global = [7 + i for i, value in enumerate(endo.cvs) if value == 1]
            cvs_text = f"CVS={endo.cvs}; {join_names(cvs_global, cvs_names, offset=7)}"

            c50_phase = "" if c50 is None or c50.phase_id is None else join_names([c50.phase_id], phase_names)
            c50_triplets = "" if c50 is None else join_names(c50.triplet_ids, triplet_names)
            c80_phase = "" if c80 is None else join_names([c80.phase_id], phase_names)

            prefix = f"v{item.private_video_id}_r{idx:04d}"
            endo_cell = img_cell(
                endo.image_path,
                f"Endoscapes {item.endoscapes_id}, frame={endo.frame_id}<br>{html.escape(cvs_text)}",
                out_dir,
                thumbs_dir,
                f"{prefix}_endo",
                args.thumb_width,
            )
            c50_cell = (
                '<div class="missing">missing</div>'
                if c50 is None
                else img_cell(
                    c50.image_path,
                    f"CholecT50 {item.cholect50_id}, {html.escape(c50_status)}<br>"
                    f"phase={html.escape(c50_phase)}<br>triplets={html.escape(c50_triplets)}",
                    out_dir,
                    thumbs_dir,
                    f"{prefix}_c50",
                    args.thumb_width,
                )
            )
            c80_cell = (
                '<div class="missing">missing</div>'
                if c80 is None
                else img_cell(
                    c80.image_path,
                    f"Cholec80 {item.cholec80_id}, {html.escape(c80_status)}<br>"
                    f"raw={html.escape(c80.raw_frame_id)}<br>phase={html.escape(c80_phase)}",
                    out_dir,
                    thumbs_dir,
                    f"{prefix}_c80",
                    args.thumb_width,
                )
            )

            rows_html.append(
                "<tr>"
                f"<td>{endo_cell}</td>"
                f"<td>{c50_cell}</td>"
                f"<td>{c80_cell}</td>"
                "</tr>"
            )
            total_rows += 1

        blocks.append(
            f"""
            <h2>private={item.private_video_id} |
                Endoscapes-{item.endoscapes_split}:{item.endoscapes_id} |
                CholecT50-{item.cholect50_split}:{item.cholect50_id} |
                Cholec80-{item.cholec80_split}:{item.cholec80_id}</h2>
            <table>
                <thead><tr><th>Endoscapes CVS</th><th>CholecT50 Triplet</th><th>Cholec80 Phase</th></tr></thead>
                <tbody>{''.join(rows_html)}</tbody>
            </table>
            """
        )

    html_text = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Triple-overlap joint audit preview</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 20px; color: #222; }}
h1 {{ margin-bottom: 4px; }}
h2 {{ margin-top: 28px; font-size: 18px; }}
table {{ border-collapse: collapse; width: 100%; margin-bottom: 24px; }}
th, td {{ border: 1px solid #ddd; vertical-align: top; padding: 8px; width: 33%; }}
th {{ background: #f4f4f4; }}
img {{ max-width: 100%; height: auto; display: block; margin-bottom: 6px; }}
code {{ font-size: 11px; color: #555; overflow-wrap: anywhere; }}
.cap {{ font-size: 12px; margin: 4px 0; line-height: 1.35; }}
.missing {{ padding: 24px; background: #fee; color: #900; font-weight: 700; }}
.meta {{ color: #555; }}
</style>
</head>
<body>
<h1>Triple-overlap joint audit preview</h1>
<p class="meta">Rows: {total_rows}. Data root: {html.escape(str(data_root))}. Frame scale: {args.endoscapes_frame_scale}.</p>
{''.join(blocks)}
</body>
</html>
"""
    index_path = out_dir / "index.html"
    index_path.write_text(html_text, encoding="utf-8")
    return index_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create an HTML preview for triple-overlap joint audit frames.")
    parser.add_argument("--data-root", default="/data/cholecdata", help="Root containing cholec80/endoscapes/cholect50.")
    parser.add_argument("--repo-root", default=str(DEFAULT_REPO_ROOT), help="MML1209 repository root.")
    parser.add_argument("--overlap-root", default=str(DEFAULT_OVERLAP_ROOT), help="camma_dataset_overlaps checkout.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory for HTML and thumbnails.")
    parser.add_argument("--cholect50-phase-index", type=int, default=14)
    parser.add_argument("--endoscapes-frame-key", default=None)
    parser.add_argument("--endoscapes-frame-regex", default=None)
    parser.add_argument("--endoscapes-frame-scale", type=float, default=1.0)
    parser.add_argument("--endoscapes-frame-offset", type=float, default=0.0)
    parser.add_argument("--thumb-width", type=int, default=360)
    parser.add_argument("--max-frames-per-video", type=int, default=0, help="0 means export every Endoscapes frame.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    index_path = build_gallery(args)
    print(f"Wrote preview: {index_path}")


if __name__ == "__main__":
    main()
