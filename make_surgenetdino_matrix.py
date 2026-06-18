#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_surgenetdino_matrix.py

For MML1209/gate:
    随机给每个标签抽 1 张正样本图片，用 SurgeNetDINO 官方权重抽特征，
    计算 110x110 cosine similarity matrix，保存为 .npy。

推荐放到：
    /data/MML1209/tools/make_surgenetdino_matrix.py

示例：
    CUDA_VISIBLE_DEVICES=0 python tools/make_surgenetdino_matrix.py \
      -c configs/你的配置.yaml \
      --variant dinov2_vitb \
      --split all \
      --out /data/MML1209/matrices/A_ext_cos_surgenetdino_vitb_seed42.npy
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode


# Same model names + official weight URLs as SurgeNetDINO/load_models.py
SURGENETDINO_SPECS: Dict[str, Dict[str, Any]] = {
    "dinov1_vits": {
        "timm_name": "vit_small_patch16_224.dino",
        "img_size": 224,
        "patch_size": 16,
        "url": "https://huggingface.co/rlpddejong/SurgeNetXL_DINOv1-v3/resolve/main/DINOv1_ViTs16_size224_SurgeNetXL.pth?download=true",
    },
    "dinov1_vitb": {
        "timm_name": "vit_base_patch16_224.dino",
        "img_size": 224,
        "patch_size": 16,
        "url": "https://huggingface.co/rlpddejong/SurgeNetXL_DINOv1-v3/resolve/main/DINOv1_ViTb16_size224_SurgeNetXL.pth?download=true",
    },
    "dinov2_vits": {
        "timm_name": "vit_small_patch14_dinov2",
        "img_size": 336,
        "patch_size": 14,
        "url": "https://huggingface.co/rlpddejong/SurgeNetXL_DINOv1-v3/resolve/main/DINOv2_ViTs14_size336_SurgeNetXL.pth?download=true",
    },
    "dinov2_vitb": {
        "timm_name": "vit_base_patch14_dinov2",
        "img_size": 336,
        "patch_size": 14,
        "url": "https://huggingface.co/rlpddejong/SurgeNetXL_DINOv1-v3/resolve/main/DINOv2_ViTb14_size336_SurgeNetXL.pth?download=true",
    },
    "dinov2_vitl": {
        "timm_name": "vit_large_patch14_dinov2",
        "img_size": 336,
        "patch_size": 14,
        "url": "https://huggingface.co/rlpddejong/SurgeNetXL_DINOv1-v3/resolve/main/DINOv2_ViTl14_size336_SurgeNetXL.pth?download=true",
    },
    "dinov3_vits": {
        "timm_name": "vit_small_patch16_dinov3.lvd1689m",
        "img_size": 336,
        "patch_size": 16,
        "url": "https://huggingface.co/rlpddejong/SurgeNetXL_DINOv1-v3/resolve/main/DINOv3_ViTs16_size336_SurgeNetXL.pth?download=true",
    },
    "dinov3_vitb": {
        "timm_name": "vit_base_patch16_dinov3.lvd1689m",
        "img_size": 336,
        "patch_size": 16,
        "url": "https://huggingface.co/rlpddejong/SurgeNetXL_DINOv1-v3/resolve/main/DINOv3_ViTb16_size336_SurgeNetXL.pth?download=true",
    },
    "dinov3_vitl": {
        "timm_name": "vit_large_patch16_dinov3.lvd1689m",
        "img_size": 336,
        "patch_size": 16,
        "url": "https://huggingface.co/rlpddejong/SurgeNetXL_DINOv1-v3/resolve/main/DINOv3_ViTl16_size336_SurgeNetXL.pth?download=true",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build 110x110 external cosine matrix from one random positive image per label using SurgeNetDINO."
    )
    parser.add_argument(
        "-c", "--config-file",
        required=True,
        help="MML1209 yaml config, e.g. configs/surgadapt+cholec.yaml"
    )
    parser.add_argument(
        "--variant",
        default="dinov2_vitb",
        choices=sorted(SURGENETDINO_SPECS.keys()),
        help="SurgeNetDINO official model variant. Default: dinov2_vitb"
    )
    parser.add_argument(
        "--weights",
        default="",
        help="Optional local .pth weight path. If empty, download/cache official SurgeNetDINO URL."
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output .npy path, e.g. /data/MML1209/matrices/A_ext_cos_surgenetdino_vitb_seed42.npy"
    )
    parser.add_argument(
        "--split",
        default="all",
        choices=["train", "val", "test", "all"],
        help="Dataset split to sample from. Default all is safer for covering all 110 labels."
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=110,
        help="Number of labels/classes for matrix. Default: 110."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling one image per class."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for feature extraction."
    )
    parser.add_argument(
        "--round",
        type=int,
        default=1,
        help="Dummy round passed into MML1209 args/config. Not used for output."
    )
    parser.add_argument(
        "--save-csv",
        action="store_true",
        help="Also save .csv for inspection."
    )
    parser.add_argument(
        "--no-preserve-multilabel",
        action="store_true",
        help="Do not force cfg.getitem=True before build_dataset. Usually leave this OFF."
    )
    return parser.parse_args()


def reset_argv_for_mml1209(config_file: str, round_id: int):
    """
    MML1209/config.py imports args.py, and args.py parses sys.argv.
    We clean sys.argv before importing config/dataset so extra args here do not break it.

    Use -t to avoid config.py creating a new train checkpoint round.
    """
    sys.argv = [
        sys.argv[0],
        "-c", config_file,
        "-t",
        "-r", str(round_id),
    ]


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_preprocess(img_size: int):
    # DINO/DINOv2/DINOv3 timm models generally use ImageNet normalization.
    return transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])


def clean_state_dict(obj: Any) -> Dict[str, torch.Tensor]:
    """
    Official SurgeNetDINO example loads the URL directly with strict=False.
    This function just handles common checkpoint wrappers/prefixes safely.
    """
    if isinstance(obj, dict):
        for key in ["state_dict", "model", "teacher", "student"]:
            if key in obj and isinstance(obj[key], dict):
                obj = obj[key]
                break

    if not isinstance(obj, dict):
        raise TypeError(f"Unsupported checkpoint type: {type(obj)}")

    cleaned = {}
    for k, v in obj.items():
        if not isinstance(v, torch.Tensor):
            continue
        nk = k
        for prefix in ["module.", "backbone.", "model."]:
            if nk.startswith(prefix):
                nk = nk[len(prefix):]
        cleaned[nk] = v
    return cleaned


def load_surgenetdino_model(variant: str, local_weights: str, device: torch.device):
    import timm

    spec = SURGENETDINO_SPECS[variant]

    model = timm.create_model(
        spec["timm_name"],
        img_size=(spec["img_size"], spec["img_size"]),
        patch_size=spec["patch_size"],
        num_classes=0,     # feature extractor; no classifier head needed
    )

    if local_weights:
        print(f"[INFO] loading local SurgeNetDINO weights: {local_weights}")
        ckpt = torch.load(local_weights, map_location="cpu")
    else:
        print(f"[INFO] downloading/loading official SurgeNetDINO weights: {variant}")
        ckpt = torch.hub.load_state_dict_from_url(
            spec["url"],
            map_location="cpu",
            check_hash=False,
        )

    state_dict = clean_state_dict(ckpt)
    msg = model.load_state_dict(state_dict, strict=False)
    print("[INFO] load_state_dict msg:", msg)

    model.to(device)
    model.eval()
    return model, spec


def to_label_tensor(label: Any) -> torch.Tensor:
    if isinstance(label, torch.Tensor):
        return label.detach().cpu().float()
    return torch.tensor(label, dtype=torch.float32)


def get_dataset_entries(dataset: Any) -> List[Tuple[str, torch.Tensor, str]]:
    """
    Your Cholec datasets store:
        dataset.data = [[impath, label, video_id], ...]
    This also works for the label-file CholecDataset path.
    """
    if not hasattr(dataset, "data"):
        raise AttributeError(f"{type(dataset)} has no .data field.")

    entries: List[Tuple[str, torch.Tensor, str]] = []
    for item in dataset.data:
        if len(item) < 2:
            continue
        impath = str(item[0])
        label = to_label_tensor(item[1])
        vid = str(item[2]) if len(item) >= 3 else ""
        if os.path.exists(impath):
            entries.append((impath, label, vid))
    return entries


def reservoir_sample_one_per_class(
    entries: Sequence[Tuple[str, torch.Tensor, str]],
    num_classes: int,
    classnames: Sequence[str],
    seed: int,
):
    """
    Uniformly sample one positive image per class using reservoir sampling.
    A multi-label image can be selected for multiple classes.
    """
    rng = random.Random(seed)
    selected: List[Any] = [None for _ in range(num_classes)]
    counts = [0 for _ in range(num_classes)]

    for impath, label, vid in entries:
        label = label[:num_classes]
        pos_classes = torch.where(label > 0.5)[0].tolist()

        for c in pos_classes:
            if c < 0 or c >= num_classes:
                continue
            counts[c] += 1
            if rng.random() < 1.0 / counts[c]:
                cname = classnames[c] if c < len(classnames) else f"class_{c}"
                selected[c] = {
                    "class_id": int(c),
                    "class_name": cname,
                    "impath": impath,
                    "video_id": vid,
                }

    missing = [i for i, x in enumerate(selected) if x is None]
    if missing:
        detail = [
            f"{i}:{classnames[i] if i < len(classnames) else 'UNKNOWN'}"
            for i in missing
        ]
        raise RuntimeError(
            "Some classes have no positive image in selected split: "
            + ", ".join(detail)
            + "\nTry --split all, or check cfg.num_classes / cholec_labels.txt / dataset labels."
        )

    return selected, counts


def load_selected_images(selected: Sequence[Dict[str, Any]], preprocess) -> torch.Tensor:
    imgs = []
    for item in selected:
        img = Image.open(item["impath"]).convert("RGB")
        imgs.append(preprocess(img))
    return torch.stack(imgs, dim=0)


def unwrap_timm_forward_features(out: Any) -> torch.Tensor:
    """
    Convert timm ViT forward_features output to [B, D].
    Handles tensor/dict/tuple outputs across timm versions.
    """
    if isinstance(out, dict):
        for key in ["x_norm_clstoken", "cls_token", "pooled", "features"]:
            if key in out and isinstance(out[key], torch.Tensor):
                feat = out[key]
                if feat.ndim == 3:
                    feat = feat[:, 0]
                return feat

        # fallback: first tensor value
        for value in out.values():
            if isinstance(value, torch.Tensor):
                feat = value
                if feat.ndim == 3:
                    feat = feat[:, 0]
                return feat

        raise RuntimeError(f"Cannot find tensor feature in forward_features dict keys={list(out.keys())}")

    if isinstance(out, (tuple, list)):
        # Prefer the last tensor-like output.
        tensors = [x for x in out if isinstance(x, torch.Tensor)]
        if not tensors:
            raise RuntimeError("forward_features returned tuple/list with no tensor.")
        feat = tensors[-1]
        if feat.ndim == 3:
            feat = feat[:, 0]
        return feat

    if isinstance(out, torch.Tensor):
        if out.ndim == 3:
            return out[:, 0]
        if out.ndim == 2:
            return out

    raise RuntimeError(f"Unsupported forward_features output type/shape: {type(out)}")


@torch.no_grad()
def extract_features(model: torch.nn.Module, images: torch.Tensor) -> torch.Tensor:
    out = model.forward_features(images)
    feat = unwrap_timm_forward_features(out)
    feat = feat.float()
    feat = F.normalize(feat, dim=-1)
    return feat


def main():
    args = parse_args()
    seed_everything(args.seed)

    spec = SURGENETDINO_SPECS[args.variant]
    preprocess = build_preprocess(spec["img_size"])

    reset_argv_for_mml1209(args.config_file, args.round)

    from config import cfg
    from dataset import build_dataset

    # Important for Cholec_Train:
    # if cfg.getitem=False, read_data() may random-pick one positive label and discard other positives.
    # We force getitem=True before building dataset so dataset.data keeps multi-label positives.
    if not args.no_preserve_multilabel and hasattr(cfg, "getitem"):
        cfg.getitem = True

    loaders = build_dataset(
        train_preprocess=preprocess,
        val_preprocess=preprocess,
        distributed=False,
    )

    train_loader = loaders[0]
    val_loader = loaders[1]
    if getattr(cfg, "val_sp", False):
        test_loader = loaders[3]
    else:
        test_loader = loaders[2]

    split_to_datasets = {
        "train": [train_loader.dataset],
        "val": [val_loader.dataset],
        "test": [test_loader.dataset],
        "all": [train_loader.dataset, val_loader.dataset, test_loader.dataset],
    }
    datasets = split_to_datasets[args.split]

    raw_classnames = list(train_loader.dataset.labels())
    # labels file may end with a trailing empty line; keep order but remove only extra tail.
    while raw_classnames and raw_classnames[-1] == "":
        raw_classnames.pop()

    cfg_num_classes = int(getattr(cfg, "num_classes", args.num_classes))
    num_classes = int(args.num_classes or cfg_num_classes)

    if num_classes != cfg_num_classes:
        print(f"[WARN] --num-classes={num_classes}, but cfg.num_classes={cfg_num_classes}")

    classnames = raw_classnames[:num_classes]
    if len(classnames) < num_classes:
        classnames += [f"class_{i}" for i in range(len(classnames), num_classes)]

    entries: List[Tuple[str, torch.Tensor, str]] = []
    for ds in datasets:
        entries.extend(get_dataset_entries(ds))

    print(f"[INFO] variant={args.variant}")
    print(f"[INFO] split={args.split}")
    print(f"[INFO] num_classes={num_classes}")
    print(f"[INFO] valid image entries={len(entries)}")

    selected, counts = reservoir_sample_one_per_class(
        entries=entries,
        num_classes=num_classes,
        classnames=classnames,
        seed=args.seed,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_surgenetdino_model(args.variant, args.weights, device)

    print("[INFO] loading selected images...")
    selected_images = load_selected_images(selected, preprocess)

    print("[INFO] extracting features...")
    feats = []
    for start in range(0, selected_images.size(0), args.batch_size):
        batch = selected_images[start:start + args.batch_size].to(device, non_blocking=True)
        feat = extract_features(model, batch)
        feats.append(feat.cpu())

    feats = torch.cat(feats, dim=0)
    feats = F.normalize(feats, dim=-1)

    matrix = feats @ feats.T
    matrix = matrix.clamp(-1.0, 1.0).numpy().astype(np.float32)
    matrix = 0.5 * (matrix + matrix.T)
    np.fill_diagonal(matrix, 1.0)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, matrix)

    meta_path = Path(str(out_path.with_suffix("")) + "_meta.json")
    meta = {
        "matrix_path": str(out_path),
        "shape": list(matrix.shape),
        "variant": args.variant,
        "timm_name": spec["timm_name"],
        "official_url": spec["url"] if not args.weights else "",
        "local_weights": args.weights,
        "seed": args.seed,
        "split": args.split,
        "num_classes": num_classes,
        "cfg_num_classes": cfg_num_classes,
        "classnames": classnames,
        "positive_counts": counts,
        "selected": selected,
        "note": "Each row/column follows class index order from cholec/cholec_labels.txt and cfg.num_classes.",
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    if args.save_csv:
        csv_path = out_path.with_suffix(".csv")
        np.savetxt(csv_path, matrix, delimiter=",", fmt="%.6f")
        print(f"[DONE] csv saved: {csv_path}")

    print(f"[DONE] matrix saved: {out_path}")
    print(f"[DONE] meta saved: {meta_path}")
    print(f"[INFO] matrix shape={matrix.shape}, min={matrix.min():.4f}, max={matrix.max():.4f}, mean={matrix.mean():.4f}")


if __name__ == "__main__":
    main()
