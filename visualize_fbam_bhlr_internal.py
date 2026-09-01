#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
00_rgb_original_gt.jpg：原始 RGB + UAV GT 框
01_tir_original_gt.jpg：原始 TIR + UAV GT 框
02_rgb_semantic_gt.jpg：640 RGB 输入 + UAV
03_tir_letterbox_gt.jpg：640 TIR 输入 + UAV
04_rgb_p3_activation.jpg：RGB P3 激活
05_tir_p3_raw_activation.jpg：对齐前 TIR P3
06_rgb_frequency_boundary.jpg：RGB 频率边界
07_tir_frequency_boundary.jpg：TIR 频率边界
08_coarse_offset_magnitude.jpg：粗对齐偏移
09_fine_offset_magnitude.jpg：细对齐偏移
10_total_offset_magnitude.jpg：最终 Offset
11_alignment_confidence.jpg：对齐置信度
12_tir_p3_aligned_activation.jpg：对齐后的 TIR P3
13_alignment_difference_before.jpg：对齐前 RGB/TIR 特征差
14_alignment_difference_after.jpg：对齐后 RGB/TIR 特征差
15_tir_image_warp_preview.jpg：根据 P3 offset 得到的 TIR 对齐可视化
16_rgb_high_gt.jpg：1280 高分辨率 RGB + UAV
17_reconstructed_high.jpg：640 RGB 上采样重建结果
18_highres_residual_heatmap.jpg：真正的高分辨率丢失信息
19_highres_residual_overlay.jpg：丢失细节叠加到高分辨率 RGB
20_detail_bank_activation.jpg：Shared Lost-detail Bank
21_support_map.jpg：BHLR 的 UAV soft support map
22_support_overlay.jpg：模型认为应该调用高分辨率信息的关键区域，同时画出真实 UAV 框
23_detail_p3_activation.jpg：压缩到 P3 的高分辨率丢失特征
24_detail_gate.jpg：Useful Detail Gate
25_usable_detail_activation.jpg：最终真正准备注入 RGB 的细节
26_rgb_p3_before_bhlr.jpg：增强前 RGB P3
27_rgb_p3_after_bhlr.jpg：增强后 RGB P3
28_bhlr_enhancement_magnitude.jpg：BHLR 到底修改了哪些位置
29_prediction_vs_gt.jpg：最终预测框与 GT 对比
overview.jpg：把上述关键步骤拼成一张总览图
metadata.json：Offset、confidence、support、gate、gamma 等数值

Notes
conda activate tcy
cd /mnt/sda/taochangyong/Projects/Model/YOLO-tcy
python visualize_fbam_bhlr_internal.py \
  --weights /mnt/sda/taochangyong/Projects/Model/YOLO-tcy/runs/fbam_bhlr/yolo26n_UAVCB_fbamP3_bhlrP3_rgb1280_sem640_tir640_seed0_V2/weights/best.pt \
  --split test \
  --device 5 \
  --seed 0 \
  --index 0 100 200 300 400 500 600
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------

import val_rgbt as base_val

from datasets.rsdt_dataset import (
    build_rsdt_dataset,
)

from datasets.rgbt_dataset import (
    load_yolo_label,
)

from train_rsdt import (
    preprocess_batch_rsdt,
)

from val_fbam_bhlr import (
    load_fbam_bhlr_checkpoint,
)


# ===========================================================================
# Basic image helpers
# ===========================================================================

def tensor_rgb_to_bgr(x: torch.Tensor) -> np.ndarray:
    """[3,H,W], [0,1] or [0,255] -> uint8 BGR."""
    x = x.detach().float().cpu()

    if x.ndim == 4:
        x = x[0]

    x = x.clamp(0, 1) if x.max() <= 1.5 else x.clamp(0, 255) / 255.0

    arr = (
        x.permute(1, 2, 0)
        .numpy()
        * 255.0
    ).round().astype(np.uint8)

    if arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)

    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def read_rgb_bgr(path: str) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return image


def read_tir_bgr(path: str) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


def ensure_bgr(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 1:
        return cv2.cvtColor(img[..., 0], cv2.COLOR_GRAY2BGR)
    return img


def normalize_map(
    x: np.ndarray,
    robust: bool = True,
) -> np.ndarray:
    """Return float32 map in [0,1]."""
    x = np.asarray(x, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    if robust:
        lo = float(np.percentile(x, 1.0))
        hi = float(np.percentile(x, 99.0))
    else:
        lo = float(x.min())
        hi = float(x.max())

    if hi <= lo + 1e-12:
        return np.zeros_like(x, dtype=np.float32)

    y = (x - lo) / (hi - lo)
    return np.clip(y, 0.0, 1.0)


def tensor_feature_map(
    x: torch.Tensor,
    mode: str = "mean_abs",
) -> np.ndarray:
    """
    [B,C,H,W] or [C,H,W] -> [H,W].
    """
    x = x.detach().float().cpu()

    if x.ndim == 4:
        x = x[0]

    if x.ndim == 2:
        return x.numpy()

    if mode == "mean":
        y = x.mean(dim=0)
    elif mode == "max_abs":
        y = x.abs().amax(dim=0)
    elif mode == "l2":
        y = torch.sqrt(
            torch.clamp(
                (x * x).mean(dim=0),
                min=0.0,
            )
        )
    else:
        y = x.abs().mean(dim=0)

    return y.numpy()


def heatmap_bgr(
    fmap: np.ndarray,
    size: Optional[Tuple[int, int]] = None,
    robust: bool = True,
) -> np.ndarray:
    """
    size: (width, height)
    """
    norm = normalize_map(fmap, robust=robust)
    img = (norm * 255.0).astype(np.uint8)

    if size is not None:
        img = cv2.resize(
            img,
            size,
            interpolation=cv2.INTER_LINEAR,
        )

    return cv2.applyColorMap(
        img,
        cv2.COLORMAP_TURBO,
    )


def overlay_heatmap(
    base: np.ndarray,
    fmap: np.ndarray,
    alpha: float = 0.45,
    robust: bool = True,
) -> np.ndarray:
    base = ensure_bgr(base).copy()
    h, w = base.shape[:2]

    hm = heatmap_bgr(
        fmap,
        size=(w, h),
        robust=robust,
    )

    return cv2.addWeighted(
        base,
        1.0 - alpha,
        hm,
        alpha,
        0.0,
    )


def add_title(
    img: np.ndarray,
    title: str,
    subtitle: Optional[str] = None,
    bar_h: int = 72,
) -> np.ndarray:
    img = ensure_bgr(img)

    canvas = np.full(
        (img.shape[0] + bar_h, img.shape[1], 3),
        245,
        dtype=np.uint8,
    )
    canvas[bar_h:] = img

    cv2.putText(
        canvas,
        title,
        (16, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )

    if subtitle:
        cv2.putText(
            canvas,
            subtitle,
            (16, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (70, 70, 70),
            1,
            cv2.LINE_AA,
        )

    return canvas


# ===========================================================================
# Box helpers
# ===========================================================================

def normalized_xywh_to_xyxy(
    boxes: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    if boxes is None or len(boxes) == 0:
        return np.zeros((0, 4), dtype=np.float32)

    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)

    cx = boxes[:, 0] * width
    cy = boxes[:, 1] * height
    bw = boxes[:, 2] * width
    bh = boxes[:, 3] * height

    out = np.stack(
        [
            cx - bw / 2,
            cy - bh / 2,
            cx + bw / 2,
            cy + bh / 2,
        ],
        axis=1,
    )

    out[:, [0, 2]] = np.clip(out[:, [0, 2]], 0, width - 1)
    out[:, [1, 3]] = np.clip(out[:, [1, 3]], 0, height - 1)

    return out


def draw_boxes(
    image: np.ndarray,
    boxes_xyxy: np.ndarray,
    label: str = "UAV GT",
    color: Tuple[int, int, int] = (0, 0, 255),
    thickness: int = 2,
) -> np.ndarray:
    out = ensure_bgr(image).copy()

    for box in boxes_xyxy:
        x1, y1, x2, y2 = [int(round(v)) for v in box]

        cv2.rectangle(
            out,
            (x1, y1),
            (x2, y2),
            color,
            thickness,
        )

        text_size = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            1,
        )[0]

        ty = max(y1 - 6, text_size[1] + 4)

        cv2.rectangle(
            out,
            (x1, ty - text_size[1] - 5),
            (x1 + text_size[0] + 6, ty + 3),
            color,
            -1,
        )

        cv2.putText(
            out,
            label,
            (x1 + 3, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return out


def draw_prediction_boxes(
    image: np.ndarray,
    pred: Dict[str, torch.Tensor],
    color: Tuple[int, int, int] = (0, 255, 0),
    conf_min: float = 0.25,
) -> np.ndarray:
    out = ensure_bgr(image).copy()

    if pred is None or pred.get("bboxes") is None:
        return out

    boxes = pred["bboxes"].detach().float().cpu().numpy()
    confs = pred["conf"].detach().float().cpu().numpy()

    for box, conf in zip(boxes, confs):
        if float(conf) < conf_min:
            continue

        x1, y1, x2, y2 = [int(round(v)) for v in box]
        label = f"Pred {float(conf):.2f}"

        cv2.rectangle(
            out,
            (x1, y1),
            (x2, y2),
            color,
            2,
        )

        cv2.putText(
            out,
            label,
            (x1, max(16, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            color,
            2,
            cv2.LINE_AA,
        )

    return out


def draw_support_region(
    image: np.ndarray,
    support_map: np.ndarray,
    gt_boxes: np.ndarray,
    support_thr: float = 0.5,
) -> np.ndarray:
    """
    Draw support heatmap + predicted high-response contours + GT boxes.
    """
    out = overlay_heatmap(
        image,
        support_map,
        alpha=0.40,
        robust=False,
    )

    h, w = image.shape[:2]

    support = cv2.resize(
        support_map.astype(np.float32),
        (w, h),
        interpolation=cv2.INTER_LINEAR,
    )

    # If 0.5 selects nothing, fall back to the top 10% response region.
    thr = float(support_thr)
    mask = (support >= thr).astype(np.uint8) * 255

    if mask.max() == 0:
        thr = float(np.percentile(support, 90.0))
        mask = (support >= thr).astype(np.uint8) * 255

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    cv2.drawContours(
        out,
        contours,
        -1,
        (255, 255, 0),
        2,
    )

    out = draw_boxes(
        out,
        gt_boxes,
        label="UAV GT",
        color=(0, 0, 255),
        thickness=2,
    )

    return out


# ===========================================================================
# Debug dictionary helpers
# ===========================================================================

def get_scale_item(
    mapping: Dict,
    scale: int,
):
    if scale in mapping:
        return mapping[scale]

    key = str(scale)
    if key in mapping:
        return mapping[key]

    raise KeyError(
        f"Scale P{scale} not found. Available keys: {list(mapping.keys())}"
    )


def get_alignment_scale_debug(
    features: Dict,
    scale: int,
) -> Dict:
    info = features.get("alignment_info", None)

    if not isinstance(info, dict):
        raise KeyError(
            "Model did not return alignment_info. "
            "Run with return_alignment_debug=True."
        )

    scale_debug = info.get(
        "scale_debug",
        info.get("scales", {}),
    )

    if not isinstance(scale_debug, dict):
        raise KeyError(
            f"Invalid alignment debug structure: {info.keys()}"
        )

    return get_scale_item(
        scale_debug,
        scale,
    )


def get_bhlr_scale_debug(
    features: Dict,
    scale: int,
) -> Tuple[Dict, Dict]:
    debug = features.get(
        "bhlr_debug",
        features.get("rsdt_debug", None),
    )

    if not isinstance(debug, dict):
        raise KeyError(
            "Model did not return BHLR debug. "
            "Run with return_rsdt_debug=True."
        )

    scales = debug.get("scales", {})

    if not isinstance(scales, dict):
        raise KeyError(
            f"Invalid BHLR debug structure: {debug.keys()}"
        )

    return debug, get_scale_item(
        scales,
        scale,
    )


# ===========================================================================
# Approximate input-level warp for visualization only
# ===========================================================================

@torch.no_grad()
def warp_tir_preview(
    tir_tensor: torch.Tensor,
    total_offset_p3: torch.Tensor,
    semantic_h: int,
    semantic_w: int,
) -> torch.Tensor:
    """
    FBAM alignment is performed on P3.

    We upsample the P3 offset field to image resolution and convert
    feature-pixel offsets to image-pixel offsets using the P3 stride.
    This is a VISUALIZATION ONLY.
    """
    if tir_tensor.ndim == 3:
        tir_tensor = tir_tensor.unsqueeze(0)

    offset = total_offset_p3

    if offset.ndim == 3:
        offset = offset.unsqueeze(0)

    h3, w3 = offset.shape[-2:]

    stride_x = float(semantic_w) / float(w3)
    stride_y = float(semantic_h) / float(h3)

    offset_img = F.interpolate(
        offset.float(),
        size=(semantic_h, semantic_w),
        mode="bilinear",
        align_corners=True,
    )

    offset_img = offset_img.clone()
    offset_img[:, 0] *= stride_x
    offset_img[:, 1] *= stride_y

    source = tir_tensor.float()
    b, _, h, w = source.shape

    ys = torch.linspace(
        -1.0,
        1.0,
        h,
        device=source.device,
        dtype=source.dtype,
    )
    xs = torch.linspace(
        -1.0,
        1.0,
        w,
        device=source.device,
        dtype=source.dtype,
    )

    yy, xx = torch.meshgrid(
        ys,
        xs,
        indexing="ij",
    )

    base_grid = (
        torch.stack((xx, yy), dim=-1)
        .unsqueeze(0)
        .expand(b, -1, -1, -1)
    )

    dx = (
        2.0
        * offset_img[:, 0]
        / max(w - 1, 1)
    )
    dy = (
        2.0
        * offset_img[:, 1]
        / max(h - 1, 1)
    )

    delta = torch.stack(
        (dx, dy),
        dim=-1,
    )

    warped = F.grid_sample(
        source,
        base_grid + delta,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )

    return warped


# ===========================================================================
# Save helpers
# ===========================================================================

def save_image(
    out_dir: Path,
    filename: str,
    image: np.ndarray,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
) -> Path:
    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    image = ensure_bgr(image)

    if title is not None:
        image = add_title(
            image,
            title,
            subtitle,
        )

    path = out_dir / filename

    ok = cv2.imwrite(
        str(path),
        image,
    )

    if not ok:
        raise RuntimeError(
            f"Failed to save: {path}"
        )

    return path


def resize_tile(
    image: np.ndarray,
    width: int = 480,
    height: int = 360,
) -> np.ndarray:
    image = ensure_bgr(image)

    h, w = image.shape[:2]

    scale = min(
        width / max(w, 1),
        height / max(h, 1),
    )

    nw = max(
        1,
        int(round(w * scale)),
    )
    nh = max(
        1,
        int(round(h * scale)),
    )

    resized = cv2.resize(
        image,
        (nw, nh),
        interpolation=cv2.INTER_AREA
        if scale < 1
        else cv2.INTER_LINEAR,
    )

    canvas = np.full(
        (height, width, 3),
        250,
        dtype=np.uint8,
    )

    x0 = (width - nw) // 2
    y0 = (height - nh) // 2

    canvas[
        y0:y0 + nh,
        x0:x0 + nw,
    ] = resized

    return canvas


def make_overview(
    panels: Sequence[Tuple[str, np.ndarray]],
    out_path: Path,
    cols: int = 4,
    tile_w: int = 480,
    tile_h: int = 390,
) -> None:
    rows = int(
        np.ceil(
            len(panels)
            / max(cols, 1)
        )
    )

    canvas = np.full(
        (
            rows * tile_h,
            cols * tile_w,
            3,
        ),
        255,
        dtype=np.uint8,
    )

    for i, (title, img) in enumerate(panels):
        row = i // cols
        col = i % cols

        tile = resize_tile(
            img,
            width=tile_w,
            height=tile_h - 48,
        )

        x0 = col * tile_w
        y0 = row * tile_h

        canvas[
            y0 + 48:y0 + tile_h,
            x0:x0 + tile_w,
        ] = tile

        cv2.putText(
            canvas,
            f"{i + 1:02d}. {title}",
            (x0 + 10, y0 + 31),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )

    cv2.imwrite(
        str(out_path),
        canvas,
    )


# ===========================================================================
# Positive sample selection
# ===========================================================================

def collect_positive_indices(
    dataset,
) -> List[int]:
    positive = []

    for i, item in enumerate(dataset.samples):
        cls, boxes = load_yolo_label(
            item["rgb_label"],
            nc=1,
            strict=True,
            target_classes=(0,),
            remap_classes=True,
        )

        if len(cls) > 0 and len(boxes) > 0:
            positive.append(i)

    return positive


# ===========================================================================
# Main visualization
# ===========================================================================

@torch.inference_mode()
def visualize(
    weights: str,
    split: str = "test",
    device_str: str = "0",
    rgb_yaml: Optional[str] = None,
    tir_yaml: Optional[str] = None,
    rgb_imgsz: Optional[int] = None,
    rgb_semantic_imgsz: Optional[int] = None,
    tir_imgsz: Optional[int] = None,
    pair_mode: Optional[str] = None,
    index: Optional[int] = None,
    seed: int = 0,
    align_scale: int = 3,
    bhlr_scale: int = 3,
    conf: float = 0.25,
    iou: float = 0.70,
    max_det: int = 300,
    support_thr: float = 0.50,
    save_dir: Optional[str] = None,
):
    # -----------------------------------------------------------------------
    # Device + checkpoint
    # -----------------------------------------------------------------------

    device = base_val.select_device(
        device_str
    )

    model, ckpt, cfg = (
        load_fbam_bhlr_checkpoint(
            weights,
            device,
        )
    )

    model.eval()

    data_cfg = cfg.get(
        "data",
        {},
    )

    if rgb_yaml is None:
        rgb_yaml = data_cfg.get("rgb")

    if tir_yaml is None:
        tir_yaml = data_cfg.get("tir")

    if rgb_imgsz is None:
        rgb_imgsz = int(
            data_cfg.get(
                "rgb_imgsz",
                data_cfg.get(
                    "rgb_high_imgsz",
                    1280,
                ),
            )
        )

    if rgb_semantic_imgsz is None:
        rgb_semantic_imgsz = int(
            data_cfg.get(
                "rgb_semantic_imgsz",
                640,
            )
        )

    if tir_imgsz is None:
        tir_imgsz = int(
            data_cfg.get(
                "tir_imgsz",
                640,
            )
        )

    if pair_mode is None:
        pair_mode = data_cfg.get(
            "pair_mode",
            "relative",
        )

    if rgb_yaml is None or tir_yaml is None:
        raise ValueError(
            "RGB/TIR dataset YAML cannot be resolved from checkpoint."
        )

    # -----------------------------------------------------------------------
    # Dataset
    # -----------------------------------------------------------------------

    dataset = build_rsdt_dataset(
        rgb_yaml=rgb_yaml,
        tir_yaml=tir_yaml,
        split=split,
        rgb_high_imgsz=rgb_imgsz,
        rgb_semantic_imgsz=rgb_semantic_imgsz,
        tir_imgsz=tir_imgsz,
        pair_mode=pair_mode,
        augment=False,
        tir_channels=3,
        strict_pair=True,
    )

    if len(dataset) == 0:
        raise RuntimeError(
            f"No samples in split={split}"
        )

    # -----------------------------------------------------------------------
    # Random positive sample
    # -----------------------------------------------------------------------

    rng = random.Random(seed)

    if index is None:
        positive_indices = (
            collect_positive_indices(
                dataset
            )
        )

        if not positive_indices:
            raise RuntimeError(
                f"No UAV-positive samples found in split={split}."
            )

        index = rng.choice(
            positive_indices
        )

    if index < 0 or index >= len(dataset):
        raise IndexError(
            f"index={index}, dataset length={len(dataset)}"
        )

    sample = dataset[index]

    # Build a batch of size 1 using the dataset's own collate_fn.
    batch = dataset.collate_fn(
        [sample]
    )

    batch = preprocess_batch_rsdt(
        batch,
        device,
    )

    # -----------------------------------------------------------------------
    # Full debug forward
    # -----------------------------------------------------------------------

    outputs = model(
        batch["rgb_img"],
        batch["tir_img"],
        batch["rgb_semantic_img"],
        return_features=True,
        return_rsdt_debug=True,
        return_alignment_debug=True,
    )

    if not isinstance(outputs, dict):
        raise TypeError(
            "Expected a debug dict from model(... return_features=True ...)"
        )

    preds_raw = outputs["pred"]

    # -----------------------------------------------------------------------
    # Debug tensors
    # -----------------------------------------------------------------------

    align_dbg = (
        get_alignment_scale_debug(
            outputs,
            align_scale,
        )
    )

    bhlr_debug_all, bhlr_dbg = (
        get_bhlr_scale_debug(
            outputs,
            bhlr_scale,
        )
    )

    scale_to_index = (
        outputs.get(
            "rsdt_scale_to_index",
            getattr(
                model,
                "rsdt_scale_to_index",
                {},
            ),
        )
    )

    p3_idx = int(
        scale_to_index.get(
            align_scale,
            scale_to_index.get(
                str(align_scale),
                -1,
            ),
        )
    )

    if p3_idx < 0:
        raise KeyError(
            f"Cannot resolve P{align_scale} backbone index."
        )

    raw_tir_p3 = (
        outputs["tir_backbone"][
            p3_idx
        ]
    )

    aligned_tir_p3 = (
        outputs["aligned_tir_backbone"][
            p3_idx
        ]
    )

    # BHLR result dictionary stores enhanced RGB after replacement.
    enhanced_rgb_p3 = (
        bhlr_dbg["enhanced_rgb"]
    )

    usable_detail = (
        bhlr_dbg["usable_detail"]
    )

    gamma = bhlr_dbg["gamma"]

    # Recover the RGB feature BEFORE BHLR:
    # enhanced = raw + gamma * usable_detail
    raw_rgb_p3 = (
        enhanced_rgb_p3
        - gamma * usable_detail
    )

    # -----------------------------------------------------------------------
    # Images + GT
    # -----------------------------------------------------------------------

    rgb_original = read_rgb_bgr(
        sample["rgb_path"]
    )

    tir_original = read_tir_bgr(
        sample["tir_path"]
    )

    rgb_semantic = tensor_rgb_to_bgr(
        batch["rgb_semantic_img"][0]
    )

    tir_letterbox = tensor_rgb_to_bgr(
        batch["tir_img"][0]
    )

    rgb_high = tensor_rgb_to_bgr(
        batch["rgb_img"][0]
    )

    # Original RGB labels
    _, rgb_boxes_original_norm = (
        load_yolo_label(
            sample["rgb_label_path"],
            nc=1,
            strict=True,
            target_classes=(0,),
            remap_classes=True,
        )
    )

    rgb_boxes_original = (
        normalized_xywh_to_xyxy(
            rgb_boxes_original_norm,
            rgb_original.shape[1],
            rgb_original.shape[0],
        )
    )

    # Original TIR labels
    _, tir_boxes_original_norm = (
        load_yolo_label(
            sample["tir_label_path"],
            nc=1,
            strict=True,
            target_classes=(0,),
            remap_classes=True,
        )
    )

    tir_boxes_original = (
        normalized_xywh_to_xyxy(
            tir_boxes_original_norm,
            tir_original.shape[1],
            tir_original.shape[0],
        )
    )

    # Semantic RGB GT
    semantic_boxes_norm = (
        sample[
            "rgb_semantic_bboxes"
        ]
        .detach()
        .float()
        .cpu()
        .numpy()
    )

    semantic_boxes = (
        normalized_xywh_to_xyxy(
            semantic_boxes_norm,
            rgb_semantic.shape[1],
            rgb_semantic.shape[0],
        )
    )

    # High RGB GT
    high_boxes_norm = (
        sample["rgb_bboxes"]
        .detach()
        .float()
        .cpu()
        .numpy()
    )

    high_boxes = (
        normalized_xywh_to_xyxy(
            high_boxes_norm,
            rgb_high.shape[1],
            rgb_high.shape[0],
        )
    )

    # TIR LetterBox GT
    tir_boxes_norm = (
        sample["tir_bboxes"]
        .detach()
        .float()
        .cpu()
        .numpy()
    )

    tir_boxes_letterbox = (
        normalized_xywh_to_xyxy(
            tir_boxes_norm,
            tir_letterbox.shape[1],
            tir_letterbox.shape[0],
        )
    )

    # -----------------------------------------------------------------------
    # Save directory
    # -----------------------------------------------------------------------

    weights_path = (
        Path(weights)
        .expanduser()
        .resolve()
    )

    stem = Path(
        sample["rgb_path"]
    ).stem

    if save_dir is None:
        save_dir_path = (
            weights_path
            .parent
            .parent
            / "visualize_internal"
            / f"{split}_{index:06d}_{stem}"
        )
    else:
        save_dir_path = (
            Path(save_dir)
            .expanduser()
            .resolve()
        )

    save_dir_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    panels: List[
        Tuple[str, np.ndarray]
    ] = []

    # -----------------------------------------------------------------------
    # 00-03 Inputs + GT
    # -----------------------------------------------------------------------

    img = draw_boxes(
        rgb_original,
        rgb_boxes_original,
        "UAV GT",
    )
    save_image(
        save_dir_path,
        "00_rgb_original_gt.jpg",
        img,
        "Original RGB + GT",
        str(sample["rgb_path"]),
    )
    panels.append(("Original RGB + GT", img))

    img = draw_boxes(
        tir_original,
        tir_boxes_original,
        "UAV GT",
    )
    save_image(
        save_dir_path,
        "01_tir_original_gt.jpg",
        img,
        "Original TIR + GT",
        str(sample["tir_path"]),
    )
    panels.append(("Original TIR + GT", img))

    img = draw_boxes(
        rgb_semantic,
        semantic_boxes,
        "UAV GT",
    )
    save_image(
        save_dir_path,
        "02_rgb_semantic_gt.jpg",
        img,
        "RGB semantic input + GT",
        f"{rgb_semantic_imgsz}x{rgb_semantic_imgsz}",
    )
    panels.append(("RGB semantic + GT", img))

    img = draw_boxes(
        tir_letterbox,
        tir_boxes_letterbox,
        "UAV GT",
    )
    save_image(
        save_dir_path,
        "03_tir_letterbox_gt.jpg",
        img,
        "TIR network input + GT",
        f"{tir_imgsz}x{tir_imgsz}",
    )
    panels.append(("TIR input + GT", img))

    # -----------------------------------------------------------------------
    # 04-15 FBAM
    # -----------------------------------------------------------------------

    rgb_p3_map = tensor_feature_map(
        raw_rgb_p3,
        "mean_abs",
    )

    img = overlay_heatmap(
        rgb_semantic,
        rgb_p3_map,
        alpha=0.50,
    )
    img = draw_boxes(
        img,
        semantic_boxes,
        "UAV GT",
    )
    save_image(
        save_dir_path,
        "04_rgb_p3_activation.jpg",
        img,
        f"RGB P{align_scale} activation",
        str(tuple(raw_rgb_p3.shape)),
    )
    panels.append(("RGB P3 activation", img))

    tir_p3_raw_map = tensor_feature_map(
        raw_tir_p3,
        "mean_abs",
    )

    img = overlay_heatmap(
        tir_letterbox,
        tir_p3_raw_map,
        alpha=0.50,
    )
    save_image(
        save_dir_path,
        "05_tir_p3_raw_activation.jpg",
        img,
        f"Raw TIR P{align_scale} activation",
        str(tuple(raw_tir_p3.shape)),
    )
    panels.append(("Raw TIR P3", img))

    rgb_boundary_map = tensor_feature_map(
        align_dbg["rgb_boundary"],
        "mean_abs",
    )

    img = overlay_heatmap(
        rgb_semantic,
        rgb_boundary_map,
        alpha=0.58,
    )
    img = draw_boxes(
        img,
        semantic_boxes,
        "UAV GT",
    )
    save_image(
        save_dir_path,
        "06_rgb_frequency_boundary.jpg",
        img,
        "FBAM RGB frequency-boundary response",
        str(tuple(align_dbg["rgb_boundary"].shape)),
    )
    panels.append(("RGB frequency boundary", img))

    tir_boundary_map = tensor_feature_map(
        align_dbg["tir_boundary"],
        "mean_abs",
    )

    img = overlay_heatmap(
        tir_letterbox,
        tir_boundary_map,
        alpha=0.58,
    )
    save_image(
        save_dir_path,
        "07_tir_frequency_boundary.jpg",
        img,
        "FBAM TIR frequency-boundary response",
        str(tuple(align_dbg["tir_boundary"].shape)),
    )
    panels.append(("TIR frequency boundary", img))

    coarse = (
        align_dbg["coarse_offset"]
        .detach()
        .float()
    )

    coarse_mag = torch.sqrt(
        coarse[:, 0].square()
        + coarse[:, 1].square()
    )[0].cpu().numpy()

    img = overlay_heatmap(
        rgb_semantic,
        coarse_mag,
        alpha=0.58,
    )
    img = draw_boxes(
        img,
        semantic_boxes,
        "UAV GT",
    )
    save_image(
        save_dir_path,
        "08_coarse_offset_magnitude.jpg",
        img,
        "FBAM coarse offset magnitude",
        "unit = P3 feature pixels",
    )
    panels.append(("Coarse offset", img))

    fine = (
        align_dbg["fine_offset"]
        .detach()
        .float()
    )

    fine_mag = torch.sqrt(
        fine[:, 0].square()
        + fine[:, 1].square()
    )[0].cpu().numpy()

    img = overlay_heatmap(
        rgb_semantic,
        fine_mag,
        alpha=0.58,
    )
    img = draw_boxes(
        img,
        semantic_boxes,
        "UAV GT",
    )
    save_image(
        save_dir_path,
        "09_fine_offset_magnitude.jpg",
        img,
        "FBAM fine offset magnitude",
        "frequency-boundary-guided refinement",
    )
    panels.append(("Fine offset", img))

    total = (
        align_dbg["total_offset"]
        .detach()
        .float()
    )

    total_mag = torch.sqrt(
        total[:, 0].square()
        + total[:, 1].square()
    )[0].cpu().numpy()

    img = overlay_heatmap(
        rgb_semantic,
        total_mag,
        alpha=0.58,
    )
    img = draw_boxes(
        img,
        semantic_boxes,
        "UAV GT",
    )
    save_image(
        save_dir_path,
        "10_total_offset_magnitude.jpg",
        img,
        "FBAM total offset magnitude",
        "coarse + fine",
    )
    panels.append(("Total offset", img))

    confidence_map = (
        align_dbg["confidence"]
        .detach()
        .float()
        .cpu()
        .numpy()[0, 0]
    )

    img = overlay_heatmap(
        rgb_semantic,
        confidence_map,
        alpha=0.52,
        robust=False,
    )
    img = draw_boxes(
        img,
        semantic_boxes,
        "UAV GT",
    )
    save_image(
        save_dir_path,
        "11_alignment_confidence.jpg",
        img,
        "FBAM alignment confidence",
        "0 = low confidence, 1 = high confidence",
    )
    panels.append(("Alignment confidence", img))

    tir_p3_aligned_map = (
        tensor_feature_map(
            aligned_tir_p3,
            "mean_abs",
        )
    )

    img = overlay_heatmap(
        rgb_semantic,
        tir_p3_aligned_map,
        alpha=0.50,
    )
    img = draw_boxes(
        img,
        semantic_boxes,
        "UAV GT",
    )
    save_image(
        save_dir_path,
        "12_tir_p3_aligned_activation.jpg",
        img,
        f"Aligned TIR P{align_scale} activation",
        "displayed in RGB-semantic coordinates",
    )
    panels.append(("Aligned TIR P3", img))

    # Difference before / after alignment relative to RGB feature.
    rgb_for_diff = raw_rgb_p3.detach().float()

    tir_before = raw_tir_p3.detach().float()
    tir_after = aligned_tir_p3.detach().float()

    # Same channels expected in current P3 implementation.
    diff_before = (
        (rgb_for_diff - tir_before)
        .abs()
        .mean(dim=1)[0]
        .cpu()
        .numpy()
    )

    diff_after = (
        (rgb_for_diff - tir_after)
        .abs()
        .mean(dim=1)[0]
        .cpu()
        .numpy()
    )

    img = overlay_heatmap(
        rgb_semantic,
        diff_before,
        alpha=0.58,
    )
    img = draw_boxes(
        img,
        semantic_boxes,
        "UAV GT",
    )
    save_image(
        save_dir_path,
        "13_alignment_difference_before.jpg",
        img,
        "RGB-TIR feature difference BEFORE FBAM",
        "mean absolute channel difference",
    )
    panels.append(("Difference before", img))

    img = overlay_heatmap(
        rgb_semantic,
        diff_after,
        alpha=0.58,
    )
    img = draw_boxes(
        img,
        semantic_boxes,
        "UAV GT",
    )
    save_image(
        save_dir_path,
        "14_alignment_difference_after.jpg",
        img,
        "RGB-TIR feature difference AFTER FBAM",
        "mean absolute channel difference",
    )
    panels.append(("Difference after", img))

    warped_tir = warp_tir_preview(
        batch["tir_img"],
        align_dbg["total_offset"],
        semantic_h=rgb_semantic.shape[0],
        semantic_w=rgb_semantic.shape[1],
    )

    warped_tir_bgr = tensor_rgb_to_bgr(
        warped_tir[0]
    )

    warped_tir_bgr = draw_boxes(
        warped_tir_bgr,
        semantic_boxes,
        "RGB UAV GT",
    )

    save_image(
        save_dir_path,
        "15_tir_image_warp_preview.jpg",
        warped_tir_bgr,
        "Approximate image-level TIR warp",
        "P3 offsets upsampled to 640; visualization only",
    )
    panels.append(("Approx. warped TIR", warped_tir_bgr))

    # -----------------------------------------------------------------------
    # 16-28 BHLR
    # -----------------------------------------------------------------------

    img = draw_boxes(
        rgb_high,
        high_boxes,
        "UAV GT",
    )
    save_image(
        save_dir_path,
        "16_rgb_high_gt.jpg",
        img,
        "RGB-high + GT",
        f"{rgb_imgsz}x{rgb_imgsz}",
    )
    panels.append(("RGB-high + GT", img))

    reconstructed_high = (
        bhlr_debug_all[
            "reconstructed_high"
        ]
    )

    reconstructed_bgr = (
        tensor_rgb_to_bgr(
            reconstructed_high[0]
        )
    )

    reconstructed_bgr = draw_boxes(
        reconstructed_bgr,
        high_boxes,
        "UAV GT",
    )

    save_image(
        save_dir_path,
        "17_reconstructed_high.jpg",
        reconstructed_bgr,
        "Reconstructed high RGB",
        "Up(RGB-semantic)",
    )
    panels.append(("Reconstructed high", reconstructed_bgr))

    residual_high = (
        bhlr_debug_all[
            "resolution_residual"
        ]
        .detach()
        .float()
    )

    residual_map = (
        residual_high[0]
        .abs()
        .mean(dim=0)
        .cpu()
        .numpy()
    )

    residual_hm = heatmap_bgr(
        residual_map,
        size=(
            rgb_high.shape[1],
            rgb_high.shape[0],
        ),
    )

    residual_hm = draw_boxes(
        residual_hm,
        high_boxes,
        "UAV GT",
    )

    save_image(
        save_dir_path,
        "18_highres_residual_heatmap.jpg",
        residual_hm,
        "High-resolution lost-detail residual",
        "|RGB-high - Up(RGB-semantic)|",
    )
    panels.append(("High-res residual", residual_hm))

    residual_overlay = overlay_heatmap(
        rgb_high,
        residual_map,
        alpha=0.55,
    )

    residual_overlay = draw_boxes(
        residual_overlay,
        high_boxes,
        "UAV GT",
    )

    save_image(
        save_dir_path,
        "19_highres_residual_overlay.jpg",
        residual_overlay,
        "Lost-detail residual over RGB-high",
        "bright regions = information not reconstructed from 640 RGB",
    )
    panels.append(("Residual overlay", residual_overlay))

    detail_raw_map = tensor_feature_map(
        bhlr_debug_all["detail_raw"],
        "mean_abs",
    )

    detail_bank_img = overlay_heatmap(
        rgb_semantic,
        detail_raw_map,
        alpha=0.55,
    )

    detail_bank_img = draw_boxes(
        detail_bank_img,
        semantic_boxes,
        "UAV GT",
    )

    save_image(
        save_dir_path,
        "20_detail_bank_activation.jpg",
        detail_bank_img,
        "Shared Lost-detail Bank",
        str(tuple(bhlr_debug_all["detail_raw"].shape)),
    )
    panels.append(("Detail Bank", detail_bank_img))

    support_map = (
        bhlr_dbg["support_map"]
        .detach()
        .float()
        .cpu()
        .numpy()[0, 0]
    )

    img = overlay_heatmap(
        rgb_semantic,
        support_map,
        alpha=0.52,
        robust=False,
    )

    img = draw_boxes(
        img,
        semantic_boxes,
        "UAV GT",
    )

    save_image(
        save_dir_path,
        "21_support_map.jpg",
        img,
        f"BHLR P{bhlr_scale} soft UAV support map",
        "boundary + RGB/TIR semantics",
    )
    panels.append(("Soft UAV support", img))

    support_overlay = draw_support_region(
        rgb_semantic,
        support_map,
        semantic_boxes,
        support_thr=support_thr,
    )

    save_image(
        save_dir_path,
        "22_support_overlay.jpg",
        support_overlay,
        "Key high-resolution retrieval region",
        "cyan contour = high support; red box = UAV GT",
    )
    panels.append(("Key retrieval region", support_overlay))

    detail_scale_map = tensor_feature_map(
        bhlr_dbg["detail_scale"],
        "mean_abs",
    )

    img = overlay_heatmap(
        rgb_semantic,
        detail_scale_map,
        alpha=0.55,
    )

    img = draw_boxes(
        img,
        semantic_boxes,
        "UAV GT",
    )

    save_image(
        save_dir_path,
        "23_detail_p3_activation.jpg",
        img,
        f"BHLR P{bhlr_scale} lost-detail feature",
        str(tuple(bhlr_dbg["detail_scale"].shape)),
    )
    panels.append(("P3 lost-detail feature", img))

    gate_map = (
        bhlr_dbg["detail_gate"]
        .detach()
        .float()
        .cpu()
        .numpy()[0, 0]
    )

    img = overlay_heatmap(
        rgb_semantic,
        gate_map,
        alpha=0.52,
        robust=False,
    )

    img = draw_boxes(
        img,
        semantic_boxes,
        "UAV GT",
    )

    save_image(
        save_dir_path,
        "24_detail_gate.jpg",
        img,
        f"BHLR P{bhlr_scale} useful-detail gate",
        "0 = suppress, 1 = retain",
    )
    panels.append(("Useful-detail gate", img))

    usable_detail_map = tensor_feature_map(
        usable_detail,
        "mean_abs",
    )

    img = overlay_heatmap(
        rgb_semantic,
        usable_detail_map,
        alpha=0.58,
    )

    img = draw_boxes(
        img,
        semantic_boxes,
        "UAV GT",
    )

    save_image(
        save_dir_path,
        "25_usable_detail_activation.jpg",
        img,
        "Detail actually injected into RGB",
        "support × gate × lost-detail",
    )
    panels.append(("Usable detail", img))

    before_map = tensor_feature_map(
        raw_rgb_p3,
        "mean_abs",
    )

    img = overlay_heatmap(
        rgb_semantic,
        before_map,
        alpha=0.50,
    )

    img = draw_boxes(
        img,
        semantic_boxes,
        "UAV GT",
    )

    save_image(
        save_dir_path,
        "26_rgb_p3_before_bhlr.jpg",
        img,
        f"RGB P{bhlr_scale} BEFORE BHLR",
        str(tuple(raw_rgb_p3.shape)),
    )
    panels.append(("RGB before BHLR", img))

    after_map = tensor_feature_map(
        enhanced_rgb_p3,
        "mean_abs",
    )

    img = overlay_heatmap(
        rgb_semantic,
        after_map,
        alpha=0.50,
    )

    img = draw_boxes(
        img,
        semantic_boxes,
        "UAV GT",
    )

    save_image(
        save_dir_path,
        "27_rgb_p3_after_bhlr.jpg",
        img,
        f"RGB P{bhlr_scale} AFTER BHLR",
        f"gamma={float(gamma.detach().cpu().item()):.5f}",
    )
    panels.append(("RGB after BHLR", img))

    enhancement = (
        (enhanced_rgb_p3 - raw_rgb_p3)
        .detach()
        .float()
    )

    enhancement_map = tensor_feature_map(
        enhancement,
        "mean_abs",
    )

    img = overlay_heatmap(
        rgb_semantic,
        enhancement_map,
        alpha=0.60,
    )

    img = draw_boxes(
        img,
        semantic_boxes,
        "UAV GT",
    )

    save_image(
        save_dir_path,
        "28_bhlr_enhancement_magnitude.jpg",
        img,
        "BHLR enhancement magnitude",
        "|RGB_after - RGB_before|",
    )
    panels.append(("BHLR enhancement", img))

    # -----------------------------------------------------------------------
    # 29 Final prediction
    # -----------------------------------------------------------------------

    pred_list = base_val.postprocess_predictions(
        preds=preds_raw,
        model=model,
        conf_thres=0.001,
        iou_thres=iou,
        max_det=max_det,
    )

    pred = (
        pred_list[0]
        if pred_list
        else None
    )

    final_img = draw_boxes(
        rgb_semantic,
        semantic_boxes,
        "GT",
        color=(0, 0, 255),
        thickness=2,
    )

    final_img = draw_prediction_boxes(
        final_img,
        pred,
        color=(0, 255, 0),
        conf_min=conf,
    )

    save_image(
        save_dir_path,
        "29_prediction_vs_gt.jpg",
        final_img,
        "Final detection: prediction vs GT",
        "red = GT, green = prediction",
    )
    panels.append(("Prediction vs GT", final_img))

    # -----------------------------------------------------------------------
    # Overview
    # -----------------------------------------------------------------------

    overview_path = (
        save_dir_path
        / "overview.jpg"
    )

    make_overview(
        panels,
        overview_path,
        cols=4,
        tile_w=480,
        tile_h=390,
    )

    # -----------------------------------------------------------------------
    # Metadata
    # -----------------------------------------------------------------------

    coarse_cpu = coarse.cpu()
    fine_cpu = fine.cpu()
    total_cpu = total.cpu()

    metadata = {
        "weights": str(
            Path(weights).resolve()
        ),
        "split": split,
        "dataset_index": int(index),
        "pair_key": str(
            sample["pair_key"]
        ),
        "rgb_path": str(
            sample["rgb_path"]
        ),
        "tir_path": str(
            sample["tir_path"]
        ),
        "rgb_imgsz": int(
            rgb_imgsz
        ),
        "rgb_semantic_imgsz": int(
            rgb_semantic_imgsz
        ),
        "tir_imgsz": int(
            tir_imgsz
        ),
        "align_scale": int(
            align_scale
        ),
        "bhlr_scale": int(
            bhlr_scale
        ),
        "tensor_shapes": {
            "rgb_high": list(
                batch["rgb_img"].shape
            ),
            "rgb_semantic": list(
                batch[
                    "rgb_semantic_img"
                ].shape
            ),
            "tir": list(
                batch["tir_img"].shape
            ),
            "rgb_p3_before": list(
                raw_rgb_p3.shape
            ),
            "tir_p3_raw": list(
                raw_tir_p3.shape
            ),
            "tir_p3_aligned": list(
                aligned_tir_p3.shape
            ),
            "detail_raw": list(
                bhlr_debug_all[
                    "detail_raw"
                ].shape
            ),
            "support_map": list(
                bhlr_dbg[
                    "support_map"
                ].shape
            ),
            "detail_scale": list(
                bhlr_dbg[
                    "detail_scale"
                ].shape
            ),
            "usable_detail": list(
                usable_detail.shape
            ),
            "rgb_p3_after": list(
                enhanced_rgb_p3.shape
            ),
        },
        "fbam": {
            "coarse_offset_mean_abs": float(
                coarse_cpu.abs().mean()
            ),
            "coarse_offset_max_abs": float(
                coarse_cpu.abs().max()
            ),
            "fine_offset_mean_abs": float(
                fine_cpu.abs().mean()
            ),
            "fine_offset_max_abs": float(
                fine_cpu.abs().max()
            ),
            "total_offset_mean_abs": float(
                total_cpu.abs().mean()
            ),
            "total_offset_max_abs": float(
                total_cpu.abs().max()
            ),
            "confidence_mean": float(
                align_dbg[
                    "confidence"
                ]
                .detach()
                .float()
                .mean()
                .cpu()
            ),
        },
        "bhlr": {
            "gamma": float(
                gamma.detach()
                .float()
                .cpu()
                .item()
            ),
            "support_mean": float(
                bhlr_dbg[
                    "support_map"
                ]
                .detach()
                .float()
                .mean()
                .cpu()
            ),
            "support_max": float(
                bhlr_dbg[
                    "support_map"
                ]
                .detach()
                .float()
                .max()
                .cpu()
            ),
            "gate_mean": float(
                bhlr_dbg[
                    "detail_gate"
                ]
                .detach()
                .float()
                .mean()
                .cpu()
            ),
            "gate_max": float(
                bhlr_dbg[
                    "detail_gate"
                ]
                .detach()
                .float()
                .max()
                .cpu()
            ),
            "residual_mean_abs": float(
                residual_high
                .abs()
                .mean()
                .cpu()
            ),
            "usable_detail_mean_abs": float(
                usable_detail
                .abs()
                .mean()
                .cpu()
            ),
            "enhancement_mean_abs": float(
                enhancement
                .abs()
                .mean()
                .cpu()
            ),
        },
        "gt": {
            "rgb_num_uav": int(
                len(rgb_boxes_original)
            ),
            "tir_num_uav": int(
                len(tir_boxes_original)
            ),
            "rgb_original_boxes_xyxy": (
                rgb_boxes_original.tolist()
            ),
            "rgb_semantic_boxes_xyxy": (
                semantic_boxes.tolist()
            ),
            "tir_original_boxes_xyxy": (
                tir_boxes_original.tolist()
            ),
        },
        "prediction": {
            "visualization_conf": float(
                conf
            ),
            "num_predictions_above_conf": int(
                sum(
                    float(c) >= conf
                    for c in (
                        pred["conf"]
                        .detach()
                        .float()
                        .cpu()
                        .numpy()
                        if pred is not None
                        else []
                    )
                )
            ),
        },
        "important_note": (
            "15_tir_image_warp_preview.jpg is an approximate image-level "
            "visualization generated by upsampling P3 feature offsets. "
            "FBAM itself aligns feature maps, not raw input pixels."
        ),
    }

    metadata_path = (
        save_dir_path
        / "metadata.json"
    )

    with metadata_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # -----------------------------------------------------------------------
    # Console summary
    # -----------------------------------------------------------------------

    print(
        "\n"
        "============================================================"
    )
    print(
        "FBAM + BHLR internal visualization completed"
    )
    print(
        "============================================================"
    )
    print(
        f"Dataset index   : {index}"
    )
    print(
        f"Pair key        : {sample['pair_key']}"
    )
    print(
        f"RGB             : {sample['rgb_path']}"
    )
    print(
        f"TIR             : {sample['tir_path']}"
    )
    print(
        f"FBAM P{align_scale}     : "
        f"offset mean={metadata['fbam']['total_offset_mean_abs']:.5f}, "
        f"max={metadata['fbam']['total_offset_max_abs']:.5f}"
    )
    print(
        f"BHLR P{bhlr_scale}     : "
        f"gamma={metadata['bhlr']['gamma']:.6f}, "
        f"support mean={metadata['bhlr']['support_mean']:.5f}, "
        f"gate mean={metadata['bhlr']['gate_mean']:.5f}"
    )
    print(
        f"Output dir      : {save_dir_path}"
    )
    print(
        f"Overview        : {overview_path}"
    )
    print(
        f"Metadata        : {metadata_path}"
    )
    print(
        "============================================================\n"
    )


# ===========================================================================
# CLI
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Visualize internal FBAM/BHLR features "
            "on one random UAV-positive test image."
        )
    )

    parser.add_argument(
        "--weights",
        type=str,
        default=(
            "/mnt/sda/taochangyong/Projects/Model/YOLO-tcy/"
            "runs/fbam_bhlr/"
            "yolo26n_UAVCB_fbamP3_bhlrP3_rgb1280_sem640_tir640_seed0/"
            "weights/best.pt"
        ),
    )

    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=[
            "train",
            "val",
            "test",
        ],
    )

    parser.add_argument(
        "--device",
        type=str,
        default="5",
    )

    parser.add_argument(
        "--rgb",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--tir",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--rgb-imgsz",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--rgb-semantic-imgsz",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--tir-imgsz",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--pair-mode",
        type=str,
        default=None,
        choices=[
            "relative",
            "stem",
            "filename",
        ],
    )

    parser.add_argument(
        "--index",
        type=int,
        nargs="+",
        default=None,
        help=(
            "One or more dataset indices. Examples: "
            "--index 0 50 100 200. "
            "If omitted, randomly choose one UAV-positive sample."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--align-scale",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--bhlr-scale",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Prediction confidence for drawing final boxes.",
    )

    parser.add_argument(
        "--iou",
        type=float,
        default=0.70,
    )

    parser.add_argument(
        "--max-det",
        type=int,
        default=300,
    )

    parser.add_argument(
        "--support-thr",
        type=float,
        default=0.50,
        help="Threshold used to draw the BHLR key-support contour.",
    )

    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
    )

    args = parser.parse_args()

    # ---------------------------------------------------------------
    # Multi-index mode
    #
    # Example:
    #   --index 10 20 30
    #
    # If --index is omitted, preserve the old behavior and randomly
    # choose one UAV-positive sample.
    # ---------------------------------------------------------------

    selected_indices = (
        args.index
        if args.index is not None
        else [None]
    )

    # Remove duplicates while preserving the requested order.
    if args.index is not None:
        selected_indices = list(
            dict.fromkeys(selected_indices)
        )

    total_selected = len(selected_indices)

    print(
        "\n"
        "============================================================"
    )
    print("FBAM + BHLR multi-index visualization")
    print(
        "============================================================"
    )

    if args.index is None:
        print("Indices         : random positive sample")
    else:
        print(f"Indices         : {selected_indices}")

    print(f"Total           : {total_selected}")
    print(
        "============================================================\n"
    )

    for position, current_index in enumerate(
        selected_indices,
        start=1,
    ):
        print(
            "\n"
            "############################################################"
        )
        print(
            f"Visualizing {position}/{total_selected}: "
            f"index={current_index}"
        )
        print(
            "############################################################\n"
        )

        current_save_dir = args.save_dir

        # visualize() already generates one unique directory per index when
        # --save-dir is omitted. If the user explicitly gives --save-dir,
        # add one child directory for every requested index to avoid overwrite.
        if (
            args.save_dir is not None
            and total_selected > 1
            and current_index is not None
        ):
            current_save_dir = str(
                Path(args.save_dir)
                .expanduser()
                .resolve()
                / f"index_{int(current_index):06d}"
            )

        visualize(
            weights=args.weights,
            split=args.split,
            device_str=args.device,
            rgb_yaml=args.rgb,
            tir_yaml=args.tir,
            rgb_imgsz=args.rgb_imgsz,
            rgb_semantic_imgsz=args.rgb_semantic_imgsz,
            tir_imgsz=args.tir_imgsz,
            pair_mode=args.pair_mode,
            index=current_index,
            seed=args.seed,
            align_scale=args.align_scale,
            bhlr_scale=args.bhlr_scale,
            conf=args.conf,
            iou=args.iou,
            max_det=args.max_det,
            support_thr=args.support_thr,
            save_dir=current_save_dir,
        )

    print(
        "\n"
        "============================================================"
    )
    print("All requested indices have been visualized.")
    print(
        "============================================================\n"
    )
