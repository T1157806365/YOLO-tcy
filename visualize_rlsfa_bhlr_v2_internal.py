#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Standalone outputs
------------------
INPUT
  00_rgb_original_gt.jpg
  01_tir_original_gt.jpg
  02_rgb_semantic_restored_gt.jpg
  03_tir_network_restored_gt.jpg

RLSFA
  04_rgb_p3_before_alignment.jpg
  05_tir_p3_before_alignment.jpg
  06_coarse_targetness.jpg
  07_tir_before_coarse_rgb_grid.jpg
  08_tir_after_coarse.jpg
  09_rgb_spatial_structure.jpg
  10_rgb_local_frequency.jpg
  11_rgb_frequency_reliability.jpg
  12_rgb_reliable_structure.jpg
  13_tir_spatial_structure.jpg
  14_tir_local_frequency.jpg
  15_tir_frequency_reliability.jpg
  16_tir_reliable_structure.jpg
  17_fine_displacement_probability.jpg
  18_total_offset.jpg
  19_tir_after_final_rlsfa.jpg
  20_aligned_tir_p3.jpg

BHLR-v2
  21_rgb_high_gt.jpg
  22_rgb_semantic_upscaled.jpg
  23_highres_residual.jpg
  24_highres_residual_overlay.jpg
  25_uav_high_zoom.jpg
  26_uav_semantic_zoom.jpg
  27_uav_residual_zoom.jpg
  28_pixel_unshuffle_detail.jpg
  29_detail_bank.jpg
  30_detail_avg_pool_p3.jpg
  31_detail_max_pool_p3.jpg
  32_detail_projection_p3.jpg
  33_bhlr_guidance_targetness.jpg
  34_bhlr_rgb_structure.jpg
  35_bhlr_tir_structure.jpg
  36_selector_logit.jpg
  37_selector_multiplier.jpg
  38_selector_multiplier_overlay.jpg
  39_usable_detail.jpg
  40_usable_detail_overlay.jpg
  41_rgb_p3_before_bhlr.jpg
  42_rgb_p3_after_bhlr.jpg
  43_bhlr_enhancement.jpg

DETECTION
  44_prediction_vs_gt.jpg

  metadata.json

Example
-------
conda activate tcy
cd /mnt/sda/taochangyong/Projects/Model/YOLO-tcy
python visualize_rlsfa_bhlr_v2_internal.py \
  --weights runs/rlsfa_bhlr_v2/yolo26n_UAVCB_RLSFA-P3_BHLRv2-P3_v2_rgb1280_sem640_tir640_seed0/weights/best.pt \
  --split test \
  --device 5 \
  --index 0 100 200 300 400 500 600
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:
    Image = ImageDraw = ImageFont = None

import val_rgbt as base_val

from datasets.rsdt_dataset import build_rsdt_dataset
from datasets.rgbt_dataset import load_yolo_label
from train_rsdt import preprocess_batch_rsdt
from val_rlsfa_bhlr_v2 import load_rlsfa_bhlr_v2_checkpoint


# =============================================================================
# Basic image / tensor helpers
# =============================================================================


def ensure_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 1:
        return cv2.cvtColor(image[..., 0], cv2.COLOR_GRAY2BGR)
    return image



def read_rgb_bgr(path: str) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return image



def read_tir_bgr(path: str) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)



def tensor_rgb_to_bgr(x: torch.Tensor) -> np.ndarray:
    x = x.detach().float().cpu()
    if x.ndim == 4:
        x = x[0]
    if x.ndim != 3:
        raise ValueError(f"Expected CHW tensor, got {tuple(x.shape)}")

    if float(x.max()) > 1.5:
        x = x / 255.0
    x = x.clamp(0.0, 1.0)

    arr = (
        x.permute(1, 2, 0).numpy() * 255.0
    ).round().astype(np.uint8)

    if arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)

    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)



def tensor_feature_map(
    x: torch.Tensor,
    mode: str = "mean_abs",
) -> np.ndarray:
    x = x.detach().float().cpu()

    if x.ndim == 4:
        x = x[0]

    if x.ndim == 2:
        return x.numpy().astype(np.float32)

    if x.ndim != 3:
        raise ValueError(f"Expected CHW/BCHW feature, got {tuple(x.shape)}")

    if mode == "mean":
        y = x.mean(dim=0)
    elif mode == "max_abs":
        y = x.abs().amax(dim=0)
    elif mode == "l2":
        y = torch.sqrt(torch.clamp((x * x).mean(dim=0), min=0.0))
    else:
        y = x.abs().mean(dim=0)

    return y.numpy().astype(np.float32)



def tensor_single_map(x: torch.Tensor) -> np.ndarray:
    x = x.detach().float().cpu()
    while x.ndim > 2:
        x = x[0]
    return x.numpy().astype(np.float32)


# =============================================================================
# LetterBox -> original-resolution mapping
# =============================================================================


def _scalar_pair(value, default=(1.0, 1.0)) -> Tuple[float, float]:
    if value is None:
        return float(default[0]), float(default[1])

    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()

    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return float(default[0]), float(default[1])
    if arr.size == 1:
        return float(arr[0]), float(arr[0])
    return float(arr[0]), float(arr[1])



def parse_ratio_pad(
    ratio_pad,
    orig_hw: Tuple[int, int],
    net_hw: Tuple[int, int],
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Return ((gain_x, gain_y), (pad_x, pad_y)).

    Supports the common custom-dataset format:
        ((gain_x, gain_y), (pad_x, pad_y))
    and falls back to geometry inferred from original/network shapes.
    """
    orig_h, orig_w = int(orig_hw[0]), int(orig_hw[1])
    net_h, net_w = int(net_hw[0]), int(net_hw[1])

    if ratio_pad is not None:
        try:
            if isinstance(ratio_pad, (list, tuple)) and len(ratio_pad) == 2:
                first, second = ratio_pad
                if isinstance(first, (list, tuple, np.ndarray)) or torch.is_tensor(first):
                    gain_yx = _scalar_pair(first)
                    # Most letterbox() functions store ratio as (r_w, r_h) or (r, r).
                    gain_x, gain_y = gain_yx[0], gain_yx[1]
                    pad_x, pad_y = _scalar_pair(second, default=(0.0, 0.0))
                    return (gain_x, gain_y), (pad_x, pad_y)
        except Exception:
            pass

    gain = min(
        net_w / max(orig_w, 1),
        net_h / max(orig_h, 1),
    )
    resized_w = int(round(orig_w * gain))
    resized_h = int(round(orig_h * gain))
    pad_x = (net_w - resized_w) / 2.0
    pad_y = (net_h - resized_h) / 2.0
    return (gain, gain), (pad_x, pad_y)



def unletterbox_image(
    image_lb: np.ndarray,
    orig_hw: Tuple[int, int],
    ratio_pad=None,
) -> np.ndarray:
    """Remove letterbox padding and resize back to the original native size."""
    image_lb = ensure_bgr(image_lb)
    net_h, net_w = image_lb.shape[:2]
    orig_h, orig_w = int(orig_hw[0]), int(orig_hw[1])

    (_, _), (pad_x, pad_y) = parse_ratio_pad(
        ratio_pad,
        orig_hw=orig_hw,
        net_hw=(net_h, net_w),
    )

    left = int(round(pad_x - 0.1))
    right = int(round(pad_x + 0.1))
    top = int(round(pad_y - 0.1))
    bottom = int(round(pad_y + 0.1))

    x1 = max(0, left)
    y1 = max(0, top)
    x2 = min(net_w, net_w - max(0, right))
    y2 = min(net_h, net_h - max(0, bottom))

    if x2 <= x1 or y2 <= y1:
        crop = image_lb
    else:
        crop = image_lb[y1:y2, x1:x2]

    return cv2.resize(
        crop,
        (orig_w, orig_h),
        interpolation=cv2.INTER_CUBIC,
    )



def feature_map_on_original_grid(
    fmap: np.ndarray,
    network_hw: Tuple[int, int],
    orig_hw: Tuple[int, int],
    ratio_pad=None,
) -> np.ndarray:
    """Map an HxW feature map through the network letterbox grid to native size."""
    net_h, net_w = int(network_hw[0]), int(network_hw[1])
    fmap = np.asarray(fmap, dtype=np.float32)

    fmap_net = cv2.resize(
        fmap,
        (net_w, net_h),
        interpolation=cv2.INTER_LINEAR,
    )

    (_, _), (pad_x, pad_y) = parse_ratio_pad(
        ratio_pad,
        orig_hw=orig_hw,
        net_hw=(net_h, net_w),
    )

    left = int(round(pad_x - 0.1))
    right = int(round(pad_x + 0.1))
    top = int(round(pad_y - 0.1))
    bottom = int(round(pad_y + 0.1))

    x1 = max(0, left)
    y1 = max(0, top)
    x2 = min(net_w, net_w - max(0, right))
    y2 = min(net_h, net_h - max(0, bottom))

    if x2 <= x1 or y2 <= y1:
        crop = fmap_net
    else:
        crop = fmap_net[y1:y2, x1:x2]

    orig_h, orig_w = int(orig_hw[0]), int(orig_hw[1])
    return cv2.resize(
        crop,
        (orig_w, orig_h),
        interpolation=cv2.INTER_LINEAR,
    )



def scale_xyxy_from_letterbox_to_original(
    boxes_xyxy: np.ndarray,
    orig_hw: Tuple[int, int],
    net_hw: Tuple[int, int],
    ratio_pad=None,
) -> np.ndarray:
    boxes = np.asarray(boxes_xyxy, dtype=np.float32).reshape(-1, 4).copy()
    if len(boxes) == 0:
        return boxes

    orig_h, orig_w = int(orig_hw[0]), int(orig_hw[1])
    (gain_x, gain_y), (pad_x, pad_y) = parse_ratio_pad(
        ratio_pad,
        orig_hw=orig_hw,
        net_hw=net_hw,
    )

    boxes[:, [0, 2]] -= float(pad_x)
    boxes[:, [1, 3]] -= float(pad_y)
    boxes[:, [0, 2]] /= max(float(gain_x), 1e-12)
    boxes[:, [1, 3]] /= max(float(gain_y), 1e-12)

    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w - 1)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h - 1)
    return boxes


# =============================================================================
# Heatmap helpers
# =============================================================================


def normalize_robust(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    if x.size == 0:
        return np.zeros_like(x, dtype=np.float32)

    lo = float(np.percentile(x, 1.0))
    hi = float(np.percentile(x, 99.0))

    if hi <= lo + 1e-12:
        lo = float(x.min())
        hi = float(x.max())

    if hi <= lo + 1e-12:
        return np.zeros_like(x, dtype=np.float32)

    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)



def normalize_fixed(x: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x = np.nan_to_num(x, nan=vmin, posinf=vmax, neginf=vmin)
    if vmax <= vmin:
        raise ValueError(f"Invalid fixed range [{vmin}, {vmax}]")
    return np.clip((x - vmin) / (vmax - vmin), 0.0, 1.0)



def normalize_signed(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    if x.size == 0:
        return np.zeros_like(x, dtype=np.float32)
    limit = float(np.percentile(np.abs(x), 99.0))
    if limit <= 1e-12:
        limit = float(np.max(np.abs(x)))
    if limit <= 1e-12:
        return np.full_like(x, 0.5, dtype=np.float32)
    return np.clip((x + limit) / (2.0 * limit), 0.0, 1.0)



def colormap_from_norm(norm: np.ndarray) -> np.ndarray:
    gray = (np.clip(norm, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    return cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)



def mapped_heatmap(
    fmap: np.ndarray,
    network_hw: Tuple[int, int],
    orig_hw: Tuple[int, int],
    ratio_pad=None,
    mode: str = "robust",
    vmin: float = 0.0,
    vmax: float = 1.0,
) -> np.ndarray:
    mapped = feature_map_on_original_grid(
        fmap,
        network_hw=network_hw,
        orig_hw=orig_hw,
        ratio_pad=ratio_pad,
    )

    if mode == "fixed":
        norm = normalize_fixed(mapped, vmin=vmin, vmax=vmax)
    elif mode == "signed":
        norm = normalize_signed(mapped)
    else:
        norm = normalize_robust(mapped)

    return colormap_from_norm(norm)



def mapped_overlay(
    base_original: np.ndarray,
    fmap: np.ndarray,
    network_hw: Tuple[int, int],
    ratio_pad=None,
    alpha: float = 0.52,
    mode: str = "robust",
    vmin: float = 0.0,
    vmax: float = 1.0,
) -> np.ndarray:
    base_original = ensure_bgr(base_original).copy()
    orig_hw = base_original.shape[:2]

    hm = mapped_heatmap(
        fmap,
        network_hw=network_hw,
        orig_hw=orig_hw,
        ratio_pad=ratio_pad,
        mode=mode,
        vmin=vmin,
        vmax=vmax,
    )

    return cv2.addWeighted(
        base_original,
        1.0 - float(alpha),
        hm,
        float(alpha),
        0.0,
    )




def letterbox_content_bounds(
    orig_hw: Tuple[int, int],
    net_hw: Tuple[int, int],
    ratio_pad=None,
) -> Tuple[int, int, int, int]:
    """Valid image region (x1,y1,x2,y2) inside a LetterBox network canvas."""
    net_h, net_w = int(net_hw[0]), int(net_hw[1])
    (gain_x, gain_y), (pad_x, pad_y) = parse_ratio_pad(
        ratio_pad,
        orig_hw=orig_hw,
        net_hw=net_hw,
    )
    orig_h, orig_w = int(orig_hw[0]), int(orig_hw[1])
    valid_w = int(round(orig_w * gain_x))
    valid_h = int(round(orig_h * gain_y))
    x1 = max(0, min(net_w, int(round(pad_x - 0.1))))
    y1 = max(0, min(net_h, int(round(pad_y - 0.1))))
    x2 = max(x1, min(net_w, x1 + valid_w))
    y2 = max(y1, min(net_h, y1 + valid_h))
    return x1, y1, x2, y2



def black_letterbox_padding(
    image_lb: np.ndarray,
    orig_hw: Tuple[int, int],
    ratio_pad=None,
) -> np.ndarray:
    """Keep the network canvas, but force all LetterBox padding to pure black."""
    image = ensure_bgr(image_lb).copy()
    h, w = image.shape[:2]
    x1, y1, x2, y2 = letterbox_content_bounds(
        orig_hw=orig_hw,
        net_hw=(h, w),
        ratio_pad=ratio_pad,
    )
    out = np.zeros_like(image)
    if x2 > x1 and y2 > y1:
        out[y1:y2, x1:x2] = image[y1:y2, x1:x2]
    return out



def network_feature_overlay(
    base_lb: np.ndarray,
    fmap: np.ndarray,
    orig_hw: Tuple[int, int],
    ratio_pad=None,
    alpha: float = 0.42,
    mode: str = "robust",
    vmin: float = 0.0,
    vmax: float = 1.0,
) -> np.ndarray:
    """
    Overlay a NETWORK feature map on the complete LetterBox canvas.

    The original image content may occupy only part of the network canvas
    (e.g. TIR 640x512 inside a 640x640 input with top/bottom padding), but
    CNN features are computed on the ENTIRE padded input. Therefore the
    padding area also has real feature activations and must be visualized.

    Background image:
        keep LetterBox padding black.
    Feature heatmap:
        cover ALL network pixels, including padding.
    """
    base = black_letterbox_padding(
        base_lb,
        orig_hw=orig_hw,
        ratio_pad=ratio_pad,
    )

    h, w = base.shape[:2]
    fmap_net = cv2.resize(
        np.asarray(fmap, dtype=np.float32),
        (w, h),
        interpolation=cv2.INTER_LINEAR,
    )

    if mode == "fixed":
        norm = normalize_fixed(fmap_net, vmin=vmin, vmax=vmax)
    elif mode == "signed":
        norm = normalize_signed(fmap_net)
    else:
        norm = normalize_robust(fmap_net)

    heat = colormap_from_norm(norm)

    # Key correction: blend over the FULL network canvas.
    # Do not crop/mask the heatmap to the 640x512 valid image area.
    return cv2.addWeighted(
        base,
        1.0 - float(alpha),
        heat,
        float(alpha),
        0.0,
    )



def shared_robust_range(
    *arrays: np.ndarray,
    low_percentile: float = 1.0,
    high_percentile: float = 99.0,
) -> Tuple[float, float]:
    """One color range shared by several maps, used by images 41 and 42."""
    vals = []
    for a in arrays:
        x = np.asarray(a, dtype=np.float32)
        x = x[np.isfinite(x)]
        if x.size:
            vals.append(x.reshape(-1))
    if not vals:
        return 0.0, 1.0
    x = np.concatenate(vals)
    lo = float(np.percentile(x, low_percentile))
    hi = float(np.percentile(x, high_percentile))
    if hi <= lo + 1e-12:
        lo = float(x.min())
        hi = float(x.max())
    if hi <= lo + 1e-12:
        hi = lo + 1.0
    return lo, hi



def warp_inside_letterbox_content(
    image_lb: np.ndarray,
    offset: torch.Tensor,
    feature_hw: Tuple[int, int],
    orig_hw: Tuple[int, int],
    ratio_pad=None,
) -> np.ndarray:
    """
    Visualization-only translation while keeping black LetterBox bars fixed.

    For 640x512 TIR -> 640x640:
        valid region = 640x512
        top/bottom padding = 64/64 px
    """
    canvas = black_letterbox_padding(image_lb, orig_hw=orig_hw, ratio_pad=ratio_pad)
    h, w = canvas.shape[:2]
    x1, y1, x2, y2 = letterbox_content_bounds(orig_hw, (h, w), ratio_pad)
    if x2 <= x1 or y2 <= y1:
        return canvas

    content = canvas[y1:y2, x1:x2].copy()
    ch, cw = content.shape[:2]
    dx, dy = offset_xy(offset)
    fh, fw = int(feature_hw[0]), int(feature_hw[1])

    # RLSFA sampling convention is source(p + delta), so visual motion ~= -delta.
    shift_x = -float(dx) * (float(cw) / max(float(fw), 1.0))
    shift_y = -float(dy) * (float(ch) / max(float(fh), 1.0))
    M = np.array([[1.0, 0.0, shift_x], [0.0, 1.0, shift_y]], dtype=np.float32)
    warped = cv2.warpAffine(
        content,
        M,
        (cw, ch),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    out = np.zeros_like(canvas)
    out[y1:y2, x1:x2] = warped
    return out



def signed_response_change_overlay(
    base_lb: np.ndarray,
    delta_map: np.ndarray,
    orig_hw: Tuple[int, int],
    ratio_pad=None,
    alpha_max: float = 0.72,
    percentile: float = 99.0,
) -> Tuple[np.ndarray, float]:
    """
    True response change map:
        delta = mean_c(|F_after|) - mean_c(|F_before|)

    red  = response increase
    blue = response decrease

    delta_map is a network feature quantity, so it is rendered on the FULL
    LetterBox canvas, including padding regions.
    """
    base = black_letterbox_padding(
        base_lb,
        orig_hw=orig_hw,
        ratio_pad=ratio_pad,
    )
    h, w = base.shape[:2]
    delta = cv2.resize(
        np.asarray(delta_map, dtype=np.float32),
        (w, h),
        interpolation=cv2.INTER_LINEAR,
    )

    # Full network grid, including padding.
    finite = delta[np.isfinite(delta)]
    if finite.size:
        limit = float(np.percentile(np.abs(finite), percentile))
        if limit <= 1e-12:
            limit = float(np.max(np.abs(finite)))
    else:
        limit = 0.0

    if limit <= 1e-12:
        return base, 0.0

    s = np.clip(delta / limit, -1.0, 1.0)
    color = np.zeros_like(base, dtype=np.float32)
    color[..., 2] = 255.0 * np.clip(s, 0.0, 1.0)
    color[..., 0] = 255.0 * np.clip(-s, 0.0, 1.0)
    a = np.abs(s)[..., None] * float(alpha_max)
    blend = base.astype(np.float32) * (1.0 - a) + color * a

    return np.clip(blend, 0, 255).astype(np.uint8), limit


# =============================================================================
# Drawing and saving helpers
# =============================================================================


def normalized_xywh_to_xyxy(
    boxes: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    if len(boxes) == 0:
        return np.zeros((0, 4), dtype=np.float32)

    cx = boxes[:, 0] * width
    cy = boxes[:, 1] * height
    bw = boxes[:, 2] * width
    bh = boxes[:, 3] * height

    xyxy = np.stack(
        [
            cx - bw / 2,
            cy - bh / 2,
            cx + bw / 2,
            cy + bh / 2,
        ],
        axis=1,
    )

    xyxy[:, [0, 2]] = np.clip(xyxy[:, [0, 2]], 0, width - 1)
    xyxy[:, [1, 3]] = np.clip(xyxy[:, [1, 3]], 0, height - 1)
    return xyxy



def draw_boxes(
    image: np.ndarray,
    boxes_xyxy: np.ndarray,
    label: str = "UAV GT",
    color: Tuple[int, int, int] = (0, 0, 255),
) -> np.ndarray:
    out = ensure_bgr(image).copy()
    h, w = out.shape[:2]
    thickness = max(2, int(round(max(h, w) / 900.0)))
    font_scale = max(0.45, min(1.1, max(h, w) / 1800.0))

    for box in np.asarray(boxes_xyxy, dtype=np.float32).reshape(-1, 4):
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(
            out,
            label,
            (x1, max(20, y1 - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            max(1, thickness // 2),
            cv2.LINE_AA,
        )

    return out



def draw_predictions_xyxy(
    image: np.ndarray,
    boxes_xyxy: np.ndarray,
    conf: np.ndarray,
    conf_min: float,
) -> np.ndarray:
    out = ensure_bgr(image).copy()
    h, w = out.shape[:2]
    thickness = max(2, int(round(max(h, w) / 900.0)))
    font_scale = max(0.45, min(1.1, max(h, w) / 1800.0))

    for box, score in zip(boxes_xyxy, conf):
        if float(score) < float(conf_min):
            continue
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), thickness)
        cv2.putText(
            out,
            f"Pred {float(score):.2f}",
            (x1, max(20, y1 - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 255, 0),
            max(1, thickness // 2),
            cv2.LINE_AA,
        )

    return out



def _unicode_font(size: int, bold: bool = False):
    if ImageFont is None:
        return None

    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold
        else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]

    for font_path in candidates:
        if Path(font_path).exists():
            try:
                return ImageFont.truetype(font_path, size=max(8, int(size)))
            except Exception:
                pass
    return None



def put_unicode_text(
    image_bgr: np.ndarray,
    text: str,
    xy: Tuple[int, int],
    font_size: int,
    color_bgr: Tuple[int, int, int] = (25, 25, 25),
    bold: bool = False,
) -> np.ndarray:
    if not text:
        return image_bgr

    font = _unicode_font(font_size, bold=bold)
    if Image is not None and ImageDraw is not None and font is not None:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        draw = ImageDraw.Draw(pil)
        color_rgb = (int(color_bgr[2]), int(color_bgr[1]), int(color_bgr[0]))
        draw.text((int(xy[0]), int(xy[1])), text, font=font, fill=color_rgb)
        return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)

    fallback = text.replace("Δ", "Delta_")
    cv2.putText(
        image_bgr,
        fallback,
        (int(xy[0]), int(xy[1] + font_size)),
        cv2.FONT_HERSHEY_SIMPLEX,
        max(0.42, font_size / 32.0),
        color_bgr,
        max(1, int(round(font_size / 18.0))),
        cv2.LINE_AA,
    )
    return image_bgr



def add_title_inside(
    image: np.ndarray,
    title: str,
    subtitle: str = "",
    annotation: str = "",
) -> np.ndarray:
    """Draw title inside the native-resolution image; pixel dimensions do not change."""
    image = ensure_bgr(image).copy()
    h, w = image.shape[:2]

    scale_ref = max(0.7, min(w / 1280.0, h / 720.0))
    base_h = 104 if annotation else 72
    band_h = int(round(base_h * scale_ref))
    band_h = min(max(46, band_h), max(46, h // 3))

    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (w, band_h), (248, 248, 248), -1)
    image = cv2.addWeighted(overlay, 0.93, image, 0.07, 0.0)

    x = max(8, int(round(14 * scale_ref)))
    title_y = max(20, int(round(29 * scale_ref)))
    sub_y = max(title_y + 18, int(round(56 * scale_ref)))

    cv2.putText(
        image,
        title,
        (x, min(title_y, band_h - 7)),
        cv2.FONT_HERSHEY_SIMPLEX,
        max(0.42, 0.70 * scale_ref),
        (20, 20, 20),
        max(1, int(round(2 * scale_ref))),
        cv2.LINE_AA,
    )

    if subtitle:
        cv2.putText(
            image,
            subtitle,
            (x, min(sub_y, band_h - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            max(0.34, 0.44 * scale_ref),
            (75, 75, 75),
            max(1, int(round(scale_ref))),
            cv2.LINE_AA,
        )

    if annotation:
        ann_y = max(sub_y + 8, int(round(70 * scale_ref)))
        image = put_unicode_text(
            image,
            annotation,
            (x, min(ann_y, max(0, band_h - int(round(22 * scale_ref))))),
            font_size=max(12, int(round(18 * scale_ref))),
            color_bgr=(30, 30, 30),
            bold=True,
        )

    return image



def save_native(
    out_dir: Path,
    number: int,
    slug: str,
    image: np.ndarray,
    title: str,
    subtitle: str = "",
    annotation: str = "",
):
    """Save only the visualization canvas; never draw a white title strip."""
    del title, subtitle, annotation
    out_dir.mkdir(parents=True, exist_ok=True)
    image = ensure_bgr(image)
    path = out_dir / f"{number:02d}_{slug}.jpg"
    ok = cv2.imwrite(
        str(path),
        image,
        [cv2.IMWRITE_JPEG_QUALITY, 97],
    )
    if not ok:
        raise RuntimeError(f"Failed to save {path}")


def crop_around_box(
    image: np.ndarray,
    box: Optional[np.ndarray],
    expand: float = 5.0,
    min_size: int = 128,
) -> np.ndarray:
    image = ensure_bgr(image)
    h, w = image.shape[:2]

    if box is None:
        side = min(h, w, max(min_size, min(h, w) // 3))
        cx, cy = w // 2, h // 2
        x1 = max(0, cx - side // 2)
        y1 = max(0, cy - side // 2)
        x2 = min(w, x1 + side)
        y2 = min(h, y1 + side)
        return image[y1:y2, x1:x2].copy()

    x1, y1, x2, y2 = [float(v) for v in box]
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)

    crop_w = max(float(min_size), bw * float(expand))
    crop_h = max(float(min_size), bh * float(expand))

    left = max(0, int(math.floor(cx - crop_w / 2)))
    top = max(0, int(math.floor(cy - crop_h / 2)))
    right = min(w, int(math.ceil(cx + crop_w / 2)))
    bottom = min(h, int(math.ceil(cy + crop_h / 2)))

    if right <= left or bottom <= top:
        return image.copy()
    return image[top:bottom, left:right].copy()



def crop_to_native_canvas(
    image: np.ndarray,
    box: Optional[np.ndarray],
    canvas_hw: Tuple[int, int],
    expand: float = 5.0,
    min_size: int = 128,
) -> np.ndarray:
    """Magnify a crop but keep the saved image at original RGB resolution."""
    crop = crop_around_box(image, box, expand=expand, min_size=min_size)
    target_h, target_w = int(canvas_hw[0]), int(canvas_hw[1])

    h, w = crop.shape[:2]
    scale = min(target_w / max(w, 1), target_h / max(h, 1))
    rw = max(1, int(round(w * scale)))
    rh = max(1, int(round(h * scale)))

    resized = cv2.resize(
        crop,
        (rw, rh),
        interpolation=cv2.INTER_CUBIC,
    )

    canvas = np.full((target_h, target_w, 3), 245, dtype=np.uint8)
    x0 = (target_w - rw) // 2
    y0 = (target_h - rh) // 2
    canvas[y0:y0 + rh, x0:x0 + rw] = resized
    return canvas


# =============================================================================
# RLSFA helpers
# =============================================================================


def get_scale_item(mapping: Mapping, scale: int):
    if scale in mapping:
        return mapping[scale]
    if str(scale) in mapping:
        return mapping[str(scale)]
    raise KeyError(f"P{scale} not found; available={list(mapping.keys())}")



def debug_get(debug: Dict, *keys: str):
    for key in keys:
        if key in debug:
            return debug[key]
    raise KeyError(
        f"None of debug keys {keys} exist. Available={list(debug.keys())}"
    )



def get_alignment_debug(outputs: Dict, scale: int) -> Dict:
    info = outputs["alignment_info"]
    mapping = info.get("scale_debug", info.get("scales", {}))
    return get_scale_item(mapping, scale)



def get_bhlr_debug(outputs: Dict, scale: int):
    all_debug = outputs.get("bhlr_debug", outputs.get("rsdt_debug", None))
    if all_debug is None:
        raise KeyError("No BHLR-v2 debug dictionary in outputs")
    return all_debug, get_scale_item(all_debug["scales"], scale)



def collect_positive_indices(dataset) -> List[int]:
    positive = []
    for i, item in enumerate(dataset.samples):
        label_path = item.get("rgb_label", item.get("rgb_label_path", None))
        if label_path is None:
            continue
        _, boxes = load_yolo_label(
            label_path,
            nc=1,
            strict=True,
            target_classes=(0,),
            remap_classes=True,
        )
        if len(boxes) > 0:
            positive.append(i)
    return positive


@torch.no_grad()
def warp_image_translation_preview(
    tir_tensor: torch.Tensor,
    offset_feature: torch.Tensor,
    feature_hw: Tuple[int, int],
) -> torch.Tensor:
    """Apply the same RLSFA global sampling translation to the TIR network image."""
    if tir_tensor.ndim == 3:
        tir_tensor = tir_tensor.unsqueeze(0)

    source = tir_tensor.float()
    b, _, h, w = source.shape
    fh, fw = feature_hw

    offset = offset_feature.detach().float().clone()
    offset[:, 0] *= float(w) / float(fw)
    offset[:, 1] *= float(h) / float(fh)
    offset = offset.expand(-1, -1, h, w)

    ys = torch.linspace(-1.0, 1.0, h, device=source.device, dtype=source.dtype)
    xs = torch.linspace(-1.0, 1.0, w, device=source.device, dtype=source.dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    base = torch.stack((xx, yy), dim=-1)[None].expand(b, -1, -1, -1)

    dx = 2.0 * offset[:, 0] / max(w - 1, 1)
    dy = 2.0 * offset[:, 1] / max(h - 1, 1)
    delta = torch.stack((dx, dy), dim=-1)

    return F.grid_sample(
        source,
        base + delta,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )



def offset_xy(offset: torch.Tensor) -> Tuple[float, float]:
    t = offset.detach().float().cpu()
    return float(t[0, 0, 0, 0]), float(t[0, 1, 0, 0])



def draw_translation_decomposition(
    base: np.ndarray,
    coarse_offset: torch.Tensor,
    fine_offset: torch.Tensor,
    total_offset: torch.Tensor,
    feature_stride_original: float,
) -> np.ndarray:
    """Draw visual content motion (-sampling displacement) on original RGB."""
    image = ensure_bgr(base).copy()
    h, w = image.shape[:2]
    start = np.array([w / 2.0, h / 2.0], dtype=np.float32)

    dc = np.asarray(offset_xy(coarse_offset), dtype=np.float32)
    df = np.asarray(offset_xy(fine_offset), dtype=np.float32)
    dt = np.asarray(offset_xy(total_offset), dtype=np.float32)

    coarse_end = start - dc * float(feature_stride_original)
    final_end = start - dt * float(feature_stride_original)

    def pt(v):
        return int(round(float(v[0]))), int(round(float(v[1])))

    thickness = max(2, int(round(max(h, w) / 700.0)))

    cv2.arrowedLine(
        image, pt(start), pt(coarse_end),
        (255, 120, 0), thickness + 1, tipLength=0.12,
    )
    cv2.arrowedLine(
        image, pt(coarse_end), pt(final_end),
        (0, 165, 255), thickness + 1, tipLength=0.18,
    )
    cv2.arrowedLine(
        image, pt(start), pt(final_end),
        (0, 0, 255), max(2, thickness - 1), tipLength=0.10,
    )

    cv2.circle(image, pt(start), max(4, thickness * 2), (255, 255, 255), -1)
    cv2.circle(image, pt(coarse_end), max(4, thickness * 2), (255, 120, 0), -1)
    cv2.circle(image, pt(final_end), max(4, thickness * 2), (0, 0, 255), -1)

    legend_y = h - max(18, int(round(h * 0.025)))
    x = max(12, int(round(w * 0.012)))
    font_scale = max(0.45, min(1.0, w / 1900.0))

    for name, color in [
        ("coarse", (255, 120, 0)),
        ("fine", (0, 165, 255)),
        ("total", (0, 0, 255)),
    ]:
        cv2.line(image, (x, legend_y), (x + 34, legend_y), color, thickness)
        cv2.putText(
            image,
            name,
            (x + 42, legend_y + 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            max(1, thickness // 2),
            cv2.LINE_AA,
        )
        x += max(120, int(round(w * 0.085)))

    return image



def render_displacement_probability(
    probability: torch.Tensor,
    fine_offset: torch.Tensor,
    fine_confidence: Optional[torch.Tensor],
    canvas_hw: Tuple[int, int],
) -> np.ndarray:
    """Render the actual fine-search candidate distribution on native RGB canvas."""
    p = probability.detach().float().cpu()
    while p.ndim > 1:
        p = p[0]
    p = p.reshape(-1)

    k = int(p.numel())
    side = int(round(math.sqrt(k)))
    if side * side != k or side % 2 == 0:
        raise ValueError(
            "fine_probability must contain an odd square number of candidates, "
            f"got K={k}"
        )

    radius = (side - 1) // 2
    prob = p.numpy().reshape(side, side)

    height, width = int(canvas_hw[0]), int(canvas_hw[1])
    canvas = np.full((height, width, 3), 246, dtype=np.uint8)

    scale_ref = max(0.65, min(width / 1280.0, height / 720.0))
    left = int(round(180 * scale_ref))
    right = int(round(80 * scale_ref))
    top = int(round(120 * scale_ref))
    bottom = int(round(150 * scale_ref))

    usable_w = max(1, width - left - right)
    usable_h = max(1, height - top - bottom)
    cell = max(12, min(usable_w // side, usable_h // side))

    grid_w = cell * side
    grid_h = cell * side
    x0 = left + max(0, (usable_w - grid_w) // 2)
    y0 = top + max(0, (usable_h - grid_h) // 2)

    pmax = float(prob.max()) if prob.size else 0.0
    color_norm = (
        np.zeros_like(prob, dtype=np.float32)
        if pmax <= 1e-12
        else np.clip(prob / pmax, 0.0, 1.0).astype(np.float32)
    )

    best_flat = int(np.argmax(prob))
    best_row, best_col = divmod(best_flat, side)
    best_dx = best_col - radius
    best_dy = best_row - radius

    for row in range(side):
        for col in range(side):
            value = float(prob[row, col])
            color_value = int(round(float(color_norm[row, col]) * 255.0))
            color = cv2.applyColorMap(
                np.array([[color_value]], dtype=np.uint8),
                cv2.COLORMAP_TURBO,
            )[0, 0]
            color = tuple(int(v) for v in color.tolist())

            xa = x0 + col * cell
            ya = y0 + row * cell
            xb = xa + cell
            yb = ya + cell

            cv2.rectangle(canvas, (xa, ya), (xb, yb), color, -1)
            cv2.rectangle(canvas, (xa, ya), (xb, yb), (35, 35, 35), 1)

            if row == best_row and col == best_col:
                cv2.rectangle(
                    canvas,
                    (xa + 3, ya + 3),
                    (xb - 3, yb - 3),
                    (255, 255, 255),
                    max(2, int(round(3 * scale_ref))),
                )

            brightness = 0.114 * color[0] + 0.587 * color[1] + 0.299 * color[2]
            text_color = (20, 20, 20) if brightness > 145 else (255, 255, 255)
            label = f"{value:.3f}"
            font_scale = max(0.36, 0.52 * scale_ref)
            thickness = max(1, int(round(scale_ref)))
            (tw, th), _ = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                thickness,
            )
            tx = xa + max(3, (cell - tw) // 2)
            ty = ya + max(th + 3, (cell + th) // 2)
            cv2.putText(
                canvas,
                label,
                (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                text_color,
                thickness,
                cv2.LINE_AA,
            )

    axis_scale = max(0.42, 0.58 * scale_ref)
    axis_thickness = max(1, int(round(scale_ref)))

    for col in range(side):
        dx = col - radius
        label = f"{dx:+d}" if dx != 0 else "0"
        (tw, _), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, axis_scale, axis_thickness
        )
        cx = x0 + col * cell + cell // 2
        cv2.putText(
            canvas,
            label,
            (cx - tw // 2, y0 - max(12, int(round(18 * scale_ref)))),
            cv2.FONT_HERSHEY_SIMPLEX,
            axis_scale,
            (35, 35, 35),
            axis_thickness,
            cv2.LINE_AA,
        )

    for row in range(side):
        dy = row - radius
        label = f"{dy:+d}" if dy != 0 else "0"
        (tw, th), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, axis_scale, axis_thickness
        )
        cy = y0 + row * cell + cell // 2
        cv2.putText(
            canvas,
            label,
            (x0 - tw - max(12, int(round(24 * scale_ref))), cy + th // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            axis_scale,
            (35, 35, 35),
            axis_thickness,
            cv2.LINE_AA,
        )

    off = fine_offset.detach().float().cpu()
    fine_dx = float(off.reshape(off.shape[0], off.shape[1], -1)[0, 0, 0])
    fine_dy = float(off.reshape(off.shape[0], off.shape[1], -1)[0, 1, 0])

    conf_text = "N/A"
    if fine_confidence is not None:
        c = fine_confidence.detach().float().cpu().reshape(-1)
        if c.numel() > 0:
            conf_text = f"{float(c[0]):.4f}"

    info_y = min(
        height - max(55, int(round(65 * scale_ref))),
        y0 + grid_h + max(35, int(round(55 * scale_ref))),
    )

    line1 = (
        f"argmax: dx={best_dx:+d}, dy={best_dy:+d} P3 px, "
        f"p_max={float(prob[best_row, best_col]):.4f}"
    )
    line2 = (
        f"pred fine offset: dx={fine_dx:+.4f}, dy={fine_dy:+.4f} P3 px, "
        f"confidence={conf_text}"
    )

    cv2.putText(
        canvas,
        line1,
        (max(16, x0 - int(round(80 * scale_ref))), info_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        max(0.38, 0.55 * scale_ref),
        (30, 30, 30),
        max(1, int(round(scale_ref))),
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        line2,
        (
            max(16, x0 - int(round(80 * scale_ref))),
            info_y + max(24, int(round(34 * scale_ref))),
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        max(0.38, 0.55 * scale_ref),
        (30, 30, 30),
        max(1, int(round(scale_ref))),
        cv2.LINE_AA,
    )

    return canvas


# =============================================================================
# One-sample visualization
# =============================================================================


@torch.inference_mode()
def visualize_one(
    model,
    dataset,
    weights: str,
    index: int,
    device: torch.device,
    split: str,
    align_scale: int,
    bhlr_scale: int,
    conf: float,
    iou: float,
    max_det: int,
    save_dir: Optional[str],
):
    sample = dataset[index]
    batch = dataset.collate_fn([sample])
    batch = preprocess_batch_rsdt(batch, device)

    outputs = model(
        batch["rgb_img"],
        batch["tir_img"],
        batch["rgb_semantic_img"],
        return_features=True,
        return_rsdt_debug=True,
        return_alignment_debug=True,
    )

    align = get_alignment_debug(outputs, align_scale)
    bhlr_all, bhlr = get_bhlr_debug(outputs, bhlr_scale)

    # -------------------------------------------------------------------------
    # Original/native images and mapping metadata
    # -------------------------------------------------------------------------
    rgb_original = read_rgb_bgr(sample["rgb_path"])
    tir_original = read_tir_bgr(sample["tir_path"])

    rgb_orig_hw = rgb_original.shape[:2]
    tir_orig_hw = tir_original.shape[:2]

    rgb_high_lb = tensor_rgb_to_bgr(batch["rgb_img"][0])
    rgb_sem_lb = tensor_rgb_to_bgr(batch["rgb_semantic_img"][0])
    tir_lb = tensor_rgb_to_bgr(batch["tir_img"][0])

    high_net_hw = rgb_high_lb.shape[:2]
    sem_net_hw = rgb_sem_lb.shape[:2]
    tir_net_hw = tir_lb.shape[:2]

    rgb_high_ratio_pad = sample.get("rgb_ratio_pad", None)
    rgb_sem_ratio_pad = sample.get("rgb_semantic_ratio_pad", None)
    tir_ratio_pad = sample.get("tir_ratio_pad", None)

    # Original GT boxes from original labels.
    _, rgb_orig_norm = load_yolo_label(
        sample["rgb_label_path"],
        nc=1,
        strict=True,
        target_classes=(0,),
        remap_classes=True,
    )
    _, tir_orig_norm = load_yolo_label(
        sample["tir_label_path"],
        nc=1,
        strict=True,
        target_classes=(0,),
        remap_classes=True,
    )

    rgb_orig_boxes = normalized_xywh_to_xyxy(
        rgb_orig_norm,
        rgb_original.shape[1],
        rgb_original.shape[0],
    )
    tir_orig_boxes = normalized_xywh_to_xyxy(
        tir_orig_norm,
        tir_original.shape[1],
        tir_original.shape[0],
    )

    first_rgb_box = rgb_orig_boxes[0] if len(rgb_orig_boxes) else None

    # -------------------------------------------------------------------------
    # Scale features
    # -------------------------------------------------------------------------
    scale_to_index = outputs.get(
        "rsdt_scale_to_index",
        getattr(model, "rsdt_scale_to_index", {}),
    )
    pidx = int(get_scale_item(scale_to_index, align_scale))

    raw_tir_p3 = outputs["tir_backbone"][pidx]
    aligned_tir_p3 = outputs["aligned_tir_backbone"][pidx]

    enhanced_rgb_p3 = bhlr["enhanced_rgb"]
    usable_detail = bhlr["usable_detail"]
    gamma = bhlr["gamma"]

    # exact inverse of BHLR-v2 residual injection
    raw_rgb_p3 = enhanced_rgb_p3 - gamma * usable_detail

    fh, fw = raw_tir_p3.shape[-2:]

    # -------------------------------------------------------------------------
    # Save directory
    # -------------------------------------------------------------------------
    weights_path = Path(weights).expanduser().resolve()
    stem = Path(sample["rgb_path"]).stem

    if save_dir is None:
        out_dir = (
            weights_path.parent.parent
            / "visualize_rlsfa_bhlr_v2_internal"
            / f"{split}_{index:06d}_{stem}"
        )
    else:
        out_dir = Path(save_dir).expanduser().resolve()

    out_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # LetterBox-aware display canvases.
    # Network images keep their black padding instead of being stretched back
    # to original aspect ratio.
    # -------------------------------------------------------------------------
    semantic_norm = sample["rgb_semantic_bboxes"].detach().float().cpu().numpy()
    tir_norm = sample["tir_bboxes"].detach().float().cpu().numpy()

    rgb_sem_boxes = normalized_xywh_to_xyxy(
        semantic_norm, sem_net_hw[1], sem_net_hw[0]
    )
    tir_net_boxes = normalized_xywh_to_xyxy(
        tir_norm, tir_net_hw[1], tir_net_hw[0]
    )
    rgb_ref_boxes_on_tir_grid = normalized_xywh_to_xyxy(
        semantic_norm, tir_net_hw[1], tir_net_hw[0]
    )

    rgb_sem_canvas = black_letterbox_padding(
        rgb_sem_lb, orig_hw=rgb_orig_hw, ratio_pad=rgb_sem_ratio_pad
    )
    tir_canvas = black_letterbox_padding(
        tir_lb, orig_hw=tir_orig_hw, ratio_pad=tir_ratio_pad
    )

    # -------------------------------------------------------------------------
    # INPUT 00-03
    # -------------------------------------------------------------------------
    save_native(
        out_dir, 0, "rgb_original_gt",
        draw_boxes(rgb_original, rgb_orig_boxes),
        "Original RGB + GT",
        f"native={rgb_original.shape[1]}x{rgb_original.shape[0]}",
    )

    save_native(
        out_dir, 1, "tir_original_gt",
        draw_boxes(tir_original, tir_orig_boxes),
        "Original TIR + GT",
        f"native={tir_original.shape[1]}x{tir_original.shape[0]}",
    )

    rgb_sem_restored = unletterbox_image(
        rgb_sem_lb,
        orig_hw=rgb_orig_hw,
        ratio_pad=rgb_sem_ratio_pad,
    )
    save_native(
        out_dir, 2, "rgb_semantic_restored_gt",
        draw_boxes(rgb_sem_restored, rgb_orig_boxes),
        "RGB semantic input restored to original RGB size",
        f"network={sem_net_hw[1]}x{sem_net_hw[0]} -> native={rgb_original.shape[1]}x{rgb_original.shape[0]}",
    )

    tir_restored = unletterbox_image(
        tir_lb,
        orig_hw=tir_orig_hw,
        ratio_pad=tir_ratio_pad,
    )
    save_native(
        out_dir, 3, "tir_network_restored_gt",
        draw_boxes(tir_restored, tir_orig_boxes),
        "TIR network input restored to native TIR size",
        f"network={tir_net_hw[1]}x{tir_net_hw[0]} -> native={tir_original.shape[1]}x{tir_original.shape[0]}",
    )

    # -------------------------------------------------------------------------
    # RLSFA 04-20 -- every quantity is standalone
    # -------------------------------------------------------------------------
    rgb_p3_overlay = draw_boxes(
        mapped_overlay(
            rgb_original,
            tensor_feature_map(raw_rgb_p3),
            network_hw=sem_net_hw,
            ratio_pad=rgb_sem_ratio_pad,
            alpha=0.52,
        ),
        rgb_orig_boxes,
    )
    save_native(
        out_dir, 4, "rgb_p3_before_alignment",
        rgb_p3_overlay,
        f"RGB P{align_scale} response before RLSFA",
        f"feature={tuple(raw_rgb_p3.shape)}",
    )

    tir_p3_overlay = draw_boxes(
        mapped_overlay(
            tir_original,
            tensor_feature_map(raw_tir_p3),
            network_hw=tir_net_hw,
            ratio_pad=tir_ratio_pad,
            alpha=0.52,
        ),
        tir_orig_boxes,
    )
    save_native(
        out_dir, 5, "tir_p3_before_alignment",
        tir_p3_overlay,
        f"Raw TIR P{align_scale} response before RLSFA",
        f"feature={tuple(raw_tir_p3.shape)}",
    )

    targetness_tensor = debug_get(align, "targetness", "coarse_targetness")
    targetness = tensor_single_map(targetness_tensor)
    save_native(
        out_dir, 6, "coarse_targetness",
        draw_boxes(
            network_feature_overlay(
                rgb_sem_canvas,
                targetness,
                orig_hw=rgb_orig_hw,
                ratio_pad=rgb_sem_ratio_pad,
                alpha=0.38,
                mode="fixed",
                vmin=0.0,
                vmax=1.0,
            ),
            rgb_sem_boxes,
        ),
        "RLSFA coarse targetness over RGB background",
        f"fixed [0,1]; min={targetness.min():.4f}, mean={targetness.mean():.4f}, max={targetness.max():.4f}",
    )

    coarse_offset = debug_get(align, "coarse_offset")
    fine_offset = debug_get(align, "fine_offset")
    total_offset = debug_get(align, "total_offset")

    coarse_dx, coarse_dy = offset_xy(coarse_offset)
    fine_dx, fine_dy = offset_xy(fine_offset)
    total_dx, total_dy = offset_xy(total_offset)

    offset_annotation = (
        f"Δcoarse=({coarse_dx:+.3f},{coarse_dy:+.3f}) | "
        f"Δfine=({fine_dx:+.3f},{fine_dy:+.3f}) | "
        f"Δtotal=({total_dx:+.3f},{total_dy:+.3f}) P3 px [sampling]"
    )

    # 07: 640x640 TIR network canvas; effective TIR content remains 640x512.
    save_native(
        out_dir, 7, "tir_before_coarse_rgb_grid",
        draw_boxes(tir_canvas, rgb_ref_boxes_on_tir_grid, label="RGB GT"),
        "TIR before coarse translation",
        f"canvas={tir_net_hw[1]}x{tir_net_hw[0]}; valid={tir_orig_hw[1]}x{tir_orig_hw[0]}",
        annotation=offset_annotation,
    )

    # 08: translate only inside the valid 640x512 TIR region; black bars stay fixed.
    coarse_preview_canvas = warp_inside_letterbox_content(
        tir_canvas,
        coarse_offset,
        feature_hw=(fh, fw),
        orig_hw=tir_orig_hw,
        ratio_pad=tir_ratio_pad,
    )
    save_native(
        out_dir, 8, "tir_after_coarse",
        draw_boxes(coarse_preview_canvas, rgb_ref_boxes_on_tir_grid, label="RGB GT"),
        "TIR after RLSFA coarse translation",
        f"canvas={tir_net_hw[1]}x{tir_net_hw[0]}; valid={tir_orig_hw[1]}x{tir_orig_hw[0]}",
        annotation=offset_annotation,
    )

    # RGB spatial/frequency/reliability/reliable structure
    rgb_spatial = debug_get(align, "rgb_spatial", "rgb_spatial_structure")
    rgb_frequency = debug_get(align, "rgb_frequency", "rgb_local_frequency")
    rgb_reliability_t = debug_get(
        align, "rgb_frequency_reliability", "rgb_reliability"
    )
    rgb_reliable = debug_get(align, "rgb_reliable_structure")

    save_native(
        out_dir, 9, "rgb_spatial_structure",
        draw_boxes(
            mapped_overlay(
                rgb_original,
                tensor_feature_map(rgb_spatial),
                network_hw=sem_net_hw,
                ratio_pad=rgb_sem_ratio_pad,
                alpha=0.55,
            ),
            rgb_orig_boxes,
        ),
        "RLSFA RGB spatial structure",
        f"feature={tuple(rgb_spatial.shape)}",
        annotation=offset_annotation,
    )

    save_native(
        out_dir, 10, "rgb_local_frequency",
        draw_boxes(
            mapped_overlay(
                rgb_original,
                tensor_feature_map(rgb_frequency),
                network_hw=sem_net_hw,
                ratio_pad=rgb_sem_ratio_pad,
                alpha=0.55,
            ),
            rgb_orig_boxes,
        ),
        "RLSFA RGB local Window-FFT response",
        f"feature={tuple(rgb_frequency.shape)}",
        annotation=offset_annotation,
    )

    rgb_rel = tensor_single_map(rgb_reliability_t)
    save_native(
        out_dir, 11, "rgb_frequency_reliability",
        draw_boxes(
            network_feature_overlay(
                rgb_sem_canvas,
                rgb_rel,
                orig_hw=rgb_orig_hw,
                ratio_pad=rgb_sem_ratio_pad,
                alpha=0.38,
                mode="fixed",
                vmin=0.0,
                vmax=1.0,
            ),
            rgb_sem_boxes,
        ),
        "RLSFA RGB frequency reliability Q_R over RGB background",
        f"fixed [0,1]; min={rgb_rel.min():.4f}, mean={rgb_rel.mean():.4f}, max={rgb_rel.max():.4f}",
        annotation=offset_annotation,
    )

    save_native(
        out_dir, 12, "rgb_reliable_structure",
        draw_boxes(
            mapped_overlay(
                rgb_original,
                tensor_feature_map(rgb_reliable),
                network_hw=sem_net_hw,
                ratio_pad=rgb_sem_ratio_pad,
                alpha=0.55,
            ),
            rgb_orig_boxes,
        ),
        "RLSFA reliable RGB structure B_R",
        "B_R = Q_R * F_R + (1-Q_R) * S_R",
        annotation=offset_annotation,
    )

    # TIR structural branch after coarse alignment.
    # Keep TIR network geometry: 640x640 canvas, effective 640x512 region.
    tir_spatial = debug_get(align, "tir_spatial", "tir_spatial_structure")
    tir_frequency = debug_get(align, "tir_frequency", "tir_local_frequency")
    tir_reliability_t = debug_get(
        align, "tir_frequency_reliability", "tir_reliability"
    )
    tir_reliable = debug_get(align, "tir_reliable_structure")

    save_native(
        out_dir, 13, "tir_spatial_structure",
        draw_boxes(
            network_feature_overlay(
                coarse_preview_canvas,
                tensor_feature_map(tir_spatial),
                orig_hw=tir_orig_hw,
                ratio_pad=tir_ratio_pad,
                alpha=0.50,
            ),
            rgb_ref_boxes_on_tir_grid,
            label="RGB GT",
        ),
        "RLSFA TIR spatial structure after coarse alignment",
        f"canvas={tir_net_hw[1]}x{tir_net_hw[0]}; valid={tir_orig_hw[1]}x{tir_orig_hw[0]}",
        annotation=offset_annotation,
    )

    save_native(
        out_dir, 14, "tir_local_frequency",
        draw_boxes(
            network_feature_overlay(
                coarse_preview_canvas,
                tensor_feature_map(tir_frequency),
                orig_hw=tir_orig_hw,
                ratio_pad=tir_ratio_pad,
                alpha=0.50,
            ),
            rgb_ref_boxes_on_tir_grid,
            label="RGB GT",
        ),
        "RLSFA TIR local high-frequency response",
        f"canvas={tir_net_hw[1]}x{tir_net_hw[0]}; valid={tir_orig_hw[1]}x{tir_orig_hw[0]}",
        annotation=offset_annotation,
    )

    tir_rel = tensor_single_map(tir_reliability_t)
    save_native(
        out_dir, 15, "tir_frequency_reliability",
        draw_boxes(
            network_feature_overlay(
                coarse_preview_canvas,
                tir_rel,
                orig_hw=tir_orig_hw,
                ratio_pad=tir_ratio_pad,
                alpha=0.38,
                mode="fixed",
                vmin=0.0,
                vmax=1.0,
            ),
            rgb_ref_boxes_on_tir_grid,
            label="RGB GT",
        ),
        "RLSFA TIR frequency reliability Q_T over TIR background",
        f"fixed [0,1]; canvas={tir_net_hw[1]}x{tir_net_hw[0]}; valid={tir_orig_hw[1]}x{tir_orig_hw[0]}",
        annotation=offset_annotation,
    )

    save_native(
        out_dir, 16, "tir_reliable_structure",
        draw_boxes(
            network_feature_overlay(
                coarse_preview_canvas,
                tensor_feature_map(tir_reliable),
                orig_hw=tir_orig_hw,
                ratio_pad=tir_ratio_pad,
                alpha=0.50,
            ),
            rgb_ref_boxes_on_tir_grid,
            label="RGB GT",
        ),
        "RLSFA reliable TIR structure B_T",
        "B_T = Q_T * F_T + (1-Q_T) * S_T",
        annotation=offset_annotation,
    )

    fine_probability = debug_get(align, "fine_probability")
    fine_confidence = debug_get(align, "fine_correlation_confidence")
    fine_prob_img = render_displacement_probability(
        probability=fine_probability,
        fine_offset=fine_offset,
        fine_confidence=fine_confidence,
        canvas_hw=rgb_orig_hw,
    )
    save_native(
        out_dir, 17, "fine_displacement_probability",
        fine_prob_img,
        "RLSFA fine residual displacement probability",
        "rows=dy, columns=dx; white box=max-probability discrete candidate",
        annotation=offset_annotation,
    )

    # Convert one P3 pixel to native RGB pixels.
    (gain_x_sem, _), _ = parse_ratio_pad(
        rgb_sem_ratio_pad,
        orig_hw=rgb_orig_hw,
        net_hw=sem_net_hw,
    )
    p3_stride_network = float(sem_net_hw[1]) / float(fw)
    feature_stride_original = p3_stride_network / max(float(gain_x_sem), 1e-12)

    offset_img = draw_translation_decomposition(
        rgb_original,
        coarse_offset=coarse_offset,
        fine_offset=fine_offset,
        total_offset=total_offset,
        feature_stride_original=feature_stride_original,
    )
    offset_img = draw_boxes(offset_img, rgb_orig_boxes)
    save_native(
        out_dir, 18, "total_offset",
        offset_img,
        "RLSFA translation decomposition",
        "blue=coarse, orange=fine residual, red=total; arrows show visual content motion",
        annotation=offset_annotation,
    )

    final_preview_canvas = warp_inside_letterbox_content(
        tir_canvas,
        total_offset,
        feature_hw=(fh, fw),
        orig_hw=tir_orig_hw,
        ratio_pad=tir_ratio_pad,
    )
    save_native(
        out_dir, 19, "tir_after_final_rlsfa",
        draw_boxes(final_preview_canvas, rgb_ref_boxes_on_tir_grid, label="RGB GT"),
        "TIR after final RLSFA translation",
        f"canvas={tir_net_hw[1]}x{tir_net_hw[0]}; valid={tir_orig_hw[1]}x{tir_orig_hw[0]}",
        annotation=offset_annotation,
    )

    aligned_p3_overlay = draw_boxes(
        network_feature_overlay(
            final_preview_canvas,
            tensor_feature_map(aligned_tir_p3),
            orig_hw=tir_orig_hw,
            ratio_pad=tir_ratio_pad,
            alpha=0.48,
        ),
        rgb_ref_boxes_on_tir_grid,
        label="RGB GT",
    )
    save_native(
        out_dir, 20, "aligned_tir_p3",
        aligned_p3_overlay,
        f"Aligned TIR P{align_scale} response",
        f"feature={tuple(aligned_tir_p3.shape)}; valid={tir_orig_hw[1]}x{tir_orig_hw[0]}",
        annotation=offset_annotation,
    )

    # -------------------------------------------------------------------------
    # BHLR-v2 21-44
    # -------------------------------------------------------------------------
    guidance_source = str(bhlr.get("guidance_source", "unknown"))

    save_native(
        out_dir, 21, "rgb_high_gt",
        draw_boxes(rgb_original, rgb_orig_boxes),
        "BHLR-v2 high-resolution RGB source + GT",
        f"native={rgb_original.shape[1]}x{rgb_original.shape[0]}",
    )

    reconstructed_high = bhlr_all.get("reconstructed_high", None)
    residual_high = bhlr_all.get("resolution_residual", None)
    residual_s2d = bhlr_all.get("residual_s2d", None)
    detail_bank = bhlr_all.get("detail_bank", bhlr_all.get("detail_raw", None))

    if reconstructed_high is None or residual_high is None or detail_bank is None:
        raise KeyError(
            "BHLR-v2 debug must expose reconstructed_high, resolution_residual, "
            "and detail_bank/detail_raw. "
            f"Available={list(bhlr_all.keys())}"
        )

    reconstructed_lb = tensor_rgb_to_bgr(reconstructed_high[0])
    reconstructed_native = unletterbox_image(
        reconstructed_lb,
        orig_hw=rgb_orig_hw,
        ratio_pad=rgb_high_ratio_pad,
    )
    save_native(
        out_dir, 22, "rgb_semantic_upscaled",
        draw_boxes(reconstructed_native, rgb_orig_boxes),
        "RGB semantic reconstruction at native RGB resolution",
        "Upsampled low-resolution RGB used to define the lost-detail residual",
    )

    residual_map = tensor_feature_map(residual_high, mode="mean_abs")
    residual_heat = mapped_heatmap(
        residual_map,
        network_hw=high_net_hw,
        orig_hw=rgb_orig_hw,
        ratio_pad=rgb_high_ratio_pad,
        mode="robust",
    )
    residual_heat = draw_boxes(residual_heat, rgb_orig_boxes)
    save_native(
        out_dir, 23, "highres_residual",
        residual_heat,
        "BHLR-v2 high-resolution lost-detail residual",
        "|RGB_high - upsample(RGB_semantic)|; background detail is expected",
    )

    residual_overlay = draw_boxes(
        mapped_overlay(
            rgb_original,
            residual_map,
            network_hw=high_net_hw,
            ratio_pad=rgb_high_ratio_pad,
            alpha=0.50,
        ),
        rgb_orig_boxes,
    )
    save_native(
        out_dir, 24, "highres_residual_overlay",
        residual_overlay,
        "High-resolution residual over original RGB",
        "Physical source of detail lost by semantic downsampling",
    )

    save_native(
        out_dir, 25, "uav_high_zoom",
        crop_to_native_canvas(
            draw_boxes(rgb_original, rgb_orig_boxes),
            first_rgb_box,
            canvas_hw=rgb_orig_hw,
            expand=5.0,
            min_size=160,
        ),
        "UAV zoom: original high-resolution RGB",
        "Standalone zoom; output canvas keeps original RGB pixel dimensions",
    )

    save_native(
        out_dir, 26, "uav_semantic_zoom",
        crop_to_native_canvas(
            draw_boxes(reconstructed_native, rgb_orig_boxes),
            first_rgb_box,
            canvas_hw=rgb_orig_hw,
            expand=5.0,
            min_size=160,
        ),
        "UAV zoom: reconstructed semantic RGB",
        "Shows what remains after low-resolution semantic input is upsampled",
    )

    save_native(
        out_dir, 27, "uav_residual_zoom",
        crop_to_native_canvas(
            residual_overlay,
            first_rgb_box,
            canvas_hw=rgb_orig_hw,
            expand=5.0,
            min_size=160,
        ),
        "UAV zoom: high-resolution lost-detail residual",
        "Standalone zoom at original RGB output resolution",
    )

    # residual_s2d is deterministic; recompute only if an older debug path omitted it.
    if residual_s2d is None:
        ratio_h = int(round(high_net_hw[0] / sem_net_hw[0]))
        ratio_w = int(round(high_net_hw[1] / sem_net_hw[1]))
        if ratio_h != ratio_w or ratio_h < 1:
            raise ValueError(
                "Cannot infer PixelUnshuffle ratio from "
                f"high={high_net_hw}, semantic={sem_net_hw}"
            )
        residual_s2d = F.pixel_unshuffle(
            residual_high,
            downscale_factor=ratio_h,
        )

    save_native(
        out_dir, 28, "pixel_unshuffle_detail",
        draw_boxes(
            mapped_overlay(
                rgb_original,
                tensor_feature_map(residual_s2d),
                network_hw=sem_net_hw,
                ratio_pad=rgb_sem_ratio_pad,
                alpha=0.54,
            ),
            rgb_orig_boxes,
        ),
        "BHLR-v2 PixelUnshuffle residual tensor",
        f"shape={tuple(residual_s2d.shape)}; local high-resolution samples moved to channels",
    )

    save_native(
        out_dir, 29, "detail_bank",
        draw_boxes(
            mapped_overlay(
                rgb_original,
                tensor_feature_map(detail_bank),
                network_hw=sem_net_hw,
                ratio_pad=rgb_sem_ratio_pad,
                alpha=0.54,
            ),
            rgb_orig_boxes,
        ),
        "BHLR-v2 lightweight shared detail bank",
        f"shape={tuple(detail_bank.shape)}; includes stride-2 detail compression",
    )

    detail_avg = bhlr.get("detail_avg", None)
    detail_max = bhlr.get("detail_max", None)
    detail_scale = bhlr.get("detail_scale", None)

    target_hw = raw_rgb_p3.shape[-2:]
    if detail_avg is None:
        detail_avg = F.adaptive_avg_pool2d(detail_bank, target_hw)
    if detail_max is None:
        detail_max = F.adaptive_max_pool2d(detail_bank, target_hw)
    if detail_scale is None:
        raise KeyError(
            "BHLR-v2 scale debug missing detail_scale. "
            f"Available={list(bhlr.keys())}"
        )

    for number, slug, tensor, title in [
        (30, "detail_avg_pool_p3", detail_avg, "BHLR-v2 average-pooled detail -> P3"),
        (31, "detail_max_pool_p3", detail_max, "BHLR-v2 max-pooled detail -> P3"),
        (32, "detail_projection_p3", detail_scale, "BHLR-v2 projected high-resolution detail D3"),
    ]:
        save_native(
            out_dir, number, slug,
            draw_boxes(
                mapped_overlay(
                    rgb_original,
                    tensor_feature_map(tensor),
                    network_hw=sem_net_hw,
                    ratio_pad=rgb_sem_ratio_pad,
                    alpha=0.54,
                ),
                rgb_orig_boxes,
            ),
            title,
            f"shape={tuple(tensor.shape)}",
        )

    # V2 guidance
    bhlr_targetness_t = bhlr["targetness"]
    bhlr_targetness = tensor_single_map(bhlr_targetness_t)
    save_native(
        out_dir, 33, "bhlr_guidance_targetness",
        draw_boxes(
            network_feature_overlay(
                rgb_sem_canvas,
                bhlr_targetness,
                orig_hw=rgb_orig_hw,
                ratio_pad=rgb_sem_ratio_pad,
                alpha=0.38,
                mode="fixed",
                vmin=0.0,
                vmax=1.0,
            ),
            rgb_sem_boxes,
        ),
        "BHLR-v2 guidance targetness over RGB background",
        f"source={guidance_source}; fixed [0,1]; mean={bhlr_targetness.mean():.4f}",
    )

    bhlr_rgb_structure = bhlr["rgb_structure"]
    save_native(
        out_dir, 34, "bhlr_rgb_structure",
        draw_boxes(
            mapped_overlay(
                rgb_original,
                tensor_feature_map(bhlr_rgb_structure),
                network_hw=sem_net_hw,
                ratio_pad=rgb_sem_ratio_pad,
                alpha=0.55,
            ),
            rgb_orig_boxes,
        ),
        "BHLR-v2 RGB guidance structure",
        f"guidance_source={guidance_source}; shape={tuple(bhlr_rgb_structure.shape)}",
    )

    bhlr_tir_structure = bhlr["tir_structure"]

    alignment_enabled = bool(outputs.get("alignment_enabled", False))
    tir_for_bhlr_aligned = alignment_enabled and bool(
        getattr(model, "alignment_use_for_rsdt", True)
    )

    if guidance_source == "external" or tir_for_bhlr_aligned:
        bhlr_tir_base = final_preview_canvas
        bhlr_tir_boxes = rgb_ref_boxes_on_tir_grid
        bhlr_tir_label = "RGB GT"
    else:
        bhlr_tir_base = tir_canvas
        bhlr_tir_boxes = tir_net_boxes
        bhlr_tir_label = "UAV GT"

    bhlr_tir_structure_img = draw_boxes(
        network_feature_overlay(
            bhlr_tir_base,
            tensor_feature_map(bhlr_tir_structure),
            orig_hw=tir_orig_hw,
            ratio_pad=tir_ratio_pad,
            alpha=0.48,
        ),
        bhlr_tir_boxes,
        label=bhlr_tir_label,
    )
    save_native(
        out_dir, 35, "bhlr_tir_structure",
        bhlr_tir_structure_img,
        "BHLR-v2 TIR guidance structure",
        f"guidance_source={guidance_source}; canvas={tir_net_hw[1]}x{tir_net_hw[0]}; valid={tir_orig_hw[1]}x{tir_orig_hw[0]}",
    )

    selector_logit_t = bhlr["selector_logit"]
    selector_logit = tensor_single_map(selector_logit_t)
    save_native(
        out_dir, 36, "selector_logit",
        draw_boxes(
            network_feature_overlay(
                rgb_sem_canvas,
                selector_logit,
                orig_hw=rgb_orig_hw,
                ratio_pad=rgb_sem_ratio_pad,
                alpha=0.38,
                mode="signed",
            ),
            rgb_sem_boxes,
        ),
        "BHLR-v2 detail-selector logit Z3 over RGB background",
        f"signed display; min={selector_logit.min():+.4f}, mean={selector_logit.mean():+.4f}, max={selector_logit.max():+.4f}",
    )

    selector_multiplier_t = bhlr["selector_multiplier"]
    selector_multiplier = tensor_single_map(selector_multiplier_t)
    save_native(
        out_dir, 37, "selector_multiplier",
        draw_boxes(
            network_feature_overlay(
                rgb_sem_canvas,
                selector_multiplier,
                orig_hw=rgb_orig_hw,
                ratio_pad=rgb_sem_ratio_pad,
                alpha=0.38,
                mode="fixed",
                vmin=0.0,
                vmax=2.0,
            ),
            rgb_sem_boxes,
        ),
        "BHLR-v2 centered detail multiplier M3 over RGB background",
        f"fixed [0,2]; min={selector_multiplier.min():.4f}, mean={selector_multiplier.mean():.4f}, max={selector_multiplier.max():.4f}",
    )

    save_native(
        out_dir, 38, "selector_multiplier_overlay",
        draw_boxes(
            network_feature_overlay(
                rgb_sem_canvas,
                selector_multiplier,
                orig_hw=rgb_orig_hw,
                ratio_pad=rgb_sem_ratio_pad,
                alpha=0.38,
                mode="fixed",
                vmin=0.0,
                vmax=2.0,
            ),
            rgb_sem_boxes,
        ),
        "BHLR-v2 detail multiplier over RGB background",
        "M3<1 suppresses detail; M3=1 keeps; M3>1 enhances",
    )

    save_native(
        out_dir, 39, "usable_detail",
        draw_boxes(
            network_feature_overlay(
                rgb_sem_canvas,
                tensor_feature_map(usable_detail),
                orig_hw=rgb_orig_hw,
                ratio_pad=rgb_sem_ratio_pad,
                alpha=0.40,
                mode="robust",
            ),
            rgb_sem_boxes,
        ),
        "BHLR-v2 usable high-resolution detail over RGB background",
        "D_use = M3 * D3",
    )

    save_native(
        out_dir, 40, "usable_detail_overlay",
        draw_boxes(
            network_feature_overlay(
                rgb_sem_canvas,
                tensor_feature_map(usable_detail),
                orig_hw=rgb_orig_hw,
                ratio_pad=rgb_sem_ratio_pad,
                alpha=0.40,
            ),
            rgb_sem_boxes,
        ),
        "BHLR-v2 usable detail over RGB background",
        "Selected lost-detail tensor before gamma residual injection",
    )

    before_map = tensor_feature_map(raw_rgb_p3)
    after_map = tensor_feature_map(enhanced_rgb_p3)
    shared_vmin, shared_vmax = shared_robust_range(before_map, after_map)

    save_native(
        out_dir, 41, "rgb_p3_before_bhlr",
        draw_boxes(
            network_feature_overlay(
                rgb_sem_canvas,
                before_map,
                orig_hw=rgb_orig_hw,
                ratio_pad=rgb_sem_ratio_pad,
                alpha=0.50,
                mode="fixed",
                vmin=shared_vmin,
                vmax=shared_vmax,
            ),
            rgb_sem_boxes,
        ),
        "RGB P3 before BHLR-v2",
        f"shared color range=[{shared_vmin:.6f}, {shared_vmax:.6f}]",
    )

    save_native(
        out_dir, 42, "rgb_p3_after_bhlr",
        draw_boxes(
            network_feature_overlay(
                rgb_sem_canvas,
                after_map,
                orig_hw=rgb_orig_hw,
                ratio_pad=rgb_sem_ratio_pad,
                alpha=0.50,
                mode="fixed",
                vmin=shared_vmin,
                vmax=shared_vmax,
            ),
            rgb_sem_boxes,
        ),
        "RGB P3 after BHLR-v2",
        f"shared color range=[{shared_vmin:.6f}, {shared_vmax:.6f}]; gamma={float(gamma.detach().float().cpu()):+.6f}",
    )

    enhancement = enhanced_rgb_p3 - raw_rgb_p3
    save_native(
        out_dir, 43, "bhlr_enhancement",
        draw_boxes(
            network_feature_overlay(
                rgb_sem_canvas,
                tensor_feature_map(enhancement),
                orig_hw=rgb_orig_hw,
                ratio_pad=rgb_sem_ratio_pad,
                alpha=0.52,
            ),
            rgb_sem_boxes,
        ),
        "Actual BHLR-v2 feature injection",
        "F_after - F_before = tanh(gamma) * D_use",
    )

    # TRUE increase/decrease: after mean-abs response minus before mean-abs response.
    response_delta = after_map - before_map
    delta_img, response_delta_limit = signed_response_change_overlay(
        rgb_sem_canvas,
        response_delta,
        orig_hw=rgb_orig_hw,
        ratio_pad=rgb_sem_ratio_pad,
        alpha_max=0.72,
        percentile=99.0,
    )
    delta_img = draw_boxes(delta_img, rgb_sem_boxes)
    save_native(
        out_dir, 44, "bhlr_response_increase_decrease",
        delta_img,
        "BHLR-v2 true response increase/decrease",
        f"red=increase; blue=decrease; symmetric 99% limit={response_delta_limit:.6f}",
    )

    # -------------------------------------------------------------------------
    # DETECTION 45 -- map semantic prediction boxes back to original RGB
    # -------------------------------------------------------------------------
    pred_list = base_val.postprocess_predictions(
        preds=outputs["pred"],
        model=model,
        conf_thres=0.001,
        iou_thres=iou,
        max_det=max_det,
    )
    pred = pred_list[0] if pred_list else None

    final_image = draw_boxes(rgb_original, rgb_orig_boxes, label="GT")

    pred_boxes_original = np.zeros((0, 4), dtype=np.float32)
    pred_conf = np.zeros((0,), dtype=np.float32)

    if pred is not None:
        pred_boxes = pred.get("bboxes", None)
        pred_conf_t = pred.get("conf", None)
        if pred_boxes is not None and pred_conf_t is not None:
            pred_boxes_np = pred_boxes.detach().float().cpu().numpy()
            pred_conf = pred_conf_t.detach().float().cpu().numpy()
            pred_boxes_original = scale_xyxy_from_letterbox_to_original(
                pred_boxes_np,
                orig_hw=rgb_orig_hw,
                net_hw=sem_net_hw,
                ratio_pad=rgb_sem_ratio_pad,
            )
            final_image = draw_predictions_xyxy(
                final_image,
                pred_boxes_original,
                pred_conf,
                conf_min=conf,
            )

    save_native(
        out_dir, 45, "prediction_vs_gt",
        final_image,
        "Final detection on original RGB",
        "red=GT, green=prediction; prediction boxes restored from semantic grid",
    )

    # -------------------------------------------------------------------------
    # Metadata
    # -------------------------------------------------------------------------
    fine_prob_cpu = fine_probability.detach().float().cpu()
    if fine_prob_cpu.ndim > 1:
        fine_prob_cpu = fine_prob_cpu[0]
    fine_prob_cpu = fine_prob_cpu.reshape(-1)
    fine_side = int(round(math.sqrt(int(fine_prob_cpu.numel()))))
    fine_radius_vis = (fine_side - 1) // 2
    fine_best = int(torch.argmax(fine_prob_cpu).item())
    fine_best_row, fine_best_col = divmod(fine_best, fine_side)
    fine_argmax_dx = int(fine_best_col - fine_radius_vis)
    fine_argmax_dy = int(fine_best_row - fine_radius_vis)

    metadata = {
        "weights": str(weights_path),
        "split": split,
        "dataset_index": int(index),
        "rgb_path": str(sample["rgb_path"]),
        "tir_path": str(sample["tir_path"]),
        "export_policy": {
            "group_panels": False,
            "overview": False,
            "title_strip": False,
            "letterbox_removed": False,
            "rgb_semantic_canvas": [int(sem_net_hw[1]), int(sem_net_hw[0])],
            "tir_canvas": [int(tir_net_hw[1]), int(tir_net_hw[0])],
            "tir_effective_region": [int(tir_orig_hw[1]), int(tir_orig_hw[0])],
            "visual_padding_color": "black",
        },
        "rlsfa": {
            "coarse_dx": coarse_dx,
            "coarse_dy": coarse_dy,
            "fine_dx": fine_dx,
            "fine_dy": fine_dy,
            "total_dx": total_dx,
            "total_dy": total_dy,
            "fine_confidence": float(fine_confidence.detach().float().cpu().reshape(-1)[0]),
            "fine_probability_max": float(fine_probability.detach().float().cpu().max()),
            "fine_argmax_dx": fine_argmax_dx,
            "fine_argmax_dy": fine_argmax_dy,
            "rgb_reliability": {
                "min": float(rgb_rel.min()),
                "mean": float(rgb_rel.mean()),
                "max": float(rgb_rel.max()),
            },
            "tir_reliability": {
                "min": float(tir_rel.min()),
                "mean": float(tir_rel.mean()),
                "max": float(tir_rel.max()),
            },
            "targetness": {
                "min": float(targetness.min()),
                "mean": float(targetness.mean()),
                "max": float(targetness.max()),
            },
        },
        "bhlr_v2": {
            "guidance_source": guidance_source,
            "gamma": float(gamma.detach().float().cpu()),
            "targetness": {
                "min": float(bhlr_targetness.min()),
                "mean": float(bhlr_targetness.mean()),
                "max": float(bhlr_targetness.max()),
            },
            "selector_logit": {
                "min": float(selector_logit.min()),
                "mean": float(selector_logit.mean()),
                "max": float(selector_logit.max()),
            },
            "selector_multiplier": {
                "min": float(selector_multiplier.min()),
                "mean": float(selector_multiplier.mean()),
                "max": float(selector_multiplier.max()),
            },
            "usable_detail_mean_abs": float(usable_detail.abs().mean().detach().cpu()),
            "actual_enhancement_mean_abs": float(enhancement.abs().mean().detach().cpu()),
            "shared_before_after_color_range": {
                "vmin": float(shared_vmin),
                "vmax": float(shared_vmax),
            },
            "response_delta": {
                "min": float(response_delta.min()),
                "mean": float(response_delta.mean()),
                "max": float(response_delta.max()),
                "symmetric_display_limit_99pct": float(response_delta_limit),
            },
            "residual_mean_abs": float(residual_high.abs().mean().detach().cpu()),
            "detail_bank_shape": list(detail_bank.shape),
        },
        "detection": {
            "num_predictions_before_display_threshold": int(len(pred_conf)),
            "display_conf_threshold": float(conf),
            "num_displayed_predictions": int(np.sum(pred_conf >= conf)) if len(pred_conf) else 0,
        },
    }

    # Single-target translation GT in P3 feature-grid coordinates.
    semantic_norm = sample["rgb_semantic_bboxes"].detach().float().cpu().numpy()
    tir_norm = sample["tir_bboxes"].detach().float().cpu().numpy()

    if semantic_norm.shape[0] == 1 and tir_norm.shape[0] == 1:
        gt_dx = float((tir_norm[0, 0] - semantic_norm[0, 0]) * fw)
        gt_dy = float((tir_norm[0, 1] - semantic_norm[0, 1]) * fh)
        metadata["rlsfa"]["gt_dx"] = gt_dx
        metadata["rlsfa"]["gt_dy"] = gt_dy
        metadata["rlsfa"]["coarse_translation_error"] = float(
            math.hypot(coarse_dx - gt_dx, coarse_dy - gt_dy)
        )
        metadata["rlsfa"]["final_translation_error"] = float(
            math.hypot(total_dx - gt_dx, total_dy - gt_dy)
        )

    with (out_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"[OK] index={index} -> {out_dir}")
    print(f"     guidance_source={guidance_source}")
    print(f"     RGB-native={rgb_original.shape[1]}x{rgb_original.shape[0]}")
    print(f"     TIR-native={tir_original.shape[1]}x{tir_original.shape[0]}")
    _tx1, _ty1, _tx2, _ty2 = letterbox_content_bounds(
        tir_orig_hw, tir_net_hw, tir_ratio_pad
    )
    print(
        f"     TIR-canvas={tir_net_hw[1]}x{tir_net_hw[0]}, "
        f"valid={_tx2-_tx1}x{_ty2-_ty1}, "
        f"bounds=(x:{_tx1}:{_tx2}, y:{_ty1}:{_ty2})"
    )
    print("     standalone JPGs=46; grouped panels=0; overview=0; white title strip=0")


# =============================================================================
# CLI
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="RLSFA + BHLR-v2 standalone native-resolution visualization"
    )

    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
    )
    parser.add_argument("--device", type=str, default="5")
    parser.add_argument("--rgb", type=str, default=None)
    parser.add_argument("--tir", type=str, default=None)
    parser.add_argument("--rgb-imgsz", type=int, default=None)
    parser.add_argument("--rgb-semantic-imgsz", type=int, default=None)
    parser.add_argument("--tir-imgsz", type=int, default=None)
    parser.add_argument(
        "--pair-mode",
        type=str,
        default=None,
        choices=["relative", "stem", "filename"],
    )
    parser.add_argument(
        "--index",
        type=int,
        nargs="+",
        default=None,
        help="One or more dataset indices, e.g. --index 0 50 100 200",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--align-scale", type=int, default=3)
    parser.add_argument("--bhlr-scale", type=int, default=3)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--save-dir", type=str, default=None)

    args = parser.parse_args()

    device = base_val.select_device(args.device)
    model, _, cfg = load_rlsfa_bhlr_v2_checkpoint(args.weights, device)
    model.eval()

    data_cfg = cfg.get("data", {})
    rgb_yaml = args.rgb or data_cfg.get("rgb")
    tir_yaml = args.tir or data_cfg.get("tir")
    rgb_imgsz = args.rgb_imgsz or int(data_cfg.get("rgb_imgsz", 1920))
    rgb_semantic_imgsz = args.rgb_semantic_imgsz or int(
        data_cfg.get("rgb_semantic_imgsz", 640)
    )
    tir_imgsz = args.tir_imgsz or int(data_cfg.get("tir_imgsz", 640))
    pair_mode = args.pair_mode or data_cfg.get("pair_mode", "relative")

    dataset = build_rsdt_dataset(
        rgb_yaml=rgb_yaml,
        tir_yaml=tir_yaml,
        split=args.split,
        rgb_high_imgsz=rgb_imgsz,
        rgb_semantic_imgsz=rgb_semantic_imgsz,
        tir_imgsz=tir_imgsz,
        pair_mode=pair_mode,
        augment=False,
        tir_channels=3,
        strict_pair=True,
    )

    if len(dataset) == 0:
        raise RuntimeError("Dataset is empty")

    if args.index is None:
        positive = collect_positive_indices(dataset)
        if not positive:
            raise RuntimeError("No positive UAV sample found")
        rng = random.Random(args.seed)
        indices = [rng.choice(positive)]
    else:
        indices = list(dict.fromkeys(args.index))

    for index in indices:
        if index < 0 or index >= len(dataset):
            raise IndexError(f"index={index}, dataset length={len(dataset)}")

        current_save_dir = args.save_dir
        if args.save_dir is not None and len(indices) > 1:
            current_save_dir = str(
                Path(args.save_dir).expanduser().resolve()
                / f"index_{index:06d}"
            )

        visualize_one(
            model=model,
            dataset=dataset,
            weights=args.weights,
            index=index,
            device=device,
            split=args.split,
            align_scale=args.align_scale,
            bhlr_scale=args.bhlr_scale,
            conf=args.conf,
            iou=args.iou,
            max_det=args.max_det,
            save_dir=current_save_dir,
        )


if __name__ == "__main__":
    main()
