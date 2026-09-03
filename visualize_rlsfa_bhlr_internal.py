"""
Outputs
-------
INPUT
00_rgb_original_gt.jpg
01_tir_original_gt.jpg
02_rgb_semantic_gt.jpg
03_tir_letterbox_gt.jpg
RLSFA ALIGNMENT -- only key stages
04_rgb_tir_p3_before_alignment.jpg
05_coarse_targetness.jpg
06_tir_after_coarse.jpg
07_local_spatial_frequency_panel.jpg
08_fine_displacement_probability.jpg
09_total_offset.jpg
10_rgb_tir_after_final_rlsfa.jpg
BHLR HIGH-RESOLUTION LOST-DETAIL RECOVERY
11_rgb_high_gt.jpg
12_rgb_semantic_upscaled.jpg
13_highres_residual.jpg
14_highres_residual_overlay.jpg
15_uav_residual_zoom.jpg
16_pixel_unshuffle_detail.jpg
17_detail_encoder_output.jpg
18_detail_avg_pool_p3.jpg
19_detail_max_pool_p3.jpg
20_detail_projection_p3.jpg
21_support_map.jpg
22_support_overlay.jpg
23_support_uav_zoom.jpg
24_detail_gate.jpg
25_detail_gate_overlay.jpg
26_usable_detail.jpg
27_usable_detail_overlay.jpg
28_rgb_before_after_bhlr.jpg
29_bhlr_enhancement.jpg
DETECTION
30_prediction_vs_gt.jpg
overview.jpg
metadata.json

Example
-------
conda activate tcy
cd /mnt/sda/taochangyong/Projects/Model/YOLO-tcy
python visualize_rlsfa_bhlr_internal.py \
  --weights runs/rlsfa_bhlr/yolo26n_UAVCB_RLSFA-P3_BHLR-P3_rgb1920_sem640_tir640_seed0/weights/best.pt \
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
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:  # Pillow is optional; OpenCV fallback remains available.
    Image = ImageDraw = ImageFont = None

import val_rgbt as base_val

from datasets.rsdt_dataset import build_rsdt_dataset
from datasets.rgbt_dataset import load_yolo_label
from train_rsdt import preprocess_batch_rsdt
from val_rlsfa_bhlr import load_rlsfa_bhlr_checkpoint


# ============================================================================
# Generic helpers
# ============================================================================


def ensure_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 1:
        return cv2.cvtColor(image[..., 0], cv2.COLOR_GRAY2BGR)
    return image


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


def normalize_map(x: np.ndarray, robust: bool = True) -> np.ndarray:
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

    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def tensor_feature_map(
    x: torch.Tensor,
    mode: str = "mean_abs",
) -> np.ndarray:
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
        y = torch.sqrt(torch.clamp((x * x).mean(dim=0), min=0.0))
    else:
        y = x.abs().mean(dim=0)

    return y.numpy()


def heatmap_bgr(
    fmap: np.ndarray,
    size: Optional[Tuple[int, int]] = None,
    robust: bool = True,
) -> np.ndarray:
    norm = normalize_map(fmap, robust=robust)
    image = (norm * 255.0).round().astype(np.uint8)

    if size is not None:
        image = cv2.resize(
            image,
            size,
            interpolation=cv2.INTER_LINEAR,
        )

    return cv2.applyColorMap(image, cv2.COLORMAP_TURBO)


def overlay_heatmap(
    base: np.ndarray,
    fmap: np.ndarray,
    alpha: float = 0.50,
    robust: bool = True,
) -> np.ndarray:
    base = ensure_bgr(base).copy()
    h, w = base.shape[:2]
    hm = heatmap_bgr(fmap, size=(w, h), robust=robust)
    return cv2.addWeighted(base, 1.0 - alpha, hm, alpha, 0.0)


def render_displacement_probability(
    probability: torch.Tensor,
    fine_offset: torch.Tensor,
    fine_confidence: Optional[torch.Tensor] = None,
) -> np.ndarray:
    """
    Visualize RLSFA fine matching as a displacement-probability matrix.

    The current RLSFA fine matcher does NOT return a spatial HxW correlation
    map. It returns a probability distribution over candidate residual
    translations. With fine_radius=2, the distribution has 25 candidates and
    is reshaped to a 5x5 matrix:

        rows    -> dy = -2, -1, 0, +1, +2
        columns -> dx = -2, -1, 0, +1, +2

    This is the faithful visualization of the actual fine-correlation search.
    """
    p = probability.detach().float().cpu()

    # Current RLSFA returns [B, K]. Keep the first visualization sample.
    while p.ndim > 1:
        p = p[0]
    p = p.reshape(-1)

    k = int(p.numel())
    side = int(round(math.sqrt(k)))
    if side * side != k or side % 2 == 0:
        raise ValueError(
            "fine_probability must contain an odd square number of candidate "
            f"translations, got K={k}"
        )

    radius = (side - 1) // 2
    prob = p.numpy().reshape(side, side)

    # Use the original RGB source resolution as the drawing canvas whenever
    # available. save_image() will therefore not need to shrink this figure.
    width = int(_EXPORT_REFERENCE_W or 1280)
    height = int(_EXPORT_REFERENCE_H or 720)
    width = max(width, 900)
    height = max(height, 700)

    canvas = np.full((height, width, 3), 246, dtype=np.uint8)

    scale_ref = max(1.0, min(width / 1280.0, height / 720.0))
    left = int(round(180 * scale_ref))
    right = int(round(80 * scale_ref))
    top = int(round(120 * scale_ref))
    bottom = int(round(150 * scale_ref))

    usable_w = max(1, width - left - right)
    usable_h = max(1, height - top - bottom)
    cell = max(20, min(usable_w // side, usable_h // side))

    grid_w = cell * side
    grid_h = cell * side
    x0 = left + max(0, (usable_w - grid_w) // 2)
    y0 = top + max(0, (usable_h - grid_h) // 2)

    # Color uses relative probability only for visualization. Actual numeric
    # probabilities are printed inside every cell.
    pmax = float(prob.max()) if prob.size else 0.0
    if pmax <= 1e-12:
        color_norm = np.zeros_like(prob, dtype=np.float32)
    else:
        color_norm = np.clip(prob / pmax, 0.0, 1.0).astype(np.float32)

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
            cv2.rectangle(
                canvas,
                (xa, ya),
                (xb, yb),
                (35, 35, 35),
                max(1, int(round(scale_ref))),
            )

            # White outline marks the highest-probability discrete candidate.
            if row == best_row and col == best_col:
                cv2.rectangle(
                    canvas,
                    (xa + 3, ya + 3),
                    (xb - 3, yb - 3),
                    (255, 255, 255),
                    max(3, int(round(4 * scale_ref))),
                )

            # Choose text color from cell brightness.
            brightness = 0.114 * color[0] + 0.587 * color[1] + 0.299 * color[2]
            text_color = (20, 20, 20) if brightness > 145 else (255, 255, 255)
            label = f"{value:.3f}"
            font_scale = max(0.42, 0.58 * scale_ref)
            thickness = max(1, int(round(scale_ref)))
            (tw, th), _ = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                thickness,
            )
            tx = xa + max(4, (cell - tw) // 2)
            ty = ya + max(th + 4, (cell + th) // 2)
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

    # dx labels along the top.
    axis_scale = max(0.50, 0.62 * scale_ref)
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
            (cx - tw // 2, y0 - int(round(18 * scale_ref))),
            cv2.FONT_HERSHEY_SIMPLEX,
            axis_scale,
            (35, 35, 35),
            axis_thickness,
            cv2.LINE_AA,
        )

    # dy labels along the left.
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
            (x0 - tw - int(round(24 * scale_ref)), cy + th // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            axis_scale,
            (35, 35, 35),
            axis_thickness,
            cv2.LINE_AA,
        )

    cv2.putText(
        canvas,
        "dx (P3 pixels)",
        (x0, max(35, y0 - int(round(65 * scale_ref)))),
        cv2.FONT_HERSHEY_SIMPLEX,
        max(0.55, 0.68 * scale_ref),
        (30, 30, 30),
        max(1, int(round(scale_ref))),
        cv2.LINE_AA,
    )

    cv2.putText(
        canvas,
        "dy",
        (max(18, x0 - int(round(110 * scale_ref))), y0 + grid_h // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        max(0.55, 0.68 * scale_ref),
        (30, 30, 30),
        max(1, int(round(scale_ref))),
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

    info_y = min(height - int(round(65 * scale_ref)), y0 + grid_h + int(round(55 * scale_ref)))
    line1 = (
        f"argmax candidate: dx={best_dx:+d}, dy={best_dy:+d} P3 px   "
        f"p_max={float(prob[best_row, best_col]):.4f}"
    )
    line2 = (
        f"predicted fine offset: dx={fine_dx:+.4f}, dy={fine_dy:+.4f} P3 px   "
        f"confidence={conf_text}"
    )

    cv2.putText(
        canvas,
        line1,
        (max(20, x0 - int(round(80 * scale_ref))), info_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        max(0.48, 0.60 * scale_ref),
        (30, 30, 30),
        max(1, int(round(scale_ref))),
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        line2,
        (max(20, x0 - int(round(80 * scale_ref))), info_y + int(round(36 * scale_ref))),
        cv2.FONT_HERSHEY_SIMPLEX,
        max(0.48, 0.60 * scale_ref),
        (30, 30, 30),
        max(1, int(round(scale_ref))),
        cv2.LINE_AA,
    )

    return canvas


# ---------------------------------------------------------------------------
# Full-resolution export
# ---------------------------------------------------------------------------
# Every standalone visualization is exported at least at the original RGB
# source resolution. Multi-panel figures use the original RGB resolution for
# EACH tile, so a 2x2 panel is intentionally larger than one source image.
# This only changes visualization/export resolution. It does NOT change model
# tensors or create new spatial information in P3/P4/P5 feature maps.
_EXPORT_REFERENCE_W: Optional[int] = None
_EXPORT_REFERENCE_H: Optional[int] = None


def set_export_reference(width: int, height: int) -> None:
    global _EXPORT_REFERENCE_W, _EXPORT_REFERENCE_H
    _EXPORT_REFERENCE_W = int(width)
    _EXPORT_REFERENCE_H = int(height)


def fit_to_export_reference(image: np.ndarray) -> np.ndarray:
    """
    Upscale a small visualization to the original RGB source canvas while
    preserving aspect ratio. Images already at or above that resolution are
    left unchanged, which avoids downsampling RGB-high and large panels.
    """
    image = ensure_bgr(image)

    if _EXPORT_REFERENCE_W is None or _EXPORT_REFERENCE_H is None:
        return image

    target_w = int(_EXPORT_REFERENCE_W)
    target_h = int(_EXPORT_REFERENCE_H)
    h, w = image.shape[:2]

    # Never reduce an already high-resolution visualization.
    if w >= target_w and h >= target_h:
        return image

    scale = min(
        target_w / max(w, 1),
        target_h / max(h, 1),
    )
    rw = max(1, int(round(w * scale)))
    rh = max(1, int(round(h * scale)))

    resized = cv2.resize(
        image,
        (rw, rh),
        interpolation=cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA,
    )

    # Keep aspect ratio; use a neutral canvas instead of stretching.
    canvas = np.full((target_h, target_w, 3), 245, dtype=np.uint8)
    x0 = (target_w - rw) // 2
    y0 = (target_h - rh) // 2
    canvas[y0:y0 + rh, x0:x0 + rw] = resized
    return canvas


def _unicode_font(size: int, bold: bool = False):
    """Return a local Unicode-capable font when Pillow is available."""
    if ImageFont is None:
        return None

    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold
        else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf" if bold
        else "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    for font_path in candidates:
        if Path(font_path).exists():
            try:
                return ImageFont.truetype(font_path, size=max(8, int(size)))
            except Exception:
                pass
    return None


def _put_unicode_text(
    image_bgr: np.ndarray,
    text: str,
    xy: Tuple[int, int],
    font_size: int,
    color_bgr: Tuple[int, int, int] = (25, 25, 25),
    bold: bool = False,
) -> np.ndarray:
    """Draw Unicode text (including Δ) with Pillow, falling back to OpenCV."""
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

    # Fallback cannot reliably render Greek Delta, so use ASCII spelling.
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


def add_title(
    image: np.ndarray,
    title: str,
    subtitle: str = "",
    annotation: str = "",
    bar_h: int = 68,
) -> np.ndarray:
    """
    Draw a white information strip INSIDE the image so saved pixel size is unchanged.

    annotation is intended for RLSFA offset decomposition, e.g.:
        Δcoarse=(dx,dy) | Δfine=(dx,dy) | Δtotal=(dx,dy)
    """
    image = ensure_bgr(image).copy()
    h, w = image.shape[:2]

    scale_ref = max(1.0, min(w / 1280.0, h / 720.0))
    base_h = 98 if annotation else 68
    band_h = int(round(max(bar_h, base_h * scale_ref)))
    band_h = min(band_h, max(base_h, h // 3))

    # Opaque white strip: easier to read than a semitransparent overlay.
    cv2.rectangle(image, (0, 0), (w, band_h), (248, 248, 248), -1)

    title_scale = 0.68 * scale_ref
    subtitle_scale = 0.45 * scale_ref
    x = int(round(14 * scale_ref))
    y_title = int(round(27 * scale_ref))
    y_sub = int(round(54 * scale_ref))

    cv2.putText(
        image,
        title,
        (x, min(y_title, band_h - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        title_scale,
        (20, 20, 20),
        max(2, int(round(2 * scale_ref))),
        cv2.LINE_AA,
    )

    if subtitle:
        cv2.putText(
            image,
            subtitle,
            (x, min(y_sub, band_h - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            subtitle_scale,
            (75, 75, 75),
            max(1, int(round(scale_ref))),
            cv2.LINE_AA,
        )

    if annotation:
        ann_y = int(round(68 * scale_ref))
        ann_size = int(round(18 * scale_ref))
        image = _put_unicode_text(
            image,
            annotation,
            (x, min(ann_y, band_h - ann_size - 4)),
            font_size=ann_size,
            color_bgr=(30, 30, 30),
            bold=True,
        )

    return image


def save_image(
    out_dir: Path,
    filename: str,
    image: np.ndarray,
    title: str,
    subtitle: str = "",
    annotation: str = "",
) -> np.ndarray:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Small 640/80-derived visualizations are exported on an original-RGB
    # resolution canvas. Large RGB-high images/panels are never downsampled.
    image = fit_to_export_reference(image)
    titled = add_title(image, title, subtitle, annotation=annotation)

    path = out_dir / filename
    params = [cv2.IMWRITE_JPEG_QUALITY, 97]
    if not cv2.imwrite(str(path), titled, params):
        raise RuntimeError(f"Failed to save {path}")
    return titled


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
    thickness: int = 2,
) -> np.ndarray:
    out = ensure_bgr(image).copy()

    for box in boxes_xyxy:
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(
            out,
            label,
            (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            color,
            1,
            cv2.LINE_AA,
        )

    return out


def draw_predictions(
    image: np.ndarray,
    pred: Optional[Dict[str, torch.Tensor]],
    conf_min: float,
) -> np.ndarray:
    out = image.copy()
    if pred is None:
        return out

    boxes = pred.get("bboxes", None)
    conf = pred.get("conf", None)
    if boxes is None or conf is None:
        return out

    boxes = boxes.detach().float().cpu().numpy()
    conf = conf.detach().float().cpu().numpy()

    for box, score in zip(boxes, conf):
        if float(score) < conf_min:
            continue
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            out,
            f"Pred {float(score):.2f}",
            (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    return out


def panel(
    items: Sequence[Tuple[str, np.ndarray]],
    cols: int,
    tile_w: int = 520,
    tile_h: int = 390,
) -> np.ndarray:
    """Create a labeled panel with aspect-ratio-preserving tiles."""
    if not items:
        raise ValueError("panel items cannot be empty")

    cols = max(1, int(cols))
    rows = int(math.ceil(len(items) / cols))

    # Use original RGB resolution for every sub-panel tile. Explicit small
    # tile_w/tile_h arguments from the old script are treated only as minima.
    if _EXPORT_REFERENCE_W is not None:
        tile_w = max(int(tile_w), int(_EXPORT_REFERENCE_W))
    if _EXPORT_REFERENCE_H is not None:
        tile_h = max(int(tile_h), int(_EXPORT_REFERENCE_H))

    scale_ref = max(1.0, min(tile_w / 1280.0, tile_h / 720.0))
    header = int(round(44 * scale_ref))

    canvas = np.full(
        (rows * (tile_h + header), cols * tile_w, 3),
        255,
        dtype=np.uint8,
    )

    for i, (label, image) in enumerate(items):
        image = ensure_bgr(image)
        row = i // cols
        col = i % cols
        x0 = col * tile_w
        y0 = row * (tile_h + header)

        h, w = image.shape[:2]
        scale = min(tile_w / max(w, 1), tile_h / max(h, 1))
        rw = max(1, int(round(w * scale)))
        rh = max(1, int(round(h * scale)))
        resized = cv2.resize(
            image,
            (rw, rh),
            interpolation=cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA,
        )

        px = x0 + (tile_w - rw) // 2
        py = y0 + header + (tile_h - rh) // 2
        canvas[py:py + rh, px:px + rw] = resized

        cv2.putText(
            canvas,
            label,
            (x0 + 10, y0 + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52 * scale_ref,
            (20, 20, 20),
            max(1, int(round(scale_ref))),
            cv2.LINE_AA,
        )

    return canvas


def crop_around_box(
    image: np.ndarray,
    box: Optional[np.ndarray],
    expand: float = 3.0,
    min_size: int = 96,
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


def draw_translation_arrow(
    base: np.ndarray,
    offset: torch.Tensor,
    feature_stride: float,
    title: str,
) -> np.ndarray:
    """
    Draw VISUAL content-motion direction.

    RLSFA offset is sampling displacement:
        output(p) = source(p + delta)
    so content visually moves approximately -delta.
    """
    dx = float(offset[0, 0, 0, 0].detach().float().cpu())
    dy = float(offset[0, 1, 0, 0].detach().float().cpu())

    image = base.copy()
    h, w = image.shape[:2]
    start = (w // 2, h // 2)
    end = (
        int(round(start[0] - dx * feature_stride)),
        int(round(start[1] - dy * feature_stride)),
    )

    cv2.arrowedLine(
        image,
        start,
        end,
        (0, 0, 255),
        4,
        tipLength=0.12,
    )

    cv2.putText(
        image,
        f"sampling dx={dx:+.3f}, dy={dy:+.3f} P3 px",
        (18, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        image,
        "arrow = visual TIR content motion",
        (18, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (0, 0, 180),
        1,
        cv2.LINE_AA,
    )

    return image


def draw_translation_decomposition(
    base: np.ndarray,
    coarse_offset: torch.Tensor,
    fine_offset: torch.Tensor,
    total_offset: torch.Tensor,
    feature_stride: float,
) -> np.ndarray:
    """
    Visualize the decomposition of RLSFA translation.

    Offsets are sampling displacements, so visual content motion is -delta.
    - coarse: center -> coarse endpoint
    - fine  : coarse endpoint -> final endpoint
    - total : center -> final endpoint
    """
    image = ensure_bgr(base).copy()
    h, w = image.shape[:2]
    start = np.array([w / 2.0, h / 2.0], dtype=np.float32)

    def _xy(t: torch.Tensor) -> np.ndarray:
        td = t.detach().float().cpu()
        return np.array(
            [float(td[0, 0, 0, 0]), float(td[0, 1, 0, 0])],
            dtype=np.float32,
        )

    dc = _xy(coarse_offset)
    df = _xy(fine_offset)
    dt = _xy(total_offset)

    coarse_end = start - dc * float(feature_stride)
    final_end = start - dt * float(feature_stride)

    def _pt(v):
        return (int(round(float(v[0]))), int(round(float(v[1]))))

    # Coarse visual motion: blue.
    cv2.arrowedLine(
        image,
        _pt(start),
        _pt(coarse_end),
        (255, 120, 0),
        4,
        tipLength=0.12,
    )

    # Fine residual motion: orange, starting where coarse ended.
    cv2.arrowedLine(
        image,
        _pt(coarse_end),
        _pt(final_end),
        (0, 165, 255),
        4,
        tipLength=0.18,
    )

    # Total motion: red, drawn slightly thinner so all three remain visible.
    cv2.arrowedLine(
        image,
        _pt(start),
        _pt(final_end),
        (0, 0, 255),
        2,
        tipLength=0.10,
    )

    # Mark the three key locations.
    cv2.circle(image, _pt(start), 6, (255, 255, 255), -1)
    cv2.circle(image, _pt(start), 6, (30, 30, 30), 2)
    cv2.circle(image, _pt(coarse_end), 6, (255, 120, 0), -1)
    cv2.circle(image, _pt(final_end), 6, (0, 0, 255), -1)

    # Compact legend near the bottom; the numerical values stay in the white header.
    legend_y = max(24, h - 28)
    legend = [
        ("coarse", (255, 120, 0)),
        ("fine", (0, 165, 255)),
        ("total", (0, 0, 255)),
    ]
    x = 18
    for name, color in legend:
        cv2.line(image, (x, legend_y), (x + 34, legend_y), color, 4)
        cv2.putText(
            image,
            name,
            (x + 42, legend_y + 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            color,
            2,
            cv2.LINE_AA,
        )
        x += 126

    return image


@torch.no_grad()
def warp_image_translation_preview(
    tir_tensor: torch.Tensor,
    offset_feature: torch.Tensor,
    feature_hw: Tuple[int, int],
) -> torch.Tensor:
    """Use the same global RLSFA sampling translation on the 640 TIR image."""
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


def get_scale_item(mapping: Dict, scale: int):
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


def debug_tensor_to_2d(
    value: torch.Tensor,
    fallback_hw: Tuple[int, int] | None = None,
) -> np.ndarray:
    """
    Convert a debug tensor to a 2-D visualization map.

    RLSFA implementations in this project use two correlation-debug styles:
      1) spatial confidence map [B,1,H,W]
      2) global confidence [B,1,1,1]

    This helper supports both, so the visualization remains compatible with
    the actual RLSFA debug keys used by the trained checkpoint.
    """
    x = value.detach().float().cpu()

    if x.ndim == 4:
        x = x[0]

    if x.ndim == 3:
        if x.shape[0] == 1:
            x = x[0]
        else:
            x = x.abs().mean(dim=0)

    if x.ndim == 1:
        x = x.mean().reshape(1, 1)

    if x.ndim == 0:
        x = x.reshape(1, 1)

    arr = x.numpy().astype(np.float32)

    if arr.size == 1 and fallback_hw is not None:
        h, w = fallback_hw
        arr = np.full((h, w), float(arr.reshape(-1)[0]), dtype=np.float32)

    return arr


def get_alignment_debug(outputs: Dict, scale: int) -> Dict:
    info = outputs["alignment_info"]
    mapping = info.get("scale_debug", info.get("scales", {}))
    return get_scale_item(mapping, scale)


def get_bhlr_debug(outputs: Dict, scale: int):
    all_debug = outputs.get("bhlr_debug", outputs.get("rsdt_debug", None))
    if all_debug is None:
        raise KeyError("No BHLR/RSD-T debug dictionary in outputs")
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


def make_overview(
    panels: Sequence[Tuple[str, np.ndarray]],
    path: Path,
    cols: int = 5,
):
    """
    High-resolution contact sheet. For ordinary 1080p-or-smaller source RGB,
    every overview tile also uses the full original RGB resolution. For very
    large sources (for example 4K), overview tiles are reduced to half-size
    only to avoid excessive RAM; all numbered output images remain full-res.
    """
    ref_w = int(_EXPORT_REFERENCE_W or 1920)
    ref_h = int(_EXPORT_REFERENCE_H or 1080)

    if ref_w * ref_h <= 2_500_000:
        thumb_w = ref_w
        thumb_h = ref_h
    else:
        thumb_w = max(960, ref_w // 2)
        thumb_h = max(540, ref_h // 2)
    scale_ref = max(1.0, min(thumb_w / 960.0, thumb_h / 540.0))
    header = int(round(38 * scale_ref))
    rows = int(math.ceil(len(panels) / cols))

    canvas = np.full(
        (rows * (thumb_h + header), cols * thumb_w, 3),
        255,
        dtype=np.uint8,
    )

    for i, (name, image) in enumerate(panels):
        row = i // cols
        col = i % cols
        x0 = col * thumb_w
        y0 = row * (thumb_h + header)

        image = ensure_bgr(image)
        h, w = image.shape[:2]
        scale = min(thumb_w / max(w, 1), thumb_h / max(h, 1))
        rw = max(1, int(round(w * scale)))
        rh = max(1, int(round(h * scale)))
        resized = cv2.resize(
            image,
            (rw, rh),
            interpolation=cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA,
        )

        px = x0 + (thumb_w - rw) // 2
        py = y0 + header + (thumb_h - rh) // 2
        canvas[py:py + rh, px:px + rw] = resized

        cv2.putText(
            canvas,
            f"{i:02d} {name}",
            (x0 + int(8 * scale_ref), y0 + int(26 * scale_ref)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48 * scale_ref,
            (25, 25, 25),
            max(1, int(round(scale_ref))),
            cv2.LINE_AA,
        )

    cv2.imwrite(
        str(path),
        canvas,
        [cv2.IMWRITE_JPEG_QUALITY, 95],
    )


# ============================================================================
# Main per-sample visualization
# ============================================================================


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
    raw_rgb_p3 = enhanced_rgb_p3 - gamma * usable_detail

    # ---------------------------------------------------------------------
    # Input images and labels
    # ---------------------------------------------------------------------
    rgb_original = read_rgb_bgr(sample["rgb_path"])
    tir_original = read_tir_bgr(sample["tir_path"])

    # All exported visualizations use the original RGB source resolution as
    # the minimum standalone canvas / per-panel tile resolution.
    set_export_reference(
        width=rgb_original.shape[1],
        height=rgb_original.shape[0],
    )

    rgb_semantic = tensor_rgb_to_bgr(batch["rgb_semantic_img"][0])
    tir_letterbox = tensor_rgb_to_bgr(batch["tir_img"][0])
    rgb_high = tensor_rgb_to_bgr(batch["rgb_img"][0])

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

    semantic_norm = (
        sample["rgb_semantic_bboxes"].detach().float().cpu().numpy()
    )
    tir_norm = sample["tir_bboxes"].detach().float().cpu().numpy()
    high_norm = sample["rgb_bboxes"].detach().float().cpu().numpy()

    semantic_boxes = normalized_xywh_to_xyxy(
        semantic_norm,
        rgb_semantic.shape[1],
        rgb_semantic.shape[0],
    )
    tir_letterbox_boxes = normalized_xywh_to_xyxy(
        tir_norm,
        tir_letterbox.shape[1],
        tir_letterbox.shape[0],
    )
    high_boxes = normalized_xywh_to_xyxy(
        high_norm,
        rgb_high.shape[1],
        rgb_high.shape[0],
    )

    # ---------------------------------------------------------------------
    # Save directory
    # ---------------------------------------------------------------------
    weights_path = Path(weights).expanduser().resolve()
    stem = Path(sample["rgb_path"]).stem

    if save_dir is None:
        out_dir = (
            weights_path.parent.parent
            / "visualize_rlsfa_bhlr_focused"
            / f"{split}_{index:06d}_{stem}"
        )
    else:
        out_dir = Path(save_dir).expanduser().resolve()

    out_dir.mkdir(parents=True, exist_ok=True)
    overview_panels: List[Tuple[str, np.ndarray]] = []

    def save(
        number: int,
        slug: str,
        image: np.ndarray,
        title: str,
        subtitle: str = "",
        annotation: str = "",
    ):
        save_image(
            out_dir,
            f"{number:02d}_{slug}.jpg",
            image,
            title,
            subtitle,
            annotation=annotation,
        )
        overview_panels.append((slug, image))

    # =====================================================================
    # 00-03 INPUT
    # =====================================================================
    save(
        0,
        "rgb_original_gt",
        draw_boxes(rgb_original, rgb_orig_boxes),
        "Original RGB + GT",
        str(sample["rgb_path"]),
    )

    save(
        1,
        "tir_original_gt",
        draw_boxes(tir_original, tir_orig_boxes),
        "Original TIR + GT",
        str(sample["tir_path"]),
    )

    save(
        2,
        "rgb_semantic_gt",
        draw_boxes(rgb_semantic, semantic_boxes),
        "RGB semantic input + GT",
        f"shape={rgb_semantic.shape[1]}x{rgb_semantic.shape[0]}",
    )

    save(
        3,
        "tir_letterbox_gt",
        draw_boxes(tir_letterbox, tir_letterbox_boxes),
        "TIR network input + GT",
        f"shape={tir_letterbox.shape[1]}x{tir_letterbox.shape[0]}",
    )

    # =====================================================================
    # 04-10 RLSFA -- only key stages
    # =====================================================================
    rgb_raw_overlay = draw_boxes(
        overlay_heatmap(
            rgb_semantic,
            tensor_feature_map(raw_rgb_p3),
            alpha=0.52,
        ),
        semantic_boxes,
    )

    tir_raw_overlay = draw_boxes(
        overlay_heatmap(
            tir_letterbox,
            tensor_feature_map(raw_tir_p3),
            alpha=0.52,
        ),
        tir_letterbox_boxes,
    )

    before_panel = panel(
        [
            ("RGB P3", rgb_raw_overlay),
            ("Raw TIR P3", tir_raw_overlay),
        ],
        cols=2,
    )

    save(
        4,
        "rgb_tir_p3_before_alignment",
        before_panel,
        "RLSFA input: RGB/TIR P3 before alignment",
        "Compare target response location before any TIR translation",
    )

    targetness = debug_get(align, "targetness", "coarse_targetness")[0, 0].detach().float().cpu().numpy()
    targetness_img = draw_boxes(
        overlay_heatmap(
            rgb_semantic,
            targetness,
            alpha=0.56,
            robust=False,
        ),
        semantic_boxes,
    )

    save(
        5,
        "coarse_targetness",
        targetness_img,
        "Coarse targetness",
        "RGB-reference support used to reduce background domination in correlation",
    )

    coarse_offset = debug_get(align, "coarse_offset")
    fine_offset = debug_get(align, "fine_offset")
    total_offset = debug_get(align, "total_offset")

    def _offset_xy(t: torch.Tensor) -> Tuple[float, float]:
        td = t.detach().float().cpu()
        return (
            float(td[0, 0, 0, 0]),
            float(td[0, 1, 0, 0]),
        )

    coarse_dx, coarse_dy = _offset_xy(coarse_offset)
    fine_dx, fine_dy = _offset_xy(fine_offset)
    total_dx, total_dy = _offset_xy(total_offset)

    offset_annotation = (
        f"Δcoarse=({coarse_dx:+.3f}, {coarse_dy:+.3f})  |  "
        f"Δfine=({fine_dx:+.3f}, {fine_dy:+.3f})  |  "
        f"Δtotal=({total_dx:+.3f}, {total_dy:+.3f})  P3 px  [sampling offset]"
    )

    fh, fw = raw_tir_p3.shape[-2:]
    coarse_preview_tensor = warp_image_translation_preview(
        batch["tir_img"],
        coarse_offset,
        (fh, fw),
    )
    coarse_preview = tensor_rgb_to_bgr(coarse_preview_tensor[0])

    coarse_panel = panel(
        [
            (
                "Raw TIR",
                draw_boxes(tir_letterbox, tir_letterbox_boxes),
            ),
            (
                "After coarse translation",
                draw_boxes(coarse_preview, semantic_boxes, label="RGB GT"),
            ),
        ],
        cols=2,
    )

    save(
        6,
        "tir_after_coarse",
        coarse_panel,
        "Coarse translation result",
        "Left: original TIR coordinate; right: TIR translated toward RGB reference",
        annotation=offset_annotation,
    )

    tir_spatial = debug_get(align, "tir_spatial", "tir_spatial_structure")
    tir_frequency = debug_get(align, "tir_frequency", "tir_local_frequency")
    tir_reliability = debug_get(
        align,
        "tir_frequency_reliability",
        "tir_reliability",
    )
    tir_reliable = debug_get(align, "tir_reliable_structure")

    structural_panel = panel(
        [
            (
                "TIR spatial structure",
                draw_boxes(
                    overlay_heatmap(
                        coarse_preview,
                        tensor_feature_map(tir_spatial),
                        alpha=0.55,
                    ),
                    semantic_boxes,
                    label="RGB GT",
                ),
            ),
            (
                "TIR local Window-FFT",
                draw_boxes(
                    overlay_heatmap(
                        coarse_preview,
                        tensor_feature_map(tir_frequency),
                        alpha=0.55,
                    ),
                    semantic_boxes,
                    label="RGB GT",
                ),
            ),
            (
                "Frequency reliability Q_T",
                draw_boxes(
                    overlay_heatmap(
                        coarse_preview,
                        tir_reliability[0, 0].detach().float().cpu().numpy(),
                        alpha=0.55,
                        robust=False,
                    ),
                    semantic_boxes,
                    label="RGB GT",
                ),
            ),
            (
                "Reliable TIR structure",
                draw_boxes(
                    overlay_heatmap(
                        coarse_preview,
                        tensor_feature_map(tir_reliable),
                        alpha=0.55,
                    ),
                    semantic_boxes,
                    label="RGB GT",
                ),
            ),
        ],
        cols=2,
    )

    save(
        7,
        "local_spatial_frequency_panel",
        structural_panel,
        "Local spatial-frequency reliability",
        "Q_T=1 prefers local frequency; Q_T=0 prefers spatial structure",
        annotation=offset_annotation,
    )

    # Fine matcher output is a probability distribution over candidate
    # residual translations, NOT an HxW spatial correlation map.
    fine_probability = debug_get(align, "fine_probability")
    fine_confidence = debug_get(
        align,
        "fine_correlation_confidence",
    )
    fine_offset_for_vis = fine_offset

    fine_prob_img = render_displacement_probability(
        probability=fine_probability,
        fine_offset=fine_offset_for_vis,
        fine_confidence=fine_confidence,
    )

    save(
        8,
        "fine_displacement_probability",
        fine_prob_img,
        "Fine residual displacement probability",
        "rows=dy, columns=dx; white box = maximum-probability candidate",
        annotation=offset_annotation,
    )

    # total_offset was already fetched above for the shared header annotation.
    feature_stride = float(rgb_semantic.shape[1]) / float(fw)
    offset_img = draw_translation_decomposition(
        rgb_semantic,
        coarse_offset=coarse_offset,
        fine_offset=fine_offset,
        total_offset=total_offset,
        feature_stride=feature_stride,
    )
    offset_img = draw_boxes(offset_img, semantic_boxes)

    save(
        9,
        "total_offset",
        offset_img,
        "Final RLSFA translation",
        "coarse + fine; translation only, no dense deformation",
        annotation=offset_annotation,
    )

    final_preview_tensor = warp_image_translation_preview(
        batch["tir_img"],
        total_offset,
        (fh, fw),
    )
    final_preview = tensor_rgb_to_bgr(final_preview_tensor[0])

    aligned_p3_overlay = draw_boxes(
        overlay_heatmap(
            rgb_semantic,
            tensor_feature_map(aligned_tir_p3),
            alpha=0.52,
        ),
        semantic_boxes,
    )

    final_panel = panel(
        [
            (
                "RGB semantic + GT",
                draw_boxes(rgb_semantic, semantic_boxes),
            ),
            (
                "Aligned TIR image preview",
                draw_boxes(final_preview, semantic_boxes, label="RGB GT"),
            ),
            (
                "Aligned TIR P3 activation",
                aligned_p3_overlay,
            ),
        ],
        cols=3,
        tile_w=430,
        tile_h=350,
    )

    save(
        10,
        "rgb_tir_after_final_rlsfa",
        final_panel,
        "Final RLSFA alignment",
        "The final transform is one global TIR translation",
        annotation=offset_annotation,
    )

    # =====================================================================
    # 11-29 BHLR -- expanded high-resolution path
    # =====================================================================
    save(
        11,
        "rgb_high_gt",
        draw_boxes(rgb_high, high_boxes),
        "High-resolution RGB + GT",
        f"shape={rgb_high.shape[1]}x{rgb_high.shape[0]}",
    )

    reconstructed_high = bhlr_all.get("reconstructed_high", None)
    residual_high = bhlr_all.get("resolution_residual", None)
    residual_s2d = bhlr_all.get("residual_s2d", None)
    detail_raw = bhlr_all.get("detail_raw", None)

    if reconstructed_high is None or residual_high is None or detail_raw is None:
        raise KeyError(
            "BHLR debug must expose reconstructed_high, resolution_residual, "
            "and detail_raw. Use the refactored bhlr_v1.py included with this script."
        )

    reconstructed_img = tensor_rgb_to_bgr(reconstructed_high[0])
    save(
        12,
        "rgb_semantic_upscaled",
        draw_boxes(reconstructed_img, high_boxes),
        "RGB semantic upscaled to high resolution",
        "This is what remains if only the low-resolution RGB is retained",
    )

    residual_map_high = tensor_feature_map(residual_high, mode="mean_abs")
    residual_heat_high = heatmap_bgr(
        residual_map_high,
        size=(rgb_high.shape[1], rgb_high.shape[0]),
        robust=True,
    )
    residual_heat_high = draw_boxes(residual_heat_high, high_boxes)

    save(
        13,
        "highres_residual",
        residual_heat_high,
        "High-resolution lost-detail residual",
        "|RGB_high - upsample(RGB_semantic)|; background detail is expected here",
    )

    residual_overlay_high = draw_boxes(
        overlay_heatmap(
            rgb_high,
            residual_map_high,
            alpha=0.50,
            robust=True,
        ),
        high_boxes,
    )

    save(
        14,
        "highres_residual_overlay",
        residual_overlay_high,
        "Lost-detail residual over high-resolution RGB",
        "Use this to locate the physical source of high-frequency residuals",
    )

    first_high_box = high_boxes[0] if len(high_boxes) > 0 else None
    zoom_panel = panel(
        [
            (
                "High-resolution UAV",
                crop_around_box(
                    draw_boxes(rgb_high, high_boxes),
                    first_high_box,
                    expand=4.0,
                    min_size=160,
                ),
            ),
            (
                "Upscaled low-resolution UAV",
                crop_around_box(
                    draw_boxes(reconstructed_img, high_boxes),
                    first_high_box,
                    expand=4.0,
                    min_size=160,
                ),
            ),
            (
                "Lost-detail residual",
                crop_around_box(
                    residual_overlay_high,
                    first_high_box,
                    expand=4.0,
                    min_size=160,
                ),
            ),
        ],
        cols=3,
        tile_w=430,
        tile_h=360,
    )

    save(
        15,
        "uav_residual_zoom",
        zoom_panel,
        "UAV-level high-resolution information loss",
        "High RGB vs upscaled low RGB vs residual around the UAV",
    )

    if residual_s2d is None:
        # Backward-compatible fallback for the original bhlr_v1.py.
        # PixelUnshuffle is deterministic and has no learnable parameters, so
        # recomputing it here changes visualization only, not the model.
        ratio_h = int(round(rgb_high.shape[0] / rgb_semantic.shape[0]))
        ratio_w = int(round(rgb_high.shape[1] / rgb_semantic.shape[1]))
        if ratio_h != ratio_w or ratio_h < 1:
            raise ValueError(
                "Cannot infer BHLR PixelUnshuffle ratio from "
                f"RGB-high={rgb_high.shape[:2]} and RGB-semantic={rgb_semantic.shape[:2]}"
            )
        residual_s2d = F.pixel_unshuffle(
            residual_high,
            downscale_factor=ratio_h,
        )

    pixel_unshuffle_img = draw_boxes(
        overlay_heatmap(
            rgb_semantic,
            tensor_feature_map(residual_s2d),
            alpha=0.54,
        ),
        semantic_boxes,
    )

    save(
        16,
        "pixel_unshuffle_detail",
        pixel_unshuffle_img,
        "PixelUnshuffle lost-detail tensor",
        f"shape={tuple(residual_s2d.shape)}; spatial detail moved into channels",
    )

    detail_raw_img = draw_boxes(
        overlay_heatmap(
            rgb_semantic,
            tensor_feature_map(detail_raw),
            alpha=0.54,
        ),
        semantic_boxes,
    )

    save(
        17,
        "detail_encoder_output",
        detail_raw_img,
        "Shared lost-detail encoder output",
        f"shape={tuple(detail_raw.shape)}",
    )

    detail_avg = bhlr.get("detail_avg", None)
    detail_max = bhlr.get("detail_max", None)
    detail_scale = bhlr.get("detail_scale", None)

    # Backward-compatible fallback for the original BHLR debug dictionary.
    # These are the exact deterministic pooling operations already used by
    # BHLRScaleAdapter._compress_detail().
    target_hw = raw_rgb_p3.shape[-2:]
    if detail_avg is None:
        detail_avg = F.adaptive_avg_pool2d(
            detail_raw,
            target_hw,
        )
    if detail_max is None:
        detail_max = F.adaptive_max_pool2d(
            detail_raw,
            target_hw,
        )
    if detail_scale is None:
        raise KeyError(
            "BHLR scale debug is missing 'detail_scale'. "
            f"Available={list(bhlr.keys())}"
        )

    save(
        18,
        "detail_avg_pool_p3",
        draw_boxes(
            overlay_heatmap(
                rgb_semantic,
                tensor_feature_map(detail_avg),
                alpha=0.54,
            ),
            semantic_boxes,
        ),
        f"Average-pooled lost detail -> P{bhlr_scale}",
        f"shape={tuple(detail_avg.shape)}",
    )

    save(
        19,
        "detail_max_pool_p3",
        draw_boxes(
            overlay_heatmap(
                rgb_semantic,
                tensor_feature_map(detail_max),
                alpha=0.54,
            ),
            semantic_boxes,
        ),
        f"Max-pooled lost detail -> P{bhlr_scale}",
        f"shape={tuple(detail_max.shape)}",
    )

    save(
        20,
        "detail_projection_p3",
        draw_boxes(
            overlay_heatmap(
                rgb_semantic,
                tensor_feature_map(detail_scale),
                alpha=0.54,
            ),
            semantic_boxes,
        ),
        f"Projected high-resolution lost detail at P{bhlr_scale}",
        "concat(AvgPool, MaxPool) -> 1x1 projection",
    )

    support = bhlr["support_map"][0, 0].detach().float().cpu().numpy()
    support_heat = heatmap_bgr(
        support,
        size=(rgb_semantic.shape[1], rgb_semantic.shape[0]),
        robust=False,
    )
    support_heat = draw_boxes(support_heat, semantic_boxes)

    save(
        21,
        "support_map",
        support_heat,
        "BHLR support map",
        "Soft target-support region, not a segmentation mask",
    )

    support_overlay = draw_boxes(
        overlay_heatmap(
            rgb_semantic,
            support,
            alpha=0.50,
            robust=False,
        ),
        semantic_boxes,
    )

    save(
        22,
        "support_overlay",
        support_overlay,
        "Support map over RGB semantic input",
        "Check whether the support covers the UAV rather than only an edge",
    )

    first_sem_box = semantic_boxes[0] if len(semantic_boxes) > 0 else None
    support_zoom = panel(
        [
            (
                "RGB UAV",
                crop_around_box(
                    draw_boxes(rgb_semantic, semantic_boxes),
                    first_sem_box,
                    expand=5.0,
                    min_size=96,
                ),
            ),
            (
                "Support overlay",
                crop_around_box(
                    support_overlay,
                    first_sem_box,
                    expand=5.0,
                    min_size=96,
                ),
            ),
        ],
        cols=2,
        tile_w=500,
        tile_h=400,
    )

    save(
        23,
        "support_uav_zoom",
        support_zoom,
        "UAV-level support-region inspection",
        "The support should represent the target region, not only a thin contour",
    )

    gate = bhlr["detail_gate"][0, 0].detach().float().cpu().numpy()
    gate_heat = heatmap_bgr(
        gate,
        size=(rgb_semantic.shape[1], rgb_semantic.shape[0]),
        robust=False,
    )
    gate_heat = draw_boxes(gate_heat, semantic_boxes)

    save(
        24,
        "detail_gate",
        gate_heat,
        "Useful-detail gate",
        "Which lost details are useful for detection",
    )

    gate_overlay = draw_boxes(
        overlay_heatmap(
            rgb_semantic,
            gate,
            alpha=0.50,
            robust=False,
        ),
        semantic_boxes,
    )

    save(
        25,
        "detail_gate_overlay",
        gate_overlay,
        "Useful-detail gate over RGB semantic input",
        "High gate value does not mean object probability; it means detail usefulness",
    )

    usable_img = draw_boxes(
        overlay_heatmap(
            rgb_semantic,
            tensor_feature_map(usable_detail),
            alpha=0.56,
        ),
        semantic_boxes,
    )

    save(
        26,
        "usable_detail",
        usable_img,
        "Usable high-resolution lost detail",
        "support x detail_gate x projected_detail",
    )

    usable_overlay = draw_boxes(
        overlay_heatmap(
            rgb_semantic,
            tensor_feature_map(usable_detail),
            alpha=0.42,
        ),
        semantic_boxes,
    )

    save(
        27,
        "usable_detail_overlay",
        usable_overlay,
        "Lost detail actually allowed into RGB P3",
        "This is more important than the raw residual in 13/14",
    )

    before_bhlr = draw_boxes(
        overlay_heatmap(
            rgb_semantic,
            tensor_feature_map(raw_rgb_p3),
            alpha=0.52,
        ),
        semantic_boxes,
    )
    after_bhlr = draw_boxes(
        overlay_heatmap(
            rgb_semantic,
            tensor_feature_map(enhanced_rgb_p3),
            alpha=0.52,
        ),
        semantic_boxes,
    )

    before_after_panel = panel(
        [
            ("RGB P3 before BHLR", before_bhlr),
            ("RGB P3 after BHLR", after_bhlr),
        ],
        cols=2,
    )

    save(
        28,
        "rgb_before_after_bhlr",
        before_after_panel,
        "RGB P3 before / after BHLR",
        f"gamma={float(gamma.detach().float().cpu()):+.6f}",
    )

    enhancement = enhanced_rgb_p3 - raw_rgb_p3
    enhancement_img = draw_boxes(
        overlay_heatmap(
            rgb_semantic,
            tensor_feature_map(enhancement),
            alpha=0.58,
        ),
        semantic_boxes,
    )

    save(
        29,
        "bhlr_enhancement",
        enhancement_img,
        "Actual BHLR feature injection",
        "F_after - F_before = tanh(gamma) * usable_detail",
    )

    # =====================================================================
    # 30 FINAL DETECTION
    # =====================================================================
    pred_list = base_val.postprocess_predictions(
        preds=outputs["pred"],
        model=model,
        conf_thres=0.001,
        iou_thres=iou,
        max_det=max_det,
    )
    pred = pred_list[0] if pred_list else None

    final_image = draw_boxes(rgb_semantic, semantic_boxes, label="GT")
    final_image = draw_predictions(final_image, pred, conf_min=conf)

    save(
        30,
        "prediction_vs_gt",
        final_image,
        "Final detection vs GT",
        "red=GT, green=prediction",
    )

    # =====================================================================
    # Metadata
    # =====================================================================
    coarse_dx = float(coarse_offset[0, 0, 0, 0].detach().float().cpu())
    coarse_dy = float(coarse_offset[0, 1, 0, 0].detach().float().cpu())
    # coarse/fine/total offsets were already extracted above for the visual header.

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

    rgb_rel = debug_get(
        align,
        "rgb_frequency_reliability",
        "rgb_reliability",
    )
    tir_rel = debug_get(
        align,
        "tir_frequency_reliability",
        "tir_reliability",
    )

    metadata = {
        "weights": str(weights_path),
        "split": split,
        "dataset_index": int(index),
        "rgb_path": str(sample["rgb_path"]),
        "tir_path": str(sample["tir_path"]),
        "rlsfa": {
            "coarse_dx": coarse_dx,
            "coarse_dy": coarse_dy,
            "fine_dx": fine_dx,
            "fine_dy": fine_dy,
            "fine_confidence": float(fine_confidence.detach().float().cpu().reshape(-1)[0]),
            "fine_probability_max": float(fine_probability.detach().float().cpu().max()),
            "fine_argmax_dx": fine_argmax_dx,
            "fine_argmax_dy": fine_argmax_dy,
            "total_dx": total_dx,
            "total_dy": total_dy,
            "rgb_frequency_reliability_mean": float(rgb_rel.mean().detach().cpu()),
            "tir_frequency_reliability_mean": float(tir_rel.mean().detach().cpu()),
            "targetness_mean": float(debug_get(align, "targetness", "coarse_targetness").mean().detach().cpu()),
        },
        "bhlr": {
            "gamma": float(gamma.detach().float().cpu()),
            "support_mean": float(bhlr["support_map"].mean().detach().cpu()),
            "detail_gate_mean": float(bhlr["detail_gate"].mean().detach().cpu()),
            "usable_detail_mean_abs": float(usable_detail.abs().mean().detach().cpu()),
            "actual_enhancement_mean_abs": float(enhancement.abs().mean().detach().cpu()),
            "residual_mean_abs": float(residual_high.abs().mean().detach().cpu()),
        },
    }

    # Single-target GT sampling translation for reference.
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

    make_overview(
        overview_panels,
        out_dir / "overview.jpg",
        cols=5,
    )

    print(f"[OK] index={index} -> {out_dir}")


# ============================================================================
# CLI
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Focused RLSFA + BHLR internal visualization"
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
    model, _, cfg = load_rlsfa_bhlr_checkpoint(args.weights, device)
    model.eval()

    data_cfg = cfg.get("data", {})
    rgb_yaml = args.rgb or data_cfg.get("rgb")
    tir_yaml = args.tir or data_cfg.get("tir")
    rgb_imgsz = args.rgb_imgsz or int(data_cfg.get("rgb_imgsz", 1280))
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
        # preserve order while removing duplicates
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
