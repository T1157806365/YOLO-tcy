"""
visualize_hra.py
================

用途
----
指定：
    1. RGB 图像
    2. TIR 图像
    3. 已训练的 RGB-T HRA checkpoint

脚本会把 HRA-v1 在真实模型中的每一步运算可视化。

默认输出：
    output/
    ├── 00_input_pair.jpg
    ├── 01_final_detection_network.jpg
    ├── 02_final_detection_original.jpg
    ├── tensor_shapes.json
    ├── model_info.json
    │
    ├── P3/
    │   ├── 01_rgb_feature.jpg
    │   ├── 02_tir_feature.jpg
    │   ├── 03_tir_projected.jpg
    │   ├── 04_base_grid.png
    │   ├── 05_coarse_tir.jpg
    │   ├── 06_offset_dx.jpg
    │   ├── 07_offset_dy.jpg
    │   ├── 08_offset_magnitude.jpg
    │   ├── 09_offset_quiver.png
    │   ├── 10_refined_grid.png
    │   ├── 11_aligned_tir.jpg
    │   ├── 12_fused_feature.jpg
    │   ├── 13_pipeline_overview.png
    │   └── 14_top_channels.png
    │
    ├── P4/
    │   └── ...
    │
    └── P5/
        └── ...

运行示例
--------
conda activate tcy

cd /mnt/sda/taochangyong/Projects/Model/YOLO-tcy

python visualize_hra.py \
    --weights runs/rgbt/yolo26n_uavcb_rgbt_concat_640_b4_seed0/weights/best.pt \
    --rgb-image /mnt/sda/taochangyong/Projects/datasets/UAV-CB/RGB/images/test/building_1_000004.jpg \
    --tir-image /mnt/sda/taochangyong/Projects/datasets/UAV-CB/TIR/images/test/building_1_000004.jpg \
    --device 5 \
    --conf 0.25 \
    --save-tensors

说明
----
1. 本脚本不会修改模型。
2. 本脚本直接使用 checkpoint 内保存的训练 config。
3. RGB / TIR 预处理与 datasets/rgbt_dataset.py 保持一致：
       RGB: BGR -> RGB
       TIR: grayscale -> repeat 3 channels
       RGB/TIR 独立 LetterBox
       uint8 -> float32 / 255
4. HRA 中间张量不是通过 hook 猜出来，而是直接调用：
       HRAFusion(..., return_debug=True)
   因此可看到：
       tir_projected
       base_grid
       coarse_tir
       raw_offset
       bounded_offset
       refined_grid
       aligned_tir
       fused
5. 最终 Detect 输出仍使用模型原来的 YOLO Neck + Detect Head。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch


# ============================================================
# 1. Project path
# ============================================================

ROOT = Path(
    __file__
).resolve().parent

if str(ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(ROOT),
    )


# ============================================================
# 2. Project imports
# ============================================================

from datasets.rgbt_dataset import (
    letterbox,
)

from models.modules.hra_fusion import (
    HRAFusion,
)

from val_rgbt import (
    load_rgbt_checkpoint,
    postprocess_predictions,
    select_device,
)


# ============================================================
# 3. Basic helpers
# ============================================================

def parse_imgsz(
    value: Union[
        int,
        Sequence[int],
    ]
) -> Tuple[int, int]:
    """
    640 -> (640, 640)

    [512, 640] -> (512, 640)
    """

    if isinstance(
        value,
        int,
    ):
        return (
            value,
            value,
        )

    if isinstance(
        value,
        (list, tuple),
    ):

        if len(value) == 1:
            return (
                int(
                    value[0]
                ),
                int(
                    value[0]
                ),
            )

        if len(value) == 2:
            return (
                int(
                    value[0]
                ),
                int(
                    value[1]
                ),
            )

    raise ValueError(
        f"无法解析 imgsz={value}"
    )


def ensure_exists(
    path: Union[
        str,
        Path,
    ],
    name: str,
) -> Path:

    path = (
        Path(
            path
        )
        .expanduser()
        .resolve()
    )

    if not path.exists():

        raise FileNotFoundError(
            f"\n{name} 不存在:\n{path}\n"
        )

    return path


def safe_json_value(
    value,
):
    """
    Convert common numpy / torch values into JSON-compatible values.
    """

    if isinstance(
        value,
        Path,
    ):
        return str(
            value
        )

    if torch.is_tensor(
        value
    ):

        if value.numel() == 1:
            return float(
                value.detach().cpu().item()
            )

        return (
            value
            .detach()
            .cpu()
            .tolist()
        )

    if isinstance(
        value,
        np.ndarray,
    ):
        return value.tolist()

    if isinstance(
        value,
        (
            np.floating,
            np.integer,
        ),
    ):
        return value.item()

    if isinstance(
        value,
        dict,
    ):

        return {
            str(k):
                safe_json_value(
                    v
                )
            for k, v
            in value.items()
        }

    if isinstance(
        value,
        (
            list,
            tuple,
        ),
    ):

        return [
            safe_json_value(
                x
            )
            for x in value
        ]

    return value


def save_json(
    path: Path,
    data: Dict,
):

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            safe_json_value(
                data
            ),
            f,
            ensure_ascii=False,
            indent=2,
        )


# ============================================================
# 4. Image preprocessing
# ============================================================

def read_rgb(
    path: Path,
) -> np.ndarray:
    """
    Read RGB image exactly like RGBTPairedDataset.

    Returns:
        RGB uint8 HWC
    """

    image = cv2.imread(
        str(
            path
        ),
        cv2.IMREAD_COLOR,
    )

    if image is None:

        raise RuntimeError(
            f"RGB 图像读取失败:\n{path}"
        )

    return cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB,
    )


def read_tir(
    path: Path,
) -> np.ndarray:
    """
    Read thermal image exactly like RGBTPairedDataset.

    TIR:
        grayscale
        ->
        repeat to 3 channels

    Returns:
        pseudo-RGB uint8 HWC
    """

    tir = cv2.imread(
        str(
            path
        ),
        cv2.IMREAD_GRAYSCALE,
    )

    if tir is None:

        raise RuntimeError(
            f"TIR 图像读取失败:\n{path}"
        )

    return np.stack(
        [
            tir,
            tir,
            tir,
        ],
        axis=-1,
    )


def preprocess_one(
    image_rgb: np.ndarray,
    imgsz: Union[
        int,
        Sequence[int],
    ],
):
    """
    Validation-style preprocessing.

    Important:
        scaleup=False

    This matches validation/inference behavior in the current dataset.
    """

    empty_boxes = np.zeros(
        (
            0,
            4,
        ),
        dtype=np.float32,
    )

    (
        image_lb,
        _,
        ratio_pad,
    ) = letterbox(
        image_rgb,
        empty_boxes,
        imgsz,
        scaleup=False,
    )

    image_chw = (
        np.ascontiguousarray(
            image_lb.transpose(
                2,
                0,
                1,
            )
        )
    )

    tensor = (
        torch.from_numpy(
            image_chw
        )
        .float()
        / 255.0
    )

    return (
        tensor,
        image_lb,
        ratio_pad,
    )


# ============================================================
# 5. Basic visualization utilities
# ============================================================

def normalize_map(
    array: np.ndarray,
    low_percentile: float = 1.0,
    high_percentile: float = 99.0,
) -> np.ndarray:
    """
    Robust normalization to [0, 1].
    """

    array = np.asarray(
        array,
        dtype=np.float32,
    )

    finite = np.isfinite(
        array
    )

    if not finite.any():

        return np.zeros_like(
            array,
            dtype=np.float32,
        )

    values = array[
        finite
    ]

    lo = float(
        np.percentile(
            values,
            low_percentile,
        )
    )

    hi = float(
        np.percentile(
            values,
            high_percentile,
        )
    )

    if (
        not np.isfinite(lo)
        or not np.isfinite(hi)
        or hi <= lo
    ):

        lo = float(
            np.min(
                values
            )
        )

        hi = float(
            np.max(
                values
            )
        )

    if hi <= lo:

        return np.zeros_like(
            array,
            dtype=np.float32,
        )

    out = (
        array - lo
    ) / (
        hi - lo
    )

    return np.clip(
        out,
        0.0,
        1.0,
    )


def feature_to_scalar_map(
    feature: torch.Tensor,
) -> np.ndarray:
    """
    Feature tensor:
        [B, C, H, W]

    ->
    one activation map:
        [H, W]

    Method:
        mean(abs(feature), channels)
    """

    if feature.ndim != 4:

        raise ValueError(
            "feature 必须为 [B,C,H,W]"
        )

    x = (
        feature[
            0
        ]
        .detach()
        .float()
        .cpu()
    )

    heat = (
        x.abs()
        .mean(
            dim=0
        )
        .numpy()
    )

    return normalize_map(
        heat
    )


def scalar_to_colormap(
    scalar_map: np.ndarray,
) -> np.ndarray:
    """
    scalar [0,1]
    ->
    RGB heatmap
    """

    image_u8 = (
        np.clip(
            scalar_map,
            0.0,
            1.0,
        )
        * 255.0
    ).astype(
        np.uint8
    )

    bgr = cv2.applyColorMap(
        image_u8,
        cv2.COLORMAP_JET,
    )

    return cv2.cvtColor(
        bgr,
        cv2.COLOR_BGR2RGB,
    )


def overlay_heatmap(
    base_rgb: np.ndarray,
    scalar_map: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    """
    Overlay feature heatmap onto image.
    """

    h, w = base_rgb.shape[:2]

    heat_rgb = scalar_to_colormap(
        scalar_map
    )

    heat_rgb = cv2.resize(
        heat_rgb,
        (
            w,
            h,
        ),
        interpolation=cv2.INTER_LINEAR,
    )

    out = cv2.addWeighted(
        base_rgb,
        1.0 - alpha,
        heat_rgb,
        alpha,
        0.0,
    )

    return out


def put_title(
    image_rgb: np.ndarray,
    title: str,
    subtitle: str = "",
) -> np.ndarray:
    """
    Add an English/ASCII title bar.
    """

    image = image_rgb.copy()

    h, w = image.shape[:2]

    bar_h = max(
        56,
        int(
            0.08 * h
        ),
    )

    canvas = np.zeros(
        (
            h + bar_h,
            w,
            3,
        ),
        dtype=np.uint8,
    )

    canvas[
        bar_h:
    ] = image

    cv2.putText(
        canvas,
        title,
        (
            12,
            24,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (
            255,
            255,
            255,
        ),
        2,
        cv2.LINE_AA,
    )

    if subtitle:

        cv2.putText(
            canvas,
            subtitle,
            (
                12,
                47,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (
                210,
                210,
                210,
            ),
            1,
            cv2.LINE_AA,
        )

    return canvas


def save_rgb_image(
    path: Path,
    image_rgb: np.ndarray,
):

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    cv2.imwrite(
        str(
            path
        ),
        cv2.cvtColor(
            image_rgb,
            cv2.COLOR_RGB2BGR,
        ),
    )


def save_feature_triplet(
    path: Path,
    feature: torch.Tensor,
    base_rgb: np.ndarray,
    title: str,
):
    """
    Save:
        base image | activation map | overlay

    This makes each HRA step easier to explain.
    """

    scalar = feature_to_scalar_map(
        feature
    )

    heat = scalar_to_colormap(
        scalar
    )

    h, w = base_rgb.shape[:2]

    heat_resized = cv2.resize(
        heat,
        (
            w,
            h,
        ),
        interpolation=cv2.INTER_LINEAR,
    )

    overlay = overlay_heatmap(
        base_rgb,
        scalar,
    )

    panel = np.concatenate(
        [
            base_rgb,
            heat_resized,
            overlay,
        ],
        axis=1,
    )

    shape_text = (
        str(
            list(
                feature.shape
            )
        )
    )

    panel = put_title(
        panel,
        title,
        f"shape={shape_text} | left=input, middle=activation, right=overlay",
    )

    save_rgb_image(
        path,
        panel,
    )


def save_scalar_triplet(
    path: Path,
    scalar_map: np.ndarray,
    base_rgb: np.ndarray,
    title: str,
    subtitle: str = "",
):
    """
    Save scalar map and overlay.
    """

    scalar = normalize_map(
        scalar_map
    )

    heat = scalar_to_colormap(
        scalar
    )

    h, w = base_rgb.shape[:2]

    heat = cv2.resize(
        heat,
        (
            w,
            h,
        ),
        interpolation=cv2.INTER_LINEAR,
    )

    overlay = overlay_heatmap(
        base_rgb,
        scalar,
    )

    panel = np.concatenate(
        [
            base_rgb,
            heat,
            overlay,
        ],
        axis=1,
    )

    panel = put_title(
        panel,
        title,
        subtitle,
    )

    save_rgb_image(
        path,
        panel,
    )


# ============================================================
# 6. Input pair visualization
# ============================================================

def save_input_pair(
    path: Path,
    rgb_input: np.ndarray,
    tir_input: np.ndarray,
    rgb_original_shape,
    tir_original_shape,
):
    """
    Save RGB and TIR network inputs side-by-side.
    """

    target_h = max(
        rgb_input.shape[0],
        tir_input.shape[0],
    )

    def resize_height(
        image,
        h,
    ):

        if image.shape[0] == h:
            return image

        scale = (
            h
            / image.shape[0]
        )

        new_w = int(
            round(
                image.shape[1]
                * scale
            )
        )

        return cv2.resize(
            image,
            (
                new_w,
                h,
            ),
            interpolation=cv2.INTER_LINEAR,
        )

    rgb_show = resize_height(
        rgb_input,
        target_h,
    )

    tir_show = resize_height(
        tir_input,
        target_h,
    )

    separator = np.full(
        (
            target_h,
            12,
            3,
        ),
        255,
        dtype=np.uint8,
    )

    panel = np.concatenate(
        [
            rgb_show,
            separator,
            tir_show,
        ],
        axis=1,
    )

    panel = put_title(
        panel,
        "RGB-T network inputs",
        (
            f"RGB original={tuple(rgb_original_shape)} | "
            f"TIR original={tuple(tir_original_shape)} | "
            f"after independent LetterBox"
        ),
    )

    save_rgb_image(
        path,
        panel,
    )


# ============================================================
# 7. Grid / offset visualization
# ============================================================

def tensor_grid_to_numpy(
    grid: torch.Tensor,
) -> np.ndarray:
    """
    [B,H,W,2] -> [H,W,2]
    """

    return (
        grid[
            0
        ]
        .detach()
        .float()
        .cpu()
        .numpy()
    )


def choose_grid_stride(
    h: int,
    w: int,
    max_lines: int = 18,
) -> int:

    return max(
        1,
        int(
            math.ceil(
                max(
                    h,
                    w,
                )
                / max_lines
            )
        ),
    )


def save_grid_plot(
    path: Path,
    base_grid: torch.Tensor,
    refined_grid: torch.Tensor | None = None,
    title: str = "Sampling grid",
):
    """
    Visualize normalized sampling grid.

    If refined_grid is given:
        regular base grid and learned refined grid are shown together.
    """

    base = tensor_grid_to_numpy(
        base_grid
    )

    refined = (
        tensor_grid_to_numpy(
            refined_grid
        )
        if refined_grid is not None
        else None
    )

    h, w = base.shape[:2]

    stride = choose_grid_stride(
        h,
        w,
    )

    fig = plt.figure(
        figsize=(
            9,
            9,
        )
    )

    ax = fig.add_subplot(
        111
    )

    # --------------------------------------------------------
    # Base grid
    # --------------------------------------------------------

    for y in range(
        0,
        h,
        stride,
    ):

        ax.plot(
            base[
                y,
                ::stride,
                0,
            ],
            base[
                y,
                ::stride,
                1,
            ],
            linewidth=0.8,
            alpha=0.55,
            label=(
                "base grid"
                if y == 0
                else None
            ),
        )

    for x in range(
        0,
        w,
        stride,
    ):

        ax.plot(
            base[
                ::stride,
                x,
                0,
            ],
            base[
                ::stride,
                x,
                1,
            ],
            linewidth=0.8,
            alpha=0.55,
        )

    # --------------------------------------------------------
    # Refined grid
    # --------------------------------------------------------

    if refined is not None:

        for y in range(
            0,
            h,
            stride,
        ):

            ax.plot(
                refined[
                    y,
                    ::stride,
                    0,
                ],
                refined[
                    y,
                    ::stride,
                    1,
                ],
                linewidth=1.1,
                alpha=0.85,
                linestyle="--",
                label=(
                    "refined grid"
                    if y == 0
                    else None
                ),
            )

        for x in range(
            0,
            w,
            stride,
        ):

            ax.plot(
                refined[
                    ::stride,
                    x,
                    0,
                ],
                refined[
                    ::stride,
                    x,
                    1,
                ],
                linewidth=1.1,
                alpha=0.85,
                linestyle="--",
            )

    ax.set_xlim(
        -1.05,
        1.05,
    )

    ax.set_ylim(
        1.05,
        -1.05,
    )

    ax.set_aspect(
        "equal",
        adjustable="box",
    )

    ax.set_xlabel(
        "normalized x"
    )

    ax.set_ylabel(
        "normalized y"
    )

    ax.set_title(
        title
    )

    ax.grid(
        True,
        alpha=0.2,
    )

    ax.legend()

    fig.tight_layout()

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        path,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


def save_offset_quiver(
    path: Path,
    bounded_offset: torch.Tensor,
    tir_hw: Tuple[int, int],
    title: str,
):
    """
    bounded_offset:
        [B,2,H,W]

    channel 0:
        dx normalized

    channel 1:
        dy normalized

    Also converts normalized displacement into approximate
    TIR feature-pixel displacement for interpretation.
    """

    offset = (
        bounded_offset[
            0
        ]
        .detach()
        .float()
        .cpu()
        .numpy()
    )

    dx = offset[
        0
    ]

    dy = offset[
        1
    ]

    h, w = dx.shape

    stride = max(
        1,
        int(
            math.ceil(
                max(
                    h,
                    w,
                )
                / 24
            )
        ),
    )

    yy, xx = np.mgrid[
        0:h:stride,
        0:w:stride,
    ]

    dx_s = dx[
        ::stride,
        ::stride,
    ]

    dy_s = dy[
        ::stride,
        ::stride,
    ]

    # align_corners=False approximate conversion:
    # normalized 2.0 spans one full feature dimension.
    tir_h, tir_w = tir_hw

    dx_px = (
        dx
        * (
            tir_w / 2.0
        )
    )

    dy_px = (
        dy
        * (
            tir_h / 2.0
        )
    )

    mag_px = np.sqrt(
        dx_px ** 2
        + dy_px ** 2
    )

    fig = plt.figure(
        figsize=(
            10,
            8,
        )
    )

    ax = fig.add_subplot(
        111
    )

    magnitude = np.sqrt(
        dx ** 2
        + dy ** 2
    )

    ax.imshow(
        magnitude,
        interpolation="nearest",
    )

    # Convert normalized sampling displacement to approximate
    # TIR-feature pixel displacement so the arrows are visible
    # and physically easier to interpret.
    dx_px_s = dx_s * (
        tir_w / 2.0
    )

    dy_px_s = dy_s * (
        tir_h / 2.0
    )

    ax.quiver(
        xx,
        yy,
        dx_px_s,
        dy_px_s,
        angles="xy",
        scale_units="xy",
        scale=1.0,
    )

    ax.set_title(
        (
            f"{title}\n"
            f"mean displacement ~= {mag_px.mean():.3f} TIR-feature pixels, "
            f"max ~= {mag_px.max():.3f}"
        )
    )

    ax.set_xlabel(
        "RGB feature x"
    )

    ax.set_ylabel(
        "RGB feature y"
    )

    fig.tight_layout()

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        path,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    return {
        "mean_dx_normalized":
            float(
                np.mean(
                    dx
                )
            ),

        "mean_dy_normalized":
            float(
                np.mean(
                    dy
                )
            ),

        "mean_abs_dx_normalized":
            float(
                np.mean(
                    np.abs(
                        dx
                    )
                )
            ),

        "mean_abs_dy_normalized":
            float(
                np.mean(
                    np.abs(
                        dy
                    )
                )
            ),

        "max_abs_dx_normalized":
            float(
                np.max(
                    np.abs(
                        dx
                    )
                )
            ),

        "max_abs_dy_normalized":
            float(
                np.max(
                    np.abs(
                        dy
                    )
                )
            ),

        "mean_displacement_tir_feature_pixels":
            float(
                mag_px.mean()
            ),

        "max_displacement_tir_feature_pixels":
            float(
                mag_px.max()
            ),
    }


# ============================================================
# 8. Channel montage
# ============================================================

def select_top_channels(
    feature: torch.Tensor,
    top_k: int,
) -> List[int]:
    """
    Choose channels with largest spatial standard deviation.
    """

    x = (
        feature[
            0
        ]
        .detach()
        .float()
        .cpu()
    )

    score = (
        x.flatten(
            1
        )
        .std(
            dim=1
        )
    )

    k = min(
        int(
            top_k
        ),
        int(
            x.shape[0]
        ),
    )

    indices = (
        torch.topk(
            score,
            k=k,
        )
        .indices
        .tolist()
    )

    return [
        int(
            x
        )
        for x in indices
    ]


def save_channel_montage(
    path: Path,
    tensors: Dict[
        str,
        torch.Tensor,
    ],
    top_k: int = 6,
):
    """
    For each tensor, show its strongest/most-varying channels.

    Recommended tensors:
        RGB
        Coarse TIR
        Aligned TIR
        Fused
    """

    items = list(
        tensors.items()
    )

    rows = len(
        items
    )

    cols = int(
        top_k
    )

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(
            3 * cols,
            3 * rows,
        ),
        squeeze=False,
    )

    for row, (
        name,
        tensor,
    ) in enumerate(
        items
    ):

        channels = select_top_channels(
            tensor,
            top_k,
        )

        for col in range(
            cols
        ):

            ax = axes[
                row,
                col,
            ]

            ax.axis(
                "off"
            )

            if col >= len(
                channels
            ):
                continue

            ch = channels[
                col
            ]

            feature_map = (
                tensor[
                    0,
                    ch,
                ]
                .detach()
                .float()
                .cpu()
                .numpy()
            )

            ax.imshow(
                normalize_map(
                    np.abs(
                        feature_map
                    )
                ),
            )

            ax.set_title(
                (
                    f"{name}\n"
                    f"channel={ch}"
                )
            )

    fig.tight_layout()

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


# ============================================================
# 9. Detection drawing
# ============================================================

def draw_detections(
    image_rgb: np.ndarray,
    pred: Dict,
    conf_thres: float,
    names: Dict,
) -> np.ndarray:
    """
    Draw model detections in current network coordinate space.
    """

    image = image_rgb.copy()

    boxes = (
        pred[
            "bboxes"
        ]
        .detach()
        .float()
        .cpu()
        .numpy()
    )

    confs = (
        pred[
            "conf"
        ]
        .detach()
        .float()
        .cpu()
        .numpy()
    )

    classes = (
        pred[
            "cls"
        ]
        .detach()
        .float()
        .cpu()
        .numpy()
    )

    h, w = image.shape[:2]

    for box, conf, cls_id in zip(
        boxes,
        confs,
        classes,
    ):

        conf = float(
            conf
        )

        if conf < conf_thres:
            continue

        cls_id = int(
            cls_id
        )

        x1, y1, x2, y2 = [
            int(
                round(
                    float(v)
                )
            )
            for v in box
        ]

        x1 = int(
            np.clip(
                x1,
                0,
                w - 1,
            )
        )

        y1 = int(
            np.clip(
                y1,
                0,
                h - 1,
            )
        )

        x2 = int(
            np.clip(
                x2,
                0,
                w - 1,
            )
        )

        y2 = int(
            np.clip(
                y2,
                0,
                h - 1,
            )
        )

        cv2.rectangle(
            image,
            (
                x1,
                y1,
            ),
            (
                x2,
                y2,
            ),
            (
                255,
                0,
                0,
            ),
            2,
        )

        class_name = str(
            names.get(
                cls_id,
                cls_id,
            )
        )

        text = (
            f"{class_name} "
            f"{conf:.2f}"
        )

        cv2.putText(
            image,
            text,
            (
                x1,
                max(
                    y1 - 5,
                    16,
                ),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (
                255,
                0,
                0,
            ),
            2,
            cv2.LINE_AA,
        )

    return image


def network_boxes_to_original(
    pred: Dict,
    ratio_pad: Dict,
    original_hw: Tuple[
        int,
        int,
    ],
) -> Dict:
    """
    Convert RGB network-space prediction boxes back to original RGB image.
    """

    boxes = (
        pred[
            "bboxes"
        ]
        .detach()
        .clone()
        .float()
        .cpu()
    )

    ratio = float(
        ratio_pad[
            "ratio"
        ]
    )

    pad_x, pad_y = (
        ratio_pad[
            "pad"
        ]
    )

    if boxes.shape[0]:

        boxes[
            :,
            [0, 2],
        ] -= float(
            pad_x
        )

        boxes[
            :,
            [1, 3],
        ] -= float(
            pad_y
        )

        boxes /= ratio

        h0, w0 = (
            original_hw
        )

        boxes[
            :,
            [0, 2],
        ].clamp_(
            0,
            w0 - 1,
        )

        boxes[
            :,
            [1, 3],
        ].clamp_(
            0,
            h0 - 1,
        )

    return {
        "bboxes":
            boxes,

        "conf":
            pred[
                "conf"
            ]
            .detach()
            .cpu(),

        "cls":
            pred[
                "cls"
            ]
            .detach()
            .cpu(),
    }


# ============================================================
# 10. Scale naming
# ============================================================

def infer_scale_names(
    model,
    rgb_outputs,
) -> Dict[int, str]:
    """
    Dynamically map fusion indices to P3/P4/P5 according to spatial size.

    Largest feature map:
        P3

    Next:
        P4

    Smallest:
        P5
    """

    info = []

    for index in (
        model.fusion_indices
    ):

        feature = (
            rgb_outputs[
                index
            ]
        )

        area = int(
            feature.shape[-2]
            * feature.shape[-1]
        )

        info.append(
            (
                index,
                area,
            )
        )

    info = sorted(
        info,
        key=lambda x:
            x[1],
        reverse=True,
    )

    return {
        int(index):
            f"P{3 + rank}"
        for rank, (
            index,
            _,
        ) in enumerate(
            info
        )
    }


# ============================================================
# 11. HRA forward with full debug tensors
# ============================================================

@torch.inference_mode()
def forward_hra_debug(
    model,
    rgb: torch.Tensor,
    tir: torch.Tensor,
):
    """
    Reproduce model forward while explicitly asking each HRAFusion
    module for all intermediate tensors.

    This does NOT alter model parameters.
    """

    rgb_outputs = (
        model.forward_rgb_backbone(
            rgb
        )
    )

    tir_outputs = (
        model.forward_tir_backbone(
            tir
        )
    )

    scale_names = infer_scale_names(
        model,
        rgb_outputs,
    )

    fused_outputs = list(
        rgb_outputs
    )

    hra_debug = {}

    for index in (
        model.fusion_indices
    ):

        fusion_module = (
            model.fusions[
                str(
                    index
                )
            ]
        )

        if not isinstance(
            fusion_module,
            HRAFusion,
        ):

            raise TypeError(
                "\n当前 checkpoint 的 fusion module 不是 HRAFusion。\n"
                f"layer={index}\n"
                f"type={type(fusion_module).__name__}\n\n"
                "请确认:\n"
                "  model.fusion = hra\n"
                "并确认 best.pt 来自 HRA 实验。\n"
            )

        debug = fusion_module(
            rgb_outputs[
                index
            ],
            tir_outputs[
                index
            ],
            return_debug=True,
        )

        fused_outputs[
            index
        ] = debug[
            "fused"
        ]

        hra_debug[
            scale_names[
                index
            ]
        ] = {
            "layer_index":
                int(
                    index
                ),

            "module":
                fusion_module,

            "debug":
                debug,
        }

    preds = model.forward_head(
        fused_outputs
    )

    return {
        "rgb_outputs":
            rgb_outputs,

        "tir_outputs":
            tir_outputs,

        "fused_outputs":
            fused_outputs,

        "hra_debug":
            hra_debug,

        "preds":
            preds,
    }


# ============================================================
# 12. Pipeline overview
# ============================================================

def get_map_for_overview(
    tensor: torch.Tensor,
) -> np.ndarray:

    return feature_to_scalar_map(
        tensor
    )


def save_pipeline_overview(
    path: Path,
    scale_name: str,
    debug: Dict[
        str,
        torch.Tensor,
    ],
):
    """
    3 x 3 HRA pipeline summary:

        RGB feature
        TIR feature
        projected TIR

        coarse TIR
        dx
        dy

        offset magnitude
        aligned TIR
        fused feature
    """

    raw_offset = (
        debug[
            "bounded_offset"
        ][
            0
        ]
        .detach()
        .float()
        .cpu()
        .numpy()
    )

    dx = raw_offset[
        0
    ]

    dy = raw_offset[
        1
    ]

    magnitude = np.sqrt(
        dx ** 2
        + dy ** 2
    )

    panels = [
        (
            "1 RGB feature",
            get_map_for_overview(
                debug[
                    "rgb"
                ]
            ),
            list(
                debug[
                    "rgb"
                ].shape
            ),
        ),

        (
            "2 TIR feature",
            get_map_for_overview(
                debug[
                    "tir"
                ]
            ),
            list(
                debug[
                    "tir"
                ].shape
            ),
        ),

        (
            "3 TIR projected",
            get_map_for_overview(
                debug[
                    "tir_projected"
                ]
            ),
            list(
                debug[
                    "tir_projected"
                ].shape
            ),
        ),

        (
            "4 Coarse TIR",
            get_map_for_overview(
                debug[
                    "coarse_tir"
                ]
            ),
            list(
                debug[
                    "coarse_tir"
                ].shape
            ),
        ),

        (
            "5 Offset dx",
            normalize_map(
                dx
            ),
            list(
                debug[
                    "bounded_offset"
                ].shape
            ),
        ),

        (
            "6 Offset dy",
            normalize_map(
                dy
            ),
            list(
                debug[
                    "bounded_offset"
                ].shape
            ),
        ),

        (
            "7 Offset magnitude",
            normalize_map(
                magnitude
            ),
            list(
                magnitude.shape
            ),
        ),

        (
            "8 Aligned TIR",
            get_map_for_overview(
                debug[
                    "aligned_tir"
                ]
            ),
            list(
                debug[
                    "aligned_tir"
                ].shape
            ),
        ),

        (
            "9 HRA fused",
            get_map_for_overview(
                debug[
                    "fused"
                ]
            ),
            list(
                debug[
                    "fused"
                ].shape
            ),
        ),
    ]

    fig, axes = plt.subplots(
        3,
        3,
        figsize=(
            15,
            14,
        ),
    )

    for ax, (
        name,
        data,
        shape,
    ) in zip(
        axes.flat,
        panels,
    ):

        ax.imshow(
            data,
        )

        ax.set_title(
            (
                f"{name}\n"
                f"shape={shape}"
            ),
            fontsize=10,
        )

        ax.axis(
            "off"
        )

    fig.suptitle(
        (
            f"HRA-v1 {scale_name} "
            "step-by-step computation"
        ),
        fontsize=16,
    )

    fig.tight_layout(
        rect=[
            0,
            0,
            1,
            0.96,
        ]
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        path,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


# ============================================================
# 13. Visualize one HRA scale
# ============================================================

def visualize_scale(
    scale_dir: Path,
    scale_name: str,
    debug: Dict,
    rgb_input: np.ndarray,
    tir_input: np.ndarray,
    top_channels: int,
    save_tensors: bool,
):
    """
    Save every important HRA step at one feature scale.
    """

    scale_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # 1. RGB feature
    # ========================================================

    save_feature_triplet(
        scale_dir
        / "01_rgb_feature.jpg",

        debug[
            "rgb"
        ],

        rgb_input,

        (
            f"{scale_name} Step 1 - "
            "RGB feature"
        ),
    )

    # ========================================================
    # 2. Raw TIR feature
    # ========================================================

    save_feature_triplet(
        scale_dir
        / "02_tir_feature.jpg",

        debug[
            "tir"
        ],

        tir_input,

        (
            f"{scale_name} Step 2 - "
            "TIR feature"
        ),
    )

    # ========================================================
    # 3. TIR projected
    # ========================================================

    save_feature_triplet(
        scale_dir
        / "03_tir_projected.jpg",

        debug[
            "tir_projected"
        ],

        tir_input,

        (
            f"{scale_name} Step 3 - "
            "TIR channel projection"
        ),
    )

    # ========================================================
    # 4. Base grid
    # ========================================================

    save_grid_plot(
        scale_dir
        / "04_base_grid.png",

        base_grid=(
            debug[
                "base_grid"
            ]
        ),

        refined_grid=None,

        title=(
            f"{scale_name} base sampling grid G0"
        ),
    )

    # ========================================================
    # 5. Coarse TIR
    # ========================================================

    save_feature_triplet(
        scale_dir
        / "05_coarse_tir.jpg",

        debug[
            "coarse_tir"
        ],

        rgb_input,

        (
            f"{scale_name} Step 5 - "
            "coarse sampled TIR"
        ),
    )

    # ========================================================
    # 6-8. Offset components
    # ========================================================

    bounded = (
        debug[
            "bounded_offset"
        ][
            0
        ]
        .detach()
        .float()
        .cpu()
        .numpy()
    )

    dx = bounded[
        0
    ]

    dy = bounded[
        1
    ]

    magnitude = np.sqrt(
        dx ** 2
        + dy ** 2
    )

    save_scalar_triplet(
        scale_dir
        / "06_offset_dx.jpg",

        dx,

        rgb_input,

        (
            f"{scale_name} Step 6 - "
            "bounded offset dx"
        ),

        (
            f"normalized mean={dx.mean():.6f}, "
            f"max_abs={np.abs(dx).max():.6f}"
        ),
    )

    save_scalar_triplet(
        scale_dir
        / "07_offset_dy.jpg",

        dy,

        rgb_input,

        (
            f"{scale_name} Step 7 - "
            "bounded offset dy"
        ),

        (
            f"normalized mean={dy.mean():.6f}, "
            f"max_abs={np.abs(dy).max():.6f}"
        ),
    )

    save_scalar_triplet(
        scale_dir
        / "08_offset_magnitude.jpg",

        magnitude,

        rgb_input,

        (
            f"{scale_name} Step 8 - "
            "offset magnitude"
        ),

        (
            f"mean={magnitude.mean():.6f}, "
            f"max={magnitude.max():.6f}"
        ),
    )

    tir_hw = (
        int(
            debug[
                "tir_projected"
            ].shape[-2]
        ),
        int(
            debug[
                "tir_projected"
            ].shape[-1]
        ),
    )

    offset_stats = save_offset_quiver(
        scale_dir
        / "09_offset_quiver.png",

        bounded_offset=(
            debug[
                "bounded_offset"
            ]
        ),

        tir_hw=tir_hw,

        title=(
            f"{scale_name} learned offset field"
        ),
    )

    # ========================================================
    # 10. Refined grid
    # ========================================================

    save_grid_plot(
        scale_dir
        / "10_refined_grid.png",

        base_grid=(
            debug[
                "base_grid"
            ]
        ),

        refined_grid=(
            debug[
                "refined_grid"
            ]
        ),

        title=(
            f"{scale_name} base grid vs refined grid"
        ),
    )

    # ========================================================
    # 11. Refined / aligned TIR
    # ========================================================

    save_feature_triplet(
        scale_dir
        / "11_aligned_tir.jpg",

        debug[
            "aligned_tir"
        ],

        rgb_input,

        (
            f"{scale_name} Step 11 - "
            "refined aligned TIR"
        ),
    )

    # ========================================================
    # 12. Fused HRA output
    # ========================================================

    save_feature_triplet(
        scale_dir
        / "12_fused_feature.jpg",

        debug[
            "fused"
        ],

        rgb_input,

        (
            f"{scale_name} Step 12 - "
            "HRA fused feature"
        ),
    )

    # ========================================================
    # 13. 3x3 overview
    # ========================================================

    save_pipeline_overview(
        scale_dir
        / "13_pipeline_overview.png",

        scale_name=scale_name,

        debug=debug,
    )

    # ========================================================
    # 14. Strong channels
    # ========================================================

    if top_channels > 0:

        save_channel_montage(
            scale_dir
            / "14_top_channels.png",

            tensors={
                "RGB":
                    debug[
                        "rgb"
                    ],

                "Coarse TIR":
                    debug[
                        "coarse_tir"
                    ],

                "Aligned TIR":
                    debug[
                        "aligned_tir"
                    ],

                "HRA Fused":
                    debug[
                        "fused"
                    ],
            },

            top_k=top_channels,
        )

    # ========================================================
    # Optional raw tensors
    # ========================================================

    if save_tensors:

        tensor_dir = (
            scale_dir
            / "tensors"
        )

        tensor_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        for key, value in (
            debug.items()
        ):

            if torch.is_tensor(
                value
            ):

                torch.save(
                    value
                    .detach()
                    .cpu(),

                    tensor_dir
                    / f"{key}.pt",
                )

    # ========================================================
    # Tensor shapes / statistics
    # ========================================================

    tensor_shapes = {}

    for key, value in (
        debug.items()
    ):

        if torch.is_tensor(
            value
        ):

            tensor_shapes[
                key
            ] = {
                "shape":
                    list(
                        value.shape
                    ),

                "dtype":
                    str(
                        value.dtype
                    ),

                "min":
                    float(
                        value.detach()
                        .float()
                        .min()
                        .cpu()
                        .item()
                    ),

                "max":
                    float(
                        value.detach()
                        .float()
                        .max()
                        .cpu()
                        .item()
                    ),

                "mean":
                    float(
                        value.detach()
                        .float()
                        .mean()
                        .cpu()
                        .item()
                    ),
            }

    return {
        "tensor_shapes":
            tensor_shapes,

        "offset_stats":
            offset_stats,
    }


# ============================================================
# 14. Main
# ============================================================

@torch.inference_mode()
def main(
    args,
):

    # ========================================================
    # Paths / device
    # ========================================================

    weights = ensure_exists(
        args.weights,
        "weights",
    )

    rgb_path = ensure_exists(
        args.rgb_image,
        "RGB image",
    )

    tir_path = ensure_exists(
        args.tir_image,
        "TIR image",
    )

    device = select_device(
        args.device
    )

    # ========================================================
    # Load checkpoint + architecture
    # ========================================================

    (
        model,
        ckpt,
        cfg,
    ) = load_rgbt_checkpoint(
        str(
            weights
        ),
        device,
    )

    fusion_type = str(
        ckpt.get(
            "fusion",
            cfg.get(
                "model",
                {}
            ).get(
                "fusion",
                "",
            ),
        )
    ).lower()

    if fusion_type != "hra":

        raise RuntimeError(
            "\n这个可视化脚本用于 HRA 模型。\n"
            f"当前 checkpoint fusion={fusion_type}\n\n"
            "请指定 HRA 训练得到的 best.pt / last.pt。\n"
        )

    # ========================================================
    # Recover input sizes from checkpoint config
    # ========================================================

    data_cfg = cfg.get(
        "data",
        {},
    )

    rgb_imgsz = (
        args.rgb_imgsz
        if args.rgb_imgsz is not None
        else data_cfg.get(
            "rgb_imgsz",
            640,
        )
    )

    tir_imgsz = (
        args.tir_imgsz
        if args.tir_imgsz is not None
        else data_cfg.get(
            "tir_imgsz",
            640,
        )
    )

    rgb_hw = parse_imgsz(
        rgb_imgsz
    )

    tir_hw = parse_imgsz(
        tir_imgsz
    )

    # ========================================================
    # Output dir
    # ========================================================

    if args.output is None:

        output_dir = (
            ROOT
            / "runs"
            / "visualize_hra"
            / (
                f"{rgb_path.stem}"
                f"__"
                f"{weights.parent.parent.name}"
            )
        )

    else:

        output_dir = (
            Path(
                args.output
            )
            .expanduser()
            .resolve()
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # Read original images
    # ========================================================

    rgb_original = read_rgb(
        rgb_path
    )

    tir_original = read_tir(
        tir_path
    )

    rgb_original_hw = (
        rgb_original.shape[
            :2
        ]
    )

    tir_original_hw = (
        tir_original.shape[
            :2
        ]
    )

    # ========================================================
    # Preprocess exactly like validation
    # ========================================================

    (
        rgb_tensor,
        rgb_input,
        rgb_ratio_pad,
    ) = preprocess_one(
        rgb_original,
        rgb_hw,
    )

    (
        tir_tensor,
        tir_input,
        tir_ratio_pad,
    ) = preprocess_one(
        tir_original,
        tir_hw,
    )

    rgb_tensor = (
        rgb_tensor
        .unsqueeze(
            0
        )
        .to(
            device
        )
    )

    tir_tensor = (
        tir_tensor
        .unsqueeze(
            0
        )
        .to(
            device
        )
    )

    # ========================================================
    # Save model inputs
    # ========================================================

    save_input_pair(
        output_dir
        / "00_input_pair.jpg",

        rgb_input=(
            rgb_input
        ),

        tir_input=(
            tir_input
        ),

        rgb_original_shape=(
            rgb_original_hw
        ),

        tir_original_shape=(
            tir_original_hw
        ),
    )

    # ========================================================
    # Full debug forward
    # ========================================================

    results = forward_hra_debug(
        model,
        rgb_tensor,
        tir_tensor,
    )

    # ========================================================
    # Final predictions
    # ========================================================

    pred_list = postprocess_predictions(
        preds=(
            results[
                "preds"
            ]
        ),

        model=model,

        conf_thres=(
            args.conf
        ),

        iou_thres=(
            args.iou
        ),

        max_det=(
            args.max_det
        ),
    )

    pred = pred_list[
        0
    ]

    names = model.names

    if not isinstance(
        names,
        dict,
    ):

        names = {
            i:
                str(name)
            for i, name
            in enumerate(
                names
            )
        }

    names = {
        int(k):
            str(v)
        for k, v
        in names.items()
    }

    # --------------------------------------------------------
    # Network-space detection
    # --------------------------------------------------------

    detection_network = draw_detections(
        rgb_input,
        pred,
        conf_thres=(
            args.conf
        ),
        names=names,
    )

    detection_network = put_title(
        detection_network,
        "Final detection - network RGB coordinate space",
        (
            f"conf={args.conf} | "
            f"detections={int(pred['bboxes'].shape[0])}"
        ),
    )

    save_rgb_image(
        output_dir
        / "01_final_detection_network.jpg",

        detection_network,
    )

    # --------------------------------------------------------
    # Original RGB-space detection
    # --------------------------------------------------------

    pred_original = network_boxes_to_original(
        pred=pred,

        ratio_pad=(
            rgb_ratio_pad
        ),

        original_hw=(
            rgb_original_hw
        ),
    )

    detection_original = draw_detections(
        rgb_original,
        pred_original,
        conf_thres=(
            args.conf
        ),
        names=names,
    )

    detection_original = put_title(
        detection_original,
        "Final detection - original RGB image",
        (
            f"conf={args.conf} | "
            f"detections={int(pred['bboxes'].shape[0])}"
        ),
    )

    save_rgb_image(
        output_dir
        / "02_final_detection_original.jpg",

        detection_original,
    )

    # ========================================================
    # HRA scales
    # ========================================================

    requested_scales = (
        {
            x.strip().upper()
            for x in args.scales.split(
                ","
            )
            if x.strip()
        }
        if args.scales.lower()
        != "all"
        else None
    )

    all_report = {}

    for scale_name, item in sorted(
        results[
            "hra_debug"
        ].items(),
        key=lambda kv:
            int(
                kv[0][1:]
            ),
    ):

        if (
            requested_scales
            is not None
            and scale_name.upper()
            not in requested_scales
        ):
            continue

        print(
            "\n"
            "============================================================"
        )

        print(
            f"Visualizing {scale_name}"
        )

        print(
            "============================================================"
        )

        debug = item[
            "debug"
        ]

        scale_dir = (
            output_dir
            / scale_name
        )

        report = visualize_scale(
            scale_dir=(
                scale_dir
            ),

            scale_name=(
                scale_name
            ),

            debug=(
                debug
            ),

            rgb_input=(
                rgb_input
            ),

            tir_input=(
                tir_input
            ),

            top_channels=(
                args.top_channels
            ),

            save_tensors=(
                args.save_tensors
            ),
        )

        report[
            "layer_index"
        ] = item[
            "layer_index"
        ]

        all_report[
            scale_name
        ] = report

        # Console shape summary
        for key in [
            "rgb",
            "tir",
            "tir_projected",
            "base_grid",
            "coarse_tir",
            "offset_input",
            "raw_offset",
            "bounded_offset",
            "refined_grid",
            "aligned_tir",
            "fusion_input",
            "fused",
        ]:

            if key in debug:

                print(
                    f"{key:<24}: "
                    f"{list(debug[key].shape)}"
                )

    # ========================================================
    # Save reports
    # ========================================================

    model_info = {
        "weights":
            str(
                weights
            ),

        "checkpoint_epoch":
            ckpt.get(
                "epoch"
            ),

        "checkpoint_stage":
            ckpt.get(
                "stage"
            ),

        "model_name":
            ckpt.get(
                "model_name"
            ),

        "fusion":
            fusion_type,

        "fusion_indices":
            list(
                model.fusion_indices
            ),

        "rgb_image":
            str(
                rgb_path
            ),

        "tir_image":
            str(
                tir_path
            ),

        "rgb_original_hw":
            list(
                rgb_original_hw
            ),

        "tir_original_hw":
            list(
                tir_original_hw
            ),

        "rgb_input_hw":
            list(
                rgb_input.shape[
                    :2
                ]
            ),

        "tir_input_hw":
            list(
                tir_input.shape[
                    :2
                ]
            ),

        "rgb_ratio_pad":
            rgb_ratio_pad,

        "tir_ratio_pad":
            tir_ratio_pad,

        "conf":
            args.conf,

        "iou":
            args.iou,

        "num_detections":
            int(
                pred[
                    "bboxes"
                ].shape[0]
            ),
    }

    save_json(
        output_dir
        / "model_info.json",

        model_info,
    )

    save_json(
        output_dir
        / "tensor_shapes.json",

        all_report,
    )

    # ========================================================
    # Final print
    # ========================================================

    print(
        "\n"
        "============================================================"
    )

    print(
        "HRA visualization completed."
    )

    print(
        "============================================================"
    )

    print(
        f"Output directory : "
        f"{output_dir}"
    )

    print(
        f"Input pair       : "
        f"{output_dir / '00_input_pair.jpg'}"
    )

    print(
        f"Detection        : "
        f"{output_dir / '02_final_detection_original.jpg'}"
    )

    print(
        f"Tensor report    : "
        f"{output_dir / 'tensor_shapes.json'}"
    )

    print(
        "============================================================\n"
    )


# ============================================================
# 15. CLI
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Visualize every HRA-v1 computation step "
            "for one RGB-T image pair."
        )
    )

    parser.add_argument(
        "--weights",

        type=str,

        required=True,

        help=(
            "Custom RGB-T HRA best.pt / last.pt"
        ),
    )

    parser.add_argument(
        "--rgb-image",

        type=str,

        required=True,

        help="RGB image path",
    )

    parser.add_argument(
        "--tir-image",

        type=str,

        required=True,

        help="TIR image path",
    )

    parser.add_argument(
        "--device",

        type=str,

        default="0",

        help="CUDA device, e.g. 0 / 5 / cpu",
    )

    parser.add_argument(
        "--rgb-imgsz",

        type=int,

        default=None,

        help=(
            "Override RGB input size. "
            "Default: recover from checkpoint config."
        ),
    )

    parser.add_argument(
        "--tir-imgsz",

        type=int,

        default=None,

        help=(
            "Override TIR input size. "
            "Default: recover from checkpoint config."
        ),
    )

    parser.add_argument(
        "--conf",

        type=float,

        default=0.25,

        help=(
            "Detection confidence threshold"
        ),
    )

    parser.add_argument(
        "--iou",

        type=float,

        default=0.7,

        help="NMS IoU threshold",
    )

    parser.add_argument(
        "--max-det",

        type=int,

        default=300,
    )

    parser.add_argument(
        "--output",

        type=str,

        default=None,

        help=(
            "Output directory. "
            "Default: runs/visualize_hra/..."
        ),
    )

    parser.add_argument(
        "--scales",

        type=str,

        default="all",

        help=(
            "all or comma-separated, e.g. P3 or P3,P4"
        ),
    )

    parser.add_argument(
        "--top-channels",

        type=int,

        default=6,

        help=(
            "Number of channels shown in channel montage. "
            "Set 0 to disable."
        ),
    )

    parser.add_argument(
        "--save-tensors",

        action="store_true",

        help=(
            "Also save every intermediate tensor as .pt"
        ),
    )

    args = parser.parse_args()

    main(
        args
    )
