"""
RGB-T Dual-Backbone YOLO Validation
===================================

Project:
    /mnt/sda/taochangyong/Projects/Model/YOLO-tcy

Main paper metrics:
    Params (M)
    FLOPs (G)
    FPS
    P (%)
    R (%)
    AP50 (%)
    AP75 (%)
    AP50:95 (%)

Architecture:
    RGB -> Backbone-R --\
                         -> Feature Fusion -> YOLO Neck -> Detect
    TIR -> Backbone-T --/

Evaluation coordinate system:
    RGB reference coordinate system

Current main GT:
    rgb_cls
    rgb_bboxes
    rgb_batch_idx

Current task:
    UAV-only detection

    cd /mnt/sda/taochangyong/Projects/Model/YOLO-tcy
    python val_rgbt.py \
    --weights runs/rgbt/yolo26n_LRDDv3_rgbt_concat_640_seed0/weights/best.pt \
    --rgb /mnt/sda/taochangyong/Projects/Model/YOLO-tcy/configs/datasets/NewDataset-RGB.yaml \
    --tir /mnt/sda/taochangyong/Projects/Model/YOLO-tcy/configs/datasets/NewDataset-TIR.yaml \
    --split val \
    --rgb-imgsz 640 \
    --tir-imgsz 640 \
    --batch 4 \
    --device 3
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from tqdm import tqdm

from ultralytics.utils import nms, ops
from ultralytics.utils.metrics import (
    DetMetrics,
    box_iou,
)


# ============================================================
# Project root
# ============================================================

ROOT = Path(__file__).resolve().parent

if str(ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(ROOT),
    )


from datasets.rgbt_dataset import (
    build_rgbt_dataset,
    build_rgbt_dataloader,
)

from models.rgbt_model import (
    RGBTDetectionModel,
)


# ============================================================
# 1. Device
# ============================================================

def select_device(
    device: str,
) -> torch.device:
    """
    Select CPU / CUDA.

    Examples
    --------
    "0"
    "1"
    "cpu"
    """

    device = str(
        device
    ).strip()

    if device.lower() == "cpu":

        return torch.device(
            "cpu"
        )

    if not torch.cuda.is_available():

        print(
            "WARNING: CUDA 不可用，自动切换 CPU。"
        )

        return torch.device(
            "cpu"
        )

    gpu_id = int(
        device
    )

    if gpu_id >= torch.cuda.device_count():

        raise ValueError(
            f"GPU {gpu_id} 不存在，"
            f"当前 GPU 数量="
            f"{torch.cuda.device_count()}"
        )

    torch.cuda.set_device(
        gpu_id
    )

    return torch.device(
        f"cuda:{gpu_id}"
    )


# ============================================================
# 2. AMP
# ============================================================

def autocast_context(
    device: torch.device,
    enabled: bool = True,
):
    """
    Automatic mixed precision context.
    """

    enabled = (
        bool(enabled)
        and device.type == "cuda"
    )

    return torch.autocast(
        device_type=device.type,

        dtype=(
            torch.float16
            if device.type == "cuda"
            else torch.bfloat16
        ),

        enabled=enabled,
    )


# ============================================================
# 3. CUDA synchronization
# ============================================================

def synchronize(
    device: torch.device,
):
    """
    Synchronize CUDA for accurate timing.
    """

    if device.type == "cuda":
        torch.cuda.synchronize(
            device
        )


# ============================================================
# 4. Batch preprocessing
# ============================================================

def preprocess_batch(
    batch: Dict,
    device: torch.device,
):
    """
    Move RGB-T batch to device.

    Images:
        uint8 [0,255]
        ->
        float32 [0,1]

    Labels:
        normalized xywh
    """

    batch["rgb_img"] = (
        batch[
            "rgb_img"
        ]
        .to(
            device,
            non_blocking=True,
        )
        .float()
        / 255.0
    )

    batch["tir_img"] = (
        batch[
            "tir_img"
        ]
        .to(
            device,
            non_blocking=True,
        )
        .float()
        / 255.0
    )

    tensor_keys = [
        "rgb_cls",
        "rgb_bboxes",
        "rgb_batch_idx",

        "tir_cls",
        "tir_bboxes",
        "tir_batch_idx",
    ]

    for key in tensor_keys:

        batch[key] = (
            batch[
                key
            ]
            .to(
                device,
                non_blocking=True,
            )
        )

    return batch


# ============================================================
# 5. Model parameters
# ============================================================

def get_params_m(
    model: torch.nn.Module,
) -> float:
    """
    Total model parameters in millions.
    """

    params = sum(
        p.numel()
        for p in model.parameters()
    )

    return (
        params / 1e6
    )


# ============================================================
# 6. FLOPs
# ============================================================

def get_flops_g(
    model: torch.nn.Module,

    rgb_imgsz: int,

    tir_imgsz: int,

    device: torch.device,
) -> float:
    """
    Calculate dual-input RGB-T FLOPs.

    Input:
        RGB:
            [1, 3, rgb_imgsz, rgb_imgsz]

        TIR:
            [1, 3, tir_imgsz, tir_imgsz]

    Output:
        GFLOPs

    Convention:
        FLOPs = MACs * 2
    """

    try:

        import thop

    except ImportError:

        print(
            "\nWARNING: 未安装 thop，"
            "FLOPs 将记为 0。"
        )

        print(
            "安装命令："
        )

        print(
            "python -m pip install ultralytics-thop "
            "-i https://mirrors.aliyun.com/pypi/simple/"
        )

        return 0.0

    was_training = (
        model.training
    )

    model.eval()

    rgb_dummy = torch.zeros(
        1,
        3,
        rgb_imgsz,
        rgb_imgsz,
        device=device,
        dtype=torch.float32,
    )

    tir_dummy = torch.zeros(
        1,
        3,
        tir_imgsz,
        tir_imgsz,
        device=device,
        dtype=torch.float32,
    )

    try:

        with torch.no_grad():

            macs, _ = thop.profile(
                model,

                inputs=(
                    rgb_dummy,
                    tir_dummy,
                ),

                verbose=False,
            )

        flops_g = (
            float(macs)
            * 2.0
            / 1e9
        )

    except Exception as e:

        print(
            "\nWARNING: FLOPs 计算失败："
        )

        print(
            e
        )

        flops_g = 0.0

    model.train(
        was_training
    )

    del rgb_dummy
    del tir_dummy

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return flops_g


# ============================================================
# 7. Standardized FPS benchmark
# ============================================================

@torch.inference_mode()
def benchmark_fps(
    model: torch.nn.Module,

    rgb_imgsz: int,

    tir_imgsz: int,

    device: torch.device,

    amp: bool = True,

    warmup: int = 20,

    iterations: int = 100,
):
    """
    Standard model-forward FPS benchmark.

    Important:
        batch = 1

    Included:
        RGB tensor
        +
        TIR tensor
        ->
        model forward

    Excluded:
        disk I/O
        dataset loading
        preprocessing
        NMS/postprocess

    Returns
    -------
    fps:
        frames/pairs per second

    latency_ms:
        model forward latency per RGB-T pair
    """

    model.eval()

    rgb = torch.randn(
        1,
        3,
        rgb_imgsz,
        rgb_imgsz,
        device=device,
    )

    tir = torch.randn(
        1,
        3,
        tir_imgsz,
        tir_imgsz,
        device=device,
    )

    # ========================================================
    # Warmup
    # ========================================================

    with autocast_context(
        device,
        amp,
    ):

        for _ in range(
            max(
                warmup,
                0,
            )
        ):

            _ = model(
                rgb,
                tir,
            )

    synchronize(
        device
    )

    # ========================================================
    # Benchmark
    # ========================================================

    start = time.perf_counter()

    with autocast_context(
        device,
        amp,
    ):

        for _ in range(
            max(
                iterations,
                1,
            )
        ):

            _ = model(
                rgb,
                tir,
            )

    synchronize(
        device
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    iterations = max(
        iterations,
        1,
    )

    latency_s = (
        elapsed
        / iterations
    )

    latency_ms = (
        latency_s
        * 1000.0
    )

    fps = (
        1.0 / latency_s
        if latency_s > 0
        else 0.0
    )

    del rgb
    del tir

    return (
        fps,
        latency_ms,
    )


# ============================================================
# 8. Model complexity
# ============================================================

def get_model_complexity(
    model: torch.nn.Module,

    rgb_imgsz: int,

    tir_imgsz: int,

    device: torch.device,
):
    """
    Return:
        Params(M)
        FLOPs(G)
    """

    params_m = get_params_m(
        model
    )

    flops_g = get_flops_g(
        model=model,

        rgb_imgsz=rgb_imgsz,

        tir_imgsz=tir_imgsz,

        device=device,
    )

    return (
        params_m,
        flops_g,
    )


# ============================================================
# 9. Prediction postprocessing
# ============================================================

def postprocess_predictions(
    preds,

    model: torch.nn.Module,

    conf_thres: float,

    iou_thres: float,

    max_det: int,
):
    """
    Ultralytics-style NMS.

    Returns
    -------
    list[dict]

    Each dict:
        bboxes: [N,4] xyxy
        conf:   [N]
        cls:    [N]
    """

    # --------------------------------------------------------
    # Custom wrapper safety
    # --------------------------------------------------------

    if isinstance(
        preds,
        dict,
    ):

        if "pred" in preds:
            preds = preds[
                "pred"
            ]

        elif "preds" in preds:
            preds = preds[
                "preds"
            ]

        elif "output" in preds:
            preds = preds[
                "output"
            ]

        else:

            raise TypeError(
                "\n无法解析 dict 类型模型输出。\n"
                f"keys={list(preds.keys())}"
            )

    outputs = (
        nms.non_max_suppression(
            preds,

            conf_thres,

            iou_thres,

            # Same detect-task behavior as
            # Ultralytics DetectionValidator
            nc=0,

            multi_label=True,

            agnostic=False,

            max_det=max_det,

            end2end=getattr(
                model,
                "end2end",
                False,
            ),

            rotated=False,
        )
    )

    results = []

    for output in outputs:

        results.append(
            {
                "bboxes":
                    output[
                        :,
                        :4
                    ],

                "conf":
                    output[
                        :,
                        4
                    ],

                "cls":
                    output[
                        :,
                        5
                    ],
            }
        )

    return results


# ============================================================
# 10. Match predictions
# ============================================================

def match_predictions(
    pred_classes: torch.Tensor,

    true_classes: torch.Tensor,

    iou: torch.Tensor,

    iouv: torch.Tensor,
):
    """
    Ultralytics-style greedy one-to-one matching.

    Output:
        [num_predictions, 10]

    IoU thresholds:
        0.50
        0.55
        0.60
        0.65
        0.70
        0.75
        0.80
        0.85
        0.90
        0.95
    """

    correct = np.zeros(
        (
            pred_classes.shape[0],
            iouv.shape[0],
        ),
        dtype=bool,
    )

    if (
        pred_classes.numel() == 0
        or true_classes.numel() == 0
    ):

        return torch.from_numpy(
            correct
        )

    # ========================================================
    # GT x predictions class match
    # ========================================================

    correct_class = (
        true_classes[:, None]
        ==
        pred_classes[None, :]
    )

    # Wrong class IoU -> 0
    iou = (
        iou
        * correct_class
    )

    iou_np = (
        iou
        .detach()
        .cpu()
        .numpy()
    )

    thresholds = (
        iouv
        .detach()
        .cpu()
        .tolist()
    )

    # ========================================================
    # Match independently at each IoU threshold
    # ========================================================

    for threshold_index, threshold in enumerate(
        thresholds
    ):

        matches = np.nonzero(
            iou_np
            >= threshold
        )

        matches = (
            np.array(
                matches
            )
            .T
        )

        if matches.shape[0] == 0:
            continue

        if matches.shape[0] > 1:

            # highest IoU first
            matches = matches[
                iou_np[
                    matches[:, 0],
                    matches[:, 1],
                ]
                .argsort()[::-1]
            ]

            # each detection -> one GT
            matches = matches[
                np.unique(
                    matches[:, 1],
                    return_index=True,
                )[1]
            ]

            # each GT -> one detection
            matches = matches[
                np.unique(
                    matches[:, 0],
                    return_index=True,
                )[1]
            ]

        pred_indices = (
            matches[:, 1]
            .astype(int)
        )

        correct[
            pred_indices,
            threshold_index,
        ] = True

    return torch.from_numpy(
        correct
    )


# ============================================================
# 11. Prepare RGB GT
# ============================================================

def prepare_rgb_ground_truth(
    batch: Dict,

    sample_index: int,
):
    """
    Convert RGB GT:

        normalized xywh
            ↓
        network-space xyxy

    Main fused detector currently uses RGB coordinate system.
    """

    mask = (
        batch[
            "rgb_batch_idx"
        ]
        == sample_index
    )

    cls = (
        batch[
            "rgb_cls"
        ][mask]
        .squeeze(-1)
    )

    boxes = (
        batch[
            "rgb_bboxes"
        ][mask]
    )

    rgb_h = int(
        batch[
            "rgb_img"
        ].shape[2]
    )

    rgb_w = int(
        batch[
            "rgb_img"
        ].shape[3]
    )

    if boxes.shape[0] > 0:

        boxes = ops.xywh2xyxy(
            boxes
        )

        scale = torch.tensor(
            [
                rgb_w,
                rgb_h,
                rgb_w,
                rgb_h,
            ],
            device=boxes.device,
            dtype=boxes.dtype,
        )

        boxes = (
            boxes
            * scale
        )

    return {
        "cls":
            cls,

        "bboxes":
            boxes,
    }


# ============================================================
# 12. One-image TP calculation
# ============================================================

def process_single_image(
    pred: Dict,

    gt: Dict,

    iouv: torch.Tensor,
):
    """
    Calculate TP matrix for one image.
    """

    pred_cls = pred[
        "cls"
    ]

    pred_boxes = pred[
        "bboxes"
    ]

    gt_cls = gt[
        "cls"
    ]

    gt_boxes = gt[
        "bboxes"
    ]

    # ========================================================
    # No predictions
    # ========================================================

    if pred_cls.shape[0] == 0:

        return np.zeros(
            (
                0,
                len(iouv),
            ),
            dtype=bool,
        )

    # ========================================================
    # Background image
    # ========================================================

    if gt_cls.shape[0] == 0:

        return np.zeros(
            (
                pred_cls.shape[0],
                len(iouv),
            ),
            dtype=bool,
        )

    # ========================================================
    # GT x Predictions IoU
    # ========================================================

    iou = box_iou(
        gt_boxes,
        pred_boxes,
    )

    correct = match_predictions(
        pred_classes=pred_cls,

        true_classes=gt_cls,

        iou=iou,

        iouv=iouv,
    )

    return (
        correct
        .cpu()
        .numpy()
    )


# ============================================================
# 13. Load custom checkpoint
# ============================================================

def load_rgbt_checkpoint(
    checkpoint_path: str,

    device: torch.device,
):
    """
    Load checkpoint generated by train_rgbt.py.

    This is NOT a standard YOLO("best.pt") checkpoint.

    Architecture is rebuilt first,
    then model_state_dict is restored.
    """

    checkpoint_path = (
        Path(
            checkpoint_path
        )
        .expanduser()
        .resolve()
    )

    if not checkpoint_path.exists():

        raise FileNotFoundError(
            "\n权重不存在:\n"
            f"{checkpoint_path}"
        )

    print(
        "\nLoading checkpoint:"
    )

    print(
        checkpoint_path
    )

    ckpt = torch.load(
        checkpoint_path,

        map_location="cpu",

        weights_only=False,
    )

    # ========================================================
    # Recover architecture
    # ========================================================

    model_name = ckpt.get(
        "model_name",
        "yolo26n",
    )

    fusion = ckpt.get(
        "fusion",
        "concat",
    )

    nc = int(
        ckpt.get(
            "nc",
            1,
        )
    )

    names = ckpt.get(
        "names",
        {
            0: "UAV",
        },
    )

    cfg = ckpt.get(
        "config",
        {},
    )

    align_mode = (
        cfg
        .get(
            "model",
            {},
        )
        .get(
            "align_mode",
            "bilinear",
        )
    )

    # ========================================================
    # Rebuild model
    # ========================================================

    model = RGBTDetectionModel(
        model_name=model_name,

        nc=nc,

        # No need to reload COCO weights.
        # Full trained state will be loaded below.
        pretrained=False,

        fusion=fusion,

        align_mode=align_mode,

        names=names,

        verbose=False,
    )

    if (
        "model_state_dict"
        not in ckpt
    ):

        raise KeyError(
            "\nCheckpoint 中没有 model_state_dict。\n"
            "请确认该权重由当前 train_rgbt.py 保存。"
        )

    model.load_state_dict(
        ckpt[
            "model_state_dict"
        ],

        strict=True,
    )

    model = model.to(
        device
    )

    model.eval()

    return (
        model,
        ckpt,
        cfg,
    )


# ============================================================
# 14. Save CSV
# ============================================================

def save_metrics_csv(
    path: Path,

    row: Dict,
):
    """
    Save one paper-style result row.
    """

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "Model",

        "Params(M)",

        "FLOPs(G)",

        "FPS",

        "P(%)",

        "R(%)",

        "AP50(%)",

        "AP75(%)",

        "AP50:95(%)",

        "RGB_imgsz",

        "TIR_imgsz",

        "Split",
    ]

    with path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerow(
            {
                key: row.get(
                    key,
                    ""
                )
                for key in fieldnames
            }
        )


# ============================================================
# 15. Main validation
# ============================================================

@torch.inference_mode()
def validate(
    checkpoint: str,

    split: str = "val",

    device_str: str = "0",

    batch_size: int = 8,

    workers: int = 4,

    conf_thres: float = 0.001,

    iou_thres: float = 0.7,

    max_det: int = 300,

    amp: bool = True,

    rgb_yaml: str | None = None,

    tir_yaml: str | None = None,

    rgb_imgsz: int | None = None,

    tir_imgsz: int | None = None,

    pair_mode: str | None = None,

    save_dir: str | None = None,

    plots: bool = False,

    speed_warmup: int = 20,

    speed_iters: int = 100,
):

    # ========================================================
    # Device
    # ========================================================

    device = select_device(
        device_str
    )

    # ========================================================
    # Load model
    # ========================================================

    (
        model,
        ckpt,
        cfg,
    ) = load_rgbt_checkpoint(
        checkpoint,
        device,
    )

    # ========================================================
    # Recover dataset settings from training config
    # ========================================================

    data_cfg = cfg.get(
        "data",
        {},
    )

    if rgb_yaml is None:

        rgb_yaml = data_cfg.get(
            "rgb"
        )

    if tir_yaml is None:

        tir_yaml = data_cfg.get(
            "tir"
        )

    if rgb_imgsz is None:

        rgb_imgsz = int(
            data_cfg.get(
                "rgb_imgsz",
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

    if rgb_yaml is None:

        raise ValueError(
            "无法确定 RGB dataset YAML。"
        )

    if tir_yaml is None:

        raise ValueError(
            "无法确定 TIR dataset YAML。"
        )

    # ========================================================
    # Save directory
    # ========================================================

    checkpoint_path = (
        Path(
            checkpoint
        )
        .expanduser()
        .resolve()
    )

    if save_dir is None:

        save_dir = (
            checkpoint_path
            .parent
            .parent
            / f"val_{split}"
        )

    else:

        save_dir = (
            Path(
                save_dir
            )
            .expanduser()
            .resolve()
        )

    save_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # Complexity
    # ========================================================

    print(
        "\nCalculating model complexity..."
    )

    (
        params_m,
        flops_g,
    ) = get_model_complexity(
        model=model,

        rgb_imgsz=rgb_imgsz,

        tir_imgsz=tir_imgsz,

        device=device,
    )

    # ========================================================
    # Dataset
    # ========================================================

    dataset = build_rgbt_dataset(
        rgb_yaml=rgb_yaml,

        tir_yaml=tir_yaml,

        split=split,

        rgb_imgsz=rgb_imgsz,

        tir_imgsz=tir_imgsz,

        pair_mode=pair_mode,

        augment=False,

        tir_channels=3,

        strict_pair=True,
    )

    loader = build_rgbt_dataloader(
        dataset,

        batch_size=batch_size,

        workers=workers,

        shuffle=False,

        pin_memory=(
            device.type
            == "cuda"
        ),
    )

    # ========================================================
    # Validation info
    # ========================================================

    print(
        "\n"
        "============================================================"
    )

    print(
        "RGB-T YOLO Validation"
    )

    print(
        "============================================================"
    )

    print(
        f"Model          : {model.model_name}"
    )

    print(
        f"Fusion         : {model.fusion_type}"
    )

    print(
        f"Checkpoint     : {checkpoint_path}"
    )

    print(
        f"Split          : {split}"
    )

    print(
        f"RGB imgsz      : {rgb_imgsz}"
    )

    print(
        f"TIR imgsz      : {tir_imgsz}"
    )

    print(
        f"Batch          : {batch_size}"
    )

    print(
        f"Workers        : {workers}"
    )

    print(
        f"Device         : {device}"
    )

    if device.type == "cuda":

        print(
            f"GPU            : "
            f"{torch.cuda.get_device_name(device)}"
        )

    print(
        f"AMP            : {amp}"
    )

    print(
        f"Confidence     : {conf_thres}"
    )

    print(
        f"NMS IoU        : {iou_thres}"
    )

    print(
        f"Max det        : {max_det}"
    )

    print(
        f"Params         : {params_m:.3f} M"
    )

    print(
        f"FLOPs          : {flops_g:.3f} G"
    )

    print(
        "============================================================\n"
    )

    # ========================================================
    # DetMetrics
    # ========================================================

    names = model.names

    if not isinstance(
        names,
        dict,
    ):

        names = {
            i: name
            for i, name
            in enumerate(
                names
            )
        }

    # normalize keys
    names = {
        int(k): str(v)
        for k, v in names.items()
    }

    metrics = DetMetrics(
        names=names
    )

    # ========================================================
    # IoU thresholds: 0.50 -> 0.95
    # ========================================================

    iouv = torch.linspace(
        0.50,
        0.95,
        10,
        device=device,
    )

    # ========================================================
    # Statistics
    # ========================================================

    seen = 0

    total_instances = 0

    preprocess_time = 0.0

    inference_time = 0.0

    postprocess_time = 0.0

    # ========================================================
    # Warmup validation model
    # ========================================================

    warmup_rgb = torch.zeros(
        1,
        3,
        rgb_imgsz,
        rgb_imgsz,
        device=device,
    )

    warmup_tir = torch.zeros(
        1,
        3,
        tir_imgsz,
        tir_imgsz,
        device=device,
    )

    with autocast_context(
        device,
        amp,
    ):

        for _ in range(
            3
        ):

            _ = model(
                warmup_rgb,
                warmup_tir,
            )

    synchronize(
        device
    )

    del warmup_rgb
    del warmup_tir

    # ========================================================
    # Progress bar
    # ========================================================

    pbar = tqdm(
        loader,

        total=len(
            loader
        ),

        desc=(
            f"{'Class':>12} "
            f"{'Images':>8} "
            f"{'Instances':>10}"
        ),

        dynamic_ncols=True,

        leave=True,
    )

    # ========================================================
    # Validation loop
    # ========================================================

    for batch in pbar:

        batch_size_current = int(
            batch[
                "rgb_img"
            ].shape[0]
        )

        seen += (
            batch_size_current
        )

        total_instances += int(
            batch[
                "rgb_cls"
            ].shape[0]
        )

        # ====================================================
        # Preprocess
        # ====================================================

        synchronize(
            device
        )

        t0 = time.perf_counter()

        batch = preprocess_batch(
            batch,
            device,
        )

        synchronize(
            device
        )

        t1 = time.perf_counter()

        preprocess_time += (
            t1 - t0
        )

        # ====================================================
        # Model inference
        # ====================================================

        synchronize(
            device
        )

        t2 = time.perf_counter()

        with autocast_context(
            device,
            amp,
        ):

            preds = model(
                batch[
                    "rgb_img"
                ],

                batch[
                    "tir_img"
                ],
            )

        synchronize(
            device
        )

        t3 = time.perf_counter()

        inference_time += (
            t3 - t2
        )

        # ====================================================
        # Postprocess / NMS
        # ====================================================

        synchronize(
            device
        )

        t4 = time.perf_counter()

        preds = postprocess_predictions(
            preds=preds,

            model=model,

            conf_thres=conf_thres,

            iou_thres=iou_thres,

            max_det=max_det,
        )

        synchronize(
            device
        )

        t5 = time.perf_counter()

        postprocess_time += (
            t5 - t4
        )

        # ====================================================
        # Per-image metrics
        # ====================================================

        for sample_index, pred in enumerate(
            preds
        ):

            gt = prepare_rgb_ground_truth(
                batch,
                sample_index,
            )

            tp = process_single_image(
                pred=pred,

                gt=gt,

                iouv=iouv,
            )

            target_cls = (
                gt[
                    "cls"
                ]
                .detach()
                .float()
                .cpu()
                .numpy()
            )

            no_pred = (
                pred[
                    "cls"
                ].shape[0]
                == 0
            )

            if no_pred:

                pred_conf = np.zeros(
                    0,
                    dtype=np.float32,
                )

                pred_cls = np.zeros(
                    0,
                    dtype=np.float32,
                )

            else:

                pred_conf = (
                    pred[
                        "conf"
                    ]
                    .detach()
                    .float()
                    .cpu()
                    .numpy()
                )

                pred_cls = (
                    pred[
                        "cls"
                    ]
                    .detach()
                    .float()
                    .cpu()
                    .numpy()
                )

            metrics.update_stats(
                {
                    "tp":
                        tp,

                    "conf":
                        pred_conf,

                    "pred_cls":
                        pred_cls,

                    "target_cls":
                        target_cls,

                    "target_img":
                        np.unique(
                            target_cls
                        ),

                    "im_name":
                        Path(
                            batch[
                                "rgb_path"
                            ][sample_index]
                        ).name,
                }
            )

        # ====================================================
        # Progress display
        # ====================================================

        gpu_mem = 0.0

        if device.type == "cuda":

            gpu_mem = (
                torch.cuda.memory_reserved(
                    device
                )
                / (1024 ** 3)
            )

        pbar.set_postfix(
            {
                "Images":
                    seen,

                "Instances":
                    total_instances,

                "GPU":
                    f"{gpu_mem:.2f}G",
            }
        )

    pbar.close()

    # ========================================================
    # Process metrics
    # ========================================================

    metrics.process(
        save_dir=save_dir,

        plot=plots,
    )

    # ========================================================
    # Main metrics
    # ========================================================

    (
        precision,
        recall,
        map50,
        map5095,
    ) = metrics.mean_results()

    # AP75
    map75 = float(
        metrics.box.map75
    )

    precision = float(
        precision
    )

    recall = float(
        recall
    )

    map50 = float(
        map50
    )

    map5095 = float(
        map5095
    )

    # ========================================================
    # Convert to %
    # ========================================================

    precision_pct = (
        precision * 100.0
    )

    recall_pct = (
        recall * 100.0
    )

    ap50_pct = (
        map50 * 100.0
    )

    ap75_pct = (
        map75 * 100.0
    )

    ap5095_pct = (
        map5095 * 100.0
    )

    # ========================================================
    # Validation pipeline timing
    # ========================================================

    preprocess_ms = (
        preprocess_time
        / max(
            seen,
            1,
        )
        * 1000.0
    )

    inference_ms_val = (
        inference_time
        / max(
            seen,
            1,
        )
        * 1000.0
    )

    postprocess_ms = (
        postprocess_time
        / max(
            seen,
            1,
        )
        * 1000.0
    )

    e2e_ms = (
        preprocess_ms
        + inference_ms_val
        + postprocess_ms
    )

    e2e_fps = (
        1000.0
        / e2e_ms
        if e2e_ms > 0
        else 0.0
    )

    # ========================================================
    # Standardized batch=1 model FPS
    # ========================================================

    print(
        "\nBenchmarking batch=1 model FPS..."
    )

    (
        fps,
        latency_ms,
    ) = benchmark_fps(
        model=model,

        rgb_imgsz=rgb_imgsz,

        tir_imgsz=tir_imgsz,

        device=device,

        amp=amp,

        warmup=speed_warmup,

        iterations=speed_iters,
    )

    # ========================================================
    # Main paper-style table
    # ========================================================

    model_display_name = (
        f"{model.model_name}-"
        f"{model.fusion_type}"
    )

    print(
        "\n"
        "=============================================================================================================="
    )

    print(
        f"{'Model':<20}"
        f"{'Params(M)':>12}"
        f"{'FLOPs(G)':>12}"
        f"{'FPS':>11}"
        f"{'P(%)':>10}"
        f"{'R(%)':>10}"
        f"{'AP50(%)':>12}"
        f"{'AP75(%)':>12}"
        f"{'AP50:95(%)':>15}"
    )

    print(
        "--------------------------------------------------------------------------------------------------------------"
    )

    print(
        f"{model_display_name:<20}"
        f"{params_m:>12.3f}"
        f"{flops_g:>12.3f}"
        f"{fps:>11.2f}"
        f"{precision_pct:>10.2f}"
        f"{recall_pct:>10.2f}"
        f"{ap50_pct:>12.2f}"
        f"{ap75_pct:>12.2f}"
        f"{ap5095_pct:>15.2f}"
    )

    print(
        "=============================================================================================================="
    )

    # ========================================================
    # Dataset information
    # ========================================================

    print(
        "\nDataset:"
    )

    print(
        f"  Images            : {seen}"
    )

    print(
        f"  UAV Instances     : {total_instances}"
    )

    print(
        f"  RGB Resolution    : {rgb_imgsz}"
    )

    print(
        f"  TIR Resolution    : {tir_imgsz}"
    )

    # ========================================================
    # Speed
    # ========================================================

    print(
        "\nSpeed:"
    )

    print(
        f"  Preprocess        : "
        f"{preprocess_ms:.3f} ms/image"
    )

    print(
        f"  Val inference     : "
        f"{inference_ms_val:.3f} ms/image "
        f"(batch={batch_size})"
    )

    print(
        f"  Postprocess       : "
        f"{postprocess_ms:.3f} ms/image"
    )

    print(
        f"  End-to-end        : "
        f"{e2e_ms:.3f} ms/image"
    )

    print(
        f"  End-to-end FPS    : "
        f"{e2e_fps:.2f}"
    )

    print(
        f"  Model latency     : "
        f"{latency_ms:.3f} ms/pair "
        f"(batch=1)"
    )

    print(
        f"  Paper FPS         : "
        f"{fps:.2f} "
        f"(batch=1, model forward only)"
    )

    # ========================================================
    # Per-class output
    # ========================================================

    if len(
        metrics.ap_class_index
    ):

        print(
            "\nPer-class:"
        )

        print(
            f"{'Class':<15}"
            f"{'Images':>10}"
            f"{'Instances':>12}"
            f"{'P(%)':>10}"
            f"{'R(%)':>10}"
            f"{'AP50(%)':>12}"
            f"{'AP75(%)':>12}"
            f"{'AP50:95(%)':>15}"
        )

        print(
            "-" * 96
        )

        for metric_i, class_id in enumerate(
            metrics.ap_class_index
        ):

            class_id = int(
                class_id
            )

            (
                class_p,
                class_r,
                class_ap50,
                class_ap5095,
            ) = metrics.class_result(
                metric_i
            )

            # all_ap:
            # index 5 = IoU 0.75
            class_ap75 = float(
                metrics.box.all_ap[
                    metric_i,
                    5,
                ]
            )

            class_name = names.get(
                class_id,
                str(class_id),
            )

            class_instances = 0

            class_images = 0

            if (
                metrics.nt_per_class
                is not None
                and class_id
                < len(
                    metrics.nt_per_class
                )
            ):

                class_instances = int(
                    metrics.nt_per_class[
                        class_id
                    ]
                )

            if (
                metrics.nt_per_image
                is not None
                and class_id
                < len(
                    metrics.nt_per_image
                )
            ):

                class_images = int(
                    metrics.nt_per_image[
                        class_id
                    ]
                )

            print(
                f"{class_name:<15}"
                f"{class_images:>10}"
                f"{class_instances:>12}"
                f"{float(class_p) * 100:>10.2f}"
                f"{float(class_r) * 100:>10.2f}"
                f"{float(class_ap50) * 100:>12.2f}"
                f"{class_ap75 * 100:>12.2f}"
                f"{float(class_ap5095) * 100:>15.2f}"
            )

    # ========================================================
    # Results dictionary
    # ========================================================

    results = {
        # ----------------------------------------------------
        # Experiment
        # ----------------------------------------------------

        "checkpoint":
            str(
                checkpoint_path
            ),

        "model":
            model.model_name,

        "fusion":
            model.fusion_type,

        "model_display_name":
            model_display_name,

        "split":
            split,

        # ----------------------------------------------------
        # Dataset
        # ----------------------------------------------------

        "images":
            int(
                seen
            ),

        "instances":
            int(
                total_instances
            ),

        "rgb_imgsz":
            int(
                rgb_imgsz
            ),

        "tir_imgsz":
            int(
                tir_imgsz
            ),

        # ----------------------------------------------------
        # Complexity
        # ----------------------------------------------------

        "complexity": {
            "params_m":
                float(
                    params_m
                ),

            "flops_g":
                float(
                    flops_g
                ),
        },

        # ----------------------------------------------------
        # Validation settings
        # ----------------------------------------------------

        "validation": {
            "conf":
                float(
                    conf_thres
                ),

            "nms_iou":
                float(
                    iou_thres
                ),

            "max_det":
                int(
                    max_det
                ),

            "batch":
                int(
                    batch_size
                ),

            "amp":
                bool(
                    amp
                ),
        },

        # ----------------------------------------------------
        # Metrics [0,1]
        # ----------------------------------------------------

        "metrics": {
            "precision":
                precision,

            "recall":
                recall,

            "AP50":
                map50,

            "AP75":
                map75,

            "AP50-95":
                map5095,
        },

        # ----------------------------------------------------
        # Paper metrics [%]
        # ----------------------------------------------------

        "paper_metrics": {
            "Params(M)":
                float(
                    params_m
                ),

            "FLOPs(G)":
                float(
                    flops_g
                ),

            "FPS":
                float(
                    fps
                ),

            "P(%)":
                float(
                    precision_pct
                ),

            "R(%)":
                float(
                    recall_pct
                ),

            "AP50(%)":
                float(
                    ap50_pct
                ),

            "AP75(%)":
                float(
                    ap75_pct
                ),

            "AP50:95(%)":
                float(
                    ap5095_pct
                ),
        },

        # ----------------------------------------------------
        # Speed
        # ----------------------------------------------------

        "speed": {
            "preprocess_ms":
                float(
                    preprocess_ms
                ),

            "validation_inference_ms":
                float(
                    inference_ms_val
                ),

            "postprocess_ms":
                float(
                    postprocess_ms
                ),

            "end_to_end_ms":
                float(
                    e2e_ms
                ),

            "end_to_end_fps":
                float(
                    e2e_fps
                ),

            "batch1_model_latency_ms":
                float(
                    latency_ms
                ),

            "batch1_model_fps":
                float(
                    fps
                ),

            "speed_warmup":
                int(
                    speed_warmup
                ),

            "speed_iterations":
                int(
                    speed_iters
                ),
        },
    }

    # ========================================================
    # Save JSON
    # ========================================================

    metrics_json = (
        save_dir
        / "metrics.json"
    )

    with metrics_json.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            results,

            f,

            ensure_ascii=False,

            indent=2,
        )

    # ========================================================
    # Save paper table CSV
    # ========================================================

    paper_row = {
        "Model":
            model_display_name,

        "Params(M)":
            round(
                params_m,
                3,
            ),

        "FLOPs(G)":
            round(
                flops_g,
                3,
            ),

        "FPS":
            round(
                fps,
                2,
            ),

        "P(%)":
            round(
                precision_pct,
                2,
            ),

        "R(%)":
            round(
                recall_pct,
                2,
            ),

        "AP50(%)":
            round(
                ap50_pct,
                2,
            ),

        "AP75(%)":
            round(
                ap75_pct,
                2,
            ),

        "AP50:95(%)":
            round(
                ap5095_pct,
                2,
            ),

        "RGB_imgsz":
            rgb_imgsz,

        "TIR_imgsz":
            tir_imgsz,

        "Split":
            split,
    }

    save_metrics_csv(
        save_dir
        / "paper_metrics.csv",

        paper_row,
    )

    # ========================================================
    # Final
    # ========================================================

    print(
        "\n"
        "============================================================"
    )

    print(
        "Validation completed."
    )

    print(
        f"Results directory : {save_dir}"
    )

    print(
        f"JSON              : {metrics_json}"
    )

    print(
        f"Paper CSV         : "
        f"{save_dir / 'paper_metrics.csv'}"
    )

    print(
        "============================================================\n"
    )

    return results


# ============================================================
# 16. CLI
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Validate RGB-T dual-backbone YOLO"
        )
    )

    # ========================================================
    # Weight
    # ========================================================

    parser.add_argument(
        "--weights",

        type=str,

        required=True,

        help=(
            "YOLO-tcy custom best.pt / last.pt"
        ),
    )

    # ========================================================
    # Dataset
    # ========================================================

    parser.add_argument(
        "--split",

        type=str,

        default="val",

        choices=[
            "train",
            "val",
            "test",
        ],
    )

    parser.add_argument(
        "--rgb",

        type=str,

        default=None,

        help=(
            "Optional RGB dataset YAML override"
        ),
    )

    parser.add_argument(
        "--tir",

        type=str,

        default=None,

        help=(
            "Optional TIR dataset YAML override"
        ),
    )

    # ========================================================
    # Resolution
    # ========================================================

    parser.add_argument(
        "--rgb-imgsz",

        type=int,

        default=None,
    )

    parser.add_argument(
        "--tir-imgsz",

        type=int,

        default=None,
    )

    # ========================================================
    # DataLoader
    # ========================================================

    parser.add_argument(
        "--batch",

        type=int,

        default=8,
    )

    parser.add_argument(
        "--workers",

        type=int,

        default=4,
    )

    # ========================================================
    # Validation thresholds
    # ========================================================

    parser.add_argument(
        "--conf",

        type=float,

        default=0.001,

        help=(
            "Validation confidence threshold"
        ),
    )

    parser.add_argument(
        "--iou",

        type=float,

        default=0.7,

        help=(
            "NMS IoU threshold"
        ),
    )

    parser.add_argument(
        "--max-det",

        type=int,

        default=300,
    )

    # ========================================================
    # Device
    # ========================================================

    parser.add_argument(
        "--device",

        type=str,

        default="0",
    )

    parser.add_argument(
        "--no-amp",

        action="store_true",

        help=(
            "Disable AMP"
        ),
    )

    # ========================================================
    # Pairing
    # ========================================================

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

    # ========================================================
    # FPS benchmark
    # ========================================================

    parser.add_argument(
        "--speed-warmup",

        type=int,

        default=20,

        help=(
            "Number of batch=1 warmup iterations"
        ),
    )

    parser.add_argument(
        "--speed-iters",

        type=int,

        default=100,

        help=(
            "Number of batch=1 timed iterations"
        ),
    )

    # ========================================================
    # Output
    # ========================================================

    parser.add_argument(
        "--save-dir",

        type=str,

        default=None,
    )

    parser.add_argument(
        "--plots",

        action="store_true",

        help=(
            "Save PR/F1/P/R curves"
        ),
    )

    # ========================================================
    # Parse
    # ========================================================

    args = parser.parse_args()

    validate(
        checkpoint=args.weights,

        split=args.split,

        device_str=args.device,

        batch_size=args.batch,

        workers=args.workers,

        conf_thres=args.conf,

        iou_thres=args.iou,

        max_det=args.max_det,

        amp=not args.no_amp,

        rgb_yaml=args.rgb,

        tir_yaml=args.tir,

        rgb_imgsz=args.rgb_imgsz,

        tir_imgsz=args.tir_imgsz,

        pair_mode=args.pair_mode,

        save_dir=args.save_dir,

        plots=args.plots,

        speed_warmup=args.speed_warmup,

        speed_iters=args.speed_iters,
    )