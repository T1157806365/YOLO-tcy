"""
Train RGB-T Dual-Backbone YOLO
==============================

Architecture:

RGB -> Backbone-R ---\
                      -> Feature Fusion -> YOLO Neck -> Detect -> RGB GT
TIR -> Backbone-T ---/

Project:

    训练
    conda activate tcy
    cd /mnt/sda/taochangyong/Projects/Model/YOLO-tcy
    python train_rgbt.py \
    --cfg configs/experiments/lrdd_rgbt_yolo26n_1280.yaml

    conda activate tcy
    cd /mnt/sda/taochangyong/Projects/Model/YOLO-tcy
    python train_rgbt.py \
    --cfg configs/experiments/uavcb_har_yolo26n_640.yaml
"""
from __future__ import annotations
import argparse
import copy
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict
import numpy as np
import torch
import yaml
from tqdm import tqdm

# ============================================================
# Project path
# ============================================================

ROOT = Path(__file__).resolve().parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from datasets.rgbt_dataset import (
    build_rgbt_dataloader,
    build_rgbt_dataset,
)

from models.rgbt_model import (
    RGBTDetectionModel,
)
from val_rgbt import (
    postprocess_predictions,
    prepare_rgb_ground_truth,
    process_single_image,
)
from ultralytics.utils.metrics import DetMetrics

# ============================================================
# 1. Reproducibility
# ============================================================

def set_seed(seed: int = 0, deterministic: bool = True):
    """
    Set random seeds.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


# ============================================================
# 2. YAML
# ============================================================

def load_config(path: str) -> Dict:
    """
    Load experiment YAML.
    """

    path = Path(path).resolve()

    if not path.exists():
        raise FileNotFoundError(
            f"配置文件不存在:\n{path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        cfg = yaml.safe_load(f)

    if cfg is None:
        cfg = {}

    return cfg


def save_config(
    cfg: Dict,
    path: Path,
):
    """
    Save final experiment configuration.
    """

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:

        yaml.safe_dump(
            cfg,
            f,
            allow_unicode=True,
            sort_keys=False,
        )


# ============================================================
# 3. Device
# ============================================================

def select_device(device: str):
    """
    Examples:
        "0"
        "1"
        "cpu"
    """

    if device.lower() == "cpu":
        return torch.device("cpu")

    if not torch.cuda.is_available():
        print(
            "WARNING: CUDA 不可用，自动切换到 CPU。"
        )
        return torch.device("cpu")

    gpu_id = int(device)

    if gpu_id >= torch.cuda.device_count():
        raise ValueError(
            f"GPU {gpu_id} 不存在，"
            f"当前 GPU 数量={torch.cuda.device_count()}"
        )

    torch.cuda.set_device(gpu_id)

    return torch.device(
        f"cuda:{gpu_id}"
    )


# ============================================================
# 4. Move batch to GPU
# ============================================================

def preprocess_batch(
    batch: Dict,
    device: torch.device,
):
    """
    RGB/TIR uint8:
        [0,255]

    -> float32:
        [0,1]

    Labels remain normalized xywh.
    """

    batch["rgb_img"] = (
        batch["rgb_img"]
        .to(
            device,
            non_blocking=True,
        )
        .float()
        / 255.0
    )

    batch["tir_img"] = (
        batch["tir_img"]
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

        batch[key] = batch[key].to(
            device,
            non_blocking=True,
        )

    return batch


# ============================================================
# 5. Optimizer
# ============================================================

def build_optimizer(
    model,
    cfg,
):
    """
    Build optimizer.
    """

    optimizer_name = (
        cfg["train"]
        .get(
            "optimizer",
            "SGD",
        )
        .lower()
    )

    lr = float(
        cfg["train"].get(
            "lr0",
            0.01,
        )
    )

    weight_decay = float(
        cfg["train"].get(
            "weight_decay",
            5e-4,
        )
    )

    momentum = float(
        cfg["train"].get(
            "momentum",
            0.937,
        )
    )

    parameters = [
        p
        for p in model.parameters()
        if p.requires_grad
    ]

    if optimizer_name == "sgd":

        optimizer = torch.optim.SGD(
            parameters,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=True,
        )

    elif optimizer_name == "adam":

        optimizer = torch.optim.Adam(
            parameters,
            lr=lr,
            weight_decay=weight_decay,
        )

    elif optimizer_name == "adamw":

        optimizer = torch.optim.AdamW(
            parameters,
            lr=lr,
            weight_decay=weight_decay,
        )

    else:

        raise ValueError(
            "\n不支持 optimizer:\n"
            f"{optimizer_name}\n\n"
            "支持:\n"
            "SGD\n"
            "Adam\n"
            "AdamW\n"
        )

    return optimizer


# ============================================================
# 6. LR Scheduler
# ============================================================

def build_scheduler(
    optimizer,
    cfg,
):
    """
    Cosine LR scheduler.

    final_lr = lr0 * lrf
    """

    epochs = int(
        cfg["train"]["epochs"]
    )

    lrf = float(
        cfg["train"].get(
            "lrf",
            0.01,
        )
    )

    lr0 = float(
        cfg["train"].get(
            "lr0",
            0.01,
        )
    )

    eta_min = (
        lr0 * lrf
    )

    scheduler = (
        torch.optim.lr_scheduler
        .CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=eta_min,
        )
    )

    return scheduler


# ============================================================
# 7. AMP
# ============================================================

def build_scaler(
    enabled: bool,
    device: torch.device,
):
    """
    PyTorch AMP GradScaler.
    """

    enabled = (
        enabled
        and device.type == "cuda"
    )

    try:

        scaler = torch.amp.GradScaler(
            "cuda",
            enabled=enabled,
        )

    except TypeError:

        # compatibility
        scaler = (
            torch.cuda.amp.GradScaler(
                enabled=enabled
            )
        )

    return scaler


# ============================================================
# 8. Autocast
# ============================================================

def autocast_context(
    device,
    enabled,
):
    """
    AMP autocast compatibility.
    """

    enabled = (
        enabled
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
# 9. Update YOLO loss hyperparameters
# ============================================================

def configure_model_loss(
    model,
    cfg,
):
    """
    Parameters used by Ultralytics detection loss.
    """

    loss_cfg = cfg.get(
        "loss",
        {}
    )

    # Standard Ultralytics detection-loss weights
    model.args.box = float(
        loss_cfg.get(
            "box",
            7.5,
        )
    )

    model.args.cls = float(
        loss_cfg.get(
            "cls",
            0.5,
        )
    )

    model.args.dfl = float(
        loss_cfg.get(
            "dfl",
            1.5,
        )
    )

    # Force criterion reconstruction
    model.criterion = None


# ============================================================
# 10. Checkpoint
# ============================================================

def save_checkpoint(
    path: Path,
    model,
    optimizer,
    scheduler,
    scaler,
    epoch: int,
    best_map5095: float,
    cfg: Dict,
    stage: str = "stage1",
    stage_epoch: int | None = None,
    global_epoch: int | None = None,
    best_val_loss: float = float("inf"),
    early_stop_counter: int = 0,
):
    """
    Save custom RGB-T checkpoint.

    IMPORTANT
    ---------
    best.pt is selected by validation mAP50-95, NOT val_loss.

    Stored training state:
        model
        optimizer
        scheduler
        AMP scaler
        best mAP50-95
        best val loss (reference only)
        early-stopping counter
        stage information

    This remains a YOLO-tcy custom checkpoint.
    Load it using RGBTDetectionModel / this training script,
    NOT directly with YOLO("best.pt").
    """

    path = Path(
        path
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if stage_epoch is None:
        stage_epoch = int(
            epoch
        )

    if global_epoch is None:
        global_epoch = int(
            epoch
        )

    checkpoint = {
        # ----------------------------------------------------
        # Epoch information
        # ----------------------------------------------------

        "epoch":
            int(
                stage_epoch
            ),

        "stage":
            str(
                stage
            ),

        "stage_epoch":
            int(
                stage_epoch
            ),

        "global_epoch":
            int(
                global_epoch
            ),

        # ----------------------------------------------------
        # Model / optimizer state
        # ----------------------------------------------------

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "scheduler_state_dict":
            scheduler.state_dict(),

        "scaler_state_dict":
            scaler.state_dict(),

        # ----------------------------------------------------
        # Model-selection / early-stop state
        # ----------------------------------------------------

        "best_map5095":
            float(
                best_map5095
            ),

        "best_val_loss":
            float(
                best_val_loss
            ),

        "early_stop_counter":
            int(
                early_stop_counter
            ),

        # ----------------------------------------------------
        # Experiment information
        # ----------------------------------------------------

        "config":
            cfg,

        "model_name":
            model.model_name,

        "fusion":
            model.fusion_type,

        "nc":
            model.nc,

        "names":
            model.names,
    }

    torch.save(
        checkpoint,
        path,
    )


# ============================================================
# 11. Resume checkpoint
# ============================================================

def resume_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    scaler,
    device,
):
    """
    Resume a stage-1 or stage-2 training checkpoint.

    Restores:
        model
        optimizer
        scheduler
        AMP scaler
        best mAP50-95
        early-stop counter
        current stage / epoch
    """

    path = Path(
        path
    )

    if not path.exists():

        raise FileNotFoundError(
            f"Checkpoint 不存在:\n{path}"
        )

    print(
        "\n"
        "============================================================"
    )

    print(
        "Resume RGB-T checkpoint"
    )

    print(
        "============================================================"
    )

    print(
        f"Checkpoint     : {path}"
    )

    ckpt = torch.load(
        path,
        map_location=device,
        weights_only=False,
    )

    if (
        "model_state_dict"
        not in ckpt
    ):

        raise KeyError(
            "Checkpoint 中没有 model_state_dict。"
        )

    model.load_state_dict(
        ckpt[
            "model_state_dict"
        ],
        strict=True,
    )

    if (
        "optimizer_state_dict"
        in ckpt
    ):

        optimizer.load_state_dict(
            ckpt[
                "optimizer_state_dict"
            ]
        )

    if (
        "scheduler_state_dict"
        in ckpt
    ):

        scheduler.load_state_dict(
            ckpt[
                "scheduler_state_dict"
            ]
        )

    if (
        "scaler_state_dict"
        in ckpt
    ):

        scaler.load_state_dict(
            ckpt[
                "scaler_state_dict"
            ]
        )

    resume_stage = str(
        ckpt.get(
            "stage",
            "stage1",
        )
    )

    stage_epoch = int(
        ckpt.get(
            "stage_epoch",
            ckpt.get(
                "epoch",
                -1,
            ),
        )
    )

    start_epoch = (
        stage_epoch
        + 1
    )

    best_map5095 = float(
        ckpt.get(
            "best_map5095",
            -1.0,
        )
    )

    best_val_loss = float(
        ckpt.get(
            "best_val_loss",
            float("inf"),
        )
    )

    early_stop_counter = int(
        ckpt.get(
            "early_stop_counter",
            0,
        )
    )

    global_epoch = int(
        ckpt.get(
            "global_epoch",
            stage_epoch,
        )
    )

    print(
        f"Stage          : {resume_stage}"
    )

    print(
        f"Next stage ep. : {start_epoch + 1}"
    )

    print(
        f"Best mAP50-95 : {best_map5095:.6f}"
    )

    print(
        f"EarlyStop      : {early_stop_counter}"
    )

    print(
        "============================================================\n"
    )

    return {
        "stage":
            resume_stage,

        "start_epoch":
            start_epoch,

        "best_map5095":
            best_map5095,

        "best_val_loss":
            best_val_loss,

        "early_stop_counter":
            early_stop_counter,

        "global_epoch":
            global_epoch,
    }


# ============================================================
# 12. Load model weights only
# ============================================================

def load_weights_only(
    path,
    model,
    device,
):
    """
    Load ONLY model parameters from a checkpoint.

    Used when entering stage 2.

    Stage 2 intentionally DOES NOT inherit:
        optimizer
        scheduler
        AMP scaler

    These states are reinitialized so the fine-tuning stage
    receives a fresh, lower-learning-rate schedule.
    """

    path = Path(
        path
    )

    if not path.exists():

        raise FileNotFoundError(
            "\nStage-2 initialization checkpoint 不存在:\n"
            f"{path}"
        )

    ckpt = torch.load(
        path,
        map_location=device,
        weights_only=False,
    )

    if (
        "model_state_dict"
        not in ckpt
    ):

        raise KeyError(
            "\nCheckpoint 中没有 model_state_dict。\n"
        )

    model.load_state_dict(
        ckpt[
            "model_state_dict"
        ],
        strict=True,
    )

    print(
        "\n"
        "============================================================"
    )

    print(
        "Load best model for Stage 2 fine-tuning"
    )

    print(
        "============================================================"
    )

    print(
        f"Checkpoint     : {path}"
    )

    print(
        "Model weights  : loaded"
    )

    print(
        "Optimizer      : NEW"
    )

    print(
        "Scheduler      : NEW"
    )

    print(
        "AMP scaler     : NEW"
    )

    print(
        "============================================================\n"
    )

    return ckpt


# ============================================================
# 13. History helper
# ============================================================

def load_existing_history(
    path: Path,
):
    """
    Load history.json when resuming an interrupted experiment.
    """

    path = Path(
        path
    )

    if not path.exists():
        return []

    try:

        with path.open(
            "r",
            encoding="utf-8",
        ) as f:

            history = json.load(
                f
            )

        if isinstance(
            history,
            list,
        ):

            return history

    except Exception as e:

        print(
            "WARNING: history.json 读取失败，"
            "将从空 history 继续。"
        )

        print(
            e
        )

    return []


# ============================================================
# Training progress utilities
# ============================================================

def get_gpu_memory_gb(
    device: torch.device,
) -> float:
    """
    Current reserved CUDA memory in GB.
    """

    if device.type != "cuda":
        return 0.0

    return (
        torch.cuda.memory_reserved(
            device
        )
        / (1024 ** 3)
    )


def normalize_loss_items(
    loss_items,
):
    """
    Convert Ultralytics loss_items into a standard dict.

    Compatible with:
        dict:
            {
                "box_loss": ...,
                "cls_loss": ...,
                "dfl_loss": ...
            }

        Tensor:
            tensor([box, cls, dfl])

        list / tuple:
            [box, cls, dfl]
    """

    # ========================================================
    # Current Ultralytics style: dict
    # ========================================================

    if isinstance(
        loss_items,
        dict,
    ):

        result = {}

        for key, value in loss_items.items():

            if torch.is_tensor(value):

                value = float(
                    value.detach().mean().item()
                )

            else:

                value = float(value)

            result[key] = value

        return result

    # ========================================================
    # Tensor style
    # ========================================================

    if torch.is_tensor(
        loss_items
    ):

        values = (
            loss_items
            .detach()
            .flatten()
            .cpu()
            .tolist()
        )

    elif isinstance(
        loss_items,
        (list, tuple),
    ):

        values = []

        for value in loss_items:

            if torch.is_tensor(value):

                values.append(
                    float(
                        value.detach().item()
                    )
                )

            else:

                values.append(
                    float(value)
                )

    else:

        return {
            "loss": float(
                loss_items
            )
        }

    # ========================================================
    # Standard detection loss names
    # ========================================================

    if len(values) == 3:

        return {
            "box_loss":
                values[0],

            "cls_loss":
                values[1],

            "dfl_loss":
                values[2],
        }

    return {
        f"loss_{i}": value
        for i, value
        in enumerate(values)
    }


def update_running_losses(
    running: dict,
    current: dict,
    batch_i: int,
):
    """
    Ultralytics-like running average:

        avg_i =
        (old_avg * batch_i + current)
        / (batch_i + 1)
    """

    for key, value in current.items():

        if key not in running:

            running[key] = value

        else:

            running[key] = (
                running[key] * batch_i
                + value
            ) / (
                batch_i + 1
            )

    return running


def get_standard_losses(
    losses: dict,
):
    """
    Safely extract box / cls / dfl losses.

    Also supports slightly different key naming.
    """

    def find_value(
        candidates,
        default=0.0,
    ):

        for key in candidates:

            if key in losses:
                return losses[key]

        return default

    box_loss = find_value(
        [
            "box_loss",
            "box",
        ]
    )

    cls_loss = find_value(
        [
            "cls_loss",
            "cls",
        ]
    )

    dfl_loss = find_value(
        [
            "dfl_loss",
            "dfl",
            "l1_loss",
            "l1",
        ]
    )

    return (
        box_loss,
        cls_loss,
        dfl_loss,
    )
# ============================================================
# 12. One training epoch
# ============================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    scaler,
    device,

    amp=True,

    grad_clip=10.0,

    epoch=0,

    epochs=1,

    rgb_imgsz=640,

    tir_imgsz=640,
):
    """
    Train one epoch.

    Progress UI is designed to resemble
    Ultralytics YOLO training output.
    """

    model.train()

    # ========================================================
    # Statistics
    # ========================================================

    running_total_loss = 0.0

    running_loss_items = {}

    num_batches = len(
        loader
    )

    # ========================================================
    # Ultralytics-like header
    # ========================================================

    print(
        (
            f"{'Epoch':>11}"
            f"{'GPU_mem':>11}"
            f"{'box_loss':>11}"
            f"{'cls_loss':>11}"
            f"{'dfl_loss':>11}"
            f"{'Instances':>11}"
            f"{'Size':>13}"
        )
    )

    # ========================================================
    # Progress bar
    # ========================================================

    pbar = tqdm(
        enumerate(loader),

        total=num_batches,

        dynamic_ncols=True,

        leave=True,

        bar_format=(
            "{desc} "
            "{percentage:3.0f}%|"
            "{bar}| "
            "{n_fmt}/{total_fmt} "
            "[{elapsed}<{remaining}, "
            "{rate_fmt}]"
        ),
    )

    # ========================================================
    # Batch loop
    # ========================================================

    for batch_i, batch in pbar:

        # ====================================================
        # Move batch to GPU
        # ====================================================

        batch = preprocess_batch(
            batch,
            device,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        # ====================================================
        # Forward
        # ====================================================

        with autocast_context(
            device,
            amp,
        ):

            loss_raw, loss_items = model(
                batch
            )

            # -----------------------------------------------
            # Ultralytics loss may contain multiple elements.
            # Sum into scalar before backward.
            # -----------------------------------------------

            loss = loss_raw.sum()

        # ====================================================
        # Backward
        # ====================================================

        scaler.scale(
            loss
        ).backward()

        # ====================================================
        # Gradient clipping
        # ====================================================

        if grad_clip > 0:

            scaler.unscale_(
                optimizer
            )

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=grad_clip,
            )

        # ====================================================
        # Optimizer
        # ====================================================

        scaler.step(
            optimizer
        )

        scaler.update()

        # ====================================================
        # Total loss
        # ====================================================

        loss_value = float(
            loss.detach().item()
        )

        running_total_loss += (
            loss_value
        )

        # ====================================================
        # Individual loss components
        # ====================================================

        current_loss_items = (
            normalize_loss_items(
                loss_items
            )
        )

        running_loss_items = (
            update_running_losses(
                running_loss_items,

                current_loss_items,

                batch_i,
            )
        )

        (
            box_loss,
            cls_loss,
            dfl_loss,
        ) = get_standard_losses(
            running_loss_items
        )

        # ====================================================
        # Instances
        #
        # Main fused detector uses RGB GT.
        # Therefore instance count = RGB targets.
        # ====================================================

        instances = int(
            batch[
                "rgb_cls"
            ].shape[0]
        )

        # ====================================================
        # GPU memory
        # ====================================================

        gpu_mem = get_gpu_memory_gb(
            device
        )

        # ====================================================
        # Size display
        #
        # Because this is RGB-T, show both resolutions.
        # ====================================================

        if (
            rgb_imgsz
            == tir_imgsz
        ):

            size_str = str(
                rgb_imgsz
            )

        else:

            size_str = (
                f"{rgb_imgsz}/"
                f"{tir_imgsz}"
            )

        # ====================================================
        # Progress description
        # ====================================================

        description = (
            f"{epoch + 1:>5}/{epochs:<5}"
            f"{gpu_mem:>10.3g}G"
            f"{box_loss:>11.4g}"
            f"{cls_loss:>11.4g}"
            f"{dfl_loss:>11.4g}"
            f"{instances:>11}"
            f"{size_str:>13}"
        )

        pbar.set_description(
            description,
            refresh=False,
        )

    pbar.close()

    # ========================================================
    # Epoch total loss
    # ========================================================

    epoch_loss = (
        running_total_loss
        / max(
            num_batches,
            1,
        )
    )

    return (
        epoch_loss,
        running_loss_items,
    )


# ============================================================
# 13. Validation loss
# ============================================================

@torch.no_grad()
def validate_loss(
    model,
    loader,
    device,
    amp=True,
    epoch=0,
    epochs=1,
):
    """
    Validation loss with tqdm progress bar.
    """

    model.eval()

    running_loss = 0.0

    num_batches = len(loader)

    pbar = tqdm(
        loader,
        total=num_batches,
        desc=f"Val   {epoch + 1}/{epochs}",
        dynamic_ncols=True,
        leave=True,
    )

    for batch_i, batch in enumerate(pbar):

        batch = preprocess_batch(
            batch,
            device,
        )

        # YOLO criterion expects training-format raw outputs.
        model.train()

        with autocast_context(
            device,
            amp,
        ):

            loss_raw, _ = model(
                batch
            )

            loss = loss_raw.sum()

        model.eval()

        loss_value = float(
            loss.detach().item()
        )

        running_loss += loss_value

        avg_loss = (
            running_loss
            / (batch_i + 1)
        )

        pbar.set_postfix(
            {
                "loss":
                    f"{loss_value:.4f}",

                "avg":
                    f"{avg_loss:.4f}",
            }
        )

    pbar.close()

    return (
        running_loss
        / max(
            num_batches,
            1,
        )
    )

# ============================================================
# 14. Detection metrics validation
# ============================================================

@torch.inference_mode()
def validate_metrics(
    model,
    loader,
    device,
    amp=True,
    conf_thres=0.001,
    iou_thres=0.7,
    max_det=300,
):
    """
    Calculate detection metrics on validation set.

    Output
    ------
    {
        "precision": ...
        "recall": ...
        "map50": ...
        "map75": ...
        "map5095": ...
        "images": ...
        "instances": ...
    }

    Important
    ---------
    Metrics use exactly the same core functions as val_rgbt.py:

        postprocess_predictions()
        prepare_rgb_ground_truth()
        process_single_image()

    Therefore training-time mAP should be consistent with
    standalone val_rgbt.py.
    """

    # ========================================================
    # Evaluation mode
    # ========================================================

    model.eval()

    # ========================================================
    # Class names
    # ========================================================

    names = model.names

    if not isinstance(
        names,
        dict,
    ):

        names = {
            i: name
            for i, name
            in enumerate(names)
        }

    names = {
        int(k): str(v)
        for k, v in names.items()
    }

    # ========================================================
    # Ultralytics detection metrics
    # ========================================================

    metrics = DetMetrics(
        names=names
    )

    # ========================================================
    # IoU thresholds:
    #
    # 0.50
    # 0.55
    # ...
    # 0.95
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

    num_batches = len(
        loader
    )

    # ========================================================
    # Ultralytics-style validation header
    # ========================================================

    header = (
        f"{'Class':>22}"
        f"{'Images':>11}"
        f"{'Instances':>11}"
        f"{'Box(P':>11}"
        f"{'R':>11}"
        f"{'mAP50':>11}"
        f"{'mAP50-95)':>13}"
    )

    print(
        header
    )

    # ========================================================
    # Progress bar
    # ========================================================

    pbar = tqdm(
        loader,

        total=num_batches,

        dynamic_ncols=True,

        leave=True,

        bar_format=(
            "{desc} "
            "{percentage:3.0f}%|"
            "{bar}| "
            "{n_fmt}/{total_fmt} "
            "[{elapsed}<{remaining}, "
            "{rate_fmt}]"
        ),
    )

    pbar.set_description(
        f"{'validating':>22}",
        refresh=False,
    )

    # ========================================================
    # Validation loop
    # ========================================================

    for batch in pbar:

        # ----------------------------------------------------
        # Number of images in current batch
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Move data to GPU
        # ----------------------------------------------------

        batch = preprocess_batch(
            batch,
            device,
        )

        # ----------------------------------------------------
        # RGB-T inference
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # YOLO postprocess
        # ----------------------------------------------------

        preds = postprocess_predictions(
            preds=preds,

            model=model,

            conf_thres=conf_thres,

            iou_thres=iou_thres,

            max_det=max_det,
        )

        # ====================================================
        # Per-image statistics
        # ====================================================

        for sample_index, pred in enumerate(
            preds
        ):

            # ------------------------------------------------
            # RGB GT
            # ------------------------------------------------

            gt = prepare_rgb_ground_truth(
                batch,
                sample_index,
            )

            # ------------------------------------------------
            # TP at IoU 0.50:0.95
            # ------------------------------------------------

            tp = process_single_image(
                pred=pred,

                gt=gt,

                iouv=iouv,
            )

            # ------------------------------------------------
            # GT classes
            # ------------------------------------------------

            target_cls = (
                gt[
                    "cls"
                ]
                .detach()
                .float()
                .cpu()
                .numpy()
            )

            # ------------------------------------------------
            # Predictions
            # ------------------------------------------------

            if (
                pred[
                    "cls"
                ].shape[0]
                == 0
            ):

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

            # ------------------------------------------------
            # Update Ultralytics metrics
            # ------------------------------------------------

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

        pbar.set_description(
            (
                f"{'all':>22}"
                f"{seen:>11}"
                f"{total_instances:>11}"
            ),
            refresh=False,
        )

    pbar.close()

    # ========================================================
    # Calculate final metrics
    # ========================================================

    metrics.process(
        plot=False
    )

    (
        precision,
        recall,
        map50,
        map5095,
    ) = metrics.mean_results()

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
    # Official-style summary line
    # ========================================================

    print(
        (
            f"{'all':>22}"
            f"{seen:>11}"
            f"{total_instances:>11}"
            f"{precision:>11.3f}"
            f"{recall:>11.3f}"
            f"{map50:>11.3f}"
            f"{map5095:>13.3f}"
        )
    )

    # ========================================================
    # Return values are 0~1, NOT percentages
    # ========================================================

    return {
        "precision":
            precision,

        "recall":
            recall,

        "map50":
            map50,

        "map75":
            map75,

        "map5095":
            map5095,

        "images":
            seen,

        "instances":
            total_instances,
    }
# ============================================================
# 17. One complete training stage
# ============================================================

def run_training_stage(
    *,
    stage_name: str,
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    scaler,
    device,
    amp: bool,
    grad_clip: float,
    epochs: int,
    start_epoch: int,
    patience: int,
    min_delta: float,
    rgb_imgsz: int,
    tir_imgsz: int,
    save_dir: Path,
    weights_dir: Path,
    cfg: Dict,
    history: list,
    best_map5095: float,
    best_val_loss: float,
    early_stop_counter: int = 0,
    global_epoch_offset: int = 0,
):
    """
    Train one stage.

    Model selection:
        validation mAP50-95

    Early stopping:
        stop when mAP50-95 has not improved for `patience`
        consecutive epochs.

    best.pt:
        global best across both stages.

    stage1_best.pt / stage2_best.pt:
        stage-specific copies when a new GLOBAL best appears.
    """

    stage_name = str(
        stage_name
    )

    epochs = int(
        epochs
    )

    start_epoch = int(
        start_epoch
    )

    patience = int(
        patience
    )

    min_delta = float(
        min_delta
    )

    early_stop_counter = int(
        early_stop_counter
    )

    epochs_ran = 0

    last_stage_epoch = (
        start_epoch
        - 1
    )

    last_global_epoch = (
        global_epoch_offset
        + last_stage_epoch
    )

    stopped_early = False

    print(
        "\n"
        "############################################################"
    )

    print(
        f"{stage_name.upper()} START"
    )

    print(
        "############################################################"
    )

    print(
        f"Max epochs       : {epochs}"
    )

    print(
        f"Start epoch      : {start_epoch + 1}"
    )

    print(
        f"Patience         : {patience}"
    )

    print(
        f"Best mAP50-95    : {best_map5095:.6f}"
    )

    print(
        f"EarlyStop count  : "
        f"{early_stop_counter}/{patience}"
    )

    print(
        "############################################################\n"
    )

    for stage_epoch in range(
        start_epoch,
        epochs,
    ):

        global_epoch = (
            global_epoch_offset
            + stage_epoch
        )

        # ====================================================
        # Train
        # ====================================================

        (
            train_loss,
            train_loss_items,
        ) = train_one_epoch(
            model=model,

            loader=train_loader,

            optimizer=optimizer,

            scaler=scaler,

            device=device,

            amp=amp,

            grad_clip=grad_clip,

            epoch=stage_epoch,

            epochs=epochs,

            rgb_imgsz=rgb_imgsz,

            tir_imgsz=tir_imgsz,
        )

        # ====================================================
        # Validation loss
        # ====================================================

        val_loss = validate_loss(
            model=model,

            loader=val_loader,

            device=device,

            amp=amp,

            epoch=stage_epoch,

            epochs=epochs,
        )

        # ====================================================
        # Detection metrics
        # ====================================================

        val_metrics = validate_metrics(
            model=model,

            loader=val_loader,

            device=device,

            amp=amp,

            conf_thres=0.001,

            iou_thres=0.7,

            max_det=300,
        )

        # ====================================================
        # Scheduler
        # ====================================================

        scheduler.step()

        current_lr = float(
            optimizer.param_groups[
                0
            ][
                "lr"
            ]
        )

        (
            train_box_loss,
            train_cls_loss,
            train_dfl_loss,
        ) = get_standard_losses(
            train_loss_items
        )

        current_map5095 = float(
            val_metrics[
                "map5095"
            ]
        )

        current_map50 = float(
            val_metrics[
                "map50"
            ]
        )

        current_map75 = float(
            val_metrics[
                "map75"
            ]
        )

        current_precision = float(
            val_metrics[
                "precision"
            ]
        )

        current_recall = float(
            val_metrics[
                "recall"
            ]
        )

        best_val_loss = min(
            float(
                best_val_loss
            ),
            float(
                val_loss
            ),
        )

        # ====================================================
        # mAP-based best model + early stopping
        # ====================================================

        map_improved = (
            current_map5095
            >
            (
                best_map5095
                + min_delta
            )
        )

        if map_improved:

            best_map5095 = (
                current_map5095
            )

            early_stop_counter = 0

            # -----------------------------------------------
            # Global best.pt
            # -----------------------------------------------

            save_checkpoint(
                weights_dir
                / "best.pt",

                model=model,

                optimizer=optimizer,

                scheduler=scheduler,

                scaler=scaler,

                epoch=stage_epoch,

                best_map5095=best_map5095,

                cfg=cfg,

                stage=stage_name,

                stage_epoch=stage_epoch,

                global_epoch=global_epoch,

                best_val_loss=best_val_loss,

                early_stop_counter=(
                    early_stop_counter
                ),
            )

            # -----------------------------------------------
            # Stage-specific best copy
            # -----------------------------------------------

            save_checkpoint(
                weights_dir
                / f"{stage_name}_best.pt",

                model=model,

                optimizer=optimizer,

                scheduler=scheduler,

                scaler=scaler,

                epoch=stage_epoch,

                best_map5095=best_map5095,

                cfg=cfg,

                stage=stage_name,

                stage_epoch=stage_epoch,

                global_epoch=global_epoch,

                best_val_loss=best_val_loss,

                early_stop_counter=(
                    early_stop_counter
                ),
            )

        else:

            early_stop_counter += 1

        # ====================================================
        # History
        # ====================================================

        record = {
            "stage":
                stage_name,

            "stage_epoch":
                stage_epoch + 1,

            "global_epoch":
                global_epoch + 1,

            "train_loss":
                float(
                    train_loss
                ),

            "box_loss":
                float(
                    train_box_loss
                ),

            "cls_loss":
                float(
                    train_cls_loss
                ),

            "dfl_loss":
                float(
                    train_dfl_loss
                ),

            "val_loss":
                float(
                    val_loss
                ),

            "precision":
                current_precision,

            "recall":
                current_recall,

            "map50":
                current_map50,

            "map75":
                current_map75,

            "map5095":
                current_map5095,

            "best_map5095":
                float(
                    best_map5095
                ),

            "early_stop_counter":
                int(
                    early_stop_counter
                ),

            "patience":
                int(
                    patience
                ),

            "lr":
                current_lr,
        }

        history.append(
            record
        )

        history_path = (
            save_dir
            / "history.json"
        )

        with history_path.open(
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                history,
                f,
                ensure_ascii=False,
                indent=2,
            )

        # ====================================================
        # last.pt
        # ====================================================

        save_checkpoint(
            weights_dir
            / "last.pt",

            model=model,

            optimizer=optimizer,

            scheduler=scheduler,

            scaler=scaler,

            epoch=stage_epoch,

            best_map5095=best_map5095,

            cfg=cfg,

            stage=stage_name,

            stage_epoch=stage_epoch,

            global_epoch=global_epoch,

            best_val_loss=best_val_loss,

            early_stop_counter=(
                early_stop_counter
            ),
        )

        save_checkpoint(
            weights_dir
            / f"{stage_name}_last.pt",

            model=model,

            optimizer=optimizer,

            scheduler=scheduler,

            scaler=scaler,

            epoch=stage_epoch,

            best_map5095=best_map5095,

            cfg=cfg,

            stage=stage_name,

            stage_epoch=stage_epoch,

            global_epoch=global_epoch,

            best_val_loss=best_val_loss,

            early_stop_counter=(
                early_stop_counter
            ),
        )

        # ====================================================
        # Epoch summary
        # ====================================================

        print(
            "\n"
            f"{stage_name} "
            f"Epoch {stage_epoch + 1}/{epochs} finished:"
        )

        print(
            f"  train_loss     = "
            f"{train_loss:.6f}"
        )

        print(
            f"  box_loss       = "
            f"{train_box_loss:.6f}"
        )

        print(
            f"  cls_loss       = "
            f"{train_cls_loss:.6f}"
        )

        print(
            f"  dfl_loss       = "
            f"{train_dfl_loss:.6f}"
        )

        print(
            f"  val_loss       = "
            f"{val_loss:.6f}"
        )

        print(
            f"  precision      = "
            f"{current_precision:.4f}"
        )

        print(
            f"  recall         = "
            f"{current_recall:.4f}"
        )

        print(
            f"  mAP50          = "
            f"{current_map50:.4f}"
        )

        print(
            f"  mAP75          = "
            f"{current_map75:.4f}"
        )

        print(
            f"  mAP50-95       = "
            f"{current_map5095:.4f}"
        )

        print(
            f"  best mAP50-95  = "
            f"{best_map5095:.4f}"
        )

        print(
            f"  EarlyStop      = "
            f"{early_stop_counter}/{patience}"
        )

        print(
            f"  lr             = "
            f"{current_lr:.8f}"
        )

        if map_improved:

            print(
                "  [BEST] "
                f"mAP50-95="
                f"{best_map5095:.6f}"
            )

        # ====================================================
        # Bookkeeping
        # ====================================================

        epochs_ran += 1

        last_stage_epoch = (
            stage_epoch
        )

        last_global_epoch = (
            global_epoch
        )

        # ====================================================
        # Early stopping
        # ====================================================

        if (
            patience > 0
            and early_stop_counter
            >= patience
        ):

            stopped_early = True

            print(
                "\n"
                "============================================================"
            )

            print(
                f"{stage_name} EARLY STOPPING"
            )

            print(
                "============================================================"
            )

            print(
                f"No mAP50-95 improvement for "
                f"{patience} consecutive epochs."
            )

            print(
                f"Best mAP50-95 : "
                f"{best_map5095:.6f}"
            )

            print(
                f"Stopped at     : "
                f"{stage_epoch + 1}/{epochs}"
            )

            print(
                "============================================================\n"
            )

            break

    return {
        "best_map5095":
            float(
                best_map5095
            ),

        "best_val_loss":
            float(
                best_val_loss
            ),

        "early_stop_counter":
            int(
                early_stop_counter
            ),

        "epochs_ran":
            int(
                epochs_ran
            ),

        "last_stage_epoch":
            int(
                last_stage_epoch
            ),

        "last_global_epoch":
            int(
                last_global_epoch
            ),

        "stopped_early":
            bool(
                stopped_early
            ),

        "history":
            history,
    }


# ============================================================
# 18. Main training
# ============================================================

def train(
    cfg: Dict,
):

    # ========================================================
    # Basic settings
    # ========================================================

    seed = int(
        cfg[
            "train"
        ].get(
            "seed",
            0,
        )
    )

    deterministic = bool(
        cfg[
            "train"
        ].get(
            "deterministic",
            True,
        )
    )

    set_seed(
        seed,
        deterministic,
    )

    device = select_device(
        str(
            cfg[
                "train"
            ].get(
                "device",
                "0",
            )
        )
    )

    print(
        "\n"
        "============================================================"
    )

    print(
        "RGB-T YOLO Two-Stage Training"
    )

    print(
        "============================================================"
    )

    print(
        f"Device        : {device}"
    )

    if device.type == "cuda":

        print(
            "GPU           : "
            f"{torch.cuda.get_device_name(device)}"
        )

    # ========================================================
    # Output
    # ========================================================

    project = Path(
        cfg[
            "output"
        ].get(
            "project",
            ROOT
            / "runs/rgbt",
        )
    )

    if not project.is_absolute():

        project = (
            ROOT
            / project
        )

    run_name = (
        cfg[
            "output"
        ].get(
            "name",
            "rgbt_exp",
        )
    )

    save_dir = (
        project
        / run_name
    ).resolve()

    weights_dir = (
        save_dir
        / "weights"
    )

    weights_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # Training-stage settings
    #
    # Defaults required by this project:
    #     Stage 1 = 300 epochs
    #     Stage 2 = 200 epochs
    #     patience = 50
    # ========================================================

    train_cfg = cfg[
        "train"
    ]

    stage1_epochs = int(
        train_cfg.get(
            "epochs",
            300,
        )
    )

    patience = int(
        train_cfg.get(
            "patience",
            50,
        )
    )

    min_delta = float(
        train_cfg.get(
            "early_stop_min_delta",
            1e-6,
        )
    )

    stage2_user_cfg = (
        train_cfg.get(
            "stage2",
            {},
        )
        or {}
    )

    stage2_enabled = bool(
        stage2_user_cfg.get(
            "enabled",
            True,
        )
    )

    stage2_epochs = int(
        stage2_user_cfg.get(
            "epochs",
            200,
        )
    )

    stage1_lr0 = float(
        train_cfg.get(
            "lr0",
            0.01,
        )
    )

    stage2_lr0 = float(
        stage2_user_cfg.get(
            "lr0",
            stage1_lr0
            * 0.1,
        )
    )

    stage2_lrf = float(
        stage2_user_cfg.get(
            "lrf",
            train_cfg.get(
                "lrf",
                0.01,
            ),
        )
    )

    stage2_patience = int(
        stage2_user_cfg.get(
            "patience",
            patience,
        )
    )

    stage2_min_delta = float(
        stage2_user_cfg.get(
            "early_stop_min_delta",
            min_delta,
        )
    )

    # Store effective defaults in saved config.
    cfg.setdefault(
        "train",
        {}
    )

    cfg[
        "train"
    ][
        "epochs"
    ] = (
        stage1_epochs
    )

    cfg[
        "train"
    ][
        "patience"
    ] = (
        patience
    )

    cfg[
        "train"
    ][
        "early_stop_min_delta"
    ] = (
        min_delta
    )

    cfg[
        "train"
    ][
        "stage2"
    ] = {
        **stage2_user_cfg,

        "enabled":
            stage2_enabled,

        "epochs":
            stage2_epochs,

        "lr0":
            stage2_lr0,

        "lrf":
            stage2_lrf,

        "patience":
            stage2_patience,

        "early_stop_min_delta":
            stage2_min_delta,
    }

    save_config(
        cfg,
        save_dir
        / "config.yaml",
    )

    # ========================================================
    # Dataset
    # ========================================================

    data_cfg = cfg[
        "data"
    ]

    rgb_yaml = data_cfg[
        "rgb"
    ]

    tir_yaml = data_cfg[
        "tir"
    ]

    rgb_imgsz = int(
        data_cfg.get(
            "rgb_imgsz",
            640,
        )
    )

    tir_imgsz = int(
        data_cfg.get(
            "tir_imgsz",
            640,
        )
    )

    pair_mode = data_cfg.get(
        "pair_mode",
        "relative",
    )

    print(
        f"RGB dataset   : {rgb_yaml}"
    )

    print(
        f"TIR dataset   : {tir_yaml}"
    )

    print(
        f"RGB imgsz     : {rgb_imgsz}"
    )

    print(
        f"TIR imgsz     : {tir_imgsz}"
    )

    # ========================================================
    # Datasets
    # ========================================================

    train_dataset = (
        build_rgbt_dataset(
            rgb_yaml=rgb_yaml,

            tir_yaml=tir_yaml,

            split="train",

            rgb_imgsz=rgb_imgsz,

            tir_imgsz=tir_imgsz,

            pair_mode=pair_mode,

            augment=True,

            fliplr=float(
                cfg[
                    "augment"
                ].get(
                    "fliplr",
                    0.5,
                )
            ),

            flipud=float(
                cfg[
                    "augment"
                ].get(
                    "flipud",
                    0.0,
                )
            ),

            tir_channels=3,

            strict_pair=True,
        )
    )

    val_dataset = (
        build_rgbt_dataset(
            rgb_yaml=rgb_yaml,

            tir_yaml=tir_yaml,

            split="val",

            rgb_imgsz=rgb_imgsz,

            tir_imgsz=tir_imgsz,

            pair_mode=pair_mode,

            augment=False,

            tir_channels=3,

            strict_pair=True,
        )
    )

    batch_size = int(
        train_cfg.get(
            "batch",
            8,
        )
    )

    workers = int(
        train_cfg.get(
            "workers",
            4,
        )
    )

    train_loader = (
        build_rgbt_dataloader(
            train_dataset,

            batch_size=batch_size,

            workers=workers,

            shuffle=True,

            pin_memory=(
                device.type
                == "cuda"
            ),
        )
    )

    val_loader = (
        build_rgbt_dataloader(
            val_dataset,

            batch_size=batch_size,

            workers=workers,

            shuffle=False,

            pin_memory=(
                device.type
                == "cuda"
            ),
        )
    )

    # ========================================================
    # Model
    # ========================================================

    model_cfg = cfg[
        "model"
    ]

    model = RGBTDetectionModel(
        model_name=model_cfg.get(
            "name",
            "yolo26n",
        ),

        nc=train_dataset.nc,

        pretrained=bool(
            model_cfg.get(
                "pretrained",
                True,
            )
        ),

        fusion=model_cfg.get(
            "fusion",
            "concat",
        ),

        align_mode=model_cfg.get(
            "align_mode",
            "bilinear",
        ),

        names=train_dataset.names,

        verbose=True,
    )

    model = model.to(
        device
    )

    configure_model_loss(
        model,
        cfg,
    )

    model.print_info()

    # ========================================================
    # Common training parameters
    # ========================================================

    amp = bool(
        train_cfg.get(
            "amp",
            True,
        )
    )

    grad_clip = float(
        train_cfg.get(
            "grad_clip",
            10.0,
        )
    )

    resume = train_cfg.get(
        "resume",
        None,
    )

    history = (
        load_existing_history(
            save_dir
            / "history.json"
        )
        if resume
        else []
    )

    best_map5095 = -1.0

    best_val_loss = float(
        "inf"
    )

    # ========================================================
    # Stage 1 optimizer / scheduler / scaler
    # ========================================================

    stage1_cfg = copy.deepcopy(
        cfg
    )

    stage1_cfg[
        "train"
    ][
        "epochs"
    ] = (
        stage1_epochs
    )

    optimizer = build_optimizer(
        model,
        stage1_cfg,
    )

    scheduler = build_scheduler(
        optimizer,
        stage1_cfg,
    )

    scaler = build_scaler(
        amp,
        device,
    )

    # ========================================================
    # Resume state
    # ========================================================

    resume_state = None

    if resume:

        resume_state = (
            resume_checkpoint(
                resume,

                model,

                optimizer,

                scheduler,

                scaler,

                device,
            )
        )

        best_map5095 = (
            resume_state[
                "best_map5095"
            ]
        )

        best_val_loss = (
            resume_state[
                "best_val_loss"
            ]
        )

    # ========================================================
    # Summary
    # ========================================================

    print(
        "\n"
        "============================================================"
    )

    print(
        "Two-stage training configuration"
    )

    print(
        "============================================================"
    )

    print(
        f"Stage 1 epochs : "
        f"{stage1_epochs}"
    )

    print(
        f"Stage 1 lr0    : "
        f"{stage1_lr0}"
    )

    print(
        f"Stage 2 enabled: "
        f"{stage2_enabled}"
    )

    print(
        f"Stage 2 epochs : "
        f"{stage2_epochs}"
    )

    print(
        f"Stage 2 lr0    : "
        f"{stage2_lr0}"
    )

    print(
        f"Patience       : "
        f"{patience}"
    )

    print(
        f"Selection      : "
        "mAP50-95"
    )

    print(
        f"Batch          : "
        f"{batch_size}"
    )

    print(
        f"Workers        : "
        f"{workers}"
    )

    print(
        f"Optimizer      : "
        f"{train_cfg.get('optimizer', 'SGD')}"
    )

    print(
        f"AMP            : "
        f"{amp}"
    )

    print(
        f"Save dir       : "
        f"{save_dir}"
    )

    print(
        "============================================================\n"
    )

    # ========================================================
    # Decide resume stage
    # ========================================================

    resume_stage = (
        resume_state[
            "stage"
        ]
        if resume_state
        is not None
        else "stage1"
    )

    # ========================================================
    # Stage 1
    # ========================================================

    stage1_result = None

    if resume_stage == "stage1":

        stage1_start_epoch = (
            resume_state[
                "start_epoch"
            ]
            if resume_state
            is not None
            else 0
        )

        stage1_counter = (
            resume_state[
                "early_stop_counter"
            ]
            if resume_state
            is not None
            else 0
        )

        stage1_result = run_training_stage(
            stage_name="stage1",

            model=model,

            train_loader=train_loader,

            val_loader=val_loader,

            optimizer=optimizer,

            scheduler=scheduler,

            scaler=scaler,

            device=device,

            amp=amp,

            grad_clip=grad_clip,

            epochs=stage1_epochs,

            start_epoch=stage1_start_epoch,

            patience=patience,

            min_delta=min_delta,

            rgb_imgsz=rgb_imgsz,

            tir_imgsz=tir_imgsz,

            save_dir=save_dir,

            weights_dir=weights_dir,

            cfg=cfg,

            history=history,

            best_map5095=best_map5095,

            best_val_loss=best_val_loss,

            early_stop_counter=(
                stage1_counter
            ),

            global_epoch_offset=0,
        )

        best_map5095 = (
            stage1_result[
                "best_map5095"
            ]
        )

        best_val_loss = (
            stage1_result[
                "best_val_loss"
            ]
        )

        history = (
            stage1_result[
                "history"
            ]
        )

        stage1_last_global_epoch = (
            stage1_result[
                "last_global_epoch"
            ]
        )

    else:

        # Resuming an already-running stage 2 checkpoint.
        stage1_last_global_epoch = (
            max(
                resume_state[
                    "global_epoch"
                ]
                - resume_state[
                    "start_epoch"
                ],
                -1,
            )
        )

    # ========================================================
    # Stop here when Stage 2 is disabled
    # ========================================================

    if not stage2_enabled:

        print(
            "\nStage 2 disabled."
        )

        print(
            f"Best mAP50-95 : "
            f"{best_map5095:.6f}"
        )

        print(
            f"Best weight   : "
            f"{weights_dir / 'best.pt'}"
        )

        return

    # ========================================================
    # Stage 2 configuration
    #
    # Fresh optimizer / scheduler / scaler.
    # ========================================================

    stage2_cfg = copy.deepcopy(
        cfg
    )

    stage2_cfg[
        "train"
    ][
        "epochs"
    ] = (
        stage2_epochs
    )

    stage2_cfg[
        "train"
    ][
        "lr0"
    ] = (
        stage2_lr0
    )

    stage2_cfg[
        "train"
    ][
        "lrf"
    ] = (
        stage2_lrf
    )

    if (
        "optimizer"
        in stage2_user_cfg
    ):

        stage2_cfg[
            "train"
        ][
            "optimizer"
        ] = (
            stage2_user_cfg[
                "optimizer"
            ]
        )

    if (
        "momentum"
        in stage2_user_cfg
    ):

        stage2_cfg[
            "train"
        ][
            "momentum"
        ] = (
            stage2_user_cfg[
                "momentum"
            ]
        )

    if (
        "weight_decay"
        in stage2_user_cfg
    ):

        stage2_cfg[
            "train"
        ][
            "weight_decay"
        ] = (
            stage2_user_cfg[
                "weight_decay"
            ]
        )

    # ========================================================
    # Stage 2 resume
    # ========================================================

    if resume_stage == "stage2":

        # The currently loaded model / optimizer / scheduler /
        # scaler already came from the stage-2 checkpoint.
        stage2_optimizer = optimizer

        stage2_scheduler = scheduler

        stage2_scaler = scaler

        stage2_start_epoch = (
            resume_state[
                "start_epoch"
            ]
        )

        stage2_counter = (
            resume_state[
                "early_stop_counter"
            ]
        )

        stage2_global_offset = (
            resume_state[
                "global_epoch"
            ]
            - (
                resume_state[
                    "start_epoch"
                ]
                - 1
            )
        )

    else:

        # ----------------------------------------------------
        # IMPORTANT:
        # Start stage 2 from the BEST stage-1 mAP checkpoint,
        # not from stage1_last.pt.
        # ----------------------------------------------------

        stage1_best_path = (
            weights_dir
            / "stage1_best.pt"
        )

        if not stage1_best_path.exists():

            stage1_best_path = (
                weights_dir
                / "best.pt"
            )

        load_weights_only(
            stage1_best_path,

            model,

            device,
        )

        # Reconfigure loss in case model mode/state changed.
        configure_model_loss(
            model,
            stage2_cfg,
        )

        stage2_optimizer = (
            build_optimizer(
                model,
                stage2_cfg,
            )
        )

        stage2_scheduler = (
            build_scheduler(
                stage2_optimizer,
                stage2_cfg,
            )
        )

        stage2_scaler = (
            build_scaler(
                amp,
                device,
            )
        )

        stage2_start_epoch = 0

        # Patience restarts at stage 2, while the metric
        # baseline remains the global stage-1 best.
        stage2_counter = 0

        stage2_global_offset = (
            stage1_last_global_epoch
            + 1
        )

    # ========================================================
    # Stage 2
    # ========================================================

    stage2_result = run_training_stage(
        stage_name="stage2",

        model=model,

        train_loader=train_loader,

        val_loader=val_loader,

        optimizer=stage2_optimizer,

        scheduler=stage2_scheduler,

        scaler=stage2_scaler,

        device=device,

        amp=amp,

        grad_clip=grad_clip,

        epochs=stage2_epochs,

        start_epoch=stage2_start_epoch,

        patience=stage2_patience,

        min_delta=stage2_min_delta,

        rgb_imgsz=rgb_imgsz,

        tir_imgsz=tir_imgsz,

        save_dir=save_dir,

        weights_dir=weights_dir,

        cfg=cfg,

        history=history,

        best_map5095=best_map5095,

        best_val_loss=best_val_loss,

        early_stop_counter=(
            stage2_counter
        ),

        global_epoch_offset=(
            stage2_global_offset
        ),
    )

    best_map5095 = (
        stage2_result[
            "best_map5095"
        ]
    )

    best_val_loss = (
        stage2_result[
            "best_val_loss"
        ]
    )

    # ========================================================
    # Final
    # ========================================================

    print(
        "\n"
        "============================================================"
    )

    print(
        "Two-stage training completed."
    )

    print(
        "============================================================"
    )

    print(
        f"Best mAP50-95 : "
        f"{best_map5095:.6f}"
    )

    print(
        f"Best val loss : "
        f"{best_val_loss:.6f}"
    )

    print(
        f"Best weight   : "
        f"{weights_dir / 'best.pt'}"
    )

    print(
        f"Last weight   : "
        f"{weights_dir / 'last.pt'}"
    )

    print(
        "============================================================"
    )


# ============================================================
# 19. CLI
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cfg",
        type=str,
        required=True,
        help="Experiment YAML",
    )

    args = parser.parse_args()

    config = load_config(
        args.cfg
    )

    train(
        config
    )
