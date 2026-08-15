"""
Train RGB-T Dual-Backbone YOLO
==============================

Architecture:

RGB -> Backbone-R ---\
                      -> Feature Fusion -> YOLO Neck -> Detect -> RGB GT
TIR -> Backbone-T ---/

Current baseline:
    - Two independent YOLO backbones
    - Both initialized from pretrained YOLO weights
    - RGB/TIR can use different input resolutions
    - P3/P4/P5 feature-level fusion
    - Main detection loss uses RGB labels
    - TIR labels are retained for future auxiliary supervision

Project:
    /mnt/sda/taochangyong/Projects/Model/YOLO-tcy

    训练
    conda activate tcy
    cd /mnt/sda/taochangyong/Projects/Model/YOLO-tcy
    python train_rgbt.py \
    --cfg configs/experiments/lrdd_rgbt_yolo26n_1280.yaml
"""
from __future__ import annotations

import argparse
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

    best_val_loss: float,

    cfg: Dict,
):
    """
    Save custom RGB-T checkpoint.

    NOTE:
    This is a YOLO-tcy custom checkpoint.
    Load it using RGBTDetectionModel,
    NOT directly using YOLO("best.pt").
    """

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = {
        "epoch":
            epoch,

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "scheduler_state_dict":
            scheduler.state_dict(),

        "scaler_state_dict":
            scaler.state_dict(),

        "best_val_loss":
            best_val_loss,

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
# 11. Load checkpoint
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
    Resume training.
    """

    path = Path(path)

    if not path.exists():

        raise FileNotFoundError(
            f"Checkpoint 不存在:\n{path}"
        )

    print(
        f"\nResume checkpoint:\n{path}"
    )

    ckpt = torch.load(
        path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        ckpt["model_state_dict"],
        strict=True,
    )

    optimizer.load_state_dict(
        ckpt[
            "optimizer_state_dict"
        ]
    )

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

    start_epoch = (
        int(
            ckpt["epoch"]
        )
        + 1
    )

    best_val_loss = float(
        ckpt.get(
            "best_val_loss",
            float("inf"),
        )
    )

    return (
        start_epoch,
        best_val_loss,
    )

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
# 14. Main training
# ============================================================

def train(cfg: Dict):

    # ========================================================
    # Basic settings
    # ========================================================

    seed = int(
        cfg["train"].get(
            "seed",
            0,
        )
    )

    deterministic = bool(
        cfg["train"].get(
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
            cfg["train"].get(
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
        "RGB-T YOLO Training"
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
        cfg["output"].get(
            "project",
            ROOT / "runs/rgbt",
        )
    )

    if not project.is_absolute():
        project = ROOT / project

    run_name = cfg[
        "output"
    ].get(
        "name",
        "rgbt_exp",
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
    # Train dataset
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
                cfg["augment"].get(
                    "fliplr",
                    0.5,
                )
            ),

            flipud=float(
                cfg["augment"].get(
                    "flipud",
                    0.0,
                )
            ),

            tir_channels=3,

            strict_pair=True,
        )
    )

    # ========================================================
    # Validation dataset
    # ========================================================

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
        cfg["train"].get(
            "batch",
            8,
        )
    )

    workers = int(
        cfg["train"].get(
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
    # Optimizer
    # ========================================================

    optimizer = build_optimizer(
        model,
        cfg,
    )

    scheduler = build_scheduler(
        optimizer,
        cfg,
    )

    # ========================================================
    # AMP
    # ========================================================

    amp = bool(
        cfg["train"].get(
            "amp",
            True,
        )
    )

    scaler = build_scaler(
        amp,
        device,
    )

    # ========================================================
    # Resume
    # ========================================================

    start_epoch = 0

    best_val_loss = float(
        "inf"
    )

    resume = cfg[
        "train"
    ].get(
        "resume",
        None,
    )

    if resume:

        (
            start_epoch,
            best_val_loss,
        ) = resume_checkpoint(
            resume,

            model,

            optimizer,

            scheduler,

            scaler,

            device,
        )

    # ========================================================
    # Training parameters
    # ========================================================

    epochs = int(
        cfg["train"].get(
            "epochs",
            300,
        )
    )

    grad_clip = float(
        cfg["train"].get(
            "grad_clip",
            10.0,
        )
    )

    print(
        f"Epochs        : {epochs}"
    )

    print(
        f"Batch         : {batch_size}"
    )

    print(
        f"Workers       : {workers}"
    )

    print(
        f"Optimizer     : "
        f"{cfg['train'].get('optimizer', 'SGD')}"
    )

    print(
        f"AMP           : {amp}"
    )

    print(
        f"Save dir      : {save_dir}"
    )

    print(
        "============================================================\n"
    )

    # ========================================================
    # Training loop
    # ========================================================

    history = []

    for epoch in range(
        start_epoch,
        epochs,
    ):

        print(
            "\n"
            "############################################################"
        )

        print(
            f"Epoch "
            f"{epoch + 1}/{epochs}"
        )

        print(
            "############################################################"
        )

        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

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

            epoch=epoch,

            epochs=epochs,

            rgb_imgsz=rgb_imgsz,

            tir_imgsz=tir_imgsz,
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        val_loss = validate_loss(
            model=model,

            loader=val_loader,

            device=device,

            amp=amp,

            epoch=epoch,

            epochs=epochs,
        )

        # ----------------------------------------------------
        # LR
        # ----------------------------------------------------

        scheduler.step()

        current_lr = (
            optimizer.param_groups[
                0
            ]["lr"]
        )

        print(
            "\n"
            f"Epoch {epoch + 1} finished:"
        )

        (
            train_box_loss,
            train_cls_loss,
            train_dfl_loss,
        ) = get_standard_losses(
            train_loss_items
        )

        print(
            f"  train_loss = "
            f"{train_loss:.6f}"
        )

        print(
            f"  box_loss   = "
            f"{train_box_loss:.6f}"
        )

        print(
            f"  cls_loss   = "
            f"{train_cls_loss:.6f}"
        )

        print(
            f"  dfl_loss   = "
            f"{train_dfl_loss:.6f}"
        )

        print(
            f"  val_loss   = "
            f"{val_loss:.6f}"
        )

        print(
            f"  lr         = "
            f"{current_lr:.8f}"
        )

        print(
            f"  val_loss   = "
            f"{val_loss:.6f}"
        )

        print(
            f"  lr         = "
            f"{current_lr:.8f}"
        )

        # ----------------------------------------------------
        # History
        # ----------------------------------------------------

        record = {
            "epoch":
                epoch + 1,

            "train_loss":
                train_loss,

            "box_loss":
                train_box_loss,

            "cls_loss":
                train_cls_loss,

            "dfl_loss":
                train_dfl_loss,

            "val_loss":
                val_loss,

            "lr":
                current_lr,
        }

        history.append(
            record
        )

        with (
            save_dir
            / "history.json"
        ).open(
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                history,
                f,
                indent=2,
            )

        # ----------------------------------------------------
        # last.pt
        # ----------------------------------------------------

        save_checkpoint(
            weights_dir
            / "last.pt",

            model,

            optimizer,

            scheduler,

            scaler,

            epoch,

            best_val_loss,

            cfg,
        )

        # ----------------------------------------------------
        # best.pt
        # ----------------------------------------------------

        if val_loss < best_val_loss:

            best_val_loss = (
                val_loss
            )

            save_checkpoint(
                weights_dir
                / "best.pt",

                model,

                optimizer,

                scheduler,

                scaler,

                epoch,

                best_val_loss,

                cfg,
            )

            print(
                f"  [BEST] "
                f"val_loss={best_val_loss:.6f}"
            )

    print(
        "\n"
        "============================================================"
    )

    print(
        "Training completed."
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
        "============================================================"
    )


# ============================================================
# 15. CLI
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
