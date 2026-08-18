"""
Train isolated RGB-T + RSD-T v1
===============================

Original files remain untouched:
    train_rgbt.py
    datasets/rgbt_dataset.py
    models/rgbt_model.py

This script only imports reusable utilities from train_rgbt.py.

Run:
    python train_rsdt.py \
        --cfg configs/experiments/uavcb_rsdt_yolo26n_1280_640.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import torch
from tqdm import tqdm

from datasets.rsdt_dataset import (
    build_rsdt_dataset,
    build_rsdt_dataloader,
)

from models.rgbt_rsdt_model import (
    RGBTRSDTDetectionModel,
)

# Reuse stable utilities from the original trainer.
from train_rgbt import (
    ROOT,
    set_seed,
    load_config,
    save_config,
    select_device,
    build_optimizer,
    build_scheduler,
    build_scaler,
    autocast_context,
    configure_model_loss,
    save_checkpoint,
    resume_checkpoint,
    get_gpu_memory_gb,
    normalize_loss_items,
    update_running_losses,
    get_standard_losses,
)


def preprocess_batch_rsdt(
    batch: Dict,
    device: torch.device,
):
    """
    Move RSD-T batch to GPU and normalize images.
    """

    for key in [
        "rgb_img",
        "rgb_semantic_img",
        "tir_img",
    ]:

        batch[key] = (
            batch[key]
            .to(
                device,
                non_blocking=True,
            )
            .float()
            / 255.0
        )

    for key in [
        "rgb_cls",
        "rgb_bboxes",
        "rgb_semantic_bboxes",
        "rgb_batch_idx",

        "tir_cls",
        "tir_bboxes",
        "tir_batch_idx",
    ]:

        batch[key] = (
            batch[key]
            .to(
                device,
                non_blocking=True,
            )
        )

    return batch


def train_one_epoch_rsdt(
    model,
    loader,
    optimizer,
    scaler,
    device,
    amp=True,
    grad_clip=10.0,
    epoch=0,
    epochs=1,
    rgb_high_imgsz=1280,
    rgb_semantic_imgsz=640,
    tir_imgsz=640,
):
    model.train()

    running_total_loss = 0.0
    running_loss_items = {}

    num_batches = len(
        loader
    )

    print(
        (
            f"{'Epoch':>11}"
            f"{'GPU_mem':>11}"
            f"{'box_loss':>11}"
            f"{'cls_loss':>11}"
            f"{'dfl_loss':>11}"
            f"{'Instances':>11}"
            f"{'Size':>19}"
        )
    )

    pbar = tqdm(
        enumerate(loader),
        total=num_batches,
        dynamic_ncols=True,
        leave=True,
    )

    for batch_i, batch in pbar:

        batch = preprocess_batch_rsdt(
            batch,
            device,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        with autocast_context(
            device,
            amp,
        ):

            loss_raw, loss_items = model(
                batch
            )

            loss = loss_raw.sum()

        scaler.scale(
            loss
        ).backward()

        if grad_clip > 0:

            scaler.unscale_(
                optimizer
            )

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=grad_clip,
            )

        scaler.step(
            optimizer
        )

        scaler.update()

        loss_value = float(
            loss.detach().item()
        )

        running_total_loss += (
            loss_value
        )

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

        instances = int(
            batch[
                "rgb_cls"
            ].shape[0]
        )

        gpu_mem = get_gpu_memory_gb(
            device
        )

        size_str = (
            f"{rgb_high_imgsz}/"
            f"{rgb_semantic_imgsz}/"
            f"{tir_imgsz}"
        )

        state = (
            model.rsdt.scalar_state()
        )

        description = (
            f"{epoch + 1:>5}/{epochs:<5}"
            f"{gpu_mem:>10.3g}G"
            f"{box_loss:>11.4g}"
            f"{cls_loss:>11.4g}"
            f"{dfl_loss:>11.4g}"
            f"{instances:>11}"
            f"{size_str:>19}"
            f" g={state['gamma']:+.3f}"
        )

        pbar.set_description(
            description,
            refresh=False,
        )

    pbar.close()

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


@torch.no_grad()
def validate_loss_rsdt(
    model,
    loader,
    device,
    amp=True,
    epoch=0,
    epochs=1,
):
    """
    Kept behavior-compatible with the original train_rgbt.py:
    temporarily uses training-format raw predictions for YOLO criterion.
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

    for batch_i, batch in enumerate(
        pbar
    ):

        batch = preprocess_batch_rsdt(
            batch,
            device,
        )

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

        running_loss += (
            loss_value
        )

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

    model_cfg = cfg[
        "model"
    ]

    data_cfg = cfg[
        "data"
    ]

    # ========================================================
    # RSD-T resolutions
    # ========================================================

    rgb_high_imgsz = int(
        data_cfg.get(
            "rgb_high_imgsz",
            1280,
        )
    )

    rgb_semantic_imgsz = int(
        data_cfg.get(
            "rgb_semantic_imgsz",
            640,
        )
    )

    tir_imgsz = int(
        data_cfg.get(
            "tir_imgsz",
            640,
        )
    )

    rgb_yaml = data_cfg[
        "rgb"
    ]

    tir_yaml = data_cfg[
        "tir"
    ]

    pair_mode = data_cfg.get(
        "pair_mode",
        "relative",
    )

    print(
        "\n"
        "============================================================"
    )

    print(
        "RGB-T + RSD-T v1 Training"
    )

    print(
        "============================================================"
    )

    print(
        f"Device          : {device}"
    )

    print(
        f"RGB high        : {rgb_high_imgsz}"
    )

    print(
        f"RGB semantic    : {rgb_semantic_imgsz}"
    )

    print(
        f"TIR             : {tir_imgsz}"
    )

    print(
        f"Guidance        : "
        f"{model_cfg.get('use_guidance', True)}"
    )

    # ========================================================
    # Output
    # ========================================================

    project = Path(
        cfg[
            "output"
        ].get(
            "project",
            ROOT / "runs/rsdt",
        )
    )

    if not project.is_absolute():
        project = (
            ROOT / project
        )

    run_name = cfg[
        "output"
    ].get(
        "name",
        "rsdt_exp",
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
    # Datasets
    # ========================================================

    augment_cfg = cfg.get(
        "augment",
        {},
    )

    train_dataset = build_rsdt_dataset(
        rgb_yaml=rgb_yaml,
        tir_yaml=tir_yaml,
        split="train",
        rgb_high_imgsz=rgb_high_imgsz,
        rgb_semantic_imgsz=(
            rgb_semantic_imgsz
        ),
        tir_imgsz=tir_imgsz,
        pair_mode=pair_mode,
        augment=True,
        fliplr=float(
            augment_cfg.get(
                "fliplr",
                0.5,
            )
        ),
        flipud=float(
            augment_cfg.get(
                "flipud",
                0.0,
            )
        ),
        tir_channels=3,
        strict_pair=True,
    )

    val_dataset = build_rsdt_dataset(
        rgb_yaml=rgb_yaml,
        tir_yaml=tir_yaml,
        split="val",
        rgb_high_imgsz=rgb_high_imgsz,
        rgb_semantic_imgsz=(
            rgb_semantic_imgsz
        ),
        tir_imgsz=tir_imgsz,
        pair_mode=pair_mode,
        augment=False,
        tir_channels=3,
        strict_pair=True,
    )

    batch_size = int(
        cfg[
            "train"
        ].get(
            "batch",
            4,
        )
    )

    workers = int(
        cfg[
            "train"
        ].get(
            "workers",
            4,
        )
    )

    train_loader = (
        build_rsdt_dataloader(
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
        build_rsdt_dataloader(
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

    model = RGBTRSDTDetectionModel(
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

        semantic_imgsz=(
            rgb_semantic_imgsz
        ),

        use_guidance=bool(
            model_cfg.get(
                "use_guidance",
                True,
            )
        ),

        detail_channels=int(
            model_cfg.get(
                "detail_channels",
                64,
            )
        ),

        stem_channels=int(
            model_cfg.get(
                "stem_channels",
                24,
            )
        ),

        guide_channels=int(
            model_cfg.get(
                "guide_channels",
                32,
            )
        ),

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
    # Optimizer / scheduler / AMP
    # ========================================================

    optimizer = build_optimizer(
        model,
        cfg,
    )

    scheduler = build_scheduler(
        optimizer,
        cfg,
    )

    amp = bool(
        cfg[
            "train"
        ].get(
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
    # Loop
    # ========================================================

    epochs = int(
        cfg[
            "train"
        ].get(
            "epochs",
            300,
        )
    )

    grad_clip = float(
        cfg[
            "train"
        ].get(
            "grad_clip",
            10.0,
        )
    )

    history = []

    for epoch in range(
        start_epoch,
        epochs,
    ):

        (
            train_loss,
            train_loss_items,
        ) = train_one_epoch_rsdt(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            amp=amp,
            grad_clip=grad_clip,
            epoch=epoch,
            epochs=epochs,
            rgb_high_imgsz=(
                rgb_high_imgsz
            ),
            rgb_semantic_imgsz=(
                rgb_semantic_imgsz
            ),
            tir_imgsz=tir_imgsz,
        )

        val_loss = validate_loss_rsdt(
            model=model,
            loader=val_loader,
            device=device,
            amp=amp,
            epoch=epoch,
            epochs=epochs,
        )

        scheduler.step()

        current_lr = (
            optimizer.param_groups[
                0
            ]["lr"]
        )

        (
            train_box_loss,
            train_cls_loss,
            train_dfl_loss,
        ) = get_standard_losses(
            train_loss_items
        )

        rsdt_state = (
            model.rsdt.scalar_state()
        )

        print(
            f"\nEpoch {epoch + 1}:"
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
            f"  alpha      = "
            f"{rsdt_state['alpha']:.6f}"
        )

        print(
            f"  gamma      = "
            f"{rsdt_state['gamma']:.6f}"
        )

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

            "rsdt_alpha":
                rsdt_state[
                    "alpha"
                ],

            "rsdt_gamma":
                rsdt_state[
                    "gamma"
                ],
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
                f"val_loss="
                f"{best_val_loss:.6f}"
            )

    print(
        "\nTraining completed."
    )

    print(
        f"Save dir: {save_dir}"
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cfg",
        type=str,
        required=True,
    )

    args = parser.parse_args()

    cfg = load_config(
        args.cfg
    )

    train(
        cfg
    )


if __name__ == "__main__":
    main()
