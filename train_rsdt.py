"""
Full-feature RSD-T trainer for YOLO-tcy
=======================================

Design goal
-----------
Keep the original full-feature train_rgbt.py UNCHANGED and reuse its complete
training engine:

- reproducibility
- SGD / Adam / AdamW
- cosine LR
- AMP
- gradient clipping
- validation loss
- validation P / R / mAP50 / mAP75 / mAP50-95 every epoch
- best.pt selected by mAP50-95
- early stopping by mAP50-95
- two-stage training
- stage2 starts from stage1 best
- independent stage2 optimizer / scheduler / AMP scaler
- resume
- history.json
- best.pt / last.pt
- stage1_best.pt / stage1_last.pt
- stage2_best.pt / stage2_last.pt

Only RSD-T-specific pieces are replaced:
- dataset
- model
- batch preprocessing
- validation inference / GT coordinate handling

Original files are not modified.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from tqdm import tqdm
from ultralytics.utils.metrics import DetMetrics

import train_rgbt as base_train

from datasets.rsdt_dataset import (
    build_rsdt_dataset,
    build_rsdt_dataloader,
)

from models.rgbt_rsdt_model import (
    RGBTRSDTDetectionModel,
)

from val_rgbt import (
    postprocess_predictions,
    process_single_image,
)

# ---------------------------------------------------------------------------
# Active config is used by adapters called internally from base_train.train().
# ---------------------------------------------------------------------------

_ACTIVE_CFG: Dict = {}


# ===========================================================================
# 1. RSD-T batch preprocessing
# ===========================================================================

def preprocess_batch_rsdt(
    batch: Dict,
    device: torch.device,
):
    """
    Move RSD-T batch to device.

    Images:
        rgb_img              = high-resolution RGB
        rgb_semantic_img     = semantic RGB
        tir_img              = TIR

    uint8 [0,255] -> float32 [0,1]
    """

    for key in (
        "rgb_img",
        "rgb_semantic_img",
        "tir_img",
    ):
        batch[key] = (
            batch[key]
            .to(
                device,
                non_blocking=True,
            )
            .float()
            / 255.0
        )

    tensor_keys = (
        "rgb_cls",
        "rgb_bboxes",
        "rgb_semantic_bboxes",
        "rgb_batch_idx",
        "tir_cls",
        "tir_bboxes",
        "tir_batch_idx",
    )

    for key in tensor_keys:
        batch[key] = (
            batch[key]
            .to(
                device,
                non_blocking=True,
            )
        )

    return batch


# ===========================================================================
# 2. Semantic-coordinate GT
# ===========================================================================

def prepare_rsdt_ground_truth(
    batch: Dict,
    sample_index: int,
):
    """
    Final YOLO detection head runs in RGB semantic coordinates.

    Therefore validation GT must use:
        rgb_semantic_bboxes
        rgb_semantic_img.shape

    NOT the high-resolution rgb_bboxes/rgb_img shape.
    """

    from ultralytics.utils import ops

    mask = (
        batch["rgb_batch_idx"]
        == sample_index
    )

    cls = (
        batch["rgb_cls"][mask]
        .squeeze(-1)
    )

    boxes = (
        batch[
            "rgb_semantic_bboxes"
        ][mask]
    )

    h = int(
        batch[
            "rgb_semantic_img"
        ].shape[2]
    )

    w = int(
        batch[
            "rgb_semantic_img"
        ].shape[3]
    )

    if boxes.shape[0] > 0:
        boxes = ops.xywh2xyxy(
            boxes
        )

        scale = torch.tensor(
            [w, h, w, h],
            device=boxes.device,
            dtype=boxes.dtype,
        )

        boxes = (
            boxes
            * scale
        )

    return {
        "cls": cls,
        "bboxes": boxes,
    }


# ===========================================================================
# 3. Full training-time detection metrics
# ===========================================================================

@torch.inference_mode()
def validate_metrics_rsdt(
    model,
    loader,
    device,
    amp=True,
    conf_thres=0.001,
    iou_thres=0.7,
    max_det=300,
):
    """
    RSD-T version of the full train_rgbt.py validation-metrics routine.

    Returns values in [0,1]:
        precision
        recall
        map50
        map75
        map5095

    This function is used by the original two-stage training engine for:
        - epoch metrics
        - best.pt selection
        - early stopping
    """

    model.eval()

    names = model.names

    if not isinstance(
        names,
        dict,
    ):
        names = {
            i: name
            for i, name in enumerate(names)
        }

    names = {
        int(k): str(v)
        for k, v in names.items()
    }

    metrics = DetMetrics(
        names=names
    )

    iouv = torch.linspace(
        0.50,
        0.95,
        10,
        device=device,
    )

    seen = 0
    total_instances = 0

    print(
        (
            f"{'Class':>22}"
            f"{'Images':>11}"
            f"{'Instances':>11}"
            f"{'Box(P':>11}"
            f"{'R':>11}"
            f"{'mAP50':>11}"
            f"{'mAP50-95)':>13}"
        )
    )

    pbar = tqdm(
        loader,
        total=len(loader),
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

    for batch in pbar:

        batch_size_current = int(
            batch["rgb_img"].shape[0]
        )

        seen += batch_size_current

        total_instances += int(
            batch["rgb_cls"].shape[0]
        )

        batch = preprocess_batch_rsdt(
            batch,
            device,
        )

        with base_train.autocast_context(
            device,
            amp,
        ):
            preds = model(
                batch["rgb_img"],
                batch["tir_img"],
                rgb_semantic=(
                    batch[
                        "rgb_semantic_img"
                    ]
                ),
            )

        preds = postprocess_predictions(
            preds=preds,
            model=model,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            max_det=max_det,
        )

        for sample_index, pred in enumerate(
            preds
        ):

            gt = prepare_rsdt_ground_truth(
                batch,
                sample_index,
            )

            tp = process_single_image(
                pred=pred,
                gt=gt,
                iouv=iouv,
            )

            target_cls = (
                gt["cls"]
                .detach()
                .float()
                .cpu()
                .numpy()
            )

            if (
                pred["cls"].shape[0]
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
                    pred["conf"]
                    .detach()
                    .float()
                    .cpu()
                    .numpy()
                )

                pred_cls = (
                    pred["cls"]
                    .detach()
                    .float()
                    .cpu()
                    .numpy()
                )

            metrics.update_stats(
                {
                    "tp": tp,
                    "conf": pred_conf,
                    "pred_cls": pred_cls,
                    "target_cls": target_cls,
                    "target_img": (
                        np.unique(
                            target_cls
                        )
                    ),
                    "im_name": Path(
                        batch[
                            "rgb_path"
                        ][sample_index]
                    ).name,
                }
            )

        pbar.set_description(
            (
                f"{'all':>22}"
                f"{seen:>11}"
                f"{total_instances:>11}"
            ),
            refresh=False,
        )

    pbar.close()

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

    return {
        "precision": precision,
        "recall": recall,
        "map50": map50,
        "map75": map75,
        "map5095": map5095,
        "images": seen,
        "instances": total_instances,
    }


# ===========================================================================
# 4. Dataset adapter for the original full trainer
# ===========================================================================

def build_rsdt_dataset_adapter(
    rgb_yaml,
    tir_yaml,
    split="train",
    rgb_imgsz=640,
    tir_imgsz=640,
    pair_mode="relative",
    augment=False,
    fliplr=0.5,
    flipud=0.0,
    tir_channels=3,
    strict_pair=True,
    **kwargs,
):
    """
    The original trainer calls build_rgbt_dataset(rgb_imgsz=...).

    For RSD-T:
        rgb_imgsz -> high-resolution RGB
        rgb_semantic_imgsz -> read from RSD-T config
    """

    data_cfg = (
        _ACTIVE_CFG
        .get(
            "data",
            {},
        )
    )

    semantic_imgsz = int(
        data_cfg.get(
            "rgb_semantic_imgsz",
            640,
        )
    )

    return build_rsdt_dataset(
        rgb_yaml=rgb_yaml,
        tir_yaml=tir_yaml,
        split=split,
        rgb_high_imgsz=rgb_imgsz,
        rgb_semantic_imgsz=(
            semantic_imgsz
        ),
        tir_imgsz=tir_imgsz,
        pair_mode=pair_mode,
        augment=augment,
        fliplr=fliplr,
        flipud=flipud,
        tir_channels=tir_channels,
        strict_pair=strict_pair,
    )


# ===========================================================================
# 5. Model adapter for the original full trainer
# ===========================================================================

def build_rsdt_model_adapter(
    model_name="yolo26n",
    nc=1,
    pretrained=True,
    fusion="concat",
    fusion_indices=None,
    align_mode="bilinear",
    names=None,
    verbose=True,
    **kwargs,
):
    model_cfg = (
        _ACTIVE_CFG
        .get(
            "model",
            {},
        )
    )

    data_cfg = (
        _ACTIVE_CFG
        .get(
            "data",
            {},
        )
    )

    return RGBTRSDTDetectionModel(
        model_name=model_name,
        nc=nc,
        pretrained=pretrained,
        fusion=fusion,
        fusion_indices=fusion_indices,
        align_mode=align_mode,
        names=names,

        semantic_imgsz=int(
            data_cfg.get(
                "rgb_semantic_imgsz",
                640,
            )
        ),

        rsdt_scales=model_cfg.get(
            "rsdt_scales",
            [3],
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

        alignment_cfg=model_cfg.get(
            "alignment",
            {},
        ),

        verbose=verbose,
    )


# ===========================================================================
# 6. Optional training progress wrapper
# ===========================================================================

_ORIGINAL_TRAIN_ONE_EPOCH = (
    base_train.train_one_epoch
)


def train_one_epoch_rsdt(
    *args,
    **kwargs,
):
    """
    Keep the complete original train-one-epoch logic.
    Only the displayed resolution string is handled by the original
    high-RGB/TIR fields. gamma/alpha are printed after every epoch by
    the stage wrapper below.
    """

    return _ORIGINAL_TRAIN_ONE_EPOCH(
        *args,
        **kwargs,
    )


# ===========================================================================
# 7. Stage wrapper: keep original behavior + print RSD-T scalar state
# ===========================================================================

_ORIGINAL_RUN_TRAINING_STAGE = (
    base_train.run_training_stage
)


def run_training_stage_rsdt(
    *args,
    **kwargs,
):
    """
    Delegates the COMPLETE stage implementation to the old trainer:
        best mAP selection
        early stopping
        history
        checkpoints
        two-stage bookkeeping

    Adds final RSD-T scalar reporting.
    """

    result = (
        _ORIGINAL_RUN_TRAINING_STAGE(
            *args,
            **kwargs,
        )
    )

    model = kwargs.get(
        "model",
        None,
    )

    if (
        model is not None
        and hasattr(
            model,
            "rsdt",
        )
    ):
        state = (
            model.rsdt.scalar_state()
        )

        print(
            "\nRSD-T state:"
        )

        print(
            f"  alpha = "
            f"{state['alpha']:.6f}"
        )

        print(
            f"  gamma = "
            f"{state['gamma']:.6f}"
        )

    return result


# ===========================================================================
# 8. Install adapters into the imported full trainer
# ===========================================================================

def install_rsdt_adapters(
    cfg: Dict,
):
    """
    Patches only the in-memory imported train_rgbt module.

    NO source file is changed on disk.
    Starting a normal:
        python train_rgbt.py ...
    process remains completely unaffected.
    """

    global _ACTIVE_CFG

    _ACTIVE_CFG = cfg

    # Allow either spelling in experiment YAML.
    data_cfg = cfg.setdefault(
        "data",
        {},
    )

    if (
        "rgb_imgsz"
        not in data_cfg
    ):
        data_cfg[
            "rgb_imgsz"
        ] = int(
            data_cfg.get(
                "rgb_high_imgsz",
                1280,
            )
        )

    data_cfg.setdefault(
        "rgb_semantic_imgsz",
        640,
    )

    # RSD-T dataset / loader.
    base_train.build_rgbt_dataset = (
        build_rsdt_dataset_adapter
    )

    base_train.build_rgbt_dataloader = (
        build_rsdt_dataloader
    )

    # RSD-T model.
    base_train.RGBTDetectionModel = (
        build_rsdt_model_adapter
    )

    # RSD-T batch structure.
    base_train.preprocess_batch = (
        preprocess_batch_rsdt
    )

    # Full train epoch remains original.
    base_train.train_one_epoch = (
        train_one_epoch_rsdt
    )

    # Training-time metrics need 3-input inference
    # and semantic-coordinate GT.
    base_train.validate_metrics = (
        validate_metrics_rsdt
    )

    # Keep complete original stage implementation,
    # wrapped only for RSD-T state reporting.
    base_train.run_training_stage = (
        run_training_stage_rsdt
    )


# ===========================================================================
# 9. Main
# ===========================================================================

def train(
    cfg: Dict,
):
    """
    Use the original FULL train_rgbt.train() after installing in-memory
    RSD-T adapters.
    """

    install_rsdt_adapters(
        cfg
    )

    print(
        "\n"
        "============================================================"
    )
    print(
        "RSD-T full-feature training"
    )
    print(
        "Base engine     : train_rgbt.py"
    )
    print(
        "Original files  : unchanged"
    )
    print(
        f"RGB high        : "
        f"{cfg['data']['rgb_imgsz']}"
    )
    print(
        f"RGB semantic    : "
        f"{cfg['data']['rgb_semantic_imgsz']}"
    )
    print(
        f"RSD-T scales    : "
        f"{cfg.get('model', {}).get('rsdt_scales', [3])}"
    )

    alignment_cfg = (
        cfg.get(
            "model",
            {},
        ).get(
            "alignment",
            {},
        )
    )

    print(
        f"Alignment       : "
        f"enabled={alignment_cfg.get('enabled', False)}, "
        f"type={alignment_cfg.get('type', 'identity')}"
    )

    print(
        f"TIR             : "
        f"{cfg['data'].get('tir_imgsz', 640)}"
    )
    print(
        "============================================================\n"
    )

    return base_train.train(
        cfg
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Full-feature isolated RSD-T training"
        )
    )

    parser.add_argument(
        "--cfg",
        type=str,
        required=True,
        help="RSD-T experiment YAML",
    )

    args = parser.parse_args()

    config = (
        base_train.load_config(
            args.cfg
        )
    )

    train(
        config
    )
