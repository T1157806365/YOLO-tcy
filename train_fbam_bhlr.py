"""
Training entry for FBAM + BHLR + explicit bbox-guided alignment supervision.

Unchanged on disk:
- train_rgbt.py
- train_rsdt.py
- datasets/rsdt_dataset.py
- models/rgbt_model.py
- models/modules/alignment/fbam.py
- models/modules/bhlr_v1.py

The existing RSD-T data/preprocess/validation adapters are reused.
"""

from __future__ import annotations

import argparse
from typing import Dict

import train_rsdt as rsdt_train

from models.rgbt_fbam_bhlr_model import (
    RGBTFBAMBHLRDetectionModel,
)


_ACTIVE_CFG: Dict = {}


def build_fbam_bhlr_model_adapter(
    model_name="yolo11m",
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
        _ACTIVE_CFG.get(
            "model",
            {},
        )
    )

    data_cfg = (
        _ACTIVE_CFG.get(
            "data",
            {},
        )
    )

    loss_cfg = (
        _ACTIVE_CFG.get(
            "loss",
            {},
        )
    )

    alignment_cfg = dict(
        model_cfg.get(
            "alignment",
            {},
        )
    )

    alignment_loss_cfg = dict(
        loss_cfg.get(
            "alignment",
            {},
        )
        or {}
    )

    return RGBTFBAMBHLRDetectionModel(
        model_name=model_name,
        nc=nc,
        pretrained=pretrained,
        fusion="concat",
        fusion_indices=fusion_indices,
        align_mode=align_mode,
        names=names,
        semantic_imgsz=int(
            data_cfg.get(
                "rgb_semantic_imgsz",
                640,
            )
        ),
        bhlr_scales=model_cfg.get(
            "bhlr_scales",
            [3],
        ),
        high_to_semantic_ratio=int(
            model_cfg.get(
                "high_to_semantic_ratio",
                3,
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
                32,
            )
        ),
        guide_channels=int(
            model_cfg.get(
                "guide_channels",
                32,
            )
        ),
        bhlr_freq_cutoff=float(
            model_cfg.get(
                "bhlr_freq_cutoff",
                0.15,
            )
        ),
        bhlr_freq_sharpness=float(
            model_cfg.get(
                "bhlr_freq_sharpness",
                24.0,
            )
        ),
        alignment_cfg=alignment_cfg,
        alignment_loss_cfg=(
            alignment_loss_cfg
        ),
        verbose=verbose,
    )


def train_one_epoch_fbam_bhlr(
    *args,
    **kwargs,
):
    """
    Keep the complete original train_rgbt.py epoch implementation.

    The model now returns alignment loss terms in loss_items, so the original
    trainer automatically averages them. This wrapper only prints them after
    each epoch; the optimization loop itself remains unchanged.
    """
    (
        epoch_loss,
        running_loss_items,
    ) = (
        rsdt_train
        ._ORIGINAL_TRAIN_ONE_EPOCH(
            *args,
            **kwargs,
        )
    )

    if isinstance(
        running_loss_items,
        dict,
    ):
        alignment_keys = (
            "align_loss",
            "align_coarse_loss",
            "align_final_loss",
            "align_weighted_loss",
            "align_valid_pairs",
        )

        if any(
            key
            in running_loss_items
            for key
            in alignment_keys
        ):
            print(
                "\nFBAM alignment-loss state:"
            )

            for key in alignment_keys:
                if key not in running_loss_items:
                    continue

                value = (
                    running_loss_items[
                        key
                    ]
                )

                if (
                    key
                    == "align_valid_pairs"
                ):
                    print(
                        f"  {key:<22} = "
                        f"{value:.2f}"
                    )
                else:
                    print(
                        f"  {key:<22} = "
                        f"{value:.6f}"
                    )

    return (
        epoch_loss,
        running_loss_items,
    )


def run_training_stage_bhlr(
    *args,
    **kwargs,
):
    result = (
        rsdt_train
        ._ORIGINAL_RUN_TRAINING_STAGE(
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
            "bhlr",
        )
    ):
        state = (
            model
            .bhlr
            .scalar_state()
        )

        print(
            "\nBHLR state:"
        )

        for key, value in (
            state.items()
        ):
            if key.startswith(
                "gamma"
            ):
                print(
                    f"  {key} = "
                    f"{value:.6f}"
                )

    return result


def install_fbam_bhlr_adapters(
    cfg: Dict,
):
    global _ACTIVE_CFG

    _ACTIVE_CFG = cfg

    # Install existing:
    # high-RGB / semantic-RGB / TIR dataset,
    # preprocess,
    # semantic-RGB validation path.
    rsdt_train.install_rsdt_adapters(
        cfg
    )

    # Replace model construction with FBAM+BHLR model.
    rsdt_train.base_train.RGBTDetectionModel = (
        build_fbam_bhlr_model_adapter
    )

    # Keep original epoch implementation, wrapped only for
    # alignment-loss reporting.
    rsdt_train.base_train.train_one_epoch = (
        train_one_epoch_fbam_bhlr
    )

    # Keep original stage behavior + BHLR status reporting.
    rsdt_train.base_train.run_training_stage = (
        run_training_stage_bhlr
    )


def train(
    cfg: Dict,
):
    install_fbam_bhlr_adapters(
        cfg
    )

    model_cfg = cfg.get(
        "model",
        {},
    )

    align_cfg = model_cfg.get(
        "alignment",
        {},
    )

    loss_cfg = cfg.get(
        "loss",
        {},
    )

    align_loss_cfg = (
        loss_cfg.get(
            "alignment",
            {},
        )
        or {}
    )

    print(
        "\n"
        "============================================================"
    )
    print(
        "FBAM + BHLR training"
    )
    print(
        "Base engine      : train_rgbt.py (unchanged)"
    )
    print(
        "Dataset          : rsdt_dataset.py (unchanged)"
    )
    print(
        f"FBAM scales      : "
        f"{align_cfg.get('scales', [3])}"
    )
    print(
        f"BHLR scales      : "
        f"{model_cfg.get('bhlr_scales', [3])}"
    )
    print(
        "Fusion           : concat (Module-3 OFF)"
    )
    print(
        f"RGB high         : "
        f"{cfg['data'].get('rgb_imgsz', 1920)}"
    )
    print(
        f"RGB semantic     : "
        f"{cfg['data'].get('rgb_semantic_imgsz', 640)}"
    )
    print(
        f"TIR              : "
        f"{cfg['data'].get('tir_imgsz', 640)}"
    )
    print(
        "Alignment loss   : "
        f"{align_loss_cfg.get('enabled', False)}"
    )

    if align_loss_cfg.get(
        "enabled",
        False,
    ):
        print(
            f"  lambda_align   : "
            f"{align_loss_cfg.get('lambda_align', 0.2)}"
        )
        print(
            f"  lambda_coarse  : "
            f"{align_loss_cfg.get('lambda_coarse', 0.5)}"
        )
        print(
            f"  loss scales    : "
            f"{align_loss_cfg.get('scales', align_cfg.get('scales', [3]))}"
        )
        print(
            "  supervision    : "
            "RGB/TIR bbox-guided sampling offsets"
        )

    print(
        "============================================================\n"
    )

    return (
        rsdt_train
        .base_train
        .train(
            cfg
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "FBAM + BHLR RGB-T training "
            "with explicit alignment supervision"
        )
    )

    parser.add_argument(
        "--cfg",
        type=str,
        required=True,
    )

    args = parser.parse_args()

    config = (
        rsdt_train
        .base_train
        .load_config(
            args.cfg
        )
    )

    train(
        config
    )
