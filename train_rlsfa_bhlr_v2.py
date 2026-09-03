"""
Training entry for RLSFA + BHLR-v2.

This file is intentionally separate from train_rlsfa_bhlr.py so that:
- BHLR-v1 experiments remain reproducible.
- BHLR-v2 has an independent model import and run namespace.
- The existing RSD-T dataset/preprocessing/two-stage trainer is still reused.
"""

from __future__ import annotations

import argparse
from typing import Dict

import train_rsdt as rsdt_train
from models.rgbt_rlsfa_bhlr_model_v2 import RGBTRLSFABHLRV2DetectionModel


_ACTIVE_CFG: Dict = {}


def build_rlsfa_bhlr_v2_model_adapter(
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
    model_cfg = _ACTIVE_CFG.get("model", {})
    data_cfg = _ACTIVE_CFG.get("data", {})
    loss_cfg = _ACTIVE_CFG.get("loss", {})

    return RGBTRLSFABHLRV2DetectionModel(
        model_name=model_name,
        nc=nc,
        pretrained=pretrained,
        fusion="concat",
        fusion_indices=fusion_indices,
        align_mode=align_mode,
        names=names,
        semantic_imgsz=int(data_cfg.get("rgb_semantic_imgsz", 640)),
        bhlr_scales=model_cfg.get("bhlr_scales", [3]),
        high_to_semantic_ratio=int(model_cfg.get("high_to_semantic_ratio", 2)),

        # BHLR-v2 lightweight defaults.
        detail_channels=int(model_cfg.get("detail_channels", 48)),
        stem_channels=int(model_cfg.get("stem_channels", 16)),
        guide_channels=int(model_cfg.get("guide_channels", 24)),

        # Kept only for constructor/backward-config compatibility.
        # BHLR-v2 itself no longer uses a global FFT boundary branch.
        bhlr_freq_cutoff=float(model_cfg.get("bhlr_freq_cutoff", 0.15)),
        bhlr_freq_sharpness=float(model_cfg.get("bhlr_freq_sharpness", 24.0)),

        alignment_cfg=dict(model_cfg.get("alignment", {})),
        alignment_loss_cfg=dict(loss_cfg.get("alignment", {}) or {}),
        verbose=verbose,
    )


def train_one_epoch_rlsfa_bhlr_v2(*args, **kwargs):
    epoch_loss, running = rsdt_train._ORIGINAL_TRAIN_ONE_EPOCH(*args, **kwargs)

    if isinstance(running, dict):
        keys = (
            "align_loss",
            "align_coarse_loss",
            "align_final_loss",
            "align_targetness_loss",
            "align_fine_reg",
            "align_weighted_loss",
            "align_valid_pairs",
        )

        if any(k in running for k in keys):
            print("\nRLSFA alignment-loss state:")
            for key in keys:
                if key not in running:
                    continue

                value = running[key]
                if key == "align_valid_pairs":
                    print(f"  {key:<24} = {value:.2f}")
                else:
                    print(f"  {key:<24} = {value:.6f}")

    return epoch_loss, running


def run_training_stage_rlsfa_bhlr_v2(*args, **kwargs):
    result = rsdt_train._ORIGINAL_RUN_TRAINING_STAGE(*args, **kwargs)

    model = kwargs.get("model", None)
    if model is not None and hasattr(model, "bhlr"):
        state = model.bhlr.scalar_state()

        print("\nBHLR-v2 state:")
        for key, value in state.items():
            if key.startswith("gamma"):
                print(f"  {key} = {value:.6f}")

    return result


def install_rlsfa_bhlr_v2_adapters(cfg: Dict):
    global _ACTIVE_CFG
    _ACTIVE_CFG = cfg

    # Reuse the current dual-resolution RGB-T dataset/preprocessing logic.
    rsdt_train.install_rsdt_adapters(cfg)

    # Replace only the in-memory constructor used by the shared trainer.
    rsdt_train.base_train.RGBTDetectionModel = build_rlsfa_bhlr_v2_model_adapter
    rsdt_train.base_train.train_one_epoch = train_one_epoch_rlsfa_bhlr_v2
    rsdt_train.base_train.run_training_stage = run_training_stage_rlsfa_bhlr_v2


def train(cfg: Dict):
    install_rlsfa_bhlr_v2_adapters(cfg)

    model_cfg = cfg.get("model", {})
    align_cfg = model_cfg.get("alignment", {})
    align_loss = (cfg.get("loss", {}).get("alignment", {}) or {})

    print("\n============================================================")
    print("RLSFA + BHLR-v2 training")
    print("============================================================")
    print(f"Alignment enabled : {align_cfg.get('enabled', False)}")
    print(f"Alignment type    : {align_cfg.get('type', 'identity')}")
    print("Geometry          : coarse translation + fine residual translation")
    print(f"RLSFA scales      : {align_cfg.get('scales', [3])}")
    print(f"BHLR-v2 scales    : {model_cfg.get('bhlr_scales', [3])}")
    print(f"BHLR guidance     : {align_cfg.get('bhlr_guidance_mode', 'auto')}")
    print(f"RGB high          : {cfg['data'].get('rgb_imgsz', 1280)}")
    print(f"RGB semantic      : {cfg['data'].get('rgb_semantic_imgsz', 640)}")
    print(f"TIR               : {cfg['data'].get('tir_imgsz', 640)}")
    print(f"Detail channels   : {model_cfg.get('detail_channels', 48)}")
    print(f"Stem channels     : {model_cfg.get('stem_channels', 16)}")
    print(f"Guide channels    : {model_cfg.get('guide_channels', 24)}")
    print(f"Alignment loss    : {align_loss.get('enabled', False)}")
    print("Fusion            : concat")
    print("============================================================\n")

    return rsdt_train.base_train.train(cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RLSFA + BHLR-v2 RGB-T training"
    )
    parser.add_argument("--cfg", type=str, required=True)
    args = parser.parse_args()

    cfg = rsdt_train.base_train.load_config(args.cfg)
    train(cfg)
