"""
Training entry for RLSFA + BHLR.

On-disk legacy files remain unchanged:
- train_rgbt.py
- train_rsdt.py
- datasets/rsdt_dataset.py
- models/rgbt_model.py
- FBAM files

This script reuses the existing complete two-stage trainer in memory.
"""

from __future__ import annotations

import argparse
from typing import Dict

import train_rsdt as rsdt_train
from models.rgbt_rlsfa_bhlr_model import RGBTRLSFABHLRDetectionModel


_ACTIVE_CFG: Dict = {}


def build_rlsfa_bhlr_model_adapter(
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

    return RGBTRLSFABHLRDetectionModel(
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
        detail_channels=int(model_cfg.get("detail_channels", 64)),
        stem_channels=int(model_cfg.get("stem_channels", 32)),
        guide_channels=int(model_cfg.get("guide_channels", 32)),
        bhlr_freq_cutoff=float(model_cfg.get("bhlr_freq_cutoff", 0.15)),
        bhlr_freq_sharpness=float(model_cfg.get("bhlr_freq_sharpness", 24.0)),
        alignment_cfg=dict(model_cfg.get("alignment", {})),
        alignment_loss_cfg=dict(loss_cfg.get("alignment", {}) or {}),
        verbose=verbose,
    )


def train_one_epoch_rlsfa(*args, **kwargs):
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


def run_training_stage_rlsfa(*args, **kwargs):
    result = rsdt_train._ORIGINAL_RUN_TRAINING_STAGE(*args, **kwargs)
    model = kwargs.get("model", None)
    if model is not None and hasattr(model, "bhlr"):
        state = model.bhlr.scalar_state()
        print("\nBHLR state:")
        for key, value in state.items():
            if key.startswith("gamma"):
                print(f"  {key} = {value:.6f}")
    return result


def install_rlsfa_bhlr_adapters(cfg: Dict):
    global _ACTIVE_CFG
    _ACTIVE_CFG = cfg

    # Dataset, preprocessing and semantic-RGB validation logic.
    rsdt_train.install_rsdt_adapters(cfg)

    # New model only; legacy source files are untouched.
    rsdt_train.base_train.RGBTDetectionModel = build_rlsfa_bhlr_model_adapter
    rsdt_train.base_train.train_one_epoch = train_one_epoch_rlsfa
    rsdt_train.base_train.run_training_stage = run_training_stage_rlsfa


def train(cfg: Dict):
    install_rlsfa_bhlr_adapters(cfg)
    model_cfg = cfg.get("model", {})
    align_cfg = model_cfg.get("alignment", {})
    align_loss = (cfg.get("loss", {}).get("alignment", {}) or {})

    print("\n============================================================")
    print("RLSFA + BHLR training")
    print("============================================================")
    print("Alignment module  : RLSFA")
    print("Geometry          : coarse translation + fine residual translation")
    print("Global FFT        : NO")
    print(f"Window FFT        : {align_cfg.get('window_sizes', [4, 8])}")
    print(f"Spatial kernels   : {align_cfg.get('spatial_kernel_sizes', [3, 5])}")
    print(f"Coarse radius     : {align_cfg.get('coarse_radius', 8)}")
    print(f"Fine radius       : {align_cfg.get('fine_radius', 2)}")
    print(f"Max fine offset   : {align_cfg.get('max_fine_offset', 1.0)}")
    print(f"RLSFA scales      : {align_cfg.get('scales', [3])}")
    print(f"BHLR scales       : {model_cfg.get('bhlr_scales', [3])}")
    print(f"RGB high          : {cfg['data'].get('rgb_imgsz', 1280)}")
    print(f"RGB semantic      : {cfg['data'].get('rgb_semantic_imgsz', 640)}")
    print(f"TIR               : {cfg['data'].get('tir_imgsz', 640)}")
    print(f"Alignment loss    : {align_loss.get('enabled', False)}")
    print("Fusion            : concat (Module-3 OFF)")
    print("============================================================\n")

    return rsdt_train.base_train.train(cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RLSFA + BHLR RGB-T training")
    parser.add_argument("--cfg", type=str, required=True)
    args = parser.parse_args()
    cfg = rsdt_train.base_train.load_config(args.cfg)
    train(cfg)
