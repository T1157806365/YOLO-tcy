"""
Training entry for FBAM + BHLR.

Unchanged:
- train_rgbt.py
- train_rsdt.py
- datasets/rsdt_dataset.py
- models/rgbt_model.py

The existing RSD-T data/preprocess/validation adapters are reused.
"""

from __future__ import annotations
import argparse
from typing import Dict

import train_rsdt as rsdt_train
from models.rgbt_fbam_bhlr_model import RGBTFBAMBHLRDetectionModel

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
    model_cfg = _ACTIVE_CFG.get("model", {})
    data_cfg = _ACTIVE_CFG.get("data", {})
    alignment_cfg = dict(model_cfg.get("alignment", {}))

    return RGBTFBAMBHLRDetectionModel(
        model_name=model_name,
        nc=nc,
        pretrained=pretrained,
        fusion="concat",
        fusion_indices=fusion_indices,
        align_mode=align_mode,
        names=names,
        semantic_imgsz=int(data_cfg.get("rgb_semantic_imgsz", 640)),
        bhlr_scales=model_cfg.get("bhlr_scales", [3]),
        high_to_semantic_ratio=int(
            model_cfg.get("high_to_semantic_ratio", 3)
        ),
        detail_channels=int(model_cfg.get("detail_channels", 64)),
        stem_channels=int(model_cfg.get("stem_channels", 32)),
        guide_channels=int(model_cfg.get("guide_channels", 32)),
        bhlr_freq_cutoff=float(
            model_cfg.get("bhlr_freq_cutoff", 0.15)
        ),
        bhlr_freq_sharpness=float(
            model_cfg.get("bhlr_freq_sharpness", 24.0)
        ),
        alignment_cfg=alignment_cfg,
        verbose=verbose,
    )


def run_training_stage_bhlr(*args, **kwargs):
    result = rsdt_train._ORIGINAL_RUN_TRAINING_STAGE(*args, **kwargs)

    model = kwargs.get("model", None)
    if model is not None and hasattr(model, "bhlr"):
        state = model.bhlr.scalar_state()
        print("\nBHLR state:")
        for key, value in state.items():
            if key.startswith("gamma"):
                print(f"  {key} = {value:.6f}")

    return result


def install_fbam_bhlr_adapters(cfg: Dict):
    global _ACTIVE_CFG
    _ACTIVE_CFG = cfg

    # Install the old high-RGB/semantic-RGB/TIR dataset and validation path.
    rsdt_train.install_rsdt_adapters(cfg)

    # Replace only model construction and status reporting in memory.
    rsdt_train.base_train.RGBTDetectionModel = (
        build_fbam_bhlr_model_adapter
    )
    rsdt_train.base_train.run_training_stage = (
        run_training_stage_bhlr
    )


def train(cfg: Dict):
    install_fbam_bhlr_adapters(cfg)

    model_cfg = cfg.get("model", {})
    align_cfg = model_cfg.get("alignment", {})

    print("\n============================================================")
    print("FBAM + BHLR training")
    print("Base engine      : train_rgbt.py (unchanged)")
    print("Dataset          : rsdt_dataset.py (unchanged)")
    print(f"FBAM scales      : {align_cfg.get('scales', [3])}")
    print(f"BHLR scales      : {model_cfg.get('bhlr_scales', [3])}")
    print("Fusion           : concat (Module-3 OFF)")
    print(f"RGB high         : {cfg['data'].get('rgb_imgsz', 1920)}")
    print(f"RGB semantic     : {cfg['data'].get('rgb_semantic_imgsz', 640)}")
    print(f"TIR              : {cfg['data'].get('tir_imgsz', 640)}")
    print("============================================================\n")

    return rsdt_train.base_train.train(cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FBAM + BHLR RGB-T training"
    )
    parser.add_argument("--cfg", type=str, required=True)
    args = parser.parse_args()

    config = rsdt_train.base_train.load_config(args.cfg)
    train(config)
