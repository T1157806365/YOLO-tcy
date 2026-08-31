"""
Full-feature validator for RGB-T + FBAM + BHLR
==============================================

Design goal
-----------
Keep the existing full-feature val_rsdt.py UNCHANGED.

Reuse from val_rsdt.py:
- P / R / AP50 / AP75 / AP50-95
- YOLO26 end-to-end and YOLO11 NMS postprocessing
- Params(M)
- FLOPs(G)
- batch=1 model FPS / latency
- preprocess / inference / postprocess timing
- end-to-end FPS
- PR/F1/P/R plots
- prediction / GT preview grids
- per-class table
- metrics.json
- paper_metrics.csv
- train / val / test split support

Only replace checkpoint model construction:
    RGBTRSDTDetectionModel
        ->
    RGBTFBAMBHLRDetectionModel
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import torch

import val_rsdt as base_val

from models.rgbt_fbam_bhlr_model import (
    RGBTFBAMBHLRDetectionModel,
)


# ===========================================================================
# 1. State-dict helper
# ===========================================================================

def _strip_module_prefix(
    state_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Remove DataParallel/DDP 'module.' prefix when present."""

    if (
        state_dict
        and all(
            key.startswith("module.")
            for key in state_dict.keys()
        )
    ):
        return {
            key[len("module."):]: value
            for key, value in state_dict.items()
        }

    return state_dict


# ===========================================================================
# 2. Load FBAM + BHLR checkpoint
# ===========================================================================

def load_fbam_bhlr_checkpoint(
    checkpoint_path: str,
    device: torch.device,
):
    checkpoint_path = (
        Path(checkpoint_path)
        .expanduser()
        .resolve()
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            "\n权重不存在:\n"
            f"{checkpoint_path}"
        )

    print("\nLoading FBAM + BHLR checkpoint:")
    print(checkpoint_path)

    ckpt = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    if "model_state_dict" not in ckpt:
        raise KeyError(
            "\nCheckpoint 中没有 model_state_dict。\n"
        )

    cfg = ckpt.get("config", {})
    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})

    bhlr_scales = model_cfg.get(
        "bhlr_scales",
        [3],
    )

    semantic_imgsz = int(
        data_cfg.get(
            "rgb_semantic_imgsz",
            640,
        )
    )

    high_imgsz = int(
        data_cfg.get(
            "rgb_imgsz",
            data_cfg.get(
                "rgb_high_imgsz",
                1280,
            ),
        )
    )

    # Prefer the value saved in YAML.
    # Otherwise infer the integer ratio from the saved input sizes.
    high_to_semantic_ratio = int(
        model_cfg.get(
            "high_to_semantic_ratio",
            max(
                1,
                high_imgsz // semantic_imgsz,
            ),
        )
    )

    model = RGBTFBAMBHLRDetectionModel(
        model_name=ckpt.get(
            "model_name",
            model_cfg.get(
                "name",
                "yolo26n",
            ),
        ),

        nc=int(
            ckpt.get(
                "nc",
                1,
            )
        ),

        # Validation must reconstruct architecture only.
        # Do not load COCO pretrained weights here.
        pretrained=False,

        # Module-3 is OFF; keep original concat fusion.
        fusion="concat",

        align_mode=model_cfg.get(
            "align_mode",
            "bilinear",
        ),

        names=ckpt.get(
            "names",
            {
                0: "UAV",
            },
        ),

        semantic_imgsz=semantic_imgsz,

        # Module-2
        bhlr_scales=bhlr_scales,

        high_to_semantic_ratio=(
            high_to_semantic_ratio
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

        # Module-1
        alignment_cfg=model_cfg.get(
            "alignment",
            {},
        ),

        verbose=False,
    )

    state_dict = _strip_module_prefix(
        ckpt["model_state_dict"]
    )

    # Strict loading is intentional:
    # if training and validation architectures do not match,
    # fail immediately instead of silently evaluating the wrong network.
    model.load_state_dict(
        state_dict,
        strict=True,
    )

    model = model.to(device)
    model.eval()

    print(
        "FBAM scales : "
        f"{list(getattr(model.alignment, 'scales', []))}"
    )
    print(
        "BHLR scales : "
        f"{list(model.bhlr_scales)}"
    )
    print(
        "Fusion      : concat (Module-3 OFF)"
    )
    print(
        "RGB high    : "
        f"{high_imgsz}"
    )
    print(
        "RGB semantic: "
        f"{semantic_imgsz}"
    )
    print(
        "HR/LR ratio : "
        f"{high_to_semantic_ratio}"
    )

    return (
        model,
        ckpt,
        cfg,
    )


# ===========================================================================
# 3. Reuse the complete old validator
# ===========================================================================

# val_rsdt.validate() resolves this global function at runtime.
# Patch only this function in memory; val_rsdt.py on disk is unchanged.
base_val.load_rsdt_checkpoint = (
    load_fbam_bhlr_checkpoint
)


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
    rgb_semantic_imgsz: int | None = None,
    tir_imgsz: int | None = None,
    pair_mode: str | None = None,
    save_dir: str | None = None,
    plots: bool = False,
    preview_num: int = 6,
    preview_conf: float = 0.25,
    preview_cols: int = 3,
    speed_warmup: int = 20,
    speed_iters: int = 100,
):
    """
    Full validation delegated to val_rsdt.validate().

    The only functional difference is that checkpoint construction now uses
    RGBTFBAMBHLRDetectionModel.
    """

    return base_val.validate(
        checkpoint=checkpoint,
        split=split,
        device_str=device_str,
        batch_size=batch_size,
        workers=workers,
        conf_thres=conf_thres,
        iou_thres=iou_thres,
        max_det=max_det,
        amp=amp,
        rgb_yaml=rgb_yaml,
        tir_yaml=tir_yaml,
        rgb_imgsz=rgb_imgsz,
        rgb_semantic_imgsz=rgb_semantic_imgsz,
        tir_imgsz=tir_imgsz,
        pair_mode=pair_mode,
        save_dir=save_dir,
        plots=plots,
        preview_num=preview_num,
        preview_conf=preview_conf,
        preview_cols=preview_cols,
        speed_warmup=speed_warmup,
        speed_iters=speed_iters,
    )


# ===========================================================================
# 4. CLI
# ===========================================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Full-feature validation for "
            "RGB-T + FBAM + BHLR"
        )
    )

    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="FBAM+BHLR best.pt / last.pt",
    )

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
        help="Optional RGB dataset YAML override",
    )

    parser.add_argument(
        "--tir",
        type=str,
        default=None,
        help="Optional TIR dataset YAML override",
    )

    parser.add_argument(
        "--rgb-imgsz",
        type=int,
        default=None,
        help="High-resolution RGB canvas size",
    )

    parser.add_argument(
        "--rgb-high-imgsz",
        dest="rgb_imgsz",
        type=int,
        default=None,
        help="Alias of --rgb-imgsz",
    )

    parser.add_argument(
        "--rgb-semantic-imgsz",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--tir-imgsz",
        type=int,
        default=None,
    )

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

    parser.add_argument(
        "--conf",
        type=float,
        default=0.001,
    )

    parser.add_argument(
        "--iou",
        type=float,
        default=0.7,
    )

    parser.add_argument(
        "--max-det",
        type=int,
        default=300,
    )

    parser.add_argument(
        "--device",
        type=str,
        default="0",
    )

    parser.add_argument(
        "--no-amp",
        action="store_true",
    )

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

    parser.add_argument(
        "--speed-warmup",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--speed-iters",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--plots",
        action="store_true",
    )

    parser.add_argument(
        "--preview-num",
        type=int,
        default=6,
    )

    parser.add_argument(
        "--preview-conf",
        type=float,
        default=0.25,
    )

    parser.add_argument(
        "--preview-cols",
        type=int,
        default=3,
    )

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
        rgb_semantic_imgsz=(
            args.rgb_semantic_imgsz
        ),
        tir_imgsz=args.tir_imgsz,
        pair_mode=args.pair_mode,
        save_dir=args.save_dir,
        plots=args.plots,
        preview_num=args.preview_num,
        preview_conf=args.preview_conf,
        preview_cols=args.preview_cols,
        speed_warmup=args.speed_warmup,
        speed_iters=args.speed_iters,
    )
