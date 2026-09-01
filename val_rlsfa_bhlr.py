"""Full-feature validator for RLSFA + BHLR, reusing val_rsdt.py."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import torch

import val_rsdt as base_val
from models.rgbt_rlsfa_bhlr_model import RGBTRLSFABHLRDetectionModel


def _strip_module_prefix(state_dict: Dict[str, torch.Tensor]):
    if state_dict and all(k.startswith("module.") for k in state_dict):
        return {k[len("module."):]: v for k, v in state_dict.items()}
    return state_dict


def load_rlsfa_bhlr_checkpoint(checkpoint_path: str, device: torch.device):
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)

    print("\nLoading RLSFA + BHLR checkpoint:")
    print(path)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if "model_state_dict" not in ckpt:
        raise KeyError("Checkpoint has no model_state_dict")

    cfg = ckpt.get("config", {})
    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})
    loss_cfg = cfg.get("loss", {})

    semantic_imgsz = int(data_cfg.get("rgb_semantic_imgsz", 640))
    high_imgsz = int(data_cfg.get("rgb_imgsz", data_cfg.get("rgb_high_imgsz", 1280)))
    ratio = int(model_cfg.get("high_to_semantic_ratio", max(1, high_imgsz // semantic_imgsz)))

    model = RGBTRLSFABHLRDetectionModel(
        model_name=ckpt.get("model_name", model_cfg.get("name", "yolo26n")),
        nc=int(ckpt.get("nc", 1)),
        pretrained=False,
        fusion="concat",
        align_mode=model_cfg.get("align_mode", "bilinear"),
        names=ckpt.get("names", {0: "UAV"}),
        semantic_imgsz=semantic_imgsz,
        bhlr_scales=model_cfg.get("bhlr_scales", [3]),
        high_to_semantic_ratio=ratio,
        detail_channels=int(model_cfg.get("detail_channels", 64)),
        stem_channels=int(model_cfg.get("stem_channels", 32)),
        guide_channels=int(model_cfg.get("guide_channels", 32)),
        bhlr_freq_cutoff=float(model_cfg.get("bhlr_freq_cutoff", 0.15)),
        bhlr_freq_sharpness=float(model_cfg.get("bhlr_freq_sharpness", 24.0)),
        alignment_cfg=model_cfg.get("alignment", {}),
        alignment_loss_cfg=(loss_cfg.get("alignment", {}) or {}),
        verbose=False,
    )

    model.load_state_dict(_strip_module_prefix(ckpt["model_state_dict"]), strict=True)
    model = model.to(device)
    model.eval()

    print(f"RLSFA scales : {list(getattr(model.alignment, 'scales', []))}")
    print(f"BHLR scales  : {list(model.bhlr_scales)}")
    print("Geometry     : translation-only")
    print("Fusion       : concat (Module-3 OFF)")
    print(f"RGB high     : {high_imgsz}")
    print(f"RGB semantic : {semantic_imgsz}")
    return model, ckpt, cfg


# val_rsdt.validate resolves this global at runtime.
base_val.load_rsdt_checkpoint = load_rlsfa_bhlr_checkpoint


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


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Full validation for RGB-T + RLSFA + BHLR")
    p.add_argument("--weights", required=True)
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--rgb", default=None)
    p.add_argument("--tir", default=None)
    p.add_argument("--rgb-imgsz", type=int, default=None)
    p.add_argument("--rgb-semantic-imgsz", type=int, default=None)
    p.add_argument("--tir-imgsz", type=int, default=None)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--conf", type=float, default=0.001)
    p.add_argument("--iou", type=float, default=0.7)
    p.add_argument("--max-det", type=int, default=300)
    p.add_argument("--device", default="0")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--pair-mode", default=None, choices=["relative", "stem", "filename"])
    p.add_argument("--save-dir", default=None)
    p.add_argument("--plots", action="store_true")
    p.add_argument("--preview-num", type=int, default=6)
    p.add_argument("--preview-conf", type=float, default=0.25)
    p.add_argument("--preview-cols", type=int, default=3)
    p.add_argument("--speed-warmup", type=int, default=20)
    p.add_argument("--speed-iters", type=int, default=100)
    a = p.parse_args()

    validate(
        checkpoint=a.weights,
        split=a.split,
        device_str=a.device,
        batch_size=a.batch,
        workers=a.workers,
        conf_thres=a.conf,
        iou_thres=a.iou,
        max_det=a.max_det,
        amp=not a.no_amp,
        rgb_yaml=a.rgb,
        tir_yaml=a.tir,
        rgb_imgsz=a.rgb_imgsz,
        rgb_semantic_imgsz=a.rgb_semantic_imgsz,
        tir_imgsz=a.tir_imgsz,
        pair_mode=a.pair_mode,
        save_dir=a.save_dir,
        plots=a.plots,
        preview_num=a.preview_num,
        preview_conf=a.preview_conf,
        preview_cols=a.preview_cols,
        speed_warmup=a.speed_warmup,
        speed_iters=a.speed_iters,
    )
