#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Validator for NEW RLSFA + BHLR-v2 ANY-LAYER checkpoints only.

Rules:
- bhlr_layers means ACTUAL backbone layer IDs.
- If config only has bhlr_scales, the CURRENT model performs the conceptual
  P3/P4/P5 -> actual-layer mapping.
- No legacy checkpoint key migration is performed.
- strict=True is always used.
- Old checkpoints whose adapter keys use conceptual P-level IDs will fail
  explicitly instead of being silently remapped.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple

import torch

import val_rsdt as base_val
from models.rgbt_rlsfa_bhlr_model_v2 import RGBTRLSFABHLRV2DetectionModel


# Validation-time override. "auto" selects by exact state_dict structural compatibility.
_FUSION_OVERRIDE = "auto"


def _fusion_candidates(
    ckpt: Dict,
    model_cfg: Dict,
    override: str = "auto",
) -> List[Tuple[str, str]]:
    """Return fusion reconstruction candidates in priority order.

    Auto mode does NOT infer fusion from ``fusions.*.fuse.*`` names because
    in this project CONCAT commonly owns a learnable 1x1 Conv/BN named
    ``fuse`` while ADD can be parameter-free.  Instead, each candidate is
    instantiated and compared against the checkpoint state_dict.
    """
    override = str(override or "auto").strip().lower()
    supported = {"auto", "concat", "add", "weighted"}
    if override not in supported:
        raise ValueError(
            f"Unsupported fusion override: {override}. "
            "Expected auto/concat/add/weighted."
        )

    if override != "auto":
        return [(override, f"CLI override --fusion {override}")]

    out: List[Tuple[str, str]] = []

    def push(value, source):
        mode = str(value or "").strip().lower()
        if mode in {"concat", "add", "weighted"} and all(mode != x[0] for x in out):
            out.append((mode, source))

    # Prefer what was explicitly saved by the trainer.
    push(ckpt.get("fusion", None), "checkpoint top-level fusion")
    push(model_cfg.get("fusion", None), "checkpoint config model.fusion")

    # Structural fallbacks.  Exact state_dict compatibility decides.
    push("concat", "structural fallback")
    push("add", "structural fallback")
    push("weighted", "structural fallback")
    return out

def _strip_module_prefix(state_dict: Dict[str, torch.Tensor]):
    if state_dict and all(k.startswith("module.") for k in state_dict):
        return {k[len("module."):]: v for k, v in state_dict.items()}
    return state_dict


def _extract_adapter_layer_ids(state_dict: Dict[str, torch.Tensor], prefix: str) -> List[int]:
    pattern = re.compile(rf"^{re.escape(prefix)}\.adapters\.(\d+)\.")
    ids = set()
    for key in state_dict.keys():
        m = pattern.match(key)
        if m:
            ids.add(int(m.group(1)))
    return sorted(ids)


def _check_new_anylayer_checkpoint(
    state_dict: Dict[str, torch.Tensor],
    model: RGBTRLSFABHLRV2DetectionModel,
) -> Tuple[List[int], List[int]]:
    expected = sorted(int(x) for x in getattr(model, "bhlr_layers", ()))
    bhlr_ids = _extract_adapter_layer_ids(state_dict, "bhlr")
    rsdt_ids = _extract_adapter_layer_ids(state_dict, "rsdt")

    if not expected:
        raise RuntimeError("Current ANY-LAYER model resolved no BHLR layer IDs.")

    if not bhlr_ids:
        raise RuntimeError(
            "Checkpoint has no bhlr.adapters.<actual_layer_id>.* keys. "
            "This validator only supports NEW ANY-LAYER checkpoints."
        )

    if rsdt_ids and rsdt_ids != bhlr_ids:
        raise RuntimeError(
            "Checkpoint BHLR/RSDT alias layer IDs disagree:\n"
            f"  bhlr.adapters = {bhlr_ids}\n"
            f"  rsdt.adapters = {rsdt_ids}"
        )

    if bhlr_ids != expected:
        raise RuntimeError(
            "\nCheckpoint is NOT compatible with NEW ANY-LAYER semantics.\n"
            "Legacy adapter-key remapping is disabled.\n\n"
            f"  Current model actual BHLR layer IDs : {expected}\n"
            f"  Checkpoint bhlr.adapters IDs        : {bhlr_ids}\n"
            f"  Checkpoint rsdt.adapters IDs        : {rsdt_ids}\n\n"
            "This usually indicates an older BHLR-v2 checkpoint where an "
            "adapter key such as '3' represented conceptual P3 rather than "
            "actual backbone layer 3."
        )

    return bhlr_ids, rsdt_ids


def load_rlsfa_bhlr_v2_checkpoint(checkpoint_path: str, device: torch.device):
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)

    print("\nLoading NEW ANY-LAYER RLSFA + BHLR-v2 checkpoint:")
    print(path)

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        raise TypeError(f"Expected checkpoint dict, got {type(ckpt)}")
    if "model_state_dict" not in ckpt:
        raise KeyError("Checkpoint has no model_state_dict")

    state_dict = _strip_module_prefix(ckpt["model_state_dict"])
    cfg = ckpt.get("config", {}) or {}
    model_cfg = cfg.get("model", {}) or {}
    data_cfg = cfg.get("data", {}) or {}
    loss_cfg = cfg.get("loss", {}) or {}

    alignment_cfg = dict(model_cfg.get("alignment", {}) or {})
    alignment_loss_cfg = dict(loss_cfg.get("alignment", {}) or {})

    semantic_imgsz = int(data_cfg.get("rgb_semantic_imgsz", 640))
    high_imgsz = int(data_cfg.get("rgb_imgsz", data_cfg.get("rgb_high_imgsz", 1280)))
    tir_imgsz = int(data_cfg.get("tir_imgsz", 640))
    ratio = int(
        model_cfg.get(
            "high_to_semantic_ratio",
            max(1, high_imgsz // max(semantic_imgsz, 1)),
        )
    )

    configured_layers = model_cfg.get("bhlr_layers", None)
    if configured_layers is not None:
        bhlr_layers = [int(x) for x in configured_layers]
        bhlr_scales = None
        layer_config_mode = "bhlr_layers (actual backbone IDs)"
    else:
        bhlr_layers = None
        bhlr_scales = [int(x) for x in model_cfg.get("bhlr_scales", [3])]
        layer_config_mode = "bhlr_scales (current-model P-level mapping)"

    guidance_mode = str(
        model_cfg.get(
            "guidance_mode",
            alignment_cfg.get("bhlr_guidance_mode", "auto"),
        )
    )
    detach_external_guidance = bool(
        model_cfg.get(
            "detach_external_guidance",
            alignment_cfg.get("bhlr_detach_external_guidance", True),
        )
    )
    external_structure_channels = int(
        model_cfg.get(
            "external_structure_channels",
            alignment_cfg.get("hidden_channels", 32),
        )
    )

    model_kwargs = dict(
        model_name=ckpt.get("model_name", model_cfg.get("name", "yolo26n")),
        nc=int(ckpt.get("nc", 1)),
        pretrained=False,
        align_mode=str(model_cfg.get("align_mode", "bilinear")),
        names=ckpt.get("names", {0: "UAV"}),
        semantic_imgsz=semantic_imgsz,
        bhlr_layers=bhlr_layers,
        bhlr_scales=bhlr_scales,
        high_to_semantic_ratio=ratio,
        detail_channels=int(model_cfg.get("detail_channels", 48)),
        stem_channels=int(model_cfg.get("stem_channels", 16)),
        guide_channels=int(model_cfg.get("guide_channels", 24)),
        bhlr_freq_cutoff=float(model_cfg.get("bhlr_freq_cutoff", 0.15)),
        bhlr_freq_sharpness=float(model_cfg.get("bhlr_freq_sharpness", 24.0)),
        guidance_mode=guidance_mode,
        detach_external_guidance=detach_external_guidance,
        external_structure_channels=external_structure_channels,
        alignment_cfg=alignment_cfg,
        alignment_loss_cfg=alignment_loss_cfg,
        verbose=False,
    )

    checkpoint_keys = set(state_dict.keys())
    model = None
    fusion_mode = None
    fusion_source = None
    fusion_diagnostics = []

    for candidate_mode, candidate_source in _fusion_candidates(
        ckpt=ckpt,
        model_cfg=model_cfg,
        override=_FUSION_OVERRIDE,
    ):
        try:
            candidate = RGBTRLSFABHLRV2DetectionModel(
                fusion=candidate_mode,
                **model_kwargs,
            )
        except Exception as exc:
            fusion_diagnostics.append(
                f"{candidate_mode}: constructor failed: {type(exc).__name__}: {exc}"
            )
            continue

        candidate_keys = set(candidate.state_dict().keys())
        missing = sorted(candidate_keys - checkpoint_keys)
        unexpected = sorted(checkpoint_keys - candidate_keys)

        if not missing and not unexpected:
            model = candidate
            fusion_mode = candidate_mode
            fusion_source = (
                f"{candidate_source}; exact state_dict key match"
            )
            break

        fusion_diagnostics.append(
            f"{candidate_mode}: missing={len(missing)}, unexpected={len(unexpected)}; "
            f"missing_sample={missing[:4]}; unexpected_sample={unexpected[:4]}"
        )
        del candidate

    if model is None:
        raise RuntimeError(
            "Unable to reconstruct a fusion mode compatible with this checkpoint.\n"
            "Tried:\n  " + "\n  ".join(fusion_diagnostics)
        )

    if _FUSION_OVERRIDE == "auto" and fusion_diagnostics:
        print("\n[Auto fusion] rejected candidates before exact match:")
        for item in fusion_diagnostics:
            print("  -", item)


    bhlr_ids, rsdt_ids = _check_new_anylayer_checkpoint(state_dict, model)

    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()

    print("\n============================================================")
    print("NEW ANY-LAYER checkpoint validation")
    print("============================================================")
    print(f"Model              : {getattr(model, 'model_name', 'unknown')}")
    print(f"Layer config mode  : {layer_config_mode}")
    print(f"BHLR actual layers : {list(getattr(model, 'bhlr_layers', []))}")
    print(f"Checkpoint BHLR IDs: {bhlr_ids}")
    print(f"Checkpoint RSDT IDs: {rsdt_ids}")
    print(f"RLSFA scales       : {list(getattr(model.alignment, 'scales', []))}")
    print(f"Guidance mode      : {guidance_mode}")
    print(f"Detach ext guide   : {detach_external_guidance}")
    print("Geometry           : translation-only")
    print(f"Fusion             : {fusion_mode}")
    print(f"Fusion resolved by : {fusion_source}")
    print(f"RGB high           : {high_imgsz}")
    print(f"RGB semantic       : {semantic_imgsz}")
    print(f"TIR                : {tir_imgsz}")
    print(f"High/semantic ratio: {ratio}")
    print("Checkpoint loading : strict=True")
    print("Legacy remapping   : OFF")
    print("============================================================\n")

    return model, ckpt, cfg


# val_rsdt.validate resolves this global at runtime.
base_val.load_rsdt_checkpoint = load_rlsfa_bhlr_v2_checkpoint


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
    fusion_mode: str = "auto",
):
    global _FUSION_OVERRIDE
    _FUSION_OVERRIDE = str(fusion_mode or "auto").strip().lower()
    if _FUSION_OVERRIDE not in {"auto", "concat", "add", "weighted"}:
        raise ValueError(
            f"--fusion must be one of auto/concat/add/weighted, got {_FUSION_OVERRIDE!r}"
        )

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
    p = argparse.ArgumentParser(
        description="Validation for NEW RLSFA + BHLR-v2 ANY-LAYER checkpoints only"
    )
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
    p.add_argument(
        "--fusion",
        default="auto",
        choices=["auto", "concat", "add", "weighted"],
        help=(
            "Fusion reconstruction mode. auto rebuilds candidate fusion modes and "
            "selects the one whose state_dict keys exactly match the checkpoint."
        ),
    )
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
        fusion_mode=a.fusion,
    )
