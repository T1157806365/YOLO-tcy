#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Five-level diagnostic for RLSFA:
    L0 NO WARP
    L1 COARSE TRANSLATION
    L2 FINAL TRANSLATION
    L3 ORACLE TRANSLATION
    L4 ORACLE AFFINE (reference only; NOT used for training)

Primary metrics:
    center error in P-level pixels
    soft IoU of warped TIR bbox mask against RGB bbox mask

Also reports:
    coarse/final direction cosine to GT sampling translation
    RGB/TIR frequency reliability in RGB UAV ROI
    targetness ROI/background means
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import val_rgbt as base_val
from datasets.rsdt_dataset import build_rsdt_dataset
from train_rsdt import preprocess_batch_rsdt
from val_rlsfa_bhlr import load_rlsfa_bhlr_checkpoint


def get_scale(mapping, scale):
    if scale in mapping:
        return mapping[scale]
    if str(scale) in mapping:
        return mapping[str(scale)]
    raise KeyError(f"P{scale} unavailable, keys={list(mapping.keys())}")


def bbox_mask(box, h, w, device):
    """Normalized xywh -> soft binary mask [1,1,H,W]."""
    cx, cy, bw, bh = box.float()
    x1 = max(0, min(w - 1, int(torch.floor((cx - bw / 2) * w).item())))
    y1 = max(0, min(h - 1, int(torch.floor((cy - bh / 2) * h).item())))
    x2 = max(x1 + 1, min(w, int(torch.ceil((cx + bw / 2) * w).item())))
    y2 = max(y1 + 1, min(h, int(torch.ceil((cy + bh / 2) * h).item())))
    m = torch.zeros(1, 1, h, w, device=device, dtype=torch.float32)
    m[:, :, y1:y2, x1:x2] = 1.0
    return m


def warp_mask(mask, offset):
    b, _, h, w = mask.shape
    if offset.shape[-2:] != (h, w):
        offset = offset.expand(-1, -1, h, w)
    ys = torch.linspace(-1, 1, h, device=mask.device)
    xs = torch.linspace(-1, 1, w, device=mask.device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    base = torch.stack((xx, yy), -1)[None].expand(b, -1, -1, -1)
    dx = 2.0 * offset[:, 0] / max(w - 1, 1)
    dy = 2.0 * offset[:, 1] / max(h - 1, 1)
    delta = torch.stack((dx, dy), -1)
    return F.grid_sample(mask, base + delta, mode="bilinear", padding_mode="zeros", align_corners=True)


def soft_iou(a, b):
    inter = torch.minimum(a, b).sum()
    union = torch.maximum(a, b).sum().clamp(min=1e-6)
    return float((inter / union).item())


def soft_center(mask):
    _, _, h, w = mask.shape
    mass = mask.sum().clamp(min=1e-6)
    yy, xx = torch.meshgrid(
        torch.arange(h, device=mask.device, dtype=mask.dtype),
        torch.arange(w, device=mask.device, dtype=mask.dtype),
        indexing="ij",
    )
    cx = (mask[0, 0] * xx).sum() / mass
    cy = (mask[0, 0] * yy).sum() / mass
    return torch.stack((cx, cy))


def center_error(mask, rgb_mask):
    return float(torch.linalg.vector_norm(soft_center(mask) - soft_center(rgb_mask)).item())


def oracle_translation(rbox, tbox, h, w, device):
    rcx, rcy = rbox[0] * w, rbox[1] * h
    tcx, tcy = tbox[0] * w, tbox[1] * h
    return torch.stack((tcx - rcx, tcy - rcy)).view(1, 2, 1, 1).to(device=device, dtype=torch.float32)


def oracle_affine(rbox, tbox, h, w, device):
    rcx, rcy, rw, rh = rbox.float()
    tcx, tcy, tw, th = tbox.float()
    rcx, rcy, rw, rh = rcx * w, rcy * h, rw * w, rh * h
    tcx, tcy, tw, th = tcx * w, tcy * h, tw * w, th * h
    yy, xx = torch.meshgrid(
        torch.arange(h, device=device, dtype=torch.float32),
        torch.arange(w, device=device, dtype=torch.float32),
        indexing="ij",
    )
    qx = tcx + (tw / rw.clamp(min=1e-6)) * (xx - rcx)
    qy = tcy + (th / rh.clamp(min=1e-6)) * (yy - rcy)
    return torch.stack((qx - xx, qy - yy), 0)[None]


def cosine(vec, gt):
    a = vec.reshape(-1).float()
    b = gt.reshape(-1).float()
    na = torch.linalg.vector_norm(a)
    nb = torch.linalg.vector_norm(b)
    if float(na) < 1e-8 or float(nb) < 1e-8:
        return 0.0
    return float(torch.dot(a, b).div(na * nb).item())


def roi_mean(map_tensor, box):
    """Map [1,1,H,W], normalized RGB bbox."""
    _, _, h, w = map_tensor.shape
    cx, cy, bw, bh = box.float()
    x1 = max(0, min(w - 1, int(torch.floor((cx - bw / 2) * w).item())))
    y1 = max(0, min(h - 1, int(torch.floor((cy - bh / 2) * h).item())))
    x2 = max(x1 + 1, min(w, int(torch.ceil((cx + bw / 2) * w).item())))
    y2 = max(y1 + 1, min(h, int(torch.ceil((cy + bh / 2) * h).item())))
    return float(map_tensor[:, :, y1:y2, x1:x2].float().mean().item())


@torch.inference_mode()
def run(args):
    device = base_val.select_device(args.device)
    model, ckpt, cfg = load_rlsfa_bhlr_checkpoint(args.weights, device)
    data = cfg.get("data", {})

    dataset = build_rsdt_dataset(
        rgb_yaml=args.rgb or data.get("rgb"),
        tir_yaml=args.tir or data.get("tir"),
        split=args.split,
        rgb_high_imgsz=args.rgb_imgsz or int(data.get("rgb_imgsz", 1280)),
        rgb_semantic_imgsz=args.rgb_semantic_imgsz or int(data.get("rgb_semantic_imgsz", 640)),
        tir_imgsz=args.tir_imgsz or int(data.get("tir_imgsz", 640)),
        pair_mode=args.pair_mode or data.get("pair_mode", "relative"),
        augment=False,
        tir_channels=3,
        strict_pair=True,
    )

    rows = []
    candidate_indices = args.index if args.index else list(range(len(dataset)))

    for idx in candidate_indices:
        if len(rows) >= args.num_samples:
            break
        sample = dataset[idx]
        # Diagnostic only uses unambiguous single-object pairs.
        if int(sample["rgb_semantic_bboxes"].shape[0]) != 1 or int(sample["tir_bboxes"].shape[0]) != 1:
            continue

        batch = dataset.collate_fn([sample])
        batch = preprocess_batch_rsdt(batch, device)
        out = model(
            batch["rgb_img"], batch["tir_img"], batch["rgb_semantic_img"],
            return_features=True, return_alignment_debug=True,
        )
        info = out["alignment_info"]
        dbg = get_scale(info.get("scale_debug", info.get("scales", {})), args.scale)

        coarse = dbg["coarse_offset"].float()
        final = dbg["total_offset"].float()
        rbox = batch["rgb_semantic_bboxes"][0].float()
        tbox = batch["tir_bboxes"][0].float()
        h, w = dbg["coarse_targetness"].shape[-2:]
        rmask = bbox_mask(rbox, h, w, device)
        tmask = bbox_mask(tbox, h, w, device)
        gt = oracle_translation(rbox, tbox, h, w, device)
        aff = oracle_affine(rbox, tbox, h, w, device)

        stages = {
            "no": tmask,
            "coarse": warp_mask(tmask, coarse),
            "final": warp_mask(tmask, final),
            "oracle_translation": warp_mask(tmask, gt),
            "oracle_affine": warp_mask(tmask, aff),
        }

        target = dbg["coarse_targetness"].float()
        qr = dbg["rgb_reliability"].float()
        qt = dbg["tir_reliability"].float()

        row = {
            "index": idx,
            "pair_key": str(sample["pair_key"]),
            "no_center_error": center_error(stages["no"], rmask),
            "coarse_center_error": center_error(stages["coarse"], rmask),
            "final_center_error": center_error(stages["final"], rmask),
            "oracle_translation_center_error": center_error(stages["oracle_translation"], rmask),
            "oracle_affine_center_error": center_error(stages["oracle_affine"], rmask),
            "no_iou": soft_iou(stages["no"], rmask),
            "coarse_iou": soft_iou(stages["coarse"], rmask),
            "final_iou": soft_iou(stages["final"], rmask),
            "oracle_translation_iou": soft_iou(stages["oracle_translation"], rmask),
            "oracle_affine_iou": soft_iou(stages["oracle_affine"], rmask),
            "coarse_cosine": cosine(coarse, gt),
            "final_cosine": cosine(final, gt),
            "gt_dx": float(gt[0, 0, 0, 0].item()),
            "gt_dy": float(gt[0, 1, 0, 0].item()),
            "coarse_dx": float(coarse[0, 0, 0, 0].item()),
            "coarse_dy": float(coarse[0, 1, 0, 0].item()),
            "final_dx": float(final[0, 0, 0, 0].item()),
            "final_dy": float(final[0, 1, 0, 0].item()),
            "targetness_roi": roi_mean(target, rbox),
            "targetness_global": float(target.mean().item()),
            "rgb_reliability_roi": roi_mean(qr, rbox),
            "tir_reliability_roi": roi_mean(qt, rbox),
        }
        rows.append(row)
        print(f"[{len(rows):03d}/{args.num_samples}] index={idx} final_err={row['final_center_error']:.3f}")

    if not rows:
        raise RuntimeError("No valid single-object RGB/TIR pairs found")

    arr = lambda k: np.asarray([r[k] for r in rows], dtype=np.float64)
    summary = {
        "samples": len(rows),
        "mean_center_error": {
            "L0_NO_WARP": float(arr("no_center_error").mean()),
            "L1_COARSE": float(arr("coarse_center_error").mean()),
            "L2_FINAL": float(arr("final_center_error").mean()),
            "L3_ORACLE_TRANSLATION": float(arr("oracle_translation_center_error").mean()),
            "L4_ORACLE_AFFINE": float(arr("oracle_affine_center_error").mean()),
        },
        "mean_soft_iou": {
            "L0_NO_WARP": float(arr("no_iou").mean()),
            "L1_COARSE": float(arr("coarse_iou").mean()),
            "L2_FINAL": float(arr("final_iou").mean()),
            "L3_ORACLE_TRANSLATION": float(arr("oracle_translation_iou").mean()),
            "L4_ORACLE_AFFINE": float(arr("oracle_affine_iou").mean()),
        },
        "improved_counts": {
            "COARSE_vs_NO": int((arr("coarse_center_error") < arr("no_center_error")).sum()),
            "FINAL_vs_NO": int((arr("final_center_error") < arr("no_center_error")).sum()),
            "FINAL_vs_COARSE": int((arr("final_center_error") < arr("coarse_center_error")).sum()),
        },
        "direction_cosine": {
            "coarse": float(arr("coarse_cosine").mean()),
            "final": float(arr("final_cosine").mean()),
        },
        "reliability": {
            "targetness_roi": float(arr("targetness_roi").mean()),
            "targetness_global": float(arr("targetness_global").mean()),
            "rgb_frequency_reliability_roi": float(arr("rgb_reliability_roi").mean()),
            "tir_frequency_reliability_roi": float(arr("tir_reliability_roi").mean()),
        },
    }

    save_dir = Path(args.save_dir).resolve() if args.save_dir else Path(args.weights).resolve().parent.parent / "rlsfa_alignment_diagnostic"
    save_dir.mkdir(parents=True, exist_ok=True)
    with (save_dir / "rlsfa_alignment_samples.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)
    with (save_dir / "rlsfa_alignment_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 88)
    print("RLSFA FINAL SUMMARY")
    print("=" * 88)
    for k, v in summary["mean_center_error"].items(): print(f"Center error {k:<24}: {v:.4f}")
    for k, v in summary["mean_soft_iou"].items(): print(f"Soft IoU     {k:<24}: {v:.4f}")
    print("Improved counts:", summary["improved_counts"])
    print("Direction cosine:", summary["direction_cosine"])
    print("Reliability:", summary["reliability"])
    print("Saved to:", save_dir)


def main():
    p = argparse.ArgumentParser(description="Five-level RLSFA alignment diagnostic")
    p.add_argument("--weights", required=True)
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--device", default="0")
    p.add_argument("--num-samples", type=int, default=100)
    p.add_argument("--scale", type=int, default=3)
    p.add_argument("--index", type=int, nargs="+", default=None)
    p.add_argument("--rgb", default=None)
    p.add_argument("--tir", default=None)
    p.add_argument("--rgb-imgsz", type=int, default=None)
    p.add_argument("--rgb-semantic-imgsz", type=int, default=None)
    p.add_argument("--tir-imgsz", type=int, default=None)
    p.add_argument("--pair-mode", default=None)
    p.add_argument("--save-dir", default=None)
    run(p.parse_args())


if __name__ == "__main__":
    main()
