#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
diagnose_fbam_coarse_fine_oracle.py
===================================

五级对齐诊断：
    L0  NO_WARP
    L1  COARSE
    L2  FINAL (当前模型 coarse + fine 后的 total_offset)
    L3  ORACLE_TRANSLATION
    L4  ORACLE_AFFINE

目的：
    判断 FBAM 的问题究竟发生在：
      1) coarse offset
      2) fine refinement
      3) warp / grid_sample 几何链
      4) 还是数据坐标映射

核心判断逻辑：
    - COARSE < NO_WARP，但 FINAL > COARSE
        => coarse 有效，fine 阶段把对齐带坏
    - COARSE > NO_WARP，FINAL 仍差
        => coarse matcher 从一开始就预测错
    - COARSE > NO_WARP，但 FINAL < COARSE
        => fine 在纠正 coarse，但纠正不够
    - ORACLE 显著优于 NO_WARP
        => warp/grid_sample 链基本没问题
    - ORACLE 也不改善
        => 应优先检查 bbox 坐标、P3映射、dx/dy、归一化、align_corners

脚本不会修改模型文件。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F

import val_rgbt as base_val
from datasets.rsdt_dataset import build_rsdt_dataset
from train_rsdt import preprocess_batch_rsdt
from val_fbam_bhlr import load_fbam_bhlr_checkpoint


# ============================================================================
# Debug helpers
# ============================================================================

def get_alignment_scale_debug(outputs: Dict, scale: int = 3) -> Dict:
    info = outputs.get("alignment_info", None)

    if not isinstance(info, dict):
        raise RuntimeError(
            "No alignment_info found. "
            "Model must run with return_alignment_debug=True."
        )

    scale_debug = info.get(
        "scale_debug",
        info.get("scales", {}),
    )

    if scale in scale_debug:
        return scale_debug[scale]

    key = str(scale)
    if key in scale_debug:
        return scale_debug[key]

    raise KeyError(
        f"P{scale} not found. Available keys: {list(scale_debug.keys())}"
    )


# ============================================================================
# Box / mask helpers
# ============================================================================

def box_xywhn_to_feature(
    boxes: torch.Tensor,
    h: int,
    w: int,
) -> Tuple[float, float, float, float]:
    """normalized xywh -> feature-coordinate xywh."""
    b = boxes.detach().float().reshape(-1, 4)[0]

    return (
        float(b[0].item() * w),
        float(b[1].item() * h),
        float(b[2].item() * w),
        float(b[3].item() * h),
    )


def boxes_to_mask(
    boxes: torch.Tensor,
    h: int,
    w: int,
    device: torch.device,
) -> torch.Tensor:
    """normalized xywh boxes -> union mask [1,1,H,W]."""
    mask = torch.zeros(
        (1, 1, h, w),
        dtype=torch.float32,
        device=device,
    )

    if boxes is None or boxes.numel() == 0:
        return mask

    boxes = boxes.detach().float().to(device).reshape(-1, 4)

    for cx, cy, bw, bh in boxes:
        x1 = int(torch.floor((cx - bw / 2) * w).item())
        y1 = int(torch.floor((cy - bh / 2) * h).item())
        x2 = int(torch.ceil((cx + bw / 2) * w).item())
        y2 = int(torch.ceil((cy + bh / 2) * h).item())

        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(x1 + 1, min(w, x2))
        y2 = max(y1 + 1, min(h, y2))

        mask[:, :, y1:y2, x1:x2] = 1.0

    return mask


def centroid(mask: torch.Tensor) -> Tuple[float, float]:
    m = mask.detach().float()[0, 0]
    mass = float(m.sum().item())

    if mass <= 1e-8:
        return float("nan"), float("nan")

    h, w = m.shape

    yy = torch.arange(
        h,
        device=m.device,
        dtype=m.dtype,
    ).view(h, 1)

    xx = torch.arange(
        w,
        device=m.device,
        dtype=m.dtype,
    ).view(1, w)

    cx = float((m * xx).sum().item() / mass)
    cy = float((m * yy).sum().item() / mass)

    return cx, cy


def center_error(
    a: torch.Tensor,
    b: torch.Tensor,
) -> float:
    ax, ay = centroid(a)
    bx, by = centroid(b)

    if any(math.isnan(v) for v in (ax, ay, bx, by)):
        return float("nan")

    return math.hypot(ax - bx, ay - by)


def soft_iou(
    a: torch.Tensor,
    b: torch.Tensor,
) -> float:
    a = a.detach().float().clamp(0, 1)
    b = b.detach().float().clamp(0, 1)

    inter = float((a * b).sum().item())
    union = float((a + b - a * b).sum().item())

    if union <= 1e-8:
        return 0.0

    return inter / union


# ============================================================================
# grid_sample warp
# ============================================================================

def build_base_grid(
    h: int,
    w: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    ys = torch.linspace(
        -1.0,
        1.0,
        h,
        device=device,
        dtype=dtype,
    )

    xs = torch.linspace(
        -1.0,
        1.0,
        w,
        device=device,
        dtype=dtype,
    )

    yy, xx = torch.meshgrid(
        ys,
        xs,
        indexing="ij",
    )

    return torch.stack(
        (xx, yy),
        dim=-1,
    ).unsqueeze(0)


def warp_plus(
    source: torch.Tensor,
    offset: torch.Tensor,
) -> torch.Tensor:
    """
    模拟当前 FBAM:
        grid = base + delta
        align_corners=True

    offset单位：
        feature pixels
    """
    source = source.float()
    offset = offset.float()

    b, _, h, w = source.shape

    if offset.shape[-2:] != (h, w):
        raise ValueError(
            f"offset spatial size {offset.shape[-2:]} != source {(h, w)}"
        )

    base = build_base_grid(
        h,
        w,
        source.device,
        source.dtype,
    ).expand(b, -1, -1, -1)

    dx_norm = 2.0 * offset[:, 0] / max(w - 1, 1)
    dy_norm = 2.0 * offset[:, 1] / max(h - 1, 1)

    delta = torch.stack(
        (dx_norm, dy_norm),
        dim=-1,
    )

    return F.grid_sample(
        source,
        base + delta,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )


# ============================================================================
# Oracle offsets
# ============================================================================

def oracle_translation_offset(
    rgb_box: torch.Tensor,
    tir_box: torch.Tensor,
    h: int,
    w: int,
    device: torch.device,
) -> torch.Tensor:
    """
    当前 FBAM:
        output(p) = input(p + delta)

    希望 TIR 内容中心 CT 映射到 RGB 中心 CR：
        CR + delta = CT
        delta = CT - CR
    """
    rcx, rcy, _, _ = box_xywhn_to_feature(
        rgb_box, h, w
    )
    tcx, tcy, _, _ = box_xywhn_to_feature(
        tir_box, h, w
    )

    offset = torch.zeros(
        (1, 2, h, w),
        dtype=torch.float32,
        device=device,
    )

    offset[:, 0] = tcx - rcx
    offset[:, 1] = tcy - rcy

    return offset


def oracle_affine_offset(
    rgb_box: torch.Tensor,
    tir_box: torch.Tensor,
    h: int,
    w: int,
    device: torch.device,
) -> torch.Tensor:
    """
    利用 RGB/TIR bbox 中心 + 宽高，构造理想的 x/y 仿射 sampling field：

        qx = cTx + (wT/wR) * (x - cRx)
        qy = cTy + (hT/hR) * (y - cRy)

        delta = q - p

    同时校正：
        - x/y 平移
        - x/y 尺度差
    """
    rcx, rcy, rw, rh = box_xywhn_to_feature(
        rgb_box, h, w
    )
    tcx, tcy, tw, th = box_xywhn_to_feature(
        tir_box, h, w
    )

    sx = tw / max(rw, 1e-6)
    sy = th / max(rh, 1e-6)

    yy = torch.arange(
        h,
        device=device,
        dtype=torch.float32,
    ).view(h, 1).expand(h, w)

    xx = torch.arange(
        w,
        device=device,
        dtype=torch.float32,
    ).view(1, w).expand(h, w)

    qx = tcx + sx * (xx - rcx)
    qy = tcy + sy * (yy - rcy)

    dx = qx - xx
    dy = qy - yy

    return torch.stack(
        (dx, dy),
        dim=0,
    ).unsqueeze(0)


# ============================================================================
# Offset diagnostics
# ============================================================================

def masked_mean_offset(
    offset: torch.Tensor,
    mask: torch.Tensor,
) -> Tuple[float, float]:
    denom = float(mask.sum().item())

    if denom <= 1e-8:
        return float("nan"), float("nan")

    dx = float(
        (offset[:, 0:1] * mask).sum().item()
        / denom
    )
    dy = float(
        (offset[:, 1:2] * mask).sum().item()
        / denom
    )

    return dx, dy


def vector_cosine(
    ax: float,
    ay: float,
    bx: float,
    by: float,
) -> float:
    na = math.hypot(ax, ay)
    nb = math.hypot(bx, by)

    if na <= 1e-8 or nb <= 1e-8:
        return float("nan")

    return (ax * bx + ay * by) / (na * nb)


# ============================================================================
# Dataset filtering
# ============================================================================

def collect_single_uav_indices(dataset):
    ids = []

    for i in range(len(dataset)):
        sample = dataset[i]

        nr = int(
            sample["rgb_semantic_bboxes"].shape[0]
        )
        nt = int(
            sample["tir_bboxes"].shape[0]
        )

        if nr == 1 and nt == 1:
            ids.append(i)

    return ids


# ============================================================================
# Main
# ============================================================================

@torch.inference_mode()
def run(args):
    device = base_val.select_device(
        args.device
    )

    model, ckpt, cfg = load_fbam_bhlr_checkpoint(
        args.weights,
        device,
    )

    model.eval()

    data_cfg = cfg.get("data", {})

    rgb_yaml = data_cfg.get("rgb")
    tir_yaml = data_cfg.get("tir")

    if rgb_yaml is None or tir_yaml is None:
        raise RuntimeError(
            "Cannot resolve data.rgb / data.tir from checkpoint config."
        )

    dataset = build_rsdt_dataset(
        rgb_yaml=rgb_yaml,
        tir_yaml=tir_yaml,
        split=args.split,
        rgb_high_imgsz=int(
            data_cfg.get(
                "rgb_imgsz",
                data_cfg.get("rgb_high_imgsz", 1280),
            )
        ),
        rgb_semantic_imgsz=int(
            data_cfg.get("rgb_semantic_imgsz", 640)
        ),
        tir_imgsz=int(
            data_cfg.get("tir_imgsz", 640)
        ),
        pair_mode=data_cfg.get(
            "pair_mode",
            "relative",
        ),
        augment=False,
        tir_channels=3,
        strict_pair=True,
    )

    ids = collect_single_uav_indices(
        dataset
    )

    if not ids:
        raise RuntimeError(
            "No single-UAV RGB/TIR positive pairs found."
        )

    rng = random.Random(
        args.seed
    )
    rng.shuffle(ids)

    ids = ids[
        :min(args.num_samples, len(ids))
    ]

    print()
    print("=" * 96)
    print("FBAM COARSE / FINE / ORACLE FIVE-LEVEL DIAGNOSIS")
    print("=" * 96)
    print(f"Weights     : {args.weights}")
    print(f"Split       : {args.split}")
    print(f"Samples     : {len(ids)}")
    print(f"Align scale : P{args.align_scale}")
    print("=" * 96)
    print()

    rows = []

    for n, idx in enumerate(ids, 1):
        sample = dataset[idx]

        batch = dataset.collate_fn(
            [sample]
        )

        batch = preprocess_batch_rsdt(
            batch,
            device,
        )

        outputs = model(
            batch["rgb_img"],
            batch["tir_img"],
            batch["rgb_semantic_img"],
            return_features=True,
            return_rsdt_debug=True,
            return_alignment_debug=True,
        )

        dbg = get_alignment_scale_debug(
            outputs,
            args.align_scale,
        )

        coarse_offset = (
            dbg["coarse_offset"]
            .detach()
            .float()
        )

        total_offset = (
            dbg["total_offset"]
            .detach()
            .float()
        )

        # 真正 Fine 对 Final 的“有效贡献”
        # 不依赖 debug 中 fine_offset 是否已经乘 confidence：
        fine_effective = (
            total_offset
            - coarse_offset
        )

        h, w = total_offset.shape[-2:]

        rgb_box = sample[
            "rgb_semantic_bboxes"
        ]
        tir_box = sample[
            "tir_bboxes"
        ]

        rgb_mask = boxes_to_mask(
            rgb_box,
            h,
            w,
            device,
        )

        tir_mask = boxes_to_mask(
            tir_box,
            h,
            w,
            device,
        )

        # ----------------------------------------------------
        # L0: NO WARP
        # ----------------------------------------------------
        mask_no = tir_mask

        # ----------------------------------------------------
        # L1: COARSE ONLY
        # ----------------------------------------------------
        mask_coarse = warp_plus(
            tir_mask,
            coarse_offset,
        )

        # ----------------------------------------------------
        # L2: FINAL
        # 当前模型真正使用的 total_offset
        # ----------------------------------------------------
        mask_final = warp_plus(
            tir_mask,
            total_offset,
        )

        # ----------------------------------------------------
        # L3: ORACLE TRANSLATION
        # ----------------------------------------------------
        oracle_t = oracle_translation_offset(
            rgb_box,
            tir_box,
            h,
            w,
            device,
        )

        mask_oracle_t = warp_plus(
            tir_mask,
            oracle_t,
        )

        # ----------------------------------------------------
        # L4: ORACLE AFFINE
        # ----------------------------------------------------
        oracle_a = oracle_affine_offset(
            rgb_box,
            tir_box,
            h,
            w,
            device,
        )

        mask_oracle_a = warp_plus(
            tir_mask,
            oracle_a,
        )

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------
        e_no = center_error(
            mask_no,
            rgb_mask,
        )

        e_coarse = center_error(
            mask_coarse,
            rgb_mask,
        )

        e_final = center_error(
            mask_final,
            rgb_mask,
        )

        e_oracle_t = center_error(
            mask_oracle_t,
            rgb_mask,
        )

        e_oracle_a = center_error(
            mask_oracle_a,
            rgb_mask,
        )

        i_no = soft_iou(
            mask_no,
            rgb_mask,
        )

        i_coarse = soft_iou(
            mask_coarse,
            rgb_mask,
        )

        i_final = soft_iou(
            mask_final,
            rgb_mask,
        )

        i_oracle_t = soft_iou(
            mask_oracle_t,
            rgb_mask,
        )

        i_oracle_a = soft_iou(
            mask_oracle_a,
            rgb_mask,
        )

        rcx, rcy, rw, rh = (
            box_xywhn_to_feature(
                rgb_box,
                h,
                w,
            )
        )

        tcx, tcy, tw, th = (
            box_xywhn_to_feature(
                tir_box,
                h,
                w,
            )
        )

        # GT sampling direction for current base+delta convention
        gt_dx = tcx - rcx
        gt_dy = tcy - rcy

        coarse_dx, coarse_dy = (
            masked_mean_offset(
                coarse_offset,
                tir_mask,
            )
        )

        fine_dx, fine_dy = (
            masked_mean_offset(
                fine_effective,
                tir_mask,
            )
        )

        final_dx, final_dy = (
            masked_mean_offset(
                total_offset,
                tir_mask,
            )
        )

        coarse_cos = vector_cosine(
            coarse_dx,
            coarse_dy,
            gt_dx,
            gt_dy,
        )

        final_cos = vector_cosine(
            final_dx,
            final_dy,
            gt_dx,
            gt_dy,
        )

        # Per-sample stage diagnosis
        eps = 1e-4

        if (
            e_coarse + eps < e_no
            and e_final > e_coarse + eps
        ):
            stage_diagnosis = (
                "COARSE_HELPED_FINE_HURT"
            )

        elif (
            e_coarse > e_no + eps
            and e_final >= e_coarse - eps
        ):
            stage_diagnosis = (
                "COARSE_BAD_FINAL_NOT_FIXED"
            )

        elif (
            e_coarse > e_no + eps
            and e_final + eps < e_coarse
        ):
            stage_diagnosis = (
                "COARSE_BAD_FINE_PARTLY_FIXED"
            )

        elif (
            e_coarse + eps < e_no
            and e_final + eps < e_coarse
        ):
            stage_diagnosis = (
                "COARSE_HELPED_FINE_HELPED"
            )

        else:
            stage_diagnosis = (
                "MIXED_OR_SMALL_CHANGE"
            )

        row = {
            "dataset_index":
                idx,

            "pair_key":
                str(sample["pair_key"]),

            "rgb_center_x_p3":
                rcx,

            "rgb_center_y_p3":
                rcy,

            "tir_center_x_p3":
                tcx,

            "tir_center_y_p3":
                tcy,

            "gt_sampling_dx_p3":
                gt_dx,

            "gt_sampling_dy_p3":
                gt_dy,

            "coarse_mean_dx_in_tir_box":
                coarse_dx,

            "coarse_mean_dy_in_tir_box":
                coarse_dy,

            "fine_effective_mean_dx_in_tir_box":
                fine_dx,

            "fine_effective_mean_dy_in_tir_box":
                fine_dy,

            "final_mean_dx_in_tir_box":
                final_dx,

            "final_mean_dy_in_tir_box":
                final_dy,

            "coarse_vs_gt_direction_cosine":
                coarse_cos,

            "final_vs_gt_direction_cosine":
                final_cos,

            "center_error_no_warp":
                e_no,

            "center_error_coarse":
                e_coarse,

            "center_error_final":
                e_final,

            "center_error_oracle_translation":
                e_oracle_t,

            "center_error_oracle_affine":
                e_oracle_a,

            "iou_no_warp":
                i_no,

            "iou_coarse":
                i_coarse,

            "iou_final":
                i_final,

            "iou_oracle_translation":
                i_oracle_t,

            "iou_oracle_affine":
                i_oracle_a,

            "stage_diagnosis":
                stage_diagnosis,
        }

        rows.append(
            row
        )

        print(
            f"[{n:03d}/{len(ids):03d}] "
            f"idx={idx:6d} | "
            f"center: "
            f"no={e_no:5.2f} "
            f"coarse={e_coarse:5.2f} "
            f"final={e_final:5.2f} "
            f"oracleT={e_oracle_t:5.2f} "
            f"oracleA={e_oracle_a:5.2f} | "
            f"{stage_diagnosis}"
        )

    # =========================================================================
    # Summary
    # =========================================================================

    def avg(key):
        vals = [
            float(r[key])
            for r in rows
            if math.isfinite(
                float(r[key])
            )
        ]

        return (
            float(np.mean(vals))
            if vals
            else float("nan")
        )

    def median(key):
        vals = [
            float(r[key])
            for r in rows
            if math.isfinite(
                float(r[key])
            )
        ]

        return (
            float(np.median(vals))
            if vals
            else float("nan")
        )

    stages = {
        "NO_WARP":
            "center_error_no_warp",

        "COARSE":
            "center_error_coarse",

        "FINAL":
            "center_error_final",

        "ORACLE_TRANSLATION":
            "center_error_oracle_translation",

        "ORACLE_AFFINE":
            "center_error_oracle_affine",
    }

    iou_keys = {
        "NO_WARP":
            "iou_no_warp",

        "COARSE":
            "iou_coarse",

        "FINAL":
            "iou_final",

        "ORACLE_TRANSLATION":
            "iou_oracle_translation",

        "ORACLE_AFFINE":
            "iou_oracle_affine",
    }

    mean_center = {
        name: avg(key)
        for name, key in stages.items()
    }

    median_center = {
        name: median(key)
        for name, key in stages.items()
    }

    mean_iou = {
        name: avg(key)
        for name, key in iou_keys.items()
    }

    n = len(rows)

    coarse_improve = sum(
        r["center_error_coarse"]
        <
        r["center_error_no_warp"]
        for r in rows
    )

    final_improve_vs_no = sum(
        r["center_error_final"]
        <
        r["center_error_no_warp"]
        for r in rows
    )

    final_improve_vs_coarse = sum(
        r["center_error_final"]
        <
        r["center_error_coarse"]
        for r in rows
    )

    oracle_t_improve = sum(
        r["center_error_oracle_translation"]
        <
        r["center_error_no_warp"]
        for r in rows
    )

    oracle_a_improve = sum(
        r["center_error_oracle_affine"]
        <
        r["center_error_no_warp"]
        for r in rows
    )

    diagnosis_counts = {}

    for r in rows:
        k = r["stage_diagnosis"]
        diagnosis_counts[k] = (
            diagnosis_counts.get(k, 0)
            + 1
        )

    coarse_cos_mean = avg(
        "coarse_vs_gt_direction_cosine"
    )

    final_cos_mean = avg(
        "final_vs_gt_direction_cosine"
    )

    # =========================================================================
    # Global conclusion
    # =========================================================================

    no_e = mean_center["NO_WARP"]
    coarse_e = mean_center["COARSE"]
    final_e = mean_center["FINAL"]
    oracle_t_e = mean_center["ORACLE_TRANSLATION"]
    oracle_a_e = mean_center["ORACLE_AFFINE"]

    oracle_works = (
        oracle_t_e < no_e * 0.5
        or oracle_a_e < no_e * 0.5
    )

    if oracle_works:
        if (
            coarse_e > no_e
            and final_e >= coarse_e
        ):
            conclusion = (
                "Oracle 显著有效，但 COARSE 已经比 NO_WARP 更差，"
                "且 FINAL 没有把它纠正回来。主要问题位于 Coarse Matcher / "
                "Coarse Offset Predictor；Fine 阶段不是首要矛盾。"
            )

        elif (
            coarse_e < no_e
            and final_e > coarse_e
        ):
            conclusion = (
                "Oracle 显著有效，COARSE 能改善对齐，但 FINAL 反而恶化。"
                "说明 Coarse Matcher 有效，而 Fine Refinement 正在破坏已有对齐；"
                "应优先检查 TIR frequency boundary、fine matcher 和 confidence gate。"
            )

        elif (
            coarse_e > no_e
            and final_e < coarse_e
        ):
            conclusion = (
                "Oracle 显著有效，但 COARSE 先把对齐带坏；Fine 能部分纠正。"
                "说明 Coarse Matcher 是主要问题，同时 Fine 有一定修复能力。"
            )

        elif (
            coarse_e < no_e
            and final_e < coarse_e
        ):
            conclusion = (
                "Coarse 和 Fine 平均都在改善对齐，但仍弱于 Oracle。"
                "说明整体结构可以学习几何对应，只是监督不足或 offset 精度不够。"
            )

        else:
            conclusion = (
                "Oracle 显著有效，但 Coarse/Final 的平均关系不够单一。"
                "请结合 stage_diagnosis 计数与方向 cosine 判断："
                "若 COARSE_BAD 类占多数，先修 coarse；"
                "若 COARSE_HELPED_FINE_HURT 占多数，先修 fine。"
            )

    else:
        conclusion = (
            "Oracle 本身也未显著改善。此时不要先改 Coarse/Fine Predictor，"
            "应先检查 bbox 坐标系、P3映射、dx/dy顺序、offset归一化和 align_corners。"
        )

    # =========================================================================
    # Save
    # =========================================================================

    if args.save_dir:
        out_dir = Path(
            args.save_dir
        ).expanduser().resolve()

    else:
        out_dir = (
            Path(
                args.weights
            )
            .expanduser()
            .resolve()
            .parent
            .parent
            / "coarse_fine_oracle_check"
        )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path = (
        out_dir
        / "coarse_fine_oracle_samples.csv"
    )

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            rows
        )

    summary = {
        "weights":
            str(
                Path(
                    args.weights
                ).expanduser().resolve()
            ),

        "split":
            args.split,

        "num_samples":
            n,

        "align_scale":
            args.align_scale,

        "mean_center_error_p3":
            mean_center,

        "median_center_error_p3":
            median_center,

        "mean_soft_iou":
            mean_iou,

        "samples_improved": {
            "coarse_vs_no_warp":
                coarse_improve,

            "final_vs_no_warp":
                final_improve_vs_no,

            "final_vs_coarse":
                final_improve_vs_coarse,

            "oracle_translation_vs_no_warp":
                oracle_t_improve,

            "oracle_affine_vs_no_warp":
                oracle_a_improve,
        },

        "stage_diagnosis_counts":
            diagnosis_counts,

        "direction_cosine_to_gt": {
            "coarse_mean":
                coarse_cos_mean,

            "final_mean":
                final_cos_mean,
        },

        "conclusion":
            conclusion,
    }

    json_path = (
        out_dir
        / "coarse_fine_oracle_summary.json"
    )

    json_path.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # =========================================================================
    # Print
    # =========================================================================

    print()
    print("=" * 96)
    print("FINAL SUMMARY")
    print("=" * 96)

    print("Mean center error (P3 pixels)")
    print(
        f"  L0 NO WARP            : "
        f"{mean_center['NO_WARP']:.4f}"
    )
    print(
        f"  L1 COARSE             : "
        f"{mean_center['COARSE']:.4f}"
    )
    print(
        f"  L2 FINAL              : "
        f"{mean_center['FINAL']:.4f}"
    )
    print(
        f"  L3 ORACLE TRANSLATION : "
        f"{mean_center['ORACLE_TRANSLATION']:.4f}"
    )
    print(
        f"  L4 ORACLE AFFINE      : "
        f"{mean_center['ORACLE_AFFINE']:.4f}"
    )

    print()
    print("Mean soft IoU")
    print(
        f"  L0 NO WARP            : "
        f"{mean_iou['NO_WARP']:.4f}"
    )
    print(
        f"  L1 COARSE             : "
        f"{mean_iou['COARSE']:.4f}"
    )
    print(
        f"  L2 FINAL              : "
        f"{mean_iou['FINAL']:.4f}"
    )
    print(
        f"  L3 ORACLE TRANSLATION : "
        f"{mean_iou['ORACLE_TRANSLATION']:.4f}"
    )
    print(
        f"  L4 ORACLE AFFINE      : "
        f"{mean_iou['ORACLE_AFFINE']:.4f}"
    )

    print()
    print("Samples improved")
    print(
        f"  COARSE vs NO WARP           : "
        f"{coarse_improve}/{n}"
    )
    print(
        f"  FINAL vs NO WARP            : "
        f"{final_improve_vs_no}/{n}"
    )
    print(
        f"  FINAL vs COARSE             : "
        f"{final_improve_vs_coarse}/{n}"
    )
    print(
        f"  ORACLE TRANSLATION vs NO    : "
        f"{oracle_t_improve}/{n}"
    )
    print(
        f"  ORACLE AFFINE vs NO         : "
        f"{oracle_a_improve}/{n}"
    )

    print()
    print("Stage diagnosis counts")
    for key, value in sorted(
        diagnosis_counts.items(),
        key=lambda kv: (-kv[1], kv[0]),
    ):
        print(
            f"  {key:<34}: "
            f"{value}/{n}"
        )

    print()
    print("Direction cosine to GT sampling offset")
    print(
        f"  COARSE mean cosine : "
        f"{coarse_cos_mean:.4f}"
    )
    print(
        f"  FINAL  mean cosine : "
        f"{final_cos_mean:.4f}"
    )
    print(
        "  解释：+1=方向一致，0=无关，-1=方向相反"
    )

    print()
    print("结论：")
    print(
        conclusion
    )

    print()
    print(
        f"CSV : {csv_path}"
    )
    print(
        f"JSON: {json_path}"
    )
    print("=" * 96)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Five-level FBAM diagnosis: "
            "No Warp -> Coarse -> Final -> Oracle Translation -> Oracle Affine"
        )
    )

    parser.add_argument(
        "--weights",
        type=str,
        default=(
            "/mnt/sda/taochangyong/Projects/Model/YOLO-tcy/"
            "runs/fbam_bhlr/"
            "yolo26n_UAVCB_fbamP3_bhlrP3_rgb1280_sem640_tir640_seed0/"
            "weights/best.pt"
        ),
    )

    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=[
            "train",
            "val",
            "test",
        ],
    )

    parser.add_argument(
        "--device",
        type=str,
        default="5",
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--align-scale",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
    )

    run(
        parser.parse_args()
    )
