"""
Full-feature RSD-T validator for YOLO-tcy
========================================

Based on the feature set of the existing full val_rgbt.py, while keeping
val_rgbt.py unchanged.

Preserved validation features
-----------------------------
- P / R / AP50 / AP75 / AP50-95
- YOLO26 end-to-end and YOLO11 NMS postprocessing
- Params(M)
- FLOPs(G)
- batch=1 model FPS / latency
- preprocess / validation inference / postprocess timing
- end-to-end FPS
- optional PR/F1/P/R plots via DetMetrics
- prediction preview grid
- ground-truth preview grid
- per-class table
- metrics.json
- paper_metrics.csv
- dataset / resolution CLI overrides
- split train / val / test

RSD-T-specific changes
----------------------
- 3 model inputs:
      RGB high + TIR + RGB semantic
- GT evaluation in RGB semantic coordinates
- preview image uses RGB semantic view so boxes are geometrically correct
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from tqdm import tqdm

from ultralytics.utils.metrics import (
    DetMetrics,
)

import val_rgbt as base_val

from datasets.rsdt_dataset import (
    build_rsdt_dataset,
    build_rsdt_dataloader,
)

from models.rgbt_rsdt_model import (
    RGBTRSDTDetectionModel,
)

from train_rsdt import (
    preprocess_batch_rsdt,
    prepare_rsdt_ground_truth,
)


# ===========================================================================
# 1. Load RSD-T checkpoint
# ===========================================================================

def load_rsdt_checkpoint(
    checkpoint_path: str,
    device: torch.device,
):
    checkpoint_path = (
        Path(
            checkpoint_path
        )
        .expanduser()
        .resolve()
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            "\n权重不存在:\n"
            f"{checkpoint_path}"
        )

    print(
        "\nLoading RSD-T checkpoint:"
    )
    print(
        checkpoint_path
    )

    ckpt = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    if (
        "model_state_dict"
        not in ckpt
    ):
        raise KeyError(
            "\nCheckpoint 中没有 model_state_dict。\n"
        )

    cfg = ckpt.get(
        "config",
        {},
    )

    model_cfg = cfg.get(
        "model",
        {},
    )

    data_cfg = cfg.get(
        "data",
        {},
    )

    model = RGBTRSDTDetectionModel(
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

        pretrained=False,

        fusion=ckpt.get(
            "fusion",
            model_cfg.get(
                "fusion",
                "concat",
            ),
        ),

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

        semantic_imgsz=int(
            data_cfg.get(
                "rgb_semantic_imgsz",
                640,
            )
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

        verbose=False,
    )

    model.load_state_dict(
        ckpt[
            "model_state_dict"
        ],
        strict=True,
    )

    model = model.to(
        device
    )

    model.eval()

    return (
        model,
        ckpt,
        cfg,
    )


# ===========================================================================
# 2. Params / FLOPs
# ===========================================================================

def get_params_m(
    model: torch.nn.Module,
) -> float:
    return (
        sum(
            p.numel()
            for p in model.parameters()
        )
        / 1e6
    )


def get_flops_g_rsdt(
    model: torch.nn.Module,
    rgb_imgsz: int,
    rgb_semantic_imgsz: int,
    tir_imgsz: int,
    device: torch.device,
) -> float:
    """
    FLOPs = MACs * 2
    """

    try:
        import thop

    except ImportError:
        print(
            "\nWARNING: 未安装 thop，FLOPs 将记为 0。"
        )
        print(
            "安装命令："
        )
        print(
            "python -m pip install ultralytics-thop "
            "-i https://mirrors.aliyun.com/pypi/simple/"
        )
        return 0.0

    was_training = (
        model.training
    )
    model.eval()

    rgb_high = torch.zeros(
        1,
        3,
        rgb_imgsz,
        rgb_imgsz,
        device=device,
        dtype=torch.float32,
    )

    tir = torch.zeros(
        1,
        3,
        tir_imgsz,
        tir_imgsz,
        device=device,
        dtype=torch.float32,
    )

    rgb_semantic = torch.zeros(
        1,
        3,
        rgb_semantic_imgsz,
        rgb_semantic_imgsz,
        device=device,
        dtype=torch.float32,
    )

    try:
        with torch.no_grad():
            macs, _ = thop.profile(
                model,
                inputs=(
                    rgb_high,
                    tir,
                    rgb_semantic,
                ),
                verbose=False,
            )

        flops_g = (
            float(macs)
            * 2.0
            / 1e9
        )

    except Exception as e:
        print(
            "\nWARNING: RSD-T FLOPs 计算失败："
        )
        print(
            e
        )
        flops_g = 0.0

    model.train(
        was_training
    )

    del rgb_high
    del tir
    del rgb_semantic

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return flops_g


def get_model_complexity_rsdt(
    model,
    rgb_imgsz,
    rgb_semantic_imgsz,
    tir_imgsz,
    device,
):
    params_m = get_params_m(
        model
    )

    flops_g = get_flops_g_rsdt(
        model=model,
        rgb_imgsz=rgb_imgsz,
        rgb_semantic_imgsz=(
            rgb_semantic_imgsz
        ),
        tir_imgsz=tir_imgsz,
        device=device,
    )

    return (
        params_m,
        flops_g,
    )


# ===========================================================================
# 3. Standardized batch=1 FPS
# ===========================================================================

@torch.inference_mode()
def benchmark_fps_rsdt(
    model,
    rgb_imgsz,
    rgb_semantic_imgsz,
    tir_imgsz,
    device,
    amp=True,
    warmup=20,
    iterations=100,
):
    """
    Batch=1 model-forward benchmark.

    Included:
        prepared RGB-high tensor
        prepared TIR tensor
        prepared RGB-semantic tensor
        model forward

    Excluded:
        disk I/O
        cv2 preprocessing
        postprocess / NMS
    """

    model.eval()

    rgb_high = torch.randn(
        1,
        3,
        rgb_imgsz,
        rgb_imgsz,
        device=device,
    )

    tir = torch.randn(
        1,
        3,
        tir_imgsz,
        tir_imgsz,
        device=device,
    )

    rgb_semantic = torch.randn(
        1,
        3,
        rgb_semantic_imgsz,
        rgb_semantic_imgsz,
        device=device,
    )

    with base_val.autocast_context(
        device,
        amp,
    ):
        for _ in range(
            max(
                warmup,
                0,
            )
        ):
            _ = model(
                rgb_high,
                tir,
                rgb_semantic,
            )

    base_val.synchronize(
        device
    )

    start = time.perf_counter()

    with base_val.autocast_context(
        device,
        amp,
    ):
        for _ in range(
            max(
                iterations,
                1,
            )
        ):
            _ = model(
                rgb_high,
                tir,
                rgb_semantic,
            )

    base_val.synchronize(
        device
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    iterations = max(
        iterations,
        1,
    )

    latency_s = (
        elapsed
        / iterations
    )

    latency_ms = (
        latency_s
        * 1000.0
    )

    fps = (
        1.0 / latency_s
        if latency_s > 0
        else 0.0
    )

    del rgb_high
    del tir
    del rgb_semantic

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return (
        fps,
        latency_ms,
    )


# ===========================================================================
# 4. Paper CSV
# ===========================================================================

def save_metrics_csv_rsdt(
    path: Path,
    row: Dict,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "Model",
        "Params(M)",
        "FLOPs(G)",
        "FPS",
        "P(%)",
        "R(%)",
        "AP50(%)",
        "AP75(%)",
        "AP50:95(%)",
        "RGB_high_imgsz",
        "RGB_semantic_imgsz",
        "TIR_imgsz",
        "Split",
    ]

    with path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerow(
            {
                key: row.get(
                    key,
                    "",
                )
                for key in fieldnames
            }
        )


# ===========================================================================
# 5. Main full validation
# ===========================================================================

@torch.inference_mode()
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
    # -----------------------------------------------------------------------
    # Device / checkpoint
    # -----------------------------------------------------------------------

    device = base_val.select_device(
        device_str
    )

    (
        model,
        ckpt,
        cfg,
    ) = load_rsdt_checkpoint(
        checkpoint,
        device,
    )

    data_cfg = cfg.get(
        "data",
        {},
    )

    if rgb_yaml is None:
        rgb_yaml = data_cfg.get(
            "rgb"
        )

    if tir_yaml is None:
        tir_yaml = data_cfg.get(
            "tir"
        )

    if rgb_imgsz is None:
        rgb_imgsz = int(
            data_cfg.get(
                "rgb_imgsz",
                data_cfg.get(
                    "rgb_high_imgsz",
                    1280,
                ),
            )
        )

    if rgb_semantic_imgsz is None:
        rgb_semantic_imgsz = int(
            data_cfg.get(
                "rgb_semantic_imgsz",
                640,
            )
        )

    if tir_imgsz is None:
        tir_imgsz = int(
            data_cfg.get(
                "tir_imgsz",
                640,
            )
        )

    if pair_mode is None:
        pair_mode = data_cfg.get(
            "pair_mode",
            "relative",
        )

    if rgb_yaml is None:
        raise ValueError(
            "无法确定 RGB dataset YAML。"
        )

    if tir_yaml is None:
        raise ValueError(
            "无法确定 TIR dataset YAML。"
        )

    # -----------------------------------------------------------------------
    # Save directory / previews
    # -----------------------------------------------------------------------

    checkpoint_path = (
        Path(
            checkpoint
        )
        .expanduser()
        .resolve()
    )

    if save_dir is None:
        save_dir = (
            checkpoint_path
            .parent
            .parent
            / f"val_{split}"
        )
    else:
        save_dir = (
            Path(
                save_dir
            )
            .expanduser()
            .resolve()
        )

    save_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    preview_dir = (
        save_dir
        / "previews"
    )

    preview_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    preview_num = max(
        int(
            preview_num
        ),
        0,
    )

    preview_cols = max(
        int(
            preview_cols
        ),
        1,
    )

    prediction_preview_tiles = []
    ground_truth_preview_tiles = []
    preview_saved = 0

    preview_paths = {
        "prediction": None,
        "ground_truth": None,
    }

    # -----------------------------------------------------------------------
    # Complexity
    # -----------------------------------------------------------------------

    print(
        "\nCalculating RSD-T model complexity..."
    )

    (
        params_m,
        flops_g,
    ) = get_model_complexity_rsdt(
        model=model,
        rgb_imgsz=rgb_imgsz,
        rgb_semantic_imgsz=(
            rgb_semantic_imgsz
        ),
        tir_imgsz=tir_imgsz,
        device=device,
    )

    # -----------------------------------------------------------------------
    # Dataset
    # -----------------------------------------------------------------------

    dataset = build_rsdt_dataset(
        rgb_yaml=rgb_yaml,
        tir_yaml=tir_yaml,
        split=split,
        rgb_high_imgsz=rgb_imgsz,
        rgb_semantic_imgsz=(
            rgb_semantic_imgsz
        ),
        tir_imgsz=tir_imgsz,
        pair_mode=pair_mode,
        augment=False,
        tir_channels=3,
        strict_pair=True,
    )

    loader = build_rsdt_dataloader(
        dataset,
        batch_size=batch_size,
        workers=workers,
        shuffle=False,
        pin_memory=(
            device.type
            == "cuda"
        ),
    )

    # -----------------------------------------------------------------------
    # Info
    # -----------------------------------------------------------------------

    print(
        "\n"
        "============================================================"
    )
    print(
        "RGB-T + RSD-T v1 Validation"
    )
    print(
        "============================================================"
    )
    print(
        f"Model          : {model.model_name}"
    )
    print(
        f"Fusion         : {model.fusion_type}"
    )
    print(
        f"Checkpoint     : {checkpoint_path}"
    )
    print(
        f"Split          : {split}"
    )
    print(
        f"RGB high       : {rgb_imgsz}"
    )
    print(
        f"RGB semantic   : {rgb_semantic_imgsz}"
    )
    print(
        f"TIR imgsz      : {tir_imgsz}"
    )
    print(
        f"Batch          : {batch_size}"
    )
    print(
        f"Workers        : {workers}"
    )
    print(
        f"Device         : {device}"
    )

    if device.type == "cuda":
        print(
            f"GPU            : "
            f"{torch.cuda.get_device_name(device)}"
        )

    print(
        f"AMP            : {amp}"
    )
    print(
        f"Confidence     : {conf_thres}"
    )
    print(
        f"NMS IoU        : {iou_thres}"
    )
    print(
        f"Max det        : {max_det}"
    )
    print(
        f"Params         : {params_m:.3f} M"
    )
    print(
        f"FLOPs          : {flops_g:.3f} G"
    )

    if hasattr(
        model,
        "rsdt",
    ):
        state = (
            model.rsdt.scalar_state()
        )
        print(
            f"RSD-T alpha    : {state['alpha']:.6f}"
        )
        print(
            f"RSD-T gamma    : {state['gamma']:.6f}"
        )

    print(
        "============================================================\n"
    )

    # -----------------------------------------------------------------------
    # Metrics
    # -----------------------------------------------------------------------

    names = model.names

    if not isinstance(
        names,
        dict,
    ):
        names = {
            i: name
            for i, name
            in enumerate(names)
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

    preprocess_time = 0.0
    inference_time = 0.0
    postprocess_time = 0.0

    # -----------------------------------------------------------------------
    # Warmup
    # -----------------------------------------------------------------------

    warmup_rgb = torch.zeros(
        1,
        3,
        rgb_imgsz,
        rgb_imgsz,
        device=device,
    )

    warmup_tir = torch.zeros(
        1,
        3,
        tir_imgsz,
        tir_imgsz,
        device=device,
    )

    warmup_semantic = torch.zeros(
        1,
        3,
        rgb_semantic_imgsz,
        rgb_semantic_imgsz,
        device=device,
    )

    with base_val.autocast_context(
        device,
        amp,
    ):
        for _ in range(3):
            _ = model(
                warmup_rgb,
                warmup_tir,
                warmup_semantic,
            )

    base_val.synchronize(
        device
    )

    del warmup_rgb
    del warmup_tir
    del warmup_semantic

    # -----------------------------------------------------------------------
    # Validation
    # -----------------------------------------------------------------------

    pbar = tqdm(
        loader,
        total=len(loader),
        desc=(
            f"{'Class':>12} "
            f"{'Images':>8} "
            f"{'Instances':>10}"
        ),
        dynamic_ncols=True,
        leave=True,
    )

    for batch in pbar:

        batch_size_current = int(
            batch["rgb_img"].shape[0]
        )

        seen += batch_size_current

        total_instances += int(
            batch["rgb_cls"].shape[0]
        )

        # ----------------------------------------------------
        # Preprocess timing
        # ----------------------------------------------------

        base_val.synchronize(
            device
        )
        t0 = time.perf_counter()

        batch = preprocess_batch_rsdt(
            batch,
            device,
        )

        base_val.synchronize(
            device
        )
        t1 = time.perf_counter()

        preprocess_time += (
            t1 - t0
        )

        # ----------------------------------------------------
        # Inference timing
        # ----------------------------------------------------

        base_val.synchronize(
            device
        )
        t2 = time.perf_counter()

        with base_val.autocast_context(
            device,
            amp,
        ):
            preds = model(
                batch["rgb_img"],
                batch["tir_img"],
                batch[
                    "rgb_semantic_img"
                ],
            )

        base_val.synchronize(
            device
        )
        t3 = time.perf_counter()

        inference_time += (
            t3 - t2
        )

        # ----------------------------------------------------
        # Postprocess timing
        # ----------------------------------------------------

        base_val.synchronize(
            device
        )
        t4 = time.perf_counter()

        preds = (
            base_val.postprocess_predictions(
                preds=preds,
                model=model,
                conf_thres=conf_thres,
                iou_thres=iou_thres,
                max_det=max_det,
            )
        )

        base_val.synchronize(
            device
        )
        t5 = time.perf_counter()

        postprocess_time += (
            t5 - t4
        )

        # ----------------------------------------------------
        # Per-image
        # ----------------------------------------------------

        for sample_index, pred in enumerate(
            preds
        ):

            gt = prepare_rsdt_ground_truth(
                batch,
                sample_index,
            )

            # ------------------------------------------------
            # Preview:
            # use SEMANTIC RGB because prediction/GT coords
            # are in semantic coordinate space.
            # ------------------------------------------------

            if (
                preview_saved
                < preview_num
                and gt[
                    "bboxes"
                ].shape[0] > 0
            ):
                image_bgr = (
                    base_val.tensor_to_bgr(
                        batch[
                            "rgb_semantic_img"
                        ][sample_index]
                    )
                )

                image_name = Path(
                    batch[
                        "rgb_path"
                    ][sample_index]
                ).name

                prediction_preview_tiles.append(
                    base_val.draw_prediction_tile(
                        image_bgr=image_bgr,
                        pred=pred,
                        image_name=image_name,
                        preview_conf=preview_conf,
                    )
                )

                ground_truth_preview_tiles.append(
                    base_val.draw_ground_truth_tile(
                        image_bgr=image_bgr,
                        gt=gt,
                        image_name=image_name,
                    )
                )

                preview_saved += 1

            tp = (
                base_val.process_single_image(
                    pred=pred,
                    gt=gt,
                    iouv=iouv,
                )
            )

            target_cls = (
                gt["cls"]
                .detach()
                .float()
                .cpu()
                .numpy()
            )

            no_pred = (
                pred["cls"].shape[0]
                == 0
            )

            if no_pred:
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

        gpu_mem = 0.0

        if device.type == "cuda":
            gpu_mem = (
                torch.cuda.memory_reserved(
                    device
                )
                / (1024 ** 3)
            )

        pbar.set_postfix(
            {
                "Images": seen,
                "Instances": (
                    total_instances
                ),
                "GPU": (
                    f"{gpu_mem:.2f}G"
                ),
            }
        )

    pbar.close()

    # -----------------------------------------------------------------------
    # Save previews
    # -----------------------------------------------------------------------

    if preview_saved > 0:
        preview_paths = (
            base_val.save_preview_grids(
                prediction_tiles=(
                    prediction_preview_tiles
                ),
                ground_truth_tiles=(
                    ground_truth_preview_tiles
                ),
                preview_dir=preview_dir,
                split=split,
                preview_conf=preview_conf,
                cols=preview_cols,
            )
        )

    # -----------------------------------------------------------------------
    # Process metrics / optional plots
    # -----------------------------------------------------------------------

    metrics.process(
        save_dir=save_dir,
        plot=plots,
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

    precision_pct = (
        precision * 100.0
    )
    recall_pct = (
        recall * 100.0
    )
    ap50_pct = (
        map50 * 100.0
    )
    ap75_pct = (
        map75 * 100.0
    )
    ap5095_pct = (
        map5095 * 100.0
    )

    # -----------------------------------------------------------------------
    # Pipeline timing
    # -----------------------------------------------------------------------

    preprocess_ms = (
        preprocess_time
        / max(
            seen,
            1,
        )
        * 1000.0
    )

    inference_ms_val = (
        inference_time
        / max(
            seen,
            1,
        )
        * 1000.0
    )

    postprocess_ms = (
        postprocess_time
        / max(
            seen,
            1,
        )
        * 1000.0
    )

    e2e_ms = (
        preprocess_ms
        + inference_ms_val
        + postprocess_ms
    )

    e2e_fps = (
        1000.0 / e2e_ms
        if e2e_ms > 0
        else 0.0
    )

    print(
        "\nBenchmarking batch=1 RSD-T model FPS..."
    )

    (
        fps,
        latency_ms,
    ) = benchmark_fps_rsdt(
        model=model,
        rgb_imgsz=rgb_imgsz,
        rgb_semantic_imgsz=(
            rgb_semantic_imgsz
        ),
        tir_imgsz=tir_imgsz,
        device=device,
        amp=amp,
        warmup=speed_warmup,
        iterations=speed_iters,
    )

    # -----------------------------------------------------------------------
    # Main paper-style table
    # -----------------------------------------------------------------------

    model_display_name = (
        f"{model.model_name}-"
        f"{model.fusion_type}-RSDT"
    )

    print(
        "\n"
        "=============================================================================================================="
    )

    print(
        f"{'Model':<20}"
        f"{'Params(M)':>12}"
        f"{'FLOPs(G)':>12}"
        f"{'FPS':>11}"
        f"{'P(%)':>10}"
        f"{'R(%)':>10}"
        f"{'AP50(%)':>12}"
        f"{'AP75(%)':>12}"
        f"{'AP50:95(%)':>15}"
    )

    print(
        "--------------------------------------------------------------------------------------------------------------"
    )

    print(
        f"{model_display_name:<20}"
        f"{params_m:>12.3f}"
        f"{flops_g:>12.3f}"
        f"{fps:>11.2f}"
        f"{precision_pct:>10.2f}"
        f"{recall_pct:>10.2f}"
        f"{ap50_pct:>12.2f}"
        f"{ap75_pct:>12.2f}"
        f"{ap5095_pct:>15.2f}"
    )

    print(
        "=============================================================================================================="
    )

    # -----------------------------------------------------------------------
    # Dataset info
    # -----------------------------------------------------------------------

    print(
        "\nDataset:"
    )
    print(
        f"  Images            : {seen}"
    )
    print(
        f"  UAV Instances     : {total_instances}"
    )
    print(
        f"  RGB High          : {rgb_imgsz}"
    )
    print(
        f"  RGB Semantic      : {rgb_semantic_imgsz}"
    )
    print(
        f"  TIR Resolution    : {tir_imgsz}"
    )

    # -----------------------------------------------------------------------
    # Speed
    # -----------------------------------------------------------------------

    print(
        "\nSpeed:"
    )
    print(
        f"  Preprocess        : "
        f"{preprocess_ms:.3f} ms/image"
    )
    print(
        f"  Val inference     : "
        f"{inference_ms_val:.3f} ms/image "
        f"(batch={batch_size})"
    )
    print(
        f"  Postprocess       : "
        f"{postprocess_ms:.3f} ms/image"
    )
    print(
        f"  End-to-end        : "
        f"{e2e_ms:.3f} ms/image"
    )
    print(
        f"  End-to-end FPS    : "
        f"{e2e_fps:.2f}"
    )
    print(
        f"  Model latency     : "
        f"{latency_ms:.3f} ms/pair "
        f"(batch=1)"
    )
    print(
        f"  Paper FPS         : "
        f"{fps:.2f} "
        f"(batch=1, model forward only)"
    )

    # -----------------------------------------------------------------------
    # Per-class
    # -----------------------------------------------------------------------

    if len(
        metrics.ap_class_index
    ):
        print(
            "\nPer-class:"
        )

        print(
            f"{'Class':<15}"
            f"{'Images':>10}"
            f"{'Instances':>12}"
            f"{'P(%)':>10}"
            f"{'R(%)':>10}"
            f"{'AP50(%)':>12}"
            f"{'AP75(%)':>12}"
            f"{'AP50:95(%)':>15}"
        )

        print(
            "-" * 96
        )

        for metric_i, class_id in enumerate(
            metrics.ap_class_index
        ):
            class_id = int(
                class_id
            )

            (
                class_p,
                class_r,
                class_ap50,
                class_ap5095,
            ) = metrics.class_result(
                metric_i
            )

            class_ap75 = float(
                metrics.box.all_ap[
                    metric_i,
                    5,
                ]
            )

            class_name = names.get(
                class_id,
                str(
                    class_id
                ),
            )

            class_instances = 0
            class_images = 0

            if (
                metrics.nt_per_class
                is not None
                and class_id
                < len(
                    metrics.nt_per_class
                )
            ):
                class_instances = int(
                    metrics.nt_per_class[
                        class_id
                    ]
                )

            if (
                metrics.nt_per_image
                is not None
                and class_id
                < len(
                    metrics.nt_per_image
                )
            ):
                class_images = int(
                    metrics.nt_per_image[
                        class_id
                    ]
                )

            print(
                f"{class_name:<15}"
                f"{class_images:>10}"
                f"{class_instances:>12}"
                f"{float(class_p) * 100:>10.2f}"
                f"{float(class_r) * 100:>10.2f}"
                f"{float(class_ap50) * 100:>12.2f}"
                f"{class_ap75 * 100:>12.2f}"
                f"{float(class_ap5095) * 100:>15.2f}"
            )

    # -----------------------------------------------------------------------
    # RSD-T state
    # -----------------------------------------------------------------------

    rsdt_state = (
        model.rsdt.scalar_state()
        if hasattr(
            model,
            "rsdt",
        )
        else {}
    )

    # -----------------------------------------------------------------------
    # Results JSON
    # -----------------------------------------------------------------------

    results = {
        "checkpoint": str(
            checkpoint_path
        ),
        "model": model.model_name,
        "fusion": model.fusion_type,
        "model_display_name": (
            model_display_name
        ),
        "split": split,

        "images": int(
            seen
        ),
        "instances": int(
            total_instances
        ),

        "rgb_imgsz": int(
            rgb_imgsz
        ),
        "rgb_semantic_imgsz": int(
            rgb_semantic_imgsz
        ),
        "tir_imgsz": int(
            tir_imgsz
        ),

        "rsdt": {
            "use_guidance": bool(
                getattr(
                    model,
                    "use_guidance",
                    True,
                )
            ),
            "alpha": rsdt_state.get(
                "alpha"
            ),
            "gamma": rsdt_state.get(
                "gamma"
            ),
        },

        "complexity": {
            "params_m": float(
                params_m
            ),
            "flops_g": float(
                flops_g
            ),
        },

        "validation": {
            "conf": float(
                conf_thres
            ),
            "nms_iou": float(
                iou_thres
            ),
            "max_det": int(
                max_det
            ),
            "batch": int(
                batch_size
            ),
            "amp": bool(
                amp
            ),
        },

        "metrics": {
            "precision": precision,
            "recall": recall,
            "AP50": map50,
            "AP75": map75,
            "AP50-95": map5095,
        },

        "paper_metrics": {
            "Params(M)": float(
                params_m
            ),
            "FLOPs(G)": float(
                flops_g
            ),
            "FPS": float(
                fps
            ),
            "P(%)": float(
                precision_pct
            ),
            "R(%)": float(
                recall_pct
            ),
            "AP50(%)": float(
                ap50_pct
            ),
            "AP75(%)": float(
                ap75_pct
            ),
            "AP50:95(%)": float(
                ap5095_pct
            ),
        },

        "previews": {
            "num_samples": int(
                preview_saved
            ),
            "preview_conf": float(
                preview_conf
            ),
            "prediction_grid": (
                str(
                    preview_paths[
                        "prediction"
                    ]
                )
                if preview_paths[
                    "prediction"
                ] is not None
                else None
            ),
            "ground_truth_grid": (
                str(
                    preview_paths[
                        "ground_truth"
                    ]
                )
                if preview_paths[
                    "ground_truth"
                ] is not None
                else None
            ),
        },

        "speed": {
            "preprocess_ms": float(
                preprocess_ms
            ),
            "validation_inference_ms": float(
                inference_ms_val
            ),
            "postprocess_ms": float(
                postprocess_ms
            ),
            "end_to_end_ms": float(
                e2e_ms
            ),
            "end_to_end_fps": float(
                e2e_fps
            ),
            "batch1_model_latency_ms": float(
                latency_ms
            ),
            "batch1_model_fps": float(
                fps
            ),
            "speed_warmup": int(
                speed_warmup
            ),
            "speed_iterations": int(
                speed_iters
            ),
        },
    }

    metrics_json = (
        save_dir
        / "metrics.json"
    )

    with metrics_json.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            results,
            f,
            ensure_ascii=False,
            indent=2,
        )

    paper_row = {
        "Model": model_display_name,
        "Params(M)": round(
            params_m,
            3,
        ),
        "FLOPs(G)": round(
            flops_g,
            3,
        ),
        "FPS": round(
            fps,
            2,
        ),
        "P(%)": round(
            precision_pct,
            2,
        ),
        "R(%)": round(
            recall_pct,
            2,
        ),
        "AP50(%)": round(
            ap50_pct,
            2,
        ),
        "AP75(%)": round(
            ap75_pct,
            2,
        ),
        "AP50:95(%)": round(
            ap5095_pct,
            2,
        ),
        "RGB_high_imgsz": (
            rgb_imgsz
        ),
        "RGB_semantic_imgsz": (
            rgb_semantic_imgsz
        ),
        "TIR_imgsz": tir_imgsz,
        "Split": split,
    }

    save_metrics_csv_rsdt(
        save_dir
        / "paper_metrics.csv",
        paper_row,
    )

    print(
        "\n"
        "============================================================"
    )
    print(
        "RSD-T validation completed."
    )
    print(
        f"Results directory : {save_dir}"
    )
    print(
        f"JSON              : {metrics_json}"
    )
    print(
        f"Paper CSV         : "
        f"{save_dir / 'paper_metrics.csv'}"
    )
    print(
        f"Preview samples   : {preview_saved}"
    )
    print(
        f"Prediction grid   : "
        f"{preview_paths['prediction']}"
    )
    print(
        f"Ground-truth grid : "
        f"{preview_paths['ground_truth']}"
    )
    print(
        "============================================================\n"
    )

    return results


# ===========================================================================
# 6. CLI
# ===========================================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Full-feature validation for "
            "RGB-T + RSD-T v1"
        )
    )

    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help=(
            "RSD-T custom best.pt / last.pt"
        ),
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
        help=(
            "Optional RGB dataset YAML override"
        ),
    )

    parser.add_argument(
        "--tir",
        type=str,
        default=None,
        help=(
            "Optional TIR dataset YAML override"
        ),
    )

    # Keep the same old CLI meaning:
    # --rgb-imgsz now means high-resolution RGB.
    parser.add_argument(
        "--rgb-imgsz",
        type=int,
        default=None,
        help=(
            "High-resolution RGB size"
        ),
    )

    parser.add_argument(
        "--rgb-high-imgsz",
        dest="rgb_imgsz",
        type=int,
        default=None,
        help=(
            "Alias of --rgb-imgsz"
        ),
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
        help=(
            "Validation confidence threshold"
        ),
    )

    parser.add_argument(
        "--iou",
        type=float,
        default=0.7,
        help=(
            "NMS IoU threshold"
        ),
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
        help="Disable AMP",
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
        help=(
            "Save PR/F1/P/R curves"
        ),
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
