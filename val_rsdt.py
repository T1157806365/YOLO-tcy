"""
Validate isolated RGB-T + RSD-T v1
==================================

Original val_rgbt.py remains untouched.

Run:
    python val_rsdt.py \
        --weights runs/rsdt/.../weights/best.pt \
        --split val \
        --device 0
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from tqdm import tqdm

from ultralytics.utils import ops
from ultralytics.utils.metrics import (
    DetMetrics,
)

from datasets.rsdt_dataset import (
    build_rsdt_dataset,
    build_rsdt_dataloader,
)

from models.rgbt_rsdt_model import (
    RGBTRSDTDetectionModel,
)

from train_rsdt import (
    preprocess_batch_rsdt,
)

# Reuse stable evaluation helpers from the original validator.
from val_rgbt import (
    select_device,
    autocast_context,
    synchronize,
    postprocess_predictions,
    process_single_image,
)


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
            f"Checkpoint not found:\n"
            f"{checkpoint_path}"
        )

    ckpt = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
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

    if (
        "model_state_dict"
        not in ckpt
    ):
        raise KeyError(
            "Checkpoint has no model_state_dict."
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
    rgb_high_imgsz: int,
    rgb_semantic_imgsz: int,
    tir_imgsz: int,
    device: torch.device,
) -> float:
    """
    Three-input RSD-T FLOPs.
    Convention matches the original validator:
        FLOPs = MACs * 2
    """

    try:
        import thop

    except ImportError:

        print(
            "WARNING: thop not installed; "
            "FLOPs will be 0."
        )

        return 0.0

    was_training = (
        model.training
    )

    model.eval()

    rgb_high = torch.zeros(
        1,
        3,
        rgb_high_imgsz,
        rgb_high_imgsz,
        device=device,
    )

    rgb_semantic = torch.zeros(
        1,
        3,
        rgb_semantic_imgsz,
        rgb_semantic_imgsz,
        device=device,
    )

    tir = torch.zeros(
        1,
        3,
        tir_imgsz,
        tir_imgsz,
        device=device,
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
            "WARNING: RSD-T FLOPs "
            f"calculation failed: {e}"
        )

        flops_g = 0.0

    model.train(
        was_training
    )

    return flops_g


@torch.inference_mode()
def benchmark_fps_rsdt(
    model: torch.nn.Module,
    rgb_high_imgsz: int,
    rgb_semantic_imgsz: int,
    tir_imgsz: int,
    device: torch.device,
    amp: bool = True,
    warmup: int = 20,
    iterations: int = 100,
):
    """
    Batch=1 pure model-forward speed.

    Inputs are already prepared tensors:
        RGB high + RGB semantic + TIR

    Dataset I/O / cv2 LetterBox / NMS are excluded.
    """

    model.eval()

    rgb_high = torch.randn(
        1,
        3,
        rgb_high_imgsz,
        rgb_high_imgsz,
        device=device,
    )

    rgb_semantic = torch.randn(
        1,
        3,
        rgb_semantic_imgsz,
        rgb_semantic_imgsz,
        device=device,
    )

    tir = torch.randn(
        1,
        3,
        tir_imgsz,
        tir_imgsz,
        device=device,
    )

    with autocast_context(
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

    synchronize(
        device
    )

    start = time.perf_counter()

    with autocast_context(
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

    synchronize(
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

    latency_ms = (
        elapsed
        / iterations
        * 1000.0
    )

    fps = (
        1000.0
        / latency_ms
        if latency_ms > 0
        else 0.0
    )

    return (
        fps,
        latency_ms,
    )


def prepare_rsdt_ground_truth(
    batch: Dict,
    sample_index: int,
):
    """
    Prediction is in semantic RGB coordinates.
    """

    mask = (
        batch[
            "rgb_batch_idx"
        ]
        == sample_index
    )

    cls = (
        batch[
            "rgb_cls"
        ][mask]
        .squeeze(-1)
    )

    boxes = (
        batch[
            "rgb_semantic_bboxes"
        ][mask]
    )

    h = int(
        batch[
            "rgb_semantic_img"
        ].shape[2]
    )

    w = int(
        batch[
            "rgb_semantic_img"
        ].shape[3]
    )

    if boxes.shape[0] > 0:

        boxes = ops.xywh2xyxy(
            boxes
        )

        scale = torch.tensor(
            [
                w,
                h,
                w,
                h,
            ],
            device=boxes.device,
            dtype=boxes.dtype,
        )

        boxes = (
            boxes
            * scale
        )

    return {
        "cls":
            cls,

        "bboxes":
            boxes,
    }


@torch.inference_mode()
def validate(
    checkpoint: str,
    split: str = "val",
    device_str: str = "0",
    batch_size: int = 4,
    workers: int = 4,
    conf_thres: float = 0.001,
    iou_thres: float = 0.7,
    max_det: int = 300,
    amp: bool = True,
    speed_warmup: int = 20,
    speed_iters: int = 100,
):
    device = select_device(
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

    rgb_yaml = data_cfg[
        "rgb"
    ]

    tir_yaml = data_cfg[
        "tir"
    ]

    rgb_high_imgsz = int(
        data_cfg.get(
            "rgb_high_imgsz",
            1280,
        )
    )

    rgb_semantic_imgsz = int(
        data_cfg.get(
            "rgb_semantic_imgsz",
            640,
        )
    )

    tir_imgsz = int(
        data_cfg.get(
            "tir_imgsz",
            640,
        )
    )

    pair_mode = data_cfg.get(
        "pair_mode",
        "relative",
    )

    dataset = build_rsdt_dataset(
        rgb_yaml=rgb_yaml,
        tir_yaml=tir_yaml,
        split=split,
        rgb_high_imgsz=(
            rgb_high_imgsz
        ),
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

    params_m = get_params_m(
        model
    )

    flops_g = get_flops_g_rsdt(
        model,
        rgb_high_imgsz,
        rgb_semantic_imgsz,
        tir_imgsz,
        device,
    )

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
        for k, v
        in names.items()
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

    model.eval()

    pbar = tqdm(
        loader,
        total=len(loader),
        dynamic_ncols=True,
        desc="RSD-T validation",
    )

    for batch in pbar:

        batch = preprocess_batch_rsdt(
            batch,
            device,
        )

        preds = model(
            batch[
                "rgb_img"
            ],

            batch[
                "tir_img"
            ],

            batch[
                "rgb_semantic_img"
            ],
        )

        preds = postprocess_predictions(
            preds=preds,
            model=model,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            max_det=max_det,
        )

        batch_size_current = int(
            batch[
                "rgb_img"
            ].shape[0]
        )

        seen += (
            batch_size_current
        )

        total_instances += int(
            batch[
                "rgb_cls"
            ].shape[0]
        )

        for sample_index, pred in enumerate(
            preds
        ):

            gt = (
                prepare_rsdt_ground_truth(
                    batch,
                    sample_index,
                )
            )

            tp = process_single_image(
                pred=pred,
                gt=gt,
                iouv=iouv,
            )

            target_cls = (
                gt[
                    "cls"
                ]
                .detach()
                .float()
                .cpu()
                .numpy()
            )

            if (
                pred[
                    "cls"
                ].shape[0]
                == 0
            ):

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
                    pred[
                        "conf"
                    ]
                    .detach()
                    .float()
                    .cpu()
                    .numpy()
                )

                pred_cls = (
                    pred[
                        "cls"
                    ]
                    .detach()
                    .float()
                    .cpu()
                    .numpy()
                )

            metrics.update_stats(
                {
                    "tp":
                        tp,

                    "conf":
                        pred_conf,

                    "pred_cls":
                        pred_cls,

                    "target_cls":
                        target_cls,

                    "target_img":
                        np.unique(
                            target_cls
                        ),

                    "im_name":
                        Path(
                            batch[
                                "rgb_path"
                            ][sample_index]
                        ).name,
                }
            )

        pbar.set_postfix(
            {
                "Images":
                    seen,

                "Instances":
                    total_instances,
            }
        )

    pbar.close()

    checkpoint_path = (
        Path(
            checkpoint
        )
        .resolve()
    )

    save_dir = (
        checkpoint_path
        .parent
        .parent
        / f"val_{split}"
    )

    save_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics.process(
        save_dir=save_dir,
        plot=False,
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

    (
        fps,
        latency_ms,
    ) = benchmark_fps_rsdt(
        model=model,
        rgb_high_imgsz=(
            rgb_high_imgsz
        ),
        rgb_semantic_imgsz=(
            rgb_semantic_imgsz
        ),
        tir_imgsz=tir_imgsz,
        device=device,
        amp=amp,
        warmup=speed_warmup,
        iterations=speed_iters,
    )

    state = (
        model.rsdt.scalar_state()
    )

    print(
        "\n"
        "=================================================================================================="
    )

    print(
        f"{'Model':<24}"
        f"{'Params(M)':>12}"
        f"{'FLOPs(G)':>12}"
        f"{'FPS':>10}"
        f"{'P(%)':>10}"
        f"{'R(%)':>10}"
        f"{'AP50':>10}"
        f"{'AP75':>10}"
        f"{'AP50:95':>12}"
    )

    print(
        "--------------------------------------------------------------------------------------------------"
    )

    print(
        f"{'RSD-T-v1':<24}"
        f"{params_m:>12.3f}"
        f"{flops_g:>12.3f}"
        f"{fps:>10.2f}"
        f"{float(precision)*100:>10.2f}"
        f"{float(recall)*100:>10.2f}"
        f"{float(map50)*100:>10.2f}"
        f"{map75*100:>10.2f}"
        f"{float(map5095)*100:>12.2f}"
    )

    print(
        "=================================================================================================="
    )

    print(
        f"RGB high      : "
        f"{rgb_high_imgsz}"
    )

    print(
        f"RGB semantic  : "
        f"{rgb_semantic_imgsz}"
    )

    print(
        f"TIR           : "
        f"{tir_imgsz}"
    )

    print(
        f"Latency       : "
        f"{latency_ms:.3f} ms/pair"
    )

    print(
        f"alpha         : "
        f"{state['alpha']:.6f}"
    )

    print(
        f"gamma         : "
        f"{state['gamma']:.6f}"
    )

    print(
        f"Save dir      : "
        f"{save_dir}"
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--weights",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--split",
        type=str,
        default="val",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="0",
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=4,
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
        "--no-amp",
        action="store_true",
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
        speed_warmup=(
            args.speed_warmup
        ),
        speed_iters=(
            args.speed_iters
        ),
    )


if __name__ == "__main__":
    main()
