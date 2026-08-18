"""
RSD-T Dataset
=============
Zero-intrusion dataset for RSD-T experiments.

This file DOES NOT modify datasets/rgbt_dataset.py.
It reuses its existing:
    - RGB/TIR pairing
    - label loading
    - letterbox()
    - shared flip helpers

For every original RGB image, this dataset generates TWO RGB views using
the repository's existing letterbox() implementation:

    original RGB
      ├─ letterbox(rgb_high_imgsz)     -> rgb_img
      └─ letterbox(rgb_semantic_imgsz) -> rgb_semantic_img

TIR is processed exactly as before:
      └─ letterbox(tir_imgsz)          -> tir_img
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.rgbt_dataset import (
    RGBTPairedDataset,
    _parse_imgsz,
    horizontal_flip,
    vertical_flip,
    letterbox,
    load_yolo_label,
)


class RSDTPairedDataset(RGBTPairedDataset):
    """
    RSD-T paired dataset.

    Output image fields:
        rgb_img              : high-resolution RGB
        rgb_semantic_img     : low-resolution RGB semantic view
        tir_img              : TIR

    Main detection supervision:
        rgb_cls
        rgb_semantic_bboxes
        rgb_batch_idx

    Additional high-resolution RGB boxes are retained as:
        rgb_bboxes
    """

    def __init__(
        self,
        rgb_yaml: Union[str, "Path"],
        tir_yaml: Union[str, "Path"],
        split: str = "train",

        rgb_high_imgsz: Union[
            int,
            Sequence[int],
        ] = 1280,

        rgb_semantic_imgsz: Union[
            int,
            Sequence[int],
        ] = 640,

        tir_imgsz: Union[
            int,
            Sequence[int],
        ] = 640,

        pair_mode: str = "relative",
        augment: bool = False,
        fliplr: float = 0.5,
        flipud: float = 0.0,
        tir_channels: int = 3,
        scaleup: Optional[bool] = None,
        strict_pair: bool = True,
        strict_label: bool = True,
    ):
        # Parent is initialized with HIGH RGB size only.
        # We reuse its pairing/indexing/metadata code unchanged.
        super().__init__(
            rgb_yaml=rgb_yaml,
            tir_yaml=tir_yaml,
            split=split,
            rgb_imgsz=rgb_high_imgsz,
            tir_imgsz=tir_imgsz,
            pair_mode=pair_mode,
            augment=augment,
            fliplr=fliplr,
            flipud=flipud,
            tir_channels=tir_channels,
            scaleup=scaleup,
            strict_pair=strict_pair,
            strict_label=strict_label,
        )

        self.rgb_high_imgsz = _parse_imgsz(
            rgb_high_imgsz
        )

        self.rgb_semantic_imgsz = _parse_imgsz(
            rgb_semantic_imgsz
        )

        # Keep inherited rgb_imgsz as the high-resolution setting.
        self.rgb_imgsz = self.rgb_high_imgsz

        print(
            f"RSD-T RGB high: {self.rgb_high_imgsz}"
        )
        print(
            f"RSD-T RGB semantic: {self.rgb_semantic_imgsz}"
        )

    def __getitem__(
        self,
        index: int,
    ) -> Dict:

        sample = self.samples[index]

        # ====================================================
        # 1. Read ORIGINAL RGB and TIR
        # ====================================================

        rgb_original = self._read_rgb(
            sample["rgb_path"]
        )

        tir = self._read_tir(
            sample["tir_path"]
        )

        rgb_ori_shape = (
            rgb_original.shape[:2]
        )

        tir_ori_shape = (
            tir.shape[:2]
        )

        # ====================================================
        # 2. Load labels in ORIGINAL coordinates
        # ====================================================

        rgb_cls, rgb_boxes_original = (
            load_yolo_label(
                sample["rgb_label"],
                nc=1,
                strict=self.strict_label,
                target_classes=(0,),
                remap_classes=True,
            )
        )

        tir_cls, tir_boxes = (
            load_yolo_label(
                sample["tir_label"],
                nc=1,
                strict=self.strict_label,
                target_classes=(0,),
                remap_classes=True,
            )
        )

        # ====================================================
        # 3. SAME original RGB -> TWO existing LetterBox paths
        # ====================================================

        (
            rgb_high,
            rgb_high_boxes,
            rgb_high_ratio_pad,
        ) = letterbox(
            rgb_original.copy(),
            rgb_boxes_original.copy(),
            self.rgb_high_imgsz,
            scaleup=self.scaleup,
        )

        (
            rgb_semantic,
            rgb_semantic_boxes,
            rgb_semantic_ratio_pad,
        ) = letterbox(
            rgb_original.copy(),
            rgb_boxes_original.copy(),
            self.rgb_semantic_imgsz,
            scaleup=self.scaleup,
        )

        (
            tir,
            tir_boxes,
            tir_ratio_pad,
        ) = letterbox(
            tir,
            tir_boxes,
            self.tir_imgsz,
            scaleup=self.scaleup,
        )

        # ====================================================
        # 4. Shared geometric augmentation
        #
        # The SAME random decision is applied to:
        #   RGB-high
        #   RGB-semantic
        #   TIR
        # ====================================================

        if self.augment:

            import random

            do_fliplr = (
                random.random()
                < self.fliplr
            )

            if do_fliplr:

                (
                    rgb_high,
                    rgb_high_boxes,
                ) = horizontal_flip(
                    rgb_high,
                    rgb_high_boxes,
                )

                (
                    rgb_semantic,
                    rgb_semantic_boxes,
                ) = horizontal_flip(
                    rgb_semantic,
                    rgb_semantic_boxes,
                )

                (
                    tir,
                    tir_boxes,
                ) = horizontal_flip(
                    tir,
                    tir_boxes,
                )

            do_flipud = (
                random.random()
                < self.flipud
            )

            if do_flipud:

                (
                    rgb_high,
                    rgb_high_boxes,
                ) = vertical_flip(
                    rgb_high,
                    rgb_high_boxes,
                )

                (
                    rgb_semantic,
                    rgb_semantic_boxes,
                ) = vertical_flip(
                    rgb_semantic,
                    rgb_semantic_boxes,
                )

                (
                    tir,
                    tir_boxes,
                ) = vertical_flip(
                    tir,
                    tir_boxes,
                )

        # ====================================================
        # 5. HWC -> CHW
        # ====================================================

        rgb_high = np.ascontiguousarray(
            rgb_high.transpose(
                2, 0, 1
            )
        )

        rgb_semantic = np.ascontiguousarray(
            rgb_semantic.transpose(
                2, 0, 1
            )
        )

        tir = np.ascontiguousarray(
            tir.transpose(
                2, 0, 1
            )
        )

        # ====================================================
        # 6. numpy -> torch
        #
        # Keep image tensors uint8.
        # train_rsdt.py / val_rsdt.py normalize /255.
        # ====================================================

        rgb_high = torch.from_numpy(
            rgb_high
        )

        rgb_semantic = torch.from_numpy(
            rgb_semantic
        )

        tir = torch.from_numpy(
            tir
        )

        rgb_cls = torch.from_numpy(
            rgb_cls
        ).float()

        rgb_high_boxes = torch.from_numpy(
            rgb_high_boxes
        ).float()

        rgb_semantic_boxes = torch.from_numpy(
            rgb_semantic_boxes
        ).float()

        tir_cls = torch.from_numpy(
            tir_cls
        ).float()

        tir_boxes = torch.from_numpy(
            tir_boxes
        ).float()

        # ====================================================
        # 7. Return
        # ====================================================

        return {
            # ---------------- RGB high ----------------
            "rgb_img":
                rgb_high,

            "rgb_bboxes":
                rgb_high_boxes,

            "rgb_ratio_pad":
                rgb_high_ratio_pad,

            # ---------------- RGB semantic ----------------
            "rgb_semantic_img":
                rgb_semantic,

            "rgb_semantic_bboxes":
                rgb_semantic_boxes,

            "rgb_semantic_ratio_pad":
                rgb_semantic_ratio_pad,

            # ---------------- shared RGB labels ----------------
            "rgb_cls":
                rgb_cls,

            "rgb_path":
                sample["rgb_path"],

            "rgb_label_path":
                sample["rgb_label"],

            "rgb_ori_shape":
                rgb_ori_shape,

            # ---------------- TIR ----------------
            "tir_img":
                tir,

            "tir_cls":
                tir_cls,

            "tir_bboxes":
                tir_boxes,

            "tir_path":
                sample["tir_path"],

            "tir_label_path":
                sample["tir_label"],

            "tir_ori_shape":
                tir_ori_shape,

            "tir_ratio_pad":
                tir_ratio_pad,

            # ---------------- pair ----------------
            "pair_key":
                sample["key"],
        }

    @staticmethod
    def collate_fn(
        batch: List[Dict],
    ) -> Dict:

        rgb_img = torch.stack(
            [
                item["rgb_img"]
                for item in batch
            ],
            dim=0,
        )

        rgb_semantic_img = torch.stack(
            [
                item[
                    "rgb_semantic_img"
                ]
                for item in batch
            ],
            dim=0,
        )

        tir_img = torch.stack(
            [
                item["tir_img"]
                for item in batch
            ],
            dim=0,
        )

        rgb_cls_list = []
        rgb_high_box_list = []
        rgb_semantic_box_list = []
        rgb_batch_idx_list = []

        tir_cls_list = []
        tir_box_list = []
        tir_batch_idx_list = []

        for i, item in enumerate(batch):

            n_rgb = len(
                item["rgb_cls"]
            )

            if n_rgb > 0:

                rgb_cls_list.append(
                    item["rgb_cls"]
                )

                rgb_high_box_list.append(
                    item["rgb_bboxes"]
                )

                rgb_semantic_box_list.append(
                    item[
                        "rgb_semantic_bboxes"
                    ]
                )

                rgb_batch_idx_list.append(
                    torch.full(
                        (n_rgb,),
                        i,
                        dtype=torch.long,
                    )
                )

            n_tir = len(
                item["tir_cls"]
            )

            if n_tir > 0:

                tir_cls_list.append(
                    item["tir_cls"]
                )

                tir_box_list.append(
                    item["tir_bboxes"]
                )

                tir_batch_idx_list.append(
                    torch.full(
                        (n_tir,),
                        i,
                        dtype=torch.long,
                    )
                )

        if rgb_cls_list:

            rgb_cls = torch.cat(
                rgb_cls_list,
                dim=0,
            )

            rgb_bboxes = torch.cat(
                rgb_high_box_list,
                dim=0,
            )

            rgb_semantic_bboxes = torch.cat(
                rgb_semantic_box_list,
                dim=0,
            )

            rgb_batch_idx = torch.cat(
                rgb_batch_idx_list,
                dim=0,
            )

        else:

            rgb_cls = torch.zeros(
                (0, 1),
                dtype=torch.float32,
            )

            rgb_bboxes = torch.zeros(
                (0, 4),
                dtype=torch.float32,
            )

            rgb_semantic_bboxes = torch.zeros(
                (0, 4),
                dtype=torch.float32,
            )

            rgb_batch_idx = torch.zeros(
                (0,),
                dtype=torch.long,
            )

        if tir_cls_list:

            tir_cls = torch.cat(
                tir_cls_list,
                dim=0,
            )

            tir_bboxes = torch.cat(
                tir_box_list,
                dim=0,
            )

            tir_batch_idx = torch.cat(
                tir_batch_idx_list,
                dim=0,
            )

        else:

            tir_cls = torch.zeros(
                (0, 1),
                dtype=torch.float32,
            )

            tir_bboxes = torch.zeros(
                (0, 4),
                dtype=torch.float32,
            )

            tir_batch_idx = torch.zeros(
                (0,),
                dtype=torch.long,
            )

        return {
            # Images
            "rgb_img":
                rgb_img,

            "rgb_semantic_img":
                rgb_semantic_img,

            "tir_img":
                tir_img,

            # RGB GT
            "rgb_cls":
                rgb_cls,

            "rgb_bboxes":
                rgb_bboxes,

            "rgb_semantic_bboxes":
                rgb_semantic_bboxes,

            "rgb_batch_idx":
                rgb_batch_idx,

            # TIR GT
            "tir_cls":
                tir_cls,

            "tir_bboxes":
                tir_bboxes,

            "tir_batch_idx":
                tir_batch_idx,

            # Paths
            "rgb_path":
                [
                    item["rgb_path"]
                    for item in batch
                ],

            "tir_path":
                [
                    item["tir_path"]
                    for item in batch
                ],

            "pair_key":
                [
                    item["pair_key"]
                    for item in batch
                ],

            # Shapes / mapping
            "rgb_ori_shape":
                [
                    item[
                        "rgb_ori_shape"
                    ]
                    for item in batch
                ],

            "tir_ori_shape":
                [
                    item[
                        "tir_ori_shape"
                    ]
                    for item in batch
                ],

            "rgb_ratio_pad":
                [
                    item[
                        "rgb_ratio_pad"
                    ]
                    for item in batch
                ],

            "rgb_semantic_ratio_pad":
                [
                    item[
                        "rgb_semantic_ratio_pad"
                    ]
                    for item in batch
                ],

            "tir_ratio_pad":
                [
                    item[
                        "tir_ratio_pad"
                    ]
                    for item in batch
                ],
        }


def build_rsdt_dataset(
    rgb_yaml,
    tir_yaml,
    split="train",
    rgb_high_imgsz=1280,
    rgb_semantic_imgsz=640,
    tir_imgsz=640,
    pair_mode="relative",
    augment=False,
    fliplr=0.5,
    flipud=0.0,
    tir_channels=3,
    strict_pair=True,
):
    return RSDTPairedDataset(
        rgb_yaml=rgb_yaml,
        tir_yaml=tir_yaml,
        split=split,
        rgb_high_imgsz=rgb_high_imgsz,
        rgb_semantic_imgsz=rgb_semantic_imgsz,
        tir_imgsz=tir_imgsz,
        pair_mode=pair_mode,
        augment=augment,
        fliplr=fliplr,
        flipud=flipud,
        tir_channels=tir_channels,
        strict_pair=strict_pair,
    )


def build_rsdt_dataloader(
    dataset: RSDTPairedDataset,
    batch_size: int = 8,
    workers: int = 4,
    shuffle: bool = True,
    pin_memory: bool = True,
):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=pin_memory,
        collate_fn=dataset.collate_fn,
        persistent_workers=(
            workers > 0
        ),
    )
