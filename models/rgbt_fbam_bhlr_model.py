"""
RGB-T detector:
  Module-1: FBAM
  Module-2: BHLR
  Fusion  : original concat
  Module-3: OFF

This version adds bbox-guided explicit alignment supervision WITHOUT changing
the FBAM architecture itself.

Alignment supervision
---------------------
Coarse:
    GT sampling translation = C_TIR - C_RGB

Final:
    bbox-affine oracle sampling field
    qx = cTx + (wT / wR) * (x - cRx)
    qy = cTy + (hT / hR) * (y - cRy)
    delta = q - p

Important:
    Current FBAM warp is:
        output(p) = input(p + delta)
    therefore the GT above is a SAMPLING offset, not visual content motion.

Only single-RGB-UAV + single-TIR-UAV samples receive alignment supervision.
Detection loss remains active for every sample.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Optional, Sequence, Union

import torch
import torch.nn.functional as F

from models.rgbt_rsdt_model import RGBTRSDTDetectionModel
from models.modules.bhlr_v1 import MultiScaleBHLRv1


class RGBTFBAMBHLRDetectionModel(RGBTRSDTDetectionModel):
    def __init__(
        self,
        model_name: str = "yolo11m",
        nc: int = 1,
        pretrained: bool = True,
        fusion: str = "concat",
        fusion_indices: Optional[Sequence[int]] = None,
        align_mode: str = "bilinear",
        names: Optional[Union[Dict[int, str], Sequence[str]]] = None,
        semantic_imgsz: int = 640,
        bhlr_scales: Iterable[int] = (3,),
        high_to_semantic_ratio: int = 3,
        detail_channels: int = 64,
        stem_channels: int = 32,
        guide_channels: int = 32,
        bhlr_freq_cutoff: float = 0.15,
        bhlr_freq_sharpness: float = 24.0,
        alignment_cfg: Optional[Dict[str, Any]] = None,
        alignment_loss_cfg: Optional[Dict[str, Any]] = None,
        verbose: bool = True,
    ):
        alignment_cfg = dict(alignment_cfg or {})
        alignment_loss_cfg = dict(alignment_loss_cfg or {})

        # New readable key, mapped to the old parent routing key.
        if "use_for_bhlr" in alignment_cfg:
            alignment_cfg["use_for_rsdt"] = bool(
                alignment_cfg["use_for_bhlr"]
            )

        bhlr_scales = tuple(
            sorted({int(s) for s in bhlr_scales})
        )

        # -----------------------------------------------------
        # Explicit alignment-loss configuration.
        # -----------------------------------------------------
        self.alignment_loss_cfg = alignment_loss_cfg
        self.alignment_loss_enabled = bool(
            alignment_loss_cfg.get("enabled", False)
        )

        self.lambda_align = float(
            alignment_loss_cfg.get("lambda_align", 0.2)
        )

        self.lambda_coarse = float(
            alignment_loss_cfg.get("lambda_coarse", 0.5)
        )

        self.alignment_smooth_l1_beta = float(
            alignment_loss_cfg.get("smooth_l1_beta", 1.0)
        )

        self.clamp_alignment_gt = bool(
            alignment_loss_cfg.get(
                "clamp_gt_to_capacity",
                True,
            )
        )

        configured_loss_scales = alignment_loss_cfg.get(
            "scales",
            None,
        )

        if configured_loss_scales is None:
            self.alignment_loss_scales = None
        else:
            self.alignment_loss_scales = tuple(
                sorted(
                    {
                        int(s)
                        for s in configured_loss_scales
                    }
                )
            )

        # Build original dual-backbone, original concat fusion and alignment.
        # The old RSD-T object is temporarily built by the parent and replaced
        # immediately below.
        super().__init__(
            model_name=model_name,
            nc=nc,
            pretrained=pretrained,
            fusion="concat",
            fusion_indices=fusion_indices,
            align_mode=align_mode,
            names=names,
            semantic_imgsz=semantic_imgsz,
            rsdt_scales=bhlr_scales,
            use_guidance=True,
            detail_channels=detail_channels,
            stem_channels=stem_channels,
            guide_channels=guide_channels,
            alignment_cfg=alignment_cfg,
            verbose=False,
        )

        # Alignment supervision is meaningful only for real FBAM.
        align_type = str(
            self.alignment_cfg.get(
                "type",
                "identity",
            )
        ).strip().lower()

        if (
            self.alignment_loss_enabled
            and (
                not self.alignment_enabled
                or align_type != "fbam"
            )
        ):
            print(
                "[Alignment loss] disabled because "
                "FBAM alignment is not active."
            )
            self.alignment_loss_enabled = False

        # -----------------------------------------------------
        # IMPORTANT:
        # FBAM contains nn.LazyConv2d projections.
        # The trainer counts parameters / builds the optimizer
        # before the first real forward, so initialize them now.
        # -----------------------------------------------------
        self._initialize_fbam_lazy_modules()

        feature_channels = self._infer_feature_channels()

        rgb_channels = {}
        tir_channels = {}

        for scale in self.rsdt_scales:
            idx = self.rsdt_scale_to_index[scale]

            rgb_channels[scale] = int(
                feature_channels[idx]["rgb"]
            )

            tir_channels[scale] = int(
                feature_channels[idx]["tir"]
            )

        del self.rsdt

        self.bhlr = MultiScaleBHLRv1(
            rgb_channels=rgb_channels,
            tir_channels=tir_channels,
            scales=self.rsdt_scales,
            high_to_semantic_ratio=int(
                high_to_semantic_ratio
            ),
            detail_channels=int(
                detail_channels
            ),
            stem_channels=int(
                stem_channels
            ),
            guide_channels=int(
                guide_channels
            ),
            freq_cutoff=float(
                bhlr_freq_cutoff
            ),
            freq_sharpness=float(
                bhlr_freq_sharpness
            ),
            exact_identity_when_equal=True,
        )

        # Parent forward_features() still calls self.rsdt(...).
        self.rsdt = self.bhlr
        self.bhlr_scales = self.rsdt_scales

        if verbose:
            print(
                "\n"
                "============================================================"
            )
            print(
                "FBAM + BHLR RGB-T Detector"
            )
            print(
                "============================================================"
            )
            print(
                f"Model              : {self.model_name}"
            )
            print(
                "Fusion             : concat (Module-3 OFF)"
            )
            print(
                f"FBAM enabled       : {self.alignment_enabled}"
            )
            print(
                "FBAM type          : "
                f"{self.alignment_cfg.get('type', 'identity')}"
            )
            print(
                "FBAM scales        : "
                f"{list(getattr(self.alignment, 'scales', []))}"
            )
            print(
                f"BHLR scales        : {list(self.bhlr_scales)}"
            )
            print(
                f"RGB semantic       : {self.semantic_imgsz}"
            )
            print(
                f"High/semantic ratio: {high_to_semantic_ratio}"
            )
            print(
                "Reference system   : RGB; TIR -> RGB"
            )
            print(
                "Alignment loss     : "
                f"{self.alignment_loss_enabled}"
            )

            if self.alignment_loss_enabled:
                print(
                    f"  lambda_align     : {self.lambda_align}"
                )
                print(
                    f"  lambda_coarse    : {self.lambda_coarse}"
                )
                print(
                    "  SmoothL1 beta    : "
                    f"{self.alignment_smooth_l1_beta}"
                )
                print(
                    "  GT clamp         : "
                    f"{self.clamp_alignment_gt}"
                )
                print(
                    "  loss scales      : "
                    f"{self.alignment_loss_scales}"
                )
                print(
                    "  target samples   : "
                    "single RGB UAV + single TIR UAV"
                )

            print(
                "============================================================\n"
            )

    # ======================================================================
    # Lazy FBAM initialization
    # ======================================================================

    @torch.no_grad()
    def _initialize_fbam_lazy_modules(self):
        """
        Materialize FBAM LazyConv2d parameters before external parameter
        counting or optimizer construction.
        """
        if not getattr(
            self,
            "alignment_enabled",
            False,
        ):
            return

        align_type = str(
            self.alignment_cfg.get(
                "type",
                "identity",
            )
        ).strip().lower()

        if align_type != "fbam":
            return

        rgb_training = self.model.training
        tir_training = self.tir_backbone.training
        align_training = self.alignment.training

        self.model.eval()
        self.tir_backbone.eval()
        self.alignment.eval()

        device = next(
            self.model.parameters()
        ).device

        dummy = torch.zeros(
            1,
            3,
            256,
            256,
            device=device,
            dtype=torch.float32,
        )

        rgb_outputs = list(
            self.forward_rgb_backbone(
                dummy
            )
        )

        tir_outputs = list(
            self.forward_tir_backbone(
                dummy
            )
        )

        _ = self.alignment(
            rgb_features=rgb_outputs,
            tir_features=tir_outputs,
            scale_to_index=(
                self.rsdt_scale_to_index
            ),
            return_debug=False,
        )

        self.model.train(
            rgb_training
        )
        self.tir_backbone.train(
            tir_training
        )
        self.alignment.train(
            align_training
        )

        from torch.nn.parameter import (
            UninitializedParameter,
        )

        uninitialized = [
            name
            for name, param
            in self.named_parameters()
            if isinstance(
                param,
                UninitializedParameter,
            )
        ]

        if uninitialized:
            raise RuntimeError(
                "FBAM lazy parameters are still uninitialized: "
                + ", ".join(
                    uninitialized
                )
            )

    # ======================================================================
    # Forward features
    # ======================================================================

    def forward_features(
        self,
        rgb,
        tir,
        rgb_semantic=None,
        return_rsdt_debug: bool = False,
        return_alignment_debug: bool = False,
    ):
        out = super().forward_features(
            rgb=rgb,
            tir=tir,
            rgb_semantic=rgb_semantic,
            return_rsdt_debug=(
                return_rsdt_debug
            ),
            return_alignment_debug=(
                return_alignment_debug
            ),
        )

        out["bhlr_scales"] = out.get(
            "rsdt_scales",
            self.bhlr_scales,
        )

        if "rsdt_debug" in out:
            out["bhlr_debug"] = (
                out["rsdt_debug"]
            )

        return out

    # ======================================================================
    # Alignment-supervision helpers
    # ======================================================================

    @staticmethod
    def _det_loss_items_to_dict(
        loss_items,
    ) -> Dict[str, torch.Tensor]:
        """
        Convert Ultralytics loss_items to a dict while preserving the
        original standard box/cls/dfl names used by train_rgbt.py.
        """
        if isinstance(
            loss_items,
            dict,
        ):
            return dict(
                loss_items
            )

        if torch.is_tensor(
            loss_items
        ):
            values = list(
                loss_items.flatten()
            )

        elif isinstance(
            loss_items,
            (list, tuple),
        ):
            values = list(
                loss_items
            )

        else:
            return {
                "loss":
                    loss_items
            }

        if len(values) == 3:
            return {
                "box_loss":
                    values[0],

                "cls_loss":
                    values[1],

                "dfl_loss":
                    values[2],
            }

        return {
            f"loss_{i}":
                value
            for i, value
            in enumerate(values)
        }

    @staticmethod
    def _get_scale_debug(
        alignment_info: Dict,
        scale: int,
    ) -> Dict:
        scale_debug = alignment_info.get(
            "scale_debug",
            alignment_info.get(
                "scales",
                {},
            ),
        )

        if scale in scale_debug:
            return scale_debug[
                scale
            ]

        key = str(
            scale
        )

        if key in scale_debug:
            return scale_debug[
                key
            ]

        raise KeyError(
            f"Alignment debug missing P{scale}. "
            f"Available={list(scale_debug.keys())}"
        )

    @staticmethod
    def _bbox_feature_xywh(
        box: torch.Tensor,
        h: int,
        w: int,
    ):
        """
        Normalized xywh -> feature-coordinate xywh.
        """
        return (
            box[0] * float(w),
            box[1] * float(h),
            box[2] * float(w),
            box[3] * float(h),
        )

    def _roi_coordinates(
        self,
        rgb_box: torch.Tensor,
        h: int,
        w: int,
        device: torch.device,
    ):
        """
        Return all integer P-level coordinates covered by the RGB bbox.

        Offset is defined on the output/RGB sampling grid, therefore the
        supervision region MUST be the RGB bbox region.
        """
        rcx, rcy, rw, rh = (
            self._bbox_feature_xywh(
                rgb_box,
                h,
                w,
            )
        )

        # Integer ROI bounds.
        x1 = max(
            0,
            min(
                w - 1,
                int(
                    math.floor(
                        float(
                            rcx - rw / 2
                        )
                    )
                ),
            ),
        )

        y1 = max(
            0,
            min(
                h - 1,
                int(
                    math.floor(
                        float(
                            rcy - rh / 2
                        )
                    )
                ),
            ),
        )

        x2 = max(
            x1 + 1,
            min(
                w,
                int(
                    math.ceil(
                        float(
                            rcx + rw / 2
                        )
                    )
                ),
            ),
        )

        y2 = max(
            y1 + 1,
            min(
                h,
                int(
                    math.ceil(
                        float(
                            rcy + rh / 2
                        )
                    )
                ),
            ),
        )

        ys = torch.arange(
            y1,
            y2,
            device=device,
            dtype=torch.long,
        )

        xs = torch.arange(
            x1,
            x2,
            device=device,
            dtype=torch.long,
        )

        yy, xx = torch.meshgrid(
            ys,
            xs,
            indexing="ij",
        )

        return (
            yy.reshape(-1),
            xx.reshape(-1),
        )

    def _capacity_for_scale(
        self,
        scale: int,
    ):
        """
        Read the actual bounded offset capacity from FBAMScaleUnit.

        coarse:
            [-max_coarse, +max_coarse]

        total:
            coarse + fine
            approximately bounded by
            [-max_coarse-max_fine, +max_coarse+max_fine]
        """
        max_coarse = None
        max_total = None

        alignment = getattr(
            self,
            "alignment",
            None,
        )

        if (
            alignment is not None
            and hasattr(
                alignment,
                "units",
            )
            and str(scale)
            in alignment.units
        ):
            unit = alignment.units[
                str(scale)
            ]

            max_coarse = float(
                getattr(
                    unit,
                    "max_coarse_offset",
                    0.0,
                )
            )

            max_fine = float(
                getattr(
                    unit,
                    "max_fine_offset",
                    0.0,
                )
            )

            max_total = (
                max_coarse
                + max_fine
            )

        return (
            max_coarse,
            max_total,
        )

    def _alignment_auxiliary_loss(
        self,
        batch: Dict,
        alignment_info: Dict,
    ):
        """
        Compute bbox-guided explicit FBAM alignment loss.

        Coarse target:
            constant sampling translation
                delta_c = C_TIR - C_RGB

        Final target:
            bbox-affine sampling field
                qx = cTx + (wT/wR) * (x-cRx)
                qy = cTy + (hT/hR) * (y-cRy)
                delta = q - p

        Only samples with exactly one RGB bbox and one TIR bbox are used.
        This avoids inventing object correspondence for multi-object samples.
        """
        scale_debug_all = (
            alignment_info.get(
                "scale_debug",
                alignment_info.get(
                    "scales",
                    {},
                ),
            )
        )

        if not scale_debug_all:
            raise RuntimeError(
                "Alignment loss enabled but "
                "alignment_info contains no scale_debug."
            )

        if self.alignment_loss_scales is None:
            scales = [
                int(s)
                for s in getattr(
                    self.alignment,
                    "scales",
                    [],
                )
            ]
        else:
            scales = list(
                self.alignment_loss_scales
            )

        coarse_losses = []
        final_losses = []

        valid_pairs = 0

        # We need both modality-specific target sets.
        required = (
            "rgb_semantic_bboxes",
            "rgb_batch_idx",
            "tir_bboxes",
            "tir_batch_idx",
        )

        for key in required:
            if key not in batch:
                raise KeyError(
                    "Alignment supervision requires "
                    f"batch['{key}']."
                )

        for scale in scales:
            dbg = (
                self._get_scale_debug(
                    alignment_info,
                    scale,
                )
            )

            coarse_offset = (
                dbg[
                    "coarse_offset"
                ].float()
            )

            total_offset = (
                dbg[
                    "total_offset"
                ].float()
            )

            batch_size, _, h, w = (
                coarse_offset.shape
            )

            max_coarse, max_total = (
                self._capacity_for_scale(
                    scale
                )
            )

            for sample_idx in range(
                batch_size
            ):
                rgb_select = (
                    batch[
                        "rgb_batch_idx"
                    ]
                    == sample_idx
                )

                tir_select = (
                    batch[
                        "tir_batch_idx"
                    ]
                    == sample_idx
                )

                # Do not guess RGB<->TIR assignment in multi-UAV images.
                if (
                    int(
                        rgb_select.sum().item()
                    )
                    != 1
                    or int(
                        tir_select.sum().item()
                    )
                    != 1
                ):
                    continue

                rgb_box = (
                    batch[
                        "rgb_semantic_bboxes"
                    ][rgb_select][0]
                    .float()
                )

                tir_box = (
                    batch[
                        "tir_bboxes"
                    ][tir_select][0]
                    .float()
                )

                (
                    rcx,
                    rcy,
                    rw,
                    rh,
                ) = self._bbox_feature_xywh(
                    rgb_box,
                    h,
                    w,
                )

                (
                    tcx,
                    tcy,
                    tw,
                    th,
                ) = self._bbox_feature_xywh(
                    tir_box,
                    h,
                    w,
                )

                yy, xx = (
                    self._roi_coordinates(
                        rgb_box,
                        h,
                        w,
                        coarse_offset.device,
                    )
                )

                if yy.numel() == 0:
                    continue

                # ----------------------------------------------------------
                # Coarse GT: translation only.
                #
                # Current warp:
                #     output(p) = input(p + delta)
                #
                # Thus:
                #     delta_gt = C_TIR - C_RGB
                # ----------------------------------------------------------
                gt_coarse_dx = (
                    tcx
                    - rcx
                )

                gt_coarse_dy = (
                    tcy
                    - rcy
                )

                gt_coarse = torch.stack(
                    (
                        gt_coarse_dx,
                        gt_coarse_dy,
                    )
                ).view(
                    2,
                    1,
                )

                gt_coarse = gt_coarse.expand(
                    2,
                    yy.numel(),
                )

                if (
                    self.clamp_alignment_gt
                    and max_coarse is not None
                    and max_coarse > 0
                ):
                    gt_coarse = gt_coarse.clamp(
                        -max_coarse,
                        max_coarse,
                    )

                pred_coarse = (
                    coarse_offset[
                        sample_idx,
                        :,
                        yy,
                        xx,
                    ]
                )

                coarse_loss = (
                    F.smooth_l1_loss(
                        pred_coarse,
                        gt_coarse,
                        reduction="mean",
                        beta=(
                            self.alignment_smooth_l1_beta
                        ),
                    )
                )

                # ----------------------------------------------------------
                # Final GT: bbox affine field.
                #
                # For each RGB/output position p=(x,y), compute the TIR
                # sampling position q that maps RGB bbox coordinates into
                # the TIR bbox coordinate system.
                # ----------------------------------------------------------
                px = xx.to(
                    dtype=total_offset.dtype
                )

                py = yy.to(
                    dtype=total_offset.dtype
                )

                sx = (
                    tw
                    / rw.clamp(
                        min=1e-6
                    )
                )

                sy = (
                    th
                    / rh.clamp(
                        min=1e-6
                    )
                )

                qx = (
                    tcx
                    + sx
                    * (
                        px
                        - rcx
                    )
                )

                qy = (
                    tcy
                    + sy
                    * (
                        py
                        - rcy
                    )
                )

                gt_final_dx = (
                    qx
                    - px
                )

                gt_final_dy = (
                    qy
                    - py
                )

                gt_final = torch.stack(
                    (
                        gt_final_dx,
                        gt_final_dy,
                    ),
                    dim=0,
                )

                if (
                    self.clamp_alignment_gt
                    and max_total is not None
                    and max_total > 0
                ):
                    gt_final = gt_final.clamp(
                        -max_total,
                        max_total,
                    )

                pred_final = (
                    total_offset[
                        sample_idx,
                        :,
                        yy,
                        xx,
                    ]
                )

                final_loss = (
                    F.smooth_l1_loss(
                        pred_final,
                        gt_final,
                        reduction="mean",
                        beta=(
                            self.alignment_smooth_l1_beta
                        ),
                    )
                )

                coarse_losses.append(
                    coarse_loss
                )

                final_losses.append(
                    final_loss
                )

                valid_pairs += 1

        if valid_pairs == 0:
            # Differentiable zero connected to FBAM parameters.
            first_scale = scales[0]

            first_dbg = (
                self._get_scale_debug(
                    alignment_info,
                    first_scale,
                )
            )

            zero = (
                first_dbg[
                    "coarse_offset"
                ].sum()
                * 0.0
            )

            return {
                "align_loss":
                    zero,

                "align_coarse_loss":
                    zero,

                "align_final_loss":
                    zero,

                "align_valid_pairs":
                    0,
            }

        coarse_loss = torch.stack(
            coarse_losses
        ).mean()

        final_loss = torch.stack(
            final_losses
        ).mean()

        align_loss = (
            final_loss
            + self.lambda_coarse
            * coarse_loss
        )

        return {
            "align_loss":
                align_loss,

            "align_coarse_loss":
                coarse_loss,

            "align_final_loss":
                final_loss,

            "align_valid_pairs":
                valid_pairs,
        }

    # ======================================================================
    # Detection + explicit alignment loss
    # ======================================================================

    def loss(
        self,
        batch: Dict,
        preds=None,
    ):
        """
        Training loss.

        Detection:
            unchanged YOLO detection criterion in RGB-semantic coordinates.

        Alignment:
            optional bbox-guided FBAM supervision.

        Final optimization objective:
            L = L_det
              + lambda_align
                * (L_final + lambda_coarse * L_coarse)

        The alignment term is multiplied by batch size before being appended
        to loss_raw because current Ultralytics detection criterion returns
        a batch-scaled raw loss vector.
        """
        if (
            not self.alignment_loss_enabled
            or preds is not None
        ):
            return super().loss(
                batch,
                preds=preds,
            )

        required = [
            "rgb_img",
            "rgb_semantic_img",
            "tir_img",
            "rgb_cls",
            "rgb_semantic_bboxes",
            "rgb_batch_idx",
            "tir_bboxes",
            "tir_batch_idx",
        ]

        for key in required:
            if key not in batch:
                raise KeyError(
                    f"FBAM+BHLR batch missing: {key}"
                )

        if self.criterion is None:
            self.criterion = (
                self.init_criterion()
            )

        # Request FBAM debug tensors WITH gradient.
        outputs = self.predict(
            batch[
                "rgb_img"
            ],
            batch[
                "tir_img"
            ],
            rgb_semantic=(
                batch[
                    "rgb_semantic_img"
                ]
            ),
            return_features=True,
            return_alignment_debug=True,
        )

        det_preds = outputs[
            "pred"
        ]

        yolo_batch = {
            "img":
                batch[
                    "rgb_semantic_img"
                ],

            "cls":
                batch[
                    "rgb_cls"
                ],

            "bboxes":
                batch[
                    "rgb_semantic_bboxes"
                ],

            "batch_idx":
                batch[
                    "rgb_batch_idx"
                ],
        }

        (
            det_loss_raw,
            det_loss_items,
        ) = self.criterion(
            det_preds,
            yolo_batch,
        )

        alignment_stats = (
            self._alignment_auxiliary_loss(
                batch,
                outputs[
                    "alignment_info"
                ],
            )
        )

        align_loss = (
            alignment_stats[
                "align_loss"
            ]
        )

        batch_size = int(
            batch[
                "rgb_semantic_img"
            ].shape[0]
        )

        # Ultralytics raw detection loss is batch-scaled.
        weighted_align_raw = (
            self.lambda_align
            * align_loss
            * float(
                batch_size
            )
        )

        # train_rgbt.py performs loss_raw.sum().
        # Append one new component so alignment loss is counted exactly once.
        if torch.is_tensor(
            det_loss_raw
        ):
            total_loss_raw = torch.cat(
                (
                    det_loss_raw.reshape(
                        -1
                    ),
                    weighted_align_raw.reshape(
                        1
                    ),
                ),
                dim=0,
            )

        else:
            raise TypeError(
                "Expected tensor det_loss_raw, got "
                f"{type(det_loss_raw)}"
            )

        loss_items = (
            self._det_loss_items_to_dict(
                det_loss_items
            )
        )

        # Reporting only: detach these values so running statistics do not
        # keep the autograd graph alive.
        loss_items[
            "align_loss"
        ] = (
            align_loss.detach()
        )

        loss_items[
            "align_coarse_loss"
        ] = (
            alignment_stats[
                "align_coarse_loss"
            ].detach()
        )

        loss_items[
            "align_final_loss"
        ] = (
            alignment_stats[
                "align_final_loss"
            ].detach()
        )

        loss_items[
            "align_weighted_loss"
        ] = (
            (
                self.lambda_align
                * align_loss
            ).detach()
        )

        loss_items[
            "align_valid_pairs"
        ] = float(
            alignment_stats[
                "align_valid_pairs"
            ]
        )

        return (
            total_loss_raw,
            loss_items,
        )
