"""
RGB-T + RSD-T v1 Detection Model
================================
Zero-intrusion extension of models/rgbt_model.py.

The original RGBTDetectionModel is NOT modified.

Inheritance:
    RGBTDetectionModel
        ↓
    RGBTRSDTDetectionModel

Only one behavior changes:
    before the original RGB-T fusion, RGB P3 is enhanced by RSD-T v1.

P4/P5, original fusion, YOLO neck, detect head and loss implementation style
remain inherited/reused.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Union

import torch
import torch.nn.functional as F

from models.rgbt_model import (
    RGBTDetectionModel,
)

from models.modules.rsd_t_v1 import (
    RSDTv1,
)


class RGBTRSDTDetectionModel(
    RGBTDetectionModel
):
    """
    RGB-T detector with RSD-T v1.

    Inputs:
        rgb            : high-resolution RGB
        tir            : TIR
        rgb_semantic   : low-resolution RGB for full RGB backbone

    Architecture:
        rgb_semantic -> RGB backbone -> RGB P3/P4/P5
        tir          -> TIR backbone -> TIR P3/P4/P5
        rgb_high + RGB P3 + TIR P3 -> RSD-T -> enhanced RGB P3
        enhanced RGB P3/P4/P5 + TIR -> original fusion
        -> original YOLO neck + detect
    """

    def __init__(
        self,
        model_name: str = "yolo26n",
        nc: int = 1,
        pretrained: bool = True,
        fusion: str = "concat",
        fusion_indices: Optional[
            Sequence[int]
        ] = None,
        align_mode: str = "bilinear",
        names: Optional[
            Union[
                Dict[int, str],
                Sequence[str],
            ]
        ] = None,

        semantic_imgsz: int = 640,

        use_guidance: bool = True,

        detail_channels: int = 64,
        stem_channels: int = 24,
        guide_channels: int = 32,

        verbose: bool = True,
    ):
        # Build the original RGB-T model unchanged.
        super().__init__(
            model_name=model_name,
            nc=nc,
            pretrained=pretrained,
            fusion=fusion,
            fusion_indices=fusion_indices,
            align_mode=align_mode,
            names=names,
            verbose=verbose,
        )

        self.semantic_imgsz = int(
            semantic_imgsz
        )

        self.use_guidance = bool(
            use_guidance
        )

        # Standard YOLO11/YOLO26 current repository returns
        # [P3, P4, P5] in ascending fusion_indices.
        if not self.fusion_indices:
            raise RuntimeError(
                "RSD-T requires at least one fusion index."
            )

        self.rsdt_index = int(
            self.fusion_indices[0]
        )

        # Infer P3 channels from the actual model, avoiding hard coding.
        feature_channels = (
            self._infer_feature_channels()
        )

        rgb_p3_channels = int(
            feature_channels[
                self.rsdt_index
            ]["rgb"]
        )

        tir_p3_channels = int(
            feature_channels[
                self.rsdt_index
            ]["tir"]
        )

        self.rsdt = RSDTv1(
            rgb_p3_channels=(
                rgb_p3_channels
            ),

            tir_p3_channels=(
                tir_p3_channels
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

            use_guidance=(
                self.use_guidance
            ),

            exact_identity_when_equal=True,
        )

        if verbose:

            print(
                "\n"
                "============================================================"
            )

            print(
                "Enable isolated RSD-T v1"
            )

            print(
                "============================================================"
            )

            print(
                f"RSD-T P3 layer : "
                f"{self.rsdt_index}"
            )

            print(
                f"RGB semantic   : "
                f"{self.semantic_imgsz}"
            )

            print(
                f"Guidance       : "
                f"{self.use_guidance}"
            )

            print(
                f"RGB P3 C       : "
                f"{rgb_p3_channels}"
            )

            print(
                f"TIR P3 C       : "
                f"{tir_p3_channels}"
            )

            print(
                "gamma init      : 0.0"
            )

            print(
                "============================================================\n"
            )

    # ========================================================
    # Semantic RGB fallback
    # ========================================================

    def _fallback_semantic_rgb(
        self,
        rgb_high: torch.Tensor,
    ) -> torch.Tensor:
        """
        Only used for dummy model tests / FLOPs / FPS when a separate
        dataset-generated semantic RGB is unavailable.

        Formal training and validation should always pass rgb_semantic.
        """

        if (
            rgb_high.shape[-2:]
            == (
                self.semantic_imgsz,
                self.semantic_imgsz,
            )
        ):
            return rgb_high

        return F.interpolate(
            rgb_high,
            size=(
                self.semantic_imgsz,
                self.semantic_imgsz,
            ),
            mode="bilinear",
            align_corners=False,
        )

    # ========================================================
    # Feature forward
    # ========================================================

    def forward_features(
        self,
        rgb: torch.Tensor,
        tir: torch.Tensor,
        rgb_semantic: Optional[
            torch.Tensor
        ] = None,
        return_rsdt_debug: bool = False,
    ) -> Dict:

        if rgb_semantic is None:
            rgb_semantic = (
                self._fallback_semantic_rgb(
                    rgb
                )
            )

        # Full RGB backbone runs ONLY on low-resolution semantic RGB.
        rgb_outputs = list(
            self.forward_rgb_backbone(
                rgb_semantic
            )
        )

        # TIR branch remains the original full backbone.
        tir_outputs = (
            self.forward_tir_backbone(
                tir
            )
        )

        rgb_p3 = rgb_outputs[
            self.rsdt_index
        ]

        tir_p3 = tir_outputs[
            self.rsdt_index
        ]

        rsdt_debug = None

        if return_rsdt_debug:

            (
                enhanced_rgb_p3,
                rsdt_debug,
            ) = self.rsdt(
                rgb_high=rgb,
                rgb_semantic=rgb_semantic,
                rgb_p3=rgb_p3,
                tir_p3=tir_p3,
                return_debug=True,
            )

        else:

            enhanced_rgb_p3 = (
                self.rsdt(
                    rgb_high=rgb,
                    rgb_semantic=rgb_semantic,
                    rgb_p3=rgb_p3,
                    tir_p3=tir_p3,
                    return_debug=False,
                )
            )

        # Only P3 is replaced.
        rgb_outputs[
            self.rsdt_index
        ] = enhanced_rgb_p3

        # Original RGB-T P3/P4/P5 fusion is reused unchanged.
        (
            fused_outputs,
            fusion_features,
        ) = self.fuse_backbone_features(
            rgb_outputs,
            tir_outputs,
        )

        result = {
            "rgb_backbone":
                rgb_outputs,

            "tir_backbone":
                tir_outputs,

            "fused_backbone":
                fused_outputs,

            "fusion_features":
                fusion_features,

            "rgb_semantic":
                rgb_semantic,

            "rsdt_index":
                self.rsdt_index,
        }

        if return_rsdt_debug:

            result[
                "rsdt_debug"
            ] = rsdt_debug

        return result

    # ========================================================
    # Prediction
    # ========================================================

    def predict(
        self,
        rgb: torch.Tensor,
        tir: torch.Tensor,
        rgb_semantic: Optional[
            torch.Tensor
        ] = None,
        return_features: bool = False,
        return_rsdt_debug: bool = False,
    ):

        features = self.forward_features(
            rgb,
            tir,
            rgb_semantic=rgb_semantic,
            return_rsdt_debug=(
                return_rsdt_debug
            ),
        )

        pred = self.forward_head(
            features[
                "fused_backbone"
            ]
        )

        if (
            return_features
            or return_rsdt_debug
        ):

            return {
                "pred":
                    pred,

                **features,
            }

        return pred

    # ========================================================
    # Standard forward
    # ========================================================

    def forward(
        self,
        rgb,
        tir=None,
        rgb_semantic=None,
        return_features: bool = False,
        return_rsdt_debug: bool = False,
    ):

        # Training batch
        if isinstance(
            rgb,
            dict,
        ):

            return self.loss(
                rgb
            )

        if tir is None:

            raise ValueError(
                "RSD-T model requires both RGB and TIR."
            )

        return self.predict(
            rgb,
            tir,
            rgb_semantic=rgb_semantic,
            return_features=(
                return_features
            ),
            return_rsdt_debug=(
                return_rsdt_debug
            ),
        )

    # ========================================================
    # Detection loss
    # ========================================================

    def loss(
        self,
        batch: Dict,
        preds=None,
    ):
        """
        Final prediction belongs to the 640 semantic-RGB reference grid.

        Therefore:
            img     = rgb_semantic_img
            bboxes  = rgb_semantic_bboxes

        High-resolution RGB labels are NOT used by the final head.
        """

        required = [
            "rgb_img",
            "rgb_semantic_img",
            "tir_img",

            "rgb_cls",
            "rgb_semantic_bboxes",
            "rgb_batch_idx",
        ]

        for key in required:

            if key not in batch:

                raise KeyError(
                    f"RSD-T batch missing: {key}"
                )

        if self.criterion is None:

            self.criterion = (
                self.init_criterion()
            )

        if preds is None:

            preds = self.predict(
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
            )

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

        return self.criterion(
            preds,
            yolo_batch,
        )

    # ========================================================
    # Info
    # ========================================================

    def print_info(
        self,
    ):

        super().print_info()

        state = self.rsdt.scalar_state()

        print(
            "RSD-T v1:"
        )

        print(
            f"  P3 layer       : "
            f"{self.rsdt_index}"
        )

        print(
            f"  semantic imgsz : "
            f"{self.semantic_imgsz}"
        )

        print(
            f"  guidance       : "
            f"{self.use_guidance}"
        )

        print(
            f"  alpha          : "
            f"{state['alpha']:.6f}"
        )

        print(
            f"  gamma          : "
            f"{state['gamma']:.6f}"
        )
