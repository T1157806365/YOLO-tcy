"""
Configurable Multi-Scale RGB-T + RSD-T Detection Model
======================================================

Zero-intrusion extension of:
    models/rgbt_model.py

Original RGB-T model is not modified.

YAML control:
    rsdt_scales: [3]
    rsdt_scales: [4]
    rsdt_scales: [5]
    rsdt_scales: [3, 4]
    rsdt_scales: [3, 4, 5]

Scale mapping:
    P3 -> fusion_indices[0]
    P4 -> fusion_indices[1]
    P5 -> fusion_indices[2]

The shared high-resolution DetailStem runs only once.
Each selected scale has an independent guidance/injection adapter.
"""

from __future__ import annotations

from typing import (
    Dict,
    Iterable,
    Optional,
    Sequence,
    Union,
)

import torch
import torch.nn.functional as F

from models.rgbt_model import (
    RGBTDetectionModel,
)

from models.modules.rsd_t_v1 import (
    MultiScaleRSDTv1,
)


class RGBTRSDTDetectionModel(
    RGBTDetectionModel
):
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

        # -----------------------------------------------------
        # New configurable scale selector
        # -----------------------------------------------------
        rsdt_scales: Iterable[int] = (3,),

        use_guidance: bool = True,

        detail_channels: int = 64,
        stem_channels: int = 24,
        guide_channels: int = 32,

        verbose: bool = True,
    ):
        # -----------------------------------------------------
        # Original RGB-T detector
        # -----------------------------------------------------

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

        self.rsdt_scales = tuple(
            sorted(
                {
                    int(s)
                    for s in rsdt_scales
                }
            )
        )

        if not self.rsdt_scales:
            raise ValueError(
                "rsdt_scales 不能为空。"
            )

        invalid = [
            s
            for s in self.rsdt_scales
            if s not in (
                3,
                4,
                5,
            )
        ]

        if invalid:
            raise ValueError(
                "rsdt_scales 只能包含 3/4/5，"
                f"当前非法值={invalid}"
            )

        # -----------------------------------------------------
        # Current RGBTDetectionModel exposes its three pyramid
        # fusion locations in P3/P4/P5 order.
        # -----------------------------------------------------

        if len(
            self.fusion_indices
        ) < 3:
            raise RuntimeError(
                "Multi-scale RSD-T requires "
                "P3/P4/P5 fusion indices."
            )

        self.rsdt_scale_to_index = {
            3:
                int(
                    self.fusion_indices[
                        0
                    ]
                ),

            4:
                int(
                    self.fusion_indices[
                        1
                    ]
                ),

            5:
                int(
                    self.fusion_indices[
                        2
                    ]
                ),
        }

        # -----------------------------------------------------
        # Infer actual channels, no hard coding.
        # -----------------------------------------------------

        feature_channels = (
            self._infer_feature_channels()
        )

        rgb_channels = {}
        tir_channels = {}

        for scale in self.rsdt_scales:

            index = (
                self.rsdt_scale_to_index[
                    scale
                ]
            )

            rgb_channels[
                scale
            ] = int(
                feature_channels[
                    index
                ][
                    "rgb"
                ]
            )

            tir_channels[
                scale
            ] = int(
                feature_channels[
                    index
                ][
                    "tir"
                ]
            )

        # -----------------------------------------------------
        # Shared DetailStem + per-scale adapters
        # -----------------------------------------------------

        self.rsdt = MultiScaleRSDTv1(
            rgb_channels=rgb_channels,
            tir_channels=tir_channels,
            scales=self.rsdt_scales,

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
                "Enable configurable Multi-Scale RSD-T v1"
            )
            print(
                "============================================================"
            )
            print(
                f"RSD-T scales    : "
                f"{list(self.rsdt_scales)}"
            )

            for scale in self.rsdt_scales:
                print(
                    f"P{scale} layer       : "
                    f"{self.rsdt_scale_to_index[scale]}"
                )

                print(
                    f"P{scale} RGB/TIR C   : "
                    f"{rgb_channels[scale]}/"
                    f"{tir_channels[scale]}"
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
                "Shared DetailStem: 1x"
            )
            print(
                "gamma init      : 0.0 per scale"
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
    # Forward features
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

        # -----------------------------------------------------
        # Original two full semantic backbones
        # -----------------------------------------------------

        rgb_outputs = list(
            self.forward_rgb_backbone(
                rgb_semantic
            )
        )

        tir_outputs = list(
            self.forward_tir_backbone(
                tir
            )
        )

        # -----------------------------------------------------
        # Build only selected scale feature dictionaries.
        # -----------------------------------------------------

        rgb_scale_features = {}
        tir_scale_features = {}

        for scale in self.rsdt_scales:

            index = (
                self.rsdt_scale_to_index[
                    scale
                ]
            )

            rgb_scale_features[
                scale
            ] = rgb_outputs[
                index
            ]

            tir_scale_features[
                scale
            ] = tir_outputs[
                index
            ]

        rsdt_debug = None

        # -----------------------------------------------------
        # Shared high-resolution detail + selected scales
        # -----------------------------------------------------

        if return_rsdt_debug:

            (
                enhanced_features,
                rsdt_debug,
            ) = self.rsdt(
                rgb_high=rgb,
                rgb_semantic=(
                    rgb_semantic
                ),
                rgb_features=(
                    rgb_scale_features
                ),
                tir_features=(
                    tir_scale_features
                ),
                return_debug=True,
            )

        else:

            enhanced_features = self.rsdt(
                rgb_high=rgb,
                rgb_semantic=rgb_semantic,
                rgb_features=(
                    rgb_scale_features
                ),
                tir_features=(
                    tir_scale_features
                ),
                return_debug=False,
            )

        # -----------------------------------------------------
        # Replace ONLY selected RGB pyramid levels.
        # Non-selected levels remain exactly original.
        # -----------------------------------------------------

        for scale in self.rsdt_scales:

            index = (
                self.rsdt_scale_to_index[
                    scale
                ]
            )

            rgb_outputs[
                index
            ] = enhanced_features[
                scale
            ]

        # -----------------------------------------------------
        # Original final RGB-T fusion stays unchanged.
        # -----------------------------------------------------

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

            "rsdt_scales":
                self.rsdt_scales,

            "rsdt_scale_to_index":
                dict(
                    self.rsdt_scale_to_index
                ),
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
        Detection head stays in semantic-RGB coordinates.
        This is independent of whether P3/P4/P5 are enhanced.
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
    # Information
    # ========================================================

    def print_info(
        self,
    ):

        super().print_info()

        state = (
            self.rsdt.scalar_state()
        )

        print(
            "Multi-Scale RSD-T v1:"
        )

        print(
            f"  scales          : "
            f"{list(self.rsdt_scales)}"
        )

        print(
            f"  semantic imgsz  : "
            f"{self.semantic_imgsz}"
        )

        print(
            f"  guidance        : "
            f"{self.use_guidance}"
        )

        print(
            "  shared detail   : "
            "DetailStem x1"
        )

        for scale in self.rsdt_scales:

            print(
                f"  P{scale} alpha       : "
                f"{state[f'alpha_p{scale}']:.6f}"
            )

            print(
                f"  P{scale} gamma       : "
                f"{state[f'gamma_p{scale}']:.6f}"
            )
