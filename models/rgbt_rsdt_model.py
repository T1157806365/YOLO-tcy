"""
RGB-T + RSD-T with Optional Pluggable Alignment
===============================================

Design goal
-----------
Keep alignment and RSD-T completely decoupled.

Current pipeline:
    RGB/TIR Backbones
        -> IdentityAlignment (alignment disabled)
        -> RSD-T
        -> original RGB-T fusion
        -> neck/detect

Future pipeline:
    RGB/TIR Backbones
        -> any BaseAlignment-compatible plugin
        -> RSD-T and/or original RGB-T fusion
        -> neck/detect

Current YAML:
    alignment:
      enabled: false

When disabled:
    aligned_tir == raw_tir

so the network behaves like the current RSD-T implementation.

Alignment can independently be routed to:
    use_for_rsdt
    use_for_fusion
"""

from __future__ import annotations

from typing import (
    Any,
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

from models.modules.alignment.builder import (
    build_alignment,
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
        # RSD-T
        # -----------------------------------------------------
        rsdt_scales: Iterable[int] = (3,),

        use_guidance: bool = True,

        detail_channels: int = 64,
        stem_channels: int = 24,
        guide_channels: int = 32,

        # -----------------------------------------------------
        # Optional alignment plugin
        # -----------------------------------------------------
        alignment_cfg: Optional[
            Dict[str, Any]
        ] = None,

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

        # =====================================================
        # RSD-T scale configuration
        # =====================================================

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

        if len(
            self.fusion_indices
        ) < 3:
            raise RuntimeError(
                "Multi-scale RSD-T requires "
                "P3/P4/P5 fusion indices."
            )

        # Map conceptual P3/P4/P5 to the actual backbone output index.
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

        # =====================================================
        # Infer actual feature channels
        # =====================================================

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

        # =====================================================
        # RSD-T
        # =====================================================

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

        # =====================================================
        # Optional Alignment Plugin
        # =====================================================

        self.alignment_cfg = dict(
            alignment_cfg or {}
        )

        self.alignment_enabled = bool(
            self.alignment_cfg.get(
                "enabled",
                False,
            )
        )

        self.alignment_use_for_rsdt = bool(
            self.alignment_cfg.get(
                "use_for_rsdt",
                True,
            )
        )

        self.alignment_use_for_fusion = bool(
            self.alignment_cfg.get(
                "use_for_fusion",
                True,
            )
        )

        # Important:
        # enabled=false -> IdentityAlignment -> no parameters / no changes.
        self.alignment = build_alignment(
            self.alignment_cfg
        )

        if verbose:
            print(
                "\n"
                "============================================================"
            )
            print(
                "RSD-T + Pluggable Alignment Framework"
            )
            print(
                "============================================================"
            )
            print(
                f"RSD-T scales       : "
                f"{list(self.rsdt_scales)}"
            )
            print(
                f"RGB semantic       : "
                f"{self.semantic_imgsz}"
            )
            print(
                f"RSD-T guidance     : "
                f"{self.use_guidance}"
            )
            print(
                "Shared DetailStem   : 1x"
            )

            print(
                "------------------------------------------------------------"
            )

            print(
                f"Alignment enabled  : "
                f"{self.alignment_enabled}"
            )
            print(
                f"Alignment type     : "
                f"{self.alignment_cfg.get('type', 'identity')}"
            )
            print(
                f"Use for RSD-T      : "
                f"{self.alignment_use_for_rsdt}"
            )
            print(
                f"Use for fusion     : "
                f"{self.alignment_use_for_fusion}"
            )

            if not self.alignment_enabled:
                print(
                    "Alignment behavior : "
                    "Identity / bypass"
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
        return_alignment_debug: bool = False,
    ) -> Dict:

        if rgb_semantic is None:
            rgb_semantic = (
                self._fallback_semantic_rgb(
                    rgb
                )
            )

        # -----------------------------------------------------
        # 1. Original semantic backbones
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

        # Keep the raw TIR stream explicitly.
        raw_tir_outputs = tir_outputs

        # -----------------------------------------------------
        # 2. Optional pluggable alignment
        #
        # Current:
        #     IdentityAlignment
        #     aligned_tir_outputs == raw_tir_outputs
        #
        # Future:
        #     Offset / correlation / deformable / other plugin
        # -----------------------------------------------------

        (
            aligned_tir_outputs,
            alignment_info,
        ) = self.alignment(
            rgb_features=rgb_outputs,
            tir_features=raw_tir_outputs,
            scale_to_index=(
                self.rsdt_scale_to_index
            ),
            return_debug=(
                return_alignment_debug
                or return_rsdt_debug
            ),
        )

        # -----------------------------------------------------
        # 3. Routing policy
        #
        # Alignment can later serve:
        #     a) RSD-T only
        #     b) fusion only
        #     c) both
        #
        # Current alignment OFF -> both routes are unchanged.
        # -----------------------------------------------------

        if (
            self.alignment_enabled
            and self.alignment_use_for_rsdt
        ):
            tir_for_rsdt = (
                aligned_tir_outputs
            )
        else:
            tir_for_rsdt = (
                raw_tir_outputs
            )

        if (
            self.alignment_enabled
            and self.alignment_use_for_fusion
        ):
            tir_for_fusion = (
                aligned_tir_outputs
            )
        else:
            tir_for_fusion = (
                raw_tir_outputs
            )

        # -----------------------------------------------------
        # 4. Select P3/P4/P5 features for RSD-T
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
            ] = tir_for_rsdt[
                index
            ]

        # -----------------------------------------------------
        # 5. RSD-T
        # -----------------------------------------------------

        rsdt_debug = None

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

            enhanced_features = (
                self.rsdt(
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

                    return_debug=False,
                )
            )

        # Replace only selected RGB pyramid levels.
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
        # 6. Original RGB-T fusion
        #
        # Current alignment OFF:
        #     tir_for_fusion == raw_tir_outputs
        #
        # Future alignment ON:
        #     can use aligned TIR if YAML requests it.
        # -----------------------------------------------------

        (
            fused_outputs,
            fusion_features,
        ) = self.fuse_backbone_features(
            rgb_outputs,
            tir_for_fusion,
        )

        result = {
            "rgb_backbone":
                rgb_outputs,

            # Raw TIR is kept for backward/debug semantics.
            "tir_backbone":
                raw_tir_outputs,

            "aligned_tir_backbone":
                aligned_tir_outputs,

            "tir_for_rsdt":
                tir_for_rsdt,

            "tir_for_fusion":
                tir_for_fusion,

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

            "alignment_enabled":
                self.alignment_enabled,

            "alignment_type":
                self.alignment_cfg.get(
                    "type",
                    "identity",
                ),
        }

        if return_rsdt_debug:
            result[
                "rsdt_debug"
            ] = rsdt_debug

        if (
            return_alignment_debug
            or return_rsdt_debug
        ):
            result[
                "alignment_info"
            ] = alignment_info

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
        return_alignment_debug: bool = False,
    ):

        features = self.forward_features(
            rgb,
            tir,

            rgb_semantic=rgb_semantic,

            return_rsdt_debug=(
                return_rsdt_debug
            ),

            return_alignment_debug=(
                return_alignment_debug
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
            or return_alignment_debug
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
        return_alignment_debug: bool = False,
    ):

        # Training batch.
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

            return_alignment_debug=(
                return_alignment_debug
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
        Final head remains in semantic-RGB coordinates.
        Alignment does not change label coordinates.
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

        state = (
            self.rsdt.scalar_state()
        )

        print(
            "RSD-T + Optional Alignment:"
        )

        print(
            f"  RSD-T scales     : "
            f"{list(self.rsdt_scales)}"
        )

        print(
            f"  semantic imgsz   : "
            f"{self.semantic_imgsz}"
        )

        print(
            f"  guidance         : "
            f"{self.use_guidance}"
        )

        print(
            f"  alignment        : "
            f"{self.alignment_enabled}"
        )

        print(
            f"  alignment type   : "
            f"{self.alignment_cfg.get('type', 'identity')}"
        )

        print(
            f"  align -> RSD-T   : "
            f"{self.alignment_use_for_rsdt}"
        )

        print(
            f"  align -> fusion  : "
            f"{self.alignment_use_for_fusion}"
        )

        for scale in self.rsdt_scales:

            print(
                f"  P{scale} alpha        : "
                f"{state[f'alpha_p{scale}']:.6f}"
            )

            print(
                f"  P{scale} gamma        : "
                f"{state[f'gamma_p{scale}']:.6f}"
            )
