"""
Configurable Multi-Scale RSD-T v1
=================================

Purpose
-------
Allow RSD-T to be inserted at any combination of:
    P3
    P4
    P5

without changing Python code.

Example YAML:
    rsdt_scales: [3]
    rsdt_scales: [4]
    rsdt_scales: [5]
    rsdt_scales: [3, 4]
    rsdt_scales: [3, 4, 5]

Design
------
The expensive/high-resolution part is SHARED:

    RGB-high + RGB-semantic
        -> resolution residual
        -> DetailStem
        -> D_R

D_R is computed ONCE.

Each selected scale owns an independent adapter:

    RGB P_s + TIR P_s
        -> Guidance_s
        -> A_s

    D_R + A_s
        -> soft enhancement
        -> adaptive AvgPool + MaxPool
        -> scale-specific projection
        -> D_s

    P_s* = P_s + gamma_s * D_s

Thus:
    DetailStem            : shared
    Guidance Generator    : scale-specific
    alpha_s               : scale-specific
    gamma_s               : scale-specific
    detail projection     : scale-specific
"""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ===========================================================================
# Basic blocks
# ===========================================================================

class ConvBNAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
    ):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(
                out_channels
            ),
            nn.SiLU(
                inplace=True
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.block(x)


class DetailStem(nn.Module):
    """
    Shared high-resolution shallow detail encoder.

    Example:
        1280
          -> 640
          -> 320
          -> 160
    """

    def __init__(
        self,
        detail_channels: int = 64,
        stem_channels: int = 24,
    ):
        super().__init__()

        stem_channels = int(
            stem_channels
        )

        detail_channels = int(
            detail_channels
        )

        mid_channels = (
            stem_channels * 2
        )

        self.net = nn.Sequential(
            # /2
            ConvBNAct(
                3,
                stem_channels,
                kernel_size=3,
                stride=2,
            ),

            # /4 depthwise
            ConvBNAct(
                stem_channels,
                stem_channels,
                kernel_size=3,
                stride=2,
                groups=stem_channels,
            ),

            # channel mixing
            ConvBNAct(
                stem_channels,
                mid_channels,
                kernel_size=1,
                stride=1,
            ),

            # /8 depthwise
            ConvBNAct(
                mid_channels,
                mid_channels,
                kernel_size=3,
                stride=2,
                groups=mid_channels,
            ),

            # shared detail embedding
            ConvBNAct(
                mid_channels,
                detail_channels,
                kernel_size=1,
                stride=1,
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.net(x)


class GuidanceGenerator(nn.Module):
    """
    Scale-specific RGB-T semantic guidance generator.

        RGB P_s -> 1x1 projection --\
                                    -> concat -> 3x3 -> 1x1 -> sigmoid
        TIR P_s -> 1x1 projection --/

    Output:
        A_s: [B, 1, H_s, W_s]
    """

    def __init__(
        self,
        rgb_channels: int,
        tir_channels: int,
        guide_channels: int = 32,
    ):
        super().__init__()

        self.rgb_proj = ConvBNAct(
            rgb_channels,
            guide_channels,
            kernel_size=1,
            stride=1,
        )

        self.tir_proj = ConvBNAct(
            tir_channels,
            guide_channels,
            kernel_size=1,
            stride=1,
        )

        self.fuse = nn.Sequential(
            ConvBNAct(
                guide_channels * 2,
                guide_channels,
                kernel_size=3,
                stride=1,
            ),

            nn.Conv2d(
                guide_channels,
                1,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            ),
        )

    @staticmethod
    def _resize_like(
        source: torch.Tensor,
        reference: torch.Tensor,
    ) -> torch.Tensor:

        if (
            source.shape[-2:]
            == reference.shape[-2:]
        ):
            return source

        return F.interpolate(
            source,
            size=reference.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    def forward(
        self,
        rgb_feat: torch.Tensor,
        tir_feat: torch.Tensor,
    ) -> torch.Tensor:

        tir_feat = self._resize_like(
            tir_feat,
            rgb_feat,
        )

        rgb_g = self.rgb_proj(
            rgb_feat
        )

        tir_g = self.tir_proj(
            tir_feat
        )

        fused = torch.cat(
            [
                rgb_g,
                tir_g,
            ],
            dim=1,
        )

        return torch.sigmoid(
            self.fuse(
                fused
            )
        )


# ===========================================================================
# One scale adapter
# ===========================================================================

class RSDTScaleAdapter(nn.Module):
    """
    One independent RSD-T adapter for one pyramid level.

    Example:
        scale = P3 / P4 / P5

    Shared D_R comes from MultiScaleRSDTv1.detail_stem.
    """

    def __init__(
        self,
        rgb_channels: int,
        tir_channels: int,
        detail_channels: int,
        guide_channels: int = 32,
        use_guidance: bool = True,
    ):
        super().__init__()

        self.rgb_channels = int(
            rgb_channels
        )

        self.tir_channels = int(
            tir_channels
        )

        self.use_guidance = bool(
            use_guidance
        )

        self.guidance = GuidanceGenerator(
            rgb_channels=rgb_channels,
            tir_channels=tir_channels,
            guide_channels=guide_channels,
        )

        # Avg + Max -> scale-specific RGB channels
        self.detail_projection = nn.Sequential(
            nn.Conv2d(
                detail_channels * 2,
                rgb_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),

            nn.BatchNorm2d(
                rgb_channels
            ),

            nn.SiLU(
                inplace=True
            ),
        )

        # Independent learnable soft-enhancement strength.
        # sigmoid(0) = 0.5
        self.alpha_logit = nn.Parameter(
            torch.tensor(
                0.0,
                dtype=torch.float32,
            )
        )

        # Independent residual injection strength.
        # tanh(0) = 0 -> exact baseline at initialization.
        self.gamma_raw = nn.Parameter(
            torch.tensor(
                0.0,
                dtype=torch.float32,
            )
        )

    @staticmethod
    def _resize(
        x: torch.Tensor,
        size: Tuple[int, int],
    ) -> torch.Tensor:

        if (
            x.shape[-2:]
            == size
        ):
            return x

        return F.interpolate(
            x,
            size=size,
            mode="bilinear",
            align_corners=False,
        )

    def _compress_detail(
        self,
        detail: torch.Tensor,
        rgb_feat: torch.Tensor,
    ) -> torch.Tensor:

        target_size = tuple(
            rgb_feat.shape[-2:]
        )

        avg_detail = (
            F.adaptive_avg_pool2d(
                detail,
                output_size=target_size,
            )
        )

        max_detail = (
            F.adaptive_max_pool2d(
                detail,
                output_size=target_size,
            )
        )

        merged = torch.cat(
            [
                avg_detail,
                max_detail,
            ],
            dim=1,
        )

        return self.detail_projection(
            merged
        )

    def forward(
        self,
        detail_raw: torch.Tensor,
        rgb_feat: torch.Tensor,
        tir_feat: torch.Tensor,
        return_debug: bool = False,
    ):

        if (
            rgb_feat.shape[1]
            != self.rgb_channels
        ):
            raise ValueError(
                "RGB feature channels mismatch: "
                f"{rgb_feat.shape[1]} != "
                f"{self.rgb_channels}"
            )

        if (
            tir_feat.shape[1]
            != self.tir_channels
        ):
            raise ValueError(
                "TIR feature channels mismatch: "
                f"{tir_feat.shape[1]} != "
                f"{self.tir_channels}"
            )

        # -------------------------------------------------------
        # 1. Scale-specific semantic guidance
        # -------------------------------------------------------

        if self.use_guidance:

            guidance = self.guidance(
                rgb_feat,
                tir_feat,
            )

            guidance_high = self._resize(
                guidance,
                size=tuple(
                    detail_raw.shape[-2:]
                ),
            )

        else:

            guidance = torch.zeros(
                (
                    rgb_feat.shape[0],
                    1,
                    rgb_feat.shape[-2],
                    rgb_feat.shape[-1],
                ),
                device=rgb_feat.device,
                dtype=rgb_feat.dtype,
            )

            guidance_high = torch.zeros(
                (
                    detail_raw.shape[0],
                    1,
                    detail_raw.shape[-2],
                    detail_raw.shape[-1],
                ),
                device=detail_raw.device,
                dtype=detail_raw.dtype,
            )

        # -------------------------------------------------------
        # 2. Soft enhancement
        # -------------------------------------------------------

        alpha = torch.sigmoid(
            self.alpha_logit
        )

        guided_detail = (
            detail_raw
            * (
                1.0
                + alpha
                * guidance_high
            )
        )

        # -------------------------------------------------------
        # 3. Compress shared detail to current P_s resolution
        # -------------------------------------------------------

        detail_scale = self._compress_detail(
            guided_detail,
            rgb_feat,
        )

        # -------------------------------------------------------
        # 4. Residual injection
        # -------------------------------------------------------

        gamma = torch.tanh(
            self.gamma_raw
        )

        enhanced_rgb = (
            rgb_feat
            + gamma
            * detail_scale
        )

        if not return_debug:
            return enhanced_rgb

        return (
            enhanced_rgb,
            {
                "guidance":
                    guidance,

                "guidance_high":
                    guidance_high,

                "guided_detail":
                    guided_detail,

                "detail_scale":
                    detail_scale,

                "alpha":
                    alpha,

                "gamma":
                    gamma,

                "enhanced_rgb":
                    enhanced_rgb,
            },
        )

    @torch.no_grad()
    def scalar_state(
        self,
    ) -> Dict[str, float]:

        return {
            "alpha":
                float(
                    torch.sigmoid(
                        self.alpha_logit
                    ).cpu().item()
                ),

            "gamma":
                float(
                    torch.tanh(
                        self.gamma_raw
                    ).cpu().item()
                ),
        }


# ===========================================================================
# Multi-scale RSD-T
# ===========================================================================

class MultiScaleRSDTv1(nn.Module):
    """
    Configurable multi-scale RSD-T.

    Parameters
    ----------
    rgb_channels:
        {3: C3, 4: C4, 5: C5}

    tir_channels:
        {3: C3_t, 4: C4_t, 5: C5_t}

    scales:
        Any non-empty subset of {3, 4, 5}.

    Example:
        scales=(3,)
        scales=(4,)
        scales=(3, 4)
        scales=(3, 4, 5)
    """

    VALID_SCALES = (
        3,
        4,
        5,
    )

    def __init__(
        self,
        rgb_channels: Mapping[int, int],
        tir_channels: Mapping[int, int],
        scales: Iterable[int] = (3,),
        detail_channels: int = 64,
        stem_channels: int = 24,
        guide_channels: int = 32,
        use_guidance: bool = True,
        exact_identity_when_equal: bool = True,
    ):
        super().__init__()

        scales = tuple(
            sorted(
                {
                    int(s)
                    for s in scales
                }
            )
        )

        if not scales:
            raise ValueError(
                "rsdt_scales 不能为空。"
            )

        invalid = [
            s
            for s in scales
            if s not in self.VALID_SCALES
        ]

        if invalid:
            raise ValueError(
                "rsdt_scales 仅支持 "
                "[3], [4], [5], [3,4], "
                "[3,5], [4,5], [3,4,5]。"
                f" 当前非法值: {invalid}"
            )

        self.scales = scales

        self.detail_channels = int(
            detail_channels
        )

        self.use_guidance = bool(
            use_guidance
        )

        self.exact_identity_when_equal = bool(
            exact_identity_when_equal
        )

        # -------------------------------------------------------
        # Shared detail branch: compute exactly once.
        # -------------------------------------------------------

        self.detail_stem = DetailStem(
            detail_channels=detail_channels,
            stem_channels=stem_channels,
        )

        # -------------------------------------------------------
        # Independent adapter for every selected pyramid level.
        # -------------------------------------------------------

        adapters = {}

        for scale in self.scales:

            if scale not in rgb_channels:
                raise KeyError(
                    f"Missing RGB channels for P{scale}."
                )

            if scale not in tir_channels:
                raise KeyError(
                    f"Missing TIR channels for P{scale}."
                )

            adapters[
                str(scale)
            ] = RSDTScaleAdapter(
                rgb_channels=int(
                    rgb_channels[scale]
                ),

                tir_channels=int(
                    tir_channels[scale]
                ),

                detail_channels=(
                    detail_channels
                ),

                guide_channels=(
                    guide_channels
                ),

                use_guidance=(
                    use_guidance
                ),
            )

        self.adapters = nn.ModuleDict(
            adapters
        )

    @staticmethod
    def _resize(
        x: torch.Tensor,
        size: Tuple[int, int],
    ) -> torch.Tensor:

        if (
            x.shape[-2:]
            == size
        ):
            return x

        return F.interpolate(
            x,
            size=size,
            mode="bilinear",
            align_corners=False,
        )

    def build_resolution_residual(
        self,
        rgb_high: torch.Tensor,
        rgb_semantic: torch.Tensor,
    ):

        reconstructed_high = self._resize(
            rgb_semantic,
            size=tuple(
                rgb_high.shape[-2:]
            ),
        )

        residual = (
            rgb_high
            - reconstructed_high
        )

        return (
            residual,
            reconstructed_high,
        )

    def forward(
        self,
        rgb_high: torch.Tensor,
        rgb_semantic: torch.Tensor,
        rgb_features: Mapping[
            int,
            torch.Tensor,
        ],
        tir_features: Mapping[
            int,
            torch.Tensor,
        ],
        return_debug: bool = False,
    ):
        """
        Returns
        -------
        enhanced_features:
            dict:
                {
                    3: enhanced P3,
                    4: enhanced P4,
                    5: enhanced P5
                }
            only selected scales are returned.

        debug:
            optional per-scale diagnostics.
        """

        equal_resolution = (
            rgb_high.shape[-2:]
            == rgb_semantic.shape[-2:]
        )

        # -------------------------------------------------------
        # Equal resolution:
        # exact identity, no detail branch computation.
        # -------------------------------------------------------

        if (
            equal_resolution
            and self.exact_identity_when_equal
        ):

            outputs = {
                scale:
                    rgb_features[
                        scale
                    ]

                for scale in self.scales
            }

            if not return_debug:
                return outputs

            return (
                outputs,
                {
                    "equal_resolution":
                        True,

                    "resolution_residual":
                        torch.zeros_like(
                            rgb_high
                        ),

                    "reconstructed_high":
                        rgb_high,

                    "detail_raw":
                        None,

                    "scales":
                        {},
                },
            )

        # -------------------------------------------------------
        # 1. Shared resolution residual
        # -------------------------------------------------------

        (
            residual,
            reconstructed_high,
        ) = self.build_resolution_residual(
            rgb_high,
            rgb_semantic,
        )

        # -------------------------------------------------------
        # 2. Shared DetailStem: ONE computation only
        # -------------------------------------------------------

        detail_raw = self.detail_stem(
            residual
        )

        enhanced_features = {}
        scale_debug = {}

        # -------------------------------------------------------
        # 3. Independent scale adapters
        # -------------------------------------------------------

        for scale in self.scales:

            adapter = self.adapters[
                str(scale)
            ]

            if return_debug:

                (
                    enhanced,
                    debug,
                ) = adapter(
                    detail_raw=detail_raw,

                    rgb_feat=(
                        rgb_features[
                            scale
                        ]
                    ),

                    tir_feat=(
                        tir_features[
                            scale
                        ]
                    ),

                    return_debug=True,
                )

                scale_debug[
                    scale
                ] = debug

            else:

                enhanced = adapter(
                    detail_raw=detail_raw,

                    rgb_feat=(
                        rgb_features[
                            scale
                        ]
                    ),

                    tir_feat=(
                        tir_features[
                            scale
                        ]
                    ),

                    return_debug=False,
                )

            enhanced_features[
                scale
            ] = enhanced

        if not return_debug:
            return enhanced_features

        return (
            enhanced_features,
            {
                "equal_resolution":
                    False,

                "resolution_residual":
                    residual,

                "reconstructed_high":
                    reconstructed_high,

                "detail_raw":
                    detail_raw,

                "scales":
                    scale_debug,
            },
        )

    @torch.no_grad()
    def scalar_state(
        self,
    ) -> Dict[str, float]:
        """
        Backward-compatible keys:
            alpha
            gamma

        For multi-scale use, these are the arithmetic means.

        Scale-specific keys:
            alpha_p3
            gamma_p3
            alpha_p4
            gamma_p4
            ...
        """

        result = {}

        alphas = []
        gammas = []

        for scale in self.scales:

            state = self.adapters[
                str(scale)
            ].scalar_state()

            alpha = state[
                "alpha"
            ]

            gamma = state[
                "gamma"
            ]

            result[
                f"alpha_p{scale}"
            ] = alpha

            result[
                f"gamma_p{scale}"
            ] = gamma

            alphas.append(
                alpha
            )

            gammas.append(
                gamma
            )

        result[
            "alpha"
        ] = (
            sum(alphas)
            / len(alphas)
        )

        result[
            "gamma"
        ] = (
            sum(gammas)
            / len(gammas)
        )

        return result


# Backward-compatible alias.
# New code should prefer MultiScaleRSDTv1.
RSDTv1 = MultiScaleRSDTv1
