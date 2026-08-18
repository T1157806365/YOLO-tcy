"""
RSD-T v1
========
Thermal-Guided Resolution Semantic-Detail Enhancement Module.

Zero-intrusion component for YOLO-tcy.

Inputs:
    rgb_high      : high-resolution RGB image tensor
    rgb_semantic  : low-resolution RGB semantic image tensor
    rgb_p3        : RGB P3 semantic feature
    tir_p3        : TIR P3 semantic feature

Output:
    enhanced RGB P3 feature

Core:
    delta = rgb_high - U(rgb_semantic)
    D = DetailStem(delta)
    A = sigmoid(Guidance(rgb_p3, tir_p3))
    D* = D * (1 + alpha * U(A))
    D3 = Project(AvgPool(D*) || MaxPool(D*))
    P3* = P3 + gamma * D3

gamma is initialized to zero, so P3* == P3 at initialization.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


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
    Lightweight /8 high-resolution detail encoder.

    Example:
        1280 -> 640 -> 320 -> 160
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

            # detail embedding
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
    RGB P3 + TIR P3 -> spatial guidance A.

    A is not a segmentation mask.
    It is a soft importance map for high-resolution RGB detail.
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
        rgb_p3: torch.Tensor,
        tir_p3: torch.Tensor,
    ) -> torch.Tensor:

        tir_p3 = self._resize_like(
            tir_p3,
            rgb_p3,
        )

        rgb_g = self.rgb_proj(
            rgb_p3
        )

        tir_g = self.tir_proj(
            tir_p3
        )

        x = torch.cat(
            [
                rgb_g,
                tir_g,
            ],
            dim=1,
        )

        return torch.sigmoid(
            self.fuse(x)
        )


class RSDTv1(nn.Module):
    def __init__(
        self,
        rgb_p3_channels: int,
        tir_p3_channels: int,
        detail_channels: int = 64,
        stem_channels: int = 24,
        guide_channels: int = 32,
        use_guidance: bool = True,
        exact_identity_when_equal: bool = True,
    ):
        super().__init__()

        self.rgb_p3_channels = int(
            rgb_p3_channels
        )

        self.tir_p3_channels = int(
            tir_p3_channels
        )

        self.use_guidance = bool(
            use_guidance
        )

        self.exact_identity_when_equal = bool(
            exact_identity_when_equal
        )

        self.detail_stem = DetailStem(
            detail_channels=detail_channels,
            stem_channels=stem_channels,
        )

        self.guidance = GuidanceGenerator(
            rgb_channels=rgb_p3_channels,
            tir_channels=tir_p3_channels,
            guide_channels=guide_channels,
        )

        self.detail_to_p3 = nn.Sequential(
            nn.Conv2d(
                detail_channels * 2,
                rgb_p3_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(
                rgb_p3_channels
            ),
            nn.SiLU(
                inplace=True
            ),
        )

        # alpha in (0,1), alpha_init = 0.5
        self.alpha_logit = nn.Parameter(
            torch.tensor(
                0.0,
                dtype=torch.float32,
            )
        )

        # gamma in (-1,1), gamma_init = 0
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

    def compress_detail_to_p3(
        self,
        detail: torch.Tensor,
        rgb_p3: torch.Tensor,
    ) -> torch.Tensor:

        target_size = tuple(
            rgb_p3.shape[-2:]
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

        detail = torch.cat(
            [
                avg_detail,
                max_detail,
            ],
            dim=1,
        )

        return self.detail_to_p3(
            detail
        )

    def forward(
        self,
        rgb_high: torch.Tensor,
        rgb_semantic: torch.Tensor,
        rgb_p3: torch.Tensor,
        tir_p3: torch.Tensor,
        return_debug: bool = False,
    ):

        if (
            rgb_high.ndim != 4
            or rgb_semantic.ndim != 4
            or rgb_p3.ndim != 4
            or tir_p3.ndim != 4
        ):
            raise ValueError(
                "RSD-T 所有输入必须为 BCHW Tensor。"
            )

        if (
            rgb_p3.shape[1]
            != self.rgb_p3_channels
        ):
            raise ValueError(
                "RGB P3 channel mismatch: "
                f"{rgb_p3.shape[1]} != "
                f"{self.rgb_p3_channels}"
            )

        if (
            tir_p3.shape[1]
            != self.tir_p3_channels
        ):
            raise ValueError(
                "TIR P3 channel mismatch: "
                f"{tir_p3.shape[1]} != "
                f"{self.tir_p3_channels}"
            )

        equal_resolution = (
            rgb_high.shape[-2:]
            == rgb_semantic.shape[-2:]
        )

        # 640+640: exact baseline behavior.
        if (
            equal_resolution
            and self.exact_identity_when_equal
        ):

            if not return_debug:
                return rgb_p3

            return (
                rgb_p3,
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

                    "guidance":
                        None,

                    "guidance_high":
                        None,

                    "guided_detail":
                        None,

                    "detail_p3":
                        torch.zeros_like(
                            rgb_p3
                        ),

                    "alpha":
                        torch.sigmoid(
                            self.alpha_logit
                        ),

                    "gamma":
                        torch.tanh(
                            self.gamma_raw
                        ),

                    "enhanced_rgb_p3":
                        rgb_p3,
                },
            )

        # 1. Resolution residual
        (
            residual,
            reconstructed_high,
        ) = self.build_resolution_residual(
            rgb_high,
            rgb_semantic,
        )

        # 2. High-resolution shallow detail
        detail_raw = self.detail_stem(
            residual
        )

        # 3. RGB-T semantic guidance
        if self.use_guidance:

            guidance = self.guidance(
                rgb_p3,
                tir_p3,
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
                    rgb_p3.shape[0],
                    1,
                    rgb_p3.shape[-2],
                    rgb_p3.shape[-1],
                ),
                device=rgb_p3.device,
                dtype=rgb_p3.dtype,
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

        # 4. Soft enhancement
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

        # 5. Detail -> P3
        detail_p3 = (
            self.compress_detail_to_p3(
                guided_detail,
                rgb_p3,
            )
        )

        # 6. Residual P3 injection
        gamma = torch.tanh(
            self.gamma_raw
        )

        enhanced_rgb_p3 = (
            rgb_p3
            + gamma
            * detail_p3
        )

        if not return_debug:
            return enhanced_rgb_p3

        debug: Dict[str, object] = {
            "equal_resolution":
                False,

            "resolution_residual":
                residual,

            "reconstructed_high":
                reconstructed_high,

            "detail_raw":
                detail_raw,

            "guidance":
                guidance,

            "guidance_high":
                guidance_high,

            "guided_detail":
                guided_detail,

            "detail_p3":
                detail_p3,

            "alpha":
                alpha,

            "gamma":
                gamma,

            "enhanced_rgb_p3":
                enhanced_rgb_p3,
        }

        return (
            enhanced_rgb_p3,
            debug,
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
