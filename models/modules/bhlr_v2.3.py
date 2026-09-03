"""
BHLR-v2: lightweight dual-mode high-resolution lost-detail recovery.

BHLR-v2 design:
----------------

    RGB-high + RGB-semantic
        -> high-resolution residual
        -> PixelUnshuffle
        -> 1x1 channel compression
        -> DWConv 3x3 stride=2
        -> 1x1 projection
        -> DWConv 3x3
        -> lightweight shared detail bank

    detail bank
        -> adaptive AvgPool + MaxPool
        -> scale-specific detail D_s

Guidance has two modes:
1) external:
       reuse RLSFA reliable RGB/TIR structures + targetness.
2) fallback:
       when RLSFA is absent (or forced off for ablation),
       generate lightweight local spatial structures + self targetness.

The old BHLR-v1:
    Global FFT -> Support -> Gate -> Support*Gate*Detail
is removed.

BHLR-v2 uses ONE centered detail selector:
    multiplier = 1 + tanh(selector_logit)  in (0, 2)
    usable_detail = multiplier * detail_scale

Final residual injection is preserved:
    enhanced_rgb = rgb_feat + tanh(gamma_raw) * usable_detail

gamma_raw starts at 0 -> exact RGB baseline at initialization.
"""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, groups=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                int(c1),
                int(c2),
                kernel_size=int(k),
                stride=int(s),
                padding=int(k) // 2,
                groups=int(groups),
                bias=False,
            ),
            nn.BatchNorm2d(int(c2)),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class SharedLostDetailEncoder(nn.Module):
    """
    Lightweight BHLR-v2 shared lost-detail encoder.

    1280 / 640, ratio=2:
        residual_s2d: [B,12,640,640]
        -> 1x1        [B,S,640,640]
        -> DW s2      [B,S,320,320]
        -> 1x1        [B,D,320,320]
        -> DW         [B,D,320,320]

    1920 / 640, ratio=3:
        residual_s2d: [B,27,640,640]
        then the same lightweight pipeline.

    Compared with BHLR-v1, the expensive feature bank is no longer kept at
    640x640 with large channel width.
    """

    def __init__(self, ratio=3, detail_channels=48, stem_channels=16):
        super().__init__()

        ratio = int(ratio)
        if ratio < 1:
            raise ValueError(f"ratio must be >= 1, got {ratio}")

        in_channels = 3 * ratio * ratio
        c = int(stem_channels)
        d = int(detail_channels)

        self.net = nn.Sequential(
            # Cross-channel mixing while still at semantic spatial size.
            ConvBNAct(in_channels, c, 1, 1),

            # Immediately reduce spatial cost.
            ConvBNAct(c, c, 3, 2, groups=c),

            # Compact detail embedding.
            ConvBNAct(c, d, 1, 1),

            # Cheap local refinement at 320x320 (for semantic 640).
            ConvBNAct(d, d, 3, 1, groups=d),
        )

    def forward(self, x):
        return self.net(x)


class MultiKernelSpatialStructure(nn.Module):
    """Fallback local spatial-structure extractor; no FFT."""

    def __init__(self, in_channels, out_channels=24):
        super().__init__()

        c = int(out_channels)

        self.proj = ConvBNAct(
            int(in_channels),
            c,
            1,
            1,
        )

        self.dw3 = ConvBNAct(
            c,
            c,
            3,
            1,
            groups=c,
        )

        self.dw5 = ConvBNAct(
            c,
            c,
            5,
            1,
            groups=c,
        )

        self.fuse = ConvBNAct(
            c * 2,
            c,
            1,
            1,
        )

    def forward(self, x):
        x = self.proj(x)
        structure = self.fuse(
            torch.cat(
                [
                    self.dw3(x),
                    self.dw5(x),
                ],
                dim=1,
            )
        )
        return x, structure


class FallbackGuidanceGenerator(nn.Module):
    """
    BHLR-v2 standalone mode.

    When RLSFA guidance is unavailable:
        RGB P_s -> local spatial structure S_R
        TIR P_s -> local spatial structure S_T

        [R, T, |R-T|] -> lightweight targetness A_self

    No global FFT is used.
    """

    def __init__(
        self,
        rgb_channels,
        tir_channels,
        guide_channels=24,
    ):
        super().__init__()

        g = int(guide_channels)

        self.rgb_structure = MultiKernelSpatialStructure(
            rgb_channels,
            out_channels=g,
        )
        self.tir_structure = MultiKernelSpatialStructure(
            tir_channels,
            out_channels=g,
        )

        self.target_head = nn.Sequential(
            ConvBNAct(
                g * 3,
                g,
                1,
                1,
            ),
            ConvBNAct(
                g,
                g,
                3,
                1,
                groups=g,
            ),
            nn.Conv2d(
                g,
                1,
                kernel_size=1,
                bias=True,
            ),
        )

        # Neutral initial targetness = 0.5.
        nn.init.zeros_(self.target_head[-1].weight)
        nn.init.zeros_(self.target_head[-1].bias)

    @staticmethod
    def _resize_like(source, reference):
        if source.shape[-2:] == reference.shape[-2:]:
            return source

        return F.interpolate(
            source,
            size=reference.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    def forward(self, rgb_feat, tir_feat):
        tir_feat = self._resize_like(
            tir_feat,
            rgb_feat,
        )

        r, sr = self.rgb_structure(
            rgb_feat
        )

        t, st = self.tir_structure(
            tir_feat
        )

        target_logits = self.target_head(
            torch.cat(
                [
                    r,
                    t,
                    (r - t).abs(),
                ],
                dim=1,
            )
        )

        targetness = torch.sigmoid(
            target_logits
        )

        return {
            "rgb_structure": sr,
            "tir_structure": st,
            "targetness": targetness,
            "targetness_logits": target_logits,
        }


class SENetChannelAttention(nn.Module):
    """
    Standard Squeeze-and-Excitation (SE) channel attention.

    Used only on the six projected selector feature groups:
        r, t, |r-t|, d, sr, st

    Targetness is intentionally NOT included in SE because it is a spatial
    soft prior rather than a normal feature channel.
    """

    def __init__(self, channels, reduction=16):
        super().__init__()

        channels = int(channels)
        reduction = max(int(reduction), 1)
        hidden = max(channels // reduction, 1)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(
                channels,
                hidden,
                kernel_size=1,
                bias=True,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                hidden,
                channels,
                kernel_size=1,
                bias=True,
            ),
            nn.Sigmoid(),
        )

    def forward(self, x):
        weight = self.fc(
            self.pool(x)
        )
        return x * weight


class CrossModalDetailSelector(nn.Module):
    """
    Single BHLR-v2 detail selector.

    All feature inputs are projected to a small channel width first:
        RGB semantic
        TIR semantic
        |RGB-TIR|
        high-resolution detail D_s
        RGB structure
        TIR structure

    The six projected groups and targetness are first fused by the existing
    1x1 Conv-BN-SiLU block. SENet is then applied to the fused guide feature
    before the existing depthwise spatial refinement.

    The output is CENTERED at identity:
        multiplier = 1 + tanh(z)

        z = 0 -> multiplier = 1
        z < 0 -> suppress irrelevant detail
        z > 0 -> enhance useful detail
    """

    def __init__(
        self,
        rgb_channels,
        tir_channels,
        detail_channels,
        guide_channels=24,
        se_reduction=16,
    ):
        super().__init__()

        g = int(guide_channels)

        self.rgb_proj = ConvBNAct(
            rgb_channels,
            g,
            1,
            1,
        )

        self.tir_proj = ConvBNAct(
            tir_channels,
            g,
            1,
            1,
        )

        self.detail_proj = ConvBNAct(
            detail_channels,
            g,
            1,
            1,
        )

        self.rgb_structure_proj = ConvBNAct(
            g,
            g,
            1,
            1,
        )

        self.tir_structure_proj = ConvBNAct(
            g,
            g,
            1,
            1,
        )

        # SENet is applied AFTER the existing 1x1 fusion:
        # [r,t,|r-t|,d,sr,st,targetness] -> 1x1 Conv-BN-SiLU -> g channels
        # -> SENet -> DWConv -> Z3.
        self.channel_attention = SENetChannelAttention(
            channels=g,
            reduction=se_reduction,
        )

        # [r,t,|r-t|,d,sr,st] -> 6*g; plus targetness -> 1.
        self.head = nn.Sequential(
            ConvBNAct(
                g * 6 + 1,
                g,
                1,
                1,
            ),
            ConvBNAct(
                g,
                g,
                3,
                1,
                groups=g,
            ),
            nn.Conv2d(
                g,
                1,
                kernel_size=1,
                bias=False,
            ),
        )

        # Selector multiplier still starts exactly at 1:
        # final 1x1 weight = 0 -> Z3 = 0 -> M3 = 1 + tanh(0) = 1.
        # No output bias is used, preventing a learned global spatial offset
        # from trivially lifting the entire Z3 map.
        nn.init.zeros_(self.head[-1].weight)

    @staticmethod
    def _resize_like(source, reference):
        if source.shape[-2:] == reference.shape[-2:]:
            return source

        return F.interpolate(
            source,
            size=reference.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    def forward(
        self,
        rgb_feat,
        tir_feat,
        detail_scale,
        rgb_structure,
        tir_structure,
        targetness,
    ):
        tir_feat = self._resize_like(
            tir_feat,
            rgb_feat,
        )

        detail_scale = self._resize_like(
            detail_scale,
            rgb_feat,
        )

        rgb_structure = self._resize_like(
            rgb_structure,
            rgb_feat,
        )

        tir_structure = self._resize_like(
            tir_structure,
            rgb_feat,
        )

        targetness = self._resize_like(
            targetness,
            rgb_feat,
        )

        r = self.rgb_proj(
            rgb_feat
        )

        t = self.tir_proj(
            tir_feat
        )

        d = self.detail_proj(
            detail_scale
        )

        sr = self.rgb_structure_proj(
            rgb_structure
        )

        st = self.tir_structure_proj(
            tir_structure
        )

        feature_groups = torch.cat(
            [
                r,
                t,
                (r - t).abs(),
                d,
                sr,
                st,
            ],
            dim=1,
        )

        # Keep the original selector input unchanged.
        selector_input = torch.cat(
            [
                feature_groups,
                targetness,
            ],
            dim=1,
        )

        # Only move SENet:
        # 145ch -> existing 1x1 Conv-BN-SiLU -> 24ch -> SENet
        # -> existing DWConv 3x3 -> existing output 1x1 Conv -> Z3.
        selector_feature = self.head[0](
            selector_input
        )

        selector_feature = self.channel_attention(
            selector_feature
        )

        selector_feature = self.head[1](
            selector_feature
        )

        selector_logit = self.head[2](
            selector_feature
        )

        selector_multiplier = (
            1.0
            + torch.tanh(
                selector_logit
            )
        )

        return (
            selector_logit,
            selector_multiplier,
        )


class BHLRScaleAdapter(nn.Module):
    """
    One BHLR-v2 adapter.

    guidance_mode:
        "auto":
            external RLSFA guidance if supplied, otherwise fallback.
        "fallback":
            always use BHLR-v2 self spatial guidance.
            This is the recommended ablation mode.
        "external":
            require RLSFA guidance.
    """

    VALID_GUIDANCE_MODES = {
        "auto",
        "fallback",
        "external",
    }

    def __init__(
        self,
        rgb_channels,
        tir_channels,
        detail_channels,
        guide_channels=24,
        freq_cutoff=0.15,        # retained only for old API compatibility
        freq_sharpness=24.0,    # retained only for old API compatibility
        guidance_mode="auto",
        detach_external_guidance=True,
        external_structure_channels=32,
    ):
        super().__init__()

        del freq_cutoff, freq_sharpness

        self.rgb_channels = int(
            rgb_channels
        )

        self.tir_channels = int(
            tir_channels
        )

        self.guide_channels = int(
            guide_channels
        )

        self.guidance_mode = str(
            guidance_mode
        ).strip().lower()

        if (
            self.guidance_mode
            not in self.VALID_GUIDANCE_MODES
        ):
            raise ValueError(
                "guidance_mode must be one of "
                f"{sorted(self.VALID_GUIDANCE_MODES)}, "
                f"got {self.guidance_mode}"
            )

        self.detach_external_guidance = bool(
            detach_external_guidance
        )

        external_structure_channels = int(
            external_structure_channels
        )

        # AvgPool + MaxPool -> current RGB feature channels.
        self.detail_projection = nn.Sequential(
            nn.Conv2d(
                int(detail_channels) * 2,
                self.rgb_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(
                self.rgb_channels
            ),
            nn.SiLU(
                inplace=True
            ),
        )

        self.fallback = FallbackGuidanceGenerator(
            rgb_channels=self.rgb_channels,
            tir_channels=self.tir_channels,
            guide_channels=self.guide_channels,
        )

        # RLSFA reliable structures are hidden-channel features (normally 32ch).
        self.external_rgb_adapter = ConvBNAct(
            external_structure_channels,
            self.guide_channels,
            1,
            1,
        )

        self.external_tir_adapter = ConvBNAct(
            external_structure_channels,
            self.guide_channels,
            1,
            1,
        )

        self.selector = CrossModalDetailSelector(
            rgb_channels=self.rgb_channels,
            tir_channels=self.tir_channels,
            detail_channels=self.rgb_channels,
            guide_channels=self.guide_channels,
        )

        # Exact original RGB feature at initialization.
        self.gamma_raw = nn.Parameter(
            torch.tensor(
                0.0,
                dtype=torch.float32,
            )
        )

    @staticmethod
    def _resize_like(source, reference):
        if source.shape[-2:] == reference.shape[-2:]:
            return source

        return F.interpolate(
            source,
            size=reference.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    def _compress_detail(
        self,
        detail_raw,
        rgb_feat,
        return_debug=False,
    ):
        target = rgb_feat.shape[-2:]

        avg = F.adaptive_avg_pool2d(
            detail_raw,
            target,
        )

        mx = F.adaptive_max_pool2d(
            detail_raw,
            target,
        )

        detail_scale = self.detail_projection(
            torch.cat(
                [
                    avg,
                    mx,
                ],
                dim=1,
            )
        )

        if not return_debug:
            return detail_scale

        return (
            detail_scale,
            avg,
            mx,
        )

    def _prepare_external_guidance(
        self,
        external_guidance,
        rgb_feat,
    ):
        required = (
            "rgb_structure",
            "tir_structure",
            "targetness",
        )

        missing = [
            key
            for key in required
            if key not in external_guidance
        ]

        if missing:
            raise KeyError(
                "BHLR-v2 external_guidance "
                f"missing keys: {missing}"
            )

        br = external_guidance[
            "rgb_structure"
        ]

        bt = external_guidance[
            "tir_structure"
        ]

        targetness = external_guidance[
            "targetness"
        ]

        if self.detach_external_guidance:
            br = br.detach()
            bt = bt.detach()
            targetness = targetness.detach()

        br = self._resize_like(
            br,
            rgb_feat,
        )

        bt = self._resize_like(
            bt,
            rgb_feat,
        )

        targetness = self._resize_like(
            targetness,
            rgb_feat,
        )

        br = self.external_rgb_adapter(
            br
        )

        bt = self.external_tir_adapter(
            bt
        )

        return {
            "rgb_structure": br,
            "tir_structure": bt,
            "targetness": targetness.clamp(
                0.0,
                1.0,
            ),
            "targetness_logits": None,
        }

    def _resolve_guidance(
        self,
        rgb_feat,
        tir_feat,
        external_guidance=None,
    ):
        if self.guidance_mode == "fallback":
            return (
                "fallback",
                self.fallback(
                    rgb_feat,
                    tir_feat,
                ),
            )

        if self.guidance_mode == "external":
            if external_guidance is None:
                raise RuntimeError(
                    "BHLR-v2 guidance_mode='external' "
                    "requires RLSFA external guidance."
                )

            return (
                "external",
                self._prepare_external_guidance(
                    external_guidance,
                    rgb_feat,
                ),
            )

        # auto
        if external_guidance is not None:
            return (
                "external",
                self._prepare_external_guidance(
                    external_guidance,
                    rgb_feat,
                ),
            )

        return (
            "fallback",
            self.fallback(
                rgb_feat,
                tir_feat,
            ),
        )

    def forward(
        self,
        detail_raw,
        rgb_feat,
        tir_feat,
        external_guidance=None,
        return_debug=False,
    ):
        if (
            rgb_feat.shape[1]
            != self.rgb_channels
        ):
            raise ValueError(
                "RGB channels mismatch: "
                f"{rgb_feat.shape[1]} "
                f"!= {self.rgb_channels}"
            )

        if (
            tir_feat.shape[1]
            != self.tir_channels
        ):
            raise ValueError(
                "TIR channels mismatch: "
                f"{tir_feat.shape[1]} "
                f"!= {self.tir_channels}"
            )

        tir_feat = self._resize_like(
            tir_feat,
            rgb_feat,
        )

        if return_debug:
            (
                detail_scale,
                detail_avg,
                detail_max,
            ) = self._compress_detail(
                detail_raw,
                rgb_feat,
                return_debug=True,
            )
        else:
            detail_scale = self._compress_detail(
                detail_raw,
                rgb_feat,
                return_debug=False,
            )
            detail_avg = None
            detail_max = None

        (
            guidance_source,
            guidance,
        ) = self._resolve_guidance(
            rgb_feat=rgb_feat,
            tir_feat=tir_feat,
            external_guidance=external_guidance,
        )

        (
            selector_logit,
            selector_multiplier,
        ) = self.selector(
            rgb_feat=rgb_feat,
            tir_feat=tir_feat,
            detail_scale=detail_scale,
            rgb_structure=guidance[
                "rgb_structure"
            ],
            tir_structure=guidance[
                "tir_structure"
            ],
            targetness=guidance[
                "targetness"
            ],
        )

        usable_detail = (
            selector_multiplier
            * detail_scale
        )

        gamma = torch.tanh(
            self.gamma_raw
        )

        enhanced_rgb = (
            rgb_feat
            + gamma
            * usable_detail
        )

        if not return_debug:
            return enhanced_rgb

        # True BHLR-v2 debug keys + compatibility aliases for the old
        # visualization script. The aliases are kept only to prevent old
        # debug code from crashing; their semantics are now BHLR-v2.
        return enhanced_rgb, {
            "guidance_source":
                guidance_source,

            "rgb_structure":
                guidance["rgb_structure"],

            "tir_structure":
                guidance["tir_structure"],

            "targetness":
                guidance["targetness"],

            "targetness_logits":
                guidance.get(
                    "targetness_logits",
                    None,
                ),

            "detail_avg":
                detail_avg,

            "detail_max":
                detail_max,

            "detail_scale":
                detail_scale,

            "selector_logit":
                selector_logit,

            "selector_multiplier":
                selector_multiplier,

            "usable_detail":
                usable_detail,

            "gamma":
                gamma,

            "enhanced_rgb":
                enhanced_rgb,

            # ---- old-name compatibility aliases ----
            "support_map":
                guidance["targetness"],

            "rgb_boundary":
                guidance["rgb_structure"],

            "tir_boundary":
                guidance["tir_structure"],

            "detail_gate":
                selector_multiplier,
        }

    @torch.no_grad()
    def scalar_state(self):
        return {
            "gamma":
                float(
                    torch.tanh(
                        self.gamma_raw
                    ).cpu().item()
                )
        }


class MultiScaleBHLRv2(nn.Module):
    """Multi-scale BHLR-v2 on any selected subset of P3/P4/P5."""

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
        high_to_semantic_ratio=3,
        detail_channels=48,
        stem_channels=16,
        guide_channels=24,
        freq_cutoff=0.15,       # old API compatibility; not used by v2
        freq_sharpness=24.0,   # old API compatibility; not used by v2
        exact_identity_when_equal=True,
        guidance_mode="auto",
        detach_external_guidance=True,
        external_structure_channels=32,
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
                "bhlr_scales cannot be empty"
            )

        invalid = [
            s
            for s in scales
            if s not in self.VALID_SCALES
        ]

        if invalid:
            raise ValueError(
                "bhlr_scales only support "
                f"P3/P4/P5, invalid={invalid}"
            )

        self.scales = scales

        self.ratio = int(
            high_to_semantic_ratio
        )

        self.guidance_mode = str(
            guidance_mode
        ).strip().lower()

        self.detach_external_guidance = bool(
            detach_external_guidance
        )

        self.exact_identity_when_equal = bool(
            exact_identity_when_equal
        )

        self.detail_stem = SharedLostDetailEncoder(
            ratio=self.ratio,
            detail_channels=int(
                detail_channels
            ),
            stem_channels=int(
                stem_channels
            ),
        )

        self.adapters = nn.ModuleDict({
            str(scale):
                BHLRScaleAdapter(
                    rgb_channels=int(
                        rgb_channels[scale]
                    ),

                    tir_channels=int(
                        tir_channels[scale]
                    ),

                    detail_channels=int(
                        detail_channels
                    ),

                    guide_channels=int(
                        guide_channels
                    ),

                    freq_cutoff=float(
                        freq_cutoff
                    ),

                    freq_sharpness=float(
                        freq_sharpness
                    ),

                    guidance_mode=(
                        self.guidance_mode
                    ),

                    detach_external_guidance=(
                        self.detach_external_guidance
                    ),

                    external_structure_channels=int(
                        external_structure_channels
                    ),
                )

            for scale in self.scales
        })

    @staticmethod
    def _resize(x, size):
        if (
            x.shape[-2:]
            == tuple(size)
        ):
            return x

        return F.interpolate(
            x,
            size=size,
            mode="bilinear",
            align_corners=False,
        )

    def build_lost_detail_bank(
        self,
        rgb_high,
        rgb_semantic,
    ):
        expected_h = (
            rgb_semantic.shape[-2]
            * self.ratio
        )

        expected_w = (
            rgb_semantic.shape[-1]
            * self.ratio
        )

        if (
            rgb_high.shape[-2:]
            != (
                expected_h,
                expected_w,
            )
        ):
            raise ValueError(
                "BHLR-v2 PixelUnshuffle requires "
                "exact integer scale ratio.\n"
                f"ratio={self.ratio}\n"
                f"rgb_high="
                f"{tuple(rgb_high.shape[-2:])}\n"
                f"rgb_semantic="
                f"{tuple(rgb_semantic.shape[-2:])}\n"
                f"expected_high="
                f"{(expected_h, expected_w)}"
            )

        reconstructed_high = self._resize(
            rgb_semantic,
            rgb_high.shape[-2:],
        )

        residual_high = (
            rgb_high
            - reconstructed_high
        )

        residual_s2d = F.pixel_unshuffle(
            residual_high,
            downscale_factor=self.ratio,
        )

        detail_raw = self.detail_stem(
            residual_s2d
        )

        # Keep the same tensor return structure used by the detector wrapper.
        return (
            detail_raw,
            reconstructed_high,
            residual_high,
        )

    @staticmethod
    def _guidance_for_scale(
        external_guidance,
        scale,
    ):
        if external_guidance is None:
            return None

        # Direct single-scale dict.
        required = (
            "rgb_structure",
            "tir_structure",
            "targetness",
        )

        if all(
            key in external_guidance
            for key in required
        ):
            return external_guidance

        if scale in external_guidance:
            return external_guidance[scale]

        if str(scale) in external_guidance:
            return external_guidance[
                str(scale)
            ]

        return None

    def forward(
        self,
        rgb_high,
        rgb_semantic,
        rgb_features: Mapping[
            int,
            torch.Tensor,
        ],
        tir_features: Mapping[
            int,
            torch.Tensor,
        ],
        external_guidance: Optional[
            Mapping
        ] = None,
        return_debug=False,
    ):
        equal = (
            rgb_high.shape[-2:]
            == rgb_semantic.shape[-2:]
        )

        if (
            equal
            and self.exact_identity_when_equal
        ):
            outputs = {
                s: rgb_features[s]
                for s in self.scales
            }

            if not return_debug:
                return outputs

            return outputs, {
                "equal_resolution": True,
                "detail_raw": None,
                "scales": {},
            }

        (
            detail_raw,
            reconstructed_high,
            residual_high,
        ) = self.build_lost_detail_bank(
            rgb_high,
            rgb_semantic,
        )

        # Debug only. Recomputed only when requested.
        residual_s2d = None

        if return_debug:
            residual_s2d = (
                F.pixel_unshuffle(
                    residual_high,
                    downscale_factor=(
                        self.ratio
                    ),
                )
            )

        outputs = {}
        debug_scales = {}

        for scale in self.scales:
            adapter = self.adapters[
                str(scale)
            ]

            guidance = (
                self._guidance_for_scale(
                    external_guidance,
                    scale,
                )
            )

            if return_debug:
                (
                    enhanced,
                    dbg,
                ) = adapter(
                    detail_raw,
                    rgb_features[scale],
                    tir_features[scale],
                    external_guidance=guidance,
                    return_debug=True,
                )

                debug_scales[
                    scale
                ] = dbg

            else:
                enhanced = adapter(
                    detail_raw,
                    rgb_features[scale],
                    tir_features[scale],
                    external_guidance=guidance,
                    return_debug=False,
                )

            outputs[
                scale
            ] = enhanced

        if not return_debug:
            return outputs

        return outputs, {
            "equal_resolution":
                False,

            "reconstructed_high":
                reconstructed_high,

            "resolution_residual":
                residual_high,

            "residual_s2d":
                residual_s2d,

            # Detail-bank debug tensor.
            "detail_raw":
                detail_raw,

            # More explicit v2 alias.
            "detail_bank":
                detail_raw,

            "scales":
                debug_scales,
        }

    @torch.no_grad()
    def scalar_state(
        self,
    ) -> Dict[str, float]:
        """
        Keep old RSD-T/BHLR logging API compatible.
        """

        result = {}
        gammas = []

        for scale in self.scales:
            gamma = (
                self.adapters[
                    str(scale)
                ].scalar_state()[
                    "gamma"
                ]
            )

            # Compatibility-only alpha.
            result[
                f"alpha_p{scale}"
            ] = 1.0

            result[
                f"gamma_p{scale}"
            ] = gamma

            gammas.append(
                gamma
            )

        result[
            "alpha"
        ] = 1.0

        result[
            "gamma"
        ] = (
            sum(gammas)
            / max(
                len(gammas),
                1,
            )
        )

        return result
