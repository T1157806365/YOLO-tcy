"""
BHLR: Boundary-guided High-resolution Lost-detail Recovery
==========================================================

Configurable on any subset of P3/P4/P5.
The interface intentionally matches MultiScaleRSDTv1 so the existing
RSD-T trainer/dataset/validation adapters can be reused.
"""

from __future__ import annotations
from typing import Dict, Iterable, Mapping
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, groups=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(c1, c2, k, s, k // 2, groups=groups, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class FrequencyBoundary(nn.Module):
    def __init__(self, cutoff=0.15, sharpness=24.0):
        super().__init__()
        self.cutoff = float(cutoff)
        self.sharpness = float(sharpness)

    def forward(self, x):
        dtype = x.dtype
        xf = x.float()
        _, _, h, w = xf.shape
        fy = torch.fft.fftfreq(h, device=xf.device, dtype=xf.dtype)
        fx = torch.fft.fftfreq(w, device=xf.device, dtype=xf.dtype)
        yy, xx = torch.meshgrid(fy, fx, indexing="ij")
        radius = torch.sqrt(xx.square() + yy.square())
        mask = torch.sigmoid(
            (radius - self.cutoff) * self.sharpness
        )[None, None]
        spec = torch.fft.fft2(xf, dim=(-2, -1), norm="ortho")
        high = torch.fft.ifft2(
            spec * mask, dim=(-2, -1), norm="ortho"
        ).real
        return high.abs().to(dtype)


class SharedLostDetailEncoder(nn.Module):
    """
    Residual is first transformed by PixelUnshuffle, so it is already at
    semantic-RGB spatial resolution. No further stride is used here.
    """

    def __init__(self, ratio=3, detail_channels=64, stem_channels=32):
        super().__init__()
        in_channels = 3 * int(ratio) * int(ratio)
        c = int(stem_channels)
        d = int(detail_channels)
        self.net = nn.Sequential(
            ConvBNAct(in_channels, c, 3, 1),
            ConvBNAct(c, c, 3, 1, groups=c),
            ConvBNAct(c, d, 1, 1),
        )

    def forward(self, x):
        return self.net(x)


class BoundarySemanticRegionGenerator(nn.Module):
    """
    Boundary + semantics -> soft UAV support region.
    This is NOT a segmentation mask.
    """

    def __init__(
        self,
        rgb_channels,
        tir_channels,
        guide_channels=32,
        freq_cutoff=0.15,
        freq_sharpness=24.0,
    ):
        super().__init__()
        g = int(guide_channels)
        self.rgb_proj = ConvBNAct(rgb_channels, g, 1, 1)
        self.tir_proj = ConvBNAct(tir_channels, g, 1, 1)
        self.freq = FrequencyBoundary(freq_cutoff, freq_sharpness)

        self.region_head = nn.Sequential(
            ConvBNAct(g * 5, g, 3, 1),
            ConvBNAct(g, g, 3, 1),
            nn.Conv2d(g, 1, 1, bias=True),
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

    def forward(self, rgb_feat, tir_feat):
        tir_feat = self._resize_like(tir_feat, rgb_feat)
        r = self.rgb_proj(rgb_feat)
        t = self.tir_proj(tir_feat)
        br = self.freq(r)
        bt = self.freq(t)

        support = torch.sigmoid(
            self.region_head(
                torch.cat([r, t, (r - t).abs(), br, bt], dim=1)
            )
        )
        return support, br, bt


class BHLRScaleAdapter(nn.Module):
    def __init__(
        self,
        rgb_channels,
        tir_channels,
        detail_channels,
        guide_channels=32,
        freq_cutoff=0.15,
        freq_sharpness=24.0,
    ):
        super().__init__()
        self.rgb_channels = int(rgb_channels)
        self.tir_channels = int(tir_channels)

        self.region = BoundarySemanticRegionGenerator(
            rgb_channels=rgb_channels,
            tir_channels=tir_channels,
            guide_channels=guide_channels,
            freq_cutoff=freq_cutoff,
            freq_sharpness=freq_sharpness,
        )

        self.detail_projection = nn.Sequential(
            nn.Conv2d(detail_channels * 2, rgb_channels, 1, bias=False),
            nn.BatchNorm2d(rgb_channels),
            nn.SiLU(inplace=True),
        )

        g = max(16, int(guide_channels))
        self.rgb_gate_proj = ConvBNAct(rgb_channels, g, 1, 1)
        self.tir_gate_proj = ConvBNAct(tir_channels, g, 1, 1)
        self.detail_gate_proj = ConvBNAct(rgb_channels, g, 1, 1)

        self.gate_head = nn.Sequential(
            ConvBNAct(g * 3, g, 3, 1),
            nn.Conv2d(g, 1, 1, bias=True),
        )

        # Zero -> exact baseline at initialization.
        self.gamma_raw = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

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

    def _compress_detail(self, detail_raw, rgb_feat):
        target = rgb_feat.shape[-2:]
        avg = F.adaptive_avg_pool2d(detail_raw, target)
        mx = F.adaptive_max_pool2d(detail_raw, target)
        return self.detail_projection(torch.cat([avg, mx], dim=1))

    def forward(self, detail_raw, rgb_feat, tir_feat, return_debug=False):
        if rgb_feat.shape[1] != self.rgb_channels:
            raise ValueError(
                f"RGB channels mismatch: {rgb_feat.shape[1]} != {self.rgb_channels}"
            )
        if tir_feat.shape[1] != self.tir_channels:
            raise ValueError(
                f"TIR channels mismatch: {tir_feat.shape[1]} != {self.tir_channels}"
            )

        tir_feat = self._resize_like(tir_feat, rgb_feat)

        # 1) Edge/boundary + semantics define the whole UAV support region.
        support, rgb_boundary, tir_boundary = self.region(
            rgb_feat, tir_feat
        )

        # 2) Only the information lost by downsampling is used.
        detail_scale = self._compress_detail(detail_raw, rgb_feat)

        # 3) Lost information is not always useful -> learned gate.
        gate = torch.sigmoid(
            self.gate_head(
                torch.cat(
                    [
                        self.rgb_gate_proj(rgb_feat),
                        self.tir_gate_proj(tir_feat),
                        self.detail_gate_proj(detail_scale),
                    ],
                    dim=1,
                )
            )
        )

        # The support map gates the WHOLE inferred UAV region, not only edge pixels.
        usable_detail = support * gate * detail_scale

        gamma = torch.tanh(self.gamma_raw)
        enhanced_rgb = rgb_feat + gamma * usable_detail

        if not return_debug:
            return enhanced_rgb

        return enhanced_rgb, {
            "support_map": support,
            "rgb_boundary": rgb_boundary,
            "tir_boundary": tir_boundary,
            "detail_scale": detail_scale,
            "detail_gate": gate,
            "usable_detail": usable_detail,
            "gamma": gamma,
            "enhanced_rgb": enhanced_rgb,
        }

    @torch.no_grad()
    def scalar_state(self):
        return {"gamma": float(torch.tanh(self.gamma_raw).cpu().item())}


class MultiScaleBHLRv1(nn.Module):
    VALID_SCALES = (3, 4, 5)

    def __init__(
        self,
        rgb_channels: Mapping[int, int],
        tir_channels: Mapping[int, int],
        scales: Iterable[int] = (3,),
        high_to_semantic_ratio=3,
        detail_channels=64,
        stem_channels=32,
        guide_channels=32,
        freq_cutoff=0.15,
        freq_sharpness=24.0,
        exact_identity_when_equal=True,
    ):
        super().__init__()

        scales = tuple(sorted({int(s) for s in scales}))
        if not scales:
            raise ValueError("bhlr_scales cannot be empty")

        invalid = [s for s in scales if s not in self.VALID_SCALES]
        if invalid:
            raise ValueError(
                f"bhlr_scales only support P3/P4/P5, invalid={invalid}"
            )

        self.scales = scales
        self.ratio = int(high_to_semantic_ratio)
        self.exact_identity_when_equal = bool(exact_identity_when_equal)

        self.detail_stem = SharedLostDetailEncoder(
            ratio=self.ratio,
            detail_channels=detail_channels,
            stem_channels=stem_channels,
        )

        self.adapters = nn.ModuleDict({
            str(scale): BHLRScaleAdapter(
                rgb_channels=int(rgb_channels[scale]),
                tir_channels=int(tir_channels[scale]),
                detail_channels=int(detail_channels),
                guide_channels=int(guide_channels),
                freq_cutoff=float(freq_cutoff),
                freq_sharpness=float(freq_sharpness),
            )
            for scale in self.scales
        })

    @staticmethod
    def _resize(x, size):
        if x.shape[-2:] == tuple(size):
            return x
        return F.interpolate(
            x, size=size, mode="bilinear", align_corners=False
        )

    def build_lost_detail_bank(self, rgb_high, rgb_semantic):
        expected_h = rgb_semantic.shape[-2] * self.ratio
        expected_w = rgb_semantic.shape[-1] * self.ratio

        if rgb_high.shape[-2:] != (expected_h, expected_w):
            raise ValueError(
                "BHLR PixelUnshuffle requires exact integer scale ratio.\n"
                f"ratio={self.ratio}\n"
                f"rgb_high={tuple(rgb_high.shape[-2:])}\n"
                f"rgb_semantic={tuple(rgb_semantic.shape[-2:])}\n"
                f"expected_high={(expected_h, expected_w)}"
            )

        reconstructed_high = self._resize(
            rgb_semantic, rgb_high.shape[-2:]
        )
        residual_high = rgb_high - reconstructed_high

        # 1920 -> 640 with ratio=3:
        # [B,3,1920,1920] -> [B,27,640,640]
        residual_s2d = F.pixel_unshuffle(
            residual_high,
            downscale_factor=self.ratio,
        )
        detail_raw = self.detail_stem(residual_s2d)

        return detail_raw, reconstructed_high, residual_high

    def forward(
        self,
        rgb_high,
        rgb_semantic,
        rgb_features: Mapping[int, torch.Tensor],
        tir_features: Mapping[int, torch.Tensor],
        return_debug=False,
    ):
        equal = rgb_high.shape[-2:] == rgb_semantic.shape[-2:]

        if equal and self.exact_identity_when_equal:
            outputs = {s: rgb_features[s] for s in self.scales}
            if not return_debug:
                return outputs
            return outputs, {
                "equal_resolution": True,
                "detail_raw": None,
                "scales": {},
            }

        detail_raw, reconstructed_high, residual_high = (
            self.build_lost_detail_bank(rgb_high, rgb_semantic)
        )

        outputs = {}
        debug_scales = {}

        for scale in self.scales:
            adapter = self.adapters[str(scale)]

            if return_debug:
                enhanced, dbg = adapter(
                    detail_raw,
                    rgb_features[scale],
                    tir_features[scale],
                    return_debug=True,
                )
                debug_scales[scale] = dbg
            else:
                enhanced = adapter(
                    detail_raw,
                    rgb_features[scale],
                    tir_features[scale],
                    return_debug=False,
                )

            outputs[scale] = enhanced

        if not return_debug:
            return outputs

        return outputs, {
            "equal_resolution": False,
            "reconstructed_high": reconstructed_high,
            "resolution_residual": residual_high,
            "detail_raw": detail_raw,
            "scales": debug_scales,
        }

    @torch.no_grad()
    def scalar_state(self) -> Dict[str, float]:
        """
        Return BHLR scalar state.

        Notes
        -----
        The current detector inherits the old RGBTRSDTDetectionModel
        print/status interface, which expects BOTH:
            alpha_p3 / gamma_p3
            alpha_p4 / gamma_p4
            alpha_p5 / gamma_p5

        BHLR does not use the old RSD-T alpha parameter, so alpha is exposed
        as a fixed compatibility value of 1.0. It is NOT a learnable BHLR
        parameter and does not participate in forward computation.
        """
        result = {}
        gammas = []

        for scale in self.scales:
            gamma = self.adapters[str(scale)].scalar_state()["gamma"]

            # Compatibility-only key required by the inherited
            # RGBTRSDTDetectionModel.print_info().
            result[f"alpha_p{scale}"] = 1.0

            # Actual learnable BHLR residual injection strength.
            result[f"gamma_p{scale}"] = gamma

            gammas.append(gamma)

        result["alpha"] = 1.0
        result["gamma"] = (
            sum(gammas) / max(len(gammas), 1)
        )

        return result
