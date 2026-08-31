"""
FBAM: Frequency-aware Boundary Alignment Module
================================================
Configurable on P3/P4/P5 through alignment.scales.
RGB is the reference coordinate system; TIR is warped toward RGB.
"""

from __future__ import annotations
from typing import Any, Dict, Mapping, Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseAlignment


class ConvBNAct(nn.Module):
    def __init__(self, c1: int, c2: int, k: int = 3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(c1, c2, k, 1, k // 2, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class LazyProj(nn.Module):
    def __init__(self, out_channels: int):
        super().__init__()
        self.conv = nn.LazyConv2d(out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class FrequencyBoundary(nn.Module):
    """Differentiable frequency-domain high-pass reconstruction."""

    def __init__(self, cutoff: float = 0.15, sharpness: float = 24.0):
        super().__init__()
        self.cutoff = float(cutoff)
        self.sharpness = float(sharpness)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        xf = x.float()
        _, _, h, w = xf.shape

        fy = torch.fft.fftfreq(h, device=xf.device, dtype=xf.dtype)
        fx = torch.fft.fftfreq(w, device=xf.device, dtype=xf.dtype)
        yy, xx = torch.meshgrid(fy, fx, indexing="ij")
        radius = torch.sqrt(xx.square() + yy.square())
        mask = torch.sigmoid(
            (radius - self.cutoff) * self.sharpness
        )[None, None, :, :]

        spectrum = torch.fft.fft2(xf, dim=(-2, -1), norm="ortho")
        high = torch.fft.ifft2(
            spectrum * mask, dim=(-2, -1), norm="ortho"
        ).real
        return high.abs().to(dtype=dtype)


def _zero_init_last_conv(module: nn.Sequential):
    for m in reversed(module):
        if isinstance(m, nn.Conv2d):
            nn.init.zeros_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
            return


class FBAMScaleUnit(nn.Module):
    def __init__(
        self,
        hidden_channels: int = 32,
        max_coarse_offset: float = 4.0,
        max_fine_offset: float = 2.0,
        freq_cutoff: float = 0.15,
        freq_sharpness: float = 24.0,
        use_confidence: bool = True,
    ):
        super().__init__()
        c = int(hidden_channels)
        self.max_coarse_offset = float(max_coarse_offset)
        self.max_fine_offset = float(max_fine_offset)
        self.use_confidence = bool(use_confidence)

        self.rgb_proj = LazyProj(c)
        self.tir_proj = LazyProj(c)

        self.coarse_head = nn.Sequential(
            ConvBNAct(c * 3, c, 3),
            ConvBNAct(c, c, 3),
            nn.Conv2d(c, 2, 3, padding=1, bias=True),
        )
        _zero_init_last_conv(self.coarse_head)

        self.freq_boundary = FrequencyBoundary(
            cutoff=freq_cutoff,
            sharpness=freq_sharpness,
        )

        self.fine_head = nn.Sequential(
            ConvBNAct(c * 3, c, 3),
            ConvBNAct(c, c, 3),
            nn.Conv2d(c, 2, 3, padding=1, bias=True),
        )
        _zero_init_last_conv(self.fine_head)

        if self.use_confidence:
            self.conf_head = nn.Sequential(
                ConvBNAct(c * 3, c, 3),
                nn.Conv2d(c, 1, 1, bias=True),
            )
            nn.init.zeros_(self.conf_head[-1].weight)
            nn.init.zeros_(self.conf_head[-1].bias)
        else:
            self.conf_head = None

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

    @staticmethod
    def _warp(source: torch.Tensor, offset_px: torch.Tensor) -> torch.Tensor:
        b, _, h, w = source.shape
        device, dtype = source.device, source.dtype

        ys = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype)
        xs = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        base = torch.stack((xx, yy), dim=-1)[None].expand(b, -1, -1, -1)

        dx = 2.0 * offset_px[:, 0] / max(w - 1, 1)
        dy = 2.0 * offset_px[:, 1] / max(h - 1, 1)
        delta = torch.stack((dx, dy), dim=-1)

        return F.grid_sample(
            source,
            base + delta,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )

    def forward(self, rgb_feat, tir_feat, return_debug: bool = False):
        tir_feat = self._resize_like(tir_feat, rgb_feat)

        r = self.rgb_proj(rgb_feat)
        t = self.tir_proj(tir_feat)

        coarse_input = torch.cat([r, t, (r - t).abs()], dim=1)
        coarse_offset = torch.tanh(
            self.coarse_head(coarse_input)
        ) * self.max_coarse_offset

        tir_coarse = self._warp(tir_feat, coarse_offset)
        t_coarse = self.tir_proj(tir_coarse)

        br = self.freq_boundary(r)
        bt = self.freq_boundary(t_coarse)

        fine_input = torch.cat([br, bt, (br - bt).abs()], dim=1)
        fine_offset = torch.tanh(
            self.fine_head(fine_input)
        ) * self.max_fine_offset

        if self.conf_head is not None:
            confidence = torch.sigmoid(self.conf_head(fine_input))
            fine_offset = fine_offset * confidence
        else:
            confidence = torch.ones(
                (rgb_feat.shape[0], 1, rgb_feat.shape[-2], rgb_feat.shape[-1]),
                device=rgb_feat.device,
                dtype=rgb_feat.dtype,
            )

        total_offset = coarse_offset + fine_offset
        aligned_tir = self._warp(tir_feat, total_offset)

        if not return_debug:
            return aligned_tir

        return aligned_tir, {
            "coarse_offset": coarse_offset,
            "fine_offset": fine_offset,
            "total_offset": total_offset,
            "confidence": confidence,
            "rgb_boundary": br,
            "tir_boundary": bt,
        }


class FBAMAlignment(BaseAlignment):
    VALID_SCALES = (3, 4, 5)

    def __init__(self, cfg: Dict[str, Any] | None = None):
        super().__init__(cfg=cfg)
        invalid = [s for s in self.scales if s not in self.VALID_SCALES]
        if invalid:
            raise ValueError(f"FBAM only supports P3/P4/P5, invalid={invalid}")

        hidden = int(self.cfg.get("hidden_channels", 32))
        max_coarse = float(self.cfg.get("max_coarse_offset", 4.0))
        max_fine = float(self.cfg.get("max_fine_offset", 2.0))
        cutoff = float(self.cfg.get("freq_cutoff", 0.15))
        sharpness = float(self.cfg.get("freq_sharpness", 24.0))
        use_conf = bool(self.cfg.get("use_confidence", True))

        self.units = nn.ModuleDict({
            str(scale): FBAMScaleUnit(
                hidden_channels=hidden,
                max_coarse_offset=max_coarse,
                max_fine_offset=max_fine,
                freq_cutoff=cutoff,
                freq_sharpness=sharpness,
                use_confidence=use_conf,
            )
            for scale in self.scales
        })

    def forward(
        self,
        rgb_features: Sequence[torch.Tensor],
        tir_features: Sequence[torch.Tensor],
        scale_to_index: Mapping[int, int],
        return_debug: bool = False,
    ):
        aligned = list(tir_features)
        info = {
            "enabled": True,
            "type": "fbam",
            "scales": list(self.scales),
        }
        debug_all = {}

        for scale in self.scales:
            if scale not in scale_to_index:
                raise KeyError(f"Missing scale mapping for P{scale}")

            idx = int(scale_to_index[scale])
            unit = self.units[str(scale)]

            if return_debug:
                aligned_feat, dbg = unit(
                    rgb_features[idx], tir_features[idx], return_debug=True
                )
                debug_all[scale] = dbg
            else:
                aligned_feat = unit(
                    rgb_features[idx], tir_features[idx], return_debug=False
                )

            aligned[idx] = aligned_feat

        if return_debug:
            info["scale_debug"] = debug_all

        return aligned, info
