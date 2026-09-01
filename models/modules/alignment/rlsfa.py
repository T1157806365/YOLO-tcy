"""
RLSFA: Reliability-guided Local Spatial-Frequency Alignment
===========================================================

中文：可靠性引导局部空频对齐模块

Core principles
---------------
1. Coarse alignment uses explicit correlation search, not direct Conv offset regression.
2. Spatial structure provides precise localization.
3. Local overlapping Window FFT provides local frequency structure; no whole-map FFT.
4. RGB/TIR estimate independent frequency reliability maps.
5. Fine alignment uses semantic + reliable structural descriptors.
6. Coarse/fine offsets are GLOBAL translations [B,2,1,1].
   The module changes target position but does not non-rigidly deform target shape.
7. RGB is the reference/output coordinate system. TIR uses sampling offset:
       output(p) = source(p + delta)
   therefore GT sampling translation is C_TIR - C_RGB.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseAlignment


class ConvBNAct(nn.Module):
    def __init__(self, c1: int, c2: int, k: int = 3, groups: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(c1, c2, k, 1, k // 2, groups=groups, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class LazyProj(nn.Module):
    def __init__(self, out_channels: int):
        super().__init__()
        self.conv = nn.LazyConv2d(out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class MultiKernelSpatialStructure(nn.Module):
    """Learnable local spatial structure: multi-kernel depthwise conv + fusion."""

    def __init__(self, channels: int, kernel_sizes: Sequence[int] = (3, 5)):
        super().__init__()
        self.channels = int(channels)
        self.kernel_sizes = tuple(int(k) for k in kernel_sizes)
        if not self.kernel_sizes:
            raise ValueError("spatial_kernel_sizes cannot be empty")
        for k in self.kernel_sizes:
            if k <= 0 or k % 2 == 0:
                raise ValueError(f"spatial kernels must be positive odd integers, got {self.kernel_sizes}")

        self.branches = nn.ModuleList([
            nn.Sequential(
                ConvBNAct(self.channels, self.channels, k=k, groups=self.channels),
                nn.Conv2d(self.channels, self.channels, 1, bias=False),
                nn.BatchNorm2d(self.channels),
                nn.SiLU(inplace=True),
            )
            for k in self.kernel_sizes
        ])
        self.fuse = ConvBNAct(self.channels * len(self.kernel_sizes), self.channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fuse(torch.cat([m(x) for m in self.branches], dim=1))


class LocalWindowFrequency(nn.Module):
    """
    Overlapping local Window FFT.

    Each scale:
        unfold -> raised Hann -> FFT2 -> radial high-pass -> IFFT2 -> abs
        -> overlap-add -> normalization.

    Multi-scale responses are fused by 1x1 Conv.
    """

    def __init__(
        self,
        channels: int,
        window_sizes: Sequence[int] = (4, 8),
        overlap: float = 0.5,
        cutoff: float = 0.15,
        sharpness: float = 24.0,
        use_hann_window: bool = True,
    ):
        super().__init__()
        self.channels = int(channels)
        self.window_sizes = tuple(int(w) for w in window_sizes)
        if not self.window_sizes:
            raise ValueError("window_sizes cannot be empty")
        if not 0.0 <= float(overlap) < 1.0:
            raise ValueError("window_overlap must satisfy 0 <= overlap < 1")
        self.overlap = float(overlap)
        self.cutoff = float(cutoff)
        self.sharpness = float(sharpness)
        self.use_hann_window = bool(use_hann_window)
        self.fuse = ConvBNAct(self.channels * len(self.window_sizes), self.channels, 1)

    @staticmethod
    def _stride(k: int, overlap: float) -> int:
        return max(1, int(round(k * (1.0 - overlap))))

    @staticmethod
    def _padding(h: int, w: int, k: int, stride: int) -> Tuple[int, int]:
        th, tw = max(h, k), max(w, k)
        rh, rw = (th - k) % stride, (tw - k) % stride
        if rh:
            th += stride - rh
        if rw:
            tw += stride - rw
        return th - h, tw - w

    @staticmethod
    def _raised_hann(k: int, device, dtype) -> torch.Tensor:
        one = torch.hann_window(k, periodic=False, device=device, dtype=dtype)
        win = one[:, None] * one[None, :]
        # Avoid exact zero overlap denominator at the global image border.
        return 0.05 + 0.95 * win

    def _single_scale(self, x: torch.Tensor, k: int) -> torch.Tensor:
        out_dtype = x.dtype
        xf = x.float()  # FFT in float32 for AMP stability.
        b, c, h, w = xf.shape
        stride = self._stride(k, self.overlap)
        pad_h, pad_w = self._padding(h, w, k, stride)
        if pad_h or pad_w:
            xf = F.pad(xf, (0, pad_w, 0, pad_h), mode="replicate")
        hp, wp = xf.shape[-2:]

        patches = F.unfold(xf, kernel_size=k, stride=stride)  # [B,C*k*k,L]
        nwin = int(patches.shape[-1])
        patches = (
            patches.view(b, c, k, k, nwin)
            .permute(0, 4, 1, 2, 3)
            .contiguous()
            .view(b * nwin, c, k, k)
        )

        if self.use_hann_window:
            win = self._raised_hann(k, patches.device, patches.dtype)[None, None]
        else:
            win = torch.ones(1, 1, k, k, device=patches.device, dtype=patches.dtype)

        pw = patches * win
        fy = torch.fft.fftfreq(k, device=pw.device, dtype=pw.dtype)
        fx = torch.fft.fftfreq(k, device=pw.device, dtype=pw.dtype)
        yy, xx = torch.meshgrid(fy, fx, indexing="ij")
        radius = torch.sqrt(xx.square() + yy.square())
        hp_mask = torch.sigmoid((radius - self.cutoff) * self.sharpness)[None, None]

        spec = torch.fft.fft2(pw, dim=(-2, -1), norm="ortho")
        high = torch.fft.ifft2(spec * hp_mask, dim=(-2, -1), norm="ortho").real.abs()
        high = high * win

        cols = (
            high.view(b, nwin, c, k, k)
            .permute(0, 2, 3, 4, 1)
            .contiguous()
            .view(b, c * k * k, nwin)
        )
        rec = F.fold(cols, output_size=(hp, wp), kernel_size=k, stride=stride)

        weights = win.view(1, k * k, 1).expand(1, k * k, nwin)
        den = F.fold(weights, output_size=(hp, wp), kernel_size=k, stride=stride)
        rec = rec / den.clamp(min=1e-6)
        return rec[:, :, :h, :w].to(out_dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = [self._single_scale(x, k) for k in self.window_sizes]
        return self.fuse(torch.cat(feats, dim=1))


def _correlation_translation(
    reference: torch.Tensor,
    source: torch.Tensor,
    weight_map: torch.Tensor,
    radius: int,
    temperature: float,
):
    """
    Explicit target-weighted displacement search.

    For candidate (dx,dy), compare reference(p) with source(p + [dx,dy]).
    This directly follows grid_sample(base + delta) sampling semantics.

    Returns:
        expected_offset [B,2,1,1]
        normalized confidence [B,1,1,1]
        probability [B,K]
    """
    r = int(radius)
    if r < 0:
        raise ValueError(f"radius must be >=0, got {r}")
    temp = max(float(temperature), 1e-4)

    ref = F.normalize(reference.float(), p=2, dim=1, eps=1e-6)
    src = F.normalize(source.float(), p=2, dim=1, eps=1e-6)
    weight = weight_map.float().clamp(min=0.0)
    den = weight.sum(dim=(-2, -1)).clamp(min=1e-6)  # [B,1]

    _, _, h, w = src.shape
    padded = F.pad(src, (r, r, r, r), mode="replicate")

    all_scores = []
    disp = []
    # Vectorize all dx candidates for each dy: only (2r+1) Python loops.
    for dy in range(-r, r + 1):
        shifted_dx = []
        for dx in range(-r, r + 1):
            y0, x0 = r + dy, r + dx
            shifted_dx.append(padded[:, :, y0:y0 + h, x0:x0 + w])
            disp.append((float(dx), float(dy)))
        shifted = torch.stack(shifted_dx, dim=2)  # [B,C,Kx,H,W]
        corr = (ref.unsqueeze(2) * shifted).sum(dim=1)  # [B,Kx,H,W]
        score = (corr * weight).sum(dim=(-2, -1)) / den  # [B,Kx]
        all_scores.append(score)

    logits = torch.cat(all_scores, dim=1)  # [B,K]
    prob = torch.softmax(logits / temp, dim=1)
    disp_t = torch.tensor(disp, device=prob.device, dtype=prob.dtype)  # [K,2]
    expected = prob @ disp_t

    k = int(prob.shape[1])
    maxp = prob.max(dim=1, keepdim=True).values
    uniform = 1.0 / max(k, 1)
    conf = ((maxp - uniform) / max(1.0 - uniform, 1e-6)).clamp(0.0, 1.0)
    return expected.view(-1, 2, 1, 1), conf.view(-1, 1, 1, 1), prob


class RLSFAScaleUnit(nn.Module):
    def __init__(
        self,
        hidden_channels: int = 32,
        coarse_radius: int = 8,
        coarse_temperature: float = 0.1,
        spatial_kernel_sizes: Sequence[int] = (3, 5),
        window_sizes: Sequence[int] = (4, 8),
        window_overlap: float = 0.5,
        use_hann_window: bool = True,
        freq_cutoff: float = 0.15,
        freq_sharpness: float = 24.0,
        reliability_channels: int = 32,
        fine_radius: int = 2,
        max_fine_offset: float = 1.0,
        fine_temperature: float = 0.1,
        use_targetness: bool = True,
    ):
        super().__init__()
        c = int(hidden_channels)
        self.hidden_channels = c
        self.coarse_radius = int(coarse_radius)
        self.coarse_temperature = float(coarse_temperature)
        self.fine_radius = int(fine_radius)
        self.max_fine_offset = float(max_fine_offset)
        self.fine_temperature = float(fine_temperature)
        self.use_targetness = bool(use_targetness)

        self.rgb_proj = LazyProj(c)
        self.tir_proj = LazyProj(c)

        self.targetness_head = nn.Sequential(
            ConvBNAct(c * 3, c, 3),
            nn.Conv2d(c, 1, 1, bias=True),
        )
        nn.init.zeros_(self.targetness_head[-1].weight)
        nn.init.zeros_(self.targetness_head[-1].bias)

        self.rgb_spatial = MultiKernelSpatialStructure(c, spatial_kernel_sizes)
        self.tir_spatial = MultiKernelSpatialStructure(c, spatial_kernel_sizes)
        self.rgb_frequency = LocalWindowFrequency(
            c, window_sizes, window_overlap, freq_cutoff, freq_sharpness, use_hann_window
        )
        self.tir_frequency = LocalWindowFrequency(
            c, window_sizes, window_overlap, freq_cutoff, freq_sharpness, use_hann_window
        )

        rc = int(reliability_channels)
        self.rgb_reliability_head = nn.Sequential(
            ConvBNAct(c * 4, rc, 3),
            nn.Conv2d(rc, 1, 1, bias=True),
        )
        self.tir_reliability_head = nn.Sequential(
            ConvBNAct(c * 4, rc, 3),
            nn.Conv2d(rc, 1, 1, bias=True),
        )
        nn.init.zeros_(self.rgb_reliability_head[-1].weight)
        nn.init.zeros_(self.rgb_reliability_head[-1].bias)
        nn.init.zeros_(self.tir_reliability_head[-1].weight)
        nn.init.zeros_(self.tir_reliability_head[-1].bias)

        self.rgb_descriptor = ConvBNAct(c * 2, c, 3)
        self.tir_descriptor = ConvBNAct(c * 2, c, 3)

    @staticmethod
    def _resize_like(source: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        if source.shape[-2:] == reference.shape[-2:]:
            return source
        return F.interpolate(source, size=reference.shape[-2:], mode="bilinear", align_corners=False)

    @staticmethod
    def _expand_translation(offset: torch.Tensor, h: int, w: int) -> torch.Tensor:
        if offset.shape[-2:] == (h, w):
            return offset
        return offset.expand(-1, -1, h, w)

    @staticmethod
    def _warp(source: torch.Tensor, offset_px: torch.Tensor) -> torch.Tensor:
        b, _, h, w = source.shape
        offset_px = RLSFAScaleUnit._expand_translation(offset_px, h, w)
        device, dtype = source.device, source.dtype
        ys = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype)
        xs = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        base = torch.stack((xx, yy), dim=-1)[None].expand(b, -1, -1, -1)
        dx = 2.0 * offset_px[:, 0] / max(w - 1, 1)
        dy = 2.0 * offset_px[:, 1] / max(h - 1, 1)
        delta = torch.stack((dx, dy), dim=-1)
        return F.grid_sample(
            source, base + delta, mode="bilinear", padding_mode="border", align_corners=True
        )

    def forward(self, rgb_feat, tir_feat, return_debug: bool = False):
        tir_feat = self._resize_like(tir_feat, rgb_feat)
        r = self.rgb_proj(rgb_feat)
        t = self.tir_proj(tir_feat)

        # 1) RGB-grid targetness for coarse/fine matching.
        target_in = torch.cat([r, t, (r - t).abs()], dim=1)
        target_logits = self.targetness_head(target_in)
        targetness = torch.sigmoid(target_logits) if self.use_targetness else torch.ones_like(target_logits)

        # 2) Coarse explicit correlation -> global sampling translation.
        coarse_offset, coarse_conf, coarse_prob = _correlation_translation(
            r, t, targetness, self.coarse_radius, self.coarse_temperature
        )
        tir_coarse = self._warp(tir_feat, coarse_offset)
        t_coarse = self.tir_proj(tir_coarse)

        # 3) Spatial + local-frequency structures.
        sr = self.rgb_spatial(r)
        st = self.tir_spatial(t_coarse)
        fr = self.rgb_frequency(r)
        ft = self.tir_frequency(t_coarse)

        # 4) Independent modality frequency reliability.
        qr_logits = self.rgb_reliability_head(torch.cat([r, sr, fr, (sr - fr).abs()], dim=1))
        qt_logits = self.tir_reliability_head(torch.cat([t_coarse, st, ft, (st - ft).abs()], dim=1))
        qr = torch.sigmoid(qr_logits)
        qt = torch.sigmoid(qt_logits)
        br = qr * fr + (1.0 - qr) * sr
        bt = qt * ft + (1.0 - qt) * st

        # 5) Semantic + reliable-structure fine descriptors.
        dr = self.rgb_descriptor(torch.cat([r, br], dim=1))
        dt = self.tir_descriptor(torch.cat([t_coarse, bt], dim=1))
        fine_expected, fine_conf, fine_prob = _correlation_translation(
            dr, dt, targetness, self.fine_radius, self.fine_temperature
        )
        fine_scale = max(float(self.fine_radius), 1.0)
        fine_offset = (
            torch.tanh(fine_expected / fine_scale)
            * self.max_fine_offset
            * fine_conf
        )

        # Global translations can be composed by direct addition.
        total_offset = coarse_offset + fine_offset
        aligned_tir = self._warp(tir_feat, total_offset)

        if not return_debug:
            return aligned_tir

        return aligned_tir, {
            "coarse_targetness_logits": target_logits,
            "coarse_targetness": targetness,
            "coarse_correlation_confidence": coarse_conf,
            "coarse_probability": coarse_prob,
            "coarse_offset": coarse_offset,
            "tir_coarse": tir_coarse,
            "rgb_spatial_structure": sr,
            "tir_spatial_structure": st,
            "rgb_local_frequency": fr,
            "tir_local_frequency": ft,
            "rgb_reliability_logits": qr_logits,
            "tir_reliability_logits": qt_logits,
            "rgb_reliability": qr,
            "tir_reliability": qt,
            "rgb_reliable_structure": br,
            "tir_reliable_structure": bt,
            "rgb_fine_descriptor": dr,
            "tir_fine_descriptor": dt,
            "fine_correlation_confidence": fine_conf,
            "fine_probability": fine_prob,
            "fine_offset": fine_offset,
            "total_offset": total_offset,
            "aligned_tir": aligned_tir,
        }


class RLSFAAlignment(BaseAlignment):
    VALID_SCALES = (3, 4, 5)

    def __init__(self, cfg: Dict[str, Any] | None = None):
        super().__init__(cfg=cfg)
        invalid = [s for s in self.scales if s not in self.VALID_SCALES]
        if invalid:
            raise ValueError(f"RLSFA supports only P3/P4/P5, invalid={invalid}")

        hidden = int(self.cfg.get("hidden_channels", 32))
        self.units = nn.ModuleDict({
            str(scale): RLSFAScaleUnit(
                hidden_channels=hidden,
                coarse_radius=int(self.cfg.get("coarse_radius", 8)),
                coarse_temperature=float(self.cfg.get("coarse_temperature", 0.1)),
                spatial_kernel_sizes=self.cfg.get("spatial_kernel_sizes", [3, 5]),
                window_sizes=self.cfg.get("window_sizes", [4, 8]),
                window_overlap=float(self.cfg.get("window_overlap", 0.5)),
                use_hann_window=bool(self.cfg.get("use_hann_window", True)),
                freq_cutoff=float(self.cfg.get("freq_cutoff", 0.15)),
                freq_sharpness=float(self.cfg.get("freq_sharpness", 24.0)),
                reliability_channels=int(self.cfg.get("reliability_channels", hidden)),
                fine_radius=int(self.cfg.get("fine_radius", 2)),
                max_fine_offset=float(self.cfg.get("max_fine_offset", 1.0)),
                fine_temperature=float(self.cfg.get("fine_temperature", 0.1)),
                use_targetness=bool(self.cfg.get("use_targetness", True)),
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
        info = {"enabled": True, "type": "rlsfa", "scales": list(self.scales)}
        debug_all = {}

        for scale in self.scales:
            if scale not in scale_to_index:
                raise KeyError(f"Missing scale mapping for P{scale}")
            idx = int(scale_to_index[scale])
            unit = self.units[str(scale)]
            if return_debug:
                aligned_feat, dbg = unit(rgb_features[idx], tir_features[idx], return_debug=True)
                debug_all[scale] = dbg
            else:
                aligned_feat = unit(rgb_features[idx], tir_features[idx], return_debug=False)
            aligned[idx] = aligned_feat

        if return_debug:
            info["scale_debug"] = debug_all
        return aligned, info
