"""
RGB-T detector:
  Module-1: FBAM
  Module-2: BHLR
  Fusion  : original concat
  Module-3: OFF

The existing RGBTRSDTDetectionModel routing is reused so P3/P4/P5 selection
continues to work exactly like the old RSD-T branch.
"""

from __future__ import annotations
from typing import Any, Dict, Iterable, Optional, Sequence, Union

import torch

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
        verbose: bool = True,
    ):
        alignment_cfg = dict(alignment_cfg or {})

        # New readable key, mapped to the old parent routing key.
        if "use_for_bhlr" in alignment_cfg:
            alignment_cfg["use_for_rsdt"] = bool(
                alignment_cfg["use_for_bhlr"]
            )

        bhlr_scales = tuple(sorted({int(s) for s in bhlr_scales}))

        # Build original dual-backbone, original concat fusion and alignment
        # routing. An old RSD-T object is temporarily built by the parent and
        # replaced immediately below.
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

        # -----------------------------------------------------
        # IMPORTANT:
        # FBAM currently contains nn.LazyConv2d projections.
        # The original trainer counts parameters / builds the
        # optimizer before the first real forward, so all lazy
        # parameters must be materialized here.
        # -----------------------------------------------------
        self._initialize_fbam_lazy_modules()

        feature_channels = self._infer_feature_channels()
        rgb_channels = {}
        tir_channels = {}

        for scale in self.rsdt_scales:
            idx = self.rsdt_scale_to_index[scale]
            rgb_channels[scale] = int(feature_channels[idx]["rgb"])
            tir_channels[scale] = int(feature_channels[idx]["tir"])

        del self.rsdt

        self.bhlr = MultiScaleBHLRv1(
            rgb_channels=rgb_channels,
            tir_channels=tir_channels,
            scales=self.rsdt_scales,
            high_to_semantic_ratio=int(high_to_semantic_ratio),
            detail_channels=int(detail_channels),
            stem_channels=int(stem_channels),
            guide_channels=int(guide_channels),
            freq_cutoff=float(bhlr_freq_cutoff),
            freq_sharpness=float(bhlr_freq_sharpness),
            exact_identity_when_equal=True,
        )

        # Parent forward_features() still calls self.rsdt(...), so use an alias
        # instead of rewriting the full forward pipeline.
        self.rsdt = self.bhlr
        self.bhlr_scales = self.rsdt_scales

        if verbose:
            print("\n============================================================")
            print("FBAM + BHLR RGB-T Detector")
            print("============================================================")
            print(f"Model              : {self.model_name}")
            print("Fusion             : concat (Module-3 OFF)")
            print(f"FBAM enabled       : {self.alignment_enabled}")
            print(f"FBAM type          : {self.alignment_cfg.get('type', 'identity')}")
            print(f"FBAM scales        : {list(getattr(self.alignment, 'scales', []))}")
            print(f"BHLR scales        : {list(self.bhlr_scales)}")
            print(f"RGB semantic       : {self.semantic_imgsz}")
            print(f"High/semantic ratio: {high_to_semantic_ratio}")
            print("Reference system   : RGB; TIR -> RGB")
            print("============================================================\n")

    @torch.no_grad()
    def _initialize_fbam_lazy_modules(self):
        """
        Materialize FBAM LazyConv2d parameters before any external
        parameter counting or optimizer construction.

        This keeps train_rgbt.py / train_rsdt.py unchanged.
        """
        if not getattr(self, "alignment_enabled", False):
            return

        align_type = str(
            self.alignment_cfg.get("type", "identity")
        ).strip().lower()

        if align_type != "fbam":
            return

        # Preserve training/eval states.
        rgb_training = self.model.training
        tir_training = self.tir_backbone.training
        align_training = self.alignment.training

        self.model.eval()
        self.tir_backbone.eval()
        self.alignment.eval()

        # At construction time the model is normally still on CPU.
        # Use the same device as the RGB backbone parameters.
        device = next(self.model.parameters()).device

        dummy = torch.zeros(
            1,
            3,
            256,
            256,
            device=device,
            dtype=torch.float32,
        )

        rgb_outputs = list(
            self.forward_rgb_backbone(dummy)
        )

        tir_outputs = list(
            self.forward_tir_backbone(dummy)
        )

        # One dry forward is enough to initialize every selected
        # FBAM scale's LazyConv2d projection.
        _ = self.alignment(
            rgb_features=rgb_outputs,
            tir_features=tir_outputs,
            scale_to_index=self.rsdt_scale_to_index,
            return_debug=False,
        )

        # Restore original states.
        self.model.train(rgb_training)
        self.tir_backbone.train(tir_training)
        self.alignment.train(align_training)

        # Safety check: fail early if any lazy parameter is still
        # uninitialized.
        from torch.nn.parameter import UninitializedParameter

        uninitialized = [
            name
            for name, param in self.named_parameters()
            if isinstance(param, UninitializedParameter)
        ]

        if uninitialized:
            raise RuntimeError(
                "FBAM lazy parameters are still uninitialized: "
                + ", ".join(uninitialized)
            )

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
            return_rsdt_debug=return_rsdt_debug,
            return_alignment_debug=return_alignment_debug,
        )
        out["bhlr_scales"] = out.get("rsdt_scales", self.bhlr_scales)
        if "rsdt_debug" in out:
            out["bhlr_debug"] = out["rsdt_debug"]
        return out
