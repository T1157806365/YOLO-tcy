"""
RLSFA + BHLR-v2 ANY-LAYER detector wrapper
===========================================

Purpose
-------
Allow BHLR-v2 to be injected after ANY RGB backbone layer by actual layer ID,
while ensuring the enhancement propagates into all later backbone layers.

Why inline injection is required
--------------------------------
Wrong:
    run complete backbone
    -> change outputs[layer_id]

For an early layer this does NOT affect later layers because they were already
computed.

Correct:
    execute backbone layer by layer
    -> after selected layer: BHLR(x)
    -> store enhanced x
    -> continue to next layer

Configuration
-------------
Use actual backbone layer IDs:

    model:
      bhlr_layers: [2]

or:

    model:
      bhlr_layers: [2, 4, 6]

For YOLO26n semantic 640, commonly:
    layer 0 : 320x320
    layer 1 : 160x160
    layer 2 : 160x160   (often treated as P2-level feature)
    layer 4 : 80x80     (P3)
    layer 6 : 40x40     (P4)
    layer 10: 20x20     (P5)

Do not assume these IDs for another model YAML; call print_backbone_table().
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Union

import torch
import torch.nn.functional as F

from models.rgbt_model import get_layer_input
from models.rgbt_rsdt_model import RGBTRSDTDetectionModel
from models.modules.bhlr_v2 import MultiScaleBHLRv2


class RGBTRLSFABHLRAnyLayerDetectionModel(RGBTRSDTDetectionModel):
    """
    Drop-in experimental wrapper for arbitrary-layer BHLR injection.

    ``bhlr_layers`` are ACTUAL backbone layer indices, not P-level labels.
    """

    def __init__(
        self,
        model_name: str = "yolo26n",
        nc: int = 1,
        pretrained: bool = True,
        fusion: str = "concat",
        fusion_indices: Optional[Sequence[int]] = None,
        align_mode: str = "bilinear",
        names: Optional[Union[Dict[int, str], Sequence[str]]] = None,
        semantic_imgsz: int = 640,

        # NEW: actual backbone layer indices.
        # Example: [2, 4, 6]
        bhlr_layers: Optional[Iterable[int]] = None,

        # Legacy compatibility with the existing trainer/config.
        # If bhlr_layers is omitted, bhlr_scales=[3,4,5] is mapped
        # through the model's P3/P4/P5 fusion indices.
        bhlr_scales: Optional[Iterable[int]] = None,

        high_to_semantic_ratio: int = 2,
        detail_channels: int = 48,
        stem_channels: int = 16,
        guide_channels: int = 24,
        bhlr_freq_cutoff: float = 0.15,
        bhlr_freq_sharpness: float = 24.0,

        guidance_mode: str = "auto",
        detach_external_guidance: bool = True,
        external_structure_channels: int = 32,

        alignment_cfg: Optional[Dict[str, Any]] = None,
        alignment_loss_cfg: Optional[Dict[str, Any]] = None,
        verbose: bool = True,
    ):
        alignment_cfg = dict(alignment_cfg or {})
        alignment_loss_cfg = dict(alignment_loss_cfg or {})

        # ----------------------------------------------------------
        # RLSFA auxiliary supervision configuration.
        # Kept compatible with the original RLSFA+BHLR-v2 trainer.
        # ----------------------------------------------------------
        self.alignment_loss_cfg = alignment_loss_cfg
        self.alignment_loss_enabled = bool(
            alignment_loss_cfg.get("enabled", False)
        )
        self.lambda_align = float(
            alignment_loss_cfg.get("lambda_align", 0.2)
        )
        self.lambda_coarse = float(
            alignment_loss_cfg.get("lambda_coarse", 0.5)
        )
        self.lambda_targetness = float(
            alignment_loss_cfg.get("lambda_targetness", 0.2)
        )
        self.lambda_fine_reg = float(
            alignment_loss_cfg.get("lambda_fine_reg", 0.02)
        )
        self.smooth_l1_beta = float(
            alignment_loss_cfg.get("smooth_l1_beta", 1.0)
        )
        self.targetness_alpha = float(
            alignment_loss_cfg.get("targetness_alpha", 0.75)
        )
        self.targetness_gamma = float(
            alignment_loss_cfg.get("targetness_gamma", 2.0)
        )
        self.clamp_gt_to_capacity = bool(
            alignment_loss_cfg.get("clamp_gt_to_capacity", True)
        )

        loss_scales = alignment_loss_cfg.get("scales", None)
        self.alignment_loss_scales = (
            None
            if loss_scales is None
            else tuple(
                sorted({int(s) for s in loss_scales})
            )
        )

        # Accept the newer spelling without breaking the old parent API.
        if "use_for_bhlr" in alignment_cfg:
            alignment_cfg["use_for_rsdt"] = bool(
                alignment_cfg["use_for_bhlr"]
            )

        # Parent is used for:
        #   dual backbone
        #   fusion
        #   head/loss
        #   alignment plugin
        #
        # Its RSD-T object is only a temporary placeholder and is replaced
        # immediately below. P3 is supplied only to satisfy the old parent API.
        super().__init__(
            model_name=model_name,
            nc=nc,
            pretrained=pretrained,
            fusion=fusion,
            fusion_indices=fusion_indices,
            align_mode=align_mode,
            names=names,
            semantic_imgsz=semantic_imgsz,
            rsdt_scales=(3,),
            use_guidance=True,
            detail_channels=detail_channels,
            stem_channels=stem_channels,
            guide_channels=guide_channels,
            alignment_cfg=alignment_cfg,
            verbose=False,
        )

        # ----------------------------------------------------------
        # RLSFA auxiliary loss is meaningful only when RLSFA is active.
        # ----------------------------------------------------------
        align_type = str(
            self.alignment_cfg.get("type", "identity")
        ).strip().lower()

        if (
            self.alignment_loss_enabled
            and (
                not self.alignment_enabled
                or align_type != "rlsfa"
            )
        ):
            print(
                "[RLSFA loss] disabled because "
                "RLSFA alignment is not active"
            )
            self.alignment_loss_enabled = False

        # Initialize possible lazy RLSFA parameters BEFORE optimizer build.
        self._initialize_rlsfa_lazy_modules()

        # ----------------------------------------------------------
        # Resolve BHLR injection locations.
        #
        # Preferred NEW interface:
        #     bhlr_layers = actual backbone layer IDs
        #
        # Legacy interface:
        #     bhlr_scales = [3,4,5]
        # maps to the current model's P3/P4/P5 backbone indices.
        # ----------------------------------------------------------
        if bhlr_layers is not None:
            resolved_layers = tuple(
                sorted({int(i) for i in bhlr_layers})
            )

        elif bhlr_scales is not None:
            legacy_scales = tuple(
                sorted({int(s) for s in bhlr_scales})
            )

            legacy_map = {
                3: int(self.fusion_indices[0]),
                4: int(self.fusion_indices[1]),
                5: int(self.fusion_indices[2]),
            }

            invalid_legacy = [
                s
                for s in legacy_scales
                if s not in legacy_map
            ]

            if invalid_legacy:
                raise ValueError(
                    "Legacy bhlr_scales only accepts conceptual P3/P4/P5 "
                    f"(3/4/5), got {invalid_legacy}. "
                    "For arbitrary backbone layers use bhlr_layers=[...]."
                )

            resolved_layers = tuple(
                sorted({
                    legacy_map[s]
                    for s in legacy_scales
                })
            )

        else:
            # Preserve the historical default: P3.
            resolved_layers = (
                int(self.fusion_indices[0]),
            )

        self.bhlr_layers = resolved_layers

        if not self.bhlr_layers:
            raise ValueError("BHLR injection layer list cannot be empty")

        bad = [
            i for i in self.bhlr_layers
            if i < 0 or i >= self.backbone_len
        ]
        if bad:
            raise ValueError(
                f"BHLR layer IDs out of range: {bad}; "
                f"valid range is 0..{self.backbone_len - 1}"
            )

        feature_channels = self._infer_any_layer_channels(
            self.bhlr_layers
        )

        rgb_channels = {
            i: int(feature_channels[i]["rgb"])
            for i in self.bhlr_layers
        }
        tir_channels = {
            i: int(feature_channels[i]["tir"])
            for i in self.bhlr_layers
        }

        # Replace the old P3/P4/P5-only module.
        del self.rsdt

        self.bhlr = MultiScaleBHLRv2(
            rgb_channels=rgb_channels,
            tir_channels=tir_channels,
            scales=self.bhlr_layers,  # now arbitrary layer IDs
            high_to_semantic_ratio=int(high_to_semantic_ratio),
            detail_channels=int(detail_channels),
            stem_channels=int(stem_channels),
            guide_channels=int(guide_channels),
            freq_cutoff=float(bhlr_freq_cutoff),
            freq_sharpness=float(bhlr_freq_sharpness),
            exact_identity_when_equal=True,
            guidance_mode=str(guidance_mode),
            detach_external_guidance=bool(detach_external_guidance),
            external_structure_channels=int(external_structure_channels),
        )

        # Compatibility aliases for older training/visualization code.
        self.rsdt = self.bhlr
        self.bhlr_scales = self.bhlr_layers
        self.rsdt_scales = self.bhlr_layers

        if verbose:
            print("\n============================================================")
            print("RLSFA + BHLR-v2 ANY-LAYER Detector")
            print("============================================================")
            print(f"Model              : {self.model_name}")
            print(f"Backbone length    : {self.backbone_len}")
            print(f"BHLR layer IDs     : {list(self.bhlr_layers)}")
            if bhlr_layers is not None:
                print("BHLR config mode   : actual backbone layer IDs")
            elif bhlr_scales is not None:
                print(
                    f"BHLR config mode   : legacy scales {list(bhlr_scales)} "
                    f"-> layers {list(self.bhlr_layers)}"
                )
            else:
                print("BHLR config mode   : default P3 -> actual layer ID")
            for i in self.bhlr_layers:
                print(
                    f"  layer {i:<2}: "
                    f"RGB C={rgb_channels[i]}, "
                    f"TIR C={tir_channels[i]}"
                )
            print(f"Alignment enabled  : {self.alignment_enabled}")
            print(f"Alignment type     : {self.alignment_cfg.get('type', 'identity')}")
            print(f"Alignment loss     : {self.alignment_loss_enabled}")
            print("Injection mode     : INLINE / propagates downstream")
            print("============================================================\n")

    @torch.no_grad()
    def _initialize_rlsfa_lazy_modules(self):
        if not getattr(self, "alignment_enabled", False):
            return
        if str(self.alignment_cfg.get("type", "identity")).strip().lower() != "rlsfa":
            return

        states = (self.model.training, self.tir_backbone.training, self.alignment.training)
        self.model.eval(); self.tir_backbone.eval(); self.alignment.eval()
        device = next(self.model.parameters()).device
        dummy = torch.zeros(1, 3, 256, 256, device=device, dtype=torch.float32)
        rgb_outputs = list(self.forward_rgb_backbone(dummy))
        tir_outputs = list(self.forward_tir_backbone(dummy))
        _ = self.alignment(
            rgb_features=rgb_outputs,
            tir_features=tir_outputs,
            scale_to_index=self.rsdt_scale_to_index,
            return_debug=False,
        )
        self.model.train(states[0]); self.tir_backbone.train(states[1]); self.alignment.train(states[2])

        from torch.nn.parameter import UninitializedParameter
        bad = [n for n, p in self.named_parameters() if isinstance(p, UninitializedParameter)]
        if bad:
            raise RuntimeError("RLSFA lazy parameters remain uninitialized: " + ", ".join(bad))


    # ------------------------------------------------------------------
    # Channel inference for arbitrary backbone indices.
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _infer_any_layer_channels(
        self,
        layer_ids: Iterable[int],
    ) -> Dict[int, Dict[str, int]]:
        layer_ids = tuple(int(i) for i in layer_ids)

        rgb_training = self.model.training
        tir_training = self.tir_backbone.training

        self.model.eval()
        self.tir_backbone.eval()

        device = next(self.model.parameters()).device
        dummy = torch.zeros(
            1, 3, 256, 256,
            device=device,
            dtype=torch.float32,
        )

        rgb_outputs = list(
            self.forward_rgb_backbone(dummy)
        )
        tir_outputs = list(
            self.forward_tir_backbone(dummy)
        )

        channels: Dict[int, Dict[str, int]] = {}

        for i in layer_ids:
            r = rgb_outputs[i]
            t = tir_outputs[i]

            if not isinstance(r, torch.Tensor):
                raise TypeError(
                    f"RGB backbone layer {i} does not return a Tensor"
                )
            if not isinstance(t, torch.Tensor):
                raise TypeError(
                    f"TIR backbone layer {i} does not return a Tensor"
                )

            channels[i] = {
                "rgb": int(r.shape[1]),
                "tir": int(t.shape[1]),
            }

        self.model.train(rgb_training)
        self.tir_backbone.train(tir_training)

        return channels

    # ------------------------------------------------------------------
    # Optional RLSFA guidance -> actual backbone layer IDs.
    #
    # RLSFA is still P-level based. If a BHLR layer corresponds to one of
    # RLSFA's P3/P4/P5 indices, external guidance is reused. Other arbitrary
    # layers automatically fall back when guidance_mode="auto".
    # ------------------------------------------------------------------
    def _alignment_guidance_by_layer(
        self,
        alignment_info: Optional[Mapping],
    ) -> Dict[int, Dict[str, torch.Tensor]]:
        if not alignment_info:
            return {}

        scale_debug = alignment_info.get(
            "scale_debug",
            alignment_info.get("scales", {}),
        )

        if not isinstance(scale_debug, Mapping):
            return {}

        result: Dict[int, Dict[str, torch.Tensor]] = {}

        for scale_key, dbg in scale_debug.items():
            try:
                scale = int(scale_key)
            except Exception:
                continue

            if scale not in self.rsdt_scale_to_index:
                continue

            layer_id = int(
                self.rsdt_scale_to_index[scale]
            )

            if not isinstance(dbg, Mapping):
                continue

            required = (
                "rgb_structure",
                "tir_structure",
                "targetness",
            )

            if all(k in dbg for k in required):
                result[layer_id] = {
                    "rgb_structure": dbg["rgb_structure"],
                    "tir_structure": dbg["tir_structure"],
                    "targetness": dbg["targetness"],
                }

        return result

    # ------------------------------------------------------------------
    # TRUE inline RGB backbone execution.
    # ------------------------------------------------------------------
    def _forward_rgb_backbone_with_bhlr(
        self,
        rgb_high: torch.Tensor,
        rgb_semantic: torch.Tensor,
        tir_features: Sequence[torch.Tensor],
        external_guidance_by_layer: Optional[Mapping[int, Mapping]] = None,
        return_debug: bool = False,
    ):
        external_guidance_by_layer = dict(
            external_guidance_by_layer or {}
        )

        # If high == semantic and exact bypass is enabled, use the normal
        # backbone because there is no lost-detail branch to inject.
        equal_resolution = (
            rgb_high.shape[-2:]
            == rgb_semantic.shape[-2:]
        )

        if equal_resolution and self.bhlr.exact_identity_when_equal:
            outputs = list(
                self.forward_rgb_backbone(rgb_semantic)
            )

            if return_debug:
                return outputs, {
                    "equal_resolution": True,
                    "detail_raw": None,
                    "detail_bank": None,
                    "layers": {},
                    "scales": {},  # compatibility
                }

            return outputs, None

        (
            detail_raw,
            reconstructed_high,
            residual_high,
        ) = self.bhlr.build_lost_detail_bank(
            rgb_high,
            rgb_semantic,
        )

        layers = list(self.model)[:self.backbone_len]

        outputs = []
        x = rgb_semantic

        debug_layers: Dict[int, Dict] = {}

        for layer_id, module in enumerate(layers):
            module_input = get_layer_input(
                module,
                x,
                outputs,
            )

            x = module(module_input)

            # ----------------------------------------------------------
            # Inject IMMEDIATELY after this selected RGB backbone layer.
            # The enhanced tensor is then stored in outputs and therefore
            # participates in every later layer that depends on it.
            # ----------------------------------------------------------
            if layer_id in self.bhlr_layers:
                guidance = external_guidance_by_layer.get(
                    layer_id,
                    None,
                )

                if return_debug:
                    x, dbg = self.bhlr.enhance_layer(
                        layer_id=layer_id,
                        detail_raw=detail_raw,
                        rgb_feat=x,
                        tir_feat=tir_features[layer_id],
                        external_guidance=guidance,
                        return_debug=True,
                    )
                    debug_layers[layer_id] = dbg
                else:
                    x = self.bhlr.enhance_layer(
                        layer_id=layer_id,
                        detail_raw=detail_raw,
                        rgb_feat=x,
                        tir_feat=tir_features[layer_id],
                        external_guidance=guidance,
                        return_debug=False,
                    )

            outputs.append(x)

        if not return_debug:
            return outputs, None

        residual_s2d = F.pixel_unshuffle(
            residual_high,
            downscale_factor=self.bhlr.ratio,
        )

        debug = {
            "equal_resolution": False,
            "reconstructed_high": reconstructed_high,
            "resolution_residual": residual_high,
            "residual_s2d": residual_s2d,
            "detail_raw": detail_raw,
            "detail_bank": detail_raw,
            "layers": debug_layers,

            # compatibility with old visualizers
            "scales": debug_layers,
        }

        return outputs, debug

    # ------------------------------------------------------------------
    # Main forward
    # ------------------------------------------------------------------
    def forward_features(
        self,
        rgb: torch.Tensor,
        tir: torch.Tensor,
        rgb_semantic: Optional[torch.Tensor] = None,
        return_rsdt_debug: bool = False,
        return_alignment_debug: bool = False,
    ) -> Dict:
        if rgb_semantic is None:
            rgb_semantic = self._fallback_semantic_rgb(
                rgb
            )

        # TIR is run once.
        raw_tir_outputs = list(
            self.forward_tir_backbone(tir)
        )

        need_alignment = bool(
            self.alignment_enabled
        )

        # RLSFA currently needs a reference RGB feature set.
        # This raw pre-pass is only required when alignment is active.
        raw_rgb_for_alignment = None
        alignment_info: Dict[str, Any] = {}

        if need_alignment:
            raw_rgb_for_alignment = list(
                self.forward_rgb_backbone(
                    rgb_semantic
                )
            )

            (
                aligned_tir_outputs,
                alignment_info,
            ) = self.alignment(
                rgb_features=raw_rgb_for_alignment,
                tir_features=raw_tir_outputs,
                scale_to_index=self.rsdt_scale_to_index,
                return_debug=(
                    return_alignment_debug
                    or return_rsdt_debug
                ),
            )
        else:
            aligned_tir_outputs = raw_tir_outputs

        if (
            self.alignment_enabled
            and self.alignment_use_for_rsdt
        ):
            tir_for_bhlr = aligned_tir_outputs
        else:
            tir_for_bhlr = raw_tir_outputs

        if (
            self.alignment_enabled
            and self.alignment_use_for_fusion
        ):
            tir_for_fusion = aligned_tir_outputs
        else:
            tir_for_fusion = raw_tir_outputs

        guidance_by_layer = self._alignment_guidance_by_layer(
            alignment_info
        )

        (
            rgb_outputs,
            bhlr_debug,
        ) = self._forward_rgb_backbone_with_bhlr(
            rgb_high=rgb,
            rgb_semantic=rgb_semantic,
            tir_features=tir_for_bhlr,
            external_guidance_by_layer=guidance_by_layer,
            return_debug=return_rsdt_debug,
        )

        (
            fused_outputs,
            fusion_features,
        ) = self.fuse_backbone_features(
            rgb_outputs,
            tir_for_fusion,
        )

        result = {
            "rgb_backbone": rgb_outputs,
            "tir_backbone": raw_tir_outputs,
            "aligned_tir_backbone": aligned_tir_outputs,
            "tir_for_rsdt": tir_for_bhlr,
            "tir_for_bhlr": tir_for_bhlr,
            "tir_for_fusion": tir_for_fusion,
            "fused_backbone": fused_outputs,
            "fusion_features": fusion_features,
            "rgb_semantic": rgb_semantic,

            "bhlr_layers": self.bhlr_layers,

            # Old-key compatibility.
            "rsdt_scales": self.bhlr_layers,
            "bhlr_scales": self.bhlr_layers,

            "alignment_enabled": self.alignment_enabled,
            "alignment_type": self.alignment_cfg.get(
                "type",
                "identity",
            ),
        }

        if raw_rgb_for_alignment is not None:
            result[
                "rgb_backbone_raw_for_alignment"
            ] = raw_rgb_for_alignment

        if return_rsdt_debug:
            result["rsdt_debug"] = bhlr_debug
            result["bhlr_debug"] = bhlr_debug

        if return_alignment_debug or return_rsdt_debug:
            result["alignment_info"] = alignment_info

        return result

    @staticmethod
    def _det_loss_items_to_dict(loss_items):
        if isinstance(loss_items, dict):
            return dict(loss_items)
        if torch.is_tensor(loss_items):
            values = list(loss_items.flatten())
        elif isinstance(loss_items, (list, tuple)):
            values = list(loss_items)
        else:
            return {"loss": loss_items}
        if len(values) == 3:
            return {"box_loss": values[0], "cls_loss": values[1], "dfl_loss": values[2]}
        return {f"loss_{i}": v for i, v in enumerate(values)}

    @staticmethod
    def _scale_debug(info: Dict, scale: int) -> Dict:
        mapping = info.get("scale_debug", info.get("scales", {}))
        if scale in mapping:
            return mapping[scale]
        if str(scale) in mapping:
            return mapping[str(scale)]
        raise KeyError(f"RLSFA debug missing P{scale}; available={list(mapping.keys())}")

    @staticmethod
    def _bbox_xywh_feature(box: torch.Tensor, h: int, w: int):
        return box[0] * w, box[1] * h, box[2] * w, box[3] * h

    @staticmethod
    def _build_rgb_target_mask(batch: Dict, h: int, w: int, device, dtype):
        bsz = int(batch["rgb_semantic_img"].shape[0])
        mask = torch.zeros(bsz, 1, h, w, device=device, dtype=dtype)
        boxes = batch["rgb_semantic_bboxes"]
        bidx = batch["rgb_batch_idx"]
        for i in range(int(boxes.shape[0])):
            bi = int(bidx[i].item())
            cx, cy, bw, bh = boxes[i]
            x1 = max(0, min(w - 1, int(torch.floor((cx - bw / 2) * w).item())))
            y1 = max(0, min(h - 1, int(torch.floor((cy - bh / 2) * h).item())))
            x2 = max(x1 + 1, min(w, int(torch.ceil((cx + bw / 2) * w).item())))
            y2 = max(y1 + 1, min(h, int(torch.ceil((cy + bh / 2) * h).item())))
            mask[bi, 0, y1:y2, x1:x2] = 1.0
        return mask

    def _focal_bce(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prob = torch.sigmoid(logits)
        ce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        pt = target * prob + (1.0 - target) * (1.0 - prob)
        alpha_t = self.targetness_alpha * target + (1.0 - self.targetness_alpha) * (1.0 - target)
        return (alpha_t * (1.0 - pt).pow(self.targetness_gamma) * ce).mean()

    def _capacity(self, scale: int):
        unit = self.alignment.units[str(scale)]
        coarse = float(unit.coarse_radius)
        total = coarse + float(unit.max_fine_offset)
        return coarse, total

    def _alignment_auxiliary_loss(self, batch: Dict, alignment_info: Dict):
        if self.alignment_loss_scales is None:
            scales = [int(s) for s in getattr(self.alignment, "scales", [])]
        else:
            scales = list(self.alignment_loss_scales)
        if not scales:
            raise RuntimeError("No RLSFA alignment-loss scales configured")

        coarse_losses, final_losses, target_losses, fine_regs = [], [], [], []
        valid_pairs = 0

        for key in ("rgb_semantic_bboxes", "rgb_batch_idx", "tir_bboxes", "tir_batch_idx"):
            if key not in batch:
                raise KeyError(f"RLSFA supervision requires batch['{key}']")

        for scale in scales:
            dbg = self._scale_debug(alignment_info, scale)
            coarse = dbg["coarse_offset"].float()  # [B,2,1,1]
            fine = dbg["fine_offset"].float()
            total = dbg["total_offset"].float()
            target_logits = dbg["coarse_targetness_logits"].float()
            bsz, _, h, w = target_logits.shape

            target_mask = self._build_rgb_target_mask(
                batch, h, w, target_logits.device, target_logits.dtype
            )
            target_losses.append(self._focal_bce(target_logits, target_mask))
            fine_regs.append(fine.abs().mean())

            max_coarse, max_total = self._capacity(scale)
            for bi in range(bsz):
                rsel = batch["rgb_batch_idx"] == bi
                tsel = batch["tir_batch_idx"] == bi
                # Avoid inventing correspondence for ambiguous multi-object samples.
                if int(rsel.sum().item()) != 1 or int(tsel.sum().item()) != 1:
                    continue

                rbox = batch["rgb_semantic_bboxes"][rsel][0].float()
                tbox = batch["tir_bboxes"][tsel][0].float()
                rcx, rcy, _, _ = self._bbox_xywh_feature(rbox, h, w)
                tcx, tcy, _, _ = self._bbox_xywh_feature(tbox, h, w)
                gt = torch.stack((tcx - rcx, tcy - rcy)).view(1, 2, 1, 1)

                gt_coarse = gt
                gt_total = gt
                if self.clamp_gt_to_capacity:
                    gt_coarse = gt_coarse.clamp(-max_coarse, max_coarse)
                    gt_total = gt_total.clamp(-max_total, max_total)

                coarse_losses.append(F.smooth_l1_loss(
                    coarse[bi:bi + 1], gt_coarse,
                    reduction="mean", beta=self.smooth_l1_beta
                ))
                final_losses.append(F.smooth_l1_loss(
                    total[bi:bi + 1], gt_total,
                    reduction="mean", beta=self.smooth_l1_beta
                ))
                valid_pairs += 1

        zero = self._scale_debug(alignment_info, scales[0])["coarse_offset"].sum() * 0.0
        coarse_loss = torch.stack(coarse_losses).mean() if coarse_losses else zero
        final_loss = torch.stack(final_losses).mean() if final_losses else zero
        target_loss = torch.stack(target_losses).mean() if target_losses else zero
        fine_reg = torch.stack(fine_regs).mean() if fine_regs else zero

        align_loss = (
            final_loss
            + self.lambda_coarse * coarse_loss
            + self.lambda_targetness * target_loss
            + self.lambda_fine_reg * fine_reg
        )
        return {
            "align_loss": align_loss,
            "align_coarse_loss": coarse_loss,
            "align_final_loss": final_loss,
            "align_targetness_loss": target_loss,
            "align_fine_reg": fine_reg,
            "align_valid_pairs": valid_pairs,
        }

    def loss(self, batch: Dict, preds=None):
        if not self.alignment_loss_enabled or preds is not None:
            return super().loss(batch, preds=preds)

        required = [
            "rgb_img", "rgb_semantic_img", "tir_img",
            "rgb_cls", "rgb_semantic_bboxes", "rgb_batch_idx",
            "tir_bboxes", "tir_batch_idx",
        ]
        for key in required:
            if key not in batch:
                raise KeyError(f"RLSFA+BHLR batch missing: {key}")

        if self.criterion is None:
            self.criterion = self.init_criterion()

        outputs = self.predict(
            batch["rgb_img"],
            batch["tir_img"],
            rgb_semantic=batch["rgb_semantic_img"],
            return_features=True,
            return_alignment_debug=True,
        )

        yolo_batch = {
            "img": batch["rgb_semantic_img"],
            "cls": batch["rgb_cls"],
            "bboxes": batch["rgb_semantic_bboxes"],
            "batch_idx": batch["rgb_batch_idx"],
        }
        det_loss_raw, det_loss_items = self.criterion(outputs["pred"], yolo_batch)
        stats = self._alignment_auxiliary_loss(batch, outputs["alignment_info"])

        bsz = int(batch["rgb_semantic_img"].shape[0])
        weighted_raw = self.lambda_align * stats["align_loss"] * float(bsz)
        if not torch.is_tensor(det_loss_raw):
            raise TypeError(f"Expected tensor detection loss, got {type(det_loss_raw)}")
        total_raw = torch.cat((det_loss_raw.reshape(-1), weighted_raw.reshape(1)), dim=0)

        items = self._det_loss_items_to_dict(det_loss_items)
        items.update({
            "align_loss": stats["align_loss"].detach(),
            "align_coarse_loss": stats["align_coarse_loss"].detach(),
            "align_final_loss": stats["align_final_loss"].detach(),
            "align_targetness_loss": stats["align_targetness_loss"].detach(),
            "align_fine_reg": stats["align_fine_reg"].detach(),
            "align_weighted_loss": (self.lambda_align * stats["align_loss"]).detach(),
            "align_valid_pairs": float(stats["align_valid_pairs"]),
        })
        return total_raw, items

    # ------------------------------------------------------------------
    # Utility: print actual layer IDs / channels / spatial sizes.
    # ------------------------------------------------------------------
    @torch.no_grad()
    def print_backbone_table(
        self,
        imgsz: int = 640,
    ):
        training = self.model.training
        self.model.eval()

        device = next(self.model.parameters()).device
        x = torch.zeros(
            1, 3, int(imgsz), int(imgsz),
            device=device,
        )

        outputs = list(
            self.forward_rgb_backbone(x)
        )

        print("\nRGB backbone layer table")
        print("-" * 74)
        print(
            f"{'ID':>4}  {'Module':<28} "
            f"{'Channels':>9}  {'H':>6}  {'W':>6}"
        )
        print("-" * 74)

        layers = list(self.model)[:self.backbone_len]

        for i, (m, y) in enumerate(zip(layers, outputs)):
            if isinstance(y, torch.Tensor):
                c = int(y.shape[1])
                h = int(y.shape[-2])
                w = int(y.shape[-1])
                shape_text = f"{c:>9}  {h:>6}  {w:>6}"
            else:
                shape_text = f"{'non-Tensor':>23}"

            print(
                f"{i:>4}  "
                f"{m.__class__.__name__:<28} "
                f"{shape_text}"
            )

        print("-" * 74)
        print(
            "Configured BHLR layers:",
            list(self.bhlr_layers),
        )
        print()

        self.model.train(training)


# ------------------------------------------------------------------
# Backward-compatible public class name expected by the existing trainer:
#
#     from models.rgbt_rlsfa_bhlr_model_v2 import (
#         RGBTRLSFABHLRV2DetectionModel
#     )
# ------------------------------------------------------------------
class RGBTRLSFABHLRV2DetectionModel(
    RGBTRLSFABHLRAnyLayerDetectionModel
):
    pass


# Optional aliases.
AnyLayerBHLRModel = RGBTRLSFABHLRAnyLayerDetectionModel
RGBTRLSFABHLRModelV2 = RGBTRLSFABHLRV2DetectionModel

__all__ = [
    "RGBTRLSFABHLRV2DetectionModel",
    "RGBTRLSFABHLRAnyLayerDetectionModel",
    "RGBTRLSFABHLRModelV2",
    "AnyLayerBHLRModel",
]
