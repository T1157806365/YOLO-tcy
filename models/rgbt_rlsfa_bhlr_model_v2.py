"""
RGB-T detector with:
  Module-1: RLSFA
  Module-2: BHLR-v2
  Fusion  : existing concat + 1x1 Conv
  Module-3: OFF

RLSFA training supervision:
  coarse translation GT = C_TIR - C_RGB
  final translation GT  = C_TIR - C_RGB
  targetness             = RGB bbox mask on feature grid
  fine regularization    = |fine_offset|

No affine supervision and no dense deformation.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Sequence, Union

import torch
import torch.nn.functional as F

from models.rgbt_rsdt_model import RGBTRSDTDetectionModel
from models.modules.bhlr_v2 import MultiScaleBHLRv2


class RGBTRLSFABHLRV2DetectionModel(RGBTRSDTDetectionModel):
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
        bhlr_scales: Iterable[int] = (3,),
        high_to_semantic_ratio: int = 2,
        detail_channels: int = 64,
        stem_channels: int = 32,
        guide_channels: int = 32,
        bhlr_freq_cutoff: float = 0.15,
        bhlr_freq_sharpness: float = 24.0,
        alignment_cfg: Optional[Dict[str, Any]] = None,
        alignment_loss_cfg: Optional[Dict[str, Any]] = None,
        verbose: bool = True,
    ):
        alignment_cfg = dict(alignment_cfg or {})
        alignment_loss_cfg = dict(alignment_loss_cfg or {})

        # BHLR-v2 settings are intentionally stored inside alignment_cfg so
        # train_rlsfa_bhlr.py / val_rlsfa_bhlr.py do NOT need to be modified.
        #
        # YAML:
        #   alignment:
        #     bhlr_guidance_mode: auto      # auto | fallback | external
        #     bhlr_detach_external_guidance: true
        self.bhlr_guidance_mode = str(
            alignment_cfg.get("bhlr_guidance_mode", "auto")
        ).strip().lower()
        self.bhlr_detach_external_guidance = bool(
            alignment_cfg.get("bhlr_detach_external_guidance", True)
        )

        if "use_for_bhlr" in alignment_cfg:
            alignment_cfg["use_for_rsdt"] = bool(alignment_cfg["use_for_bhlr"])

        self.alignment_loss_cfg = alignment_loss_cfg
        self.alignment_loss_enabled = bool(alignment_loss_cfg.get("enabled", False))
        self.lambda_align = float(alignment_loss_cfg.get("lambda_align", 0.2))
        self.lambda_coarse = float(alignment_loss_cfg.get("lambda_coarse", 0.5))
        self.lambda_targetness = float(alignment_loss_cfg.get("lambda_targetness", 0.2))
        self.lambda_fine_reg = float(alignment_loss_cfg.get("lambda_fine_reg", 0.02))
        self.smooth_l1_beta = float(alignment_loss_cfg.get("smooth_l1_beta", 1.0))
        self.targetness_alpha = float(alignment_loss_cfg.get("targetness_alpha", 0.75))
        self.targetness_gamma = float(alignment_loss_cfg.get("targetness_gamma", 2.0))
        self.clamp_gt_to_capacity = bool(alignment_loss_cfg.get("clamp_gt_to_capacity", True))

        loss_scales = alignment_loss_cfg.get("scales", None)
        self.alignment_loss_scales = (
            None if loss_scales is None
            else tuple(sorted({int(s) for s in loss_scales}))
        )

        bhlr_scales = tuple(sorted({int(s) for s in bhlr_scales}))

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

        align_type = str(self.alignment_cfg.get("type", "identity")).strip().lower()
        if self.alignment_loss_enabled and (not self.alignment_enabled or align_type != "rlsfa"):
            print("[RLSFA loss] disabled because RLSFA alignment is not active")
            self.alignment_loss_enabled = False

        self._initialize_rlsfa_lazy_modules()

        feature_channels = self._infer_feature_channels()
        rgb_channels, tir_channels = {}, {}
        for scale in self.rsdt_scales:
            idx = self.rsdt_scale_to_index[scale]
            rgb_channels[scale] = int(feature_channels[idx]["rgb"])
            tir_channels[scale] = int(feature_channels[idx]["tir"])

        del self.rsdt
        self.bhlr = MultiScaleBHLRv2(
            rgb_channels=rgb_channels,
            tir_channels=tir_channels,
            scales=self.rsdt_scales,
            high_to_semantic_ratio=int(high_to_semantic_ratio),
            detail_channels=int(detail_channels),
            stem_channels=int(stem_channels),
            guide_channels=int(guide_channels),
            # Kept arguments for old train/val API compatibility.
            # BHLR-v2 no longer performs its own global FFT.
            freq_cutoff=float(bhlr_freq_cutoff),
            freq_sharpness=float(bhlr_freq_sharpness),

            # BHLR-v2 dual-mode guidance.
            guidance_mode=self.bhlr_guidance_mode,
            detach_external_guidance=self.bhlr_detach_external_guidance,

            # RLSFA reliable structures use hidden_channels channels.
            external_structure_channels=int(
                self.alignment_cfg.get("hidden_channels", 32)
            ),

            exact_identity_when_equal=True,
        )
        self.rsdt = self.bhlr
        self.bhlr_scales = self.rsdt_scales

        if verbose:
            print("\n============================================================")
            print("RLSFA + BHLR-v2 RGB-T Detector")
            print("============================================================")
            print(f"Model              : {self.model_name}")
            print("Fusion             : concat (Module-3 OFF)")
            print(f"RLSFA enabled      : {self.alignment_enabled}")
            print(f"RLSFA scales       : {list(getattr(self.alignment, 'scales', []))}")
            print(f"BHLR scales        : {list(self.bhlr_scales)}")
            print(f"BHLR guidance      : {self.bhlr_guidance_mode}")
            print(f"Detach ext guide   : {self.bhlr_detach_external_guidance}")
            print(f"RGB semantic       : {self.semantic_imgsz}")
            print(f"High/semantic ratio: {high_to_semantic_ratio}")
            print(f"Alignment loss     : {self.alignment_loss_enabled}")
            print("Geometry           : translation-only; no affine/dense deformation")
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

    @staticmethod
    def _extract_bhlr_external_guidance(
        alignment_info: Dict,
        scales,
    ):
        """
        Convert the CURRENT RLSFA debug outputs to a BHLR-v2-neutral API.

        BHLR-v2 does not import or know about RLSFA.
        It only receives:
            rgb_structure
            tir_structure
            targetness
        """
        mapping = alignment_info.get(
            "scale_debug",
            alignment_info.get("scales", {}),
        )

        result = {}

        for scale in scales:
            if scale in mapping:
                dbg = mapping[scale]
            elif str(scale) in mapping:
                dbg = mapping[str(scale)]
            else:
                continue

            required = (
                "rgb_reliable_structure",
                "tir_reliable_structure",
                "coarse_targetness",
            )

            if not all(
                key in dbg
                for key in required
            ):
                continue

            result[scale] = {
                "rgb_structure":
                    dbg["rgb_reliable_structure"],

                "tir_structure":
                    dbg["tir_reliable_structure"],

                "targetness":
                    dbg["coarse_targetness"],
            }

        return result

    def forward_features(
        self,
        rgb,
        tir,
        rgb_semantic=None,
        return_rsdt_debug: bool = False,
        return_alignment_debug: bool = False,
    ):
        if rgb_semantic is None:
            rgb_semantic = self._fallback_semantic_rgb(
                rgb
            )

        # -----------------------------------------------------
        # 1. Original semantic RGB/TIR backbones.
        # -----------------------------------------------------
        rgb_outputs = list(
            self.forward_rgb_backbone(
                rgb_semantic
            )
        )

        raw_tir_outputs = list(
            self.forward_tir_backbone(
                tir
            )
        )

        # -----------------------------------------------------
        # 2. Optional RLSFA alignment.
        #
        # For BHLR-v2 external/auto mode, we need the CURRENT
        # RLSFA reliable structures. RLSFA already computes these;
        # return_debug=True only exposes them, it does not introduce
        # another alignment branch.
        # -----------------------------------------------------
        align_type = str(
            self.alignment_cfg.get(
                "type",
                "identity",
            )
        ).strip().lower()

        can_use_external = (
            self.alignment_enabled
            and align_type == "rlsfa"
            and self.alignment_use_for_rsdt
        )

        wants_external = (
            self.bhlr_guidance_mode
            in (
                "auto",
                "external",
            )
        )

        if (
            self.bhlr_guidance_mode
            == "external"
            and not can_use_external
        ):
            raise RuntimeError(
                "BHLR-v2 guidance_mode='external' "
                "requires active RLSFA and "
                "alignment.use_for_bhlr=true."
            )

        need_alignment_debug = (
            return_alignment_debug
            or return_rsdt_debug
            or (
                wants_external
                and can_use_external
            )
        )

        (
            aligned_tir_outputs,
            alignment_info,
        ) = self.alignment(
            rgb_features=rgb_outputs,
            tir_features=raw_tir_outputs,
            scale_to_index=self.rsdt_scale_to_index,
            return_debug=need_alignment_debug,
        )

        # -----------------------------------------------------
        # 3. Existing routing policy.
        # -----------------------------------------------------
        if (
            self.alignment_enabled
            and self.alignment_use_for_rsdt
        ):
            tir_for_bhlr = (
                aligned_tir_outputs
            )
        else:
            tir_for_bhlr = (
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
        # 4. Optional RLSFA -> BHLR-v2 structural guidance.
        #
        # auto:
        #   RLSFA available -> use external
        #   RLSFA absent    -> BHLR fallback
        #
        # fallback:
        #   force BHLR self spatial guidance even when RLSFA exists
        #
        # external:
        #   require RLSFA guidance
        # -----------------------------------------------------
        external_guidance = None

        if (
            can_use_external
            and self.bhlr_guidance_mode
            in (
                "auto",
                "external",
            )
        ):
            external_guidance = (
                self._extract_bhlr_external_guidance(
                    alignment_info,
                    self.bhlr_scales,
                )
            )

            if (
                self.bhlr_guidance_mode
                == "external"
            ):
                missing = [
                    scale
                    for scale in self.bhlr_scales
                    if scale
                    not in external_guidance
                ]

                if missing:
                    raise RuntimeError(
                        "RLSFA external guidance "
                        "missing for BHLR scales "
                        f"{missing}."
                    )

        # -----------------------------------------------------
        # 5. Select scale features and run BHLR-v2.
        # -----------------------------------------------------
        rgb_scale_features = {}
        tir_scale_features = {}

        for scale in self.bhlr_scales:
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
            ] = tir_for_bhlr[
                index
            ]

        bhlr_debug = None

        if return_rsdt_debug:
            (
                enhanced_features,
                bhlr_debug,
            ) = self.bhlr(
                rgb_high=rgb,
                rgb_semantic=rgb_semantic,
                rgb_features=rgb_scale_features,
                tir_features=tir_scale_features,
                external_guidance=external_guidance,
                return_debug=True,
            )
        else:
            enhanced_features = self.bhlr(
                rgb_high=rgb,
                rgb_semantic=rgb_semantic,
                rgb_features=rgb_scale_features,
                tir_features=tir_scale_features,
                external_guidance=external_guidance,
                return_debug=False,
            )

        # Replace only selected RGB pyramid features.
        for scale in self.bhlr_scales:
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
        # 6. Existing RGB-T concat fusion / original neck-head.
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

            "tir_backbone":
                raw_tir_outputs,

            "aligned_tir_backbone":
                aligned_tir_outputs,

            # Old compatibility key.
            "tir_for_rsdt":
                tir_for_bhlr,

            "tir_for_bhlr":
                tir_for_bhlr,

            "tir_for_fusion":
                tir_for_fusion,

            "fused_backbone":
                fused_outputs,

            "fusion_features":
                fusion_features,

            "rgb_semantic":
                rgb_semantic,

            "rsdt_scales":
                self.bhlr_scales,

            "bhlr_scales":
                self.bhlr_scales,

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

            "bhlr_guidance_mode":
                self.bhlr_guidance_mode,
        }

        if return_rsdt_debug:
            result[
                "rsdt_debug"
            ] = bhlr_debug

            result[
                "bhlr_debug"
            ] = bhlr_debug

        if (
            return_alignment_debug
            or return_rsdt_debug
        ):
            result[
                "alignment_info"
            ] = alignment_info

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
