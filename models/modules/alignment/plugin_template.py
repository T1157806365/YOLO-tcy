"""
Template for a future RGB-T alignment plugin.

This file is ONLY a development template.
It intentionally performs identity pass-through.

To implement a real alignment method:
1. inherit BaseAlignment
2. create learnable layers in __init__
3. replace selected TIR feature maps in forward()
4. keep the return interface unchanged
5. set YAML target to this class

Example YAML:
alignment:
  enabled: true
  type: custom
  target: models.modules.alignment.my_alignment:MyAlignment
  scales: [3]
  use_for_rsdt: true
  use_for_fusion: true
"""

from __future__ import annotations

from typing import Mapping, Sequence

import torch

from .base import BaseAlignment


class AlignmentPluginTemplate(
    BaseAlignment
):
    """
    Replace this class with a real alignment implementation later.
    """

    def __init__(
        self,
        cfg=None,
    ):
        super().__init__(
            cfg=cfg
        )

        # Future example:
        # hidden_channels = int(
        #     self.cfg.get(
        #         "hidden_channels",
        #         32,
        #     )
        # )
        #
        # self.offset_head = ...
        # self.confidence_head = ...

    def forward(
        self,
        rgb_features: Sequence[torch.Tensor],
        tir_features: Sequence[torch.Tensor],
        scale_to_index: Mapping[int, int],
        return_debug: bool = False,
    ):
        aligned_tir = list(
            tir_features
        )

        # ---------------------------------------------------
        # Future implementation skeleton
        # ---------------------------------------------------
        #
        # for scale in self.scales:
        #
        #     index = scale_to_index[
        #         scale
        #     ]
        #
        #     rgb_feat = rgb_features[
        #         index
        #     ]
        #
        #     tir_feat = tir_features[
        #         index
        #     ]
        #
        #     aligned_feat = self.align_one_scale(
        #         rgb_feat,
        #         tir_feat,
        #     )
        #
        #     aligned_tir[
        #         index
        #     ] = aligned_feat
        #
        # ---------------------------------------------------

        info = {
            "enabled":
                True,

            "type":
                "template",

            "scales":
                list(
                    self.scales
                ),

            "has_alignment":
                False,
        }

        if return_debug:
            info[
                "message"
            ] = (
                "Template only: no real alignment "
                "has been implemented."
            )

        return (
            aligned_tir,
            info,
        )
