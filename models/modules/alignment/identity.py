"""
Identity alignment.

This is the current default.

It performs NO RGB-T alignment:
    aligned_tir == raw_tir

Properties
----------
- no learnable parameters
- no resampling
- no warping
- no feature modification
- compatible with old RSD-T behavior

When alignment.enabled=false, builder.py returns this module.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import torch

from .base import BaseAlignment


class IdentityAlignment(
    BaseAlignment
):
    """Pass TIR features through unchanged."""

    def forward(
        self,
        rgb_features: Sequence[torch.Tensor],
        tir_features: Sequence[torch.Tensor],
        scale_to_index: Mapping[int, int],
        return_debug: bool = False,
    ):
        # New list container, but every Tensor object is unchanged.
        aligned_tir = list(
            tir_features
        )

        info = {
            "enabled":
                False,

            "type":
                "identity",

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
                "IdentityAlignment: "
                "TIR features are passed through unchanged."
            )

        return (
            aligned_tir,
            info,
        )
