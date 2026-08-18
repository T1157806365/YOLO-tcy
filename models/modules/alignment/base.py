"""
Base interface for pluggable RGB-T alignment modules.

Contract
--------
Every alignment plugin must implement:

    forward(
        rgb_features,
        tir_features,
        scale_to_index,
        return_debug=False,
    )

and must return:

    aligned_tir_features, alignment_info

where:
    rgb_features         : list/tuple of RGB backbone outputs
    tir_features         : list/tuple of TIR backbone outputs
    scale_to_index       : {3: idx_p3, 4: idx_p4, 5: idx_p5}
    aligned_tir_features : list with the same structure as tir_features
    alignment_info       : dict, optional debug/offset/confidence information

The rest of the detector does NOT need to know how alignment is implemented.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

import torch
import torch.nn as nn


class BaseAlignment(nn.Module):
    """Unified interface for all future RGB-T alignment plugins."""

    def __init__(
        self,
        cfg: Dict[str, Any] | None = None,
    ):
        super().__init__()

        self.cfg = dict(
            cfg or {}
        )

        self.enabled = bool(
            self.cfg.get(
                "enabled",
                False,
            )
        )

        self.scales = tuple(
            int(s)
            for s in self.cfg.get(
                "scales",
                [3],
            )
        )

    def forward(
        self,
        rgb_features: Sequence[torch.Tensor],
        tir_features: Sequence[torch.Tensor],
        scale_to_index: Mapping[int, int],
        return_debug: bool = False,
    ):
        raise NotImplementedError

    def extra_repr(
        self,
    ) -> str:
        return (
            f"enabled={self.enabled}, "
            f"scales={list(self.scales)}"
        )
