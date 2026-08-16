"""
HRA-v1: Heterogeneous-Resolution Alignment Fusion
=================================================

Purpose
-------
Align a lower-resolution TIR feature map to a higher-resolution RGB
feature map using a learnable sampling grid, then fuse the two modalities.

Main computation
----------------

RGB:
    F_R ∈ R^{B×C_r×H_r×W_r}

TIR:
    F_T ∈ R^{B×C_t×H_t×W_t}

1) Channel projection
    F_T^p ∈ R^{B×C_r×H_t×W_t}

2) Build base grid in RGB reference space
    G_0 ∈ R^{B×H_r×W_r×2}

3) Coarse sampling
    F_T^coarse = grid_sample(F_T^p, G_0)
    F_T^coarse ∈ R^{B×C_r×H_r×W_r}

4) Offset prediction
    [F_R, F_T^coarse]
        ∈ R^{B×2C_r×H_r×W_r}

    Δ = phi([F_R, F_T^coarse])
    Δ ∈ R^{B×2×H_r×W_r}

5) Bounded offset
    Δ_b = max_offset * tanh(Δ)
    Δ_b -> permute
    Δ_b ∈ R^{B×H_r×W_r×2}

6) Refined sampling grid
    G = G_0 + Δ_b
    G ∈ R^{B×H_r×W_r×2}

7) Refined TIR sampling
    F_T^align = grid_sample(F_T^p, G)
    F_T^align ∈ R^{B×C_r×H_r×W_r}

8) Fusion
    [F_R, F_T^align]
        ∈ R^{B×2C_r×H_r×W_r}

    F_HRA = Conv1x1([F_R, F_T^align])
    F_HRA ∈ R^{B×C_r×H_r×W_r}


Design notes
------------
1. The final offset-head convolution is zero initialized.
   Therefore the module starts with Δ = 0 and refined sampling is initially
   identical to coarse sampling. This makes training more stable.

2. The final 1×1 fusion convolution is initialized as an RGB pass-through
   when out_channels == rgb_channels:
       F_HRA ≈ F_R at initialization.
   The TIR contribution is then learned gradually.

3. `max_offset` is measured in normalized grid coordinates [-1, 1].
   Example:
       max_offset = 0.10
   means each refined sampling coordinate can move at most ±0.10 in x/y
   after tanh bounding.

4. This v1 intentionally contains NO reliability attention, NO scale
   attention and NO auxiliary loss. Those should be separate modules /
   ablations later.
"""

from __future__ import annotations

from typing import Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. Basic Conv block used inside HRA
# ============================================================

class HRAConvBNAct(nn.Module):
    """
    Conv2d -> BatchNorm2d -> SiLU
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
    ):
        super().__init__()

        padding = (
            kernel_size // 2
        )

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )

        self.bn = nn.BatchNorm2d(
            out_channels
        )

        self.act = nn.SiLU(
            inplace=True
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        return self.act(
            self.bn(
                self.conv(
                    x
                )
            )
        )


# ============================================================
# 2. HRA-v1
# ============================================================

class HRAFusion(nn.Module):
    """
    Heterogeneous-Resolution Alignment Fusion.

    Parameters
    ----------
    rgb_channels:
        RGB input feature channels C_r.

    tir_channels:
        TIR input feature channels C_t.

    out_channels:
        HRA output channels.
        Default = rgb_channels.

    align_mode:
        Sampling mode used by torch.nn.functional.grid_sample.
        Recommended:
            "bilinear"

        Also supports:
            "nearest"
            "bicubic"

    max_offset:
        Maximum normalized offset after tanh bounding.

        The grid_sample coordinate range is [-1, 1].
        Example:
            max_offset = 0.10

        Then:
            delta_x ∈ [-0.10, +0.10]
            delta_y ∈ [-0.10, +0.10]

    offset_hidden_channels:
        Hidden channels inside Offset Predictor.
        If None:
            max(32, rgb_channels // 2)

    padding_mode:
        grid_sample padding mode:
            "zeros"
            "border"
            "reflection"

        "border" is recommended for v1.

    align_corners:
        grid_sample align_corners.
        Keep False for consistency with modern PyTorch resizing.
    """

    def __init__(
        self,
        rgb_channels: int,
        tir_channels: int,
        out_channels: Optional[int] = None,
        align_mode: str = "bilinear",
        max_offset: float = 0.10,
        offset_hidden_channels: Optional[int] = None,
        padding_mode: str = "border",
        align_corners: bool = False,
    ):
        super().__init__()

        # ----------------------------------------------------
        # Basic metadata
        # ----------------------------------------------------

        self.rgb_channels = int(
            rgb_channels
        )

        self.tir_channels = int(
            tir_channels
        )

        self.out_channels = int(
            out_channels
            if out_channels is not None
            else rgb_channels
        )

        self.align_mode = str(
            align_mode
        ).lower()

        self.max_offset = float(
            max_offset
        )

        self.padding_mode = str(
            padding_mode
        ).lower()

        self.align_corners = bool(
            align_corners
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        if self.rgb_channels <= 0:

            raise ValueError(
                "rgb_channels 必须 > 0"
            )

        if self.tir_channels <= 0:

            raise ValueError(
                "tir_channels 必须 > 0"
            )

        if self.out_channels <= 0:

            raise ValueError(
                "out_channels 必须 > 0"
            )

        if self.max_offset < 0:

            raise ValueError(
                "max_offset 必须 >= 0"
            )

        valid_sampling_modes = {
            "bilinear",
            "nearest",
            "bicubic",
        }

        if (
            self.align_mode
            not in valid_sampling_modes
        ):

            raise ValueError(
                "\nHRA grid_sample 不支持 align_mode:\n"
                f"{self.align_mode}\n\n"
                "支持:\n"
                "  bilinear\n"
                "  nearest\n"
                "  bicubic\n"
            )

        valid_padding_modes = {
            "zeros",
            "border",
            "reflection",
        }

        if (
            self.padding_mode
            not in valid_padding_modes
        ):

            raise ValueError(
                "\nHRA grid_sample 不支持 padding_mode:\n"
                f"{self.padding_mode}\n"
            )

        # ====================================================
        # Step 1:
        # TIR channel projection
        #
        # F_T:
        #   [B, C_t, H_t, W_t]
        #
        # ->
        #
        # F_T^p:
        #   [B, C_r, H_t, W_t]
        # ====================================================

        if (
            self.tir_channels
            != self.rgb_channels
        ):

            self.tir_proj = (
                HRAConvBNAct(
                    in_channels=(
                        self.tir_channels
                    ),
                    out_channels=(
                        self.rgb_channels
                    ),
                    kernel_size=1,
                )
            )

        else:

            self.tir_proj = (
                nn.Identity()
            )

        # ====================================================
        # Step 2:
        # Offset Predictor
        #
        # Input:
        #   concat(
        #       F_R,
        #       F_T^coarse
        #   )
        #
        # Shape:
        #   [B, 2*C_r, H_r, W_r]
        #
        # Output:
        #   raw_offset
        #
        # Shape:
        #   [B, 2, H_r, W_r]
        #
        # Channel 0:
        #   delta_x
        #
        # Channel 1:
        #   delta_y
        # ====================================================

        if offset_hidden_channels is None:

            offset_hidden_channels = max(
                32,
                self.rgb_channels // 2,
            )

        self.offset_hidden_channels = int(
            offset_hidden_channels
        )

        self.offset_stem = nn.Sequential(
            HRAConvBNAct(
                in_channels=(
                    self.rgb_channels
                    * 2
                ),
                out_channels=(
                    self.offset_hidden_channels
                ),
                kernel_size=3,
            ),

            HRAConvBNAct(
                in_channels=(
                    self.offset_hidden_channels
                ),
                out_channels=(
                    self.offset_hidden_channels
                ),
                kernel_size=3,
            ),
        )

        self.offset_head = nn.Conv2d(
            in_channels=(
                self.offset_hidden_channels
            ),
            out_channels=2,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )

        # ----------------------------------------------------
        # IMPORTANT:
        # start from zero offset
        #
        # raw_offset = 0
        # bounded_offset = 0
        # refined_grid = base_grid
        #
        # This gives a stable initialization.
        # ----------------------------------------------------

        nn.init.zeros_(
            self.offset_head.weight
        )

        nn.init.zeros_(
            self.offset_head.bias
        )

        # ====================================================
        # Step 3:
        # Fusion projection
        #
        # Input:
        #   [F_R, F_T^align]
        #
        # Shape:
        #   [B, 2*C_r, H_r, W_r]
        #
        # Output:
        #   F_HRA
        #
        # Shape:
        #   [B, C_out, H_r, W_r]
        #
        # NOTE:
        # We intentionally use a plain 1x1 Conv here rather
        # than Conv+BN+SiLU so that RGB-pass-through
        # initialization can be exact.
        # ====================================================

        self.fuse = nn.Conv2d(
            in_channels=(
                self.rgb_channels
                * 2
            ),
            out_channels=(
                self.out_channels
            ),
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

        self._init_fusion_projection()

    # ========================================================
    # 3. RGB-preserving fusion initialization
    # ========================================================

    def _init_fusion_projection(
        self,
    ):
        """
        Initialize final fusion projection.

        If:
            out_channels == rgb_channels

        initialize:
            output = RGB

        i.e.:
            RGB weights = identity
            TIR weights = zero

        This avoids immediately destroying pretrained RGB
        feature semantics at the beginning of HRA training.
        """

        with torch.no_grad():

            self.fuse.weight.zero_()

            if (
                self.fuse.bias
                is not None
            ):

                self.fuse.bias.zero_()

            if (
                self.out_channels
                == self.rgb_channels
            ):

                for c in range(
                    self.rgb_channels
                ):

                    self.fuse.weight[
                        c,
                        c,
                        0,
                        0,
                    ] = 1.0

            else:

                # Non-standard case:
                # use normal Kaiming initialization.
                nn.init.kaiming_normal_(
                    self.fuse.weight,
                    mode="fan_out",
                    nonlinearity="linear",
                )

    # ========================================================
    # 4. Base grid
    # ========================================================

    @staticmethod
    def build_base_grid(
        batch_size: int,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Construct normalized base sampling grid.

        Output:
            G_0
            [B, H, W, 2]

        Last dimension:
            [..., 0] = x
            [..., 1] = y

        Coordinate range:
            approximately [-1, 1]

        IMPORTANT
        ---------
        These coordinates correspond to PIXEL CENTERS for
        align_corners=False.

        For x position j:
            x_j = 2 * (j + 0.5) / W - 1

        For y position i:
            y_i = 2 * (i + 0.5) / H - 1
        """

        if height <= 0 or width <= 0:

            raise ValueError(
                "height / width 必须 > 0"
            )

        y = (
            (
                torch.arange(
                    height,
                    device=device,
                    dtype=dtype,
                )
                + 0.5
            )
            * (
                2.0
                / float(
                    height
                )
            )
            - 1.0
        )

        x = (
            (
                torch.arange(
                    width,
                    device=device,
                    dtype=dtype,
                )
                + 0.5
            )
            * (
                2.0
                / float(
                    width
                )
            )
            - 1.0
        )

        grid_y, grid_x = torch.meshgrid(
            y,
            x,
            indexing="ij",
        )

        base_grid = torch.stack(
            [
                grid_x,
                grid_y,
            ],
            dim=-1,
        )

        # [H, W, 2]
        # ->
        # [1, H, W, 2]
        # ->
        # [B, H, W, 2]

        base_grid = (
            base_grid
            .unsqueeze(
                0
            )
            .expand(
                batch_size,
                -1,
                -1,
                -1,
            )
        )

        return base_grid

    # ========================================================
    # 5. Grid sampling helper
    # ========================================================

    def sample_feature(
        self,
        feature: torch.Tensor,
        grid: torch.Tensor,
    ) -> torch.Tensor:
        """
        Wrapper around torch.nn.functional.grid_sample.

        feature:
            [B, C, H_source, W_source]

        grid:
            [B, H_target, W_target, 2]

        output:
            [B, C, H_target, W_target]
        """

        return F.grid_sample(
            input=feature,
            grid=grid,
            mode=self.align_mode,
            padding_mode=(
                self.padding_mode
            ),
            align_corners=(
                self.align_corners
            ),
        )

    # ========================================================
    # 6. Forward
    # ========================================================

    def forward(
        self,
        rgb: torch.Tensor,
        tir: torch.Tensor,
        return_debug: bool = False,
    ) -> Union[
        torch.Tensor,
        Dict[str, torch.Tensor],
    ]:
        """
        Parameters
        ----------
        rgb:
            [B, C_r, H_r, W_r]

        tir:
            [B, C_t, H_t, W_t]

        return_debug:
            False:
                return only F_HRA.

            True:
                return all important intermediate tensors.
                Useful for visualization / debugging.
        """

        # ----------------------------------------------------
        # Input validation
        # ----------------------------------------------------

        if rgb.ndim != 4:

            raise ValueError(
                "\nRGB feature 必须是 4D Tensor:\n"
                "[B, C, H, W]"
            )

        if tir.ndim != 4:

            raise ValueError(
                "\nTIR feature 必须是 4D Tensor:\n"
                "[B, C, H, W]"
            )

        if (
            rgb.shape[0]
            != tir.shape[0]
        ):

            raise ValueError(
                "\nRGB / TIR batch size 不一致:\n"
                f"RGB={rgb.shape[0]}, "
                f"TIR={tir.shape[0]}"
            )

        if (
            rgb.shape[1]
            != self.rgb_channels
        ):

            raise ValueError(
                "\nRGB feature channels 错误:\n"
                f"got={rgb.shape[1]}, "
                f"expected={self.rgb_channels}"
            )

        if (
            tir.shape[1]
            != self.tir_channels
        ):

            raise ValueError(
                "\nTIR feature channels 错误:\n"
                f"got={tir.shape[1]}, "
                f"expected={self.tir_channels}"
            )

        batch_size = int(
            rgb.shape[0]
        )

        rgb_h = int(
            rgb.shape[-2]
        )

        rgb_w = int(
            rgb.shape[-1]
        )

        # ====================================================
        # STEP 1
        # TIR channel projection
        #
        # [B, C_t, H_t, W_t]
        # ->
        # [B, C_r, H_t, W_t]
        # ====================================================

        tir_projected = self.tir_proj(
            tir
        )

        # ====================================================
        # STEP 2
        # Base grid in RGB reference resolution
        #
        # [B, H_r, W_r, 2]
        # ====================================================

        base_grid = self.build_base_grid(
            batch_size=(
                batch_size
            ),
            height=(
                rgb_h
            ),
            width=(
                rgb_w
            ),
            device=(
                rgb.device
            ),
            dtype=(
                rgb.dtype
            ),
        )

        # ====================================================
        # STEP 3
        # Coarse sampling
        #
        # Source:
        #   tir_projected
        #   [B, C_r, H_t, W_t]
        #
        # Grid:
        #   base_grid
        #   [B, H_r, W_r, 2]
        #
        # Output:
        #   coarse_tir
        #   [B, C_r, H_r, W_r]
        # ====================================================

        coarse_tir = self.sample_feature(
            feature=(
                tir_projected
            ),
            grid=(
                base_grid
            ),
        )

        # ====================================================
        # STEP 4
        # Offset prediction
        #
        # concat:
        #   RGB
        #   [B, C_r, H_r, W_r]
        #
        #   coarse TIR
        #   [B, C_r, H_r, W_r]
        #
        # ->
        #
        # offset_input:
        #   [B, 2*C_r, H_r, W_r]
        # ====================================================

        offset_input = torch.cat(
            [
                rgb,
                coarse_tir,
            ],
            dim=1,
        )

        offset_feature = (
            self.offset_stem(
                offset_input
            )
        )

        raw_offset = (
            self.offset_head(
                offset_feature
            )
        )

        # raw_offset:
        # [B, 2, H_r, W_r]
        #
        # channel 0 = dx
        # channel 1 = dy

        # ====================================================
        # STEP 5
        # Bound the offset
        #
        # [-inf, +inf]
        #
        # ->
        #
        # tanh
        #
        # ->
        #
        # [-1, +1]
        #
        # ->
        #
        # * max_offset
        #
        # ->
        #
        # [-max_offset, +max_offset]
        # ====================================================

        bounded_offset = (
            torch.tanh(
                raw_offset
            )
            * self.max_offset
        )

        # [B, 2, H_r, W_r]
        # ->
        # [B, H_r, W_r, 2]

        bounded_offset_grid = (
            bounded_offset
            .permute(
                0,
                2,
                3,
                1,
            )
            .contiguous()
        )

        # ====================================================
        # STEP 6
        # Refined grid
        #
        # G = G_0 + delta_b
        #
        # [B, H_r, W_r, 2]
        # ====================================================

        refined_grid = (
            base_grid
            + bounded_offset_grid
        )

        # Keep grid numerically inside valid normalized range.
        refined_grid = torch.clamp(
            refined_grid,
            min=-1.0,
            max=1.0,
        )

        # ====================================================
        # STEP 7
        # Refined TIR sampling
        #
        # Source:
        #   tir_projected
        #   [B, C_r, H_t, W_t]
        #
        # Grid:
        #   refined_grid
        #   [B, H_r, W_r, 2]
        #
        # Output:
        #   aligned_tir
        #   [B, C_r, H_r, W_r]
        # ====================================================

        aligned_tir = self.sample_feature(
            feature=(
                tir_projected
            ),
            grid=(
                refined_grid
            ),
        )

        # ====================================================
        # STEP 8
        # RGB-T fusion
        #
        # RGB:
        #   [B, C_r, H_r, W_r]
        #
        # aligned TIR:
        #   [B, C_r, H_r, W_r]
        #
        # concat:
        #   [B, 2*C_r, H_r, W_r]
        #
        # ->
        #
        # F_HRA:
        #   [B, C_out, H_r, W_r]
        # ====================================================

        fusion_input = torch.cat(
            [
                rgb,
                aligned_tir,
            ],
            dim=1,
        )

        fused = self.fuse(
            fusion_input
        )

        # ----------------------------------------------------
        # Normal model path
        # ----------------------------------------------------

        if not return_debug:

            return fused

        # ----------------------------------------------------
        # Debug path
        #
        # Useful later for:
        #   offset visualization
        #   sampling-grid visualization
        #   HRA analysis figures
        # ----------------------------------------------------

        return {
            "rgb":
                rgb,

            "tir":
                tir,

            "tir_projected":
                tir_projected,

            "base_grid":
                base_grid,

            "coarse_tir":
                coarse_tir,

            "offset_input":
                offset_input,

            "raw_offset":
                raw_offset,

            "bounded_offset":
                bounded_offset,

            "bounded_offset_grid":
                bounded_offset_grid,

            "refined_grid":
                refined_grid,

            "aligned_tir":
                aligned_tir,

            "fusion_input":
                fusion_input,

            "fused":
                fused,
        }

    # ========================================================
    # 7. Module representation
    # ========================================================

    def extra_repr(
        self,
    ) -> str:

        return (
            f"rgb_channels={self.rgb_channels}, "
            f"tir_channels={self.tir_channels}, "
            f"out_channels={self.out_channels}, "
            f"align_mode='{self.align_mode}', "
            f"max_offset={self.max_offset}, "
            f"offset_hidden_channels="
            f"{self.offset_hidden_channels}"
        )


# ============================================================
# 8. Standalone sanity test
# ============================================================

if __name__ == "__main__":

    torch.manual_seed(
        0
    )

    # --------------------------------------------------------
    # P3 example:
    #
    # RGB 1280 input:
    #   P3 = 160x160
    #
    # TIR 640 input:
    #   P3 = 80x80
    # --------------------------------------------------------

    batch = 2

    channels = 64

    rgb = torch.randn(
        batch,
        channels,
        160,
        160,
        requires_grad=True,
    )

    tir = torch.randn(
        batch,
        channels,
        80,
        80,
        requires_grad=True,
    )

    hra = HRAFusion(
        rgb_channels=channels,
        tir_channels=channels,
        out_channels=channels,
        align_mode="bilinear",
        max_offset=0.10,
    )

    debug = hra(
        rgb,
        tir,
        return_debug=True,
    )

    print(
        "\n"
        "============================================================"
    )

    print(
        "HRA-v1 sanity test"
    )

    print(
        "============================================================"
    )

    for key, value in (
        debug.items()
    ):

        if torch.is_tensor(
            value
        ):

            print(
                f"{key:<24} "
                f"{list(value.shape)}"
            )

    # --------------------------------------------------------
    # Verify initialization behaviour
    #
    # Offset head starts at zero.
    # Fusion starts as exact RGB pass-through.
    # --------------------------------------------------------

    initial_offset_max = (
        debug[
            "raw_offset"
        ]
        .detach()
        .abs()
        .max()
        .item()
    )

    initial_rgb_diff = (
        (
            debug[
                "fused"
            ]
            - rgb
        )
        .detach()
        .abs()
        .max()
        .item()
    )

    print(
        "------------------------------------------------------------"
    )

    print(
        f"initial raw offset max : "
        f"{initial_offset_max:.8f}"
    )

    print(
        f"initial fused-RGB diff : "
        f"{initial_rgb_diff:.8f}"
    )

    # --------------------------------------------------------
    # Backward test
    # --------------------------------------------------------

    loss = (
        debug[
            "fused"
        ]
        .square()
        .mean()
    )

    loss.backward()

    print(
        "------------------------------------------------------------"
    )

    print(
        f"loss                   : "
        f"{loss.item():.8f}"
    )

    print(
        f"RGB grad exists        : "
        f"{rgb.grad is not None}"
    )

    print(
        f"TIR grad exists        : "
        f"{tir.grad is not None}"
    )

    print(
        "============================================================"
    )

    print(
        "HRA-v1 forward/backward test passed."
    )
