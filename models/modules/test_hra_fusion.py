"""
Quick integration test for HRA-v1.

Run from project root:

    conda activate tcy
    cd /mnt/sda/taochangyong/Projects/Model/YOLO-tcy

    python tests/test_hra_fusion.py
"""

import torch

from models.modules.hra_fusion import (
    HRAFusion,
)


def run_case(
    name,
    rgb_shape,
    tir_shape,
):

    print(
        "\n"
        "============================================================"
    )

    print(
        name
    )

    print(
        "============================================================"
    )

    rgb = torch.randn(
        *rgb_shape,
        requires_grad=True,
    )

    tir = torch.randn(
        *tir_shape,
        requires_grad=True,
    )

    module = HRAFusion(
        rgb_channels=rgb_shape[1],
        tir_channels=tir_shape[1],
        out_channels=rgb_shape[1],
        align_mode="bilinear",
        max_offset=0.10,
    )

    debug = module(
        rgb,
        tir,
        return_debug=True,
    )

    expected_shape = (
        rgb_shape[0],
        rgb_shape[1],
        rgb_shape[2],
        rgb_shape[3],
    )

    assert (
        tuple(
            debug[
                "fused"
            ].shape
        )
        == expected_shape
    )

    assert (
        tuple(
            debug[
                "coarse_tir"
            ].shape
        )
        == expected_shape
    )

    assert (
        tuple(
            debug[
                "aligned_tir"
            ].shape
        )
        == expected_shape
    )

    assert (
        tuple(
            debug[
                "raw_offset"
            ].shape
        )
        == (
            rgb_shape[0],
            2,
            rgb_shape[2],
            rgb_shape[3],
        )
    )

    assert (
        tuple(
            debug[
                "base_grid"
            ].shape
        )
        == (
            rgb_shape[0],
            rgb_shape[2],
            rgb_shape[3],
            2,
        )
    )

    print(
        f"RGB              : "
        f"{list(rgb.shape)}"
    )

    print(
        f"TIR              : "
        f"{list(tir.shape)}"
    )

    print(
        f"TIR projected    : "
        f"{list(debug['tir_projected'].shape)}"
    )

    print(
        f"Base grid        : "
        f"{list(debug['base_grid'].shape)}"
    )

    print(
        f"Coarse TIR       : "
        f"{list(debug['coarse_tir'].shape)}"
    )

    print(
        f"Raw offset       : "
        f"{list(debug['raw_offset'].shape)}"
    )

    print(
        f"Refined grid     : "
        f"{list(debug['refined_grid'].shape)}"
    )

    print(
        f"Aligned TIR      : "
        f"{list(debug['aligned_tir'].shape)}"
    )

    print(
        f"Fused            : "
        f"{list(debug['fused'].shape)}"
    )

    initial_diff = (
        debug[
            "fused"
        ]
        .detach()
        .sub(
            rgb.detach()
        )
        .abs()
        .max()
        .item()
    )

    print(
        f"Initial RGB diff : "
        f"{initial_diff:.8f}"
    )

    loss = (
        debug[
            "fused"
        ]
        .square()
        .mean()
    )

    loss.backward()

    assert (
        rgb.grad
        is not None
    )

    assert (
        tir.grad
        is not None
    )

    print(
        "Backward         : OK"
    )


if __name__ == "__main__":

    # Same resolution baseline
    run_case(
        "Case 1: RGB640 / TIR640 style feature",
        (
            2,
            64,
            80,
            80,
        ),
        (
            2,
            64,
            80,
            80,
        ),
    )

    # Heterogeneous-resolution P3 example
    run_case(
        "Case 2: RGB1280 / TIR640 P3",
        (
            2,
            64,
            160,
            160,
        ),
        (
            2,
            64,
            80,
            80,
        ),
    )

    # Channel mismatch test
    run_case(
        "Case 3: channel projection",
        (
            2,
            96,
            80,
            80,
        ),
        (
            2,
            64,
            40,
            40,
        ),
    )

    print(
        "\nAll HRA-v1 tests passed."
    )
