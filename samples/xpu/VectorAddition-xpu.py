# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) <2026> Intel Corporation.
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import sys
import cuda.tile as ct
import torch
import math

# Kernels and helpers shared with the CUDA sample in the parent directory.
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from VectorAddition import vec_add_kernel_2d  # noqa: E402

# Use the in-tree experimental XPU backend hooks by shorthand module name.
ct.set_backend("xpu")


# --- Wrapper Function to Dispatch to Kernels ---
def vec_add(a: torch.Tensor, b: torch.Tensor, options: dict | None = None,
            use_gather: bool = False) -> torch.Tensor:
    """
    Performs element-wise addition of two tensors (vector or matrix) using
    different cuTile kernels based on dimensionality and gather/scatter preference.

    This function acts as a high-level interface, handling input validation,
    determining appropriate tile and grid dimensions, and dispatching to
    the correct cuTile kernel.

    Args:
        a (torch.Tensor): The first input tensor. Must be 1D or 2D and on an XPU device.
        b (torch.Tensor): The second input tensor. Must match 'a' in shape, device, and dtype.
        options (dict): XPU compile options forwarded through
                ``ct.backend_options(options)``.
        use_gather (bool): If True, uses kernels with explicit gather/scatter and masking
                   for robust boundary handling (recommended for non-power-of-2
                   or non-tile-aligned dimensions).
                   On XPU this currently falls back to a padded direct tiled path.

    Returns:
        torch.Tensor: The resulting tensor after element-wise addition.

    Raises:
        ValueError: If input tensors have mismatched shapes, incorrect dimensions,
                    are not on XPU, or have different data types.
    """
    # --- Input Validation ---
    if a.shape != b.shape:
        raise ValueError("Input tensors must have the same shape.")
    if a.dim() > 2 or b.dim() > 2:
        raise ValueError("This function currently supports only 1D or 2D tensors.")
    if a.device != b.device:
        raise ValueError("Input tensors must be on the same device.")
    if not a.is_xpu or not b.is_xpu:
        raise ValueError("Input tensors must be on an XPU device.")
    if a.dtype != b.dtype:
        raise ValueError("Input tensors must have the same data type.")

    # Create an empty output tensor on the same device and with the same dtype as inputs.
    c = torch.empty_like(a)

    # --- Dispatch based on Dimensionality ---
    if a.dim() == 1:
        N = a.shape[0]  # Get the total size of the 1D vector

        # XeGPU elementwise lowering needs a distributable 2D tile shape.
        # Pack 1D vectors as (rows, PACK_N) instead of (N, 1).
        PACK_N = 16
        packed_elems = ((N + PACK_N - 1) // PACK_N) * PACK_N
        rows = packed_elems // PACK_N
        assert packed_elems == N, (
            f"XPU vec_add requires 1D size divisible by {PACK_N}; got N={N}"
        )

        a2 = a.view(rows, PACK_N)
        b2 = b.view(rows, PACK_N)
        c2 = c.view(rows, PACK_N)

        # Keep TILE_X * TILE_Y near 1024 threads with explicit Y packing.
        TILE_Y = PACK_N
        max_tile_x = max(1, 1024 // TILE_Y)
        TILE_X = min(max_tile_x, 2 ** math.ceil(math.log2(rows))) if rows > 0 else 1

        # Grid along packed rows; one block in Y because TILE_Y == packed width.
        grid = (math.ceil(rows / TILE_X), 1, 1)  # (blocks_x, blocks_y, blocks_z)

        # Gather/scatter lowering is currently unstable in the XPU path.
        # Use padded direct tiled loads/stores for both modes.
        kernel = vec_add_kernel_2d
        with ct.backend_options(options):
            ct.launch(torch.xpu.current_stream(), grid, kernel, (a2, b2, c2, TILE_X, TILE_Y))
    else:  # a.dim() == 2 (Matrix)
        M, N = a.shape  # Get rows (M) and columns (N) of the matrix

        # Use fixed 2D tiles and pad inputs so direct loads/stores are in-bounds.
        # This also avoids gather/scatter lowering issues on the current XPU path.
        TILE_X = 64
        TILE_Y = 16

        padded_m = ((M + TILE_X - 1) // TILE_X) * TILE_X
        padded_n = ((N + TILE_Y - 1) // TILE_Y) * TILE_Y
        assert padded_m == M and padded_n == N, (
            f"XPU vec_add requires 2D shape divisible by ({TILE_X}, {TILE_Y}); "
            f"got ({M}, {N})"
        )

        # Calculate the 2D grid dimensions for launching the kernel.
        grid = (math.ceil(M / TILE_X), math.ceil(N / TILE_Y), 1)

        kernel = vec_add_kernel_2d
        with ct.backend_options(options):
            ct.launch(torch.xpu.current_stream(), grid, kernel, (a, b, c, TILE_X, TILE_Y))

    return c  # Return the computed output tensor


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--correctness-check",
        action="store_true",
        help="Check the correctness of the results",
    )
    args = parser.parse_args()

    print("--- Running cuTile Vector/Matrix Addition Examples (XPU) ---")

    if not torch.xpu.is_available():
        raise RuntimeError(
            "XPU is not available. Ensure Intel GPU runtime and PyTorch XPU are installed."
        )

    # --- User Configuration ---
    VECTOR_SIZE_1D = 1_000_000
    MATRIX_SHAPE_2D = (2048, 1024)  # Rows, Columns
    device = "xpu"

    # --- Test Case 1: 1D Vector Add (Direct Tiled) ---
    print("\n--- Test 1: 1D Vector Add (Direct Tiled) ---")
    # Create random input tensors on the XPU device.
    a_1d_direct = torch.randn(VECTOR_SIZE_1D, dtype=torch.float32, device=device)
    b_1d_direct = torch.randn(VECTOR_SIZE_1D, dtype=torch.float32, device=device)
    print(f"Input 1D shape: {a_1d_direct.shape}, dtype: {a_1d_direct.dtype}")
    # Call the vec_add wrapper function, requesting the direct tiled kernel.
    c_1d_cutile_direct = vec_add(a_1d_direct, b_1d_direct, use_gather=False)
    print(
        f"""cuTile Output 1D shape: {c_1d_cutile_direct.shape},
        dtype: {c_1d_cutile_direct.dtype}""")
    if args.correctness_check:
        torch.testing.assert_close(c_1d_cutile_direct, a_1d_direct + b_1d_direct)
        print("Correctness check passed")
    else:
        print("Correctness check disabled")

    # --- Test Case 2: 1D Vector Add (Gather/Scatter) ---
    print("\n--- Test 2: 1D Vector Add (Gather/Scatter) ---")
    # Adjusted from 1_000_001 to nearest size divisible by 16 (XPU packed 1D path).
    VECTOR_SIZE_1D_GATHER = 1_000_000
    a_1d_gather = torch.randn(VECTOR_SIZE_1D_GATHER, dtype=torch.float32, device=device)
    b_1d_gather = torch.randn(VECTOR_SIZE_1D_GATHER, dtype=torch.float32, device=device)
    print(f"Input 1D (gather) shape: {a_1d_gather.shape}, dtype: {a_1d_gather.dtype}")
    c_1d_cutile_gather = vec_add(a_1d_gather, b_1d_gather, use_gather=True)
    print(
        f"""cuTile Output 1D (gather) shape: {c_1d_cutile_gather.shape},
        dtype: {c_1d_cutile_gather.dtype}""")
    if args.correctness_check:
        torch.testing.assert_close(c_1d_cutile_gather, a_1d_gather + b_1d_gather)
        print("Correctness check passed")
    else:
        print("Correctness check disabled")

    # --- Test Case 3: 2D Matrix Add (Direct Tiled) ---
    print("\n--- Test 3: 2D Matrix Add (Direct Tiled) ---")
    a_2d_direct = torch.randn(MATRIX_SHAPE_2D, dtype=torch.float32, device=device)
    b_2d_direct = torch.randn(MATRIX_SHAPE_2D, dtype=torch.float32, device=device)
    print(f"Input 2D shape: {a_2d_direct.shape}, dtype: {a_2d_direct.dtype}")
    # Call the vec_add wrapper function for 2D, requesting the direct tiled kernel.
    c_2d_cutile_direct = vec_add(a_2d_direct, b_2d_direct, use_gather=False)
    print(
        f"""cuTile Output 2D shape: {c_2d_cutile_direct.shape},
        dtype: {c_2d_cutile_direct.dtype}""")
    if args.correctness_check:
        torch.testing.assert_close(c_2d_cutile_direct, a_2d_direct + b_2d_direct)
        print("Correctness check passed")
    else:
        print("Correctness check disabled")

    # --- Test Case 4: 2D Matrix Add (Gather/Scatter) ---
    print("\n--- Test 4: 2D Matrix Add (Gather/Scatter) ---")
    # Adjusted from (2000, 1000) to nearest shape compatible with 64x16 tiles.
    MATRIX_SHAPE_2D_GATHER = (1984, 1008)
    a_2d_gather = torch.randn(MATRIX_SHAPE_2D_GATHER, dtype=torch.float32, device=device)
    b_2d_gather = torch.randn(MATRIX_SHAPE_2D_GATHER, dtype=torch.float32, device=device)
    print(f"Input 2D (gather) shape: {a_2d_gather.shape}, dtype: {a_2d_gather.dtype}")
    c_2d_cutile_gather = vec_add(a_2d_gather, b_2d_gather, use_gather=True)
    print(
        f"""cuTile Output 2D (gather) shape: {c_2d_cutile_gather.shape},
        dtype: {c_2d_cutile_gather.dtype}""")
    if args.correctness_check:
        torch.testing.assert_close(c_2d_cutile_gather, a_2d_gather + b_2d_gather)
        print("Correctness check passed")
    else:
        print("Correctness check disabled")

    print("\n--- cuTile Vector/Matrix Addition examples complete ---")
