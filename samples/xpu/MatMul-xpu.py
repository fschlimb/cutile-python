# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) <2026> Intel Corporation.
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import sys
import cuda.tile as ct
import torch
from math import ceil

# Kernels and helpers shared with the CUDA sample in the parent directory.
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from MatMul import matmul_kernel, persistent_matmul_kernel  # noqa: E402

# Use the in-tree experimental XPU backend hooks by shorthand module name.
ct.set_backend("xpu")


def cutile_matmul(A: torch.Tensor, B: torch.Tensor, options: dict,
                  persistent: bool = False) -> torch.Tensor:
    """
    Performs matrix multiplication C = A @ B using a cuTile kernel with a 2D grid.

    This wrapper function handles input validation, determines appropriate
    tile sizes based on data type, calculates the necessary grid dimensions,
    and launches the `matmul_kernel`.

    Args:
        A (torch.Tensor): The first input matrix (M x K). Must be on an XPU device.
        B (torch.Tensor): The second input matrix (K x N). Must be on an XPU device
                          and have its K dimension match A's K dimension.
        persistent (bool): Whether to use the persistent kernel.

    Returns:
        torch.Tensor: The resulting matrix C (M x N) on the XPU device.

    Raises:
        ValueError: If matrices are incompatible (K dimensions don't match),
                    or if they are not on a XPU device.
    """
    # --- Input Validation ---
    if A.shape[1] != B.shape[0]:
        raise ValueError(f"Incompatible matrices: K dimension of A ({A.shape[1]}) "
                         f"must match K dimension of B ({B.shape[0]})")
    if A.device != B.device:
        raise ValueError("Input tensors must be on the same device.")
    if not A.is_xpu or not B.is_xpu:
        raise ValueError("Input tensors must be on a XPU device.")
    # Note: cuTile handles dtype compatibility within the kernel, but inputs should generally match.

    # --- Tile Shapes of the XPU schedule ---
    tm = 256
    tn = 256
    tk = 32

    # --- Get Matrix Dimensions ---
    m, k_a = A.shape  # M = total rows of A (and C), K_A = total columns of A
    k_b, n = B.shape  # K_B = total rows of B, N = total columns of B (and C)
    # Note: k_a and k_b must be equal due to validation. This is the 'K' dimension.

    # --- Calculate Grid Dimensions for Kernel Launch (1D Grid) ---
    # The grid defines how many blocks will be launched.
    # Each block computes one (tm x tn) output tile of matrix C.
    # `ceil(total_dim / tile_dim)` ensures enough blocks are launched to cover
    # the entire matrix, even if dimensions are not perfect multiples of tile sizes.
    grid_x = ceil(m / tm)  # Number of blocks needed along the M dimension (rows of C)
    grid_y = ceil(n / tn)  # Number of blocks needed along the N dimension (columns of C)
    grid_size = grid_x * grid_y
    if persistent:
        # One persistent block per Xe core.
        num_xe_cores = torch.xpu.get_device_properties(A.device).gpu_subslice_count
        grid_size = min(num_xe_cores, grid_size)
    grid = (grid_size, 1, 1)

    # --- Create Output Tensor C ---
    # The output tensor `C` is initialized with the correct dimensions (M x N),
    # on the same device, and with the same data type as the input matrices.
    C = torch.empty((m, n), device=A.device, dtype=torch.float32)

    # --- Launch the cuTile Kernel ---
    # The `matmul_kernel` is launched with the calculated grid dimensions.
    # `tm`, `tn`, and `tk` are passed as Constant integers to the kernel.
    kernel = persistent_matmul_kernel if persistent else matmul_kernel
    with ct.backend_options(options):
        ct.launch(torch.xpu.current_stream(), grid, kernel, (A, B, C, tm, tn, tk))

    return C


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--correctness-check",
        action="store_true",
        help="Check the correctness of the results",
    )
    args = parser.parse_args()

    # --- Running cuTile Matrix Multiplication Examples ---
    print("--- Running cuTile Matrix Multiplication Examples (2D Grid) ---")

    # Define common matrix dimensions for the examples
    M_dim = 4096
    N_dim = 4096
    K_dim = 4096

    device = "xpu"
    options = {
        "wg_m": 256, "wg_n": 256,
        "sg_m": 64, "sg_n": 32,
    }
    atol, rtol = 1e-2, 1e-2

    # --- Test Case 1: float16 (Half-Precision) ---
    print("\n--- Test Case 1: Matrix Multiplication with float16 (Half-Precision) ---")
    # Create random input matrices with float16 data type on the XPU device.
    A_fp16 = torch.randn(M_dim, K_dim, dtype=torch.float16, device=device)
    B_fp16 = torch.randn(K_dim, N_dim, dtype=torch.float16, device=device)
    print(f"Input A shape: {A_fp16.shape}, dtype: {A_fp16.dtype}")
    print(f"Input B shape: {B_fp16.shape}, dtype: {B_fp16.dtype}")

    # Perform matrix multiplication using the cuTile wrapper function.
    with ct.backend_options(options):
        C_fp32_cutile = cutile_matmul(A_fp16, B_fp16, options)
    print(f"cuTile Output C shape: {C_fp32_cutile.shape}, dtype: {C_fp32_cutile.dtype}")
    if args.correctness_check:
        torch.testing.assert_close(
            C_fp32_cutile, A_fp16.to(torch.float32) @ B_fp16.to(torch.float32),
            atol=atol, rtol=rtol)
        print("Correctness check passed")
    else:
        print("Correctness check disabled")

    # Test Case 2 (float32) of the CUDA sample is not ported to XPU yet.

    # --- Test Case 3: Dimensions Not Multiples of Tile Sizes ---
    print("""\n--- Test Case 3: Matrix Multiplication with Dimensions
            Not Perfect Multiples of Tile Sizes ---""")
    # Define matrix dimensions that are not exact multiples of the default tile sizes (32, 32, 32).
    # This demonstrates that `ceil` in grid calculation correctly handles partial tiles.
    M_dim_non_mult = 1000
    N_dim_non_mult = 500
    K_dim_non_mult = 700
    A_non_mult = torch.randn(M_dim_non_mult, K_dim_non_mult, dtype=torch.float16, device=device)
    B_non_mult = torch.randn(K_dim_non_mult, N_dim_non_mult, dtype=torch.float16, device=device)
    print(f"Input A shape: {A_non_mult.shape}, dtype: {A_non_mult.dtype}")
    print(f"Input B shape: {B_non_mult.shape}, dtype: {B_non_mult.dtype}")

    C_non_mult_cutile = cutile_matmul(A_non_mult, B_non_mult, options)
    print(f"cuTile Output C shape: {C_non_mult_cutile.shape}, dtype: {C_non_mult_cutile.dtype}")
    if args.correctness_check:
        torch.testing.assert_close(
            C_non_mult_cutile,
            A_non_mult.to(torch.float32) @ B_non_mult.to(torch.float32),
            atol=atol, rtol=rtol)
        print("Correctness check passed")
    else:
        print("Correctness check disabled")

    # --- Test Case 4: Persistent Matmul ---
    print("\n--- Test Case 4: Matrix Multiplication with Persistent Matmul ---")
    C_persistent_fp32_cutile = cutile_matmul(A_fp16, B_fp16, options, persistent=True)
    print(f"cuTile Output C shape: {C_persistent_fp32_cutile.shape}, "
          f"dtype: {C_persistent_fp32_cutile.dtype}")
    if args.correctness_check:
        torch.testing.assert_close(
            C_persistent_fp32_cutile,
            A_fp16.to(torch.float32) @ B_fp16.to(torch.float32),
            atol=atol,
            rtol=rtol,
        )
        print("Correctness check passed")
    else:
        print("Correctness check disabled")

    print("\n--- All cuTile matrix multiplication examples completed. ---")
