# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) <2026> Intel Corporation.
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import sys
import cuda.tile as ct
import torch
from math import ceil  # Required for host-side grid calculation

# Kernels and helpers shared with the CUDA sample in the parent directory.
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from MatMul import matmul_kernel, persistent_matmul_kernel  # noqa: E402


ct.set_backend("cpu")


def cutile_matmul(A: torch.Tensor, B: torch.Tensor, persistent: bool = False,
                  num_cpu_threads: int = 0) -> torch.Tensor:
    """
    Performs matrix multiplication C = A @ B using a cuTile kernel with a 2D grid.

    This wrapper function handles input validation, determines appropriate
    tile sizes based on data type, calculates the necessary grid dimensions,
    and launches the `matmul_kernel`.

    Args:
        A (torch.Tensor): The first input matrix (M x K). Must be on the CPU.
        B (torch.Tensor): The second input matrix (K x N). Must be on the CPU
                          and have its K dimension match A's K dimension.
        persistent (bool): Whether to use the persistent kernel.

    Returns:
        torch.Tensor: The resulting matrix C (M x N) on the CPU.

    Raises:
        ValueError: If matrices are incompatible (K dimensions don't match),
                    or if they are not on the CPU.
    """
    # --- Input Validation ---
    if A.shape[1] != B.shape[0]:
        raise ValueError(f"Incompatible matrices: K dimension of A ({A.shape[1]}) "
                         f"must match K dimension of B ({B.shape[0]})")
    if A.device != B.device:
        raise ValueError("Input tensors must be on the same device.")
    if not A.is_cpu or not B.is_cpu:
        raise ValueError("Input tensors must be on the CPU.")
    # Note: cuTile handles dtype compatibility within the kernel, but inputs should generally match.

    # --- Determine Tile Shapes based on Data Type for Optimization ---
    # This logic selects optimal tile sizes (tm, tn, tk) based on whether
    # the input is half-precision (e.g., float16, bfloat16, where itemsize=2 bytes)
    # which can often leverage Tensor Cores for higher throughput,
    # or full-precision (e.g., float32, where itemsize=4 bytes).
    if A.dtype.itemsize == 2:  # Likely torch.float16 or torch.bfloat16
        tm, tn, tk = 128, 256, 64  # Larger tiles for Tensor Core friendly types
    else:  # Likely torch.float32 or other
        tm, tn, tk = 32, 32, 32   # Smaller, more general tiles

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
        num_workers = num_cpu_threads or torch.get_num_threads()
        grid_size = min(num_workers, grid_size)
    grid = (grid_size, 1, 1)

    # --- Create Output Tensor C ---
    # The output tensor `C` is initialized with the correct dimensions (M x N),
    # on the same device, and with the same data type as the input matrices.
    C = torch.empty((m, n), device=A.device, dtype=A.dtype)

    # --- Launch the cuTile Kernel ---
    # The `matmul_kernel` is launched with the calculated grid dimensions.
    # `tm`, `tn`, and `tk` are passed as Constant integers to the kernel.
    kernel = persistent_matmul_kernel if persistent else matmul_kernel
    with ct.backend_options({"num_cpu_threads": num_cpu_threads}):
        ct.launch(None, grid, kernel, (A, B, C, tm, tn, tk))

    return C


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--correctness-check",
        action="store_true",
        help="Check the correctness of the results",
    )
    parser.add_argument("--num-cpu-threads", type=int, default=0)
    args = parser.parse_args()

    # --- Running cuTile Matrix Multiplication Examples ---
    print("--- Running cuTile Matrix Multiplication Examples (2D Grid) ---")

    # Define common matrix dimensions for the examples
    M_dim = 512
    N_dim = 512
    K_dim = 768

    # --- Test Case 1: float16 (Half-Precision) ---
    print("\n--- Test Case 1: Matrix Multiplication with float16 (Half-Precision) ---")
    # Create random input matrices with float16 data type on the CPU.
    A_fp16 = torch.randn(M_dim, K_dim, dtype=torch.float16, device='cpu')
    B_fp16 = torch.randn(K_dim, N_dim, dtype=torch.float16, device='cpu')
    print(f"Input A shape: {A_fp16.shape}, dtype: {A_fp16.dtype}")
    print(f"Input B shape: {B_fp16.shape}, dtype: {B_fp16.dtype}")

    # Perform matrix multiplication using the cuTile wrapper function.
    C_fp16_cutile = cutile_matmul(
        A_fp16, B_fp16, num_cpu_threads=args.num_cpu_threads)
    print(f"cuTile Output C shape: {C_fp16_cutile.shape}, dtype: {C_fp16_cutile.dtype}")
    if args.correctness_check:
        torch.testing.assert_close(C_fp16_cutile, A_fp16 @ B_fp16, atol=1e-4, rtol=1e-3)
        print("Correctness check passed")
    else:
        print("Correctness check disabled")

    # --- Test Case 2: float32 (Single-Precision) ---
    torch.set_float32_matmul_precision("high")
    print("\n--- Test Case 2: Matrix Multiplication with float32 (Single-Precision) ---")
    # Create random input matrices with float32 data type on the CPU.
    A_fp32 = torch.randn(M_dim, K_dim, dtype=torch.float32, device='cpu')
    B_fp32 = torch.randn(K_dim, N_dim, dtype=torch.float32, device='cpu')
    print(f"Input A shape: {A_fp32.shape}, dtype: {A_fp32.dtype}")
    print(f"Input B shape: {B_fp32.shape}, dtype: {B_fp32.dtype}")

    atol, rtol = 2e-4, 1e-3

    # Perform matrix multiplication using the cuTile wrapper function.
    C_fp32_cutile = cutile_matmul(
        A_fp32, B_fp32, num_cpu_threads=args.num_cpu_threads)
    print(f"cuTile Output C shape: {C_fp32_cutile.shape}, dtype: {C_fp32_cutile.dtype}")
    if args.correctness_check:
        torch.testing.assert_close(C_fp32_cutile, A_fp32 @ B_fp32, atol=atol, rtol=rtol)
        print("Correctness check passed")
    else:
        print("Correctness check disabled")

    # --- Test Case 3: Additional Matrix Dimensions ---
    print("\n--- Test Case 3: Matrix Multiplication with Additional Dimensions ---")
    # CPU MMA currently requires dimensions to be multiples of the tile sizes (32, 32, 32).
    M_dim_non_mult = 992
    N_dim_non_mult = 512
    K_dim_non_mult = 704
    A_non_mult = torch.randn(M_dim_non_mult, K_dim_non_mult, dtype=torch.float32, device='cpu')
    B_non_mult = torch.randn(K_dim_non_mult, N_dim_non_mult, dtype=torch.float32, device='cpu')
    print(f"Input A shape: {A_non_mult.shape}, dtype: {A_non_mult.dtype}")
    print(f"Input B shape: {B_non_mult.shape}, dtype: {B_non_mult.dtype}")

    C_non_mult_cutile = cutile_matmul(
        A_non_mult, B_non_mult, num_cpu_threads=args.num_cpu_threads)
    print(f"cuTile Output C shape: {C_non_mult_cutile.shape}, dtype: {C_non_mult_cutile.dtype}")
    if args.correctness_check:
        torch.testing.assert_close(C_non_mult_cutile, A_non_mult @ B_non_mult, atol=atol, rtol=rtol)
        print("Correctness check passed")
    else:
        print("Correctness check disabled")

    # --- Test Case 4: Persistent Matmul ---
    print("\n--- Test Case 4: Matrix Multiplication with Persistent Matmul ---")
    C_persistent_fp32_cutile = cutile_matmul(
        A_fp32, B_fp32, persistent=True,
        num_cpu_threads=args.num_cpu_threads)
    print(f"cuTile Output C shape: {C_persistent_fp32_cutile.shape}, "
          f"dtype: {C_persistent_fp32_cutile.dtype}")
    if args.correctness_check:
        torch.testing.assert_close(C_persistent_fp32_cutile, A_fp32 @ B_fp32, atol=atol, rtol=rtol)
        print("Correctness check passed")
    else:
        print("Correctness check disabled")
    torch.set_float32_matmul_precision("highest")

    print("\n--- All cuTile matrix multiplication examples completed. ---")
