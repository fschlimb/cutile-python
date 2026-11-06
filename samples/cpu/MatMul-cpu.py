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
from MatMul import matmul_kernel  # noqa: E402


ct.set_backend("cpu")


def cutile_matmul(A: torch.Tensor, B: torch.Tensor,
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

    # BF16 uses AMX-friendly tiles; other dtypes retain their generic tile sizes.
    if A.dtype == torch.bfloat16:
        tm, tn, tk = 32, 32, 32
    elif A.dtype.itemsize == 2:
        tm, tn, tk = 128, 256, 64
    else:
        tm, tn, tk = 32, 32, 32

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
    grid = (grid_size, 1, 1)

    # --- Create Output Tensor C ---
    # The output tensor `C` is initialized with the correct dimensions (M x N),
    # on the same device, and with the same data type as the input matrices.
    C = torch.empty((m, n), device=A.device, dtype=A.dtype)

    # --- Launch the cuTile Kernel ---
    # The `matmul_kernel` is launched with the calculated grid dimensions.
    # `tm`, `tn`, and `tk` are passed as Constant integers to the kernel.
    assume_in_bounds = m % tm == 0 and n % tn == 0 and k_a % tk == 0
    with ct.backend_options({
        "num_cpu_threads": num_cpu_threads,
        "assume_in_bounds": assume_in_bounds,
    }):
        ct.launch(None, grid, matmul_kernel, (A, B, C, tm, tn, tk))

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

    for M_dim, N_dim, K_dim in ((512, 512, 768), (992, 512, 704)):
        A = torch.randn(M_dim, K_dim, dtype=torch.bfloat16, device="cpu")
        B = torch.randn(K_dim, N_dim, dtype=torch.bfloat16, device="cpu")
        C_cutile = cutile_matmul(A, B, num_cpu_threads=args.num_cpu_threads)
        print(f"Input shapes: {A.shape}, {B.shape}")
        print(f"cuTile output shape: {C_cutile.shape}, dtype: {C_cutile.dtype}")
        if args.correctness_check:
            torch.testing.assert_close(C_cutile, A @ B, atol=1e-1, rtol=1e-2)
            print("Correctness check passed")
        else:
            print("Correctness check disabled")
