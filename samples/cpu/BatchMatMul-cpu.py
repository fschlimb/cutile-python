# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) <2026> Intel Corporation.
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import sys
from math import ceil
import cuda.tile as ct
import torch

# Kernels and helpers shared with the CUDA sample in the parent directory.
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from BatchMatMul import batch_matmul_kernel  # noqa: E402


ct.set_backend("cpu")


def bmm(a: torch.Tensor, b: torch.Tensor, out_dtype: torch.dtype,
        num_cpu_threads: int = 0) -> torch.Tensor:
    """
    Batch Matrix Multiplication using cuTile's standard tiled kernel.

    Args:
        a (torch.Tensor): Input tensor A with shape (Batch, M, K).
        b (torch.Tensor): Input tensor B with shape (Batch, K, N).

    Returns:
        Output tensor C with shape (Batch, M, N).
    """
    # --- Input Validation ---
    if a.ndim != 3 or b.ndim != 3:
        raise ValueError("Input tensors for BMM must be 3D (Batch, M, K) and (Batch, K, N).")
    if a.shape[0] != b.shape[0]:
        raise ValueError(f"""Batch dimensions must match:
                         A.shape[0]={a.shape[0]}, B.shape[0]={b.shape[0]}.""")
    if a.device != b.device or not a.is_cpu or not b.is_cpu or a.dtype != b.dtype:
        raise ValueError("""Input tensors must be on the CPU
                         and have the same data type.""")

    # Get M, K, N dimensions
    Batch, M, K = a.shape
    _, K_b, N = b.shape
    assert K == K_b, f"Incompatible K dimensions: A's K is {K}, B's K is {K_b}"

    # Create output tensor
    output = torch.empty((Batch, M, N), device=a.device, dtype=out_dtype)

    # --- Determine Tile Shapes for Optimization (Fixed for float16 as per previous request) ---
    tm_val, tn_val, tk_val = 128, 256, 128  # Larger tiles for Tensor Core benefits

    # --- Grid calculation for standard 3D tiled kernel ---
    grid = (Batch, ceil(M / tm_val), ceil(N / tn_val))

    # --- Launch kernel ---
    with ct.backend_options({"num_cpu_threads": num_cpu_threads}):
        ct.launch(None, grid, batch_matmul_kernel,
                  (a, b, output, tm_val, tn_val, tk_val))

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--correctness-check",
        action="store_true",
        help="Check the correctness of the results",
    )
    parser.add_argument("--num-cpu-threads", type=int, default=0)
    args = parser.parse_args()
    print("--- Running cuTile Batched Matrix Multiplication (Standard Tiled) Sample ---")

    # --- User Configuration for BMM Example ---
    BATCH_DIM = 4
    M_DIM = 512
    K_DIM = 256
    N_DIM = 1024

    # --- Test Case 1: Standard BMM (float16) ---
    print("\n--- Test 1: Standard BMM (float16) ---")
    A_fp16 = torch.randn(BATCH_DIM, M_DIM, K_DIM, dtype=torch.float16, device='cpu')
    B_fp16 = torch.randn(BATCH_DIM, K_DIM, N_DIM, dtype=torch.float16, device='cpu')
    print(f"Input A shape: {A_fp16.shape}, dtype: {A_fp16.dtype}")
    print(f"Input B shape: {B_fp16.shape}, dtype: {B_fp16.dtype}")

    C_bmm_cutile_fp16 = bmm(
        A_fp16, B_fp16, A_fp16.dtype, args.num_cpu_threads)
    print(f"""cuTile Standard BMM Output C
            shape:{C_bmm_cutile_fp16.shape},
            dtype: {C_bmm_cutile_fp16.dtype}""")
    if args.correctness_check:
        torch.testing.assert_close(C_bmm_cutile_fp16, A_fp16 @ B_fp16)
        print("Correctness check passed")
    else:
        print("Correctness check disabled")

    # --- Test Case 2: Standard BMM (float8_e4m3fn) ---
    print("\n--- Test 2: Standard BMM (float8_e4m3fn) ---")
    print("skip: CPU does not support float8 matrix multiplication")

    print("\n--- cuTile Batched Matrix Multiplication (Standard Tiled) examples complete ---")
