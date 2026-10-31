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
from utils.benchmark import report_benchmark

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
    tm_val, tn_val, tk_val = 32, 32, 32  # Larger tiles for Tensor Core benefits

    # --- Grid calculation for standard 3D tiled kernel ---
    grid = (Batch, ceil(M / tm_val), ceil(N / tn_val))

    # --- Launch kernel ---
    with ct.backend_options({"num_cpu_threads": num_cpu_threads, "assume_in_bounds": True}):
        ct.launch(None, grid, batch_matmul_kernel,
                  (a, b, output, tm_val, tn_val, tk_val))

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-cpu-threads", type=int, default=0)
    args = parser.parse_args()

    BATCH_DIM = 4
    M_DIM = 4096
    K_DIM = 4096
    N_DIM = 4096

    A_fp16 = torch.randn(BATCH_DIM, M_DIM, K_DIM, dtype=torch.bfloat16, device='cpu')
    B_fp16 = torch.randn(BATCH_DIM, K_DIM, N_DIM, dtype=torch.bfloat16, device='cpu')

    output = bmm(A_fp16, B_fp16, torch.float32, args.num_cpu_threads)
    torch.testing.assert_close(
        output, torch.bmm(A_fp16, B_fp16).to(torch.float32),
        atol=1e-2, rtol=1e-2,
    )
    print("Correctness check passed")

    stats_cutile = report_benchmark(
        bmm, (A_fp16, B_fp16, torch.float32, args.num_cpu_threads))
    stats_torch = report_benchmark(torch.bmm, (A_fp16, B_fp16))
    print("Benchmark results:")
    print(f"  cuTile Batch MatMul: {stats_cutile['mean_time_ms']:.5f} ms")
    print(f"  torch Batch MatMul: {stats_torch['mean_time_ms']:.5f} ms")
    speedup = stats_torch["mean_time_ms"] / stats_cutile["mean_time_ms"]
    print(f"Speedup: {speedup:.3f}x")
