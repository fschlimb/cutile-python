# SPDX-FileCopyrightText: Copyright (c) <2026> Intel Corporation.
# SPDX-License-Identifier: Apache-2.0

import argparse
from math import ceil
import cuda.tile as ct
import torch
from cuda.tile._backend import cpu
from utils.benchmark import report_benchmark


ct.set_backend("cpu")


ConstInt = ct.Constant[int]


@ct.kernel
def batch_matmul_kernel(A, B, C, tm: ConstInt, tn: ConstInt, tk: ConstInt):
    """CuTile kernel for batch matrix multiplication
    A has shape (Batch, M, K), B has shape (Batch, K, N) and C has shape (Batch, M, N)
    Each block computes one (tm x tn) tile for a specific batch item.
    The grid is 3D: (Batch_idx, M_tile_idx, N_tile_idx).
    """
    pid_batch = ct.bid(0)  # Batch dimension
    pidx = ct.bid(1)  # M dimension
    pidy = ct.bid(2)  # N dimension

    # Calculate number of K tiles
    # A is (Batch, M, K), so K is axis 2
    # Use A.shape[2] for the total K dimension and ct.cdiv for ceiling division
    num_k_tiles = ct.cdiv(A.shape[2], tk)

    # Initialize accumulator
    accumulator = ct.full((tm, tn), 0.0, dtype=ct.float32)
    zero_pad = ct.PaddingMode.ZERO
    # K-dimension loop
    for k in range(num_k_tiles):
        # Load tiles with 3D index and 3D shape
        # A is (Batch, M, K), load (1, tm, tk) tile
        a = ct.load(A, index=(pid_batch, pidx, k), shape=(1, tm, tk), padding_mode=zero_pad)
        a = ct.reshape(a, (tm, tk))  # Reshape to 2D for ct.mma

        # B is (Batch, K, N), load (1, tk, tn) tile
        b = ct.load(B, index=(pid_batch, k, pidy), shape=(1, tk, tn), padding_mode=zero_pad)
        b = ct.reshape(b, (tk, tn))  # Reshape to 2D for ct.mma

        accumulator = ct.mma(a, b, acc=accumulator)

    # Convert to output dtype and store
    result = ct.astype(accumulator, C.dtype)
    # Store with 3D index and 3D shape, C is (Batch, M, N)
    result_3d = ct.reshape(result, (1, tm, tn))
    ct.store(C, index=(pid_batch, pidx, pidy), tile=result_3d)


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
    with cpu.compile_options({"num_cpu_threads": num_cpu_threads, "assume_in_bounds": True}):
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
