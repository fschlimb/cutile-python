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
from utils.benchmark import report_benchmark

# Kernels and helpers shared with the CUDA sample in the parent directory.
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from VectorAddition import (  # noqa: E402
    vec_add_kernel_1d,
    vec_add_kernel_2d,
    vec_add_kernel_1d_gather,
    vec_add_kernel_2d_gather,
)


ct.set_backend("cpu")


# --- Wrapper Function to Dispatch to Kernels ---
def vec_add(a: torch.Tensor, b: torch.Tensor, use_gather: bool = False,
            num_cpu_threads: int = 0) -> torch.Tensor:
    """
    Performs element-wise addition using a cuTile kernel selected by input
    dimensionality and the gather/scatter option.

    Args:
        a (torch.Tensor): First input tensor, 1D or 2D and on the CPU.
        b (torch.Tensor): Second input tensor with the same shape, device, and dtype.
        use_gather (bool): Select the gather/scatter kernel variant.
        num_cpu_threads (int): Number of CPU threads used to run the kernel.

    Returns:
        torch.Tensor: The element-wise sum of `a` and `b`.
    """
    if a.shape != b.shape:
        raise ValueError("Input tensors must have the same shape.")
    if a.dim() > 2 or b.dim() > 2:
        raise ValueError("This function currently supports only 1D or 2D tensors.")
    if a.device != b.device:
        raise ValueError("Input tensors must be on the same device.")
    if not a.is_cpu or not b.is_cpu:
        raise ValueError("Input tensors must be on the CPU.")
    if a.dtype != b.dtype:
        raise ValueError("Input tensors must have the same data type.")

    c = torch.empty_like(a)
    if a.dim() == 1:
        n = a.shape[0]
        tile = min(1024, 2 ** math.ceil(math.log2(n))) if n > 0 else 1
        grid = (math.ceil(n / tile), 1, 1)
        kernel = vec_add_kernel_1d_gather if use_gather else vec_add_kernel_1d
        kernel_args = (a, b, c, tile)
        in_bounds = n % tile == 0
    else:
        m, n = a.shape
        tile_y = min(1024, 2 ** math.ceil(math.log2(n))) if n > 0 else 1
        tile_x = max(1, 1024 // tile_y)
        grid = (math.ceil(m / tile_x), math.ceil(n / tile_y), 1)
        kernel = vec_add_kernel_2d_gather if use_gather else vec_add_kernel_2d
        kernel_args = (a, b, c, tile_x, tile_y)
        in_bounds = m % tile_x == 0 and n % tile_y == 0

    # Partial edge tiles rely on bounds checks, so only full tilings may skip them.
    options = {"num_cpu_threads": num_cpu_threads, "assume_in_bounds": in_bounds}
    with ct.backend_options(options):
        ct.launch(None, grid, kernel, kernel_args)
    return c


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-cpu-threads", type=int, default=0)
    args = parser.parse_args()

    vector_size = 1_000_000
    matrix_shape = (2048, 1024)
    gather_vector_size = 1_000_001
    gather_matrix_shape = (2000, 1000)

    a_1d_direct = torch.randn(vector_size, dtype=torch.bfloat16, device="cpu")
    b_1d_direct = torch.randn(vector_size, dtype=torch.bfloat16, device="cpu")
    a_1d_gather = torch.randn(gather_vector_size, dtype=torch.bfloat16, device="cpu")
    b_1d_gather = torch.randn(gather_vector_size, dtype=torch.bfloat16, device="cpu")
    a_2d_direct = torch.randn(matrix_shape, dtype=torch.bfloat16, device="cpu")
    b_2d_direct = torch.randn(matrix_shape, dtype=torch.bfloat16, device="cpu")
    a_2d_gather = torch.randn(gather_matrix_shape, dtype=torch.bfloat16, device="cpu")
    b_2d_gather = torch.randn(gather_matrix_shape, dtype=torch.bfloat16, device="cpu")

    benchmarks = (
        ("1D direct", a_1d_direct, b_1d_direct, False),
        ("1D gather", a_1d_gather, b_1d_gather, True),
        ("2D direct", a_2d_direct, b_2d_direct, False),
        ("2D gather", a_2d_gather, b_2d_gather, True),
    )

    for _, a, b, use_gather in benchmarks:
        output = vec_add(a, b, use_gather, args.num_cpu_threads)
        torch.testing.assert_close(output, torch.add(a, b))
    print("Correctness check passed")

    print("Benchmark results:")
    for name, a, b, use_gather in benchmarks:
        stats_cutile = report_benchmark(vec_add, (a, b, use_gather, args.num_cpu_threads))
        stats_torch = report_benchmark(torch.add, (a, b))
        cutile_time = stats_cutile["mean_time_ms"]
        torch_time = stats_torch["mean_time_ms"]
        print(f"  {name} cuTile: {cutile_time:.5f} ms")
        print(f"  {name} torch: {torch_time:.5f} ms")
        print(f"  {name} speedup: {torch_time / cutile_time:.3f}x")
