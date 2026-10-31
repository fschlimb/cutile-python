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
from Transpose import transpose_kernel  # noqa: E402

# Use the in-tree experimental XPU backend hooks by shorthand module name.
ct.set_backend("xpu")


def cutile_transpose(x: torch.Tensor) -> torch.Tensor:
    """
    Performs matrix transposition C = X.T using a cuTile kernel.

    Args:
        x (torch.Tensor): The input matrix (M x N). This tensor must be 2D and
                           on an XPU device.

    Returns:
        torch.Tensor: The transposed matrix (N x M) on the same XPU device.

    Raises:
        ValueError: If the input tensor is not on XPU, is not 2D, or its shape
                    does not meet the XPU direct-tile alignment requirements.
    """
    if not x.is_xpu:
        raise ValueError("Input tensor must be on an XPU device.")
    if x.ndim != 2:
        raise ValueError("Transpose kernel currently supports only 2D tensors.")

    m, n = x.shape
    tm, tn = 64, 16
    if m % tm != 0 or n % tn != 0:
        raise ValueError(f"XPU transpose requires M and N divisible by {tm} and {tn}; "
                         f"got M={m}, N={n}.")

    grid = (ceil(m / tm), ceil(n / tn), 1)
    y = torch.empty((n, m), device=x.device, dtype=x.dtype)
    options = {
        "block_threads": (1, 32, 16),
        "assume_in_bounds": True,
    }
    with ct.backend_options(options):
        ct.launch(torch.xpu.current_stream(), grid, transpose_kernel, (x, y, tm, tn))

    return y


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--correctness-check",
        action="store_true",
        help="Check the correctness of the results",
    )
    args = parser.parse_args()

    if not torch.xpu.is_available():
        raise RuntimeError(
            "XPU is not available. Ensure Intel GPU runtime and PyTorch XPU are installed."
        )

    print("--- Running cuTile Matrix Transposition Examples (XPU) ---")

    # Define common matrix dimensions for the examples.
    M_dim = 1024
    N_dim = 512

    # --- Test Case 1: float16 (Half-Precision) ---
    print("\n--- Test Case 1: Matrix Transposition with float16 (Half-Precision) ---")
    x_fp16 = torch.randn(M_dim, N_dim, dtype=torch.float16, device="xpu")
    print(f"Input x shape: {x_fp16.shape}, dtype: {x_fp16.dtype}")

    y_fp16_cutile = cutile_transpose(x_fp16)
    print(f"cuTile Output y shape: {y_fp16_cutile.shape}, dtype: {y_fp16_cutile.dtype}")
    if args.correctness_check:
        torch.testing.assert_close(y_fp16_cutile, x_fp16.T)
        print("Correctness check passed")
    else:
        print("Correctness check disabled")

    # --- Test Case 2: float32 (Single-Precision) ---
    print("\n--- Test Case 2: Matrix Transposition with float32 (Single-Precision) ---")
    x_fp32 = torch.randn(M_dim, N_dim, dtype=torch.float32, device="xpu")
    print(f"Input x shape: {x_fp32.shape}, dtype: {x_fp32.dtype}")

    y_fp32_cutile = cutile_transpose(x_fp32)
    print(f"cuTile Output y shape: {y_fp32_cutile.shape}, dtype: {y_fp32_cutile.dtype}")
    if args.correctness_check:
        torch.testing.assert_close(y_fp32_cutile, x_fp32.T)
        print("Correctness check passed")
    else:
        print("Correctness check disabled")

    # --- Test Case 3: Non-square Matrix ---
    print("\n--- Test Case 3: Matrix Transposition with Non-Square Dimensions ---")
    M_dim_non_square = 960
    N_dim_non_square = 496
    x_non_square = torch.randn(M_dim_non_square, N_dim_non_square,
                               dtype=torch.float32, device="xpu")
    print(f"Input x shape: {x_non_square.shape}, dtype: {x_non_square.dtype}")

    y_non_square_cutile = cutile_transpose(x_non_square)
    print(f"cuTile Output y shape: {y_non_square_cutile.shape}, "
          f"dtype: {y_non_square_cutile.dtype}")
    if args.correctness_check:
        torch.testing.assert_close(y_non_square_cutile, x_non_square.T)
        print("Correctness check passed")
    else:
        print("Correctness check disabled")

    print("\n--- All cuTile matrix transposition examples completed. ---")
