# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import argparse
from math import ceil

import cuda.tile as ct
import torch
from cuda.tile._backend import xpu

# Use the in-tree experimental XPU backend hooks by shorthand module name.
ct.set_backend("xpu")


ConstInt = ct.Constant[int]


@ct.kernel
def transpose_kernel(x, y,
                     tm: ConstInt,  # Tile size along M dimension (rows of original x)
                     tn: ConstInt):  # Tile size along N dimension (columns of original x)
    """
    cuTile kernel to transpose a 2D matrix by processing data in tiles.

    Each block is responsible for computing a `tn` x `tm` tile
    of the output (transposed) matrix `y`. This involves loading a `tm` x `tn`
    tile from the input matrix `x`, transposing it locally, and then storing
    the `tn` x `tm` result to the correct location in `y`.

    Args:
        x: Input matrix (M x N).
        y: Output matrix (N x M), which will be the transpose of x.
        tm (ConstInt): The height of the input tile (number of rows from x)
                       processed by this block.
        tn (ConstInt): The width of the input tile (number of columns from x)
                       processed by this block.
    """
    # Get the global IDs of the current block in a 2D grid.
    # `ct.bid(0)` gives the block ID along the X-axis of the grid, which corresponds
    # to the M-tile index (rows) of the original input matrix `x`.
    # `ct.bid(1)` gives the block ID along the Y-axis of the grid, which corresponds
    # to the N-tile index (columns) of the original input matrix `x`.
    bidx = ct.bid(0)
    bidy = ct.bid(1)

    # Load a tile from the input matrix 'x'.
    # `ct.load` reads a `tm` x `tn` chunk of data from global memory `x`
    # at the specified `index=(bidx, bidy)`.
    input_tile = ct.load(x, index=(bidx, bidy), shape=(tm, tn))

    # Transpose the loaded tile.
    # `ct.transpose` without explicit axes defaults to swapping the last two dimensions.
    # For a 2D tile of shape (tm, tn), this operation transforms it into a (tn, tm) tile.
    transposed_tile = ct.transpose(input_tile)

    # Store the transposed tile to the output matrix 'y'.
    # The store index is swapped (`bidy`, `bidx`) because `y` is the transpose of `x`.
    ct.store(y, index=(bidy, bidx), tile=transposed_tile)


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
        "wg_m": tm,
        "wg_n": tn,
        "block_threads": (1, 32, 16),
        "assume_in_bounds": True,
    }
    with xpu.compile_options(options):
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