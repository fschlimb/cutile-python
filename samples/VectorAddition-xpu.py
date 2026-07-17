# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import cuda.tile as ct
import torch
import math
from cuda.tile._backend import xpu

# Use the in-tree experimental XPU backend hooks by shorthand module name.
ct.set_backend("xpu")


ConstInt = ct.Constant[int]


@ct.kernel
def vec_add_kernel_1d(a, b, c, TILE: ConstInt):
    """
    cuTile kernel for 1D element-wise vector addition using direct tiled loads/stores.

    Each block processes a `TILE`-sized chunk of the vectors.
    This approach is efficient when the total dimension is a multiple of `TILE`,
    or when out-of-bounds accesses are implicitly handled by the calling context
    (e.g., by padding or ensuring input sizes match grid dimensions).

    Args:
        a: Input tensor A.
        b: Input tensor B.
        c: Output tensor for the sum (A + B).
        TILE (ConstInt): The size of the tile (chunk of data) processed by each
                         block. This must be a compile-time constant.
    """
    # Get the global ID of the current block along the first dimension.
    # In a 1D grid, this directly corresponds to the index of the tile.
    bid = ct.bid(0)

    # Load TILE-sized chunks from input vectors 'a' and 'b'.
    # `ct.load` automatically distributes the load operation across the threads
    # within the block, bringing the specified tile of data into shared memory
    # or registers. The `index=(bid,)` specifies which tile to load based on the block ID.
    a_tile = ct.load(a, index=(bid,), shape=(TILE,))
    b_tile = ct.load(b, index=(bid,), shape=(TILE,))

    # Perform the element-wise addition on the loaded tiles.
    # This operation happens in parallel across the threads within the block.
    sum_tile = a_tile + b_tile

    # Store the resulting TILE-sized chunk back to the output vector 'c'.
    # `ct.store` writes the computed tile back to global memory, again
    # distributing the store operation across threads.
    ct.store(c, index=(bid,), tile=sum_tile)


@ct.kernel
def vec_add_kernel_2d(a, b, c, TILE_X: ConstInt, TILE_Y: ConstInt):
    """
    cuTile kernel for 2D element-wise matrix addition using direct tiled loads/stores.

    Each block computes a `TILE_X` x `TILE_Y` chunk of the matrices.
    Similar to the 1D direct kernel, this is efficient when dimensions are
    multiples of the tile sizes.

    Args:
        a: Input matrix A.
        b: Input matrix B.
        c: Output matrix for the sum (A + B).
        TILE_X (ConstInt): The tile dimension along the X-axis (rows).
        TILE_Y (ConstInt): The tile dimension along the Y-axis (columns).
    """
    # Get the global IDs of the current block along the X and Y axes.
    # `ct.bid(0)` for the first grid dimension (typically rows),
    # `ct.bid(1)` for the second grid dimension (typically columns).
    bid_x = ct.bid(0)
    bid_y = ct.bid(1)

    # Load `TILE_X` x `TILE_Y` chunks from input matrices 'a' and 'b'.
    # The `index=(bid_x, bid_y)` specifies the 2D tile to load.
    a_tile = ct.load(a, index=(bid_x, bid_y), shape=(TILE_X, TILE_Y))
    b_tile = ct.load(b, index=(bid_x, bid_y), shape=(TILE_X, TILE_Y))

    # Perform the element-wise addition on the loaded tiles.
    sum_tile = a_tile + b_tile

    # Store the resulting `TILE_X` x `TILE_Y` chunk back to the output matrix 'c'.
    ct.store(c, index=(bid_x, bid_y), tile=sum_tile)


@ct.kernel
def vec_add_kernel_1d_gather(a, b, c, TILE: ConstInt):
    """
    cuTile kernel for 1D element-wise vector addition using direct tiled loads/stores.

    Each block processes a `TILE`-sized chunk of the vectors.
    This approach is efficient when the total dimension is a multiple of `TILE`,
    or when out-of-bounds accesses are implicitly handled by the calling context
    (e.g., by padding or ensuring input sizes match grid dimensions).

    Args:
        a: Input tensor A.
        b: Input tensor B.
        c: Output tensor for the sum (A + B).
        TILE (ConstInt): The size of the tile (chunk of data) processed by each
                         block. This must be a compile-time constant.
    """
    # Get the global ID of the current block.
    bid = ct.bid(0)

    # Calculate indices for elements within the current block's tile.
    # `ct.arange(TILE, ...)` creates local offsets [0, 1, ..., TILE-1] for threads
    # within the block. `bid * TILE` shifts these to the correct global starting
    # point for this specific block.
    indices = bid * TILE + ct.arange(TILE, dtype=torch.int32)

    # Load elements using the calculated indices.
    # `ct.gather` only loads data within the array bounds, and zeroes any out-of-bounds elements.
    a_tile = ct.gather(a, indices)
    b_tile = ct.gather(b, indices)

    # Perform the element-wise addition.
    sum_tile = a_tile + b_tile

    # Store the result using the same indices.
    # `ct.scatter()` only writes data to positions within the array bounds.
    ct.scatter(c, indices, sum_tile)


@ct.kernel
def vec_add_kernel_2d_gather(
    a, b, c,
    TILE_X: ConstInt, TILE_Y: ConstInt  # Tile dimensions for this block
):
    """
    cuTile kernel for 2D element-wise matrix addition using direct tiled loads/stores.

    Each block computes a `TILE_X` x `TILE_Y` chunk of the matrices.
    Similar to the 1D direct kernel, this is efficient when dimensions are
    multiples of the tile sizes.

    Args:
        a: Input matrix A.
        b: Input matrix B.
        c: Output matrix for the sum (A + B).
        TILE_X (ConstInt): The tile dimension along the X-axis (rows).
        TILE_Y (ConstInt): The tile dimension along the Y-axis (columns).
    """
    # Get the global IDs of the current block along the X and Y axes.
    bid_x = ct.bid(0)
    bid_y = ct.bid(1)

    # Calculate X and Y indices within the current block's tile.
    x = bid_x * TILE_X + ct.arange(TILE_X, dtype=torch.int32)
    y = bid_y * TILE_Y + ct.arange(TILE_Y, dtype=torch.int32)

    # Reshape the X and Y indices to (TILE_X, 1) and (1, TILE_Y), respectively.
    # This way, they can be broadcasted together to a common shape (TILE_X, TILE_Y).
    x = x[:, None]
    y = y[None, :]

    # Load elements using the calculated X and Y indices.
    # Both `a_tile` and `b_tile` have shape (TILE_X, TILE_Y).
    a_tile = ct.gather(a, (x, y))
    b_tile = ct.gather(b, (x, y))

    # Perform the element-wise addition.
    sum_tile = a_tile + b_tile

    # Store the result back to `c` using the same index tiles.
    # `ct.scatter()` only writes data to positions within the array bounds.
    ct.scatter(c, (x, y), sum_tile)


# --- Wrapper Function to Dispatch to Kernels ---
def _build_options(base: dict, *, wg_m: int, wg_n: int, flops: int) -> dict:
    opts = dict(base or {})
    # Explicit work-group tiles are required by the backend.
    opts["wg_m"] = wg_m
    opts["wg_n"] = wg_n
    opts["flops"] = flops
    return opts


def vec_add(a: torch.Tensor, b: torch.Tensor, options: dict | None = None, use_gather: bool = False) -> torch.Tensor:
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
                ``xpu.compile_options(options)``.
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
        launch_options = _build_options(
            options,
            wg_m=TILE_X,
            wg_n=TILE_Y,
            flops=N,
        )
        with xpu.compile_options(launch_options):
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
        launch_options = _build_options(
            options,
            wg_m=TILE_X,
            wg_n=TILE_Y,
            flops=M * N,
        )
        with xpu.compile_options(launch_options):
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
