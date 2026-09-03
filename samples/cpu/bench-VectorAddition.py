# SPDX-FileCopyrightText: Copyright (c) <2026> Intel Corporation.
# SPDX-License-Identifier: Apache-2.0

import argparse
import cuda.tile as ct
import torch
import math
from cuda.tile._backend import cpu
from utils.benchmark import report_benchmark


ct.set_backend("cpu")


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
    else:
        m, n = a.shape
        tile_y = min(1024, 2 ** math.ceil(math.log2(n))) if n > 0 else 1
        tile_x = max(1, 1024 // tile_y)
        grid = (math.ceil(m / tile_x), math.ceil(n / tile_y), 1)
        kernel = vec_add_kernel_2d_gather if use_gather else vec_add_kernel_2d
        kernel_args = (a, b, c, tile_x, tile_y)

    with cpu.compile_options({"num_cpu_threads": num_cpu_threads, "assume_in_bounds": True}):
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
