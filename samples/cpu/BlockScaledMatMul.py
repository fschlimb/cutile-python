# SPDX-FileCopyrightText: Copyright (c) <2026> Intel Corporation.
# SPDX-License-Identifier: Apache-2.0

import argparse
import cuda.tile as ct
import torch
from cuda.tile._backend import cpu


ct.set_backend("cpu")


def block_quantize(x: torch.Tensor, block_size: int, dtype: torch.dtype = torch.float16):

    """
    Args:
        x (torch.Tensor):       input tensor.
        block_size (int):       size of block.
        dtype (torch.dtype):    datatype used to carry quantized values and scales on CPU.

    Returns:
        tuple[torch.Tensor, torch.Tensor, int]: A tuple containing:
            - x (torch.Tensor):         in dtype.
            - scale (torch.Tensor):     in dtype.

    Raises:
        ValueError: If the requested block size is larger than the inner most
                    dimension of the input tensor.
    """

    dtype_max = torch.finfo(torch.float8_e4m3fn).max

    if x.shape[-1] < block_size:
        raise ValueError(f'The requested block size {block_size} is larger than the inner most '
                         f'dimension of the input tensor {x.shape[-1]}')

    if x.shape[-1] % block_size != 0:
        print('[WARNING] block size is not a multiple of the inner most '
              'dimension of the input tensor')
        print('          padding the input tensor...')
        pad_len = (block_size - (x.shape[-1] % block_size)) % block_size
        x = torch.nn.functional.pad(x, (0, pad_len), value=0)

    x_block = x.reshape(*x.shape[:-1], x.shape[-1] // block_size, block_size)

    scale = torch.max(x_block.abs(), dim=-1, keepdims=True)[0]
    scale = torch.clamp(scale / dtype_max, min=1e-12)
    scale = torch.pow(2.0, torch.ceil(torch.log2(scale)))

    x_q = x_block / scale

    x_q = x_q.to(dtype).reshape(x.shape)
    scale = scale.to(dtype).squeeze(-1)

    return x_q, scale


def swizzle_32_4_4(scale):
    '''
    Prepare the original scale tensor to align with the expected tmem layout.
    With the innermost dimensions being (m1=32, m2=4, k1=4), and the outer dimensions
    being (m0=(M // m1 * m2), k0=(K_s // k1)).

    Reference: PTX ISA tcgen05.mma scale factor layout.
    https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-scale-factor-a-layout-1x
    '''
    m1, m2, k1 = 32, 4, 4

    M, K_s = scale.shape
    m0 = M // (m1 * m2)
    k0 = K_s // k1
    scale = scale.reshape(m0, m2, m1, k0, k1).permute(0, 3, 2, 1, 4).contiguous()
    return scale.reshape(m0, k0, 32, 16)


ConstInt = ct.Constant[int]


def swizzle_2d_from_bid(M, N, tm, tn, GROUP_SIZE_M, bid):
    # Get the global IDs of a given block in a 1D grid.
    num_bid_m = ct.cdiv(M, tm)
    num_bid_n = ct.cdiv(N, tn)
    num_bid_in_group = GROUP_SIZE_M * num_bid_n
    group_id = bid // num_bid_in_group
    first_bid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_bid_m - first_bid_m, GROUP_SIZE_M)
    bid_m = first_bid_m + (bid % group_size_m)
    bid_n = (bid % num_bid_in_group) // group_size_m
    return bid_m, bid_n


def swizzle_2d(M, N, tm, tn, GROUP_SIZE_M):
    # Get the global IDs of the current block in a 1D grid.
    bid = ct.bid(0)
    return swizzle_2d_from_bid(M, N, tm, tn, GROUP_SIZE_M, bid)


def unswizzle_32_4_4(tile_swizzled_scale):
    '''
    Kernel-side inverse of ``swizzle_32_4_4``: take a tile loaded
    from the host swizzled scale tensor and recover the ``(M, K_s)``
    logical scale view used by the block-scaled matmul.
    '''
    m1, m2, k1 = 32, 4, 4
    m0, k0, _, _ = tile_swizzled_scale.shape

    return (tile_swizzled_scale.reshape((m0, k0, m1, m2, k1))
                               .permute((0, 3, 2, 1, 4))
                               .reshape((m0 * m2 * m1, k0 * k1)))


@ct.kernel
def block_scaled_matmul_kernel(
                    A, A_scale, B, B_scale, C,
                    tm: ConstInt,         # Tile size along M dimension (rows of C)
                    tn: ConstInt,         # Tile size along N dimension (columns of C)
                    tk: ConstInt,        # Tile size along K dimension (inner product dimension)
                    scaling_block_size: ConstInt):

    """
    cuTile kernel for block-scaled matrix multiplication.

    Computes C = (A * A_scale) @ (B * B_scale), accumulating in float32.
    Each TileBlock computes one tm x tn output tile. The K dimension is processed
    in chunks of tk, with tks scale values per K tile.

    If packed swizzle scales are passed, they get unswizzled into logical 2D scale tiles.

    Args:
        A:              Input matrix A (M x K).
        A_scale:        2D scale of (M, K // scaling_block_size) or swizzle scale of
                        (M // 32 // 4, K // scaling_block_size // 4, 32, 4, 4) reshaped
                        into (M // 32 // 4, K // scaling_block_size // 4, 32, 16).
        B:              Input matrix B (K x N).
        B_scale:        2D scale of (M, K // scaling_block_size) or swizzled scale of
                        (M // 32 // 4, K // scaling_block_size // 4, 32, 4, 4) reshaped
                        into (M // 32 // 4, K // scaling_block_size // 4, 32, 16).
        C:              Output matrix C (M x N).
        tm (ConstInt):  The height of the output tile computed by this block.
                        Corresponds to rows of A and C.
        tn (ConstInt):  The width of the output tile computed by this block.
                        Corresponds to columns of B and C.
        tk (ConstInt):  The depth of the inner loop (K-dimension) tile size.
                        Corresponds to columns of A and rows of B.
        scaling_block_size (ConstInt): the scaling block size.
    """
    GROUP_SIZE_M = 8
    M = A.shape[0]
    N = B.shape[1]
    bidx, bidy = swizzle_2d(M, N, tm, tn, GROUP_SIZE_M)

    tks = tk // scaling_block_size

    # Calculate the total number of tiles along the K-dimension that need to be processed.
    # `ct.num_tiles(A, axis=1, shape=(tm, tk))` means:
    #   "View A as an MxK tensor tiled by (tm, tk), and return the number of tiles along
    #    axis 1 (the K dimension)."
    # We pass shape=(tm, tk) to describe the 2D tiling, only `tk` matters for axis=1.
    num_tiles_k = ct.num_tiles(A, axis=1, shape=(tm, tk))

    # Initialize an accumulator for the current output tile (tm x tn).
    # It's common practice to use `float32` for accumulation even with `float16` inputs
    # to maintain higher precision during the sum-reduction of the matrix multiplication.
    accumulator = ct.full((tm, tn), 0, dtype=ct.float32)
    zero_pad = ct.PaddingMode.ZERO

    # K-dimension loop: Iterate over the K-dimension in chunks of 'tk'.
    # In each iteration, a `tm` x `tk` tile from A and a `tk` x `tn` tile from B
    # are loaded, multiplied, and accumulated.
    for k in range(num_tiles_k):
        # Load tile from matrix A.
        # The `index=(bidx, k_tile_idx)` specifies which (M-tile, K-tile) to load
        # from global memory A. `shape=(tm, tk)` defines the size of this tile.
        a = ct.load(A, index=(bidx, k), shape=(tm, tk), padding_mode=zero_pad)

        if len(A_scale.shape) == 2:
            # 2D scale path. A_scale is already stored in logical shape (M, K_s).
            a_scale = ct.load(A_scale, index=(bidx, k), shape=(tm, tks), padding_mode=zero_pad)
        else:
            # Load the packed scale tile, unswizzle it, and reshape to
            # the logical block-scaled matmul shape (tm, tks).

            # unswizzle
            a_scale_swizzled = ct.load(A_scale, index=(bidx, k, 0, 0),
                                       shape=(tm // scaling_block_size // 4, tks // 4, 32, 16),
                                       padding_mode=zero_pad)
            a_scale = unswizzle_32_4_4(a_scale_swizzled)

        # Load tile from matrix B.
        # The `index=(k_tile_idx, bidy)` specifies which (K-tile, N-tile) to load
        # from global memory B. `shape=(tk, tn)` defines the size of this tile.
        b = ct.load(B, index=(k, bidy), shape=(tk, tn), padding_mode=zero_pad)

        if len(B_scale.shape) == 2:
            b_scale = ct.load(B_scale, index=(k, bidy), shape=(tks, tn), padding_mode=zero_pad)
        else:
            # B scales are stored N-major. Unswizzle it, reshape it to
            # (tn, tks), then transpose it to the logical block-scaled shape (tks, tn).

            # unswizzle
            b_scale_swizzled = ct.load(B_scale, index=(bidy, k, 0, 0),
                                       shape=(tn // scaling_block_size // 4, tks // 4, 32, 16),
                                       padding_mode=zero_pad)
            b_scale = unswizzle_32_4_4(b_scale_swizzled).permute((1, 0))

        # Expand one scale factor per K block and apply it before ordinary MMA.
        a_scale = ct.broadcast_to(a_scale.reshape((tm, tks, 1)),
                      (tm, tks, scaling_block_size)).reshape((tm, tk))
        b_scale = ct.broadcast_to(b_scale.reshape((tks, 1, tn)),
                      (tks, scaling_block_size, tn)).reshape((tk, tn))
        accumulator = ct.mma(a * a_scale, b * b_scale, accumulator)

    # Store the computed tile to the global memory of the output matrix C.
    # The `(bidx, bidy)` directly corresponds to the tile's position in the 2D output matrix.
    ct.store(C, index=(bidx, bidy), tile=accumulator)


def cutile_block_scaled_matmul(A: torch.Tensor, A_scale: torch.Tensor,
                               B: torch.Tensor, B_scale: torch.Tensor,
                               num_cpu_threads: int = 0) -> torch.Tensor:

    """
    Performs block-scaled matrix multiplication using a cuTile kernel.

    This wrapper function handles input validation, determines appropriate
    tile sizes based on data type, calculates the necessary grid dimensions,
    and launches the `block_scaled_matmul_kernel`.

    Args:
        A (torch.Tensor):       The first input matrix (M x K). Must be on the CPU.
        B (torch.Tensor):       The second input matrix (K x N). Must be on the CPU
                                and have its K dimension match A's K dimension.
        A_scale (torch.Tensor): Either 2D scale with shape (M, K // scaling_block_size) or
                                swizzled scale of (M // scaling_block_size // 4,
                                K // scaling_block_size // 4, 32, 16).
        B_scale (torch.Tensor): Either 2D scale with shape (M, K // scaling_block_size) or
                                swizzled scale of (M // scaling_block_size // 4,
                                K // scaling_block_size // 4, 32, 16).

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
    if A.device != B.device or A.device != A_scale.device or A.device != B_scale.device:
        raise ValueError("Input tensors must be on the same device.")
    if not A.is_cpu or not A_scale.is_cpu or not B.is_cpu or not B_scale.is_cpu:
        raise ValueError("Input tensors must be on the CPU.")
    # Note: cuTile handles dtype compatibility within the kernel,
    # but inputs should generally match.

    tm, tn, tk, scaling_block_size = 256, 256, 128, 32

    # --- Get Matrix Dimensions ---
    m, _ = A.shape
    _, n = B.shape

    # --- Calculate Grid Dimensions for Kernel Launch (1D Grid) ---
    # The grid defines how many blocks will be launched.
    # Each block computes one (tm x tn) output tile of matrix C.
    # `ct.cdiv(total_dim, tile_dim)` ensures enough blocks are launched to cover
    # the entire matrix, even if dimensions are not perfect multiples of tile sizes.
    grid_x = ct.cdiv(m, tm)  # Number of blocks needed along the M dimension (rows of C)
    grid_y = ct.cdiv(n, tn)  # Number of blocks needed along the N dimension (columns of C)
    grid_size = grid_x * grid_y

    grid = (grid_size, 1, 1)

    # --- Create Output Tensor C ---
    # The output tensor `C` is initialized with the correct dimensions (M x N),
    # on the same device, and with the same data type as the input matrices.
    C = torch.empty((m, n), device=A.device, dtype=torch.float32)

    # --- Launch the cuTile Kernel ---
    # The `block_scaled_matmul_kernel` is launched with the calculated grid dimensions.
    # `tm`, `tn`, and `tk` are passed as Constant integers to the kernel.
    kernel = block_scaled_matmul_kernel
    with cpu.compile_options({"num_cpu_threads": num_cpu_threads}):
        ct.launch(None, grid, kernel, (
            A, A_scale, B, B_scale, C, tm, tn, tk, scaling_block_size))

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

    # --- Running cuTile Block Scaled Matrix Multiplication Examples ---
    print("--- Running cuTile Block Scaled Matrix Multiplication Examples ---")

    # Define common matrix dimensions for the examples
    M_dim = 512
    N_dim = 512
    K_dim = 768

    scaling_block_size = 32
    KS_dim = K_dim // scaling_block_size

    print("\n--- Test Case: Block Scaled Matrix Multiplication with M = 512, N = 512, "
          "K = 768, Scaling Block Size = 32 ---")

    A = torch.rand((M_dim, K_dim), device='cpu')
    B = torch.rand((N_dim, K_dim), device='cpu')

    A, A_scale = block_quantize(A, scaling_block_size)
    B, B_scale = block_quantize(B, scaling_block_size)

    B = B.T
    B_scale = B_scale.T

    k = A.shape[-1]
    ks = k // scaling_block_size

    A_s_swizzled = swizzle_32_4_4(A_scale)
    B_s_swizzled = swizzle_32_4_4(B_scale.T.contiguous())

    print(f"Input A shape: {A.shape}, dtype: {A.dtype}")
    print(f"Input B shape: {B.shape}, dtype: {B.dtype}")

    atol, rtol = 1e-4, 1e-3

    # Perform matrix multiplication using the cuTile wrapper function.
    C_cutile = cutile_block_scaled_matmul(
        A, A_scale, B, B_scale, args.num_cpu_threads)
    C_cutile_swizzled = cutile_block_scaled_matmul(
        A, A_s_swizzled, B, B_s_swizzled, args.num_cpu_threads)
    print(f"cuTile Output C shape: {C_cutile.shape}, dtype: {C_cutile.dtype}")

    if args.correctness_check:
        ref_A_scale = torch.repeat_interleave(A_scale, scaling_block_size, dim=1).to(torch.float32)
        ref_B_scale = torch.repeat_interleave(B_scale, scaling_block_size, dim=0).to(torch.float32)
        ref = (A.to(torch.float32) * ref_A_scale) @ (B.to(torch.float32) * ref_B_scale)

        torch.testing.assert_close(C_cutile, ref, atol=atol, rtol=rtol)
        torch.testing.assert_close(C_cutile_swizzled, ref, atol=atol, rtol=rtol)
        print("Correctness check passed")
    else:
        print("Correctness check disabled")

    print("\n--- cuTile block scaled matrix multiplication example completed. ---")
