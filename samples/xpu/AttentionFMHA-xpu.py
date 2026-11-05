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

from torch.nn.functional import scaled_dot_product_attention

# Kernels and helpers shared with the CUDA sample in the parent directory.
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from AttentionFMHA import fmha_kernel  # noqa: E402

# Use the in-tree experimental XPU backend hooks by shorthand module name.
ct.set_backend("xpu")


# --- Wrapper function to launch the FMHA kernel ---
def cutile_fmha(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                qk_scale: float | None = None,
                input_pos: int = 0,
                tile_m: int = 128,
                tile_n: int = 128,
                query_group_size: int = 1,
                causal: bool = False,
                options: dict | None = None,
                kernel=fmha_kernel) -> torch.Tensor:
    """
    Performs Fused Multi-Head Attention (FMHA) using a cuTile kernel.

    Args:
        Q (torch.Tensor): Query tensor (Batch, Heads, SeqLen_Q, D_k).
        K (torch.Tensor): Key tensor (Batch, KV_Heads, SeqLen_KV, D_k).
        V (torch.Tensor): Value tensor (Batch, KV_Heads, SeqLen_KV, D_v).
        qk_scale (float, optional): Scaling factor for QK dot product. Defaults to 1/sqrt(D_k).
        input_pos (int, optional): Global start pos for queries (causal masking). Defaults to 0.
        tile_m (int): Tile size for Query sequence length (M dimension).
        tile_n (int): Tile size for Key/Value sequence length (N dimension).
        query_group_size (int): Number of query heads per key/value head.
        causal (bool): If True, applies causal masking.
        options (dict): XPU compile options forwarded through
                ``ct.backend_options(options)``.

    Returns:
        torch.Tensor: Output tensor (Batch, Heads, SeqLen_Q, D_v).
    """
    # --- Input Validation ---
    if Q.ndim != 4 or K.ndim != 4 or V.ndim != 4:
        raise ValueError("Input tensors Q, K, V must be 4D (Batch, Heads, SeqLen, Dim).")
    if Q.shape[0] != K.shape[0] or Q.shape[0] != V.shape[0]:
        raise ValueError("Batch dimensions must match for Q, K, V.")
    if Q.shape[1] % query_group_size != 0:
        raise ValueError("Number of query heads must be divisible by query_group_size.")
    if K.shape[1] * query_group_size != Q.shape[1]:
        raise ValueError("K_Heads * query_group_size must equal Q_Heads.")
    if Q.shape[3] != K.shape[3]:
        raise ValueError("D_k (last dim of Q and K) must match.")
    if K.shape[2] != V.shape[2]:
        raise ValueError("SeqLen_KV (dim 2 of K and V) must match.")
    if Q.device != K.device or Q.device != V.device or not Q.is_xpu:
        raise ValueError("All input tensors must be on the same XPU device.")
    if Q.dtype != K.dtype or Q.dtype != V.dtype:
        raise ValueError("All input tensors must have the same data type.")

    Batch, Heads, SeqLen_Q, D_k = Q.shape
    _, KV_Heads, SeqLen_KV, D_v = V.shape
    even_k = (SeqLen_KV % tile_n) == 0

    if qk_scale is None:
        qk_scale = 1.0 / math.sqrt(D_k)

    # --- Create Output Tensor ---
    Out = torch.empty((Batch, Heads, SeqLen_Q, D_v), dtype=Q.dtype, device=Q.device)

    # --- Calculate Grid Dimensions ---
    grid_x = math.ceil(SeqLen_Q / tile_m)
    grid_y = Batch * Heads
    grid = (grid_x, grid_y, 1)

    # --- Build XPU options ---
    launch_options = {
        "block_threads": (1, 32, 16),
        "assume_in_bounds": False,
        **(options or {}),
    }

    # --- Launch the FMHA Kernel ---
    with ct.backend_options(launch_options):
        ct.launch(torch.xpu.current_stream(), grid, kernel, (
            Q, K, V, Out,
            qk_scale,
            input_pos,
            D_k,
            Heads,
            tile_m,
            tile_n,
            query_group_size,
            causal,
            even_k
        ))

    return Out


def torch_fmha(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
               is_causal: bool, enable_gqa: bool) -> torch.Tensor:
    # The CUDA-specific sdpa backends (cuDNN/Flash) are unavailable on XPU.
    return scaled_dot_product_attention(Q, K, V,
                                        is_causal=is_causal,
                                        enable_gqa=enable_gqa)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--correctness-check",
        action="store_true",
        help="Check the correctness of the results",
    )
    args = parser.parse_args()
    print("--- Running cuTile Fused Multi-Head Attention (FMHA) Sample (XPU) ---")

    if not torch.xpu.is_available():
        raise RuntimeError(
            "XPU is not available. Ensure Intel GPU runtime and PyTorch XPU are installed."
        )

    # --- User Configuration ---
    BATCH_SIZE = 2
    NUM_HEADS = 8
    SEQ_LEN_Q = 512
    SEQ_LEN_KV = 512
    D_K = 64
    D_V = 64

    QUERY_GROUP_SIZE = 1

    DTYPE = torch.float16
    device = "xpu"

    Q_input = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN_Q, D_K, dtype=DTYPE, device=device)
    K_input = torch.randn(BATCH_SIZE, NUM_HEADS // QUERY_GROUP_SIZE, SEQ_LEN_KV, D_K,
                          dtype=DTYPE, device=device)
    V_input = torch.randn(BATCH_SIZE, NUM_HEADS // QUERY_GROUP_SIZE, SEQ_LEN_KV, D_V,
                          dtype=DTYPE, device=device)

    print("  Configuration:")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Number of Heads: {NUM_HEADS}")
    print(f"  Query Sequence Length: {SEQ_LEN_Q}")
    print(f"  KV Sequence Length: {SEQ_LEN_KV}")
    print(f"  Head Dimension (D_k): {D_K}")
    print(f"  Value Dimension (D_v): {D_V}")
    print(f"  Data Type: {DTYPE}")
    print(f"Input Q shape: {Q_input.shape}")
    print(f"Input K shape: {K_input.shape}")
    print(f"Input V shape: {V_input.shape}")

    # Test 1: Non-Causal Attention
    print("\n--- Test 1: Non-Causal Attention ---")
    output_fmha_cutile_non_causal = cutile_fmha(
        Q=Q_input, K=K_input, V=V_input,
        tile_m=128, tile_n=128,  # Increased tile sizes
        causal=False,
        query_group_size=QUERY_GROUP_SIZE
    )
    print(f"""cuTile FMHA Output shape (Non-Causal):{output_fmha_cutile_non_causal.shape},
        dtype:{output_fmha_cutile_non_causal.dtype}""")
    if args.correctness_check:
        ref_fmha = torch_fmha(Q_input, K_input, V_input,
                              is_causal=False, enable_gqa=False)
        torch.testing.assert_close(output_fmha_cutile_non_causal, ref_fmha, atol=1e-3, rtol=1e-3)
        print("Correctness check passed")
    else:
        print("Correctness check disabled")

    # Test 2: Causal Attention
    print("\n--- Test 2: Causal Attention ---")
    output_fmha_cutile_causal = cutile_fmha(
        Q=Q_input, K=K_input, V=V_input,
        tile_m=128, tile_n=128,  # Increased tile sizes
        causal=True,
        query_group_size=QUERY_GROUP_SIZE
    )
    print(f"""cuTile FMHA Output shape (Causal): {output_fmha_cutile_causal.shape},
            dtype: {output_fmha_cutile_causal.dtype}""")
    if args.correctness_check:
        ref_fmha = torch_fmha(Q_input, K_input, V_input,
                              is_causal=True, enable_gqa=False)
        torch.testing.assert_close(output_fmha_cutile_causal, ref_fmha, atol=1e-3, rtol=1e-3)
        print("Correctness check passed")
    else:
        print("Correctness check disabled")

    # Test 3 (autotuning and benchmarking) of the CUDA sample is not ported to XPU yet.

    print("\n--- cuTile Fused Multi-Head Attention (FMHA) Sample execution complete ---")
