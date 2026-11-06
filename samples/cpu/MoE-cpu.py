# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) <2026> Intel Corporation.
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import sys
import math

import torch
import cuda.tile as ct

# Kernels and helpers shared with the CUDA sample in the parent directory.
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from MoE import (  # noqa: E402
    fused_moe_kernel,
    silu_and_mul_kernel,
    moe_align_tile_size_torch,
    torch_moe,
    next_power_of_2,
)


ct.set_backend("cpu")


# --- cuTile MoE Wrapper ------------------------------------------------------
def cutile_moe(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    num_cpu_threads: int = 0,
) -> torch.Tensor:
    """
    Executes a Mixture-of-Experts (MoE) forward pass using the fused cuTile kernel.

    Args:
        hidden_states: Token activations, shape (num_tokens, hidden_size)
        w1: Expert gate+up projection weights,
            shape (num_experts, intermediate_size * 2, hidden_size)
        w2: Expert down projection weights,
            shape (num_experts, hidden_size, intermediate_size)
        topk_weights: Router weights per token, shape (num_tokens, topk)
        topk_ids: Expert indices per token, shape (num_tokens, topk)
        tile_m/n/k: Tile sizes for cuTile kernel launch
        num_cpu_threads: Number of CPU threads used to run each kernel

    Returns:
        Tensor with the same shape/dtype as `hidden_states`.
    """
    out_dtype = hidden_states.dtype
    device = hidden_states.device

    num_tokens, hidden_size = hidden_states.shape
    num_experts, _, intermediate_size = w2.shape
    _, topk = topk_ids.shape

    if w1.shape[1] != intermediate_size * 2:
        raise ValueError("w1 must have 2 * intermediate_size rows (gate + up projection)")

    intermediate_cache1 = torch.zeros(
        (num_tokens, topk, intermediate_size * 2),
        device=device,
        dtype=out_dtype,
    )
    intermediate_cache2 = torch.zeros(
        (num_tokens * topk, intermediate_size),
        device=device,
        dtype=out_dtype,
    )
    intermediate_cache3 = torch.zeros(
        (num_tokens, topk, hidden_size),
        device=device,
        dtype=out_dtype,
    )

    sorted_token_ids, sorted_expert_ids = moe_align_tile_size_torch(
        topk_ids,
        tile_m,
        num_experts,
    )

    invoke_fused_moe_kernel(
        hidden_states,
        w1,
        intermediate_cache1,
        topk_weights,
        sorted_token_ids,
        sorted_expert_ids,
        mul_routed_weight=False,
        num_token_replicas=topk,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        num_cpu_threads=num_cpu_threads,
    )

    invoke_silu_and_mul_kernel(
        intermediate_cache1.view(-1, intermediate_cache1.shape[-1]),
        intermediate_cache2,
        num_cpu_threads=num_cpu_threads,
    )

    invoke_fused_moe_kernel(
        intermediate_cache2,
        w2,
        intermediate_cache3,
        topk_weights,
        sorted_token_ids,
        sorted_expert_ids,
        mul_routed_weight=True,
        num_token_replicas=1,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        num_cpu_threads=num_cpu_threads,
    )

    return torch.sum(intermediate_cache3, dim=1)


# --- Helper Utilities -------------------------------------------------------
def invoke_fused_moe_kernel(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    topk_weights: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    sorted_expert_ids: torch.Tensor,
    mul_routed_weight: bool,
    num_token_replicas: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    num_cpu_threads: int = 0,
) -> None:
    m = sorted_token_ids.shape[0]
    n = B.shape[1]

    grid = (math.ceil(m / tile_m) * math.ceil(n / tile_n),)
    topk_weights = topk_weights.view(-1)
    C = C.view(-1, C.shape[2])

    with ct.backend_options({"num_cpu_threads": num_cpu_threads}):
        ct.launch(
            None,
            grid,
            fused_moe_kernel,
            (
                A,
                B,
                C,
                topk_weights,
                sorted_token_ids,
                sorted_expert_ids,
                num_token_replicas,
                mul_routed_weight,
                tile_m,
                tile_n,
                tile_k,
            ),
        )


def invoke_silu_and_mul_kernel(
    AB: torch.Tensor,
    C: torch.Tensor,
    num_cpu_threads: int = 0,
):
    A, B = AB.chunk(2, dim=-1)
    with ct.backend_options({"num_cpu_threads": num_cpu_threads}):
        ct.launch(
            None,
            (AB.shape[0],),
            silu_and_mul_kernel,
            (
                A,
                B,
                C,
                next_power_of_2(C.shape[-1])
            )
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--correctness-check",
        action="store_true",
        help="Check the correctness of the cuTile MoE output against a torch reference.",
    )
    parser.add_argument("--num-cpu-threads", type=int, default=0)
    args = parser.parse_args()

    print("--- Running cuTile Mixture-of-Experts (MoE) Sample ---")

    num_tokens = 48
    hidden_size = 512
    num_experts = 64
    intermediate_size = 1024
    topk = 8
    dtype = torch.bfloat16

    device = "cpu"
    print(
        f"Tokens: {num_tokens}, Hidden: {hidden_size}, "
        f"Experts: {num_experts}, Intermediate: {intermediate_size}, "
        f"TopK: {topk}, dtype: {dtype}"
    )

    hidden_states = torch.empty(
        num_tokens, hidden_size, device=device, dtype=dtype
    ).normal_(0, 0.5)
    w1 = torch.empty(
        num_experts, intermediate_size * 2, hidden_size, device=device, dtype=dtype
    ).normal_(0, 0.1)
    w2 = torch.empty(
        num_experts, hidden_size, intermediate_size, device=device, dtype=dtype
    ).normal_(0, 0.1)

    # Unique expert IDs for each token (no repeating elements per row)
    topk_ids = torch.stack([
        torch.randperm(num_experts, device=device)[:topk]
        for _ in range(num_tokens)
    ])
    topk_weights = torch.softmax(
        torch.randn(num_tokens, topk, device=device), dim=-1
    ).to(dtype)

    print("\n--- Executing cuTile MoE ---")
    # The generic CPU lowering unrolls whole tiles (including the A gather), so
    # small tiles keep compilation fast.
    output_cutile = cutile_moe(hidden_states, w1, w2, topk_weights, topk_ids,
                               tile_m=32, tile_n=64, tile_k=32,
                               num_cpu_threads=args.num_cpu_threads)
    print(f"cuTile MoE output shape: {output_cutile.shape}, "
          "dtype: {output_cutile.dtype}")

    if args.correctness_check:
        print("\n--- Running correctness check against torch reference ---")
        ref_output = torch_moe(hidden_states, w1, w2, topk_weights, topk_ids)
        torch.testing.assert_close(output_cutile, ref_output, rtol=1e-1, atol=1e-1)
        print("Correctness check passed")
    else:
        print("Correctness check disabled (use --correctness-check to enable)")

    print("\n--- cuTile Mixture-of-Experts (MoE) Sample complete ---")
