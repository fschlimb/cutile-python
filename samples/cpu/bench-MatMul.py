# SPDX-FileCopyrightText: Copyright (c) <2026> Intel Corporation.
# SPDX-License-Identifier: Apache-2.0

import argparse

import torch

import cuda.tile as ct
from utils.benchmark import report_benchmark
from utils.sfc_matmul import prepare_sfc_matmul


@ct.autotune(
    configs=[
        ct.tune.Config({"blocking_factor_k": 1}),
        ct.tune.Config({"blocking_factor_k": 2}),
        ct.tune.Config({"blocking_factor_k": 4}),
    ],
    quiet=False,
)
def autotuned_prepare_sfc_matmul(a, b, *, options=None, blocking_factor_k=1):
    return prepare_sfc_matmul(
        a,
        b,
        blocking_factor_k=blocking_factor_k,
        options=options,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-cpu-threads", type=int, default=0)
    args = parser.parse_args()

    M_dim, N_dim, K_dim = 1024, 2048, 4096
    A = torch.randn(M_dim, K_dim, dtype=torch.bfloat16, device="cpu")
    B = torch.randn(K_dim, N_dim, dtype=torch.bfloat16, device="cpu")

    options = {
        "assume_in_bounds": True,
        "num_cpu_threads": args.num_cpu_threads,
    }
    selected = autotuned_prepare_sfc_matmul.prepare(
        A,
        B,
        options=options,
    )
    prepared_output = autotuned_prepare_sfc_matmul.launch_prepared()
    torch.testing.assert_close(
        prepared_output, torch.matmul(A, B), atol=1e-2, rtol=1e-2)
    print("Correctness check passed")
    print(f"Selected configuration: {selected}")

    stats_cutile = report_benchmark(
        autotuned_prepare_sfc_matmul.launch_prepared,
        (),
    )
    stats_torch = report_benchmark(torch.matmul, (A, B))
    print("Benchmark results:")
    print(f"  cuTile MatMul: {stats_cutile['mean_time_ms']:.5f} ms")
    print(f"  torch MatMul: {stats_torch['mean_time_ms']:.5f} ms")
    speedup = stats_torch["mean_time_ms"] / stats_cutile["mean_time_ms"]
    print(f"Speedup: {speedup:.3f}x")
