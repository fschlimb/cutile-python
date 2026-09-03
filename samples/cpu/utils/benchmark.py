# SPDX-FileCopyrightText: Copyright (c) <2026> Intel Corporation.
# SPDX-License-Identifier: Apache-2.0

import gc
import time
from math import ceil

import cuda.tile as ct
from cuda.tile._backend import cpu


_WARMUP_ITER_GUESS = 5
_MIN_ROUND_TIME_NS = 200_000_000
_ROUNDS = 50
_WARMUP_ROUNDS = 1
_MAX_ITERS = 1_000_000


def _measure_block_ns(f, tuple_of_args, iterations: int) -> int:
    gc_was_enabled = gc.isenabled()
    if gc_was_enabled:
        gc.disable()
    try:
        start = time.perf_counter_ns()
        for _ in range(iterations):
            f(*tuple_of_args)
        return time.perf_counter_ns() - start
    finally:
        if gc_was_enabled:
            gc.enable()


def _estimate_bench_iter(f, tuple_of_args):
    f(*tuple_of_args)
    elapsed_ns = _measure_block_ns(
        f, tuple_of_args, _WARMUP_ITER_GUESS)
    if elapsed_ns <= 0:
        raise RuntimeError("CPU benchmark timer did not advance")

    iterations = ceil(
        _MIN_ROUND_TIME_NS * _WARMUP_ITER_GUESS / elapsed_ns)
    iterations = min(_MAX_ITERS, max(_ROUNDS, iterations))
    return _WARMUP_ROUNDS, iterations, _ROUNDS


def _time_ms(f, tuple_of_args, warmup: int, iters: int, rounds: int) -> float:
    for _ in range(warmup):
        f(*tuple_of_args)

    run_iters = max(iters, rounds)
    elapsed_ns = sum(
        _measure_block_ns(f, tuple_of_args, run_iters)
        for _ in range(rounds)
    )
    return elapsed_ns / (rounds * run_iters * 1_000_000)


def _report_benchmark(f, tuple_of_args) -> dict[str, float]:
    warmup_rounds, iterations, rounds = _estimate_bench_iter(
        f, tuple_of_args)
    mean_time_ms = _time_ms(
        f, tuple_of_args, warmup_rounds, iterations, rounds)
    return {"mean_time_ms": mean_time_ms}


def report_benchmark(f, tuple_of_args, *, kernel=None, grid=None,
                     kernel_args=None, options=None) -> dict[str, float]:
    """Benchmark a callable, optionally using one resolved CPU kernel binary.

    When ``kernel`` is provided, ``kernel_args`` are used to resolve the
    backend binary once and each measured iteration launches that binary
    directly. The original callable and ``tuple_of_args`` remain the regular
    benchmark interface for callers that do not need this optimization.
    """
    if kernel is None:
        return _report_benchmark(f, tuple_of_args)
    if grid is None or kernel_args is None:
        raise TypeError(
            "grid and kernel_args are required when kernel is provided")

    kernel_args = tuple(kernel_args)
    with cpu.compile_options(dict(options or {})):
        compiled = ct.compile_kernel_for_launch(kernel, kernel_args)

        def direct_call():
            return ct.launch_compiled(None, grid, compiled, kernel_args)

        return _report_benchmark(direct_call, ())
