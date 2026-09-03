# SPDX-FileCopyrightText: Copyright (c) <2026> Intel Corporation.
# SPDX-License-Identifier: Apache-2.0

import gc

import pytest

from samples.utils.cpu import benchmark


def _clock(values):
    values = iter(values)
    return lambda: next(values)


def test_calibration_targets_two_hundred_milliseconds(monkeypatch):
    calls = 0

    def function():
        nonlocal calls
        calls += 1

    monkeypatch.setattr(
        benchmark.time, "perf_counter_ns", _clock((0, 5_000_000)))

    warmup, iterations, rounds = benchmark._estimate_bench_iter(function, ())

    assert (warmup, iterations, rounds) == (1, 200, 5)
    assert calls == 6


def test_calibration_caps_iteration_count(monkeypatch):
    monkeypatch.setattr(
        benchmark.time, "perf_counter_ns", _clock((0, 1)))

    _, iterations, _ = benchmark._estimate_bench_iter(lambda: None, ())

    assert iterations == benchmark._MAX_ITERS


def test_calibration_rejects_stalled_timer(monkeypatch):
    monkeypatch.setattr(
        benchmark.time, "perf_counter_ns", _clock((7, 7)))

    with pytest.raises(RuntimeError, match="timer did not advance"):
        benchmark._estimate_bench_iter(lambda: None, ())


def test_report_benchmark_returns_arithmetic_mean(monkeypatch):
    calls = 0

    def function():
        nonlocal calls
        calls += 1

    round_times_ns = (100_000_000, 200_000_000, 300_000_000,
                      400_000_000, 500_000_000)
    clock_values = [0, 5_000_000]
    for elapsed_ns in round_times_ns:
        clock_values.extend((0, elapsed_ns))
    monkeypatch.setattr(
        benchmark.time, "perf_counter_ns", _clock(clock_values))

    result = benchmark.report_benchmark(function, ())

    assert result == {"mean_time_ms": 1.5}
    assert calls == 1007


def test_measurement_restores_enabled_gc():
    gc.enable()

    def fail():
        raise ValueError("expected")

    with pytest.raises(ValueError, match="expected"):
        benchmark._measure_block_ns(fail, (), 1)

    assert gc.isenabled()


def test_measurement_preserves_disabled_gc():
    gc.disable()
    try:
        benchmark._measure_block_ns(lambda: None, (), 1)
        assert not gc.isenabled()
    finally:
        gc.enable()
