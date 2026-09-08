# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

import cuda.tile as ct
import cuda.tile.tune._autotune as autotune_mod
from cuda.tile.tune import Config, autotune


@ct.kernel
def _kernel(x, N: ct.Constant[int], TILE: ct.Constant[int], FLAG: ct.Constant[bool]):
    pass


@ct.kernel
def _runtime_kernel(x, scale: float, TILE: ct.Constant[int]):
    pass


def _fake_runtime(monkeypatch, events=None):
    tuned = []
    launched = []
    active_options = []

    def search(search_space, stream, *, grid_fn, kernel, args_fn,
               hints_fn, quiet, context_fn=None):
        config = search_space[0]
        if context_fn is None:
            tuned.append((config, args_fn(config), grid_fn(config)))
        else:
            with context_fn(config):
                tuned.append((config, args_fn(config), grid_fn(config)))
        return SimpleNamespace(best=SimpleNamespace(config=config))

    def compile_kernel(kernel, args):
        if events is not None and active_options:
            events.append(("compile", dict(active_options[-1])))
        return SimpleNamespace(binary=object(), symbol="kernel", signature=None)

    def launch_compiled(stream, grid, compiled, args):
        if events is not None and active_options:
            events.append(("launch", dict(active_options[-1])))
        launched.append((grid, args))

    def launch(stream, grid, kernel, args):
        if events is not None and active_options:
            events.append(("launch", dict(active_options[-1])))
        launched.append((grid, args))

    @contextmanager
    def compile_options(options):
        options = dict(options)
        if events is not None:
            events.append(("enter", options))
        active_options.append(options)
        try:
            yield
        finally:
            active_options.pop()
            if events is not None:
                events.append(("exit", options))

    monkeypatch.setattr(autotune_mod, "exhaustive_search", search)
    monkeypatch.setattr(autotune_mod, "compile_for_launch", compile_kernel)
    monkeypatch.setattr(
        autotune_mod._backend, "get_launch_compiled_fn", lambda: launch_compiled)
    monkeypatch.setattr(
        autotune_mod._backend, "get_compile_options_fn", lambda: compile_options)
    monkeypatch.setattr(autotune_mod._execution, "launch", launch)
    return tuned, launched


def test_autotune_injects_configured_constants_and_reuses_tuning(monkeypatch):
    tuned, launched = _fake_runtime(monkeypatch)
    autotuned = autotune(
        configs=[Config({"TILE": 4}), Config({"TILE": 8})],
        grid=lambda meta: (meta["x"].shape[0] // meta["TILE"],),
    )(_kernel)

    x = torch.empty(16)
    args = (x, 16, True)
    autotuned.prepare(None, args)
    autotuned(None, args)
    autotuned(None, args)

    assert len(tuned) == 1
    assert len(launched) == 2
    assert launched[0] == ((4,), (x, 16, 4, True))
    assert autotuned.best_config == tuned[0][0]


def test_launch_prepared_uses_compiled_kernel(monkeypatch):
    _tuned, launched = _fake_runtime(monkeypatch)
    autotuned = autotune(
        configs=[Config({"TILE": 4})],
        grid=(1,),
    )(_runtime_kernel)
    args = (torch.empty(16), 1.0)

    with pytest.raises(RuntimeError, match="prepared"):
        autotuned.launch_prepared(None)
    autotuned.prepare(None, args)
    autotuned.launch_prepared(None)

    assert launched == [((1,), (args[0], 1.0, 4))]


def test_autotune_options_scope_tuning_compile_and_launch(monkeypatch):
    events = []
    _tuned, launched = _fake_runtime(monkeypatch, events)
    options = {"assume_in_bounds": True, "num_cpu_threads": 4}
    autotuned = autotune(
        configs=[Config({"TILE": 4})],
        grid=(1,),
        options=options,
    )(_runtime_kernel)
    args = (torch.empty(16), 1.0)

    autotuned.prepare(None, args)
    autotuned.launch_prepared(None)

    assert launched == [((1,), (args[0], 1.0, 4))]
    assert events == [
        ("enter", options), ("exit", options),
        ("enter", options), ("compile", options), ("exit", options),
        ("enter", options), ("launch", options), ("exit", options),
    ]


def test_autotune_options_callable_receives_configured_meta(monkeypatch):
    _tuned, _launched = _fake_runtime(monkeypatch)
    received = []

    def options(meta):
        received.append(meta)
        return {"assume_in_bounds": meta["N"] % meta["TILE"] == 0}

    autotuned = autotune(
        configs=[Config({"TILE": 4})],
        grid=(1,),
        options=options,
    )(_kernel)
    x = torch.empty(16)
    autotuned(None, (x, 16, True))

    assert len(received) == 2
    assert all(meta["x"] is x and meta["N"] == 16
               and meta["TILE"] == 4 and meta["FLAG"] is True
               for meta in received)


def test_autotune_rejects_invalid_options_callable_result(monkeypatch):
    _fake_runtime(monkeypatch)
    autotuned = autotune(
        configs=[Config({"TILE": 4})],
        grid=(1,),
        options=lambda _meta: None,
    )(_runtime_kernel)

    with pytest.raises(TypeError, match="options callable must return a mapping"):
        autotuned(None, (torch.empty(16), 1.0))


def test_autotune_retunes_for_shape_and_unconfigured_constant_changes(monkeypatch):
    tuned, launched = _fake_runtime(monkeypatch)
    autotuned = autotune(
        configs=[Config({"TILE": 4})],
        grid=lambda meta: (meta["x"].shape[0] // meta["TILE"],),
    )(_kernel)

    autotuned(None, (torch.empty(16), 16, True))
    autotuned(None, (torch.empty(16), 32, True))

    assert len(tuned) == 2
    assert [entry[2] for entry in tuned] == [(4,), (4,)]


def test_unkeyed_runtime_scalar_cannot_reuse_stale_grid(monkeypatch):
    tuned, launched = _fake_runtime(monkeypatch)
    autotuned = autotune(
        configs=[Config({"TILE": 4})],
        grid=lambda meta: (int(meta["scale"]),),
    )(_runtime_kernel)

    x = torch.empty(16)
    autotuned(None, (x, 1.0))
    autotuned(None, (x, 2.0))

    assert len(tuned) == 1
    assert [entry[0] for entry in launched] == [(1,), (2,)]


def test_autotune_rejects_inconsistent_configured_constants():
    with pytest.raises(ValueError, match="same constant parameters"):
        autotune(
            configs=[Config({"TILE": 4}), Config({"N": 16})],
            grid=(1,),
        )(_kernel)


def test_autotune_rejects_unknown_constant_parameter():
    with pytest.raises(ValueError, match="unknown constant parameters"):
        autotune(configs=[Config({"UNKNOWN": 4})], grid=(1,))(_kernel)


def test_autotune_forwards_generic_compiler_hints(monkeypatch):
    received = []

    def replace_hints(kernel, **hints):
        received.append(hints)
        return kernel

    monkeypatch.setattr(
        type(_runtime_kernel), "replace_hints", replace_hints)
    autotune(
        configs=[Config({"TILE": 4}, future_compiler_hint="value")],
        grid=(1,),
    )(_runtime_kernel)

    assert received == [{"future_compiler_hint": "value"}]


def test_autotune_rejects_options_for_factory():
    with pytest.raises(TypeError, match="options is only valid"):
        autotune(configs=[Config({})], options={})(lambda: lambda: None)


def test_autotune_factory_reuses_prepared_multi_kernel_operation(monkeypatch):
    prepared_calls = []
    factory_calls = []
    timer_calls = []

    def factory(x, blocking_factor_k=1):
        factory_calls.append(blocking_factor_k)

        def prepared():
            prepared_calls.append(blocking_factor_k)

        return prepared

    def timer(stream, prepared, args):
        timer_calls.append(prepared)
        prepared(*args)
        return 1.0

    def search(search_space, stream, *, grid_fn, kernel, args_fn,
               hints_fn, quiet, benchmark_fn):
        config = search_space[0]
        benchmark_fn(stream, grid_fn(config), kernel, args_fn(config))
        return SimpleNamespace(best=SimpleNamespace(config=config))

    monkeypatch.setattr(autotune_mod, "exhaustive_search", search)
    monkeypatch.setattr(
        autotune_mod._backend, "get_benchmark_callable_fn", lambda: timer)

    autotuned = autotune(
        configs=[
            Config({"blocking_factor_k": 1}),
            Config({"blocking_factor_k": 2}),
        ],
    )(factory)
    x = torch.empty(16)
    autotuned.prepare(x)
    autotuned.launch_prepared()
    autotuned.launch_prepared()

    assert factory_calls == [1]
    assert len(timer_calls) == 1
    assert prepared_calls == [1, 1, 1]


def test_factory_rebuilds_operation_for_different_storage(monkeypatch):
    factory_calls = []

    def factory(x):
        factory_calls.append(x)
        return lambda: None

    def search(search_space, stream, *, args_fn, **kwargs):
        config = search_space[0]
        args_fn(config)
        return SimpleNamespace(best=SimpleNamespace(config=config))

    monkeypatch.setattr(autotune_mod, "exhaustive_search", search)
    monkeypatch.setattr(
        autotune_mod._backend,
        "get_benchmark_callable_fn",
        lambda: lambda stream, fn, args: 1.0,
    )
    autotuned = autotune(configs=[Config({})])(factory)
    first = torch.empty(16)
    second = torch.empty(16)
    autotuned.prepare(first)
    autotuned.prepare(second)

    assert factory_calls == [first, second]