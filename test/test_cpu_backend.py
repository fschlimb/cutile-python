# SPDX-FileCopyrightText: Copyright (c) <2026> Intel Corporation.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
import sys
from types import ModuleType

import numpy as np
import pytest

import cuda.tile as ct
from cuda.tile import _backend
from cuda.tile._backend import cpu
from cuda.tile._backend._signature import build_signature


ConstInt = ct.Constant[int]


@ct.kernel
def _vector_add(a, b, c, tile_size: ConstInt):
    block = ct.bid(0)
    a_tile = ct.load(a, index=(block,), shape=(tile_size,))
    b_tile = ct.load(b, index=(block,), shape=(tile_size,))
    ct.store(c, index=(block,), tile=a_tile + b_tile)


def _arguments():
    return (
        np.empty(64, dtype=np.float32),
        np.empty(64, dtype=np.float32),
        np.empty(64, dtype=np.float32),
        16,
    )


def test_signature_flattens_arrays_and_omits_constants():
    signature = build_signature(_vector_add, _arguments())

    values, types = cpu._flatten_arguments(signature, _arguments())

    assert list(types.values()) == ["*i8", "i32", "i32"] * 3
    assert values[1:3] == [64, 1]
    assert values[4:6] == [64, 1]
    assert values[7:9] == [64, 1]


def test_signature_uses_actual_array_alignment():
    arguments = list(_arguments())
    arguments[0] = arguments[0][1:]

    signature = build_signature(_vector_add, tuple(arguments))

    assert signature.parameters[0].base_addr_divisible_by == 1


def test_signature_rejects_negative_strides():
    arguments = list(_arguments())
    arguments[0] = arguments[0][::-1]

    with pytest.raises(ValueError, match="negative array strides"):
        build_signature(_vector_add, tuple(arguments))


def test_lower_tileir_uses_cpu_pipeline(monkeypatch):
    observed = {}
    monkeypatch.setattr(cpu, "resolve_tool", lambda *args: "/tool")

    def run_tool(backend, argv, bytecode):
        observed.update(backend=backend, argv=argv, bytecode=bytecode)
        return b"module {}"

    monkeypatch.setattr(cpu, "run_tool", run_tool)

    assert cpu._lower_tileir(b"bytecode") == b"module {}"
    assert observed["backend"] == "CPU"
    assert observed["bytecode"] == b"bytecode"
    assert "target=cpu" in observed["argv"][1]
    assert "append-grid-args=true" in observed["argv"][1]
    assert "drop-rounding-modes=true" in observed["argv"][1]
    assert "--convert-memref-args-to-ptr-args" in observed["argv"]
    assert "--mlir-print-ir-after-all" not in observed["argv"]


def test_triton_cpu_reports_missing_package(monkeypatch):
    monkeypatch.setattr(cpu.importlib.util, "find_spec", lambda name: None)

    with pytest.raises(ImportError, match="requires the 'cpu' extra"):
        cpu._triton_cpu()


def test_triton_cpu_preserves_nested_import_error(monkeypatch):
    monkeypatch.setattr(
        cpu.importlib.util, "find_spec", lambda name: object())
    for name in tuple(sys.modules):
        if name == "triton" or name.startswith("triton."):
            monkeypatch.delitem(sys.modules, name, raising=False)
    triton = ModuleType("triton")
    triton.__path__ = []
    monkeypatch.setitem(sys.modules, "triton", triton)

    with pytest.raises(ModuleNotFoundError, match=r"triton\._C"):
        cpu._triton_cpu()


def test_backend_registration_discovers_compile_cache_key():
    _backend.set_backend("cpu")
    try:
        assert _backend.get_compile_cache_key_fn() is cpu.compile_cache_key
    finally:
        _backend.clear_backend()


def test_compile_cache_key_ignores_thread_count(monkeypatch):
    backend = SimpleNamespace(
        cpu_arch="x86_64", cpu_name="cpu", cpu_features={"feature"})
    options = SimpleNamespace(hash=lambda: "options")
    monkeypatch.setattr(
        cpu, "_cpu_compiler",
        lambda: (None, backend, options, None, None))
    monkeypatch.setattr(
        cpu, "_cpu_identity_modules",
        lambda: (
            "triton-version",
            SimpleNamespace(__file__="/triton-compiler.py"),
            SimpleNamespace(__file__="/libtriton.so"),
            SimpleNamespace(_find_compiler=lambda language: "/cc"),
            SimpleNamespace(build=SimpleNamespace(impl=None)),
        ))
    monkeypatch.setattr(cpu, "resolve_tool", lambda *_args: "/tool")
    monkeypatch.setattr(
        cpu, "file_fingerprint", lambda path: f"fingerprint:{path}")

    with cpu.compile_options({"num_cpu_threads": 1}):
        first = cpu.compile_cache_key()
    with cpu.compile_options({"num_cpu_threads": 8}):
        second = cpu.compile_cache_key()

    assert first == second


@pytest.mark.parametrize(
    "grid, expected",
    [(7, (7, 1, 1)), ((7, 3), (7, 3, 1)), ((7, 3, 2), (7, 3, 2))],
)
def test_triplet(grid, expected):
    assert cpu._triplet(grid) == expected


def test_launch_reuses_triton_launcher(monkeypatch):
    arguments = _arguments()
    calls = []

    class Launcher:
        def __init__(self, source, metadata):
            calls.append(("create", list(source.signature.values()), metadata))

        def __call__(self, *args):
            calls.append(("launch", args))

    class Utils:
        def load_binary(self, symbol, binary, shared, device):
            calls.append(("load", symbol, binary, shared, device))
            return object(), 1234, 0, 0, 0

    monkeypatch.setattr(
        cpu, "_triton_cpu",
        lambda: (None, None, None, Launcher, Utils),
    )
    monkeypatch.setattr(
        ct, "compile_kernel", lambda kernel, signature: (b"object", signature.symbol))
    cpu._launch_cache.clear()

    cpu.launch(None, (4,), _vector_add, arguments)
    cpu.launch(None, (4,), _vector_add, arguments)

    assert [call[0] for call in calls].count("create") == 1
    assert [call[0] for call in calls].count("load") == 1
    launches = [call for call in calls if call[0] == "launch"]
    assert len(launches) == 2
    assert launches[0][1][:5] == (4, 1, 1, 0, 1234)
    assert isinstance(launches[0][1][5], SimpleNamespace)