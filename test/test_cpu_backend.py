# SPDX-FileCopyrightText: Copyright (c) <2026> Intel Corporation.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import cuda.tile as ct
from cuda.tile import _backend
from cuda.tile._backend import cpu
from cuda.tile._backend._signature import build_signature
import numpy as np
import pytest
import torch

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


def test_flatten_arguments_reuses_signature_layout():
    first_arguments = _arguments()
    second_arguments = _arguments()
    signature = build_signature(_vector_add, first_arguments)
    cpu._argument_layout_cache.clear()

    first_values, first_types = cpu._flatten_arguments(signature, first_arguments)
    second_values, second_types = cpu._flatten_arguments(signature, second_arguments)

    assert first_types is second_types
    assert first_values[0] == first_arguments[0].ctypes.data
    assert second_values[0] == second_arguments[0].ctypes.data


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


@ct.kernel
def _scalar_and_array(x, flag: bool, scale: float, count: int):
    pass


def test_signature_matches_native_launcher_types():
    x = np.zeros(4, dtype=np.bool_)

    parameters = build_signature(_scalar_and_array, (x, True, 1.0, 1)).parameters

    assert parameters[0].dtype is ct.bool_
    assert [p.dtype for p in parameters[1:]] == [ct.bool_, ct.float32, ct.int32]


def test_signature_detects_internally_aliasing_arrays():
    base = np.zeros(16, dtype=np.float32)
    broadcast = np.lib.stride_tricks.as_strided(base, shape=(4, 4), strides=(0, 4))

    aliasing = build_signature(_scalar_and_array, (broadcast, True, 1.0, 1))
    disjoint = build_signature(_scalar_and_array, (base.reshape(4, 4), True, 1.0, 1))

    assert aliasing.parameters[0].may_alias_internally
    assert not disjoint.parameters[0].may_alias_internally


def test_signature_skips_layout_assumptions_beyond_five_dimensions():
    x = np.zeros((16,) * 6, dtype=np.float32)

    [array, *_] = build_signature(_scalar_and_array, (x, True, 1.0, 1)).parameters

    assert array.stride_constant == (None,) * 6
    assert array.shape_divisible_by == (1,) * 6


def test_flatten_arguments_rejects_non_host_arrays():
    arguments = _arguments()
    signature = build_signature(_vector_add, arguments)
    device_tensor = torch.empty(64, dtype=torch.float32, device="meta")

    with pytest.raises(ValueError, match="expected host memory"):
        cpu._flatten_arguments(signature, (device_tensor, *arguments[1:]))


def test_lower_tileir_uses_cpu_pipeline(monkeypatch):
    observed = {}
    monkeypatch.setattr(cpu, "resolve_tool", lambda *args: "/tool")

    def run_tool(backend, argv, bytecode):
        observed.update(backend=backend, argv=argv, bytecode=bytecode)
        return b"module {}"

    monkeypatch.setattr(cpu, "run_tool", run_tool)

    assert cpu._lower_tileir(b"bytecode", cpu.Options(True)) == b"module {}"
    assert observed["backend"] == "CPU"
    assert observed["bytecode"] == b"bytecode"
    assert "target=cpu" in observed["argv"][1]
    assert "append-grid-args=true" in observed["argv"][1]
    assert "drop-rounding-modes=true" in observed["argv"][1]
    assert "assume-in-bounds=true" in observed["argv"][1]
    assert "--convert-memref-args-to-ptr-args" in observed["argv"]
    assert not any(arg.startswith("--mlir-print-ir") for arg in observed["argv"])


def test_triton_cpu_reports_missing_package(monkeypatch):
    monkeypatch.setattr(cpu.importlib.util, "find_spec", lambda name: None)
    cpu._triton_cpu.cache_clear()

    with pytest.raises(ImportError, match="requires the 'cpu' extra"):
        cpu._triton_cpu()


def test_triton_cpu_preserves_nested_import_error(monkeypatch):
    def import_module(name):
        raise ModuleNotFoundError(f"No module named {name!r}", name=name)

    monkeypatch.setattr(
        cpu.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(cpu.importlib, "import_module", import_module)
    cpu._triton_cpu.cache_clear()

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
        lambda assume_in_bounds: (None, backend, options, None, None))
    monkeypatch.setattr(
        cpu, "_cpu_identity_modules",
        lambda: (
            "triton-version",
            SimpleNamespace(__file__="/nonexistent/triton-compiler.py"),
            SimpleNamespace(__file__="/libtriton.so"),
            SimpleNamespace(_find_compiler=lambda language: "/cc"),
            SimpleNamespace(build=SimpleNamespace(impl=None)),
        ))
    monkeypatch.setattr(cpu, "resolve_tool", lambda *_args: "/tool")
    monkeypatch.setattr(
        cpu, "file_fingerprint", lambda path: f"fingerprint:{path}")
    cpu._compile_cache_key_cached.cache_clear()

    with cpu.compile_options({"num_cpu_threads": 1}):
        first = cpu.compile_cache_key()
    with cpu.compile_options({"num_cpu_threads": 8}):
        second = cpu.compile_cache_key()
    with cpu.compile_options({"assume_in_bounds": True}):
        in_bounds = cpu.compile_cache_key()
    cpu._compile_cache_key_cached.cache_clear()

    assert first is not None
    assert first == second
    assert first != in_bounds


def test_options_reject_non_bool_flags():
    with pytest.raises(TypeError, match="must be a bool"):
        cpu.normalize_options({"assume_in_bounds": "false"})


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
        ct, "compile_kernel",
        lambda kernel, signature: (b"object", signature.symbol))
    cpu._launch_cache.clear()

    cpu.launch(None, (4,), _vector_add, arguments)
    cpu.launch(None, (4,), _vector_add, arguments)

    assert [call[0] for call in calls].count("create") == 1
    assert [call[0] for call in calls].count("load") == 1
    launches = [call for call in calls if call[0] == "launch"]
    assert len(launches) == 2
    assert launches[0][1][:5] == (4, 1, 1, 0, 1234)
    assert isinstance(launches[0][1][5], SimpleNamespace)
