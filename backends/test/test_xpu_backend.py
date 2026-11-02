# SPDX-FileCopyrightText: Copyright (c) <2026> Intel Corporation. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import struct
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("cuda.tile._backend.xpu._level_zero",
                    reason="needs the Level Zero extension")

import cuda.tile as ct  # noqa: E402
from cuda.tile import _backend  # noqa: E402
from cuda.tile._backend import xpu  # noqa: E402
from cuda.tile._backend import _custom  # noqa: E402
from cuda.tile._backend._signature import build_signature, constant_parameters  # noqa: E402

_OPTIONS = {"wg_m": 128, "wg_n": 64, "sg_m": 32, "sg_n": 32}


@ct.kernel
def _scaled(x, count: int, TILE: ct.Constant[int]):
    pass


@ct.kernel
def _counter(count: int, TILE: ct.Constant[int]):
    pass


def _mock_toolchain(monkeypatch):
    xpu._ocloc_identity.cache_clear()
    monkeypatch.setattr(xpu, "native_compiler", lambda _: SimpleNamespace(
        __file__="/compiler", xpu_identity=lambda: "ocloc-1.0:igc"))
    monkeypatch.setattr(
        xpu, "file_fingerprint", lambda path: f"fingerprint:{path}")


def _identity(**options):
    return xpu.compiler_identity(xpu.normalize_options({**_OPTIONS, **options}))


def test_xpu_module_is_a_backend():
    _backend.set_backend("xpu")
    try:
        assert _backend.get_backend() is xpu
    finally:
        _backend.set_backend(None)


def _runtime(kernel, args):
    constants = constant_parameters(kernel)
    return _custom._runtime_arguments(
        args, constants, _custom._describe(constants, args))


def test_options_reject_unknown_keys():
    with pytest.raises(ValueError, match="unknown options 'wg_k'"):
        xpu.normalize_options({**_OPTIONS, "wg_k": 1})


def test_launch_block_needs_no_tile_sizes():
    assert xpu.normalize_options({}).block == xpu._DEFAULT_BLOCK
    assert xpu.normalize_options({"block_threads": (1, 32, 16)}).block == (1, 32, 16)
    assert xpu.normalize_options(_OPTIONS).block == (128, 1, 1)


@pytest.mark.parametrize("options", [
    {"wg_m": 128, "wg_n": 64},
    {**_OPTIONS, "block_threads": 256},
    {**_OPTIONS, "sg_m": 0},
    {**_OPTIONS, "sg_n": 48},
])
def test_launch_block_rejects_inconsistent_tile_sizes(options):
    with pytest.raises(ValueError, match="XPU backend"):
        xpu.normalize_options(options)


def test_compiler_identity_changes_with_compiler_options(monkeypatch):
    _mock_toolchain(monkeypatch)

    base = _identity()

    assert base is not None
    assert base != _identity(sg_m=64)
    assert base != _identity(large_register_file=False)


def test_compiler_identity_reports_missing_native_dependencies(monkeypatch):
    _mock_toolchain(monkeypatch)

    def unavailable(_):
        raise ImportError("missing libocloc")

    monkeypatch.setattr(xpu, "native_compiler", unavailable)
    with pytest.raises(ImportError, match="missing libocloc"):
        _identity()


def test_compiler_identity_caches_versions_but_tracks_rebuilds(monkeypatch):
    _mock_toolchain(monkeypatch)
    calls = []
    fingerprint = ["first"]

    def query():
        calls.append(True)
        return "ocloc-1.0:igc"

    monkeypatch.setattr(xpu, "native_compiler", lambda _: SimpleNamespace(
        __file__="/compiler", xpu_identity=query))
    monkeypatch.setattr(xpu, "file_fingerprint", lambda path:
                        fingerprint[0] if path == "/compiler" else path)
    first = _identity()
    assert _identity() == first
    assert _identity(large_register_file=False) != first
    assert len(calls) == 1
    fingerprint[0] = "rebuilt"
    assert _identity() != first
    assert len(calls) == 2


def test_compiler_identity_does_not_hide_or_cache_query_failures(monkeypatch):
    _mock_toolchain(monkeypatch)
    calls = []

    def query():
        calls.append(True)
        if len(calls) == 1:
            raise RuntimeError("libocloc query failed")
        return "ocloc-1.0:igc"

    monkeypatch.setattr(xpu, "native_compiler", lambda _: SimpleNamespace(
        __file__="/compiler", xpu_identity=query))
    with pytest.raises(RuntimeError, match="libocloc query failed"):
        _identity()
    assert _identity() is not None
    assert len(calls) == 2


def test_options_are_validated_and_normalized():
    options = xpu.normalize_options({"block_threads": 256})

    assert options.block == (256, 1, 1)
    with pytest.raises(TypeError, match="must be a bool"):
        xpu.normalize_options({**_OPTIONS, "assume_in_bounds": "false"})
    with pytest.raises(ValueError, match="xegpu_op_level"):
        xpu.normalize_options({**_OPTIONS, "xegpu_op_level": "grid"})


def test_compile_passes_options_to_the_native_compiler(monkeypatch):
    observed = {}

    def invoke(backend, method, bytecode, *options):
        observed.update(backend=backend, method=method, bytecode=bytecode, options=options)
        return b"binary"

    monkeypatch.setattr(xpu, "invoke", invoke)
    options = xpu.normalize_options(
        {"block_threads": 256, "assume_in_bounds": True})
    signature = build_signature(_counter, (7, 4))

    assert xpu.compile(b"bytecode", signature, options) == b"binary"
    assert observed["backend"] == "xpu"
    assert observed["method"] == "compile_xpu"
    assert observed["bytecode"] == b"bytecode"
    assert observed["options"][:5] == ((256, 1, 1), True, "workgroup", True, "bmg")
    assert observed["options"][-1] == (-1,)


def test_launch_uses_the_compiled_block_and_symbol(monkeypatch):
    kernels = []

    class Kernel:
        def __init__(self, binary, symbol):
            self.binary, self.symbol, self.launches = binary, symbol, []
            kernels.append(self)

        def launch(self, *args):
            self.launches.append(args)

    monkeypatch.setattr(xpu, "_LevelZeroKernel", Kernel)
    signature = build_signature(_counter, (7, 4))
    options = xpu.normalize_options(
        {"wg_m": 64, "wg_n": 32, "sg_m": 32, "sg_n": 16})

    loaded = xpu.load(b"binary", signature, options)
    xpu.launch(loaded, None, (2,), _runtime(_counter, (7, 4)))
    xpu.launch(loaded, None, (3,), _runtime(_counter, (8, 4)))

    [kernel] = kernels
    assert (kernel.binary, kernel.symbol) == (b"binary", signature.symbol)
    assert kernel.launches == [
        ([struct.pack("i", 7)], (2, 1, 1), (64, 1, 1)),
        ([struct.pack("i", 8)], (3, 1, 1), (64, 1, 1)),
    ]


def test_runtime_arguments_pass_arrays_as_memrefs():
    x = torch.empty((4, 6))[:, ::2]
    signature = build_signature(_scaled, (x, 3, 16))
    [array, count] = xpu._runtime_arguments(
        xpu.runtime_parameters("XPU", signature),
        (_custom.array_metadata(x)._replace(device="xpu"), 3))

    assert array == (x.data_ptr(), (4, 3), (6, 2))
    assert count == struct.pack("i", 3)


def test_runtime_arguments_reject_devices_other_than_xpu0():
    x = torch.empty(16)
    signature = build_signature(_scaled, (x, 3, 16))

    with pytest.raises(ValueError, match="xpu:1, but kernels run on xpu:0"):
        xpu._runtime_arguments(
            xpu.runtime_parameters("XPU", signature),
            (_custom.array_metadata(x)._replace(device="xpu", device_index=1), 3))


def test_runtime_arguments_reject_host_tensors():
    arguments = (torch.empty(16), 3, 16)
    signature = build_signature(_scaled, arguments)

    with pytest.raises(ValueError, match="expected 'xpu'"):
        xpu._runtime_arguments(xpu.runtime_parameters("XPU", signature),
                               _runtime(_scaled, arguments))
