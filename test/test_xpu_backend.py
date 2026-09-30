# SPDX-FileCopyrightText: Copyright (c) <2026> Intel Corporation. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch

pytest.importorskip("mlir.ir", reason="needs the MLIR Python bindings")
pytest.importorskip("cuda.tile._level_zero", reason="needs the Level Zero extension")

import cuda.tile as ct  # noqa: E402
from cuda.tile import _backend  # noqa: E402
from cuda.tile._backend import xpu  # noqa: E402
from cuda.tile._backend._signature import build_signature  # noqa: E402
from mlir import ir  # noqa: E402

_OPTIONS = {"wg_m": 128, "wg_n": 64}


@ct.kernel
def _scaled(x, count: int, TILE: ct.Constant[int]):
    pass


@ct.kernel
def _counter(count: int, TILE: ct.Constant[int]):
    pass


def _mock_toolchain(monkeypatch):
    monkeypatch.setattr(xpu, "resolve_tool", lambda *_args: "/tileir-to-mlir")
    monkeypatch.setattr(xpu.shutil, "which", lambda name: f"/{name}")
    monkeypatch.setattr(xpu, "_ocloc_identity", lambda path: ("1.0", "igc"))
    monkeypatch.setattr(xpu, "_mlir_fingerprint", lambda: {"mlir": "1"})
    monkeypatch.setattr(
        xpu, "file_fingerprint", lambda path: f"fingerprint:{path}")
    xpu._compiler_identity.cache_clear()


def _identity(**options):
    return xpu.compiler_identity(xpu.normalize_options({**_OPTIONS, **options}))


def test_xpu_module_is_a_backend():
    _backend.set_backend("xpu")
    try:
        assert _backend.get_backend() is xpu
    finally:
        _backend.clear_backend()


def test_options_ignore_unknown_keys():
    assert (xpu.normalize_options({**_OPTIONS, "unused": 1})
            == xpu.normalize_options({**_OPTIONS, "unused": 2}))


def test_compiler_identity_changes_with_compiler_options(monkeypatch):
    _mock_toolchain(monkeypatch)

    base = _identity()

    assert base is not None
    assert base != _identity(block_threads=256)
    assert base != _identity(large_register_file=False)


def test_compiler_identity_requires_ocloc(monkeypatch):
    _mock_toolchain(monkeypatch)
    monkeypatch.setattr(xpu.shutil, "which", lambda name: None)

    assert _identity() is None


def test_options_are_validated_and_normalized():
    options = xpu.normalize_options({**_OPTIONS, "block_threads": 256})

    assert options.block == (256, 1, 1)
    with pytest.raises(TypeError, match="must be a bool"):
        xpu.normalize_options({**_OPTIONS, "assume_in_bounds": "false"})
    with pytest.raises(ValueError, match="xegpu_op_level"):
        xpu.normalize_options({**_OPTIONS, "xegpu_op_level": "grid"})


def test_compile_passes_options_to_the_toolchain_in_an_mlir_context(monkeypatch):
    observed = {}

    def run_tool(backend, argv, bytecode):
        observed["argv"] = argv
        return b"module {}"

    def xeas(mlir, **options):
        assert ir.Context.current is not None
        observed["xeas"] = options
        return b"binary"

    monkeypatch.setattr(xpu, "resolve_tool", lambda *_args: "/tileir-to-mlir")
    monkeypatch.setattr(xpu, "run_tool", run_tool)
    monkeypatch.setattr(xpu, "xeas", xeas)
    options = xpu.normalize_options(
        {**_OPTIONS, "block_threads": 256, "assume_in_bounds": True})

    assert xpu.compile(b"bytecode", None, options) == b"binary"
    assert "known-block-size=256,1,1" in observed["argv"][1]
    assert "assume-in-bounds=true" in observed["argv"][1]
    assert observed["xeas"]["chip"] == "bmg"


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
    with _backend.compile_options(_OPTIONS):
        xpu.launch(loaded, None, (2,), (7, 4))
        xpu.launch(loaded, None, (3,), (8, 4))

    [kernel] = kernels
    assert (kernel.binary, kernel.symbol) == (b"binary", signature.symbol)
    assert kernel.launches == [
        ([np.int32(7).tobytes()], (2, 1, 1), (64, 1, 1)),
        ([np.int32(8).tobytes()], (3, 1, 1), (64, 1, 1)),
    ]


def test_runtime_arguments_reject_host_tensors():
    arguments = (torch.empty(16), 3, 16)
    signature = build_signature(_scaled, arguments)

    with pytest.raises(ValueError, match="expected an XPU tensor"):
        xpu._runtime_kernel_args(signature, arguments)
