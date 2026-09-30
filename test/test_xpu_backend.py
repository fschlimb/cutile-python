# SPDX-FileCopyrightText: Copyright (c) <2026> Intel Corporation.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch

pytest.importorskip("mlir.ir", reason="needs the MLIR Python bindings")
pytest.importorskip("cuda.tile._level_zero", reason="needs the Level Zero extension")

import cuda.tile as ct  # noqa: E402
from cuda.tile._backend import xpu  # noqa: E402
from cuda.tile._backend._signature import build_signature  # noqa: E402
from mlir import ir  # noqa: E402


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
    xpu._compile_cache_key_cached.cache_clear()


def test_compile_cache_key_uses_effective_options(monkeypatch):
    _mock_toolchain(monkeypatch)

    with xpu.compile_options({"wg_m": 128, "wg_n": 64, "unused": 1}):
        first = xpu.compile_cache_key()
    with xpu.compile_options({"wg_m": 128, "wg_n": 64, "unused": 2}):
        second = xpu.compile_cache_key()

    assert first is not None
    assert first == second


def test_compile_cache_key_changes_with_compiler_options(monkeypatch):
    _mock_toolchain(monkeypatch)

    with xpu.compile_options({"wg_m": 128, "wg_n": 64}):
        base = xpu.compile_cache_key()
    with xpu.compile_options({
            "wg_m": 128, "wg_n": 64, "block_threads": 256}):
        different_block = xpu.compile_cache_key()
    with xpu.compile_options({
            "wg_m": 128, "wg_n": 64,
            "large_register_file": False}):
        different_registers = xpu.compile_cache_key()

    assert base != different_block
    assert base != different_registers


def test_options_are_validated_and_normalized():
    options = xpu.normalize_options(
        {"wg_m": 128, "wg_n": 64, "block_threads": 256})

    assert options.block == (256, 1, 1)
    with pytest.raises(TypeError, match="must be a bool"):
        xpu.normalize_options(
            {"wg_m": 128, "wg_n": 64, "assume_in_bounds": "false"})
    with pytest.raises(ValueError, match="xegpu_op_level"):
        xpu.normalize_options(
            {"wg_m": 128, "wg_n": 64, "xegpu_op_level": "grid"})


def test_compile_passes_normalized_options_to_the_toolchain(monkeypatch):
    observed = {}

    def run_tool(backend, argv, bytecode):
        observed["argv"] = argv
        return b"module {}"

    def xeas(mlir, **options):
        observed["xeas"] = options
        return b"binary"

    monkeypatch.setattr(xpu, "resolve_tool", lambda *_args: "/tileir-to-mlir")
    monkeypatch.setattr(xpu, "run_tool", run_tool)
    monkeypatch.setattr(xpu, "xeas", xeas)

    with xpu.compile_options({"wg_m": 64, "wg_n": 64, "block_threads": 256,
                              "assume_in_bounds": True}):
        xpu.compile_tileir(
            b"bytecode", symbol="kernel", sm_arch=xpu.sm_arch, signature=None)

    assert "known-block-size=256,1,1" in observed["argv"][1]
    assert "assume-in-bounds=true" in observed["argv"][1]
    assert observed["xeas"]["chip"] == "bmg"


def test_compile_cache_key_requires_ocloc(monkeypatch):
    _mock_toolchain(monkeypatch)
    monkeypatch.setattr(xpu.shutil, "which", lambda name: None)

    with xpu.compile_options({"wg_m": 128, "wg_n": 64}):
        assert xpu.compile_cache_key() is None


def test_compile_tileir_provides_mlir_context(monkeypatch):
    def xeas(mlir, **_options):
        assert ir.Context.current is not None
        return b"binary"

    monkeypatch.setattr(xpu, "resolve_tool", lambda *_args: "/tileir-to-mlir")
    monkeypatch.setattr(xpu, "run_tool", lambda *_args: b"module {}")
    monkeypatch.setattr(xpu, "xeas", xeas)

    with xpu.compile_options({"wg_m": 64, "wg_n": 64}):
        binary = xpu.compile_tileir(
            b"bytecode", symbol="kernel", sm_arch=xpu.sm_arch, signature=None)

    assert binary == b"binary"


def test_launch_uses_compiled_symbol_and_option_block(monkeypatch):
    launches = []
    monkeypatch.setattr(
        ct, "compile_kernel", lambda kernel, sig: (b"binary", sig.symbol))
    monkeypatch.setattr(
        xpu, "launch_level_zero_module_kernel",
        lambda *args: launches.append(args))

    options = {"wg_m": 64, "wg_n": 32, "sg_m": 32, "sg_n": 16}
    with xpu.compile_options(options):
        xpu.launch(None, (2,), _counter, (7, 4))

    [(binary, symbol, arguments, grid, block)] = launches
    assert binary == b"binary"
    assert symbol == build_signature(_counter, (7, 4)).symbol
    assert arguments == [np.int32(7).tobytes()]
    assert grid == (2, 1, 1)
    assert block == (64, 1, 1)


def test_runtime_arguments_reject_host_tensors():
    arguments = (torch.empty(16), 3, 16)
    signature = build_signature(_scaled, arguments)

    with pytest.raises(ValueError, match="expected an XPU tensor"):
        xpu._runtime_kernel_args(signature, arguments)
