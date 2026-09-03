# SPDX-FileCopyrightText: Copyright (c) <2026> Intel Corporation.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from cuda.tile._backend import xpu


def _mock_toolchain(monkeypatch):
    monkeypatch.setattr(xpu, "resolve_tool", lambda *_args: "/tileir-to-mlir")
    monkeypatch.setattr(xpu.shutil, "which", lambda name: f"/{name}")
    monkeypatch.setattr(
        xpu.importlib, "import_module",
        lambda name: SimpleNamespace(__file__="/mlir.so"))
    monkeypatch.setattr(
        xpu, "file_fingerprint", lambda path: f"fingerprint:{path}")


def test_compile_cache_key_uses_effective_options(monkeypatch):
    _mock_toolchain(monkeypatch)

    with xpu.compile_options({"wg_m": 128, "wg_n": 64, "unused": 1}):
        first = xpu.compile_cache_key()
    with xpu.compile_options({"wg_m": 128, "wg_n": 64, "unused": 2}):
        second = xpu.compile_cache_key()

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


def test_compile_cache_key_requires_ocloc(monkeypatch):
    _mock_toolchain(monkeypatch)
    monkeypatch.setattr(xpu.shutil, "which", lambda name: None)

    with xpu.compile_options({"wg_m": 128, "wg_n": 64}):
        assert xpu.compile_cache_key() is None
