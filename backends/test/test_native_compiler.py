# SPDX-FileCopyrightText: Copyright (c) <2026> Intel Corporation. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

import cuda.tile as ct
from cuda.tile._backends import _native, cpu
from cuda.tile._cext import default_tile_context
from cuda.tile._exception import TileCompilerTimeoutError


def test_native_compiler_rejects_wrong_variant(monkeypatch):
    monkeypatch.setattr(
        _native.importlib, "import_module", lambda _: SimpleNamespace(backend="cpu"))
    with pytest.raises(RuntimeError, match="targets cpu, not xpu"):
        _native.compiler("xpu")


def test_default_compilation_stays_in_process(monkeypatch):
    monkeypatch.setattr(
        default_tile_context.config, "compiler_timeout_sec", None)
    monkeypatch.setattr(_native, "compiler", lambda _: SimpleNamespace(
        compile_cpu=lambda data, flag: data + bytes([flag])))
    monkeypatch.setattr(
        _native.subprocess, "run",
        lambda *args, **kwargs: pytest.fail("unexpected compiler subprocess"))
    assert _native.invoke("cpu", "compile_cpu", b"input", True) == b"input\x01"


def test_explicit_timeout_uses_worker_and_reports_diagnostics(monkeypatch, capsys):
    monkeypatch.setattr(default_tile_context.config, "compiler_timeout_sec", 7.5)
    monkeypatch.setattr(_native, "compiler", lambda _: SimpleNamespace(__file__="/compiler"))

    def run(command, **kwargs):
        assert Path(command[0]).is_absolute()
        assert kwargs["timeout"] == 7.5
        assert kwargs["input"] == b"input"
        return SimpleNamespace(returncode=0, stdout=b"binary", stderr=b"warning\n")

    monkeypatch.setattr(_native.subprocess, "run", run)
    assert _native.invoke("cpu", "compile_cpu", b"input", True) == b"binary"
    assert capsys.readouterr().err == "warning\n"


def test_native_timeout_is_reported(monkeypatch):
    monkeypatch.setattr(default_tile_context.config, "compiler_timeout_sec", 0.1)
    monkeypatch.setattr(_native, "compiler", lambda _: SimpleNamespace(__file__="/compiler"))

    def run(command, **kwargs):
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(_native.subprocess, "run", run)
    with pytest.raises(TileCompilerTimeoutError, match="exceeded"):
        _native.invoke("cpu", "compile_cpu", b"input")


def test_worker_failure_is_not_a_binary(monkeypatch):
    monkeypatch.setattr(default_tile_context.config, "compiler_timeout_sec", 1)
    monkeypatch.setattr(_native, "compiler", lambda _: SimpleNamespace(__file__="/compiler"))
    monkeypatch.setattr(_native.subprocess, "run", lambda *args, **kwargs:
                        SimpleNamespace(returncode=-6, stdout=b"", stderr=b"compiler error"))
    with pytest.raises(RuntimeError, match="compiler error"):
        _native.invoke("cpu", "compile_cpu", b"input")


@pytest.mark.parametrize("code, exception", [(2, ValueError), (3, RuntimeError)])
def test_worker_preserves_known_exception_types(monkeypatch, code, exception):
    monkeypatch.setattr(default_tile_context.config, "compiler_timeout_sec", 1)
    monkeypatch.setattr(_native, "compiler", lambda _: SimpleNamespace(__file__="/compiler"))
    monkeypatch.setattr(_native.subprocess, "run", lambda *args, **kwargs:
                        SimpleNamespace(returncode=code, stdout=b"", stderr=b"native error"))
    with pytest.raises(exception) as caught:
        _native.invoke("cpu", "compile_cpu", b"input")
    assert str(caught.value) == "native error"


def test_file_fingerprint_tracks_rebuilds_and_resolves_symlinks(tmp_path):
    path = tmp_path / "compiler.so"
    path.write_bytes(b"first")
    alias = tmp_path / "alias.so"
    alias.symlink_to(path)
    first = _native.file_fingerprint(str(path))
    assert _native.file_fingerprint(str(alias)) == first
    path.write_bytes(b"rebuilt compiler")
    assert _native.file_fingerprint(str(path)) != first


@ct.kernel
def _empty():
    pass


def _divide_with_rounding(mode):
    @ct.kernel
    def divide(x, y, out):
        a = ct.load(x, index=(0,), shape=(1,))
        b = ct.load(y, index=(0,), shape=(1,))
        ct.store(out, index=(0,), tile=ct.truediv(a, b, rounding_mode=mode))

    return divide


@pytest.mark.parametrize("mode", [ct.RoundingMode.RZ, ct.RoundingMode.RM,
                                 ct.RoundingMode.RP])
@pytest.mark.parametrize("timeout", [None, 30])
def test_native_rejects_unsupported_rounding(native, bytecode, monkeypatch, mode, timeout):
    kernel = _divide_with_rounding(mode)
    args = tuple(np.ones(1, dtype=np.float32) for _ in range(3))
    data, signature = bytecode(kernel, args, cpu)
    monkeypatch.setattr(default_tile_context.config, "compiler_timeout_sec", timeout)
    with pytest.raises(ValueError, match=f"unsupported rounding mode {mode.value}"):
        if native.backend == "cpu":
            cpu.compile(data, signature, cpu.Options(False))
        else:
            _native.invoke("xpu", "compile_xpu", data, (32, 1, 1), False,
                           "workgroup", True, "bmg", False)


def test_native_worker_compiles_bytecode(native, bytecode, monkeypatch):
    if native.backend == "cpu":
        backend, options = cpu, cpu.Options(False)
    else:
        from cuda.tile._backends import xpu
        backend = xpu
        options = xpu.normalize_options({"block_threads": 32})
    data, signature = bytecode(_empty, (), backend)
    monkeypatch.setattr(default_tile_context.config, "compiler_timeout_sec", 30)
    assert backend.compile(data, signature, options).startswith(b"\x7fELF")
