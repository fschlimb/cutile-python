# SPDX-FileCopyrightText: Copyright (c) <2026> Intel Corporation. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
from types import SimpleNamespace
import warnings

import numpy as np
import pytest

import cuda.tile as ct
from cuda.tile._backend import _custom, _native, cpu
from cuda.tile._backend._signature import build_signature, constant_parameters
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
def _waves(x, out, TILE: ct.Constant[int]):
    value = ct.load(x, index=(ct.bid(0),), shape=(TILE,))
    ct.store(out, index=(ct.bid(0),), tile=ct.sin(value) + ct.cos(value))


@ct.kernel
def _empty():
    pass


@ct.kernel
def _add_one(x, out, TILE: ct.Constant[int]):
    value = ct.load(x, index=(ct.bid(0),), shape=(TILE,))
    ct.store(out, index=(ct.bid(0),), tile=value + 1)


def _bytecode(kernel, arguments, backend):
    return _custom.compile_bytecode(kernel, backend, build_signature(kernel, arguments))


@pytest.fixture
def native(monkeypatch):
    module = pytest.importorskip("cuda.tile._backend._compiler")
    monkeypatch.setattr(default_tile_context.config, "compiler_timeout_sec", None)
    return module


@pytest.mark.parametrize("concurrent", [False, True])
def test_cpu_compiles_and_links_without_external_tools(native, monkeypatch, concurrent):
    if native.backend != "cpu":
        pytest.skip("requires the CPU compiler")
    monkeypatch.setenv("PATH", "")
    monkeypatch.setattr(
        _native.subprocess, "run",
        lambda *args, **kwargs: pytest.fail("unexpected compiler subprocess"))
    x = np.linspace(-1, 1, 128, dtype=np.float32)
    out = np.empty_like(x)
    arguments = (x, out, 16)
    bytecode, signature = _bytecode(_waves, arguments, cpu)
    options = cpu.Options(False)
    identity = cpu.compiler_identity(options)
    assert isinstance(identity, str) and len(identity) == 64
    if concurrent:
        with ThreadPoolExecutor(max_workers=4) as pool:
            binaries = list(pool.map(lambda _: cpu.compile(bytecode, signature, options), range(4)))
    else:
        binaries = [cpu.compile(bytecode, signature, options) for _ in range(3)]
    constants = constant_parameters(_waves)
    runtime = _custom._runtime_arguments(
        arguments, constants, _custom._describe(constants, arguments))
    for binary in binaries:
        assert binary.startswith(b"\x7fELF")
        out.fill(np.nan)
        loaded = cpu.load(binary, signature, options)
        cpu.launch(loaded, None, (8,), runtime)
        np.testing.assert_allclose(out, np.sin(x) + np.cos(x), rtol=2e-6, atol=2e-6)


def test_invalid_bytecode_does_not_poison_compiler(native, tmp_path):
    if native.backend != "cpu":
        pytest.skip("requires the CPU compiler")
    with pytest.raises(ValueError, match="invalid TileIR"):
        native.compile_cpu(b"invalid", cpu._lowering_pipeline(), False, str(tmp_path))
    bytecode, signature = _bytecode(_empty, (), cpu)
    assert cpu.compile(bytecode, signature, cpu.Options(False)).startswith(b"\x7fELF")


def test_native_mlir_errors_include_diagnostics(native):
    if native.backend != "cpu":
        pytest.skip("requires the CPU compiler")
    with pytest.raises(RuntimeError, match="parsing MLIR"):
        native.lower_cpu_mlir(b"not MLIR", cpu._lowering_pipeline())


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
def test_native_rejects_unsupported_rounding(native, monkeypatch, mode, timeout):
    kernel = _divide_with_rounding(mode)
    args = tuple(np.ones(1, dtype=np.float32) for _ in range(3))
    bytecode, signature = _bytecode(kernel, args, cpu)
    monkeypatch.setattr(default_tile_context.config, "compiler_timeout_sec", timeout)
    with pytest.raises(ValueError, match=f"unsupported rounding mode {mode.value}"):
        if native.backend == "cpu":
            cpu.compile(bytecode, signature, cpu.Options(False))
        else:
            _native.invoke("xpu", "compile_xpu", bytecode, (32, 1, 1), False,
                           "workgroup", True, "bmg", False)


def test_native_cpu_default_rounding_still_executes(native):
    if native.backend != "cpu":
        pytest.skip("requires the CPU compiler")
    kernel = _divide_with_rounding(ct.RoundingMode.RN)
    x, y = np.array([1.0], dtype=np.float32), np.array([3.0], dtype=np.float32)
    out = np.zeros_like(x)
    bytecode, signature = _bytecode(kernel, (x, y, out), cpu)
    loaded = cpu.load(cpu.compile(bytecode, signature, cpu.Options(False)),
                      signature, cpu.Options(False))
    runtime = _custom._runtime_arguments(
        (x, y, out), (False,) * 3, _custom._describe((False,) * 3, (x, y, out)))
    cpu.launch(loaded, None, (1,), runtime)
    np.testing.assert_array_equal(out, x / y)


def test_cpu_compiles_in_forked_child_after_parent_compiled(native):
    if native.backend != "cpu":
        pytest.skip("requires the CPU compiler")
    x = np.zeros(128, dtype=np.float32)
    bytecode, signature = _bytecode(_waves, (x, np.empty_like(x), 16), cpu)
    options = cpu.Options(False)
    cpu.compile(bytecode, signature, options)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        pid = os.fork()
    if pid == 0:
        status = 1
        try:
            signal.alarm(60)
            status = 0 if cpu.compile(bytecode, signature, options).startswith(b"\x7fELF") else 2
        finally:
            os._exit(status)
    _, status = os.waitpid(pid, 0)
    assert os.waitstatus_to_exitcode(status) == 0


_MALLOC_SHIM = b"""
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stddef.h>
void *malloc(size_t size) {
  static void *(*next)(size_t);
  if (!next)
    next = (void *(*)(size_t))dlsym(RTLD_NEXT, "malloc");
  return next(size);
}
"""

_PRELOADED_COMPILE = """\
import sys, tempfile
from cuda.tile._backend import _compiler
from cuda.tile._backend import cpu
with tempfile.TemporaryDirectory() as directory:
    sys.stdout.buffer.write(_compiler.compile_cpu(
        sys.stdin.buffer.read(), cpu._lowering_pipeline(), False, directory))
"""


def test_cpu_links_system_libraries_despite_preloaded_allocator(native, tmp_path):
    if native.backend != "cpu":
        pytest.skip("requires the CPU compiler")
    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("building the malloc shim requires cc")
    shim = tmp_path / "libmalloc-shim.so"
    subprocess.run([compiler, "-shared", "-fPIC", "-x", "c", "-", "-o", str(shim), "-ldl"],
                   input=_MALLOC_SHIM, check=True)
    x = np.zeros(128, dtype=np.float32)
    bytecode, _ = _bytecode(_waves, (x, np.empty_like(x), 16), cpu)
    binary = subprocess.run(
        [sys.executable, "-c", _PRELOADED_COMPILE], input=bytecode,
        env={**os.environ, "LD_PRELOAD": str(shim)},
        capture_output=True, check=True, timeout=300).stdout
    assert binary.startswith(b"\x7fELF")
    assert b"libm.so.6" in binary
    assert str(shim).encode() not in binary


def test_native_worker_compiles_bytecode(native, monkeypatch):
    if native.backend == "cpu":
        backend, options = cpu, cpu.Options(False)
    else:
        from cuda.tile._backend import xpu
        backend = xpu
        options = xpu.normalize_options({"block_threads": 32})
    bytecode, signature = _bytecode(_empty, (), backend)
    monkeypatch.setattr(default_tile_context.config, "compiler_timeout_sec", 30)
    assert backend.compile(bytecode, signature, options).startswith(b"\x7fELF")


@pytest.mark.parametrize("timeout", [None, 30])
def test_native_errors_are_consistent_with_and_without_worker(
        native, monkeypatch, tmp_path, timeout):
    if native.backend != "cpu":
        pytest.skip("requires the CPU compiler")
    monkeypatch.setattr(default_tile_context.config, "compiler_timeout_sec", timeout)
    with pytest.raises(ValueError, match="invalid TileIR bytecode") as caught:
        _native.invoke("cpu", "compile_cpu", b"invalid", cpu._lowering_pipeline(),
                       False, str(tmp_path))
    assert "Traceback" not in str(caught.value)
    bytecode, _ = _bytecode(_empty, (), cpu)
    with pytest.raises(RuntimeError, match="parsing the CPU pipeline") as caught:
        _native.invoke("cpu", "compile_cpu", bytecode, "builtin.module(not-a-pass)",
                       False, str(tmp_path))
    assert "Traceback" not in str(caught.value)


@pytest.mark.parametrize("large_register_file", [False, True])
@pytest.mark.parametrize("with_memory", [False, True])
def test_xpu_compiles_bytecode_without_tools_or_mlir_python(
        native, monkeypatch, tmp_path, large_register_file, with_memory):
    if native.backend != "xpu":
        pytest.skip("requires the XPU compiler")
    from cuda.tile._backend import xpu

    monkeypatch.setenv("PATH", "")
    monkeypatch.setattr(
        _native.subprocess, "run",
        lambda *args, **kwargs: pytest.fail("unexpected compiler subprocess"))
    monkeypatch.chdir(tmp_path)
    arguments = (np.zeros(256, dtype=np.float32), np.empty(256, dtype=np.float32), 256)
    bytecode, signature = _bytecode(
        _add_one if with_memory else _empty, arguments if with_memory else (), xpu)
    options = xpu.normalize_options({
        "block_threads": 32,
        "large_register_file": large_register_file})
    identity = xpu.compiler_identity(options)
    assert isinstance(identity, str) and len(identity) == 64
    assert xpu.compiler_identity(options) == identity
    assert not (tmp_path / "IGC_REVISION").exists()
    assert xpu.compile(bytecode, signature, options).startswith(b"\x7fELF")


def test_xpu_large_register_file_reaches_offline_compiler(native):
    if native.backend != "xpu":
        pytest.skip("requires the XPU compiler")
    from cuda.tile._backend import xpu

    arguments = (np.zeros(256, dtype=np.float32), np.empty(256, dtype=np.float32), 256)
    bytecode, signature = _bytecode(_add_one, arguments, xpu)
    default, large = (xpu.compile(bytecode, signature, xpu.normalize_options({
        "block_threads": 32, "large_register_file": large}))
        for large in (False, True))
    assert default != large


def test_xpu_compiles_concurrently(native, monkeypatch):
    if native.backend != "xpu":
        pytest.skip("requires the XPU compiler")
    from cuda.tile._backend import xpu

    monkeypatch.setenv("PATH", "")
    arguments = (np.zeros(256, dtype=np.float32), np.empty(256, dtype=np.float32), 256)
    bytecode, signature = _bytecode(_add_one, arguments, xpu)
    options = xpu.normalize_options({"block_threads": 32})
    with ThreadPoolExecutor(max_workers=4) as pool:
        binaries = list(pool.map(lambda _: xpu.compile(bytecode, signature, options), range(4)))
    assert all(binary.startswith(b"\x7fELF") for binary in binaries)


@ct.kernel
def _unused_array(unused, count: int, out):
    ct.store(out, index=(0,), tile=ct.full((32,), count, ct.float32))


@ct.kernel
def _array_shape_only(unused, count: int, out):
    ct.store(out, index=(0,), tile=ct.full((32,), unused.shape[0] + count, ct.float32))


def _xpu_argument_sizes(binary):
    return {int(index): int(size) for size, index in re.findall(
        rb"size:\s+(\d+)\s+arg_index:\s+(\d+)", binary)}


@pytest.mark.parametrize("shape", [(), (13,), (3, 5)])
@pytest.mark.parametrize("shape_only", [False, True])
def test_xpu_unused_arrays_keep_ranked_abi(native, capfd, shape, shape_only):
    if native.backend != "xpu":
        pytest.skip("requires the XPU compiler")
    if shape_only and not shape:
        pytest.skip("rank-zero arrays have no shape axis")
    from cuda.tile._backend import xpu

    kernel = _array_shape_only if shape_only else _unused_array
    args = (np.zeros(shape, dtype=np.float32), 7, np.zeros(32, dtype=np.float32))
    bytecode, _ = _bytecode(kernel, args, xpu)
    binary = native.compile_xpu(bytecode, (32, 1, 1), False, "workgroup", True,
                                "bmg", True, (len(shape), -1, 1))
    assert binary.startswith(b"\x7fELF")
    lowered = capfd.readouterr().err
    assert "memref<*" not in lowered
    signature = next(line for line in lowered.splitlines() if "gpu.func @" in line)
    assert signature.count("%arg") == 3
    assert signature.count("memref<") == 2
    expected = [8] * (3 + 2 * len(shape)) + [4] + [8] * 5
    assert _xpu_argument_sizes(binary) == dict(enumerate(expected))


@ct.kernel
def _xpu_scalars(out, flag: bool, scale: float, count: int, wide: ct.ScalarInt64):
    value = scale + count if flag else -scale
    ct.store(out, index=(0,), tile=ct.full((32,), value, ct.float32))


def test_xpu_scalar_sizes_match_the_native_abi(native):
    if native.backend != "xpu":
        pytest.skip("requires the XPU compiler")
    from cuda.tile._backend import xpu

    args = (np.zeros(32, dtype=np.float32), True, 1.5, -7, 1 << 40)
    bytecode, signature = _bytecode(_xpu_scalars, args, xpu)
    binary = xpu.compile(bytecode, signature, xpu.normalize_options({"block_threads": 32}))
    assert _xpu_argument_sizes(binary) == dict(enumerate([8] * 5 + [1, 4, 4, 8]))
