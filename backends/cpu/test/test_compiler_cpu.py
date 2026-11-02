# SPDX-FileCopyrightText: Copyright (c) <2026> Intel Corporation. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from concurrent.futures import ThreadPoolExecutor
import os
import shutil
import signal
import subprocess
import sys
import warnings

import numpy as np
import pytest

import cuda.tile as ct
from cuda.tile._backends import _custom, _native, cpu
from cuda.tile._backends._signature import constant_parameters
from cuda.tile._cext import default_tile_context


@pytest.fixture(autouse=True)
def _cpu_compiler(native):
    if native.backend != "cpu":
        pytest.skip("requires the CPU compiler")


@ct.kernel
def _waves(x, out, TILE: ct.Constant[int]):
    value = ct.load(x, index=(ct.bid(0),), shape=(TILE,))
    ct.store(out, index=(ct.bid(0),), tile=ct.sin(value) + ct.cos(value))


@ct.kernel
def _empty():
    pass


@pytest.mark.parametrize("concurrent", [False, True])
def test_cpu_compiles_and_links_without_external_tools(native, bytecode, monkeypatch, concurrent):
    monkeypatch.setenv("PATH", "")
    monkeypatch.setattr(
        _native.subprocess, "run",
        lambda *args, **kwargs: pytest.fail("unexpected compiler subprocess"))
    x = np.linspace(-1, 1, 128, dtype=np.float32)
    out = np.empty_like(x)
    arguments = (x, out, 16)
    data, signature = bytecode(_waves, arguments, cpu)
    options = cpu.Options(False)
    identity = cpu.compiler_identity(options)
    assert isinstance(identity, str) and len(identity) == 64
    if concurrent:
        with ThreadPoolExecutor(max_workers=4) as pool:
            binaries = list(pool.map(lambda _: cpu.compile(data, signature, options), range(4)))
    else:
        binaries = [cpu.compile(data, signature, options) for _ in range(3)]
    constants = constant_parameters(_waves)
    runtime = _custom._runtime_arguments(
        arguments, constants, _custom._describe(constants, arguments))
    for binary in binaries:
        assert binary.startswith(b"\x7fELF")
        out.fill(np.nan)
        loaded = cpu.load(binary, signature, options)
        cpu.launch(loaded, None, (8,), runtime)
        np.testing.assert_allclose(out, np.sin(x) + np.cos(x), rtol=2e-6, atol=2e-6)


def test_invalid_bytecode_does_not_poison_compiler(native, bytecode, tmp_path):
    with pytest.raises(ValueError, match="invalid TileIR"):
        native.compile_cpu(b"invalid", cpu._TRANSFORM_SCHEDULE, False, str(tmp_path))
    data, signature = bytecode(_empty, (), cpu)
    assert cpu.compile(data, signature, cpu.Options(False)).startswith(b"\x7fELF")


def test_native_mlir_errors_include_diagnostics(native):
    with pytest.raises(RuntimeError, match="parsing MLIR"):
        native.lower_cpu_mlir(b"not MLIR", cpu._TRANSFORM_SCHEDULE)


@ct.kernel
def _divide(x, y, out):
    a = ct.load(x, index=(0,), shape=(1,))
    b = ct.load(y, index=(0,), shape=(1,))
    ct.store(out, index=(0,), tile=ct.truediv(a, b, rounding_mode=ct.RoundingMode.RN))


def test_native_cpu_default_rounding_still_executes(native, bytecode):
    x, y = np.array([1.0], dtype=np.float32), np.array([3.0], dtype=np.float32)
    out = np.zeros_like(x)
    data, signature = bytecode(_divide, (x, y, out), cpu)
    loaded = cpu.load(cpu.compile(data, signature, cpu.Options(False)),
                      signature, cpu.Options(False))
    runtime = _custom._runtime_arguments(
        (x, y, out), (False,) * 3, _custom._describe((False,) * 3, (x, y, out)))
    cpu.launch(loaded, None, (1,), runtime)
    np.testing.assert_array_equal(out, x / y)


def test_cpu_compiles_in_forked_child_after_parent_compiled(native, bytecode):
    x = np.zeros(128, dtype=np.float32)
    data, signature = bytecode(_waves, (x, np.empty_like(x), 16), cpu)
    options = cpu.Options(False)
    cpu.compile(data, signature, options)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        pid = os.fork()
    if pid == 0:
        status = 1
        try:
            signal.alarm(60)
            status = 0 if cpu.compile(data, signature, options).startswith(b"\x7fELF") else 2
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
from cuda.tile._backends import _compiler
from cuda.tile._backends import cpu
with tempfile.TemporaryDirectory() as directory:
    sys.stdout.buffer.write(_compiler.compile_cpu(
        sys.stdin.buffer.read(), cpu._TRANSFORM_SCHEDULE, False, directory))
"""


def test_cpu_links_system_libraries_despite_preloaded_allocator(native, bytecode, tmp_path):
    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("building the malloc shim requires cc")
    shim = tmp_path / "libmalloc-shim.so"
    subprocess.run([compiler, "-shared", "-fPIC", "-x", "c", "-", "-o", str(shim), "-ldl"],
                   input=_MALLOC_SHIM, check=True)
    x = np.zeros(128, dtype=np.float32)
    data, _ = bytecode(_waves, (x, np.empty_like(x), 16), cpu)
    binary = subprocess.run(
        [sys.executable, "-c", _PRELOADED_COMPILE], input=data,
        env={**os.environ, "LD_PRELOAD": str(shim)},
        capture_output=True, check=True, timeout=300).stdout
    assert binary.startswith(b"\x7fELF")
    assert b"libm.so.6" in binary
    assert str(shim).encode() not in binary


@pytest.mark.parametrize("timeout", [None, 30])
def test_native_errors_are_consistent_with_and_without_worker(
        native, bytecode, monkeypatch, tmp_path, timeout):
    monkeypatch.setattr(default_tile_context.config, "compiler_timeout_sec", timeout)
    with pytest.raises(ValueError, match="invalid TileIR bytecode") as caught:
        _native.invoke("cpu", "compile_cpu", b"invalid", cpu._TRANSFORM_SCHEDULE,
                       False, str(tmp_path))
    assert "Traceback" not in str(caught.value)
    data, _ = bytecode(_empty, (), cpu)
    bad_schedule = tmp_path / "invalid-schedule.mlir"
    bad_schedule.write_text("not MLIR")
    with pytest.raises(RuntimeError, match="running the pass pipeline") as caught:
        _native.invoke("cpu", "compile_cpu", data, str(bad_schedule),
                       False, str(tmp_path))
    assert "Traceback" not in str(caught.value)
