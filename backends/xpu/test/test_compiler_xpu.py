# SPDX-FileCopyrightText: Copyright (c) <2026> Intel Corporation. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from concurrent.futures import ThreadPoolExecutor
import re

import numpy as np
import pytest

import cuda.tile as ct
from cuda.tile._backends import _native


@pytest.fixture(autouse=True)
def _xpu_compiler(native):
    if native.backend != "xpu":
        pytest.skip("requires the XPU compiler")


@ct.kernel
def _empty():
    pass


@ct.kernel
def _add_one(x, out, TILE: ct.Constant[int]):
    value = ct.load(x, index=(ct.bid(0),), shape=(TILE,))
    ct.store(out, index=(ct.bid(0),), tile=value + 1)


@pytest.mark.parametrize("large_register_file", [False, True])
@pytest.mark.parametrize("with_memory", [False, True])
def test_xpu_compiles_bytecode_without_tools_or_mlir_python(
        native, bytecode, monkeypatch, tmp_path, large_register_file, with_memory):
    from cuda.tile._backends import xpu

    monkeypatch.setenv("PATH", "")
    monkeypatch.setattr(
        _native.subprocess, "run",
        lambda *args, **kwargs: pytest.fail("unexpected compiler subprocess"))
    monkeypatch.chdir(tmp_path)
    arguments = (np.zeros(256, dtype=np.float32), np.empty(256, dtype=np.float32), 256)
    data, signature = bytecode(
        _add_one if with_memory else _empty, arguments if with_memory else (), xpu)
    options = xpu.normalize_options({
        "block_threads": 32,
        "large_register_file": large_register_file})
    identity = xpu.compiler_identity(options)
    assert isinstance(identity, str) and len(identity) == 64
    assert xpu.compiler_identity(options) == identity
    assert not (tmp_path / "IGC_REVISION").exists()
    assert xpu.compile(data, signature, options).startswith(b"\x7fELF")


def test_xpu_large_register_file_reaches_offline_compiler(native, bytecode):
    from cuda.tile._backends import xpu

    arguments = (np.zeros(256, dtype=np.float32), np.empty(256, dtype=np.float32), 256)
    data, signature = bytecode(_add_one, arguments, xpu)
    default, large = (xpu.compile(data, signature, xpu.normalize_options({
        "block_threads": 32, "large_register_file": large}))
        for large in (False, True))
    assert default != large


def test_xpu_compiles_concurrently(native, bytecode, monkeypatch):
    from cuda.tile._backends import xpu

    monkeypatch.setenv("PATH", "")
    arguments = (np.zeros(256, dtype=np.float32), np.empty(256, dtype=np.float32), 256)
    data, signature = bytecode(_add_one, arguments, xpu)
    options = xpu.normalize_options({"block_threads": 32})
    with ThreadPoolExecutor(max_workers=4) as pool:
        binaries = list(pool.map(lambda _: xpu.compile(data, signature, options), range(4)))
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
def test_xpu_unused_arrays_keep_ranked_abi(native, bytecode, capfd, shape, shape_only):
    from cuda.tile._backends import xpu

    if shape_only and not shape:
        pytest.skip("rank-zero arrays have no shape axis")
    kernel = _array_shape_only if shape_only else _unused_array
    args = (np.zeros(shape, dtype=np.float32), 7, np.zeros(32, dtype=np.float32))
    data, _ = bytecode(kernel, args, xpu)
    binary = native.compile_xpu(data, (32, 1, 1), False, "workgroup", True,
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


def test_xpu_scalar_sizes_match_the_native_abi(native, bytecode):
    from cuda.tile._backends import xpu

    args = (np.zeros(32, dtype=np.float32), True, 1.5, -7, 1 << 40)
    data, signature = bytecode(_xpu_scalars, args, xpu)
    binary = xpu.compile(data, signature, xpu.normalize_options({"block_threads": 32}))
    assert _xpu_argument_sizes(binary) == dict(enumerate([8] * 5 + [1, 4, 4, 8]))
