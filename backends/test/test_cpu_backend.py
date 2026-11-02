# SPDX-FileCopyrightText: Copyright (c) <2026> Intel Corporation. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import ctypes
import importlib.util
from importlib.resources import files
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
from types import SimpleNamespace
from typing import Annotated

import cuda.tile as ct
from cuda.tile import _backend
from cuda.tile._backend import _custom, cpu
from cuda.tile._backend._signature import build_signature, constant_parameters, runtime_parameters
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


def _runtime(kernel, args):
    """The runtime arguments ``ct.launch`` passes to the backend."""
    constants = constant_parameters(kernel)
    return _custom._runtime_arguments(
        args, constants, _custom._describe(constants, args))


def _parameters(kernel, args):
    return runtime_parameters("CPU", build_signature(kernel, args))


def test_signature_flattens_arrays_and_omits_constants():
    arguments = _arguments()
    parameters = _parameters(_vector_add, arguments)

    values = cpu._flatten_arguments(parameters, _runtime(_vector_add, arguments))

    assert cpu._argument_formats(parameters) == ["P", "i", "i"] * 3
    assert values[0] == arguments[0].ctypes.data
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
    device_tensor = torch.empty(64, dtype=torch.float32, device="meta")

    with pytest.raises(ValueError, match="expected 'cpu'"):
        cpu._flatten_arguments(_parameters(_vector_add, arguments),
                               _runtime(_vector_add, (device_tensor, *arguments[1:])))


def test_flatten_arguments_rejects_mismatched_kinds():
    arguments = _arguments()
    runtime = _runtime(_vector_add, arguments)

    with pytest.raises(TypeError, match="expected a 1-D array"):
        cpu._flatten_arguments(_parameters(_vector_add, arguments), (1, *runtime[1:]))
    with pytest.raises(ValueError):
        cpu._flatten_arguments(_parameters(_vector_add, arguments), runtime[:2])


def test_signature_rejects_array_lists():
    with pytest.raises(TypeError, match="only supported on CUDA"):
        build_signature(_scalar_and_array, ([np.zeros(4)], True, 1.0, 1))


def test_backend_package_contains_lowering_schedule():
    schedule = files(cpu).joinpath("cpu_schedule.mlir")
    assert str(schedule) == cpu._TRANSFORM_SCHEDULE
    for entry_point in ("__transform_nano", "__transform_nano_x86", "__transform_nano_amx"):
        assert f"transform.named_sequence @{entry_point}(" in schedule.read_text()


@pytest.mark.usefixtures("native_cpu")
def test_native_compiler_selects_transform_entry_point():
    if cpu.platform.machine() != "x86_64":
        assert cpu._transform_entry_point() == "__transform_nano"
    elif cpu._amx_enabled():
        assert cpu._transform_entry_point() == "__transform_nano_amx"
    else:
        assert cpu._transform_entry_point() == "__transform_nano_x86"


@pytest.mark.usefixtures("native_cpu")
def test_native_lowering_pipeline_builds_pass_manager():
    native = cpu.native_compiler("cpu")
    source = b"""
module {
  func.func @identity(%value: i32) -> i32 {
    %result = arith.addi %value, %value : i32
    return %result : i32
  }
}
"""
    llvm_ir = native.lower_cpu_mlir(
        source, cpu._TRANSFORM_SCHEDULE, "__transform_nano")
    assert b"define i32 @identity" in llvm_ir


def test_backend_registration_accepts_builtin_cpu():
    _backend.set_backend("cpu")
    try:
        assert _backend.get_backend() is cpu
    finally:
        _backend.set_backend(None)


def test_compiler_identity_tracks_toolchain_cpu_and_compile_options(monkeypatch):
    monkeypatch.setattr(
        cpu, "native_compiler", lambda _: SimpleNamespace(__file__="/compiler"))
    monkeypatch.setattr(
        cpu, "file_fingerprint", lambda path: f"fingerprint:{path}")
    monkeypatch.setattr(cpu, "_host_cpu", lambda: {"flags": "avx2"})
    monkeypatch.setattr(cpu, "_transform_entry_point", lambda: "__transform_nano_x86")

    first = cpu.compiler_identity(cpu.normalize_options({"num_cpu_threads": 1}))
    second = cpu.compiler_identity(cpu.normalize_options({"num_cpu_threads": 8}))
    in_bounds = cpu.compiler_identity(
        cpu.normalize_options({"assume_in_bounds": True}))

    assert first is not None
    assert first == second
    assert first != in_bounds

    monkeypatch.setattr(cpu, "_host_cpu", lambda: {"flags": "avx2 avx512f"})
    assert first != cpu.compiler_identity(cpu.Options(False))
    monkeypatch.setattr(cpu, "_host_cpu", lambda: {"flags": "avx2"})
    # Whether AMX is enabled selects the schedule, so it is part of the identity.
    monkeypatch.setattr(cpu, "_transform_entry_point", lambda: "__transform_nano_amx")
    assert first != cpu.compiler_identity(cpu.Options(False))
    monkeypatch.setattr(cpu, "_transform_entry_point", lambda: "__transform_nano_x86")
    monkeypatch.setattr(
        cpu, "file_fingerprint",
        lambda path: (f"rebuilt:{path}" if path == cpu._TRANSFORM_SCHEDULE
                      else f"fingerprint:{path}"))
    assert first != cpu.compiler_identity(cpu.Options(False))
    monkeypatch.setattr(
        cpu, "file_fingerprint", lambda path: f"rebuilt:{path}")
    assert first != cpu.compiler_identity(cpu.Options(False))


def test_compiler_identity_disables_disk_cache_for_missing_native_file(monkeypatch):
    monkeypatch.setattr(
        cpu, "native_compiler", lambda _: SimpleNamespace(__file__="/compiler"))

    def missing_file(*_args):
        raise FileNotFoundError("missing compiler")

    monkeypatch.setattr(cpu, "file_fingerprint", missing_file)
    assert cpu.compiler_identity(cpu.Options(False)) is None


def test_compile_calls_native_pipeline_and_cleans_temporary_files(monkeypatch):
    observed = {}
    monkeypatch.setattr(cpu, "_transform_entry_point", lambda: "__transform_nano")
    monkeypatch.setenv("CUTILE_CPU_DUMP_MLIR", "/dump.mlir")
    monkeypatch.setenv("MLIR_ENABLE_DUMP", "1")

    def invoke(backend, method, bytecode, schedule_path, in_bounds, directory, dump,
               print_ir):
        observed.update(backend=backend, method=method, bytecode=bytecode,
                        schedule_path=schedule_path, in_bounds=in_bounds,
                        directory=directory, dump=dump, print_ir=print_ir)
        assert Path(directory).is_dir()
        return b"shared object"

    monkeypatch.setattr(cpu, "invoke", invoke)
    signature = build_signature(_vector_add, _arguments())
    assert cpu.compile(b"bytecode", signature, cpu.Options(True)) == b"shared object"
    assert observed["backend"] == "cpu"
    assert observed["method"] == "compile_cpu"
    assert observed["bytecode"] == b"bytecode"
    assert observed["schedule_path"] == cpu._TRANSFORM_SCHEDULE
    assert observed["in_bounds"] and observed["print_ir"]
    assert observed["dump"] == "/dump.mlir"
    assert not Path(observed["directory"]).exists()


def test_options_reject_non_bool_flags():
    with pytest.raises(TypeError, match="must be a bool"):
        cpu.normalize_options({"assume_in_bounds": "false"})


def test_options_reject_unknown_keys(monkeypatch):
    with pytest.raises(ValueError, match="unknown options 'assume_inbounds'"):
        cpu.normalize_options({"assume_inbounds": True})
    monkeypatch.setenv("CUTILE_CPU_ASSUME_IN_BOUNDS", "1")
    assert cpu.normalize_options({}) == cpu.Options(assume_in_bounds=False)


def test_load_prepares_native_launcher_once(monkeypatch):
    arguments = _arguments()
    calls = []
    signature = build_signature(_vector_add, arguments)

    class Kernel:
        def __init__(self, path, symbol, size):
            calls.append(("load", Path(path).read_bytes(), symbol, size))

        def launch(self, *args):
            calls.append(("launch", args))

    monkeypatch.setitem(sys.modules, "cuda.tile._backend.cpu._cpu", SimpleNamespace(Kernel=Kernel))

    runtime = _runtime(_vector_add, arguments)
    loaded = cpu.load(b"object", signature, cpu.Options(False))
    cpu.launch(loaded, None, (4,), runtime)
    with ct.backend_options({"num_cpu_threads": 3}):
        cpu.launch(loaded, None, (4,), runtime)

    assert calls[0] == ("load", b"object", signature.symbol, 9 * 8)
    assert isinstance(loaded.kernel, Kernel)
    launches = [call for call in calls if call[0] == "launch"]
    assert len(launches) == 2
    assert launches[0][1][1:] == ((4, 1, 1), 0)
    assert loaded.packer.unpack(launches[0][1][0]) == tuple(
        cpu._flatten_arguments(loaded.parameters, runtime))
    assert [args[2] for _, args in launches] == [0, 3]


@pytest.mark.usefixtures("native_cpu")
@pytest.mark.parametrize("grid", [(-1,), (1 << 31,), (1 << 22,) * 3])
def test_launch_rejects_overflowing_grids(grid):
    out = np.zeros((1, 1, 1), dtype=np.int32)
    compiled = _backend.compile_for_launch(_grid_ids, (out,))
    with pytest.raises(ValueError, match="grid"):
        _backend.launch_compiled(None, grid, compiled, (out,))


@pytest.mark.usefixtures("native_cpu")
@pytest.mark.parametrize("threads", [-1, 1 << 31, 1.5, "3"])
def test_launch_rejects_invalid_thread_counts(threads):
    out = np.zeros((1, 1, 1), dtype=np.int32)
    compiled = _backend.compile_for_launch(_grid_ids, (out,))
    with ct.backend_options({"num_cpu_threads": threads}):
        with pytest.raises((TypeError, ValueError)):
            _backend.launch_compiled(None, (1,), compiled, (out,))


def test_integer_scalars_do_not_silently_wrap():
    with pytest.raises(struct.error):
        cpu._argument_packer(("i",)).pack(1 << 31)
    with pytest.raises(struct.error):
        cpu._argument_packer(("q",)).pack(1 << 63)
    packer = cpu._argument_packer(("q",))
    assert packer.unpack(packer.pack(1 << 40)) == (1 << 40,)


def test_packed_argument_slots_have_fixed_offsets():
    packer = cpu._argument_packer(("?", "i", "f", "q"))
    data = packer.pack(True, -3, 1.5, 1 << 40)
    assert len(data) == 4 * 8
    assert struct.unpack_from("@?", data, 0) == (True,)
    assert struct.unpack_from("@i", data, 8) == (-3,)
    assert struct.unpack_from("@f", data, 16) == (1.5,)
    assert struct.unpack_from("@q", data, 24) == (1 << 40,)


@pytest.fixture
def native_cpu():
    pytest.importorskip("cuda.tile._backend.cpu._cpu")
    compiler = pytest.importorskip("cuda.tile._backend._compiler")
    if compiler.backend != "cpu":
        pytest.skip("requires the native CPU compiler")
    _backend.set_backend("cpu")
    yield
    _backend.set_backend(None)


@pytest.mark.usefixtures("native_cpu")
def test_lowering_pipeline_expands_integer_powers():
    source = b"""
module {
  func.func @cube(%base: f32) -> f32 {
    %power = arith.constant 3 : i32
    %result = math.fpowi %base, %power : f32, i32
    return %result : f32
  }
}
"""
    llvm_ir = cpu.native_compiler("cpu").lower_cpu_mlir(
        source, cpu._TRANSFORM_SCHEDULE)
    assert b"fmul float" in llvm_ir
    assert b"llvm.powi" not in llvm_ir
    assert b"define float @cube" in llvm_ir


@pytest.mark.usefixtures("native_cpu")
def test_lowering_pipeline_converts_loop_carried_amx_tiles():
    source = b"""
module {
  func.func @amx_loop(%a: memref<16x32xbf16>, %b: memref<16x32xbf16>,
                     %c: memref<16x16xf32>, %n: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %zero = x86.amx.tile_zero : !x86.amx.tile<16x16xf32>
    %result = scf.for %i = %c0 to %n step %c1
        iter_args(%acc = %zero) -> (!x86.amx.tile<16x16xf32>) {
      %lhs = x86.amx.tile_load %a[%c0, %c0]
          : memref<16x32xbf16> into !x86.amx.tile<16x32xbf16>
      %rhs = x86.amx.tile_load %b[%c0, %c0]
          : memref<16x32xbf16> into !x86.amx.tile<16x32xbf16>
      %next = x86.amx.tile_mulf %lhs, %rhs, %acc
          : !x86.amx.tile<16x32xbf16>, !x86.amx.tile<16x32xbf16>,
            !x86.amx.tile<16x16xf32>
      scf.yield %next : !x86.amx.tile<16x16xf32>
    }
    x86.amx.tile_store %c[%c0, %c0], %result
        : memref<16x16xf32>, !x86.amx.tile<16x16xf32>
    return
  }
}
"""
    llvm_ir = cpu.native_compiler("cpu").lower_cpu_mlir(
        source, cpu._TRANSFORM_SCHEDULE)
    assert b"unrealized_conversion_cast" not in llvm_ir
    assert b"phi x86_amx" in llvm_ir
    assert b"llvm.x86.tdpbf16ps.internal" in llvm_ir


@pytest.mark.skipif(
    sys.platform != "linux" or "amx_tile" not in cpu._host_cpu().get("flags", "").split(),
    reason="requires Linux with AMX tile support",
)
def test_native_import_enables_amx_tile_data():
    pytest.importorskip("cuda.tile._backend.cpu._cpu")
    assert cpu._amx_enabled()
    subprocess.run([sys.executable, "-c", """
import ctypes
import torch

# Create an OpenMP worker pool before enabling process-wide AMX permission.
torch.ones(64, 64).sum()
import cuda.tile._backend.cpu._cpu

libc = ctypes.CDLL(None, use_errno=True)
permissions = ctypes.c_ulong()
result = libc.syscall(ctypes.c_long(158), ctypes.c_long(0x1022),
                      ctypes.byref(permissions))
assert result == 0, ctypes.get_errno()
assert permissions.value & (1 << 18)
"""], check=True)


@pytest.mark.usefixtures("native_cpu")
@pytest.mark.parametrize("dtype", [
    np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32,
    np.uint64, np.float16, np.float32, np.float64])
def test_native_vector_add(dtype):
    a = np.arange(140, dtype=dtype)[::2]
    b = np.ones(70, dtype=dtype)
    c = np.zeros(140, dtype=dtype)[::2]
    with ct.backend_options({"num_cpu_threads": 3}):
        ct.launch(None, (5,), _vector_add, (a, b, c, 16))
    np.testing.assert_array_equal(c, a + b)


@ct.kernel
def _gather_scatter_add(a, b, out, tile_x: ConstInt, tile_y: ConstInt):
    x = ct.bid(0) * tile_x + ct.arange(tile_x, dtype=ct.int32)
    y = ct.bid(1) * tile_y + ct.arange(tile_y, dtype=ct.int32)
    indices = (x[:, None], y[None, :])
    value = ct.gather(a, indices) + ct.gather(b, indices)
    ct.scatter(out, indices, value)


@pytest.mark.usefixtures("native_cpu")
@pytest.mark.parametrize("tile", [(1, 16), (2, 8)])
def test_native_2d_gather_scatter_with_partial_tiles(tile):
    a = np.arange(7 * 38, dtype=np.float32).reshape(7, 38)[:, ::2]
    b = np.ones(a.shape, dtype=np.float32)
    storage = np.full((7, 38), -1, dtype=np.float32)
    out = storage[:, ::2]
    grid = tuple((size + extent - 1) // extent for size, extent in zip(a.shape, tile))
    with ct.backend_options({"num_cpu_threads": 3}):
        ct.launch(None, grid, _gather_scatter_add, (a, b, out, *tile))
    np.testing.assert_array_equal(out, a + b)
    np.testing.assert_array_equal(storage[:, 1::2], -1)


@ct.kernel
def _grid_ids(out):
    x, y, z = ct.bid(0), ct.bid(1), ct.bid(2)
    value = x + 10 * y + 100 * z
    value += ct.num_blocks(0) * 1000 + ct.num_blocks(1) * 10000
    value += ct.num_blocks(2) * 100000
    ct.store(out, index=(x, y, z), tile=ct.full((1, 1, 1), value, ct.int32))


@pytest.mark.usefixtures("native_cpu")
@pytest.mark.parametrize("grid", [(1,), (4,), (3, 2), (3, 2, 2), (0, 2, 2)])
@pytest.mark.parametrize("threads", [1, 3])
def test_native_grid_ids_and_sizes(grid, threads):
    shape = _backend._util.normalize_dims(grid)
    out = np.full(shape, -1, dtype=np.int32)
    with ct.backend_options({"num_cpu_threads": threads}):
        ct.launch(None, grid, _grid_ids, (out,))
    x, y, z = np.indices(shape)
    expected = x + y * 10 + z * 100 + shape[0] * 1000 + shape[1] * 10000 + shape[2] * 100000
    np.testing.assert_array_equal(out, expected)


@ct.kernel
def _scalar_store(out, flag: bool, scale: float, count: int):
    value = scale * count if flag else -scale
    ct.store(out, index=(0,), tile=ct.full((1,), value, ct.float32))


@pytest.mark.usefixtures("native_cpu")
@pytest.mark.parametrize("flag", [True, False])
def test_native_scalars(flag):
    out = torch.empty(1, dtype=torch.float32)
    ct.launch(None, (1,), _scalar_store, (out, flag, 1.5, -7))
    assert out.item() == (-10.5 if flag else -1.5)


@ct.kernel
def _constant_store(out, value: ct.Constant[float]):
    ct.store(out, index=(0,), tile=ct.full((1,), value, ct.float32))


@pytest.mark.usefixtures("native_cpu")
def test_native_constant_cache_preserves_signed_zero():
    out = np.zeros(1, dtype=np.float32)
    for value in (0.0, -0.0, 0.0):
        ct.launch(None, (1,), _constant_store, (out, value))
        assert np.signbit(out[0]) == np.signbit(value)


@ct.kernel
def _reduce_and_exp(a, out):
    row = ct.bid(0)
    values = ct.load(a, index=(row, 0), shape=(1, 16))
    result = ct.sum(ct.exp(values), axis=1)
    ct.store(out, index=(row,), tile=result)


@pytest.mark.usefixtures("native_cpu")
def test_native_reduction_and_libm():
    a = torch.linspace(-1, 1, 64).reshape(4, 16)
    out = torch.empty(4)
    with ct.backend_options({"num_cpu_threads": 3}):
        ct.launch(None, (4,), _reduce_and_exp, (a, out))
    torch.testing.assert_close(out, a.exp().sum(dim=1))


@pytest.mark.usefixtures("native_cpu")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_native_matmul(dtype):
    path = Path(__file__).resolve().parents[2] / "samples" / "MatMul.py"
    spec = importlib.util.spec_from_file_location("cuda_sample_matmul", path)
    sample = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sample)
    torch.manual_seed(0)
    a = torch.randn((16, 23), dtype=dtype)
    b = torch.randn((23, 21), dtype=dtype)
    out = torch.empty((16, 21), dtype=dtype)
    with ct.backend_options({"num_cpu_threads": 3}):
        ct.launch(None, (6,), sample.matmul_kernel, (a, b, out, 8, 8, 8))
    torch.testing.assert_close(out, a @ b, rtol=2e-2, atol=2e-2)


@pytest.mark.usefixtures("native_cpu")
@pytest.mark.skipif(
    sys.platform != "linux" or "amx_bf16" not in cpu._host_cpu().get("flags", "").split(),
    reason="requires AMX BF16 support",
)
@pytest.mark.parametrize("blocking_factor_k", [1, 2, 4])
@pytest.mark.parametrize("threads", [1, 3])
def test_native_sfc_bfloat16_matmul(blocking_factor_k, threads):
    path = (Path(__file__).resolve().parents[2]
            / "samples" / "cpu" / "utils" / "sfc_matmul.py")
    spec = importlib.util.spec_from_file_location("cpu_sample_sfc_matmul", path)
    sample = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sample)
    torch.manual_seed(0)
    a = torch.randn((64, 256), dtype=torch.bfloat16)
    b = torch.randn((256, 64), dtype=torch.bfloat16)
    prepared = sample.prepare_sfc_matmul(
        a, b, blocking_factor_k=blocking_factor_k,
        options={"assume_in_bounds": True, "num_cpu_threads": threads})
    for _ in range(2):
        torch.testing.assert_close(prepared(), a @ b, rtol=1e-2, atol=1e-2)


@pytest.mark.usefixtures("native_cpu")
def test_only_the_amx_schedule_emits_amx(monkeypatch, tmp_path):
    from cuda.tile._cext import default_tile_context

    monkeypatch.setattr(default_tile_context.config, "cache_dir", None)
    dumps = []
    invoke = cpu.invoke

    def dumping_invoke(*args):
        dumps.append(tmp_path / f"{len(dumps)}.mlir")
        return invoke(*args[:6], str(dumps[-1]), *args[7:])

    monkeypatch.setattr(cpu, "invoke", dumping_invoke)
    path = Path(__file__).resolve().parents[2] / "samples" / "cpu" / "utils" / "sfc_matmul.py"
    spec = importlib.util.spec_from_file_location("cpu_sample_sfc_matmul_amx", path)
    sample = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sample)
    sample.prepare_sfc_matmul(
        torch.randn((64, 256), dtype=torch.bfloat16),
        torch.randn((256, 64), dtype=torch.bfloat16),
        options={"assume_in_bounds": True})

    def amx_kernels(entry_point):
        return sum(b"x86_amx" in cpu.native_compiler("cpu").lower_cpu_mlir(
            dump.read_bytes(), cpu._TRANSFORM_SCHEDULE, entry_point)
                   for dump in dumps)

    assert dumps
    assert amx_kernels("__transform_nano_amx") > 0
    assert amx_kernels("__transform_nano_x86") == amx_kernels("__transform_nano") == 0


@pytest.mark.usefixtures("native_cpu")
@pytest.mark.parametrize("scaling_block_size", [16, 32])
def test_swizzled_scale_tile_shape_is_independent_of_scaling_block_size(scaling_block_size):
    path = Path(__file__).resolve().parents[2] / "samples" / "cpu" / "BlockScaledMatMul-cpu.py"
    spec = importlib.util.spec_from_file_location("cpu_sample_block_scaled_matmul", path)
    sample = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sample)
    k = 4 * scaling_block_size
    scales = sample.swizzle_32_4_4(torch.ones((128, 4), dtype=torch.bfloat16))
    args = (torch.ones((128, k), dtype=torch.bfloat16), scales,
            torch.ones((k, 128), dtype=torch.bfloat16), scales,
            torch.empty((128, 128)), 128, 128, k, scaling_block_size)
    kernel = sample.block_scaled_matmul_kernel
    bytecode, _ = _custom.compile_bytecode(kernel, cpu, build_signature(kernel, args))
    assert bytecode


@ct.kernel
def _wide_scalar(out: ct.IndexedWithInt64, value: ct.ScalarInt64):
    ct.store(out, index=(0,), tile=ct.full((1,), value, ct.int64))


@pytest.mark.usefixtures("native_cpu")
def test_native_int64_scalar_and_array_metadata():
    out = np.zeros(1, dtype=np.int64)
    value = (1 << 40) + 37
    ct.launch(None, (1,), _wide_scalar, (out, value))
    assert out[0] == value


@ct.kernel
def _copy_static_shape(x: Annotated[ct.Array, ct.ArrayAnnotation(static_shape_dims=(0,))],
                       y, size: ConstInt):
    ct.store(y, index=(0,), tile=ct.load(x, index=(0,), shape=(size,)))


@ct.kernel
def _copy_matrix(x, y, rows: ConstInt, cols: ConstInt):
    ct.store(y, index=(0, 0), tile=ct.load(x, index=(0, 0), shape=(rows, cols)))


@pytest.mark.usefixtures("native_cpu")
def test_native_static_and_size_one_shapes():
    x = np.arange(16, dtype=np.float32)
    y = np.zeros_like(x)
    ct.launch(None, (1,), _copy_static_shape, (x, y, 16))
    np.testing.assert_array_equal(y, x)

    x, y = x.reshape(1, 16), np.zeros((1, 16), dtype=np.float32)
    ct.launch(None, (1,), _copy_matrix, (x, y, 1, 16))
    np.testing.assert_array_equal(y, x)


@ct.kernel
def _empty_kernel():
    pass


@pytest.mark.usefixtures("native_cpu")
def test_native_kernel_without_arguments():
    with ct.backend_options({"num_cpu_threads": 3}):
        ct.launch(None, (3, 2), _empty_kernel, ())


@pytest.mark.usefixtures("native_cpu")
def test_existing_flat_cpu_abi_still_executes(monkeypatch, tmp_path):
    args = (np.arange(64, dtype=np.float32), np.ones(64, dtype=np.float32),
            np.zeros(64, dtype=np.float32), 16)
    bytecode, signature = _custom.compile_bytecode(
        _vector_add, cpu, build_signature(_vector_add, args))
    binary = cpu.native_compiler("cpu").compile_cpu(
        bytecode, cpu._TRANSFORM_SCHEDULE, False,
        str(tmp_path), "", False, False)
    path = tmp_path / "flat.so"
    path.write_bytes(binary)
    library = ctypes.CDLL(str(path))
    function = getattr(library, signature.symbol)
    function.argtypes = [*([ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32] * 3),
                         *([ctypes.c_int32] * 6)]
    function.restype = None
    values = cpu._flatten_arguments(runtime_parameters("CPU", signature),
                                    _runtime(_vector_add, args))
    for block in range(4):
        function(*values, block, 0, 0, 4, 1, 1)
    np.testing.assert_array_equal(args[2], args[0] + args[1])


@pytest.fixture(scope="module")
def openmp_probe(tmp_path_factory):
    native = pytest.importorskip("cuda.tile._backend.cpu._cpu")
    path = tmp_path_factory.mktemp("cpu-probe") / "probe.so"
    source = b"""
#include <stdint.h>
#include <string.h>

void probe(const void *arguments, int32_t x, int32_t y, int32_t z,
           int32_t nx, int32_t ny, int32_t nz) {
  struct { int32_t *output; int (*thread_id)(void); int (*team_size)(void); } args;
  memcpy(&args, arguments, sizeof(args));
  int64_t index = x + (int64_t)nx * (y + (int64_t)ny * z);
  args.output[index * 2] = args.thread_id();
  args.output[index * 2 + 1] = args.team_size();
}
"""
    compiler = shutil.which(os.environ.get("CC", "cc"))
    if compiler is None:
        pytest.skip("building the OpenMP probe requires a C compiler")
    subprocess.run([compiler, "-shared", "-fPIC", "-x", "c", "-", "-o", str(path)],
                   input=source, check=True, timeout=60)
    runtime = ctypes.CDLL(native.__file__)
    return native, path, runtime


@pytest.mark.parametrize("threads", [0, 1, 3])
def test_native_openmp_thread_count_and_static_order(openmp_probe, threads):
    native, path, runtime = openmp_probe
    old_dynamic = runtime.omp_get_dynamic()
    runtime.omp_set_dynamic(0)
    try:
        expected_threads = threads or runtime.omp_get_max_threads()
        grid = (expected_threads * 2, 2, 2)
        out = np.full((np.prod(grid), 2), -1, dtype=np.int32)
        packer = cpu._argument_packer(("P", "P", "P"))
        args = packer.pack(out.ctypes.data,
                           ctypes.cast(runtime.omp_get_thread_num, ctypes.c_void_p).value,
                           ctypes.cast(runtime.omp_get_num_threads, ctypes.c_void_p).value)
        kernel = native.Kernel(str(path), "probe", packer.size)
        kernel.launch(args, grid, threads)
        np.testing.assert_array_equal(out[:, 0], np.repeat(np.arange(expected_threads), 8))
        np.testing.assert_array_equal(out[:, 1], expected_threads)
    finally:
        runtime.omp_set_dynamic(old_dynamic)


def test_native_loader_reports_errors(openmp_probe):
    native, path, _ = openmp_probe
    with pytest.raises(RuntimeError, match="unable to load kernel"):
        native.Kernel(str(path) + ".missing", "probe", 24)
    with pytest.raises(RuntimeError, match="unable to find kernel"):
        native.Kernel(str(path), "missing_symbol", 24)
    with pytest.raises(ValueError, match="multiple of 8"):
        native.Kernel(str(path), "probe", 3)
    kernel = native.Kernel(str(path), "probe", 24)
    with pytest.raises(ValueError, match="buffer size"):
        kernel.launch(b"", (1, 1, 1), 1)
