from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import os
import platform
import tempfile
import threading
from types import SimpleNamespace

import cuda.tile as ct
from cuda.tile.compilation import ArrayConstraint, ConstantConstraint, ScalarConstraint

from ._signature import array_metadata, build_signature
from ._toolchain import resolve_tool, run_tool


sm_arch = "3000"
bytecode_version = "13.3"

_tls = threading.local()
_cache_lock = threading.Lock()
_launch_cache = {}
_ASSUME_IN_BOUNDS = os.environ.get(
    "CUTILE_CPU_ASSUME_IN_BOUNDS", "0").lower() in ("1", "on", "true", "yes")


@contextlib.contextmanager
def compile_options(options: dict):
    previous = getattr(_tls, "options", None)
    _tls.options = dict(options or {})
    try:
        yield
    finally:
        _tls.options = previous


def _current_options() -> dict:
    return getattr(_tls, "options", None) or {}


def _triton_cpu():
    if importlib.util.find_spec("triton") is None:
        raise ImportError(
            "CPU backend requires the 'cpu' extra; run "
            "scripts/sync-backend.sh cpu")
    from triton._C.libtriton import ir
    from triton.backends.compiler import GPUTarget
    from triton.backends.cpu.compiler import CPUBackend
    from triton.backends.cpu.driver import CPULauncher, CPUUtils
    return ir, GPUTarget, CPUBackend, CPULauncher, CPUUtils


def _lower_tileir(bytecode: bytes) -> bytes:
    tool = resolve_tool("CPU", "CUTILE_CPU_TILEIR_TO_MLIR", "tileir-to-mlir")
    assume_in_bounds = str(_ASSUME_IN_BOUNDS).lower()
    argv = [
        tool,
        "--tileir-to-mlir-pipeline="
        f"target=cpu append-grid-args=true drop-rounding-modes=true "
        f"assume-in-bounds={assume_in_bounds}",
        "--loop-invariant-code-motion",
        "--convert-memref-args-to-ptr-args",
        "--cse",
        "--canonicalize",
    ]
    output = run_tool("CPU", argv, bytecode)
    dump = os.environ.get("CUTILE_CPU_DUMP_MLIR")
    if dump:
        with open(dump, "wb") as file:
            file.write(output)
    return output


def compile_tileir(tileir_bytecode, *, symbol, sm_arch, signature):
    del symbol, sm_arch, signature
    ir, GPUTarget, CPUBackend, _CPULauncher, _CPUUtils = _triton_cpu()
    target = GPUTarget("cpu", platform.machine(), 0)
    backend = CPUBackend(target)
    options = backend.parse_options({
        "assume_in_bounds": _ASSUME_IN_BOUNDS,
    })
    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)
    mlir = _lower_tileir(tileir_bytecode)
    with tempfile.NamedTemporaryFile(suffix=".tttcir") as source:
        source.write(mlir)
        source.flush()
        module = ir.parse_mlir_module(source.name, context)
    module.context = context
    metadata = {
        "num_ctas": 0,
        "num_warps": 0,
        "num_stages": 0,
        "cluster_dims": (1, 1, 1),
    }
    module = backend.make_tttcir(module, metadata, options, from_tileir=True)
    llvm_ir = backend.make_llir(module, metadata, options, from_tileir=True)
    assembly = backend.make_asm(llvm_ir, metadata, options)
    return backend.make_so(assembly, metadata, options)


_SCALAR_TYPES = {
    ct.int8: "i8", ct.int16: "i16", ct.int32: "i32", ct.int64: "i64",
    ct.uint8: "u8", ct.uint16: "u16", ct.uint32: "u32", ct.uint64: "u64",
    ct.float32: "fp32", ct.float64: "fp64",
}


def _flatten_arguments(signature, arguments):
    values = []
    types = []
    for parameter, argument in zip(signature.parameters, arguments):
        if isinstance(parameter, ConstantConstraint):
            continue
        if isinstance(parameter, ScalarConstraint):
            try:
                types.append(_SCALAR_TYPES[parameter.dtype])
            except KeyError:
                raise TypeError(
                    f"CPU backend: unsupported scalar dtype {parameter.dtype}") from None
            values.append(argument)
            continue
        if not isinstance(parameter, ArrayConstraint):
            raise TypeError(
                f"CPU backend: unsupported parameter constraint "
                f"{type(parameter).__name__}")
        pointer, shape, strides, _dtype = array_metadata(argument)
        index_type = "i64" if parameter.index_dtype is ct.int64 else "i32"
        types.extend(["*i8", *([index_type] * parameter.ndim * 2)])
        values.extend([pointer, *shape, *strides])
    return values, {index: value for index, value in enumerate(types)}


def _triplet(grid) -> tuple[int, int, int]:
    values = (grid,) if isinstance(grid, int) else tuple(grid)
    if not 1 <= len(values) <= 3:
        raise ValueError("launch dimensions must have length 1 to 3")
    return values + (1,) * (3 - len(values))


def launch(stream, grid, kernel, args):
    if stream not in (None, 0):
        synchronize = getattr(stream, "synchronize", None)
        if synchronize is None:
            raise TypeError("CPU backend stream must be None, zero, or synchronizable")
        synchronize()

    signature = build_signature(kernel, args)
    binary, symbol = ct.compile_kernel(kernel, signature)
    values, triton_signature = _flatten_arguments(signature, args)
    _ir, _target, _backend, CPULauncher, CPUUtils = _triton_cpu()
    key = (hashlib.sha256(binary).digest(), symbol,
           tuple(triton_signature.values()))
    with _cache_lock:
        cached = _launch_cache.get(key)
        if cached is None:
            source = SimpleNamespace(signature=triton_signature, constants={})
            launcher = CPULauncher(source, None)
            module, function, *_unused = CPUUtils().load_binary(
                symbol, binary, 0, 0)
            cached = module, function, launcher
            _launch_cache[key] = cached
    _module, function, launcher = cached
    metadata = SimpleNamespace(
        num_cpu_threads=int(_current_options().get("num_cpu_threads", 0)))
    x, y, z = _triplet(grid)
    launcher(x, y, z, 0, function, metadata, None, None, None, *values)