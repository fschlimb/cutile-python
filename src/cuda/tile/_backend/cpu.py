from __future__ import annotations

import contextlib
import functools
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import shutil
import tempfile
import threading
from types import SimpleNamespace

import cuda.tile as ct
from cuda.tile.compilation import ArrayConstraint, ConstantConstraint, ScalarConstraint

from ._custom import compile_for_launch, normalize_dims
from ._signature import array_metadata
from ._toolchain import file_fingerprint, resolve_tool, run_tool


sm_arch = "3000"
bytecode_version = "13.3"

_tls = threading.local()
_cache_lock = threading.Lock()
_launch_cache = {}


def _assume_in_bounds():
    value = os.environ.get("CUTILE_CPU_ASSUME_IN_BOUNDS")
    if value is not None:
        return value.lower() in ("1", "on", "true", "yes")
    return _current_options().get("assume_in_bounds", False)


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
    libtriton = importlib.import_module("triton._C.libtriton")
    compiler = importlib.import_module("triton.backends.compiler")
    cpu_compiler = importlib.import_module("triton.backends.cpu.compiler")
    cpu_driver = importlib.import_module("triton.backends.cpu.driver")
    return (libtriton.ir, compiler.GPUTarget, cpu_compiler.CPUBackend,
            cpu_driver.CPULauncher, cpu_driver.CPUUtils)


@functools.lru_cache(maxsize=1)
def _cpu_backend():
    """Return the process-local Triton CPU backend and launcher APIs."""

    ir, GPUTarget, CPUBackend, CPULauncher, CPUUtils = _triton_cpu()
    backend = CPUBackend(GPUTarget("cpu", platform.machine(), 0))
    return ir, backend, CPULauncher, CPUUtils


def _cpu_compiler(assume_in_bounds):
    """Return the CPU backend with its current compile-time options."""

    ir, backend, CPULauncher, CPUUtils = _cpu_backend()
    options = backend.parse_options({
        "assume_in_bounds": assume_in_bounds,
    })
    return ir, backend, options, CPULauncher, CPUUtils


@functools.lru_cache(maxsize=1)
def _cpu_identity_modules():
    """Load stable Triton modules used to identify generated CPU binaries."""

    return (
        importlib.metadata.version("triton"),
        importlib.import_module("triton.backends.cpu.compiler"),
        importlib.import_module("triton._C.libtriton"),
        importlib.import_module("triton.runtime.build"),
        importlib.import_module("triton.knobs"),
    )


@functools.lru_cache(maxsize=16)
def _compile_cache_key_cached(environment, assume_in_bounds):
    """Build the CPU compiler identity for one compile environment."""

    try:
        _ir, backend, options, _CPULauncher, _CPUUtils = _cpu_compiler(
            assume_in_bounds)
        (triton_version, triton_cpu_compiler, libtriton,
         triton_build, triton_knobs) = _cpu_identity_modules()
        if triton_knobs.build.impl is not None:
            return None
        triton_library_dir = os.path.dirname(libtriton.__file__)
        compiler = triton_build._find_compiler("c")
        compiler = shutil.which(compiler) or compiler
        llvm_pass_plugin = os.environ.get("LLVM_PASS_PLUGIN_PATH")

        tool = resolve_tool(
            "CPU", "CUTILE_CPU_TILEIR_TO_MLIR", "tileir-to-mlir")
        identity = {
            "schema": 1,
            "triton_version": triton_version,
            "triton_library_dir": os.path.realpath(triton_library_dir),
            "libtriton": file_fingerprint(libtriton.__file__),
            "cpu_runtime": file_fingerprint(os.path.join(
                triton_library_dir, "libTritonCPURuntime.so")),
            "sleef": file_fingerprint(os.path.join(
                triton_library_dir, "libsleef.so")),
            "triton_cpu_compiler": file_fingerprint(
                triton_cpu_compiler.__file__),
            "cutile_cpu_backend": file_fingerprint(__file__),
            "tileir_to_mlir": file_fingerprint(tool),
            "cpu_arch": backend.cpu_arch,
            "cpu_name": backend.cpu_name,
            "cpu_features": sorted(backend.cpu_features),
            "platform": platform.platform(),
            "libc": platform.libc_ver(),
            "host_compiler_path": os.path.realpath(compiler),
            "host_compiler": file_fingerprint(compiler),
            "options": options.hash(),
            "environment": environment,
            "llvm_pass_plugin": (
                file_fingerprint(llvm_pass_plugin)
                if llvm_pass_plugin else None),
        }
    except (AttributeError, ImportError, OSError, TypeError):
        return None
    return hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def compile_cache_key():
    """Identify all CPU toolchain and option inputs affecting compilation."""

    assume_in_bounds = _assume_in_bounds()
    try:
        *_modules, triton_knobs = _cpu_identity_modules()
    except (AttributeError, ImportError, OSError, TypeError):
        return None
    if triton_knobs.build.impl is not None:
        return None
    environment = tuple(os.environ.get(name) for name in (
        "TRITON_CPU_FAST_MATH",
        "TRITON_CPU_UKERNELS_LIB",
        "TRITON_CPU_DOT_PROD_HORIZ_SUM",
        "TRITON_DISABLE_LINE_INFO",
        "DISABLE_LLVM_OPT",
        "LLVM_PASS_PLUGIN_PATH",
        "TRITON_ENABLE_ASAN",
    ))
    return _compile_cache_key_cached(environment, assume_in_bounds)


def _lower_tileir(bytecode: bytes) -> bytes:
    tool = resolve_tool("CPU", "CUTILE_CPU_TILEIR_TO_MLIR", "tileir-to-mlir")
    assume_in_bounds = str(_assume_in_bounds()).lower()
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
    print(argv)
    output = run_tool("CPU", argv, bytecode)
    dump = os.environ.get("CUTILE_CPU_DUMP_MLIR")
    if dump:
        with open(dump, "wb") as file:
            file.write(output)
    return output


def compile_tileir(tileir_bytecode, *, symbol, sm_arch, signature):
    del symbol, sm_arch, signature
    ir, backend, options, _CPULauncher, _CPUUtils = _cpu_compiler(
        _assume_in_bounds())
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


_triplet = normalize_dims


def launch(stream, grid, kernel, args):
    if stream not in (None, 0):
        synchronize = getattr(stream, "synchronize", None)
        if synchronize is None:
            raise TypeError("CPU backend stream must be None, zero, or synchronizable")
        synchronize()

    compiled = compile_for_launch(kernel, args)
    return launch_compiled(stream, grid, compiled, args)


def launch_compiled(stream, grid, compiled, args):
    """Launch a previously compiled binary without compilation/cache lookup."""
    signature = compiled.signature
    binary = compiled.binary
    symbol = compiled.symbol
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