# SPDX-FileCopyrightText: Copyright (c) <2026> Intel Corporation. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import dataclasses
import functools
import glob
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import shutil
import tempfile
from types import SimpleNamespace
from typing import Any, Mapping, NamedTuple

import cuda.tile as ct
from cuda.tile.compilation import ArrayConstraint
from cuda.tile.compilation import ConstantConstraint
from cuda.tile.compilation import ScalarConstraint

from ._custom import bool_option
from ._custom import compile_options  # noqa: F401  (backend API)
from ._custom import current_options
from ._custom import normalize_dims
from ._signature import array_device_type
from ._signature import array_metadata
from ._toolchain import file_fingerprint
from ._toolchain import resolve_tool
from ._toolchain import run_tool

sm_arch = "3000"
bytecode_version = "13.3"

# Environment variables read by the Triton CPU compiler.
_COMPILER_ENVIRONMENT = (
    "TRITON_CPU_FAST_MATH",
    "TRITON_CPU_UKERNELS_LIB",
    "TRITON_CPU_DOT_PROD_HORIZ_SUM",
    "TRITON_DISABLE_LINE_INFO",
    "DISABLE_LLVM_OPT",
    "LLVM_PASS_PLUGIN_PATH",
    "TRITON_ENABLE_ASAN",
)


@dataclasses.dataclass(frozen=True)
class Options:
    """Validated compile options."""

    assume_in_bounds: bool


def normalize_options(options: Mapping[str, Any]) -> Options:
    """Validate and normalize the options of a :func:`compile_options` block.

    ``CUTILE_CPU_ASSUME_IN_BOUNDS`` overrides ``assume_in_bounds``.
    """
    value = os.environ.get("CUTILE_CPU_ASSUME_IN_BOUNDS")
    if value is not None:
        return Options(assume_in_bounds=value.lower() in ("1", "on", "true", "yes"))
    return Options(
        assume_in_bounds=bool_option("CPU", options, "assume_in_bounds", False))


@functools.lru_cache(maxsize=1)
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
def _compiler_identity(environment, options: Options):
    """Build the CPU compiler identity for one compile environment."""

    try:
        _ir, backend, triton_options, _CPULauncher, _CPUUtils = _cpu_compiler(
            options.assume_in_bounds)
        (triton_version, triton_cpu_compiler, libtriton,
         triton_build, triton_knobs) = _cpu_identity_modules()
        if triton_knobs.build.impl is not None:
            return None
        triton_library_dir = os.path.dirname(libtriton.__file__)
        triton_cpu_dir = os.path.dirname(triton_cpu_compiler.__file__)
        compiler = triton_build._find_compiler("c")
        compiler = shutil.which(compiler) or compiler
        llvm_pass_plugin = os.environ.get("LLVM_PASS_PLUGIN_PATH")

        tool = resolve_tool(
            "CPU", "CUTILE_CPU_TILEIR_TO_MLIR", "tileir-to-mlir")
        identity = {
            "schema": 2,
            "triton_version": triton_version,
            "triton_library_dir": os.path.realpath(triton_library_dir),
            "libtriton": file_fingerprint(libtriton.__file__),
            "cpu_runtime": file_fingerprint(os.path.join(
                triton_library_dir, "libTritonCPURuntime.so")),
            "sleef": file_fingerprint(os.path.join(
                triton_library_dir, "libsleef.so")),
            "triton_cpu_backend": [
                file_fingerprint(path) for path in
                sorted(glob.glob(os.path.join(triton_cpu_dir, "*.py")))],
            "cutile_cpu_backend": file_fingerprint(__file__),
            "tileir_to_mlir": file_fingerprint(tool),
            "cpu_arch": backend.cpu_arch,
            "cpu_name": backend.cpu_name,
            "cpu_features": sorted(backend.cpu_features),
            "platform": platform.platform(),
            "libc": platform.libc_ver(),
            "host_compiler_path": os.path.realpath(compiler),
            "host_compiler": file_fingerprint(compiler),
            "options": dataclasses.asdict(options),
            "triton_options": triton_options.hash(),
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


def compiler_identity(options: Options):
    """Identify all CPU toolchain inputs affecting ``compile`` for ``options``."""

    try:
        *_modules, triton_knobs = _cpu_identity_modules()
    except (AttributeError, ImportError, OSError, TypeError):
        return None
    if triton_knobs.build.impl is not None:
        return None
    environment = tuple(os.environ.get(name) for name in _COMPILER_ENVIRONMENT)
    return _compiler_identity(environment, options)


def _lower_tileir(bytecode: bytes, options: Options) -> bytes:
    tool = resolve_tool("CPU", "CUTILE_CPU_TILEIR_TO_MLIR", "tileir-to-mlir")
    assume_in_bounds = str(options.assume_in_bounds).lower()
    argv = [
        tool,
        "--tileir-to-mlir-pipeline=target=cpu append-grid-args=true "
        f"drop-rounding-modes=true assume-in-bounds={assume_in_bounds}",
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


def compile(bytecode: bytes, signature, options: Options) -> bytes:
    """Compile TileIR bytecode into a shared object via Triton CPU."""
    del signature
    ir, backend, triton_options, _CPULauncher, _CPUUtils = _cpu_compiler(
        options.assume_in_bounds)
    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)
    mlir = _lower_tileir(bytecode, options)
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
    module = backend.make_tttcir(module, metadata, triton_options, from_tileir=True)
    llvm_ir = backend.make_llir(module, metadata, triton_options)
    assembly = backend.make_asm(llvm_ir, metadata, triton_options)
    return backend.make_so(assembly, metadata, triton_options)


_SCALAR_TYPES = {
    ct.int8: "i8", ct.int16: "i16", ct.int32: "i32", ct.int64: "i64",
    ct.uint8: "u8", ct.uint16: "u16", ct.uint32: "u32", ct.uint64: "u64",
    ct.float32: "fp32", ct.float64: "fp64", ct.bool_: "i1",
}


def _argument_layout(signature):
    """Per parameter kind (``None`` for constants) and the Triton signature."""
    kinds = []
    types = []
    for parameter in signature.parameters:
        if isinstance(parameter, ConstantConstraint):
            kinds.append(None)
            continue
        if isinstance(parameter, ScalarConstraint):
            try:
                types.append(_SCALAR_TYPES[parameter.dtype])
            except KeyError:
                raise TypeError(
                    f"CPU backend: unsupported scalar dtype {parameter.dtype}") from None
            kinds.append("scalar")
            continue
        if not isinstance(parameter, ArrayConstraint):
            raise TypeError(
                f"CPU backend: unsupported parameter constraint "
                f"{type(parameter).__name__}")
        index_type = "i64" if parameter.index_dtype is ct.int64 else "i32"
        types.extend(["*i8", *([index_type] * parameter.ndim * 2)])
        kinds.append("array")

    return tuple(kinds), {index: value for index, value in enumerate(types)}


def _flatten_arguments(kinds, arguments):
    values = []
    for index, (kind, argument) in enumerate(zip(kinds, arguments)):
        if kind is None:
            continue
        if kind == "scalar":
            values.append(argument)
            continue
        device = array_device_type(argument)
        if device != "cpu":
            raise ValueError(
                f"CPU backend: argument #{index} is on device '{device}', "
                "expected host memory")
        pointer, shape, strides, _dtype = array_metadata(argument)
        values.extend([pointer, *shape, *strides])
    return values


class _Loaded(NamedTuple):
    kinds: tuple
    module: Any
    function: Any
    launcher: Any


def load(binary: bytes, signature, options: Options) -> _Loaded:
    """Load the shared object and build a Triton launcher for its signature."""
    del options
    _ir, _target, _backend, CPULauncher, CPUUtils = _triton_cpu()
    kinds, types = _argument_layout(signature)
    launcher = CPULauncher(SimpleNamespace(signature=types, constants={}), None)
    module, function, *_unused = CPUUtils().load_binary(
        signature.symbol, binary, 0, 0)
    return _Loaded(kinds, module, function, launcher)


def synchronize(stream):
    """Wait for work queued on ``stream``; CPU kernels run synchronously."""
    if stream not in (None, 0):
        stream_synchronize = getattr(stream, "synchronize", None)
        if stream_synchronize is None:
            raise TypeError("CPU backend stream must be None, zero, or synchronizable")
        stream_synchronize()


def launch(loaded: _Loaded, stream, grid, args):
    """Run a loaded kernel; ``num_cpu_threads`` sets the worker count."""
    synchronize(stream)
    values = _flatten_arguments(loaded.kinds, args)
    metadata = SimpleNamespace(
        num_cpu_threads=int(current_options().get("num_cpu_threads", 0)))
    x, y, z = normalize_dims(grid)
    loaded.launcher(x, y, z, 0, loaded.function, metadata, None, None, None,
                    *values)
