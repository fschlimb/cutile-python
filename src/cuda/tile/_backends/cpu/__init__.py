# SPDX-FileCopyrightText: Copyright (c) <2026> Intel Corporation. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""CPU backend: native compilation and synchronous OpenMP execution.

TileIRToMLIR and MLIR passes, LLVM optimization and object emission, and the
LLD ELF linker run through C++ APIs in the private native compiler. Shared
libraries use cuTile's binary cache. Installed kernels need no LLVM tools or
MLIR Python bindings.

TileIRToMLIR emits a uniform ABI: a pointer to immutable eight-byte argument
slots followed by six i32 block-id/grid-dimension arguments. A single
nanobind/OpenMP launcher serves all kernels; no launcher is compiled per kernel.
Blocks use static OpenMP scheduling without a grid-sized coordinate array.
One worker and single-block grids bypass parallel dispatch.
The launcher uses the system GCC OpenMP runtime (libgomp.so.1), not a bundled copy.

Before general lowering, cpu_schedule.mlir removes leading unit vector
dimensions, including on transfer operations, on any host. It selects x86
contraction-loop preparation on x86-64, and AMX lowering only when
AMX tile, BF16, and INT8 are available. Final LLVM conversion lowers AMX
operations and loop-carried tile types together with control flow. On Linux
x86-64, importing the launcher requests process-wide tile-data permission.
If permission is denied, schedule selection uses the x86 path without AMX.
After the schedule, math functions except sin and cos are replaced by
polynomial approximations so they stay vectorized; LLVM would otherwise
scalarize vector math intrinsics into per-element libm calls.

The compiler and TileIRToMLIR are LLVM external projects in the persistent
build/llvm tree. Shared compiler invocation and timeout handling live in
cuda.tile._backends._native. See backends/cpu/README.md for build and usage.
"""

from __future__ import annotations

import dataclasses
import functools
import operator
import os
import platform
import struct
import tempfile
from typing import Any, Mapping, NamedTuple

import cuda.tile as ct
from cuda.tile.compilation import ArrayConstraint

from .._custom import current_options
from .._native import BYTECODE_VERSION
from .._native import compiler as native_compiler
from .._native import file_fingerprint, identity_digest, invoke
from .._signature import ArrayArgument, check_argument, runtime_parameters, scalar_format
from .._util import bool_option, check_options, normalize_dims
from .._util import synchronize_stream as synchronize

name = "cpu"
bytecode_version = BYTECODE_VERSION

_TRANSFORM_SCHEDULE = os.path.join(os.path.dirname(__file__), "cpu_schedule.mlir")


# Options of a ``backend_options`` block; ``num_cpu_threads`` only applies at launch.
_OPTIONS = ("assume_in_bounds", "num_cpu_threads")


@dataclasses.dataclass(frozen=True)
class Options:
    """Validated compile options."""

    assume_in_bounds: bool


def normalize_options(options: Mapping[str, Any]) -> Options:
    """Validate and normalize the options of a ``backend_options`` block."""
    check_options("CPU", options, _OPTIONS)
    return Options(
        assume_in_bounds=bool_option("CPU", options, "assume_in_bounds", False))


@functools.lru_cache(maxsize=1)
def _host_cpu():
    """Include the host ISA in cache keys for native LLVM code generation."""
    with open("/proc/cpuinfo") as file:
        fields = dict(
            line.split(":", 1) for line in file.read().split("\n\n", 1)[0].splitlines()
            if ":" in line)
    return {key.strip(): value.strip() for key, value in fields.items()
            if key.strip() in ("model name", "flags", "Features", "CPU architecture",
                               "CPU implementer", "CPU part", "CPU variant", "CPU revision")}


def _amx_enabled() -> bool:
    """Whether the launcher enabled AMX tile, BF16 and INT8 for this process."""
    from ._cpu import amx_enabled

    return amx_enabled


def _transform_entry_point() -> str:
    """The native compiler's schedule entry point for this host."""
    return native_compiler("cpu").cpu_transform_entry_point()


def compiler_identity(options: Options):
    """Identify the native toolchain, host ISA, and effective compile options."""

    try:
        identity = {
            "schema": 6,
            "compiler": file_fingerprint(native_compiler("cpu").__file__),
            "cutile_cpu_backend": file_fingerprint(__file__),
            "cutile_cpu_schedule": file_fingerprint(_TRANSFORM_SCHEDULE),
            "entry_point": _transform_entry_point(),
            "cpu": _host_cpu(),
            "platform": (platform.system(), platform.machine()),
            "libc": platform.libc_ver(),
            "options": dataclasses.asdict(options),
        }
    except OSError:
        return None
    return identity_digest(identity)


def compile(bytecode: bytes, signature, options: Options) -> bytes:
    """Compile TileIR into a shared object with the private native compiler."""
    del signature
    with tempfile.TemporaryDirectory() as directory:
        return invoke(
            "cpu", "compile_cpu", bytecode, _TRANSFORM_SCHEDULE,
            options.assume_in_bounds, directory,
            os.environ.get("CUTILE_CPU_DUMP_MLIR", ""),
            os.environ.get("MLIR_ENABLE_DUMP") == "1")


def _argument_packer(formats) -> struct.Struct:
    """Match TileIRToMLIR's eight-byte slots without scalar alignment gaps."""
    return struct.Struct("@" + "".join(
        f"{fmt}{8 - struct.calcsize(fmt)}x" for fmt in formats))


def _argument_formats(parameters) -> list[str]:
    """Struct formats of the flat native ABI: a pointer, sizes and strides per array."""
    formats = []
    for parameter in parameters:
        if isinstance(parameter, ArrayConstraint):
            index = "q" if parameter.index_dtype is ct.int64 else "i"
            formats += ["P", *([index] * parameter.ndim * 2)]
        else:
            formats.append(scalar_format("CPU", parameter.dtype))
    return formats


class _Loaded(NamedTuple):
    parameters: tuple
    kernel: Any
    packer: struct.Struct
    native_plan: tuple


def make_launch_cache(constants, limit):
    from ._cpu import LaunchCache

    return LaunchCache(constants, limit)


def load(binary: bytes, signature, options: Options) -> _Loaded:
    """Load the shared object through the generic nanobind/OpenMP launcher."""
    from ._cpu import Kernel

    del options
    parameters = runtime_parameters("CPU", signature)
    packer = _argument_packer(_argument_formats(parameters))
    native_plan = tuple(
        (parameter.ndim, "q" if parameter.index_dtype is ct.int64 else "i")
        if isinstance(parameter, ArrayConstraint)
        else scalar_format("CPU", parameter.dtype)
        for parameter in parameters)
    # dlopen needs a file; TMPDIR selects where it goes.
    with tempfile.NamedTemporaryFile(suffix=".so") as file:
        file.write(binary)
        file.flush()
        kernel = Kernel(file.name, signature.symbol, packer.size)
    return _Loaded(parameters, kernel, packer, native_plan)


def _flatten_arguments(parameters, args) -> list:
    values = []
    for parameter, argument in zip(parameters, args, strict=True):
        check_argument("CPU", "cpu", parameter, argument)
        if isinstance(argument, ArrayArgument):
            values += (argument.pointer, *argument.shape, *argument.strides)
        else:
            values.append(argument)
    return values


def launch(loaded: _Loaded, stream, grid, args):
    """Run a loaded kernel; the ``num_cpu_threads`` option sets the worker count."""
    launch_flat(loaded, stream, grid, _flatten_arguments(loaded.parameters, args))


def launch_flat(loaded: _Loaded, stream, grid, values):
    """Run a loaded kernel with arguments already flattened as by ``launch``."""
    synchronize(stream)
    dimensions = tuple(operator.index(value) for value in normalize_dims(grid))
    threads = operator.index(current_options().get("num_cpu_threads", 0))
    loaded.kernel.launch(loaded.packer.pack(*values), dimensions, threads)
