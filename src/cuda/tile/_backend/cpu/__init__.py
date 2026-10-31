# SPDX-FileCopyrightText: Copyright (c) <2026> Intel Corporation. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import dataclasses
import functools
import hashlib
import json
import operator
import os
import platform
import struct
import tempfile
from typing import Any, Mapping, NamedTuple

import cuda.tile as ct
from cuda.tile.compilation import ArrayConstraint

from .._custom import bool_option, check_options, current_options, normalize_dims
from .._custom import synchronize_stream as synchronize
from .._signature import ArrayArgument, check_argument, runtime_parameters, scalar_format
from .._native import compiler as native_compiler
from .._native import file_fingerprint, invoke

sm_arch = "3000"
bytecode_version = "13.3"

_TRANSFORM_SCHEDULE = os.path.join(os.path.dirname(__file__), "cpu_schedule.mlir")


def _lowering_pipeline(entry_point: str | None = None) -> str:
    """The MLIR pipeline lowering TileIRToMLIR output on this host."""
    entry_point = entry_point or _transform_entry_point()
    return f"""builtin.module(
    transform-preload-library{{transform-library-paths={json.dumps(_TRANSFORM_SCHEDULE)}}},
    transform-interpreter{{entry-point={entry_point}}},
    canonicalize,
    loop-invariant-code-motion,
    cse,
    symbol-dce,
    func.func(lower-vector-multi-reduction{{lowering-strategy=inner-reduction}}),
    fold-memref-alias-ops,
    expand-strided-metadata,
    canonicalize,
    convert-vector-to-scf{{full-unroll=true target-rank=1 lower-tensors=false}},
    lower-affine,
    finalize-memref-to-llvm,
    convert-scf-to-cf,
    convert-index-to-llvm,
    convert-vector-to-llvm{{
        reassociate-fp-reductions=true
        vector-contract-lowering=outerproduct
    }},
    math-expand-ops{{ops=fpowi}},
    convert-math-to-llvm{{approximate-log1p=true}},
    convert-math-to-libm,
    reconcile-unrealized-casts,
    convert-arith-to-llvm,
    convert-ub-to-llvm,
    convert-to-llvm,
    reconcile-unrealized-casts,
    canonicalize,
    cse,
    symbol-dce
)"""


# Options of a ``backend_options`` block; ``num_cpu_threads`` only applies at launch.
_OPTIONS = ("assume_in_bounds", "num_cpu_threads")

# CPU flags (as Linux reports them) required by the AMX lowering.
_AMX_FLAGS = frozenset({"amx_tile", "amx_bf16", "amx_int8"})


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


def _transform_entry_point() -> str:
    """The ``cpu_schedule.mlir`` entry point whose code the host can execute."""
    if platform.machine() != "x86_64":
        return "__transform_nano"
    if _AMX_FLAGS <= set(_host_cpu().get("flags", "").split()):
        return "__transform_nano_amx"
    return "__transform_nano_x86"


def compiler_identity(options: Options):
    """Identify the native toolchain, host ISA, and effective compile options."""

    try:
        identity = {
            "schema": 5,
            "compiler": file_fingerprint(native_compiler("cpu").__file__),
            "cutile_cpu_backend": file_fingerprint(__file__),
            "cutile_cpu_schedule": file_fingerprint(_TRANSFORM_SCHEDULE),
            "cpu": _host_cpu(),
            "platform": (platform.system(), platform.machine()),
            "libc": platform.libc_ver(),
            "options": dataclasses.asdict(options),
        }
    except OSError:
        return None
    return hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def compile(bytecode: bytes, signature, options: Options) -> bytes:
    """Compile TileIR into a shared object with the private native compiler."""
    del signature
    with tempfile.TemporaryDirectory() as directory:
        return invoke(
            "cpu", "compile_cpu", bytecode, _lowering_pipeline(),
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


def load(binary: bytes, signature, options: Options) -> _Loaded:
    """Load the shared object through the generic nanobind/OpenMP launcher."""
    from ._cpu import Kernel

    del options
    parameters = runtime_parameters("CPU", signature)
    packer = _argument_packer(_argument_formats(parameters))
    # dlopen needs a file; TMPDIR selects where it goes.
    with tempfile.NamedTemporaryFile(suffix=".so") as file:
        file.write(binary)
        file.flush()
        kernel = Kernel(file.name, signature.symbol, packer.size)
    return _Loaded(parameters, kernel, packer)


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
    synchronize(stream)
    values = _flatten_arguments(loaded.parameters, args)
    dimensions = tuple(operator.index(value) for value in normalize_dims(grid))
    threads = operator.index(current_options().get("num_cpu_threads", 0))
    loaded.kernel.launch(loaded.packer.pack(*values), dimensions, threads)
