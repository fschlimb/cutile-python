# SPDX-FileCopyrightText: Copyright (c) <2026> Intel Corporation. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""cuTile backend targeting Intel XPUs through XeGPU and Level Zero.

TileIRToMLIR and XeVM passes run in the private native compiler, linked against
LLVM/MLIR and TileIRToMLIR through nanobind. Device-code generation calls
libocloc in-process and is serialized because its thread safety is undocumented.
Installed kernels need no LLVM tools or MLIR Python bindings.

Array parameters use a uniform ranked-memref ABI, including unused arrays and
arrays used only for shape or strides. Offsets, sizes, and strides are 64-bit.
Compilation rejects access patterns that cannot be represented by this ABI.
The launch block derived from tile sizes is
((wg_m // sg_m) * (wg_n // sg_n) * 16, 1, 1) and is baked into the binary.

Kernels run synchronously on Level Zero device 0 (``xpu:0``), in a private
context rather than the caller's. The native _level_zero extension waits for
the caller's stream before launching and checks Level Zero calls. Each compiled
kernel owns its module, kernel, and command list. Launches of the same kernel
are serialized; different kernels can run independently. Driver and launch
failures raise RuntimeError with module build logs when available.
The MLIR Level Zero runtime is neither built nor packaged.

Shared compiler invocation and timeout handling live in cuda.tile._backend._native.
See backends/xpu/README.md for build, device selection, and usage.
"""
import dataclasses
import functools
import operator
import os
import struct
from typing import Any, Mapping, NamedTuple

from cuda.tile.compilation import ArrayConstraint

from .._native import BYTECODE_VERSION
from .._native import compiler as native_compiler
from .._native import file_fingerprint, identity_digest, invoke
from .._signature import ArrayArgument, check_argument, runtime_parameters, scalar_format
from .._util import bool_option, check_options, normalize_dims
from .._util import synchronize_stream as synchronize
from ._level_zero import Kernel as _LevelZeroKernel

name = "xpu"
bytecode_version = BYTECODE_VERSION

# The ocloc chip of each supported architecture.
_ARCHS = {"b70": "bmg", "b50": "bmg", "pvc": "pvc"}
_DEFAULT_ARCH = "b70"


def _chip(arch: str) -> str:
    if arch not in _ARCHS:
        raise ValueError(
            f"XPU backend: unknown architecture {arch!r}; expected one of "
            f"{', '.join(sorted(_ARCHS))}.")
    return _ARCHS[arch]


_CHIP = _chip(os.environ.get("CUTILE_XPU_ARCH", _DEFAULT_ARCH))

_SUBGROUP_SIZE = 16

# Launch block used when neither ``block_threads`` nor tile sizes are given.
_DEFAULT_BLOCK = (512, 1, 1)

_TILE_OPTIONS = ("wg_m", "wg_n", "sg_m", "sg_n")
_OPTIONS = ("block_threads", *_TILE_OPTIONS, "assume_in_bounds",
            "xegpu_op_level", "large_register_file")


def _launch_block(options: Mapping[str, Any]) -> tuple[int, int, int]:
    """Return ``block_threads``, or derive ``((wg_m // sg_m) * (wg_n // sg_n) * 16, 1, 1)``.

    Without either, :data:`_DEFAULT_BLOCK` applies.
    """
    tiles = [name for name in _TILE_OPTIONS if name in options]
    if "block_threads" in options:
        if tiles:
            raise ValueError(
                "XPU backend: give either block_threads or the tile sizes "
                f"{', '.join(_TILE_OPTIONS)}, not both")
        return normalize_dims(options["block_threads"])
    if not tiles:
        return _DEFAULT_BLOCK
    if len(tiles) != len(_TILE_OPTIONS):
        raise ValueError(
            f"XPU backend: the tile sizes {', '.join(_TILE_OPTIONS)} must be given together")
    wg_m, wg_n, sg_m, sg_n = (operator.index(options[name]) for name in _TILE_OPTIONS)
    if min(sg_m, sg_n) <= 0 or wg_m <= 0 or wg_n <= 0 or wg_m % sg_m or wg_n % sg_n:
        raise ValueError(
            "XPU backend: wg_m and wg_n must be positive multiples of sg_m and sg_n")
    return ((wg_m // sg_m) * (wg_n // sg_n) * _SUBGROUP_SIZE, 1, 1)


_XEGPU_OP_LEVELS = ("workgroup", "subgroup", "lane")


@dataclasses.dataclass(frozen=True)
class Options:
    """Validated compile options; ``block`` is baked into the binary."""

    block: tuple[int, int, int]
    assume_in_bounds: bool
    xegpu_op_level: str
    large_register_file: bool


def normalize_options(options: Mapping[str, Any]) -> Options:
    """Validate and normalize the options of a ``backend_options`` block.

    =================== ================================================
    key                 meaning
    =================== ================================================
    block_threads       launch block
    wg_m, wg_n          work-group tile size; derive the launch block
    sg_m, sg_n          subgroup tile size, given together with wg_m/wg_n
    assume_in_bounds    mark transfers in-bounds
    xegpu_op_level      initial XeGPU op level
    large_register_file enable large register file (default)
    =================== ================================================
    """
    check_options("XPU", options, _OPTIONS)
    xegpu_op_level = options.get("xegpu_op_level", "workgroup")
    if xegpu_op_level not in _XEGPU_OP_LEVELS:
        raise ValueError(
            f"XPU backend: xegpu_op_level must be one of "
            f"{', '.join(_XEGPU_OP_LEVELS)}, got {xegpu_op_level!r}")
    return Options(
        block=_launch_block(options),
        assume_in_bounds=bool_option("XPU", options, "assume_in_bounds", False),
        xegpu_op_level=xegpu_op_level,
        large_register_file=bool_option(
            "XPU", options, "large_register_file", True),
    )


@functools.lru_cache(maxsize=1)
def _ocloc_identity(_compiler_fingerprint: str) -> str:
    """Cache successful version queries until the compiler file changes."""
    return native_compiler("xpu").xpu_identity()


def compiler_identity(options: Options):
    """Build the XPU compiler identity for normalized effective options."""

    try:
        compiler = native_compiler("xpu")
        fingerprint = file_fingerprint(compiler.__file__)
        identity = {
            "schema": 4,
            "chip": _CHIP,
            "options": dataclasses.asdict(options),
            "compiler": fingerprint,
            "ocloc": _ocloc_identity(fingerprint),
            "cutile_xpu_backend": file_fingerprint(__file__),
        }
    except OSError:
        return None
    return identity_digest(identity)


def compile(bytecode: bytes, signature, options: Options) -> bytes:
    """Compile TileIR bytecode into an Xe binary without command-line tools."""
    ranks = tuple(parameter.ndim if isinstance(parameter, ArrayConstraint) else -1
                  for parameter in runtime_parameters("XPU", signature))
    return invoke(
        "xpu", "compile_xpu", bytecode, options.block, options.assume_in_bounds,
        options.xegpu_op_level, options.large_register_file, _CHIP,
        bool(os.environ.get("CUTILE_XPU_DUMP_MLIR")), ranks)


def _runtime_arguments(parameters, args) -> list:
    """Scalars as ABI-typed bytes and arrays as ``(data, shape, strides)`` memrefs."""
    runtime = []
    for parameter, argument in zip(parameters, args, strict=True):
        check_argument("XPU", "xpu", parameter, argument)
        if isinstance(argument, ArrayArgument):
            if argument.device_index != 0:
                raise ValueError(
                    f"XPU backend: array is on xpu:{argument.device_index}, but kernels "
                    f"run on xpu:0; select the device with ZE_AFFINITY_MASK")
            runtime.append((argument.pointer, argument.shape, argument.strides))
        else:
            runtime.append(struct.pack(scalar_format("XPU", parameter.dtype), argument))
    return runtime


class _Loaded(NamedTuple):
    kernel: _LevelZeroKernel
    parameters: tuple
    block: tuple[int, int, int]


def load(binary: bytes, signature, options: Options) -> _Loaded:
    """Wrap a binary for Level Zero, bound to the block it was compiled for."""
    return _Loaded(_LevelZeroKernel(binary, signature.symbol),
                   runtime_parameters("XPU", signature), options.block)


def launch(loaded: _Loaded, stream, grid, args):
    """Run a loaded kernel synchronously through checked Level Zero calls."""
    runtime_args = _runtime_arguments(loaded.parameters, args)

    # The launcher enqueues onto its own immediate command list, which
    # is unordered with respect to ``stream``. Drain the caller's pending work
    # (e.g. the tensor initialisation) first, otherwise it can land after the
    # kernel and overwrite its results. The launch below blocks until the kernel
    # completed, so ordering is restored on the way out.
    synchronize(stream)

    loaded.kernel.launch(runtime_args, normalize_dims(grid), loaded.block)
