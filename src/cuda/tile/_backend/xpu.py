# SPDX-FileCopyrightText: Copyright (c) <2026> Intel Corporation. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""cuTile backend targeting Intel XPUs through XeGPU and Level Zero.

Compilation runs the native ``tileir-to-mlir`` tool followed by the XeVM
pipeline (:mod:`cuda.tile._backend.xeas`); the resulting device binary is
launched through the native :mod:`cuda.tile._level_zero` extension.

Select the backend and supply the mandatory tuning parameters::

    ct.set_backend("xpu")
    with xpu.compile_options({"wg_m": 128, "wg_n": 128}):
        ct.launch(stream, grid, matmul_kernel, args)
"""
import dataclasses
import functools
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Mapping, NamedTuple

import numpy as np

import cuda.tile as ct
from cuda.tile.compilation import KernelSignature, ScalarConstraint, ConstantConstraint

from ._custom import bool_option, normalize_dims
from ._custom import compile_options  # noqa: F401  (backend API)
from ._toolchain import file_fingerprint, resolve_tool, run_tool


from mlir import ir  # noqa: E402
from cuda.tile._level_zero import launch_level_zero_module_kernel  # noqa: E402

from .xeas import xeas

# cuTile identifies a compile target by an arch string. Xe devices get values
# that cannot collide with a CUDA SM number, plus the ocloc chip to target.
_ARCHS = {
    "b70": ("1070", "bmg"),
    "b50": ("1050", "bmg"),
    "pvc": ("2000", "pvc"),
}
_DEFAULT_ARCH = "b70"


def _arch(arch: str) -> tuple[str, str]:
    if arch not in _ARCHS:
        raise ValueError(
            f"XPU backend: unknown architecture {arch!r}; expected one of "
            f"{', '.join(sorted(_ARCHS))}.")
    return _ARCHS[arch]


# Pinning the target means cuTile never probes a CUDA device (which would
# dlopen libcuda) when this backend is active.
sm_arch, _CHIP = _arch(os.environ.get("CUTILE_XPU_ARCH", _DEFAULT_ARCH))

# Pin the TileIR bytecode version so cuTile does not probe the `tileiras`
# compiler (which requires a CUDA toolkit) to auto-detect it.
bytecode_version = "13.3"

_SUBGROUP_SIZE = 16

# Launch block used when no subgroup tile sizes are given.
_DEFAULT_BLOCK = (512, 1, 1)


def _workgroup_tiles(options: dict) -> tuple[int, int]:
    """Read the mandatory work-group tile sizes from options."""
    missing = [k for k in ("wg_m", "wg_n") if k not in options]
    if missing:
        raise ValueError(
            "XPU backend: missing required compile_options keys: "
            f"{', '.join(missing)}"
        )
    return int(options["wg_m"]), int(options["wg_n"])


def _launch_block(options: Mapping[str, Any]) -> tuple[int, int, int]:
    """Derive the launch block ``((wg_m // sg_m) * (wg_n // sg_n) * 16, 1, 1)``.

    ``block_threads`` overrides the derived value; without ``sg_m``/``sg_n``
    there is nothing to derive from and :data:`_DEFAULT_BLOCK` applies.
    """
    wg_m, wg_n = _workgroup_tiles(options)
    block_threads = options.get("block_threads")
    if block_threads is not None:
        return normalize_dims(block_threads)
    if "sg_m" not in options or "sg_n" not in options:
        return _DEFAULT_BLOCK
    sg_m, sg_n = int(options["sg_m"]), int(options["sg_n"])
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
    """Validate and normalize the options of a ``compile_options`` block.

    ``wg_m``/``wg_n`` are required. Combined with ``sg_m``/``sg_n`` and the
    fixed subgroup size of 16 they derive the launch block
    ``((wg_m // sg_m) * (wg_n // sg_n) * 16, 1, 1)``.

    =================== ========== ================================
    key                 required?  meaning
    =================== ========== ================================
    block_threads       optional   launch block (overrides derived)
    wg_m, wg_n          yes        work-group tile sizes
    sg_m, sg_n          optional   subgroup tile size
    assume_in_bounds    optional   mark transfers in-bounds
    xegpu_op_level      optional   initial XeGPU op level
    large_register_file optional   enable large register file
    =================== ========== ================================
    """
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


@functools.lru_cache(maxsize=4)
def _ocloc_identity(ocloc: str) -> tuple[str, ...]:
    """Versions of ocloc and of the IGC code generator it loads at run time."""
    # `ocloc query` also writes its answer into the working directory.
    with tempfile.TemporaryDirectory() as cwd:
        return tuple(
            subprocess.run([ocloc, *args], cwd=cwd, capture_output=True,
                           check=True, timeout=60).stdout.decode().strip()
            for args in (("--version",), ("query", "IGC_REVISION")))


def _mlir_fingerprint() -> dict[str, str]:
    """Fingerprint the native MLIR libraries that run the XeVM pipeline."""
    libs = os.path.join(os.path.dirname(ir.__file__), "_mlir_libs")
    return {name: file_fingerprint(os.path.join(libs, name))
            for name in sorted(os.listdir(libs)) if ".so" in name}


@functools.lru_cache(maxsize=32)
def _compiler_identity(options: Options, tool: str, ocloc: str):
    """Build the XPU compiler identity for normalized effective options."""

    try:
        identity = {
            "schema": 2,
            "sm_arch": sm_arch,
            "chip": _CHIP,
            "options": dataclasses.asdict(options),
            "tileir_to_mlir": file_fingerprint(tool),
            "ocloc": _ocloc_identity(ocloc),
            "mlir": _mlir_fingerprint(),
            "xeas": file_fingerprint(
                os.path.join(os.path.dirname(__file__), "xeas.py")),
            "cutile_xpu_backend": file_fingerprint(__file__),
        }
    except (OSError, subprocess.SubprocessError):
        return None
    return hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def compiler_identity(options: Options):
    """Identify all XPU toolchain inputs affecting ``compile`` for ``options``."""

    ocloc = shutil.which("ocloc")
    try:
        tool = resolve_tool(
            "XPU", "CUTILE_XPU_TILEIR_TO_MLIR", "tileir-to-mlir")
    except FileNotFoundError:
        return None
    if ocloc is None:
        return None
    return _compiler_identity(
        options, os.path.realpath(tool), os.path.realpath(ocloc))


_torch = None


def _torch_api():
    """Import torch on first use; only tensor arguments need it."""
    global _torch
    if _torch is None:
        import torch
        _torch = torch
    return _torch


def _run_tileir_to_mlir(tool: str, bytecode: bytes, options: Options) -> str:
    """Lower TileIR bytecode to 'outlined' MLIR text via the tileir-to-mlir CLI.

    Raises ``RuntimeError`` with the captured stderr on failure. Set
    ``CUTILE_XPU_DUMP_MLIR`` to mirror the produced MLIR to stderr.
    """
    argv = [
        tool,
        "--tileir-to-mlir-pipeline=drop-rounding-modes=true "
        f"known-block-size={','.join(map(str, options.block))} "
        f"assume-in-bounds={str(options.assume_in_bounds).lower()}",
        "--convert-memref-args-to-ranked-memref=remove-unused=assumed-memref-dependent",
    ]
    mlir = run_tool("XPU", argv, bytecode).decode(errors="replace")
    if os.environ.get("CUTILE_XPU_DUMP_MLIR"):
        sys.stderr.write(mlir)
        sys.stderr.flush()
    return mlir


def compile(bytecode: bytes, signature, options: Options) -> bytes:
    """Compile TileIR bytecode into an Xe device binary blob.

    Runs the native tileir-to-mlir tool (overridable with
    ``CUTILE_XPU_TILEIR_TO_MLIR``) followed by the XeVM pipeline in
    :func:`~cuda.tile._backend.xeas.xeas`.
    """
    del signature
    tool = resolve_tool("XPU", "CUTILE_XPU_TILEIR_TO_MLIR", "tileir-to-mlir")
    mlir = _run_tileir_to_mlir(tool, bytecode, options)
    with ir.Context():
        return xeas(mlir, xegpu_op_level=options.xegpu_op_level,
                    large_register_file=options.large_register_file,
                    chip=_CHIP)


# ABI type of a runtime scalar, keyed by its signature dtype.
_SCALAR_NUMPY_DTYPE = {
    ct.int8: np.int8, ct.int16: np.int16,
    ct.int32: np.int32, ct.int64: np.int64,
    ct.uint8: np.uint8, ct.uint16: np.uint16,
    ct.uint32: np.uint32, ct.uint64: np.uint64,
    ct.float32: np.float32, ct.float64: np.float64,
    ct.bool_: np.bool_,
}


def _scalar_arg(value, constraint: ScalarConstraint):
    """Convert a runtime scalar to the NumPy value matching its ABI type."""
    dtype = _SCALAR_NUMPY_DTYPE.get(constraint.dtype)
    if dtype is None:
        raise TypeError(f"XPU backend: unsupported scalar dtype {constraint.dtype}")
    return dtype(value)


def _runtime_kernel_args(signature: KernelSignature, args: tuple) -> list:
    """Build the kernel argument list expected by the Xe kernel.

    Compile-time constants are dropped, tensors become memref metadata, and
    scalars become bytes with the ABI type carried by the signature.
    """
    torch = _torch_api()
    flat = []
    for index, (arg, param) in enumerate(zip(args, signature.parameters)):
        if isinstance(param, ConstantConstraint):
            continue
        if isinstance(arg, torch.Tensor):
            if arg.device.type != "xpu":
                raise ValueError(
                    f"XPU backend: argument #{index} is on device "
                    f"'{arg.device}', expected an XPU tensor")
            flat.append((arg.data_ptr(), tuple(arg.shape), arg.stride()))
        elif isinstance(param, ScalarConstraint):
            flat.append(_scalar_arg(arg, param).tobytes())
        else:
            raise TypeError(
                f"XPU backend: unsupported runtime argument {type(arg).__name__}")
    return flat


class _Loaded(NamedTuple):
    binary: bytes
    signature: KernelSignature
    block: tuple[int, int, int]


def load(binary: bytes, signature: KernelSignature, options: Options) -> _Loaded:
    """Bind a binary to the launch block it was compiled for."""
    return _Loaded(binary, signature, options.block)


def synchronize(stream):
    """Wait for work queued on ``stream``; XPU launches block until done."""
    if stream is not None:
        stream.synchronize()


def launch(loaded: _Loaded, stream, grid, args):
    """Run a loaded kernel synchronously through the Level Zero runtime."""
    runtime_args = _runtime_kernel_args(loaded.signature, args)

    # The Level Zero runtime enqueues onto its own immediate command list, which
    # is unordered with respect to ``stream``. Drain the caller's pending work
    # (e.g. the tensor initialisation) first, otherwise it can land after the
    # kernel and overwrite its results. The launch below blocks until the kernel
    # completed, so ordering is restored on the way out.
    synchronize(stream)

    launch_level_zero_module_kernel(
        loaded.binary, loaded.signature.symbol, runtime_args,
        normalize_dims(grid), loaded.block)
