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
import contextlib
import os
import sys
import threading

import numpy as np

import cuda.tile as ct
from cuda.tile.compilation import KernelSignature, ScalarConstraint, ConstantConstraint

from ._signature import build_signature
from ._toolchain import resolve_tool, run_tool


from mlir import ir  # noqa: E402
from cuda.tile._level_zero import launch_level_zero_module_kernel  # noqa: E402

from .xeas import xeas

# cuTile identifies a compile target by an `sm_<number>` style string. Xe
# devices have no such number, so each supported device gets a distinct value
# that cannot collide with a real SM.
_ARCH_IDS = {
    "b70": "1070",
    "b50": "1050",
    "pvc": "2000",
}
_DEFAULT_ARCH = "b70"


def _arch_id(arch: str) -> str:
    if arch not in _ARCH_IDS:
        raise ValueError(
            f"XPU backend: unknown architecture {arch!r}; expected one of "
            f"{', '.join(sorted(_ARCH_IDS))}.")
    return _ARCH_IDS[arch]


# Reporting a target here makes `get_sm_arch_override()` short-circuit, so
# cuTile never probes a CUDA device (which would dlopen libcuda) when this
# backend is active.
sm_arch = _arch_id(os.environ.get("CUTILE_XPU_ARCH", _DEFAULT_ARCH))

# Pin the TileIR bytecode version so cuTile does not probe the `tileiras`
# compiler (which requires a CUDA toolkit) to auto-detect it.
bytecode_version = "13.3"

# Single process-wide MLIR context, created once and reused by every launch.
_context_lock = threading.Lock()
_mlir_context = None


def _xe_context():
    """Return the shared MLIR context, creating it on first use."""
    global _mlir_context
    with _context_lock:
        if _mlir_context is None:
            _mlir_context = ir.Context()
    return _mlir_context


_SUBGROUP_SIZE = 16

# Per-launch compiler options. `compile_options(...)` sets extra values passed
# to the launcher, scoped to the enclosing `ct.launch` call (it runs on the
# same thread, so a thread-local keeps concurrent calls isolated).
_tls = threading.local()


@contextlib.contextmanager
def compile_options(options: dict):
    """Provide tuning parameters for the XPU compile, per kernel launch.

    ``options`` is a flat dict scoped to launches inside the ``with`` block.
    Its keys are forwarded as ``xeas`` keyword arguments or used to derive the
    launch block.

    The work-group tiles ``wg_m``/``wg_n`` must be set explicitly in
    ``options``. Combined with ``sg_m``/``sg_n`` and the fixed subgroup size of
    16 they derive the launch block
    ``((wg_m // sg_m) * (wg_n // sg_n) * 16, 1, 1)``.

        Accepted options:

        =================== ========== ========================== ==========================
        key                 kernels    required?                  meaning
        =================== ========== ========================== ==========================
        block_threads       all        optional                   launch block (overrides derived)
        wg_m, wg_n          all        yes                        work-group tile sizes
        sg_m, sg_n          all        optional                   subgroup tile size
        assume_in_bounds    all        optional                   mark transfers in-bounds
        xegpu_op_level      all        optional                   initial XeGPU op level
        large_register_file all        optional                   enable large register file
        =================== ========== ========================== ==========================

    Example::

        with xpu.compile_options({"wg_m": 128, "wg_n": 128}):
            ct.launch(stream, grid, matmul_kernel, args)
    """
    prev = getattr(_tls, "options", None)
    _tls.options = dict(options or {})
    try:
        yield
    finally:
        _tls.options = prev


def _current_options() -> dict:
    """Options of the innermost enclosing :func:`compile_options` block."""
    return getattr(_tls, "options", None) or {}


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


def _launch_block(options: dict) -> tuple:
    """Derive the launch block ``((wg_m // sg_m) * (wg_n // sg_n) * 16, 1, 1)``.

    ``block_threads`` overrides the derived value; without ``sg_m``/``sg_n``
    there is nothing to derive from and :data:`_DEFAULT_BLOCK` applies.
    """
    wg_m, wg_n = _workgroup_tiles(options)
    block_threads = options.get("block_threads")
    if block_threads is not None:
        return block_threads
    if "sg_m" not in options or "sg_n" not in options:
        return _DEFAULT_BLOCK
    sg_m, sg_n = options["sg_m"], options["sg_n"]
    return ((wg_m // sg_m) * (wg_n // sg_n) * _SUBGROUP_SIZE, 1, 1)


def _triplet(value) -> tuple:
    values = (value,) if isinstance(value, int) else tuple(value)
    if not 1 <= len(values) <= 3:
        raise ValueError("launch dimensions must have length 1 to 3")
    return values + (1,) * (3 - len(values))


def _xeas_options(options: dict) -> dict:
    """Build the :func:`~cuda.tile._backend.xeas.xeas` keyword arguments."""
    return {
        "xegpu_op_level": options.get("xegpu_op_level", "workgroup"),
        "large_register_file": options.get("large_register_file", True),
    }


_torch = None


def _torch_api():
    """Import torch on first use; only tensor arguments need it."""
    global _torch
    if _torch is None:
        import torch
        _torch = torch
    return _torch


def _run_tileir_to_mlir(tool: str, bytecode: bytes, options: dict) -> str:
    """Lower TileIR bytecode to 'outlined' MLIR text via the tileir-to-mlir CLI.

    Raises ``RuntimeError`` with the captured stderr on failure. Set
    ``CUTILE_XPU_DUMP_MLIR`` to mirror the produced MLIR to stderr.
    """
    block = _launch_block(options)
    assume_in_bounds = str(options.get("assume_in_bounds", False)).lower()
    argv = [tool,
            f"--tileir-to-mlir-pipeline=drop-rounding-modes=true known-block-size={','.join(map(str, block))} assume-in-bounds={assume_in_bounds}",
            "--convert-memref-args-to-ranked-memref=remove-unused=assumed-memref-dependent",
            # "--loop-invariant-code-motion", "-canonicalize", "-cse",
            # "--mlir-print-ir-before-all",
            # "--mlir-print-ir-after-all",
    ]
    mlir = run_tool("XPU", argv, bytecode).decode(errors="replace")
    if os.environ.get("CUTILE_XPU_DUMP_MLIR"):
        sys.stderr.write(mlir)
        sys.stderr.flush()
    return mlir


def compile_tileir(tileir_bytecode, *, symbol, sm_arch, signature):
    """Compile TileIR bytecode into an Xe device binary blob.

    Runs the native tileir-to-mlir tool (overridable with
    ``CUTILE_XPU_TILEIR_TO_MLIR``) followed by the XeVM pipeline in
    :func:`~cuda.tile._backend.xeas.xeas`.
    """
    # symbol/sm_arch/signature are part of the backend hook protocol; the Xe
    # toolchain derives everything it needs from the bytecode itself.
    tool = resolve_tool("XPU", "CUTILE_XPU_TILEIR_TO_MLIR", "tileir-to-mlir")
    options = _current_options()
    mlir = _run_tileir_to_mlir(tool, tileir_bytecode, options)
    return xeas(mlir, **_xeas_options(options))


# ABI type of a runtime scalar, keyed by its signature dtype.
_SCALAR_NUMPY_DTYPE = {
    ct.int8: np.int8, ct.int16: np.int16,
    ct.int32: np.int32, ct.int64: np.int64,
    ct.uint8: np.uint8, ct.uint16: np.uint16,
    ct.uint32: np.uint32, ct.uint64: np.uint64,
    ct.float32: np.float32, ct.float64: np.float64,
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
    for arg, param in zip(args, signature.parameters):
        if isinstance(param, ConstantConstraint):
            continue
        if isinstance(arg, torch.Tensor):
            flat.append((arg.data_ptr(), tuple(arg.shape), arg.stride()))
        elif isinstance(param, ScalarConstraint):
            flat.append(_scalar_arg(arg, param).tobytes())
        else:
            raise TypeError(
                f"XPU backend: unsupported runtime argument {type(arg).__name__}")
    return flat


def launch(stream, grid, kernel, args):
    """Compile and synchronously launch a kernel on the XPU.

    Compilation goes through :func:`compile_tileir`; the resulting device binary
    is loaded and run by the Level Zero runtime with the runtime arguments only
    (compile-time constants are baked into the binary).
    """
    options = _current_options()
    block = _launch_block(options)

    sig = build_signature(kernel, args)
    with _xe_context():
        binary, _ = ct.compile_kernel(kernel, sig)

    runtime_args = _runtime_kernel_args(sig, args)

    # The Level Zero runtime enqueues onto its own immediate command list, which
    # is unordered with respect to ``stream``. Drain the caller's pending work
    # (e.g. the tensor initialisation) first, otherwise it can land after the
    # kernel and overwrite its results. The launch below blocks until the kernel
    # completed, so ordering is restored on the way out.
    if stream is not None:
        stream.synchronize()

    launch_level_zero_module_kernel(
        binary, sig.symbol, runtime_args, _triplet(grid), _triplet(block))
