# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""cuTile backend targeting Intel XPUs through XeGPU and Level Zero.

Compilation runs the native ``tileir-to-mlir`` tool followed by the XeVM
pipeline (:mod:`cuda.tile._backend.xeas`); the resulting device binary is
launched through :mod:`cuda.tile._backend.level_zero_ctypes`.

Select the backend and supply the mandatory tuning parameters::

    ct.set_backend("xpu")
    with xpu.compile_options({"wg_m": 128, "wg_n": 128}):
        ct.launch(stream, grid, matmul_kernel, args)

The Level Zero runtime wrapper library is located through ``LZ_RT_LIB_PATH``.
"""
import os
import sys
import shutil
import subprocess
import contextlib
import threading

import numpy as np

import cuda.tile as ct
from cuda.tile.compilation import (
    KernelSignature, CallingConvention,
    ArrayConstraint, ScalarConstraint, ConstantConstraint,
)

from mlir import ir

from .xeas import xeas
from .level_zero_ctypes import launch_level_zero_module_kernel

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


_ALIGN_BYTES = 16
_SHAPE_DIVISOR = 16
_BITS_PER_BYTE = 8
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


def _xeas_options(options: dict) -> dict:
    """Build the :func:`~cuda.tile._backend.xeas.xeas` keyword arguments."""
    return {
        "xegpu_op_level": options.get("xegpu_op_level", "workgroup"),
        "large_register_file": options.get("large_register_file", True),
    }


def _runtime_library_path() -> str:
    """Path to the Level Zero runtime wrapper library."""
    path = os.environ.get("LZ_RT_LIB_PATH")
    if path is None:
        raise RuntimeError(
            "XPU backend: LZ_RT_LIB_PATH must be set to your "
            "libmlir_levelzero_runtime.so")
    return path


_torch = None
_TORCH_TO_CT_DTYPE: dict = {}


def _torch_api():
    """Import torch on first use; only tensor arguments need it."""
    global _torch
    if _torch is None:
        import torch
        _TORCH_TO_CT_DTYPE.update({
            torch.float32: ct.float32, torch.float16: ct.float16,
            torch.bfloat16: ct.bfloat16, torch.float64: ct.float64,
            torch.int32: ct.int32, torch.int64: ct.int64,
            torch.int16: ct.int16, torch.int8: ct.int8,
            torch.uint8: ct.uint8, torch.bool: ct.int8,
        })
        _torch = torch
    return _torch


def _torch_to_ct_dtype(dtype):
    _torch_api()
    try:
        return _TORCH_TO_CT_DTYPE[dtype]
    except KeyError:
        raise TypeError(
            f"XPU backend: unsupported tensor dtype {dtype}") from None


def _array_constraint_from_torch(tensor,
                                 *,
                                 index_dtype,
                                 base_addr_divisible_by) -> ArrayConstraint:
    # cuTile assumes a dense, C-contiguous layout for the pointer it receives.
    if not tensor.is_contiguous():
        raise ValueError(
            f"XPU backend: expected a contiguous tensor (shape={tuple(tensor.shape)}, "
            f"strides={tuple(tensor.stride())}); call .contiguous() before launch."
        )

    dtype = _torch_to_ct_dtype(tensor.dtype)
    shape = tuple(tensor.shape)
    strides = tuple(tensor.stride())      # real strides, in elements
    ndim = len(shape)
    bits = dtype.bitwidth

    if index_dtype is ct.int32:
        i32_max = (1 << 31) - 1
        for i, (d, s) in enumerate(zip(shape, strides)):
            if d > i32_max or s > i32_max:
                raise TypeError(
                    f"XPU backend: shape={shape} dim {i} (size={d}, stride={s}) exceeds "
                    f"int32 index range; annotate the parameter with ct.int64 index dtype."
                )

    # The alignment guarantee is in bytes; it only translates into an element
    # count when the element size divides it evenly.
    align_bits = _ALIGN_BYTES * _BITS_PER_BYTE
    stride_divisor = align_bits // bits if align_bits % bits == 0 else 1

    stride_constant, stride_divisible_by, shape_divisible_by = [], [], []
    for i in range(ndim):
        s, d = strides[i], shape[i]
        stride_constant.append(1 if s == 1 else None)
        shape_divisible_by.append(_SHAPE_DIVISOR if d % _SHAPE_DIVISOR == 0 else 1)
        stride_divisible_by.append(stride_divisor if s % stride_divisor == 0 else 1)

    return ArrayConstraint(
        dtype=dtype,
        ndim=ndim,
        index_dtype=index_dtype,
        stride_lower_bound_incl=0,
        alias_groups=(),
        may_alias_internally=False,
        stride_constant=tuple(stride_constant),
        stride_divisible_by=tuple(stride_divisible_by),
        shape_divisible_by=tuple(shape_divisible_by),
        base_addr_divisible_by=base_addr_divisible_by,
    )


def _scalar_constraint(value, *, int64) -> ScalarConstraint:
    if isinstance(value, bool):
        return ScalarConstraint(ct.int8)
    if isinstance(value, int):
        return ScalarConstraint(ct.int64 if int64 else ct.int32)
    if isinstance(value, float):
        return ScalarConstraint(ct.float32)
    raise TypeError(f"XPU backend: unsupported scalar type: {type(value).__name__}")


def build_signature(kernel, args,
                    *,
                    calling_convention=None,
                    base_addr_divisible_by=16,
                    symbol=None) -> KernelSignature:
    """cuTile signature for XPU (or any non-CUDA) contiguous tensors, bypassing
    the CUDA-only inspection in get_parameter_constraints_from_pyargs.

    Constant / scalar / array classification is driven by the kernel's own
    parameter annotations, not by guessing from Python types.
    """
    torch = _torch_api()
    cc = calling_convention or CallingConvention.cutile_python_v1()

    af = kernel._annotated_function
    const_mask = af.constant_parameter_mask
    i64_index_mask = af.int64_index_parameter_mask
    i64_scalar_mask = af.int64_parameter_mask

    n = len(const_mask)
    if len(args) != n:
        raise TypeError(f"XPU backend: kernel expects {n} arguments, got {len(args)}")

    constraints = []
    for i, a in enumerate(args):
        if const_mask[i]:
            if not isinstance(a, bool | int | float):
                raise TypeError(
                    f"XPU backend: constant parameter #{i} must be bool/int/float, "
                    f"got {type(a).__name__}")
            constraints.append(ConstantConstraint(a))
        elif isinstance(a, torch.Tensor):
            constraints.append(
                _array_constraint_from_torch(
                    a,
                    index_dtype=ct.int64 if i64_index_mask[i] else ct.int32,
                    base_addr_divisible_by=base_addr_divisible_by))
        else:
            constraints.append(_scalar_constraint(a, int64=i64_scalar_mask[i]))

    sig = KernelSignature(constraints, cc, symbol)
    if symbol is None:
        sig = sig.with_mangled_symbol(af.pyfunc.__name__)
    return sig


def _resolve_tool(env_var: str, default_name: str) -> str:
    """Resolve a toolchain executable.

    Uses the path in ``env_var`` if set, otherwise looks up ``default_name`` on
    ``$PATH``. Raises a clear error if the tool cannot be found.
    """
    override = os.environ.get(env_var)
    candidate = override or default_name
    resolved = shutil.which(candidate)
    if resolved is not None:
        return resolved
    # Allow an explicit path that shutil.which() may miss (e.g. not on PATH).
    if override and os.path.isfile(override) and os.access(override, os.X_OK):
        return override
    hint = f"{env_var}={override!r}" if override else f"$PATH (or set {env_var})"
    raise FileNotFoundError(
        f"XPU backend: could not find '{default_name}' via {hint}.")


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
            "--loop-invariant-code-motion", "-canonicalize", "-cse"]
    try:
        proc = subprocess.run(argv, input=bytecode, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, check=False)
    except OSError as exc:
        raise RuntimeError(
            f"XPU backend: failed to execute tileir-to-mlir ({tool}): {exc}") from exc
    if proc.returncode != 0:
        stderr = proc.stderr.decode(errors="replace").strip()
        raise RuntimeError(
            f"XPU backend: tileir-to-mlir failed (exit code {proc.returncode}).\n"
            f"  command: {' '.join(argv)}\n"
            f"  stderr:  {stderr or '<empty>'}")

    # MLIR pass-manager print flags (for example --mlir-print-ir-before-all)
    # emit to stderr. We capture stderr above to improve failures, so mirror it
    # back to the process stderr on success to keep debug output visible.
    if proc.stderr:
        sys.stderr.write(proc.stderr.decode(errors="replace"))
        sys.stderr.flush()

    mlir = proc.stdout.decode(errors="replace")
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
    tool = _resolve_tool("CUTILE_XPU_TILEIR_TO_MLIR", "tileir-to-mlir")
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

    Compile-time constants are dropped (they are baked into the binary), tensors
    are forwarded as-is (the Level Zero launcher expands them into memref
    descriptors) and scalars become NumPy values carrying their signature dtype.
    """
    torch = _torch_api()
    flat = []
    for arg, param in zip(args, signature.parameters):
        if isinstance(param, ConstantConstraint):
            continue
        if isinstance(arg, torch.Tensor):
            flat.append(arg)
        elif isinstance(param, ScalarConstraint):
            flat.append(_scalar_arg(arg, param))
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
    library_path = _runtime_library_path()
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
        binary, sig.symbol, runtime_args, [], grid, block,
        library_path=library_path)
