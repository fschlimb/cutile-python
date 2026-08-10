import os
import sys
import shutil
import subprocess
import contextlib
import tempfile
import threading

import numpy as np

import cuda.tile as ct
from cuda.tile.compilation import (
    KernelSignature, CallingConvention,
    ArrayConstraint, ScalarConstraint, ConstantConstraint,
)

from lighthouse.pipeline.xegpu.xeas import xeas
from lighthouse.execution.xegpu.xelaunch import xelaunch
from lighthouse import dialects as _lh_dialects
from mlir import ir

# Target string reported to the compiler. Setting this here makes
# `get_sm_arch_override()` short-circuit so cuTile never probes a CUDA device
# (which would dlopen libcuda) when this backend is active.
# We need an integer, provide a somewhat random mapping from identifiers to ints
def get_sm_arch_xpu(arch: str) -> int:
    _xpus = {"b70": "1070",
             "b50": "1050",
             "pvc": "2000",
             }
    return _xpus.get(arch, 0)

sm_arch = get_sm_arch_xpu("b70")

# Pin the TileIR bytecode version so cuTile does not probe the `tileiras`
# compiler (which requires a CUDA toolkit) to auto-detect it.
bytecode_version = "13.3"

# Host launcher / entry function name shared by the compile stages and xelaunch.
_ENTRY_POINT = "payload"
_BENCH_NAME = "benchmark"

# Single process-wide MLIR context with the lighthouse dialects registered.
# The dialect extensions attach process-global transform-dialect interface
# models on load; registering them more than once corrupts the transform
# interpreter (aborting later with a failed ``cast<TransformOpInterface>``), so
# the context is created and populated exactly once and reused by every launch.
_mlir_context = None


def _xe_context():
    """Return the shared MLIR context, creating and populating it on first use."""
    global _mlir_context
    if _mlir_context is None:
        ctx = ir.Context()
        with ctx:
            _lh_dialects.register_and_load()
        _mlir_context = ctx
    return _mlir_context


_DIVISOR_16 = 16
_BYTE_BITWIDTH = 8
_NB_WORKITEMS = 16

# Per-launch compiler options. `compile_options(...)` sets extra values passed
# to the launcher, scoped to the enclosing `ct.launch` call (it runs on the
# same thread, so a thread-local keeps concurrent calls isolated).
_tls = threading.local()


@contextlib.contextmanager
def compile_options(options: dict):
    """Provide tuning parameters for the XPU compile, per kernel launch.

    ``options`` is a flat dict scoped to launches inside the ``with`` block.
    Its keys are passed straight through to the lighthouse xe* Python APIs:
    the matmul parameters go to ``xeas`` as its ``params`` dict and the
    remaining keys are forwarded as ``xeas`` keyword arguments.

    The work-group tiles ``wg_m``/``wg_n`` must be set explicitly in
    ``options``. ``k_tile`` is optional and only needed by pipelines that use
    it. Combined with ``sg_m``/``sg_n`` and a fixed ``nb_workitems=16``
    they derive the launch block
    ``((wg_m // sg_m) * (wg_n // sg_n) * 16, 1, 1)``.

        Accepted options:

        =================== ========== ========================== ==========================
        key                 kernels    required?                  meaning
        =================== ========== ========================== ==========================
        block               all        optional                   launch block (overrides derived)
        wg_m, wg_n          all        yes                        work-group tile sizes
        m, n, k             matmul     by matmul schedule         problem sizes
        k_tile              matmul     optional                   reduction tile size
        transpose_a         matmul     optional                   A is stored transposed
        transpose_b         matmul     optional                   B is stored transposed
        device              matmul     optional                   target GPU for selection
        sg_m, sg_n          all        optional                   subgroup tile size
        load_a_m, load_a_k  matmul     optional                   DPAS load tile for A
        load_b_k, load_b_n  matmul     optional                   DPAS load tile for B
        load_m, load_n      elemwise   optional                   elementwise load tile sizes
        prefetch_a_m,       matmul     optional                   cooperative prefetch A
            prefetch_a_k
        prefetch_b_k,       matmul     optional                   cooperative prefetch B
            prefetch_b_n
        prefetch_a_nb       matmul     optional                   initial A prefetch count
        prefetch_b_nb       matmul     optional                   initial B prefetch count
        assume_in_bounds    all        optional                   mark transfers in-bounds
        xegpu_op_level      all        optional                   initial XeGPU op level
        large_register_file all        optional                   enable large register file
        flops               all        optional                   benchmark flop count (xelaunch)
        nwarmup             all        optional                   benchmark warmup runs
        nruns               all        optional                   benchmark timed runs
        =================== ========== ========================== ==========================

        For elemwise-like kernels (no ``m/n/k``), if ``sg_m/sg_n/load_m/load_n`` are
        omitted, this backend derives compatible defaults from ``wg_m/wg_n``.

    The matmul subgroup / prefetch tile keys (``sg_*`` through
    ``prefetch_*_nb``) may be given partially: xeas keeps what is provided and
    derives the parameters that depend on it.

    Example::

        with xpu.compile_options({"m": M, "n": N, "k": K}):
            ct.launch(stream, grid, matmul_kernel, args)
    """
    prev = getattr(_tls, "options", None)
    _tls.options = dict(options or {})
    try:
        yield
    finally:
        _tls.options = prev


# Matmul parameters understood by xeas (its ``params`` dict). ``compile_options``
# keys matching these names are passed straight through; ``m``/``n``/``k`` are
# required and the subgroup/prefetch tile keys are optional (xeas derives the
# missing ones). ``wg_m``/``wg_n``/``k_tile`` are provided separately as
# explicit compile options and injected into this params dict.
_XEAS_PARAM_KEYS = (
    "m", "n", "k", "device", "transpose_a", "transpose_b",
    "sg_m", "sg_n",
    "load_a_m", "load_a_k", "load_b_k", "load_b_n",
    "load_m", "load_n",
    "prefetch_a_m", "prefetch_a_k", "prefetch_b_k", "prefetch_b_n",
    "prefetch_a_nb", "prefetch_b_nb",
)

# Fallback launch block when the tile options needed to derive it are absent
# (matches xeaddlauncher's own default).
_DEFAULT_BLOCK = (512, 1, 1)


def _tile_from_options(options: dict) -> tuple[int, int, int | None]:
    """Read required work-group tiles and optional reduction tile from options."""
    missing = [k for k in ("wg_m", "wg_n") if k not in options]
    if missing:
        raise ValueError(
            "XPU backend: missing required compile_options keys: "
            f"{', '.join(missing)}"
        )
    k_tile = options.get("k_tile")
    return int(options["wg_m"]), int(options["wg_n"]), (None if k_tile is None else int(k_tile))


def _xeas_params(options: dict, wg_m: int, wg_n: int, k_tile: int | None) -> dict:
    """Build ``xeas`` params from flat options and optional signature tiles."""
    params = {k: options[k] for k in _XEAS_PARAM_KEYS if k in options}
    if wg_m is not None and wg_n is not None:
        params.setdefault("wg_m", wg_m)
        params.setdefault("wg_n", wg_n)

    # Elementwise schedules can be launched with only wg tiles provided.
    # In that case xeas would otherwise inject fixed defaults (sg_n=32,
    # load_n=16), which can violate constraints for skinny tiles such as
    # wg_n=1. Derive compatible defaults from wg tiles unless the caller
    # already provided explicit sg/load values.
    is_elemwise_like = not all(k in options for k in ("m", "n", "k"))
    if is_elemwise_like and wg_m is not None and wg_n is not None:
        sg_m = int(params.setdefault("sg_m", min(32, wg_m)))
        sg_n = int(params.setdefault("sg_n", min(32, wg_n)))
        params.setdefault("load_m", min(8, sg_m))
        params.setdefault("load_n", min(16, sg_n))

    if k_tile is not None:
        params.setdefault("k_tile", k_tile)
    return params


def _block(options: dict, block_threads, wg_m: int, wg_n: int) -> tuple:
    """Launch block ``((wg_m // sg_m) * (wg_n // sg_n) * 16, 1, 1)``.
    ``wg_m``/``wg_n`` are supplied from options. Falls back to
    :data:`_DEFAULT_BLOCK` unless ``sg_m``/``sg_n`` are both provided.
    """
    if block_threads is not None:
        return block_threads
    if wg_m is None or wg_n is None:
        return _DEFAULT_BLOCK
    keys = ("sg_m", "sg_n")
    if not all(k in options for k in keys):
        return _DEFAULT_BLOCK
    sg_m, sg_n = (options[k] for k in keys)
    return ((wg_m // sg_m) * (wg_n // sg_n) * _NB_WORKITEMS, 1, 1)


def _xeas_api(options: dict,
              wg_m: int | None,
              wg_n: int | None,
              k_tile: int | None) -> tuple[dict, dict]:
    """Build ``(params, kwargs)`` for xeas from flat options."""
    params = _xeas_params(options, wg_m, wg_n, k_tile)
    # print(options)
    kwargs = {
        "assume_in_bounds": options.get("assume_in_bounds", False),
        "xegpu_op_level": options.get("xegpu_op_level", "workgroup"),
        "large_register_file": options.get("large_register_file", True),
    }
    return params, kwargs


def _xelaunch_api(options: dict) -> dict:
    """Build ``kwargs`` for xelaunch from flat options."""
    zpath = os.environ.get("LZ_RT_LIB_PATH", None)
    assert zpath is not None, "XPU backend: LZ_RT_LIB_PATH must be set to your libmlir_levelzero_runtime.so"
    kwargs = {
        # "flops": options.get("flops", None),
        # "nwarmup": options.get("nwarmup", 500),
        # "nruns": options.get("nruns", 1000),
        "library_path": zpath,
    }
    return kwargs


# Map torch dtypes to cuTile DTypes. Extend as needed.
def _torch_to_ct_dtype(t):
    import torch
    return {
        torch.float32: ct.float32, torch.float16: ct.float16,
        torch.bfloat16: ct.bfloat16, torch.float64: ct.float64,
        torch.int32: ct.int32, torch.int64: ct.int64,
        torch.int16: ct.int16, torch.int8: ct.int8,
        torch.uint8: ct.uint8, torch.bool: ct.int8,
    }[t]


# Map torch dtypes to MLIR element type names, matching tileir-to-mlir output.
def _torch_to_mlir_elem(t) -> str | None:
    import torch
    return {
        torch.float32: "f32", torch.float16: "f16",
        torch.bfloat16: "bf16", torch.float64: "f64",
        torch.int32: "i32", torch.int64: "i64",
        torch.int16: "i16", torch.int8: "i8",
        torch.uint8: "i8", torch.bool: "i8",
    }.get(t)


def _input_shape_from_args(signature: KernelSignature, args: tuple) -> str | None:
    """Build the xeas ``input_shape`` string from the runtime tensor arguments.

    Produces one ``D0xD1x...xTYPE`` descriptor per runtime (non-constant) kernel
    argument, in the order the outlined ``gpu.func`` receives its memref
    arguments. Passing this lets xeas rewrite the kernel's dynamically shaped
    memrefs to static shapes, which is required for the XeGPU block-descriptor
    lowering to succeed. Returns ``None`` when a runtime argument is not a
    shaped tensor (the rewrite needs a memref descriptor for every argument).
    """
    import torch
    descriptors = []
    for arg, param in zip(args, signature.parameters):
        if isinstance(param, ConstantConstraint):
            continue
        if not isinstance(arg, torch.Tensor) or arg.ndim == 0:
            return None
        elem = _torch_to_mlir_elem(arg.dtype)
        if elem is None:
            return None
        dims = "x".join(str(int(d)) for d in arg.shape)
        descriptors.append(f"{dims}x{elem}")
    return ",".join(descriptors) if descriptors else None


def _array_constraint_from_torch(tensor,
                                 *,
                                 index_dtype,
                                 base_addr_divisible_by,
                                 alias_group=None) -> ArrayConstraint:
    # cuTile assumes a dense, C-contiguous layout for the pointer it receives.
    if not tensor.is_contiguous():
        raise ValueError(
            f"Expected a contiguous tensor (shape={tuple(tensor.shape)}, "
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
                    f"shape={shape} dim {i} (size={d}, stride={s}) exceeds int32 "
                    f"index range; annotate the parameter with ct.int64 index dtype."
                )

    div16_bits = _DIVISOR_16 * _BYTE_BITWIDTH
    stride_divisor = div16_bits // bits if div16_bits % bits == 0 else 1

    stride_constant, stride_divisible_by, shape_divisible_by = [], [], []
    for i in range(ndim):
        s, d = strides[i], shape[i]
        stride_constant.append(1 if s == 1 else None)
        shape_divisible_by.append(_DIVISOR_16 if d % _DIVISOR_16 == 0 else 1)
        stride_divisible_by.append(stride_divisor if s % stride_divisor == 0 else 1)

    return ArrayConstraint(
        dtype=dtype,
        ndim=ndim,
        index_dtype=index_dtype,
        stride_lower_bound_incl=0,
        alias_groups=() if alias_group is None else (alias_group,),
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
    raise TypeError(f"Unsupported scalar type: {type(value).__name__}")


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
    import torch
    cc = calling_convention or CallingConvention.cutile_python_v1()

    af = kernel._annotated_function
    const_mask = af.constant_parameter_mask
    i64_index_mask = af.int64_index_parameter_mask
    i64_scalar_mask = af.int64_parameter_mask

    n = len(const_mask)
    if len(args) != n:
        raise TypeError(f"kernel expects {n} arguments, got {len(args)}")

    constraints = []
    for i, a in enumerate(args):
        print(f"arg {i}: {a} (type={type(a).__name__}) {'constant' if const_mask[i] else 'runtime'}")
        if const_mask[i]:
            if not isinstance(a, bool | int | float):
                raise TypeError(
                    f"constant parameter #{i} must be bool/int/float, "
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


def _run_tileir_to_mlir(tool: str, bytecode: bytes) -> str:
    """Lower TileIR bytecode to 'outlined' MLIR text via the tileir-to-mlir CLI.

    This stage is a native tool (not a lighthouse Python module), so it is still
    run as a subprocess. Raises ``RuntimeError`` with the captured stderr on
    failure.
    """
    options = getattr(_tls, "options", None) or {}
    block_threads = options.get("block_threads")
    wg_m, wg_n, _ = _tile_from_options(options)
    block = _block(options, block_threads, wg_m, wg_n)
    assume_in_bounds = str(options.get("assume_in_bounds", False)).lower()
    argv = [tool,
            f"--tileir-to-mlir-pipeline=drop-rounding-modes=true known-block-size={','.join(map(str, block))} assume-in-bounds={assume_in_bounds}",
            "--convert-memref-args-to-ranked-memref=remove-unused=assumed-memref-dependent",
            "--loop-invariant-code-motion", "-canonicalize", "-cse",
            "--mlir-print-ir-after-all"]
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
    print(proc.stdout.decode())
    return proc.stdout.decode()


def compile_tileir(tileir_bytecode, *, symbol, sm_arch, signature):
    """Compile TileIR bytecode to an XPU shared library via the XeGPU toolchain.

    Runs two stages:
        tileir-to-mlir (CLI) -> xeas (Python)

    The native tileir-to-mlir tool can be overridden with ``CUTILE_XPU_TILEIR_TO_MLIR``.
    """
    tileir_to_mlir = _resolve_tool("CUTILE_XPU_TILEIR_TO_MLIR", "tileir-to-mlir")
    options = getattr(_tls, "options", None) or {}
    wg_m, wg_n, k_tile = _tile_from_options(options)
    xeas_params, xeas_kwargs = _xeas_api(options, wg_m, wg_n, k_tile)

    # Rewrite the kernel's dynamic memref args to static shapes so xeas can
    # build XeGPU block descriptors; explicit option overrides take precedence.
    input_shape = options.get("input_shape") or getattr(_tls, "input_shape", None)
    if input_shape:
        xeas_kwargs["input_shape"] = input_shape

    mlir = _run_tileir_to_mlir(tileir_to_mlir, tileir_bytecode)
    # print(xeas_params, xeas_kwargs, file=sys.stderr)
    result = xeas(mlir, xeas_params, **xeas_kwargs)
    return result


def _scalar_arg(value, constraint: ScalarConstraint):
    """Convert a runtime scalar to the NumPy value matching its ABI type."""
    dtype = {
        ct.int8: np.int8, ct.int16: np.int16,
        ct.int32: np.int32, ct.int64: np.int64,
        ct.uint8: np.uint8, ct.uint16: np.uint16,
        ct.uint32: np.uint32, ct.uint64: np.uint64,
        ct.float32: np.float32, ct.float64: np.float64,
    }.get(constraint.dtype)
    if dtype is None:
        raise TypeError(f"XPU backend: unsupported scalar dtype {constraint.dtype}")
    return dtype(value)


def _runtime_kernel_args(signature: KernelSignature, args: tuple) -> list:
    """Build the kernel argument list expected by the Xe kernel.

    Compile-time constants are dropped (they are baked into the binary), tensors
    are forwarded as-is (``xelaunch`` expands them into memref descriptors) and
    scalars become NumPy values carrying their signature dtype (``xelaunch``
    converts both to the kernel ABI).
    """
    import torch
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
    """Compile and launch kernel on the XPU.

    Compiles the kernel to a shared library, writes it to a temporary file and
    hands it to the ``xelaunch`` Python API together with the runtime tensor
    arguments (compile-time constants removed), which loads the library and runs
    the entry function once.
    """
    options = getattr(_tls, "options", None) or {}
    run_kwargs = _xelaunch_api(options)

    sig = build_signature(kernel, args)
    input_shape = _input_shape_from_args(sig, args)
    with _xe_context():
        _tls.grid = grid
        _tls.input_shape = input_shape
        binary, _ = ct.compile_kernel(kernel, sig)
            
        # xelaunch only sees runtime values. Drop compile-time constants from args
        # and expand tensors into the memref descriptors the kernel ABI expects.
        runtime_args = _runtime_kernel_args(sig, args)

        # The Level Zero runtime behind xelaunch enqueues onto its own immediate
        # command list, which is unordered with respect to ``stream``. Drain the
        # caller's pending work (e.g. the tensor initialisation) first, otherwise
        # it can land after the kernel and overwrite its results. xelaunch blocks
        # until the kernel completed, so ordering is restored on the way out.
        if stream is not None and hasattr(stream, "synchronize"):
            stream.synchronize()

        block_threads = options.get("block_threads")
        wg_m, wg_n, _ = _tile_from_options(options)
        block = _block(options, block_threads, wg_m, wg_n)
        xelaunch(binary, sig.symbol, runtime_args, grid, block, **run_kwargs)
