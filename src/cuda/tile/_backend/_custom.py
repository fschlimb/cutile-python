# SPDX-FileCopyrightText: Copyright (c) <2026> Intel Corporation. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""Backend registry and launch dispatch for non-CUDA backends."""

from __future__ import annotations

from collections import OrderedDict
import contextlib
from dataclasses import dataclass
import gc
import importlib
import logging
import threading
import time
from types import MappingProxyType
from typing import Any, Hashable

from cuda.tile._cext import _synchronize_context, default_tile_context
from cuda.tile._cext import launch as _cext_launch
from cuda.tile.compilation import ConstantConstraint, KernelSignature, mangle_kernel_name
from cutile_cache._cache import cache_key, cache_lookup, cache_store, evict_lru

from ._signature import ArrayArgument, array_metadata, build_signature, constant_parameters

logger = logging.getLogger(__name__)

# Attributes every backend provides; see the package docstring.
_BACKEND_ATTRIBUTES = (
    "sm_arch", "bytecode_version", "normalize_options", "compiler_identity",
    "compile", "load", "launch", "synchronize",
)

# Per kernel, at most this many argument layouts and binaries stay cached.
_CACHE_LIMIT = 256

# Base address alignments above this many bytes (log2) share a cache entry.
_MAX_TRACKED_ALIGNMENT_LOG2 = 12

_NO_OPTIONS = MappingProxyType({})

_backend = None
_tls = threading.local()


def _import_backend_module(module: str):
    # Support shorthand names (e.g. "xpu") for built-in backends.
    candidates = [module]
    if "." not in module:
        candidates.insert(0, f"cuda.tile._backend.{module}")

    for name in candidates:
        try:
            return importlib.import_module(name)
        except ModuleNotFoundError as exc:
            # Only ignore "module not found" for the candidate itself.
            if exc.name != name:
                raise
    raise ModuleNotFoundError(
        f"Could not import backend module '{module}'. "
        f"Tried: {', '.join(candidates)}"
    )


def set_backend(backend) -> None:
    """Route :func:`cuda.tile.launch` to a non-CUDA backend.

    Args:
        backend: A backend module (or object) or its name. Short names such
            as ``"cpu"`` or ``"xpu"`` select the built-in backends. ``None``
            restores the default CUDA path.
    """
    global _backend
    if isinstance(backend, str):
        backend = _import_backend_module(backend)
    if backend is not None:
        missing = [name for name in _BACKEND_ATTRIBUTES
                   if not hasattr(backend, name)]
        if missing:
            raise TypeError(
                f"{backend!r} is not a cuTile backend; it lacks "
                f"{', '.join(missing)}")
        hash(backend)  # identifies cached binaries
    _backend = backend


def clear_backend() -> None:
    """Restore the default CUDA path."""
    set_backend(None)


def get_backend():
    """Return the active backend, or ``None`` for CUDA."""
    return _backend


@contextlib.contextmanager
def backend_options(options):
    """Scope options of the active backend for the compiles and launches in the block.

    The backend's ``normalize_options`` validates them. Options that affect
    code generation select the compiled binary; the others, such as the CPU
    backend's ``num_cpu_threads``, only apply when launching.
    """
    previous = getattr(_tls, "options", None)
    _tls.options = dict(options or {})
    try:
        yield
    finally:
        _tls.options = previous


def current_options():
    """Options of the innermost enclosing :func:`backend_options` block."""
    return getattr(_tls, "options", None) or _NO_OPTIONS


def _normalized_options(backend):
    options = current_options()
    memo = getattr(_tls, "normalized", None)
    if memo is not None and memo[0] is backend and memo[1] is options:
        return memo[2]
    normalized = backend.normalize_options(options)
    _tls.normalized = (backend, options, normalized)
    return normalized


@dataclass(frozen=True, eq=False)
class CompiledKernel:
    """A backend binary with everything needed to launch it."""

    backend: Any
    binary: bytes
    signature: KernelSignature
    options: Hashable
    loaded: Any
    constants: tuple[bool, ...]

    @property
    def symbol(self) -> str:
        return self.signature.symbol


class _KernelCache:
    __slots__ = ("constants", "launches", "binaries", "lock")

    def __init__(self, constants: tuple[bool, ...]):
        # Whether each kernel parameter is a compile-time constant.
        self.constants = constants
        # (backend, options, builder, argument layouts) -> CompiledKernel.
        self.launches = OrderedDict()
        # (backend, options, mangled name, symbol) -> CompiledKernel.
        self.binaries = OrderedDict()
        self.lock = threading.Lock()


def _kernel_cache(kernel) -> _KernelCache:
    try:
        return kernel._cutile_backend_cache
    except AttributeError:
        return kernel.__dict__.setdefault(
            "_cutile_backend_cache", _KernelCache(constant_parameters(kernel)))


def _lru_get(cache: OrderedDict, key):
    value = cache.get(key)
    if value is not None:
        try:
            cache.move_to_end(key)
        except KeyError:  # evicted by another thread
            pass
    return value


def _lru_put(cache: OrderedDict, key, value) -> None:
    cache[key] = value
    if len(cache) > _CACHE_LIMIT:
        cache.popitem(last=False)


def _describe(constants, args) -> tuple:
    """Return each argument's :class:`ArrayArgument`, or ``None`` if it is none."""
    if len(args) != len(constants):
        raise TypeError(
            f"kernel expects {len(constants)} arguments, got {len(args)}")
    arrays = []
    for value, constant in zip(args, constants):
        array = None
        if not constant:
            try:
                array = array_metadata(value)
            except TypeError:
                pass
        arrays.append(array)
    return tuple(arrays)


def _argument_key(value, constant: bool, array: ArrayArgument | None):
    """Everything about ``value`` that a signature builder may depend on."""
    if constant:
        return type(value), value
    if array is None:
        return type(value)
    pointer = array.pointer
    alignment = ((pointer & -pointer).bit_length() - 1 if pointer
                 else _MAX_TRACKED_ALIGNMENT_LOG2)
    return (array.dtype, array.shape, array.strides,
            min(alignment, _MAX_TRACKED_ALIGNMENT_LOG2))


def _runtime_arguments(args, constants, arrays) -> tuple:
    """The arguments a backend's ``launch`` receives; see the package docstring."""
    return tuple(value if array is None else array
                 for value, constant, array in zip(args, constants, arrays)
                 if not constant)


def _compile_for_launch(backend, kernel, cache, args, arrays, signature_builder):
    options = _normalized_options(backend)
    key = (backend, options, signature_builder,
           tuple(map(_argument_key, args, cache.constants, arrays)))
    compiled = _lru_get(cache.launches, key)
    if compiled is None:
        with cache.lock:
            compiled = cache.launches.get(key)
            if compiled is None:
                compiled = _compile(kernel, cache, backend,
                                    signature_builder(kernel, args), options)
                _lru_put(cache.launches, key, compiled)
    return compiled


def _active_backend():
    backend = _backend
    if backend is None:
        raise RuntimeError("no backend is active; see cuda.tile.set_backend()")
    return backend


def compile_for_launch(kernel, args, *, signature_builder=build_signature):
    """Return the active backend's :class:`CompiledKernel` for ``args``.

    ``signature_builder(kernel, args)`` may add assumptions to the signature,
    as long as they only depend on constants, scalar types and array
    dtypes, shapes, strides and alignment.
    """
    backend = _active_backend()
    args = tuple(args)
    cache = _kernel_cache(kernel)
    arrays = _describe(cache.constants, args)
    return _compile_for_launch(
        backend, kernel, cache, args, arrays, signature_builder)


def _compile(kernel, cache, backend, signature, options):
    name = kernel._annotated_function.pyfunc.__name__
    key = (backend, options, mangle_kernel_name(name, signature), signature.symbol)
    compiled = _lru_get(cache.binaries, key)
    if compiled is None:
        binary, signature = _compile_binary(kernel, backend, signature, options)
        loaded = backend.load(binary, signature, options)
        constants = tuple(isinstance(parameter, ConstantConstraint)
                          for parameter in signature.parameters)
        compiled = CompiledKernel(
            backend, binary, signature, options, loaded, constants)
        _lru_put(cache.binaries, key, compiled)
    return compiled


def _compile_binary(kernel, backend, signature, options):
    from cuda.tile._compile import compile_tile, parse_bytecode_version

    context = default_tile_context
    result = compile_tile(
        kernel._annotated_function, (signature,), backend.sm_arch,
        kernel._compiler_options, context,
        bytecode_version=parse_bytecode_version(backend.bytecode_version),
        return_bytecode=True, return_cubin=False)
    [signature] = result.kernel_signatures
    bytecode = bytes(result.bytecode)

    cache_dir = context.config.cache_dir
    disk_key = None
    if cache_dir is not None:
        identity = backend.compiler_identity(options)
        if identity is None:
            logger.warning("disk cache disabled: backend compiler identity is unknown")
        else:
            opt_level = kernel._compiler_options.opt_level_for_target(backend.sm_arch)
            disk_key = cache_key(f"custom-backend-v2:{identity}", backend.sm_arch,
                                 opt_level, bytecode)
            binary = cache_lookup(cache_dir, disk_key)
            if binary is not None:
                return binary, signature

    binary = backend.compile(bytecode, signature, options)
    if not isinstance(binary, bytes):
        raise TypeError("backend compile() must return bytes")
    if disk_key is not None:
        cache_store(cache_dir, disk_key, binary)
        evict_lru(cache_dir, context.config.cache_size_limit)
    return binary, signature


def launch(stream, grid, kernel, kernel_args, /):
    """Launch a cuTile kernel.

    Runs on the backend selected with :func:`cuda.tile.set_backend`, if any,
    and on CUDA otherwise.

    Args:
        stream: The stream to execute the |kernel| on.
        grid: Tuple of up to 3 grid dimensions to execute the |kernel| over.
        kernel: The |kernel| to execute.
        kernel_args: Positional arguments to pass to the kernel.
    """
    backend = _backend
    if backend is None:
        return _cext_launch(stream, grid, kernel, kernel_args)
    args = tuple(kernel_args)
    cache = _kernel_cache(kernel)
    arrays = _describe(cache.constants, args)
    compiled = _compile_for_launch(
        backend, kernel, cache, args, arrays, build_signature)
    backend.launch(compiled.loaded, stream, grid,
                   _runtime_arguments(args, cache.constants, arrays))


def launch_compiled(stream, grid, compiled: CompiledKernel, kernel_args, /):
    """Launch ``compiled`` without dispatch lookups.

    ``kernel_args`` must match the arguments ``compiled`` was built for.
    """
    args = tuple(kernel_args)
    arrays = _describe(compiled.constants, args)
    compiled.backend.launch(compiled.loaded, stream, grid,
                            _runtime_arguments(args, compiled.constants, arrays))


def _synchronize(stream):
    if _backend is None:
        _synchronize_context()
    else:
        _backend.synchronize(stream)


def benchmark_callable(stream, fn, args=()) -> float:
    """Time one synchronous call of ``fn(*args)`` in microseconds."""
    _synchronize(stream)
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        start = time.perf_counter_ns()
        fn(*args)
        _synchronize(stream)
        elapsed_ns = time.perf_counter_ns() - start
    finally:
        if gc_was_enabled:
            gc.enable()
    return elapsed_ns / 1_000.0


def benchmark(stream, grid, kernel, args) -> float:
    """Time one launch on the active backend in microseconds."""
    compiled = compile_for_launch(kernel, args)
    return benchmark_callable(
        stream, launch_compiled, (stream, grid, compiled, args))


def synchronize_stream(stream) -> None:
    """Wait for a framework stream; ``None`` and ``0`` stand for no stream."""
    if stream is None or (isinstance(stream, int) and stream == 0):
        return
    synchronize = getattr(stream, "synchronize", None)
    if synchronize is None:
        raise TypeError("stream must be None, 0, or provide synchronize()")
    synchronize()


def normalize_dims(value) -> tuple[int, int, int]:
    """Normalize one to three launch dimensions to a three-tuple."""

    values = (value,) if isinstance(value, int) else tuple(value)
    if not 1 <= len(values) <= 3:
        raise ValueError("launch dimensions must have length 1 to 3")
    return values + (1,) * (3 - len(values))


def check_options(backend: str, options, known) -> None:
    """Reject option keys the backend does not know, e.g. misspelled ones."""
    unknown = sorted(set(options) - set(known))
    if unknown:
        raise ValueError(
            f"{backend} backend: unknown options {', '.join(map(repr, unknown))}; "
            f"expected any of {', '.join(sorted(known))}")


def bool_option(backend: str, options, key: str, default: bool) -> bool:
    """Read a boolean option, rejecting look-alikes such as ``"false"``."""

    value = options.get(key, default)
    if not isinstance(value, bool):
        raise TypeError(
            f"{backend} backend: option {key!r} must be a bool, got {value!r}")
    return value
