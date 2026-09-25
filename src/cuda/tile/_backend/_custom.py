# SPDX-FileCopyrightText: Copyright (c) <2026> Intel Corporation.
# SPDX-License-Identifier: Apache-2.0
"""Shared launch preparation for non-CUDA backends."""

from dataclasses import dataclass
import threading

import cuda.tile as ct
from cuda.tile.compilation import KernelSignature

from ._signature import array_metadata
from ._signature import build_signature

_SIGNATURE_CACHE_LIMIT = 128
_COMPILED_LAUNCH_CACHE_LIMIT = 128

# Base address assumptions beyond this many bytes are not distinguished by the cache key.
_MAX_TRACKED_ALIGNMENT_LOG2 = 12


@dataclass(frozen=True)
class CompiledKernel:
    """A backend binary and the concrete signature that produced it."""

    binary: bytes
    symbol: str
    signature: KernelSignature


def _compiler_cache_key():
    from cuda.tile import _backend

    compile_fn = _backend.get_compile_fn()
    cache_key_fn = _backend.get_compile_cache_key_fn()
    if compile_fn is None or cache_key_fn is None:
        return None
    compiler_identity = cache_key_fn()
    if not isinstance(compiler_identity, str | bytes):
        return None
    return (
        id(compile_fn),
        compiler_identity,
        _backend.get_sm_arch_override(),
        _backend.get_bytecode_version_override(),
    )


def launch_compiled(stream, grid, compiled: CompiledKernel, args):
    """Launch a previously resolved custom-backend binary."""
    from cuda.tile import _backend

    launch_fn = _backend.get_launch_compiled_fn()
    if launch_fn is None:
        raise RuntimeError("custom backend does not provide launch_compiled()")
    return launch_fn(stream, grid, compiled, args)


def _argument_key(value):
    """Cheap stand-in for the constraint that `build_signature` derives from `value`."""

    try:
        pointer, shape, strides, dtype = array_metadata(value)
    except TypeError:
        return type(value), value

    if pointer == 0:
        alignment = _MAX_TRACKED_ALIGNMENT_LOG2
    else:
        alignment = min((pointer & -pointer).bit_length() - 1,
                        _MAX_TRACKED_ALIGNMENT_LOG2)
    return dtype, shape, strides, alignment


def _build_signature_cached(kernel, args, signature_builder):
    cache = kernel._launch_signature_cache
    key = (signature_builder, tuple(_argument_key(a) for a in args))
    signature = cache.get(key)
    if signature is None:
        signature = signature_builder(kernel, args)
        if len(cache) >= _SIGNATURE_CACHE_LIMIT:
            cache.clear()
        cache[key] = signature
    return signature


def compile_for_launch(kernel, args, *, signature_builder=build_signature):
    """Build a runtime signature and compile or retrieve its backend binary."""

    signature = _build_signature_cached(kernel, args, signature_builder)
    compiler_key = _compiler_cache_key()
    if compiler_key is None:
        binary, symbol = ct.compile_kernel(kernel, signature)
        return CompiledKernel(binary, symbol, signature)

    cache = getattr(kernel, "_custom_launch_cache", None)
    if cache is None:
        cache = {}
        kernel._custom_launch_cache = cache
    key = (id(signature_builder), id(signature), compiler_key)
    compiled = cache.get(key)
    if compiled is not None:
        return compiled

    lock = getattr(kernel, "_custom_compile_lock", None)
    if lock is None:
        lock = threading.RLock()
        kernel._custom_compile_lock = lock
    with lock:
        compiled = cache.get(key)
        if compiled is None:
            binary, symbol = ct.compile_kernel(kernel, signature)
            compiled = CompiledKernel(binary, symbol, signature)
            if len(cache) >= _COMPILED_LAUNCH_CACHE_LIMIT:
                cache.clear()
            cache[key] = compiled
    return compiled


def normalize_dims(value) -> tuple[int, int, int]:
    """Normalize one to three launch dimensions to a three-tuple."""

    values = (value,) if isinstance(value, int) else tuple(value)
    if not 1 <= len(values) <= 3:
        raise ValueError("launch dimensions must have length 1 to 3")
    return values + (1,) * (3 - len(values))
