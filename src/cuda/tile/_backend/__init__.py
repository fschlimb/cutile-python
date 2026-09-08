# SPDX-FileCopyrightText: Copyright (c) <2026> Intel Corporation & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""Pluggable backend hooks for retargeting cuTile to a non-CUDA backend.

cuTile normally compiles a kernel from TileIR bytecode into a CUDA cubin
(see ``cuda.tile._compile.compile_tile``) and launches it through the CUDA
driver in the C++ extension. To run on a different architecture you only need
to provide the core hooks and register them once:

* ``compile_fn`` -- turns TileIR bytecode into an opaque backend binary.
* ``compile_cache_key_fn`` -- identifies the compiler and effective options.
* ``launch_fn``  -- executes a kernel on the backend.
* ``benchmark_fn`` -- measures one synchronous kernel launch.
* ``benchmark_callable_fn`` -- measures a synchronous host callable.
* ``compile_options_fn`` -- scopes backend compile and launch options.

Everything else (parsing Python source to TileIR, building kernel signatures,
caching) is reused unchanged.

Example::

    import cuda.tile as ct

    def my_compile(tileir_bytecode, *, symbol, sm_arch, signature):
        # Hand the TileIR bytecode to your own toolchain.
        return my_toolchain.compile(tileir_bytecode)  # -> bytes

    def my_launch(stream, grid, kernel, args):
        # Build a signature for the concrete runtime arguments, then compile
        # (routes through `my_compile`) and run on your backend.
        signature = build_signature(kernel, args)
        binary, symbol = ct.compile_kernel(kernel, signature)
        my_runtime.run(binary, symbol, grid, stream, args)

    ct.set_backend(compile_fn=my_compile, launch_fn=my_launch)

Either hook may be ``None`` to leave the corresponding stage on the default
CUDA path.
"""
from __future__ import annotations

import importlib
import threading
from typing import Any, Callable, ContextManager, Optional

__all__ = ("set_backend", "clear_backend", "backend_active")

# compile_fn(tileir_bytecode: bytes, *, symbol: str, sm_arch: str,
#            signature: KernelSignature) -> bytes
CompileFn = Callable[..., bytes]

# compile_cache_key_fn() -> str | bytes | None
CompileCacheKeyFn = Callable[[], str | bytes | None]

# launch_fn(stream, grid, kernel, args: tuple) -> Any
LaunchFn = Callable[..., Any]
LaunchCompiledFn = Callable[..., Any]
BenchmarkFn = Callable[..., float]
BenchmarkCallableFn = Callable[..., float]
CompileOptionsFn = Callable[[dict[str, Any]], ContextManager[Any]]

_lock = threading.RLock()
_compile_fn: Optional[CompileFn] = None
_compile_cache_key_fn: Optional[CompileCacheKeyFn] = None
_launch_fn: Optional[LaunchFn] = None
_launch_compiled_fn: Optional[LaunchCompiledFn] = None
_benchmark_fn: Optional[BenchmarkFn] = None
_benchmark_callable_fn: Optional[BenchmarkCallableFn] = None
_compile_options_fn: Optional[CompileOptionsFn] = None
_sm_arch: Optional[str] = None
_bytecode_version: Optional[str] = None


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


def set_backend(module: Optional[str] = None, *,
                compile_fn: Optional[CompileFn] = None,
                compile_cache_key_fn: Optional[CompileCacheKeyFn] = None,
                launch_fn: Optional[LaunchFn] = None,
                launch_compiled_fn: Optional[LaunchCompiledFn] = None,
                benchmark_fn: Optional[BenchmarkFn] = None,
                benchmark_callable_fn: Optional[BenchmarkCallableFn] = None,
                compile_options_fn: Optional[CompileOptionsFn] = None,
                sm_arch: Optional[str] = None,
                bytecode_version: Optional[str] = None) -> None:
    """Register custom backend hooks.

    Args:
        module: Optional module name to import hooks from. When passed as a
            short name (for example ``"xpu"``), the built-in path
            ``cuda.tile._backend.<module>`` is tried first. Keyword arguments
            override module values.
        compile_fn: Converts TileIR bytecode into a backend binary. Called as
            ``compile_fn(tileir_bytecode, symbol=..., sm_arch=..., signature=...)``
            and must return ``bytes``. When ``None`` the default CUDA cubin
            compilation is used.
        compile_cache_key_fn: Returns a deterministic ``str`` or ``bytes``
            identifying all compiler inputs downstream of TileIR. Returning
            ``None`` disables custom-backend compilation caching.
        launch_fn: Executes a kernel on the backend. Called as
            ``launch_fn(stream, grid, kernel, args)``. When ``None`` the default
            CUDA launch path (the C++ extension) is used.
        benchmark_fn: Measures one synchronous kernel launch. Called as
            ``benchmark_fn(stream, grid, kernel, args)`` and returns microseconds.
        benchmark_callable_fn: Measures a synchronous host callable. Called as
            ``benchmark_callable_fn(stream, callable, args)`` and returns microseconds.
        compile_options_fn: Returns a context manager that scopes backend
            compile and launch options. Called as ``compile_options_fn(options)``.
        sm_arch: Optional target string passed to the compiler instead of
            probing a local CUDA device. Useful when no NVIDIA GPU is present.
        bytecode_version: Optional TileIR bytecode version string (for example
            ``"13.1"``) to pin instead of probing the ``tileiras`` compiler.
            Useful when no CUDA toolkit is present.
    """
    global _compile_fn, _compile_cache_key_fn, _launch_fn, _launch_compiled_fn
    global _benchmark_fn, _benchmark_callable_fn, _compile_options_fn
    global _sm_arch, _bytecode_version
    if module is not None:
        mod = _import_backend_module(module)
        if compile_fn is None:
            compile_fn = getattr(mod, "compile_tileir", None)
            if compile_cache_key_fn is None:
                compile_cache_key_fn = getattr(
                    mod, "compile_cache_key", None)
        launch_fn = launch_fn or getattr(mod, "launch", None)
        launch_compiled_fn = launch_compiled_fn or getattr(
            mod, "launch_compiled", None)
        benchmark_fn = benchmark_fn or getattr(mod, "benchmark", None)
        benchmark_callable_fn = benchmark_callable_fn or getattr(
            mod, "benchmark_callable", None)
        compile_options_fn = compile_options_fn or getattr(
            mod, "compile_options", None)
        sm_arch = sm_arch or getattr(mod, "sm_arch", None)
        bytecode_version = bytecode_version or getattr(mod, "bytecode_version", None)

    with _lock:
        _compile_fn = compile_fn
        _compile_cache_key_fn = compile_cache_key_fn
        _launch_fn = launch_fn
        _launch_compiled_fn = launch_compiled_fn
        _benchmark_fn = benchmark_fn
        _benchmark_callable_fn = benchmark_callable_fn
        _compile_options_fn = compile_options_fn
        _sm_arch = sm_arch
        _bytecode_version = bytecode_version


def clear_backend() -> None:
    """Remove any registered backend hooks and restore the default CUDA path."""
    set_backend(module=None, compile_fn=None, compile_cache_key_fn=None,
                launch_fn=None, launch_compiled_fn=None, benchmark_fn=None,
                benchmark_callable_fn=None, compile_options_fn=None,
                sm_arch=None, bytecode_version=None)


def backend_active() -> bool:
    """Return ``True`` if a custom launch hook is registered."""
    return _launch_fn is not None


def get_compile_fn() -> Optional[CompileFn]:
    return _compile_fn


def get_compile_cache_key_fn() -> Optional[CompileCacheKeyFn]:
    """Return the active custom compiler identity callback."""

    return _compile_cache_key_fn


def get_launch_fn() -> Optional[LaunchFn]:
    return _launch_fn


def get_launch_compiled_fn() -> Optional[LaunchCompiledFn]:
    return _launch_compiled_fn


def get_benchmark_fn() -> Optional[BenchmarkFn]:
    return _benchmark_fn


def get_benchmark_callable_fn() -> Optional[BenchmarkCallableFn]:
    return _benchmark_callable_fn


def get_compile_options_fn() -> Optional[CompileOptionsFn]:
    """Return the active backend compile/launch option scoping callback."""

    return _compile_options_fn


def get_sm_arch_override() -> Optional[str]:
    return _sm_arch


def get_bytecode_version_override() -> Optional[str]:
    return _bytecode_version
