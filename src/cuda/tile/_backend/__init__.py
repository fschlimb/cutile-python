# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""Pluggable backend hooks for retargeting cuTile to a non-CUDA backend.

cuTile normally compiles a kernel from TileIR bytecode into a CUDA cubin
(see ``cuda.tile._compile.compile_tile``) and launches it through the CUDA
driver in the C++ extension. To run on a different architecture you only need
to provide two functions and register them once:

* ``compile_fn`` -- turns TileIR bytecode into an opaque backend binary.
* ``launch_fn``  -- executes a kernel on the backend.

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
from typing import Any, Callable, Optional

__all__ = ("set_backend", "clear_backend", "backend_active")

# compile_fn(tileir_bytecode: bytes, *, symbol: str, sm_arch: str,
#            signature: KernelSignature) -> bytes
CompileFn = Callable[..., bytes]

# launch_fn(stream, grid, kernel, args: tuple) -> Any
LaunchFn = Callable[..., Any]

_lock = threading.RLock()
_compile_fn: Optional[CompileFn] = None
_launch_fn: Optional[LaunchFn] = None
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
                launch_fn: Optional[LaunchFn] = None,
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
        launch_fn: Executes a kernel on the backend. Called as
            ``launch_fn(stream, grid, kernel, args)``. When ``None`` the default
            CUDA launch path (the C++ extension) is used.
        sm_arch: Optional target string passed to the compiler instead of
            probing a local CUDA device. Useful when no NVIDIA GPU is present.
        bytecode_version: Optional TileIR bytecode version string (for example
            ``"13.1"``) to pin instead of probing the ``tileiras`` compiler.
            Useful when no CUDA toolkit is present.
    """
    global _compile_fn, _launch_fn, _sm_arch, _bytecode_version
    if module is not None:
        mod = _import_backend_module(module)
        compile_fn = compile_fn or getattr(mod, "compile_tileir", None)
        launch_fn = launch_fn or getattr(mod, "launch", None)
        sm_arch = sm_arch or getattr(mod, "sm_arch", None)
        bytecode_version = bytecode_version or getattr(mod, "bytecode_version", None)

    with _lock:
        _compile_fn = compile_fn
        _launch_fn = launch_fn
        _sm_arch = sm_arch
        _bytecode_version = bytecode_version


def clear_backend() -> None:
    """Remove any registered backend hooks and restore the default CUDA path."""
    set_backend(module=None, compile_fn=None, launch_fn=None, sm_arch=None,
                bytecode_version=None)


def backend_active() -> bool:
    """Return ``True`` if a custom launch hook is registered."""
    return _launch_fn is not None


def get_compile_fn() -> Optional[CompileFn]:
    return _compile_fn


def get_launch_fn() -> Optional[LaunchFn]:
    return _launch_fn


def get_sm_arch_override() -> Optional[str]:
    return _sm_arch


def get_bytecode_version_override() -> Optional[str]:
    return _bytecode_version
