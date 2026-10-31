# SPDX-FileCopyrightText: Copyright (c) <2026> Intel Corporation. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""Pluggable backends for running cuTile kernels on non-CUDA targets.

cuTile compiles a kernel's Python source to TileIR bytecode. A backend turns
that bytecode into a binary for its target and launches it. Select one with
:func:`cuda.tile.set_backend`, e.g. ``ct.set_backend("cpu")``; afterwards
:func:`cuda.tile.launch` compiles and runs kernels on it, using the options of
the enclosing :func:`cuda.tile.backend_options` block. Signature derivation,
in-memory and on-disk caching are shared by all backends.

A backend is a module (or any object) providing:

* ``sm_arch``, ``bytecode_version``: compile target and TileIR version.
* ``normalize_options(options) -> Hashable``: validate the options of a
  :func:`backend_options` block, rejecting unknown keys, into a hashable value
  with a stable ``repr`` covering everything that affects ``compile``.
* ``compiler_identity(options) -> str | None``: identify everything besides
  the bytecode that affects ``compile``; ``None`` disables the disk cache.
* ``compile(bytecode, signature, options) -> bytes``.
* ``load(binary, signature, options)``: prepare a binary for launching; called
  once per binary.
* ``launch(loaded, stream, grid, args)``: run ``load``'s result synchronously.
  ``args`` holds the kernel arguments without compile-time constants, with
  each array described by an :class:`ArrayArgument`.
* ``synchronize(stream)``: wait for work queued on ``stream``.
"""

from ._custom import (
    CompiledKernel,
    backend_options,
    benchmark,
    benchmark_callable,
    clear_backend,
    compile_for_launch,
    current_options,
    get_backend,
    launch,
    launch_compiled,
    set_backend,
)
from ._signature import ArrayArgument

__all__ = (
    "ArrayArgument",
    "CompiledKernel",
    "backend_options",
    "benchmark",
    "benchmark_callable",
    "clear_backend",
    "compile_for_launch",
    "current_options",
    "get_backend",
    "launch",
    "launch_compiled",
    "set_backend",
)
