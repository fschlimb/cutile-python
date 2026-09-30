# SPDX-FileCopyrightText: Copyright (c) <2026> Intel Corporation. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""Pluggable backends for running cuTile kernels on non-CUDA targets.

cuTile compiles a kernel's Python source to TileIR bytecode. A backend turns
that bytecode into a binary for its target and launches it. Select one with
:func:`cuda.tile.set_backend`, e.g. ``ct.set_backend("cpu")``; afterwards
:func:`cuda.tile.launch` compiles and runs kernels on it. Signature
derivation, in-memory and on-disk caching are shared by all backends.

A backend is a module (or any object) providing:

* ``sm_arch``, ``bytecode_version``: compile target and TileIR version.
* ``normalize_options(options) -> Hashable``: validate the options of a
  :func:`compile_options` block into a hashable value with a stable ``repr``.
* ``compiler_identity(options) -> str | None``: identify everything besides
  the bytecode that affects ``compile``; ``None`` disables the disk cache.
* ``compile(bytecode, signature, options) -> bytes``.
* ``load(binary, signature, options)``: prepare a binary for launching; called
  once per binary.
* ``launch(loaded, stream, grid, args)``: run ``load``'s result synchronously.
* ``synchronize(stream)``: wait for work queued on ``stream``.
"""

from ._custom import (
    CompiledKernel,
    benchmark,
    benchmark_callable,
    clear_backend,
    compile_for_launch,
    compile_options,
    current_options,
    get_backend,
    launch,
    launch_compiled,
    set_backend,
)

__all__ = (
    "CompiledKernel",
    "benchmark",
    "benchmark_callable",
    "clear_backend",
    "compile_for_launch",
    "compile_options",
    "current_options",
    "get_backend",
    "launch",
    "launch_compiled",
    "set_backend",
)
