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

A backend is a module or object implementing the :class:`Backend` protocol.
Launches with an array argument on a CUDA device always run on CUDA.
"""

from ._custom import (
    Backend,
    CompiledKernel,
    backend_for,
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
    "Backend",
    "CompiledKernel",
    "backend_for",
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
