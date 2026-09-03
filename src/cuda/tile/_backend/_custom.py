# SPDX-FileCopyrightText: Copyright (c) <2026> Intel Corporation.
# SPDX-License-Identifier: Apache-2.0
"""Shared launch preparation for non-CUDA backends."""

from dataclasses import dataclass

import cuda.tile as ct
from cuda.tile.compilation import KernelSignature

from ._signature import build_signature


@dataclass(frozen=True)
class CompiledKernel:
    """A backend binary and the concrete signature that produced it."""

    binary: bytes
    symbol: str
    signature: KernelSignature


def launch_compiled(stream, grid, compiled: CompiledKernel, args):
    """Launch a previously resolved custom-backend binary."""
    from cuda.tile import _backend

    launch_fn = _backend.get_launch_compiled_fn()
    if launch_fn is None:
        raise RuntimeError("custom backend does not provide launch_compiled()")
    return launch_fn(stream, grid, compiled, args)


def compile_for_launch(kernel, args, *, signature_builder=build_signature):
    """Build a runtime signature and compile or retrieve its backend binary."""

    signature = signature_builder(kernel, args)
    binary, symbol = ct.compile_kernel(kernel, signature)
    return CompiledKernel(binary, symbol, signature)


def normalize_dims(value) -> tuple[int, int, int]:
    """Normalize one to three launch dimensions to a three-tuple."""

    values = (value,) if isinstance(value, int) else tuple(value)
    if not 1 <= len(values) <= 3:
        raise ValueError("launch dimensions must have length 1 to 3")
    return values + (1,) * (3 - len(values))
