# SPDX-FileCopyrightText: Copyright (c) <2026> Intel Corporation. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""CUDA-free stand-in for the native ``_cext`` extension.

The extension is built only when cuda.h is found (see ``BUILD_CUDA_EXTENSION``)
and then shadows this module, because Python prefers extension modules over
source modules of the same name. What remains here is the part of the API that
a non-CUDA backend such as XPU needs while compiling a kernel.
"""

import inspect
from dataclasses import dataclass

from ._context import init_context_config_from_env


def __getattr__(name: str):
    """Keep CUDA-only APIs importable, but fail when one is actually called."""
    if name.startswith("__") and name.endswith("__"):
        raise AttributeError(name)

    def cuda_unavailable(*args, **kwargs):
        raise RuntimeError(
            f"cuda.tile._cext.{name}() requires CUDA, but cuda.tile was built "
            "without it because cuda.h was not found. Select a non-CUDA "
            "backend, e.g. cuda.tile.set_backend('xpu'), or install the CUDA "
            "headers and rebuild.")

    return cuda_unavailable


@dataclass(frozen=True)
class CallingConvention:
    version: int

    @staticmethod
    def cutile_python_v1() -> "CallingConvention":
        return _CUTILE_PYTHON_V1

    @staticmethod
    def from_code(code: str, /) -> "CallingConvention":
        if code == _CUTILE_PYTHON_V1.code:
            return _CUTILE_PYTHON_V1
        raise ValueError(f"Unknown calling convention code '{code}'")

    @property
    def name(self) -> str:
        return f"cutile_python_v{self.version}"

    @property
    def code(self) -> str:
        return f"t{self.version}"

    def __repr__(self) -> str:
        return f"CallingConvention({self.name!r}, {self.code!r})"


_CUTILE_PYTHON_V1 = CallingConvention(1)


class TileDispatcher:
    def __new__(cls, *args, **kwargs):
        # `cuda.tile.kernel` forwards its decorator arguments here.
        return super().__new__(cls)

    def __init__(self, constant_arg_flags, int64_index_flags, int64_param_flags):
        # The flags only drive the CUDA launch path, which is unavailable here.
        pass


class TileContext:
    def __init__(self, *, config):
        self.config = config
        self.autotune_cache = None


default_tile_context = TileContext(config=init_context_config_from_env())


def dev_features_enabled() -> bool:
    return False


def run_coroutine(coro):
    """Drive nested compiler coroutines on an explicit stack.

    ``resume_after`` yields a coroutine instead of awaiting it directly, so
    deeply nested compilation runs at a constant Python stack depth.
    """
    if not inspect.iscoroutine(coro):
        raise TypeError("Expected a coroutine")

    stack = [coro]
    result = None
    error = None
    try:
        while stack:
            try:
                if error is None:
                    nested = stack[-1].send(result)
                else:
                    nested, error = stack[-1].throw(error), None
                result = None
            except StopIteration as stop:
                stack.pop()
                result, error = stop.value, None
                continue
            except BaseException as exc:
                stack.pop()
                result, error = None, exc
                continue

            if not inspect.iscoroutine(nested):
                raise TypeError("Expected a continuation coroutine")
            stack.append(nested)
    finally:
        for pending in reversed(stack):
            pending.close()

    if error is not None:
        raise error
    return result
