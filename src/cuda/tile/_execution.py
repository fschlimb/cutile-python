# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
import dataclasses
import functools
from types import FunctionType
from typing import TYPE_CHECKING

from cuda.tile._by_target import ByTarget
from cuda.tile._cext import TileDispatcher, TileContext
from cuda.tile._cext import launch as _cext_launch
from cuda.tile._cext import default_tile_context
from cuda.tile._dispatch_mode import DispatchMode

if TYPE_CHECKING:
    from cuda.tile.compilation import KernelSignature

__all__ = ("function", "kernel", "stub", "launch", "compile_kernel")


###############################################################################
# Decorators


def function(func=None, /, *, host=False, tile=True):
    """*Tile functions* are functions that are usable in |tile code|.

    This decorator indicates what |execution spaces| a function can be called from.
    With no arguments, it denotes a tile-only function.

    When an unannotated function is called by a |tile function|, tile shall be added to the
    unannotated function's execution space.
    This process is recursive.
    No explicit annotation is required.

    The types usable as parameters to a |tile function| are described in the |data model|.

    Args:
        host (bool, optional): Whether the function can be called from |host code|.
            Default is False.
        tile (bool, optional): Whether the function can be called from |tile code|.
            Default is True.
    """
    def decorator(func):
        if host:
            return func
        else:
            @functools.wraps(func)
            def wrapped(*args, **kwargs):
                return DispatchMode.get_current().call_tile_function_from_host(
                        wrapped, args, kwargs)
            wrapped._cutile_function_wrapper = True
            return wrapped

    if func is None:
        return decorator
    else:
        return decorator(func)


class kernel(TileDispatcher):
    """A *tile kernel* is a function executed by each |block| in a |grid|.

    Functions with this decorator are |kernels|.

    |Kernels| are the entry points of |tile code|.
    Their |execution space| shall be only |tile code|; they cannot be called from |host code|.

    Kernels cannot be called directly. Instead, use :py:func:`launch` to
    queue a kernel for execution over a grid.

    The types usable as parameters to a |kernel| are described in the |data model|.

    Args:
        num_ctas: Number of CTAs in a CGA. Must be a power of 2 between 1 and 16, inclusive.
            Default: None (auto).
        occupancy: Expected number of active CTAs per SM, [1, 32]. Default: None (auto).
        opt_level: Optimization level [0, 3], default 3.
        num_worker_warps: Number of warps in the CUDA core warp groups in a
            warp-specialized kernel. The compiler may add warps
            (e.g., for asynchronous memory transfers) that are not counted here.
            This value does not represent the total warp count.
            It's worth tuning when a warp-specialized kernel has high register pressure
            that other approaches cannot resolve.
            Normalization-style kernels with large tiles are the canonical cases.
            Must be either 4 or 8.
            Default: None (auto).
            Since CTK 13.3. Ignored with a warning otherwise.

    Target-specific values for the compiler options above can be provided
    using a :py:class:`ByTarget` object.
    """
    def __new__(cls, function=None, /, **kwargs):
        if function is None:
            def decorate(func):
                return kernel(func, **kwargs)
            return decorate

        return super().__new__(cls, function, **kwargs)

    def __init__(self,
                 function=None,
                 /, *,
                 num_ctas: None | int | ByTarget[int] = None,
                 occupancy: None | int | ByTarget[int] = None,
                 opt_level: None | int | ByTarget[int] = 3,
                 num_worker_warps: None | int | ByTarget[int] = None):
        if not isinstance(function, FunctionType):
            raise TypeError("`kernel` decorator must be applied to a Python function")

        from cuda.tile._compiler_options import CompilerOptions
        from cuda.tile._annotated_function import get_annotated_function

        ann_func = get_annotated_function(function)
        compiler_options = CompilerOptions(
            num_ctas=num_ctas,
            occupancy=occupancy,
            opt_level=opt_level,
            num_worker_warps=num_worker_warps
        )
        super().__init__(ann_func.constant_parameter_mask, ann_func.int64_index_parameter_mask,
                         ann_func.int64_parameter_mask)
        self._annotated_function = ann_func
        self._compiler_options = compiler_options

    def _compile(self, signature: KernelSignature, context: TileContext):
        from cuda.tile._compile import compile_tile, get_sm_arch, parse_bytecode_version
        from cuda.tile import _backend

        sm_arch = _backend.get_sm_arch_override() or get_sm_arch()
        compile_fn = _backend.get_compile_fn()
        if compile_fn is not None:
            # Custom backend: compile to TileIR bytecode and let the hook turn
            # it into a backend binary instead of a CUDA cubin.
            bc_ver = _backend.get_bytecode_version_override()
            bytecode_version = parse_bytecode_version(bc_ver) if bc_ver else None
            result = compile_tile(self._annotated_function, (signature,),
                                  sm_arch, self._compiler_options, context,
                                  bytecode_version=bytecode_version,
                                  return_bytecode=True, return_cubin=False)
            [kernel_sig] = result.kernel_signatures
            binary = compile_fn(bytes(result.bytecode),
                                symbol=kernel_sig.symbol,
                                sm_arch=sm_arch,
                                signature=kernel_sig)
            return binary, kernel_sig.symbol, None, []

        result = compile_tile(self._annotated_function, (signature,),
                              sm_arch, self._compiler_options, context)
        [kernel_sig] = result.kernel_signatures
        return result.cubin, kernel_sig.symbol, None, []

    @property
    def _pyfunc(self):
        return self._annotated_function.pyfunc

    def replace_hints(self, **hints):
        """Return a new kernel with updated compiler hints.

        Notes::

            Because hints affects compilation, the returned object will have its
            own JIT cache.

        Examples:

        .. testcode::
            :template: setup_only.py

            @ct.kernel(occupancy=2)
            def kernel():
                pass

            # compile
            ct.launch(torch.cuda.current_stream(), (1,), kernel, ())
            # cache hit
            ct.launch(torch.cuda.current_stream(), (1,), kernel, ())

            new_kernel = kernel.replace_hints(occupancy=4)

            # compile with new hints
            ct.launch(torch.cuda.current_stream(), (1,), new_kernel, ())
            # cache hit
            ct.launch(torch.cuda.current_stream(), (1,), new_kernel, ())
        """
        compiler_options = dataclasses.replace(self._compiler_options, **hints)
        return kernel(self._pyfunc, **dataclasses.asdict(compiler_options))

    def __call__(self, *args, **kwargs):
        raise TypeError("Tile kernels cannot be called directly. Use cuda.tile.launch() instead.")


def stub(func=None, /, *, host=False):
    def decorate(func):
        func = function(func, host=host)
        func._cutile_python_stub = True
        return func

    if func is None:
        return decorate
    else:
        return decorate(func)


def is_stub(func) -> bool:
    while True:
        if getattr(func, "_cutile_python_stub", False):
            return True
        func = getattr(func, "__wrapped__", None)
        if func is None:
            return False


def is_function_wrapper(func) -> bool:
    return getattr(func, "_cutile_function_wrapper", False)


def compile_kernel(kernel: "kernel",
                   signature: "KernelSignature",
                   context: TileContext = default_tile_context):
    """Compile a kernel for a single signature into a launchable binary.

    Routes through the registered backend ``compile_fn`` when one is set
    (see :func:`cuda.tile.set_backend`), otherwise produces a CUDA cubin.

    Returns:
        ``(binary, symbol)``: the compiled binary bytes and the kernel symbol
        name.
    """
    binary, symbol, _dyn_smem, _tensor_maps = kernel._compile(signature, context)
    return binary, symbol


def launch(stream, grid, kernel, kernel_args, /):
    """Queue a kernel for execution over a grid.

    When a custom backend launch hook is registered via
    :func:`cuda.tile.set_backend`, execution is delegated to it; otherwise the
    default CUDA driver launch path is used.
    """
    from cuda.tile import _backend
    launch_fn = _backend.get_launch_fn()
    if launch_fn is not None:
        return launch_fn(stream, grid, kernel, kernel_args)
    return _cext_launch(stream, grid, kernel, kernel_args)
