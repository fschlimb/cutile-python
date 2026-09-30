# SPDX-FileCopyrightText: Copyright (c) <2026> Intel Corporation. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import numpy as np

import cuda.tile as ct
from cuda.tile._ir.typing_support import to_dtype
from cuda.tile.compilation import (
    ArrayConstraint,
    CallingConvention,
    ConstantConstraint,
    KernelSignature,
    ScalarConstraint,
)


# The native launcher (cext/tile_kernel.cpp) derives layout assumptions for at
# most this many dimensions; matching it keeps signatures backend-independent.
_MAX_SPECIALIZED_NDIM = 5


def array_metadata(value: Any) -> tuple[int, tuple[int, ...], tuple[int, ...], Any]:
    """Return ``(pointer, shape, strides, dtype)`` with strides in elements."""
    if hasattr(value, "data_ptr") and callable(value.data_ptr):
        pointer = int(value.data_ptr())
        shape = tuple(map(int, value.shape))
        strides = tuple(map(int, value.stride()))
        return pointer, shape, strides, value.dtype
    if hasattr(value, "__array_interface__"):
        array = np.asarray(value)
        itemsize = array.dtype.itemsize
        if any(stride % itemsize for stride in array.strides):
            raise ValueError("array strides must be multiples of the element size")
        strides = tuple(stride // itemsize for stride in array.strides)
        return array.__array_interface__["data"][0], array.shape, strides, array.dtype
    raise TypeError(
        f"expected an array with data_ptr() or __array_interface__, got "
        f"{type(value).__name__}")


def array_device_type(value: Any) -> str:
    """Device type of an array, e.g. ``"cpu"`` or ``"xpu"``."""
    device = getattr(value, "device", "cpu")
    return getattr(device, "type", device)


def _elements_disjoint(shape, strides) -> bool:
    # Same rule as the native launcher: sorted by stride, every stride must be
    # positive and step over the whole extent of the previous dimension.
    dims = sorted(zip(strides, shape))
    if not dims:
        return True
    return dims[0][0] > 0 and all(
        stride > 0 and stride >= prev_stride * prev_size
        for (prev_stride, prev_size), (stride, _) in zip(dims, dims[1:]))


def _array_constraint(value: Any, *, index_dtype, base_addr_divisible_by: int):
    pointer, shape, strides, source_dtype = array_metadata(value)
    try:
        dtype = to_dtype(source_dtype)
    except KeyError:
        raise TypeError(f"unsupported array dtype {source_dtype}") from None

    if index_dtype is ct.int32:
        i32_max = (1 << 31) - 1
        if any(size > i32_max or stride > i32_max
               for size, stride in zip(shape, strides)):
            raise TypeError(
                "array shape or stride exceeds int32; annotate the parameter "
                "with ct.int64 index dtype")
    if any(stride < 0 for stride in strides):
        raise ValueError("negative array strides are not supported")

    specialize = len(shape) <= _MAX_SPECIALIZED_NDIM
    align_bits = 16 * 8
    stride_divisor = (align_bits // dtype.bitwidth
                      if align_bits % dtype.bitwidth == 0 else 1)
    return ArrayConstraint(
        dtype=dtype,
        ndim=len(shape),
        index_dtype=index_dtype,
        stride_lower_bound_incl=0,
        alias_groups=(),
        may_alias_internally=not _elements_disjoint(shape, strides),
        stride_constant=tuple(
            1 if specialize and stride == 1 else None for stride in strides),
        stride_divisible_by=tuple(
            stride_divisor if specialize and stride % stride_divisor == 0 else 1
            for stride in strides),
        shape_divisible_by=tuple(
            16 if specialize and size % 16 == 0 else 1 for size in shape),
        base_addr_divisible_by=(base_addr_divisible_by
                                if pointer % base_addr_divisible_by == 0 else 1),
    )


def _scalar_constraint(value: Any, *, int64: bool) -> ScalarConstraint:
    if isinstance(value, bool):
        return ScalarConstraint(ct.bool_)
    if isinstance(value, int):
        return ScalarConstraint(ct.int64 if int64 else ct.int32)
    if isinstance(value, float):
        return ScalarConstraint(ct.float32)
    raise TypeError(f"unsupported kernel argument type: {type(value).__name__}")


def build_signature(kernel, args, *, calling_convention=None,
                    base_addr_divisible_by=16, symbol=None) -> KernelSignature:
    annotated = kernel._annotated_function
    constants = annotated.constant_parameter_mask
    int64_indices = annotated.int64_index_parameter_mask
    int64_scalars = annotated.int64_parameter_mask
    if len(args) != len(constants):
        raise TypeError(
            f"kernel expects {len(constants)} arguments, got {len(args)}")

    parameters = []
    for index, value in enumerate(args):
        if constants[index]:
            parameters.append(ConstantConstraint(value))
            continue
        try:
            array_metadata(value)
        except TypeError:
            parameters.append(
                _scalar_constraint(value, int64=int64_scalars[index]))
        else:
            parameters.append(_array_constraint(
                value,
                index_dtype=ct.int64 if int64_indices[index] else ct.int32,
                base_addr_divisible_by=base_addr_divisible_by,
            ))

    convention = calling_convention or CallingConvention.cutile_python_v1()
    signature = KernelSignature(parameters, convention, symbol)
    if symbol is None:
        signature = signature.with_mangled_symbol(annotated.pyfunc.__name__)
    return signature
