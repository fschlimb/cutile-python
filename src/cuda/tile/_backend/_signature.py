# SPDX-FileCopyrightText: Copyright (c) <2026> Intel Corporation. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, NamedTuple

import cuda.tile as ct
from cuda.tile._annotated_function import LeafAnnotationNode
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

# Struct format of each runtime scalar dtype, as passed to a backend kernel.
_SCALAR_FORMATS = {
    ct.bool_: "?", ct.int8: "b", ct.uint8: "B", ct.int16: "h", ct.uint16: "H",
    ct.int32: "i", ct.uint32: "I", ct.int64: "q", ct.uint64: "Q",
    ct.float32: "f", ct.float64: "d",
}


class ArrayArgument(NamedTuple):
    """An array kernel argument as passed to a backend's ``launch``."""

    pointer: int
    shape: tuple[int, ...]
    strides: tuple[int, ...]  # in elements
    dtype: Any
    device: str


def array_metadata(value: Any) -> ArrayArgument:
    """Describe an array argument; raise ``TypeError`` for anything else."""
    data_ptr = getattr(value, "data_ptr", None)
    if callable(data_ptr):
        device = getattr(value, "device", "cpu")
        return ArrayArgument(int(data_ptr()), tuple(map(int, value.shape)),
                             tuple(map(int, value.stride())), value.dtype,
                             getattr(device, "type", device))
    if not hasattr(value, "__array_interface__"):
        raise TypeError(
            f"expected an array with data_ptr() or __array_interface__, got "
            f"{type(value).__name__}")
    import numpy as np

    value = np.asarray(value)
    itemsize = value.dtype.itemsize
    if any(stride % itemsize for stride in value.strides):
        raise ValueError("array strides must be multiples of the element size")
    strides = tuple(stride // itemsize for stride in value.strides)
    return ArrayArgument(value.ctypes.data, value.shape, strides, value.dtype, "cpu")


def scalar_format(backend: str, dtype) -> str:
    """Struct format of a runtime scalar of ``dtype``."""
    try:
        return _SCALAR_FORMATS[dtype]
    except KeyError:
        raise TypeError(f"{backend} backend: unsupported scalar dtype {dtype}") from None


def runtime_parameters(backend: str, signature: KernelSignature) -> tuple:
    """The signature's parameters that receive runtime arguments."""
    parameters = tuple(parameter for parameter in signature.parameters
                       if not isinstance(parameter, ConstantConstraint))
    for parameter in parameters:
        if isinstance(parameter, ScalarConstraint):
            scalar_format(backend, parameter.dtype)
        elif not isinstance(parameter, ArrayConstraint):
            raise TypeError(f"{backend} backend: unsupported parameter constraint "
                            f"{type(parameter).__name__}")
    return parameters


def check_argument(backend: str, device: str, parameter, argument) -> None:
    """Check that a runtime argument matches its signature parameter."""
    if isinstance(parameter, ArrayConstraint):
        if (not isinstance(argument, ArrayArgument)
                or len(argument.shape) != parameter.ndim):
            raise TypeError(f"{backend} backend: expected a {parameter.ndim}-D array, "
                            f"got {type(argument).__name__}")
        if argument.device != device:
            raise ValueError(f"{backend} backend: array is on device "
                             f"'{argument.device}', expected '{device}'")
    elif isinstance(argument, ArrayArgument):
        raise TypeError(f"{backend} backend: expected a scalar, got an array")


def _elements_disjoint(shape, strides) -> bool:
    # Same rule as the native launcher: sorted by stride, every stride must be
    # positive and step over the whole extent of the previous dimension.
    dims = sorted(zip(strides, shape))
    if not dims:
        return True
    return dims[0][0] > 0 and all(
        stride > 0 and stride >= prev_stride * prev_size
        for (prev_stride, prev_size), (stride, _) in zip(dims, dims[1:]))


def _normalized_dims(dims, ndim: int, name: str) -> tuple[int, ...]:
    normalized = []
    for dim in dims:
        if not -ndim <= dim < ndim:
            raise ValueError(f"{name} axis {dim} is out of range for a {ndim}-D array")
        dim %= ndim
        if dim in normalized:
            raise ValueError(f"{name} contains duplicate axis {dim}")
        normalized.append(dim)
    return tuple(normalized)


def _array_constraint(array: ArrayArgument, annotation, *, base_addr_divisible_by: int):
    try:
        dtype = to_dtype(array.dtype)
    except KeyError:
        raise TypeError(f"unsupported array dtype {array.dtype}") from None

    shape, strides = array.shape, array.strides
    ndim = len(shape)
    index_dtype = ct.int32 if annotation is None else annotation.index_dtype
    static_shape_dims = () if annotation is None else _normalized_dims(
        annotation.static_shape_dims, ndim, "static_shape_dims")
    static_stride_dims = () if annotation is None else _normalized_dims(
        annotation.static_stride_dims, ndim, "static_stride_dims")
    if index_dtype is ct.int32:
        i32_max = (1 << 31) - 1
        if any(size > i32_max or stride > i32_max
               for size, stride in zip(shape, strides)):
            raise TypeError(
                "array shape or stride exceeds int32; annotate the parameter "
                "with ct.int64 index dtype")
    if any(stride < 0 for stride in strides):
        raise ValueError("negative array strides are not supported")

    specialize = ndim <= _MAX_SPECIALIZED_NDIM
    align_bits = 16 * 8
    stride_divisor = (align_bits // dtype.bitwidth
                      if align_bits % dtype.bitwidth == 0 else 1)
    # As in the native launcher, annotated static strides replace the inference
    # of unit strides for all dimensions.
    if static_stride_dims:
        stride_constant = tuple(strides[i] if i in static_stride_dims else None
                                for i in range(ndim))
    else:
        stride_constant = tuple(
            1 if specialize and stride == 1 else None for stride in strides)
    return ArrayConstraint(
        dtype=dtype,
        ndim=ndim,
        index_dtype=index_dtype,
        stride_lower_bound_incl=0,
        alias_groups=(),
        may_alias_internally=not _elements_disjoint(shape, strides),
        stride_constant=stride_constant,
        shape_constant=tuple(shape[i] if i in static_shape_dims else None
                             for i in range(ndim)),
        stride_divisible_by=tuple(
            stride_divisor if specialize and stride % stride_divisor == 0 else 1
            for stride in strides),
        shape_divisible_by=tuple(
            16 if specialize and size % 16 == 0 else 1 for size in shape),
        base_addr_divisible_by=(base_addr_divisible_by
                                if array.pointer % base_addr_divisible_by == 0 else 1),
    )


def _scalar_constraint(value: Any, annotation) -> ScalarConstraint:
    if isinstance(value, bool):
        return ScalarConstraint(ct.bool_)
    if isinstance(value, int):
        return ScalarConstraint(ct.int32 if annotation is None else annotation.dtype)
    if isinstance(value, float):
        return ScalarConstraint(ct.float32)
    if isinstance(value, (list, tuple)):
        raise TypeError("array list arguments are only supported on CUDA")
    raise TypeError(f"unsupported kernel argument type: {type(value).__name__}")


def _leaf_annotations(kernel) -> tuple[LeafAnnotationNode, ...]:
    annotations = kernel._annotated_function.parameter_annotations
    for annotation in annotations:
        if not isinstance(annotation, LeafAnnotationNode):
            raise TypeError("tuple parameters are only supported on CUDA")
        if annotation.list is not None:
            raise TypeError("array list parameters are only supported on CUDA")
    return annotations


def constant_parameters(kernel) -> tuple[bool, ...]:
    """Whether each parameter of ``kernel`` is a ``ct.Constant`` scalar or array."""
    return tuple(isinstance(annotation, LeafAnnotationNode) and annotation.constant
                 for annotation in kernel._annotated_function.parameter_annotations)


def build_signature(kernel, args, *, calling_convention=None,
                    base_addr_divisible_by=16, symbol=None) -> KernelSignature:
    """Python counterpart of ``KernelSignature.from_kernel_args`` for any backend."""
    annotations = _leaf_annotations(kernel)
    if len(args) != len(annotations):
        raise TypeError(
            f"kernel expects {len(annotations)} arguments, got {len(args)}")

    parameters = []
    for annotation, value in zip(annotations, args):
        if annotation.constant:
            parameters.append(ConstantConstraint(value))
            continue
        try:
            array = array_metadata(value)
        except TypeError:
            parameters.append(_scalar_constraint(value, annotation.scalar))
        else:
            parameters.append(_array_constraint(
                array, annotation.array,
                base_addr_divisible_by=base_addr_divisible_by,
            ))

    convention = calling_convention or CallingConvention.cutile_python_v1()
    signature = KernelSignature(parameters, convention, symbol)
    if symbol is None:
        signature = signature.with_mangled_symbol(
            kernel._annotated_function.pyfunc.__name__)
    return signature
