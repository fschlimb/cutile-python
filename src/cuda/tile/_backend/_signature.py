from __future__ import annotations

from typing import Any

import cuda.tile as ct
from cuda.tile.compilation import (
    ArrayConstraint,
    CallingConvention,
    ConstantConstraint,
    KernelSignature,
    ScalarConstraint,
)


_DTYPES = {
    "bool": ct.int8,
    "bool_": ct.int8,
    "bfloat16": ct.bfloat16,
    "float16": ct.float16,
    "float32": ct.float32,
    "float64": ct.float64,
    "int8": ct.int8,
    "int16": ct.int16,
    "int32": ct.int32,
    "int64": ct.int64,
    "uint8": ct.uint8,
    "uint16": ct.uint16,
    "uint32": ct.uint32,
    "uint64": ct.uint64,
}


def _dtype_name(dtype: Any) -> str:
    return str(dtype).removeprefix("torch.")


def array_metadata(value: Any) -> tuple[int, tuple[int, ...], tuple[int, ...], Any]:
    if hasattr(value, "data_ptr") and callable(value.data_ptr):
        pointer = int(value.data_ptr())
        shape = tuple(map(int, value.shape))
        strides = tuple(map(int, value.stride()))
        dtype = value.dtype
    elif hasattr(value, "__array_interface__"):
        interface = value.__array_interface__
        pointer = int(interface["data"][0])
        shape = tuple(map(int, value.shape))
        itemsize = int(value.dtype.itemsize)
        byte_strides = value.strides
        if byte_strides is None:
            strides = []
            stride = 1
            for size in reversed(shape):
                strides.append(stride)
                stride *= size
            strides = tuple(reversed(strides))
        else:
            strides = tuple(int(stride // itemsize) for stride in byte_strides)
        dtype = value.dtype
    else:
        raise TypeError(
            f"expected an array with data_ptr() or __array_interface__, got "
            f"{type(value).__name__}")
    return pointer, shape, strides, dtype


def _array_constraint(value: Any, *, index_dtype, base_addr_divisible_by: int):
    pointer, shape, strides, source_dtype = array_metadata(value)
    try:
        dtype = _DTYPES[_dtype_name(source_dtype)]
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

    align_bits = 16 * 8
    stride_divisor = (align_bits // dtype.bitwidth
                      if align_bits % dtype.bitwidth == 0 else 1)
    return ArrayConstraint(
        dtype=dtype,
        ndim=len(shape),
        index_dtype=index_dtype,
        stride_lower_bound_incl=0,
        alias_groups=(),
        may_alias_internally=False,
        stride_constant=tuple(1 if stride == 1 else None for stride in strides),
        stride_divisible_by=tuple(
            stride_divisor if stride % stride_divisor == 0 else 1
            for stride in strides),
        shape_divisible_by=tuple(16 if size % 16 == 0 else 1 for size in shape),
        base_addr_divisible_by=(base_addr_divisible_by
                    if pointer % base_addr_divisible_by == 0 else 1),
    )


def _scalar_constraint(value: Any, *, int64: bool) -> ScalarConstraint:
    if isinstance(value, bool):
        return ScalarConstraint(ct.int8)
    if isinstance(value, int):
        return ScalarConstraint(ct.int64 if int64 else ct.int32)
    if isinstance(value, float):
        return ScalarConstraint(ct.float32)
    raise TypeError(f"unsupported scalar type: {type(value).__name__}")


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