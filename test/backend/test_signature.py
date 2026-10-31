# SPDX-FileCopyrightText: Copyright (c) <2026> Intel Corporation.
# SPDX-License-Identifier: Apache-2.0
"""Keep the Python signature builder of custom backends in line with the CUDA launcher."""

from typing import Annotated

import numpy as np
import pytest

import cuda.tile as ct
from cuda.tile._backend._signature import build_signature, constant_parameters
from cuda.tile.compilation import (
    ArrayConstraint, CallingConvention, ConstantConstraint, KernelSignature, ScalarConstraint)

_StaticStrides = Annotated[ct.Array, ct.ArrayAnnotation(static_stride_dims=(0,))]
_StaticShape = Annotated[ct.Array, ct.ArrayAnnotation(static_shape_dims=(-1,))]


@ct.kernel
def _kernel(x, y: ct.IndexedWithInt64, z: _StaticStrides, flag: bool, scale: float,
            count: int, wide: ct.ScalarInt64, tile: ct.Constant[int]):
    pass


@ct.kernel
def _static_shape_kernel(x: _StaticShape):
    pass


@ct.kernel
def _tuple_kernel(pair: tuple[int, int]):
    pass


def test_constant_parameters():
    assert constant_parameters(_kernel) == (False,) * 7 + (True,)
    assert constant_parameters(_tuple_kernel) == (False,)


def test_tuple_parameters_are_cuda_only():
    with pytest.raises(TypeError, match="tuple parameters"):
        build_signature(_tuple_kernel, ((1, 2),))


def test_build_signature_follows_annotations():
    x = np.zeros((32, 16), dtype=np.float32)
    signature = build_signature(_kernel, (x, x, x, True, 1.5, 3, 1 << 40, 16))
    plain, int64_indexed, static_strides, *scalars, tile = signature.parameters

    assert all(isinstance(p, ArrayConstraint) for p in (plain, int64_indexed, static_strides))
    assert (plain.index_dtype, int64_indexed.index_dtype) == (ct.int32, ct.int64)
    assert plain.stride_constant == (None, 1)
    # Annotated static strides replace the inference of unit strides.
    assert static_strides.stride_constant == (16, None)
    assert plain.shape_constant == (None, None)
    assert scalars == [ScalarConstraint(ct.bool_), ScalarConstraint(ct.float32),
                       ScalarConstraint(ct.int32), ScalarConstraint(ct.int64)]
    assert tile == ConstantConstraint(16)


def test_static_shape_dims_require_cutile_python_v2():
    x = np.zeros((32, 16), dtype=np.float32)
    with pytest.raises(ValueError):
        build_signature(_static_shape_kernel, (x,))
    signature = build_signature(_static_shape_kernel, (x,),
                                calling_convention=CallingConvention.cutile_python_v2())
    assert signature.parameters[0].shape_constant == (None, 16)


@ct.kernel
def _cuda_kernel(x, y, flag: bool, scale: float, count: int, wide: ct.ScalarInt64,
                 big: ct.IndexedWithInt64, z: _StaticStrides, tile: ct.Constant[int]):
    pass


def test_build_signature_matches_the_cuda_launcher():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    base = torch.zeros((64, 64), device="cuda")
    for x, y in [(base, base[1:, ::2]), (base[:, :17], base.t()),
                 (base.as_strided((16, 16), (0, 1)), base.view(-1)[1:])]:
        args = (x, y, True, 1.5, 3, 1 << 40, y, x, 16)
        assert build_signature(_cuda_kernel, args) == KernelSignature.from_kernel_args(
            _cuda_kernel, args, CallingConvention.cutile_python_v1())
