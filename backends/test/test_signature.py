# SPDX-FileCopyrightText: Copyright (c) <2026> Intel Corporation. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""Keep the Python signature builder of custom backends in line with the CUDA launcher."""

from typing import Annotated

import numpy as np
import pytest

import cuda.tile as ct
from cuda.tile._backends._signature import build_signature, constant_parameters
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
        build_signature(_static_shape_kernel, (x,),
                        calling_convention=CallingConvention.cutile_python_v1())
    signature = build_signature(_static_shape_kernel, (x,))
    assert signature.calling_convention == CallingConvention.cutile_python_v2()
    assert signature.parameters[0].shape_constant == (None, 16)


def test_size_one_dims_become_static_shapes_as_for_cuda_launches():
    x = np.zeros((32, 16), dtype=np.float32)
    default = build_signature(_kernel, (x[:1], x, x, True, 1.5, 3, 1, 16))
    assert default.calling_convention == CallingConvention.cutile_python_v2()
    assert default.parameters[0].shape_constant == (1, None)
    assert default.parameters[1].shape_constant == (None, None)

    unit_free = build_signature(_kernel, (x, x, x, True, 1.5, 3, 1, 16))
    assert unit_free.calling_convention == CallingConvention.cutile_python_v1()

    v1 = build_signature(_kernel, (x[:1], x, x, True, 1.5, 3, 1, 16),
                         calling_convention=CallingConvention.cutile_python_v1())
    assert v1.parameters[0].shape_constant == (None, None)


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
                 (base.as_strided((16, 16), (0, 1)), base.view(-1)[1:]),
                 (base[:1], base[:, :1])]:
        args = (x, y, True, 1.5, 3, 1 << 40, y, x, 16)
        # cutile_python_v2 enables the static size-one shapes of CUDA launches.
        for convention in (CallingConvention.cutile_python_v1(),
                           CallingConvention.cutile_python_v2()):
            assert build_signature(_cuda_kernel, args, calling_convention=convention) \
                == KernelSignature.from_kernel_args(_cuda_kernel, args, convention)
