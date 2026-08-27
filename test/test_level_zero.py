# SPDX-FileCopyrightText: Copyright (c) <2026> Intel Corporation. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from cuda.tile import _level_zero


def test_kernel_argument_expansion():
    arguments = [(1234, (2, 3), (3, 1)), (7).to_bytes(4, "little")]
    assert _level_zero._kernel_parameter_count(arguments) == 8


@pytest.mark.parametrize(
    "arguments, match",
    [
        ([(1234, (2,), (1, 1))], "equal length"),
        ([b"abc"], "1, 2, 4, or 8"),
        ([(1234, (-1,), (1,))], "non-negative"),
    ],
)
def test_kernel_argument_validation(arguments, match):
    with pytest.raises(ValueError, match=match):
        _level_zero._kernel_parameter_count(arguments)


def test_launch_validation_does_not_initialize_runtime():
    with pytest.raises(ValueError, match="must not be empty"):
        _level_zero.launch_level_zero_module_kernel(
            b"", "kernel", [], (1, 1, 1), (1, 1, 1)
        )

    with pytest.raises(TypeError):
        _level_zero.launch_level_zero_module_kernel(
            b"", "kernel", [], (1, 1), (1, 1, 1)
        )