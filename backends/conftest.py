# SPDX-FileCopyrightText: Copyright (c) <2026> Intel Corporation. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from cuda.tile._backends import _custom
from cuda.tile._backends._signature import build_signature
from cuda.tile._cext import default_tile_context


@pytest.fixture(autouse=True)
def _compile_in_process(monkeypatch):
    """Exercise the default in-process compilation; tests opt into a timeout worker."""
    monkeypatch.setattr(default_tile_context.config, "compiler_timeout_sec", None)


@pytest.fixture
def native():
    return pytest.importorskip("cuda.tile._backends._compiler")


@pytest.fixture
def bytecode():
    def compile(kernel, arguments, backend):
        return _custom.compile_bytecode(kernel, backend, build_signature(kernel, arguments))

    return compile
