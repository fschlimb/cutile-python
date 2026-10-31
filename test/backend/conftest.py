# SPDX-FileCopyrightText: Copyright (c) <2026> Intel Corporation.
# SPDX-License-Identifier: Apache-2.0

import pytest

from cuda.tile._cext import default_tile_context


@pytest.fixture(autouse=True)
def _compile_in_process(monkeypatch):
    """Exercise the default in-process compilation; tests opt into a timeout worker."""
    monkeypatch.setattr(default_tile_context.config, "compiler_timeout_sec", None)


@pytest.fixture(autouse=True)
def reset_torch_device():
    """Override the CUDA device reset of test/conftest.py; custom backends need no CUDA."""
