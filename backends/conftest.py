# SPDX-FileCopyrightText: Copyright (c) <2026> Intel Corporation. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from cuda.tile._cext import default_tile_context

# The submodule's own tests are not cuTile tests.
collect_ignore = ["tileir-to-mlir"]


@pytest.fixture(autouse=True)
def _compile_in_process(monkeypatch):
    """Exercise the default in-process compilation; tests opt into a timeout worker."""
    monkeypatch.setattr(default_tile_context.config, "compiler_timeout_sec", None)
