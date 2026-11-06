# SPDX-FileCopyrightText: Copyright (c) <2026> Intel Corporation. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""Run every CPU sample with its correctness check, like samples/test_samples.py for CUDA."""

import os
from pathlib import Path
import subprocess
import sys

import pytest

_SAMPLES = sorted((Path(__file__).resolve().parents[3] / "samples" / "cpu").glob("*-cpu.py"))


@pytest.fixture(scope="module", autouse=True)
def _native_cpu():
    pytest.importorskip("cuda.tile._backends.cpu._cpu")
    compiler = pytest.importorskip("cuda.tile._backends._compiler")
    if compiler.backend != "cpu":
        pytest.skip("requires the native CPU compiler")


@pytest.mark.parametrize("sample", _SAMPLES, ids=lambda path: path.name)
def test_cpu_sample_passes_correctness_check(sample):
    # As in conftest.py, compile without pytest.ini's per-compile timeout, which is
    # meant for tileiras: native CPU compiles of large tiles take minutes.
    environment = dict(os.environ)
    environment.pop("CUDA_TILE_COMPILER_TIMEOUT_SEC", None)
    result = subprocess.run([sys.executable, "-u", str(sample), "--correctness-check"],
                            capture_output=True, text=True, timeout=1800, env=environment)
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "Correctness check passed" in result.stdout, output
    assert "Correctness check disabled" not in result.stdout, output
