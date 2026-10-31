# SPDX-FileCopyrightText: Copyright (c) <2026> Intel Corporation. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import shutil
import subprocess

import pytest


@pytest.fixture
def llvm_build(tmp_path):
    cmake, ninja = shutil.which("cmake"), shutil.which("ninja")
    if cmake is None or ninja is None:
        pytest.skip("compiler build tests require CMake and Ninja")
    source = tmp_path / "source with spaces"
    (source / "llvm").mkdir(parents=True)
    cmakelists = source / "llvm" / "CMakeLists.txt"
    cmakelists.write_text("""
cmake_minimum_required(VERSION 3.20)
project(CompilerBuildProbe NONE)
if(TEST_VARIANT STREQUAL "fail-configure")
    message(FATAL_ERROR "configuration failed")
endif()
file(APPEND "${CMAKE_BINARY_DIR}/configured.txt" "${TEST_VARIANT}:${TEST_LIST}\\n")
if(TEST_VARIANT STREQUAL "fail-build")
    add_custom_target(probe COMMAND "${CMAKE_COMMAND}" -E false)
else()
    add_custom_target(probe COMMAND "${CMAKE_COMMAND}" -E echo "Built ${TEST_VARIANT}")
endif()
""")
    binary = tmp_path / "shared build"
    template = (Path(__file__).resolve().parents[2] / "backends" / "cmake"
                / "BuildLLVM.cmake.in").read_text()

    def run(variant, check=True):
        script = tmp_path / f"build-{variant}.cmake"
        contents = template
        for key, value in {
            "CUTILE_LLVM_SOURCE_DIR": str(source),
            "CUTILE_LLVM_BINARY_DIR": str(binary),
            "_cutile_llvm_cmake_args": f"-DTEST_VARIANT={variant};-DTEST_LIST=mlir|lld",
            "_cutile_llvm_build_targets": "probe",
        }.items():
            contents = contents.replace(f"@{key}@", value)
        script.write_text(contents)
        return subprocess.run([cmake, "-P", str(script)], check=check,
                              capture_output=True, text=True, timeout=60)

    return run, binary, cmakelists


def test_shared_build_only_configures_when_needed(llvm_build):
    run, binary, cmakelists = llvm_build
    run("cpu")
    run("cpu")
    log = binary / "configured.txt"
    assert log.read_text().splitlines() == ["cpu:mlir;lld"]
    run("xpu")
    run("xpu")
    run("cpu")
    assert log.read_text().splitlines() == ["cpu:mlir;lld", "xpu:mlir;lld", "cpu:mlir;lld"]
    # Manual cache edits must not defeat variant/configuration restoration.
    cache = binary / "CMakeCache.txt"
    cache.write_text(cache.read_text().replace(
        "TEST_VARIANT:UNINITIALIZED=cpu", "TEST_VARIANT:UNINITIALIZED=manual"))
    run("cpu")
    assert log.read_text().splitlines()[-1] == "cpu:mlir;lld"
    assert len(log.read_text().splitlines()) == 4
    # Ninja must still regenerate when a project's CMake sources change.
    cmakelists.write_text(cmakelists.read_text() + '\nmessage(STATUS "source changed")\n')
    run("cpu")
    assert len(log.read_text().splitlines()) == 5
    run("cpu")
    assert len(log.read_text().splitlines()) == 5


@pytest.mark.parametrize("variant", ["fail-configure", "fail-build"])
def test_failed_configuration_or_build_is_not_marked_successful(llvm_build, variant):
    run, binary, _ = llvm_build
    run("cpu")
    result = run(variant, check=False)
    assert result.returncode != 0
    assert not (binary / "cutile-configure.stamp").exists()
    run("cpu")
    assert (binary / "cutile-configure.stamp").exists()


def test_missing_cache_triggers_reconfiguration(llvm_build):
    run, binary, _ = llvm_build
    run("cpu")
    (binary / "CMakeCache.txt").unlink()
    run("cpu")
    assert (binary / "configured.txt").read_text().splitlines() == [
        "cpu:mlir;lld", "cpu:mlir;lld"]
