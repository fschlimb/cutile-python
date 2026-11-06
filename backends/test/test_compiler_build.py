# SPDX-FileCopyrightText: Copyright (c) <2026> Intel Corporation. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import hashlib
from pathlib import Path
import shutil
import subprocess
import tarfile

import pytest


@pytest.fixture
def compiler_sources(tmp_path):
    cmake = shutil.which("cmake")
    if cmake is None:
        pytest.skip("compiler fetch tests require CMake")
    source = tmp_path / "project"
    source.mkdir()
    binary = tmp_path / "build"
    fetch = Path(__file__).resolve().parents[1] / "cmake" / "FetchTileIRToMLIR.cmake"
    (source / "CMakeLists.txt").write_text(f"""
cmake_minimum_required(VERSION 3.24)
project(CompilerFetchProbe NONE)
include("{fetch}")
fetch_tileir_to_mlir()
file(WRITE "${{CMAKE_BINARY_DIR}}/sources.txt"
    "${{CUTILE_TILEIR_TO_MLIR_DIR}}\n${{CUTILE_CUDA_TILE_SOURCE_DIR}}\n")
""")
    args = []
    archives = []
    for name, prefix, license_name in [
        ("tileir_to_mlir", "TILEIR_TO_MLIR", "LICENSE"),
        ("cuda_tile", "CUDA_TILE", "LICENSE.txt"),
    ]:
        contents = tmp_path / name
        contents.mkdir()
        (contents / "CMakeLists.txt").write_text(
            'message(FATAL_ERROR "dependency must not be configured here")\n')
        (contents / license_name).write_text(name)
        (contents / "llvm-revision.txt").write_text("llvm-pin\n")
        repository = tmp_path / f"{name}-repository"
        (repository / "archive").mkdir(parents=True)
        archive = repository / "archive" / "test-pin.tar.gz"
        with tarfile.open(archive, "w:gz") as bundle:
            bundle.add(contents, arcname=name)
        archives.append(archive)
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        args.extend([
            f"-D{prefix}_REPOSITORY={repository.as_uri()}",
            f"-D{prefix}_REVISION=test-pin",
            f"-D{prefix}_REVISION_SHA256={digest}",
        ])

    def run(extra_args=(), check=True):
        return subprocess.run([cmake, "-S", str(source), "-B", str(binary),
                               *args, *extra_args], check=check,
                              capture_output=True, text=True, timeout=60)

    return run, binary, archives


def test_compiler_sources_are_fetched_and_reused(compiler_sources):
    run, binary, archives = compiler_sources
    run()
    tileir, cuda_tile = map(Path, (binary / "sources.txt").read_text().splitlines())
    assert tileir == binary / "_deps" / "tileir_to_mlir-src"
    assert cuda_tile == binary / "_deps" / "cuda_tile-src"
    assert (tileir / "llvm-revision.txt").read_text() == "llvm-pin\n"
    assert (tileir / "LICENSE").read_text() == "tileir_to_mlir"
    assert (cuda_tile / "LICENSE.txt").read_text() == "cuda_tile"
    for archive in archives:
        archive.unlink()
    run()


@pytest.mark.parametrize("prefix", ["TILEIR_TO_MLIR", "CUDA_TILE"])
def test_compiler_source_hash_mismatch_is_rejected(compiler_sources, prefix):
    run, _, _ = compiler_sources
    result = run([f"-D{prefix}_REVISION_SHA256={'0' * 64}"], check=False)
    assert result.returncode != 0
    assert "SHA256 hash" in result.stdout + result.stderr


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
    run("other")
    run("other")
    run("cpu")
    assert log.read_text().splitlines() == ["cpu:mlir;lld", "other:mlir;lld", "cpu:mlir;lld"]
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


def test_cpu_toolchain_settings(tmp_path):
    cmake = shutil.which("cmake")
    if cmake is None:
        pytest.skip("compiler build tests require CMake")
    toolchain = Path(__file__).resolve().parents[1] / "cpu" / "toolchain.cmake"
    script = tmp_path / "toolchain.cmake"
    script.write_text(f'include("{toolchain}")\n' + """
if(NOT _cutile_llvm_targets STREQUAL "host" OR
   NOT _cutile_llvm_projects STREQUAL "mlir|lld" OR
   NOT _cutile_llvm_experimental_targets STREQUAL "" OR
   NOT _cutile_disable_ocloc OR
   NOT _cutile_backend_cmake_args STREQUAL "")
    message(FATAL_ERROR "CPU toolchain settings changed")
endif()
""")
    subprocess.run([cmake, "-P", str(script)], check=True,
                   capture_output=True, text=True, timeout=60)
