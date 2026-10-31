# SPDX-FileCopyrightText: Copyright (c) <2026> Intel Corporation. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

set(CUTILE_TOOLCHAIN_DIR "" CACHE PATH
    "Existing native compiler installation for CUTILE_BUILD_VARIANT; skips building LLVM")

execute_process(
    COMMAND "${Python_EXECUTABLE}" -c
            "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))"
    OUTPUT_VARIABLE _compiler_suffix OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY)
set(_compiler_name "_compiler${_compiler_suffix}")

set(_tileir_to_mlir_dir "${CUTILE_TILEIR_TO_MLIR_DIR}")
if(NOT EXISTS "${_tileir_to_mlir_dir}/CMakeLists.txt")
    message(FATAL_ERROR
        "TileIRToMLIR sources are missing. Initialize the tileir-to-mlir submodule.")
endif()

if(CUTILE_TOOLCHAIN_DIR)
    set(CUTILE_COMPILER_PATH "${CUTILE_TOOLCHAIN_DIR}/lib/${_compiler_name}")
    set(CUTILE_LLVM_LICENSE_PATH "${CUTILE_TOOLCHAIN_DIR}/share/cutile/llvm/LICENSE.TXT")
    foreach(_path IN ITEMS CUTILE_COMPILER_PATH CUTILE_LLVM_LICENSE_PATH)
        if(NOT EXISTS "${${_path}}")
            message(FATAL_ERROR "CUTILE_TOOLCHAIN_DIR must contain ${${_path}}")
        endif()
    endforeach()
    return()
endif()

set(_tileir_required_llvm_revision_file
    "${_tileir_to_mlir_dir}/llvm-revision.txt")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
             "${_tileir_required_llvm_revision_file}")
file(STRINGS "${_tileir_required_llvm_revision_file}"
     _tileir_required_llvm_revision LIMIT_COUNT 1)
set(_cutile_llvm_override "")
if(DEFINED ENV{USE_LLVM_REVISION} AND NOT "$ENV{USE_LLVM_REVISION}" STREQUAL "")
    set(_cutile_llvm_override "$ENV{USE_LLVM_REVISION}")
elseif(DEFINED USE_LLVM_REVISION AND NOT USE_LLVM_REVISION STREQUAL "")
    set(_cutile_llvm_override "${USE_LLVM_REVISION}")
endif()

set(CUTILE_LLVM_SOURCE_DIR "${CMAKE_SOURCE_DIR}/.llvm-project" CACHE PATH
    "Persistent LLVM source checkout")
set(CUTILE_LLVM_LICENSE_PATH "${CUTILE_LLVM_SOURCE_DIR}/llvm/LICENSE.TXT")
set(CUTILE_LLVM_BINARY_DIR "${CMAKE_SOURCE_DIR}/build/llvm" CACHE PATH
    "LLVM/MLIR build directory")
set(CUTILE_LLVM_BUILD_TYPE "Release" CACHE STRING
    "Build type for LLVM, MLIR, and TileIRToMLIR")
# Assertions abort the process that compiles in-process; enable them only to debug.
option(CUTILE_LLVM_ENABLE_ASSERTIONS "Enable LLVM/MLIR assertions" OFF)

find_package(Git REQUIRED)
find_program(CUTILE_CLANG_EXECUTABLE NAMES clang REQUIRED)
find_program(CUTILE_CLANGXX_EXECUTABLE NAMES clang++ REQUIRED)
find_program(CUTILE_LLD_EXECUTABLE NAMES ld.lld REQUIRED)
if(CUTILE_BUILD_VARIANT STREQUAL "xpu")
    find_path(OCLOC_INCLUDE_DIR NAMES ocloc_api.h
        HINTS ${OCLOC_PACKAGE_DIR} ENV CPATH ENV CPLUS_INCLUDE_PATH
        PATH_SUFFIXES include REQUIRED)
endif()

include(ExternalProject)
set(_required_llvm_revisions_file
    "${CMAKE_BINARY_DIR}/tileir-toolchain/required-llvm-revisions.txt")
file(CONFIGURE OUTPUT "${_required_llvm_revisions_file}"
    CONTENT "tileir=${_tileir_required_llvm_revision}\noverride=${_cutile_llvm_override}\n" @ONLY)

set(CUTILE_COMPILER_PATH "${CUTILE_LLVM_BINARY_DIR}/lib/${_compiler_name}")

set(_cutile_checkout_script "${CMAKE_CURRENT_LIST_DIR}/CheckoutLLVM.cmake")
set(_cutile_checkout_command
    "${CMAKE_COMMAND}"
    "-DGIT_EXECUTABLE=${GIT_EXECUTABLE}"
    "-DSOURCE_DIR=${CUTILE_LLVM_SOURCE_DIR}"
    "-DTILEIR_REQUIRED_LLVM_REVISION=${_tileir_required_llvm_revision}"
    "-DREVISION=${_cutile_llvm_override}"
    -P "${_cutile_checkout_script}")

if(CUTILE_BUILD_VARIANT STREQUAL "cpu")
    set(_cutile_llvm_targets host)
    set(_cutile_llvm_projects "mlir|lld")
    set(_cutile_llvm_experimental_targets "")
    set(_cutile_disable_ocloc ON)
    set(_cutile_llvm_build_targets _compiler)
    set(_cutile_llvm_byproducts "${CUTILE_COMPILER_PATH}")
else()
    set(_cutile_llvm_targets host)
    set(_cutile_llvm_projects mlir)
    set(_cutile_llvm_experimental_targets SPIRV)
    set(_cutile_disable_ocloc OFF)
    set(_cutile_llvm_build_targets _compiler)
    set(_cutile_llvm_byproducts "${CUTILE_COMPILER_PATH}")
endif()

set(_cutile_llvm_cmake_args
    "-DCMAKE_BUILD_TYPE:STRING=${CUTILE_LLVM_BUILD_TYPE}"
    "-DCMAKE_C_COMPILER:FILEPATH=${CUTILE_CLANG_EXECUTABLE}"
    "-DCMAKE_CXX_COMPILER:FILEPATH=${CUTILE_CLANGXX_EXECUTABLE}"
    "-DCMAKE_LINKER:FILEPATH=${CUTILE_LLD_EXECUTABLE}"
    "-DLLVM_ENABLE_PROJECTS:STRING=${_cutile_llvm_projects}"
    "-DLLVM_EXTERNAL_PROJECTS:STRING=TileIRToMLIR|cuTileCompiler"
    "-DLLVM_EXTERNAL_TILEIRTOMLIR_SOURCE_DIR:PATH=${_tileir_to_mlir_dir}"
    # Explicit, so that a shared LLVM tree never keeps another checkout's path.
    "-DTILEIR_TO_MLIR_CUDA_TILE_SOURCE_DIR:PATH=${_tileir_to_mlir_dir}/third_party/cuda-tile"
    "-DLLVM_EXTERNAL_CUTILECOMPILER_SOURCE_DIR:PATH=${CMAKE_CURRENT_SOURCE_DIR}/compiler"
    "-DCUTILE_BUILD_VARIANT:STRING=${CUTILE_BUILD_VARIANT}"
    "-DUSE_LLVM_REVISION:STRING=${_cutile_llvm_override}"
    -DLLVM_USE_LINKER:STRING=lld
    "-DLLVM_TARGETS_TO_BUILD:STRING=${_cutile_llvm_targets}"
    "-DLLVM_EXPERIMENTAL_TARGETS_TO_BUILD:STRING=${_cutile_llvm_experimental_targets}"
    "-DLLVM_ENABLE_ASSERTIONS:BOOL=${CUTILE_LLVM_ENABLE_ASSERTIONS}"
    -DLLVM_INSTALL_UTILS:BOOL=ON
    -DLLVM_INCLUDE_TESTS:BOOL=OFF
    -DLLVM_INCLUDE_BENCHMARKS:BOOL=OFF
    -DLLVM_INCLUDE_EXAMPLES:BOOL=OFF
    -DLLVM_BUILD_LLVM_DYLIB:BOOL=OFF
    -DLLVM_LINK_LLVM_DYLIB:BOOL=OFF
    # Optional host libraries add runtime dependencies the compiler never uses.
    -DLLVM_ENABLE_ZLIB:STRING=OFF
    -DLLVM_ENABLE_ZSTD:STRING=OFF
    -DLLVM_ENABLE_LZMA:STRING=OFF
    -DLLVM_ENABLE_LIBXML2:STRING=OFF
    "-DCMAKE_DISABLE_FIND_PACKAGE_ocloc:BOOL=${_cutile_disable_ocloc}"
    # Clear results that the shared LLVM cache would otherwise retain from
    # another variant.
    -UOCLOC_FOUND
    -UCMAKE_PROJECT_INCLUDE
    -DMLIR_INCLUDE_TESTS:BOOL=OFF
    -DMLIR_ENABLE_EXECUTION_ENGINE:BOOL=ON
    -DMLIR_ENABLE_LEVELZERO_RUNNER:BOOL=OFF
    -DMLIR_ENABLE_BINDINGS_PYTHON:BOOL=OFF
    "-DPython3_EXECUTABLE:FILEPATH=${Python_EXECUTABLE}"
    "-DPython_EXECUTABLE:FILEPATH=${Python_EXECUTABLE}"
    "-Dnanobind_DIR:PATH=${nanobind_DIR}")
if(CUTILE_BUILD_VARIANT STREQUAL "xpu")
    list(APPEND _cutile_llvm_cmake_args
        "-DOCLOC_INCLUDE_DIR:PATH=${OCLOC_INCLUDE_DIR}"
        "-DOCLOC_PACKAGE_DIR:PATH=${OCLOC_PACKAGE_DIR}")
endif()

set(_cutile_llvm_build_script "${CMAKE_BINARY_DIR}/tileir-toolchain/BuildLLVM.cmake")
configure_file("${CMAKE_CURRENT_LIST_DIR}/BuildLLVM.cmake.in"
    "${_cutile_llvm_build_script}" @ONLY)

ExternalProject_Add(cutile_llvm
    PREFIX "${CMAKE_BINARY_DIR}/tileir-toolchain/llvm"
    SOURCE_DIR "${CUTILE_LLVM_SOURCE_DIR}"
    BINARY_DIR "${CUTILE_LLVM_BINARY_DIR}"
    DOWNLOAD_COMMAND ${_cutile_checkout_command}
    UPDATE_COMMAND ""
    PATCH_COMMAND ""
    CONFIGURE_COMMAND ""
    BUILD_COMMAND "${CMAKE_COMMAND}" -P "${_cutile_llvm_build_script}"
    INSTALL_COMMAND ""
    BUILD_BYPRODUCTS ${_cutile_llvm_byproducts}
    BUILD_ALWAYS ON
    USES_TERMINAL_DOWNLOAD TRUE
    USES_TERMINAL_CONFIGURE TRUE
    USES_TERMINAL_BUILD TRUE)

ExternalProject_Add_StepDependencies(cutile_llvm download
    "${_required_llvm_revisions_file}")

if(_cutile_llvm_override)
    message(STATUS "LLVM revision override: ${_cutile_llvm_override}")
else()
    message(STATUS "LLVM revision: ${_tileir_required_llvm_revision}")
endif()
message(STATUS "LLVM source: ${CUTILE_LLVM_SOURCE_DIR}")
message(STATUS "Native cuTile compiler: ${CUTILE_COMPILER_PATH}")

unset(_cutile_checkout_command)
unset(_cutile_checkout_script)
unset(_required_llvm_revisions_file)
unset(_cutile_llvm_override)
unset(_tileir_required_llvm_revision)
unset(_tileir_required_llvm_revision_file)
unset(_tileir_to_mlir_dir)
