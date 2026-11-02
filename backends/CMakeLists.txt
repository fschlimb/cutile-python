# SPDX-FileCopyrightText: Copyright (c) <2026> Intel Corporation. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# LLVM builds the compiler as an external project, separately from the launchers.
if(TARGET TileIRToMLIR)
    include("${CMAKE_CURRENT_SOURCE_DIR}/compiler.cmake")
    return()
endif()

# Builds the native parts of the non-CUDA backend selected by CUTILE_BUILD_VARIANT:
# the native compiler and the backend's launcher extension. Installing
# the build (`cmake --install <build> --prefix <site> --component cuTileBackend`)
# stages them and their runtime files into <site>/cuda/tile/_backends, next to the
# Python backend in src/cuda/tile/_backends.

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(FATAL_ERROR "The native CPU/XPU compilers currently support Linux only")
endif()

if (NOT CUTILE_BUILD_VARIANT MATCHES "^(cpu|xpu)$")
    message(FATAL_ERROR "CUTILE_BUILD_VARIANT must be one of: cuda, cpu, xpu")
endif()

# Installing only this component skips the install rules of cuTile's other
# dependencies, such as dlpack.
set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME cuTileBackend)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_VARIABLE nanobind_ROOT
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY)
find_package(nanobind CONFIG REQUIRED)

# Package directory of the backends, relative to the install prefix.
set(CUTILE_BACKEND_INSTALL_DIR cuda/tile/_backends)

set(CUTILE_TILEIR_TO_MLIR_DIR "${CMAKE_CURRENT_SOURCE_DIR}/tileir-to-mlir")
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/TileIRToMLIR.cmake")

install(FILES "${CUTILE_COMPILER_PATH}" DESTINATION "${CUTILE_BACKEND_INSTALL_DIR}")
install(FILES "${CUTILE_LLVM_LICENSE_PATH}"
    DESTINATION "${CUTILE_BACKEND_INSTALL_DIR}/_licenses/llvm")
install(FILES "${CUTILE_TILEIR_TO_MLIR_DIR}/LICENSE"
    DESTINATION "${CUTILE_BACKEND_INSTALL_DIR}/_licenses/tileir-to-mlir")
install(FILES "${CUTILE_TILEIR_TO_MLIR_DIR}/third_party/cuda-tile/LICENSE.txt"
    DESTINATION "${CUTILE_BACKEND_INSTALL_DIR}/_licenses/cuda-tile")

add_subdirectory("${CUTILE_BUILD_VARIANT}")
