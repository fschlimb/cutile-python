# SPDX-FileCopyrightText: Copyright (c) <2026> Intel Corporation. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

if(NOT TARGET TileIRToMLIR)
    message(FATAL_ERROR "Build cuTileCompiler alongside the TileIRToMLIR LLVM external project")
endif()

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(_compiler FREE_THREADED mlir/compiler.cpp)
target_include_directories(_compiler PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_include_directories(_compiler SYSTEM PRIVATE
    "${LLVM_MAIN_SRC_DIR}/../mlir/include"
    "${LLVM_BINARY_DIR}/tools/mlir/include")
target_link_libraries(_compiler PRIVATE
    TileIRToMLIR CudaTileBytecodeReader
    MLIRParser MLIRTransforms MLIRRegisterAllDialects MLIRRegisterAllExtensions
    MLIRBuiltinToLLVMIRTranslation MLIRLLVMToLLVMIRTranslation
    MLIRVectorTransforms)
target_sources(_compiler PRIVATE mlir/LowerVectorScan.cpp)
target_link_options(_compiler PRIVATE "LINKER:--exclude-libs,ALL")
set_target_properties(_compiler PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${LLVM_BINARY_DIR}/lib"
    BUILD_WITH_INSTALL_RPATH TRUE)

include("${CMAKE_CURRENT_SOURCE_DIR}/${CUTILE_BUILD_VARIANT}/compiler.cmake")

if(NOT LLVM_ENABLE_RTTI)
    set_source_files_properties(
        mlir/LowerVectorScan.cpp
        PROPERTIES COMPILE_OPTIONS "-fno-rtti")
endif()

install(TARGETS _compiler LIBRARY DESTINATION lib COMPONENT cuTileCompiler)
install(FILES "${LLVM_MAIN_SRC_DIR}/LICENSE.TXT"
    DESTINATION share/cutile/llvm COMPONENT cuTileCompiler)
