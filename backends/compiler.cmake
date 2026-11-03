# SPDX-FileCopyrightText: Copyright (c) <2026> Intel Corporation. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

if(NOT TARGET TileIRToMLIR)
    message(FATAL_ERROR "Build cuTileCompiler alongside the TileIRToMLIR LLVM external project")
endif()

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(_compiler FREE_THREADED compiler.cpp)
target_include_directories(_compiler PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_include_directories(_compiler SYSTEM PRIVATE
    "${LLVM_MAIN_SRC_DIR}/../mlir/include"
    "${LLVM_BINARY_DIR}/tools/mlir/include")
target_link_libraries(_compiler PRIVATE
    TileIRToMLIR CudaTileBytecodeReader
    MLIRParser MLIRTransforms MLIRRegisterAllDialects MLIRRegisterAllExtensions
    MLIRBuiltinToLLVMIRTranslation MLIRLLVMToLLVMIRTranslation)
target_link_options(_compiler PRIVATE "LINKER:--exclude-libs,ALL")
set_target_properties(_compiler PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${LLVM_BINARY_DIR}/lib"
    BUILD_WITH_INSTALL_RPATH TRUE)

if(CUTILE_BUILD_VARIANT STREQUAL "cpu")
    target_sources(_compiler PRIVATE cpu/compiler_cpu.cpp cpu/LowerVectorScan.cpp)
    if(NOT LLVM_ENABLE_RTTI)
        set_source_files_properties(cpu/LowerVectorScan.cpp PROPERTIES
            COMPILE_OPTIONS "-fno-rtti")
    endif()
    target_include_directories(_compiler SYSTEM PRIVATE
        "${LLVM_MAIN_SRC_DIR}/../lld/include")
    target_link_libraries(_compiler PRIVATE
        MLIRTransformDialectTransforms
        MLIRMemRefTransforms MLIRVectorTransforms MLIRMathTransforms
        MLIRAffineToStandard MLIRVectorToSCF
        MLIRSCFToControlFlow MLIRVectorToLLVMPass
        MLIRMathToLibm MLIRConvertToLLVMPass
        MLIRReconcileUnrealizedCasts
        MLIRTargetLLVMIRExport lldELF lldCommon ${CMAKE_DL_LIBS} m)
    llvm_map_components_to_libnames(_llvm_libs nativecodegen passes)
    target_link_libraries(_compiler PRIVATE ${_llvm_libs})
else()
    if(NOT OCLOC_FOUND OR LLVM_BUILD_LLVM_DYLIB)
        message(FATAL_ERROR "The XPU compiler requires MLIR's in-process libocloc support")
    endif()
    target_sources(_compiler PRIVATE xpu/compiler_xpu.cpp)
    target_include_directories(_compiler PRIVATE "${OCLOC_INCLUDE_DIR}")
    target_link_libraries(_compiler PRIVATE
        MLIRGPUPipelines MLIRXeVMTarget
        MLIRGPUToLLVMIRTranslation
        "${OCLOC_LIBRARY}")
endif()

install(TARGETS _compiler LIBRARY DESTINATION lib COMPONENT cuTileCompiler)
install(FILES "${LLVM_MAIN_SRC_DIR}/LICENSE.TXT"
    DESTINATION share/cutile/llvm COMPONENT cuTileCompiler)
