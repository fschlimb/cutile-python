# SPDX-FileCopyrightText: Copyright (c) <2026> Intel Corporation. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

target_sources(_compiler PRIVATE cpu/compiler_cpu.cpp cpu/ApproximateMath.cpp)
target_include_directories(_compiler SYSTEM PRIVATE
    "${LLVM_MAIN_SRC_DIR}/../lld/include")
target_link_libraries(_compiler PRIVATE
    MLIRTransformDialectTransforms
    MLIRMemRefTransforms MLIRMathTransforms
    MLIRAffineToStandard MLIRVectorToSCF
    MLIRSCFToControlFlow MLIRVectorToLLVMPass
    MLIRMathToLibm MLIRConvertToLLVMPass
    MLIRReconcileUnrealizedCasts
    MLIRTargetLLVMIRExport lldELF lldCommon ${CMAKE_DL_LIBS} m)
llvm_map_components_to_libnames(_llvm_libs nativecodegen passes)
target_link_libraries(_compiler PRIVATE ${_llvm_libs})

if(NOT LLVM_ENABLE_RTTI)
    set_source_files_properties(cpu/ApproximateMath.cpp
        PROPERTIES COMPILE_OPTIONS "-fno-rtti")
endif()