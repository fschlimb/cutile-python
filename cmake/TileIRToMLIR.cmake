if(CUTILE_BUILD_VARIANT MATCHES "^(xpu|cpu)$")
    set(_cutile_build_tileir_default ON)
else()
    set(_cutile_build_tileir_default OFF)
endif()
option(BUILD_TILEIR_TO_MLIR
       "Build TileIRToMLIR for non-CUDA backends"
       ${_cutile_build_tileir_default})
unset(_cutile_build_tileir_default)
set(CUTILE_XPU_MONOLITHIC_INSTALL_DIR "" CACHE PATH
    "Existing LLVM/MLIR and TileIRToMLIR installation")

if(NOT BUILD_TILEIR_TO_MLIR)
    return()
endif()

set(_tileir_to_mlir_dir "${CMAKE_CURRENT_LIST_DIR}/../tileir-to-mlir")
if(NOT EXISTS "${_tileir_to_mlir_dir}/CMakeLists.txt")
    message(FATAL_ERROR
        "TileIRToMLIR sources are missing. Initialize the tileir-to-mlir submodule.")
endif()

if(CUTILE_BUILD_VARIANT STREQUAL "xpu" AND CUTILE_XPU_MONOLITHIC_INSTALL_DIR)
    set(CUTILE_XPU_TOOL_PATH
        "${CUTILE_XPU_MONOLITHIC_INSTALL_DIR}/bin/tileir-to-mlir${CMAKE_EXECUTABLE_SUFFIX}")
    find_library(CUTILE_XPU_RUNTIME_PATH
        NAMES mlir_levelzero_runtime
        PATHS "${CUTILE_XPU_MONOLITHIC_INSTALL_DIR}/lib"
              "${CUTILE_XPU_MONOLITHIC_INSTALL_DIR}/lib64"
        NO_DEFAULT_PATH REQUIRED)
    if(NOT EXISTS "${CUTILE_XPU_TOOL_PATH}")
        message(FATAL_ERROR
            "CUTILE_XPU_MONOLITHIC_INSTALL_DIR must contain bin/tileir-to-mlir")
    endif()
    add_custom_target(cutile_xpu_toolchain)
    return()
endif()

set(_tileir_required_llvm_revision_file
    "${_tileir_to_mlir_dir}/llvm-revision.txt")
set(_triton_required_llvm_revision_file
    "${CMAKE_CURRENT_LIST_DIR}/triton-cpu-required-llvm-revision.txt")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
             "${_tileir_required_llvm_revision_file}")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
             "${_triton_required_llvm_revision_file}")
file(STRINGS "${_tileir_required_llvm_revision_file}"
     _tileir_required_llvm_revision LIMIT_COUNT 1)
file(STRINGS "${_triton_required_llvm_revision_file}"
     _triton_required_llvm_revision LIMIT_COUNT 1)
set(_cutile_llvm_override "")
if(DEFINED ENV{USE_LLVM_REVISION} AND NOT "$ENV{USE_LLVM_REVISION}" STREQUAL "")
    set(_cutile_llvm_override "$ENV{USE_LLVM_REVISION}")
elseif(DEFINED USE_LLVM_REVISION AND NOT USE_LLVM_REVISION STREQUAL "")
    set(_cutile_llvm_override "${USE_LLVM_REVISION}")
endif()

set(CUTILE_LLVM_SOURCE_DIR "${CMAKE_SOURCE_DIR}/.llvm-project" CACHE PATH
    "Persistent LLVM source checkout")
set(CUTILE_LLVM_BINARY_DIR "${CMAKE_SOURCE_DIR}/build/llvm" CACHE PATH
    "LLVM/MLIR build directory")
set(CUTILE_TILEIR_BINARY_DIR "${CMAKE_SOURCE_DIR}/build/tileir-to-mlir" CACHE PATH
    "TileIRToMLIR build directory")
set(CUTILE_LLVM_BUILD_TYPE "Release" CACHE STRING
    "Build type for LLVM, MLIR, and TileIRToMLIR")

find_package(Git REQUIRED)
find_program(CUTILE_CLANG_EXECUTABLE NAMES clang REQUIRED)
find_program(CUTILE_CLANGXX_EXECUTABLE NAMES clang++ REQUIRED)
if(CUTILE_BUILD_VARIANT STREQUAL "xpu")
    find_path(LevelZeroRuntime_INCLUDE_DIR
        NAMES level_zero/ze_api.h
        HINTS ${LEVEL_ZERO_DIR} ENV LEVEL_ZERO_DIR
              ENV CPATH ENV C_INCLUDE_PATH ENV CPLUS_INCLUDE_PATH
        PATH_SUFFIXES include)
    find_library(LevelZeroRuntime_LIBRARY
        NAMES ze_loader
        HINTS ${LEVEL_ZERO_DIR} ENV LEVEL_ZERO_DIR
              ENV LIBRARY_PATH ENV LD_LIBRARY_PATH
        PATH_SUFFIXES lib lib64 lib/x86_64-linux-gnu)
    if(NOT LevelZeroRuntime_INCLUDE_DIR OR NOT LevelZeroRuntime_LIBRARY)
        message(FATAL_ERROR
            "Level Zero was not found. Source the Intel environment scripts or set LEVEL_ZERO_DIR.")
    endif()
endif()

include(ExternalProject)
set(_required_llvm_revisions_file
    "${CMAKE_BINARY_DIR}/tileir-toolchain/required-llvm-revisions.txt")
file(CONFIGURE OUTPUT "${_required_llvm_revisions_file}"
    CONTENT "tileir=${_tileir_required_llvm_revision}\ntriton=${_triton_required_llvm_revision}\noverride=${_cutile_llvm_override}\n" @ONLY)

set(CUTILE_XPU_RUNTIME_PATH
    "${CUTILE_LLVM_BINARY_DIR}/lib/${CMAKE_SHARED_LIBRARY_PREFIX}mlir_levelzero_runtime${CMAKE_SHARED_LIBRARY_SUFFIX}")
set(CUTILE_XPU_MLIR_PYTHON_DIR
    "${CUTILE_LLVM_BINARY_DIR}/tools/mlir/python_packages/mlir_core")
set(CUTILE_XPU_TOOL_PATH
    "${CUTILE_LLVM_BINARY_DIR}/bin/tileir-to-mlir${CMAKE_EXECUTABLE_SUFFIX}")

set(_cutile_checkout_script "${CMAKE_CURRENT_LIST_DIR}/CheckoutLLVM.cmake")
set(_cutile_checkout_command
    "${CMAKE_COMMAND}"
    "-DGIT_EXECUTABLE=${GIT_EXECUTABLE}"
    "-DSOURCE_DIR=${CUTILE_LLVM_SOURCE_DIR}"
    "-DTILEIR_REQUIRED_LLVM_REVISION=${_tileir_required_llvm_revision}"
    "-DTRITON_REQUIRED_LLVM_REVISION=${_triton_required_llvm_revision}"
    "-DREVISION=${_cutile_llvm_override}"
    -P "${_cutile_checkout_script}")

if(CUTILE_BUILD_VARIANT STREQUAL "cpu")
    set(_cutile_llvm_targets host)
    set(_cutile_llvm_projects mlir)
    set(_cutile_llvm_experimental_targets "")
    set(_cutile_llvm_build_targets all)
    set(_cutile_llvm_build_always OFF)
    set(_cutile_llvm_byproduct_args)
else()
    set(_cutile_llvm_targets host)
    set(_cutile_llvm_projects mlir)
    set(_cutile_llvm_experimental_targets SPIRV)
    set(_cutile_llvm_build_targets
        mlir_levelzero_runtime MLIRPythonModules MLIROptLib)
    set(_cutile_llvm_build_always ON)
    set(_cutile_llvm_byproduct_args BUILD_BYPRODUCTS
        "${CUTILE_XPU_RUNTIME_PATH}"
        "${CUTILE_XPU_MLIR_PYTHON_DIR}/mlir/ir.py")
endif()

set(_cutile_llvm_cmake_args
    "-DCMAKE_BUILD_TYPE:STRING=${CUTILE_LLVM_BUILD_TYPE}"
    "-DCMAKE_C_COMPILER:FILEPATH=${CUTILE_CLANG_EXECUTABLE}"
    "-DCMAKE_CXX_COMPILER:FILEPATH=${CUTILE_CLANGXX_EXECUTABLE}"
    "-DCMAKE_LINKER:FILEPATH=${CUTILE_LLD_EXECUTABLE}"
    "-DLLVM_ENABLE_PROJECTS:STRING=${_cutile_llvm_projects}"
    -DLLVM_USE_LINKER:STRING=lld
    "-DLLVM_TARGETS_TO_BUILD:STRING=${_cutile_llvm_targets}"
    "-DLLVM_EXPERIMENTAL_TARGETS_TO_BUILD:STRING=${_cutile_llvm_experimental_targets}"
    -DLLVM_ENABLE_ASSERTIONS:BOOL=ON
    -DLLVM_INSTALL_UTILS:BOOL=ON
    -DLLVM_INCLUDE_TESTS:BOOL=OFF
    -DLLVM_INCLUDE_BENCHMARKS:BOOL=OFF
    -DLLVM_INCLUDE_EXAMPLES:BOOL=OFF
    -DMLIR_INCLUDE_TESTS:BOOL=OFF
    -DMLIR_ENABLE_EXECUTION_ENGINE:BOOL=ON
    -DMLIR_ENABLE_LEVELZERO_RUNNER:BOOL=OFF
    -DMLIR_ENABLE_BINDINGS_PYTHON:BOOL=OFF
    "-DPython3_EXECUTABLE:FILEPATH=${Python_EXECUTABLE}"
    "-Dnanobind_DIR:PATH=${CUTILE_NANOBIND_DIR}")
if(CUTILE_BUILD_VARIANT STREQUAL "xpu")
    list(APPEND _cutile_llvm_cmake_args
        -DMLIR_ENABLE_LEVELZERO_RUNNER:BOOL=ON
        -DMLIR_ENABLE_BINDINGS_PYTHON:BOOL=ON
        "-DCMAKE_PROJECT_INCLUDE:FILEPATH=${CMAKE_CURRENT_LIST_DIR}/ExplicitLevelZeroInclude.cmake"
        "-DCUTILE_LEVEL_ZERO_INCLUDE_DIR:PATH=${LevelZeroRuntime_INCLUDE_DIR}"
        "-DLevelZeroRuntime_INCLUDE_DIR:PATH=${LevelZeroRuntime_INCLUDE_DIR}"
        "-DLevelZeroRuntime_INCLUDE_DIRS:STRING=${LevelZeroRuntime_INCLUDE_DIR}"
        "-DLevelZeroRuntime_LIBRARY:FILEPATH=${LevelZeroRuntime_LIBRARY}")
endif()

ExternalProject_Add(cutile_llvm
    PREFIX "${CMAKE_BINARY_DIR}/tileir-toolchain/llvm"
    SOURCE_DIR "${CUTILE_LLVM_SOURCE_DIR}"
    SOURCE_SUBDIR llvm
    BINARY_DIR "${CUTILE_LLVM_BINARY_DIR}"
    DOWNLOAD_COMMAND ${_cutile_checkout_command}
    UPDATE_COMMAND ""
    PATCH_COMMAND ""
    CMAKE_GENERATOR Ninja
    LIST_SEPARATOR "|"
    CMAKE_ARGS ${_cutile_llvm_cmake_args}
    BUILD_COMMAND
        "${CMAKE_COMMAND}" --build "<BINARY_DIR>" --parallel
        --target ${_cutile_llvm_build_targets}
    INSTALL_COMMAND ""
    ${_cutile_llvm_byproduct_args}
    BUILD_ALWAYS ${_cutile_llvm_build_always}
    EXCLUDE_FROM_ALL TRUE
    USES_TERMINAL_DOWNLOAD TRUE
    USES_TERMINAL_CONFIGURE TRUE
    USES_TERMINAL_BUILD TRUE)

ExternalProject_Add_StepDependencies(cutile_llvm download
    "${_required_llvm_revisions_file}")

ExternalProject_Add(cutile_tileir_to_mlir
    PREFIX "${CMAKE_BINARY_DIR}/tileir-toolchain/tileir-to-mlir"
    SOURCE_DIR "${_tileir_to_mlir_dir}"
    BINARY_DIR "${CUTILE_TILEIR_BINARY_DIR}"
    DOWNLOAD_COMMAND ""
    UPDATE_COMMAND ""
    PATCH_COMMAND ""
    CMAKE_GENERATOR Ninja
    CMAKE_ARGS
        "-DCMAKE_BUILD_TYPE:STRING=${CUTILE_LLVM_BUILD_TYPE}"
        "-DCMAKE_C_COMPILER:FILEPATH=${CUTILE_CLANG_EXECUTABLE}"
        "-DCMAKE_CXX_COMPILER:FILEPATH=${CUTILE_CLANGXX_EXECUTABLE}"
        "-DCMAKE_LINKER:FILEPATH=${CUTILE_LLD_EXECUTABLE}"
        "-DMLIR_DIR:PATH=${CUTILE_LLVM_BINARY_DIR}/lib/cmake/mlir"
        "-DLLVM_DIR:PATH=${CUTILE_LLVM_BINARY_DIR}/lib/cmake/llvm"
        "-DCMAKE_RUNTIME_OUTPUT_DIRECTORY:PATH=${CUTILE_LLVM_BINARY_DIR}/bin"
        -DTILEIR_TO_MLIR_BUILD_CUDA_TILE:BOOL=ON
        -DCUDA_TILE_ENABLE_TESTING:BOOL=OFF
        -DCUDA_TILE_ENABLE_TOOLS:BOOL=OFF
        -DCUDA_TILE_ENABLE_CAPI:BOOL=OFF
        -DCUDA_TILE_ENABLE_BINDINGS_PYTHON:BOOL=OFF
    BUILD_COMMAND
        "${CMAKE_COMMAND}" --build "<BINARY_DIR>" --parallel
        --target tileir-to-mlir
    INSTALL_COMMAND ""
    BUILD_ALWAYS TRUE
    BUILD_BYPRODUCTS "${CUTILE_XPU_TOOL_PATH}"
    EXCLUDE_FROM_ALL TRUE
    USES_TERMINAL_CONFIGURE TRUE
    USES_TERMINAL_BUILD TRUE)

add_dependencies(cutile_tileir_to_mlir cutile_llvm)

if(CUTILE_BUILD_VARIANT STREQUAL "xpu")
    add_custom_target(cutile_xpu_toolchain)
    add_dependencies(cutile_xpu_toolchain cutile_llvm cutile_tileir_to_mlir)
else()
    add_custom_target(cutile_cpu_prerequisites)
    add_dependencies(cutile_cpu_prerequisites cutile_llvm cutile_tileir_to_mlir)
endif()

if(_cutile_llvm_override)
    message(STATUS "LLVM revision override: ${_cutile_llvm_override}")
else()
    message(STATUS "LLVM revision: newest required TileIR/Triton revision")
endif()
message(STATUS "LLVM source: ${CUTILE_LLVM_SOURCE_DIR}")
if(CUTILE_BUILD_VARIANT STREQUAL "xpu")
    message(STATUS "XPU: add to PYTHONPATH: ${CUTILE_XPU_MLIR_PYTHON_DIR}")
endif()

unset(_cutile_checkout_command)
unset(_cutile_checkout_script)
unset(_required_llvm_revisions_file)
unset(_cutile_llvm_override)
unset(_tileir_required_llvm_revision)
unset(_tileir_required_llvm_revision_file)
unset(_triton_required_llvm_revision)
unset(_triton_required_llvm_revision_file)
unset(_tileir_to_mlir_dir)
