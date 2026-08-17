option(BUILD_TILEIR_TO_MLIR
       "Build TileIRToMLIR for the experimental XPU backend" ON)
set(CUTILE_XPU_MONOLITHIC_INSTALL_DIR "" CACHE PATH
    "Monolithic LLVM/MLIR install containing bin/tileir-to-mlir")
set(CUTILE_XPU_LLVM_REVISION "16ca9a2e1a5b6f687adee1ec980bbc40c448b760")

if (NOT BUILD_TILEIR_TO_MLIR)
    return()
endif()

set(_tileir_to_mlir_dir "${CMAKE_CURRENT_LIST_DIR}/../tileir-to-mlir")
if (NOT EXISTS "${_tileir_to_mlir_dir}/CMakeLists.txt")
    message(FATAL_ERROR
        "TileIRToMLIR sources are missing. Initialize the tileir-to-mlir submodule.")
endif()

# Reuse a prebuilt install: nothing is downloaded or built, but the stub target
# keeps the _cext dependency valid.
if (CUTILE_XPU_MONOLITHIC_INSTALL_DIR)
    set(_monolithic_tool
        "${CUTILE_XPU_MONOLITHIC_INSTALL_DIR}/bin/tileir-to-mlir${CMAKE_EXECUTABLE_SUFFIX}")
    if (NOT EXISTS "${_monolithic_tool}")
        message(FATAL_ERROR
            "CUTILE_XPU_MONOLITHIC_INSTALL_DIR must contain bin/tileir-to-mlir: "
            "${CUTILE_XPU_MONOLITHIC_INSTALL_DIR}")
    endif()
    message(STATUS "Reusing prebuilt TileIRToMLIR from ${_monolithic_tool}")
    add_custom_target(tileir-to-mlir)
    return()
endif()

message(STATUS "Building TileIRToMLIR, LLVM, and MLIR from source")

# Intel's setvars scripts advertise Level Zero through the compiler environment
# variables rather than a package config; these are the names MLIR looks up.
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
if (NOT LevelZeroRuntime_INCLUDE_DIR OR NOT LevelZeroRuntime_LIBRARY)
    message(FATAL_ERROR
        "Level Zero was not found. Source the Intel environment scripts (see "
        "act.sh) or set LEVEL_ZERO_DIR to the Level Zero installation root.")
endif()
message(STATUS "Using Level Zero from ${LevelZeroRuntime_LIBRARY}")

set(_cutile_llvm_source_dir
    "${CMAKE_BINARY_DIR}/_deps/cutile_xpu_llvm-src")
find_package(Git REQUIRED)

if (NOT EXISTS "${_cutile_llvm_source_dir}/llvm/CMakeLists.txt")
    message(STATUS "Cloning LLVM ${CUTILE_XPU_LLVM_REVISION}")
    file(REMOVE_RECURSE "${_cutile_llvm_source_dir}")
    execute_process(
        COMMAND "${GIT_EXECUTABLE}" clone --filter=blob:none
                https://github.com/llvm/llvm-project.git
                "${_cutile_llvm_source_dir}"
        COMMAND_ERROR_IS_FATAL ANY)
    execute_process(
        COMMAND "${GIT_EXECUTABLE}" -C "${_cutile_llvm_source_dir}"
                checkout --detach "${CUTILE_XPU_LLVM_REVISION}"
        COMMAND_ERROR_IS_FATAL ANY)
endif()

execute_process(
    COMMAND "${GIT_EXECUTABLE}" -C "${_cutile_llvm_source_dir}" rev-parse HEAD
    OUTPUT_VARIABLE _llvm_revision
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY)
if (NOT _llvm_revision STREQUAL CUTILE_XPU_LLVM_REVISION)
    message(FATAL_ERROR
        "TileIRToMLIR requires LLVM ${CUTILE_XPU_LLVM_REVISION}, but "
        "${_cutile_llvm_source_dir} is at ${_llvm_revision}.")
endif()

set(LLVM_ENABLE_PROJECTS "mlir" CACHE STRING "" FORCE)
set(LLVM_TARGETS_TO_BUILD "host" CACHE STRING "" FORCE)
set(LLVM_EXPERIMENTAL_TARGETS_TO_BUILD "SPIRV" CACHE STRING "" FORCE)
set(LLVM_ENABLE_ASSERTIONS ON CACHE BOOL "" FORCE)
set(LLVM_INSTALL_UTILS ON CACHE BOOL "" FORCE)
set(MLIR_ENABLE_LEVELZERO_RUNNER ON CACHE BOOL "" FORCE)
set(MLIR_ENABLE_BINDINGS_PYTHON ON CACHE BOOL "" FORCE)
if (Python_EXECUTABLE)
    set(Python3_EXECUTABLE "${Python_EXECUTABLE}" CACHE FILEPATH "" FORCE)
endif()

set(LLVM_EXTERNAL_PROJECTS "tileir-to-mlir" CACHE STRING "" FORCE)
set(LLVM_EXTERNAL_TILEIR_TO_MLIR_SOURCE_DIR "${_tileir_to_mlir_dir}"
    CACHE PATH "" FORCE)

# LLVM processes LLVM_EXTERNAL_PROJECTS before it initializes this variable.
set(LLVM_MAIN_SRC_DIR "${_cutile_llvm_source_dir}/llvm")
add_subdirectory("${_cutile_llvm_source_dir}/llvm"
                 "${CMAKE_BINARY_DIR}/llvm" EXCLUDE_FROM_ALL)