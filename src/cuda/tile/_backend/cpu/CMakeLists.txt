# SPDX-FileCopyrightText: Copyright (c) <2026> Intel Corporation.
# SPDX-License-Identifier: Apache-2.0

find_package(OpenMP REQUIRED COMPONENTS CXX)

nanobind_add_module(_cpu launcher.cpp)
target_link_libraries(_cpu PRIVATE OpenMP::OpenMP_CXX ${CMAKE_DL_LIBS})

install(TARGETS _cpu DESTINATION "${CUTILE_BACKEND_INSTALL_DIR}/cpu")

find_program(CUTILE_PATCHELF_EXECUTABLE NAMES patchelf REQUIRED)
set(_runtime_dirs)
foreach(_library IN LISTS OpenMP_CXX_LIBRARIES)
    get_filename_component(_directory "${_library}" DIRECTORY)
    list(APPEND _runtime_dirs "${_directory}")
endforeach()
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/../cmake/InstallCompiler.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/InstallCompiler.cmake" @ONLY)
install(SCRIPT "${CMAKE_CURRENT_BINARY_DIR}/InstallCompiler.cmake")
