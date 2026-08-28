# SPDX-FileCopyrightText: Copyright (c) <2026> Intel Corporation. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

nanobind_add_module(_level_zero level_zero.cpp)
target_link_libraries(_level_zero PRIVATE ${CMAKE_DL_LIBS})
target_compile_definitions(_level_zero PRIVATE
    CUTILE_LEVEL_ZERO_RUNTIME_NAME="${CMAKE_SHARED_LIBRARY_PREFIX}mlir_levelzero_runtime${CMAKE_SHARED_LIBRARY_SUFFIX}")
set_target_properties(_level_zero PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/xpu"
    INSTALL_RPATH "$ORIGIN"
    BUILD_WITH_INSTALL_RPATH TRUE)

set(_cutile_staged_runtime
    "${CMAKE_BINARY_DIR}/xpu/${CMAKE_SHARED_LIBRARY_PREFIX}mlir_levelzero_runtime${CMAKE_SHARED_LIBRARY_SUFFIX}")
add_custom_command(OUTPUT "${_cutile_staged_runtime}"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CUTILE_XPU_RUNTIME_PATH}" "${_cutile_staged_runtime}"
    DEPENDS "${CUTILE_XPU_RUNTIME_PATH}")
add_custom_target(_level_zero_runtime DEPENDS "${_cutile_staged_runtime}")
add_dependencies(_level_zero_runtime cutile_xpu_toolchain)
add_dependencies(_level_zero _level_zero_runtime)

install(TARGETS _level_zero
    LIBRARY DESTINATION cuda/tile COMPONENT Python
    RUNTIME DESTINATION cuda/tile
    COMPONENT Python)
install(FILES "${_cutile_staged_runtime}"
    DESTINATION cuda/tile
    COMPONENT Python)