# SPDX-FileCopyrightText: Copyright (c) <2026> Intel Corporation. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set(_runtime_name
    "${CMAKE_SHARED_LIBRARY_PREFIX}mlir_levelzero_runtime${CMAKE_SHARED_LIBRARY_SUFFIX}")

nanobind_add_module(_level_zero level_zero.cpp)
target_link_libraries(_level_zero PRIVATE ${CMAKE_DL_LIBS})
target_compile_definitions(_level_zero PRIVATE
    CUTILE_LEVEL_ZERO_RUNTIME_NAME="${_runtime_name}")
set_target_properties(_level_zero PROPERTIES
    INSTALL_RPATH "$ORIGIN"
    BUILD_WITH_INSTALL_RPATH TRUE)

# Stage a dereferenced copy of the runtime so it is installed as a regular file
# next to the extension, which finds it through its $ORIGIN RPATH.
set(_staged_runtime "${CMAKE_CURRENT_BINARY_DIR}/${_runtime_name}")
add_custom_command(OUTPUT "${_staged_runtime}"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CUTILE_XPU_RUNTIME_PATH}" "${_staged_runtime}"
    DEPENDS "${CUTILE_XPU_RUNTIME_PATH}")
add_custom_target(_level_zero_runtime ALL DEPENDS "${_staged_runtime}")
if (TARGET cutile_llvm)
    add_dependencies(_level_zero_runtime cutile_llvm)
endif()

install(TARGETS _level_zero DESTINATION "${CUTILE_BACKEND_INSTALL_DIR}/xpu")
install(FILES "${_staged_runtime}" DESTINATION "${CUTILE_BACKEND_INSTALL_DIR}/xpu")
