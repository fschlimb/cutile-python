# SPDX-FileCopyrightText: Copyright (c) <2026> Intel Corporation. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

find_path(LevelZeroRuntime_INCLUDE_DIR
    NAMES level_zero/ze_api.h
    HINTS ${LEVEL_ZERO_DIR} ENV LEVEL_ZERO_DIR
          ENV CPATH ENV C_INCLUDE_PATH ENV CPLUS_INCLUDE_PATH
    PATH_SUFFIXES include REQUIRED)
find_library(LevelZeroRuntime_LIBRARY
    NAMES ze_loader
    HINTS ${LEVEL_ZERO_DIR} ENV LEVEL_ZERO_DIR
          ENV LIBRARY_PATH ENV LD_LIBRARY_PATH
    PATH_SUFFIXES lib lib64 lib/x86_64-linux-gnu REQUIRED)

nanobind_add_module(_level_zero FREE_THREADED launcher_xpu.cpp)
target_include_directories(_level_zero PRIVATE "${LevelZeroRuntime_INCLUDE_DIR}")
target_link_libraries(_level_zero PRIVATE "${LevelZeroRuntime_LIBRARY}")

install(TARGETS _level_zero DESTINATION "${CUTILE_BACKEND_INSTALL_DIR}/xpu")
