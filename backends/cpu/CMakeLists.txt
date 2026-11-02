# SPDX-FileCopyrightText: Copyright (c) <2026> Intel Corporation. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

find_package(OpenMP REQUIRED COMPONENTS CXX)

nanobind_add_module(_cpu FREE_THREADED launcher.cpp)
target_link_libraries(_cpu PRIVATE OpenMP::OpenMP_CXX ${CMAKE_DL_LIBS})

install(TARGETS _cpu DESTINATION "${CUTILE_BACKEND_INSTALL_DIR}/cpu")
