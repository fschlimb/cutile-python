# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

function(_CUDAToolkit_find root)
    if (EXISTS "${root}/include/cuda.h")
        message(STATUS "Found cuda.h in ${root}/include")
    else()
        message(STATUS "No cuda.h found in ${root}/include")
        return()
    endif()

    set(CUDAToolkit_INCLUDE_DIR "${root}/include" CACHE PATH "" FORCE)
endfunction()

if (NOT CUDAToolkit_INCLUDE_DIRS)
    message(STATUS "Looking for CUDA Toolkit")
    if (DEFINED CUDAToolkit_ROOT)
        _CUDAToolkit_find("${CUDAToolkit_ROOT}")
    endif()
    if (UNIX)
        _CUDAToolkit_find("/usr/local/cuda")
    endif()
    if (MSVC)
        _CUDAToolkit_find("$ENV{CUDA_PATH}")
    endif()
    if (DEFINED ENV{CUDAToolkit_ROOT})
        _CUDAToolkit_find("$ENV{CUDAToolkit_ROOT}")
    endif()

    # cuda.h is the only CUDA build dependency (the driver is loaded at
    # runtime), so the pip CUDA wheels are enough when no toolkit is installed.
    set(_CUDAToolkit_site_packages)
    if (Python_EXECUTABLE)
        execute_process(
            COMMAND "${Python_EXECUTABLE}" -c
                "import sysconfig; print(sysconfig.get_paths()['purelib'])"
            OUTPUT_VARIABLE _CUDAToolkit_purelib
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET)
        list(APPEND _CUDAToolkit_site_packages "${_CUDAToolkit_purelib}")
    endif()
    # Build frontends isolate the build interpreter from the target venv.
    if (DEFINED ENV{VIRTUAL_ENV})
        file(GLOB _CUDAToolkit_venv_site_packages
             "$ENV{VIRTUAL_ENV}/lib/python*/site-packages"
             "$ENV{VIRTUAL_ENV}/Lib/site-packages")
        list(APPEND _CUDAToolkit_site_packages ${_CUDAToolkit_venv_site_packages})
    endif()
    foreach(_CUDAToolkit_site_dir IN LISTS _CUDAToolkit_site_packages)
        foreach(_CUDAToolkit_wheel nvidia/cu13 nvidia/cuda_runtime)
            if (NOT CUDAToolkit_INCLUDE_DIR)
                _CUDAToolkit_find("${_CUDAToolkit_site_dir}/${_CUDAToolkit_wheel}")
            endif()
        endforeach()
    endforeach()
endif()

find_package_handle_standard_args(CUDAToolkit
    REQUIRED_VARS
        CUDAToolkit_INCLUDE_DIR
)

if(CUDAToolkit_FOUND)
    set(CUDAToolkit_INCLUDE_DIRS ${CUDAToolkit_INCLUDE_DIR})
endif()
