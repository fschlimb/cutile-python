# SPDX-FileCopyrightText: Copyright (c) <2026> Intel Corporation. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

set(TILEIR_TO_MLIR_REPOSITORY "https://github.com/libxsmm/tileir-to-mlir"
    CACHE STRING "TileIRToMLIR repo URL")
set(TILEIR_TO_MLIR_REVISION "0a194dac40b22fb8396742562a976b578d267e04"
    CACHE STRING "TileIRToMLIR revision")
set(TILEIR_TO_MLIR_REVISION_SHA256
    "6461fa92b206ab6f2fc01b95b1ee65ae43c512b21dc413301bc7da7b935bf1d1"
    CACHE STRING "Expected SHA256 of the pinned TileIRToMLIR archive")

set(CUDA_TILE_REPOSITORY "https://github.com/NVIDIA/cuda-tile"
    CACHE STRING "CUDA Tile repo URL")
set(CUDA_TILE_REVISION "e11729ed4e228af005dc8cc113d7b84157e9d654"
    CACHE STRING "CUDA Tile revision required by TileIRToMLIR")
set(CUDA_TILE_REVISION_SHA256
    "56c03d732ac5f23e0ae9c0c184ff65aa282c24bd30a1bde723457f8e986e855e"
    CACHE STRING "Expected SHA256 of the pinned CUDA Tile archive")

macro(fetch_tileir_to_mlir)
    include(FetchContent)
    message(STATUS "Fetching TileIRToMLIR: ${TILEIR_TO_MLIR_REVISION}")
    FetchContent_Declare(
        tileir_to_mlir
        URL "${TILEIR_TO_MLIR_REPOSITORY}/archive/${TILEIR_TO_MLIR_REVISION}.tar.gz"
        URL_HASH "SHA256=${TILEIR_TO_MLIR_REVISION_SHA256}"
        SOURCE_SUBDIR does-not-exist
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
    message(STATUS "Fetching CUDA Tile: ${CUDA_TILE_REVISION}")
    FetchContent_Declare(
        cuda_tile
        URL "${CUDA_TILE_REPOSITORY}/archive/${CUDA_TILE_REVISION}.tar.gz"
        URL_HASH "SHA256=${CUDA_TILE_REVISION_SHA256}"
        SOURCE_SUBDIR does-not-exist
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
    FetchContent_MakeAvailable(tileir_to_mlir cuda_tile)
    set(CUTILE_TILEIR_TO_MLIR_DIR "${tileir_to_mlir_SOURCE_DIR}")
    set(CUTILE_CUDA_TILE_SOURCE_DIR "${cuda_tile_SOURCE_DIR}")
endmacro()