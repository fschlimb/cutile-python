<!--- SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0 -->

cuTile Python
=============

cuTile Python is a programming language for NVIDIA GPUs. The official documentation can be found
on [docs.nvidia.com](https://docs.nvidia.com/cuda/cutile-python),
or built from source located in the [docs](docs/) folder.

--------------------------------------------------------
--------------------------------------------------------

XPU Backend (uv)
----------------

The experimental XPU backend targets Intel GPUs and requires
[tileir-to-mlir](https://github.com/libxsmm/tileir-to-mlir),
included in this repository as the `tileir-to-mlir` submodule. The required
LLVM/MLIR and TileIRToMLIR toolchain is built automatically.

Prerequisites
-------------

Building requires Git, Clang, LLD, and the Intel GPU software including Level
Zero headers and its loader. The XPU dependencies currently require Python 3.12
on Linux x86-64. A CUDA Toolkit is not required for an XPU-only build, and an
XPU device is required only to run kernels. Install `uv` using the
[installation guide](https://docs.astral.sh/uv/getting-started/installation/),
then initialize the environment from the repository root. `uv sync` installs
the required CMake, Ninja, and Python build packages into `.venv`.

```bash
git submodule update --init --recursive
uv venv --python 3.12
source .venv/bin/activate
. <path-to>/intel/setvars.sh --force
. <path-to>/intel-gpu/latest/intel_gpu_vars.sh
```

Initialize the Intel environment once in each fresh shell. Adjust the script
paths for the local Intel software installation. Without environment scripts,
set `LEVEL_ZERO_DIR` to the Level Zero installation root and ensure that Intel
GPU tools such as `ocloc` are on `PATH`.

Backend builds
--------------

CUDA, XPU, and CPU installations are mutually exclusive. Use the backend sync
wrapper so the native build and selected uv extra agree:

```bash
scripts/sync-backend.sh xpu
```

No additional build-isolation options are needed. The first build checks out and
builds the LLVM revision required by TileIRToMLIR. Later builds reuse the
checkout and build outputs, recompiling only changed sources and dependencies.

Override the required LLVM revision when needed:

```bash
USE_LLVM_REVISION=commit-or-tag scripts/sync-backend.sh xpu
```

Revision locations
------------------

The related revisions have distinct purposes:

- `pyproject.toml`, under `tool.uv.sources.triton`, pins the Triton CPU source
   commit installed by uv.
- The `tileir-to-mlir` submodule entry in this repository pins the
   TileIRToMLIR source commit.
- `cmake/triton-cpu-required-llvm-revision.txt` records the minimum LLVM commit
   required by the pinned Triton CPU source.
- `tileir-to-mlir/llvm-revision.txt` records the minimum LLVM commit required
   by TileIRToMLIR. This filename belongs to the TileIRToMLIR submodule and is
   therefore kept as defined upstream.

The build selects the newer compatible LLVM requirement. `USE_LLVM_REVISION`
may select a still newer descendant; older or divergent revisions are rejected.

CPU Backend
-----------

The experimental CPU backend uses the pinned Triton CPU fork and the LLVM
revision required by TileIRToMLIR. If Triton CPU requires a newer revision, the
newer descendant is selected; older or divergent overrides are rejected.

```bash
scripts/sync-backend.sh cpu
python samples/VectorAddition-cpu.py
```

The first build creates persistent LLVM and revision-scoped Triton caches in
`build/llvm` and `build/triton-cpu-<revision>`. Pass `CUTILE_LLVM_SOURCE_DIR`, `CUTILE_LLVM_BINARY_DIR`, or
`CUTILE_TILEIR_BINARY_DIR` through `CMAKE_ARGS`, and set `TRITON_BUILD_DIR` in
the environment, to reuse custom builds. CPU launches are synchronous and currently
support NumPy-compatible arrays, scalar arguments, constants, and one- to
three-dimensional grids.

Reusing custom LLVM and TileIRToMLIR builds
-------------------------------------------

Install the custom LLVM/MLIR and TileIRToMLIR builds into a common prefix, then
pass that prefix to cuTile. The installation must use the LLVM revision required
by TileIRToMLIR and provide `tileir-to-mlir`, the MLIR Level Zero runtime, and
the MLIR Python bindings.

```bash
CMAKE_ARGS="-DCUTILE_XPU_MONOLITHIC_INSTALL_DIR=/path/to/prefix" \
   scripts/sync-backend.sh xpu
```

Running on XPU
--------------

1. Log in to a node with an XPU device and initialize a fresh shell once:

   ```bash
   source .venv/bin/activate
   . <path-to>/intel/setvars.sh --force
   . <path-to>/intel-gpu/latest/intel_gpu_vars.sh
   ```

   Adjust the script paths for the local Intel software installation. The MLIR
   Level Zero runtime is packaged with the native launcher.
2. From the repository root, add the MLIR Python bindings to `PYTHONPATH`. The
   regular build prints the exact command when it finishes; for a monolithic
   installation, use its install prefix:

   ```bash
   export PYTHONPATH=build/llvm/tools/mlir/python_packages/mlir_core:${PYTHONPATH}
   # Or: export PYTHONPATH=/path/to/install/python_packages/mlir_core:${PYTHONPATH}
   ```

3. Run an example

   ```bash
   python -u samples/MatMul-xpu.py --correctness-check
   ```

   Optional environment variables: `CUTILE_XPU_ARCH` selects the target architecture,
   `CUTILE_XPU_DUMP_MLIR` dumps the generated MLIR, and
   `CUTILE_XPU_TILEIR_TO_MLIR` overrides the compiler tool path.

How to adapt an existing cuTile program
---------------------------------------

1. Import `xpu` with `from cuda.tile._backend import xpu` and select it with
   `ct.set_backend("xpu")`.
2. Use XPU tensors and streams: replace `tensor.is_cuda` with `tensor.is_xpu`, and
   `torch.cuda.current_stream()` with `torch.xpu.current_stream()`.
3. Remove CUDA-specific kernel hints such as `occupancy` and `num_ctas`. Wrap each launch in
   `xpu.compile_options(...)`, providing the workgroup shape (`wg_m`, `wg_n`, and
   `block_threads`).
4. Respect the XPU schedule's tile-alignment requirements. Validate the input dimensions before
   launch, and reshape or pad inputs when the selected tile sizes require it.
5. Work around uses of CAS spin loops or lock-based atomic accumulation.

Notes
-----

This is an experimental extension. The implementation is not complete and you should not
expect good performance when running the kernels. At this point it is merely a POC for running
cutile/tileir on XPU. Getting good performance requires a more sophisticated compiler pipeline
in MLIR and some changes to the cutile code as well to optimize block sizes and such.
Due to limitations in the upstream MLIR lowering to Xe HW some of the ported examples don't
even compile yet or produce incorrect results.

End of XPU backend

--------------------------------------------------------
--------------------------------------------------------

Example
-------
```python
# This examples uses CuPy which can be installed via `pip install cupy-cuda13x`
# Make sure cuda toolkit 13.1+ is installed: https://developer.nvidia.com/cuda-downloads

import cuda.tile as ct
import cupy
import numpy as np

TILE_SIZE = 16

# cuTile kernel for adding two dense vectors. It runs in parallel on the GPU.
@ct.kernel
def vector_add_kernel(a, b, result):
    block_id = ct.bid(0)
    a_tile = ct.load(a, index=(block_id,), shape=(TILE_SIZE,))
    b_tile = ct.load(b, index=(block_id,), shape=(TILE_SIZE,))
    result_tile = a_tile + b_tile
    ct.store(result, index=(block_id,), tile=result_tile)

# Generate input arrays
rng = cupy.random.default_rng()
a = rng.random(128)
b = rng.random(128)
expected = cupy.asnumpy(a) + cupy.asnumpy(b)

# Allocate an output array and launch the kernel
result = cupy.zeros_like(a)
grid = (ct.cdiv(a.shape[0], TILE_SIZE), 1, 1)
ct.launch(cupy.cuda.get_current_stream(), grid, vector_add_kernel, (a, b, result))

# Verify the results
result_np = cupy.asnumpy(result)
np.testing.assert_array_almost_equal(result_np, expected)
```

More examples can be found at [Samples](samples/) and [TileGym](https://github.com/NVIDIA/TileGym).

System Requirements
-------------------
cuTile Python generates kernels based on [Tile IR](https://docs.nvidia.com/cuda/tile-ir/)
which requires NVIDIA Driver r580 or later to run.
Furthermore, the `tileiras` compiler (version 13.2) only supports Blackwell GPU and Ampere/Ada
GPU. Hopper GPU will be supported in the coming versions.
Checkout the [prerequisites](https://docs.nvidia.com/cuda/cutile-python/quickstart.html#prerequisites)
for full list of requirements.


Installing from PyPI
--------------------
cuTile Python is published on [PyPI](https://pypi.org/) under the
[cuda-tile](https://pypi.org/project/cuda-tile/) package name and can be installed with `pip`:

```bash
pip install cuda-tile[cuda]
```

The optional `cuda` dependency installs the `tileiras` compiler directly into your Python
environment.


If you do not want to have `tileiras` inside the python environment, run
```
pip install cuda-tile
```
and install [CUDA Toolkit 13.1+](https://developer.nvidia.com/cuda-downloads) separately.

On a Debian-based system, use `apt-get install cuda-tileiras-13.2
cuda-compiler-13.2` instead of `apt-get install cuda-toolkit-13.2` if you wish
to avoid installing the full CUDA Toolkit.


Building from Source
--------------------
cuTile is written mostly in Python, but includes a C++ extension which needs to be built.
You will need:
- A C++17-capable compiler, such as GNU C++ or MSVC;
- CMake 3.24+;
- Ninja and LLD on Linux, or msbuild on Windows;
- Python 3.10+ with development headers (`venv` module is recommended but optional);
- [CUDA Toolkit 13.1+](https://developer.nvidia.com/cuda-downloads) headers when
   building the CUDA extension.

On an Ubuntu system, the first four dependencies can be installed with APT:
```
sudo apt-get update && sudo apt-get install build-essential cmake ninja-build lld python3-dev python3-venv
```

The CMakeLists.txt script will also automatically download
the [DLPack](https://github.com/dmlc/dlpack) dependency from GitHub.
If you wish to disable this behavior and provide your own copy of DLPack,
set the `CUDA_TILE_CMAKE_DLPACK_PATH` environment variable to a local path
to the DLPack source tree.

Unless you are already using a Python virtual environment, it is recommended to create one
in order to avoid installing cuTile globally:

```
python3 -m venv env
source env/bin/activate
```

Once the build dependencies are in place, the simplest way to build cuTile is to install it
in editable mode. Install the build requirements into the environment and disable
build isolation so CMake records persistent paths:

```
pip install "setuptools==80.10.2" wheel "nanobind>=2.9,<3"
CMAKE_ARGS="-DBUILD_TILEIR_TO_MLIR=OFF" pip install --no-build-isolation -e .
```

`BUILD_TILEIR_TO_MLIR=OFF` disables the XPU backend; the CUDA extension is built
when CUDA headers are available. Omit that option after initializing the Intel
environment when building the XPU backend. The command creates the `build`
directory and invokes the CMake build and install steps. After changing C++
code, rebuild and restage the native modules with:

```
cmake --build build --parallel
cmake --install build --prefix src --component Python
```

Experimental Features (Optional)
--------------------------------
cuTile now provides an experimental package containing APIs that are still under active development.
These are **not** part of the stable `cuda.tile` API and may change.

To enable the experimental features when working from a source checkout, install the experimental
package from the repository root:
```
pip install ./experimental/tile_experimental
```

You can also install it directly from a GitHub repository subdirectory:
```
pip install \
  "git+https://github.com/NVIDIA/cutile-python.git#egg=cuda-tile-experimental&subdirectory=experimental/tile_experimental"
```

For example, this will make the experimental namespace available for autotuner:
```
from cuda.tile_experimental import autotune_launch, clear_autotune_cache
```

Running Tests
-------------
cuTile uses the [pytest](https://pytest.org) framework for testing.
Tests have extra dependencies, such as PyTorch, which can be installed with

For Python non-free-threading build:
```
pip install -r test/requirements.txt
```

Or for Python free-threading build:
```
pip install -r test/requirements-ft.txt
```

The tests are located in the [test/](test/) directory. To run a specific test file,
for example `test_copy.py`, use the following command:
```
pytest test/test_copy.py
```

Copyright and License Information
---------------------------------
Copyright © 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

cuTile-Python is licensed under the Apache 2.0 license. See the [LICENSES](LICENSES/) folder for the full license text.
