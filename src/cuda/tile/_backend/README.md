<!--- SPDX-FileCopyrightText: Copyright (c) <2026> Intel Corporation. All rights reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0 -->

Non-CUDA Backends
=================

`cuda.tile.set_backend("cpu")` or `set_backend("xpu")` routes `cuda.tile.launch` to an
experimental backend; `clear_backend()` restores CUDA. Backend options are scoped with
`cuda.tile.backend_options({...})`; each backend rejects keys it does not know.
The backend protocol is described in [`__init__.py`](__init__.py).

XPU Backend (uv)
----------------

The experimental XPU backend targets Intel GPUs and requires
[tileir-to-mlir](https://github.com/libxsmm/tileir-to-mlir),
included in this repository as the `tileir-to-mlir` submodule. The required
LLVM/MLIR and TileIRToMLIR toolchain is built automatically.

Prerequisites
-------------

Building requires Git, GCC, Clang, LLD, and the Intel GPU software including
Level Zero headers and its loader. Clang and LLD build LLVM/MLIR; GCC, the
default host compiler, builds cuTile's extensions, because the `_cext`
extension does not build with Clang on Linux. The XPU dependencies currently
require Python 3.12 on Linux x86-64. No CUDA Toolkit, CUDA driver, or NVIDIA
GPU is required, and an XPU device is required only to run kernels. Install `uv` using the
[installation guide](https://docs.astral.sh/uv/getting-started/installation/),
then initialize the environment from the repository root. `uv sync` installs
the required CMake, Ninja, and Python build packages into `.venv`, including
the CUDA headers from the `cuda-toolkit` wheels needed to build cuTile's `_cext`
extension, which every variant builds. Set `CUDAToolkit_ROOT` to use the headers
of a local CUDA Toolkit instead.

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

CUDA, XPU, and CPU installations are mutually exclusive. Run `uv sync` from
the repository root, selecting the matching extra and build variant:

```bash
CUTILE_BUILD_VARIANT=xpu uv sync --extra xpu
```

No additional build-isolation options are needed. The first build checks out and
builds the LLVM revision required by TileIRToMLIR. Later builds reuse the
checkout and build outputs, recompiling only changed sources and dependencies.
LLVM is reconfigured only when the requested settings or its actual CMake cache
change; sequential CPU/XPU builds can still share the LLVM build directory.

`CUTILE_BUILD_VARIANT` defaults to `cuda`, which builds exactly as described in
[Building from Source](../../../../README.md#building-from-source). All `cpu` and `xpu`
backend code lives in this directory: the Python backends, their native launchers,
the native compiler, and the CMake build, with tests in
[test/backend](../../../../test/backend) and samples in
[samples/cpu](../../../../samples/cpu) and [samples/xpu](../../../../samples/xpu).
The samples import the kernels they share with the CUDA samples in
[samples](../../../../samples). These variants stage `_cext` like CUDA builds and
copy their backend extensions and runtime files into `cuda/tile/_backend` with
`cmake --install`. After changing their C++ code, rerun `uv sync`, or rebuild
and restage manually:

```bash
cmake --build build && cmake --install build --prefix src --component cuTileBackend
```

CPU/XPU compilation uses a private nanobind extension linked against the pinned
LLVM/MLIR and TileIRToMLIR libraries. Installed kernels do not invoke LLVM tools
or require MLIR Python bindings. The original CUDA compiler and build path are
unchanged. These native backend builds currently support Linux.

Compilation runs in-process by default, releasing the GIL and using a fresh
MLIR context per call. XPU device-code generation is serialized because
`libocloc` does not document thread safety. An explicit `ct.compiler_timeout(...)` or
`CUDA_TILE_COMPILER_TIMEOUT_SEC` runs the same native compiler in a Python worker
so the timeout can safely terminate compilation. Without a timeout, compiler
crashes also affect the calling process. LLVM/MLIR assertions are disabled; add
`-DCUTILE_LLVM_ENABLE_ASSERTIONS=ON` to `CMAKE_ARGS` to debug the compiler.
Known compiler failures retain their `ValueError` or `RuntimeError` exception
type in the timeout worker, without an additional worker traceback.

Override the required LLVM revision when needed:

```bash
USE_LLVM_REVISION=commit-or-tag CUTILE_BUILD_VARIANT=xpu uv sync --extra xpu
```

Revision locations
------------------

The related revisions have distinct purposes:

- The `tileir-to-mlir` submodule entry in this repository pins the
   TileIRToMLIR source commit.
- `tileir-to-mlir/llvm-revision.txt` records the minimum LLVM commit required
   by TileIRToMLIR. This filename belongs to the TileIRToMLIR submodule and is
   therefore kept as defined upstream.

The build uses the LLVM revision required by TileIRToMLIR. `USE_LLVM_REVISION`
may select a newer descendant; older or divergent revisions are rejected.

CPU Backend
-----------

The experimental CPU backend uses LLVM/MLIR libraries and a generic
nanobind/OpenMP launcher built once with cuTile. Building it requires GCC with
OpenMP development headers and runtime, besides the Clang toolchain for LLVM. Kernels
run the existing TileIRToMLIR and MLIR
passes, LLVM optimization and native object emission, and the LLD ELF linker
through their C++ APIs. Shared libraries remain managed by cuTile's binary
cache. TileIRToMLIR emits a uniform
CPU ABI: one pointer to immutable eight-byte argument slots, followed by six
`i32` block-id/grid-dimension arguments. No launcher is compiled per kernel.

Before general MLIR lowering, CPU compilation applies a schedule from
[cpu_schedule.mlir](cpu/cpu_schedule.mlir) selected by the host: generic patterns
on any host, x86 contraction-loop preparation on x86-64, and the AMX lowering only
on hosts reporting `amx_tile`, `amx_bf16`, and `amx_int8`. Final LLVM conversion
lowers x86 AMX operations and loop-carried tile types together with control
flow. On Linux x86-64 hosts with AMX tile support, the
native launcher requests the required process-wide tile-data permission when
imported, before executing AMX kernels; permission failures raise an error.

```bash
CUTILE_BUILD_VARIANT=cpu uv sync --extra cpu
python samples/cpu/VectorAddition-cpu.py --correctness-check
```

The first build creates a persistent LLVM build in `build/llvm`, with
TileIRToMLIR and the native compiler as LLVM external projects. Pass
`CUTILE_LLVM_SOURCE_DIR` or `CUTILE_LLVM_BINARY_DIR` through `CMAKE_ARGS` to
reuse custom locations. Installed CPU packages include their native runtime
libraries; no C compiler, LLVM executables, development files, or checkout are
needed at run time. Packaging requires `patchelf`, supplied by the build group.
Locally built wheels inherit the build host's architecture and glibc baseline.
The system C and C++ runtimes are not bundled, as for manylinux wheels.
Runtime-library license notices are collected from RPM/dpkg metadata where
available; verify redistribution notices before publishing wheels.

CPU launches are synchronous and support host NumPy-compatible arrays and
PyTorch tensors, scalar arguments, constants, and one- to three-dimensional
grids. Blocks use static OpenMP scheduling without allocating a grid-sized
coordinate array. In a `ct.backend_options` block, `num_cpu_threads` selects
the worker count; zero (the default) uses OpenMP's maximum thread count and
honors `OMP_NUM_THREADS`. One thread and single-block grids bypass parallel
dispatch. The thread count is a launch option and does not recompile kernels.
The `assume_in_bounds` option skips bounds checks of tile loads and stores.
`CUTILE_CPU_DUMP_MLIR` writes the lowered MLIR to a file. Kernels are loaded from
temporary files, so `TMPDIR` must allow executable files.

Reusing custom LLVM and TileIRToMLIR builds
-------------------------------------------

Build LLVM with the TileIRToMLIR and cuTileCompiler external projects, then
install into a common prefix. The installation must use the required LLVM
revision and provide `lib/_compiler<EXT_SUFFIX>` for the active Python version
and XPU variant, plus the MLIR Level Zero runtime and
`share/cutile/llvm/LICENSE.TXT`. Installing the `cuTileCompiler` CMake component
stages the compiler and this notice together. Consuming the prefix does not
require an LLVM source checkout.

```bash
CMAKE_ARGS="-DCUTILE_XPU_MONOLITHIC_INSTALL_DIR=/path/to/prefix" \
   CUTILE_BUILD_VARIANT=xpu uv sync --extra xpu
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
2. The native compiler uses `libocloc` in-process. Intel's offline compiler
   libraries, Level Zero loader, and GPU driver remain external dependencies
   provided by the environment scripts above; the build requires the
   `ocloc_api.h` development header. No MLIR `PYTHONPATH` setup is needed.
   Run an example:

   ```bash
   python -u samples/xpu/MatMul-xpu.py --correctness-check
   ```

   Optional environment variables: `CUTILE_XPU_ARCH` selects the target architecture,
   `CUTILE_XPU_DUMP_MLIR` dumps the generated MLIR.

How to adapt an existing cuTile program
---------------------------------------

1. Select the backend with `ct.set_backend("xpu")`.
2. Use XPU tensors and streams: replace `tensor.is_cuda` with `tensor.is_xpu`, and
   `torch.cuda.current_stream()` with `torch.xpu.current_stream()`.
3. Remove CUDA-specific kernel hints such as `occupancy` and `num_ctas`. Optionally wrap
   launches in `ct.backend_options(...)` to set the launch block, either directly with
   `block_threads` or derived from the tile sizes `wg_m`, `wg_n`, `sg_m`, and `sg_n`
   (given together).
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
