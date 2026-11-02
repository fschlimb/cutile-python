<!--- SPDX-FileCopyrightText: Copyright (c) <2026> Intel Corporation. All rights reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0 -->

# CPU backend

Use the CPU backend to run cuTile kernels on host arrays, without a GPU.
It accepts NumPy-compatible arrays and CPU PyTorch tensors, along with scalars
and constants. Grids can have one, two, or three dimensions. The backend is
experimental; check correctness and measure performance for your workload.

The [backend overview](../README.md) explains selection, caching, and limitations
shared with XPU.

## Prerequisites

- Linux, Git, and [uv](https://docs.astral.sh/uv/getting-started/installation/).
- GCC with OpenMP development headers and runtime.
- Clang and LLD to build LLVM/MLIR. GCC builds cuTile's extensions; the `_cext`
  extension does not build with Clang on Linux.

No NVIDIA GPU, CUDA driver, or system CUDA Toolkit is required. The build
dependency group supplies CUDA headers for cuTile's `_cext` extension, along with
CMake, Ninja, and Python build packages.

## Build and install

From the repository root:

```bash
git submodule update --init --recursive
uv venv --python 3.12
source .venv/bin/activate
CUTILE_BUILD_VARIANT=cpu uv sync --group cpu --no-build-isolation-package cuda-tile
```

The first build also builds LLVM/MLIR and TileIRToMLIR, so it takes longer than
an incremental build. Later runs reuse `build/llvm`. Keep
`--no-build-isolation-package cuda-tile`: the persistent build trees refer to
tools in this environment.

Running kernels does not require a compiler toolchain, LLVM executables,
development files, or a source checkout. The target system must provide the
C/C++ runtimes and GCC's OpenMP runtime, `libgomp.so.1`; these are not bundled.
Install the OpenMP runtime if it is not already available:

```bash
# Debian / Ubuntu
sudo apt-get install libgomp1

# Fedora / RHEL
sudo dnf install libgomp
```

## Run a sample

With the environment active, from the repository root:

```bash
python samples/cpu/VectorAddition-cpu.py --correctness-check
```

More examples and benchmarks are in [samples/cpu](../../samples/cpu). Check each
script's `--help` for supported sizes, correctness checks, and thread settings.

## Use in a program

Select the backend, keep arrays on the CPU, and pass `None` as the stream:

```python
import cuda.tile as ct

ct.set_backend("cpu")
with ct.backend_options({"num_cpu_threads": 4}):
    ct.launch(None, grid, kernel, cpu_args)
```

Here, `kernel` is your decorated cuTile kernel, `grid` is its block grid, and
`cpu_args` contains its arguments. The call returns after all blocks finish.
When porting a CUDA program, keep the kernel where supported, move its arrays to
the CPU, and remove CUDA stream and device assumptions from the host code.
`ct.set_backend(None)` explicitly restores the CUDA launch path.

### Choose the thread count

`num_cpu_threads` controls how many OpenMP workers execute blocks. The default,
`0`, uses OpenMP's maximum and honors `OMP_NUM_THREADS`. Changing it does not
recompile the kernel.

Bounds checks are enabled by default. You can set `assume_in_bounds=True` in
the same options block when you know every tile load and store is in bounds.
Do not use it to work around an out-of-bounds access.

### Diagnose a problem

Set `CUTILE_CPU_DUMP_MLIR` to a file path to inspect the lowered MLIR.
If importing the CPU launcher reports a missing `libgomp.so.1`, install the
system OpenMP runtime as described above.
If loading a compiled kernel fails, check that `TMPDIR` allows executable
files: the backend loads kernels from temporary shared libraries.

## Run tests

From the repository root, install the test runner and plugins, then run the
backend tests:

```bash
uv pip install pytest pytest-benchmark pytest-env
python -m pytest backends
```

Tests requiring a different native build variant are skipped.

## Distribute a build

Locally built wheels inherit the build host's architecture and glibc baseline.
System C, C++, and OpenMP runtimes are not bundled. The target system's
`libgomp.so.1` must support the OpenMP symbols used by the build.

For compiler passes, argument packing, and host-specific lowering, see the
[CPU entry module](../../src/cuda/tile/_backends/cpu/__init__.py).
