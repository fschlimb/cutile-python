<!--- SPDX-FileCopyrightText: Copyright (c) <2026> Intel Corporation. All rights reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0 -->

# Non-CUDA backends

cuTile kernels can run on a host CPU. The CPU backend is experimental and
currently supports Linux.

The cuTile frontend produces architecture-independent TileIR bytecode. The
selected backend compiles that bytecode for its target, loads the binary, and
runs it.

- **[CPU: build and run](cpu/README.md)** — host arrays and OpenMP execution.

Start with the CPU guide. It covers the required software,
installation, a sample run, and the changes needed in your own program.

## Select a backend

```python
import cuda.tile as ct

ct.set_backend("cpu")
ct.launch(stream, grid, kernel, args)
ct.set_backend(None)   # Restore the CUDA launch path.
```

`set_backend` controls where subsequent launches run. Passing a CUDA tensor does
not override a CPU selection; incompatible arguments raise an error.
You can switch explicitly between available backends, but the setting applies
to the whole process, so threads must coordinate changes.

Runtime selection is separate from installation. Each environment is built for
one variant: CUDA or CPU. Selecting another backend does not make its
compiler or runtime available. The default build variant is CUDA; see the main
[build instructions](../README.md#building-from-source) for that setup.

CPU launches are synchronous. CUDA launch options such as
`programmatic_dependent_launch` are only accepted on the CUDA path.

## Configure launches and tuning

Use `ct.backend_options({...})` around the launches you want to configure.
Options apply only within the block and only to the current thread. The backend
guides list supported settings; unknown keys are rejected.

Compiled kernels are cached in memory and on disk. Changing a code-generation
option selects a different binary; changing a launch-only option, such as the
CPU thread count, does not recompile. Autotuning distinguishes both kinds of
options, as well as the backend and array device/layout.

Repeated CPU launches with PyTorch tensors, NumPy arrays, and builtin
scalars use a shared native cache lookup and argument packer. This includes
`torch.nn.Parameter`. Compilation and validation on cache misses remain in
Python. Tensor subclasses, unsupported DLPack exports, and empty or singleton
dimensions use the Python fallback. The native cache is bounded and keeps
arguments alive through synchronous execution.

CUDA-specific kernel hints such as `num_ctas` and `occupancy` are ignored.
The [CPU samples](../samples/cpu) reuse kernels
from the [CUDA samples](../samples) where supported.

## Compatibility and diagnostics

The CPU backend accepts nearest-even, full-precision, and approximate floating-point
operations, and toward-zero integer conversion. Directed floating-point rounding
and ties-away rounding are unsupported and raise `ValueError`. Integer floor and
ceiling division preserve their rounding semantics.

Compilation runs in-process by default. Set `CUDA_TILE_COMPILER_TIMEOUT_SEC` or
use `ct.compiler_timeout(...)` to compile in a worker that can be terminated on
timeout. Without a timeout, a native compiler crash also affects the caller.
Backend-specific MLIR dump settings are listed in the individual guides.

## Rebuild after source changes

Rerun the `uv sync` command in your backend's guide. The build reuses its LLVM
and compiler build trees rather than starting from scratch. For native code
changes in an already configured tree, you can also build and restage directly:

```bash
cmake --build build
cmake --install build --prefix src --component cuTileBackend
```

For custom toolchain locations, LLVM revisions, or prebuilt compilers, see the
build notes at the top of the
[shared native compiler module](../src/cuda/tile/_backends/_native.py).

## Implement a backend

Native CPU sources live in `cpu/`, which contains
`compiler_cpu.cpp`, `launcher_cpu.cpp`, backend-specific build rules in
`compiler.cmake` and `toolchain.cmake`, and a `test/` directory.
Shared MLIR compiler sources live in `mlir/`, common compiler build rules
live directly in this directory, and shared tests live in
`test/`. The launchers share native argument extraction and cache handling in
`launch.h`; each supplies its own ABI packing and execution adapter.

Measure warmed CPU launch overhead with
`python backends/cpu/test/bench_launch.py --compare`. The benchmark alternates
Python and native dispatch in one process, includes real vector-add execution,
and reports medians after compilation and warmup.

Run all backend tests with `python -m pytest backends`, or select
`backends/cpu/test` or `backends/test` for a subset.

A custom backend is a module or object implementing the
[`Backend` protocol](../src/cuda/tile/_backends/_custom.py), selected with
`ct.set_backend(backend)`. Signature construction and binary caching are shared;
the backend supplies option validation, compilation, loading, launching, and
synchronization.

Implementation details are documented in the
[CPU entry module](../src/cuda/tile/_backends/cpu/__init__.py),
and
[shared native compiler module](../src/cuda/tile/_backends/_native.py).
