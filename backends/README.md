<!--- SPDX-FileCopyrightText: Copyright (c) <2026> Intel Corporation. All rights reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0 -->

# Non-CUDA backends

cuTile kernels can run on a host CPU or an Intel GPU. Both backends are
experimental and currently support Linux.

The cuTile frontend produces architecture-independent TileIR bytecode. The
selected backend compiles that bytecode for its target, loads the binary, and
runs it.

- **[CPU: build and run](cpu/README.md)** — host arrays and OpenMP execution.
- **[XPU: build and run](xpu/README.md)** — Intel GPU tensors and Level Zero execution.

Start with the guide for your target. Each covers the required software,
installation, a sample run, and the changes needed in your own program.

## Select a backend

```python
import cuda.tile as ct

ct.set_backend("cpu")  # Or "xpu" in an XPU installation.
ct.launch(stream, grid, kernel, args)
ct.set_backend(None)   # Restore the CUDA launch path.
```

`set_backend` controls where subsequent launches run. Passing a CUDA tensor does
not override a CPU or XPU selection; incompatible arguments raise an error.
You can switch explicitly between available backends, but the setting applies
to the whole process, so threads must coordinate changes.

Runtime selection is separate from installation. Each environment is built for
one variant: CUDA, CPU, or XPU. Selecting another backend does not make its
compiler or runtime available. The default build variant is CUDA; see the main
[build instructions](../README.md#building-from-source) for that setup.

CPU and XPU launches are synchronous. CUDA launch options such as
`programmatic_dependent_launch` are only accepted on the CUDA path.

## Configure launches and tuning

Use `ct.backend_options({...})` around the launches you want to configure.
Options apply only within the block and only to the current thread. The backend
guides list supported settings; unknown keys are rejected.

Compiled kernels are cached in memory and on disk. Changing a code-generation
option selects a different binary; changing a launch-only option, such as the
CPU thread count, does not recompile. Autotuning distinguishes both kinds of
options, as well as the backend and array device/layout.

CUDA-specific kernel hints such as `num_ctas` and `occupancy` are ignored.
The [CPU samples](../samples/cpu) and [XPU samples](../samples/xpu) reuse kernels
from the [CUDA samples](../samples) where supported.

## Compatibility and diagnostics

Both backends accept nearest-even, full-precision, and approximate floating-point
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
[shared native compiler module](../src/cuda/tile/_backend/_native.py).

## Implement a backend

A custom backend is a module or object implementing the
[`Backend` protocol](../src/cuda/tile/_backend/_custom.py), selected with
`ct.set_backend(backend)`. Signature construction and binary caching are shared;
the backend supplies option validation, compilation, loading, launching, and
synchronization.

Implementation details are documented in the
[CPU entry module](../src/cuda/tile/_backend/cpu/__init__.py),
[XPU entry module](../src/cuda/tile/_backend/xpu/__init__.py), and
[shared native compiler module](../src/cuda/tile/_backend/_native.py).
