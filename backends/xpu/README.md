<!--- SPDX-FileCopyrightText: Copyright (c) <2026> Intel Corporation. All rights reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0 -->

# Intel XPU backend

Use the XPU backend to run cuTile kernels on Intel GPUs through Level Zero.
This is experimental: operation coverage and performance are limited, and some
examples may fail to compile or produce incorrect results. Run correctness
checks for the kernels and shapes you intend to use.

The [backend overview](../README.md) explains selection, caching, and limitations
shared with CPU.

## Prerequisites

- Linux x86-64 and Python 3.12 for the XPU dependency group.
- Git and [uv](https://docs.astral.sh/uv/getting-started/installation/).
- GCC, Clang, and LLD. Clang/LLD build LLVM/MLIR; GCC builds cuTile's extensions,
  because `_cext` does not build with Clang on Linux.
- Intel GPU software: Level Zero development headers and loader, the offline
  compiler libraries (`libocloc`) and `ocloc_api.h` development header.
- An Intel GPU and its driver to run kernels; a device is not required to build.

No NVIDIA GPU, CUDA driver, or system CUDA Toolkit is required. The build group
supplies CUDA headers for cuTile's `_cext` extension, CMake, Ninja, and Python
build packages.

## Build and install

From the repository root:

```bash
git submodule update --init --recursive
uv venv --python 3.12
source .venv/bin/activate
. <path-to>/intel/setvars.sh --force
. <path-to>/intel-gpu/latest/intel_gpu_vars.sh
CUTILE_BUILD_VARIANT=xpu uv sync --group xpu --no-build-isolation-package cuda-tile
```

Replace the Intel script paths with your installation paths. Initialize the
Intel environment once in each fresh shell. Without these scripts, set
`LEVEL_ZERO_DIR` to the Level Zero installation root and make the Intel GPU
tools, including `ocloc`, available on `PATH`.

The first build also builds LLVM/MLIR and TileIRToMLIR. Later runs reuse those
builds. Keep `--no-build-isolation-package cuda-tile`: the persistent build trees
refer to tools in this environment.

## Run a sample

On a node with an Intel GPU, activate the environment and initialize the Intel
software in a fresh shell:

```bash
source .venv/bin/activate
. <path-to>/intel/setvars.sh --force
. <path-to>/intel-gpu/latest/intel_gpu_vars.sh
python -u samples/xpu/MatMul-xpu.py --correctness-check
```

Run from the repository root. Intel's offline compiler libraries, Level Zero
loader, and GPU driver are external runtime dependencies. No MLIR `PYTHONPATH`
setup is needed. More examples are in [samples/xpu](../../samples/xpu).

## Use in a program

Select the backend and pass XPU tensors to your kernel:

```python
import cuda.tile as ct
import torch

ct.set_backend("xpu")
ct.launch(torch.xpu.current_stream(), grid, kernel, xpu_args)
```

Here, `kernel` is your decorated cuTile kernel, `grid` is its block grid, and
`xpu_args` contains its arguments.

When adapting a CUDA program:

1. Allocate tensors on `xpu:0` and replace `tensor.is_cuda` checks with
   `tensor.is_xpu`.
2. Replace CUDA streams with `torch.xpu.current_stream()`.
3. Validate tile-alignment requirements. Reshape or pad inputs when necessary.
4. Avoid CAS spin loops and lock-based atomic accumulation.

Launches wait for the caller's stream and complete before returning. Only
Level Zero device 0 (`xpu:0`) is supported; select the visible device with
`ZE_AFFINITY_MASK` before starting Python. Arrays on other devices are rejected.
`ct.set_backend(None)` explicitly restores the CUDA launch path.

### Configure the launch

Use `ct.backend_options({...})` around launches:

| Option | Default | Effect |
| --- | --- | --- |
| `block_threads` | `(512, 1, 1)` | Launch block dimensions. |
| `wg_m`, `wg_n`, `sg_m`, `sg_n` | Unset | Derive the launch block from tile sizes. Supply all four instead of `block_threads`; work-group sizes must be positive multiples of subgroup sizes. |
| `assume_in_bounds` | `False` | Skip tile load/store bounds checks; requires all accesses to be in bounds. |
| `xegpu_op_level` | `"workgroup"` | Initial XeGPU level: `"workgroup"`, `"subgroup"`, or `"lane"`. |
| `large_register_file` | `True` | Enable the large register file. |

Set `CUTILE_XPU_ARCH` before importing the backend to select `b70` (default),
`b50`, or `pvc`; choose the target that matches your GPU.

### Diagnose a problem

If a kernel fails, first check its input alignment and tile sizes against the
corresponding sample. Set `CUTILE_XPU_DUMP_MLIR` to inspect the generated MLIR.
Driver and launch errors raise `RuntimeError`, with module build logs when
available. A successful launch is not a substitute for checking the output.

## Run tests

From the repository root, install the test runner and plugins, then run the
backend tests:

```bash
uv pip install pytest pytest-benchmark pytest-env
python -m pytest backends
```

Tests requiring a different native build variant are skipped.

For the compiler pipeline, memref ABI, and Level Zero resource ownership, see the
[XPU entry module](../../src/cuda/tile/_backends/xpu/__init__.py).
