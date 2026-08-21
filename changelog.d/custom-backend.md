<!--- SPDX-FileCopyrightText: Copyright (c) <2026> Intel Corporation. All rights reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0 -->

- Added pluggable backend hooks via ``cuda.tile.set_backend(module="my_backend")`` or positional arguments to retarget compilation (TileIR bytecode → backend binary) and launch to a custom, non-CUDA backend, plus the ``compile_kernel`` / ``clear_backend`` / ``backend_active`` helpers.

## Custom backend

By default a kernel goes TileIR bytecode → CUDA cubin → ``cuLaunchKernelEx``.
Two hooks let you retarget the last two stages to another architecture; source
parsing, signature construction and caching are unchanged.

- ``compile_fn`` replaces the cubin step in ``kernel._compile``: it gets the
  TileIR bytecode for one signature and returns a backend binary.
- ``launch_fn`` replaces the CUDA launch path: it gets the runtime args and
  grid, compiles (via ``compile_kernel``, which calls ``compile_fn``), and runs.

Leave either as ``None`` to keep that stage on the default CUDA path.

### API

| Symbol | Notes |
| --- | --- |
| ``set_backend(module=None, *, compile_fn=None, launch_fn=None, sm_arch=None)`` | Register hooks. ``module`` can be a string name of a module to import hooks from. ``sm_arch`` (e.g. ``"sm_120"``) overrides device probing, so no NVIDIA GPU is needed. |
| ``clear_backend()`` | Restore the default CUDA path. |
| ``backend_active()`` | ``True`` while a custom ``launch_fn`` is registered. |
| ``compile_kernel(kernel, signature, context=...)`` | Compile one signature → ``(binary, symbol)``; routes through ``compile_fn`` if set. |

### Hooks

```python
def compile_fn(tileir_bytecode: bytes, *,
               symbol: str,          # mangled entry-point name
               sm_arch: str,         # target arch / sm_arch override
               signature) -> bytes:  # cuda.tile.compilation.KernelSignature

def launch_fn(stream, grid, kernel, args) -> None:
```

``compile_fn`` returns the binary as ``bytes``, opaque to cuTile: it is never
loaded or validated, so it can be anything your runtime understands (object
file, serialized module, even a path encoded as bytes). It is forwarded
verbatim: ``compile_kernel`` hands it back as ``(binary, symbol)``. The
``symbol`` is tracked separately, so it need not be embedded in the binary.

Note: if you set ``compile_fn`` but no ``launch_fn``, the bytes must be a real
cubin, since the built-in launcher will ``cuLibraryLoadData`` them. A foreign
backend always needs both hooks.

### Usage

```python
import cuda.tile as ct
from cuda.tile.compilation import KernelSignature, CallingConvention

def my_compile(tileir_bytecode, *, symbol, sm_arch, signature):
    return my_toolchain.compile(tileir_bytecode)        # -> bytes

def my_launch(stream, grid, kernel, args):
    sig = KernelSignature.from_kernel_args(
        kernel, args, CallingConvention.cutile_python_v1())
    binary, symbol = ct.compile_kernel(kernel, sig)     # -> my_compile
    my_runtime.run(binary, symbol, grid, stream, args)

ct.set_backend(compile_fn=my_compile, launch_fn=my_launch, sm_arch="sm_120")
# Alternatively, if hooks are in my_backend.py:
# ct.set_backend("my_backend", sm_arch="sm_120")
# ct.launch(...) now dispatches to my_launch
ct.clear_backend()
```

``KernelSignature.from_kernel_args`` is the launch-time way to specialize a
kernel for concrete args; it runs the same argument inspection as the default
launcher (torch / DLPack / ``__cuda_array_interface__`` / scalars / lists), so
identical shapes/dtypes/strides hit the same cached binary. As with the default
JIT, this can bake in incidental assumptions (e.g. 16-byte base alignment) from
the example args.

Without concrete arrays (AOT), or to relax such assumptions, build the
``KernelSignature`` directly from ``ArrayConstraint`` / ``ScalarConstraint`` /
``ConstantConstraint``; ``cuda/tile/jax/_jax.py`` (``_array_constraint``) is a
worked reference. ``compile_fn`` also backs ``compilation.export_kernel``, so
one hook covers both JIT and AOT.

Hooks can instead be monkey-patched onto ``cuda.tile._execution.launch`` and
``kernel._compile``, but ``set_backend`` avoids import-ordering pitfalls.
