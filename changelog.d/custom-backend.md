<!--- SPDX-FileCopyrightText: Copyright (c) <2026> Intel Corporation. All rights reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0 -->

- Added pluggable non-CUDA backends: ``cuda.tile.set_backend("cpu")`` or
  ``set_backend("xpu")`` (or any backend module) routes ``cuda.tile.launch`` to
  the backend; ``clear_backend()`` restores CUDA.

## Custom backend

cuTile compiles a kernel to TileIR bytecode; a backend turns the bytecode into
a binary and launches it. Signature derivation (same rules as the CUDA
launcher), in-memory caching per kernel and argument layout, and the on-disk
cache are shared by all backends. A backend is a module providing:

| Attribute | Purpose |
| --- | --- |
| ``sm_arch``, ``bytecode_version`` | Compile target and TileIR version; no CUDA device or toolkit is probed. |
| ``normalize_options(options) -> Hashable`` | Validate the options of a ``compile_options`` block into a hashable value. |
| ``compiler_identity(options) -> str \| None`` | Identify everything besides the bytecode that affects ``compile``; ``None`` disables the disk cache only. |
| ``compile(bytecode, signature, options) -> bytes`` | Produce the backend binary. |
| ``load(binary, signature, options)`` | Prepare a binary for launching; called once per binary. |
| ``launch(loaded, stream, grid, args)`` | Run ``load``'s result synchronously. |
| ``synchronize(stream)`` | Wait for work queued on ``stream``. |

Options are scoped with ``cuda.tile._backend.compile_options(options)`` (also
available as ``cpu.compile_options`` / ``xpu.compile_options``); the options in
effect when a kernel is compiled are part of its cache key and are passed to
``compile`` and ``load``.

``cuda.tile._backend.compile_for_launch(kernel, args)`` returns the compiled
kernel for concrete arguments and ``launch_compiled(stream, grid, compiled,
args)`` launches it without dispatch lookups, e.g. for benchmarking.
