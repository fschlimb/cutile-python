# SPDX-FileCopyrightText: Copyright (c) <2026> Intel Corporation. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""The native compiler shared by the built-in backends.

The private nanobind extension links the pinned LLVM/MLIR and TileIRToMLIR
libraries. In-process compilation releases the GIL and uses a fresh MLIR context
per call. An explicit compiler timeout selects a Python worker running the same
extension so a timeout can terminate compilation safely. Without a timeout,
native compiler crashes affect the calling process.

The worker preserves known ValueError and RuntimeError failures without an
additional worker traceback. Installed kernels do not invoke LLVM tools or
require MLIR Python bindings. LLVM/MLIR assertions are disabled by default;
CMAKE_ARGS=-DCUTILE_LLVM_ENABLE_ASSERTIONS=ON enables them for debugging.

Build configuration
-------------------
The backends/tileir-to-mlir submodule pins TileIRToMLIR; its llvm-revision.txt
defines the required LLVM commit. USE_LLVM_REVISION may select a newer
descendant, but older or divergent commits are rejected. CMAKE_ARGS accepts
CUTILE_LLVM_SOURCE_DIR and CUTILE_LLVM_BINARY_DIR for custom build locations.
CUDAToolkit_ROOT selects local CUDA headers instead of the build dependencies.

To reuse a prebuilt compiler, pass
CMAKE_ARGS="-DCUTILE_TOOLCHAIN_DIR=/path/to/prefix" to the backend's uv sync
command. The prefix must match the required LLVM revision, active Python
version, and build variant. It must contain lib/_compiler<EXT_SUFFIX> and
share/cutile/llvm/LICENSE.TXT; the cuTileCompiler CMake install component
stages both. Consuming the prefix needs neither an LLVM checkout nor network
access.
"""

import hashlib
import importlib
import json
import os
import subprocess
import sys

from cuda.tile._cext import default_tile_context
from cuda.tile._exception import TileCompilerTimeoutError

# TileIR bytecode version read by the native compiler's TileIRToMLIR.
BYTECODE_VERSION = "13.3"


def file_fingerprint(path: str) -> str:
    """Identify a file by canonical path and stat data, so any rebuild changes it."""
    resolved = os.path.realpath(path)
    stat = os.stat(resolved)
    return (f"{resolved}:{stat.st_dev}:{stat.st_ino}:{stat.st_size}:"
            f"{stat.st_mtime_ns}")


def identity_digest(identity: dict) -> str:
    """Digest of a JSON-serializable compiler identity."""
    return hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def compiler(backend: str):
    module = importlib.import_module("cuda.tile._backends._compiler")
    if module.backend != backend:
        raise RuntimeError(
            f"The installed native compiler targets {module.backend}, not {backend}; "
            f"rebuild with CUTILE_BUILD_VARIANT={backend}.")
    return module


# Loads only the compiler extension, not the cuda.tile package.
_WORKER = """\
import importlib.machinery
import importlib.util
import json
import sys
path, method, arguments = json.loads(sys.argv[1])
loader = importlib.machinery.ExtensionFileLoader("_compiler", path)
_compiler = importlib.util.module_from_spec(
    importlib.util.spec_from_loader("_compiler", loader))
loader.exec_module(_compiler)
try:
    result = getattr(_compiler, method)(sys.stdin.buffer.read(), *arguments)
except (ValueError, RuntimeError) as error:
    sys.stderr.write(str(error))
    sys.exit(2 if isinstance(error, ValueError) else 3)
sys.stdout.buffer.write(result)
"""


def invoke(backend: str, method: str, bytecode: bytes, *arguments) -> bytes:
    module = compiler(backend)
    timeout = default_tile_context.config.compiler_timeout_sec
    if timeout is None:
        return getattr(module, method)(bytecode, *arguments)
    # An explicit hard timeout needs a killable worker, not a C++ thread.
    try:
        process = subprocess.run(
            [sys.executable, "-c", _WORKER,
             json.dumps((module.__file__, method, arguments))],
            input=bytecode, capture_output=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        raise TileCompilerTimeoutError(
            f"{backend} native compiler exceeded {timeout}s.",
            method, None) from None
    if process.returncode in (2, 3):
        exception = ValueError if process.returncode == 2 else RuntimeError
        raise exception(process.stderr.decode(errors="replace"))
    if process.returncode:
        raise RuntimeError(
            f"{backend} native compiler failed (exit code {process.returncode}).\n"
            + process.stderr.decode(errors="replace"))
    if process.stderr:
        sys.stderr.write(process.stderr.decode(errors="replace"))
        sys.stderr.flush()
    return process.stdout
