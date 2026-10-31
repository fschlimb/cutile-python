# SPDX-FileCopyrightText: Copyright (c) <2026> Intel Corporation. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""The native compiler shared by the built-in backends."""

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
    module = importlib.import_module("cuda.tile._backend._compiler")
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
