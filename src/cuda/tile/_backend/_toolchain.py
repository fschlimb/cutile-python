# SPDX-FileCopyrightText: Copyright (c) <2026> Intel Corporation. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import shutil
import subprocess
import sys

from cuda.tile._cext import default_tile_context
from cuda.tile._exception import TileCompilerTimeoutError


def file_fingerprint(path: str) -> str:
    """Identify a file by canonical path and stat data, so any rebuild changes it."""

    resolved = os.path.realpath(path)
    stat = os.stat(resolved)
    return (f"{resolved}:{stat.st_dev}:{stat.st_ino}:{stat.st_size}:"
            f"{stat.st_mtime_ns}")


def buildtree_dir() -> str:
    """Return the active or in-tree CMake build directory."""

    override = os.environ.get("CUDA_TILE_CEXT_BUILD_DIR")
    if override:
        return override
    root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), *[os.pardir] * 4))
    return os.path.join(root, "build")


def resolve_tool(backend: str, env_var: str, name: str) -> str:
    """Resolve a backend tool from an override, PATH, or the build tree."""

    override = os.environ.get(env_var)
    candidate = override or name
    resolved = shutil.which(candidate)
    if resolved is not None:
        return resolved
    if override and os.path.isfile(override) and os.access(override, os.X_OK):
        return override
    if not override:
        in_tree = os.path.join(buildtree_dir(), "llvm", "bin", name)
        if os.access(in_tree, os.X_OK):
            return in_tree
    source = f"{env_var}={override!r}" if override else f"$PATH (or set {env_var})"
    raise FileNotFoundError(
        f"{backend} backend: could not find '{name}' via {source}.")


def run_tool(backend: str, argv: list[str], input_bytes: bytes) -> bytes:
    """Run a backend tool with byte input and report captured failures.

    The tool is bounded by the compiler timeout (see ``ct.compiler_timeout``).
    """

    timeout = default_tile_context.config.compiler_timeout_sec
    try:
        process = subprocess.run(
            argv, input=input_bytes, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, check=False, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise TileCompilerTimeoutError(
            f"{backend} backend: {argv[0]} exceeded the compiler timeout of "
            f"{timeout}s.", " ".join(argv[1:]), None) from None
    except OSError as error:
        raise RuntimeError(
            f"{backend} backend: failed to execute {argv[0]}: {error}") from error
    if process.returncode != 0:
        stderr = process.stderr.decode(errors="replace").strip()
        raise RuntimeError(
            f"{backend} backend: tool failed (exit code {process.returncode}).\n"
            f"  command: {' '.join(argv)}\n"
            f"  stderr:  {stderr or '<empty>'}")
    if process.stderr:
        sys.stderr.write(process.stderr.decode(errors="replace"))
        sys.stderr.flush()
    return process.stdout
