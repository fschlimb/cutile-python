from __future__ import annotations

import os
import shutil
import subprocess
import sys


def build_tree_dir() -> str:
    override = os.environ.get("CUDA_TILE_CEXT_BUILD_DIR")
    if override:
        return override
    root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), *[os.pardir] * 4))
    return os.path.join(root, "build")


def resolve_tool(backend: str, env_var: str, name: str) -> str:
    override = os.environ.get(env_var)
    candidate = override or name
    resolved = shutil.which(candidate)
    if resolved is not None:
        return resolved
    if override and os.path.isfile(override) and os.access(override, os.X_OK):
        return override
    if not override:
        in_tree = os.path.join(build_tree_dir(), "llvm", "bin", name)
        if os.access(in_tree, os.X_OK):
            return in_tree
    source = f"{env_var}={override!r}" if override else f"$PATH (or set {env_var})"
    raise FileNotFoundError(
        f"{backend} backend: could not find '{name}' via {source}.")


def run_tool(backend: str, argv: list[str], input_bytes: bytes) -> bytes:
    try:
        process = subprocess.run(
            argv, input=input_bytes, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, check=False)
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