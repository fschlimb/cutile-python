#!/usr/bin/env bash
set -euo pipefail

variant=${1:-}
case "$variant" in
  cuda|xpu|cpu) ;;
  *) echo "usage: $0 {cuda|xpu|cpu} [uv sync options]" >&2; exit 2 ;;
esac
shift

root=$(realpath "$(dirname "${BASH_SOURCE[0]}")/..")
cd "$root"
export CUTILE_BUILD_VARIANT=$variant

if [[ $variant == cpu ]]; then
  uv sync --only-group build --no-install-project --inexact
  python_executable=${VIRTUAL_ENV:-$root/.venv}/bin/python
  CUDA_TILE_CEXT_BUILD_DIR="$root/build" \
    "$python_executable" "$root/setup.py" build_ext --inplace
  llvm_source=$(sed -n 's/^CUTILE_LLVM_SOURCE_DIR:[^=]*=//p' "$root/build/CMakeCache.txt")
  llvm_build=$(sed -n 's/^CUTILE_LLVM_BINARY_DIR:[^=]*=//p' "$root/build/CMakeCache.txt")
  triton_revision=$("$python_executable" -c \
    'import pathlib, sys, tomllib; print(tomllib.loads((pathlib.Path(sys.argv[1]) / "pyproject.toml").read_text())["tool"]["uv"]["sources"]["triton"]["rev"])' \
    "$root")
  export LLVM_SYSPATH=$llvm_build
  export TRITON_BUILD_DIR=${CUTILE_TRITON_CPU_BINARY_DIR:-$root/build/triton-cpu-${triton_revision:0:10}}
  export TRITON_BUILD_PROTON=0
  export TRITON_APPEND_CMAKE_ARGS="-DTRITON_CPU_ONLY=ON -DTRITON_BUILD_PROTON=OFF -DTRITON_BUILD_UT=OFF -DCMAKE_DISABLE_FIND_PACKAGE_dnnl=ON -DLLVM_INCLUDE_DIRS=$llvm_source/llvm/include;$llvm_build/include -DLLVM_LIBRARY_DIR=$llvm_build/lib -DLLVM_DIR=$llvm_build/lib/cmake/llvm -DMLIR_DIR=$llvm_build/lib/cmake/mlir ${TRITON_APPEND_CMAKE_ARGS:-}"
  export CUDA_TILE_SKIP_CMAKE_BUILD=1
fi

exec uv sync --extra "$variant" "$@"