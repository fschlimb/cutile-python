# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
from distutils import file_util
from setuptools import setup
from setuptools.command.build_ext import build_ext
from setuptools.extension import Extension
import os
import shlex
import shutil
import sys


project_root = os.path.dirname(os.path.realpath(__file__))
is_windows = sys.platform == "win32"


class BuildExtWithCmake(build_ext):
    user_options = build_ext.user_options + [
        ('disable-internal', None, 'Disable building internal extension'),
        ('enable-dev-features', None, 'Enable development-only features'),
    ]

    def initialize_options(self):
        super().initialize_options()
        self.disable_internal = False
        self.enable_dev_features = False

    def finalize_options(self):
        super().finalize_options()

    def _cmake_executable(self, build_dir: str) -> str:
        cache_path = os.path.join(build_dir, "CMakeCache.txt")
        if os.path.isfile(cache_path):
            with open(cache_path, encoding="utf-8") as cache_file:
                for line in cache_file:
                    if line.startswith("CMAKE_COMMAND:INTERNAL="):
                        cached_cmake = line.partition("=")[2]
                        if os.path.isfile(cached_cmake):
                            return cached_cmake

        isolated_prefix = os.path.realpath(sys.prefix)
        for directory in os.get_exec_path():
            candidate = os.path.join(directory, "cmake")
            if (os.path.isfile(candidate) and os.access(candidate, os.X_OK)
                    and os.path.commonpath((os.path.realpath(candidate), isolated_prefix))
                    != isolated_prefix):
                return candidate
        return shutil.which("cmake") or "cmake"

    def _python_executable(self) -> str:
        if self.editable_mode:
            scripts_dir = "Scripts" if is_windows else "bin"
            python_name = "python.exe" if is_windows else "python"
            project_python = os.path.join(project_root, ".venv", scripts_dir,
                                          python_name)
            if os.path.isfile(project_python):
                return project_python
        return sys.executable

    @staticmethod
    def _nanobind_cmake_dir() -> str:
        import nanobind
        return nanobind.cmake_dir()

    @staticmethod
    def _cache_value(build_dir: str, key: str) -> str | None:
        cache_path = os.path.join(build_dir, "CMakeCache.txt")
        if not os.path.isfile(cache_path):
            return None
        with open(cache_path, encoding="utf-8") as cache_file:
            for line in cache_file:
                if line.startswith(f"{key}:"):
                    return line.partition("=")[2].strip()
        return None

    def _make(self, cmake_executable: str, build_dir: str, build_type: str,
              parallel: int):
        if is_windows:
            self.spawn(["msbuild", f"{build_dir}/cuda-tile-python.sln",
                        f"-maxcpucount:{parallel}",
                        f"/p:Configuration={build_type}",
                        "/t:_cext"])
        else:
            self.spawn([cmake_executable, "--build", build_dir,
                        "--parallel", str(parallel)])
        # TODO: ideally, we should "make install" the library somewhere, so that CMake removes
        #   any build RPATHs etc. But I'll leave that for another day.

    def _cmake(self, cmake_executable: str, build_dir: str, build_type: str,
               dlpack_path: str, xla_path: str):
        python_executable = self._python_executable()
        cmake_cmd = [cmake_executable, "-B", build_dir, project_root,
                     f"-DDLPACK_PATH={dlpack_path}",
                     f"-DXLA_PATH={xla_path}",
                     f"-DCMAKE_BUILD_TYPE={build_type}",
                 f"-DPython_EXECUTABLE={python_executable}",
                 f"-Dnanobind_DIR={self._nanobind_cmake_dir()}",
                     "-DCMAKE_POLICY_VERSION_MINIMUM=3.5"]
        generator = self._cache_value(build_dir, "CMAKE_GENERATOR")
        if not is_windows and generator and generator != "Ninja":
            raise RuntimeError(
                f"Existing build directory uses {generator!r}, not Ninja: {build_dir}. "
                "Remove its CMakeCache.txt and CMakeFiles before rebuilding.")
        if not is_windows and generator is None:
            cmake_cmd.extend(["-G", "Ninja"])
        cmake_cmd.extend(shlex.split(os.environ.get("CMAKE_ARGS", "")))
        if self.disable_internal:
            cmake_cmd.append("-DDISABLE_INTERNAL=1")
        if self.enable_dev_features:
            cmake_cmd.append("-DENABLE_DEV_FEATURES=1")
        self.spawn(cmake_cmd)

    def run(self):
        build_dir = os.getenv("CUDA_TILE_CEXT_BUILD_DIR")
        if build_dir is None or build_dir == "":
            if self.editable_mode:
                build_dir = os.path.join(project_root, "build")
            else:
                build_dir = self.build_temp

        build_type = "Debug" if self.debug else "Release"
        dlpack_path = os.getenv("CUDA_TILE_CMAKE_DLPACK_PATH", "")
        xla_path = os.getenv("CUDA_TILE_CMAKE_XLA_PATH", "")
        parallel = (os.cpu_count() or 1) if self.parallel is None else self.parallel
        cmake_executable = self._cmake_executable(build_dir)
        self._cmake(cmake_executable, build_dir, build_type, dlpack_path, xla_path)
        self._make(cmake_executable, build_dir, build_type, parallel)

        if self._cache_value(build_dir, "BUILD_CUDA_EXTENSION") != "ON":
            # No CUDA headers: cuda/tile/_cext.py is used instead.
            return

        for ext in self.extensions:
            src_dir = _get_csrc_dir(ext.name)
            ext_name = _get_build_lib_filename(ext.name)
            if is_windows:
                ext_build_path = os.path.join(build_dir, src_dir, build_type, ext_name)
            else:
                ext_build_path = os.path.join(build_dir, src_dir, ext_name)
            ext_path = self.get_ext_fullpath(ext.name)
            # Create a symlink to the build directory if in editable mode, otherwise copy
            link = "sym" if self.editable_mode else None
            file_util.copy_file(ext_build_path, ext_path, update=1, link=link,
                                dry_run=self.dry_run)


def _get_csrc_dir(ext_name: str):
    prefix = "cuda.tile._"
    assert ext_name.startswith(prefix)
    return ext_name[len(prefix):]


def _get_build_lib_filename(ext_name: str):
    name = ext_name.split(".")[-1]
    if is_windows:
        return f"{name}.dll"
    else:
        return f"lib{name}.so"


setup(
    ext_modules=[
        Extension("cuda.tile._cext", []),
    ],
    cmdclass=dict(
        build_ext=BuildExtWithCmake,
    )
)
