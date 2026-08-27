# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
from setuptools import setup
from setuptools.command.build_ext import build_ext
from setuptools.extension import Extension
import os
import shlex
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

    def _python_executable(self) -> str:
        scripts_dir = "Scripts" if is_windows else "bin"
        python_name = "python.exe" if is_windows else "python"
        environments = [
            os.environ.get("VIRTUAL_ENV", ""),
            os.path.join(project_root, ".venv"),
        ]
        for environment in environments:
            candidate = os.path.join(os.path.realpath(environment), scripts_dir,
                                     python_name)
            if os.path.isfile(candidate):
                return candidate
        return sys.executable

    def _configure(self, build_dir: str, build_type: str):
        cmake_cmd = ["cmake", "-S", project_root, "-B", build_dir,
                     f"-DDLPACK_PATH={os.getenv('CUDA_TILE_CMAKE_DLPACK_PATH', '')}",
                     f"-DXLA_PATH={os.getenv('CUDA_TILE_CMAKE_XLA_PATH', '')}",
                     f"-DCMAKE_BUILD_TYPE={build_type}",
                     f"-DPython_EXECUTABLE={self._python_executable()}"]
        if not is_windows and not os.path.isfile(
                os.path.join(build_dir, "CMakeCache.txt")):
            cmake_cmd.extend(["-G", "Ninja"])
        cmake_cmd.extend(shlex.split(os.environ.get("CMAKE_ARGS", "")))
        if self.disable_internal:
            cmake_cmd.append("-DDISABLE_INTERNAL=1")
        if self.enable_dev_features:
            cmake_cmd.append("-DENABLE_DEV_FEATURES=1")
        self.spawn(cmake_cmd)

    def run(self):
        build_dir = os.getenv("CUDA_TILE_CEXT_BUILD_DIR")
        if not build_dir:
            if self.editable_mode or self.inplace:
                build_dir = os.path.join(project_root, "build")
            else:
                build_dir = self.build_temp

        build_type = "Debug" if self.debug else "Release"
        parallel = (os.cpu_count() or 1) if self.parallel is None else self.parallel
        install_prefix = (os.path.join(project_root, "src")
                          if self.editable_mode or self.inplace
                          else os.path.abspath(self.build_lib))
        self._configure(build_dir, build_type)
        self.spawn(["cmake", "--build", build_dir,
                    "--parallel", str(parallel)])
        self.spawn(["cmake", "--install", build_dir,
                    "--prefix", install_prefix,
                    "--component", "Python"])


setup(
    ext_modules=[
        Extension("cuda.tile._cext", []),
        Extension("cuda.tile._level_zero", []),
    ],
    cmdclass=dict(
        build_ext=BuildExtWithCmake,
    )
)
