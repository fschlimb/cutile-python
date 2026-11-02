# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
from distutils import file_util
from setuptools import setup
from setuptools.command.build_ext import build_ext
from setuptools.extension import Extension
import os
import shlex
import sys
import tempfile


project_root = os.path.dirname(os.path.realpath(__file__))
is_windows = sys.platform == "win32"

# Every build variant builds the _cext extension. The non-CUDA variants (see
# backends/README.md) also build their backend from backends/ and stage it, with
# its runtime files, into cuda/tile/_backends with `cmake --install`.
_backend_extensions = {
    "cpu": "cuda.tile._backends.cpu._cpu",
    "xpu": "cuda.tile._backends.xpu._level_zero",
}
build_variant = os.getenv("CUTILE_BUILD_VARIANT", "cuda")
if build_variant != "cuda" and build_variant not in _backend_extensions:
    raise RuntimeError("CUTILE_BUILD_VARIANT must be one of: "
                       f"{', '.join(['cuda', *_backend_extensions])}")


def _backend_cmake_args():
    """CMake arguments of the non-CUDA variants, including those from CMAKE_ARGS."""
    args = shlex.split(os.getenv("CMAKE_ARGS", ""))
    if "CUDAToolkit_ROOT" in os.environ:
        return args
    # Without a CUDA Toolkit, take the headers _cext needs from the
    # cuda-toolkit[cudart,nvvm] wheels.
    try:
        import nvidia
    except ImportError:
        return args
    for path in nvidia.__path__:
        root = os.path.join(path, "cu13")
        if os.path.isfile(os.path.join(root, "include", "cuda.h")):
            return [f"-DCUDAToolkit_ROOT={root}", *args]
    return args


class BuildExtWithCmake(build_ext):
    user_options = build_ext.user_options + [
        ('disable-internal', None, 'Disable building internal extension'),
        ('minimal-internal', None, 'Enable building minimal internal extension'),
        ('enable-dev-features', None, 'Enable development-only features'),
        ('custom-static-libnvvm=', None, 'Include a custom NVVM built from a static library'),
        ('custom-libdevice=', None, 'Include a custom libdevice object'),
        ('custom-ptxas=', None, 'Include a custom ptxas binary'),
    ]

    def initialize_options(self):
        super().initialize_options()
        self.disable_internal = False
        self.minimal_internal = False
        self.enable_dev_features = False
        self.custom_static_libnvvm = None
        self.custom_libdevice = None
        self.custom_ptxas = None

    def finalize_options(self):
        super().finalize_options()

    def _make(self, build_dir: str, build_type: str, parallel: int):
        if is_windows:
            self.spawn(["msbuild", f"{build_dir}/cuda-tile-python.sln",
                        f"-maxcpucount:{parallel}",
                        f"/p:Configuration={build_type}",
                        "/t:_cext"])
        else:
            self.spawn(["cmake", "--build", build_dir, "--parallel", str(parallel)])
        # TODO: ideally, we should "make install" the library somewhere, so that CMake removes
        #   any build RPATHs etc. But I'll leave that for another day.

    def _cmake(self, build_dir: str, build_type: str, dlpack_path: str, xla_path: str):
        cmake_cmd = ["cmake", "-B", build_dir, project_root,
                     f"-DDLPACK_PATH={dlpack_path}",
                     f"-DXLA_PATH={xla_path}",
                     f"-DCMAKE_BUILD_TYPE={build_type}",
                     f"-DPython_EXECUTABLE={sys.executable}",
                     "-DCMAKE_POLICY_VERSION_MINIMUM=3.5",
                     f"-DCUTILE_BUILD_VARIANT={build_variant}"]
        if build_variant != "cuda":
            cmake_cmd.extend(_backend_cmake_args())
        if self.disable_internal:
            cmake_cmd.append("-DDISABLE_INTERNAL=1")
        if self.minimal_internal:
            cmake_cmd.append("-DMINIMAL_INTERNAL=1")
        if self.enable_dev_features:
            cmake_cmd.append("-DENABLE_DEV_FEATURES=1")
        if self.custom_static_libnvvm is not None:
            cmake_cmd.append(f"-DCUSTOM_STATIC_LIBNVVM={self.custom_static_libnvvm}")
        if self.custom_libdevice is not None:
            cmake_cmd.append(f"-DCUSTOM_LIBDEVICE={self.custom_libdevice}")
        if self.custom_ptxas is not None:
            cmake_cmd.append(f"-DCUSTOM_PTXAS={self.custom_ptxas}")
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
        parallel = 1 if self.parallel is None else self.parallel
        self._cmake(build_dir, build_type, dlpack_path, xla_path)
        self._make(build_dir, build_type, parallel)

        if build_variant != "cuda":
            prefix = os.path.join(project_root, "src") if self.editable_mode else self.build_lib
            self.spawn(["cmake", "--install", build_dir, "--prefix", os.path.abspath(prefix),
                        "--component", "cuTileBackend"])

        for ext in self.extensions:
            if ext.name in _backend_extensions.values():
                continue  # installed by `cmake --install` above
            src_dir = _get_csrc_dir(ext.name)
            ext_name = _get_build_lib_filename(ext.name)
            if is_windows:
                ext_build_path = os.path.join(build_dir, src_dir, build_type, ext_name)
            else:
                ext_build_path = os.path.join(build_dir, src_dir, ext_name)
            ext_path = self.get_ext_fullpath(ext.name)
            if self.editable_mode:
                _link_editable_extension(ext_build_path, ext_path)
            else:
                file_util.copy_file(ext_build_path, ext_path, update=1)


def _link_editable_extension(src: str, dst: str):
    src = os.path.abspath(src)
    if not os.path.isfile(src):
        raise FileNotFoundError(src)
    if os.path.islink(dst) and os.readlink(dst) == src:
        return
    with tempfile.TemporaryDirectory(dir=os.path.dirname(dst)) as temp_dir:
        link = os.path.join(temp_dir, os.path.basename(dst))
        os.symlink(src, link)
        os.replace(link, dst)


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
    options=({"build": {"build_base": "build/python-" + build_variant}}
             if build_variant != "cuda" else {}),
    ext_modules=[
        Extension("cuda.tile._cext", []),
        *([Extension(_backend_extensions[build_variant], [])]
          if build_variant != "cuda" else []),
    ],
    cmdclass=dict(
        build_ext=BuildExtWithCmake,
    )
)
