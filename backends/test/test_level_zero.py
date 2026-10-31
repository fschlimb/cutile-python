# SPDX-FileCopyrightText: Copyright (c) <2026> Intel Corporation. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

_level_zero = pytest.importorskip(
    "cuda.tile._backend.xpu._level_zero", reason="needs the Level Zero extension")


def test_kernel_argument_expansion():
    arguments = [(1234, (2, 3), (3, 1)), (7).to_bytes(4, "little")]
    assert _level_zero._kernel_parameter_count(arguments) == 8


@pytest.mark.parametrize(
    "arguments, match",
    [
        ([(1234, (2,), (1, 1))], "equal length"),
        ([b"abc"], "1, 2, 4, or 8"),
        ([(1234, (-1,), (1,))], "non-negative"),
        ([(1234, (1,), (-1,))], "non-negative"),
    ],
)
def test_kernel_argument_validation(arguments, match):
    with pytest.raises(ValueError, match=match):
        _level_zero._kernel_parameter_count(arguments)


def test_launch_validation_does_not_initialize_runtime():
    with pytest.raises(ValueError, match="must not be empty"):
        _level_zero.Kernel(b"", "kernel")
    with pytest.raises(ValueError, match="must not be empty"):
        _level_zero.Kernel(b"module", "")

    with pytest.raises(TypeError):
        _level_zero.Kernel(b"module", "kernel").launch([], (1, 1), (1, 1, 1))


@pytest.mark.parametrize("grid, block", [
    ((-1, 1, 1), (1, 1, 1)),
    ((1 << 32, 1, 1), (1, 1, 1)),
    ((1, 1, 1), (0, 1, 1)),
    ((1, 1, 1), (1 << 32, 1, 1)),
])
def test_invalid_dimensions_do_not_initialize_runtime(grid, block):
    with pytest.raises(ValueError, match="dimensions"):
        _level_zero.Kernel(b"module", "kernel").launch([], grid, block)


def test_zero_grid_does_not_initialize_runtime():
    _level_zero.Kernel(b"module", "kernel").launch([], (0, 1, 1), (1, 1, 1))


def test_cpu_and_xpu_launcher_types_do_not_collide():
    cpu = pytest.importorskip("cuda.tile._backend.cpu._cpu")
    assert cpu.Kernel is not _level_zero.Kernel


_FAKE_LOADER = r"""
#include <level_zero/ze_api.h>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <thread>

struct Resource { std::atomic<unsigned> next{0}; bool mixed = false; };
static std::atomic<int> live{0};
static bool fails(const char *name) {
  const char *failure = std::getenv("CUTILE_TEST_ZE_FAIL");
  return failure && std::strcmp(name, failure) == 0;
}
template <typename T> static ze_result_t make(const char *name, T *out) {
  if (fails(name)) return ZE_RESULT_ERROR_UNKNOWN;
  *out = reinterpret_cast<T>(new Resource);
  ++live;
  return ZE_RESULT_SUCCESS;
}
template <typename T> static ze_result_t destroy(const char *name, T handle) {
  delete reinterpret_cast<Resource *>(handle);
  --live;
  return fails(name) ? ZE_RESULT_ERROR_UNKNOWN : ZE_RESULT_SUCCESS;
}
#define FAIL(name) if (fails(name)) return ZE_RESULT_ERROR_UNKNOWN
#define DESTROY(type, name) \
  ZE_APIEXPORT ze_result_t ZE_APICALL name(type handle) { return destroy(#name, handle); }
extern "C" {
int cutile_test_live_handles() { return live.load(); }
ZE_APIEXPORT ze_result_t ZE_APICALL zeInit(ze_init_flags_t) { FAIL("zeInit"); return ZE_RESULT_SUCCESS; }
ZE_APIEXPORT ze_result_t ZE_APICALL zeDriverGet(uint32_t *count, ze_driver_handle_t *out) {
  FAIL("zeDriverGet"); *count = 1;
  if (out) *out = reinterpret_cast<ze_driver_handle_t>(1);
  return ZE_RESULT_SUCCESS;
}
ZE_APIEXPORT ze_result_t ZE_APICALL zeDeviceGet(ze_driver_handle_t, uint32_t *count, ze_device_handle_t *out) {
  FAIL("zeDeviceGet"); *count = 1;
  if (out) *out = reinterpret_cast<ze_device_handle_t>(1);
  return ZE_RESULT_SUCCESS;
}
ZE_APIEXPORT ze_result_t ZE_APICALL zeContextCreate(ze_driver_handle_t, const ze_context_desc_t *, ze_context_handle_t *out) {
  return make("zeContextCreate", out);
}
DESTROY(ze_context_handle_t, zeContextDestroy)
ZE_APIEXPORT ze_result_t ZE_APICALL zeModuleCreate(ze_context_handle_t, ze_device_handle_t, const ze_module_desc_t *, ze_module_handle_t *out, ze_module_build_log_handle_t *log) {
  make("", log);
  return make("zeModuleCreate", out);
}
DESTROY(ze_module_handle_t, zeModuleDestroy)
DESTROY(ze_module_build_log_handle_t, zeModuleBuildLogDestroy)
ZE_APIEXPORT ze_result_t ZE_APICALL zeModuleBuildLogGetString(ze_module_build_log_handle_t, size_t *size, char *out) {
  FAIL("zeModuleBuildLogGetString");
  const char text[] = "test module build log";
  *size = sizeof(text);
  if (out) std::memcpy(out, text, sizeof(text));
  return ZE_RESULT_SUCCESS;
}
ZE_APIEXPORT ze_result_t ZE_APICALL zeKernelCreate(ze_module_handle_t, const ze_kernel_desc_t *desc, ze_kernel_handle_t *out) {
  ze_result_t result = make("zeKernelCreate", out);
  if (result == ZE_RESULT_SUCCESS)
    reinterpret_cast<Resource *>(*out)->mixed = std::strcmp(desc->pKernelName, "mixed") == 0;
  return result;
}
DESTROY(ze_kernel_handle_t, zeKernelDestroy)
ZE_APIEXPORT ze_result_t ZE_APICALL zeKernelGetProperties(ze_kernel_handle_t handle, ze_kernel_properties_t *out) {
  FAIL("zeKernelGetProperties");
  out->numKernelArgs = reinterpret_cast<Resource *>(handle)->mixed ? 6 : 2;
  return ZE_RESULT_SUCCESS;
}
ZE_APIEXPORT ze_result_t ZE_APICALL zeDeviceGetCommandQueueGroupProperties(ze_device_handle_t, uint32_t *count, ze_command_queue_group_properties_t *out) {
  FAIL("zeDeviceGetCommandQueueGroupProperties"); *count = 1;
  if (out) { out->flags = ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE; out->numQueues = 1; }
  return ZE_RESULT_SUCCESS;
}
ZE_APIEXPORT ze_result_t ZE_APICALL zeCommandListCreateImmediate(ze_context_handle_t, ze_device_handle_t, const ze_command_queue_desc_t *, ze_command_list_handle_t *out) {
  return make("zeCommandListCreateImmediate", out);
}
DESTROY(ze_command_list_handle_t, zeCommandListDestroy)
ZE_APIEXPORT ze_result_t ZE_APICALL zeKernelSetGroupSize(ze_kernel_handle_t handle, uint32_t, uint32_t, uint32_t) {
  FAIL("zeKernelSetGroupSize"); reinterpret_cast<Resource *>(handle)->next = 0;
  return ZE_RESULT_SUCCESS;
}
ZE_APIEXPORT ze_result_t ZE_APICALL zeKernelSetArgumentValue(ze_kernel_handle_t handle, uint32_t index, size_t size, const void *) {
  FAIL("zeKernelSetArgumentValue");
  auto *kernel = reinterpret_cast<Resource *>(handle);
  size_t expected = kernel->mixed ? (index < 5 ? 8 : 4) : (index == 0 ? 4 : 1);
  if (index != kernel->next++ || size != expected) return ZE_RESULT_ERROR_INVALID_ARGUMENT;
  if (index == 0) std::this_thread::sleep_for(std::chrono::milliseconds(1));
  return ZE_RESULT_SUCCESS;
}
ZE_APIEXPORT ze_result_t ZE_APICALL zeCommandListAppendLaunchKernel(ze_command_list_handle_t, ze_kernel_handle_t handle, const ze_group_count_t *, ze_event_handle_t, uint32_t, ze_event_handle_t *) {
  FAIL("zeCommandListAppendLaunchKernel");
  auto *kernel = reinterpret_cast<Resource *>(handle);
  return kernel->next == (kernel->mixed ? 6u : 2u) ? ZE_RESULT_SUCCESS : ZE_RESULT_ERROR_INVALID_ARGUMENT;
}
ZE_APIEXPORT ze_result_t ZE_APICALL zeCommandListHostSynchronize(ze_command_list_handle_t, uint64_t) {
  FAIL("zeCommandListHostSynchronize"); return ZE_RESULT_SUCCESS;
}
}
"""

_LOADER_PROBE = """
from concurrent.futures import ThreadPoolExecutor
import ctypes
import gc
import importlib.util
import json
import os
import struct
import sys

spec = importlib.util.spec_from_file_location("_level_zero", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
mode = sys.argv[3]
kernel = module.Kernel(b"binary", "mixed" if mode == "mixed" else "scalar")
args = ([(1234, (13,), (1,)), struct.pack("i", 7)] if mode == "mixed"
        else [struct.pack("i", 7), struct.pack("?", True)])
if mode == "count":
    args.pop()
result = {}
try:
    if mode == "retry":
        try:
            kernel.launch(args, (1, 1, 1), (32, 1, 1))
        except RuntimeError as error:
            result["first_error"] = str(error)
        os.environ["CUTILE_TEST_ZE_FAIL"] = ""
        kernel.launch(args, (1, 1, 1), (32, 1, 1))
    elif mode == "threads":
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(lambda _: kernel.launch(args, (1, 1, 1), (32, 1, 1)), range(16)))
    else:
        kernel.launch(args, (1, 1, 1), (32, 1, 1))
except (RuntimeError, ValueError) as error:
    result = {"error": type(error).__name__, "message": str(error)}
del kernel
gc.collect()
result["live"] = ctypes.CDLL(sys.argv[2]).cutile_test_live_handles()
print(json.dumps(result))
"""


@pytest.fixture(scope="module")
def fake_loader(tmp_path_factory):
    compiler = shutil.which(os.environ.get("CXX", "c++"))
    include = Path(os.environ.get("LEVEL_ZERO_DIR", "/usr")) / "include"
    if compiler is None or not (include / "level_zero" / "ze_api.h").is_file():
        pytest.skip("fake loader tests require C++ and Level Zero headers (LEVEL_ZERO_DIR)")
    library = tmp_path_factory.mktemp("level-zero-loader") / "fake_loader.so"
    subprocess.run([compiler, "-shared", "-fPIC", "-pthread", "-std=c++17",
                    f"-I{include}", "-x", "c++", "-", "-o", str(library)],
                   input=_FAKE_LOADER.encode(), check=True, timeout=60)

    def run(*, failure="", mode="scalar"):
        environment = {**os.environ, "CUTILE_TEST_ZE_FAIL": failure,
                       "LD_PRELOAD": f"{library} {os.environ.get('LD_PRELOAD', '')}".strip()}
        process = subprocess.run(
            [sys.executable, "-c", _LOADER_PROBE, _level_zero.__file__, str(library), mode],
            env=environment, capture_output=True, text=True, check=True, timeout=30)
        return json.loads(process.stdout), process.stderr

    return run


@pytest.mark.parametrize("operation", [
    "zeInit", "zeDriverGet", "zeDeviceGet", "zeContextCreate", "zeModuleCreate",
    "zeKernelCreate", "zeKernelGetProperties", "zeDeviceGetCommandQueueGroupProperties",
    "zeCommandListCreateImmediate", "zeKernelSetGroupSize", "zeKernelSetArgumentValue",
    "zeCommandListAppendLaunchKernel", "zeCommandListHostSynchronize",
])
def test_driver_failures_raise_and_release_resources(fake_loader, operation):
    result, _ = fake_loader(failure=operation)
    assert result["error"] == "RuntimeError"
    assert operation in result["message"]
    assert result["live"] == 0
    if operation == "zeModuleCreate":
        assert "test module build log" in result["message"]


@pytest.mark.parametrize("mode", ["scalar", "mixed", "threads"])
def test_launch_argument_sizes_cleanup_and_serialization(fake_loader, mode):
    result, stderr = fake_loader(mode=mode)
    assert result == {"live": 0}
    assert stderr == ""


def test_argument_count_mismatch_raises(fake_loader):
    result, _ = fake_loader(mode="count")
    assert result["error"] == "ValueError"
    assert "argument count" in result["message"]
    assert result["live"] == 0


@pytest.mark.parametrize("operation", [
    "zeModuleBuildLogDestroy", "zeCommandListDestroy", "zeKernelDestroy",
    "zeModuleDestroy", "zeContextDestroy",
])
def test_cleanup_errors_are_reported_without_aborting(fake_loader, operation):
    result, stderr = fake_loader(failure=operation)
    assert result == {"live": 0}
    assert "resource cleanup failed" in stderr


@pytest.mark.parametrize("operation", [
    "zeModuleCreate", "zeKernelSetArgumentValue", "zeCommandListHostSynchronize",
])
def test_failed_launch_can_retry_after_driver_recovers(fake_loader, operation):
    result, _ = fake_loader(failure=operation, mode="retry")
    assert operation in result["first_error"]
    assert "error" not in result
    assert result["live"] == 0
