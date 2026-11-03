// SPDX-FileCopyrightText: Copyright (c) <2026> Intel Corporation. All rights
// reserved.
//
// SPDX-License-Identifier: Apache-2.0

#include <level_zero/ze_api.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <array>
#include <cstdio>
#include <cstring>
#include <limits>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;

namespace {

void check(ze_result_t status, const char *operation,
           const std::string &detail = {}) {
  if (status == ZE_RESULT_SUCCESS)
    return;
  std::ostringstream message;
  message << "XPU backend: " << operation << " failed (Level Zero status 0x"
          << std::hex << static_cast<uint32_t>(status) << ")";
  if (!detail.empty())
    message << '\n' << detail;
  throw std::runtime_error(message.str());
}

#define ZE_CHECK(call) check(call, #call)

template <typename T, ze_result_t (*Destroy)(T)> class Handle {
public:
  Handle() = default;
  Handle(const Handle &) = delete;
  Handle &operator=(const Handle &) = delete;
  ~Handle() {
    if (value) {
      ze_result_t result = Destroy(value);
      if (result != ZE_RESULT_SUCCESS)
        std::fprintf(stderr,
                     "XPU backend: resource cleanup failed (Level Zero "
                     "status 0x%x)\n",
                     static_cast<uint32_t>(result));
    }
  }
  T get() const { return value; }
  T *put() { return &value; }
  void swap(Handle &other) { std::swap(value, other.value); }

private:
  T value = nullptr;
};

struct Parameter {
  uint64_t value;
  size_t size;
};

std::vector<Parameter> marshalArguments(nb::iterable arguments) {
  std::vector<Parameter> params;
  for (nb::handle argument : arguments) {
    if (nb::isinstance<nb::bytes>(argument)) {
      nb::bytes scalar(argument);
      if (scalar.size() != 1 && scalar.size() != 2 && scalar.size() != 4 &&
          scalar.size() != 8)
        throw nb::value_error(
            "scalar arguments must contain 1, 2, 4, or 8 bytes");
      uint64_t value = 0;
      std::memcpy(&value, scalar.data(), scalar.size());
      params.push_back({value, scalar.size()});
      continue;
    }
    if (!nb::isinstance<nb::tuple>(argument))
      throw nb::type_error("kernel arguments must be bytes scalars or (data, "
                           "shape, strides) memref tuples");
    nb::tuple memref(argument);
    if (memref.size() != 3)
      throw nb::value_error("memref arguments must be (data, shape, strides)");
    const uintptr_t data = nb::cast<uintptr_t>(memref[0]);
    const auto shape = nb::cast<std::vector<int64_t>>(memref[1]);
    const auto strides = nb::cast<std::vector<int64_t>>(memref[2]);
    if (shape.size() != strides.size())
      throw nb::value_error("memref shape and strides must have equal length");
    params.push_back({data, sizeof(uintptr_t)});
    params.push_back({data, sizeof(uintptr_t)});
    params.push_back({0, sizeof(int64_t)});
    for (int64_t size : shape) {
      if (size < 0)
        throw nb::value_error("memref sizes must be non-negative");
      params.push_back({static_cast<uint64_t>(size), sizeof(int64_t)});
    }
    for (int64_t stride : strides) {
      if (stride < 0)
        throw nb::value_error("memref strides must be non-negative");
      params.push_back({static_cast<uint64_t>(stride), sizeof(int64_t)});
    }
  }
  return params;
}

class XpuKernel {
public:
  XpuKernel(nb::bytes moduleBlob, std::string kernelName)
      : blob(moduleBlob.c_str(), moduleBlob.size()),
        name(std::move(kernelName)) {
    if (blob.empty())
      throw nb::value_error("module_blob must not be empty");
    if (name.empty())
      throw nb::value_error("kernel_name must not be empty");
  }

  void launch(nb::iterable arguments, const std::array<int64_t, 3> &grid,
              const std::array<int64_t, 3> &block) {
    auto params = marshalArguments(arguments);
    for (int64_t value : grid)
      if (value < 0 || value > std::numeric_limits<uint32_t>::max())
        throw nb::value_error(
            "XPU grid dimensions must be nonnegative uint32 values");
    for (int64_t value : block)
      if (value <= 0 || value > std::numeric_limits<uint32_t>::max())
        throw nb::value_error(
            "XPU block dimensions must be positive uint32 values");
    if (grid[0] == 0 || grid[1] == 0 || grid[2] == 0)
      return;

    nb::gil_scoped_release release;
    // Kernel arguments are mutable Level Zero state, so serialize each kernel.
    std::lock_guard<std::mutex> lock(mutex);
    ensureLoaded();
    if (params.size() != parameterCount)
      throw nb::value_error(
          "XPU argument count does not match the compiled kernel");
    ZE_CHECK(
        zeKernelSetGroupSize(function.get(), block[0], block[1], block[2]));
    for (uint32_t index = 0; index < params.size(); ++index)
      ZE_CHECK(zeKernelSetArgumentValue(
          function.get(), index, params[index].size, &params[index].value));
    ze_group_count_t groups{static_cast<uint32_t>(grid[0]),
                            static_cast<uint32_t>(grid[1]),
                            static_cast<uint32_t>(grid[2])};
    ZE_CHECK(zeCommandListAppendLaunchKernel(commands.get(), function.get(),
                                             &groups, nullptr, 0, nullptr));
    ZE_CHECK(zeCommandListHostSynchronize(commands.get(), UINT64_MAX));
  }

private:
  void ensureLoaded() {
    if (function.get())
      return;
    ZE_CHECK(zeInit(ZE_INIT_FLAG_GPU_ONLY));
    uint32_t count = 0;
    ZE_CHECK(zeDriverGet(&count, nullptr));
    if (count == 0)
      throw std::runtime_error(
          "XPU backend: no Level Zero GPU driver is available");
    std::vector<ze_driver_handle_t> drivers(count);
    ZE_CHECK(zeDriverGet(&count, drivers.data()));
    ze_driver_handle_t driver = drivers.front();
    // Count-only Level Zero queries require a fresh output count.
    count = 0;
    ZE_CHECK(zeDeviceGet(driver, &count, nullptr));
    if (count == 0)
      throw std::runtime_error(
          "XPU backend: no Level Zero GPU device is available");
    std::vector<ze_device_handle_t> devices(count);
    ZE_CHECK(zeDeviceGet(driver, &count, devices.data()));
    ze_device_handle_t device = devices.front();

    Handle<ze_context_handle_t, zeContextDestroy> newContext;
    ze_context_desc_t contextDesc{};
    contextDesc.stype = ZE_STRUCTURE_TYPE_CONTEXT_DESC;
    ZE_CHECK(zeContextCreate(driver, &contextDesc, newContext.put()));
    Handle<ze_module_handle_t, zeModuleDestroy> newModule;
    Handle<ze_module_build_log_handle_t, zeModuleBuildLogDestroy> buildLog;
    ze_module_desc_t moduleDesc{};
    moduleDesc.stype = ZE_STRUCTURE_TYPE_MODULE_DESC;
    moduleDesc.format = ZE_MODULE_FORMAT_NATIVE;
    moduleDesc.inputSize = blob.size();
    moduleDesc.pInputModule = reinterpret_cast<const uint8_t *>(blob.data());
    ze_result_t result = zeModuleCreate(newContext.get(), device, &moduleDesc,
                                        newModule.put(), buildLog.put());
    if (result != ZE_RESULT_SUCCESS) {
      std::string detail;
      if (buildLog.get()) {
        size_t size = 0;
        ZE_CHECK(zeModuleBuildLogGetString(buildLog.get(), &size, nullptr));
        detail.resize(size);
        if (size)
          ZE_CHECK(
              zeModuleBuildLogGetString(buildLog.get(), &size, detail.data()));
      }
      check(result, "zeModuleCreate", detail);
    }
    Handle<ze_kernel_handle_t, zeKernelDestroy> newFunction;
    ze_kernel_desc_t kernelDesc{};
    kernelDesc.stype = ZE_STRUCTURE_TYPE_KERNEL_DESC;
    kernelDesc.pKernelName = name.c_str();
    ZE_CHECK(zeKernelCreate(newModule.get(), &kernelDesc, newFunction.put()));
    ze_kernel_properties_t properties{};
    properties.stype = ZE_STRUCTURE_TYPE_KERNEL_PROPERTIES;
    ZE_CHECK(zeKernelGetProperties(newFunction.get(), &properties));

    count = 0;
    ZE_CHECK(zeDeviceGetCommandQueueGroupProperties(device, &count, nullptr));
    std::vector<ze_command_queue_group_properties_t> queues(count);
    for (auto &queue : queues)
      queue.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES;
    ZE_CHECK(
        zeDeviceGetCommandQueueGroupProperties(device, &count, queues.data()));
    uint32_t ordinal = 0;
    while (ordinal < count && !(queues[ordinal].flags &
                                ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE))
      ++ordinal;
    if (ordinal == count)
      throw std::runtime_error(
          "XPU backend: no compute command queue is available");
    Handle<ze_command_list_handle_t, zeCommandListDestroy> newCommands;
    ze_command_queue_desc_t queueDesc{};
    queueDesc.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
    queueDesc.ordinal = ordinal;
    queueDesc.mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS;
    queueDesc.priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL;
    ZE_CHECK(zeCommandListCreateImmediate(newContext.get(), device, &queueDesc,
                                          newCommands.put()));
    context.swap(newContext);
    module.swap(newModule);
    function.swap(newFunction);
    commands.swap(newCommands);
    parameterCount = properties.numKernelArgs;
  }

  std::string blob;
  std::string name;
  std::mutex mutex;
  uint32_t parameterCount = 0;
  Handle<ze_context_handle_t, zeContextDestroy> context;
  Handle<ze_module_handle_t, zeModuleDestroy> module;
  Handle<ze_kernel_handle_t, zeKernelDestroy> function;
  Handle<ze_command_list_handle_t, zeCommandListDestroy> commands;
};

} // namespace

NB_MODULE(_level_zero, module) {
  nb::class_<XpuKernel>(module, "Kernel")
      .def(nb::init<nb::bytes, std::string>(), "module_blob"_a, "kernel_name"_a)
      .def("launch", &XpuKernel::launch, "arguments"_a, "grid_size"_a,
           "block_size"_a);
  module.def("_kernel_parameter_count", [](nb::iterable arguments) {
    return marshalArguments(arguments).size();
  });
}
