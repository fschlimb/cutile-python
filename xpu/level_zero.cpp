// SPDX-FileCopyrightText: Copyright (c) <2026> Intel Corporation. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * Nanobind bridge from the Python XPU backend to MLIR's Level Zero runtime.
 *
 * The binding accepts a compiled device module plus scalar bytes and memref
 * metadata prepared by xpu.py, flattens them into MLIR's kernel ABI, and runs
 * the kernel synchronously. Runtime streams and modules are owned entirely by
 * this call and are released before control returns to Python.
 */

#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <dlfcn.h>
#include <mutex>
#include <string>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;

// MLIR's runtime wrappers (CUDA/ROCm/SYCL/Vulkan/Level Zero) all export
// identically named `mgpu*` C functions, so directly linking against
// `mlir_levelzero_runtime` risks silently resolving to a same-named symbol
// from a different runtime wrapper loaded elsewhere in the process. Loading
// the exact library by name and resolving each symbol from that handle keeps
// this module talking to the Level Zero runtime.
struct StreamWrapper;

namespace {

template <typename T> T loadSymbol(void *library, const char *name) {
  void *symbol = dlsym(library, name);
  if (!symbol)
    throw std::runtime_error(std::string("missing Level Zero runtime symbol: ") +
                             name);
  return reinterpret_cast<T>(symbol);
}

// Resolved once per process and reused by every launch.
struct RuntimeApi {
  using ModuleLoad = void *(*)(const void *, size_t);
  using ModuleGetFunction = void *(*)(void *, const char *);
  using ModuleUnload = void (*)(void *);
  using StreamCreate = StreamWrapper *(*)();
  using StreamSynchronize = void (*)(StreamWrapper *);
  using StreamDestroy = void (*)(StreamWrapper *);
  using LaunchKernel = void (*)(void *, size_t, size_t, size_t, size_t, size_t,
                                size_t, int32_t, StreamWrapper *, void **,
                                void **, size_t);

  RuntimeApi(const RuntimeApi &) = delete;
  RuntimeApi &operator=(const RuntimeApi &) = delete;

  RuntimeApi() {
    library = dlopen(CUTILE_LEVEL_ZERO_RUNTIME_NAME, RTLD_NOW | RTLD_LOCAL);
    if (!library)
      throw std::runtime_error(std::string("unable to load Level Zero runtime: ") +
                               dlerror());
    moduleLoad = loadSymbol<ModuleLoad>(library, "mgpuModuleLoad");
    moduleGetFunction =
        loadSymbol<ModuleGetFunction>(library, "mgpuModuleGetFunction");
    moduleUnload = loadSymbol<ModuleUnload>(library, "mgpuModuleUnload");
    streamCreate = loadSymbol<StreamCreate>(library, "mgpuStreamCreate");
    streamSynchronize =
        loadSymbol<StreamSynchronize>(library, "mgpuStreamSynchronize");
    streamDestroy = loadSymbol<StreamDestroy>(library, "mgpuStreamDestroy");
    launchKernel = loadSymbol<LaunchKernel>(library, "mgpuLaunchKernel");
  }

  void *library;
  ModuleLoad moduleLoad;
  ModuleGetFunction moduleGetFunction;
  ModuleUnload moduleUnload;
  StreamCreate streamCreate;
  StreamSynchronize streamSynchronize;
  StreamDestroy streamDestroy;
  LaunchKernel launchKernel;
};

RuntimeApi &runtimeApi() {
  static RuntimeApi api;
  return api;
}

// Serializes calls into the runtime across concurrent launches.
std::mutex runtimeMutex;

// Give opaque runtime handles deterministic cleanup on every exception path.
// Non-copyable: duplicating a handle would destroy the same resource twice.
template <typename T> class Handle {
public:
  Handle(T value, void (*destroy)(T)) : value(value), destroy(destroy) {}
  Handle(const Handle &) = delete;
  Handle &operator=(const Handle &) = delete;
  ~Handle() {
    if (value)
      destroy(value);
  }
  T get() const { return value; }

private:
  T value;
  void (*destroy)(T);
};

struct KernelParams {
  // mgpuLaunchKernel receives pointers to argument values. `pointers` is only
  // built by finish(), once every value has its final address, so plain
  // contiguous storage is safe here.
  std::vector<uint64_t> values;
  std::vector<void *> pointers;

  void add(uint64_t value) { values.push_back(value); }

  void finish() {
    pointers.reserve(values.size());
    for (uint64_t &value : values)
      pointers.push_back(&value);
  }
};

void marshalArguments(nb::iterable arguments, KernelParams &params) {
  for (nb::handle argument : arguments) {
    // xpu.py has already converted signature-typed scalars to their ABI bytes.
    if (nb::isinstance<nb::bytes>(argument)) {
      nb::bytes scalar(argument);
      if (scalar.size() != 1 && scalar.size() != 2 && scalar.size() != 4 &&
          scalar.size() != 8)
        throw nb::value_error("scalar arguments must contain 1, 2, 4, or 8 bytes");
      uint64_t value = 0;
      std::memcpy(&value, scalar.data(), scalar.size());
      params.add(value);
      continue;
    }

    if (!nb::isinstance<nb::tuple>(argument))
      throw nb::type_error(
          "kernel arguments must be bytes scalars or (data, shape, strides) "
          "memref tuples");
    nb::tuple memref(argument);
    if (memref.size() != 3)
      throw nb::value_error("memref arguments must be (data, shape, strides)");
    const uintptr_t data = nb::cast<uintptr_t>(memref[0]);
    const std::vector<int64_t> shape =
        nb::cast<std::vector<int64_t>>(memref[1]);
    const std::vector<int64_t> strides =
        nb::cast<std::vector<int64_t>>(memref[2]);
    if (shape.size() != strides.size())
      throw nb::value_error("memref shape and strides must have equal length");

    // MLIR's non-bare memref ABI is
    // (allocated pointer, aligned pointer, offset, sizes..., strides...).
    params.add(data);
    params.add(data);
    params.add(0);
    for (int64_t size : shape) {
      if (size < 0)
        throw nb::value_error("memref sizes must be non-negative");
      params.add(static_cast<uint64_t>(size));
    }
    for (int64_t stride : strides)
      params.add(static_cast<uint64_t>(stride));
  }
  params.finish();
}

void launchLevelZeroModuleKernel(nb::bytes moduleBlob,
                                 const std::string &kernelName,
                                 nb::iterable arguments,
                                 const std::array<size_t, 3> &grid,
                                 const std::array<size_t, 3> &block) {
  if (moduleBlob.size() == 0)
    throw nb::value_error("module_blob must not be empty");
  if (kernelName.empty())
    throw nb::value_error("kernel_name must not be empty");
  KernelParams params;
  marshalArguments(arguments, params);
  RuntimeApi &api = runtimeApi();

  // Python objects are no longer accessed below. Release the GIL so other
  // Python threads keep running during the blocking device calls, and
  // serialize access to the runtime's global Level Zero context, whose
  // thread-safety under concurrent use is undocumented.
  nb::gil_scoped_release release;
  std::lock_guard<std::mutex> lock(runtimeMutex);

  Handle<void *> module(api.moduleLoad(moduleBlob.data(), moduleBlob.size()),
                        api.moduleUnload);
  if (!module.get())
    throw std::runtime_error("mgpuModuleLoad returned a null handle");

  void *kernel = api.moduleGetFunction(module.get(), kernelName.c_str());
  if (!kernel)
    throw std::runtime_error("mgpuModuleGetFunction returned a null handle");

  Handle<StreamWrapper *> stream(api.streamCreate(), api.streamDestroy);
  if (!stream.get())
    throw std::runtime_error("mgpuStreamCreate returned a null handle");

  api.launchKernel(kernel, grid[0], grid[1], grid[2], block[0], block[1],
                   block[2], 0, stream.get(), params.pointers.data(), nullptr,
                   params.pointers.size());
  // Arguments, stream, and module must remain alive until execution completes.
  api.streamSynchronize(stream.get());
}

} // namespace

NB_MODULE(_level_zero, module) {
  module.def("launch_level_zero_module_kernel", &launchLevelZeroModuleKernel,
             "module_blob"_a, "kernel_name"_a, "arguments"_a,
             "grid_size"_a, "block_size"_a);
  module.def("_kernel_parameter_count", [](nb::iterable arguments) {
    KernelParams params;
    marshalArguments(arguments, params);
    return params.pointers.size();
  });
}