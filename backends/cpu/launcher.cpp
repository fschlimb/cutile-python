// SPDX-FileCopyrightText: Copyright (c) <2026> Intel Corporation. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/string.h>

#include <array>
#include <cstdint>
#include <dlfcn.h>
#include <limits>
#include <memory>
#include <omp.h>
#include <stdexcept>
#include <string>

#if defined(__linux__) && defined(__x86_64__)
#include <asm/prctl.h>
#include <cpuid.h>
#include <sys/syscall.h>
#include <unistd.h>
#ifndef ARCH_REQ_XCOMP_PERM
#define ARCH_REQ_XCOMP_PERM 0x1023
#endif
#endif

namespace nb = nanobind;
using namespace nb::literals;

namespace {

// Whether the host has AMX tile, BF16 and INT8 and this process may use tile
// data. Linux grants that permission process-wide; without it, kernels use the
// non-AMX lowering.
bool enableAMX() {
#if defined(__linux__) && defined(__x86_64__)
  constexpr unsigned int amx = (1u << 22) | (1u << 24) | (1u << 25);
  unsigned int eax, ebx, ecx, edx;
  if (!__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx) || (edx & amx) != amx)
    return false;
  constexpr unsigned long tileData = 18;
  return syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, tileData) == 0;
#else
  return false;
#endif
}

struct LibraryCloser {
  void operator()(void *handle) const { dlclose(handle); }
};

class Kernel {
public:
  Kernel(const std::string &path, const std::string &symbol,
         size_t argumentBytes)
      : library(dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL)),
        argumentBytes(argumentBytes) {
    if (!library)
      throw std::runtime_error(
          std::string("CPU backend: unable to load kernel (TMPDIR must allow "
                      "executable files): ") +
          dlerror());
    dlerror();
    function = reinterpret_cast<Entry>(dlsym(library.get(), symbol.c_str()));
    const char *error = dlerror();
    if (error || !function)
      throw std::runtime_error("CPU backend: unable to find kernel '" + symbol +
                               "': " + (error ? error : "null entry point"));
    if (argumentBytes % 8 != 0)
      throw nb::value_error("CPU argument buffer size must be a multiple of 8");
  }

  void launch(nb::bytes arguments, const std::array<int64_t, 3> &grid,
              int64_t numThreads) const {
    if (arguments.size() != argumentBytes)
      throw nb::value_error(
          "CPU argument buffer size does not match the kernel");
    for (int64_t dimension : grid) {
      if (dimension < 0 || dimension > std::numeric_limits<int32_t>::max())
        throw nb::value_error(
            "CPU grid dimensions must be nonnegative int32 values");
    }
    if (numThreads < 0 || numThreads > std::numeric_limits<int32_t>::max())
      throw nb::value_error(
          "num_cpu_threads must be a nonnegative int32 value");
    if (grid[0] == 0 || grid[1] == 0 || grid[2] == 0)
      return;
    int64_t blocks = 1;
    for (int64_t dimension : grid) {
      if (blocks > std::numeric_limits<int64_t>::max() / dimension)
        throw nb::value_error("CPU grid block count must fit in int64");
      blocks *= dimension;
    }

    const void *data = arguments.data();
    const int32_t gridX = grid[0], gridY = grid[1], gridZ = grid[2];
    nb::gil_scoped_release release;
    if (blocks == 1) {
      function(data, 0, 0, 0, 1, 1, 1);
      return;
    }
    const int threads = numThreads > 0 ? numThreads : omp_get_max_threads();
    auto invoke = [&](int64_t index) {
      const int32_t x = index % gridX;
      const int32_t y = (index / gridX) % gridY;
      const int32_t z = index / (int64_t(gridX) * gridY);
      function(data, x, y, z, gridX, gridY, gridZ);
    };
    if (threads == 1) {
      for (int64_t index = 0; index < blocks; ++index)
        invoke(index);
      return;
    }
#pragma omp parallel for schedule(static) num_threads(threads)
    for (int64_t index = 0; index < blocks; ++index)
      invoke(index);
  }

private:
  // The compiler loads each argument from an eight-byte slot. Grid coordinates
  // stay separate so all blocks can share the immutable argument buffer.
  using Entry = void (*)(const void *, int32_t, int32_t, int32_t, int32_t,
                         int32_t, int32_t);
  std::unique_ptr<void, LibraryCloser> library;
  Entry function;
  size_t argumentBytes;
};

} // namespace

NB_MODULE(_cpu, module) {
  module.attr("amx_enabled") = enableAMX();
  nb::class_<Kernel>(module, "Kernel")
      .def(nb::init<const std::string &, const std::string &, size_t>(),
           "path"_a, "symbol"_a, "argument_bytes"_a)
      .def("launch", &Kernel::launch, "arguments"_a, "grid"_a,
           "num_cpu_threads"_a = 0);
}
