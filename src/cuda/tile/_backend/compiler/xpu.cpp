// SPDX-FileCopyrightText: Copyright (c) <2026> Intel Corporation.
// SPDX-License-Identifier: Apache-2.0

#include "compiler.h"
#include "mlir/Conversion/TileIRToMLIR/Passes.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Pipelines/Passes.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Target/LLVMIR/Dialect/GPU/GPUToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/XeVM/XeVMToLLVMIRTranslation.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/TargetSelect.h"
#include <array>
#include <mutex>
#include <nanobind/stl/array.h>
#include <ocloc_api.h>
#include <stdexcept>

namespace cutile {

const char *const backendName = "xpu";

// libocloc does not document thread safety; the former Python XeVM path ran
// under the GIL, so its use stays serialized.
static std::mutex oclocMutex;

void registerBackendTranslations(mlir::DialectRegistry &registry) {
  mlir::registerGPUDialectTranslation(registry);
  mlir::registerXeVMDialectTranslation(registry);
}

void registerBackendPasses() {
  mlir::gpu::registerGPUToXeVMPipeline();
  LLVMInitializeSPIRVTargetInfo();
  LLVMInitializeSPIRVTarget();
  LLVMInitializeSPIRVTargetMC();
  LLVMInitializeSPIRVAsmPrinter();
}

static std::string compileXPU(llvm::StringRef bytecode,
                              const std::array<int32_t, 3> &block,
                              bool assumeInBounds, const std::string &opLevel,
                              bool largeRegisters, const std::string &chip,
                              bool dumpIR) {
  Compilation compilation;
  auto module = compilation.readTileIR(bytecode);
  compilation.lowerTileIR(
      *module,
      "drop-rounding-modes=true assume-in-bounds=" +
          std::string(assumeInBounds ? "true" : "false") +
          " known-block-size=" + std::to_string(block[0]) + "," +
          std::to_string(block[1]) + "," + std::to_string(block[2]),
      [](mlir::OpPassManager &pm) {
        pm.addPass(mlir::createConvertMemrefArgsToRankedMemrefPass(
            {mlir::MemrefArgRemovalMode::AssumedMemrefDependent}));
      });
  if (dumpIR)
    llvm::errs() << printModule(*module) << '\n';
  std::string pipeline =
      "gpu-lower-to-xevm-pipeline{enable-vector-to-xegpu=true xegpu-op-level=" +
      opLevel + " zebin-chip=" + chip;
  if (largeRegisters)
    pipeline += " igc-cmd-options=-ze-opt-large-register-file";
  pipeline += "}";
  mlir::PassManager pm(&compilation.context);
  llvm::raw_string_ostream errors(compilation.diagnostics);
  if (failed(mlir::parsePassPipeline(pipeline, pm, errors)))
    compilation.fail("constructing the XeVM pipeline");
  {
    std::lock_guard<std::mutex> guard(oclocMutex);
    compilation.run(*module, pm);
  }
  auto binaries = module->getOps<mlir::gpu::BinaryOp>();
  if (!llvm::hasSingleElement(binaries))
    throw std::runtime_error("cuTile compiler: expected a single gpu.binary");
  auto objects = (*binaries.begin()).getObjects();
  if (objects.size() != 1)
    throw std::runtime_error("cuTile compiler: expected a single GPU object");
  return mlir::cast<mlir::gpu::ObjectAttr>(objects[0])
      .getObject()
      .getValue()
      .str();
}

static std::string compilerIdentity() {
  std::lock_guard<std::mutex> guard(oclocMutex);
  std::string identity;
  const char *queries[][3] = {{"ocloc", "--version", nullptr},
                              {"ocloc", "query", "IGC_REVISION"}};
  for (auto &args : queries) {
    uint32_t count = 0;
    uint8_t **data = nullptr;
    uint64_t *lengths = nullptr;
    char **names = nullptr;
    auto cleanup = llvm::scope_exit(
        [&] { oclocFreeOutput(&count, &data, &lengths, &names); });
    int result =
        oclocInvoke(args[2] ? 3 : 2, args, 0, nullptr, nullptr, nullptr, 0,
                    nullptr, nullptr, nullptr, &count, &data, &lengths, &names);
    std::string output;
    for (uint32_t i = 0; i < count; ++i) {
      output += names[i];
      output.append(reinterpret_cast<const char *>(data[i]), lengths[i]);
    }
    if (result != OCLOC_SUCCESS)
      throw std::runtime_error(
          "cuTile compiler: libocloc version query failed\n" + output);
    identity += output;
  }
  return identity;
}

void bindBackend(nanobind::module_ &module) {
  namespace nb = nanobind;
  module.def("compile_xpu",
             [](nb::bytes bytecode, const std::array<int32_t, 3> &block,
                bool assumeInBounds, const std::string &opLevel,
                bool largeRegisters, const std::string &chip, bool dumpIR) {
               llvm::StringRef input(bytecode.c_str(), bytecode.size());
               std::string binary;
               {
                 nb::gil_scoped_release release;
                 binary = compileXPU(input, block, assumeInBounds, opLevel,
                                     largeRegisters, chip, dumpIR);
               }
               return nb::bytes(binary.data(), binary.size());
             });
  module.def("xpu_identity", compilerIdentity,
             nb::call_guard<nb::gil_scoped_release>());
}

} // namespace cutile
