// SPDX-FileCopyrightText: Copyright (c) <2026> Intel Corporation. All rights
// reserved.
//
// SPDX-License-Identifier: Apache-2.0

#include "mlir/Conversion/TileIRToMLIR/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Pipelines/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/LowerVectorScan.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Target/LLVMIR/Dialect/GPU/GPUToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/XeVM/XeVMToLLVMIRTranslation.h"
#include "mlir/compiler.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/TargetSelect.h"
#include <array>
#include <mutex>
#include <nanobind/stl/array.h>
#include <nanobind/stl/vector.h>
#include <ocloc_api.h>
#include <stdexcept>

namespace cutile {

const char *const backendName = "xpu";

// libocloc does not document thread safety; the former Python XeVM path ran
// under the GIL, so its use stays serialized.
static std::mutex oclocMutex;

// Give unused arrays the same ranked-memref ABI as arrays accessed by the
// kernel.
static void prepareArrayArguments(mlir::ModuleOp module,
                                  const std::vector<int> &ranks) {
  module.walk([&](mlir::gpu::GPUFuncOp function) {
    if (!function.isKernel())
      return;
    uint64_t expected = 0;
    for (int rank : ranks) {
      if (rank < -1)
        throw std::invalid_argument(
            "XPU parameter ranks must be nonnegative or -1");
      expected += rank < 0 ? 1 : 1 + 2 * uint64_t(rank);
    }
    if (function.getNumArguments() != expected)
      throw std::invalid_argument(
          "XPU parameters do not match the kernel signature");
    mlir::OpBuilder builder = mlir::OpBuilder::atBlockBegin(&function.front());
    unsigned index = 0;
    for (int rank : ranks) {
      auto argument = function.getArgument(index++);
      if (rank < 0)
        continue;
      auto type = mlir::dyn_cast<mlir::UnrankedMemRefType>(argument.getType());
      if (!type)
        throw std::invalid_argument(
            "expected an unranked array parameter before promotion");
      if (argument.use_empty()) {
        llvm::SmallVector<mlir::OpFoldResult> sizes, strides;
        for (int dim = 0; dim < 2 * rank; ++dim) {
          auto value = mlir::arith::IndexCastOp::create(
              builder, function.getLoc(), builder.getIndexType(),
              function.getArgument(index + dim));
          (dim < rank ? sizes : strides).push_back(value.getResult());
        }
        llvm::SmallVector<int64_t> dynamic(rank, mlir::ShapedType::kDynamic);
        auto ranked = mlir::MemRefType::get(
            dynamic, type.getElementType(),
            mlir::StridedLayoutAttr::get(module.getContext(), 0, dynamic),
            type.getMemorySpace());
        mlir::memref::ReinterpretCastOp::create(
            builder, function.getLoc(), ranked, argument,
            builder.getIndexAttr(0), sizes, strides);
      }
      index += 2 * rank;
    }
  });
}

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
                              bool dumpIR, const std::vector<int> &ranks) {
  Compilation compilation;
  auto module = compilation.readTileIR(bytecode);
  compilation.lowerTileIR(
      *module,
      "assume-in-bounds=" + std::string(assumeInBounds ? "true" : "false") +
          " known-block-size=" + std::to_string(block[0]) + "," +
          std::to_string(block[1]) + "," + std::to_string(block[2]),
      [](mlir::OpPassManager &pm) {
        pm.addPass(cutile::createLowerVectorScanPass());
      });
  prepareArrayArguments(*module, ranks);
  mlir::PassManager promotion(&compilation.context);
  promotion.addPass(mlir::createConvertMemrefArgsToRankedMemrefPass(
      {mlir::MemrefArgRemovalMode::AssumedMemrefDependent}));
  compilation.run(*module, promotion);
  module->walk([](mlir::gpu::GPUFuncOp function) {
    if (function.isKernel() &&
        llvm::any_of(function.getArgumentTypes(), [](mlir::Type type) {
          return mlir::isa<mlir::UnrankedMemRefType>(type);
        }))
      throw std::invalid_argument("XPU backend: array accesses cannot be "
                                  "represented by the ranked-memref ABI");
  });
  if (dumpIR)
    llvm::errs() << printModule(*module) << '\n';
  std::string pipeline = "gpu-lower-to-xevm-pipeline{enable-vector-to-xegpu="
                         "true use-64bit-index=true "
                         "xegpu-op-level=" +
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
  using namespace nb::literals;
  module.def(
      "compile_xpu",
      [](nb::bytes bytecode, const std::array<int32_t, 3> &block,
         bool assumeInBounds, const std::string &opLevel, bool largeRegisters,
         const std::string &chip, bool dumpIR, const std::vector<int> &ranks) {
        llvm::StringRef input(bytecode.c_str(), bytecode.size());
        std::string binary;
        {
          nb::gil_scoped_release release;
          binary = compileXPU(input, block, assumeInBounds, opLevel,
                              largeRegisters, chip, dumpIR, ranks);
        }
        return nb::bytes(binary.data(), binary.size());
      },
      "bytecode"_a, "block"_a, "assume_in_bounds"_a, "op_level"_a,
      "large_register_file"_a, "chip"_a, "dump_ir"_a,
      "parameter_ranks"_a = std::vector<int>{});
  module.def("xpu_identity", compilerIdentity,
             nb::call_guard<nb::gil_scoped_release>());
}

} // namespace cutile
