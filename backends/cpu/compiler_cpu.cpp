// SPDX-FileCopyrightText: Copyright (c) <2026> Intel Corporation. All rights
// reserved.
//
// SPDX-License-Identifier: Apache-2.0

#include "amx.h"
#include "lld/Common/Driver.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Conversion/TileIRToMLIR/Passes.h"
#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/Transform/Transforms/Passes.h"
#include "mlir/Dialect/Vector/Transforms/Passes.h"
#include "mlir/LowerVectorScan.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/compiler.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/TargetParser/SubtargetFeature.h"
#include "llvm/TargetParser/Triple.h"
#include <dlfcn.h>
#include <link.h>
#include <mutex>
#include <stdexcept>

LLD_HAS_DRIVER(elf)

namespace cutile {

const char *const backendName = "cpu";

void registerBackendTranslations(mlir::DialectRegistry &) {}

void registerBackendPasses() {
  mlir::registerTransformsPasses();
  mlir::registerLowerAffinePass();
  mlir::registerFinalizeMemRefToLLVMConversionPass();
  mlir::registerSCFToControlFlowPass();
  mlir::registerConvertIndexToLLVMPass();
  mlir::registerConvertVectorToLLVMPass();
  mlir::registerConvertVectorToSCF();
  mlir::registerConvertMathToLLVMPass();
  mlir::registerConvertMathToLibmPass();
  mlir::registerArithToLLVMConversionPass();
  mlir::registerUBToLLVMConversionPass();
  mlir::registerConvertToLLVMPass();
  mlir::registerReconcileUnrealizedCastsPass();
  mlir::math::registerMathPasses();
  mlir::memref::registerMemRefPasses();
  mlir::vector::registerVectorPasses();
  mlir::transform::registerTransformPasses();
  if (llvm::InitializeNativeTarget() ||
      llvm::InitializeNativeTargetAsmPrinter())
    throw std::runtime_error(
        "cuTile compiler: native LLVM target is unavailable");
}

static llvm::StringRef cpuTransformEntryPoint() {
  static const char *entryPoint = [] {
    if (llvm::Triple(llvm::sys::getProcessTriple()).getArch() !=
        llvm::Triple::x86_64)
      return "__transform_nano";
    return cutile::cpu::enableAMX() ? "__transform_nano_amx"
                                    : "__transform_nano_x86";
  }();
  return entryPoint;
}

static std::unique_ptr<llvm::Module>
lowerToLLVM(Compilation &compilation, mlir::ModuleOp module,
            llvm::StringRef schedulePath, llvm::StringRef entryPoint,
            llvm::LLVMContext &llvmContext) {
  mlir::PassManager pm(&compilation.context);

  mlir::transform::PreloadLibraryPassOptions preloadOptions;
  preloadOptions.transformLibraryPaths.push_back(schedulePath.str());
  pm.addPass(
      mlir::transform::createPreloadLibraryPass(std::move(preloadOptions)));
  mlir::transform::InterpreterPassOptions interpreterOptions;
  interpreterOptions.entryPoint =
      (entryPoint.empty() ? cpuTransformEntryPoint() : entryPoint).str();
  pm.addPass(
      mlir::transform::createInterpreterPass(std::move(interpreterOptions)));

  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createLoopInvariantCodeMotionPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createSymbolDCEPass());
  pm.nest<mlir::func::FuncOp>().addPass(
      mlir::vector::createLowerVectorMultiReductionPass(
          mlir::vector::VectorMultiReductionLowering::InnerReduction));
  pm.addPass(mlir::memref::createFoldMemRefAliasOpsPass());
  pm.addPass(mlir::memref::createExpandStridedMetadataPass());
  pm.addPass(mlir::createCanonicalizerPass());

  mlir::VectorTransferToSCFOptions vectorToSCFOptions;
  vectorToSCFOptions.enableFullUnroll().setTargetRank(1);
  pm.addPass(mlir::createConvertVectorToSCFPass(vectorToSCFOptions));
  pm.addPass(mlir::createLowerAffinePass());
  pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
  pm.addPass(mlir::createSCFToControlFlowPass());
  pm.addPass(mlir::createConvertIndexToLLVMPass());

  mlir::ConvertVectorToLLVMPassOptions vectorToLLVMOptions;
  vectorToLLVMOptions.reassociateFPReductions = true;
  vectorToLLVMOptions.vectorContractLowering =
      mlir::vector::VectorContractLowering::OuterProduct;
  pm.addPass(
      mlir::createConvertVectorToLLVMPass(std::move(vectorToLLVMOptions)));

  mlir::math::MathExpandOpsPassOptions mathExpandOptions;
  mathExpandOptions.opMnemonics.push_back("fpowi");
  pm.addPass(mlir::math::createMathExpandOpsPass(std::move(mathExpandOptions)));
  mlir::ConvertMathToLLVMPassOptions mathToLLVMOptions;
  mathToLLVMOptions.approximateLog1p = true;
  pm.addPass(mlir::createConvertMathToLLVMPass(std::move(mathToLLVMOptions)));
  pm.addPass(mlir::createConvertMathToLibmPass());
  pm.addPass(mlir::createReconcileUnrealizedCastsPass());
  pm.addPass(mlir::createArithToLLVMConversionPass());
  pm.addPass(mlir::createUBToLLVMConversionPass());
  pm.addPass(mlir::createConvertToLLVMPass());
  pm.addPass(mlir::createReconcileUnrealizedCastsPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createSymbolDCEPass());

  compilation.run(module, pm);
  auto result = mlir::translateModuleToLLVMIR(module, llvmContext);
  if (!result)
    compilation.fail("translating to LLVM IR");
  return result;
}

static void writeFile(llvm::StringRef path, llvm::StringRef data) {
  std::error_code error;
  llvm::raw_fd_ostream output(path, error);
  if (error)
    throw std::runtime_error("cuTile compiler: " + path.str() + ": " +
                             error.message());
  output << data;
  output.close();
  if (output.has_error())
    throw std::runtime_error("cuTile compiler: unable to write " + path.str());
}

// Resolve by SONAME: a symbol address would name an interposer such as an
// LD_PRELOADed allocator instead of the system library.
static std::string loadedLibrary(const char *soname) {
  void *handle = dlopen(soname, RTLD_LAZY | RTLD_NOLOAD);
  link_map *map = nullptr;
  std::string path;
  if (handle && dlinfo(handle, RTLD_DI_LINKMAP, &map) == 0 && map)
    path = map->l_name;
  if (handle)
    dlclose(handle);
  if (path.empty())
    throw std::runtime_error(std::string("cuTile compiler: ") + soname +
                             " is not loaded");
  return path;
}

static std::string linkObject(llvm::StringRef object,
                              llvm::StringRef directory) {
  llvm::SmallString<128> objectPath(directory), sharedPath(directory);
  llvm::sys::path::append(objectPath, "kernel.o");
  llvm::sys::path::append(sharedPath, "kernel.so");
  writeFile(objectPath, object);
  std::string math = loadedLibrary("libm.so.6");
  std::string libc = loadedLibrary("libc.so.6");
  std::string runtime = loadedLibrary("libgcc_s.so.1");
  // One thread: LLD's default thread pool would persist in the host process
  // and deadlock links in fork()ed children.
  const char *args[] = {"ld.lld",
                        "-shared",
                        "-z",
                        "defs",
                        "--threads=1",
                        "--as-needed",
                        objectPath.c_str(),
                        math.c_str(),
                        libc.c_str(),
                        runtime.c_str(),
                        "-o",
                        sharedPath.c_str()};
  // LLD has process-global state. It recovers from its fatal errors, but must
  // not run again in the same process afterwards.
  static std::mutex linkerMutex;
  static bool linkerUsable = true;
  std::lock_guard<std::mutex> guard(linkerMutex);
  if (!linkerUsable)
    throw std::runtime_error(
        "cuTile compiler: the linker failed fatally before; restart the "
        "process or set a compiler timeout to link in a worker process");
  std::string output, errors;
  llvm::raw_string_ostream stdoutStream(output), stderrStream(errors);
  lld::DriverDef drivers[] = {{lld::Gnu, &lld::elf::link}};
  lld::Result result = lld::lldMain(args, stdoutStream, stderrStream, drivers);
  linkerUsable = result.canRunAgain;
  if (result.retCode)
    throw std::runtime_error("cuTile compiler: linking failed\n" + errors +
                             output);
  if (!errors.empty())
    llvm::errs() << errors;
  auto binary = llvm::MemoryBuffer::getFile(sharedPath);
  if (!binary)
    throw std::runtime_error("cuTile compiler: unable to read linked kernel: " +
                             binary.getError().message());
  return (*binary)->getBuffer().str();
}

static std::string compileCPU(llvm::StringRef bytecode,
                              llvm::StringRef schedulePath, bool assumeInBounds,
                              llvm::StringRef directory,
                              llvm::StringRef dumpPath, bool printIR,
                              bool packArguments) {
  Compilation compilation(printIR);
  auto module = compilation.readTileIR(bytecode);
  compilation.lowerTileIR(
      *module,
      "target=cpu append-grid-args=true "
      "assume-in-bounds=" +
          std::string(assumeInBounds ? "true" : "false"),
      [&](mlir::OpPassManager &pm) {
        pm.addPass(createLowerVectorScanPass());
        pm.addPass(mlir::createLoopInvariantCodeMotionPass());
        pm.addPass(mlir::createConvertMemrefArgsToPtrArgsPass());
        if (packArguments)
          pm.addPass(mlir::createPackCPUKernelArgumentsPass());
        pm.addPass(mlir::createCSEPass());
        pm.addPass(mlir::createCanonicalizerPass());
      });
  if (!dumpPath.empty())
    writeFile(dumpPath, printModule(*module));
  llvm::LLVMContext llvmContext;
  auto llvmModule =
      lowerToLLVM(compilation, *module, schedulePath, {}, llvmContext);
  auto triple = llvm::sys::getProcessTriple();
  std::string error;
  auto *target =
      llvm::TargetRegistry::lookupTarget(llvm::Triple(triple), error);
  if (!target)
    throw std::runtime_error("cuTile compiler: " + error);
  llvm::SubtargetFeatures features;
  for (const auto &feature : llvm::sys::getHostCPUFeatures())
    features.AddFeature(feature.first(), feature.second);
  std::unique_ptr<llvm::TargetMachine> machine(target->createTargetMachine(
      llvm::Triple(triple), llvm::sys::getHostCPUName(), features.getString(),
      llvm::TargetOptions(), llvm::Reloc::PIC_, std::nullopt,
      llvm::CodeGenOptLevel::Aggressive));
  if (!machine)
    throw std::runtime_error("cuTile compiler: unable to create native target");
  llvmModule->setTargetTriple(llvm::Triple(triple));
  llvmModule->setDataLayout(machine->createDataLayout());
  llvm::PassBuilder builder(llvmContext, machine.get());
  llvm::LoopAnalysisManager loops;
  llvm::FunctionAnalysisManager functions;
  llvm::CGSCCAnalysisManager callGraph;
  llvm::ModuleAnalysisManager modules;
  builder.registerLoopAnalyses(loops);
  builder.registerFunctionAnalyses(functions);
  builder.registerCGSCCAnalyses(callGraph);
  builder.registerModuleAnalyses(modules);
  builder.crossRegisterProxies(loops, functions, callGraph, modules);
  auto optimizer =
      builder.buildPerModuleDefaultPipeline(llvm::OptimizationLevel::O3);
  optimizer.run(*llvmModule, modules);
  llvm::raw_string_ostream errors(compilation.diagnostics);
  if (llvm::verifyModule(*llvmModule, &errors))
    compilation.fail("verifying LLVM IR");
  llvm::SmallVector<char, 0> object;
  llvm::raw_svector_ostream stream(object);
  llvm::legacy::PassManager codegen;
  if (machine->addPassesToEmitFile(codegen, stream, nullptr,
                                   llvm::CodeGenFileType::ObjectFile))
    throw std::runtime_error(
        "cuTile compiler: native object emission is unavailable");
  codegen.run(*llvmModule);
  return linkObject(llvm::StringRef(object.data(), object.size()), directory);
}

void bindBackend(nanobind::module_ &module) {
  namespace nb = nanobind;
  using namespace nb::literals;
  module.def(
      "compile_cpu",
      [](nb::bytes bytecode, const std::string &schedulePath,
         bool assumeInBounds, const std::string &directory,
         const std::string &dumpPath, bool printIR, bool packArguments) {
        llvm::StringRef input(bytecode.c_str(), bytecode.size());
        std::string binary;
        {
          nb::gil_scoped_release release;
          binary = compileCPU(input, schedulePath, assumeInBounds, directory,
                              dumpPath, printIR, packArguments);
        }
        return nb::bytes(binary.data(), binary.size());
      },
      "bytecode"_a, "schedule_path"_a, "assume_in_bounds"_a, "directory"_a,
      "dump_path"_a = "", "print_ir"_a = false, "pack_arguments"_a = true);
  module.def(
      "lower_cpu_mlir",
      [](nb::bytes source, const std::string &schedulePath,
         const std::string &entryPoint) {
        llvm::StringRef input(source.c_str(), source.size());
        std::string result;
        {
          nb::gil_scoped_release release;
          Compilation compilation;
          auto module = compilation.readMLIR(input);
          llvm::LLVMContext llvmContext;
          auto llvmModule = lowerToLLVM(compilation, *module, schedulePath,
                                        entryPoint, llvmContext);
          llvm::raw_string_ostream stream(result);
          llvmModule->print(stream, nullptr);
        }
        return nb::bytes(result.data(), result.size());
      },
      "source"_a, "schedule_path"_a, "entry_point"_a = "");
  module.def("cpu_transform_entry_point",
             [] { return cpuTransformEntryPoint().str(); });
}

} // namespace cutile
