// SPDX-FileCopyrightText: Copyright (c) <2026> Intel Corporation. All rights
// reserved.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <string>

namespace cutile {

// Backend hooks, implemented by exactly one backend's compiler source.
extern const char *const backendName;
void registerBackendTranslations(mlir::DialectRegistry &registry);
void registerBackendPasses();
void bindBackend(nanobind::module_ &module);

class Compilation {
public:
  explicit Compilation(bool printIR = false);
  mlir::OwningOpRef<mlir::ModuleOp> readTileIR(llvm::StringRef bytecode);
  mlir::OwningOpRef<mlir::ModuleOp> readMLIR(llvm::StringRef source);
  // Runs the TileIR pipeline configured by `options`, then the backend passes.
  void
  lowerTileIR(mlir::ModuleOp module, llvm::StringRef options,
              llvm::function_ref<void(mlir::OpPassManager &)> addBackendPasses);
  void run(mlir::ModuleOp module, mlir::PassManager &pm);
  [[noreturn]] void fail(llvm::StringRef stage);
  mlir::MLIRContext context;
  std::string diagnostics;

private:
  bool printIR;
  mlir::ScopedDiagnosticHandler handler;
};

std::string printModule(mlir::ModuleOp module);

} // namespace cutile
