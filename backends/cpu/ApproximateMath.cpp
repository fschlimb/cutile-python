// SPDX-FileCopyrightText: Copyright (c) <2026> Intel Corporation. All rights
// reserved.
//
// SPDX-License-Identifier: Apache-2.0

#include "ApproximateMath.h"

#include <utility>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Math/Transforms/Passes.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace cutile {
namespace {

class ApproximateMathPass
    : public mlir::PassWrapper<ApproximateMathPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ApproximateMathPass)

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::arith::ArithDialect, mlir::math::MathDialect,
                    mlir::vector::VectorDialect>();
  }

  void runOnOperation() override {
    auto approximate = [](llvm::StringRef name) {
      return name != mlir::math::SinOp::getOperationName() &&
             name != mlir::math::CosOp::getOperationName();
    };
    mlir::RewritePatternSet patterns(&getContext());
    // exp2 has no approximation of its own; express it through exp.
    mlir::math::populateExpansionPatterns(patterns, {"exp2"});
    mlir::populateMathF32ExpansionPatterns(patterns, approximate);
    mlir::populateMathPolynomialApproximationPatterns(patterns, approximate);
    if (mlir::failed(
            mlir::applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createApproximateMathPass() {
  return std::make_unique<ApproximateMathPass>();
}

} // namespace cutile
