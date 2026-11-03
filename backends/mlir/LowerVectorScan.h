// SPDX-FileCopyrightText: Copyright (c) <2026> Intel Corporation. All rights
// reserved.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>

namespace mlir {
class Pass;
}

namespace cutile {
// Lowers vector.scan to target-independent vector and arithmetic operations.
std::unique_ptr<mlir::Pass> createLowerVectorScanPass();
} // namespace cutile
