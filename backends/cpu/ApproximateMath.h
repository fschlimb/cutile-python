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
// Replaces math functions with vectorizable polynomial approximations, except
// sin and cos, whose approximation is inaccurate for large arguments.
std::unique_ptr<mlir::Pass> createApproximateMathPass();
} // namespace cutile
