// SPDX-FileCopyrightText: Copyright (c) <2026> Intel Corporation. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

// Entry points by host capability; the CPU backend selects the one it can run.
module attributes {transform.with_named_sequence} {
  // Any host.
  transform.named_sequence @__transform_nano(%root: !transform.any_op {transform.readonly}) {
    %funcs = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %funcs {
      transform.apply_patterns.vector.cast_away_vector_leading_one_dim
      transform.apply_patterns.vector.drop_unit_dims_with_shape_cast
    } : !transform.any_op
    transform.yield
  }

  // x86 hosts.
  transform.named_sequence @__transform_nano_x86(%root: !transform.any_op {transform.readonly}) {
    transform.include @__transform_nano failures(propagate) (%root) : (!transform.any_op) -> ()
    %funcs = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %funcs {
      transform.apply_patterns.x86.prepare_vector_contract_loops
    } : !transform.any_op
    transform.apply_dce to %funcs : !transform.any_op
    transform.apply_cse to %funcs : !transform.any_op
    transform.yield
  }

  // x86 hosts with AMX tile, BF16 and INT8 support.
  transform.named_sequence @__transform_nano_amx(%root: !transform.any_op {transform.readonly}) {
    transform.include @__transform_nano_x86 failures(propagate) (%root) : (!transform.any_op) -> ()
    %funcs = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %funcs {
      transform.apply_patterns.x86.vector_contract_multi_level_unroll_for_amx
      transform.apply_patterns.x86.vector_contract_to_amx_dot_product
    } : !transform.any_op
    transform.yield
  }
}
