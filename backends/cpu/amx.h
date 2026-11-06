// SPDX-FileCopyrightText: Copyright (c) <2026> Intel Corporation. All rights
// reserved.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#if defined(__linux__) && defined(__x86_64__)
#include <asm/prctl.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <cpuid.h>

#ifndef ARCH_REQ_XCOMP_PERM
#define ARCH_REQ_XCOMP_PERM 0x1023
#endif
#ifndef ARCH_XCOMP_TILEDATA
#define ARCH_XCOMP_TILEDATA 18
#endif
#endif

namespace cutile::cpu {

// Checks the required AMX features and requests Linux tile-data permission.
inline bool enableAMX() {
#if defined(__linux__) && defined(__x86_64__)
  // CPUID leaf 7, subleaf 0: AMX-BF16, AMX-TILE, and AMX-INT8 are EDX bits 22,
  // 24, and 25.
  constexpr unsigned int amxBf16 = 1u << 22;
  constexpr unsigned int amxTile = 1u << 24;
  constexpr unsigned int amxInt8 = 1u << 25;
  constexpr unsigned int requiredFeatures = amxBf16 | amxTile | amxInt8;
  // CPUID returns four register values; AMX feature bits are in EDX.
  unsigned int cpuidEax, cpuidEbx, cpuidEcx, cpuidEdx;
  if (!__get_cpuid_count(7, 0, &cpuidEax, &cpuidEbx, &cpuidEcx, &cpuidEdx))
    return false;
  if ((cpuidEdx & requiredFeatures) != requiredFeatures)
    return false;

  // Linux requires per-thread permission for the XTILEDATA xstate component.
  return syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, ARCH_XCOMP_TILEDATA) == 0;
#else
  return false;
#endif
}

} // namespace cutile::cpu
