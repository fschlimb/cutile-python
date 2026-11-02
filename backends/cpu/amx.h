// SPDX-FileCopyrightText: Copyright (c) <2026> Intel Corporation. All rights
// reserved.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#if defined(__linux__) && defined(__x86_64__)
#include <sys/syscall.h>
#include <unistd.h>

#include <cpuid.h>

#ifndef ARCH_REQ_XCOMP_PERM
#define ARCH_REQ_XCOMP_PERM 0x1023
#endif
#endif

namespace cutile::cpu {

// Checks the required AMX features and requests Linux tile-data permission.
inline bool enableAMX() {
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

} // namespace cutile::cpu
