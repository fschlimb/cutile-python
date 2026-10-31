<!--- SPDX-FileCopyrightText: Copyright (c) <2026> Intel Corporation. All rights reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0 -->

- Added experimental CPU and Intel XPU backends, built with `CUTILE_BUILD_VARIANT=cpu`
  or `xpu` (see `backends/README.md`). `ct.set_backend("cpu")`,
  `ct.set_backend("xpu")`, or any module implementing the `Backend` protocol of
  `cuda.tile._backend` runs `ct.launch` on that backend unless an array argument is on
  a CUDA device; `ct.clear_backend()` restores CUDA. `ct.backend_options({...})` scopes
  backend options for compiles and launches; unknown options are rejected.
