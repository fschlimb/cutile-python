<!--- SPDX-FileCopyrightText: Copyright (c) <2026> Intel Corporation. All rights reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0 -->

- Added experimental CPU and Intel XPU backends, built with `CUTILE_BUILD_VARIANT=cpu`
  or `xpu` (see `src/cuda/tile/_backend/README.md`). `ct.set_backend("cpu")`,
  `ct.set_backend("xpu")`, or any module implementing the backend protocol of
  `cuda.tile._backend` routes `ct.launch` to that backend; `ct.clear_backend()` restores
  CUDA. `ct.backend_options({...})` scopes backend options for compiles and launches;
  unknown options are rejected.
