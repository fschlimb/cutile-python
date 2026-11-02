<!--- SPDX-FileCopyrightText: Copyright (c) <2026> Intel Corporation. All rights reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0 -->

- Added experimental CPU and Intel XPU backends, built with `CUTILE_BUILD_VARIANT=cpu`
  or `xpu` (see `backends/README.md`). `ct.set_backend("cpu")`,
  `ct.set_backend("xpu")`, or any module implementing the `Backend` protocol from
  `cuda.tile._backends._custom`, runs `ct.launch` on that backend regardless of
  argument devices; `ct.set_backend(None)` restores CUDA.
  `ct.backend_options({...})` scopes backend options for compiles and launches;
  unknown options are rejected.
