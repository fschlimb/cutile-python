<!--- SPDX-FileCopyrightText: Copyright (c) <2026> Intel Corporation. All rights reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0 -->

- Added `ct.autotune`, a decorator that tunes `ct.Constant` parameters and compiler hints of a
  kernel, or the parameters of a factory preparing several launches, by benchmarking each
  `ct.tune.Config` once per input layout and reusing the best one.
