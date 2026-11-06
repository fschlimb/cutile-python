# SPDX-FileCopyrightText: Copyright (c) <2026> Intel Corporation. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import statistics
import timeit

import torch

import cuda.tile as ct
from cuda.tile._backends import cpu
from cuda.tile._cext import default_tile_context


@ct.kernel
def vector_add(a, b, out, block: ct.Constant[int]):
    index = ct.bid(0)
    left = ct.load(a, index=(index,), shape=(block,))
    right = ct.load(b, index=(index,), shape=(block,))
    ct.store(out, index=(index,), tile=left + right)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--disable-native", action="store_true")
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--iterations", type=int, default=10000)
    args = parser.parse_args()
    default_tile_context.config.cache_dir = None
    if args.disable_native:
        cpu.make_launch_cache = None
    factory = getattr(cpu, "make_launch_cache", None)
    torch.set_num_threads(1)
    ct.set_backend("cpu")
    print(f"native cache: {bool(getattr(cpu, 'make_launch_cache', None))}")
    for blocks, threads in ((0, 1), (1, 1), (64, 1), (64, 4)):
        size = max(blocks, 1) * 256
        left, right = torch.randn(size), torch.randn(size)
        out = torch.empty_like(left)
        values = (left, right, out, 256)
        grid = (blocks,)
        with ct.backend_options({"num_cpu_threads": threads}):
            for _ in range(20):
                ct.launch(None, grid, vector_add, values)
            samples = {"python": [], "native": []} if args.compare else {"selected": []}
            for repetition in range(9):
                modes = list(samples)
                if repetition % 2:
                    modes.reverse()
                for mode in modes:
                    cpu.make_launch_cache = None if mode == "python" else factory
                    ct.launch(None, grid, vector_add, values)
                    elapsed = timeit.timeit(
                        lambda: ct.launch(None, grid, vector_add, values),
                        number=args.iterations)
                    samples[mode].append(elapsed * 1e6 / args.iterations)
        if blocks:
            torch.testing.assert_close(out, left + right)
        medians = {mode: statistics.median(times) for mode, times in samples.items()}
        result = " ".join(f"{mode}={median:.3f} us" for mode, median in medians.items())
        if args.compare:
            result += f" speedup={medians['python'] / medians['native']:.2f}x"
        print(f"blocks={blocks:2} threads={threads}: {result}")
    cpu.make_launch_cache = factory


if __name__ == "__main__":
    main()
