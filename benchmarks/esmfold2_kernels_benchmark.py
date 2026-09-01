# Copyright © 2026 Apple Inc.

"""Benchmark ESMFold2's fused Metal kernels against the pure-MLX path.

    python benchmarks/esmfold2_kernels_benchmark.py

Checks correctness first, then times both paths at the M = B*L*L sizes the
folding trunk actually sees. Requires Metal; on any other backend the two paths
are the same code and the timing is meaningless.
"""

import argparse
import time

import mlx.core as mx

from mlx_lm.models.esmfold2_kernels import trimul_epilogue_kernel, trimul_epilogue_ops

# ESMFold2-Fast: 2 trimuls per block * 24 blocks * num_loops.
CALLS_PER_FOLD = 2 * 24 * 3


def timeit(fn, *args, iters=20):
    mx.eval(fn(*args))
    mx.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        mx.eval(fn(*args))
    mx.synchronize()
    return (time.perf_counter() - t) / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--c-z", type=int, default=256)
    ap.add_argument("--lengths", type=int, nargs="+", default=[100, 300, 500, 700, 1000])
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()

    if not mx.metal.is_available():
        print("Metal not available -- both paths are identical here. Nothing to time.")
        return

    K = N = args.c_z
    print(f"device={mx.default_device()} c_z={K}\n")
    hdr = f"{'L':>6}{'M':>12}{'ops ms':>10}{'kernel ms':>12}{'speedup':>10}{'max|d|':>11}{'per fold':>11}"
    print(hdr)
    print("-" * len(hdr))

    for L in args.lengths:
        M = L * L
        xv = mx.random.normal((M, K)).astype(mx.bfloat16)
        xg = mx.random.normal((M, K)).astype(mx.bfloat16)
        we = mx.random.normal((N, K)).astype(mx.bfloat16)
        wg = mx.random.normal((N, K)).astype(mx.bfloat16)
        res = mx.random.normal((M, N)).astype(mx.bfloat16)
        mx.eval(xv, xg, we, wg, res)

        a = trimul_epilogue_ops(xv, xg, we, wg, res)
        b = trimul_epilogue_kernel(xv, xg, we, wg, res)
        mx.eval(a, b)
        d = float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))))

        t_ops = timeit(trimul_epilogue_ops, xv, xg, we, wg, res, iters=args.iters)
        t_ker = timeit(trimul_epilogue_kernel, xv, xg, we, wg, res, iters=args.iters)
        saved = (t_ops - t_ker) * CALLS_PER_FOLD
        print(f"{L:>6}{M:>12}{t_ops*1e3:>10.2f}{t_ker*1e3:>12.2f}"
              f"{t_ops/t_ker:>9.2f}x{d:>11.3g}{saved:>10.2f}s")

    print(f"\n'per fold' = (ops - kernel) * {CALLS_PER_FOLD} trimul calls per fold.")
    print("Negative means the kernel is slower than MLX's tuned GEMMs -- if so the")
    print("fusion does not pay and trimul_epilogue should keep use_kernel=False.")


if __name__ == "__main__":
    main()
