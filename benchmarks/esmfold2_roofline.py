# Copyright © 2026 Apple Inc.

"""What does MLX's own GEMM achieve at ESMFold2's shapes, versus the whole fold?

    python benchmarks/esmfold2_roofline.py --lengths 578 1000

Times each GEMM the pair stack actually issues, in isolation, and reports
achieved TFLOPS. Compare against the fold's end-to-end TFLOPS (printed at the
bottom from a wall-clock you supply with --fold-seconds).

The point is to size the headroom before writing another kernel. If the isolated
GEMMs run far faster than the fold does, the loss is between them and worth
chasing. If they run at the same rate, the GEMMs are the floor and no kernel
helps.

Also times the triangular contraction from its native [B,L,L,D] layout, where
MLX must transpose to [B,D,L,L] first -- that copy is a candidate cost, and
esm's Triton path avoids it by reading the native layout directly.
"""

import argparse
import time

import mlx.core as mx


def timeit(fn, iters=10, warmup=3):
    for _ in range(warmup):
        mx.eval(fn())
    mx.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        mx.eval(fn())
    mx.synchronize()
    return (time.perf_counter() - t) / iters


def row(label, secs, flops, note=""):
    print(f"  {label:<44}{secs*1e3:>10.1f}{flops/secs/1e12:>10.2f}  {note}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths", type=int, nargs="+", default=[578, 1000])
    ap.add_argument("--c-z", type=int, default=256)
    ap.add_argument("--loops", type=int, default=3)
    ap.add_argument("--blocks", type=int, default=24)
    ap.add_argument("--fold-seconds", type=float, nargs="*", default=None,
                    help="measured fold wall-clock, one per --lengths entry")
    ap.add_argument("--iters", type=int, default=10)
    args = ap.parse_args()

    D = args.c_z
    dt = mx.bfloat16
    print(f"device={mx.default_device()}  c_z={D}  dtype=bfloat16\n")

    for i, L in enumerate(args.lengths):
        M = L * L
        n_tri = args.loops * args.blocks * 2      # trimul calls per fold
        n_trans = args.loops * args.blocks        # pair transitions per fold
        print(f"=== L={L}  (M = L^2 = {M})")
        print(f"  {'op':<44}{'ms':>10}{'TFLOPS':>10}")

        # 1. Triangular contraction, contiguous [B,D,L,L] @ [B,D,L,L]^T.
        a = mx.random.normal((1, D, L, L)).astype(dt)
        b = mx.random.normal((1, D, L, L)).astype(dt)
        mx.eval(a, b)
        f_contract = 2 * D * L**3
        t_c = timeit(lambda: a @ b.transpose(0, 1, 3, 2), args.iters)
        row("contraction, pre-transposed (best case)", t_c, f_contract)

        # 2. The same contraction as the model issues it: from [B,L,L,D],
        #    which forces MLX to transpose into [B,D,L,L] first.
        la = mx.random.normal((1, L, L, D)).astype(dt)
        rb = mx.random.normal((1, L, L, D)).astype(dt)
        mx.eval(la, rb)

        def native():
            l = la.transpose(0, 3, 1, 2)
            r = rb.transpose(0, 3, 1, 2)
            return (l @ r.transpose(0, 1, 3, 2)).transpose(0, 2, 3, 1)

        t_n = timeit(native, args.iters)
        row("contraction, from native layout (actual)", t_n, f_contract,
            f"{t_n/t_c:.2f}x the pre-transposed cost")

        # 3. proj_bundle: [M,D] @ [D,4D]
        x = mx.random.normal((M, D)).astype(dt)
        w4 = mx.random.normal((4 * D, D)).astype(dt)
        mx.eval(x, w4)
        f_bundle = 2 * M * D * 4 * D
        t_b = timeit(lambda: x @ w4.T, args.iters)
        row("proj_bundle  [M,256] @ [256,1024]", t_b, f_bundle)

        # 4. transition w12: [M,D] @ [D,2*4D]
        w12 = mx.random.normal((2 * 4 * D, D)).astype(dt)
        mx.eval(w12)
        f_w12 = 2 * M * D * 2 * 4 * D
        t_w = timeit(lambda: x @ w12.T, args.iters)
        row("transition w12  [M,256] @ [256,2048]", t_w, f_w12)

        # Projected fold time if every GEMM ran at its isolated rate.
        total_flops = (n_tri * f_contract + n_tri * f_bundle
                       + n_trans * (f_w12 + 2 * M * 4 * D * D))
        floor = (n_tri * t_n + n_tri * t_b
                 + n_trans * (t_w + t_w * 0.5))
        print(f"\n  pair-stack FLOPs per fold      : {total_flops/1e12:8.1f} TFLOP")
        print(f"  time if GEMM-bound at above rates: {floor:8.1f} s")
        if args.fold_seconds and i < len(args.fold_seconds):
            meas = args.fold_seconds[i]
            print(f"  measured fold                    : {meas:8.1f} s")
            r = meas / floor
            if r < 0.9:
                verdict = ("floor exceeds the measured fold -- settings differ "
                           "between this run and the fold; compare like for like")
            elif r < 1.5:
                verdict = "the GEMMs ARE the fold. A custom kernel will not help."
            else:
                verdict = (f"{r:.1f}x the GEMM floor -- most of the time is NOT in "
                           "the GEMMs. Worth locating before writing a kernel.")
            print(f"  => {verdict}")
        print()


if __name__ == "__main__":
    main()
