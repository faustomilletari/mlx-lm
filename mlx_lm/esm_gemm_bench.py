"""Hunt the real GEMM ceiling.

The trunk is 86% GEMM running at 90-99% of a measured 6.03 TFLOP/s roof, and
every non-GEMM component is already at a bandwidth roof. So nothing else in
the model matters until we know whether 6.03 is actually the limit.

Three questions:

  1. What is the best TFLOP/s MLX reaches at ANY size? The M4 Pro's ALU peak
     is roughly 8.6 TFLOP/s, so 6.03 may be leaving 30% unclaimed.
  2. Is bf16 the fastest dtype here? Apple GPUs have run fp16 at full rate for
     years; bf16 simdgroup support is newer and may not match it. If fp16 is
     faster it applies to 86% of the trunk.
  3. L=500 makes 500x500 matrices in the contraction. GEMM tiles are 64 or
     128, so 500 is 7.8 tiles -- ragged. Does padding to 512 pay for itself?
"""

from __future__ import annotations

import mlx.core as mx

from .esm_profiler import _time_op, measure_ceilings, render_ceilings, save_json

DTYPES = {"fp32": mx.float32, "bf16": mx.bfloat16, "fp16": mx.float16}


def _gemm(a, b, flops, iters):
    t = _time_op(lambda: a @ b, iters=iters)
    return t, flops / t / 1e12


def _peak_hunt(sizes, iters):
    print("== 1. peak hunt: square GEMMs, every dtype")
    hdr = f"  {'N':>6}" + "".join(f"{k + ' TF':>12}" for k in DTYPES)
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    best = {k: 0.0 for k in DTYPES}
    for n in sizes:
        line = f"  {n:>6}"
        for k, dt in DTYPES.items():
            a = mx.random.normal((n, n)).astype(dt)
            b = mx.random.normal((n, n)).astype(dt)
            mx.eval(a, b)
            _, tf = _gemm(a, b, 2.0 * n ** 3, max(iters // 2, 3))
            best[k] = max(best[k], tf)
            line += f"{tf:>12.2f}"
            del a, b
            mx.clear_cache()
        print(line)
    print("  " + "-" * (len(hdr) - 2))
    print(f"  {'best':>6}" + "".join(f"{best[k]:>12.2f}" for k in DTYPES))
    top = max(best.values())
    print()
    print(f"  Best overall {top:.2f} TFLOP/s.")
    print(f"  The trunk's GEMMs run at ~5.6-5.9, so headroom against MLX's")
    print(f"  own peak is {100 * (top / 5.75 - 1):+.0f}%.")
    win = max(best, key=best.get)
    print(f"  Fastest dtype at peak: {win}.")
    return best


def _trunk_shapes(seq_lens, d, iters):
    print()
    print("== 2. the trunk's four GEMM shapes, every dtype")
    cases = []
    for L in seq_lens:
        m = L * L
        print()
        print(f"  L={L}  M={m}")
        hdr = (f"  {'op':<24}{'K':>6}{'N':>8}"
               + "".join(f"{k + ' TF':>10}" for k in DTYPES) + f"{'best':>8}")
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        specs = [
            ("proj_bundle  W@x.T", "nt", d, 4 * d),
            ("proj_emit / proj_gate", "std", d, d),
            ("pair_transition w12", "std", d, 8 * d),
            ("pair_transition w3", "std", 4 * d, d),
        ]
        for name, kind, k_, n_ in specs:
            flops = 2.0 * m * k_ * n_
            rates = {}
            line = f"  {name:<24}{k_:>6}{n_:>8}"
            for key, dt in DTYPES.items():
                if kind == "nt":
                    # W(N,K) @ x.T(K,M): the channel-first orientation
                    a = mx.random.normal((n_, k_)).astype(dt)
                    b = mx.random.normal((k_, m)).astype(dt)
                else:
                    a = mx.random.normal((m, k_)).astype(dt)
                    b = mx.random.normal((k_, n_)).astype(dt)
                mx.eval(a, b)
                _, tf = _gemm(a, b, flops, iters)
                rates[key] = tf
                line += f"{tf:>10.2f}"
                del a, b
                mx.clear_cache()
            print(line + f"{max(rates, key=rates.get):>8}")
            cases.append({"L": L, "op": name, "rates": rates})
    return cases


def _ragged(lengths, d, dtype, iters, ref):
    """Per-element cost against L, plus the break-even for padding to it.

    Padding a real L up to L_pad multiplies the work by (L_pad/L)^2, so a
    faster per-element rate only pays if it beats that. The `net vs ref`
    column does that arithmetic: below 1.00 means padding a sequence of
    length `ref` up to this L is a net win.
    """
    print()
    print("== 3. the contraction: is L ragged against the tile size?")
    hdr = (f"  {'L':>6}{'L%64':>6}{'L%128':>7}{'ms':>10}{'TFLOP/s':>9}"
           f"{'ns/elem':>10}{'pad cost':>10}{'net vs ref':>12}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    cases, per_of = [], {}
    for L in lengths:
        a = mx.random.normal((1, d, L, L)).astype(dtype)
        b = mx.random.normal((1, d, L, L)).astype(dtype)
        mx.eval(a, b)
        t, tf = _gemm(a, b.transpose(0, 1, 3, 2), 2.0 * d * L ** 3,
                      max(iters // 2, 3))
        per_of[L] = t * 1e9 / (d * L * L)
        cases.append({"L": L, "ms": t * 1e3, "tflops": tf,
                      "ns_per_elem": per_of[L]})
        del a, b
        mx.clear_cache()

    base = per_of.get(ref)
    for L in lengths:
        per = per_of[L]
        c = next(x for x in cases if x["L"] == L)
        if base and L >= ref:
            pad = (L / ref) ** 2
            net = (per / base) * pad
            pad_s, net_s = f"{pad:>10.3f}", f"{net:>12.3f}"
            c["net_vs_ref"] = net
        else:
            pad_s, net_s = f"{'-':>10}", f"{'-':>12}"
        print(f"  {L:>6}{L % 64:>6}{L % 128:>7}{c['ms']:>10.2f}"
              f"{c['tflops']:>9.2f}{per:>10.3f}{pad_s}{net_s}")

    print()
    print(f"  Read 'net vs ref', where ref = L={ref}. Padding an L={ref}")
    print("  sequence up to this L multiplies the work by 'pad cost', so it")
    print("  only pays if the per-element rate improves by more than that.")
    print("  net < 1.00 means padding wins. net > 1.00 means it does not.")
    return cases


def cmd_gemm(args):
    c = measure_ceilings(dtype=args.mx_dtype, gemm_n=args.gemm_n)
    print(render_ceilings(c))
    print()
    out = {"ceilings": c.as_dict()}
    out["peak"] = _peak_hunt(args.sizes, args.iters)
    out["shapes"] = _trunk_shapes(args.seq_len, args.c_z, args.iters)
    lengths = sorted(set(args.seq_len) | set(args.pad_check))
    out["ragged"] = _ragged(lengths, args.c_z, args.mx_dtype, args.iters,
                            ref=min(args.seq_len))
    if args.json:
        save_json(args.json, out)
        print()
        print(f"wrote {args.json}")
