# Copyright © 2026 Apple Inc.

"""ESMFold2 benchmarks.

    python -m mlx_lm.esmfold2_bench roofline --lengths 1000 --fold-seconds 189.3
    python -m mlx_lm.esmfold2_bench profile  --seq-len 578 --steps 50 14
    python -m mlx_lm.esmfold2_bench kernels

roofline  Times each GEMM the pair stack issues, in isolation, and compares the
          sum against a fold wall-clock you supply. Answers whether a custom
          kernel can help at all. Needs nothing but mlx.
profile   Splits one real fold into ESMC-6B / trunk / sampler. Needs the
          checkpoint and the `esm` featurizer.
kernels   Correctness + speed for the opt-in fused Metal kernels.
"""

import argparse
import time

import mlx.core as mx

INT_KEYS = {
    "token_index", "residue_index", "asym_id", "sym_id", "entity_id", "mol_type",
    "res_type", "input_ids", "ref_element", "ref_atom_name_chars", "atom_to_token",
    "distogram_atom_idx", "msa",
}


def timeit(fn, iters=10, warmup=3):
    for _ in range(warmup):
        mx.eval(fn())
    mx.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        mx.eval(fn())
    mx.synchronize()
    return (time.perf_counter() - t) / iters


def _to_mlx(features_torch, dtype):
    out = {}
    for k, v in features_torch.items():
        if not hasattr(v, "detach"):
            continue
        a = mx.array(v.detach().cpu().numpy())
        if k in INT_KEYS:
            a = a.astype(mx.int32)
        elif a.dtype in (mx.float16, mx.float32, mx.float64, mx.bfloat16):
            a = a.astype(dtype)
        out[k] = a
    return out


# ---------------------------------------------------------------------------
# roofline
# ---------------------------------------------------------------------------


def cmd_roofline(args):
    D, dt = args.c_z, mx.bfloat16
    print(f"device={mx.default_device()}  c_z={D}  dtype=bfloat16\n")

    for i, L in enumerate(args.lengths):
        M = L * L
        n_tri = args.loops * args.blocks * 2
        n_trans = args.loops * args.blocks
        print(f"=== L={L}  (M = L^2 = {M})")
        print(f"  {'op':<44}{'ms':>10}{'TFLOPS':>10}")

        def row(label, secs, flops, note=""):
            print(f"  {label:<44}{secs*1e3:>10.1f}{flops/secs/1e12:>10.2f}  {note}")

        a = mx.random.normal((1, D, L, L)).astype(dt)
        b = mx.random.normal((1, D, L, L)).astype(dt)
        mx.eval(a, b)
        f_contract = 2 * D * L**3
        t_c = timeit(lambda: a @ b.transpose(0, 1, 3, 2), args.iters)
        row("contraction, pre-transposed (best case)", t_c, f_contract)
        del a, b

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
        del la, rb

        x = mx.random.normal((M, D)).astype(dt)
        w4 = mx.random.normal((4 * D, D)).astype(dt)
        mx.eval(x, w4)
        f_bundle = 2 * M * D * 4 * D
        t_b = timeit(lambda: x @ w4.T, args.iters)
        row("proj_bundle  [M,256] @ [256,1024]", t_b, f_bundle)

        w12 = mx.random.normal((2 * 4 * D, D)).astype(dt)
        mx.eval(w12)
        f_w12 = 2 * M * D * 2 * 4 * D
        t_w = timeit(lambda: x @ w12.T, args.iters)
        row("transition w12  [M,256] @ [256,2048]", t_w, f_w12)
        del x, w4, w12
        mx.clear_cache()

        total = n_tri * (f_contract + f_bundle) + n_trans * (f_w12 + 2 * M * 4 * D * D)
        floor = n_tri * (t_n + t_b) + n_trans * (t_w * 1.5)
        print(f"\n  pair-stack FLOPs per fold        : {total/1e12:8.1f} TFLOP")
        print(f"  time if GEMM-bound at above rates: {floor:8.1f} s")
        if args.fold_seconds and i < len(args.fold_seconds):
            meas = args.fold_seconds[i]
            r = meas / floor
            print(f"  measured fold                    : {meas:8.1f} s")
            if r < 0.9:
                v = ("floor exceeds the measured fold -- settings differ between "
                     "this run and the fold; compare like for like")
            elif r < 1.5:
                v = "the GEMMs ARE the fold. A custom kernel will not help."
            else:
                v = (f"{r:.1f}x the GEMM floor -- most of the time is NOT in the "
                     "GEMMs. Worth locating before writing a kernel.")
            print(f"  => {v}")
        print()


# ---------------------------------------------------------------------------
# profile
# ---------------------------------------------------------------------------


def cmd_profile(args):
    from esm.models.esmfold2 import (
        ESMFold2InputBuilder,
        ProteinInput,
        StructurePredictionInput,
    )

    from mlx_lm.models.esmfold2 import ESMFold2Model

    results = []

    def phase(label, fn):
        mx.synchronize()
        mx.reset_peak_memory()
        t = time.perf_counter()
        out = fn()
        mx.eval(out)
        mx.synchronize()
        results.append((label, time.perf_counter() - t,
                        mx.get_peak_memory() / 2**30))
        return out

    seq = args.sequence or "A" * args.seq_len
    print(f"device={mx.default_device()}  L={len(seq)}  loops={args.loops}\n")
    model = ESMFold2Model.from_pretrained(args.repo)
    builder = ESMFold2InputBuilder()
    spi = StructurePredictionInput(sequences=[ProteinInput(id="A", sequence=seq)])
    feats_torch, _ = builder.prepare_input(spi, seed=0, device=model.device)
    feats = _to_mlx(feats_torch, model.z_init_1.weight.dtype)

    lm = phase("ESMC-6B (once)", lambda: model.compute_lm_hidden_states(
        feats["input_ids"],
        asym_id=feats.get("asym_id"),
        residue_index=feats.get("residue_index"),
        mol_type=feats.get("mol_type"),
        token_mask=feats.get("token_attention_mask"),
    ))
    z, x_inputs, aux = phase(
        f"trunk ({args.loops} loops x 24 blocks)",
        lambda: model.trunk(feats, lm, num_loops=args.loops),
    )
    for steps in args.steps:
        mx.random.seed(0)
        phase(f"sampler ({steps} steps)", lambda s=steps:
              model.structure_head.sample(
                  z_trunk=z, s_inputs=x_inputs,
                  relative_position_encoding=aux["relpos"],
                  ref_pos=aux["ref_pos"], ref_charge=aux["ref_charge"],
                  ref_mask=aux["ref_mask"], ref_element=aux["ref_element_oh"],
                  ref_atom_name_chars=aux["ref_name_oh"],
                  ref_space_uid=aux["ref_space_uid"],
                  tok_idx=aux["atom_to_token"], n_tokens=aux["n_tokens"],
                  token_attention_mask=aux["tok_mask"],
                  num_diffusion_samples=1, num_sampling_steps=s,
              ))

    n_atoms = int(aux["ref_mask"].sum().item())
    print(f"atoms = {n_atoms}  ({n_atoms/len(seq):.1f} per residue)\n")
    hdr = f"{'phase':<30}{'seconds':>10}{'peak GB':>10}"
    print(hdr)
    print("-" * len(hdr))
    for label, dt, gb in results:
        print(f"{label:<30}{dt:>10.2f}{gb:>10.2f}")
    base = {l: d for l, d, _ in results}
    lm_t = base["ESMC-6B (once)"]
    tr_t = base[f"trunk ({args.loops} loops x 24 blocks)"]
    for steps in args.steps:
        sm = base[f"sampler ({steps} steps)"]
        tot = lm_t + tr_t + sm
        print(f"\nfold total @ {steps:>3} steps: {tot:6.2f}s   LM {100*lm_t/tot:4.1f}%"
              f"  trunk {100*tr_t/tot:4.1f}%  sampler {100*sm/tot:4.1f}%")


# ---------------------------------------------------------------------------
# kernels
# ---------------------------------------------------------------------------


def cmd_kernels(args):
    from mlx_lm.models.esmfold2_kernels import (
        USE_DUAL_KERNEL,
        USE_EPILOGUE_KERNEL,
        trimul_epilogue_kernel,
        trimul_epilogue_ops,
        trimul_gated_dual_kernel_only,
        trimul_gated_dual_ops,
    )

    if not mx.metal.is_available():
        print("Metal not available -- both paths are identical here.")
        return
    K = args.c_z
    calls = args.loops * args.blocks * 2
    print(f"device={mx.default_device()}  c_z={K}  "
          f"(epilogue={USE_EPILOGUE_KERNEL}, dual={USE_DUAL_KERNEL})\n")

    for title, mk_inputs, ops, ker in (
        ("input stage: gated dual GEMM",
         lambda M: (mx.random.normal((M, K)).astype(mx.bfloat16),
                    mx.random.normal((2 * K, K)).astype(mx.bfloat16),
                    mx.random.normal((2 * K, K)).astype(mx.bfloat16),
                    mx.ones((M,)).astype(mx.bfloat16)),
         trimul_gated_dual_ops, trimul_gated_dual_kernel_only),
        ("output stage: gated GEMM + residual",
         lambda M: (mx.random.normal((M, K)).astype(mx.bfloat16),
                    mx.random.normal((M, K)).astype(mx.bfloat16),
                    mx.random.normal((K, K)).astype(mx.bfloat16),
                    mx.random.normal((K, K)).astype(mx.bfloat16),
                    mx.random.normal((M, K)).astype(mx.bfloat16)),
         trimul_epilogue_ops, trimul_epilogue_kernel),
    ):
        print(f"== {title}")
        hdr = (f"{'L':>6}{'M':>12}{'ops ms':>10}{'kernel ms':>12}"
               f"{'speedup':>10}{'max|d|':>11}{'per fold':>11}")
        print(hdr)
        print("-" * len(hdr))
        for L in args.lengths:
            M = L * L
            ins = mk_inputs(M)
            mx.eval(*ins)
            a, c = ops(*ins), ker(*ins)
            mx.eval(a, c)
            d = float(mx.max(mx.abs(a.astype(mx.float32) - c.astype(mx.float32))))
            t_o = timeit(lambda: ops(*ins), args.iters)
            t_k = timeit(lambda: ker(*ins), args.iters)
            print(f"{L:>6}{M:>12}{t_o*1e3:>10.2f}{t_k*1e3:>12.2f}"
                  f"{t_o/t_k:>9.2f}x{d:>11.3g}{(t_o-t_k)*calls:>10.2f}s")
            del ins, a, c
            mx.clear_cache()
        print()


def main():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--c-z", type=int, default=256)
    common.add_argument("--loops", type=int, default=3)
    common.add_argument("--blocks", type=int, default=24)
    common.add_argument("--iters", type=int, default=10)

    p = argparse.ArgumentParser(description="ESMFold2 benchmarks", parents=[common])
    sub = p.add_subparsers(dest="cmd")

    r = sub.add_parser("roofline", parents=[common],
                       help="are the GEMMs the floor?")
    r.add_argument("--lengths", type=int, nargs="+", default=[578, 1000])
    r.add_argument("--fold-seconds", type=float, nargs="*", default=None)
    r.set_defaults(func=cmd_roofline)

    pr = sub.add_parser("profile", parents=[common], help="split a real fold into phases")
    pr.add_argument("--repo", default="biohub/ESMFold2-Fast")
    pr.add_argument("--sequence", default=None)
    pr.add_argument("--seq-len", type=int, default=460)
    pr.add_argument("--steps", type=int, nargs="+", default=[50, 14])
    pr.set_defaults(func=cmd_profile)

    k = sub.add_parser("kernels", parents=[common], help="fused Metal kernels vs pure MLX")
    k.add_argument("--lengths", type=int, nargs="+", default=[100, 300, 500, 1000])
    k.set_defaults(func=cmd_kernels)

    args = p.parse_args()
    if not getattr(args, "func", None):
        p.print_help()
        raise SystemExit(1)
    args.func(args)


if __name__ == "__main__":
    main()
