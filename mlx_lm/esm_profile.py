"""Profile the MLX ESM models layer by layer, across sequence lengths.

    python -m mlx_lm.esm_profile ceilings
    python -m mlx_lm.esm_profile esmc     --seq-len 128 256 384 500
    python -m mlx_lm.esm_profile esmfold2 --seq-len 128 256 384 500

Pass several lengths and the model is built once, then swept. The sweep fits a
scaling exponent per module class, which is the number worth having: a class
at 10% of runtime growing as L^3 overtakes one at 30% growing as L^1. Ranking
by current share alone is how you optimise the wrong layer.

Weights are not needed. Performance follows shapes and dtype, not values, so
the model is built from its config with random init. That skips a multi-GB
download and lets a profile run anywhere. Pass --weights to load the real
checkpoint when you want to confirm.

Inputs are synthesised, following tests/test_models.py::test_esmfold2. One
caveat: the synthetic token mask has no padding, and ESMC short-circuits its
attention mask to None when every token shares a chain id, so a real padded
or multi-chain input takes a slightly different path. Use --chains 2 to
exercise the masked path.
"""

from __future__ import annotations

import argparse
import json
import time

import mlx.core as mx

from .esm_profiler import (
    LayerProfiler,
    bypass_compile,
    capture,
    measure_ceilings,
    memory_info,
    render,
    render_ceilings,
    render_modes,
    render_scaling,
    render_dtype_mix,
    render_memory,
    render_shapes,
    render_sweep,
    render_verdict,
    save_json,
    time_plain,
)

FLOAT = (mx.float32, mx.float16, mx.bfloat16)
DTYPES = {"bfloat16": mx.bfloat16, "float16": mx.float16, "float32": mx.float32}


# ---------------------------------------------------------------------------
# synthetic inputs
# ---------------------------------------------------------------------------


def synth_feats(n_tokens: int, atoms_per_token: int = 8, chains: int = 1) -> dict:
    """Features with the shapes ESMFold2InputBuilder produces, random values."""
    n_atoms = n_tokens * atoms_per_token
    a2t = mx.repeat(mx.arange(n_tokens), atoms_per_token)[None]
    per = max(n_tokens // chains, 1)
    chain_id = mx.minimum(mx.arange(n_tokens) // per, chains - 1)[None]
    return {
        "token_index": mx.arange(n_tokens)[None],
        "residue_index": mx.arange(n_tokens)[None],
        "asym_id": chain_id,
        "entity_id": chain_id,
        "sym_id": mx.zeros((1, n_tokens), mx.int32),
        "mol_type": mx.zeros((1, n_tokens), mx.int32),
        "res_type": mx.random.randint(4, 24, (1, n_tokens)),
        "input_ids": mx.random.randint(4, 24, (1, n_tokens)),
        "token_bonds": mx.zeros((1, n_tokens, n_tokens, 1)),
        "token_attention_mask": mx.ones((1, n_tokens), mx.bool_),
        "ref_pos": mx.random.normal((1, n_atoms, 3)) * 10.0,
        "ref_element": mx.full((1, n_atoms), 6, mx.int32),
        "ref_charge": mx.zeros((1, n_atoms)),
        "ref_atom_name_chars": mx.random.randint(0, 64, (1, n_atoms, 4)),
        "ref_space_uid": a2t,
        "atom_attention_mask": mx.ones((1, n_atoms), mx.bool_),
        "atom_to_token": a2t,
        "distogram_atom_idx": (mx.arange(n_tokens) * atoms_per_token)[None],
    }


TINY_CONFIG = {
    "d_pair": 8, "d_single": 8, "lm_d_model": 8, "lm_num_layers": 2,
    "inputs": {"d_inputs": 71, "atom_encoder": {
        "d_atom": 16, "d_token": 8, "n_blocks": 1, "n_heads": 2,
        "swa_window_size": 8, "expansion_ratio": 2,
        "spatial_rope_base_frequency": 20.0, "n_spatial_rope_pairs_per_axis": 1,
        "n_uid_rope_pairs": 1, "uid_rope_base_frequency": 10000.0}},
    "folding_trunk": {"n_layers": 1},
    "lm_encoder": {"n_layers": 1, "enabled": True},
    "parcae": {"coda_n_layers": 1},
    "structure_head": {"distogram_bins": 4, "diffusion_module": {
        "c_atom": 16, "c_token": 8, "c_z": 8, "c_s_inputs": 71,
        "sigma_data": 16.0, "fourier_dim": 8, "atom_num_blocks": 1,
        "atom_num_heads": 2, "token_num_blocks": 1, "token_num_heads": 2,
        "transition_multiplier": 2}},
}

TINY_ESMC = dict(hidden_size=8, num_attention_heads=2, num_hidden_layers=2)


# ---------------------------------------------------------------------------
# model construction
# ---------------------------------------------------------------------------


def build_esmc(args):
    from .models import esmc
    if args.weights:
        return esmc.from_pretrained(args.repo, dtype=args.mx_dtype)
    cfg = dict(TINY_ESMC) if args.tiny else dict(
        hidden_size=args.hidden, num_attention_heads=args.heads,
        num_hidden_layers=args.layers)
    m = esmc.Model(esmc.ModelArgs(**cfg))
    m.set_dtype(args.mx_dtype)
    m.eval()
    mx.eval(m.parameters())
    return m


def build_esmfold2(args):
    from .models import esmc, esmfold2
    if args.weights:
        return esmfold2.ESMFold2Model.from_pretrained(args.repo, dtype=args.mx_dtype)
    if args.tiny:
        cfg = TINY_CONFIG
    elif args.config:
        cfg = json.load(open(args.config))
    else:
        from huggingface_hub import hf_hub_download
        cfg = json.load(open(hf_hub_download(args.repo, "config.json")))
    m = esmfold2.ESMFold2Model(cfg)
    # ESMC is ~16 GB of bf16 weights and is dead during trunk and sampler.
    # trunk() takes lm_hidden_states as an argument, so when the lm phase is
    # skipped we never build it and feed a synthetic tensor instead.
    if not args.skip_lm:
        m._esmc = esmc.Model(
            esmc.ModelArgs(**TINY_ESMC) if args.tiny
            else esmc.ModelArgs(hidden_size=cfg.get("lm_d_model", 2560),
                                num_attention_heads=args.heads,
                                num_hidden_layers=cfg.get("lm_num_layers", 80)))
    m._lm_d_model = cfg.get("lm_d_model", 2560)
    m._lm_num_layers = cfg.get("lm_num_layers", 80)
    m.set_dtype(args.mx_dtype)
    m.eval()
    mx.eval(m.parameters())
    return m


# ---------------------------------------------------------------------------
# phases: one closure per thing we want timed separately
# ---------------------------------------------------------------------------


def esmc_phases(model, L, args):
    ids = mx.random.randint(4, 24, (1, L))
    per = max(L // args.chains, 1)
    amask = mx.ones((1, L), mx.bool_) if args.chains == 1 else None
    sid = (None if args.chains == 1
           else mx.minimum(mx.arange(L) // per, args.chains - 1)[None])
    mx.eval(ids)
    return {"encode": lambda: model.encode(ids, attention_mask=amask,
                                           sequence_id=sid)}


def esmfold2_phases(model, L, args):
    feats = {k: (v.astype(args.mx_dtype) if v.dtype in FLOAT else v)
             for k, v in synth_feats(L, args.atoms_per_token, args.chains).items()}
    mx.eval(list(feats.values()))

    lm_kw = dict(asym_id=feats.get("asym_id"),
                 residue_index=feats.get("residue_index"),
                 mol_type=feats.get("mol_type"),
                 token_mask=feats.get("token_attention_mask"))
    if args.skip_lm:
        # LanguageModelShim consumes (B, L, num_layers+1, d_model).
        lm = mx.random.normal(
            (1, L, model._lm_num_layers + 1, model._lm_d_model)
        ).astype(args.mx_dtype)
    else:
        lm = model.compute_lm_hidden_states(feats["input_ids"], **lm_kw)
    mx.eval(lm)
    z, x_inputs, aux = model.trunk(feats, lm, num_loops=args.loops)
    mx.eval(z, x_inputs)

    def sampler():
        mx.random.seed(0)
        return model.structure_head.sample(
            z_trunk=z, s_inputs=x_inputs,
            relative_position_encoding=aux["relpos"], ref_pos=aux["ref_pos"],
            ref_charge=aux["ref_charge"], ref_mask=aux["ref_mask"],
            ref_element=aux["ref_element_oh"],
            ref_atom_name_chars=aux["ref_name_oh"],
            ref_space_uid=aux["ref_space_uid"], tok_idx=aux["atom_to_token"],
            n_tokens=aux["n_tokens"], token_attention_mask=aux["tok_mask"],
            num_diffusion_samples=1, num_sampling_steps=args.steps)

    phases = {}
    if not args.skip_lm:
        phases["lm"] = lambda: model.compute_lm_hidden_states(
            feats["input_ids"], **lm_kw)
    phases["trunk"] = lambda: model.trunk(feats, lm, num_loops=args.loops)
    phases["sampler"] = sampler
    return phases


# ---------------------------------------------------------------------------
# sweep driver
# ---------------------------------------------------------------------------


def _profile_once(model, fn, root, shapes=False):
    """Warm up, then measure the same call in eval mode and in lazy mode."""
    mx.eval(fn())
    mx.synchronize()
    mx.clear_cache()
    mx.reset_peak_memory()

    prof = LayerProfiler(mode="eval", record_shapes=shapes).attach(model, root=root)
    with prof:
        fn()

    lazy = LayerProfiler(mode="lazy", record_shapes=False).attach(model, root=root)
    with lazy:
        out = fn()
    mx.eval(out)
    mx.synchronize()
    del out
    return prof, lazy, mx.get_peak_memory() / 2**30


def sweep(model, make_phases, args, ceilings):
    """Profile every phase at every length. The model is built once, outside.

    Three timings per phase, because they answer different questions:
      plain      compiled, no instrumentation -- the honest wall clock
      bypassed   compiled regions routed back through Python -- comparable
                 baseline for the profiled run, and the cost of losing fusion
      profiled   per-layer attribution, only valid while bypassed
    """
    rows, raw = [], {}
    per_class: dict[str, dict[int, dict]] = {}
    per_layer: dict[str, dict[int, dict]] = {}
    detail = args.detail or ("full" if len(args.seq_len) == 1 else "summary")

    for L in args.seq_len:
        phases = make_phases(model, L, args)
        row = {"L": L, "phase_s": {}, "plain_s": {}, "bypassed_s": {},
               "lazy_s": {}, "peak_gb": 0.0}
        raw[L] = {}
        for label, fn in phases.items():
            row["plain_s"][label] = time_plain(fn)
            mx.clear_cache()

            with bypass_compile(model) as n_bypassed:
                row["bypassed_s"][label] = time_plain(fn) if n_bypassed else \
                    row["plain_s"][label]
                prof, lazy, peak = _profile_once(model, fn, args.root,
                                                 shapes=args.shapes)

            row["phase_s"][label] = prof.wall_s
            row["lazy_s"][label] = lazy.wall_s
            row["peak_gb"] = max(row["peak_gb"], peak)
            row.setdefault("bypassed_sites", n_bypassed)
            raw[L][label] = {"eval": prof.as_dict(), "lazy": lazy.as_dict(),
                             "peak_gb": peak,
                             "plain_s": row["plain_s"][label],
                             "bypassed_s": row["bypassed_s"][label]}

            for store, items in ((per_class, prof.by_class().values()),
                                 (per_layer, prof.active())):
                for st in items:
                    key = st.cls if store is per_class else st.path
                    d = store.setdefault(key, {}).setdefault(
                        L, {"calls": 0, "excl_s": 0.0, "incl_s": 0.0,
                            "moved_bytes": 0, "flops": 0.0})
                    d["calls"] += st.calls
                    d["excl_s"] += st.excl_s
                    d["incl_s"] += st.incl_s
                    d["moved_bytes"] += st.moved_bytes
                    d["flops"] += st.flops

            print(f"\n-- L={L} {label}")
            print(render_dtype_mix(prof.dtype_bytes,
                                   title=f"L={L} {label}: traffic by dtype"))
            if args.shapes and L == args.seq_len[-1]:
                print(f"\n{'#'*78}\n# L={L}  {label}: real shapes and per-call "
                      f"cost\n{'#'*78}")
                print(render_shapes(prof.active(), ceilings, top=args.top,
                                    title=f"L={L} {label}"))
            if detail == "full":
                print(f"\n{'#'*78}\n# L={L}  {label}   "
                      f"(peak {peak:.2f} GB)\n{'#'*78}")
                print(render_modes(prof.wall_s, lazy.wall_s))
                print()
                print(render(prof.active(), ceilings, top=args.top,
                             title=f"L={L} {label}: slowest layers"))
                print()
                print(render(list(prof.by_class().values()), ceilings,
                             top=args.top, label="class",
                             title=f"L={L} {label}: by module class"))
            del prof, lazy
            mx.clear_cache()

        row["total_s"] = sum(row["plain_s"].values())
        row["profiled_total_s"] = sum(row["phase_s"].values())
        tot_lazy = sum(row["lazy_s"].values())
        row["lazy_share"] = (tot_lazy / row["profiled_total_s"]
                             if row["profiled_total_s"] else 0.0)
        rows.append(row)
        if detail == "summary":
            parts = "  ".join(f"{k} {v:.3f}s" for k, v in row["plain_s"].items())
            print(f"  L={L:<5} {parts}   total {row['total_s']:.3f}s"
                  f"   lazy {100*row['lazy_share']:.1f}%"
                  f"   peak {row['peak_gb']:.2f}GB")

    _finalize(per_class)
    _finalize(per_layer)
    return rows, per_class, per_layer, raw


def _finalize(store):
    """Fill in achieved rates, suppressed where children did the work."""
    for byL in store.values():
        for d in byL.values():
            container = d["incl_s"] > 0 and (d["excl_s"] / d["incl_s"]) < 0.5
            dead = container or d["excl_s"] <= 0
            d["gbs"] = None if dead else d["moved_bytes"] / d["excl_s"] / 1e9
            d["tflops"] = (None if dead or d["flops"] <= 0
                           else d["flops"] / d["excl_s"] / 1e12)


def render_compile(rows, phases) -> str:
    """What mx.compile is worth, per phase."""
    out = ["== cost of bypassing mx.compile",
           "  (attribution needs the bypass; this is what it costs, so you can",
           "   tell a real layer cost from a lost-fusion artefact)"]
    head = f"{'L':>6}" + "".join(f"{p[:9]:>22}" for p in phases)
    out += [head, "-" * len(head)]
    for r in rows:
        line = f"{r['L']:>6}"
        for p in phases:
            a, b = r["plain_s"].get(p), r["bypassed_s"].get(p)
            line += f"{a:>9.2f}->{b:>7.2f} {b/a if a else 0:>3.1f}x"
        out.append(line)
    return "\n".join(out)


def _report(rows, per_class, per_layer, args, ceilings, extra=None):
    phases = list(rows[0]["plain_s"]) if rows else []
    peak = max((r["peak_gb"] for r in rows), default=0.0)
    print()
    print(render_memory(peak))
    print()
    print(render_sweep(rows, phases))
    print()
    if rows and rows[0].get("bypassed_sites"):
        print(render_compile(rows, phases))
        print()
    print(render_scaling(per_class, args.seq_len, ceilings, top=args.top))
    print()
    print(render_scaling(per_layer, args.seq_len, ceilings, top=args.top,
                         title="per-layer scaling", label="layer"))
    print()
    print(render_verdict(rows, per_class, args.seq_len, ceilings,
                         top=args.verdict_top))
    if args.json:
        payload = {"ceilings": ceilings.as_dict(), "memory": memory_info(),
                   "rows": rows, "per_class": per_class, "per_layer": per_layer,
                   "args": {k: v for k, v in vars(args).items()
                            if isinstance(v, (int, float, str, bool, list))}}
        if extra:
            payload.update(extra)
        save_json(args.json, payload)
        print(f"\nwrote {args.json}")


# ---------------------------------------------------------------------------
# subcommands
# ---------------------------------------------------------------------------


def cmd_esmc(args):
    model = build_esmc(args)
    a = model.args
    ceilings = measure_ceilings(dtype=args.mx_dtype, gemm_n=args.gemm_n)
    print(render_ceilings(ceilings))
    print(f"\nESMC  hidden={a.hidden_size} heads={a.num_attention_heads} "
          f"layers={a.num_hidden_layers} ffn={a.ffn_hidden}  "
          f"dtype={args.dtype}  weights={'real' if args.weights else 'random'}")
    print(f"sweeping L = {args.seq_len}\n")
    rows, per_class, per_layer, raw = sweep(model, esmc_phases, args, ceilings)
    _report(rows, per_class, per_layer, args, ceilings,
            extra={"cmd": "esmc", "raw": raw} if args.raw else {"cmd": "esmc"})
    _maybe_trace(args, model, esmc_phases)


def cmd_esmfold2(args):
    model = build_esmfold2(args)
    ceilings = measure_ceilings(dtype=args.mx_dtype, gemm_n=args.gemm_n)
    print(render_ceilings(ceilings))
    print(f"\nESMFold2  loops={args.loops} steps={args.steps} "
          f"atoms/token={args.atoms_per_token}  dtype={args.dtype}  "
          f"weights={'real' if args.weights else 'random'}")
    print(f"sweeping L = {args.seq_len}\n")
    rows, per_class, per_layer, raw = sweep(model, esmfold2_phases, args, ceilings)
    _report(rows, per_class, per_layer, args, ceilings,
            extra={"cmd": "esmfold2", "raw": raw} if args.raw
            else {"cmd": "esmfold2"})
    _maybe_trace(args, model, esmfold2_phases)


def _maybe_trace(args, model, make_phases):
    if not args.gputrace:
        return
    L = args.seq_len[0]
    phases = make_phases(model, L, args)
    label = args.trace_phase or list(phases)[-1]
    with capture(args.gputrace) as ok:
        if ok:
            mx.eval(phases[label]())
    print(f"\nwrote {args.gputrace}  (L={L}, phase={label})" if ok
          else "\nno Metal device; no trace written")


def cmd_micro(args):
    """Isolate the pair-stack primitives at their real shapes.

    The sweep says where time goes; this says whether the op itself is slow.
    Everything here runs on a (1, L, L, D) pair tensor, which is the shape
    that dominates the trunk.
    """
    import mlx.nn as nn

    from .esm_profiler import _time_op

    c = measure_ceilings(dtype=args.mx_dtype, gemm_n=args.gemm_n)
    print(render_ceilings(c))
    D, dt = args.c_z, args.mx_dtype
    out = {"ceilings": c.as_dict(), "cases": []}

    for L in args.seq_len:
        M = L * L
        z = mx.random.normal((1, L, L, D)).astype(dt)
        z2 = mx.random.normal((M, D)).astype(dt)
        g = mx.ones((D,)).astype(dt)
        b = mx.zeros((D,)).astype(dt)
        ln = nn.LayerNorm(D, eps=1e-5)
        ln.set_dtype(dt)
        mx.eval(z, z2, g, b, ln.parameters())
        nbytes = z.nbytes
        print(f"\n=== L={L}  pair tensor (1,{L},{L},{D}) {dt.__str__()}  "
              f"{nbytes/2**20:.0f} MiB")
        hdr = f"  {'op':<40}{'ms':>9}{'GB/s':>9}{'%BW':>6}{'vs floor':>10}"
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))

        floor = None
        rows = []

        def row(name, fn, moved):
            nonlocal floor
            t = _time_op(fn, iters=args.iters)
            gbs = moved / t / 1e9
            if floor is None:
                floor = t
            print(f"  {name:<40}{t*1e3:>9.2f}{gbs:>9.1f}"
                  f"{100*gbs/c.peak_bw_gbs:>6.0f}{t/floor:>9.2f}x")
            rows.append({"L": L, "op": name, "ms": t * 1e3, "gbs": gbs})

        # read once, write once: the hard floor for any normalisation
        row("copy floor (z + 1)", lambda: z + 1, 2 * nbytes)
        row("nn.LayerNorm (4D)", lambda: ln(z), 2 * nbytes)
        row("mx.fast.layer_norm (4D)",
            lambda: mx.fast.layer_norm(z, g, b, 1e-5), 2 * nbytes)
        row("mx.fast.layer_norm (2D, reshaped)",
            lambda: mx.fast.layer_norm(z.reshape(M, D), g, b, 1e-5),
            2 * nbytes)
        row("mx.fast.layer_norm (2D, native)",
            lambda: mx.fast.layer_norm(z2, g, b, 1e-5), 2 * nbytes)
        row("mx.fast.rms_norm (4D)",
            lambda: mx.fast.rms_norm(z, g, 1e-5), 2 * nbytes)
        row("manual mean/var LayerNorm", lambda: _manual_ln(z, g, b),
            2 * nbytes)
        row("sigmoid (elementwise ref)", lambda: mx.sigmoid(z), 2 * nbytes)

        # The model does not hand LayerNorm a contiguous tensor.
        # TriangleMultiplicativeUpdate._contract builds (B, D, i, j) and
        # returns out.transpose(0, 2, 3, 1), so the axis LayerNorm reduces
        # over is strided by i*j. Time that, not just the tidy case.
        print(f"\n  -- as the model actually calls it: strided reduction axis")
        raw = mx.random.normal((1, D, L, L)).astype(mx.float32)
        mx.eval(raw)
        sv = raw.transpose(0, 2, 3, 1)          # (1, L, L, D), D strided
        pe = nn.Linear(D, D, bias=False)
        pe.set_dtype(dt)
        mx.eval(pe.parameters())

        row("astype(bf16) on strided view", lambda: sv.astype(dt), 2 * nbytes)
        row("layer_norm(strided astype)  <- MODEL PATH",
            lambda: mx.fast.layer_norm(sv.astype(dt), g, b, 1e-5), 2 * nbytes)
        row("layer_norm(contiguous(strided))",
            lambda: mx.fast.layer_norm(
                mx.contiguous(sv).astype(dt), g, b, 1e-5), 2 * nbytes)
        row("proj_emit(layer_norm(strided))",
            lambda: pe(mx.fast.layer_norm(sv.astype(dt), g, b, 1e-5)),
            2 * nbytes)
        row("proj_emit(layer_norm(contiguous))",
            lambda: pe(mx.fast.layer_norm(
                mx.contiguous(sv).astype(dt), g, b, 1e-5)), 2 * nbytes)

        out["cases"].extend(rows)
        del z, z2, g, b, ln, raw, sv, pe
        mx.clear_cache()

    # The trunk runs fp32, not the requested bf16, and its norms see widths
    # of 256 and 512. The first version of this benchmark tested bf16 at 256
    # only, which is the one combination the model never uses at that point.
    print("\n== LayerNorm across the dtype and width combinations the model "
          "actually uses")
    hdr = (f"  {'dtype':<7}{'width':>7}{'MiB':>8}{'floor ms':>10}"
           f"{'ln ms':>9}{'ln GB/s':>9}{'%BW':>6}{'vs floor':>10}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    L = max(args.seq_len)
    for name, edt in (("bf16", mx.bfloat16), ("f32", mx.float32)):
        for W in args.ln_widths:
            x = mx.random.normal((1, L, L, W)).astype(edt)
            gw = mx.ones((W,)).astype(edt)
            bw_ = mx.zeros((W,)).astype(edt)
            mx.eval(x, gw, bw_)
            moved = 2 * x.nbytes
            t_f = _time_op(lambda: x + 1, iters=args.iters)
            t_n = _time_op(lambda: mx.fast.layer_norm(x, gw, bw_, 1e-5),
                           iters=args.iters)
            gbs = moved / t_n / 1e9
            print(f"  {name:<7}{W:>7}{x.nbytes/2**20:>8.0f}{t_f*1e3:>10.2f}"
                  f"{t_n*1e3:>9.2f}{gbs:>9.1f}"
                  f"{100*gbs/c.peak_bw_gbs:>6.0f}{t_n/t_f:>9.2f}x")
            out["cases"].append({"L": L, "op": f"layer_norm/{name}/{W}",
                                 "ms": t_n * 1e3, "gbs": gbs,
                                 "floor_ms": t_f * 1e3})
            del x, gw, bw_
            mx.clear_cache()

    print("\n  Read it like this: any LayerNorm row far above the copy floor "
          "is\n  a kernel problem, not an unavoidable cost. The floor is the "
          "least\n  any op touching this tensor can possibly take.")
    print("  If MODEL PATH is far above the contiguous row, the cost is the "
          "stride,\n  not the normalisation, and one mx.contiguous fixes it.")
    if args.json:
        save_json(args.json, out)
        print(f"\nwrote {args.json}")


def _manual_ln(x, g, b, eps=1e-5):
    f = x.astype(mx.float32)
    mu = mx.mean(f, axis=-1, keepdims=True)
    var = mx.var(f, axis=-1, keepdims=True)
    return (((f - mu) * mx.rsqrt(var + eps)).astype(x.dtype)) * g + b


def cmd_ceilings(args):
    c = measure_ceilings(dtype=args.mx_dtype, gemm_n=args.gemm_n)
    print(render_ceilings(c))
    if args.json:
        save_json(args.json, c.as_dict())


def main():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--dtype", default="bfloat16", choices=list(DTYPES))
    common.add_argument("--top", type=int, default=20)
    common.add_argument("--verdict-top", type=int, default=5)
    common.add_argument("--detail", default=None, choices=["full", "summary"],
                        help="default: full for one length, summary for a sweep")
    common.add_argument("--json", default=None, help="write records here")
    common.add_argument("--raw", action="store_true",
                        help="include every per-length record in the json")
    common.add_argument("--gputrace", default=None,
                        help=".gputrace path (needs MTL_CAPTURE_ENABLED=1)")
    common.add_argument("--trace-phase", default=None)
    common.add_argument("--gemm-n", type=int, default=4096)
    common.add_argument("--root", default="model")
    common.add_argument("--tiny", action="store_true",
                        help="toy dimensions, for validating the harness")
    common.add_argument("--weights", action="store_true",
                        help="load the real checkpoint instead of random init")
    common.add_argument("--chains", type=int, default=1)
    common.add_argument("--shapes", action="store_true",
                        help="print real input shapes and ms/call for the "
                             "slowest layers at the longest length")
    common.add_argument("--skip-lm", action="store_true",
                        help="do not build ESMC (~16 GB); synthesise its "
                             "hidden states and profile trunk + sampler only")

    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("ceilings", parents=[common])
    c.set_defaults(fn=cmd_ceilings, seq_len=[0])

    e = sub.add_parser("esmc", parents=[common])
    e.add_argument("--seq-len", type=int, nargs="+",
                   default=[64, 128, 192, 256, 320, 384, 448, 500])
    e.add_argument("--hidden", type=int, default=2560)
    e.add_argument("--heads", type=int, default=40)
    e.add_argument("--layers", type=int, default=80)
    e.add_argument("--repo", default="biohub/ESMC-6B")
    e.set_defaults(fn=cmd_esmc)

    f = sub.add_parser("esmfold2", parents=[common])
    f.add_argument("--seq-len", type=int, nargs="+",
                   default=[64, 128, 192, 256, 320, 384, 448, 500])
    f.add_argument("--atoms-per-token", type=int, default=8)
    f.add_argument("--loops", type=int, default=3)
    f.add_argument("--steps", type=int, default=14)
    f.add_argument("--heads", type=int, default=40)
    f.add_argument("--repo", default="biohub/ESMFold2-Fast")
    f.add_argument("--config", default=None, help="local config.json")
    f.set_defaults(fn=cmd_esmfold2)

    mi = sub.add_parser("micro", parents=[common])
    mi.add_argument("--seq-len", type=int, nargs="+", default=[256, 500])
    mi.add_argument("--c-z", type=int, default=256)
    mi.add_argument("--iters", type=int, default=20)
    mi.add_argument("--ln-widths", type=int, nargs="+",
                    default=[256, 512, 1024])
    mi.set_defaults(fn=cmd_micro)

    args = p.parse_args()
    args.mx_dtype = DTYPES[args.dtype]
    args.seq_len = sorted(set(args.seq_len))
    t0 = time.perf_counter()
    args.fn(args)
    print(f"\ntotal harness time {time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    main()
