"""End-to-end fold cost from 100 to 1000 residues, with memory split by part.

Everything else in this harness runs `--skip-lm`, which is right for trunk
work but never shows what a fold actually costs. This builds the whole thing,
ESMC included, and separates the two halves:

  time   the ESMC encode, then the trunk and sampler
  memory the weights of each model, and the activation peak of each phase

Weights are counted at build time from `mx.get_active_memory()`, so the
activation peak of a phase is its peak minus the weights that were already
resident. Random weights: timing and footprint follow shapes and dtype, not
values.

Peak grows as L^2, so a projected length that would exceed the device's
recommended working set is skipped rather than left to swap. `--max-peak-gb`
overrides.
"""

from __future__ import annotations

import time

import mlx.core as mx

from .esm_profiler import memory_info, save_json

GB = 2 ** 30


def _synth(L, args):
    from .esm_profile import feats_from_args

    dtype = args.mx_dtype
    feats = {k: (v.astype(dtype) if v.dtype in
                 (mx.float32, mx.float16, mx.bfloat16) else v)
             for k, v in feats_from_args(L, args).items()}
    mx.eval(list(feats.values()))
    return feats


def _phase(fn):
    """Time a phase and report its peak, with the graph actually forced."""
    from .esm_profiler import _force

    _force(fn())          # warm up and compile outside the measurement
    mx.synchronize()
    mx.clear_cache()
    mx.reset_peak_memory()
    t0 = time.perf_counter()
    out = fn()
    _force(out)
    mx.synchronize()
    dt = time.perf_counter() - t0
    return dt, mx.get_peak_memory() / GB, out


def cmd_fold(args):
    from .models import esmc, esmfold2

    info = memory_info()
    rec_gb = info.get("max_recommended_working_set_size", 0) / GB
    cap = args.max_peak_gb or (rec_gb * 0.95 if rec_gb else 1e9)

    # --- build, measuring each model's weights separately ---------------
    mx.clear_cache()
    base = mx.get_active_memory()
    if args.config:
        import json
        cfg = json.load(open(args.config))
    else:
        import json

        from huggingface_hub import hf_hub_download
        cfg = json.load(open(hf_hub_download(args.repo, "config.json")))

    mx.random.seed(0)
    model = esmfold2.ESMFold2Model(cfg)
    model.set_dtype(args.mx_dtype)
    model.eval()
    mx.eval(model.parameters())
    w_fold = (mx.get_active_memory() - base) / GB

    model._esmc = esmc.Model(esmc.ModelArgs(
        hidden_size=cfg.get("lm_d_model", 2560),
        num_attention_heads=args.heads,
        num_hidden_layers=cfg.get("lm_num_layers", 80)))
    model._esmc.set_dtype(args.mx_dtype)
    model._esmc.eval()
    mx.eval(model._esmc.parameters())
    w_esmc = (mx.get_active_memory() - base) / GB - w_fold

    print(f"device {info.get('architecture', '?')}   "
          f"unified {info.get('memory_size', 0)/GB:.0f} GB   "
          f"recommended working set {rec_gb:.2f} GB")
    from .esm_profile import describe_input
    print(f"dtype={args.dtype}  loops={args.loops}  steps={args.steps}  "
          f"atoms/token={args.atoms_per_token}  weights=random")
    print(f"input: {describe_input(args)}\n")
    print("== weights resident")
    print(f"  ESMFold2 {w_fold:8.2f} GB")
    print(f"  ESMC     {w_esmc:8.2f} GB")
    print(f"  total    {w_fold + w_esmc:8.2f} GB\n")

    rows, projected = [], None
    for L in args.seq_len:
        if projected is not None and projected > cap:
            print(f"  L={L:<5} SKIPPED: projected peak {projected:.1f} GB "
                  f"exceeds the {cap:.1f} GB cap")
            continue
        feats = _synth(L, args)
        lm_kw = dict(asym_id=feats["asym_id"],
                     residue_index=feats["residue_index"],
                     mol_type=feats["mol_type"],
                     token_mask=feats["token_attention_mask"])

        t_lm, p_lm, lm = _phase(
            lambda: model.compute_lm_hidden_states(feats["input_ids"], **lm_kw))
        t_tr, p_tr, tz = _phase(
            lambda: model.trunk(feats, lm, num_loops=args.loops))
        z, x_inputs, aux = tz

        def sampler():
            mx.random.seed(0)
            return model.structure_head.sample(
                z_trunk=z, s_inputs=x_inputs,
                relative_position_encoding=aux["relpos"],
                ref_pos=aux["ref_pos"], ref_charge=aux["ref_charge"],
                ref_mask=aux["ref_mask"], ref_element=aux["ref_element_oh"],
                ref_atom_name_chars=aux["ref_name_oh"],
                ref_space_uid=aux["ref_space_uid"],
                tok_idx=aux["atom_to_token"], n_tokens=aux["n_tokens"],
                token_attention_mask=aux["tok_mask"],
                num_diffusion_samples=1, num_sampling_steps=args.steps)

        t_sa, p_sa, _ = _phase(sampler)
        p_fold = max(p_tr, p_sa)
        rows.append({"L": L, "atoms": L * args.atoms_per_token,
                     "esmc_s": t_lm, "trunk_s": t_tr, "sampler_s": t_sa,
                     "fold_s": t_tr + t_sa, "total_s": t_lm + t_tr + t_sa,
                     "esmc_peak_gb": p_lm, "fold_peak_gb": p_fold,
                     "peak_gb": max(p_lm, p_fold),
                     "w_fold_gb": w_fold, "w_esmc_gb": w_esmc})
        print(f"  L={L:<5} ESMC {t_lm:6.2f}s  trunk {t_tr:7.2f}s  "
              f"sampler {t_sa:5.2f}s  total {rows[-1]['total_s']:7.2f}s  "
              f"peak {rows[-1]['peak_gb']:5.2f} GB")
        del feats, lm, z, x_inputs, aux, tz
        mx.clear_cache()
        # peak is driven by the pair tensor, so project the next L by L^2
        if len(args.seq_len) > 1:
            nxt = [x for x in args.seq_len if x > L]
            if nxt:
                act = rows[-1]["peak_gb"] - (w_fold + w_esmc)
                projected = (w_fold + w_esmc) + act * (nxt[0] / L) ** 2

    if not rows:
        return

    print("\n== fold cost, seconds")
    hdr = (f"{'L':>6}{'atoms':>8}{'ESMC':>9}{'trunk':>10}{'sampler':>9}"
           f"{'FOLD':>10}{'TOTAL':>10}{'ESMC %':>8}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['L']:>6}{r['atoms']:>8}{r['esmc_s']:>9.2f}"
              f"{r['trunk_s']:>10.2f}{r['sampler_s']:>9.2f}"
              f"{r['fold_s']:>10.2f}{r['total_s']:>10.2f}"
              f"{100*r['esmc_s']/r['total_s']:>7.0f}%")

    print("\n== memory, GB")
    hdr2 = (f"{'L':>6}{'ESMC w':>9}{'fold w':>9}{'ESMC act':>10}"
            f"{'fold act':>10}{'ESMC peak':>11}{'fold peak':>11}{'PEAK':>8}")
    print(hdr2)
    print("-" * len(hdr2))
    for r in rows:
        w = r["w_fold_gb"] + r["w_esmc_gb"]
        print(f"{r['L']:>6}{r['w_esmc_gb']:>9.2f}{r['w_fold_gb']:>9.2f}"
              f"{max(r['esmc_peak_gb']-w, 0):>10.2f}"
              f"{max(r['fold_peak_gb']-w, 0):>10.2f}"
              f"{r['esmc_peak_gb']:>11.2f}{r['fold_peak_gb']:>11.2f}"
              f"{r['peak_gb']:>8.2f}")
    print("\n  'act' is the phase peak minus the resident weights, so it is")
    print("  what the activations cost. Weights are constant in L.")

    if len(rows) >= 2:
        from .esm_profiler import fit_exponent
        print("\n== scaling")
        Ls = [r["L"] for r in rows]
        for key, label in (("esmc_s", "ESMC"), ("trunk_s", "trunk"),
                           ("sampler_s", "sampler"), ("total_s", "TOTAL"),
                           ("peak_gb", "peak memory")):
            k = fit_exponent(Ls, [r[key] for r in rows])
            if k is not None:
                print(f"  {label:<14} L^{k:.2f}")

    if rec_gb:
        worst = max(r["peak_gb"] for r in rows)
        print(f"\n  worst peak {worst:.2f} GB of {rec_gb:.2f} GB recommended"
              f"  ({100*worst/rec_gb:.0f}%)")
    if args.json:
        save_json(args.json, {"weights": {"fold_gb": w_fold,
                                          "esmc_gb": w_esmc},
                              "memory": info, "rows": rows})
        print(f"\nwrote {args.json}")
