"""Profile the MLX ESM models layer by layer.

    python -m mlx_lm.esm_profile ceilings
    python -m mlx_lm.esm_profile esmc     --seq-len 578
    python -m mlx_lm.esm_profile esmfold2 --repo biohub/ESMFold2-Fast --seq-len 578

Weights are not needed. Performance depends on shapes and dtype, not values,
so by default the model is built from its config with random init. That skips
a multi-GB download and lets a profile run anywhere. Pass --weights to load
the real checkpoint when you want to confirm.

Inputs are synthesised, following tests/test_models.py::test_esmfold2. One
caveat worth knowing: the synthetic token mask has no padding, and ESMC
short-circuits its attention mask to None when every token shares a chain id.
A real padded or multi-chain input therefore takes a slightly different path.
Use --chains 2 to exercise the masked path.
"""

from __future__ import annotations

import argparse
import json
import time

import mlx.core as mx

from .esm_profiler import (
    LayerProfiler,
    capture,
    measure_ceilings,
    render,
    render_ceilings,
    render_modes,
    save_json,
)

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
# driver
# ---------------------------------------------------------------------------


def _run_phase(label, model, fn, args, ceilings, out):
    """Warm up, then profile the same call in eval mode and lazy mode."""
    mx.eval(fn())          # warm up: first call pays JIT and allocation
    mx.synchronize()
    mx.clear_cache()

    prof = LayerProfiler(mode="eval").attach(model, root=args.root)
    with prof:
        fn()
    eval_wall = prof.wall_s

    lazy = LayerProfiler(mode="lazy", record_shapes=False).attach(
        model, root=args.root)
    with lazy:
        r = lazy_out = fn()
    lazy_wall = lazy.wall_s
    mx.eval(lazy_out)
    mx.synchronize()
    del r, lazy_out

    print(f"\n{'#'*78}\n# {label}   (peak {mx.get_peak_memory()/2**30:.2f} GB)\n{'#'*78}")
    print(render_modes(eval_wall, lazy_wall))
    print()
    print(render(prof.active(), ceilings, total_s=None, top=args.top,
                 title=f"{label}: slowest layers by exclusive device time"))
    print()
    print(render(list(prof.by_class().values()), ceilings, top=args.top,
                 title=f"{label}: rolled up by module class", label="class"))
    print()
    print(render(lazy.active(), None, top=min(args.top, 12),
                 title=f"{label}: graph-build cost only (no device work)"))
    out[label] = {"eval": prof.as_dict(), "lazy": lazy.as_dict(),
                  "peak_gb": mx.get_peak_memory() / 2**30}
    return prof


def load_esmfold2(args):
    from .models import esmc, esmfold2
    if args.tiny:
        cfg, model = TINY_CONFIG, None
    elif args.config:
        cfg = json.load(open(args.config))
    else:
        from huggingface_hub import hf_hub_download
        cfg = json.load(open(hf_hub_download(args.repo, "config.json")))

    if args.weights:
        return esmfold2.ESMFold2Model.from_pretrained(args.repo, dtype=args.mx_dtype)
    model = esmfold2.ESMFold2Model(cfg)
    model._esmc = esmc.Model(
        esmc.ModelArgs(**TINY_ESMC) if args.tiny
        else esmc.ModelArgs(hidden_size=cfg.get("lm_d_model", 2560),
                            num_attention_heads=40,
                            num_hidden_layers=cfg.get("lm_num_layers", 80)))
    model.set_dtype(args.mx_dtype)
    model.eval()
    mx.eval(model.parameters())
    return model


def cmd_esmfold2(args):
    model = load_esmfold2(args)
    feats = synth_feats(args.seq_len, args.atoms_per_token, args.chains)
    feats = {k: (v.astype(args.mx_dtype)
                 if v.dtype in (mx.float32, mx.float16, mx.bfloat16) else v)
             for k, v in feats.items()}
    mx.eval(list(feats.values()))

    ceilings = measure_ceilings(dtype=args.mx_dtype, gemm_n=args.gemm_n)
    print(render_ceilings(ceilings))
    print(f"\nL={args.seq_len}  atoms={args.seq_len*args.atoms_per_token}  "
          f"loops={args.loops}  steps={args.steps}  dtype={args.dtype}  "
          f"weights={'real' if args.weights else 'random'}")
    out = {"cmd": "esmfold2", "ceilings": ceilings.as_dict(),
           "seq_len": args.seq_len, "loops": args.loops,
           "atoms_per_token": args.atoms_per_token, "phases": {}}

    mx.reset_peak_memory()
    lm = model.compute_lm_hidden_states(
        feats["input_ids"], asym_id=feats.get("asym_id"),
        residue_index=feats.get("residue_index"), mol_type=feats.get("mol_type"),
        token_mask=feats.get("token_attention_mask"))
    mx.eval(lm)
    _run_phase("phase 1/3  ESMC language model", model,
               lambda: model.compute_lm_hidden_states(
                   feats["input_ids"], asym_id=feats.get("asym_id"),
                   residue_index=feats.get("residue_index"),
                   mol_type=feats.get("mol_type"),
                   token_mask=feats.get("token_attention_mask")),
               args, ceilings, out["phases"])

    mx.reset_peak_memory()
    _run_phase(f"phase 2/3  trunk ({args.loops} loops)", model,
               lambda: model.trunk(feats, lm, num_loops=args.loops),
               args, ceilings, out["phases"])

    z, x_inputs, aux = model.trunk(feats, lm, num_loops=args.loops)
    mx.eval(z, x_inputs)
    mx.reset_peak_memory()

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

    _run_phase(f"phase 3/3  diffusion sampler ({args.steps} steps)", model,
               sampler, args, ceilings, out["phases"])

    if args.gputrace:
        with capture(args.gputrace) as ok:
            if ok:
                mx.eval(model.trunk(feats, lm, num_loops=1))
        print(f"\nwrote {args.gputrace}" if ok else "\nno Metal device; no trace")
    if args.json:
        save_json(args.json, out)
        print(f"\nwrote {args.json}")


def cmd_esmc(args):
    from .models import esmc
    if args.weights:
        model = esmc.from_pretrained(args.repo, dtype=args.mx_dtype)
    else:
        cfg = dict(TINY_ESMC) if args.tiny else dict(
            hidden_size=args.hidden, num_attention_heads=args.heads,
            num_hidden_layers=args.layers)
        model = esmc.Model(esmc.ModelArgs(**cfg))
        model.set_dtype(args.mx_dtype)
        model.eval()
        mx.eval(model.parameters())

    a = model.args
    ceilings = measure_ceilings(dtype=args.mx_dtype, gemm_n=args.gemm_n)
    print(render_ceilings(ceilings))
    print(f"\nL={args.seq_len}  hidden={a.hidden_size}  heads={a.num_attention_heads}"
          f"  layers={a.num_hidden_layers}  ffn={a.ffn_hidden}  dtype={args.dtype}"
          f"  weights={'real' if args.weights else 'random'}")

    ids = mx.random.randint(4, 24, (1, args.seq_len))
    per = max(args.seq_len // args.chains, 1)
    amask = mx.ones((1, args.seq_len), mx.bool_) if args.chains == 1 else None
    sid = (None if args.chains == 1
           else mx.minimum(mx.arange(args.seq_len) // per, args.chains - 1)[None])
    mx.eval(ids)
    out = {"cmd": "esmc", "ceilings": ceilings.as_dict(),
           "seq_len": args.seq_len, "config": {
               "hidden": a.hidden_size, "heads": a.num_attention_heads,
               "layers": a.num_hidden_layers, "ffn": a.ffn_hidden},
           "phases": {}}
    mx.reset_peak_memory()
    _run_phase(f"ESMC encode (L={args.seq_len})", model,
               lambda: model.encode(ids, attention_mask=amask, sequence_id=sid),
               args, ceilings, out["phases"])
    if args.gputrace:
        with capture(args.gputrace) as ok:
            if ok:
                mx.eval(model.encode(ids, attention_mask=amask, sequence_id=sid))
        print(f"\nwrote {args.gputrace}" if ok else "\nno Metal device; no trace")
    if args.json:
        save_json(args.json, out)
        print(f"\nwrote {args.json}")


def cmd_ceilings(args):
    c = measure_ceilings(dtype=args.mx_dtype, gemm_n=args.gemm_n)
    print(render_ceilings(c))
    if args.json:
        save_json(args.json, c.as_dict())


DTYPES = {"bfloat16": mx.bfloat16, "float16": mx.float16, "float32": mx.float32}


def main():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--dtype", default="bfloat16", choices=list(DTYPES))
    common.add_argument("--top", type=int, default=25)
    common.add_argument("--json", default=None, help="write raw records here")
    common.add_argument("--gputrace", default=None,
                        help="also write a .gputrace (needs MTL_CAPTURE_ENABLED=1)")
    common.add_argument("--gemm-n", type=int, default=4096)
    common.add_argument("--root", default="model")
    common.add_argument("--tiny", action="store_true",
                        help="toy dimensions, for validating the harness")
    common.add_argument("--weights", action="store_true",
                        help="load the real checkpoint instead of random init")
    common.add_argument("--chains", type=int, default=1)

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("ceilings", parents=[common])
    c.set_defaults(fn=cmd_ceilings, seq_len=0)

    e = sub.add_parser("esmc", parents=[common])
    e.add_argument("--seq-len", type=int, default=578)
    e.add_argument("--hidden", type=int, default=2560)
    e.add_argument("--heads", type=int, default=40)
    e.add_argument("--layers", type=int, default=80)
    e.add_argument("--repo", default="biohub/ESMC-6B")
    e.set_defaults(fn=cmd_esmc)

    f = sub.add_parser("esmfold2", parents=[common])
    f.add_argument("--seq-len", type=int, default=578)
    f.add_argument("--atoms-per-token", type=int, default=8)
    f.add_argument("--loops", type=int, default=3)
    f.add_argument("--steps", type=int, default=14)
    f.add_argument("--repo", default="biohub/ESMFold2-Fast")
    f.add_argument("--config", default=None, help="local config.json")
    f.set_defaults(fn=cmd_esmfold2)

    args = p.parse_args()
    args.mx_dtype = DTYPES[args.dtype]
    t0 = time.perf_counter()
    args.fn(args)
    print(f"\ntotal harness time {time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    main()
