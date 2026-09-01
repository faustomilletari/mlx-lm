# Copyright © 2026 Apple Inc.

"""Where does an ESMFold2 fold actually spend its time?

    python benchmarks/esmfold2_profile.py --seq-len 460
    python benchmarks/esmfold2_profile.py --sequence MYSEQ... --steps 50 14

Splits one fold into the three phases that matter -- the ESMC-6B forward, the
folding trunk, and the diffusion sampler -- with an mx.eval barrier between
each, and reports wall time and peak memory for each. Sweeping --steps shows
what the sampler actually costs, since the checkpoint config asks for
inference_num_steps=14 while callers often pass 50.

Needs the checkpoint and the `esm` featurizer. Timings are only meaningful on
Metal.
"""

import argparse
import time

import mlx.core as mx


def phase(label, fn, results):
    mx.synchronize()
    mx.reset_peak_memory()
    t = time.perf_counter()
    out = fn()
    mx.eval(out)
    mx.synchronize()
    dt = time.perf_counter() - t
    results.append((label, dt, mx.get_peak_memory() / 2**30))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="biohub/ESMFold2-Fast")
    ap.add_argument("--sequence", default=None)
    ap.add_argument("--seq-len", type=int, default=460, help="synthetic poly-A length")
    ap.add_argument("--loops", type=int, default=3)
    ap.add_argument("--steps", type=int, nargs="+", default=[50, 14])
    args = ap.parse_args()

    from esm.models.esmfold2 import (
        ESMFold2InputBuilder,
        ProteinInput,
        StructurePredictionInput,
    )

    from mlx_lm.models.esmfold2 import ESMFold2Model

    seq = args.sequence or "A" * args.seq_len
    print(f"device={mx.default_device()}  L={len(seq)}  loops={args.loops}\n")

    model = ESMFold2Model.from_pretrained(args.repo)
    builder = ESMFold2InputBuilder()
    spi = StructurePredictionInput(sequences=[ProteinInput(id="A", sequence=seq)])
    feats_torch, _ = builder.prepare_input(spi, seed=0, device=model.device)
    feats = model._to_mlx_features(feats_torch) if hasattr(
        model, "_to_mlx_features"
    ) else _to_mlx(feats_torch, model.z_init_1.weight.dtype)

    results = []
    lm = phase(
        "ESMC-6B (once)",
        lambda: model.compute_lm_hidden_states(
            feats["input_ids"],
            asym_id=feats.get("asym_id"),
            residue_index=feats.get("residue_index"),
            mol_type=feats.get("mol_type"),
            token_mask=feats.get("token_attention_mask"),
        ),
        results,
    )
    z, x_inputs, aux = phase(
        f"trunk ({args.loops} loops x 24 blocks)",
        lambda: model.trunk(feats, lm, num_loops=args.loops),
        results,
    )

    for steps in args.steps:
        mx.random.seed(0)
        phase(
            f"sampler ({steps} steps)",
            lambda s=steps: model.structure_head.sample(
                z_trunk=z,
                s_inputs=x_inputs,
                relative_position_encoding=aux["relpos"],
                ref_pos=aux["ref_pos"],
                ref_charge=aux["ref_charge"],
                ref_mask=aux["ref_mask"],
                ref_element=aux["ref_element_oh"],
                ref_atom_name_chars=aux["ref_name_oh"],
                ref_space_uid=aux["ref_space_uid"],
                tok_idx=aux["atom_to_token"],
                n_tokens=aux["n_tokens"],
                token_attention_mask=aux["tok_mask"],
                num_diffusion_samples=1,
                num_sampling_steps=s,
            ),
            results,
        )

    n_atoms = int(aux["ref_mask"].sum().item())
    print(f"atoms = {n_atoms}  ({n_atoms / len(seq):.1f} per residue)\n")
    hdr = f"{'phase':<30}{'seconds':>10}{'peak GB':>10}"
    print(hdr)
    print("-" * len(hdr))
    for label, dt, gb in results:
        print(f"{label:<30}{dt:>10.2f}{gb:>10.2f}")

    base = dict((l, d) for l, d, _ in results)
    lm_t = base["ESMC-6B (once)"]
    tr_t = base[f"trunk ({args.loops} loops x 24 blocks)"]
    for steps in args.steps:
        total = lm_t + tr_t + base[f"sampler ({steps} steps)"]
        print(f"\nfold total @ {steps:>3} steps: {total:6.2f}s   "
              f"LM {100*lm_t/total:4.1f}%  trunk {100*tr_t/total:4.1f}%  "
              f"sampler {100*base[f'sampler ({steps} steps)']/total:4.1f}%")


def _to_mlx(features_torch, dtype):
    INT_KEYS = {
        "token_index", "residue_index", "asym_id", "sym_id", "entity_id", "mol_type",
        "res_type", "input_ids", "ref_element", "ref_atom_name_chars", "atom_to_token",
        "distogram_atom_idx", "msa",
    }
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


if __name__ == "__main__":
    main()
