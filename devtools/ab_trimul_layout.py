"""Paired A/B of the two TriMul layouts on a production-sized trunk.

The expected gain is ~5%, which is the cross-session noise floor, so comparing
two separate runs cannot resolve it. This runs both variants in ONE process,
alternating A/B/A/B, and reports the ratio per pair. Machine drift affects both
halves of a pair equally, so the ratio is far more stable than either time.

    python devtools/ab_trimul_layout.py [L] [pairs]

Defaults to L=500 and 5 pairs. Builds FoldingTrunk(24, d_pair=256) directly,
so no checkpoint and no network are needed. One measurement runs the stack 4
times, mirroring num_loops=3, giving 192 TriMul calls against the real 204.

The trap this avoids: FoldingTrunk caches mx.compile(self._apply_blocks) in
__init__. Patching TriangleMultiplicativeUpdate after that leaves the traced
graph untouched, so both halves would measure the same code. The compiled
function is rebuilt after every patch.
"""

import statistics
import sys
import time

import mlx.core as mx

from mlx_lm.models.esmfold2 import FoldingTrunk, TriangleMultiplicativeUpdate

L = int(sys.argv[1]) if len(sys.argv) > 1 else 500
PAIRS = int(sys.argv[2]) if len(sys.argv) > 2 else 5
D_PAIR, N_LAYERS, PASSES = 256, 24, 4


# --- the two variants, written out so neither depends on the checkout -----


def contract_old(self, left, right):
    """(B, i, j, C) in. Permuting C to the batch axis leaves neither inner
    axis unit-stride, so MLX copies both operands."""
    perm = (0, 3, 1, 2) if self.outgoing else (0, 3, 2, 1)
    l = left.transpose(*perm)
    r = right.transpose(*perm)
    return (l @ r.transpose(0, 1, 3, 2)).transpose(0, 2, 3, 1)


def call_old(self, z, mask=None):
    normalized = self.norm_start(z)
    bundled = self.proj_bundle(normalized)
    signal, gate_logits = mx.split(bundled, 2, axis=-1)
    routed = signal * mx.sigmoid(gate_logits)
    if mask is not None:
        routed = routed * mask[..., None]
    left, right = mx.split(routed.astype(mx.bfloat16), 2, axis=-1)
    contracted = contract_old(self, left, right).astype(z.dtype)
    mixed = self.proj_emit(self.norm_mix(contracted))
    return mixed * mx.sigmoid(self.proj_gate(normalized))


def contract_new(self, left, right):
    """(C, B, i, j) in. Only batch axes move, so nothing is copied."""
    perm = (1, 0, 2, 3) if self.outgoing else (1, 0, 3, 2)
    l = left.transpose(*perm)
    r = right.transpose(*perm)
    return (l @ r.transpose(0, 1, 3, 2)).transpose(0, 2, 3, 1)


def call_new(self, z, mask=None):
    normalized = self.norm_start(z)
    b, n = normalized.shape[0], normalized.shape[1]
    bundled = (
        self.proj_bundle.weight @ normalized.reshape(-1, self.dim).T
    ).reshape(4 * self.dim, b, n, n)
    signal, gate_logits = mx.split(bundled, 2, axis=0)
    routed = signal * mx.sigmoid(gate_logits)
    if mask is not None:
        routed = routed * mask[None]
    left, right = mx.split(routed.astype(mx.bfloat16), 2, axis=0)
    contracted = contract_new(self, left, right).astype(z.dtype)
    mixed = self.proj_emit(self.norm_mix(contracted))
    return mixed * mx.sigmoid(self.proj_gate(normalized))


VARIANTS = {"old": call_old, "new": call_new}


def select(trunk, which):
    TriangleMultiplicativeUpdate.__call__ = VARIANTS[which]
    # Must rebuild: the cached trace still holds the previous variant.
    trunk._compiled = mx.compile(trunk._apply_blocks)


def run(trunk, z, mask):
    out = z
    for _ in range(PASSES):
        out = trunk(out, mask=mask)
    mx.eval(out)
    return out


def measure(trunk, z, mask, which):
    select(trunk, which)
    run(trunk, z, mask)          # warm up and compile
    mx.synchronize()
    t0 = time.perf_counter()
    run(trunk, z, mask)
    mx.synchronize()
    return time.perf_counter() - t0


def main():
    mx.random.seed(0)
    trunk = FoldingTrunk(n_layers=N_LAYERS, d_pair=D_PAIR)
    trunk.set_dtype(mx.bfloat16)
    trunk.eval()
    z = mx.random.normal((1, L, L, D_PAIR)).astype(mx.float32)
    mask = mx.ones((1, L, L)).astype(mx.float32)
    mx.eval(trunk.parameters(), z, mask)

    print(f"L={L}  d_pair={D_PAIR}  blocks={N_LAYERS}  passes={PASSES}  "
          f"-> {N_LAYERS*PASSES*2} TriMul calls per measurement")
    print(f"pairs={PAIRS}  dtype=bfloat16\n")

    # equivalence first: a speedup on a different answer is worthless
    select(trunk, "old")
    a = run(trunk, z, mask).astype(mx.float32)
    select(trunk, "new")
    b = run(trunk, z, mask).astype(mx.float32)
    mx.eval(a, b)
    dmax = float(mx.max(mx.abs(a - b)))
    rel = float(mx.sum(mx.abs(a - b)) / mx.maximum(mx.sum(mx.abs(a)), 1e-12))
    print(f"equivalence: max abs diff {dmax:.3e}   rel {rel:.3e}"
          f"   {'IDENTICAL' if dmax == 0 else 'differs'}")
    del a, b
    mx.clear_cache()

    print(f"\n{'pair':>5}{'old s':>10}{'new s':>10}{'ratio':>9}{'gain %':>9}")
    print("-" * 43)
    olds, news, ratios = [], [], []
    for i in range(PAIRS):
        t_old = measure(trunk, z, mask, "old")
        t_new = measure(trunk, z, mask, "new")
        olds.append(t_old)
        news.append(t_new)
        ratios.append(t_old / t_new)
        print(f"{i+1:>5}{t_old:>10.3f}{t_new:>10.3f}{ratios[-1]:>8.3f}x"
              f"{100*(1-t_new/t_old):>9.1f}")

    print("-" * 43)
    med = statistics.median(ratios)
    spread = (max(ratios) - min(ratios)) / min(ratios)
    print(f"{'median':>5}{statistics.median(olds):>10.3f}"
          f"{statistics.median(news):>10.3f}{med:>8.3f}x"
          f"{100*(1-1/med):>9.1f}")
    print(f"\nratio spread across pairs: {100*spread:.1f}%")
    if PAIRS >= 3:
        print(f"ratio stdev              : {statistics.stdev(ratios):.4f}")
    if min(ratios) > 1.0:
        print("=> every pair favours the new layout. Real.")
    elif max(ratios) < 1.0:
        print("=> every pair favours the old layout. Discard the branch.")
    else:
        print("=> pairs disagree. Inside noise; raise --pairs or L.")
    print(f"\npeak memory {mx.get_peak_memory()/2**30:.2f} GB")


if __name__ == "__main__":
    main()
