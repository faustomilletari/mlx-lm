"""Does the bf16 trunk change the answer?

Runs the same tiny fold twice in one process, fp32 blocks then bf16 blocks,
and reports the divergence. bf16 has ~3 decimal digits, so a relative
difference around 1e-2 on a 24-block residual stack is expected. Anything
structural -- NaN, inf, a dtype that does not round-trip, or a relative error
near 1 -- is a bug, not rounding.
"""
import os
import sys

import mlx.core as mx

sys.path.insert(0, ".")
from mlx_lm.esm_profile import TINY_CONFIG, TINY_ESMC, synth_feats  # noqa: E402
from mlx_lm.models import esmc, esmfold2  # noqa: E402

L, APT, LOOPS = 24, 2, 2
mx.random.seed(0)

model = esmfold2.ESMFold2Model(TINY_CONFIG)
model._esmc = esmc.Model(esmc.ModelArgs(**TINY_ESMC))
model.set_dtype(mx.bfloat16)
model.eval()
mx.eval(model.parameters())

feats = synth_feats(L, APT, 1)
feats = {k: (v.astype(mx.bfloat16)
             if v.dtype in (mx.float32, mx.float16, mx.bfloat16) else v)
         for k, v in feats.items()}
mx.eval(list(feats.values()))
lm = mx.random.normal((1, L, TINY_CONFIG["lm_num_layers"] + 1,
                       TINY_CONFIG["lm_d_model"])).astype(mx.bfloat16)
mx.eval(lm)

out = {}
for flag in (False, True):
    esmfold2._BF16_TRUNK = flag
    mx.random.seed(0)
    z, x_inputs, aux = model.trunk(feats, lm, num_loops=LOOPS)
    mx.eval(z)
    out[flag] = z
    print(f"_BF16_TRUNK={str(flag):<5}  z.dtype={z.dtype}  "
          f"finite={bool(mx.all(mx.isfinite(z.astype(mx.float32))).item())}")

a = out[False].astype(mx.float32)
b = out[True].astype(mx.float32)
den = mx.maximum(mx.abs(a), 1e-6)
rel = mx.abs(a - b) / den
print()
print(f"dtype round-trips      : {out[False].dtype == out[True].dtype}")
print(f"max |fp32|             : {float(mx.max(mx.abs(a))):.5f}")
print(f"max abs diff           : {float(mx.max(mx.abs(a - b))):.6f}")
print(f"mean abs diff          : {float(mx.mean(mx.abs(a - b))):.6f}")
print(f"max relative diff      : {float(mx.max(rel)):.5f}")
print(f"mean relative diff     : {float(mx.mean(rel)):.5f}")
corr_num = float(mx.mean((a - mx.mean(a)) * (b - mx.mean(b))).item())
corr_den = (float(mx.sqrt(mx.var(a)).item()) * float(mx.sqrt(mx.var(b)).item()))
print(f"correlation            : {corr_num/max(corr_den,1e-12):.6f}")
