# ESMFold2 MLX performance ledger

Every measurement taken on this branch, so a future change is compared against
recorded numbers rather than anyone's recollection. Append, never overwrite.

This branch is instrumentation only and never merges. Optimisations go on their
own branches off main, carrying only the change that buys the speedup.

## Machine

48 GB M4 Pro, `applegpu_g16s`. Max recommended working set 37.44 GB.
Spec bandwidth 273 GB/s; the streaming probe reaches ~190 GB/s, about 71%.

| Quantity | Range across runs |
|---|---|
| Streaming bandwidth | 174 - 214 GB/s, typically ~190 |
| bf16 GEMM | 5.95 - 6.04 TFLOP/s |
| Ridge point | 31.5 - 32.4 FLOP/byte |

Target hardware is an M5 Ultra (~1.2 TB/s, compute unknown). The M4 Pro is a
biased proxy: bandwidth differs ~5.7x but compute ~13x, so the proxy
understates memory-side wins and overstates compute-side ones. Use
`--target M5-Ultra 1200 80` and read arithmetic intensity, not percentages.

## Canonical command

```
python -m mlx_lm.esm_profile esmfold2 --skip-lm --detail summary \
  --seq-len 500 --steps 14 --target M5-Ultra 1200 80 --target H100 3350 990
```

`loops=3`, `steps=14`, `atoms_per_token=8`, random weights, synthetic inputs.
Always `--skip-lm` unless profiling the LM: ESMC at real shape is ~16 GB of
bf16 weights and pushes peak from 7.7 GB to 33 GB.

## Headline

| Stage | trunk | sampler | total | peak | vs main |
|---|---|---|---|---|---|
| 0. main, fp32 pair stack | 25.013s | 1.116s | 26.129s | 7.71 GB | — |
| 1. + bf16 FoldingTrunk | 21.200s | 1.115s | 22.314s | 5.33 GB | 1.18x |
| 2. + bf16 TriMul contraction | **20.108s** | 1.114s | **21.222s** | **4.64 GB** | **1.244x** |

The 1.244x compares across sessions. The paired same-session sweep below gives
**1.300x** at L=500, which is the number to quote.

Stages 1 and 2 are now merged to main (PRs #6 and #7). Structural output at
stage 2 is bit-identical to stage 1: once the trunk is bf16, the old
`routed.astype(mx.float32)` is a lossless upcast and MLX accumulates the matmul
in fp32 either way.

## Per-class detail at L=500

Exclusive device time. `container` means children did the work.

| Class | Stage 0 (fp32) | Stage 1 (bf16 trunk) | Stage 2 (both) | Status at stage 2 |
|---|---|---|---|---|
| `Linear` | 16.417s (53.1%) | 14.499s (56.0%) | 14.472s (59.6%) | **94% of GEMM roof** |
| `TriangleMultiplicativeUpdate` | 8.379s (27.1%) | 7.463s (28.8%) | **5.887s (24.3%)** | **next target** |
| `SwiGLUMLP` | 2.207s (7.1%) | 1.147s (4.4%) | 1.150s (4.7%) | fusable |
| `LayerNorm` | 1.900s (6.1%) | 1.178s (4.6%) | 1.186s (4.9%) | 62% of BW roof |
| `PairUpdateBlock` | 0.996s (3.2%) | 0.538s (2.1%) | 0.539s (2.2%) | residual adds |
| `SWA3DRoPEAttention` | 0.453s (1.5%) | 0.456s (1.8%) | 0.454s (1.9%) | 0.6 GB/s, only `HEADROOM` left |

`Linear` at 94% of roof, `w12` individually at 99%. Untouchable without
changing the maths.

| | Stage 0 | Stage 2 |
|---|---|---|
| Trunk traffic | 1308.83 GB f32 (97.8%) | 681.67 GB bf16 (99.4%) |
| `mx.compile` benefit | 1.13x | 1.09x |
| Sampler traffic | 98.8% f32 | **still 98.8% f32** |

## Length sweep, stage 0 only

With ESMC, `steps=14`:

| L | lm | trunk | sampler | total | peak |
|---|---|---|---|---|---|
| 128 | 0.650s | 1.550s | 0.732s | 2.933s | 26.14 GB |
| 192 | 0.846s | 3.530s | 0.855s | 5.231s | 26.37 GB |
| 256 | 1.058s | 6.415s | 0.979s | 8.453s | 27.20 GB |
| 320 | 1.523s | 10.046s | 1.111s | 12.680s | 28.64 GB |
| 384 | 1.737s | 14.564s | 1.323s | 17.624s | 29.36 GB |
| 448 | 1.845s | 20.103s | 1.526s | 23.473s | 31.24 GB |
| 500 | 1.924s | 25.668s | 1.712s | 29.304s | 32.98 GB |

Fits: total L^1.70, `lm` L^0.86, **trunk L^2.05**, sampler L^0.62.

With `--skip-lm`, four lengths, the reference fp32 sweep. **Pin this by SHA:
`8a8efde`.** `main` no longer reproduces it, since the bf16 PRs landed.

| L | trunk | sampler | total | peak |
|---|---|---|---|---|
| 128 | 1.495s | 0.224s | 1.718s | 2.36 GB |
| 256 | 6.219s | 0.448s | 6.667s | 2.96 GB |
| 384 | 14.470s | 0.756s | 15.226s | 4.87 GB |
| 500 | 26.062s | 1.113s | 27.175s | 7.71 GB |

Fits: total L^2.02, **trunk L^2.09**, sampler L^1.17.

**The trunk scales L^2, not L^3. The O(L^3) contraction is not the limiter.**

### Per-class exponents, fp32, from the same sweep

| Class | L=500 | share | L^k |
|---|---|---|---|
| `Linear` | 16.492s | 53.3% | **1.69** |
| `TriangleMultiplicativeUpdate` | 8.351s | 27.0% | **2.08** |
| `SwiGLUMLP` | 2.184s | 7.1% | 1.89 |
| `LayerNorm` | 1.882s | 6.1% | 1.70 |
| `PairUpdateBlock` | 0.984s | 3.2% | 1.61 |
| `SWA3DRoPEAttention` | 0.453s | 1.5% | 1.52 |

`Linear` below L^2 despite M = L^2 work: GEMM efficiency rises with M, so
short lengths are less efficient. At L=128, M=16384 and `w12` tiles poorly.

`TriMul` is the fastest-growing significant class. Extrapolating to L=1000 its
share goes 27% -> ~32% while `Linear` falls 53% -> ~48%. **Fusing TriMul gets
more valuable at longer chains and on a bigger chip, not less.**

### Paired sweep: fp32 (`8a8efde`) vs bf16 (`0f89403`)

Run back to back in one session, ceilings 188.1 vs 186.8 GB/s (0.7% apart).
**This is the trustworthy speedup measurement**; cross-session numbers carry
the ~4% variance recorded below.

| L | trunk fp32 | trunk bf16 | speedup | total fp32 | total bf16 | speedup | peak fp32 | peak bf16 |
|---|---|---|---|---|---|---|---|---|
| 128 | 1.495s | 1.177s | 1.270x | 1.718s | 1.396s | 1.231x | 2.36 GB | 1.46 GB |
| 256 | 6.219s | 4.886s | 1.273x | 6.667s | 5.321s | 1.253x | 2.96 GB | 1.90 GB |
| 384 | 14.470s | 11.341s | 1.276x | 15.226s | 12.088s | 1.260x | 4.87 GB | 3.04 GB |
| 500 | 26.062s | 20.052s | **1.300x** | 27.175s | 21.163s | **1.284x** | 7.71 GB | 4.64 GB |

Trunk exponent 2.09 -> 2.08, total 2.02 -> 1.99. **bf16 is a constant-factor
win, not a change of scaling**, so it should hold at L=1000 and beyond.
Speedup is flat-to-slightly-rising with length, 1.27x -> 1.30x. Peak memory is
a consistent 0.60-0.64x.

Sampler is untouched by both changes: 1.113s -> 1.112s at L=500, still 98.8%
fp32, scaling L^1.18.

### Per-class exponents, bf16, same sweep

| Class | L=500 | share | L^k | vs fp32 exponent |
|---|---|---|---|---|
| `Linear` | 14.458s | 59.7% | 1.66 | 1.69 |
| `TriangleMultiplicativeUpdate` | 5.859s | 24.2% | **2.16** | 2.08 (**rose**) |
| `LayerNorm` | 1.183s | 4.9% | 1.49 | 1.70 |
| `SwiGLUMLP` | 1.135s | 4.7% | 1.75 | 1.89 |
| `PairUpdateBlock` | 0.537s | 2.2% | 1.45 | 1.61 |
| `SWA3DRoPEAttention` | 0.453s | 1.9% | 1.51 | 1.52 |

Every exponent fell except TriMul's, which rose. It is now the fastest-growing
significant class by a clear margin.

Extrapolated to L=1000 from the bf16 fits: trunk ~85s, with `Linear` ~54%,
**`TriMul` ~31%** (up from 24.2%), `SwiGLUMLP` ~4.5%, `LayerNorm` ~3.9%,
sampler ~3%.

## Run-to-run variance

The same fp32 config at L=500 measured 25.013s, 25.132s and 26.062s across
three sessions: a spread of ~4%. The ceiling probe varies similarly, 174-214
GB/s.

**Consequence: a claimed gain under ~5% is inside noise.** Stage 2's 1.054x
sits right at that boundary. Quote the conservative baseline, or repeat the
run, before trusting a small delta. Stage 1 at 1.18x is comfortably outside
it.

## Microbenchmarks at L=500

Pair tensor `(1,500,500,256)` bf16, 122 MiB. Moved bytes = 2x that.

| op | ms | GB/s | vs copy floor |
|---|---|---|---|
| copy floor (`z + 1`) | 1.30 | 196.9 | 1.00x |
| `nn.LayerNorm` 4D | 1.32 | 193.3 | 1.02x |
| `mx.fast.layer_norm` 4D | 1.33 | 191.9 | 1.03x |
| `mx.fast.layer_norm` 2D reshaped | 1.33 | 193.1 | 1.02x |
| `mx.fast.rms_norm` 4D | 1.73 | 148.1 | 1.33x |
| manual mean/var LayerNorm | 17.56 | 14.6 | 13.51x |
| `sigmoid` | 1.29 | 198.6 | 0.99x |

Strided reduction axis, as `_contract`'s transpose produces:

| op | ms | vs floor |
|---|---|---|
| `astype(bf16)` on strided view | 1.86 | 1.43x |
| `layer_norm(strided)` | 5.17 | 3.97x |
| `layer_norm(mx.contiguous(strided))` | **5.81** | **4.47x** |

Making it contiguous is *slower*. The stride is not worth fixing this way.

LayerNorm across every dtype and width the model uses:

| dtype | width | MiB | floor ms | ln ms | GB/s |
|---|---|---|---|---|---|
| bf16 | 256 | 122 | 1.40 | 1.34 | 191.8 |
| bf16 | 512 | 244 | 2.62 | 2.45 | 208.6 |
| bf16 | 1024 | 488 | 4.64 | 4.73 | 216.4 |
| f32 | 256 | 244 | 2.41 | 2.45 | 209.1 |
| f32 | 512 | 488 | 5.02 | 4.71 | 217.4 |
| f32 | 1024 | 977 | 9.89 | 9.21 | 222.3 |

At the roofline everywhere. `mx.fast.layer_norm` is not a problem.

Trunk GEMM shapes, M = L² = 250000:

| op | K | N | f32 ms | f32 TF | bf16 ms | bf16 TF | bf16 win |
|---|---|---|---|---|---|---|---|
| trimul `proj_bundle` | 256 | 1024 | 25.2 | 5.19 | 22.1 | 5.93 | 1.14x |
| trimul `proj_emit`/`gate` | 256 | 256 | 6.6 | 4.98 | 6.1 | 5.39 | 1.08x |
| `pair_transition.w12` | 256 | 2048 | 50.2 | 5.23 | 44.2 | 5.93 | 1.13x |
| `pair_transition.w3` | 1024 | 256 | 25.6 | 5.12 | 22.6 | 5.81 | 1.14x |

**bf16 does not raise the compute ceiling on a pre-M5 Apple GPU.** No matmul
accelerators. The ~1.13x is halved bytes. Roughly 2/3 of the stage-1 win came
from traffic, 1/3 from the faster matmul.

bf16 matmul accumulates in fp32: relative error flat at 0.00141 for K = 64,
256, 500, 1024, 4096. A bf16 accumulator would grow as sqrt(K) and reach ~4%
at the production K=500.

## TriMul phase 1, L=500, c_z=256, bf16

Ceilings that run: 195.9 GB/s, 6.02 TFLOP/s.

### A. Attribution ground truth -- it checks out

| variant | ms |
|---|---|
| `TriangleMultiplicativeUpdate(z, mask)` | 65.08 |
| same inside `mx.compile` | 59.79 (0.92x) |

The profiler's *exclusive* number is 5.859s / 204 = 28.7ms, which excludes
children. Adding the children back: `proj_bundle` 22.1 + `proj_emit` 6.1 +
`proj_gate` 6.1 + two norms 2.7 + own body 28.7 = **65.7ms predicted against
65.08ms measured**. The attribution chain is sound.

### B. The contraction -- the input transposes are real

| variant | ms | TFLOP/s | vs as-written |
|---|---|---|---|
| `_contract` as written (3 transposes) | 17.57 | 3.64 | — |
| pre-transposed in, transpose out | **11.87** | **5.39** | 0.68x |
| pre-transposed in, no transpose out | 11.87 | 5.39 | 0.68x |
| `mx.einsum bikd,bjkd->bijd` | 17.57 | 3.64 | 1.00x |
| transpose+contiguous only, no matmul | 2.33 | — | 0.13x |

- The **input** transposes cost 17.57 - 11.87 = **5.70ms per call**, which is
  ~2x the 2.33ms standalone copy. They materialise.
- The **output** transpose is free: identical with and without.
- `mx.einsum` lowers to the same path. No free win there.
- The matmul itself reaches 5.39 of 6.02 TFLOP/s, **90% of roof**. Irreducible.

Input transposes across the trunk: 5.70ms x 204 = **1.16s, 5.8% of 20.05s**.

### C. The gating -- mx.compile already did it

| variant | ms | GB/s | vs as-written |
|---|---|---|---|
| split + sigmoid + mul + mask | 7.93 | 96.9 | — |
| contiguous halves, same maths | 8.62 | 89.1 | **1.09x, slower** |
| **inside `mx.compile`** | **3.62** | **212.3** | **0.46x** |
| floor: read bundled, write one half | 3.53 | 217.6 | 0.45x |

`mx.compile` reaches **97% of the floor**. `mx.split`'s stride costs nothing;
forcing contiguity is *worse*. **A fused kernel here would buy ~0.09ms per
call. Nothing.**

### D. The epilogue -- also already done

| variant | ms | GB/s |
|---|---|---|
| two steps, as the model does | 4.36 | 117.3 |
| **inside `mx.compile`** | **2.31** | **221.4** |
| floor: 3 reads 1 write | 3.35 | 152.9 |

Compiled beats my floor row, which was not compiled and so made two passes.
At 221 GB/s it is above the measured streaming ceiling, so it is cache-assisted
and already optimal. **Nothing to win.**

### Verdict

The trunk runs inside a compiled `FoldingTrunk`, so C and D are already fused
in production. **Phase 3 as proposed is dead.** The only non-arithmetic cost
left in TriMul is the input transposes: 1.16s, 5.8% here.

Removing them needs a GEMM that writes transposed output, which is what the
reference's `fused_gated_dual_gemm_split` plus `(D,B,L,L)`-native
`trimul_einsum_triton` do. That is a kernel, for 5.8% on this chip.

On an M5 Ultra the transposes are memory-bound while the GEMMs are
compute-bound, so their share roughly doubles to **~11%**. The case is better
on the target than on the proxy.

## TriMul phase 2: channel-first layout

Section E2 of the trimul bench, L=500, bf16:

| variant | ms |
|---|---|
| gating + contraction, old layout, compiled | 43.02 |
| gating + contraction, channel-first, compiled | **38.22 (0.89x)** |
| `x @ W.T` -> (M=250000, N=1024) | 22.86 |
| `W @ x.T` -> (M=1024, N=250000) | 22.06 (0.96x) |
| `(B,i,j,C)` in: permute + matmul | 18.14 |
| `(C,B,i,j)` in: batch permute only | **11.88 (0.65x)** |

The feared risk did not materialise: flipping the GEMM orientation is neutral
(3.5%, inside noise), while removing the two copies is worth 34.5% on the
contraction.

### Paired A/B on a production trunk

`devtools/ab_trimul_layout.py 500 5`. Both variants in one process,
alternating, because the expected ~5% gain sits on the cross-session noise
floor. 24 blocks x 4 passes = 192 TriMul calls per measurement.

| pair | old | new | ratio |
|---|---|---|---|
| 1 | 18.768s | 17.887s | 1.049x |
| 2 | 18.786s | 17.774s | 1.057x |
| 3 | 18.759s | 17.682s | 1.061x |
| 4 | 18.905s | 17.721s | 1.067x |
| 5 | 18.805s | 17.681s | 1.064x |
| **median** | 18.786s | 17.721s | **1.061x, 5.7%** |

Output bit-identical, max abs diff 0.000e+00. Ratio spread 1.7%, stdev 0.0068.
Every pair favours the new layout.

**Why pairing was necessary:** absolute times vary 0.8% within a session and
~4% across sessions, against a 5.7% effect. The ratio is stable because drift
hits both halves of a pair equally.

Predicted 4.9% from E2, measured 5.7%. Pessimistic by 16% this time, after
being optimistic by ~33% twice.

## Cumulative

| Stage | trunk at L=500 | vs fp32 |
|---|---|---|
| fp32 baseline (`8a8efde`) | 26.062s | — |
| + bf16 trunk + contraction | 20.052s | 1.300x |
| + channel-first TriMul | **~18.90s** | **~1.379x** |

## The GPU is saturated. This closes a whole line of enquiry.

`devtools/gpu_busy.py`, L=500, powermetrics:

| workload | residency | power | freq |
|---|---|---|---|
| one large GEMM (control) | 94.6% | 21.8 W | 1503 MHz |
| one large elementwise (control) | 94.9% | 9.2 W | 1509 MHz |
| **trunk** | **100.0%** (min 100, max 100) | **23.0 W** | **1578 MHz** |

The trunk is *more* saturated than a bare GEMM loop, and draws more power at a
higher clock than either control.

**There are no dispatch bubbles.** Every wall-clock "% of roof" number in this
file is therefore honest -- none of them was hiding idle GPU. It also means
reducing kernel count buys nothing. The only levers left are better kernels
and less work.

## LayerNorm -> GEMM fusion: measured ceiling, then dropped

Section F of the trimul bench, L=500:

| variant | ms |
|---|---|
| `layer_norm(z)` alone | 1.33 |
| `proj(layer_norm(z))` as the model runs it | 23.21 |
| `proj(pre-normalised)`, GEMM only | 22.08 |
| row mean+var only | 3.88 |
| implied best fused | 25.96, **worse** |

The norm adds only **1.13ms on top of the GEMM**, because MLX already overlaps
it. Ceiling is 1.13ms x 306 pairs = 0.35s, **1.8%**, and a real fused kernel
still needs a stats pass so it would get less. I had estimated 5%; that was
LayerNorm's whole share, which fusion cannot remove. **Do not write it.**

## ESMC, first look

Never profiled before. L=500, random weights, bf16:

| class | time | share | rate |
|---|---|---|---|
| `Linear` | 1.285s | 81.0% | 4.94 TFLOP/s, **82% of GEMM roof** |
| `Attention` | 0.136s | 8.6% | container |
| `LayerNorm` | 0.065s | 4.1% | 25.5 GB/s, 15% of BW |
| rest | ~0.1s | 6% | — |

About 1.6s total, ~8% of a full fold. `Linear` sits at **396.9 FLOP/byte**, far
above every ridge point, so ESMC is strongly compute-bound and gets faster for
free on an M5 Ultra. Peak memory 11.86 GB for the 7.8B-param shim.

Little to win: 82% of roof on 81% of the phase.

## Sampler: SWA3DRoPEAttention builds a dense N x N mask

Found by reading, not measuring. For a +/-64 sliding window it materialises the
full mask and calls dense attention. At N=4000 atoms, per call:

| step | intermediate |
|---|---|
| `rank[:,:,None] - rank[:,None,:]` | 61 MB int32 |
| `mx.abs(...)` | 61 MB int32 |
| `<=`, the two `&`, `eye`, `\|` | 76 MB bool |
| **~mask traffic** | **~397 MB, 2.2ms at 190 GB/s** |

| attention | GFLOP |
|---|---|
| dense N x N | 8.19 |
| needed (N x 129) | 0.26 |
| **wasted** | **31x** |

Measured 7.19ms per call, 63 calls, 0.460s. **And the mask is identical on all
63 calls** -- `valid` and `N` are constant for a fold, so 397 MB is rebuilt 62
times for nothing. At N=8000 that intermediate is 244 MB.

Two fixes: cache the mask (trivial), or block-sparse the attention (real work).
Together ~1.5-2% of the run, and it stops the mask exploding at long chains.

## The GEMM ceiling hunt. All three levers are dead.

The trunk is 86% GEMM at 90-99% of a measured roof, every non-GEMM part is at
a bandwidth roof, and the GPU is 100% resident. So the only remaining
questions were about the ceiling itself. All three are now answered.

### 1. MLX's GEMM peak is 6.09 TFLOP/s, and we are within 6% of it

| N | fp32 | bf16 | fp16 |
|---|---|---|---|
| 512 | 0.96 | 1.27 | 1.37 |
| 1024 | 3.71 | 4.11 | 4.20 |
| 2048 | 5.09 | 5.73 | 5.72 |
| 4096 | 5.22 | 6.01 | 6.03 |
| 8192 | 5.28 | **6.09** | 6.07 |

The trunk's GEMMs run at 5.54-5.9. Headroom against MLX's own peak: **+6%**.
The ~8.6 TFLOP/s ALU figure is not reachable through MLX's GEMM; that gap is
upstream MLX's kernels, not anything in this port.

### 2. bf16 is the fastest dtype. fp16 gives nothing.

| shape | K | N | fp32 | bf16 | fp16 | best |
|---|---|---|---|---|---|---|
| `proj_bundle W@x.T` | 256 | 1024 | 5.18 | **5.95** | 5.93 | bf16 |
| `proj_emit`/`proj_gate` | 256 | 256 | 4.97 | **5.78** | 5.56 | bf16 |
| `pair_transition w12` | 256 | 2048 | 4.89 | 5.78 | 5.78 | tie |
| `pair_transition w3` | 1024 | 256 | 5.21 | **5.94** | 5.94 | tie |

bf16 wins or ties everywhere. The fp16 hypothesis -- that Apple GPUs run fp16
at full rate while bf16 support is newer -- is **false on this chip**.

### 3. Padding is a net loss, despite the contraction liking it

| L | L%64 | ms | TFLOP/s | ns/elem | pad cost | net vs 500 |
|---|---|---|---|---|---|---|
| 448 | 0 | 7.98 | 5.77 | 0.155 | — | — |
| 500 | 52 | 12.28 | 5.21 | 0.192 | 1.000 | 1.000 |
| 504 | 56 | 11.85 | 5.53 | 0.182 | 1.016 | **0.965** |
| 512 | **0** | 11.78 | **5.83** | 0.176 | 1.049 | **0.959** |
| 576 | 0 | 17.20 | 5.69 | 0.203 | 1.327 | 1.401 |

L=500 runs the contraction at 5.21 TFLOP/s against 5.83 at L=512, so
alignment is worth 12% *on the contraction*. But the contraction is only
**12.9%** of a PairUpdateBlock, and padding the pair tensor makes the other
87.1% do 4.9% more work:

| pad target | contraction | everything else | net on trunk |
|---|---|---|---|
| 504 | -0.45% | +1.39% | **+0.94% LOSS** |
| 512 | -0.53% | +4.27% | **+3.74% LOSS** |

Padding only inside `_contract` and cropping after is worse still: saves
1.00ms per block, costs 3.85ms in the pad and crop copies.

Note L=576 is 64-aligned yet worse per element than L=500, and L=448 is the
best of all. So the trend is mostly cache footprint, with alignment a
secondary bonus. Alignment alone does not explain it.

## TriMul's 72% of GEMM roof is a mixing artefact, not headroom

With raw-matmul FLOPs now counted, TriMul reads 9.169s, 39.6%, 4.34 TFLOP/s =
72% of roof, 760.5 FLOP/byte. That 72% is one rate for a mixed module. Split
it:

| part | ms | rate |
|---|---|---|
| `proj_bundle` | 22.06 | 5.95 TFLOP/s, **99% of roof** |
| contraction | 11.87 | 5.39 TFLOP/s, **90% of roof** |
| gating (compiled) | 3.63 | 211.6 GB/s, **at BW floor** |
| epilogue (compiled) | 2.30 | 218.7 GB/s, **above streaming ceiling** |

Its GEMM portion runs at ~95% of roof. The 72% is diluted by the
bandwidth-bound elementwise work, which has no FLOPs, plus instrumentation
sync across 204 calls. **Nothing to win.**

## Where this leaves the trunk

Trunk 18.965s at L=500, and every component is at a roof:

| class | share | limiter |
|---|---|---|
| `Linear` | 43.3% | 92% of GEMM roof |
| `TriangleMultiplicativeUpdate` | 39.6% | ~95% of GEMM roof on its GEMM part |
| `LayerNorm` | 5.2% | 62% of BW roof, fusion ceiling 1.8% |
| `SwiGLUMLP` | 5.0% | at `mx.compile` floor |
| `SWA3DRoPEAttention` | 2.0% | **0.6 GB/s -- the only one left** |

`Linear` at 179.9 FLOP/byte and TriMul at 760.5 stay **compute-bound on an M5
Ultra**, so 83% of the trunk takes the full M5 compute uplift for free.

## Harness bugs that invalidated earlier numbers

Read this before comparing against anything older than the commit named.

| Bug | Effect | Fixed in |
|---|---|---|
| `mx.synchronize()` does not flush a lazy graph | Every module absorbed its parent's unevaluated upstream ops. `LayerNorm` read as 32% of the trunk; it is really 4.9%. | `245aec7` |
| `mx.compile` hides submodules | `FoldingTrunk` replays a traced graph, so child `__call__` never runs. The whole pair stack was charged to the parent. | `a18a68c` |
| Container modules reported bandwidth | Full tensor bytes over exclusive-only time. `DiffusionModule` showed 1081% of peak. | `a18a68c` |
| Param bytes double-counted in class rollups | Inflated aggregate traffic | `0200862` |
| `_elide` destroyed numeric path components | `abbr("17")` returned `"1"`, so seventeen blocks rendered as one row | `e941afa` |
| `mx.eval` on a dataclass evaluates nothing | ESMC's `encode()` returns `EsmcOutput`, so every ESMC phase timing was graph construction: 0.001s at every length, scaling `L^-0.26` | `1e4c7b2` |
| raw matmuls are not counted as FLOPs | channel-first TriMul calls `proj_bundle.weight @ x.T` directly, bypassing `nn.Linear.__call__`. Its GEMM moved out of `Linear` (14.458s -> 10.031s) into TriMul's body (5.859s -> 9.147s), which then read as 5.7 GB/s and `HEADROOM` on what is mostly a GEMM at 92% of roof | `1e4c7b2` |

The second of those is worth remembering as a design smell, not just a harness
bug: bypassing `nn.Linear` also means any future Linear-level work, quantisation
above all, will silently skip `proj_bundle`.

Phase totals, scaling exponents and the dtype audit were never affected: they
come from uninstrumented runs or from byte counts.

## Hypothesis log

| Hypothesis | Verdict | Evidence |
|---|---|---|
| Dispatch / Python bound | FALSE | lazy share 0.1% |
| O(L^3) contraction is the bottleneck | FALSE | trunk scales L^2.05; TriMul own body was 3.7% pre-fix |
| Custom Metal kernels for TriMul will pay | FALSE | see above; reference's own `trimul_einsum_triton` says "Triton fwd loses to cuBLAS bgemm" |
| `mx.fast.layer_norm` is slow | FALSE | 101-117% of copy floor at every dtype and width |
| LayerNorm cost is profiler sync overhead | FALSE | total instrumentation overhead 3.7% |
| Strided reduction axis from `_contract` | FALSE | `mx.contiguous` is slower, 5.81 vs 5.17 ms |
| `w3` limited by its N=256 shape | FALSE | 5.12 TFLOP/s in isolation, same as the others |
| **Trunk silently runs fp32** | **TRUE** | 4 independent byte-count confirmations; became stages 1 and 2 |

## Open candidates, ranked

| Candidate | Est. | Basis |
|---|---|---|
| Fuse TriMul gating + mask + residual into the final GEMM | ~17% | 5.887s TriMul body is ~2.2s contraction matmul, ~3.7s elementwise and transposes. Mirrors `fused_dual_gemm` and `trimul_with_residual`. Needs `mx.fast.metal_kernel`. |
| Fuse LN + w12 + SwiGLU | ~5% | `SwiGLUMLP` 1.150s. Mirrors `fused_lnlin_swiglu` forward. |
| bf16 the sampler | ~2% | sampler traffic still 98.8% fp32 |
| `SWA3DRoPEAttention` at 0.6 GB/s | ~1% | only `HEADROOM` flag left, but small |
| Anything touching `Linear` | 0% | 94% of GEMM roof |

All estimates are arithmetic, not measurements. The last two estimates I made
this way were optimistic by about a third.
