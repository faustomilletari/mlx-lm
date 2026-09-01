# Copyright © 2026 Apple Inc.

"""Fused Metal kernels for ESMFold2's pair stack.

Two kernels, both folding a sigmoid gate into a GEMM epilogue so the wide
intermediate never reaches memory.

:func:`trimul_gated_dual` -- the TriMul *input* stage:

    routed = (x @ w_sig.T) * sigmoid(x @ w_gate.T) * mask

Unfused, ``proj_bundle`` writes an [M, 4*c_z] tensor -- four times the width of
the pair tensor -- and the gate reads it straight back. Fused, one input tile
feeds both GEMMs and only the [M, 2*c_z] result is written. Mirrors esm's
``fused_gated_dual_gemm``.

:func:`trimul_epilogue` -- the TriMul *output* stage:

    out = residual + (x_value @ w_emit.T) * sigmoid(x_gate @ w_gate.T)

Unfused that is two GEMMs, each writing a full [M, N] intermediate, then an
elementwise pass that reads both back plus the residual and writes the output --
eight passes over an [M, c_z] tensor. Fused it is four: read x_value, read
x_gate, read residual, write out. The two intermediates never reach memory.

This mirrors esm's ``trimul_with_residual`` Triton kernel, which folds the same
gate and residual into the final GEMM epilogue.

Shapes, with M = B*L*L and K = N = c_z:

    x_value, x_gate : [M, K]
    w_emit, w_gate  : [N, K]   (nn.Linear layout: [out, in])
    residual, out   : [M, N]
"""

import os
from typing import Optional

import mlx.core as mx

#: Both kernels are OFF by default. Measured on an M-series machine they cost
#: memory and returned little speed, for two reasons that apply to any custom
#: kernel placed inside the trunk:
#:
#: 1. A custom kernel cannot donate its output buffer. ``pair + delta`` writes
#:    the sum into delta's buffer and frees the old pair in place; a kernel
#:    always allocates. MLX caches freed buffers, so a differently sized
#:    molecule next cannot reuse any of it.
#: 2. FoldingTrunk._apply_blocks is mx.compile'd and a metal_kernel is opaque
#:    to that trace, which costs the cross-block fusion the trunk relied on.
#:
#: Turn one on to measure it:
#:   MLX_ESMFOLD2_EPILOGUE_KERNEL=1   TriMul output stage
#:   MLX_ESMFOLD2_DUAL_KERNEL=1       TriMul input stage
USE_EPILOGUE_KERNEL = os.environ.get("MLX_ESMFOLD2_EPILOGUE_KERNEL", "0") == "1"
USE_DUAL_KERNEL = os.environ.get("MLX_ESMFOLD2_DUAL_KERNEL", "0") == "1"

# Output tile per threadgroup, and the K step. TN and TK must divide N and K;
# c_z is 256 in every published checkpoint, so 32 divides both. SIMDGROUPS * 8
# must equal TM: each simdgroup owns one 8-row strip of the tile.
TM, TN, TK = 32, 32, 32
SIMDGROUPS = TM // 8

_HEADER = """
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;
"""

# Grid is (N/TN, ceil(M/TM), 1) threadgroups of (32, SIMDGROUPS, 1) threads.
# MLX takes the grid in threads, so the caller multiplies out.
_SOURCE = f"""
    constexpr int kTM = {TM};
    constexpr int kTN = {TN};
    constexpr int kTK = {TK};
    constexpr int kSG = {SIMDGROUPS};

    const uint n0 = threadgroup_position_in_grid.x * kTN;
    const uint m0 = threadgroup_position_in_grid.y * kTM;
    const uint sg = simdgroup_index_in_threadgroup;   // 0..kSG-1
    const uint lane = thread_index_in_simdgroup;      // 0..31
    const uint tid = sg * 32 + lane;
    const uint nthreads = kSG * 32;

    // One staging buffer, carved up. Tiles are float, not T: MLX's own steel
    // GEMM runs A, B and C as float and converts on load (see steel/gemm/mma.h),
    // and simdgroup_load requires the matrix and pointer element types to match.
    threadgroup float tg[4 * kTM * kTK];
    threadgroup float *Av = tg;
    threadgroup float *Ag = tg + kTM * kTK;
    threadgroup float *Bv = tg + 2 * kTM * kTK;   // stored transposed: [k][n]
    threadgroup float *Bg = tg + 3 * kTM * kTK;

    // Zero the accumulators from a zeroed patch. simdgroup_matrix<float>(v)
    // fills the DIAGONAL, not the matrix, so it cannot be used here.
    for (uint i = tid; i < 64; i += nthreads) {{
        tg[i] = 0.0f;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_matrix<float, 8, 8> acc_v[kTN / 8];
    simdgroup_matrix<float, 8, 8> acc_g[kTN / 8];
    for (int j = 0; j < kTN / 8; ++j) {{
        simdgroup_load(acc_v[j], tg, 8);
        simdgroup_load(acc_g[j], tg, 8);
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint k0 = 0; k0 < (uint)K; k0 += kTK) {{
        for (uint i = tid; i < kTM * kTK; i += nthreads) {{
            uint r = i / kTK, c = i % kTK;
            uint gm = m0 + r;
            bool ok = (gm < (uint)M);
            Av[i] = ok ? static_cast<float>(x_value[gm * (uint)K + k0 + c]) : 0.0f;
            Ag[i] = ok ? static_cast<float>(x_gate[gm * (uint)K + k0 + c]) : 0.0f;
        }}
        for (uint i = tid; i < kTN * kTK; i += nthreads) {{
            uint r = i / kTK, c = i % kTK;         // r over n, c over k
            uint gn = n0 + r;
            Bv[c * kTN + r] = static_cast<float>(w_emit[gn * (uint)K + k0 + c]);
            Bg[c * kTN + r] = static_cast<float>(w_gate[gn * (uint)K + k0 + c]);
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int kk = 0; kk < kTK; kk += 8) {{
            simdgroup_matrix<float, 8, 8> a_v, a_g, b_v, b_g;
            simdgroup_load(a_v, Av + sg * 8 * kTK + kk, kTK);
            simdgroup_load(a_g, Ag + sg * 8 * kTK + kk, kTK);
            for (int j = 0; j < kTN / 8; ++j) {{
                simdgroup_load(b_v, Bv + kk * kTN + j * 8, kTN);
                simdgroup_load(b_g, Bg + kk * kTN + j * 8, kTN);
                simdgroup_multiply_accumulate(acc_v[j], a_v, b_v, acc_v[j]);
                simdgroup_multiply_accumulate(acc_g[j], a_g, b_g, acc_g[j]);
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    // Epilogue: gate and residual applied in one pass. Reuses the staging
    // buffer -- the A/B tiles are dead after the last barrier above.
    threadgroup float *Ov = tg;
    threadgroup float *Og = tg + kTM * kTN;
    for (int j = 0; j < kTN / 8; ++j) {{
        simdgroup_store(acc_v[j], Ov + sg * 8 * kTN + j * 8, kTN);
        simdgroup_store(acc_g[j], Og + sg * 8 * kTN + j * 8, kTN);
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = tid; i < kTM * kTN; i += nthreads) {{
        uint r = i / kTN, c = i % kTN;
        uint gm = m0 + r, gn = n0 + c;
        if (gm < (uint)M) {{
            float g = 1.0f / (1.0f + metal::exp(-Og[i]));
            uint o = gm * (uint)N + gn;
            out[o] = static_cast<T>(static_cast<float>(residual[o]) + Ov[i] * g);
        }}
    }}
"""


def _make_trimul_epilogue_kernel():
    if not mx.metal.is_available():
        return None
    return mx.fast.metal_kernel(
        name="trimul_gated_gemm_residual",
        input_names=["x_value", "x_gate", "w_emit", "w_gate", "residual", "M", "N", "K"],
        output_names=["out"],
        header=_HEADER,
        source=_SOURCE,
    )


_trimul_epilogue_kernel = _make_trimul_epilogue_kernel()


def trimul_epilogue_ops(
    x_value: mx.array,
    x_gate: mx.array,
    w_emit: mx.array,
    w_gate: mx.array,
    residual: mx.array,
) -> mx.array:
    """Pure-MLX reference. Two tuned GEMMs, then a fused elementwise pass."""
    value = x_value @ w_emit.T
    gate = x_gate @ w_gate.T
    return residual + value * mx.sigmoid(gate)


def trimul_epilogue_kernel(
    x_value: mx.array,
    x_gate: mx.array,
    w_emit: mx.array,
    w_gate: mx.array,
    residual: mx.array,
) -> mx.array:
    M, K = x_value.shape
    N = w_emit.shape[0]
    return _trimul_epilogue_kernel(
        inputs=[x_value, x_gate, w_emit, w_gate, residual, M, N, K],
        template=[("T", x_value.dtype)],
        grid=((N // TN) * 32, (M + TM - 1) // TM * SIMDGROUPS, 1),
        threadgroup=(32, SIMDGROUPS, 1),
        output_shapes=[(M, N)],
        output_dtypes=[x_value.dtype],
    )[0]


def trimul_epilogue(
    x_value: mx.array,
    x_gate: mx.array,
    w_emit: mx.array,
    w_gate: mx.array,
    residual: mx.array,
    use_kernel: bool = True,
) -> mx.array:
    """``residual + (x_value @ w_emit.T) * sigmoid(x_gate @ w_gate.T)``.

    Falls back to :func:`trimul_epilogue_ops` off Metal, on unsupported dtypes,
    or when the tile sizes do not divide the problem.
    """
    N, K = w_emit.shape
    usable = (
        use_kernel
        and USE_EPILOGUE_KERNEL
        and _trimul_epilogue_kernel is not None
        and mx.default_device() == mx.gpu
        and x_value.dtype == x_gate.dtype == w_emit.dtype == w_gate.dtype
        and residual.dtype == x_value.dtype
        and x_value.ndim == 2
        and N % TN == 0
        and K % TK == 0
    )
    if not usable:
        return trimul_epilogue_ops(x_value, x_gate, w_emit, w_gate, residual)
    return trimul_epilogue_kernel(x_value, x_gate, w_emit, w_gate, residual)


# ---------------------------------------------------------------------------
# TriMul input stage: gated dual GEMM
# ---------------------------------------------------------------------------

# Both GEMMs share one input tile, so A is staged once.
_DUAL_SOURCE_TMPL = """
    constexpr int kTM = {TM};
    constexpr int kTN = {TN};
    constexpr int kTK = {TK};
    constexpr int kSG = {SG};

    const uint n0 = threadgroup_position_in_grid.x * kTN;
    const uint m0 = threadgroup_position_in_grid.y * kTM;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint tid = sg * 32 + lane;
    const uint nthreads = kSG * 32;

    threadgroup float tg[3 * kTM * kTK];
    threadgroup float *A  = tg;
    threadgroup float *Bs = tg + kTM * kTK;       // signal weights, [k][n]
    threadgroup float *Bg = tg + 2 * kTM * kTK;   // gate weights,   [k][n]

    for (uint i = tid; i < 64; i += nthreads) {{
        tg[i] = 0.0f;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_matrix<float, 8, 8> acc_s[kTN / 8];
    simdgroup_matrix<float, 8, 8> acc_g[kTN / 8];
    for (int j = 0; j < kTN / 8; ++j) {{
        simdgroup_load(acc_s[j], tg, 8);
        simdgroup_load(acc_g[j], tg, 8);
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint k0 = 0; k0 < (uint)K; k0 += kTK) {{
        for (uint i = tid; i < kTM * kTK; i += nthreads) {{
            uint r = i / kTK, c = i % kTK;
            uint gm = m0 + r;
            A[i] = (gm < (uint)M) ? static_cast<float>(x[gm * (uint)K + k0 + c]) : 0.0f;
        }}
        for (uint i = tid; i < kTN * kTK; i += nthreads) {{
            uint r = i / kTK, c = i % kTK;
            uint gn = n0 + r;
            Bs[c * kTN + r] = static_cast<float>(w_sig[gn * (uint)K + k0 + c]);
            Bg[c * kTN + r] = static_cast<float>(w_gate[gn * (uint)K + k0 + c]);
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int kk = 0; kk < kTK; kk += 8) {{
            simdgroup_matrix<float, 8, 8> a, b_s, b_g;
            simdgroup_load(a, A + sg * 8 * kTK + kk, kTK);
            for (int j = 0; j < kTN / 8; ++j) {{
                simdgroup_load(b_s, Bs + kk * kTN + j * 8, kTN);
                simdgroup_load(b_g, Bg + kk * kTN + j * 8, kTN);
                simdgroup_multiply_accumulate(acc_s[j], a, b_s, acc_s[j]);
                simdgroup_multiply_accumulate(acc_g[j], a, b_g, acc_g[j]);
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    threadgroup float *Os = tg;
    threadgroup float *Og = tg + kTM * kTN;
    for (int j = 0; j < kTN / 8; ++j) {{
        simdgroup_store(acc_s[j], Os + sg * 8 * kTN + j * 8, kTN);
        simdgroup_store(acc_g[j], Og + sg * 8 * kTN + j * 8, kTN);
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = tid; i < kTM * kTN; i += nthreads) {{
        uint r = i / kTN, c = i % kTN;
        uint gm = m0 + r, gn = n0 + c;
        if (gm < (uint)M) {{
            float g = 1.0f / (1.0f + metal::exp(-Og[i]));
            float v = Os[i] * g;
            {MASK}
            out[gm * (uint)N + gn] = static_cast<T>(v);
        }}
    }}
"""


def _make_trimul_dual_kernel(has_mask: bool):
    if not mx.metal.is_available():
        return None
    names = ["x", "w_sig", "w_gate", "M", "N", "K"]
    if has_mask:
        names.insert(3, "mask")
    return mx.fast.metal_kernel(
        name=f"trimul_gated_dual{'_mask' if has_mask else ''}",
        input_names=names,
        output_names=["out"],
        header=_HEADER,
        source=_DUAL_SOURCE_TMPL.format(
            TM=TM, TN=TN, TK=TK, SG=SIMDGROUPS,
            MASK="v *= static_cast<float>(mask[gm]);" if has_mask else "",
        ),
    )


_trimul_dual_kernel = _make_trimul_dual_kernel(False)
_trimul_dual_kernel_masked = _make_trimul_dual_kernel(True)


def trimul_gated_dual_ops(
    x: mx.array,
    w_sig: mx.array,
    w_gate: mx.array,
    mask: Optional[mx.array] = None,
) -> mx.array:
    """Pure-MLX reference for the gated dual GEMM."""
    out = (x @ w_sig.T) * mx.sigmoid(x @ w_gate.T)
    if mask is not None:
        out = out * mask[:, None]
    return out


def trimul_gated_dual(
    x: mx.array,
    w_sig: mx.array,
    w_gate: mx.array,
    mask: Optional[mx.array] = None,
    use_kernel: bool = True,
) -> mx.array:
    """``(x @ w_sig.T) * sigmoid(x @ w_gate.T)``, optional row-shared mask.

    ``x`` is [M, K]; ``w_sig`` and ``w_gate`` are [N, K]; ``mask`` is [M].
    """
    M, K = x.shape
    N = w_sig.shape[0]
    kernel = _trimul_dual_kernel_masked if mask is not None else _trimul_dual_kernel
    usable = (
        use_kernel
        and USE_DUAL_KERNEL
        and kernel is not None
        and mx.default_device() == mx.gpu
        and x.dtype == w_sig.dtype == w_gate.dtype
        and x.ndim == 2
        and N % TN == 0
        and K % TK == 0
        and (mask is None or (mask.ndim == 1 and mask.dtype == x.dtype))
    )
    if not usable:
        return trimul_gated_dual_ops(x, w_sig, w_gate, mask)
    inputs = [x, w_sig, w_gate, M, N, K]
    if mask is not None:
        inputs.insert(3, mask)
    return kernel(
        inputs=inputs,
        template=[("T", x.dtype)],
        grid=((N // TN) * 32, (M + TM - 1) // TM * SIMDGROUPS, 1),
        threadgroup=(32, SIMDGROUPS, 1),
        output_shapes=[(M, N)],
        output_dtypes=[x.dtype],
    )[0]
