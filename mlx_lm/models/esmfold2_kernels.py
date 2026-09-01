# Copyright © 2026 Apple Inc.

"""Fused Metal kernels for ESMFold2's pair stack.

The one kernel here is the TriMul output stage:

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

from typing import Optional

import mlx.core as mx

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
    const uint tid = sg * 32 + lane;                  // 0..kSG*32-1
    const uint nthreads = kSG * 32;

    threadgroup T Av[kTM * kTK];
    threadgroup T Ag[kTM * kTK];
    threadgroup T Bv[kTK * kTN];   // stored transposed: [k][n]
    threadgroup T Bg[kTK * kTN];

    // Each simdgroup owns rows [sg*8, sg*8+8) of the tile, all kTN columns.
    simdgroup_matrix<float, 8, 8> acc_v[kTN / 8];
    simdgroup_matrix<float, 8, 8> acc_g[kTN / 8];
    for (int j = 0; j < kTN / 8; ++j) {{
        acc_v[j] = simdgroup_matrix<float, 8, 8>(0.0f);
        acc_g[j] = simdgroup_matrix<float, 8, 8>(0.0f);
    }}

    for (uint k0 = 0; k0 < (uint)K; k0 += kTK) {{
        // Cooperative load of the A tiles: kTM*kTK elements over nthreads.
        for (uint i = tid; i < kTM * kTK; i += nthreads) {{
            uint r = i / kTK, c = i % kTK;
            uint gm = m0 + r;
            bool ok = (gm < (uint)M);
            Av[i] = ok ? x_value[gm * (uint)K + k0 + c] : (T)0;
            Ag[i] = ok ? x_gate[gm * (uint)K + k0 + c] : (T)0;
        }}
        // B tiles, transposed on the way in so the MMA sees [k][n].
        for (uint i = tid; i < kTN * kTK; i += nthreads) {{
            uint r = i / kTK, c = i % kTK;         // r over n, c over k
            uint gn = n0 + r;
            Bv[c * kTN + r] = w_emit[gn * (uint)K + k0 + c];
            Bg[c * kTN + r] = w_gate[gn * (uint)K + k0 + c];
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

    // Epilogue: gate and residual in register, one write.
    threadgroup float Ov[kTM * kTN];
    threadgroup float Og[kTM * kTN];
    for (int j = 0; j < kTN / 8; ++j) {{
        simdgroup_store(acc_v[j], Ov + sg * 8 * kTN + j * 8, kTN);
        simdgroup_store(acc_g[j], Og + sg * 8 * kTN + j * 8, kTN);
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = tid; i < kTM * kTN; i += nthreads) {{
        uint r = i / kTN, c = i % kTN;
        uint gm = m0 + r, gn = n0 + c;
        if (gm < (uint)M) {{
            float v = Ov[i];
            float g = 1.0f / (1.0f + metal::exp(-Og[i]));
            uint o = gm * (uint)N + gn;
            out[o] = (T)((float)residual[o] + v * g);
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


@mx.compile
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
