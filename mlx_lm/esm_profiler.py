"""Module-level profiler for the MLX ESM models.

Instrumentation only. This file is never merged; optimizations land elsewhere.

Two measurement modes, because MLX is lazy and both numbers matter:

  eval  Sync on every module boundary. Gives true per-layer device time, but
        the serialization inflates the total. Use for attribution.
  lazy  Never sync. Measures only Python + graph construction. If the lazy
        total is a large slice of the eval total, the model is dispatch-bound
        and no kernel work will help.

Bandwidth is reported as achieved GB/s against a ceiling measured on the
machine at runtime, so numbers stay meaningful across an M4 Pro and an Ultra.
Caveat: bytes counted are logical tensor traffic, not DRAM traffic. Unified
memory and cache mean a reused tensor may never reach DRAM, so achieved GB/s
is an upper bound. Confirm the top offenders with a .gputrace.
"""

from __future__ import annotations

import json
import math
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

# ---------------------------------------------------------------------------
# tensor / module introspection
# ---------------------------------------------------------------------------


def iter_arrays(obj: Any) -> Iterator[mx.array]:
    """Every mx.array reachable through tuples, lists and dicts."""
    if isinstance(obj, mx.array):
        yield obj
    elif isinstance(obj, (list, tuple)):
        for o in obj:
            yield from iter_arrays(o)
    elif isinstance(obj, dict):
        for o in obj.values():
            yield from iter_arrays(o)


_DT_SHORT = {"float32": "f32", "bfloat16": "bf16", "float16": "f16",
             "int32": "i32", "int64": "i64", "bool": "b8", "uint32": "u32",
             "int8": "i8", "uint8": "u8", "int16": "i16"}


def _dt(dtype) -> str:
    name = str(dtype).rsplit(".", 1)[-1]
    return _DT_SHORT.get(name, name)


def tensor_bytes(obj: Any) -> int:
    return sum(a.nbytes for a in iter_arrays(obj))


def _direct_submodules(m: nn.Module) -> Iterator[nn.Module]:
    """Direct child modules, including those held inside lists and dicts."""
    for v in m.children().values():
        yield from _walk_for_modules(v)


def _walk_for_modules(v: Any) -> Iterator[nn.Module]:
    if isinstance(v, nn.Module):
        yield v
    elif isinstance(v, (list, tuple)):
        for o in v:
            yield from _walk_for_modules(o)
    elif isinstance(v, dict):
        for o in v.values():
            yield from _walk_for_modules(o)


def _all_param_bytes(m: nn.Module) -> int:
    return sum(v.nbytes for _, v in tree_flatten(m.parameters())
               if isinstance(v, mx.array))


def _linear_shape(m: nn.Module) -> tuple:
    """(in_features, out_features) for a plain Linear, else ()."""
    w = m.get("weight") if hasattr(m, "get") else None
    if not isinstance(w, mx.array) or w.ndim != 2:
        return ()
    if type(m).__name__ not in ("Linear", "QuantizedLinear"):
        return ()
    out_f, in_f = w.shape
    return (in_f, out_f)


def own_param_bytes(m: nn.Module) -> int:
    """Parameter bytes belonging to this module and not to a descendant."""
    total = _all_param_bytes(m)
    child = sum(_all_param_bytes(c) for c in _direct_submodules(m))
    return max(total - child, 0)


# ---------------------------------------------------------------------------
# device ceilings, measured here rather than looked up
# ---------------------------------------------------------------------------


@dataclass
class Ceilings:
    device: str
    peak_bw_gbs: float
    peak_gemm_tflops: float
    detail: dict = field(default_factory=dict)

    @property
    def ridge(self) -> float:
        """FLOP per byte at which compute and memory take equally long.

        Hardware-independent way to read a layer: intensity above the ridge is
        compute-bound, below it is memory-bound. Because the ridge differs per
        chip, the same layer can be compute-bound on one and memory-bound on
        another -- which is why a percentage measured on one machine does not
        transfer to another.
        """
        return (self.peak_gemm_tflops * 1e12) / (self.peak_bw_gbs * 1e9)

    def as_dict(self) -> dict:
        return {"device": self.device, "peak_bw_gbs": self.peak_bw_gbs,
                "peak_gemm_tflops": self.peak_gemm_tflops,
                "ridge_flop_per_byte": self.ridge, "detail": self.detail}


def _time_op(fn: Callable[[], Any], iters: int = 20, warmup: int = 5) -> float:
    for _ in range(warmup):
        mx.eval(fn())
    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        mx.eval(fn())
    mx.synchronize()
    return (time.perf_counter() - t0) / iters


def measure_ceilings(dtype=mx.bfloat16, bw_mb: int = 256, gemm_n: int = 4096,
                     iters: int = 20) -> Ceilings:
    """Achievable streaming bandwidth and GEMM throughput on this machine.

    Bandwidth uses a strided-free elementwise add: one array read, one written.
    Both are lower bounds on the hardware spec and upper bounds on what any
    real layer can reach, which is exactly the comparison we want.
    """
    n = (bw_mb * 2 ** 20) // mx.zeros(1, dtype=dtype).itemsize
    x = mx.random.normal((n,)).astype(dtype)
    mx.eval(x)
    t_bw = _time_op(lambda: x + 1, iters)
    moved = 2 * x.nbytes
    peak_bw = moved / t_bw / 1e9
    del x

    a = mx.random.normal((gemm_n, gemm_n)).astype(dtype)
    b = mx.random.normal((gemm_n, gemm_n)).astype(dtype)
    mx.eval(a, b)
    t_gemm = _time_op(lambda: a @ b, max(iters // 2, 3))
    peak_tf = (2 * gemm_n ** 3) / t_gemm / 1e12
    del a, b
    mx.clear_cache()

    return Ceilings(
        device=str(mx.default_device()),
        peak_bw_gbs=peak_bw,
        peak_gemm_tflops=peak_tf,
        detail={"bw_probe_mb": bw_mb, "bw_probe_ms": t_bw * 1e3,
                "gemm_n": gemm_n, "gemm_probe_ms": t_gemm * 1e3,
                "dtype": str(dtype)},
    )


# ---------------------------------------------------------------------------
# per-module records
# ---------------------------------------------------------------------------


@dataclass
class ModuleStat:
    path: str
    cls: str
    calls: int = 0
    incl_s: float = 0.0
    excl_s: float = 0.0
    in_bytes: int = 0
    out_bytes: int = 0
    own_params: int = 0     # static size of this module's own weights
    param_bytes: int = 0    # own_params accumulated over calls
    alloc_delta: int = 0
    flops: float = 0.0      # exact, counted only where we can (Linear)
    gemm_shape: tuple = ()  # (in_features, out_features) for Linear
    shapes: list = field(default_factory=list)

    @property
    def moved_bytes(self) -> int:
        """Logical traffic charged to this module's own work."""
        return self.in_bytes + self.out_bytes + self.param_bytes

    @property
    def is_container(self) -> bool:
        """True when children did most of the work.

        Byte counts are per call, but exclusive time is not. A module that
        just forwards a large tensor to its children gets charged the full
        tensor with almost no time, which reports as thousands of GB/s. That
        number is meaningless, so suppress it.
        """
        return self.incl_s > 0 and (self.excl_s / self.incl_s) < 0.5

    def gbs(self) -> Optional[float]:
        if self.excl_s <= 0 or self.is_container:
            return None
        return self.moved_bytes / self.excl_s / 1e9

    def intensity(self) -> Optional[float]:
        """FLOP per byte. Only meaningful where FLOPs are counted exactly."""
        if self.flops <= 0 or self.moved_bytes <= 0:
            return None
        return self.flops / self.moved_bytes

    def tflops(self) -> Optional[float]:
        """Achieved compute rate. None when we cannot count FLOPs exactly.

        Counting FLOPs generically is not possible from a module hook, so we
        only count them where the shape makes it unambiguous. A GEMM near the
        compute ceiling has no headroom no matter what its GB/s says, which is
        the distinction that decides whether a layer is worth touching.
        """
        if self.excl_s <= 0 or self.flops <= 0 or self.is_container:
            return None
        return self.flops / self.excl_s / 1e12

    def as_dict(self) -> dict:
        return {"path": self.path, "cls": self.cls, "calls": self.calls,
                "incl_s": self.incl_s, "excl_s": self.excl_s,
                "in_bytes": self.in_bytes, "out_bytes": self.out_bytes,
                "own_params": self.own_params, "param_bytes": self.param_bytes,
                "flops": self.flops, "tflops": self.tflops(),
                "alloc_delta": self.alloc_delta,
                "moved_bytes": self.moved_bytes, "gbs": self.gbs(),
                "shapes": self.shapes[:4]}


# ---------------------------------------------------------------------------
# the profiler
# ---------------------------------------------------------------------------


class LayerProfiler:
    """Times every submodule of a model tree.

    `obj(...)` resolves __call__ on the type, not the instance, so instance
    patching does not intercept anything. We patch each distinct class once
    and dispatch on id(self) through a registry. Instances outside the
    registry fall straight through to the original.
    """

    def __init__(self, mode: str = "eval", record_shapes: bool = True,
                 max_depth: Optional[int] = None):
        if mode not in ("eval", "lazy"):
            raise ValueError(f"mode must be 'eval' or 'lazy', got {mode!r}")
        self.mode = mode
        self.record_shapes = record_shapes
        self.max_depth = max_depth
        self.stats: dict[str, ModuleStat] = {}
        self.dtype_bytes: dict[str, int] = {}
        self._registry: dict[int, str] = {}
        self._patched: list[tuple[type, Callable]] = []
        self._stack: list[list] = []   # [path, t_start, child_time, alloc0]
        self._attached = False
        self.wall_s = 0.0

    # -- attach / detach ---------------------------------------------------

    def attach(self, model: nn.Module, root: str = "") -> "LayerProfiler":
        if self._attached:
            raise RuntimeError("already attached")
        classes: dict[type, None] = {}
        for name, mod in model.named_modules():
            path = f"{root}.{name}" if root and name else (root or name or "<root>")
            if self.max_depth is not None and path.count(".") > self.max_depth:
                continue
            self._registry[id(mod)] = path
            self.stats[path] = ModuleStat(
                path=path, cls=type(mod).__name__,
                own_params=own_param_bytes(mod),
                gemm_shape=_linear_shape(mod))
            for klass in type(mod).__mro__:
                if "__call__" in klass.__dict__ and klass is not nn.Module:
                    classes[klass] = None
                    break
        for klass in classes:
            self._patch(klass)
        self._attached = True
        return self

    def _patch(self, klass: type) -> None:
        original = klass.__dict__["__call__"]
        prof = self

        def wrapper(self, *args, **kwargs):
            path = prof._registry.get(id(self))
            if path is None:
                return original(self, *args, **kwargs)
            return prof._run(path, self, original, args, kwargs)

        wrapper.__name__ = getattr(original, "__name__", "__call__")
        klass.__call__ = wrapper
        self._patched.append((klass, original))

    def detach(self) -> None:
        for klass, original in reversed(self._patched):
            klass.__call__ = original
        self._patched.clear()
        self._registry.clear()
        self._attached = False

    def __enter__(self) -> "LayerProfiler":
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        if self.mode == "eval":
            mx.synchronize()
        self.wall_s = time.perf_counter() - self._t0
        self.detach()

    # -- the hot path ------------------------------------------------------

    def _run(self, path, module, original, args, kwargs):
        st = self.stats[path]
        sync = self.mode == "eval"
        if sync:
            # Flush the parent's pending ops so their cost lands on the parent
            # and not on this child.
            #
            # mx.synchronize() alone does NOT do this. MLX graphs are lazy, and
            # synchronize only waits on work already submitted to the stream;
            # un-evaluated nodes stay pending. The inputs must be mx.eval'd.
            #
            # Getting this wrong cost real time. Every module was absorbing its
            # parent's unevaluated upstream ops, so the residual adds, the
            # gating, the astypes and the SwiGLU epilogue were all charged to
            # whichever leaf happened to consume them. LayerNorm read as 32% of
            # the trunk while an isolated norm of the same shape and dtype sat
            # at the bandwidth roof.
            #
            # This runs before the timer starts, so the flushed work is billed
            # to the enclosing module's exclusive time, which is where it
            # belongs.
            pending = list(iter_arrays(args)) + list(iter_arrays(kwargs))
            if pending:
                mx.eval(pending)
            mx.synchronize()
        in_bytes = tensor_bytes(args) + tensor_bytes(kwargs)
        if self.record_shapes and len(st.shapes) < 4:
            st.shapes.append([f"{list(a.shape)}{_dt(a.dtype)}"
                              for a in iter_arrays(args)][:4])
        alloc0 = mx.get_active_memory()

        frame = [path, time.perf_counter(), 0.0]
        self._stack.append(frame)
        try:
            out = original(module, *args, **kwargs)
            if sync:
                arrays = list(iter_arrays(out))
                if arrays:
                    mx.eval(arrays)
                mx.synchronize()
        finally:
            self._stack.pop()
            incl = time.perf_counter() - frame[1]

        st.calls += 1
        st.incl_s += incl
        st.excl_s += incl - frame[2]
        st.in_bytes += in_bytes
        st.out_bytes += tensor_bytes(out)
        st.param_bytes += st.own_params
        for a in iter_arrays(args):
            k = _dt(a.dtype)
            self.dtype_bytes[k] = self.dtype_bytes.get(k, 0) + a.nbytes
        for a in iter_arrays(out):
            k = _dt(a.dtype)
            self.dtype_bytes[k] = self.dtype_bytes.get(k, 0) + a.nbytes
        if st.gemm_shape:
            in_f, out_f = st.gemm_shape
            rows = sum(a.size for a in iter_arrays(args)) // max(in_f, 1)
            st.flops += 2.0 * rows * in_f * out_f
        st.alloc_delta += mx.get_active_memory() - alloc0
        if self._stack:
            self._stack[-1][2] += incl
        return out

    # -- reporting ---------------------------------------------------------

    def by_class(self) -> dict[str, ModuleStat]:
        agg: dict[str, ModuleStat] = {}
        for st in self.stats.values():
            if st.calls == 0:
                continue
            a = agg.setdefault(st.cls, ModuleStat(path=f"<all {st.cls}>",
                                                  cls=st.cls))
            a.calls += st.calls
            a.incl_s += st.incl_s
            a.excl_s += st.excl_s
            a.in_bytes += st.in_bytes
            a.out_bytes += st.out_bytes
            a.own_params += st.own_params
            a.param_bytes += st.param_bytes
            a.flops += st.flops
            a.alloc_delta += st.alloc_delta
        return agg

    def active(self) -> list[ModuleStat]:
        return [s for s in self.stats.values() if s.calls > 0]

    def as_dict(self) -> dict:
        return {"mode": self.mode, "wall_s": self.wall_s,
                "dtype_bytes": self.dtype_bytes,
                "modules": [s.as_dict() for s in self.active()],
                "by_class": [s.as_dict() for s in self.by_class().values()]}


# ---------------------------------------------------------------------------
# gputrace capture
# ---------------------------------------------------------------------------


@contextmanager
def capture(path: str):
    """Write a .gputrace for the enclosed work.

    Needs MTL_CAPTURE_ENABLED=1 in the environment before the process starts.
    Open the result in Xcode for per-dispatch ALU utilisation, real DRAM bytes
    and limiter analysis. Keep the region small; traces get huge fast.
    """
    if not mx.metal.is_available():
        yield False
        return
    mx.synchronize()
    mx.metal.start_capture(path)
    try:
        yield True
    finally:
        mx.synchronize()
        mx.metal.stop_capture()


# ---------------------------------------------------------------------------
# text report
# ---------------------------------------------------------------------------


def _fmt_bytes(n: int) -> str:
    for unit, div in (("TB", 2**40), ("GB", 2**30), ("MB", 2**20), ("KB", 2**10)):
        if abs(n) >= div:
            return f"{n/div:.2f}{unit}"
    return f"{n}B"


def _elide(path: str, width: int) -> str:
    """Shorten a dotted path without losing what distinguishes it.

    Cutting either end collides. `atom_encoder` and `atom_decoder` differ in
    the middle, and their heads and tails are identical. So keep the tail
    components whole and abbreviate the leading ones to initials:
    ``...diffusion_module.atom_encoder.atom_transformer.blocks.0.attn``
    becomes ``dm.ae.at.blocks.0.attn``, which still separates ae from ad.
    """
    if len(path) <= width:
        return path
    parts = path.split(".")

    def abbr(c):
        if c.isdigit():          # a block index is the identity of the row
            return c
        return "".join(w[0] for w in c.split("_") if w) or c[:1]

    for keep in range(min(4, len(parts)), 0, -1):
        cand = ".".join([abbr(c) for c in parts[:-keep]] + parts[-keep:])
        if len(cand) <= width:
            return cand
    return cand[-width:]


def _limiter(gbs, tflops, c: Optional[Ceilings]):
    """(verdict, bw_fraction, flop_fraction). Names the binding ceiling.

    A layer is only worth touching if it is far from BOTH ceilings. Near
    either one, the work is already done and a rewrite buys nothing.
    """
    if c is None:
        return "", None, None
    bw = gbs / c.peak_bw_gbs if gbs else None
    fl = tflops / c.peak_gemm_tflops if tflops else None
    best = max([x for x in (bw, fl) if x is not None], default=None)
    if best is None:
        return "container", bw, fl
    which = "GEMM" if (fl is not None and fl == best) else "BW"
    if best >= 0.6:
        return f"at {which} roof", bw, fl
    if best >= 0.25:
        return f"part {which}", bw, fl
    return "HEADROOM", bw, fl


def _pct(x):
    return f"{100*x:.0f}" if x is not None else "-"


def render(stats: list[ModuleStat], ceilings: Optional[Ceilings] = None,
           total_s: Optional[float] = None, top: int = 25,
           title: str = "per-layer", label: str = "layer") -> str:
    rows = sorted(stats, key=lambda s: s.excl_s, reverse=True)
    total = total_s if total_s is not None else sum(s.excl_s for s in rows)

    w = 40
    head = (f"{label:<{w}}{'calls':>7}{'excl s':>9}{'%':>6}"
            f"{'GB/s':>8}{'%BW':>5}{'TFLOP/s':>9}{'%GEMM':>7}")
    if ceilings:
        head += f"{'limiter':>15}"
    out = [f"== {title}", head, "-" * len(head)]
    for s in rows[:top]:
        g, t = s.gbs(), s.tflops()
        v, bw, fl = _limiter(g, t, ceilings)
        pct = 100 * s.excl_s / total if total else 0.0
        line = (f"{_elide(s.path, w-1):<{w}}{s.calls:>7}{s.excl_s:>9.3f}"
                f"{pct:>6.1f}"
                f"{(f'{g:.1f}' if g else '-'):>8}{_pct(bw):>5}"
                f"{(f'{t:.2f}' if t else '-'):>9}{_pct(fl):>7}")
        if ceilings:
            line += f"{v:>15}"
        out.append(line)
    shown = sum(s.excl_s for s in rows[:top])
    out.append(f"{'':<{w}}{'':>7}{shown:>9.3f}"
               f"{100*shown/total if total else 0:>6.1f}   <- shown")
    return "\n".join(out)


def render_ceilings(c: Ceilings) -> str:
    return (f"== device ceilings (measured now, not looked up)\n"
            f"  device                    : {c.device}\n"
            f"  streaming bandwidth       : {c.peak_bw_gbs:8.1f} GB/s\n"
            f"  bf16 GEMM throughput      : {c.peak_gemm_tflops:8.2f} TFLOPS\n"
            f"  bytes per flop at ceiling : "
            f"{c.peak_bw_gbs/1e3/c.peak_gemm_tflops:8.4f}\n"
            f"  ridge point               : {c.ridge:8.1f} FLOP/byte"
            f"   <- above this a layer is compute-bound")


def render_modes(eval_wall: float, lazy_wall: float) -> str:
    frac = 100 * lazy_wall / eval_wall if eval_wall else 0.0
    if frac >= 40:
        v = "DISPATCH-BOUND. Python and graph build dominate; fuse or compile."
    elif frac >= 15:
        v = "meaningful dispatch overhead. Worth reducing op count."
    else:
        v = "device-bound. Overhead is small; optimise the kernels."
    return (f"== dispatch overhead\n"
            f"  graph build only (lazy)   : {lazy_wall:8.3f} s\n"
            f"  with per-layer sync (eval): {eval_wall:8.3f} s\n"
            f"  lazy share                : {frac:8.1f} %\n"
            f"  => {v}")


def save_json(path: str, payload: dict) -> None:
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


# ---------------------------------------------------------------------------
# sweep across sequence lengths
# ---------------------------------------------------------------------------


def fit_exponent(xs: list[float], ys: list[float]) -> Optional[float]:
    """Least-squares slope of log(y) against log(x), i.e. k in y ~ x**k.

    Returns None when there are fewer than two usable points. Timing noise at
    short lengths inflates k, so read it as a shape, not a measurement: ~1 is
    linear in length, ~2 pairwise, ~3 a triangular contraction.
    """
    pts = [(math.log(x), math.log(y)) for x, y in zip(xs, ys) if x > 0 and y > 0]
    if len(pts) < 2:
        return None
    n = len(pts)
    mx_ = sum(p[0] for p in pts) / n
    my = sum(p[1] for p in pts) / n
    num = sum((p[0] - mx_) * (p[1] - my) for p in pts)
    den = sum((p[0] - mx_) ** 2 for p in pts)
    return num / den if den > 0 else None


def render_sweep(rows: list[dict], phases: list[str],
                 key: str = "plain_s") -> str:
    """Per-length phase totals, from the uninstrumented compiled runs.

    `key` selects which timing to table. plain_s is the honest wall clock;
    phase_s is the instrumented one and runs slower by construction.
    """
    head = f"{'L':>6}" + "".join(f"{p[:11]+' s':>13}" for p in phases)
    head += f"{'total s':>10}{'lazy %':>8}{'peak GB':>9}"
    out = ["== phase time vs sequence length  (uninstrumented, compiled)",
           head, "-" * len(head)]
    for r in rows:
        line = f"{r['L']:>6}"
        for p in phases:
            line += f"{r[key].get(p, float('nan')):>13.3f}"
        line += (f"{r['total_s']:>10.3f}{100*r['lazy_share']:>8.1f}"
                 f"{r['peak_gb']:>9.2f}")
        out.append(line)
    if len(rows) >= 2:
        k = fit_exponent([r["L"] for r in rows], [r["total_s"] for r in rows])
        out.append("")
        out.append(f"  total scales as L^{k:.2f}" if k is not None
                   else "  not enough points to fit a scaling exponent")
        for p in phases:
            kp = fit_exponent([r["L"] for r in rows],
                              [r[key].get(p, 0.0) for r in rows])
            if kp is not None:
                out.append(f"    {p:<12} L^{kp:.2f}")
    return "\n".join(out)


def render_scaling(per_class: dict, lengths: list[int],
                   ceilings: Optional[Ceilings] = None, top: int = 20,
                   title: str = "per-class scaling",
                   label: str = "class") -> str:
    """Rank module classes by cost at the longest length, with an exponent.

    Share alone misleads. A class at 10% with k=3 overtakes one at 30% with
    k=1, so both columns have to be read together.
    """
    Lmax = max(lengths)
    rank = sorted(per_class.items(),
                  key=lambda kv: kv[1].get(Lmax, {}).get("excl_s", 0.0),
                  reverse=True)
    total = sum(v.get(Lmax, {}).get("excl_s", 0.0) for _, v in rank) or 1.0

    w = 32
    head = (f"{label:<{w}}{'calls':>7}{f'L={Lmax} s':>11}{'%':>6}{'L^k':>7}"
            f"{'GB/s':>8}{'%BW':>5}{'TFLOP/s':>9}{'%GEMM':>7}")
    if ceilings:
        head += f"{'limiter':>15}"
    out = [f"== {title}  (ranked at L={Lmax})", head, "-" * len(head)]
    for cls, byL in rank[:top]:
        at = byL.get(Lmax)
        if not at:
            continue
        k = fit_exponent(lengths,
                         [byL.get(L, {}).get("excl_s", 0.0) for L in lengths])
        g, t = at.get("gbs"), at.get("tflops")
        v, bw, fl = _limiter(g, t, ceilings)
        line = (f"{_elide(cls, w-1):<{w}}{at['calls']:>7}{at['excl_s']:>11.3f}"
                f"{100*at['excl_s']/total:>6.1f}"
                f"{(f'{k:.2f}' if k is not None else '-'):>7}"
                f"{(f'{g:.1f}' if g else '-'):>8}{_pct(bw):>5}"
                f"{(f'{t:.2f}' if t else '-'):>9}{_pct(fl):>7}")
        if ceilings:
            line += f"{v:>15}"
        out.append(line)
    return "\n".join(out)


def render_verdict(rows: list[dict], per_class: dict, lengths: list[int],
                   ceilings: Ceilings, top: int = 5) -> str:
    """The short answer: what to look at first, and why."""
    Lmax = max(lengths)
    lazy = rows[-1]["lazy_share"] if rows else 0.0
    rank = sorted(per_class.items(),
                  key=lambda kv: kv[1].get(Lmax, {}).get("excl_s", 0.0),
                  reverse=True)
    total = sum(v.get(Lmax, {}).get("excl_s", 0.0) for _, v in rank) or 1.0

    out = [f"== where the time goes at L={Lmax}"]
    if lazy >= 0.40:
        out.append(f"  Dispatch-bound: {100*lazy:.0f}% is Python and graph "
                   "build. Cut op count. Kernels will not help.")
    elif lazy >= 0.15:
        out.append(f"  {100*lazy:.0f}% is Python and graph build. Real, but "
                   "not the main cost.")
    else:
        out.append(f"  Device-bound: only {100*lazy:.1f}% is Python and graph "
                   "build. The time is in kernels.")
    out.append("")
    for cls, byL in rank[:top]:
        at = byL.get(Lmax)
        if not at or at["excl_s"] <= 0:
            continue
        k = fit_exponent(lengths,
                         [byL.get(L, {}).get("excl_s", 0.0) for L in lengths])
        grow = (f", L^{k:.1f}" if k is not None else "")
        g, t = at.get("gbs"), at.get("tflops")
        v, bw, fl = _limiter(g, t, ceilings)
        share = 100 * at["excl_s"] / total
        if v == "container":
            out.append(f"  {share:5.1f}%  {cls:<30} container{grow}")
            out.append("           -> cost is in its children")
            continue
        rate = []
        if g:
            rate.append(f"{g:.0f} GB/s = {100*bw:.0f}% of BW")
        if t:
            rate.append(f"{t:.2f} TFLOP/s = {100*fl:.0f}% of GEMM")
        out.append(f"  {share:5.1f}%  {cls:<30} {', '.join(rate)}{grow}")
        if v == "HEADROOM":
            out.append("           -> FAR from both ceilings. Real headroom "
                       "here; this is where to look.")
        elif v.startswith("at "):
            out.append(f"           -> already {v}. Nothing to win without "
                       "changing the maths.")
        else:
            out.append(f"           -> {v}; partial headroom.")
    return "\n".join(out)



# ---------------------------------------------------------------------------
# seeing inside mx.compile
# ---------------------------------------------------------------------------


COMPILED_ATTR_PAIRS = (("_compiled", "_apply_blocks"),)


@contextmanager
def bypass_compile(model: nn.Module,
                   pairs=COMPILED_ATTR_PAIRS) -> Iterator[int]:
    """Route compiled call sites back through their Python implementation.

    mx.compile traces once and replays the captured graph, so submodule
    __call__ never runs again after the first call. A profiler sees nothing
    inside a compiled region and charges the whole cost to the enclosing
    module. FoldingTrunk does exactly this, which hides the entire pair stack.

    mlx.gc_func exposes no handle on the original function, so the bypass has
    to know the convention: a module holding `_compiled = mx.compile(self.f)`
    also still has `f`. Swap the attribute, restore on exit.

    Bypassing is not free: the uncompiled path loses fusion, so absolute times
    grow. Use it for attribution, and time the compiled path separately for
    the honest number.
    """
    saved = []
    for _, mod in model.named_modules():
        for compiled_attr, plain_attr in pairs:
            fn = getattr(mod, plain_attr, None)
            if compiled_attr in mod.__dict__ and callable(fn):
                saved.append((mod, compiled_attr, mod.__dict__[compiled_attr]))
                mod.__dict__[compiled_attr] = fn
    try:
        yield len(saved)
    finally:
        for mod, attr, original in saved:
            mod.__dict__[attr] = original


def time_plain(fn: Callable[[], Any], warmup: int = 1) -> float:
    """Wall time with nothing attached: the number to trust for totals."""
    for _ in range(warmup):
        mx.eval(fn())
    mx.synchronize()
    t0 = time.perf_counter()
    out = fn()
    mx.eval(out)
    mx.synchronize()
    dt = time.perf_counter() - t0
    del out
    return dt


# ---------------------------------------------------------------------------
# memory headroom, i.e. how close we are to swapping
# ---------------------------------------------------------------------------


def memory_info() -> dict:
    # mx.metal.device_info is deprecated in favour of mx.device_info.
    getter = getattr(mx, "device_info", None) or mx.metal.device_info
    try:
        info = getter()
    except Exception:
        return {}
    if not info.get("memory_size"):
        return {}
    return {
        "memory_size": info.get("memory_size", 0),
        "max_recommended_working_set_size": info.get(
            "max_recommended_working_set_size", 0),
        "architecture": info.get("architecture", "?"),
    }


def render_memory(peak_gb: float, info: Optional[dict] = None) -> str:
    info = info if info is not None else memory_info()
    if not info:
        return f"== memory\n  peak {peak_gb:.2f} GB  (no Metal device info)"
    total = info.get("memory_size", 0) / 2**30
    rec = info.get("max_recommended_working_set_size", 0) / 2**30
    out = ["== memory",
           f"  architecture              : {info.get('architecture','?')}",
           f"  unified memory            : {total:8.2f} GB",
           f"  max recommended workingset: {rec:8.2f} GB",
           f"  peak this run             : {peak_gb:8.2f} GB"]
    if rec > 0:
        frac = peak_gb / rec
        out.append(f"  used of recommended       : {100*frac:8.1f} %")
        if frac >= 1.0:
            out.append("  => OVER the recommended working set. Metal is spilling "
                       "to swap and every timing above is suspect.")
        elif frac >= 0.85:
            out.append("  => close to the limit. Drop --seq-len or pass --skip-lm "
                       "before trusting the long lengths.")
        else:
            out.append("  => headroom is fine; timings are not swap-contaminated.")
    return "\n".join(out)


def render_shapes(stats: list[ModuleStat], ceilings: Optional[Ceilings] = None,
                  top: int = 25, title: str = "actual shapes") -> str:
    """Per-call cost next to the real input shape.

    Aggregates hide this. A class total tells you nothing about whether one
    call is slow or there are simply many, and an assumed shape is how you
    end up benchmarking a tensor the model never produces.
    """
    rows = sorted(stats, key=lambda s: s.excl_s, reverse=True)
    w = 38
    head = (f"{'layer':<{w}}{'cls':>18}{'calls':>7}{'ms/call':>10}"
            f"{'MB/call':>9}{'GB/s':>8}  input shapes")
    out = [f"== {title}", head, "-" * (len(head) + 18)]
    for s in rows[:top]:
        g = s.gbs()
        shp = "; ".join(str(x) for x in (s.shapes[0] if s.shapes else [])) or "-"
        out.append(
            f"{_elide(s.path, w-1):<{w}}{s.cls[:17]:>18}{s.calls:>7}"
            f"{1e3*s.excl_s/max(s.calls,1):>10.2f}"
            f"{s.moved_bytes/max(s.calls,1)/2**20:>9.1f}"
            f"{(f'{g:.1f}' if g else '-'):>8}  {shp[:72]}")
    return "\n".join(out)


def render_dtype_mix(dtype_bytes: dict, title: str = "traffic by dtype") -> str:
    """Where the bytes actually go, by element type.

    Asking for bfloat16 does not mean you get it. A single fp32 tensor early
    in the graph promotes every downstream op, and nothing in a timing table
    shows that -- the layer just looks twice as expensive as it should.
    """
    total = sum(dtype_bytes.values()) or 1
    out = [f"== {title}",
           f"  {'dtype':<10}{'GB':>10}{'%':>8}", "  " + "-" * 26]
    for k, v in sorted(dtype_bytes.items(), key=lambda kv: -kv[1]):
        out.append(f"  {k:<10}{v/1e9:>10.2f}{100*v/total:>8.1f}")
    f32 = dtype_bytes.get("f32", 0)
    if f32 / total >= 0.5:
        out.append(f"  => {100*f32/total:.0f}% of traffic is fp32. Every "
                   "bandwidth-bound op is paying 2x.")
        out.append("     Find the first fp32 tensor and cast it; promotion "
                   "spreads downstream.")
    return "\n".join(out)


def render_portability(per_class: dict, lengths: list[int], here: Ceilings,
                       targets: list[tuple[str, float, float]],
                       top: int = 10) -> str:
    """Would this layer still be the bottleneck on a different chip?

    A percentage measured on one machine does not transfer to another. What
    does transfer is arithmetic intensity: FLOP per byte is a property of the
    computation, not the hardware. Compare it against each chip's ridge point
    and the binding constraint falls out.

    Only rows with exactly-counted FLOPs can be placed. Memory-bound rows with
    no FLOP count (the norms, the elementwise epilogues) stay memory-bound on
    every chip listed here, since every ridge point is far above 1 FLOP/byte.
    """
    Lmax = max(lengths)
    rank = sorted(per_class.items(),
                  key=lambda kv: kv[1].get(Lmax, {}).get("excl_s", 0.0),
                  reverse=True)
    total = sum(v.get(Lmax, {}).get("excl_s", 0.0) for _, v in rank) or 1.0
    chips = [(f"{here.device} (measured)", here.peak_bw_gbs,
              here.peak_gemm_tflops)] + list(targets)

    out = ["== does this transfer to another chip?",
           "  Arithmetic intensity is a property of the maths, not the "
           "machine. Compare",
           "  it to each chip's ridge point to see what binds there.", ""]
    out.append("  ridge points (FLOP/byte):")
    for name, bw, tf in chips:
        out.append(f"    {name:<28}{(tf*1e12)/(bw*1e9):8.1f}")
    out.append("")
    head = f"  {'class':<26}{'%here':>7}{'FLOP/byte':>11}" + \
           "".join(f"{n.split()[0][:11]:>13}" for n, _, _ in chips)
    out += [head, "  " + "-" * (len(head) - 2)]
    for cls, byL in rank[:top]:
        at = byL.get(Lmax)
        if not at or at["excl_s"] <= 0:
            continue
        flops, mb = at.get("flops", 0.0), at.get("moved_bytes", 0)
        share = 100 * at["excl_s"] / total
        if at.get("gbs") is None:
            # Container: its children did the work, so it has no rate of its
            # own and cannot be placed on either side of a ridge.
            line = (f"  {cls[:25]:<26}{share:>7.1f}{'-':>11}"
                    + "".join(f"{'n/a':>13}" for _ in chips))
        elif flops <= 0 or mb <= 0:
            line = (f"  {cls[:25]:<26}{share:>7.1f}{'-':>11}"
                    + "".join(f"{'memory':>13}" for _ in chips))
        else:
            ai = flops / mb
            line = f"  {cls[:25]:<26}{share:>7.1f}{ai:>11.1f}"
            for _, bw, tf in chips:
                line += f"{('compute' if ai >= (tf*1e12)/(bw*1e9) else 'memory'):>13}"
        out.append(line)
    out.append("")
    out.append("  A row that flips to 'memory' on the target is one where a "
               "fusion that looks\n  marginal here pays more there. A row that "
               "stays 'compute' gains from a\n  faster GEMM, which the target "
               "already provides for free.")
    return "\n".join(out)
