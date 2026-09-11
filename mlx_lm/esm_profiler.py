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
import time
from collections import defaultdict
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

    def as_dict(self) -> dict:
        return {"device": self.device, "peak_bw_gbs": self.peak_bw_gbs,
                "peak_gemm_tflops": self.peak_gemm_tflops, "detail": self.detail}


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
    shapes: list = field(default_factory=list)

    @property
    def moved_bytes(self) -> int:
        """Logical traffic charged to this module's own work."""
        return self.in_bytes + self.out_bytes + self.param_bytes

    def gbs(self) -> float:
        return self.moved_bytes / self.excl_s / 1e9 if self.excl_s > 0 else 0.0

    def as_dict(self) -> dict:
        return {"path": self.path, "cls": self.cls, "calls": self.calls,
                "incl_s": self.incl_s, "excl_s": self.excl_s,
                "in_bytes": self.in_bytes, "out_bytes": self.out_bytes,
                "own_params": self.own_params, "param_bytes": self.param_bytes,
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
                own_params=own_param_bytes(mod))
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
            # Flush the parent's own pending ops so their cost is charged to
            # the parent, not to this child.
            mx.synchronize()
        in_bytes = tensor_bytes(args) + tensor_bytes(kwargs)
        if self.record_shapes and len(st.shapes) < 4:
            st.shapes.append([list(a.shape) for a in iter_arrays(args)][:4])
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
            a.alloc_delta += st.alloc_delta
        return agg

    def active(self) -> list[ModuleStat]:
        return [s for s in self.stats.values() if s.calls > 0]

    def as_dict(self) -> dict:
        return {"mode": self.mode, "wall_s": self.wall_s,
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
    abbr = lambda c: "".join(w[0] for w in c.split("_") if w) or c[:1]
    for keep in range(min(4, len(parts)), 0, -1):
        cand = ".".join([abbr(c) for c in parts[:-keep]] + parts[-keep:])
        if len(cand) <= width:
            return cand
    return cand[-width:]


def render(stats: list[ModuleStat], ceilings: Optional[Ceilings] = None,
           total_s: Optional[float] = None, top: int = 25,
           title: str = "per-layer", label: str = "layer") -> str:
    rows = sorted(stats, key=lambda s: s.excl_s, reverse=True)
    total = total_s if total_s is not None else sum(s.excl_s for s in rows)
    peak = ceilings.peak_bw_gbs if ceilings else None

    w = 44
    head = (f"{label:<{w}}{'calls':>7}{'excl s':>9}{'%':>6}"
            f"{'incl s':>9}{'traffic':>10}{'GB/s':>9}")
    if peak:
        head += f"{'%BW':>7}{'verdict':>14}"
    out = [f"== {title}", head, "-" * len(head)]
    for s in rows[:top]:
        name = _elide(s.path, w - 1)
        pct = 100 * s.excl_s / total if total else 0.0
        line = (f"{name:<{w}}{s.calls:>7}{s.excl_s:>9.3f}{pct:>6.1f}"
                f"{s.incl_s:>9.3f}{_fmt_bytes(s.moved_bytes):>10}"
                f"{s.gbs():>9.1f}")
        if peak:
            frac = s.gbs() / peak
            if frac >= 0.6:
                v = "BW-BOUND"
            elif frac >= 0.25:
                v = "part BW"
            else:
                v = "not BW"
            line += f"{100*frac:>7.0f}{v:>14}"
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
            f"{c.peak_bw_gbs/1e3/c.peak_gemm_tflops:8.4f}"
            f"   <- a layer below this ratio is compute-bound")


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
