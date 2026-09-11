"""Is the GPU actually busy during a fold, or idle between kernels?

Every "% of roof" number in PROFILING_RESULTS.md is wall-clock based, so it
cannot tell a slow kernel from an idle GPU. A layer sitting at 0.3% of BOTH
the compute and the bandwidth ceiling -- which the sampler's atom transformer
does -- is the signature of idle, not slow. Nothing else we have measures it.

powermetrics reports GPU HW active residency, which is the fraction of
wall-clock the GPU was executing anything at all. Run a workload under it:

    sudo python devtools/gpu_busy.py trunk 500
    sudo python devtools/gpu_busy.py sampler 500

Residency near 100% means the GPU is saturated and the only way forward is
better kernels. Residency well below means there are dispatch bubbles, and
the fix is fewer, larger kernels -- a completely different job.

Needs sudo because powermetrics does. If you would rather not, pass --manual
and it prints the two commands to run in separate terminals.
"""

import argparse
import json
import re
import subprocess
import sys
import threading
import time

import mlx.core as mx

RE_RESIDENCY = re.compile(r"GPU HW active residency:\s+([\d.]+)%")
RE_FREQ = re.compile(r"GPU HW active frequency:\s+([\d.]+)\s*MHz")
RE_POWER = re.compile(r"GPU Power:\s+([\d.]+)\s*mW")


def build_trunk(L, d_pair=256, n_layers=24):
    from mlx_lm.models.esmfold2 import FoldingTrunk

    mx.random.seed(0)
    trunk = FoldingTrunk(n_layers=n_layers, d_pair=d_pair)
    trunk.set_dtype(mx.bfloat16)
    trunk.eval()
    z = mx.random.normal((1, L, L, d_pair)).astype(mx.float32)
    mask = mx.ones((1, L, L)).astype(mx.float32)
    mx.eval(trunk.parameters(), z, mask)
    return lambda: trunk(z, mask=mask)


def build_gemm(L, d_pair=256):
    """A single large GEMM, as a positive control. This should saturate."""
    M = L * L
    x = mx.random.normal((M, d_pair)).astype(mx.bfloat16)
    w = mx.random.normal((4 * d_pair, d_pair)).astype(mx.bfloat16)
    mx.eval(x, w)
    return lambda: x @ w.T


def build_elementwise(L, d_pair=256):
    """A single large elementwise op, the bandwidth control."""
    z = mx.random.normal((1, L, L, d_pair)).astype(mx.bfloat16)
    mx.eval(z)
    return lambda: z + 1.0


WORKLOADS = {"trunk": build_trunk, "gemm": build_gemm,
             "elementwise": build_elementwise}


def spin(fn, seconds, stop):
    """Keep the GPU fed for `seconds` so powermetrics has something to see."""
    n = 0
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds and not stop.is_set():
        mx.eval(fn())
        n += 1
    mx.synchronize()
    return n, time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("workload", choices=list(WORKLOADS))
    ap.add_argument("seq_len", type=int, nargs="?", default=500)
    ap.add_argument("--seconds", type=float, default=12.0)
    ap.add_argument("--interval-ms", type=int, default=200)
    ap.add_argument("--manual", action="store_true")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    if args.manual:
        print("Terminal 1:")
        print(f"  sudo powermetrics --samplers gpu_power -i {args.interval_ms}"
              f" -n {int(args.seconds*1000/args.interval_ms)} > gpu.txt")
        print("Terminal 2, immediately after:")
        print(f"  python devtools/gpu_busy.py {args.workload} {args.seq_len}"
              " --seconds 12 --no-sample")
        print("Then grep gpu.txt for 'GPU HW active residency'.")
        return

    fn = WORKLOADS[args.workload](args.seq_len)
    mx.eval(fn())          # warm up and compile outside the measurement
    mx.synchronize()

    n_samples = max(int(args.seconds * 1000 / args.interval_ms), 4)
    cmd = ["powermetrics", "--samplers", "gpu_power",
           "-i", str(args.interval_ms), "-n", str(n_samples)]
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL, text=True)
    except (FileNotFoundError, PermissionError) as e:
        print(f"could not start powermetrics ({e}). Re-run under sudo, or "
              "use --manual.")
        sys.exit(1)

    stop = threading.Event()
    iters, elapsed = spin(fn, args.seconds, stop)
    out, _ = proc.communicate(timeout=30)

    res = [float(x) for x in RE_RESIDENCY.findall(out)]
    freq = [float(x) for x in RE_FREQ.findall(out)]
    pwr = [float(x) for x in RE_POWER.findall(out)]
    if not res:
        print("powermetrics produced no GPU samples. It needs root: try\n"
              f"  sudo {' '.join(sys.argv)}")
        sys.exit(1)

    # Drop the first sample: it straddles the start and reads low.
    res, freq, pwr = res[1:] or res, freq[1:] or freq, pwr[1:] or pwr
    mean_res = sum(res) / len(res)

    print(f"workload={args.workload}  L={args.seq_len}  "
          f"{iters} iterations in {elapsed:.1f}s")
    print(f"samples={len(res)} at {args.interval_ms}ms\n")
    print(f"  GPU active residency : {mean_res:6.1f} %   "
          f"(min {min(res):.1f}, max {max(res):.1f})")
    if freq:
        print(f"  GPU active frequency : {sum(freq)/len(freq):6.0f} MHz")
    if pwr:
        print(f"  GPU power            : {sum(pwr)/len(pwr):6.0f} mW")
    print()
    if mean_res >= 95:
        print("  => SATURATED. The GPU is never idle. Only better kernels help.")
    elif mean_res >= 80:
        print(f"  => {100-mean_res:.0f}% idle. Some dispatch bubbles, but the "
              "kernels dominate.")
    else:
        print(f"  => {100-mean_res:.0f}% IDLE. The GPU is waiting, not "
              "computing.\n     Fewer and larger kernels, not faster ones.")
    print("\n  Compare against the 'gemm' and 'elementwise' workloads: those")
    print("  are single large kernels and should both read near 100%. Any gap")
    print("  between them and 'trunk' is dispatch overhead the trunk pays.")

    if args.json:
        with open(args.json, "w") as f:
            json.dump({"workload": args.workload, "seq_len": args.seq_len,
                       "residency_pct": res, "freq_mhz": freq,
                       "power_mw": pwr, "iters": iters,
                       "elapsed_s": elapsed}, f, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
