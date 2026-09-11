import mlx.core as mx, mlx.nn as nn
from mlx_lm.esm_profiler import LayerProfiler, measure_ceilings, render, render_ceilings

class Block(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.a = nn.Linear(d, 4*d)
        self.b = nn.Linear(4*d, d)
        self.norm = nn.LayerNorm(d)
    def __call__(self, x):
        return x + self.b(nn.gelu(self.a(self.norm(x))))

class Net(nn.Module):
    def __init__(self, d, n):
        super().__init__()
        self.blocks = [Block(d) for _ in range(n)]
        self.head = nn.Linear(d, 7)
    def __call__(self, x):
        for blk in self.blocks:
            x = blk(x)
        return self.head(x)

D, N = 256, 4
net = Net(D, N)
x = mx.random.normal((64, D))
mx.eval(net.parameters(), x)

# outsider: same class as registered modules, must NOT be counted
outsider = nn.Linear(D, D)
mx.eval(outsider.parameters())

orig_linear_call = nn.Linear.__call__

prof = LayerProfiler(mode="eval")
prof.attach(net, root="net")
with prof:
    y = net(x)
    mx.eval(y)
    _ = outsider(x); mx.eval(_)

print("=== 1. __call__ restored after detach:",
      nn.Linear.__call__ is orig_linear_call)

act = prof.active()
root = prof.stats["net"]
excl_sum = sum(s.excl_s for s in act)
print(f"=== 2. exclusive sum {excl_sum:.6f} vs root inclusive {root.incl_s:.6f}  "
      f"ratio {excl_sum/root.incl_s:.4f}")

print("=== 3. outsider excluded:", prof.stats["net"].calls == 1,
      "| registered module count:", len(act))

names = sorted(s.path for s in act)
print("=== 4. paths:", names)

# weights charged once per call
la = prof.stats["net.blocks.0.a"]
expect = (D*4*D + 4*D) * 4   # fp32 weight + bias
print(f"=== 5. blocks.0.a own_params={la.own_params} expected={expect} "
      f"calls={la.calls} param_bytes={la.param_bytes} "
      f"ok={la.own_params==expect and la.param_bytes==expect*la.calls}")

by_cls = prof.by_class()
lin = by_cls["Linear"]
n_lin = 2*N + 1
print(f"=== 6. Linear aggregated calls={lin.calls} expected={n_lin} "
      f"param_bytes consistency ok="
      f"{lin.param_bytes == sum(prof.stats[p].param_bytes for p in prof.stats if prof.stats[p].cls=='Linear')}")

print()
c = measure_ceilings(dtype=mx.float32, bw_mb=64, gemm_n=1024, iters=5)
print(render_ceilings(c)); print()
print(render(act, ceilings=c, top=8, title="synthetic net"))
