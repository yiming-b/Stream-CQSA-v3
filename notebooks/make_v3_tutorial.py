"""
Generate (and execute) the Stream-CQSA v3 tutorial notebook.

    python -P notebooks/make_v3_tutorial.py --no-exec      # write notebooks/stream_cqsa_v3_tutorial.ipynb
    python -P notebooks/make_v3_tutorial.py                # write + execute (needs a GPU; sbatch slurm/v3/notebook.slurm)

Every cell runs in seconds on one A100-80GB; the device budgets (`hardware=`) stand in for a smaller card so the
decomposition paths run at sizes that fit a tutorial. Every comparison against SDPA is made on a call that was
forced to decompose and asserted to have decomposed (tests/conftest.py's rule: an unpinned call that fits IS SDPA).
"""
import argparse, os, time
import nbformat as nbf

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "stream_cqsa_v3_tutorial.ipynb")

cells = []
def md(s): cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
def code(s): cells.append(nbf.v4.new_code_cell(s.strip("\n")))

md("""
# Stream-CQSA v3 tutorial: exact attention that always fits

Stream-CQSA is **exact out-of-memory recovery for attention**. When an attention call does not fit in device
memory, it decomposes the pair set over a cyclic quorum set into independent subproblems, runs them one at a
time (or a few at a time), and recomposes the exact result. It defines no attention rule of its own: it
reproduces whatever kernel it wraps, to the kernel's own precision.

There is **one function**, `stream_cqsa.attention`, with the signature of
`torch.nn.functional.scaled_dot_product_attention`. Below the memory boundary the call *is* the normal kernel;
above it the planner chooses the decomposition from the memory available and the call runs exactly, recovering
from out-of-memory on its own. Gradients flow when the inputs require them. Grouped-query attention works.

This notebook: 1 install and the one call · 2 what happens when it does not fit · 3 the planner's choices ·
4 a dry run before a huge call · 5 inputs in host memory · 6 inside a model (Q/K/V already on a full GPU) ·
7 training · 8 grouped-query attention · 9 pinning a setting · 10 routing an existing model ·
11 where the time and memory go, and where to read more.

A **device budget** (`hardware={"cuda:0": "...GiB", "host": "...GiB"}`) describes a smaller card so the recovery
paths run at sizes that fit a tutorial. On a real small card you pass nothing: the budget is detected.
""")

md("""
## 1. Install, and the one call

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126   # match your CUDA
pip install stream-cqsa                                                # Triton kernels, no compilation
stream-cqsa-doctor                                                     # what loaded, and how far this machine can go
```

The PyPI package carries the Triton kernels and runs on any recent NVIDIA GPU with no compiler. The CUDA
kernels (the classic module and the wave kernel, the fastest forward and backward) come as prebuilt wheels
per python / torch / CUDA / GPU from the GitHub release page, or from a source build in a checkout
(`pip install -e . --no-build-isolation`). The package uses whatever is installed; nothing else changes.
""")
code("""
import torch, stream_cqsa, torch.nn.functional as F
print("stream_cqsa", stream_cqsa.__version__, "|", torch.cuda.get_device_name(0))
print("kernels:", stream_cqsa.kernels_available())
""")
code("""
# The call. Q/K/V in SDPA's [B, H, N, D] layout; fp16 or bf16; host or device tensors.
B, H, N, D = 1, 8, 262_144, 64
q, k, v = (torch.randn(B, H, N, D, device="cuda", dtype=torch.float16) for _ in range(3))
out, info = stream_cqsa.attention(q, k, v, is_causal=True, verbose=True, return_info=True)
ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
print(tuple(out.shape), out.dtype, "| engine:", info["engine"], "| max |out - SDPA| =", (out.float() - ref.float()).abs().max().item())
""")
md("""
The call fits on this card, so what ran **is** the SDPA kernel (`engine: monolithic`): the difference of 0 is the
pass-through, not evidence about the decomposition. `verbose=True` printed the one line that states every
automatic choice: monolithic or decomposed (depth `itr`, quorum size `c`, number of subproblems), where Q/K/V
are, where the accumulator is, the engine and kernel, and where the output goes.
""")

md("""
## 2. The decomposition, and when it does not fit

What the decomposition is: the N tokens are cut into c = 7 chunks; each of the 7 subproblems attends 3 of them
(its chunk and two quorum partners, the perfect difference set {0, 1, 3} mod 7), so any two chunks meet in exactly
one subproblem and the pair set is **partitioned**: no pair is computed twice, none is missed. Each subproblem is
3/7 of the tokens; its partial output and log-sum-exp are merged into the result with the same max-shifted
arithmetic FlashAttention uses inside one kernel. Depth 2 applies the same cut to each subproblem (49 subproblems
of 9/49 of the tokens), and so on.

Pinning the depth (`itr=1`) forces the decomposition even though the call fits. This is how the README's
measurements isolate what decomposing costs: at 1M tokens on an A100 the 7-subproblem forward runs in
0.98-1.02x the time of one FlashAttention-2 call.
""")
code("""
out1, info = stream_cqsa.attention(q, k, v, is_causal=True, itr=1, verbose=True, return_info=True)
assert info["engine"] != "monolithic" and info["n_subproblems"] == 7, info
err = lambda a, b: ((a.float() - b.float()).norm() / b.float().norm()).item()
print("ran:", info["engine"], "engine on the", info["kernel"], "kernel,", info["n_subproblems"], "subproblems of 3/7 of the tokens"
      + (f" in {info['n_waves']} launch(es)" if info.get("n_waves") else ""), "| rel error vs SDPA:", f"{err(out1, ref):.1e}")
""")
md("""
The decomposed result agrees with SDPA to the fp16 kernel's own rounding (a norm-relative error of about 1e-4; the
monolithic fp16 kernel is at the same distance from a float64 reference).

Now the case the package exists for: a card that cannot hold the monolithic call. Describe one (Q, K, V and the
output of this call are 0.25 GiB each; the kernel needs about 1.2 GiB) and the planner answers with the fastest
configuration that fits the budget. Here Q/K/V are already on the device and count against the budget, so it
releases them to pinned host memory for the call, streams the subproblems from there, keeps the fp32 accumulator
in host memory, delivers the output on the device and brings Q/K/V back. `return_plan=True` hands back the plan
with its reason.
""")
code("""
small = {"cuda:0": "1.05GiB", "host": "64GiB"}
out2, plan, info = stream_cqsa.attention(q, k, v, is_causal=True, hardware=small, verbose=True, return_plan=True, return_info=True)
assert info["engine"] != "monolithic" and info["n_subproblems"] > 1, info     # the comparison below is only meaningful for a decomposed run
print("plan:", plan.name())
print("why:", plan.reason)
print(f"predicted: {plan.est_time_s:.2f} s, {plan.est_peak_gib:.2f} GiB on the device, {plan.est_host_gib:.2f} GiB of host memory")
print("ran:", info["engine"], "engine,", info["n_subproblems"], "subproblems | output on", out2.device, "| Q back on", q.device,
      "| rel error vs SDPA:", f"{err(out2, ref):.1e}")
""")
md("""
Why not simply keep Q/K/V on the device and decompose? With the fp32 accumulator and output next to Q/K/V, a
device-resident decomposition needs *more* memory than the monolithic kernel (which is why, with nothing pinned,
the monolithic call is taken whenever it fits). Under a budget the planner moves the O(N) terms off the device:
the accumulator first (section 7), the inputs when that is not enough (sections 5, 6). The `README` measurements
(E0-E12) give the time and memory of every configuration at 256K-8M tokens on an A100.
""")

md("""
## 3. What the planner considered

The plan is chosen by a memory model and a cost model over every candidate: depth, quorum size, which engine
(the **classic** engine runs one subproblem per kernel launch and streams; the **wave** engine runs several
subproblems in one launch), the wave budget, whether the subproblems' Q/K/V are gathered or read in place, how
many subproblems are in flight, and where the inputs and the accumulator live. Among the feasible candidates,
the fastest wins, except that a configuration with a smaller device peak is taken when it is predicted within
15% of the fastest (`tol=`, `CQSA_PLAN_TOL`): the trade that puts the fp32 accumulator in host memory from
about 1M tokens up on an A100. `plan_only=True` returns the plan without computing anything.
""")
code("""
p = stream_cqsa.attention(q, k, v, is_causal=True, hardware={"cuda:0": "2GiB", "host": "64GiB"}, plan_only=True)
print("pick:", p.name(), f"| {p.est_time_s:.2f} s predicted, {p.est_peak_gib:.2f} GiB")
feas = sorted([c for c in p.candidates if c["ok"]], key=lambda c: c["time"])
print(f"{len(p.candidates)} candidates, {len(feas)} fit the 2 GiB budget; the fastest eight:")
print(f"{'engine':8} {'itr':>3} {'c':>3} {'acc':>4} {'Q/K/V':>6} {'W tokens':>9} {'in flight':>9} {'peak GiB':>9} {'time s':>7}")
for c in feas[:8]:
    print(f"{c.get('engine', 'classic'):8} {c['itr']:>3} {c['c']:>3} {c['acc']:>4} {'host' if c['stream_from_host'] else 'device':>6} "
          f"{c.get('wave_tokens', 0) or '-':>9} {c.get('n_par', 1):>9} {c['peak']:>9.2f} {c['time']:>7.2f}")
""")

md("""
## 4. A dry run before a huge call

`stream_cqsa.estimate(N=...)` prints what `attention` would do for a shape on this machine, forward and backward,
and what it would cost, without allocating anything. `stream_cqsa.calibrate()` (about a minute, cached in
`~/.cache/stream_cqsa`) fits the cost model's rates to the GPU at hand; the defaults are an A100's.
""")
code("""
est = stream_cqsa.estimate(N=4_194_304, H=8, D=64)
""")

md("""
## 5. Inputs in host memory

On a shared or small GPU, creating the inputs is the one step outside the package's control. Create them in
host memory and `attention` streams them: only the subproblem in flight is ever on the device. The output comes
back where the inputs were. `stream_cqsa.place_inputs(q, k, v)` is the guardrail in between: it moves to the GPU
what fits and leaves the rest pinned in host memory.

Here the inputs are host tensors and the budget says the device holds 4 GiB: the planner streams the subproblems
from the host and keeps the fp32 accumulator in host memory too (`acc=cpu`), so the device footprint is
O(subproblem), not O(N). On this card without a budget, a 1M call would simply be copied over and run monolithically.

One habit: store host tensors **token-major** (`[B, N, H, D]` in memory, viewed as `[B, H, N, D]`). The streaming
engine gathers whole tokens; a head-major host tensor costs one full copy per call (a warning says so).
""")
code("""
Nh = 1_048_576
qh, kh, vh = (torch.randn(B, Nh, H, D, dtype=torch.float16).transpose(1, 2) for _ in range(3))   # 1 GiB each, token-major, [B, H, N, D] views
torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
outh, plan, info = stream_cqsa.attention(qh, kh, vh, is_causal=True, hardware={"cuda:0": "4GiB", "host": "64GiB"}, verbose=True,
                                         return_plan=True, return_info=True)
assert info["engine"] != "monolithic", info
print("output on", outh.device, "| device peak during the call:", round(torch.cuda.max_memory_allocated() / 2**30, 2), "GiB",
      "| plan:", plan.name(), "| Q/K/V streamed from the host:", plan.stream_from_host, "| accumulator on:", plan.acc)
# spot check against SDPA: the first 64K rows attend the first 64K keys only (causal), so their exact reference is cheap
r = 65536
refh = F.scaled_dot_product_attention(qh[:, :, :r].cuda(), kh[:, :, :r].cuda(), vh[:, :, :r].cuda(), is_causal=True)
print("rel error vs SDPA on the first 64K rows:", f"{err(outh[:, :, :r].cuda(), refh):.1e}")
del refh, qh, kh, vh, outh; torch.cuda.empty_cache()
""")

md("""
## 6. Inside a model: Q/K/V already on a full GPU

In a model, Q/K/V are on the device and little room is left next to them. `attention` then releases them to
pinned host memory for the computation (the caller's tensors are rebound, so the device memory is really freed),
streams from there, puts the output on the device first, and brings Q/K/V back if room remains.
`evict_inputs=` / `restore_inputs=` take `"auto"` (default), `True` or `False`; `return_info=True` reports what
happened. Memory is only freed when nothing else references the storage (`del` a fused QKV tensor first).
""")
code("""
qd, kd, vd = (torch.randn(B, H, N, D, device="cuda", dtype=torch.float16) for _ in range(3))     # 0.25 GiB each, on the device
# a budget that cannot hold the monolithic call (1.2 GiB) or a decomposition next to the resident 0.75 GiB of Q/K/V:
# they are evicted, the call streams from the pinned copies, the output lands on the device, and Q/K/V come back
# because the budget has room for them next to the output. With "0.76GiB" only two of the three would come back;
# the verbose line says which, and t.device tells the caller where each tensor is.
outd, info = stream_cqsa.attention(qd, kd, vd, is_causal=True, hardware={"cuda:0": "1.05GiB", "host": "64GiB"}, verbose=True, return_info=True)
assert info["engine"] != "monolithic" and info["inputs_evicted"], info
print("output on", outd.device, "| evicted:", info["inputs_evicted"], f"({info.get('evict_released_gib', 0)} GiB released)",
      "| restored:", info["inputs_restored"], "| Q/K/V now on", [str(t.device) for t in (qd, kd, vd)])
print("max |out - SDPA| =", (outd.float() - F.scaled_dot_product_attention(qd.cuda(), kd.cuda(), vd.cuda(), is_causal=True).float()).abs().max().item())
del qd, kd, vd, outd; torch.cuda.empty_cache()
""")

md("""
## 7. Training

Gradients flow when the inputs require them; the backward is decomposed on its own depth (`bwd_itr=`, planned
by default). Where the fp32 accumulators live is the planner's call by the same 15% rule: for the backward they are
the three gradients dQ/dK/dV, three times the size of the forward's accumulator, so from about 1M tokens up on
an A100 they are accumulated in host memory (10-15% slower, 40-70% less device memory). `accumulate_on_gpu=True`
pins the device accumulator, `tol=0` takes the fastest configuration whatever it holds on the device.
""")
code("""
leaves = [torch.randn(B, H, N, D, device="cuda", dtype=torch.float16, requires_grad=True) for _ in range(3)]
out, info = stream_cqsa.attention(*leaves, is_causal=True, hardware={"cuda:0": "1.5GiB", "host": "64GiB"}, verbose=True, return_info=True)
assert info["engine"] != "monolithic", info
dout = torch.randn_like(out)
out.backward(dout)
grads = [t.grad.clone() for t in leaves]
for t in leaves: t.grad = None
ref = F.scaled_dot_product_attention(*leaves, is_causal=True); ref.backward(dout)
rel = lambda a, b: ((a.float() - b.float()).norm() / b.float().norm()).item()
print("forward rel", f"{rel(out, ref):.1e}", "| dQ, dK, dV rel:", [f"{rel(g, t.grad):.1e}" for g, t in zip(grads, leaves)],
      "| gradients on", grads[0].device, "| accumulator on gpu:", info["accumulate_on_gpu"])
del leaves, out, dout, grads, ref; torch.cuda.empty_cache()
""")

md("""
## 8. Grouped-query attention

K and V may have fewer heads than Q (any divisor; query head *h* attends K/V head *h // (H / H_kv)*, as in
FlashAttention and SDPA's `enable_gqa=True`, which is accepted and not required). The CUDA kernels read K/V
through the head ratio and the backward sums dK/dV over each group, so dK and dV come back with K/V's head
count; the memory model counts K, V, dK, dV with their own heads. The Triton kernels reject it with a message.
The shape below is Llama-3's (32 query heads, 8 K/V heads, head dim 128).
""")
code("""
Hq, Hkv, Dg, Ng = 32, 8, 128, 131_072
qg = torch.randn(1, Hq, Ng, Dg, device="cuda", dtype=torch.float16, requires_grad=True)
kg, vg = (torch.randn(1, Hkv, Ng, Dg, device="cuda", dtype=torch.float16, requires_grad=True) for _ in range(2))
outg, info = stream_cqsa.attention(qg, kg, vg, is_causal=True, itr=1, verbose=True, return_info=True)   # itr=1 forces the decomposition
assert info["engine"] != "monolithic", info
dout = torch.randn_like(outg); outg.backward(dout)
gq, gk, gv = (t.grad.clone() for t in (qg, kg, vg))
for t in (qg, kg, vg): t.grad = None
refg = F.scaled_dot_product_attention(qg, kg, vg, is_causal=True, enable_gqa=True); refg.backward(dout)
print("out rel", f"{rel(outg, refg):.1e}", "| dQ rel", f"{rel(gq, qg.grad):.1e}", "| dK rel", f"{rel(gk, kg.grad):.1e}", "| dV rel", f"{rel(gv, vg.grad):.1e}")
print("dK shape", tuple(gk.shape), "(K/V's 8 heads, summed over each group of 4 query heads)")
del qg, kg, vg, outg, dout, gq, gk, gv, refg; torch.cuda.empty_cache()
""")

md("""
## 9. Pinning a setting

Everything the planner decides can be pinned by keyword: `itr=` (depth), `c=` (quorum size: 3, 7, 13, 21, 31, 57, 73,
91, 133; the matching quorum set follows), `kernel=` (`"cuda"` / `"triton"` for the classic engine, `"wave-cuda"` /
`"wave-triton"` for the wave engine), `max_wave_tokens=`, `gather=`, `max_parallel=`, `accumulate_on_gpu=`,
`bwd_itr=`. A pinned depth or quorum size forces the decomposition even when the call would fit, which is how
the measurements in the README isolate the cost of decomposing: at 1M tokens on an A100 the decomposed forward
(7 subproblems, device-resident) runs in 0.98-1.02x the time of one FlashAttention-2 call, forward+backward in
1.01x, with 0.65x its device memory; deeper depths and larger quorum sizes buy memory for time. (At this
size the planner's own wave budget is one subproblem per launch: its model puts the seven-launch and the
one-launch wave within 2% of each other and takes the smaller peak; `max_wave_tokens=` pins the budget.)
""")
code("""
def ms(fn, it=5):
    fn(); fn(); torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record(); [fn() for _ in range(it)]; e1.record(); torch.cuda.synchronize(); return e0.elapsed_time(e1) / it
torch.cuda.empty_cache()
rows = [("SDPA (FlashAttention-2), monolithic", lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True), {})]
for name, kw in [("itr=1, c=7 (7 subproblems), classic CUDA", dict(itr=1, kernel="cuda")),
                 ("itr=1, c=7, wave CUDA, 7 subproblems in one launch", dict(itr=1, kernel="wave-cuda", max_wave_tokens=800_000)),
                 ("   the same, Q/K/V read in place (no gathered copies)", dict(itr=1, kernel="wave-cuda", max_wave_tokens=800_000, gather=False)),
                 ("itr=2, c=7 (49 subproblems)", dict(itr=2)),
                 ("itr=1, c=31 (31 subproblems of 6/31)", dict(itr=1, c=31)),
                 ("itr=1, c=7, Triton kernel", dict(itr=1, kernel="triton"))]:
    rows.append((name, lambda kw=kw: stream_cqsa.attention(q, k, v, is_causal=True, **kw), kw))
print(f"N={N:,}, H={H}, D={D}, fp16, causal, device-resident (5 timed calls each):")
for name, fn, kw in rows:
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    t = ms(fn)
    if kw:
        _, info = stream_cqsa.attention(q, k, v, is_causal=True, return_info=True, **kw); assert info["engine"] != "monolithic", info
    print(f"  {name:56} {t:7.1f} ms   peak {torch.cuda.max_memory_allocated() / 2**30:5.2f} GiB")
""")

md("""
## 10. Routing an existing model

`stream_cqsa.patch_sdpa()` (or `with stream_cqsa.patched_sdpa(): ...`) routes
`torch.nn.functional.scaled_dot_product_attention` through `attention` for the calls it handles (4-D fp16/bf16,
no mask, no dropout, GQA included, at least `min_tokens` tokens); everything else goes to the original. Below the
memory boundary the routed call is still the monolithic kernel, so the patch changes nothing until a call would
have run out of memory.
""")
code("""
import os
class Block(torch.nn.Module):
    def __init__(self, dim, heads):
        super().__init__(); self.h = heads; self.qkv = torch.nn.Linear(dim, 3 * dim, bias=False, dtype=torch.float16)
    def forward(self, x):
        B_, N_, C_ = x.shape
        q_, k_, v_ = self.qkv(x).view(B_, N_, 3, self.h, C_ // self.h).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(q_, k_, v_, is_causal=True)
        return y.transpose(1, 2).reshape(B_, N_, C_)
block = Block(512, 8).cuda()
x = torch.randn(1, N, 512, device="cuda", dtype=torch.float16)
os.environ["CQSA_VERBOSE"] = "1"                              # the routed calls print their configuration line
with stream_cqsa.patched_sdpa(min_tokens=65536):
    y = block(x)                                              # fits here: the routed call is the monolithic kernel
os.environ.pop("CQSA_VERBOSE")
y_ref = block(x)
print("patched vs unpatched max diff:", (y.float() - y_ref.float()).abs().max().item(), "(0: below the boundary the patch is a pass-through)")
del block, x, y, y_ref; torch.cuda.empty_cache()
""")

md("""
## 11. Where the time and memory go, and where to read more

Measured on an A100-SXM4-80GB (fp16, B=1 H=8 D=64, causal; `README.md`, "Measurements", `results/v3/`):

- **Device-resident, 7 subproblems (E1):** forward 0.98-1.02x one FlashAttention-2 call from 1M tokens, forward+backward
  1.01x, at 0.65x its device memory. Below 256K the launches show (1.3-1.6x).
- **Host-resident inputs (E2):** forward 1.17x FA-2 at 1M on 2.6 GiB of device memory (FA-2: 3.6 GiB for the call,
  with Q/K/V resident), forward+backward 1.29x; at 4M and 8M FA-2 cannot run on one card and Stream-CQSA does.
- **The accumulator in host memory (E4, E8):** 10-14% of the forward's time at 1M for 40% less device memory, 3% at 4M.
- **Quorum size (E3, E10):** host-resident, the device peak follows the subproblem size l/c, from 3.9 GiB at c=7 to
  0.6 GiB at c=133 at 1M; time is flat up to c=31 and rises past it.
- **Correctness (E0, E0b):** output 0.96x FA-2's distance from a float64 reference, gradients within 1.24x, at every
  N, c, depth, kernel and residency; every row pinned to a real decomposition.

Under `attention` sits the engine layer, for kernel and systems developers: `stream_cqsa_forward` /
`stream_cqsa_backward` / `stream_cqsa_attn` (the streaming engine), `stream_cqsa.native_wave` (the wave engine),
`autoconfig.plan` / `calibrate`, `devkit.compare_kernels` (your kernel against the monolithic one on the same
subproblems), `adapters.flex_inner` (FlexAttention-style masks) and `distributed.dist_stream_cqsa_forward` /
`_backward` (several GPUs). `notebooks/stream_cqsa_v2_demo.ipynb` part B walks through it.
""")

nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})
ap = argparse.ArgumentParser(); ap.add_argument("--no-exec", action="store_true"); ap.add_argument("--out", default=OUT)
a = ap.parse_args()
if not a.no_exec:
    from nbclient import NotebookClient
    t0 = time.time()
    NotebookClient(nb, timeout=1800, kernel_name="python3", resources={"metadata": {"path": HERE}}).execute()
    print(f"executed in {time.time() - t0:.0f} s")
nbf.write(nb, a.out); print("wrote", a.out)
