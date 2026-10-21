"""
U2 wave A/B: the native wave kernel v2 (cqsa_native_v2: census + ALIGNED run-table kernels on the uniform wave
layout, kernels/native_v2) against the first wave module (cqsa_native: v11's loop, one launch per wave), the
per-task classic kernels (v12f forward / v12b backward, one launch per subproblem) and monolithic
flash_attn / SDPA-flash, ONE process on ONE GPU (A100-SXM4-80GB in slurm/ultimate/wave_ab.slurm).

Configs (causal, D=64, fp16, H=8): N=131072 c=7 (W=7, one launch) and c=73 (W=73, one launch); N=1048576 c=7
with the waves the engine's default budget plans (default_wave_tokens(N) = 1M packed tokens: 2 subproblems of
L~450K per launch). Kernel-level arms run on prepared inputs (the gathered wave, the block map, the per-task
copies), so every timed region is launches only; a wave arm's time is the sum over the planned waves of the
per-wave median (one wave's inputs are prepared at a time: the 1M backward's four waves would hold ~50 GiB):

  fwd  v2_wave (gathered) / v2_wave_inplace (block map into the original [N, H, D]) / old_wave (cqsa_native,
       gathered) / old_wave_inplace / v12f_loop (one fwd_cqs_group_bits per task) / v2_w1 and v12f_w1 (the
       smallest task alone: the W=1 wave against the classic call) / monolithic flash_attn and SDPA-flash at N
  bwd  v2_bwd_wave / old_bwd_wave / v12b_loop / v2_bwd_w1 and v12b_w1 / monolithic flash_attn and SDPA-flash
       backward at N (autograd)
  engine  wave_forward / wave_backward end to end (tables, gathers, merges) on cqsa_native_v2 and cqsa_native

Identity in every part (NaN-pre-filled outputs): v2 wave out/lse == v12f per task (torch.equal) and == the old
wave; v2 dk/dv == v12b per task, dq within the atomics' drift. Targets (docs/ultimate/03_kernels_report.md 9):
W=1 within 1% of v12f / v12b; c=73 one-launch forward <= 1.10x monolithic flash_attn; wave backward >= 2x
faster than the old bwd_wave. Timing: 1 warm-up + 5 timed runs, CUDA events, median (min stored); GPU name in
every row. Output: results/ultimate/wave_ab.json (+ wave_ab.md by --report).

    SNAP=$(bash slurm/ultimate/snapshot.sh) && sbatch --export=ALL,CQSA_SRC=$SNAP,NATIVE_V2_DIR=<frozen> slurm/ultimate/wave_ab.slurm
    python -P benchmarks/ultimate/wave_ab.py --smoke          # correctness only, shared GPU, < 3 GiB
    python -P benchmarks/ultimate/wave_ab.py --report results/ultimate/wave_ab.json
"""
import argparse, importlib, json, os, sys, time, traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("CQSA_CPU_THREADS", "4")

import torch

import fwd_ab as FA
import bwd_ab as BA
from stream_cqsa import native_wave as NW
from stream_cqsa.autoconfig import QUORUM_SETS

DEV = torch.device("cuda")
DT = torch.float16
H = 8
D = 64
GPU = FA.GPU
NAN = float("nan")
SEG_ALIGN = NW.SEG_ALIGN
OUT_DEFAULT = "/scratch/gpfs/AKEY/yb2807/Stream-CQSA-v3/results/ultimate/wave_ab.json"
NATIVE = {}      # "v2": cqsa_native_v2, "old": cqsa_native


def load_native():
    """cqsa_native_v2 (CQSA_NATIVE_MODULE overrides the name: a dev build) and the first wave module cqsa_native."""
    for key, name in (("v2", os.environ.get("CQSA_NATIVE_MODULE", "cqsa_native_v2")), ("old", "cqsa_native")):
        if key not in NATIVE:
            NATIVE[key] = importlib.import_module(name)
    assert NW.wave_api(NATIVE["v2"]) >= 2 and NW.wave_api(NATIVE["old"]) == 1
    return NATIVE


def use_native(key):
    """Point the engine at one wave module (the probe cache is per module)."""
    NW._ext = NATIVE[key]
    NW._SUPPORT.clear()


def timed(fn, iters):
    return FA.timed(fn, iters, warm=1)


def arm(r, name, fn, iters):
    """Time one arm into r[name_ms] / r[name_min_ms]; a failing arm records r[name_error] (NaN times) and the
    later arms of the part still run (the combined job must not lose a part to one arm)."""
    try:
        r[f"{name}_ms"], r[f"{name}_min_ms"] = timed(fn, iters)
    except Exception as e:  # noqa
        r[f"{name}_ms"], r[f"{name}_min_ms"], r[f"{name}_error"] = float("nan"), float("nan"), f"{type(e).__name__}: {e}"
        print(f"!! arm {name} failed: {r[name + '_error']}", flush=True)
        torch.cuda.empty_cache()
    return r[f"{name}_ms"]


def tasks_for(N, c):
    return sorted(FA.c_tasks(N, c, 1), key=lambda t: (int(t.local_size), t.path))


def first_index(tasks, task):
    """Position of a task in the list by identity (the dataclass's == would compare its tensors)."""
    return next(i for i, t in enumerate(tasks) if t is task)


def wave_plan(tasks, N):
    """The waves the engine plans at its default budget (131K: one wave; 1M: ~2 subproblems per wave)."""
    return NW.plan_waves(tasks, NW.default_wave_tokens(N))


# ---------------------------------------------------------------------------- prepared inputs
class Prepared:
    """One wave's inputs for both modules: uniform tables (v2 and the old causal layout agree: S, bits, block map),
    the gathered packed Q/K/V [W*S, H, D] (zero padding), the block map into the original [N, H, D] views."""

    def __init__(self, wave, qt, kt, vt):
        self.tb = NW.build_wave_tables(wave, DEV, uniform=True)
        tb = self.tb
        valid = tb.tok >= 0
        self.pos = torch.nonzero(valid).squeeze(1)
        self.src = tb.tok[valid]
        self.qs, self.ks, self.vs = (self.gather(t) for t in (qt, kt, vt))
        self.bb = NW.block_base_for(tb, lambda g: g).to(DEV)
        self.qt, self.kt, self.vt = qt, kt, vt
        # lens on the device with lens_even stated: no host-to-device copy (a stream sync) inside the timed launch
        self.lens, self.lens_even = tb.lens.to(DEV), bool((tb.lens % 128 == 0).all())

    def gather(self, t):
        g = torch.zeros((self.tb.total, H, D), device=DEV, dtype=t.dtype)
        g[self.pos] = t[self.src]
        return g

    def pack_rows(self, x):       # [H, N] -> [W, H, S]
        p = torch.zeros((H, self.tb.total), device=DEV, dtype=torch.float32)
        p[:, self.pos] = x[:, self.src]
        return p.view(H, self.tb.W, self.tb.S).transpose(0, 1).contiguous()


def fwd_v2(mod, p, in_place, fill=None):
    tb = p.tb
    out = torch.empty((tb.total, H, D), device=DEV, dtype=torch.float32) if fill is None else torch.full((tb.total, H, D), fill, device=DEV, dtype=torch.float32)
    lse = torch.empty((tb.W, H, tb.S), device=DEV, dtype=torch.float32) if fill is None else torch.full((tb.W, H, tb.S), fill, device=DEV, dtype=torch.float32)
    if in_place:
        mod.fwd_wave(p.qt, p.kt, p.vt, p.lens, int(tb.S), tb.bits, tb.blk_or, tb.blk_and, p.bb, SEG_ALIGN, D ** -0.5, True, out, lse,
                     lens_even=p.lens_even)
    else:
        mod.fwd_wave(p.qs, p.ks, p.vs, p.lens, int(tb.S), tb.bits, tb.blk_or, tb.blk_and, None, 0, D ** -0.5, True, out, lse,
                     lens_even=p.lens_even)
    return out, lse


def fwd_old(mod, p, in_place, fill=None):
    """cqsa_native's causal uniform layout: the same S / bits / summaries / block map (the engine's tables)."""
    tb = p.tb
    out = torch.empty((tb.total, H, D), device=DEV, dtype=torch.float32) if fill is None else torch.full((tb.total, H, D), fill, device=DEV, dtype=torch.float32)
    lse = torch.empty((tb.W, H, tb.S), device=DEV, dtype=torch.float32) if fill is None else torch.full((tb.W, H, tb.S), fill, device=DEV, dtype=torch.float32)
    if in_place:
        mod.fwd_wave(p.qt, p.kt, p.vt, tb.cu, int(tb.max_L), int(tb.total), tb.bits, tb.blk_or, tb.blk_and, tb.blk_cu, p.bb, tb.bb_cu,
                     SEG_ALIGN, D ** -0.5, True, out, lse, uniform_S=int(tb.S), uniform_W=int(tb.W))
    else:
        mod.fwd_wave(p.qs, p.ks, p.vs, tb.cu, int(tb.max_L), int(tb.total), tb.bits, tb.blk_or, tb.blk_and, tb.blk_cu, None, None,
                     0, D ** -0.5, True, out, lse, uniform_S=int(tb.S), uniform_W=int(tb.W))
    return out, lse


def task_inputs(tasks, q, k, v):
    """Per-task gathered [1, L, H, D] copies (the classic kernels' inputs) and tables."""
    items = []
    for t in tasks:
        idx = t.token_ids.to(DEV)
        qi, ki, vi = (x[0].index_select(1, idx).transpose(0, 1).unsqueeze(0).contiguous() for x in (q, k, v))
        items.append((t, qi, ki, vi, FA.task_tables(t)))
    return items


def fwd_loop(mod, items, fill=None):
    outs = []
    for t, qi, ki, vi, tabs in items:
        outs.append(FA.cqs_fwd(mod, qi, ki, vi, tabs, True, fill=fill)[:2])
    return outs


def lse_rows(lse):
    """[W, H, S] -> [W*S, H]: the lse of every packed row (padding rows included)."""
    return lse.transpose(1, 2).reshape(-1, lse.shape[1])


def equal_wave_vs_tasks(p, out, lse, refs):
    """v2's packed rows of every subproblem == the per-task (acc [1, L, H, D], lse [1, H, L])."""
    eq, maxd = True, 0.0
    for w in range(p.tb.W):
        L = int(p.tb.lens[w]); sl = slice(w * p.tb.S, w * p.tb.S + L)
        acc, lse_t = refs[w]
        eq &= bool(torch.equal(out[sl], acc[0]) and torch.equal(lse[w, :, :L], lse_t[0]))
        maxd = max(maxd, (out[sl] - acc[0]).abs().max().item())
    return eq, maxd


# ---------------------------------------------------------------------------- forward part
def part_fwd(res, N, c, iters):
    key = f"fwd_N{N}_c{c}"
    tasks = tasks_for(N, c)
    waves = wave_plan(tasks, N)
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, H, N, D, device=DEV, dtype=DT) for _ in range(3))
    qt, kt, vt = (x[0].transpose(0, 1) for x in (q, k, v))
    items = task_inputs(tasks, q, k, v)
    r = dict(gpu=GPU, part="fwd", N=N, c=c, W=len(tasks), n_waves=len(waves), wave_sizes=[len(w) for w in waves],
             S=[], L_min=min(int(t.local_size) for t in tasks), L_max=max(int(t.local_size) for t in tasks),
             sum_L=sum(int(t.local_size) for t in tasks), causal=True, D=D, H=H)
    v2, old, v12f = NATIVE["v2"], NATIVE["old"], FA.MODS["v12f"]
    refs = fwd_loop(v12f, items, fill=NAN)
    wave_arms = [("v2_wave", lambda p: fwd_v2(v2, p, False)), ("v2_wave_inplace", lambda p: fwd_v2(v2, p, True)),
                 ("old_wave", lambda p: fwd_old(old, p, False)), ("old_wave_inplace", lambda p: fwd_old(old, p, True))]
    for name, _ in wave_arms:
        r[f"{name}_ms"], r[f"{name}_min_ms"] = 0.0, 0.0
    eq_v12f, eq_old, maxd, routes, ktot = True, True, 0.0, set(), {}
    for wave in waves:
        p = Prepared(wave, qt, kt, vt)
        r["S"].append(int(p.tb.S))
        # identity (NaN-filled buffers), then the timings
        w0 = first_index(tasks, p.tb.tasks[0])
        out, lse = fwd_v2(v2, p, False, fill=NAN)
        e, d = equal_wave_vs_tasks(p, out, lse, refs[w0:w0 + p.tb.W]); eq_v12f &= e; maxd = max(maxd, d)
        out_i, lse_i = fwd_v2(v2, p, True, fill=NAN)
        eq_v12f &= bool(torch.equal(out_i[p.pos], out[p.pos]) and torch.equal(lse_rows(lse_i)[p.pos], lse_rows(lse)[p.pos]))
        out_o, lse_o = fwd_old(old, p, False, fill=NAN)
        eq_old &= bool(torch.equal(out_o[p.pos], out[p.pos]) and torch.equal(lse_rows(lse_o)[p.pos], lse_rows(lse)[p.pos]))
        del out, lse, out_i, lse_i, out_o, lse_o
        routes |= {int(x) for x in v2.fwd_cqs_census(p.tb.bits, p.tb.blk_or, p.tb.blk_and, True, D, p.tb.lens, int(p.tb.S))[0].tolist()}
        for name, fn in wave_arms:
            acc = dict(ms=0.0, min_ms=0.0)
            arm(acc, "", lambda: fn(p), iters)
            r[f"{name}_ms"] += acc["_ms"]; r[f"{name}_min_ms"] += acc["_min_ms"]
            if "_error" in acc:
                r[f"{name}_error"] = acc["_error"]
        for kk, vv in FA.kernel_times(lambda: fwd_v2(v2, p, False)).items():
            if isinstance(vv, float):
                ktot[kk] = ktot.get(kk, 0.0) + vv
        del p
        torch.cuda.empty_cache()
    r["v2_equal_v12f_all_tasks"], r["v2_maxdiff_vs_v12f"], r["v2_equal_old_wave"] = eq_v12f, maxd, eq_old
    r["census_routes"], r["v2_wave_kernel_us"] = sorted(routes), ktot
    del refs
    loops = [("v12f_loop", lambda: fwd_loop(v12f, items))]
    if N <= 131072:      # v11 is the superseded kernel (v12f is measured against it in fwd_ab); its 1M loop costs minutes
        loops.append(("v11_loop", lambda: fwd_loop(FA.MODS["v11"], items)))
    for name, fn in loops:
        arm(r, name, fn, iters)
    for name, _ in wave_arms + loops:
        print(f"[{key}] {name:18s} {r[name + '_ms']:9.2f} ms", flush=True)
    # W=1: the smallest task alone, as a wave and as the classic call
    t0 = tasks[0]
    p1 = Prepared([t0], qt, kt, vt)
    i1 = [it for it in items if it[0] is t0]
    r["w1_L"] = int(t0.local_size)
    arm(r, "v2_w1", lambda: fwd_v2(v2, p1, False), iters)
    arm(r, "v2_w1_inplace", lambda: fwd_v2(v2, p1, True), iters)
    arm(r, "old_w1", lambda: fwd_old(old, p1, False), iters)
    arm(r, "v12f_w1", lambda: fwd_loop(v12f, i1), iters)
    r["v2_w1_over_v12f_w1"] = r["v2_w1_ms"] / r["v12f_w1_ms"]
    del p1, items
    torch.cuda.empty_cache()
    fa = FA.fa2_arms(N, D, True, iters)
    r.update({f"monolithic_{kk}": vv for kk, vv in fa.items()})
    for name in ("v2_wave", "v2_wave_inplace", "old_wave", "old_wave_inplace", "v12f_loop", "v11_loop"):
        r[f"{name}_over_flash_attn"] = r.get(f"{name}_ms", float("nan")) / fa["flash_attn_ms"]
        r[f"{name}_over_sdpa"] = r.get(f"{name}_ms", float("nan")) / fa["sdpa_flash_ms"]
    r["v2_wave_over_v12f_loop"] = r["v2_wave_ms"] / r["v12f_loop_ms"]
    r["v2_wave_over_old_wave"] = r["v2_wave_ms"] / r["old_wave_ms"]
    res["rows"].append(r)
    print(f"[{key}] W={len(tasks)} waves={r['wave_sizes']} L {r['L_min']}-{r['L_max']}: v2 {r['v2_wave_ms']:.1f} ms = "
          f"{r['v2_wave_over_flash_attn']:.3f}x flash_attn ({fa['flash_attn_ms']:.1f}), {r['v2_wave_over_v12f_loop']:.3f}x v12f loop, "
          f"{r['v2_wave_over_old_wave']:.3f}x old wave; W=1 v2/v12f {r['v2_w1_over_v12f_w1']:.3f}; "
          f"equal v12f={eq_v12f} old={eq_old} routes={r['census_routes']}", flush=True)
    del q, k, v, qt, kt, vt
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------- backward part
def global_lse_dpsum(q, k, v, dout):
    """The global lse [1, H, N] and rowsum(dO * O) [1, H, N] of the causal attention, from flash_attn."""
    from flash_attn import flash_attn_func
    o, lse, _ = flash_attn_func(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), causal=True, softmax_scale=D ** -0.5,
                                return_attn_probs=True)
    dpsum = (o.float() * dout.transpose(1, 2).float()).sum(-1).transpose(1, 2).contiguous()     # [1, H, N]
    return lse.contiguous(), dpsum


class PreparedBwd(Prepared):
    def __init__(self, wave, qt, kt, vt, dot, lse, dpsum):
        super().__init__(wave, qt, kt, vt)
        self.dos = self.gather(dot)
        self.lse_p, self.dps_p = self.pack_rows(lse[0]), self.pack_rows(dpsum[0])
        # the old module's varlen layout: packed rows without padding, dpsum with 128 rows of slack per subproblem
        self.tbv = NW.build_wave_tables(list(wave), DEV, uniform=False)
        tbv = self.tbv
        self.qv, self.kv, self.vv, self.dov = (torch.index_select(t, 0, tbv.tok) for t in (qt, kt, vt, dot))
        self.lse_v = torch.index_select(lse[0], 1, tbv.tok).contiguous()
        W = tbv.W
        self.cu = cu = [0] + [int(x) for x in torch.cumsum(torch.tensor([int(t.local_size) for t in wave]), 0)]
        pos = torch.cat([torch.arange(cu[w] + 128 * w, cu[w] + 128 * w + int(t.local_size)) for w, t in enumerate(wave)]).to(DEV)
        self.dps_v = torch.zeros((H, int(tbv.total) + 128 * W), device=DEV, dtype=torch.float32)
        self.dps_v.index_copy_(1, pos, torch.index_select(dpsum[0], 1, tbv.tok))


def bwd_v2(mod, p, fill=None):
    tb = p.tb
    grads = [torch.empty((tb.total, H, D), device=DEV, dtype=DT) if fill is None else torch.full((tb.total, H, D), fill, device=DEV, dtype=DT)
             for _ in range(3)]
    mod.bwd_wave(p.dos, p.qs, p.ks, p.vs, p.lse_p, p.dps_p, p.lens, int(tb.S), tb.bits, tb.blk_or, tb.blk_and, D ** -0.5, True, *grads,
                 lens_even=p.lens_even)
    return grads


def bwd_old(mod, p, fill=None):
    tbv = p.tbv
    grads = [torch.empty((tbv.total, H, D), device=DEV, dtype=DT) if fill is None else torch.full((tbv.total, H, D), fill, device=DEV, dtype=DT)
             for _ in range(3)]
    mod.bwd_wave(p.dov, p.qv, p.kv, p.vv, p.lse_v, p.dps_v, tbv.cu, int(tbv.max_L), int(tbv.total), tbv.bits, tbv.blk_or, tbv.blk_and,
                 tbv.blk_cu, D ** -0.5, True, *grads)
    return grads


def bwd_task_inputs(tasks, q, k, v, dout, lse, dpsum):
    items = []
    for t in tasks:
        idx = t.token_ids.to(DEV)
        qi, ki, vi, doi = (x[0].index_select(1, idx).transpose(0, 1).unsqueeze(0).contiguous() for x in (q, k, v, dout))
        items.append((t, qi, ki, vi, doi, lse[0].index_select(1, idx).unsqueeze(0).contiguous(),
                      dpsum[0].index_select(1, idx).unsqueeze(0).contiguous(), BA.task_tables(t)))
    return items


def bwd_loop(mod, items, fill=None):
    outs = []
    for t, qi, ki, vi, doi, lse_i, dps_i, tabs in items:
        outs.append(BA.cqs_bwd(mod, doi, qi, ki, vi, doi, lse_i, tabs, True, dsum=dps_i, fill=fill))
    return outs


def part_bwd(res, N, c, iters):
    key = f"bwd_N{N}_c{c}"
    tasks = tasks_for(N, c)
    waves = wave_plan(tasks, N)
    torch.manual_seed(0)
    q, k, v, dout = (torch.randn(1, H, N, D, device=DEV, dtype=DT) for _ in range(4))
    lse, dpsum = global_lse_dpsum(q, k, v, dout)
    qt, kt, vt, dot = (x[0].transpose(0, 1) for x in (q, k, v, dout))
    items = bwd_task_inputs(tasks, q, k, v, dout, lse, dpsum)
    r = dict(gpu=GPU, part="bwd", N=N, c=c, W=len(tasks), n_waves=len(waves), wave_sizes=[len(w) for w in waves],
             S=[], L_min=min(int(t.local_size) for t in tasks), L_max=max(int(t.local_size) for t in tasks),
             causal=True, D=D, H=H)
    v2, old, v12b = NATIVE["v2"], NATIVE["old"], BA.MODS["v12b"]
    refs = bwd_loop(v12b, items, fill=NAN)
    wave_arms = [("v2_bwd_wave", lambda p: bwd_v2(v2, p)), ("old_bwd_wave", lambda p: bwd_old(old, p))]
    for name, _ in wave_arms:
        r[f"{name}_ms"], r[f"{name}_min_ms"] = 0.0, 0.0
    eq, dq_max, eq_old, old_max, routes = True, 0.0, True, 0.0, set()
    for wave in waves:
        p = PreparedBwd(wave, qt, kt, vt, dot, lse, dpsum)
        r["S"].append(int(p.tb.S))
        w0 = first_index(tasks, p.tb.tasks[0])
        dq, dk, dv = bwd_v2(v2, p, fill=NAN)
        for w in range(p.tb.W):
            L = int(p.tb.lens[w]); sl = slice(w * p.tb.S, w * p.tb.S + L)
            dq_r, dk_r, dv_r = refs[w0 + w]
            eq &= bool(torch.equal(dk[sl], dk_r[0]) and torch.equal(dv[sl], dv_r[0]))
            dq_max = max(dq_max, BA.maxdiff(dq[sl], dq_r[0]))
        dqo, dko, dvo = bwd_old(old, p, fill=NAN)
        for w in range(p.tb.W):
            L = int(p.tb.lens[w]); sl = slice(w * p.tb.S, w * p.tb.S + L); c0 = p.cu[w]
            eq_old &= bool(torch.equal(dk[sl], dko[c0:c0 + L]) and torch.equal(dv[sl], dvo[c0:c0 + L]))
            old_max = max(old_max, BA.maxdiff(dq[sl], dqo[c0:c0 + L]))
        del dq, dk, dv, dqo, dko, dvo
        routes |= {int(x) for x in v2.bwd_cqs_census(p.tb.bits, p.tb.blk_or, p.tb.blk_and, True, D, p.tb.lens, int(p.tb.S))[0].tolist()}
        for name, fn in wave_arms:
            acc = dict(ms=0.0, min_ms=0.0)
            arm(acc, "", lambda: fn(p), iters)
            r[f"{name}_ms"] += acc["_ms"]; r[f"{name}_min_ms"] += acc["_min_ms"]
            if "_error" in acc:
                r[f"{name}_error"] = acc["_error"]
        del p
        torch.cuda.empty_cache()
    r["v2_dkdv_equal_v12b_all_tasks"], r["v2_maxdiff_dq_vs_v12b"] = eq, dq_max
    r["v2_dkdv_equal_old_wave"], r["v2_maxdiff_dq_vs_old_wave"] = eq_old, old_max
    r["census_routes"] = sorted(routes)
    del refs
    loops = [("v12b_loop", lambda: bwd_loop(v12b, items))]
    if N <= 131072:
        loops.append(("v11_loop", lambda: bwd_loop(BA.MODS["v11"], items)))
    for name, fn in loops:
        arm(r, name, fn, iters)
    for name, _ in wave_arms + loops:
        print(f"[{key}] {name:18s} {r[name + '_ms']:9.2f} ms", flush=True)
    t0 = tasks[0]
    p1 = PreparedBwd([t0], qt, kt, vt, dot, lse, dpsum)
    i1 = [it for it in items if it[0] is t0]
    r["w1_L"] = int(t0.local_size)
    arm(r, "v2_bwd_w1", lambda: bwd_v2(v2, p1), iters)
    arm(r, "old_bwd_w1", lambda: bwd_old(old, p1), iters)
    arm(r, "v12b_w1", lambda: bwd_loop(v12b, i1), iters)
    r["v2_bwd_w1_over_v12b_w1"] = r["v2_bwd_w1_ms"] / r["v12b_w1_ms"]
    del p1, items
    torch.cuda.empty_cache()
    fa = BA.fa2_arms(N, D, True, iters)
    r.update({f"monolithic_{kk}": vv for kk, vv in fa.items()})
    for name in ("v2_bwd_wave", "old_bwd_wave", "v12b_loop", "v11_loop"):
        r[f"{name}_over_flash_attn"] = r.get(f"{name}_ms", float("nan")) / fa["flash_attn_bwd_ms"]
        r[f"{name}_over_sdpa"] = r.get(f"{name}_ms", float("nan")) / fa["sdpa_flash_bwd_ms"]
    r["v2_bwd_wave_over_v12b_loop"] = r["v2_bwd_wave_ms"] / r["v12b_loop_ms"]
    r["old_over_v2_bwd_wave"] = r["old_bwd_wave_ms"] / r["v2_bwd_wave_ms"]
    res["rows"].append(r)
    print(f"[{key}] W={len(tasks)} waves={r['wave_sizes']}: v2 {r['v2_bwd_wave_ms']:.1f} ms = {r['v2_bwd_wave_over_flash_attn']:.3f}x "
          f"flash_attn bwd ({fa['flash_attn_bwd_ms']:.1f}), {r['v2_bwd_wave_over_v12b_loop']:.3f}x v12b loop, old/v2 {r['old_over_v2_bwd_wave']:.2f}x; "
          f"W=1 v2/v12b {r['v2_bwd_w1_over_v12b_w1']:.3f}; dk/dv equal v12b={eq} old={eq_old} dq max|d| {dq_max:.1e}", flush=True)
    del q, k, v, dout, lse, dpsum, qt, kt, vt, dot
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------- engine part
def part_engine(res, N, c, iters, backward=True):
    key = f"engine_N{N}_c{c}"
    iset = QUORUM_SETS[c]
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, H, N, D, device=DEV, dtype=DT) for _ in range(3))
    r = dict(gpu=GPU, part="engine", N=N, c=c, causal=True, D=D, H=H)
    outs = {}
    for key_m in ("v2", "old"):
        use_native(key_m)
        for gather in (True, False):
            name = f"{key_m}_fwd_{'gathered' if gather else 'inplace'}"
            fn = lambda: NW.wave_forward(q, k, v, causal=True, itr=1, c=c, interest_set=iset, kernel="cuda", gather=gather)
            out, info = fn()
            outs[name] = out
            arm(r, name, fn, iters)
            r[f"{name}_waves"] = info["wave_sizes"]
            print(f"[{key}] {name:22s} {r[name + '_ms']:9.2f} ms  waves {info['wave_sizes']} gather={info['gather']}", flush=True)
    r["engine_v2_equal_old_fwd"] = bool(torch.equal(outs["v2_fwd_gathered"], outs["old_fwd_gathered"]))
    r["engine_v2_inplace_equal_gathered"] = bool(torch.equal(outs["v2_fwd_gathered"], outs["v2_fwd_inplace"]))
    r["engine_v2_vs_old_maxdiff"] = (outs["v2_fwd_gathered"] - outs["old_fwd_gathered"]).abs().max().item()
    if backward:
        use_native("v2")
        out, info = NW.wave_forward(q, k, v, causal=True, itr=1, c=c, interest_set=iset, kernel="cuda")
        dout = torch.randn_like(q)
        grads = {}
        for key_m in ("v2", "old"):
            use_native(key_m)
            fn = lambda: NW.wave_backward(q, k, v, out, dout, info["lse"], causal=True, itr=1, c=c, interest_set=iset)
            grads[key_m] = fn()[0]
            arm(r, f"{key_m}_bwd", fn, iters)
            print(f"[{key}] {key_m + '_bwd':22s} {r[key_m + '_bwd_ms']:9.2f} ms", flush=True)
        r["engine_bwd_v2_vs_old_rel"] = max(((a.float() - b.float()).norm() / b.float().norm()).item() for a, b in zip(grads["v2"], grads["old"]))
        r["engine_old_over_v2_bwd"] = r["old_bwd_ms"] / r["v2_bwd_ms"]
        del out, dout, grads
    del outs
    res["rows"].append(r)
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------- report
def _rows(res, part):
    return [r for r in res["rows"] if r["part"] == part]


def report(res):
    info = res["info"]
    L = [f"# U2 wave A/B: cqsa_native_v2 vs cqsa_native vs per-task v12f / v12b vs monolithic FlashAttention",
         "",
         f"GPU {info['gpu']}; node {info.get('node')}; job {info.get('slurm_job')}; torch {info['torch']}; flash_attn {info.get('flash_attn')}; "
         f"nvidia-smi `{info.get('nvidia_smi')}`. {res['timing']}. fp16, H={H}, D={D}, causal; every arm on prepared inputs "
         f"(launches only) except the engine rows. Modules: {json.dumps(res.get('modules'))}.", ""]
    fr = _rows(res, "fwd")
    if fr:
        L += ["## Forward (ms; one launch per wave for the wave arms, one launch per task for the loops)", "",
              "| N | c | W | waves | L | v2 wave | v2 in place | old wave | old in place | v12f loop | v11 loop | flash_attn mono | SDPA mono | v2 / flash_attn | v2 / v12f loop | v2 / old | W=1 v2 / v12f | equal v12f | equal old | routes |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in fr:
            L.append(f"| {r['N']} | {r['c']} | {r['W']} | {r['wave_sizes']} | {r['L_min']}-{r['L_max']} | {r['v2_wave_ms']:.2f} | {r['v2_wave_inplace_ms']:.2f} | "
                     f"{r['old_wave_ms']:.2f} | {r['old_wave_inplace_ms']:.2f} | {r['v12f_loop_ms']:.2f} | {r.get('v11_loop_ms', float('nan')):.2f} | {r['monolithic_flash_attn_ms']:.2f} | "
                     f"{r['monolithic_sdpa_flash_ms']:.2f} | **{r['v2_wave_over_flash_attn']:.3f}** | {r['v2_wave_over_v12f_loop']:.3f} | {r['v2_wave_over_old_wave']:.3f} | "
                     f"**{r['v2_w1_over_v12f_w1']:.3f}** ({r['v2_w1_ms']:.2f} / {r['v12f_w1_ms']:.2f} ms at L={r['w1_L']}) | {r['v2_equal_v12f_all_tasks']} | {r['v2_equal_old_wave']} | {r['census_routes']} |")
        L += ["", "Device us per kernel kind of one v2 wave call (census / aligned / legacy no-op / memset): "
              + "; ".join(f"N={r['N']} c={r['c']}: {json.dumps({k: round(v, 1) for k, v in r['v2_wave_kernel_us'].items()} if isinstance(r['v2_wave_kernel_us'], dict) else r['v2_wave_kernel_us'])}" for r in fr), ""]
    br = _rows(res, "bwd")
    if br:
        L += ["## Backward (ms; global lse and rowsum(dO*O) supplied; one launch per wave / per task)", "",
              "| N | c | W | waves | v2 bwd wave | old bwd wave | v12b loop | v11 loop | flash_attn mono bwd | SDPA mono bwd | v2 / flash_attn | v2 / v12b loop | old / v2 | W=1 v2 / v12b | dk/dv equal v12b | dq max d | dk/dv equal old | routes |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in br:
            L.append(f"| {r['N']} | {r['c']} | {r['W']} | {r['wave_sizes']} | {r['v2_bwd_wave_ms']:.2f} | {r['old_bwd_wave_ms']:.2f} | {r['v12b_loop_ms']:.2f} | "
                     f"{r.get('v11_loop_ms', float('nan')):.2f} | {r['monolithic_flash_attn_bwd_ms']:.2f} | {r['monolithic_sdpa_flash_bwd_ms']:.2f} | **{r['v2_bwd_wave_over_flash_attn']:.3f}** | "
                     f"{r['v2_bwd_wave_over_v12b_loop']:.3f} | **{r['old_over_v2_bwd_wave']:.2f}x** | **{r['v2_bwd_w1_over_v12b_w1']:.3f}** ({r['v2_bwd_w1_ms']:.2f} / {r['v12b_w1_ms']:.2f} ms at L={r['w1_L']}) | "
                     f"{r['v2_dkdv_equal_v12b_all_tasks']} | {r['v2_maxdiff_dq_vs_v12b']:.1e} | {r['v2_dkdv_equal_old_wave']} | {r['census_routes']} |")
        L.append("")
    er = _rows(res, "engine")
    if er:
        L += ["## Engine end to end (wave_forward / wave_backward: tables, gathers or block maps, merges, scatter-adds)", "",
              "| N | c | v2 fwd gathered | v2 fwd in place | old fwd gathered | old fwd in place | v2 bwd | old bwd | old / v2 bwd | v2 == old fwd | in place == gathered |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in er:
            L.append(f"| {r['N']} | {r['c']} | {r['v2_fwd_gathered_ms']:.1f} | {r['v2_fwd_inplace_ms']:.1f} | {r['old_fwd_gathered_ms']:.1f} | {r['old_fwd_inplace_ms']:.1f} | "
                     f"{r.get('v2_bwd_ms', float('nan')):.1f} | {r.get('old_bwd_ms', float('nan')):.1f} | {r.get('engine_old_over_v2_bwd', float('nan')):.2f} | "
                     f"{r['engine_v2_equal_old_fwd']} | {r['engine_v2_inplace_equal_gathered']} |")
        L.append("")
    L += ["## Targets (docs/ultimate/03_kernels_report.md section 9, package U2)", ""]
    for r in fr:
        L.append(f"- forward N={r['N']} c={r['c']}: W=1 v2 / v12f = {r['v2_w1_over_v12f_w1']:.3f} (target within 1%: "
                 f"{'met' if abs(r['v2_w1_over_v12f_w1'] - 1) <= 0.01 else 'NOT met'}); one-launch wave / monolithic flash_attn = "
                 f"{r['v2_wave_over_flash_attn']:.3f}" + (f" (target <= 1.10 at c=73: {'met' if r['v2_wave_over_flash_attn'] <= 1.10 else 'NOT met'})" if r['c'] == 73 else ""))
    for r in br:
        L.append(f"- backward N={r['N']} c={r['c']}: W=1 v2 / v12b = {r['v2_bwd_w1_over_v12b_w1']:.3f} (target within 1%: "
                 f"{'met' if abs(r['v2_bwd_w1_over_v12b_w1'] - 1) <= 0.01 else 'NOT met'}); old bwd_wave / v2 = {r['old_over_v2_bwd_wave']:.2f}x "
                 f"(target >= 2x: {'met' if r['old_over_v2_bwd_wave'] >= 2 else 'NOT met'}); v2 / monolithic flash_attn bwd = {r['v2_bwd_wave_over_flash_attn']:.3f}")
    arm_errors = [(r["part"], r["N"], r["c"], k, v) for r in res["rows"] for k, v in r.items() if k.endswith("_error")]
    if arm_errors:
        L += ["", "## Failed arms", ""] + [f"- {p} N={N} c={c} {k[:-6]}: {v}" for p, N, c, k, v in arm_errors]
    if res.get("errors"):
        L += ["", "## Errors", ""] + [f"```\n{e}\n```" for e in res["errors"]]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="small N, 2 iterations: correctness on a shared GPU")
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--out", default=os.environ.get("WAVE_AB_OUT", OUT_DEFAULT))
    ap.add_argument("--report", metavar="JSON", help="only write the markdown report of an existing JSON next to it")
    a = ap.parse_args()
    if a.report:
        res = json.load(open(a.report))
        md = a.report.replace(".json", ".md")
        open(md, "w").write(report(res))
        print(f"wrote {md}")
        return
    FA.load_modules(("v11", "v12f"))
    BA.load_modules(("v11", "v12b"))
    load_native()
    iters = a.iters
    configs = [(131072, 7), (131072, 73), (1 << 20, 7)]
    if a.smoke:
        configs, iters = [(16384, 7), (32768, 73)], 2
        a.out = a.out.replace(".json", "_smoke.json")
    t0 = time.time()
    res = dict(info=FA.gpu_info(), rows=[], H=H, D=D, dtype="float16", iters=iters, smoke=a.smoke,
               timing=f"1 warm-up + {iters} timed runs, CUDA events, median (min also stored)",
               modules={k: getattr(m, "__file__", k) for k, m in list(FA.MODS.items()) + list(BA.MODS.items()) + list(NATIVE.items())})
    print(json.dumps(res["info"]), flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    parts = [(part_fwd, (N, c, iters)) for N, c in configs] + [(part_bwd, (N, c, iters)) for N, c in configs] \
        + [(part_engine, (N, c, iters)) for N, c in configs]
    for part, args in parts:
        try:
            part(res, *args)
        except Exception as e:  # noqa
            res.setdefault("errors", []).append(traceback.format_exc())
            print(f"!! {part.__name__}{args} failed: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
        res["kernel_names"] = FA._KERNEL_NAMES
        json.dump(res, open(a.out, "w"), indent=1)
    res["elapsed_s"] = time.time() - t0
    json.dump(res, open(a.out, "w"), indent=1)
    md = report(res)
    print(md)
    if not a.smoke:
        open(a.out.replace(".json", ".md"), "w").write(md)
    print(f"\nwrote {a.out}  ({res['elapsed_s']:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
