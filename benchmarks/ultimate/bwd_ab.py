"""
P2 backward A/B: v12b (census + ALIGNED run-table backward, LEGACY = v11's CQS backward) vs v11's CQS backward,
v11x's NO_CQS backward (upstream's loop with our contract), flash_attn (the FA-2 package) and SDPA-flash, ONE
process on ONE GPU.

Kernel trees (next/kernel/<variant>, module cqsa_cuda_next_<variant>): v11 = the production CQS backward (every tile
visited, dead ones guarded off, per-element mask path present); v11x = v11 built with -DCQSA_BWD_NO_CQS (CQS compiled
out of the backward: upstream's loop, mask ignored, a timing baseline only); v12b = the census + ALIGNED backward
(this package): a census kernel classifies every tile the backward visits and writes per-column runs of live m
blocks; on a 128-aligned task every tile is fully live or fully dead and the ALIGNED instantiation (upstream's loop
over the runs) computes the call, otherwise LEGACY (v11's code) does, decided on the device.

Arms per config (causal/non-causal, D, L, pattern):
  flash_attn    flash_attn_func forward once, torch.autograd.grad timed (the package's backward: dot_do_o, the
                dq/dk/dv kernel, convert_dQ)
  sdpa_flash    torch SDPA, FLASH_ATTENTION backend forced, same timing (the FA-2 arm of every earlier job)
  v11x_nocqs    v11x `bwd`, CQS off (the upstream loop with our contract; design basis of the 0.92-0.95x route)
  v12b_plain    v12b `bwd`, CQS off (the ALIGNED loop without a run table)
  v11/<pat>     v11 `bwd` with the task's group bits + summaries (the engine's call)
  v12b/<pat>    v12b `bwd`, same arguments: census + ALIGNED (+ the LEGACY launch whose CTAs return at once)
  v12b_legacy   v12b `bwd` with cqs_bwd_flags = FORCE_LEGACY (v11's kernel in v12b's build, no census)
Patterns: real (the task's group bits), zero (all-zero bits with CQS on: every tile live), plain (CQS off).
Every CQS arm computes dsoftmax_sum in its own preprocess pass (as flash_attn does), with the census inside the
timed region. Checks per row: dk/dv torch.equal to v11's, dq max |d| vs v11 and across reruns, the census route.

Configs: causal D=64 on the aligned c=7 task at N in {32768, 65536, 131072} (L ~14K/28K/56K; real + zero);
c=73 N=131072 causal D=64 real, summed over the 73 tasks (one timed region for all of them); causal D=128 and
non-causal D=64 at L~56K; deterministic=True on the L~56K causal real task (item 3: 3 reruns torch.equal).
Timing: 1 warm-up + 5 timed runs, CUDA events, median (min stored); GPU name in every row.

    SNAP=$(bash slurm/ultimate/snapshot.sh) && sbatch --export=ALL,CQSA_SRC=$SNAP,V12B_DIR=<frozen> slurm/ultimate/bwd_ab.slurm
    python -P benchmarks/ultimate/bwd_ab.py --smoke          # correctness only, shared GPU, < 2 GiB
    python -P benchmarks/ultimate/bwd_ab.py --report results/ultimate/bwd_ab.json   # writes bwd_ab.md
"""
import argparse, importlib, json, os, socket, statistics, subprocess, sys, time, traceback

import numpy as np

os.environ.setdefault("CQSA_CUDA_MODULE", "cqsa_cuda_next_v11")
os.environ.setdefault("CQSA_CUDA_MODULE_NONCAUSAL", "cqsa_cuda_next_v9")
os.environ["CQSA_BACKWARD"] = "cuda"          # never Triton
os.environ.setdefault("CQSA_CPU_THREADS", "4")

import torch
import torch.nn.functional as F

torch.set_num_threads(int(os.environ["CQSA_CPU_THREADS"]))

from stream_cqsa.autoconfig import QUORUM_SETS
from stream_cqsa.interface import cqs_block_summaries, CQS_BLK_SIZE
from stream_cqsa.stable_stream import build_tasks

MODULES = {"v11": "cqsa_cuda_next_v11", "v11x": "cqsa_cuda_next_v11x", "v12b": "cqsa_cuda_next_v12b"}
MODS = {}
FORCE_LEGACY = 1                      # cqs_bwd_flags of v12b's bwd (kCqsBwdForceLegacy)
RUN_SLOTS = 16                        # kCqsBwdRunSlots (cqs_census_bwd.h)
H = 8
DT = torch.float16
DEV = torch.device("cuda")
GPU = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no GPU"
SEG_ALIGN = 128
OUT_DEFAULT = "/scratch/gpfs/AKEY/yb2807/Stream-CQSA-ultimate/results/ultimate/bwd_ab.json"


def load_modules(names=tuple(MODULES)):
    for name in names:
        if name not in MODS:
            MODS[name] = importlib.import_module(MODULES[name])
    return MODS


# ---------------------------------------------------------------------------- helpers
def gpu_info():
    info = dict(gpu=GPU, torch=torch.__version__, node=socket.gethostname(), slurm_job=os.environ.get("SLURM_JOB_ID"),
                modules={k: getattr(m, "__file__", k) for k, m in MODS.items()})
    try:
        import flash_attn
        info["flash_attn"] = f"{flash_attn.__version__} {flash_attn.__file__}"
    except Exception as e:  # noqa
        info["flash_attn"] = f"unavailable: {e!r}"
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name,clocks.sm,clocks.max.sm,memory.total",
                            "--format=csv,noheader"], capture_output=True, text=True, timeout=30)
        info["nvidia_smi"] = r.stdout.strip()
    except Exception as e:  # pragma: no cover
        info["nvidia_smi"] = f"unavailable: {e!r}"
    return info


def timed(fn, iters, warm=1):
    """`warm` warm-up + `iters` timed runs with CUDA events; returns (median_ms, min_ms)."""
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record(); fn(); e1.record(); torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1))
    return statistics.median(ts), min(ts)


def to_dev(t):
    return t.to(DEV).contiguous()


def maxdiff(a, b):
    return (a.float() - b.float()).abs().max().item()


def tables(bits):
    """(bits, blk_or, blk_and) on the device for a per-token bits vector."""
    bo, ba = (to_dev(x) for x in cqs_block_summaries(bits, CQS_BLK_SIZE))
    return to_dev(bits), bo, ba


def task_tables(task):
    return to_dev(task.group_bits), to_dev(task.extra["blk_or"]), to_dev(task.extra["blk_and"])


def c_tasks(N, c, itr=1, seg_align=SEG_ALIGN, D=64):
    return build_tasks(N, itr, B=1, H=H, D=D, itemsize=2, sorted_gather=True, pin=False,
                       c=c, interest_set=QUORUM_SETS[c], seg_align=seg_align)


def aligned_task(N, c=7):
    """The c/itr=1 task at N with seg_align=128 of smallest L (N=131072, c=7: L=56064, the map's row)."""
    return min(c_tasks(N, c), key=lambda t: (int(t.local_size), t.path))


def plain_fwd(mod, q, k, v, causal):
    """CQS off: the extension's `fwd`. Returns (out16, lse)."""
    out, lse, _, _ = mod.fwd(q, k, v, None, None, 0.0, q.shape[-1] ** -0.5, bool(causal), -1, -1, 0.0, False, None)
    return out, lse


def cqs_fwd(mod, q, k, v, tabs, causal):
    """CQS on, as interface.flash_attn_func_cqs_group_bits calls it (fp32 out). Returns (out16, lse). Use v11
    for real patterns: v11x/v12b are built with -DCQSA_NO_ELEMENT_PATH and poison a real pattern's forward."""
    bits, bo, ba = tabs
    out = torch.empty_like(q)
    acc = torch.empty(q.shape, device=q.device, dtype=torch.float32)
    out16, lse, _, _ = mod.fwd_cqs_group_bits(q, k, v, out, None, 0.0, q.shape[-1] ** -0.5, bool(causal), -1, -1, 0.0,
                                              False, None, bits, bo, ba, CQS_BLK_SIZE, acc, None, None)
    return out16, lse


def cqs_bwd(mod, do, q, k, v, o16, lse, tabs, causal, flags=None, deterministic=False, dsum=None):
    """The extension's `bwd` with the engine's global-lse contract: CQS on with tabs = (bits, blk_or, blk_and),
    CQS off with tabs = None; v12b's cqs_bwd_flags when given. Returns (dq, dk, dv)."""
    dq, dk, dv = (torch.empty_like(x) for x in (q, k, v))
    args = [do, q, k, v, o16, lse, dq, dk, dv, None, 0.0, q.shape[-1] ** -0.5, bool(causal), -1, -1, 0.0,
            bool(deterministic), None, None]
    args += [None, None, None, None, dsum] if tabs is None else [tabs[0], tabs[1], tabs[2], CQS_BLK_SIZE, dsum]
    if flags is not None:
        args.append(int(flags))
    mod.bwd(*args)
    return dq, dk, dv


def census(mod, tabs, causal, D):
    """v12b's census on the device: (route, run table [num_n, slots, 2] as a numpy array, (block_m, block_n))."""
    route, runs, tile = mod.bwd_cqs_census(tabs[0], tabs[1], tabs[2], bool(causal), int(D))
    torch.cuda.synchronize()
    return int(route.item()), runs.cpu().numpy(), tuple(int(x) for x in tile)


def census_ref(bits, block_m, block_n, causal, slots=RUN_SLOTS):
    """Reference of v12b's census (cqs_census_bwd.cu) from the per-token bits: a tile is live when no row and
    column share a group bit (row OR & col OR == 0), dead when all do (row AND & col AND != 0), mixed otherwise;
    the tiles visited per column block n are m in [m_min, num_m) with m_min = n * block_n // block_m under causal
    (else 0); live tiles are grouped into descending runs {hi, lo}, at most slots-1 per column block, closed by
    {-1, -1}. Returns (route, [list of runs per column block])."""
    bits = np.asarray(bits, dtype=np.int64).reshape(-1)
    L = bits.size

    def summ(n_blk, blk):
        pad = n_blk * blk - L
        o = np.concatenate([bits, np.zeros(pad, np.int64)]).reshape(n_blk, blk)
        a = np.concatenate([bits, np.full(pad, -1, np.int64)]).reshape(n_blk, blk)
        return np.bitwise_or.reduce(o, axis=1), np.bitwise_and.reduce(a, axis=1)

    num_m, num_n = -(-L // block_m), -(-L // block_n)
    r_or, r_and = summ(num_m, block_m)
    c_or, c_and = summ(num_n, block_n)
    route, cols = 0, []
    for n in range(num_n):
        m_min = (n * block_n) // block_m if causal else 0
        ms = np.arange(num_m - 1, m_min - 1, -1)                 # descending
        live = (r_or[ms] & c_or[n]) == 0
        if ((~live) & ((r_and[ms] & c_and[n]) == 0)).any():
            route |= 1
        runs, hi, prev = [], None, None
        for m, lv in zip(ms, live):
            if lv and hi is None:
                hi = int(m)
            if not lv and hi is not None:
                runs.append([hi, int(prev)])
                hi = None
            prev = m
        if hi is not None:
            runs.append([hi, int(ms[-1])])
        if len(runs) > slots - 1:
            route |= 2
            runs = runs[:slots - 1]
        cols.append(runs)
    return route, cols


def runs_prefix(runs_col):
    """The runs of one column block of the device table, up to the {-1, -1} sentinel."""
    out = []
    for hi, lo in runs_col:
        if hi < 0:
            return out
        out.append([int(hi), int(lo)])
    raise AssertionError("run table column without a sentinel")


def live_tile_fraction(bits, block_m, block_n, causal):
    """Live tiles / visited tiles of the backward's tiling (from the reference census)."""
    _, cols = census_ref(bits, block_m, block_n, causal, slots=10 ** 9)
    L = np.asarray(bits).size
    num_m, num_n = -(-L // block_m), -(-L // block_n)
    visited = sum(num_m - ((n * block_n) // block_m if causal else 0) for n in range(num_n))
    live = sum(hi - lo + 1 for runs in cols for hi, lo in runs)
    return live / visited


_KERNEL_NAMES = {}


def kernel_names(tag, fn):
    """Profiler names of the CUDA kernels fn launches (recorded once per tag)."""
    if tag not in _KERNEL_NAMES:
        try:
            from torch.profiler import profile, ProfilerActivity
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                fn(); torch.cuda.synchronize()
            _KERNEL_NAMES[tag] = sorted({e.name for e in prof.events() if e.device_type.name == "CUDA"})
        except Exception as e:  # noqa
            _KERNEL_NAMES[tag] = [f"profiler unavailable: {type(e).__name__}: {e}"]
    return _KERNEL_NAMES[tag]


def fa2_arms(L, D, causal, iters):
    """flash_attn (the package) on [1,L,H,D] and SDPA-flash on [1,H,L,D], backward only: dict of ms."""
    from flash_attn import flash_attn_func
    from torch.nn.attention import sdpa_kernel, SDPBackend
    torch.manual_seed(1)
    r = {}
    q, k, v = (torch.randn(1, L, H, D, device=DEV, dtype=DT, requires_grad=True) for _ in range(3))
    o = flash_attn_func(q, k, v, causal=causal, softmax_scale=D ** -0.5)
    do = torch.randn_like(o)
    fn = lambda: torch.autograd.grad(o, (q, k, v), do, retain_graph=True)
    r["flash_attn_bwd_ms"], r["flash_attn_bwd_min_ms"] = timed(fn, iters)
    kernel_names("flash_attn_bwd", fn)
    del q, k, v, o, do
    q, k, v = (torch.randn(1, H, L, D, device=DEV, dtype=DT, requires_grad=True) for _ in range(3))
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        o = F.scaled_dot_product_attention(q, k, v, is_causal=causal, scale=D ** -0.5)
        do = torch.randn_like(o)
        fn = lambda: torch.autograd.grad(o, (q, k, v), do, retain_graph=True)
        r["sdpa_flash_bwd_ms"], r["sdpa_flash_bwd_min_ms"] = timed(fn, iters)
        kernel_names("sdpa_flash_bwd", fn)
    del q, k, v, o, do
    torch.cuda.empty_cache()
    return r


# ============================================================================ one config
def part_config(res, key, N, task, D, causal, iters, patterns=("real", "zero"), legacy_arm=True):
    """All arms on one task: FA-2 arms at L, v11x/v12b plain, v11 and v12b per pattern, v12b forced LEGACY."""
    L = int(task.local_size)
    real = task_tables(task)
    zero = tables(torch.zeros(L, dtype=torch.int64))
    pats = {"real": real, "zero": zero}
    torch.manual_seed(0)
    q, k, v, do = (torch.randn(1, L, H, D, device=DEV, dtype=DT) for _ in range(4))
    fa = fa2_arms(L, D, causal, iters)
    base = dict(gpu=GPU, config=key, N=N, L=L,
                L_mod_128=L % 128, path=list(task.path), causal=causal, D=D, **fa)
    rows = []

    def row(arm, pattern, fn, ref=None, route=None):
        r = dict(base, arm=arm, pattern=pattern)
        r["ms"], r["min_ms"] = timed(fn, iters)
        r["ratio_vs_flash_attn"] = r["ms"] / fa["flash_attn_bwd_ms"]
        r["ratio_vs_sdpa"] = r["ms"] / fa["sdpa_flash_bwd_ms"]
        g1 = [x.clone() for x in fn()]
        g2 = fn()
        r["rerun_maxdiff_dq_dk_dv"] = [maxdiff(a, b) for a, b in zip(g1, g2)]
        r["rerun_dk_dv_equal"] = bool(torch.equal(g1[1], g2[1]) and torch.equal(g1[2], g2[2]))
        if ref is not None:
            r["maxdiff_dq_dk_dv_vs_v11"] = [maxdiff(a, b) for a, b in zip(g1, ref)]
            r["dk_dv_equal_vs_v11"] = bool(torch.equal(g1[1], ref[1]) and torch.equal(g1[2], ref[2]))
        if route is not None:
            r["route"] = route
        rows.append(r)
        res["rows"].append(r)
        print(f"[{key}] L={L:6d} D={D:3d} causal={int(causal)} {arm:12s} {pattern:5s} {r['ms']:8.2f} ms"
              f"  /fa {r['ratio_vs_flash_attn']:.3f} /sdpa {r['ratio_vs_sdpa']:.3f}"
              f"  rerun dq {r['rerun_maxdiff_dq_dk_dv'][0]:.1e} dk/dv equal={r['rerun_dk_dv_equal']}"
              + (f"  vs v11 dq {r['maxdiff_dq_dk_dv_vs_v11'][0]:.1e} dk/dv equal={r['dk_dv_equal_vs_v11']}" if ref is not None else "")
              + (f"  route={route}" if route is not None else ""), flush=True)
        return g1

    # CQS off: the upstream loop in v11x (NO_CQS) and in v12b (ALIGNED without runs)
    o16p, lsep = plain_fwd(MODS["v11"], q, k, v, causal)
    ref_plain = [x.clone() for x in cqs_bwd(MODS["v11"], do, q, k, v, o16p, lsep, None, causal)]
    row("v11x_nocqs", "plain", lambda: cqs_bwd(MODS["v11x"], do, q, k, v, o16p, lsep, None, causal), ref_plain)
    row("v12b_plain", "plain", lambda: cqs_bwd(MODS["v12b"], do, q, k, v, o16p, lsep, None, causal), ref_plain)
    row("v11_plain", "plain", lambda: cqs_bwd(MODS["v11"], do, q, k, v, o16p, lsep, None, causal))
    del ref_plain, o16p, lsep
    for pattern in patterns:
        tabs = pats[pattern]
        o16, lse = cqs_fwd(MODS["v11"], q, k, v, tabs, causal)
        route = census(MODS["v12b"], tabs, causal, D)[0]
        ref = row("v11", pattern, lambda: cqs_bwd(MODS["v11"], do, q, k, v, o16, lse, tabs, causal))
        row("v12b", pattern, lambda: cqs_bwd(MODS["v12b"], do, q, k, v, o16, lse, tabs, causal, flags=0), ref, route)
        if legacy_arm and pattern == "real":
            row("v12b_legacy", pattern, lambda: cqs_bwd(MODS["v12b"], do, q, k, v, o16, lse, tabs, causal,
                                                         flags=FORCE_LEGACY), ref)
        del ref, o16, lse
    kernel_names("v12b_real", lambda: cqs_bwd(MODS["v12b"], do, q, k, v, *cqs_fwd(MODS["v11"], q, k, v, real, causal),
                                              real, causal, flags=0))
    del q, k, v, do
    torch.cuda.empty_cache()
    return rows


def part_deterministic(res, task, iters):
    """Item 3: deterministic=True on the causal real task (split dq_accum, gridDim.x = ceil(num_sm / (b*h)))."""
    L, D = int(task.local_size), 64
    tabs = task_tables(task)
    torch.manual_seed(0)
    q, k, v, do = (torch.randn(1, L, H, D, device=DEV, dtype=DT) for _ in range(4))
    o16, lse = cqs_fwd(MODS["v11"], q, k, v, tabs, True)
    r = dict(gpu=GPU, L=L, D=D, causal=True, pattern="real", H=H, b=1,
             sm_count=torch.cuda.get_device_properties(0).multi_processor_count)
    for name, det in (("atomics", False), ("deterministic", True)):
        fn = lambda: cqs_bwd(MODS["v12b"], do, q, k, v, o16, lse, tabs, True, flags=0, deterministic=det)
        r[f"{name}_ms"], r[f"{name}_min_ms"] = timed(fn, iters)
        gs = [[x.clone() for x in fn()] for _ in range(3)]
        r[f"{name}_reruns_equal"] = all(torch.equal(a, b) for g in gs[1:] for a, b in zip(gs[0], g))
        r[f"{name}_rerun_maxdiff_dq"] = max(maxdiff(gs[0][0], g[0]) for g in gs[1:])
        r[f"{name}_grads"] = gs[0]
    r["det_vs_atomics_maxdiff_dq_dk_dv"] = [maxdiff(a, b) for a, b in zip(r.pop("deterministic_grads"), r.pop("atomics_grads"))]
    r["grid_x"] = -(-r["sm_count"] // H)
    res["deterministic"] = r
    print(f"[det] L={L} atomics {r['atomics_ms']:.2f} ms (reruns equal {r['atomics_reruns_equal']}, dq drift "
          f"{r['atomics_rerun_maxdiff_dq']:.1e})  deterministic {r['deterministic_ms']:.2f} ms (reruns equal "
          f"{r['deterministic_reruns_equal']})  det vs atomics {['%.1e' % x for x in r['det_vs_atomics_maxdiff_dq_dk_dv']]}",
          flush=True)
    del q, k, v, do, o16, lse
    torch.cuda.empty_cache()


def part_c73(res, N, iters):
    """c=73, N=131072, causal D=64, real: all 73 tasks in one timed region per arm; FA-2 arms per distinct L,
    weighted by multiplicity. The census of every task is inside v12b's region."""
    D = 64
    tasks = sorted(c_tasks(N, 73), key=lambda t: (int(t.local_size), t.path))
    Ls = [int(t.local_size) for t in tasks]
    Lmax = max(Ls)
    torch.manual_seed(0)
    q, k, v, do = (torch.randn(1, Lmax, H, D, device=DEV, dtype=DT) for _ in range(4))
    items = []
    for t, L in zip(tasks, Ls):
        tabs = task_tables(t)
        qs, ks, vs, dos = (x[:, :L] for x in (q, k, v, do))
        o16, lse = cqs_fwd(MODS["v11"], qs, ks, vs, tabs, True)
        items.append((qs, ks, vs, dos, o16, lse, tabs))
    routes = sorted({census(MODS["v12b"], it[6], True, D)[0] for it in items})

    def run_all(mod, flags=None, cqs=True):
        out = None
        for qs, ks, vs, dos, o16, lse, tabs in items:
            out = cqs_bwd(mod, dos, qs, ks, vs, o16, lse, tabs if cqs else None, True, flags=flags)
        return out

    r = dict(gpu=GPU, N=N, c=73, tasks=len(tasks), L_min=min(Ls), L_max=Lmax, sum_L=sum(Ls), D=D, causal=True,
             pattern="real", routes=routes)
    for arm, fn in (("v11", lambda: run_all(MODS["v11"])), ("v12b", lambda: run_all(MODS["v12b"], flags=0)),
                    ("v12b_legacy", lambda: run_all(MODS["v12b"], flags=FORCE_LEGACY)),
                    ("v11x_nocqs", lambda: run_all(MODS["v11x"], cqs=False))):
        r[f"{arm}_ms"], r[f"{arm}_min_ms"] = timed(fn, iters)
    # identity on every task: v12b dk/dv == v11's
    eq, dq_max = True, 0.0
    for qs, ks, vs, dos, o16, lse, tabs in items:
        a = cqs_bwd(MODS["v11"], dos, qs, ks, vs, o16, lse, tabs, True)
        b = cqs_bwd(MODS["v12b"], dos, qs, ks, vs, o16, lse, tabs, True, flags=0)
        eq &= bool(torch.equal(a[1], b[1]) and torch.equal(a[2], b[2]))
        dq_max = max(dq_max, maxdiff(a[0], b[0]))
    r["dk_dv_equal_vs_v11_all_tasks"], r["maxdiff_dq_vs_v11"] = eq, dq_max
    del items, q, k, v, do
    torch.cuda.empty_cache()
    fa_tot, sd_tot = 0.0, 0.0
    for L in sorted(set(Ls)):
        fa = fa2_arms(L, D, True, iters)
        m = Ls.count(L)
        fa_tot += m * fa["flash_attn_bwd_ms"]
        sd_tot += m * fa["sdpa_flash_bwd_ms"]
    r["flash_attn_bwd_sum_ms"], r["sdpa_flash_bwd_sum_ms"] = fa_tot, sd_tot
    mono = fa2_arms(N, D, True, iters)
    r["monolithic_flash_attn_bwd_ms"], r["monolithic_sdpa_flash_bwd_ms"] = mono["flash_attn_bwd_ms"], mono["sdpa_flash_bwd_ms"]
    for arm in ("v11", "v12b", "v12b_legacy", "v11x_nocqs"):
        r[f"{arm}_ratio_vs_flash_attn"] = r[f"{arm}_ms"] / fa_tot
        r[f"{arm}_ratio_vs_sdpa"] = r[f"{arm}_ms"] / sd_tot
    res["c73"] = r
    print(f"[c73] N={N} {len(tasks)} tasks L {min(Ls)}-{Lmax}: " + "  ".join(
        f"{arm} {r[arm + '_ms']:.1f} ms (/fa {r[arm + '_ratio_vs_flash_attn']:.3f})" for arm in ("v11", "v12b", "v12b_legacy", "v11x_nocqs"))
          + f"  | flash_attn sum {fa_tot:.1f}  sdpa sum {sd_tot:.1f}  monolithic flash_attn {mono['flash_attn_bwd_ms']:.1f}"
          f"  routes={routes} dk/dv equal={eq} dq max|d| {dq_max:.1e}", flush=True)


# ============================================================================ report
def _row(res, config, arm, pattern):
    for r in res.get("rows", []):
        if r["config"] == config and r["arm"] == arm and r["pattern"] == pattern:
            return r
    return None


def report(res, path_md):
    info = res["info"]
    L = []
    L.append("# Backward A/B (P2): v12b vs v11 / v11x / flash_attn / SDPA-flash")
    L.append("")
    L.append(f"Job **{info.get('slurm_job')}**, {info.get('node')}, **{info.get('gpu')}** (`{info.get('nvidia_smi')}`), "
             f"torch {info.get('torch')}, flash_attn {str(info.get('flash_attn', '?')).split()[0]}, fp16, H={res['H']}, "
             f"{res['timing']}. Harness `benchmarks/ultimate/bwd_ab.py` (`slurm/ultimate/bwd_ab.slurm`), raw rows "
             f"`bwd_ab.json`. Every CQS arm runs its own dsoftmax_sum preprocess (as flash_attn does); v12b's census is "
             f"inside its timed region. Modules: " + ", ".join(f"{k} = `{v}`" for k, v in info.get("modules", {}).items()) + ".")
    L.append("")
    L.append("## Per-config rows (ms, median; ratio vs flash_attn bwd / vs SDPA-flash bwd)")
    L.append("")
    L.append("| config | L | D | causal | flash_attn | SDPA | v11x NO_CQS | v12b plain | v11 real | v12b real | v12b LEGACY real | v11 zero | v12b zero | v12b route (real) | dk/dv == v11 (real, zero) | dq max vs v11 (real) |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for cfg in res.get("configs", []):
        g = lambda a, p: _row(res, cfg, a, p)
        base = g("v11x_nocqs", "plain")
        if base is None:
            continue
        fa, sd = base["flash_attn_bwd_ms"], base["sdpa_flash_bwd_ms"]
        cell = lambda r: "-" if r is None else f"{r['ms']:.2f} ({r['ms'] / fa:.3f} / {r['ms'] / sd:.3f})"
        rr, rz = g("v12b", "real"), g("v12b", "zero")
        eq = f"{rr.get('dk_dv_equal_vs_v11') if rr else '-'}, {rz.get('dk_dv_equal_vs_v11') if rz else '-'}"
        L.append(f"| {cfg} | {base['L']} | {base['D']} | {int(base['causal'])} | {fa:.2f} | {sd:.2f} | {cell(base)} | "
                 f"{cell(g('v12b_plain', 'plain'))} | {cell(g('v11', 'real'))} | {cell(rr)} | {cell(g('v12b_legacy', 'real'))} | "
                 f"{cell(g('v11', 'zero'))} | {cell(rz)} | {rr.get('route') if rr else '-'} | {eq} | "
                 f"{('%.1e' % rr['maxdiff_dq_dk_dv_vs_v11'][0]) if rr else '-'} |")
    L.append("")
    L.append("## Targets")
    L.append("")
    L.append("| target | value | pass |")
    L.append("|---|---|---|")
    for cfg in res.get("configs", []):
        z, n = _row(res, cfg, "v12b", "zero"), _row(res, cfg, "v11x_nocqs", "plain")
        if z and n:
            v = z["ms"] / n["ms"]
            L.append(f"| {cfg}: v12b zero / v11x NO_CQS <= 1.05 | {v:.3f} | {v <= 1.05} |")
    for cfg in res.get("configs", []):
        r = _row(res, cfg, "v12b", "real")
        if r and r["L"] > 50000:
            if r["causal"] and r["D"] == 64:
                L.append(f"| {cfg}: v12b real / flash_attn bwd <= 0.85 | {r['ratio_vs_flash_attn']:.3f} | {r['ratio_vs_flash_attn'] <= 0.85} |")
                L.append(f"| {cfg}: v12b real / SDPA bwd <= 1.30 (design P2.1 gate) | {r['ratio_vs_sdpa']:.3f} | {r['ratio_vs_sdpa'] <= 1.30} |")
            elif not r["causal"]:
                L.append(f"| {cfg}: non-causal v12b real / SDPA bwd <= 1.10 (design P2.1 gate) | {r['ratio_vs_sdpa']:.3f} | {r['ratio_vs_sdpa'] <= 1.10} |")
                L.append(f"| {cfg}: non-causal v12b real / flash_attn bwd (recorded) | {r['ratio_vs_flash_attn']:.3f} | - |")
            else:
                L.append(f"| {cfg}: v12b real / flash_attn bwd (recorded) | {r['ratio_vs_flash_attn']:.3f} | - |")
    c = res.get("c73")
    if c:
        per = c["v12b_ms"] / c["sdpa_flash_bwd_sum_ms"]
        L.append(f"| c=73 N={c['N']} (L {c['L_min']}-{c['L_max']}): v12b sum / SDPA sum <= 1.45 (design: L=16159 causal) | {per:.3f} | {per <= 1.45} |")
    L.append("")
    if c:
        L.append(f"## c=73, N={c['N']}, causal D=64, real: {c['tasks']} tasks (L {c['L_min']}-{c['L_max']}) in one timed region per arm")
        L.append("")
        L.append("| arm | ms | / flash_attn sum | / SDPA sum |")
        L.append("|---|---|---|---|")
        for arm in ("v11", "v12b", "v12b_legacy", "v11x_nocqs"):
            L.append(f"| {arm} | {c[arm + '_ms']:.1f} | {c[arm + '_ratio_vs_flash_attn']:.3f} | {c[arm + '_ratio_vs_sdpa']:.3f} |")
        L.append(f"| flash_attn per task, summed | {c['flash_attn_bwd_sum_ms']:.1f} | 1 | - |")
        L.append(f"| SDPA-flash per task, summed | {c['sdpa_flash_bwd_sum_ms']:.1f} | - | 1 |")
        L.append(f"| monolithic flash_attn at N={c['N']} | {c['monolithic_flash_attn_bwd_ms']:.1f} | - | - |")
        L.append("")
        L.append(f"Routes {c['routes']}; v12b dk/dv torch.equal to v11 on every task: {c['dk_dv_equal_vs_v11_all_tasks']}; "
                 f"dq max |d| vs v11 {c['maxdiff_dq_vs_v11']:.1e}.")
        L.append("")
    d = res.get("deterministic")
    if d:
        L.append(f"## deterministic=True (item 3), L={d['L']} causal real D=64, H={d['H']}, b=1, {d['sm_count']} SMs")
        L.append("")
        L.append(f"atomics {d['atomics_ms']:.2f} ms (3 reruns torch.equal: {d['atomics_reruns_equal']}, dq drift "
                 f"{d['atomics_rerun_maxdiff_dq']:.1e}); deterministic {d['deterministic_ms']:.2f} ms "
                 f"({d['deterministic_ms'] / d['atomics_ms']:.3f}x; 3 reruns torch.equal: {d['deterministic_reruns_equal']}); "
                 f"deterministic vs atomics max |d| dq/dk/dv {', '.join('%.1e' % x for x in d['det_vs_atomics_maxdiff_dq_dk_dv'])}. "
                 f"Grid x = ceil({d['sm_count']} / (b*H)) = {d['grid_x']} CTAs per head, each walking its column blocks "
                 f"serially: free only when W*H is close to a multiple of the SM count (design 4.7 item 5).")
        L.append("")
    L.append("## Kernels recorded by the profiler")
    L.append("")
    for tag, names in res.get("kernel_names", {}).items():
        L.append(f"- {tag}: " + "; ".join(f"`{n[:140]}`" for n in names))
    L.append("")
    open(path_md, "w").write("\n".join(L) + "\n")
    return path_md


# ============================================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="small N, 2 iterations: correctness on a shared GPU")
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--out", default=os.environ.get("BWD_AB_OUT", OUT_DEFAULT))
    ap.add_argument("--report", default=None, help="write the .md next to this json and exit")
    ap.add_argument("--only", default=None, help="comma list of parts: c7,c73,d128,nc,det")
    a = ap.parse_args()
    if a.report:
        res = json.load(open(a.report))
        print(report(res, a.report.replace(".json", ".md")))
        return
    load_modules()
    only = set(a.only.split(",")) if a.only else {"c7", "c73", "d128", "nc", "det"}
    Ns = [32768, 65536, 131072]
    N_big, N73 = 131072, 131072
    if a.smoke:
        Ns, N_big, N73, a.iters = [8192, 16384], 16384, 32768, 2
        a.out = a.out.replace(".json", "_smoke.json")
    t0 = time.time()
    res = dict(info=gpu_info(), iters=a.iters, H=H, dtype="float16", smoke=a.smoke, rows=[], configs=[],
               timing=f"1 warm-up + {a.iters} timed runs, CUDA events, median (min also stored)")
    print(json.dumps(res["info"]), flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    parts = []
    if "c7" in only:
        for N in Ns:
            parts.append((f"c7_N{N}_causal_d64", lambda N=N: part_config(res, f"c7_N{N}_causal_d64", N, aligned_task(N), 64, True, a.iters)))
    if "d128" in only:
        parts.append((f"c7_N{N_big}_causal_d128", lambda: part_config(res, f"c7_N{N_big}_causal_d128", N_big, aligned_task(N_big), 128, True, a.iters)))
    if "nc" in only:
        parts.append((f"c7_N{N_big}_noncausal_d64", lambda: part_config(res, f"c7_N{N_big}_noncausal_d64", N_big, aligned_task(N_big), 64, False, a.iters)))
    if "c73" in only:
        parts.append(("c73", lambda: part_c73(res, N73, a.iters)))
    if "det" in only:
        parts.append(("det", lambda: part_deterministic(res, aligned_task(N_big), a.iters)))
    for name, fn in parts:
        try:
            if name.startswith("c7_"):
                res["configs"].append(name)
            fn()
        except Exception as e:  # noqa
            res[f"error_{name}"] = traceback.format_exc()
            print(f"!! {name} failed: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
        res["kernel_names"] = _KERNEL_NAMES
        json.dump(res, open(a.out, "w"), indent=1)
    res["elapsed_s"] = time.time() - t0
    json.dump(res, open(a.out, "w"), indent=1)
    try:
        print(open(report(res, a.out.replace(".json", ".md"))).read())
    except Exception:  # noqa
        traceback.print_exc()
    print(f"\nwrote {a.out}  ({res['elapsed_s']:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
