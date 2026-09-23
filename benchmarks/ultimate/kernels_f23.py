"""
F23 combined kernel A/B: the rows the four kernel reviews asked for, for the review-fixed builds v12f b3 and
v12b b5, ONE process on ONE GPU (A100-SXM4-80GB in slurm/ultimate/kernels_f23.slurm).

Why: the F23 fixes are host-side (routing gates, a TORCH_CHECK, an env parse, a comment); the SASS of every
ALIGNED / LEGACY / census kernel is compared with the measured builds separately (v12f b2, v12b b4). This job
re-times the headline rows on the fixed builds and adds the rows the reviews found missing:

  fwd_headline   causal D=64 L=56064 (N=131072 c=7), H=8 and H=32: flash_attn / SDPA-flash (3 warm-ups, 10
                 runs), v11 plain (the in-tree upstream loop), v12f plain / zero / real with the default order
                 and both forced orders (descending = flags 2, ascending = flags 4): the tax figure with the same
                 order on both sides, the descending gain as its own line, the order default at B*H = 32.
  fwd_ragged     causal D=64, N=100003 c=7: the even task (L=42752) and the smallest ragged one (L=42787, the
                 Is_even_MN=false ALIGNED instantiation, never timed before): v11 / v12f plain, zero, real.
  fwd_small      c=73 itr=2 N=131072 (L 1664-2176, the production itr=2 regime): 73 tasks evenly spaced over the
                 5329, one timed region per arm (the per-call census + no-op LEGACY + memset floor at small L),
                 per-kernel device sums from the profiler.
  bwd_headline   causal D=64 L=56064: flash_attn (autograd and direct), v11x NO_CQS, v12b plain / zero / real,
                 v11 real; deterministic=True time and peak memory next to the atomics path.
  bwd_ragged     causal D=64, N=100003 c=7 L=42752 vs 42787 (Is_even_MN=false ALIGNED backward).
  bwd_legacy     the AUTOMATIC LEGACY route (census -> ALIGNED exit -> LEGACY) that every unaligned
                 (seg_align=0) task takes, against v11 and forced LEGACY, at L=14043 and 56173; forced LEGACY vs
                 v11 re-measured on the aligned L=13952 task (the +1.0% row of job 14323878).
  bwd_small      c=13 itr=3 (L 3584-4096) and c=31 itr=2 (L 4608-5248) at N=131072: 64 tasks each, one timed
                 region per arm, per-kernel device sums.

Identity in every part: CQS outputs written into NaN-pre-filled buffers; v12f == v11 (out32, lse, out16), v12b
dk/dv torch.equal to v11 and dq within the atomics' drift. GPU name in every row.

    SNAP=$(bash slurm/ultimate/snapshot.sh) && sbatch --export=ALL,CQSA_SRC=$SNAP,V12F_DIR=..,V12B_DIR=.. slurm/ultimate/kernels_f23.slurm
    python -P benchmarks/ultimate/kernels_f23.py --smoke        # correctness only (shared vis1 GPU, small N)
    python -P benchmarks/ultimate/kernels_f23.py --report results/ultimate/kernels_f23.json
"""
import argparse, functools, json, os, sys, time, traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("CQSA_CPU_THREADS", "4")

import torch

import fwd_ab as F
import bwd_ab as B

DEV = torch.device("cuda")
DT = torch.float16
GPU = F.GPU
NAN = float("nan")
OUT_DEFAULT = "/scratch/gpfs/AKEY/yb2807/Stream-CQSA-ultimate/results/ultimate/kernels_f23.json"
ITERS, WARM = 7, 2               # our arms
FA_ITERS, FA_WARM = 10, 3        # FA arms (review: the SDPA arm was bimodal with 1 warm-up / 5 runs)


def timed(fn, iters=None, warm=None):
    return F.timed(fn, ITERS if iters is None else iters, WARM if warm is None else warm)


def kind(name):
    n = name.lower()
    if "census" in n:
        return "census"
    if "aligned" in n:
        return "aligned"
    if "flash_fwd_kernel" in name or "dq_dk_dv" in n:
        return "legacy"
    if "memset" in n:
        return "memset"
    return "other"


def kernel_us(fn):
    """Device us per kernel kind over ONE call of fn (a call may be a loop over tasks)."""
    try:
        from torch.profiler import profile, ProfilerActivity
        fn(); torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            fn(); torch.cuda.synchronize()
        out, n = {}, {}
        for e in prof.events():
            if e.device_type.name == "CUDA":
                k = kind(e.name)
                out[k] = out.get(k, 0.0) + float(e.time_range.elapsed_us())
                n[k] = n.get(k, 0) + 1
        return {"us": out, "launches": n}
    except Exception as e:  # noqa
        return {"error": f"{type(e).__name__}: {e}"}


def fa_fwd_arms(L, D, causal, H):
    from flash_attn import flash_attn_func
    from torch.nn.attention import sdpa_kernel, SDPBackend
    torch.manual_seed(1)
    r = {}
    q, k, v = (torch.randn(1, L, H, D, device=DEV, dtype=DT) for _ in range(3))
    r["flash_attn_ms"], r["flash_attn_min_ms"] = timed(lambda: flash_attn_func(q, k, v, causal=causal, softmax_scale=D ** -0.5),
                                                       FA_ITERS, FA_WARM)
    del q, k, v
    q, k, v = (torch.randn(1, H, L, D, device=DEV, dtype=DT) for _ in range(3))
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        r["sdpa_flash_ms"], r["sdpa_flash_min_ms"] = timed(
            lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=causal, scale=D ** -0.5),
            FA_ITERS, FA_WARM)
    del q, k, v
    torch.cuda.empty_cache()
    return r


def gathered(task, N, D, H, n=3, seed=0):
    torch.manual_seed(seed)
    idx = task.token_ids.to(DEV)
    out = []
    for _ in range(n):
        x = torch.randn(1, N, H, D, device=DEV, dtype=DT)
        out.append(x.index_select(1, idx).contiguous())
        del x
    return out


def eq_fwd(a, b):
    return all(torch.equal(x, y) for x, y in zip(a, b) if x is not None and y is not None)


def pr(tag, r):
    keep = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()
            if k not in ("gpu", "kernel_us") and not isinstance(v, (list, dict))}
    print(f"[{tag}] {json.dumps(keep)}" + (f" kernels={r['kernel_us'].get('us')}" if isinstance(r.get("kernel_us"), dict) else ""),
          flush=True)


# ============================================================================ forward
def fwd_task_rows(res, part, task, N, D, causal, H, arms, zero_pattern=True):
    """Rows for one task: FA arms at L, then (module, pattern, flags) arms; identity vs v11 with NaN fill."""
    L = int(task.local_size)
    q, k, v = gathered(task, N, D, H)
    pats = {"real": F.task_tables(task)}
    if zero_pattern:
        pats["zero"] = F.tables(torch.zeros(L, dtype=torch.int64))
    fa = fa_fwd_arms(L, D, causal, H)
    routes = {p: F.census(F.MODS["v12f"], pats[p], causal, D)[0] for p in pats}
    ref = {}
    for module, pattern, flags in arms:
        if pattern not in pats and pattern != "plain":
            continue
        r = dict(gpu=GPU, part=part, N=N, L=L, L_mod_128=L % 128, H=H, D=D, causal=causal, module=module,
                 pattern=pattern, flags=flags, **fa)
        fn = lambda: F.run_arm(module, pattern, flags, q, k, v, pats, causal)
        outs = F.run_arm(module, pattern, flags, q, k, v, pats, causal, fill=NAN)
        r["ms"], r["min_ms"] = timed(fn)
        r["ratio_vs_flash_attn"] = r["ms"] / fa["flash_attn_ms"]
        if module == "v11" and pattern not in ref:
            ref[pattern] = outs
        else:
            r["equal_vs_v11"] = eq_fwd(outs, ref[pattern])
        if module == "v12f" and pattern != "plain":
            r["census_route"] = routes[pattern]
            r["kernel_us"] = kernel_us(fn)
        res["rows"].append(r)
        pr(part, r)
        del outs
    del q, k, v, ref
    torch.cuda.empty_cache()


FWD_ARMS = [("v11", "plain", None), ("v11", "zero", None), ("v11", "real", None), ("v12f", "plain", None),
            ("v12f", "zero", 0), ("v12f", "real", 0), ("v12f", "zero", F.REVERSE_M), ("v12f", "zero", F.ASCENDING_M),
            ("v12f", "real", F.REVERSE_M), ("v12f", "real", F.ASCENDING_M), ("v11", "plain", "again")]


def run_arm_again(arms):
    # ("v11", "plain", "again"): the first arm re-timed at the end (drift check); run_arm ignores flags for plain
    return [(m, p, None if f == "again" else f) if f != "again" else (m, p, None) for m, p, f in arms]


def gpu_spin(seconds):
    """Back-to-back fp16 GEMMs for `seconds`: SM clocks at their steady state before the first timed row (review:
    the first FA arm of jobs 14319938 / 14324080 ran at 18.2 ms at L=56064 while the same kernel took 17.1 ms at
    L=56173 later in the same job)."""
    a = torch.randn(8192, 8192, device=DEV, dtype=DT)
    t0 = time.time()
    while time.time() - t0 < seconds:
        for _ in range(10):
            a @ a
        torch.cuda.synchronize()
    del a
    torch.cuda.empty_cache()


def part_fwd_headline(res, N, H_list=(8, 32)):
    task = min(F.c_tasks(N, 7), key=lambda t: int(t.local_size))
    L = int(task.local_size)
    for H in H_list:
        arms = [a for a in FWD_ARMS]
        rows0 = len(res["rows"])
        fwd_task_rows(res, "fwd_headline", task, N, 64, True, H, run_arm_again(arms))
        # label the drift row
        rs = res["rows"][rows0:]
        plain11 = [r for r in rs if r["module"] == "v11" and r["pattern"] == "plain"]
        if len(plain11) == 2:
            plain11[1]["again"] = True
        # the FA arms re-timed after every other arm of the group (drift / first-arm effect)
        res.setdefault("fa_again", []).append(dict(gpu=GPU, part="fwd_headline", L=L, H=H, D=64, causal=True,
                                                   **fa_fwd_arms(L, 64, True, H)))
        pr("fa_again", res["fa_again"][-1])


def part_fwd_ragged(res, N):
    tasks = sorted(F.c_tasks(N, 7), key=lambda t: (int(t.local_size), t.path))
    even = min((t for t in tasks if int(t.local_size) % 128 == 0), key=lambda t: int(t.local_size))
    ragged = min((t for t in tasks if int(t.local_size) % 128), key=lambda t: int(t.local_size))
    arms = [("v11", "plain", None), ("v11", "zero", None), ("v11", "real", None), ("v12f", "plain", None),
            ("v12f", "zero", 0), ("v12f", "real", 0)]
    for t in (even, ragged):
        fwd_task_rows(res, "fwd_ragged", t, N, 64, True, 8, arms)


@functools.lru_cache(maxsize=None)
def tasks_of(N, c, itr, seg_align=128):
    return tuple(F.c_tasks(N, c, itr, seg_align=seg_align))


def spaced(tasks, k):
    tasks = sorted(tasks, key=lambda t: (int(t.local_size), t.path))
    step = max(1, len(tasks) // k)
    return tasks[::step][:k]


def part_fwd_small(res, N, c, itr, k, D=64, causal=True, H=8):
    """k tasks of the c/itr decomposition in one timed region per arm (prefix views of one Lmax set)."""
    tasks = spaced(tasks_of(N, c, itr), k)
    Ls = [int(t.local_size) for t in tasks]
    Lmax = max(Ls)
    torch.manual_seed(0)
    q, k_, v = (torch.randn(1, Lmax, H, D, device=DEV, dtype=DT) for _ in range(3))
    items = []
    zero_by_L = {}
    for t, L in zip(tasks, Ls):
        if L not in zero_by_L:
            zero_by_L[L] = F.tables(torch.zeros(L, dtype=torch.int64))
        items.append((q[:, :L], k_[:, :L], v[:, :L], {"real": F.task_tables(t), "zero": zero_by_L[L]}))
    routes = sorted({F.census(F.MODS["v12f"], it[3]["real"], causal, D)[0] for it in items})
    fa = B.fa_module()

    def loop(module, pattern, flags=None):
        def call():
            out = None
            for qs, ks, vs, pats in items:
                out = F.run_arm(module, pattern, flags, qs, ks, vs, pats, causal)
            return out
        return call

    def loop_fa():
        out = None
        for qs, ks, vs, _ in items:
            out = F.plain_fwd(fa, qs, ks, vs, causal)
        return out

    r = dict(gpu=GPU, part=f"fwd_small_c{c}_itr{itr}", N=N, c=c, itr=itr, tasks=len(items), of_tasks=len(tasks_of(N, c, itr)),
             L_min=min(Ls), L_max=Lmax, sum_L=sum(Ls), H=H, D=D, causal=causal, routes=routes)
    arms = {"flash_attn_direct": loop_fa, "v11_plain": loop("v11", "plain"), "v11_real": loop("v11", "real"),
            "v12f_plain": loop("v12f", "plain"), "v12f_zero": loop("v12f", "zero", 0), "v12f_real": loop("v12f", "real", 0),
            "v12f_legacy_real": loop("v12f", "real", F.FORCE_LEGACY)}
    for name, fn in arms.items():
        r[f"{name}_ms"], r[f"{name}_min_ms"] = timed(fn)
    for name in ("v12f_real", "v12f_zero", "v12f_plain", "v11_real"):
        r[f"{name}_kernel_us"] = kernel_us(arms[name])
    eq = True
    for qs, ks, vs, pats in items:
        a = F.run_arm("v11", "real", None, qs, ks, vs, pats, causal, fill=NAN)
        b = F.run_arm("v12f", "real", 0, qs, ks, vs, pats, causal, fill=NAN)
        eq &= eq_fwd(a, b)
    r["v12f_real_equal_v11_all_tasks"] = eq
    res["small"].append(r)
    print(f"[{r['part']}] {len(items)} tasks L {min(Ls)}-{Lmax}: " + "  ".join(f"{n} {r[n + '_ms']:.2f}" for n in arms)
          + f"  routes={routes} equal={eq}  v12f_real kernels={r['v12f_real_kernel_us'].get('us')}", flush=True)
    del items, q, k_, v
    torch.cuda.empty_cache()


# ============================================================================ backward
def fa_bwd_arms(L, D, causal, H=8):
    from flash_attn import flash_attn_func
    from torch.nn.attention import sdpa_kernel, SDPBackend
    torch.manual_seed(1)
    r = {}
    q, k, v = (torch.randn(1, L, H, D, device=DEV, dtype=DT, requires_grad=True) for _ in range(3))
    o = flash_attn_func(q, k, v, causal=causal, softmax_scale=D ** -0.5)
    do = torch.randn_like(o)
    r["flash_attn_bwd_ms"], r["flash_attn_bwd_min_ms"] = timed(
        lambda: torch.autograd.grad(o, (q, k, v), do, retain_graph=True), FA_ITERS, FA_WARM)
    fa = B.fa_module()
    qd, kd, vd, dod = (x.detach() for x in (q, k, v, do))
    od, lsed = B.plain_fwd(fa, qd, kd, vd, causal)
    r["flash_attn_direct_bwd_ms"], r["flash_attn_direct_bwd_min_ms"] = timed(
        lambda: B.cqs_bwd(fa, dod, qd, kd, vd, od, lsed, None, causal), FA_ITERS, FA_WARM)
    del q, k, v, o, do, qd, kd, vd, dod, od, lsed
    q, k, v = (torch.randn(1, H, L, D, device=DEV, dtype=DT, requires_grad=True) for _ in range(3))
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        o = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=causal, scale=D ** -0.5)
        do = torch.randn_like(o)
        r["sdpa_flash_bwd_ms"], r["sdpa_flash_bwd_min_ms"] = timed(
            lambda: torch.autograd.grad(o, (q, k, v), do, retain_graph=True), FA_ITERS, FA_WARM)
    del q, k, v, o, do
    torch.cuda.empty_cache()
    return r


def bwd_task_rows(res, part, task, N, D, causal, arms, H=8, with_fa=True, extra=None):
    """arms: (name, module, pattern, flags); pattern in real / zero / plain. v11 rows are the identity reference."""
    L = int(task.local_size)
    torch.manual_seed(0)
    q, k, v, do = (torch.randn(1, L, H, D, device=DEV, dtype=DT) for _ in range(4))
    pats = {"real": B.task_tables(task), "zero": B.tables(torch.zeros(L, dtype=torch.int64))}
    fa = fa_bwd_arms(L, D, causal, H) if with_fa else {}
    fw = {p: B.cqs_fwd(B.MODS["v11"], q, k, v, pats[p], causal) for p in pats}
    fw["plain"] = B.plain_fwd(B.MODS["v11"], q, k, v, causal)
    ref = {}
    for name, module, pattern, flags in arms:
        o16, lse = fw[pattern]
        tabs = None if pattern == "plain" else pats[pattern]
        fn = lambda: B.cqs_bwd(B.MODS[module], do, q, k, v, o16, lse, tabs, causal, flags=flags)
        r = dict(gpu=GPU, part=part, N=N, L=L, L_mod_128=L % 128, D=D, causal=causal, arm=name, module=module,
                 pattern=pattern, flags=flags, **fa, **(extra or {}))
        r["ms"], r["min_ms"] = timed(fn)
        if fa:
            r["ratio_vs_flash_attn"] = r["ms"] / fa["flash_attn_bwd_ms"]
        g = [x.clone() for x in B.cqs_bwd(B.MODS[module], do, q, k, v, o16, lse, tabs, causal, flags=flags, fill=NAN)]
        if module == "v11" and pattern not in ref:
            ref[pattern] = g
        elif pattern in ref:
            r["dk_dv_equal_vs_v11"] = bool(torch.equal(g[1], ref[pattern][1]) and torch.equal(g[2], ref[pattern][2]))
            r["dq_maxdiff_vs_v11"] = B.maxdiff(g[0], ref[pattern][0])
        r["all_finite"] = bool(all(torch.isfinite(x).all() for x in g))
        if module == "v12b" and tabs is not None:
            r["census_route"] = B.census(B.MODS["v12b"], tabs, causal, D)[0]
            r["kernel_us"] = kernel_us(fn)
        res["rows"].append(r)
        pr(part, r)
    del q, k, v, do, fw, ref
    torch.cuda.empty_cache()


BWD_ARMS = [("v11_real", "v11", "real", None), ("v11x_nocqs", "v11x", "plain", None), ("v12b_plain", "v12b", "plain", None),
            ("v12b_real", "v12b", "real", 0), ("v12b_zero", "v12b", "zero", 0), ("v11_zero", "v11", "zero", None),
            ("v11x_nocqs_again", "v11x", "plain", None)]


def part_bwd_headline(res, N):
    task = B.aligned_task(N)
    bwd_task_rows(res, "bwd_headline", task, N, 64, True, BWD_ARMS)
    # deterministic=True next to the atomics path: time and peak memory (review: 1.6 GB of split dq_accum unrecorded)
    L, D, H = int(task.local_size), 64, 8
    tabs = B.task_tables(task)
    torch.manual_seed(0)
    q, k, v, do = (torch.randn(1, L, H, D, device=DEV, dtype=DT) for _ in range(4))
    o16, lse = B.cqs_fwd(B.MODS["v11"], q, k, v, tabs, True)
    r = dict(gpu=GPU, part="bwd_deterministic", L=L, D=D, H=H, b=1, sm_count=torch.cuda.get_device_properties(0).multi_processor_count)
    for name, det in (("atomics", False), ("deterministic", True)):
        fn = lambda: B.cqs_bwd(B.MODS["v12b"], do, q, k, v, o16, lse, tabs, True, flags=0, deterministic=det)
        r[f"{name}_ms"], r[f"{name}_min_ms"] = timed(fn)
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        fn(); torch.cuda.synchronize()
        r[f"{name}_peak_extra_MiB"] = (torch.cuda.max_memory_allocated() - base) / 2 ** 20
    res["deterministic"] = r
    pr("bwd_deterministic", r)
    del q, k, v, do, o16, lse
    torch.cuda.empty_cache()


def part_bwd_ragged(res, N):
    tasks = sorted(B.c_tasks(N, 7), key=lambda t: (int(t.local_size), t.path))
    even = min((t for t in tasks if int(t.local_size) % 128 == 0), key=lambda t: int(t.local_size))
    ragged = min((t for t in tasks if int(t.local_size) % 128), key=lambda t: int(t.local_size))
    arms = [a for a in BWD_ARMS if a[0] != "v11x_nocqs_again"]
    for t in (even, ragged):
        bwd_task_rows(res, "bwd_ragged", t, N, 64, True, arms)


def part_bwd_legacy(res, Ns):
    """Automatic LEGACY route on unaligned (seg_align=0) tasks; forced LEGACY on the aligned 32768 task."""
    arms = [("v11_real", "v11", "real", None), ("v12b_auto", "v12b", "real", 0),
            ("v12b_forced_legacy", "v12b", "real", B.FORCE_LEGACY), ("v11_real_again", "v11", "real", None)]
    for N in Ns:
        task = min(B.c_tasks(N, 7, seg_align=0), key=lambda t: int(t.local_size))
        bwd_task_rows(res, "bwd_legacy_unaligned", task, N, 64, True, arms, with_fa=False, extra={"seg_align": 0})
    task = B.aligned_task(Ns[0])
    bwd_task_rows(res, "bwd_legacy_aligned", task, Ns[0], 64, True,
                  [("v11_real", "v11", "real", None), ("v12b_forced_legacy", "v12b", "real", B.FORCE_LEGACY),
                   ("v11_real_again", "v11", "real", None), ("v12b_forced_legacy_again", "v12b", "real", B.FORCE_LEGACY)],
                  with_fa=False, extra={"seg_align": 128})


def part_bwd_small(res, N, c, itr, k, D=64, causal=True, H=8):
    tasks = spaced(tasks_of(N, c, itr), k)
    Ls = [int(t.local_size) for t in tasks]
    Lmax = max(Ls)
    torch.manual_seed(0)
    q, k_, v, do = (torch.randn(1, Lmax, H, D, device=DEV, dtype=DT) for _ in range(4))
    items = []
    zero_by_L = {}
    for t, L in zip(tasks, Ls):
        if L not in zero_by_L:
            zero_by_L[L] = B.tables(torch.zeros(L, dtype=torch.int64))
        qs, ks, vs, dos = (x[:, :L] for x in (q, k_, v, do))
        real = B.task_tables(t)
        items.append(dict(x=(qs, ks, vs, dos), real=real, zero=zero_by_L[L],
                          f_real=B.cqs_fwd(B.MODS["v11"], qs, ks, vs, real, causal),
                          f_zero=B.cqs_fwd(B.MODS["v11"], qs, ks, vs, zero_by_L[L], causal),
                          f_plain=B.plain_fwd(B.MODS["v11"], qs, ks, vs, causal)))
    routes = sorted({B.census(B.MODS["v12b"], it["real"], causal, D)[0] for it in items})
    fa = B.fa_module()
    fa_items = [(it["x"],) + tuple(B.plain_fwd(fa, *it["x"][:3], causal)) for it in items]

    def loop(module, pattern, flags=None):
        def call():
            out = None
            for it in items:
                qs, ks, vs, dos = it["x"]
                o16, lse = it[f"f_{pattern}"]
                out = B.cqs_bwd(B.MODS[module], dos, qs, ks, vs, o16, lse, None if pattern == "plain" else it[pattern],
                                causal, flags=flags)
            return out
        return call

    def loop_fa():
        out = None
        for (qs, ks, vs, dos), o, l in fa_items:
            out = B.cqs_bwd(fa, dos, qs, ks, vs, o, l, None, causal)
        return out

    r = dict(gpu=GPU, part=f"bwd_small_c{c}_itr{itr}", N=N, c=c, itr=itr, tasks=len(items), of_tasks=len(tasks_of(N, c, itr)),
             L_min=min(Ls), L_max=Lmax, sum_L=sum(Ls), H=H, D=D, causal=causal, routes=routes)
    arms = {"flash_attn_direct": loop_fa, "v11x_nocqs": loop("v11x", "plain"), "v12b_plain": loop("v12b", "plain"),
            "v11_real": loop("v11", "real"), "v12b_real": loop("v12b", "real", 0), "v12b_zero": loop("v12b", "zero", 0),
            "v12b_legacy_real": loop("v12b", "real", B.FORCE_LEGACY)}
    for name, fn in arms.items():
        r[f"{name}_ms"], r[f"{name}_min_ms"] = timed(fn)
    for name in ("v12b_real", "v12b_zero", "v12b_plain", "v11_real"):
        r[f"{name}_kernel_us"] = kernel_us(arms[name])
    eq, dq = True, 0.0
    for it in items:
        qs, ks, vs, dos = it["x"]
        o16, lse = it["f_real"]
        a = B.cqs_bwd(B.MODS["v11"], dos, qs, ks, vs, o16, lse, it["real"], causal, fill=NAN)
        b = B.cqs_bwd(B.MODS["v12b"], dos, qs, ks, vs, o16, lse, it["real"], causal, flags=0, fill=NAN)
        eq &= bool(torch.equal(a[1], b[1]) and torch.equal(a[2], b[2]))
        dq = max(dq, B.maxdiff(a[0], b[0]))
    r["dk_dv_equal_vs_v11_all_tasks"], r["dq_maxdiff_vs_v11"] = eq, dq
    res["small"].append(r)
    print(f"[{r['part']}] {len(items)} tasks L {min(Ls)}-{Lmax}: " + "  ".join(f"{n} {r[n + '_ms']:.2f}" for n in arms)
          + f"  routes={routes} dk/dv equal={eq} dq {dq:.1e}  v12b_real kernels={r['v12b_real_kernel_us'].get('us')}", flush=True)
    del items, fa_items, q, k_, v, do
    torch.cuda.empty_cache()


# ============================================================================ report
def _f(x, nd=2):
    return "-" if x is None else f"{x:.{nd}f}"


def report(res, path_md):
    info = res["info"]
    rows = res["rows"]
    L = ["# F23 kernel A/B: review-fixed builds v12f b3 / v12b b5", "",
         f"Job **{info.get('slurm_job')}**, {info.get('node')}, **{info.get('gpu')}** (`{info.get('nvidia_smi')}`), torch "
         f"{info.get('torch')}, flash_attn {str(info.get('flash_attn', '?')).split()[0]}, fp16. {res['timing']}. Harness "
         f"`benchmarks/ultimate/kernels_f23.py`, raw rows `kernels_f23.json`. Modules: "
         + ", ".join(f"{k} = `{v}`" for k, v in info.get("modules", {}).items()) + ".", ""]

    def sel(part, **kw):
        return [r for r in rows if r["part"] == part and all(r.get(k) == v for k, v in kw.items())]

    def one(part, **kw):
        s = sel(part, **kw)
        return s[0] if s else None

    # forward task rows
    for part, title in (("fwd_headline", "Forward, causal D=64, c=7 aligned task (headline re-time, H=8 and H=32)"),
                        ("fwd_ragged", "Forward, causal D=64, N=100003 c=7: even vs ragged (Is_even_MN=false) task")):
        s = sel(part)
        if not s:
            continue
        L.append(f"## {title}")
        L.append("")
        L.append("| L | L%128 | H | arm | flags | ms (median) | min | / flash_attn | / v11 plain (in-tree) | == v11 | route | device us (census / aligned / legacy / memset) |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for (Lx, Hx) in sorted({(r["L"], r["H"]) for r in s}):
            grp = [r for r in s if r["L"] == Lx and r["H"] == Hx]
            p11 = next((r for r in grp if r["module"] == "v11" and r["pattern"] == "plain"), None)
            fa = grp[0]
            L.append(f"| {Lx} | {Lx % 128} | {Hx} | flash_attn | - | {fa['flash_attn_ms']:.2f} | {fa['flash_attn_min_ms']:.2f} | 1 | "
                     f"{fa['flash_attn_ms'] / p11['ms']:.3f} | - | - | - |")
            L.append(f"| {Lx} | {Lx % 128} | {Hx} | SDPA-flash | - | {fa['sdpa_flash_ms']:.2f} | {fa['sdpa_flash_min_ms']:.2f} | "
                     f"{fa['sdpa_flash_ms'] / fa['flash_attn_ms']:.3f} | {fa['sdpa_flash_ms'] / p11['ms']:.3f} | - | - | - |")
            for r in grp:
                ku = (r.get("kernel_us") or {}).get("us", {})
                kus = " / ".join(_f(ku.get(k), 1) for k in ("census", "aligned", "legacy", "memset")) if ku else "-"
                L.append(f"| {Lx} | {Lx % 128} | {Hx} | {r['module']} {r['pattern']}{' (again)' if r.get('again') else ''} | "
                         f"{r['flags']} | {r['ms']:.2f} | {r['min_ms']:.2f} | {r['ratio_vs_flash_attn']:.3f} | {r['ms'] / p11['ms']:.3f} | "
                         f"{r.get('equal_vs_v11', '-')} | {r.get('census_route', '-')} | {kus} |")
        L.append("")
    # derived forward lines
    for Hx in (8, 32):
        g = lambda m, p, f: next((r for r in sel("fwd_headline", H=Hx, module=m, pattern=p, flags=f)), None)
        p11, zd, za, rd, ra, r0, z0 = (g("v11", "plain", None), g("v12f", "zero", F.REVERSE_M), g("v12f", "zero", F.ASCENDING_M),
                                       g("v12f", "real", F.REVERSE_M), g("v12f", "real", F.ASCENDING_M), g("v12f", "real", 0),
                                       g("v12f", "zero", 0))
        if p11 and za and zd:
            L.append(f"- H={Hx}: all-live tax, same (ascending) order both sides: v12f zero asc / v11 plain = {za['ms'] / p11['ms']:.3f}; "
                     f"descending gain on its own: zero desc / zero asc = {zd['ms'] / za['ms']:.3f}, real desc / real asc = "
                     f"{rd['ms'] / ra['ms']:.3f}; default (flags 0) real / v11 plain (in-tree) = {r0['ms'] / p11['ms']:.3f}, "
                     f"/ flash_attn = {r0['ratio_vs_flash_attn']:.3f}; default zero / v11 plain = {z0['ms'] / p11['ms']:.3f}.")
    for fa in res.get("fa_again", []):
        first = next((r for r in rows if r["part"] == "fwd_headline" and r["H"] == fa["H"]), None)
        if first:
            L.append(f"- H={fa['H']}: FA arms re-timed at the end of the group: flash_attn {first['flash_attn_ms']:.2f} -> "
                     f"{fa['flash_attn_ms']:.2f} ms (min {fa['flash_attn_min_ms']:.2f}), SDPA-flash {first['sdpa_flash_ms']:.2f} -> "
                     f"{fa['sdpa_flash_ms']:.2f} ms (min {fa['sdpa_flash_min_ms']:.2f}).")
    ev = sel("fwd_ragged", module="v12f", pattern="real", flags=0)
    if len(ev) == 2:
        e, rg = sorted(ev, key=lambda r: r["L_mod_128"])
        ez, rz = (one("fwd_ragged", L=x["L"], module="v12f", pattern="zero", flags=0) for x in (e, rg))
        p_e, p_r = (one("fwd_ragged", L=x["L"], module="v11", pattern="plain") for x in (e, rg))
        L.append(f"- Ragged ALIGNED (Is_even_MN=false), L={rg['L']} vs even L={e['L']}: real / v11 plain {rg['ms'] / p_r['ms']:.3f} vs "
                 f"{e['ms'] / p_e['ms']:.3f}; zero / v11 plain {rz['ms'] / p_r['ms']:.3f} vs {ez['ms'] / p_e['ms']:.3f}; "
                 f"real / flash_attn {rg['ratio_vs_flash_attn']:.3f} vs {e['ratio_vs_flash_attn']:.3f}.")
    L.append("")
    # small-L parts
    for r in res.get("small", []):
        L.append(f"## {r['part']}: {r['tasks']} of {r['of_tasks']} tasks (L {r['L_min']}-{r['L_max']}, sum L {r['sum_L']}), "
                 f"causal D={r['D']} H={r['H']}, one timed region per arm")
        L.append("")
        arms = [k[:-3] for k in r if k.endswith("_ms") and not k.endswith("_min_ms")]
        base = r.get("flash_attn_direct_ms")
        ref = r.get("v11_plain_ms") or r.get("v11x_nocqs_ms")
        L.append("| arm | ms | min | / flash_attn loop | / in-tree plain loop | device us: census / aligned / legacy (= flash_fwd_kernel or the dq_dk_dv kernel, also the CQS-off forward) / memset / other (launches) |")
        L.append("|---|---|---|---|---|---|")
        for a in arms:
            ku = r.get(f"{a}_kernel_us") or {}
            us, n = ku.get("us", {}), ku.get("launches", {})
            kus = " / ".join(f"{_f(us.get(k), 0)} ({n.get(k, 0)})" for k in ("census", "aligned", "legacy", "memset", "other")) if us else "-"
            L.append(f"| {a} | {r[a + '_ms']:.2f} | {r[a + '_min_ms']:.2f} | {r[a + '_ms'] / base:.3f} | {r[a + '_ms'] / ref:.3f} | {kus} |")
        L.append("")
        L.append(f"Routes {r['routes']}; identity on every task: "
                 + (f"v12f real == v11 {r.get('v12f_real_equal_v11_all_tasks')}" if "v12f_real_equal_v11_all_tasks" in r
                    else f"v12b dk/dv == v11 {r.get('dk_dv_equal_vs_v11_all_tasks')}, dq max |d| {r.get('dq_maxdiff_vs_v11', 0):.1e}") + ".")
        L.append("")
    # backward task rows
    for part, title in (("bwd_headline", "Backward, causal D=64, c=7 aligned task L=56064 (headline re-time)"),
                        ("bwd_ragged", "Backward, causal D=64, N=100003 c=7: even vs ragged task"),
                        ("bwd_legacy_unaligned", "Backward, AUTOMATIC LEGACY route (unaligned seg_align=0 c=7 tasks)"),
                        ("bwd_legacy_aligned", "Backward, forced LEGACY vs v11 on the aligned N=32768 task")):
        s = sel(part)
        if not s:
            continue
        L.append(f"## {title}")
        L.append("")
        L.append("| L | L%128 | arm | ms (median) | min | / flash_attn (autograd) | / flash_attn direct | / v11 real (first) | dk/dv == v11 | dq max vs v11 | route | device us (census / aligned / legacy) |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for Lx in sorted({r["L"] for r in s}):
            grp = [r for r in s if r["L"] == Lx]
            v11r = next((r for r in grp if r["arm"] == "v11_real"), None)
            if "flash_attn_bwd_ms" in grp[0]:
                fa = grp[0]
                L.append(f"| {Lx} | {Lx % 128} | flash_attn autograd / direct / SDPA | {fa['flash_attn_bwd_ms']:.2f} / "
                         f"{fa['flash_attn_direct_bwd_ms']:.2f} / {fa['sdpa_flash_bwd_ms']:.2f} | {fa['flash_attn_bwd_min_ms']:.2f} / "
                         f"{fa['flash_attn_direct_bwd_min_ms']:.2f} / {fa['sdpa_flash_bwd_min_ms']:.2f} | - | - | - | - | - | - | - |")
            for r in grp:
                ku = (r.get("kernel_us") or {}).get("us", {})
                kus = " / ".join(_f(ku.get(k), 1) for k in ("census", "aligned", "legacy")) if ku else "-"
                fa_r = f"{r['ms'] / r['flash_attn_bwd_ms']:.3f}" if "flash_attn_bwd_ms" in r else "-"
                fd_r = f"{r['ms'] / r['flash_attn_direct_bwd_ms']:.3f}" if "flash_attn_direct_bwd_ms" in r else "-"
                L.append(f"| {Lx} | {Lx % 128} | {r['arm']} | {r['ms']:.2f} | {r['min_ms']:.2f} | {fa_r} | {fd_r} | "
                         f"{r['ms'] / v11r['ms']:.3f} | {r.get('dk_dv_equal_vs_v11', '-')} | "
                         f"{('%.1e' % r['dq_maxdiff_vs_v11']) if 'dq_maxdiff_vs_v11' in r else '-'} | {r.get('census_route', '-')} | {kus} |"
                         if v11r else f"| {Lx} | | {r['arm']} | {r['ms']:.2f} | | | | | | | | |")
        L.append("")
    d = res.get("deterministic")
    if d:
        L.append(f"## deterministic=True, L={d['L']} causal real D=64, H={d['H']}, b=1, {d['sm_count']} SMs")
        L.append("")
        L.append(f"atomics {d['atomics_ms']:.2f} ms (peak extra {d['atomics_peak_extra_MiB']:.0f} MiB), deterministic "
                 f"{d['deterministic_ms']:.2f} ms ({d['deterministic_ms'] / d['atomics_ms']:.3f}x, peak extra "
                 f"{d['deterministic_peak_extra_MiB']:.0f} MiB).")
        L.append("")
    errs = {k: v.splitlines()[-1] for k, v in res.items() if k.startswith("error_")}
    if errs:
        L.append("## Errors")
        L.append("")
        for k, v in errs.items():
            L.append(f"- {k}: `{v}`")
        L.append("")
    open(path_md, "w").write("\n".join(L) + "\n")
    return path_md


# ============================================================================ main
def main():
    global ITERS, WARM, FA_ITERS, FA_WARM
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--report", default=None)
    ap.add_argument("--only", default=None, help="comma list: fwd_headline,fwd_ragged,fwd_small,bwd_headline,bwd_ragged,bwd_legacy,bwd_small")
    a = ap.parse_args()
    if a.report:
        res = json.load(open(a.report))
        print(open(report(res, a.report.replace(".json", ".md"))).read())
        return
    F.load_modules(("v11", "v11x", "v12f"))
    B.load_modules(("v11", "v11x", "v12b"))
    N56, N100, N_legacy, N_small = 131072, 100003, [32768, 131072], 131072
    k73, k_small = 73, 64
    H_list = (8, 32)
    if a.smoke:
        ITERS, WARM, FA_ITERS, FA_WARM = 2, 1, 2, 1
        N56, N100, N_legacy, N_small, k73, k_small, H_list = 16384, 20011, [8192, 16384], 16384, 8, 8, (8, 16)
        a.out = a.out.replace(".json", "_smoke.json")
    t0 = time.time()
    res = dict(info=F.gpu_info(), rows=[], small=[], smoke=a.smoke,
               timing=f"our arms {WARM} warm-up + {ITERS} timed runs, FA arms {FA_WARM} + {FA_ITERS}; CUDA events, median (min stored)")
    res["info"]["modules"] = {k: getattr(m, "__file__", k) for k, m in {**F.MODS, **B.MODS}.items()}
    print(json.dumps(res["info"]), flush=True)
    fa_fwd_arms(4096, 64, True, 8); fa_bwd_arms(4096, 64, True)       # first-use init outside every timed row
    gpu_spin(5.0 if not a.smoke else 0.5)                               # clocks up before the first timed row
    parts = {
        "fwd_headline": lambda: part_fwd_headline(res, N56, H_list),
        "fwd_ragged": lambda: part_fwd_ragged(res, N100),
        "fwd_small": lambda: part_fwd_small(res, N_small, 73, 2, k73),
        "bwd_headline": lambda: part_bwd_headline(res, N56),
        "bwd_ragged": lambda: part_bwd_ragged(res, N100),
        "bwd_legacy": lambda: part_bwd_legacy(res, N_legacy),
        "bwd_small": lambda: (part_bwd_small(res, N_small, 13, 3, k_small), part_bwd_small(res, N_small, 31, 2, k_small)),
    }
    only = a.only.split(",") if a.only else list(parts)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    for name in only:
        try:
            parts[name]()
        except Exception as e:  # noqa
            res[f"error_{name}"] = traceback.format_exc()
            print(f"!! {name} failed: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
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
