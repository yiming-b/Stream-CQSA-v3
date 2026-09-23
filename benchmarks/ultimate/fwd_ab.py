"""
P3 forward-kernel A/B: v12f (ALIGNED run-table forward + LEGACY v11 fallback) against v11, v11x, v9,
flash_attn and SDPA-flash, ONE process on ONE GPU.

v12f = next/kernel/v12f (module cqsa_cuda_next_v12f): a census kernel classifies every tile the forward
visits from the 64-token OR/AND summaries; when none is mixed (every 128-aligned task) the ALIGNED
instantiation -- upstream's KV loop driven by a per-row-block run table of live tiles, no verdict loads,
no per-element mask path, one Check_inf softmax -- computes the call, otherwise LEGACY (v11's CQS forward,
unchanged). Both launch on the stream; the losing one's CTAs return at once (no host sync).

Rows (docs/ultimate/01_design_a100.md 7.1 P3.1; package P3 of the ultimate program):
  aligned c=7 task, N=131072 (L=56064): causal and non-causal x D in {64, 128};
  aligned c=7 task, N=524288 (L~225K): causal D=64;
  c=73, N=131072 (73 tasks, L~16K): per-task times summed, causal D=64.
Arms: flash_attn (the FA-2 package), SDPA-flash (torch SDPA, FLASH_ATTENTION backend; the arm of every
earlier results/ultimate row), v11 plain/zero/real, v11x plain/zero (compiled-out element path), v9
plain/zero/real (non-causal: today's production non-causal module), v12f plain/zero/real and v12f
zero/real with reversed row-block order. plain = CQS off (the extension's `fwd`), zero = CQS on with
all-zero bits (the all-live tax), real = the task's own bits.
Identity (in-job): v12f's out (fp32), lse and fp16 out equal v11's on every pattern (torch.equal), the
census routes every aligned pattern to ALIGNED (route 0).
Timing: 1 warm-up + 5 timed runs, CUDA events, median (min also stored); GPU name in every row.
Output: results/ultimate/fwd_ab.json (+ fwd_ab.md written by --report).

    sbatch slurm/ultimate/fwd_ab.slurm                  # A100-SXM4-80GB, one gpu-test job
    python -P benchmarks/ultimate/fwd_ab.py --smoke     # correctness only, shared GPU, < 2 GiB
    python -P benchmarks/ultimate/fwd_ab.py --report results/ultimate/fwd_ab.json
"""
import argparse, importlib, json, os, socket, statistics, subprocess, time, traceback

os.environ.setdefault("CQSA_CPU_THREADS", "4")

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend

torch.set_num_threads(int(os.environ["CQSA_CPU_THREADS"]))

from stream_cqsa.autoconfig import QUORUM_SETS
from stream_cqsa.interface import cqs_block_summaries, CQS_BLK_SIZE
from stream_cqsa.stable_stream import build_tasks

MODULES = {"v11": "cqsa_cuda_next_v11", "v11x": "cqsa_cuda_next_v11x", "v9": "cqsa_cuda_next_v9",
           "v12f": "cqsa_cuda_next_v12f"}
MODS = {}
FORCE_LEGACY, REVERSE_M = 1, 2        # cqs_fwd_flags of v12f's fwd_cqs_group_bits
RUN_SLOTS = 16                        # kCqsRunSlots (cqs_census.h)
ALIGNED_TILE = {64: (128, 128), 128: (128, 64)}   # CqsAlignedTile (cqs_census.h)

H = 8
DT = torch.float16
DEV = torch.device("cuda")
GPU = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no GPU"     # the CPU tests import this module
SEG_ALIGN = 128
OUT_DIR = "/scratch/gpfs/AKEY/yb2807/Stream-CQSA-ultimate/results/ultimate"


def load_modules(names=tuple(MODULES)):
    """Import the kernel extensions (next/kernel/{stable,v11x,v12f} on PYTHONPATH); fails loudly on a missing one."""
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
    """1 warm-up + `iters` timed runs with CUDA events; returns (median_ms, min_ms)."""
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


def tables(bits):
    """(bits, blk_or, blk_and) on the device for a per-token bits vector."""
    bo, ba = (to_dev(x) for x in cqs_block_summaries(bits, CQS_BLK_SIZE))
    return to_dev(bits), bo, ba


def task_tables(task):
    return to_dev(task.group_bits), to_dev(task.extra["blk_or"]), to_dev(task.extra["blk_and"])


def plain_fwd(mod, q, k, v, causal):
    """CQS off: the extension's `fwd`. Returns (out16, lse)."""
    out, lse, _, _ = mod.fwd(q, k, v, None, None, 0.0, q.shape[-1] ** -0.5, bool(causal), -1, -1, 0.0, False, None)
    return out, lse


def cqs_fwd(mod, q, k, v, tabs, causal, flags=None, block_base=None):
    """CQS on, as interface.flash_attn_func_cqs_group_bits calls it (fp32 out, contiguous outputs), with v12f's
    cqs_fwd_flags when given. q/k/v may be strided in-place views when block_base (seg_align 128) is set.
    Returns (out32, lse, out16)."""
    bits, bo, ba = tabs
    shape = tuple(q.shape)
    out = torch.empty(shape, device=q.device, dtype=q.dtype)
    acc = torch.empty(shape, device=q.device, dtype=torch.float32)
    args = [q, k, v, out, None, 0.0, q.shape[-1] ** -0.5, bool(causal), -1, -1, 0.0, False, None, bits, bo, ba,
            CQS_BLK_SIZE, acc, block_base, SEG_ALIGN if block_base is not None else None]
    if flags is not None:
        args.append(int(flags))
    out16, lse, _, _ = mod.fwd_cqs_group_bits(*args)
    return acc, lse, out16


def census(mod, tabs, causal, D):
    """v12f's census on the device: (route int, runs [M, RUN_SLOTS, 2] numpy, (block_m, block_n))."""
    route, runs, tile = mod.fwd_cqs_census(*tabs, bool(causal), int(D))
    return int(route.item()), runs.cpu().numpy(), tuple(int(x) for x in tile)


def census_ref(bits, block_m, block_n, causal, slots=RUN_SLOTS):
    """Reference of v12f's census (cqs_census.cu) from the per-token bits: a tile is live when no row and column
    share a group bit (row OR & col OR == 0), dead when all do (row AND & col AND != 0), mixed otherwise;
    visited tiles are [0, n_block_max) with the causal limit; live tiles are grouped into descending runs
    {hi, lo}, at most slots-1 per row block, closed by {-1, -1}. Returns (route, [list of runs per row block])."""
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
    route, rows = 0, []
    for m in range(num_m):
        n_max = min(num_n, -(-((m + 1) * block_m) // block_n)) if causal else num_n
        live = (r_or[m] & c_or[:n_max]) == 0
        if ((~live) & ((r_and[m] & c_and[:n_max]) == 0)).any():
            route |= 1
        runs = []
        for n in range(n_max - 1, -1, -1):
            if not live[n]:
                continue
            if runs and runs[-1][1] == n + 1:
                runs[-1][1] = n
            else:
                runs.append([n, n])
        if len(runs) > slots - 1:
            route |= 2
            runs = runs[:slots - 1]
        rows.append(runs)
    return route, rows


def runs_prefix(runs_row):
    """The runs of one row block of the device table, up to the {-1, -1} sentinel."""
    out = []
    for hi, lo in runs_row:
        if hi < 0:
            return out
        out.append([int(hi), int(lo)])
    raise AssertionError("run table row without a sentinel")


def c_tasks(N, c, itr=1, seg_align=SEG_ALIGN, D=64):
    return build_tasks(N, itr, B=1, H=H, D=D, itemsize=2, sorted_gather=True, pin=False,
                       c=c, interest_set=QUORUM_SETS[c], seg_align=seg_align)


def gathered_qkv(task, N, D, seed=0):
    """Random [1,N,H,D] Q/K/V, gathered to the task's tokens (the engine's packed layout)."""
    torch.manual_seed(seed)
    idx = task.token_ids.to(DEV)
    out = []
    for _ in range(3):
        x = torch.randn(1, N, H, D, device=DEV, dtype=DT)
        out.append(x.index_select(1, idx).contiguous())
        del x
    return out


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


def _kernel_kind(name):
    if "census" in name.lower():
        return "census"
    if "flash_fwd_aligned_kernel" in name:
        return "aligned"
    if "flash_fwd_kernel" in name:
        return "legacy"
    return "other"


def kernel_times(fn):
    """Device time per kernel kind (census / aligned / legacy / other, us) of ONE call of fn, from the profiler.
    v12f launches census -> ALIGNED -> LEGACY on every CQS call; the losing kernel's CTAs return at once, so this
    separates the census's cost (it is also inside every timed v12f row) from the kernel that did the work."""
    try:
        from torch.profiler import profile, ProfilerActivity
        fn(); torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            fn(); torch.cuda.synchronize()
        out = {}
        for e in prof.events():
            if e.device_type.name == "CUDA":
                k = _kernel_kind(e.name)
                out[k] = out.get(k, 0.0) + float(e.time_range.elapsed_us())
        return out
    except Exception as e:  # noqa
        return {"error": f"{type(e).__name__}: {e}"}


def fa2_arms(L, D, causal, iters):
    """flash_attn (the package) on [1,L,H,D] and SDPA-flash on [1,H,L,D]: dict of ms."""
    from flash_attn import flash_attn_func
    torch.manual_seed(1)
    r = {}
    q, k, v = (torch.randn(1, L, H, D, device=DEV, dtype=DT) for _ in range(3))
    fn = lambda: flash_attn_func(q, k, v, causal=causal, softmax_scale=D ** -0.5)
    r["flash_attn_ms"], r["flash_attn_min_ms"] = timed(fn, iters)
    kernel_names("flash_attn", fn)
    del q, k, v
    q, k, v = (torch.randn(1, H, L, D, device=DEV, dtype=DT) for _ in range(3))
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        fn = lambda: F.scaled_dot_product_attention(q, k, v, is_causal=causal, scale=D ** -0.5)
        r["sdpa_flash_ms"], r["sdpa_flash_min_ms"] = timed(fn, iters)
        kernel_names("sdpa_flash", fn)
    return r


def arms_for(causal):
    """(module, pattern, flags) in timing order; v11 first so its outputs are the identity reference."""
    arms = [("v11", "plain", None), ("v11", "zero", None), ("v11", "real", None),
            ("v11x", "plain", None), ("v11x", "zero", None)]
    if not causal:
        arms += [("v9", "plain", None), ("v9", "zero", None), ("v9", "real", None)]
    arms += [("v12f", "plain", None), ("v12f", "zero", 0), ("v12f", "real", 0),
             ("v12f", "zero", REVERSE_M), ("v12f", "real", REVERSE_M)]
    return arms


def run_arm(module, pattern, flags, q, k, v, pats, causal):
    """One call of an arm: (out, lse, out16 or None)."""
    mod = MODS[module]
    if pattern == "plain":
        o16, lse = plain_fwd(mod, q, k, v, causal)
        return o16, lse, None
    return cqs_fwd(mod, q, k, v, pats[pattern], causal, flags=flags)


def equal_outputs(a, b):
    return all(torch.equal(x, y) for x, y in zip(a, b) if x is not None and y is not None)


# ============================================================================ single-task rows
def part_task(res, N, configs, iters, key):
    """Rows for the c=7 min-L aligned task at N, one per (config, arm)."""
    task = min(c_tasks(N, 7), key=lambda t: int(t.local_size))
    L = int(task.local_size)
    rows = res.setdefault(key, [])
    for causal, D in configs:
        q, k, v = gathered_qkv(task, N, D)
        zero = torch.zeros(L, dtype=torch.int64)
        pats = {"real": task_tables(task), "zero": tables(zero)}
        fa = fa2_arms(L, D, causal, iters)
        routes = {p: census(MODS["v12f"], pats[p], causal, D)[0] for p in pats}
        ref = {}
        for module, pattern, flags in arms_for(causal):
            row = dict(gpu=GPU, N=N, L=L, L_mod_128=L % 128, path=list(task.path), causal=causal, D=D, module=module,
                       pattern=pattern, flags=flags, **fa)
            if module == "v12f" and pattern != "plain":
                row["census_route"] = routes[pattern]
            outs = run_arm(module, pattern, flags, q, k, v, pats, causal)
            row["ms"], row["min_ms"] = timed(lambda: run_arm(module, pattern, flags, q, k, v, pats, causal), iters)
            row["ratio_vs_flash_attn"] = row["ms"] / fa["flash_attn_ms"]
            row["ratio_vs_sdpa"] = row["ms"] / fa["sdpa_flash_ms"]
            if module == "v12f" and pattern != "plain":
                row["kernel_us"] = kernel_times(lambda: run_arm(module, pattern, flags, q, k, v, pats, causal))
            if module == "v11":
                ref[pattern] = outs
            elif module in ("v12f", "v11x", "v9"):
                row["equal_vs_v11"] = equal_outputs(outs, ref[pattern])
                row["maxdiff_vs_v11"] = (outs[0].float() - ref[pattern][0].float()).abs().max().item()
            rows.append(row)
            print(f"[{key}] causal={int(causal)} D={D:3d} L={L} {module:4s} {pattern:5s} flags={flags} {row['ms']:8.2f} ms"
                  f"  /fa {row['ratio_vs_flash_attn']:.3f} /sdpa {row['ratio_vs_sdpa']:.3f}"
                  + (f"  equal_v11={row['equal_vs_v11']}" if "equal_vs_v11" in row else "")
                  + (f"  route={row['census_route']}" if "census_route" in row else "")
                  + (f"  kernels(us)={ {k: round(v, 1) for k, v in row['kernel_us'].items()} }" if "kernel_us" in row else ""), flush=True)
            del outs
        del q, k, v, ref, pats
        torch.cuda.empty_cache()


# ============================================================================ c=73 summed over tasks
def part_c73(res, N, iters, D=64):
    """c=73, N=131072, causal D=64: every task's median summed per arm; flash_attn/SDPA per task at its L."""
    tasks = c_tasks(N, 73)
    arms = [("v11", "plain", None), ("v11", "real", None), ("v12f", "plain", None), ("v12f", "real", 0),
            ("v12f", "real", REVERSE_M)]
    tot = {f"{m}_{p}" + ("" if f in (None, 0) else "_rev"): 0.0 for m, p, f in arms}
    tot.update(flash_attn=0.0, sdpa_flash=0.0)
    equal, routes, Ls, ktot = True, set(), [], {}
    for t in tasks:
        L = int(t.local_size)
        Ls.append(L)
        q, k, v = gathered_qkv(t, N, D, seed=t.path_idx)
        pats = {"real": task_tables(t)}
        fa = fa2_arms(L, D, True, iters)
        tot["flash_attn"] += fa["flash_attn_ms"]
        tot["sdpa_flash"] += fa["sdpa_flash_ms"]
        routes.add(census(MODS["v12f"], pats["real"], True, D)[0])
        ref = None
        for module, pattern, flags in arms:
            name = f"{module}_{pattern}" + ("" if flags in (None, 0) else "_rev")
            outs = run_arm(module, pattern, flags, q, k, v, pats, True)
            tot[name] += timed(lambda: run_arm(module, pattern, flags, q, k, v, pats, True), iters)[0]
            if (module, pattern, flags) == ("v12f", "real", 0):
                for kk, vv in kernel_times(lambda: run_arm(module, pattern, flags, q, k, v, pats, True)).items():
                    if isinstance(vv, float):
                        ktot[kk] = ktot.get(kk, 0.0) + vv
            if (module, pattern) == ("v11", "real"):
                ref = outs
            elif pattern == "real":
                equal &= equal_outputs(outs, ref)
            del outs
        del q, k, v
    mono = fa2_arms(N, D, True, iters)
    res["c73"] = dict(gpu=GPU, N=N, c=73, n_tasks=len(tasks), L_min=min(Ls), L_max=max(Ls), causal=True, D=D,
                      sum_ms=tot, census_routes=sorted(routes), v12f_real_equal_v11_all_tasks=equal,
                      v12f_real_kernel_us_sum=ktot,
                      monolithic_flash_attn_ms=mono["flash_attn_ms"], monolithic_sdpa_flash_ms=mono["sdpa_flash_ms"])
    print(f"[c73] N={N} tasks={len(tasks)} L {min(Ls)}-{max(Ls)} sums (ms): "
          + "  ".join(f"{k} {v:.1f}" for k, v in tot.items())
          + f"  | monolithic flash_attn {mono['flash_attn_ms']:.1f}  routes={sorted(routes)} equal={equal}"
          + f"  v12f real kernel sums (us) { {k: round(v, 1) for k, v in ktot.items()} }", flush=True)


# ============================================================================ report
def _row(rows, **sel):
    for r in rows:
        if all(r.get(k) == v for k, v in sel.items()):
            return r
    return None


def report(res):
    """Markdown summary of fwd_ab.json: the task rows, the c=73 sums and the package targets."""
    info = res["info"]
    lines = [f"# Forward A/B (P3): v12f vs v11 / v11x / v9 / flash_attn / SDPA-flash", "",
             f"Job **{info.get('slurm_job')}**, {info.get('node')}, **{info.get('gpu')}**, torch {info.get('torch')}, "
             f"flash_attn {info.get('flash_attn', '?').split()[0]}, fp16, H={res['H']}, {res['timing']}. "
             f"Harness `benchmarks/ultimate/fwd_ab.py`, raw rows `fwd_ab.json`. Kernels: flash_attn "
             f"{res.get('kernel_names', {}).get('flash_attn')}; SDPA-flash {res.get('kernel_names', {}).get('sdpa_flash')}.", ""]
    for key in ("task56k", "task225k"):
        rows = res.get(key, [])
        if not rows:
            continue
        lines += [f"## aligned c=7 task, N={rows[0]['N']} (L={rows[0]['L']}), ms (ratio vs flash_attn / vs SDPA-flash)", "",
                  "v12f kernel columns: device us of one profiled call (census / ALIGNED / LEGACY; the losing kernel's CTAs "
                  "return at once). The census is inside every timed v12f CQS row.", "",
                  "| causal | D | flash_attn | SDPA | module | pattern | flags | ms | /flash_attn | /SDPA | equal v11 | route "
                  "| census us | ALIGNED us | LEGACY us |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in rows:
            ku = r.get("kernel_us", {})
            kc = " | ".join(f"{ku[k]:.1f}" if isinstance(ku.get(k), float) else "-" for k in ("census", "aligned", "legacy"))
            lines.append(f"| {int(r['causal'])} | {r['D']} | {r['flash_attn_ms']:.2f} | {r['sdpa_flash_ms']:.2f} | {r['module']} | "
                         f"{r['pattern']} | {r['flags'] if r['flags'] is not None else '-'} | {r['ms']:.2f} | "
                         f"{r['ratio_vs_flash_attn']:.3f} | {r['ratio_vs_sdpa']:.3f} | {r.get('equal_vs_v11', '-')} | "
                         f"{r.get('census_route', '-')} | {kc} |")
        lines.append("")
    c = res.get("c73")
    if c:
        s = c["sum_ms"]
        lines += [f"## c=73, N={c['N']}: {c['n_tasks']} tasks, L {c['L_min']}-{c['L_max']}, causal D={c['D']}, per-task medians summed", "",
                  "| arm | sum ms | /flash_attn sum | /monolithic flash_attn |", "|---|---|---|---|"]
        for k, v in s.items():
            lines.append(f"| {k} | {v:.1f} | {v / s['flash_attn']:.3f} | {v / c['monolithic_flash_attn_ms']:.3f} |")
        lines += ["", f"Monolithic flash_attn at N={c['N']}: {c['monolithic_flash_attn_ms']:.1f} ms (SDPA-flash "
                      f"{c['monolithic_sdpa_flash_ms']:.1f}). Census routes over the tasks: {c['census_routes']}; v12f real "
                      f"equal to v11 on every task: {c['v12f_real_equal_v11_all_tasks']}. v12f real (flags 0) device time "
                      f"summed over the tasks by kernel (us): {c.get('v12f_real_kernel_us_sum')}.", ""]
    lines += ["## Targets", ""]
    rows = res.get("task56k", [])
    for causal, D in ((True, 64), (True, 128), (False, 64), (False, 128)):
        z = _row(rows, causal=causal, D=D, module="v12f", pattern="zero", flags=0)
        p11 = _row(rows, causal=causal, D=D, module="v11", pattern="plain")
        p12 = _row(rows, causal=causal, D=D, module="v12f", pattern="plain")
        rl = _row(rows, causal=causal, D=D, module="v12f", pattern="real", flags=0)
        if z and p11:
            lines.append(f"- causal={int(causal)} D={D}: v12f zero / v11 plain = **{z['ms'] / p11['ms']:.3f}** "
                         f"(target <= 1.05); v12f zero / v12f plain = {z['ms'] / p12['ms']:.3f}; v12f real / flash_attn = "
                         f"**{rl['ratio_vs_flash_attn']:.3f}**" + (" (target <= 0.82)" if causal and D == 64 else ""))
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="small N, 2 iterations: correctness on a shared GPU")
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--out", default=os.environ.get("FWD_AB_OUT", f"{OUT_DIR}/fwd_ab.json"))
    ap.add_argument("--report", metavar="JSON", help="only write the markdown report of an existing JSON next to it")
    a = ap.parse_args()
    if a.report:
        res = json.load(open(a.report))
        md = a.report.replace(".json", ".md")
        open(md, "w").write(report(res))
        print(f"wrote {md}")
        return
    load_modules()
    n56, n225, n73, iters = 131072, 524288, 131072, a.iters
    if a.smoke:
        n56, n225, n73, iters = 16384, 32768, 16384 * 4, 2
        a.out = a.out.replace(".json", "_smoke.json")
    t0 = time.time()
    res = dict(info=gpu_info(), H=H, dtype="float16", iters=iters, smoke=a.smoke,
               timing=f"1 warm-up + {iters} timed runs, CUDA events, median (min also stored)",
               fa2_arms="flash_attn.flash_attn_func (package) and torch SDPA with SDPBackend.FLASH_ATTENTION forced",
               v12f="census + ALIGNED run-table forward, LEGACY = v11's CQS forward; built with CQSA_BWD_NO_CQS")
    print(json.dumps(res["info"]), flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    parts = ((part_task, (n56, ((True, 64), (True, 128), (False, 64), (False, 128)), iters, "task56k")),
             (part_task, (n225, ((True, 64),), iters, "task225k")),
             (part_c73, (n73, iters)))
    for part, args in parts:
        try:
            part(res, *args)
        except Exception as e:  # noqa
            res.setdefault("errors", []).append(traceback.format_exc())
            print(f"!! {part.__name__}{args[-1:]} failed: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
        res["kernel_names"] = _KERNEL_NAMES
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
