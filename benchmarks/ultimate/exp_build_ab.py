"""
Phase 1 experiment-build A/B: v11 vs v11x vs v9 vs SDPA-flash, ONE process on ONE GPU.

v11x = v11 built with -DCQSA_NO_ELEMENT_PATH (forward per-element mask path compiled
out; valid for the plain and all-zero patterns only, a real pattern comes back as NaN)
and -DCQSA_BWD_NO_CQS (backward CQS compiled out: upstream's loop with our contract).
Kernel trees: next/kernel/v11 and next/kernel/v11x (module names cqsa_cuda_next_*).

Answers (docs/ultimate/00_map.md gaps 2-3, judge synthesis Phase 1):
  forward   how much of the 1.21-1.27x all-live tax is the per-element path's PRESENCE
            (v11 zero vs v11x zero) versus the live loop itself (v11x zero vs plain vs
            SDPA-flash); causal, non-causal (v9 vs v11x) and hdim128, aligned task L=56064.
  backward  the route rule: fix the fused kernel iff the NO_CQS backward (v11x) is
            <= 1.25x SDPA-flash's backward; v11 CQS backward plain/zero/real alongside,
            at L in {14044, 28088, 56175} (classic layout, path (0,)).
  identity  v11x vs v11 on the patterns where v11x is valid (out, lse, dk, dv exactly
            equal; dq within the atomics' drift), and the NaN poison on the real pattern.

FA-2 arm: flash_attn is NOT installed, so it is torch SDPA with the FLASH_ATTENTION
backend forced (~5% slower than the flash_attn package, results/kernel_ab_clean_v11.json).
Timing: 1 warm-up + 5 timed runs, CUDA events, median (min also stored). The GPU name is
stored in every row. Output: results/ultimate/exp_build_ab.json + a printed table.

    sbatch slurm/ultimate/exp_build_ab.slurm             # A100-SXM4-80GB, 24G, one gpu-test job
    python -P benchmarks/ultimate/exp_build_ab.py --smoke  # correctness only, shared GPU, < 2 GiB
"""
import argparse, functools, importlib, json, os, socket, statistics, subprocess, time, traceback

os.environ.setdefault("CQSA_CUDA_MODULE", "cqsa_cuda_next_v11")
os.environ.setdefault("CQSA_CUDA_MODULE_NONCAUSAL", "cqsa_cuda_next_v9")
os.environ["CQSA_BACKWARD"] = "cuda"          # the CUDA backward, never Triton
os.environ.setdefault("CQSA_CPU_THREADS", "4")

import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend

torch.set_num_threads(int(os.environ["CQSA_CPU_THREADS"]))

from stream_cqsa import interface
from stream_cqsa.interface import (flash_attn_func, flash_attn_func_cqs_group_bits, flash_attn_bwd_cqs_global_lse,
                                   cqs_block_summaries, CQS_BLK_SIZE)
from stream_cqsa.stable_stream import build_tasks

MODULES = {"v11": "cqsa_cuda_next_v11", "v11x": "cqsa_cuda_next_v11x", "v9": "cqsa_cuda_next_v9"}
MODS = {name: importlib.import_module(mod) for name, mod in MODULES.items()}   # all three, or fail loudly

H = 8
DT = torch.float16
DEV = torch.device("cuda")
GPU = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no GPU"     # the CPU tests import this module
SEG_ALIGN = 128
FWD_CONFIGS = ((True, 64), (False, 64), (True, 128), (False, 128))   # (causal, D)


class use_module:
    """Route the interface's forward/backward calls to one extension module."""
    def __init__(self, name):
        self.mod = MODS[name]

    def __enter__(self):
        self.saved = (interface.cqsa_cuda, interface.cqsa_cuda_noncausal)
        interface.cqsa_cuda = interface.cqsa_cuda_noncausal = self.mod

    def __exit__(self, *exc):
        interface.cqsa_cuda, interface.cqsa_cuda_noncausal = self.saved


# ---------------------------------------------------------------------------- helpers
def gpu_info():
    info = dict(gpu=GPU, torch=torch.__version__, node=socket.gethostname(),
                slurm_job=os.environ.get("SLURM_JOB_ID"), modules={k: getattr(m, "__file__", k) for k, m in MODS.items()})
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


def maxdiff(a, b):
    return (a.float() - b.float()).abs().max().item()


def patterns_for(task):
    """(bits, blk_or, blk_and) for the real and the all-zero pattern of a task."""
    bits = to_dev(task.group_bits)
    zero = torch.zeros_like(bits)
    zo, za = (to_dev(x) for x in cqs_block_summaries(zero, CQS_BLK_SIZE))
    return {"real": (bits, to_dev(task.extra["blk_or"]), to_dev(task.extra["blk_and"])), "zero": (zero, zo, za)}


def poison_rows(bits, L, causal):
    """Rows v11x poisons on a 128-aligned pattern: those of every 128-row block whose masking-step tiles are not
    clear -- causal: the diagonal tile, plus the one before it when L % 128 != 0 (2 masking steps); non-causal:
    the last column tile. A fully masked tile met in the main loop is skipped by the search loop exactly as in
    v11 and needs no element path; a mixed tile there would poison too, but the aligned layout has none."""
    n_blk = (L + 127) // 128
    padded = torch.zeros(n_blk * 128, dtype=bits.dtype, device=bits.device)
    padded[:L] = bits
    blk_or = functools.reduce(torch.bitwise_or, padded.view(n_blk, 128).unbind(1))       # [n_blk]
    if causal:
        hit = blk_or != 0                          # rows and columns of the diagonal tile share the block's bits
        if L % 128:
            hit[1:] |= (blk_or[1:] & blk_or[:-1]) != 0
    else:
        hit = (blk_or & blk_or[-1]) != 0
    return hit.repeat_interleave(128)[:L]


def fwd(q, k, v, pattern, pat, causal):
    """Forward through the interface: plain -> `fwd` (CQS off), else the CQS entry with fp32 out. Returns (out, lse)."""
    scale = q.shape[-1] ** -0.5
    if pattern == "plain":
        return flash_attn_func(q, k, v, softmax_scale=scale, causal=causal, return_lse=True)
    bits, bo, ba = pat[pattern]
    out, lse, _ = flash_attn_func_cqs_group_bits(q, k, v, bits, softmax_scale=scale, causal=causal, cqs_blk_or=bo,
                                                 cqs_blk_and=ba, cqs_blk_size=CQS_BLK_SIZE, fp32_out=True, return_attn_probs=True)
    return out, lse


def bwd(mod, do, q, k, v, o16, lse, pattern, pat):
    """Causal backward with the global-lse contract. plain -> the extension's `bwd` with CQS off (the interface
    always passes bits, so this mirrors its call with the CQS arguments absent); zero/real -> the interface."""
    scale = q.shape[-1] ** -0.5
    if pattern == "plain":
        dq, dk, dv = (torch.empty_like(x) for x in (q, k, v))
        mod.bwd(do, q, k, v, o16, lse, dq, dk, dv, None, 0.0, float(scale), True, -1, -1, 0.0, False, None, None,
                None, None, None, None, None)
        return dq, dk, dv
    bits, bo, ba = pat[pattern]
    return flash_attn_bwd_cqs_global_lse(do, q, k, v, o16, lse, bits, softmax_scale=scale, causal=True,
                                         cqs_blk_or=bo, cqs_blk_and=ba, cqs_blk_size=CQS_BLK_SIZE)


_SDPA_NAMES = None


def sdpa_flash(L, D, causal, iters, backward=False):
    """FA-2 arm: torch SDPA with the FlashAttention backend forced, at length L. Returns dict of ms."""
    global _SDPA_NAMES
    torch.manual_seed(1)
    q, k, v = (torch.randn(1, H, L, D, device=DEV, dtype=DT, requires_grad=backward) for _ in range(3))
    scale = D ** -0.5
    r = {}
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        fn = lambda: F.scaled_dot_product_attention(q, k, v, is_causal=causal, scale=scale)
        r["fwd_ms"], r["fwd_min_ms"] = timed(fn, iters)
        if _SDPA_NAMES is None:
            try:
                from torch.profiler import profile, ProfilerActivity
                with profile(activities=[ProfilerActivity.CUDA]) as prof:
                    fn(); torch.cuda.synchronize()
                _SDPA_NAMES = sorted({e.name for e in prof.events() if "flash" in e.name.lower() or "attention" in e.name.lower()})
            except Exception as e:  # noqa
                _SDPA_NAMES = [f"profiler unavailable: {type(e).__name__}: {e}"]
        if backward:
            o = fn()
            do = torch.randn_like(o)
            r["bwd_ms"], r["bwd_min_ms"] = timed(lambda: torch.autograd.grad(o, (q, k, v), do, retain_graph=True), iters)
    return r


def aligned_task(N):
    """The c=7 itr=1 task at N with seg_align=128; prefers the one with L == 56064 at N=131072 (the map's row)."""
    tasks = build_tasks(N, 1, B=1, H=H, D=64, itemsize=2, sorted_gather=True, pin=False,
                        c=7, interest_set=(0, 1, 3), seg_align=SEG_ALIGN)
    return min(tasks, key=lambda t: int(t.local_size))


# ============================================================================ forward
def part_fwd(res, N, iters):
    res["fwd"] = []
    task = aligned_task(N)
    L = int(task.local_size)
    idx = task.token_ids.to(DEV)
    pat = patterns_for(task)
    for causal, D in FWD_CONFIGS:
        # v11x's real-pattern row is valid (timed, compared) only when nothing gets poisoned: with the last
        # chunk owned, the non-causal kernel meets no tile needing the element path
        pz = poison_rows(pat["real"][0], L, causal)
        torch.manual_seed(0)
        qt, kt, vt = (torch.randn(1, N, H, D, device=DEV, dtype=DT) for _ in range(3))
        q, k, v = (x.index_select(1, idx).contiguous() for x in (qt, kt, vt))
        del qt, kt, vt
        sdpa = sdpa_flash(L, D, causal, iters)
        ref = {}                                          # v11's outputs, so v11 goes first
        modules = ("v11", "v11x") if causal else ("v11", "v9", "v11x")
        for module in modules:
            for pattern in ("plain", "zero", "real"):
                row = dict(gpu=GPU, N=N, L=L, L_mod_128=L % 128, path=list(task.path), causal=causal, D=D,
                           module=module, pattern=pattern, valid=not (module == "v11x" and pattern == "real" and bool(pz.any())),
                           sdpa_flash_ms=sdpa["fwd_ms"], sdpa_flash_min_ms=sdpa["fwd_min_ms"])
                with use_module(module):
                    out, lse = fwd(q, k, v, pattern, pat, causal)
                    if row["valid"]:
                        row["ms"], row["min_ms"] = timed(lambda: fwd(q, k, v, pattern, pat, causal), iters)
                        row["ratio_vs_sdpa"] = row["ms"] / sdpa["fwd_ms"]
                if module == "v11":
                    ref[pattern] = (out, lse)
                elif row["valid"]:
                    row["maxdiff_out_vs_v11"] = maxdiff(out, ref[pattern][0])
                    row["maxdiff_lse_vs_v11"] = maxdiff(lse, ref[pattern][1])
                    row["equal_vs_v11"] = bool(torch.equal(out, ref[pattern][0]) and torch.equal(lse, ref[pattern][1]))
                else:
                    # the NaN poison: every row of a poisoned block is NaN in out, and in lse too unless the row is
                    # fully masked (row_sum == 0: a leading non-owner chunk under causal), where normalize_softmax_lse
                    # writes +inf in v11 and v11x alike; every other row is v11's
                    o, l = out[0], lse[0].T                              # [L,H,D], [L,H]
                    o_ref, l_ref = ref[pattern][0][0], ref[pattern][1][0].T
                    lp, lr = l[pz], l_ref[pz]
                    row["poison_rows_frac"] = pz.float().mean().item()
                    row["poison_out_nan"] = bool(o[pz].isnan().all())
                    row["poison_lse_ok"] = bool((lp.isnan() | (lr.isinf() & (lp == lr))).all())
                    row["poison_lse_inf_rows_frac"] = lr.isinf().float().mean().item()
                    row["clean_rows_equal_v11"] = bool(torch.equal(o[~pz], o_ref[~pz]) and torch.equal(l[~pz], l_ref[~pz]))
                    row["poison_ok"] = row["poison_out_nan"] and row["poison_lse_ok"] and row["clean_rows_equal_v11"]
                res["fwd"].append(row)
                print(f"[fwd] causal={int(causal)} D={D:3d} L={L} {module:4s} {pattern:5s} "
                      + (f"{row['ms']:8.2f} ms  /sdpa {row['ratio_vs_sdpa']:.3f}" if row["valid"] else "  (invalid: poison check)")
                      + (f"  equal_v11={row['equal_vs_v11']} maxdiff {row['maxdiff_out_vs_v11']:.2e}" if "equal_vs_v11" in row else "")
                      + (f"  poison_ok={row['poison_ok']} (rows {row['poison_rows_frac']*100:.1f}%)" if "poison_ok" in row else ""),
                      flush=True)
                del out, lse
        del q, k, v, ref
        torch.cuda.empty_cache()


# ============================================================================ backward
def part_bwd(res, Ns, iters):
    res["bwd"] = []
    D = 64
    for N in Ns:
        task = build_tasks(N, 1, B=1, H=H, D=D, itemsize=2, sorted_gather=True, pin=False,
                           c=7, interest_set=(0, 1, 3), seg_align=0)[0]
        L = int(task.local_size)
        pat = patterns_for(task)
        torch.manual_seed(0)
        q, k, v, do = (torch.randn(1, L, H, D, device=DEV, dtype=DT) for _ in range(4))
        # one (o16, lse) per pattern from v11's forward, shared by every backward arm
        fwd_out = {}
        with use_module("v11"):
            for pattern in ("plain", "zero", "real"):
                out, lse = fwd(q, k, v, pattern, pat, True)
                fwd_out[pattern] = (out.to(DT).contiguous(), lse.contiguous())
                del out
        sdpa = sdpa_flash(L, D, True, iters, backward=True)
        ref = {}
        for module in ("v11", "v11x"):
            mod = MODS[module]
            for pattern in ("plain", "zero", "real"):
                o16, lse = fwd_out[pattern]
                row = dict(gpu=GPU, N=N, L=L, L_mod_128=L % 128, path=list(task.path), causal=True, D=D, module=module,
                           pattern=pattern, valid=not (module == "v11x" and pattern == "real"),
                           sdpa_flash_bwd_ms=sdpa["bwd_ms"], sdpa_flash_bwd_min_ms=sdpa["bwd_min_ms"],
                           sdpa_flash_fwd_ms=sdpa["fwd_ms"])
                with use_module(module):
                    fn = lambda: bwd(mod, do, q, k, v, o16, lse, pattern, pat)
                    row["bwd_ms"], row["bwd_min_ms"] = timed(fn, iters)
                    row["bwd_ratio_vs_sdpa"] = row["bwd_ms"] / sdpa["bwd_ms"]
                    g1 = [x.clone() for x in fn()]
                    g2 = fn()
                    row["rerun_maxdiff_dq_dk_dv"] = [maxdiff(a, b) for a, b in zip(g1, g2)]
                    row["fwd_ms"], row["fwd_min_ms"] = timed(lambda: fwd(q, k, v, pattern, pat, True), iters)
                    row["fwd_ratio_vs_sdpa"] = row["fwd_ms"] / sdpa["fwd_ms"]
                if module == "v11":
                    ref[pattern] = g1
                else:
                    row["maxdiff_dq_dk_dv_vs_v11"] = [maxdiff(a, b) for a, b in zip(g1, ref[pattern])]
                    row["dk_dv_equal_vs_v11"] = bool(torch.equal(g1[1], ref[pattern][1]) and torch.equal(g1[2], ref[pattern][2]))
                del g1, g2
                res["bwd"].append(row)
                print(f"[bwd] L={L:6d} {module:4s} {pattern:5s} bwd {row['bwd_ms']:8.2f} ms /sdpa {row['bwd_ratio_vs_sdpa']:.3f}"
                      f"  fwd {row['fwd_ms']:7.2f} /sdpa {row['fwd_ratio_vs_sdpa']:.3f}"
                      f"  rerun {['%.1e' % x for x in row['rerun_maxdiff_dq_dk_dv']]}"
                      + (f"  vs v11 {['%.1e' % x for x in row['maxdiff_dq_dk_dv_vs_v11']]} dk/dv equal={row['dk_dv_equal_vs_v11']}"
                         if "maxdiff_dq_dk_dv_vs_v11" in row else "")
                      + ("" if row["valid"] else "  [NO_CQS ignores the mask: timing only]"), flush=True)
        del q, k, v, do, fwd_out, ref
        torch.cuda.empty_cache()


# ============================================================================ table + main
def print_table(res):
    print(f"\n==================== FORWARD, aligned c=7 task (fp16, H=8; ms, median of {res['iters']}) ====================")
    print("FA-2 arm = torch SDPA, FLASH_ATTENTION backend forced (flash_attn not installed; ~5% slower than the package). Kernels:",
          res.get("sdpa_flash_kernel_names"))
    print(f"{'causal':>6} {'D':>3} {'L':>6} {'module':>6} {'pattern':>7} | {'ms':>8} {'sdpaF':>8} {'ratio':>6} | {'equal v11':>9} {'maxdiff':>9} | poison")
    for r in res.get("fwd", []):
        print(f"{int(r['causal']):>6} {r['D']:>3} {r['L']:>6} {r['module']:>6} {r['pattern']:>7} | "
              + (f"{r['ms']:8.2f} {r['sdpa_flash_ms']:8.2f} {r['ratio_vs_sdpa']:6.3f}" if r["valid"] else f"{'-':>8} {r['sdpa_flash_ms']:8.2f} {'-':>6}")
              + " | " + (f"{str(r['equal_vs_v11']):>9} {r['maxdiff_out_vs_v11']:9.2e}" if "equal_vs_v11" in r else f"{'':>9} {'':>9}")
              + " | " + (f"ok={r['poison_ok']} rows={r['poison_rows_frac']*100:.1f}%" if "poison_ok" in r else ""))
    print("\n-- forward tax split (zero/plain ratios vs SDPA-flash) --")
    for causal, D in FWD_CONFIGS:
        rows = {(r["module"], r["pattern"]): r for r in res.get("fwd", []) if r["causal"] == causal and r["D"] == D}
        if not rows:
            continue
        line = f"  causal={int(causal)} D={D}:"
        for module in ("v9", "v11", "v11x"):
            for pattern in ("plain", "zero", "real"):
                r = rows.get((module, pattern))
                if r and r["valid"]:
                    line += f"  {module}/{pattern} {r['ratio_vs_sdpa']:.3f}"
        print(line)
    print(f"\n==================== BACKWARD, classic layout path (0,), causal, D=64 (ms, median of {res['iters']}) ====================")
    print(f"{'L':>6} {'module':>6} {'pattern':>7} | {'bwd ms':>8} {'sdpaF':>8} {'ratio':>6} | {'fwd ms':>8} {'ratio':>6} | rerun dq/dk/dv | vs v11 dq/dk/dv | dk,dv equal")
    for r in res.get("bwd", []):
        print(f"{r['L']:>6} {r['module']:>6} {r['pattern']:>7} | {r['bwd_ms']:8.2f} {r['sdpa_flash_bwd_ms']:8.2f} {r['bwd_ratio_vs_sdpa']:6.3f} | "
              f"{r['fwd_ms']:8.2f} {r['fwd_ratio_vs_sdpa']:6.3f} | {' '.join('%.1e' % x for x in r['rerun_maxdiff_dq_dk_dv'])} | "
              + (f"{' '.join('%.1e' % x for x in r['maxdiff_dq_dk_dv_vs_v11'])} | {r['dk_dv_equal_vs_v11']}" if "maxdiff_dq_dk_dv_vs_v11" in r else "")
              + ("" if r["valid"] else "  [timing only]"))
    rule = [r for r in res.get("bwd", []) if r["module"] == "v11x" and r["pattern"] == "plain"]
    if rule:
        worst = max(r["bwd_ratio_vs_sdpa"] for r in rule)
        print(f"\nroute rule: NO_CQS backward / SDPA-flash backward = {', '.join('%.3f' % r['bwd_ratio_vs_sdpa'] for r in rule)}"
              f" (L = {', '.join(str(r['L']) for r in rule)}); worst {worst:.3f} -> "
              + ("fix the fused kernel (<= 1.25x)" if worst <= 1.25 else "port the Triton two-kernel structure (> 1.25x)"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="small N, 2 iterations: correctness on a shared GPU")
    ap.add_argument("--n-fwd", type=int, default=131072)
    ap.add_argument("--n-bwd", type=int, nargs="+", default=[32768, 65536, 131072])
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--out", default=os.environ.get("EXP_AB_OUT", "/scratch/gpfs/AKEY/yb2807/Stream-CQSA-ultimate/results/ultimate/exp_build_ab.json"))
    a = ap.parse_args()
    if a.smoke:
        a.n_fwd, a.n_bwd, a.iters = 16384, [8192, 16384], 2
        a.out = a.out.replace(".json", "_smoke.json")
    t0 = time.time()
    res = dict(info=gpu_info(), fa2_arm="torch SDPA with SDPBackend.FLASH_ATTENTION forced (flash_attn not installed in CQSA311)",
               timing=f"1 warm-up + {a.iters} timed runs, CUDA events, median (min also stored)", iters=a.iters, H=H, dtype="float16",
               v11x="v11 + -DCQSA_NO_ELEMENT_PATH (fwd) + -DCQSA_BWD_NO_CQS (bwd) + -lineinfo", smoke=a.smoke)
    print(json.dumps(res["info"]), flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    for part, args in ((part_fwd, (a.n_fwd, a.iters)), (part_bwd, (a.n_bwd, a.iters))):
        try:
            part(res, *args)
        except Exception as e:  # noqa
            res[f"error_{part.__name__}"] = traceback.format_exc()
            print(f"!! {part.__name__} failed: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
        json.dump(res, open(a.out, "w"), indent=1)
    res["sdpa_flash_kernel_names"] = _SDPA_NAMES
    res["elapsed_s"] = time.time() - t0
    json.dump(res, open(a.out, "w"), indent=1)
    try:
        print_table(res)
    except Exception:  # noqa
        traceback.print_exc()
    print(f"\nwrote {a.out}  ({res['elapsed_s']:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
