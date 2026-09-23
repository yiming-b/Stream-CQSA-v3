"""
M0.1b: the kernel rows nobody has measured, in ONE process on ONE GPU (companion of benchmarks/layout_ab.py,
whose task builders, timing loop and SDPA-flash arm this reuses; layout_ab.py itself is not touched).

Rows (fp16, H=8, c=7, 128-aligned chunk layout unless stated; ms = median of 5 timed runs after 1 warm-up):
  (a) hdim128 causal, N=131072 itr=1: v11's hdim128 instantiation (classic extension; the wave has none) real/zero
      vs SDPA-flash hdim128, plus one classic (seg_align=0) reference row.
  (b) non-causal hdim64, N=131072 itr=1, through the v9 module (CQSA_CUDA_MODULE_NONCAUSAL, is_causal=False):
      real/zero vs SDPA-flash non-causal, plus one classic reference row.
  (c) bf16 hdim64 causal: the dev v11 build is an fp16-only kernel set, so the probe runs in a subprocess per module
      (the extension is chosen at import): first the dev module, then the release extension `cqsa_cuda` if importable.
  (d) synthetic patterns on the aligned L~56K task: all-ones bits and random 3-bit bits. All-ones masks EVERY pair,
      the diagonal included: the group-bits test bits[r] & bits[c] != 0 cannot mask everything except the diagonal.
  (e) ragged N=100003 itr=1: aligned vs unaligned, per task and recomposed.
  (f) itr=2 N=131072 (49 tasks, L~24K): per-task sums of v11 aligned / unaligned / SDPA-flash, the wave kernel's
      one-launch time (W=49) packed and in-place, and the wave rows vs v11 per task.

Bit-identity: the aligned and unaligned tasks of one path do NOT hold the same tokens (alignment moves the chunk
boundaries), so the cross-layout check is on the recomposed [N,H,D] output (max-shifted merge of every task with
reference.merge_one_contribution), which agrees to fp32 rounding, not bit for bit. Bit-for-bit (maxdiff 0.0) is
expected between gathered and in-place reads and between the wave rows and v11.

FA-2 arm: torch SDPA with the FLASH_ATTENTION backend forced (flash_attn is not installed); it is ~5% slower than
the flash_attn package at these shapes (results/kernel_ab_clean_v11.json), so every ratio here is ~5% kinder than
the same ratio vs flash_attn. The kernel names seen by the profiler are recorded for every arm.

Run (see slurm/ultimate/kernel_rows.slurm):
    python -P -u benchmarks/ultimate/kernel_rows.py [--smoke] [--parts abcdef]
--smoke shrinks N and the timed runs so the whole script fits a shared 40 GB GPU (correctness only, no timings).
"""
import argparse, importlib.util, json, os, subprocess, sys, time, traceback

os.environ.setdefault("CQSA_CUDA_MODULE", "cqsa_cuda_next_v11")
os.environ.setdefault("CQSA_CUDA_MODULE_NONCAUSAL", "cqsa_cuda_next_v9")
os.environ.setdefault("CQSA_CPU_THREADS", "8")

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
OUT_DIR = os.environ.get("KERNEL_ROWS_OUT", os.path.join(REPO, "results", "ultimate"))
OUT_JSON = os.path.join(OUT_DIR, "kernel_rows.json")
# layout_ab creates its output directory at import: point it here so importing its helpers never touches another checkout.
os.environ.setdefault("LAYOUT_AB_OUT", OUT_DIR)
sys.path.insert(0, os.path.join(REPO, "benchmarks"))

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend

import layout_ab as LAB
from layout_ab import timed, gpu_info, tile_census, to_dev, first_boundary, wave_pack_index, NATIVE, NATIVE_ERR
import stream_cqsa.interface as I
from stream_cqsa.interface import flash_attn_func_cqs_group_bits, cqs_block_summaries, CQS_BLK_SIZE
from stream_cqsa.stable_stream import build_tasks
from stream_cqsa.reference import merge_one_contribution, finalize
from stream_cqsa.native_wave import build_wave_tables, block_base_for, SEG_ALIGN

H = 8
C, ISET = 7, (0, 1, 3)
DEV = torch.device("cuda")
ITERS = 5
SDPA_CAVEAT = ("FA-2 arm = torch SDPA with SDPBackend.FLASH_ATTENTION forced (flash_attn not installed); "
               "SDPA-flash is ~5% slower than the flash_attn package, so every ratio vs it is ~5% kinder than vs flash_attn")


# ---------------------------------------------------------------------------- helpers
def dtype_name(dt):
    return str(dt).replace("torch.", "")


def kernel_names(fn):
    """Names of the attention kernels one call launches: the proof of which binary produced a row."""
    try:
        from torch.profiler import profile, ProfilerActivity
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            fn(); torch.cuda.synchronize()
        return sorted({e.name for e in prof.events()
                       if any(s in e.name.lower() for s in ("flash", "cqs", "wave", "attention"))})
    except Exception as e:  # noqa
        return [f"profiler unavailable: {type(e).__name__}: {e}"]


def cqs_fwd(q, k, v, bits, bo, ba, *, causal, block_base=None, seg_align=0, want_lse=False):
    """CQS forward through the interface: causal -> CQSA_CUDA_MODULE (v11), non-causal -> CQSA_CUDA_MODULE_NONCAUSAL (v9).
    fp32 normalised out [1,L,H,D]; with want_lse also the lse [1,H,L]."""
    r = flash_attn_func_cqs_group_bits(q, k, v, bits, softmax_scale=q.shape[-1] ** -0.5, causal=causal,
                                       cqs_blk_or=bo, cqs_blk_and=ba, cqs_blk_size=CQS_BLK_SIZE, fp32_out=True,
                                       return_attn_probs=want_lse, cqs_block_base=block_base, cqs_seg_align=seg_align)
    return (r[0], r[1]) if want_lse else r


class SdpaArm:
    """layout_ab's FA-2 arm generalised over (D, dtype, causal): plain SDPA-flash call at length L, cached per shape."""

    def __init__(self):
        self.cache, self.names = {}, {}

    def __call__(self, L, D=64, dtype=torch.float16, causal=True):
        key = (int(L), int(D), dtype_name(dtype), bool(causal))
        if key not in self.cache:
            torch.manual_seed(1)
            q, k, v = (torch.randn(1, H, L, D, device=DEV, dtype=dtype) for _ in range(3))
            fn = lambda: F.scaled_dot_product_attention(q, k, v, is_causal=causal, scale=D ** -0.5)
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                med, mn, _ = timed(fn, it=ITERS)
                arm = f"sdpa_flash_hdim{D}_{dtype_name(dtype)}_{'causal' if causal else 'noncausal'}"
                if arm not in self.names:
                    self.names[arm] = kernel_names(fn)
            del q, k, v
            self.cache[key] = (med, mn)
        return self.cache[key]


SDPA = SdpaArm()


def tasks_for(N, itr, seg_align, D=64):
    return build_tasks(N, itr, B=1, H=H, D=D, itemsize=2, sorted_gather=True, pin=False,
                       c=C, interest_set=ISET, seg_align=seg_align)


def gather(qt, kt, vt, ids):
    """Token-major [1,N,H,D] -> the task's packed [1,L,H,D] copies."""
    idx = ids.to(DEV)
    return tuple(x.index_select(1, idx).contiguous() for x in (qt, kt, vt))


def task_tables(t):
    return to_dev(t.group_bits), to_dev(t.extra["blk_or"]), to_dev(t.extra["blk_and"])


def tables_for(bits):
    """Kernel tables (bits, blk_or, blk_and) on the device for an arbitrary per-token bits vector."""
    bo, ba = (to_dev(x) for x in cqs_block_summaries(bits, CQS_BLK_SIZE))
    return to_dev(bits), bo, ba


def allones_bits(L):
    return torch.full((L,), -1, dtype=torch.int64)


def random_bits(L, seed=0):
    """3 random bits per token as tests/test_stable_stream.py builds them; bits 0 (an 'owner') happens 1/8 of the time."""
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 8, (L,), generator=g, dtype=torch.int64)


def fp64_rowblock_check(out32, q, k, v, bits, r0, *, causal, nrows=128):
    """max |out32[rows] - fp64 reference| over rows [r0, r0+nrows) of one task (q/k/v [1,L,H,D]).
    Rows without a live key are counted, not compared: the reference softmax is undefined there."""
    L, D = q.shape[1], q.shape[-1]
    r1 = min(L, r0 + nrows)
    rows = torch.arange(r0, r1, device=DEV)
    cols = torch.arange(L, device=DEV)
    keep = (bits[rows][:, None] & bits[None, :]) == 0
    if causal:
        keep &= cols[None, :] <= rows[:, None]
    live = keep.any(1)
    res = dict(rows=[int(r0), int(r1)], n_empty_rows=int((~live).sum()), max_abs=None, max_abs_ref=None)
    if not bool(live.any()):
        return res
    maxabs, maxref = 0.0, 0.0
    for h in range(H):
        s = (q[0, r0:r1, h].double() @ k[0, :, h].double().T) * D ** -0.5
        p = torch.softmax(s.masked_fill(~keep, float("-inf")), dim=-1)[live]
        ref = p @ v[0, :, h].double()
        got = out32[0, r0:r1, h].double()[live]
        maxabs = max(maxabs, (got - ref).abs().max().item())
        maxref = max(maxref, ref.abs().max().item())
        del s, p, ref, got
    res.update(max_abs=maxabs, max_abs_ref=maxref)
    return res


def fp64_global_check(out_full, qt, kt, vt, r0, *, causal, nrows=64):
    """Recomposed output [1,H,N,D] vs fp64 attention over ALL N keys on rows [r0, r0+nrows) (qt/kt/vt token-major)."""
    N, D = qt.shape[1], qt.shape[-1]
    r1 = min(N, r0 + nrows)
    rows = torch.arange(r0, r1, device=DEV)
    cols = torch.arange(N, device=DEV)
    maxabs, maxref = 0.0, 0.0
    for h in range(H):
        s = (qt[0, r0:r1, h].double() @ kt[0, :, h].double().T) * D ** -0.5
        if causal:
            s = s.masked_fill(cols[None, :] > rows[:, None], float("-inf"))
        ref = torch.softmax(s, dim=-1) @ vt[0, :, h].double()
        got = out_full[0, h, r0:r1].double()
        maxabs = max(maxabs, (got - ref).abs().max().item())
        maxref = max(maxref, ref.abs().max().item())
        del s, ref, got
    return dict(rows=[int(r0), int(r1)], max_abs=maxabs, max_abs_ref=maxref)


class Recomposer:
    """Max-shifted merge of every task's (normalised fp32 out, lse) into the global [1,H,N,D] output.

    A normalised partial with local_l = 1 and local_m = lse gets the weight exp(lse_t - M) in
    reference.merge_one_contribution, which is exactly the merge of normalised partials. Rows a task
    cannot see come back with a non-finite lse (FA-2 writes +inf for an empty row); they carry nothing
    and are mapped to -inf so the merge gives them weight 0 instead of a poisoned maximum."""

    def __init__(self, N, D, device=DEV):
        self.acc = torch.zeros(1, H, N, D, device=device)
        self.l = torch.zeros(1, H, N, device=device)
        self.m = torch.full((1, H, N), float("-inf"), device=device)

    def add(self, ids, out32, lse):
        fin = torch.isfinite(lse)
        acc = out32.permute(0, 2, 1, 3)
        acc = torch.where(fin[..., None], acc, torch.zeros_like(acc))    # weight 0 must not meet a NaN/inf sentinel
        lse = torch.where(fin, lse, torch.full_like(lse, float("-inf")))
        merge_one_contribution(self.acc, self.l, self.m, ids.to(self.acc.device), acc, torch.ones_like(lse), lse)

    def output(self):
        return finalize(self.acc, self.l)


def task_row(t, qt, kt, vt, *, causal, layout, sdpa_key, zero=True, fp64=False, recomp=None):
    """One gathered task: CQS real (+ zero pattern) vs the SDPA arm at its L. Returns (row, out32, lse)."""
    L = int(t.local_size)
    D = qt.shape[-1]
    q_i, k_i, v_i = gather(qt, kt, vt, t.token_ids)
    bits, bo, ba = task_tables(t)
    row = dict(layout=layout, path=list(t.path), L=L, L_mod_128=L % 128, census=tile_census(bo, ba, L))
    row["cqs_real_ms"], row["cqs_real_min_ms"], _ = timed(lambda: cqs_fwd(q_i, k_i, v_i, bits, bo, ba, causal=causal), it=ITERS)
    if zero:
        z, zo, za = tables_for(torch.zeros(L, dtype=torch.int64))
        row["cqs_zero_ms"], row["cqs_zero_min_ms"], _ = timed(lambda: cqs_fwd(q_i, k_i, v_i, z, zo, za, causal=causal), it=ITERS)
    row["sdpa_flash_ms"], row["sdpa_flash_min_ms"] = SDPA(L, **sdpa_key)
    row["real_over_sdpa"] = row["cqs_real_ms"] / row["sdpa_flash_ms"]
    if zero:
        row["zero_over_sdpa"] = row["cqs_zero_ms"] / row["sdpa_flash_ms"]
    out, lse = cqs_fwd(q_i, k_i, v_i, bits, bo, ba, causal=causal, want_lse=True)
    if fp64:
        row["fp64_check_gathered"] = fp64_rowblock_check(out, q_i, k_i, v_i, bits, max(0, first_boundary(t.group_bits) - 64), causal=causal)
    if recomp is not None:
        recomp.add(t.token_ids, out, lse)
    del q_i, k_i, v_i
    return row, out, lse


def inplace_row(t, qt, kt, vt, out_g, *, causal):
    """Segmented in-place reads of the same aligned task straight from the token-major tensors, vs the gathered output."""
    L, D = int(t.local_size), qt.shape[-1]
    bits, bo, ba = task_tables(t)
    seg_base = to_dev(t.extra["seg_base"])
    q_s, k_s, v_s = (x.as_strided((1, L, H, D), x.stride()) for x in (qt, kt, vt))
    fn = lambda: cqs_fwd(q_s, k_s, v_s, bits, bo, ba, causal=causal, block_base=seg_base, seg_align=SEG_ALIGN)
    ms, mn, _ = timed(fn, it=ITERS)
    out_ip = fn()
    r = dict(cqs_inplace_ms=ms, cqs_inplace_min_ms=mn, maxdiff_inplace_vs_gathered=(out_ip - out_g).abs().max().item())
    del out_ip
    return r


def fmt_ratio(row, key="real_over_sdpa"):
    return f"{row[key]:.3f}" if key in row else "  n/a"


def layout_pass(N, itr, seg_align, layout, qt, kt, vt, *, causal, sdpa_key, fp64_paths, zero=True, on_task=None):
    """All tasks of one layout: per-task rows plus the recomposed output. `on_task(w, t, out)` sees every task's output."""
    tasks = tasks_for(N, itr, seg_align, qt.shape[-1])
    rec = Recomposer(N, qt.shape[-1])
    rows = []
    for w, t in enumerate(tasks):
        row, out, lse = task_row(t, qt, kt, vt, causal=causal, layout=layout, sdpa_key=sdpa_key, zero=zero,
                                 fp64=t.path in fp64_paths, recomp=rec)
        if on_task is not None:
            on_task(w, t, out)
        rows.append(row)
        del out, lse
    return tasks, rows, rec.output()


def sums(rows):
    s = dict(n_tasks=len(rows), L_min=min(r["L"] for r in rows), L_max=max(r["L"] for r in rows),
             sum_cqs_real_ms=sum(r["cqs_real_ms"] for r in rows), sum_sdpa_flash_ms=sum(r["sdpa_flash_ms"] for r in rows),
             mixed_frac_mean=sum(r["census"]["mixed"] for r in rows) / len(rows),
             masked_frac_mean=sum(r["census"]["masked"] for r in rows) / len(rows))
    s["real_over_sdpa"] = s["sum_cqs_real_ms"] / s["sum_sdpa_flash_ms"]
    if all("cqs_zero_ms" in r for r in rows):
        s["sum_cqs_zero_ms"] = sum(r["cqs_zero_ms"] for r in rows)
        s["zero_over_sdpa"] = s["sum_cqs_zero_ms"] / s["sum_sdpa_flash_ms"]
    return s


def cross_layout(out_al, out_un, qt, kt, vt, *, causal):
    """The recomposed outputs of the two layouts (same tokens, different task partition) and an fp64 spot check of both."""
    d = (out_al - out_un).abs().max().item()
    rel = ((out_al - out_un).norm() / out_un.norm()).item()
    r0 = qt.shape[1] // 2
    return dict(recomposed_maxabs_aligned_vs_unaligned=d, recomposed_rel_aligned_vs_unaligned=rel,
                fp64_check_recomposed_aligned=fp64_global_check(out_al, qt, kt, vt, r0, causal=causal),
                fp64_check_recomposed_unaligned=fp64_global_check(out_un, qt, kt, vt, r0, causal=causal))


def inputs(N, D, dtype=torch.float16, seed=0):
    torch.manual_seed(seed)
    return tuple(torch.randn(1, N, H, D, device=DEV, dtype=dtype) for _ in range(3))   # token-major, in place


# ============================================================================ (a) hdim128 causal
def part_a(res, N):
    D = 128
    qt, kt, vt = inputs(N, D)
    sk = dict(D=D, dtype=torch.float16, causal=True)
    rows = []
    for layout, seg_align in (("aligned_128", SEG_ALIGN), ("classic_align1", 0)):
        tasks = tasks_for(N, 1, seg_align, D)
        for t in tasks if seg_align else tasks[:1]:            # classic: one reference row (path (0,))
            row, out, _ = task_row(t, qt, kt, vt, causal=True, layout=layout, sdpa_key=sk, fp64=(t.path == (0,)))
            if seg_align:
                row.update(inplace_row(t, qt, kt, vt, out, causal=True))
            if not rows:
                bits, bo, ba = task_tables(t)
                q_i, k_i, v_i = gather(qt, kt, vt, t.token_ids)
                res["kernel_names"]["v11_hdim128_causal"] = kernel_names(lambda: cqs_fwd(q_i, k_i, v_i, bits, bo, ba, causal=True))
                del q_i, k_i, v_i
            rows.append(row)
            print(f"[a] hdim128 {layout:14s} path={t.path} L={row['L']:6d} v11 real {row['cqs_real_ms']:8.2f} zero {row['cqs_zero_ms']:8.2f} "
                  f"sdpaF {row['sdpa_flash_ms']:8.2f}  real/F {fmt_ratio(row)} zero/F {fmt_ratio(row, 'zero_over_sdpa')}"
                  + (f"  inplace {row['cqs_inplace_ms']:8.2f} maxdiff {row['maxdiff_inplace_vs_gathered']:.1e}" if "cqs_inplace_ms" in row else ""), flush=True)
            del out
    res["a_hdim128_causal"] = dict(N=N, D=D, rows=rows, summary_aligned=sums([r for r in rows if r["layout"] == "aligned_128"]))
    del qt, kt, vt
    torch.cuda.empty_cache()


# ============================================================================ (b) non-causal hdim64 via v9
def part_b(res, N):
    D = 64
    qt, kt, vt = inputs(N, D)
    sk = dict(D=D, dtype=torch.float16, causal=False)
    rows = []
    for layout, seg_align in (("aligned_128", SEG_ALIGN), ("classic_align1", 0)):
        tasks = tasks_for(N, 1, seg_align, D)
        for t in tasks if seg_align else tasks[:1]:
            row, out, _ = task_row(t, qt, kt, vt, causal=False, layout=layout, sdpa_key=sk, fp64=(t.path == (0,)))
            if seg_align:
                row.update(inplace_row(t, qt, kt, vt, out, causal=False))
            if not rows:
                bits, bo, ba = task_tables(t)
                q_i, k_i, v_i = gather(qt, kt, vt, t.token_ids)
                res["kernel_names"]["v9_hdim64_noncausal"] = kernel_names(lambda: cqs_fwd(q_i, k_i, v_i, bits, bo, ba, causal=False))
                del q_i, k_i, v_i
            rows.append(row)
            print(f"[b] noncausal {layout:14s} path={t.path} L={row['L']:6d} v9 real {row['cqs_real_ms']:8.2f} zero {row['cqs_zero_ms']:8.2f} "
                  f"sdpaF {row['sdpa_flash_ms']:8.2f}  real/F {fmt_ratio(row)} zero/F {fmt_ratio(row, 'zero_over_sdpa')}"
                  + (f"  inplace {row['cqs_inplace_ms']:8.2f} maxdiff {row['maxdiff_inplace_vs_gathered']:.1e}" if "cqs_inplace_ms" in row else ""), flush=True)
            del out
    res["b_noncausal_hdim64"] = dict(N=N, D=D, module=I.CQSA_CUDA_MODULE_NONCAUSAL, rows=rows,
                                     summary_aligned=sums([r for r in rows if r["layout"] == "aligned_128"]))
    del qt, kt, vt
    torch.cuda.empty_cache()


# ============================================================================ (c) bf16 hdim64 causal
def ext_record(ext):
    """Which .so a module is: path, size and build time (the stable copies and the variant-dir builds differ)."""
    f = getattr(ext, "__file__", None)
    if f is None or not os.path.exists(f):
        return dict(extension=f)
    st = os.stat(f)
    return dict(extension=f, extension_bytes=st.st_size, extension_mtime=time.strftime("%Y-%m-%d %H:%M", time.localtime(st.st_mtime)))


def bf16_worker(N):
    """Subprocess body: CQSA_CUDA_MODULE is already the module under test. bf16 hdim64 causal on the aligned path-(0,)
    task, with the same task in fp16 through the same binary so the bf16 row is tied to a same-binary fp16 number."""
    r = dict(module=I.CQSA_CUDA_MODULE, N=N, **ext_record(I.cqsa_cuda))
    try:
        t = tasks_for(N, 1, SEG_ALIGN, 64)[0]
        L = int(t.local_size)
        bits, bo, ba = task_tables(t)
        z, zo, za = tables_for(torch.zeros(L, dtype=torch.int64))
        r.update(L=L, path=list(t.path), census=tile_census(bo, ba, L))
        for dt, tag in ((torch.bfloat16, "bf16"), (torch.float16, "fp16")):
            torch.manual_seed(0)
            q, k, v = (torch.randn(1, L, H, 64, device=DEV, dtype=dt) for _ in range(3))
            out = cqs_fwd(q, k, v, bits, bo, ba, causal=True)          # raises on an fp16-only kernel set
            r[f"{tag}_kernel_names"] = kernel_names(lambda: cqs_fwd(q, k, v, bits, bo, ba, causal=True))
            r[f"{tag}_cqs_real_ms"], r[f"{tag}_cqs_real_min_ms"], _ = timed(lambda: cqs_fwd(q, k, v, bits, bo, ba, causal=True), it=ITERS)
            r[f"{tag}_cqs_zero_ms"], r[f"{tag}_cqs_zero_min_ms"], _ = timed(lambda: cqs_fwd(q, k, v, z, zo, za, causal=True), it=ITERS)
            r[f"{tag}_sdpa_flash_ms"], r[f"{tag}_sdpa_flash_min_ms"] = SDPA(L, D=64, dtype=dt, causal=True)
            r[f"{tag}_real_over_sdpa"] = r[f"{tag}_cqs_real_ms"] / r[f"{tag}_sdpa_flash_ms"]
            r[f"{tag}_zero_over_sdpa"] = r[f"{tag}_cqs_zero_ms"] / r[f"{tag}_sdpa_flash_ms"]
            r[f"{tag}_fp64_check_gathered"] = fp64_rowblock_check(out, q, k, v, bits, max(0, first_boundary(t.group_bits) - 64), causal=True)
            del q, k, v, out
        r["sdpa_kernel_names"] = SDPA.names
        r["ran"] = True
    except Exception as e:  # noqa
        r["ran"] = False
        r["error"] = f"{type(e).__name__}: {str(e).splitlines()[0] if str(e) else ''}"
    print("RESULT " + json.dumps(r), flush=True)


# bf16 candidates: (module, extra sys.path entry placed right after the package, what it is). The recipe's
# kernel/stable copies are fp16-only; the variant-dir build of the same v11 source carries the common
# (fp16+bf16) set; cqsa_cuda in next/pkg is the SHIPPED pre-v11 kernel, a reference row only.
BF16_CANDIDATES = (
    (os.environ["CQSA_CUDA_MODULE"], None, "recipe module (kernel/stable copy)"),
    (os.environ["CQSA_CUDA_MODULE"], "/scratch/gpfs/AKEY/yb2807/Stream-CQSA-dev/next/kernel/v11", "variant-dir build of the same v11 source"),
    ("cqsa_cuda", None, "shipped pre-v11 kernel (next/pkg), reference only"),
)


def part_c(res, N):
    rows = []
    for mod, extra, what in BF16_CANDIDATES:
        env = dict(os.environ, CQSA_CUDA_MODULE=mod)
        if extra is not None:
            # after the package (so the variant dir's build stub never shadows stream_cqsa), before the stable copies
            env["PYTHONPATH"] = os.pathsep.join([REPO, extra, os.environ.get("PYTHONPATH", "")])
        p = subprocess.run([sys.executable, "-P", os.path.abspath(__file__), "--bf16-worker", str(N)],
                           capture_output=True, text=True, env=env)
        lines = [x for x in p.stdout.splitlines() if x.startswith("RESULT ")]
        r = json.loads(lines[-1][7:]) if lines else dict(module=mod, ran=False, error=f"worker died: {p.stderr[-800:]}")
        r["what"] = what
        rows.append(r)
        print(f"[c] bf16 hdim64 causal via {mod} [{what}] {r.get('extension')}: "
              + (f"L={r['L']} bf16 real {r['bf16_cqs_real_ms']:.2f} zero {r['bf16_cqs_zero_ms']:.2f} sdpaF {r['bf16_sdpa_flash_ms']:.2f} "
                 f"real/F {r['bf16_real_over_sdpa']:.3f} zero/F {r['bf16_zero_over_sdpa']:.3f} | same binary fp16 real {r['fp16_cqs_real_ms']:.2f} "
                 f"sdpaF {r['fp16_sdpa_flash_ms']:.2f} real/F {r['fp16_real_over_sdpa']:.3f}" if r["ran"] else f"NOT RUN ({r['error']})"), flush=True)
    res["c_bf16_hdim64_causal"] = dict(N=N, attempts=rows)


# ============================================================================ (d) synthetic patterns at L~56K
def part_d(res, N):
    D = 64
    t = tasks_for(N, 1, SEG_ALIGN, D)[0]
    L = int(t.local_size)
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, L, H, D, device=DEV, dtype=torch.float16) for _ in range(3))
    sdpa_ms, _ = SDPA(L, D=D, dtype=torch.float16, causal=True)
    rows = []
    for name, bits_cpu in (("real", t.group_bits), ("zero", torch.zeros(L, dtype=torch.int64)),
                           ("allones", allones_bits(L)), ("random3bit", random_bits(L))):
        bits, bo, ba = tables_for(bits_cpu)
        row = dict(pattern=name, L=L, census=tile_census(bo, ba, L), sdpa_flash_ms=sdpa_ms)
        row["cqs_ms"], row["cqs_min_ms"], _ = timed(lambda: cqs_fwd(q, k, v, bits, bo, ba, causal=True), it=ITERS)
        row["ratio_vs_sdpa"] = row["cqs_ms"] / sdpa_ms
        out, lse = cqs_fwd(q, k, v, bits, bo, ba, causal=True, want_lse=True)
        row["out_all_finite"] = bool(torch.isfinite(out).all())
        row["lse_nonfinite_rows"] = int((~torch.isfinite(lse)).sum())
        row["lse_min"], row["lse_max"] = lse.min().item(), lse.max().item()
        r0 = 0 if name in ("allones", "random3bit") else max(0, first_boundary(t.group_bits) - 64)
        row["fp64_check"] = fp64_rowblock_check(out, q, k, v, bits, r0, causal=True)
        rows.append(row)
        print(f"[d] L={L} {name:10s} v11 {row['cqs_ms']:8.2f} ms  /sdpaF {row['ratio_vs_sdpa']:.3f}  masked {row['census']['masked']*100:5.1f}% "
              f"mixed {row['census']['mixed']*100:5.2f}%  finite={row['out_all_finite']} lse[{row['lse_min']:.3g},{row['lse_max']:.3g}] "
              f"nonfinite rows {row['lse_nonfinite_rows']}  fp64 {row['fp64_check']}", flush=True)
        del out, lse
    res["d_patterns_L56K"] = dict(N=N, D=D, L=L, layout="aligned_128", path=list(t.path), rows=rows,
                                  note="allones (-1) masks every pair including the diagonal: bits[i]&bits[i]!=0 whenever bits[i]!=0")
    del q, k, v
    torch.cuda.empty_cache()


# ============================================================================ (e) ragged N
def part_e(res, N):
    D = 64
    qt, kt, vt = inputs(N, D)
    sk = dict(D=D, dtype=torch.float16, causal=True)
    out = {}
    for layout, seg_align in (("aligned_128", SEG_ALIGN), ("classic_align1", 0)):
        tasks, rows, out_full = layout_pass(N, 1, seg_align, layout, qt, kt, vt, causal=True, sdpa_key=sk, fp64_paths=((0,),))
        if seg_align:
            t = tasks[0]
            og, _ = cqs_fwd(*gather(qt, kt, vt, t.token_ids), *task_tables(t), causal=True, want_lse=True)
            rows[0].update(inplace_row(t, qt, kt, vt, og, causal=True))
            del og
        out[layout] = dict(rows=rows, summary=sums(rows))
        for r in rows:
            print(f"[e] N={N} {layout:14s} path={r['path']} L={r['L']:6d} (L%128={r['L_mod_128']:3d}) v11 real {r['cqs_real_ms']:8.2f} zero {r['cqs_zero_ms']:8.2f} "
                  f"sdpaF {r['sdpa_flash_ms']:8.2f} real/F {fmt_ratio(r)} zero/F {fmt_ratio(r, 'zero_over_sdpa')} mixed {r['census']['mixed']*100:.2f}%"
                  + (f"  inplace {r['cqs_inplace_ms']:8.2f} maxdiff {r['maxdiff_inplace_vs_gathered']:.1e}" if "cqs_inplace_ms" in r else ""), flush=True)
        out[layout]["_out"] = out_full
    x = cross_layout(out["aligned_128"].pop("_out"), out["classic_align1"].pop("_out"), qt, kt, vt, causal=True)
    x["sdpa_flash_monolithic_N_ms"] = SDPA(N, **sk)[0]
    res["e_ragged"] = dict(N=N, D=D, layouts=out, **x)
    print(f"[e] recomposed aligned vs unaligned: maxabs {x['recomposed_maxabs_aligned_vs_unaligned']:.2e} rel {x['recomposed_rel_aligned_vs_unaligned']:.2e}; "
          f"fp64 aligned {x['fp64_check_recomposed_aligned']} unaligned {x['fp64_check_recomposed_unaligned']}; mono sdpaF {x['sdpa_flash_monolithic_N_ms']:.2f} ms", flush=True)
    del qt, kt, vt
    torch.cuda.empty_cache()


# ============================================================================ (f) itr=2
def part_f(res, N):
    D = 64
    qt, kt, vt = inputs(N, D)
    sk = dict(D=D, dtype=torch.float16, causal=True)
    r = dict(N=N, D=D, itr=2)
    # the wave first: one launch over all aligned tasks (uniform layout), packed and in-place
    wave_out = None
    if NATIVE is not None:
        try:
            atasks = tasks_for(N, 2, SEG_ALIGN, D)
            tb = build_wave_tables(atasks, DEV, uniform=True)
            pidx = wave_pack_index(atasks, tb.S)
            qp, kp, vp = (x[0].index_select(0, pidx).contiguous() for x in (qt, kt, vt))
            r["wave_packed_ms"], r["wave_packed_min_ms"], _ = timed(lambda: LAB.wave_call(tb, qp, kp, vp), it=ITERS)
            res["kernel_names"]["wave_hdim64_causal"] = kernel_names(lambda: LAB.wave_call(tb, qp, kp, vp))
            wave_out = LAB.wave_call(tb, qp, kp, vp)[0]
            bb = block_base_for(tb, lambda g: g).to(DEV)
            q3, k3, v3 = (x[0] for x in (qt, kt, vt))
            r["wave_inplace_ms"], r["wave_inplace_min_ms"], _ = timed(lambda: LAB.wave_call(tb, q3, k3, v3, bb), it=ITERS)
            # only the real rows: the padding rows of the two layouts hold different tokens (packed padding repeats
            # the last real row, in-place padding blocks alias the last real 128-block) and are discarded by the merge
            real = torch.cat([torch.arange(w * tb.S, w * tb.S + int(t.local_size)) for w, t in enumerate(atasks)]).to(DEV)
            r["maxdiff_wave_inplace_vs_packed"] = (LAB.wave_call(tb, q3, k3, v3, bb)[0][real] - wave_out[real]).abs().max().item()
            del real
            r["wave_W"], r["wave_S"] = int(tb.W), int(tb.S)
            r["wave_uniform_packing"] = sum(int(t.local_size) for t in atasks) / (tb.W * tb.S)
            del qp, kp, vp, tb, bb
        except Exception as e:  # noqa
            r["wave_error"] = f"{type(e).__name__}: {e}"
            traceback.print_exc()
    else:
        r["wave_error"] = NATIVE_ERR
    diffs = []

    def vs_wave(w, t, out):
        if wave_out is not None:
            diffs.append((wave_out[w * r["wave_S"]: w * r["wave_S"] + int(t.local_size)] - out[0]).abs().max().item())

    outs = {}
    for layout, seg_align in (("aligned_128", SEG_ALIGN), ("classic_align1", 0)):
        tasks, rows, outs[layout] = layout_pass(N, 2, seg_align, layout, qt, kt, vt, causal=True, sdpa_key=sk, fp64_paths=((0, 0),),
                                                on_task=vs_wave if seg_align else None)
        r[layout] = dict(rows=rows, summary=sums(rows))
        s = r[layout]["summary"]
        print(f"[f] itr=2 {layout:14s}: {s['n_tasks']} tasks L {s['L_min']}-{s['L_max']}; sum v11 real {s['sum_cqs_real_ms']:.1f} zero {s['sum_cqs_zero_ms']:.1f} "
              f"sdpaF {s['sum_sdpa_flash_ms']:.1f} ms; real/F {s['real_over_sdpa']:.3f} zero/F {s['zero_over_sdpa']:.3f}; mixed {s['mixed_frac_mean']*100:.2f}% masked {s['masked_frac_mean']*100:.1f}%", flush=True)
    if diffs:
        r["maxdiff_wave_rows_vs_v11_per_task_max"] = max(diffs)
    del wave_out
    r.update(cross_layout(outs["aligned_128"], outs["classic_align1"], qt, kt, vt, causal=True))
    r["sdpa_flash_monolithic_N_ms"] = SDPA(N, **sk)[0]
    res["f_itr2"] = r
    print(f"[f] wave one launch W={r.get('wave_W')} S={r.get('wave_S')}: packed {r.get('wave_packed_ms', float('nan')):.1f} inplace {r.get('wave_inplace_ms', float('nan')):.1f} ms "
          f"(packing {r.get('wave_uniform_packing', float('nan')):.3f}); wave rows vs v11 maxdiff {r.get('maxdiff_wave_rows_vs_v11_per_task_max')}; "
          f"recomposed aligned vs unaligned maxabs {r['recomposed_maxabs_aligned_vs_unaligned']:.2e} rel {r['recomposed_rel_aligned_vs_unaligned']:.2e}; "
          f"mono sdpaF {r['sdpa_flash_monolithic_N_ms']:.1f} ms" + (f"; wave error {r['wave_error']}" if "wave_error" in r else ""), flush=True)
    del qt, kt, vt, outs
    torch.cuda.empty_cache()


# ============================================================================ table + main
def print_table(res):
    print("\n==================== KERNEL ROWS (ms, median of 5 after 1 warm-up; H=8; " + res["info"]["torch_device"] + ") ====================")
    print(SDPA_CAVEAT)
    hdr = f"{'row':>28} {'layout':>15} {'path':>7} {'L':>7} {'L%128':>5} | {'cqs real':>9} {'cqs zero':>9} {'sdpaF':>9} | {'real/F':>6} {'zero/F':>6} | {'inplace':>8} {'ip diff':>8} | {'mixed%':>6} {'masked%':>7}"
    print(hdr)

    def line(tag, r):
        print(f"{tag:>28} {r['layout']:>15} {str(r['path']):>7} {r['L']:>7} {r['L_mod_128']:>5} | {r['cqs_real_ms']:9.2f} {r.get('cqs_zero_ms', float('nan')):9.2f} {r['sdpa_flash_ms']:9.2f} | "
              f"{r['real_over_sdpa']:6.3f} {r.get('zero_over_sdpa', float('nan')):6.3f} | {r.get('cqs_inplace_ms', float('nan')):8.2f} {r.get('maxdiff_inplace_vs_gathered', float('nan')):8.1e} | "
              f"{r['census']['mixed']*100:6.2f} {r['census']['masked']*100:7.1f}")

    if "a_hdim128_causal" in res:
        for r in res["a_hdim128_causal"]["rows"]:
            line("(a) hdim128 fp16 causal v11", r)
    if "b_noncausal_hdim64" in res:
        for r in res["b_noncausal_hdim64"]["rows"]:
            line(f"(b) noncausal hdim64 {res['b_noncausal_hdim64']['module'].replace('cqsa_cuda_next_', '')}", r)
    if "c_bf16_hdim64_causal" in res:
        for r in res["c_bf16_hdim64_causal"]["attempts"]:
            print(f"{'(c) bf16 hdim64 causal':>28} via {r['module']} [{r['what']}] {r.get('extension')}: "
                  + (f"L={r['L']} bf16 real {r['bf16_cqs_real_ms']:.2f} zero {r['bf16_cqs_zero_ms']:.2f} sdpaF {r['bf16_sdpa_flash_ms']:.2f} "
                     f"real/F {r['bf16_real_over_sdpa']:.3f} zero/F {r['bf16_zero_over_sdpa']:.3f} fp64 {r['bf16_fp64_check_gathered']['max_abs']:.2e} | "
                     f"same binary fp16 real {r['fp16_cqs_real_ms']:.2f} zero {r['fp16_cqs_zero_ms']:.2f} sdpaF {r['fp16_sdpa_flash_ms']:.2f} real/F {r['fp16_real_over_sdpa']:.3f} "
                     f"zero/F {r['fp16_zero_over_sdpa']:.3f}" if r["ran"] else f"NOT RUN: {r['error']}"))
    if "d_patterns_L56K" in res:
        d = res["d_patterns_L56K"]
        for r in d["rows"]:
            print(f"{'(d) pattern ' + r['pattern']:>28} {d['layout']:>15} {str(d['path']):>7} {r['L']:>7} {r['L'] % 128:>5} | {r['cqs_ms']:9.2f} {'':>9} {r['sdpa_flash_ms']:9.2f} | "
                  f"{r['ratio_vs_sdpa']:6.3f} {'':>6} | {'':>8} {'':>8} | {r['census']['mixed']*100:6.2f} {r['census']['masked']*100:7.1f}"
                  f"   finite={r['out_all_finite']} nonfinite-lse rows={r['lse_nonfinite_rows']} fp64={r['fp64_check']}")
    for key, tag in (("e_ragged", "(e) ragged N=%d"), ("f_itr2", "(f) itr=2 N=%d")):
        if key not in res:
            continue
        p = res[key]
        lay = p["layouts"] if key == "e_ragged" else {k_: p[k_] for k_ in ("aligned_128", "classic_align1")}
        for layout, v in lay.items():
            if key == "e_ragged":
                for r in v["rows"]:
                    line(tag % p["N"], r)
            s = v["summary"]
            print(f"{tag % p['N']:>28} {layout:>15} SUM over {s['n_tasks']} tasks (L {s['L_min']}-{s['L_max']}): real {s['sum_cqs_real_ms']:.1f}  zero {s.get('sum_cqs_zero_ms', float('nan')):.1f}  "
                  f"sdpaF {s['sum_sdpa_flash_ms']:.1f} ms  real/F {s['real_over_sdpa']:.3f} zero/F {s.get('zero_over_sdpa', float('nan')):.3f}  mixed {s['mixed_frac_mean']*100:.2f}% masked {s['masked_frac_mean']*100:.1f}%")
        print(f"{'':>28} recomposed aligned vs unaligned: maxabs {p['recomposed_maxabs_aligned_vs_unaligned']:.2e} rel {p['recomposed_rel_aligned_vs_unaligned']:.2e}; "
              f"fp64 aligned {p['fp64_check_recomposed_aligned']['max_abs']:.2e} unaligned {p['fp64_check_recomposed_unaligned']['max_abs']:.2e} (ref max {p['fp64_check_recomposed_aligned']['max_abs_ref']:.3f}); "
              f"monolithic sdpaF at N {p['sdpa_flash_monolithic_N_ms']:.1f} ms")
        if key == "f_itr2":
            print(f"{'':>28} wave one launch W={p.get('wave_W')} S={p.get('wave_S')}: packed {p.get('wave_packed_ms', float('nan')):.1f} ms  inplace {p.get('wave_inplace_ms', float('nan')):.1f} ms  "
                  f"uniform packing {p.get('wave_uniform_packing', float('nan')):.3f}; wave rows vs v11 per task maxdiff {p.get('maxdiff_wave_rows_vs_v11_per_task_max')}; "
                  f"wave inplace vs packed maxdiff {p.get('maxdiff_wave_inplace_vs_packed')}" + (f"; wave error: {p['wave_error']}" if "wave_error" in p else ""))
    print("\nkernel names:", json.dumps(res["kernel_names"], indent=1))


def main():
    global ITERS
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="small N, 2 timed runs: correctness on a shared GPU, not timings")
    ap.add_argument("--parts", default="abcdef")
    ap.add_argument("--bf16-worker", type=int, metavar="N")
    a = ap.parse_args()
    if a.bf16_worker:
        return bf16_worker(a.bf16_worker)
    N_main, N_ragged = (32768, 25003) if a.smoke else (131072, 100003)
    if a.smoke:
        ITERS = 2
    t0 = time.time()
    res = dict(info=gpu_info(), fa2_arm=SDPA_CAVEAT, timing=f"1 warm-up + {ITERS} timed runs, CUDA events, median (min also stored)",
               smoke=a.smoke, H=H, c=C, interest_set=list(ISET),
               cuda_module=dict(name=I.CQSA_CUDA_MODULE, **ext_record(I.cqsa_cuda)),
               cuda_module_noncausal=dict(name=I.CQSA_CUDA_MODULE_NONCAUSAL, **ext_record(I.cqsa_cuda_noncausal)),
               native_ext=(None if NATIVE is None else getattr(NATIVE, "__file__", "cqsa_native")), native_import_error=NATIVE_ERR,
               kernel_names={})
    print(json.dumps(res["info"]), flush=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    out_json = OUT_JSON.replace(".json", "_smoke.json") if a.smoke else OUT_JSON
    parts = dict(a=(part_a, N_main), b=(part_b, N_main), c=(part_c, N_main), d=(part_d, N_main), e=(part_e, N_ragged), f=(part_f, N_main))
    for key in a.parts:
        fn, N = parts[key]
        try:
            fn(res, N)
        except Exception as e:  # noqa
            res[f"error_{fn.__name__}"] = traceback.format_exc()
            print(f"!! {fn.__name__} failed: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
        res["kernel_names"].update(SDPA.names)
        json.dump(res, open(out_json, "w"), indent=1)
    res["elapsed_s"] = time.time() - t0
    json.dump(res, open(out_json, "w"), indent=1)
    try:
        print_table(res)
    except Exception:  # noqa
        traceback.print_exc()
    print(f"\nwrote {out_json}  ({res['elapsed_s']:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
