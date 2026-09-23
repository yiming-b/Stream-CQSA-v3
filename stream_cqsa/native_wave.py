"""
next/native: the WAVE engine -- Stream-CQSA on the native multi-subproblem kernel.

The previous engine ran one attention kernel per subproblem and used CUDA
streams to overlap the gaps between them; the profiler showed the kernels
themselves never overlap (each fills the device), so `max_parallel` only ever
bought the gaps. This engine makes the batching native to the kernel instead:

* a **wave** is a set of subproblems whose token count fits the memory budget;
* the ``cqsa_native`` extension runs the CQS forward for the whole wave in ONE
  launch (``fwd_wave``): every subproblem is a "sequence" of a varlen batch,
  with its own group bits, tile summaries and -- when the inputs are device
  resident -- a block map that lets the kernel read the ORIGINAL Q/K/V in place
  (no gather, no per-subproblem copy);
* one deterministic merge kernel (``wave_merge``) folds all of the wave's
  ``(out_i, lse_i)`` into the max-shifted accumulator, in a fixed order
  (packed rows sorted by global token, stable), so the result is exact and
  reproducible with no atomics;
* the backward runs the same way (``bwd_wave`` + ``wave_scatter_add``) on the
  global log-sum-exp, so per-subproblem gradients simply add.

Memory stays linear: the wave's fp32 partial outputs are the only per-wave
allocation, and the wave size is chosen from the budget. With host-resident
Q/K/V a wave's chunks are streamed into a device chunk pool first.
"""
from __future__ import annotations

import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import torch

from .stable_stream import SubproblemTask, build_tasks, effective_free_bytes, is_oom, _GIB, _cpu_threads

# torch CPU ops fork one OpenMP thread per core; on an 80-core host a 19 MB copy then takes
# 96 ms instead of 0.2 ms (measured). Host-side copies/gathers run under this cap.
CPU_THREADS = int(os.environ.get("CQSA_CPU_THREADS", "8"))
# Default cap on packed tokens per wave. Batching pays for the launch gap and the tail of
# each subproblem's grid, which is a fixed cost per subproblem; past a few million packed
# tokens per launch nothing is left to gain, while the wave's fp32 partial outputs
# (H*D*4 bytes per packed token) are its memory cost. 2M tokens = 4 GiB at H=8, D=64.
DEFAULT_MAX_WAVE_TOKENS = int(os.environ.get("CQSA_MAX_WAVE_TOKENS", str(2 << 20)))
MIN_WAVE_BUDGET_TOKENS = int(os.environ.get("CQSA_MIN_WAVE_TOKENS", 1 << 19))   # floor of the default wave budget


def merge_async_enabled() -> bool:
    """acc=CPU: fold each wave's partial on a worker thread while the next wave runs (CQSA_WAVE_MERGE_ASYNC=0
    keeps the synchronous fold for A/B). Read per call so a test can flip it without a re-import."""
    return os.environ.get("CQSA_WAVE_MERGE_ASYNC", "1") != "0"


def fold_workers() -> int:
    """acc=CPU asynchronous fold: worker threads, each owning a fixed range of token ids (CQSA_WAVE_FOLD_WORKERS,
    default 1; read per call for A/B). Read per call so a harness can flip it without a re-import."""
    return max(1, int(os.environ.get("CQSA_WAVE_FOLD_WORKERS", "1")))


from .reference import chunk_layout
from .progress import Progress, verbose_enabled, describe_call, expected_seconds

SEG_ALIGN = 128
_ext = None


def native_ext():
    """The ``cqsa_native`` extension (name overridable with CQSA_NATIVE_MODULE)."""
    global _ext
    if _ext is None:
        import importlib
        _ext = importlib.import_module(os.environ.get("CQSA_NATIVE_MODULE", "cqsa_native"))
    return _ext


def native_available() -> bool:
    try:
        native_ext()
        return True
    except ImportError:
        return False


_SUPPORT: dict = {}


def native_supports(dtype, D: int, device="cuda") -> bool:
    """Whether the compiled ``cqsa_native`` has kernels for this dtype and head dim.

    The build's kernel set is not recorded in the extension, so this probes once per
    (dtype, D) with a 128-token wave and caches the answer; a build without the
    instantiation fails the probe with a clear TORCH_CHECK message."""
    key = (str(dtype), int(D))
    if key in _SUPPORT:
        return _SUPPORT[key]
    ok = False
    try:
        ext = native_ext()
        dev = torch.device(device)
        q = torch.zeros((128, 1, int(D)), device=dev, dtype=dtype)
        bits = torch.zeros(128, dtype=torch.int64, device=dev)
        summ = torch.zeros(2, dtype=torch.int64, device=dev)
        cu = torch.tensor([0, 128], dtype=torch.int32, device=dev)
        ext.fwd_wave(q, q, q, cu, 128, 128, bits, summ, summ, cu, None, None, 0, float(D) ** -0.5, True,
                     uniform_S=128, uniform_W=1)
        ok = True
    except Exception:
        ok = False
    _SUPPORT[key] = ok
    return ok


def triton_wave_available() -> bool:
    try:
        from . import triton_kernel  # noqa: F401
        return True
    except Exception:
        return False


def resolve_wave_kernel(kernel: str, dtype, D: int) -> str:
    """'auto' -> 'cuda' when cqsa_native is importable and built for (dtype, D), else 'triton'."""
    k = (kernel or "auto").lower().replace("-", "_").replace("wave_", "")
    if k == "auto":
        if native_available() and native_supports(dtype, int(D)):
            return "cuda"
        if triton_wave_available():
            return "triton"
        raise RuntimeError("no wave kernel available: build cqsa_native or pip install triton")
    if k == "cuda":
        if not native_available():
            raise RuntimeError("wave kernel 'cuda' needs the cqsa_native extension")
        return "cuda"
    if k == "triton":
        if not triton_wave_available():
            raise RuntimeError("wave kernel 'triton' needs triton (pip install triton)")
        return "triton"
    raise ValueError(f"wave kernel must be 'auto', 'cuda' or 'triton' (got {kernel!r})")


# ---------------------------------------------------------------------------
# Tasks (shape-only; cached)
# ---------------------------------------------------------------------------
_TASKS: dict = {}


def wave_tasks(N: int, itr: int, c: int, interest_set: Sequence[int], *, H: int, D: int, itemsize: int) -> list[SubproblemTask]:
    key = (int(N), int(itr), int(c), tuple(int(x) for x in interest_set))
    t = _TASKS.get(key)
    if t is None:
        t = build_tasks(int(N), int(itr), B=1, H=H, D=D, itemsize=itemsize, sorted_gather=True, pin=False,
                        c=int(c), interest_set=tuple(interest_set), seg_align=SEG_ALIGN)
        if len(_TASKS) > 16:
            _TASKS.pop(next(iter(_TASKS)))
        _TASKS[key] = t
    return t


def level1_chunks(task: SubproblemTask, c: int, interest_set: Sequence[int]) -> list[int]:
    """The level-1 chunks a task's tokens live in (its owner's quorum)."""
    return sorted({(int(task.path[0]) + int(off)) % int(c) for off in interest_set})


# ---------------------------------------------------------------------------
# Wave tables
# ---------------------------------------------------------------------------
@dataclass
class WaveTables:
    tasks: list
    S: int                      # > 0: uniform layout (W subproblems padded to S rows, causal); 0: varlen layout
    W: int
    cu: torch.Tensor            # [W+1] int32 (device)   (varlen)
    max_L: int
    total: int                  # packed rows (varlen: sum L; uniform: W*S)
    bits: torch.Tensor          # [total] int64
    blk_or: torch.Tensor
    blk_and: torch.Tensor
    blk_cu: torch.Tensor        # [W+1] int32 (varlen)
    tok: torch.Tensor           # [total] int64 global token of every packed row (uniform: -1 on padding)
    order: torch.Tensor         # [n_real] int64 packed rows sorted by token (stable), padding excluded
    seg: torch.Tensor           # [U+1] int32
    uniq: torch.Tensor          # [U] int64
    seg_base: torch.Tensor      # [sum blocks] int32 GLOBAL row of every 128-block (CPU; uniform: padded)
    bb_cu: torch.Tensor         # [W+1] int32 (device, varlen)
    bb_cu_cpu: np.ndarray = field(default=None)


def build_wave_tables(tasks: Sequence[SubproblemTask], device, uniform: bool = False) -> WaveTables:
    """Pack the CQS tables of a wave.

    ``uniform=True`` (causal forward): every subproblem padded to S = ceil(max_L/128)*128
    rows; the kernel then runs as an ordinary batch of W sequences whose tables sit at
    ``w * stride`` (see Flash_fwd_params::cqs_wave_tok_stride). Padding tokens carry zero
    bits and token id -1 (excluded from the merge); padding blocks of the block map alias
    the subproblem's last real block. ``uniform=False``: FlashAttention's varlen layout."""
    # numpy for the host-side concatenations: torch CPU ops fork 80 OpenMP threads per call here,
    # which cost ~200 ms per cat on 56K-element tensors (1.1 s per wave, measured; kernel 158 ms).
    npcat = lambda xs: torch.from_numpy(np.ascontiguousarray(np.concatenate([x.numpy() for x in xs])))
    sizes = [int(t.local_size) for t in tasks]
    W = len(tasks); max_L = max(sizes)
    sb = [t.extra["seg_base"] for t in tasks]
    if uniform:
        S = -(-max_L // SEG_ALIGN) * SEG_ALIGN
        nblk, nbb = S // 64, S // SEG_ALIGN
        bits_np = np.zeros(W * S, dtype=np.int64); or_np = np.zeros(W * nblk, dtype=np.int64); and_np = np.zeros(W * nblk, dtype=np.int64)
        tok_np = np.full(W * S, -1, dtype=np.int64); bb_np = np.zeros(W * nbb, dtype=np.int32)
        for w, t in enumerate(tasks):
            L = sizes[w]
            bits_np[w * S: w * S + L] = t.group_bits.numpy()
            o = t.extra["blk_or"].numpy(); a = t.extra["blk_and"].numpy()
            or_np[w * nblk: w * nblk + o.shape[0]] = o; and_np[w * nblk: w * nblk + a.shape[0]] = a
            tok_np[w * S: w * S + L] = t.token_ids.numpy()
            b = sb[w].numpy(); bb_np[w * nbb: w * nbb + b.shape[0]] = b; bb_np[w * nbb + b.shape[0]: (w + 1) * nbb] = b[-1]
        bits = torch.from_numpy(bits_np).to(device, non_blocking=True)
        blk_or = torch.from_numpy(or_np).to(device, non_blocking=True); blk_and = torch.from_numpy(and_np).to(device, non_blocking=True)
        tok_cpu = torch.from_numpy(tok_np); seg_base = torch.from_numpy(bb_np)
        cu_np = np.arange(W + 1, dtype=np.int64) * S; blk_cu_np = (np.arange(W + 1) * nblk).astype(np.int32)
        bb_cu_np = (np.arange(W + 1) * nbb).astype(np.int32); total = W * S
    else:
        S = 0
        cu_np = np.zeros(W + 1, dtype=np.int64); cu_np[1:] = np.cumsum(sizes)
        bits = npcat([t.group_bits for t in tasks]).to(device, non_blocking=True)
        blk_or = npcat([t.extra["blk_or"] for t in tasks]).to(device, non_blocking=True)
        blk_and = npcat([t.extra["blk_and"] for t in tasks]).to(device, non_blocking=True)
        blk_cu_np = np.zeros(W + 1, dtype=np.int32); blk_cu_np[1:] = np.cumsum([int(t.extra["blk_or"].numel()) for t in tasks])
        tok_cpu = npcat([t.token_ids for t in tasks])
        seg_base = npcat(sb) if sb[0] is not None else None
        bb_cu_np = np.zeros(W + 1, dtype=np.int32)
        if seg_base is not None:
            bb_cu_np[1:] = np.cumsum([int(x.numel()) for x in sb])
        total = int(cu_np[-1])
    tok = tok_cpu.to(device, non_blocking=True)
    # stable sort keeps the rows of one token in subproblem order -> deterministic merge order
    srt, order = torch.sort(tok, stable=True)
    if uniform:
        n_pad = W * S - sum(sizes)
        srt, order = srt[n_pad:], order[n_pad:]        # padding rows (token -1) sort first
    uniq, counts = torch.unique_consecutive(srt, return_counts=True)
    seg = torch.zeros(uniq.numel() + 1, dtype=torch.int32, device=device)
    seg[1:] = torch.cumsum(counts, 0).to(torch.int32)
    return WaveTables(
        tasks=list(tasks), S=S, W=W, cu=torch.from_numpy(cu_np.astype(np.int32)).to(device), max_L=max_L, total=int(total),
        bits=bits, blk_or=blk_or, blk_and=blk_and, blk_cu=torch.from_numpy(blk_cu_np).to(device),
        tok=tok, order=order.contiguous(), seg=seg, uniq=uniq, seg_base=seg_base,
        bb_cu=torch.from_numpy(bb_cu_np).to(device), bb_cu_cpu=bb_cu_np)


def block_base_for(tables: WaveTables, row_of_global) -> torch.Tensor:
    """Block map for the kernel: the row (in the tensor it reads) of every 128-block.

    ``row_of_global`` maps global rows (int64 CPU tensor) to rows of that tensor (identity
    for device-resident inputs, pool rows for a chunk pool). Uniform layout: absolute rows.
    Varlen layout: relative to the subproblem's packed offset."""
    rows = row_of_global(tables.seg_base.to(torch.int64)).numpy()
    if tables.S > 0:
        return torch.from_numpy(rows.astype(np.int32))
    cu = np.concatenate([[0], np.cumsum([int(t.local_size) for t in tables.tasks])[:-1]]).astype(np.int64)
    counts = np.diff(tables.bb_cu_cpu).astype(np.int64)
    rel = rows - np.repeat(cu, counts)
    return torch.from_numpy(rel.astype(np.int32))


block_base_relative = block_base_for   # older name


# ---------------------------------------------------------------------------
# Wave planning
# ---------------------------------------------------------------------------
def fwd_bytes_per_token(H: int, D: int) -> int:
    # fp32 partial out + lse + bits + tok/order/sort scratch
    return H * D * 4 + H * 4 + 8 + 8 * 4


def bwd_bytes_per_token(H: int, D: int, itemsize: int) -> int:
    Dr = -(-D // 32) * 32
    # packed q,k,v,dout + dq,dk,dv + fp32 dq_accum + lse/dpsum + indices
    return 7 * H * D * itemsize + H * Dr * 4 + H * 8 + 8 * 4


def default_wave_tokens(N: int) -> int:
    """The default packed-token budget per wave (v2.2.1): min(DEFAULT_MAX_WAVE_TOKENS, max(MIN_WAVE_BUDGET_TOKENS, N)).
    The wave engine's device peak is a*N + b*W_wave (profile sweep); a cap of about N tokens cost <= 1.3% of
    time on an A100-SXM4-80GB (131K-512K, itr=1) while cutting the packed term by two thirds below the 2M
    saturation. Small calls still fit in one wave."""
    return min(DEFAULT_MAX_WAVE_TOKENS, max(MIN_WAVE_BUDGET_TOKENS, int(N)))


def task_tokens_estimate(N: int, itr: int, c: int, l: int) -> int:
    """Upper estimate of the largest subproblem of (N, itr, c, l) without building the tasks: the block-aligned
    layout rounds every chunk up to 128 rows at each level, so a task is at most l blocks longer than the
    ideal N * (l/c)^itr per level."""
    L = float(N)
    for _ in range(int(itr)):
        L = -(-L * l / c // SEG_ALIGN) * SEG_ALIGN + (l - 1) * SEG_ALIGN
    return int(L)


def wave_token_ladder(N: int, itr: int, c: int, l: int, start: Optional[int] = None) -> list[int]:
    """Wave budgets the planner may choose from, largest first: the default budget halved down to the floor
    of one subproblem per wave (below that the budget changes nothing: plan_waves never splits a task)."""
    floor = task_tokens_estimate(N, itr, c, l)
    w = int(default_wave_tokens(N) if start is None else start)
    ladder = []
    while w > floor:
        ladder.append(w); w //= 2
    ladder.append(floor)
    return ladder


def wave_token_floor(N: int, itr: int, c: int, interest_set: Sequence[int], *, H: int, D: int, itemsize: int) -> int:
    """The exact floor of the wave budget: the largest task's tokens (from the cached task list)."""
    return max(int(t.local_size) for t in wave_tasks(N, itr, c, interest_set, H=H, D=D, itemsize=itemsize))


def in_place_possible(N: int, causal: bool) -> bool:
    """Whether the CUDA wave kernel may read device-resident Q/K/V in place. The causal (uniform) layout loads
    full 128-row tiles without predication, so the ragged last block of the sequence would run past the
    tensor unless N is 128-aligned; the non-causal (varlen) layout predicates and is always safe."""
    return int(N) % SEG_ALIGN == 0 or not causal


def estimate_wave_peak_bytes(N: int, wave_tokens: int, *, B: int, H: int, D: int, itemsize: int, itr: int, c: int, l: int,
                             stream_from_host: bool, accumulate_on_gpu: bool, gather: bool,
                             direction: str = "fwd", out_bytes_per_el: int = 4) -> int:
    """
    Predicted peak device memory of the wave engine: ``a*N + b*W`` like the measured fits
    (results/profile_sweep/fits.json: itr=1 acc=GPU 7.149 GiB/M tokens + 1.78 GiB/M packed tokens;
    this analytic count gives 7.0 + 2.07 at H=8, D=64, fp16 -- the a-term within 2%, the wave term
    16% conservative because the sort scratch it counts is transient).

      resident   Q/K/V when they are read in place (device-resident inputs), or the chunk pool of the
                 level-1 chunks a wave needs (host-resident: ``min(c, l * tasks_per_wave)`` slots of one
                 aligned chunk each); the fp32 output and accumulator when the accumulator is on the
                 device, else the delivered output (``out_bytes_per_el``, as the classic estimate).
      per wave   ``fwd_bytes_per_token`` per packed token (fp32 partial output, lse, tables, merge
                 scratch), plus the per-token merge accumulators with a host accumulator, plus the
                 three gathered fp16 copies when ``gather`` (in place: none).
      backward   the wave backward keeps fp32 dq/dk/dv and dpsum resident next to q/k/v/dout/out and
                 always gathers (``bwd_bytes_per_token``); the peak of the fwd+bwd step is its own.

    ``wave_tokens`` below one task's tokens is raised to it: a wave holds at least one subproblem.
    """
    el = int(N) * int(B) * int(H) * int(D)
    L = task_tokens_estimate(N, itr, c, l)
    W = max(int(wave_tokens), L)
    if direction == "bwd":
        floor = (3 * 4) * el + 2 * N * B * H * 4                     # fp32 gradients, lse + dpsum
        if not stream_from_host:
            floor += (4 * itemsize + 4) * el                          # q, k, v, dout and the saved fp32 out
        floor += 3 * itemsize * el                                    # the gradients cast to the input dtype at the end
        return int(floor + W * bwd_bytes_per_token(H, D, itemsize))
    floor = 0
    if not stream_from_host:
        floor += 3 * itemsize * el
    else:
        chunk = -(-math.ceil(N / c) // SEG_ALIGN) * SEG_ALIGN
        slots = min(int(c), int(l) * -(-W // L))
        floor += slots * chunk * 3 * H * D * itemsize
    if accumulate_on_gpu:
        floor += 4 * el + 4 * N * H * D + 8 * N * H
    else:
        floor += int(out_bytes_per_el) * el
    per_tok = fwd_bytes_per_token(H, D)
    if not accumulate_on_gpu:
        per_tok += 4 * H * D + 8 * H
    if gather:
        per_tok += 3 * H * D * itemsize
    return int(floor + W * per_tok)


def plan_waves(tasks: Sequence[SubproblemTask], max_tokens: int, max_subproblems: Optional[int] = None) -> list[list[SubproblemTask]]:
    """Greedy contiguous grouping under a token budget (and an optional count cap)."""
    waves, cur, cur_tok = [], [], 0
    for t in tasks:
        L = int(t.local_size)
        if cur and (cur_tok + L > max_tokens or (max_subproblems and len(cur) >= max_subproblems)):
            waves.append(cur); cur, cur_tok = [], 0
        cur.append(t); cur_tok += L
    if cur:
        waves.append(cur)
    return waves


def _budget_tokens(device, per_token: int, reserved: int, fraction: float, cap: Optional[int]) -> int:
    free = effective_free_bytes(device)
    avail = int(free * fraction) - int(reserved)
    n = max(SEG_ALIGN, avail // max(1, per_token))
    n = min(n, int(cap) if cap is not None else DEFAULT_MAX_WAVE_TOKENS)
    return int(n)


# ---------------------------------------------------------------------------
# Forward
# ---------------------------------------------------------------------------
MERGE_CHUNK_TOKENS = 1 << 18


def merge_rows_torch(acc, acc_l, acc_m, out_pack, lse_pack, order, seg, uniq):
    """
    The wave merge in plain torch (the Triton path; no extension): rows ``order`` are grouped
    by token (``seg`` segments, tokens ``uniq``); each token's rows are combined in a fixed
    order into a [maxc]-wide gather, then folded into the accumulators with the max-shifted
    formula. Deterministic (no atomics). out_pack [total, H, D] fp32, lse_pack [total, H] fp32.
    """
    dev = out_pack.device
    n = int(order.numel()); U = int(uniq.numel())
    if n == 0:
        return
    H, D = out_pack.shape[1], out_pack.shape[2]
    counts = (seg[1:] - seg[:-1]).to(torch.int64)
    maxc = int(counts.max().item())
    sentinel = out_pack.shape[0]
    out_ext = torch.cat([out_pack, out_pack.new_zeros((1, H, D))])
    lse_ext = torch.cat([lse_pack, lse_pack.new_full((1, H), float("-inf"))])
    row_u = torch.repeat_interleave(torch.arange(U, device=dev), counts)
    col = torch.arange(n, device=dev) - torch.repeat_interleave(seg[:-1].to(torch.int64), counts)
    idx = torch.full((U, maxc), sentinel, dtype=torch.int64, device=dev)
    idx[row_u, col] = order
    for u0 in range(0, U, MERGE_CHUNK_TOKENS):
        ix = idx[u0:u0 + MERGE_CHUNK_TOKENS]                           # [u, maxc]
        tok = uniq[u0:u0 + MERGE_CHUNK_TOKENS]
        lse_g = lse_ext[ix]                                            # [u, maxc, H]
        m_w = lse_g.amax(dim=1)                                        # [u, H]
        m_safe = torch.where(torch.isfinite(m_w), m_w, torch.zeros_like(m_w))
        w = torch.exp(lse_g - m_safe[:, None, :])                      # sentinel / -inf rows -> 0
        l_w = w.sum(1)
        acc_w = torch.einsum("ujh,ujhd->uhd", w, out_ext[ix])
        m_old = acc_m[tok]
        m_new = torch.maximum(m_old, m_w)
        m_new_safe = torch.where(torch.isfinite(m_new), m_new, torch.zeros_like(m_new))
        w_old = torch.exp(m_old - m_new_safe); w_new = torch.exp(m_w - m_new_safe)
        acc[tok] = acc[tok] * w_old.unsqueeze(-1) + acc_w * w_new.unsqueeze(-1)
        acc_l[tok] = acc_l[tok] * w_old + l_w * w_new
        acc_m[tok] = m_new


def _fold_host(acc_h, l_h, m_h, idx, sa, sl, sm):
    """Fold one wave's merged partial (``sa, sl, sm`` over the tokens ``idx``) into the host accumulators
    with the max-shifted formula the device merge uses; runs under the CPU thread cap on whichever
    thread calls it (the OpenMP team is per thread, so the worker sets its own)."""
    with _cpu_threads(CPU_THREADS):
        m_old = m_h[idx]
        m_new = torch.maximum(m_old, sm)
        w_old = torch.exp(m_old - m_new)           # 0 where the token was untouched so far (m_old = -inf)
        w_new = torch.exp(sm - m_new)              # every uniq token has at least one row: sm is finite
        acc_h[idx] = acc_h[idx] * w_old.unsqueeze(-1) + sa * w_new.unsqueeze(-1)
        l_h[idx] = l_h[idx] * w_old + sl * w_new
        m_h[idx] = m_new


class _HostMerge:
    """
    acc=CPU merge of the waves: ``merge_fn(acc, l, m, uniq)`` merges a wave's packed rows on the device
    into accumulators over the U tokens it touches (``tb.uniq`` remapped to 0..U-1); the triple lands
    in a pinned slot by a non-blocking D2H and is folded into the host accumulators by ``_fold_host``,
    so the result equals the acc=GPU path up to fp32 summation order.

    Asynchronous mode (the classic engine's CQSA_FWD_MERGE_ASYNC design): the fold of wave k runs on
    a worker thread after the wave's D2H event, while the main thread streams and launches wave k+1;
    the GPU no longer idles through the fold. With ``workers`` > 1 each worker owns a fixed range of
    token ids and folds that part of every wave (``tb.uniq`` is sorted, so a part is a slice): the
    fold is elementwise per token and every token is folded by one worker in submission order, so the
    fold order per token is the wave order and the result is bit-identical to the synchronous mode for
    any worker count. (Workers taking whole waves would race on the tokens two waves share.) Two
    pinned slots bound the waves in flight: a slot is reused only once every fold that read it has
    returned, which is also where a fold's exception resurfaces on the caller's thread.
    ``fold_s``/``wait_s``: seconds spent folding (summed over workers) and blocked waiting for a slot.
    """

    def __init__(self, acc_h, l_h, m_h, u_max: int, dev, async_merge: bool, workers: int = 1):
        H, D = acc_h.shape[1], acc_h.shape[2]
        self.acc_h, self.l_h, self.m_h, self.dev = acc_h, l_h, m_h, dev
        n_slots = 2 if async_merge else 1
        # pinned at allocation: .pin_memory() on a pageable tensor would first allocate a pageable copy
        self.slots = [(torch.empty((u_max, H, D), dtype=torch.float32, pin_memory=True),
                       torch.empty((u_max, H), dtype=torch.float32, pin_memory=True),
                       torch.empty((u_max, H), dtype=torch.float32, pin_memory=True),
                       torch.empty(u_max, dtype=torch.int64, pin_memory=True)) for _ in range(n_slots)]
        self.pending = [[] for _ in range(n_slots)]      # futures of the folds reading each slot (async)
        n_workers = max(1, int(workers)) if async_merge else 0
        self.pools = [ThreadPoolExecutor(max_workers=1) for _ in range(n_workers)]
        N = acc_h.shape[0]
        self.bounds = torch.tensor([N * j // max(1, n_workers) for j in range(n_workers + 1)], dtype=torch.int64)
        self.fold_s = [0.0] * max(1, n_workers)
        self.wait_s = 0.0
        self.n = 0

    def _wait(self, slot: int):
        futures, self.pending[slot] = self.pending[slot], []
        t0 = time.perf_counter()
        for f in futures:
            f.result()                        # re-raises the fold's exception here, not in the worker
        self.wait_s += time.perf_counter() - t0

    def _fold(self, j: int, landed, si, sa, sl, sm):
        landed.synchronize()
        t0 = time.perf_counter()
        if len(self.pools) > 1:
            a, b = torch.searchsorted(si, self.bounds[j:j + 2]).tolist()
            si, sa, sl, sm = si[a:b], sa[a:b], sl[a:b], sm[a:b]
        _fold_host(self.acc_h, self.l_h, self.m_h, si, sa, sl, sm)
        self.fold_s[j] += time.perf_counter() - t0

    def merge(self, merge_fn, tb: WaveTables):
        slot = self.n % len(self.slots); self.n += 1
        self._wait(slot)
        U = int(tb.uniq.numel())
        H, D = self.acc_h.shape[1], self.acc_h.shape[2]
        dev = self.dev
        acc_w = torch.zeros((U, H, D), device=dev, dtype=torch.float32)
        l_w = torch.zeros((U, H), device=dev, dtype=torch.float32)
        m_w = torch.full((U, H), float("-inf"), device=dev, dtype=torch.float32)
        local = torch.arange(U, device=dev, dtype=tb.uniq.dtype)
        merge_fn(acc_w, l_w, m_w, local)
        sa, sl, sm, si = (t[:U] for t in self.slots[slot])
        sa.copy_(acc_w, non_blocking=True); sl.copy_(l_w, non_blocking=True); sm.copy_(m_w, non_blocking=True)
        si.copy_(tb.uniq, non_blocking=True)
        landed = torch.cuda.Event(); landed.record(torch.cuda.current_stream(dev))
        del acc_w, l_w, m_w, local           # stream-ordered: the allocator reuses them only after the D2H
        if not self.pools:
            self._fold(0, landed, si, sa, sl, sm)
        else:
            self.pending[slot] = [pool.submit(self._fold, j, landed, si, sa, sl, sm) for j, pool in enumerate(self.pools)]

    def drain(self):
        """Every submitted fold has been applied (or its exception raised)."""
        for slot in range(len(self.slots)):
            self._wait(slot)

    def close(self):
        """Release the workers and the pinned slots; never raises (used on the error path too)."""
        for pool in self.pools:
            pool.shutdown(wait=True)
        self.pools = []
        self.slots = []


def wave_forward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, causal: bool = True, scale: Optional[float] = None,
                 itr: int = 1, c: int = 7, interest_set: Sequence[int] = (0, 1, 3),
                 max_wave_tokens: Optional[int] = None, max_wave_subproblems: Optional[int] = None,
                 memory_fraction: float = 0.85, device=None, pool_slots: Optional[int] = None,
                 accumulate_on_gpu: bool = True, kernel: str = "auto", gather: Optional[bool] = None,
                 verbose: Optional[bool] = None):
    """
    Exact attention by CQS decomposition on the wave kernel: ``kernel="cuda"`` (cqsa_native,
    the fast path), ``"triton"`` (the same wave layout on the Triton kernel, no build), or
    ``"auto"`` (CUDA when the extension is built for this dtype and head dim, else Triton).

    q/k/v: ``[B, H, N, D]`` fp16/bf16, on the device or on the host (streamed into a device
    chunk pool wave by wave). Device-resident inputs are either gathered into three packed
    copies per wave (``gather=True``; 3-6% faster kernel time, layout_ab) or read in place
    through the block map (``gather=False``; no copies, only when ``in_place_possible``);
    ``None`` reads in place whenever possible. The Triton kernel and the chunk pool have no
    choice (gathered / in place). ``info["gather"]`` records what ran.
    Returns ``(out [B, H, N, D] fp32, info)``; ``info["lse"]`` is ``[B, H, N]``.

    ``accumulate_on_gpu=False`` (the classic engine's acc=CPU): every wave is merged on
    the device into accumulators over the tokens it touches only, which are then folded
    into fp32 accumulators in host memory with the same max-shifted merge; ``out`` and
    ``lse`` come back on the host and the device never holds the full fp32 output.
    """
    B, H, N, D = q.shape
    kern = resolve_wave_kernel(kernel, q.dtype, D)
    ext = native_ext() if kern == "cuda" else None
    if kern == "triton":
        from .triton_kernel import cqs_attention_forward_wave
    dev = torch.device(device) if device is not None else (q.device if q.is_cuda else torch.device("cuda"))
    scale = float(D ** -0.5 if scale is None else scale)
    tasks = wave_tasks(N, itr, c, interest_set, H=H, D=D, itemsize=q.element_size())
    on_device = q.is_cuda
    acc_gpu = bool(accumulate_on_gpu)
    # device accumulators: the whole output (acc=GPU), or one wave's touched tokens (acc=CPU)
    acc_bytes = (N * H * D * 4 + 2 * N * H * 4) if acc_gpu else 0
    if kern == "triton" or not on_device:
        gather = kern == "triton"                # the Triton kernel has no block map; the pool is read in place
    elif gather is None or (not gather and not in_place_possible(N, causal)):
        gather = not in_place_possible(N, causal)
    per_tok = fwd_bytes_per_token(H, D) + (0 if acc_gpu else 4 * H * D + 8 * H)
    if on_device and gather:
        per_tok += 3 * H * D * q.element_size()   # the three packed copies of a gathered wave

    if max_wave_tokens is None:
        max_wave_tokens = default_wave_tokens(N)
    if on_device:
        max_tokens = _budget_tokens(dev, per_tok, acc_bytes + (256 << 20), memory_fraction, max_wave_tokens)
        waves = plan_waves(tasks, max_tokens, max_wave_subproblems)
        pool = None
    else:
        pool = ChunkPool(N, c, H, D, q.dtype, dev)
        n_slots = pool_slots
        if n_slots is None:
            # split the budget between the pool and the packed partial outputs:
            # a wave of w subproblems needs about l*w chunks (fewer when they share) and w*L tokens
            L = max(int(t.local_size) for t in tasks)
            l = len(interest_set)
            free = int(effective_free_bytes(dev) * memory_fraction) - acc_bytes - (256 << 20)
            per_task = l * pool.slot_bytes + L * per_tok
            w = max(1, min(len(tasks), free // max(1, per_task)))
            n_slots = min(c, l * w)
        pool.allocate(n_slots)
        max_tokens = _budget_tokens(dev, per_tok, acc_bytes + pool.bytes + (256 << 20), memory_fraction, max_wave_tokens)
        waves = pool.plan(tasks, interest_set, max_tokens, max_wave_subproblems)

    acc_dev = dev if acc_gpu else torch.device("cpu")
    out = torch.empty((B, H, N, D), device=acc_dev, dtype=torch.float32)
    lse = torch.empty((B, H, N), device=acc_dev, dtype=torch.float32)
    acc = torch.zeros((N, H, D), device=acc_dev, dtype=torch.float32)
    acc_l = torch.zeros((N, H), device=acc_dev, dtype=torch.float32)
    acc_m = torch.empty((N, H), device=acc_dev, dtype=torch.float32)
    merger = None
    if not acc_gpu:
        # pinned slots for a wave's merged partial, sized to the most tokens a wave can touch: a task's
        # tokens lie in its level-1 chunks, so the union of those chunks bounds U tighter than sum L
        sizes = chunk_layout(N, c, SEG_ALIGN)[0]
        u_max = max(min(sum(int(t.local_size) for t in w),
                        sum(sizes[j] for j in set().union(*[set(level1_chunks(t, c, interest_set)) for t in w])))
                    for w in waves)
        merger = _HostMerge(acc, acc_l, acc_m, u_max, dev, merge_async_enabled(), fold_workers())
    info = dict(n_subproblems=len(tasks), n_waves=len(waves), wave_sizes=[len(w) for w in waves],
                wave_tokens=[sum(int(t.local_size) for t in w) for w in waves], max_wave_tokens=int(max_tokens),
                itr=int(itr), c=int(c), interest_set=tuple(interest_set), host_resident=not on_device,
                accumulate_on_gpu=acc_gpu, kernel=kern, pool_slots=(None if pool is None else pool.n_slots),
                gather=bool(gather), merge_async=(None if merger is None else bool(merger.pools)),
                fold_workers=(None if merger is None else len(merger.pools)))
    total_tokens = sum(int(t.local_size) for t in tasks) * B
    progress = Progress(
        total_tokens, enabled=verbose_enabled(verbose), unit="tok",
        banner=describe_call(what=f"forward (wave engine, {kern} kernel)", N=N, B=B, H=H, D=D, c=int(c), itr=int(itr),
                             n_tasks=len(tasks), causal=bool(causal), device=str(dev),
                             extra=f"{len(waves)} wave{'s' if len(waves) != 1 else ''} of {info['wave_sizes']} subproblems"
                                   + (f", chunk pool of {pool.n_slots} slots" if pool is not None else
                                      (", Q/K/V gathered per wave" if gather else ", Q/K/V read in place"))),
        expected_s=expected_seconds(N=N, B=B, H=H, D=D, itr=int(itr), c=int(c), causal=bool(causal),
                                    stream_from_host=not on_device))
    q_tm = k_tm = v_tm = None
    if not on_device:
        # token-major host views (a copy only if the storage is not already token-major)
        with _cpu_threads(CPU_THREADS):
            q_tm, k_tm, v_tm = (t.transpose(1, 2).contiguous() for t in (q, k, v))
    try:
      for b in range(B):
        acc.zero_(); acc_l.zero_(); acc_m.fill_(float("-inf"))
        if on_device:
            qt, kt, vt = (t[b].transpose(0, 1) for t in (q, k, v))       # [N, H, D] strided views, read in place
        for wave in waves:
            if kern == "triton":
                # uniform layout for both causal and non-causal (the kernel masks keys by length);
                # the wave is gathered into a packed tensor (the Triton kernel has no block map)
                tb = build_wave_tables(wave, dev, uniform=True)
                valid = tb.tok >= 0
                pos = torch.nonzero(valid).squeeze(1)
                if on_device:
                    src = tb.tok[valid]
                    srcs = (qt, kt, vt)
                else:
                    with _cpu_threads(CPU_THREADS):
                        pool.load(wave, interest_set, b, q_tm, k_tm, v_tm)
                    src = pool.rows_of_global(tb.tok[valid].cpu()).to(dev, non_blocking=True)
                    srcs = (pool.q, pool.k, pool.v)
                def gather_t(t):
                    g = torch.zeros((tb.total, H, D), device=dev, dtype=t.dtype)
                    g[pos] = t[src]
                    return g
                qs, ks, vs = (gather_t(t) for t in srcs)
                lens = torch.tensor([int(t.local_size) for t in wave], dtype=torch.int32, device=dev)
                out_pack, lse_pack = cqs_attention_forward_wave(qs, ks, vs, tb.bits, tb.blk_or, tb.blk_and, lens,
                                                                S=int(tb.S), causal=bool(causal), scale=scale)
                del qs, ks, vs
                merge_fn = lambda a, l, m, uniq: merge_rows_torch(a, l, m, out_pack, lse_pack, tb.order, tb.seg, uniq)
                if acc_gpu:
                    merge_fn(acc, acc_l, acc_m, tb.uniq)
                else:
                    merger.merge(merge_fn, tb)
                if progress.enabled:
                    torch.cuda.synchronize(dev)
                progress.update(sum(int(t.local_size) for t in wave))
                del out_pack, lse_pack, tb
                continue
            # causal: uniform layout (batch of W padded sequences); non-causal: varlen layout
            tb = build_wave_tables(wave, dev, uniform=bool(causal))
            if on_device and not gather:
                # in place: the block map addresses the original tensor (see in_place_possible)
                bb = block_base_for(tb, lambda g: g).to(dev, non_blocking=True)
                qs, ks, vs = qt, kt, vt
            elif on_device:
                # gathered wave with finite (zero) padding rows
                valid = tb.tok >= 0
                pos = torch.nonzero(valid).squeeze(1); src = tb.tok[valid]
                def gather_rows(t):
                    g = torch.zeros((tb.total, H, D), device=dev, dtype=t.dtype)
                    g[pos] = t[src]
                    return g
                qs, ks, vs = gather_rows(qt), gather_rows(kt), gather_rows(vt); bb = None
            else:
                with _cpu_threads(CPU_THREADS):
                    pool.load(wave, interest_set, b, q_tm, k_tm, v_tm)
                bb = block_base_for(tb, pool.rows_of_global).to(dev, non_blocking=True)
                qs, ks, vs = pool.q, pool.k, pool.v
            out_pack, lse_pack = ext.fwd_wave(qs, ks, vs, tb.cu, int(tb.max_L), int(tb.total), tb.bits, tb.blk_or, tb.blk_and,
                                              tb.blk_cu, bb, tb.bb_cu, SEG_ALIGN, scale, bool(causal),
                                              uniform_S=int(tb.S), uniform_W=int(tb.W))
            if acc_gpu:
                ext.wave_merge(acc, acc_l, acc_m, out_pack, lse_pack, tb.order, tb.seg, tb.uniq, int(tb.S))
            else:
                merge_fn = lambda a, l, m, uniq: ext.wave_merge(a, l, m, out_pack, lse_pack, tb.order, tb.seg, uniq, int(tb.S))
                merger.merge(merge_fn, tb)
            if progress.enabled:
                torch.cuda.synchronize(dev)      # the bar reports finished work, not queued launches
            progress.update(sum(int(t.local_size) for t in wave))
            del out_pack, lse_pack, tb, bb
        if merger is not None:
            merger.drain()                  # the host accumulators are complete for this batch
        with _cpu_threads(CPU_THREADS):
            out[b].copy_((acc / acc_l.clamp_min(1e-30).unsqueeze(-1)).transpose(0, 1))
            lse_b = acc_m + torch.log(acc_l.clamp_min(1e-30))
            lse[b].copy_(torch.where(torch.isfinite(acc_m), lse_b, torch.full_like(lse_b, float("-inf"))).transpose(0, 1))
    finally:
        if merger is not None:
            merger.close()                  # on an error the workers finish their queues, then exit
            info.update(merge_fold_s=sum(merger.fold_s), merge_wait_s=merger.wait_s)
    info["lse"] = lse
    del acc, acc_l, acc_m, merger
    if pool is not None:
        pool.release()
    progress.close(f"{len(tasks)} subproblems in {len(waves)} wave{'s' if len(waves) != 1 else ''} ({kern} kernel)")
    return out, info


# ---------------------------------------------------------------------------
# Backward
# ---------------------------------------------------------------------------
def bwd_wave_budget(device, N: int, H: int, D: int, itemsize: int, memory_fraction: float, cap: Optional[int]) -> int:
    """Packed tokens per backward wave: what the free memory holds next to the fp32 dq/dk/dv (+ dpsum),
    capped by ``cap`` (``DEFAULT_MAX_WAVE_TOKENS`` when None)."""
    grad_bytes = 3 * N * H * D * 4 + N * H * 4
    return _budget_tokens(device, bwd_bytes_per_token(H, D, itemsize), grad_bytes + (256 << 20), memory_fraction, cap)


def wave_backward(q, k, v, out, dout, lse, *, causal: bool = True, scale: Optional[float] = None,
                  itr: int = 1, c: int = 7, interest_set: Sequence[int] = (0, 1, 3),
                  max_wave_tokens: Optional[int] = None, max_wave_subproblems: Optional[int] = None,
                  memory_fraction: float = 0.85, device=None, verbose: Optional[bool] = None):
    """
    Exact attention backward on the native wave kernel (global-lse formulation).

    q/k/v/dout ``[B, H, N, D]`` (same dtype), out ``[B, H, N, D]`` (fp32 or the
    input dtype), lse ``[B, H, N]`` fp32 from ``wave_forward``. Device-resident
    inputs are gathered per wave on the device; host-resident ones are gathered
    on the host and streamed. Returns ``(dq, dk, dv)`` in the input dtype.
    """
    ext = native_ext()
    B, H, N, D = q.shape
    dev = torch.device(device) if device is not None else (q.device if q.is_cuda else torch.device("cuda"))
    scale = float(D ** -0.5 if scale is None else scale)
    itemsize = q.element_size()
    tasks = wave_tasks(N, itr, c, interest_set, H=H, D=D, itemsize=itemsize)
    on_device = q.is_cuda
    max_tokens = bwd_wave_budget(dev, N, H, D, itemsize, memory_fraction, max_wave_tokens)
    waves = plan_waves(tasks, max_tokens, max_wave_subproblems)
    # rowsum(dO * O) is a global per-token quantity: once, in fp32
    dpsum = torch.empty((B, H, N), device=dev, dtype=torch.float32)
    for b in range(B):
        o_b = out[b].to(dev, non_blocking=True)
        do_b = dout[b].to(dev, non_blocking=True)
        dpsum[b] = (o_b.float() * do_b.float()).sum(-1)
        del o_b, do_b
    dq = torch.zeros((B, N, H, D), device=dev, dtype=torch.float32)
    dk = torch.zeros((B, N, H, D), device=dev, dtype=torch.float32)
    dv = torch.zeros((B, N, H, D), device=dev, dtype=torch.float32)
    lse_d = lse.to(dev, non_blocking=True)
    q_tm = k_tm = v_tm = do_tm = None
    if not on_device:
        with _cpu_threads(CPU_THREADS):
            q_tm, k_tm, v_tm, do_tm = (t.transpose(1, 2).contiguous() for t in (q, k, v, dout))
    info = dict(n_subproblems=len(tasks), n_waves=len(waves), wave_sizes=[len(w) for w in waves], max_wave_tokens=int(max_tokens))
    progress = Progress(
        sum(int(t.local_size) for t in tasks) * B, enabled=verbose_enabled(verbose), unit="tok",
        banner=describe_call(what="backward (native wave kernel)", N=N, B=B, H=H, D=D, c=int(c), itr=int(itr),
                             n_tasks=len(tasks), causal=bool(causal), device=str(dev),
                             extra=f"{len(waves)} wave{'s' if len(waves) != 1 else ''} of {info['wave_sizes']} subproblems"),
        expected_s=expected_seconds(N=N, B=B, H=H, D=D, itr=int(itr), c=int(c), causal=bool(causal),
                                    direction="bwd", stream_from_host=not on_device))
    for b in range(B):
        if on_device:
            qt, kt, vt, dot = (t[b].transpose(0, 1) for t in (q, k, v, dout))     # [N, H, D] views
        for wave in waves:
            tb = build_wave_tables(wave, dev)
            W = len(wave)
            if on_device:
                qp, kp, vp, dop = (torch.index_select(t, 0, tb.tok) for t in (qt, kt, vt, dot))
            else:
                tok_cpu = torch.from_numpy(np.concatenate([t.token_ids.numpy() for t in wave]))
                with _cpu_threads(CPU_THREADS):
                    qp, kp, vp, dop = (torch.index_select(t[b], 0, tok_cpu).pin_memory().to(dev, non_blocking=True)
                                       for t in (q_tm, k_tm, v_tm, do_tm))
            lse_p = torch.index_select(lse_d[b], 1, tb.tok).contiguous()                 # [H, T]
            # dpsum in the kernel's padded varlen layout: subproblem w at cu[w] + 128*w
            cu_cpu = np.concatenate([[0], np.cumsum([int(t.local_size) for t in wave])])
            pos = torch.from_numpy(np.concatenate([np.arange(cu_cpu[w] + 128 * w, cu_cpu[w] + 128 * w + int(t.local_size), dtype=np.int64)
                                                   for w, t in enumerate(wave)])).to(dev, non_blocking=True)
            dps_p = torch.zeros((H, int(tb.total) + 128 * W), device=dev, dtype=torch.float32)
            dps_p.index_copy_(1, pos, torch.index_select(dpsum[b], 1, tb.tok))
            dq_p, dk_p, dv_p = ext.bwd_wave(dop, qp, kp, vp, lse_p, dps_p, tb.cu, int(tb.max_L), int(tb.total),
                                            tb.bits, tb.blk_or, tb.blk_and, tb.blk_cu, scale, bool(causal))
            del qp, kp, vp, dop, lse_p, dps_p, pos
            ext.wave_scatter_add(dq[b], dq_p, tb.order, tb.seg, tb.uniq)
            ext.wave_scatter_add(dk[b], dk_p, tb.order, tb.seg, tb.uniq)
            ext.wave_scatter_add(dv[b], dv_p, tb.order, tb.seg, tb.uniq)
            if progress.enabled:
                torch.cuda.synchronize(dev)
            progress.update(sum(int(t.local_size) for t in wave))
            del dq_p, dk_p, dv_p, tb
    res = tuple(g.transpose(1, 2).to(q.dtype) for g in (dq, dk, dv))
    del dq, dk, dv
    progress.close(f"{len(tasks)} subproblems in {len(waves)} wave{'s' if len(waves) != 1 else ''}")
    return res, info


# ---------------------------------------------------------------------------
# Autograd
# ---------------------------------------------------------------------------
def wave_backward_recovering(q, k, v, out, dout, lse, *, causal: bool, scale: Optional[float], itr: int, c: int,
                             interest_set: Sequence[int], max_wave_tokens: Optional[int] = None,
                             classic_fallback: bool = False, verbose: Optional[bool] = None):
    """
    ``wave_backward`` with the forward's out-of-memory ladder (design 5.2). The backward runs during
    ``loss.backward()``, outside ``attention()``'s own recovery, so it recovers here: on an OOM the wave
    budget is halved down to one subproblem per wave (the backward always gathers: there is no in-place
    rung); below that, with ``classic_fallback`` (``kernel="auto"``), the classic streamed backward runs
    from the same saved ``out``/``lse`` -- exact at any partition, every subproblem scores against the
    global lse -- with its fp32 gradients in host memory, and escalates the depth itself. A forced wave
    kernel re-raises instead. Returns ``((dq, dk, dv), record)``; ``record`` says which engine produced
    the gradients and what the recovery did.
    """
    B, H, N, D = q.shape
    dev = q.device if q.is_cuda else torch.device("cuda")
    itemsize = q.element_size()
    floor = wave_token_floor(N, itr, c, interest_set, H=H, D=D, itemsize=itemsize)
    vb = verbose_enabled(verbose)
    W, retries = max_wave_tokens, 0
    while True:
        W = bwd_wave_budget(dev, N, H, D, itemsize, 0.85, W)       # the budget this attempt runs with
        try:
            grads, info = wave_backward(q, k, v, out, dout, lse, causal=causal, scale=scale, itr=itr, c=c,
                                        interest_set=interest_set, max_wave_tokens=W, verbose=verbose)
            return grads, dict(engine="wave", wave_oom_retries=retries, max_wave_tokens=info["max_wave_tokens"],
                               n_waves=info["n_waves"])
        except Exception as exc:                                     # noqa: BLE001
            if not is_oom(exc):
                raise
            torch.cuda.empty_cache()
            retries += 1
            if W > floor:
                W = max(floor, W // 2)
                if vb:
                    print(f"Stream-CQSA: the wave backward ran out of memory; halving the wave budget to {W:,} tokens", flush=True)
                continue
            if not classic_fallback:
                raise
            break
    if vb:
        print("Stream-CQSA: the wave backward ran out of memory at one subproblem per wave; "
              "retrying on the streaming engine with host-memory gradients", flush=True)
    from .stable_stream import stream_cqsa_backward
    host = not q.is_cuda
    place = (lambda t: t.cpu()) if host else (lambda t: t.to(dev))
    bwd_info: dict = {}
    grads = stream_cqsa_backward(place(q), place(k), place(v), place(dout), place(out), place(lse), itr="auto",
                                 min_itr=int(itr), causal=causal, scale=scale, c=c, interest_set=tuple(interest_set),
                                 stream_from_host=host, accumulate_on_gpu=False, bwd_info=bwd_info, verbose=verbose)
    # cast where the fp32 gradient sits (host memory), then move the half-size result
    grads = tuple(g.to(q.dtype).to(q.device) for g in grads)
    return grads, dict(engine="classic", wave_oom_retries=retries, itr=bwd_info.get("itr_reached"),
                       depth_escalations=bwd_info.get("depth_escalations", 0))


class _WaveAttn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal, scale, itr, c, interest_set, bwd_itr, max_wave_tokens, gather, verbose,
                classic_fallback, bwd_info):
        out, info = wave_forward(q, k, v, causal=causal, scale=scale, itr=itr, c=c, interest_set=interest_set,
                                 max_wave_tokens=max_wave_tokens, gather=gather, verbose=verbose)
        ctx.save_for_backward(q, k, v, out, info["lse"])
        ctx.meta = (causal, scale, itr if bwd_itr is None else bwd_itr, c, tuple(interest_set), max_wave_tokens, verbose)
        ctx.classic_fallback, ctx.bwd_info = bool(classic_fallback), bwd_info
        if bwd_info is not None:
            bwd_info["forward_gather"] = info["gather"]              # what ran: a ragged causal N always gathers
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, out, lse = ctx.saved_tensors
        causal, scale, itr, c, interest_set, mwt, verbose = ctx.meta
        (dq, dk, dv), rec = wave_backward_recovering(q, k, v, out, dout.to(q.dtype), lse, causal=causal, scale=scale,
                                                     itr=itr, c=c, interest_set=interest_set, max_wave_tokens=mwt,
                                                     classic_fallback=ctx.classic_fallback, verbose=verbose)
        if ctx.bwd_info is not None:
            ctx.bwd_info.update(rec)
        return dq, dk, dv, None, None, None, None, None, None, None, None, None, None, None


def wave_attention(q, k, v, *, causal=True, scale=None, itr=1, c=7, interest_set=(0, 1, 3), bwd_itr=None, max_wave_tokens=None,
                   gather=None, verbose=None, classic_fallback: bool = False, bwd_info: Optional[dict] = None):
    """Differentiable Stream-CQSA attention on the native wave kernel; returns fp32 ``[B, H, N, D]``.
    ``gather`` is the forward's (the backward always gathers). An out-of-memory in the backward halves
    the wave budget, then (``classic_fallback=True``) moves to the classic streamed backward; see
    ``wave_backward_recovering``. ``bwd_info``, when given, is filled with the forward's gather mode and,
    after ``.backward()``, the backward's record (engine, wave OOM retries, budget)."""
    return _WaveAttn.apply(q, k, v, causal, scale, itr, c, tuple(interest_set), bwd_itr, max_wave_tokens, gather, verbose,
                           classic_fallback, bwd_info)


# ---------------------------------------------------------------------------
# Chunk pool for host-resident inputs
# ---------------------------------------------------------------------------
class ChunkPool:
    """Device-resident copies of the level-1 chunks a wave needs, streamed from pinned staging."""

    def __init__(self, N: int, c: int, H: int, D: int, dtype, device):
        self.N, self.c, self.H, self.D, self.dtype, self.device = int(N), int(c), int(H), int(D), dtype, device
        sizes, starts, ends = chunk_layout(N, c, SEG_ALIGN)
        self.sizes, self.starts, self.ends = sizes, starts, ends
        self.slot_len = -(-max(sizes) // SEG_ALIGN) * SEG_ALIGN
        self.slot_bytes = 3 * self.slot_len * H * D * torch.tensor([], dtype=dtype).element_size()
        self.n_slots = 0
        self.q = self.k = self.v = None
        self.slot_of: dict[int, int] = {}
        self.batch = None                     # the batch index the resident chunks belong to
        self._stage = None

    @property
    def bytes(self) -> int:
        return self.n_slots * self.slot_bytes

    def allocate(self, n_slots: int):
        self.n_slots = int(n_slots)
        R = self.n_slots * self.slot_len
        # zeros, not empty: the uniform layout reads a slot's padding rows (finite garbage is
        # harmless, NaN is not: masked P is 0 and 0 * NaN poisons the PV product)
        self.q, self.k, self.v = (torch.zeros((R, self.H, self.D), device=self.device, dtype=self.dtype) for _ in range(3))
        self._stage = [torch.empty((self.slot_len, self.H, self.D), dtype=self.dtype).pin_memory() for _ in range(2)]
        self._stage_ev = [None, None]
        self.slot_of, self.batch = {}, None

    def release(self):
        self.q = self.k = self.v = None
        self._stage = None
        self.slot_of, self.batch = {}, None

    def plan(self, tasks, interest_set, max_tokens, max_subproblems=None):
        waves, cur, cur_tok, cur_chunks = [], [], 0, set()
        for t in tasks:
            L = int(t.local_size)
            ch = set(level1_chunks(t, self.c, interest_set))
            if cur and (cur_tok + L > max_tokens or len(cur_chunks | ch) > self.n_slots
                        or (max_subproblems and len(cur) >= max_subproblems)):
                waves.append(cur); cur, cur_tok, cur_chunks = [], 0, set()
            cur.append(t); cur_tok += L; cur_chunks |= ch
        if cur:
            waves.append(cur)
        return waves

    def load(self, wave, interest_set, b: int, q_tm, k_tm, v_tm):
        need = sorted(set().union(*[set(level1_chunks(t, self.c, interest_set)) for t in wave]))
        assert len(need) <= self.n_slots, "wave needs more chunks than the pool has slots"
        if b != self.batch:
            # a resident chunk holds one batch's rows: a new batch reloads every chunk (keeping them
            # by chunk id alone fed batch b the previous batch's Q/K/V)
            self.slot_of, self.batch = {}, b
        # keep resident chunks where they are, evict the rest
        keep = {cid: s for cid, s in self.slot_of.items() if cid in need}
        free_slots = [s for s in range(self.n_slots) if s not in keep.values()]
        new = {}
        for cid in need:
            if cid in keep:
                new[cid] = keep[cid]
            else:
                new[cid] = free_slots.pop()
        i = 0
        cur = torch.cuda.current_stream(self.device)
        for cid in need:
            if cid in keep:
                continue
            s0, n = self.starts[cid], self.sizes[cid]
            slot = new[cid]
            for src, dst in ((q_tm, self.q), (k_tm, self.k), (v_tm, self.v)):
                st = self._stage[i % 2]
                if self._stage_ev[i % 2] is not None:
                    self._stage_ev[i % 2].synchronize()
                st[:n].copy_(src[b, s0:s0 + n])
                dst[slot * self.slot_len: slot * self.slot_len + n].copy_(st[:n], non_blocking=True)
                ev = torch.cuda.Event(); ev.record(cur); self._stage_ev[i % 2] = ev
                i += 1
        self.slot_of = new
        self._starts_t = torch.tensor(self.starts, dtype=torch.int64)
        self._slot_t = torch.tensor([new.get(j, -1) for j in range(self.c)], dtype=torch.int64)

    def rows_of_global(self, g: torch.Tensor) -> torch.Tensor:
        gn = g.numpy(); starts = self._starts_t.numpy(); slots = self._slot_t.numpy()
        j = np.searchsorted(starts, gn, side="right") - 1
        slot = slots[j]
        assert bool((slot >= 0).all()), "block refers to a chunk that is not resident"
        return torch.from_numpy(slot * self.slot_len + (gn - starts[j]))
