"""
Multi-device Stream-CQSA forward and backward.

Design. The c**itr subproblems are independent and recompose through a
max-shifted merge of per-token (m, l, acc) statistics, and that merge is
associative and commutative. So:

  1. Every rank builds the identical task list and computes the identical
     assignment of it (``stream_cqsa.scheduler``): each rank contributes its
     measured or seeded live-tile rate and memory budget to an all_gathered
     table, and the assignment is a pure function of the task list and that
     table (LPT on exact live-tile work, a local descent, and -- when the
     makespan is still > 5% above the ideal -- splitting the largest unit of the
     most loaded rank into its ``c`` children, which partition its pairs
     exactly). A digest of the assignment is compared across ranks. The
     previous round-robin shard is ``scheduler="rr"``.
  2. Each rank runs the unmodified single-device engine on its units, one call
     per depth (``task_subset``), and ends with one or two normalized partial
     outputs out_p and partial log-sum-exps lse_p (-inf on tokens its units
     never touched). These are exactly "subproblem results" in the engine's own
     merge_lse() sense, with local l = 1, so the cross-rank merge is the same
     formula:

         m'   = max_{r,p} lse_p                  (all_reduce MAX, per token)
         w_p  = exp(lse_p - m')                  (0 where lse_p = -inf)
         acc  = sum  w_p * out_p                 (SUM over ranks)
         l    = sum  w_p                         (SUM over ranks)
         out  = acc / l          lse = m' + log l

     Both exponents are <= 0, so nothing overflows -- the same reason the
     single-device merge is safe.
  3. The reduction is chunked over the token axis so no fp32 [B,H,N,D]
     temporary is ever whole. ``reduce="allreduce"``: one all_reduce(SUM) of the
     packed (w*out, w) per chunk; the result is written in place into the rank's
     first partial (replicated output). ``reduce="reduce_scatter"``: rank r owns
     tokens [r*N/world, (r+1)*N/world); per step every rank packs the weighted
     partials of every rank's next piece, one reduce_scatter(SUM) hands each rank
     its own piece reduced (ring volume (n-1)/n of the buffer instead of
     2(n-1)/n), and ``output="sharded"`` returns just that token range;
     ``output="replicated"`` all_gathers the pieces. The global lse is always
     returned whole (32 B per token at H=8: the backward needs it on every rank).

The backward is the same shape: the same scheduler (its work is the same live
tiles), one engine call per depth, the depth partials summed locally, then one
chunked all_reduce(SUM) or reduce_scatter(SUM) per gradient.

Only standard collectives, NCCL or gloo (CPU tensors run the engine's pure-torch
path, so a gloo group exercises everything but the CUDA kernel). The result is
the single-device function up to fp32 summation order (and, for split units, the
kernel's fp16 rounding of a different tile packing).
"""
from __future__ import annotations

import time
from typing import Any

import torch
import torch.distributed as dist

from .stable_stream import (stream_cqsa_forward, stream_cqsa_backward, build_tasks_cached,
                            task_seg_align_for, estimate_task_bytes, effective_free_bytes)
from . import scheduler as _sched


_FORWARD_ONLY_KW = ("segmented", "shared_chunks", "chunk_pool_slots")
_GIB = float(1 << 30)


def _shard(tasks, rank: int, world: int) -> list[int]:
    """path_idx of this rank's round-robin share of the task list (``scheduler="rr"``)."""
    return [int(t.path_idx) for t in tasks[rank::world]]


def _task_seg_align(engine_kw: dict[str, Any], *, in_place: bool) -> int | None:
    """The engine builds its task list from these same kwargs; the layout must
    match it exactly for the shard (by path_idx) to be a partition."""
    return task_seg_align_for(engine_kw.get("seg_align", 128), in_place=in_place)


def _host_resident(q: torch.Tensor, engine_kw: dict[str, Any]) -> bool:
    return bool(engine_kw.get("stream_from_host")) and q.device.type == "cpu"


def _flash_inner(q: torch.Tensor, engine_kw: dict[str, Any]) -> bool:
    """Whether the engine will run its CUDA kernel (the in-place readers exist only there)."""
    return engine_kw.get("inner") is None and (q.is_cuda or (_host_resident(q, engine_kw) and torch.cuda.is_available()))


def _in_place(q: torch.Tensor, engine_kw: dict[str, Any], depth: int) -> bool:
    """The engine's own rule (stream_cqsa_forward): segmented reads of device-resident inputs, or the
    shared chunk pool, which exists at itr=1 only (the driver drops it for deeper calls)."""
    seg = engine_kw.get("seg_align", 128)
    use_seg = (bool(engine_kw.get("segmented")) and (seg is None or bool(seg)) and q.is_cuda
               and bool(engine_kw.get("sorted_gather", True)))
    shared = bool(engine_kw.get("shared_chunks")) and depth == 1 and _flash_inner(q, engine_kw)
    return use_seg or shared


def _fwd_kw_for_depth(engine_kw: dict[str, Any], depth: int) -> dict[str, Any]:
    kw = dict(engine_kw)
    if depth != 1:
        kw.pop("shared_chunks", None)       # the engine's chunk pool is an itr=1 feature
        kw.pop("chunk_pool_slots", None)
    return kw


def _compute_device(q: torch.Tensor, engine_kw: dict[str, Any]) -> torch.device:
    if q.is_cuda:
        return q.device
    if _host_resident(q, engine_kw) and torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def _result_device(q: torch.Tensor, engine_kw: dict[str, Any]) -> torch.device:
    """Where the engine puts its fp32 output (used when this rank ran nothing)."""
    comp = _compute_device(q, engine_kw)
    od = engine_kw.get("out_device")
    if od == "acc":
        acc_gpu = bool(engine_kw.get("accumulate_on_gpu", True)) and not bool(engine_kw.get("low_memory"))
        return comp if (acc_gpu and comp.type == "cuda") else torch.device("cpu")
    return comp if od is None else torch.device(od)


def _comm_device(group) -> torch.device:
    backend = dist.get_backend(group)
    if str(backend) == "gloo" or not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device("cuda", torch.cuda.current_device())


def _nb(dev: torch.device) -> bool:
    """non_blocking only for copies INTO a device: a non_blocking device-to-host copy returns before
    the data lands (the gloo path stages CUDA partials through host memory)."""
    return dev.type == "cuda"


def _sync(dev: torch.device) -> None:
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


def _gpu_name(dev: torch.device) -> str:
    if dev.type == "cuda":
        return torch.cuda.get_device_name(dev)
    return "cpu"


def rank_table(group=None, *, rate: float | None = None, budget_bytes: int | None = None,
               dev: torch.device | None = None) -> list[dict]:
    """The all_gathered table of every rank's (rate, budget, GPU name, rate source). ``rate`` in live
    tiles/s (``scheduler.calibrate_rate``); None = this GPU model's cached calibration or seed.
    ``budget_bytes`` None = 90% of the free device memory, rounded down to whole GiB (so small
    fluctuations do not change the assignment)."""
    dev = dev or _comm_device(group)
    name = _gpu_name(dev)
    if rate is None:
        dr = _sched.device_rate(name)
        r, src = dr.tile_rate, dr.source
    else:
        r, src = float(rate), "given"
    if budget_bytes is None:
        b = int(effective_free_bytes(dev) * 0.9) // (1 << 30) * (1 << 30) if dev.type == "cuda" else 0
    else:
        b = int(budget_bytes)
    me = dict(rate=float(r), budget_bytes=int(b), gpu=name, source=src, rank=dist.get_rank(group))
    table = [None] * dist.get_world_size(group)
    dist.all_gather_object(table, me, group=group)
    return table


# ---------------------------------------------------------------------------
# assignment (pure; also the dry-run entry point)
# ---------------------------------------------------------------------------

_WORK_CACHE: dict[tuple, _sched.WorkModel] = {}
_PLAN_CACHE: dict[tuple, _sched.Assignment] = {}
_CACHE_MAX = 16


def _cache_put(cache: dict, key, val):
    if len(cache) >= _CACHE_MAX:
        cache.pop(next(iter(cache)))
    cache[key] = val


def plan_distributed(N: int, itr: int, ranks: list, *, B: int = 1, H: int = 8, D: int = 64, itemsize: int = 2,
                     causal: bool = True, c: int = 7, interest_set=(0, 1, 3), sorted_gather: bool = True,
                     seg_align_for_depth=None, scheduler: str = "lpt", split: str = "children",
                     direction: str = "fwd", tol: float = 0.05, max_extra_depth: int = 2,
                     n_par: int = 2, base_bytes: int = 0, depth_bytes: int = 0,
                     tasks=None) -> _sched.Assignment:
    """
    The assignment of one call's subproblems to ``ranks`` (``scheduler.RankSpec`` or the dicts of
    ``rank_table``). Deterministic: every rank calls this with the same arguments. ``tasks``: the
    depth-``itr`` task list if already built (else built here with ``seg_align_for_depth(itr)``).
    """
    ranks = [r if isinstance(r, _sched.RankSpec) else
             _sched.RankSpec(rate=float(r["rate"]), budget_bytes=int(r.get("budget_bytes", 0)),
                             name=str(r.get("gpu", "")), source=str(r.get("source", ""))) for r in ranks]
    align_of = seg_align_for_depth or (lambda d: None)
    aligns = tuple(align_of(d) for d in range(int(itr), int(itr) + max_extra_depth + 1))
    wkey = (int(N), int(c), tuple(interest_set), bool(causal), bool(sorted_gather), int(itr), aligns)
    p = _sched.CostParams(BH=int(B * H), d_factor=_sched.head_dim_factor(D) * (3.0 if direction == "bwd" else 1.0),
                          hd_scale=B * H * D / 512.0, base_bytes=int(base_bytes), depth_bytes=int(depth_bytes),
                          unit_bytes_per_tok=int(estimate_task_bytes(1, B, H, D, itemsize)), n_par=int(n_par))
    pkey = (wkey, tuple((r.rate, r.budget_bytes) for r in ranks), scheduler, split, direction, float(tol),
            int(max_extra_depth), repr(p))
    hit = _PLAN_CACHE.get(pkey)
    if hit is not None:
        return hit
    wm = _WORK_CACHE.get(wkey)
    if wm is None:
        if tasks is None:
            tasks = build_tasks_cached(N, itr, B=B, H=H, D=D, itemsize=itemsize, sorted_gather=sorted_gather,
                                       pin=False, c=c, interest_set=tuple(interest_set), seg_align=align_of(itr))
        wm = _sched.WorkModel(N, c=c, interest_set=interest_set, causal=causal, sorted_gather=sorted_gather,
                              seg_align_for_depth=align_of)
        wm.base_units = wm.units_from_tasks(tasks, int(itr))
        _cache_put(_WORK_CACHE, wkey, wm)
    a = _sched.schedule(wm.base_units, ranks, params=p, mode=scheduler, split=split, work=wm, tol=tol,
                        max_extra_depth=max_extra_depth)
    _cache_put(_PLAN_CACHE, pkey, a)
    return a


def _resolve_split(scheduler: str, split: str) -> str:
    if scheduler == "rr":
        return "none"
    if split == "auto":
        return "children"
    if split == "rows":
        raise NotImplementedError(
            "row-span units need the kernel's row-range launch parameter (dependency U1: the CQS group-bits "
            "kernels require seqlen_q == seqlen_k); use split='children' (exact, runs today) or 'none'")
    if split not in ("none", "children"):
        raise ValueError(f"split must be 'auto', 'children', 'none' (or 'rows' once U1 lands), got {split!r}")
    return split


def _assign(q, *, itr, causal, group, scheduler, split, tol, max_extra_depth, rate, budget_bytes, engine_kw,
            direction, dev, n_tasks_hint=None):
    """(assignment, rank table, digest check time) -- identical on every rank."""
    rank, world = dist.get_rank(group), dist.get_world_size(group)
    B, H, N, D = q.shape
    c = int(engine_kw.get("c", 7))
    interest_set = tuple(engine_kw.get("interest_set", (0, 1, 3)))
    sorted_gather = bool(engine_kw.get("sorted_gather", True))
    if direction == "fwd":
        align_of = lambda d: _task_seg_align(engine_kw, in_place=_in_place(q, engine_kw, d))  # noqa: E731
    else:
        align_of = lambda d: _task_seg_align(engine_kw, in_place=False)                     # noqa: E731
    split = _resolve_split(scheduler, split)
    t0 = time.perf_counter()
    table = rank_table(group, rate=rate, budget_bytes=budget_bytes, dev=dev)
    res_dev = _result_device(q, engine_kw) if direction == "fwd" else q.device
    acc_on_dev = res_dev.type == "cuda"
    per_tok = B * H * (D * 4 + 8)                                     # fp32 out/acc + m, l
    base = per_tok * N if acc_on_dev else 0
    mp = engine_kw.get("max_parallel")
    # the engine's own cache key (pinned when it computes on CUDA; the backward always pins), so the
    # engine call below reuses this list instead of building it again
    pin = True if direction == "bwd" else _compute_device(q, engine_kw).type == "cuda"
    tasks = build_tasks_cached(N, int(itr), B=B, H=H, D=D, itemsize=q.element_size(), sorted_gather=sorted_gather,
                               pin=pin, c=c, interest_set=interest_set, seg_align=align_of(int(itr)))
    a = plan_distributed(N, int(itr), table, B=B, H=H, D=D, itemsize=q.element_size(), causal=causal, c=c,
                         interest_set=interest_set, sorted_gather=sorted_gather, seg_align_for_depth=align_of,
                         scheduler=scheduler, split=split, direction=direction, tol=tol,
                         max_extra_depth=max_extra_depth, n_par=int(mp) if mp else 2,
                         base_bytes=base, depth_bytes=base, tasks=tasks)
    digests = [None] * world
    dist.all_gather_object(digests, a.digest(), group=group)
    if len(set(digests)) != 1:
        raise RuntimeError(f"rank {rank}: the ranks computed different assignments {digests} "
                           f"(rank tables must be identical; mine {table})")
    return a, table, time.perf_counter() - t0, len(tasks)


# ---------------------------------------------------------------------------
# cross-rank reductions
# ---------------------------------------------------------------------------

def _bounds(N: int, world: int) -> list[int]:
    return [(N * r) // world for r in range(world + 1)]


def _weights(lse_c: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    return torch.where(torch.isfinite(lse_c), torch.exp(lse_c - m), torch.zeros_like(lse_c))


def merge_forward(partials: list, N: int, *, B: int, H: int, D: int, group=None, dev: torch.device,
                  reduce: str = "allreduce", output: str = "replicated", chunk_tokens: int = 1 << 18,
                  result_device: torch.device | None = None) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """
    Cross-rank merge of this rank's forward partials ``[(out_p [B,H,N,D] fp32, lse_p [B,H,N]), ...]``
    (possibly none). Returns ``(out, lse, stats)``: ``out`` is ``[B,H,N,D]`` (replicated; written in
    place into the first partial when there is one) or ``[B,H,hi-lo,D]`` (sharded: this rank's token
    range ``stats["token_range"]``); ``lse`` is the global ``[B,H,N]`` on every rank.
    """
    if reduce not in ("allreduce", "reduce_scatter"):
        raise ValueError(f"reduce must be 'allreduce' or 'reduce_scatter' (got {reduce!r})")
    if output not in ("replicated", "sharded"):
        raise ValueError(f"output must be 'replicated' or 'sharded' (got {output!r})")
    rank, world = dist.get_rank(group), dist.get_world_size(group)
    bnd = _bounds(N, world)
    lo, hi = bnd[rank], bnd[rank + 1]
    rdev = partials[0][0].device if partials else (result_device or dev)
    ldev = partials[0][1].device if partials else rdev
    ninf = float("-inf")
    lse_out = torch.empty((B, H, N), dtype=torch.float32, device=ldev)
    if output == "replicated":
        out = partials[0][0] if partials else torch.empty((B, H, N, D), dtype=torch.float32, device=rdev)
    else:
        out = torch.empty((B, H, hi - lo, D), dtype=torch.float32, device=rdev)
    n_coll = 0
    bytes_coll = 0

    def weighted(s: int, e: int, m: torch.Tensor, acc: torch.Tensor) -> None:
        """acc[..., :D] = sum_p w_p * out_p[s:e], acc[..., D] = sum_p w_p (acc zero-filled or overwritten)."""
        first = True
        for out_p, lse_p in partials:
            w = _weights(lse_p[:, :, s:e].to(dev, non_blocking=_nb(dev)).float(), m)
            x = torch.nan_to_num(out_p[:, :, s:e, :].to(dev, non_blocking=_nb(dev)) * w.unsqueeze(-1), nan=0.0)
            if first:
                acc[..., :D].copy_(x); acc[..., D].copy_(w)
                first = False
            else:
                acc[..., :D].add_(x); acc[..., D].add_(w)
        if first:
            acc.zero_()

    def local_max(s: int, e: int) -> torch.Tensor:
        m = None
        for _, lse_p in partials:
            x = lse_p[:, :, s:e].to(dev, non_blocking=_nb(dev)).float()
            m = x.clone() if m is None else torch.maximum(m, x)
        return m if m is not None else torch.full((B, H, e - s), ninf, dtype=torch.float32, device=dev)

    if reduce == "allreduce":
        for s in range(0, N, chunk_tokens):
            e = min(N, s + chunk_tokens)
            m = local_max(s, e).contiguous()
            dist.all_reduce(m, op=dist.ReduceOp.MAX, group=group)
            acc = torch.empty((B, H, e - s, D + 1), dtype=torch.float32, device=dev)
            weighted(s, e, m, acc)
            dist.all_reduce(acc, op=dist.ReduceOp.SUM, group=group)
            n_coll += 2
            bytes_coll += (m.numel() + acc.numel()) * 4
            l = acc[..., D]
            res = acc[..., :D] / l.clamp_min(1e-30).unsqueeze(-1)
            lse_g = torch.where(torch.isfinite(m), m + torch.log(l.clamp_min(1e-30)), torch.full_like(m, ninf))
            if output == "replicated":
                out[:, :, s:e, :].copy_(res.to(rdev))
            else:
                a, b = max(s, lo), min(e, hi)
                if b > a:
                    out[:, :, a - lo:b - lo, :].copy_(res[:, :, a - s:b - s, :].to(rdev))
            lse_out[:, :, s:e].copy_(lse_g.to(ldev))
            del m, acc, l, res, lse_g
    else:
        lens = [bnd[r + 1] - bnd[r] for r in range(world)]
        step = max(1, chunk_tokens // world)
        for j in range(0, max(lens), step):
            n = min(step, max(lens) - j)
            spans = [(bnd[r] + min(j, lens[r]), bnd[r] + min(j + n, lens[r])) for r in range(world)]
            m_all = torch.full((world, B, H, n), ninf, dtype=torch.float32, device=dev)
            for r, (s, e) in enumerate(spans):
                if e > s:
                    m_all[r, :, :, :e - s] = local_max(s, e)
            dist.all_reduce(m_all, op=dist.ReduceOp.MAX, group=group)
            buf = torch.zeros((world, B, H, n, D + 1), dtype=torch.float32, device=dev)
            for r, (s, e) in enumerate(spans):
                if e > s:
                    weighted(s, e, m_all[r, :, :, :e - s], buf[r, :, :, :e - s, :])
            mine = torch.empty((B, H, n, D + 1), dtype=torch.float32, device=dev)
            # stacked along dim 0 (gloo checks every other dim; NCCL only the element count)
            dist.reduce_scatter_tensor(mine, buf.view(world * B, H, n, D + 1), op=dist.ReduceOp.SUM, group=group)
            n_coll += 2
            bytes_coll += (m_all.numel() + buf.numel()) * 4
            s0, e0 = spans[rank]
            k = e0 - s0
            m_me = m_all[rank]
            l = mine[..., D]
            res = mine[..., :D] / l.clamp_min(1e-30).unsqueeze(-1)
            lse_me = torch.where(torch.isfinite(m_me), m_me + torch.log(l.clamp_min(1e-30)), torch.full_like(m_me, ninf))
            lse_all = torch.empty((world, B, H, n), dtype=torch.float32, device=dev)
            dist.all_gather_into_tensor(lse_all.view(world * B, H, n), lse_me.contiguous(), group=group)
            n_coll += 1
            for r, (s, e) in enumerate(spans):
                if e > s:
                    lse_out[:, :, s:e].copy_(lse_all[r, :, :, :e - s].to(ldev))
            if output == "replicated":
                gath = torch.empty((world, B, H, n, D), dtype=torch.float32, device=dev)
                dist.all_gather_into_tensor(gath.view(world * B, H, n, D), res.contiguous(), group=group)
                n_coll += 1
                bytes_coll += gath.numel() * 4
                for r, (s, e) in enumerate(spans):
                    if e > s:
                        out[:, :, s:e, :].copy_(gath[r, :, :, :e - s, :].to(rdev))
                del gath
            elif k > 0:
                out[:, :, s0 - lo:e0 - lo, :].copy_(res[:, :, :k, :].to(rdev))
            del m_all, buf, mine, res, lse_me, lse_all
    stats = dict(token_range=(lo, hi), n_collectives=n_coll, collective_bytes=bytes_coll)
    return out, lse_out, stats


def reduce_gradients(grad_sets: list, N: int, *, B: int, H: int, D: int, group=None, dev: torch.device,
                     reduce: str = "allreduce", chunk_tokens: int = 1 << 18,
                     result_device: torch.device | None = None) -> tuple[list, dict]:
    """
    Sum this rank's gradient partials ``[(dq, dk, dv), ...]`` (one set per depth call, possibly none)
    over the ranks. ``allreduce``: every rank ends with the full fp32 gradients (reduced in place into
    its first set). ``reduce_scatter``: rank r ends with its token range ``[r*N/world, (r+1)*N/world)``
    of each gradient (``[B,H,hi-lo,D]``, fp32).
    """
    if reduce not in ("allreduce", "reduce_scatter"):
        raise ValueError(f"reduce must be 'allreduce' or 'reduce_scatter' (got {reduce!r})")
    rank, world = dist.get_rank(group), dist.get_world_size(group)
    bnd = _bounds(N, world)
    lo, hi = bnd[rank], bnd[rank + 1]
    gdev = grad_sets[0][0].device if grad_sets else (result_device or dev)
    n_coll = 0
    outs = []
    for gi in range(3):
        parts = [gs[gi] for gs in grad_sets]

        def local_sum(s: int, e: int, dst: torch.Tensor) -> None:
            if not parts:
                dst.zero_()
                return
            dst.copy_(parts[0][:, :, s:e, :].to(dev, non_blocking=_nb(dev)))
            for g in parts[1:]:
                dst.add_(g[:, :, s:e, :].to(dev, non_blocking=_nb(dev)))

        if reduce == "allreduce":
            res = parts[0] if parts else torch.empty((B, H, N, D), dtype=torch.float32, device=gdev)
            for s in range(0, N, chunk_tokens):
                e = min(N, s + chunk_tokens)
                buf = torch.empty((B, H, e - s, D), dtype=torch.float32, device=dev)
                local_sum(s, e, buf)
                dist.all_reduce(buf, op=dist.ReduceOp.SUM, group=group)
                n_coll += 1
                res[:, :, s:e, :].copy_(buf.to(res.device))
                del buf
        else:
            res = torch.empty((B, H, hi - lo, D), dtype=torch.float32, device=gdev)
            lens = [bnd[r + 1] - bnd[r] for r in range(world)]
            step = max(1, chunk_tokens // world)
            for j in range(0, max(lens), step):
                n = min(step, max(lens) - j)
                spans = [(bnd[r] + min(j, lens[r]), bnd[r] + min(j + n, lens[r])) for r in range(world)]
                buf = torch.zeros((world, B, H, n, D), dtype=torch.float32, device=dev)
                for r, (s, e) in enumerate(spans):
                    if e > s:
                        local_sum(s, e, buf[r, :, :, :e - s, :])
                mine = torch.empty((B, H, n, D), dtype=torch.float32, device=dev)
                dist.reduce_scatter_tensor(mine, buf.view(world * B, H, n, D), op=dist.ReduceOp.SUM, group=group)
                n_coll += 1
                s0, e0 = spans[rank]
                if e0 > s0:
                    res[:, :, s0 - lo:e0 - lo, :].copy_(mine[:, :, :e0 - s0, :].to(gdev))
                del buf, mine
        outs.append(res)
    return outs, dict(token_range=(lo, hi), n_collectives=n_coll)


# ---------------------------------------------------------------------------
# drivers
# ---------------------------------------------------------------------------

def dist_stream_cqsa_forward(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *,
    itr: int, causal: bool = False, group=None,
    chunk_tokens: int = 1 << 18, output: str = "replicated",
    scheduler: str = "lpt", split: str = "auto", reduce: str = "allreduce",
    rate: float | None = None, budget_bytes: int | None = None,
    tol: float = 0.05, max_extra_depth: int = 2,
    **engine_kw: Any,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    q/k/v: [B, H, N, D], identical on every rank (host or device resident).
    Returns (out fp32, info). ``output="replicated"``: every rank holds the full [B,H,N,D] output;
    ``"sharded"``: rank r holds ``[B,H,hi-lo,D]`` for its tokens ``info["token_range"] = (lo, hi)`` =
    ``(r*N//world, (r+1)*N//world)``. ``info["lse"]`` is the global [B,H,N] lse on every rank.

    ``scheduler``: ``"lpt"`` (cost-aware, the default) or ``"rr"`` (the previous round-robin shard).
    ``split``: ``"auto"`` (= ``"children"`` under lpt), ``"children"``, ``"none"``. ``reduce``:
    ``"allreduce"`` or ``"reduce_scatter"``. ``rate``: this rank's live-tile rate (tiles/s,
    ``scheduler.calibrate_rate``), None = the GPU model's cached calibration or seed. ``budget_bytes``:
    this rank's device budget for the call (None = 90% of free memory).
    """
    if not (dist.is_available() and dist.is_initialized()):
        raise RuntimeError("torch.distributed must be initialised (torchrun)")
    rank, world = dist.get_rank(group), dist.get_world_size(group)
    dev = _comm_device(group)
    B, H, N, D = q.shape
    itr = int(itr)

    asg, table, t_sched, n_tasks = _assign(
        q, itr=itr, causal=causal, group=group, scheduler=scheduler, split=split, tol=tol,
        max_extra_depth=max_extra_depth, rate=rate, budget_bytes=budget_bytes, engine_kw=engine_kw,
        direction="fwd", dev=dev)
    subsets = asg.subsets(rank)

    comp = _compute_device(q, engine_kw)
    t0 = time.perf_counter()
    partials = []
    calls = []
    info0: dict[str, Any] = {}
    for d, subset in subsets.items():
        kw_d = _fwd_kw_for_depth(engine_kw, d)
        td = time.perf_counter()
        out_d, info_d = stream_cqsa_forward(q, k, v, itr=d, causal=causal, task_subset=subset, **kw_d)
        _sync(comp)
        calls.append(dict(depth=d, n_units=len(subset), t_s=time.perf_counter() - td))
        partials.append((out_d, info_d["lse"]))
        if not info0:
            info0 = info_d
    _sync(comp)
    t_local = time.perf_counter() - t0

    t1 = time.perf_counter()
    out, lse, mstats = merge_forward(partials, N, B=B, H=H, D=D, group=group, dev=dev, reduce=reduce,
                                     output=output, chunk_tokens=chunk_tokens,
                                     result_device=_result_device(q, engine_kw))
    del partials
    _sync(dev)
    t_merge = time.perf_counter() - t1

    info = {k_: v_ for k_, v_ in info0.items() if k_ != "lse"}
    base = [p for d, ps in subsets.items() if d == itr for p in ps]
    info.update(rank=rank, world=world, tasks_mine=base, n_tasks=n_tasks,
                units_mine=[[d, p] for d, ps in subsets.items() for p in ps], depth_calls=calls,
                t_local_s=t_local, t_merge_s=t_merge, t_schedule_s=t_sched, lse=lse, output=output,
                scheduler=scheduler, split=asg.split, reduce=reduce, token_range=mstats["token_range"],
                n_collectives=mstats["n_collectives"], collective_bytes=mstats["collective_bytes"],
                predicted_local_s=asg.loads[rank], predicted_makespan_s=asg.makespan, ideal_s=asg.ideal,
                predicted_ratio=asg.ratio, assignment=asg.summary(), rank_table=table, gpu=_gpu_name(dev))
    return out, info


def dist_stream_cqsa_backward(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    dout: torch.Tensor, out: torch.Tensor, lse: torch.Tensor, *,
    itr: int, causal: bool = False, group=None,
    chunk_tokens: int = 1 << 18,
    scheduler: str = "lpt", split: str = "auto", reduce: str = "allreduce",
    rate: float | None = None, budget_bytes: int | None = None,
    tol: float = 0.05, max_extra_depth: int = 2,
    **engine_kw: Any,
) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], dict[str, Any]]:
    """
    Multi-device backward. q/k/v/dout/out are [B, H, N, D] and lse is [B, H, N]
    (the GLOBAL lse from dist_stream_cqsa_forward), identical on every rank.

    Each rank runs the unmodified single-device backward on its units (one call per depth,
    ``task_subset``) and obtains fp32 partial dq/dk/dv restricted to its pair set. The gradient of
    each token is a plain sum over the subproblems that touch it (split units: over the children,
    whose pairs partition the parent's), so the cross-rank step is one chunked all_reduce(SUM)
    (every rank ends with the full fp32 gradients) or reduce_scatter(SUM) (rank r ends with its token
    range ``info["token_range"]`` of each gradient, ``[B,H,hi-lo,D]``) per gradient.
    """
    if not (dist.is_available() and dist.is_initialized()):
        raise RuntimeError("torch.distributed must be initialised (torchrun)")
    rank, world = dist.get_rank(group), dist.get_world_size(group)
    dev = _comm_device(group)
    B, H, N, D = q.shape
    itr = int(itr)
    # A forward/backward pair is driven with one kwargs dict; the forward's in-place read modes
    # have no backward counterpart (the backward always gathers), so they are not passed on.
    bwd_kw = {k_: v_ for k_, v_ in engine_kw.items() if k_ not in _FORWARD_ONLY_KW}
    asg, table, t_sched, n_tasks = _assign(
        q, itr=itr, causal=causal, group=group, scheduler=scheduler, split=split, tol=tol,
        max_extra_depth=max_extra_depth, rate=rate, budget_bytes=budget_bytes, engine_kw=bwd_kw,
        direction="bwd", dev=dev)
    subsets = asg.subsets(rank)

    comp = _compute_device(q, bwd_kw)
    t0 = time.perf_counter()
    grad_sets = []
    calls = []
    for d, subset in subsets.items():
        td = time.perf_counter()
        g = stream_cqsa_backward(q, k, v, dout, out, lse, itr=d, causal=causal, task_subset=subset, **bwd_kw)
        _sync(comp)
        calls.append(dict(depth=d, n_units=len(subset), t_s=time.perf_counter() - td))
        grad_sets.append(g)
    _sync(comp)
    t_local = time.perf_counter() - t0

    t1 = time.perf_counter()
    grads, rstats = reduce_gradients(grad_sets, N, B=B, H=H, D=D, group=group, dev=dev, reduce=reduce,
                                     chunk_tokens=chunk_tokens, result_device=q.device)
    del grad_sets
    _sync(dev)
    t_merge = time.perf_counter() - t1
    base = [p for d, ps in subsets.items() if d == itr for p in ps]
    info = dict(rank=rank, world=world, tasks_mine=base, n_tasks=n_tasks,
                units_mine=[[d, p] for d, ps in subsets.items() for p in ps], depth_calls=calls,
                t_local_s=t_local, t_merge_s=t_merge, t_schedule_s=t_sched, scheduler=scheduler, split=asg.split,
                reduce=reduce, token_range=rstats["token_range"], n_collectives=rstats["n_collectives"],
                predicted_local_s=asg.loads[rank], predicted_makespan_s=asg.makespan, ideal_s=asg.ideal,
                predicted_ratio=asg.ratio, assignment=asg.summary(), rank_table=table, gpu=_gpu_name(dev))
    return (grads[0], grads[1], grads[2]), info
