"""
Multi-device scheduler for Stream-CQSA: which rank runs which subproblem.

The ``c**itr`` subproblems of one attention call are independent and recompose
through the associative (m, l, acc) merge, so a multi-device call is a
scheduling problem: assign the subproblems to ranks so that the slowest rank
finishes as early as possible. Three things make round-robin (the previous
driver) a poor answer: the subproblems are few (7 at itr=1: on 4 ranks the
busiest holds 2/7 of the work, 1.14x the ideal), their work is not identical
(aligned chunks round to whole 128-row blocks), and devices differ (an A100-PCIe
next to an A100-SXM4, or a GPU shared with other work).

Work model. The kernel skips every 128x128 tile whose pairs are all CQS-masked
and runs every other one, so the work of a subproblem is its number of LIVE
tiles, known exactly from the task's block summaries (the census definition of
benchmarks/align_census.py: a tile is masked iff ``blk_and[i] & blk_and[j] !=
0``; causal tasks count the lower triangle, diagonal included). A unit on rank r
costs ``tiles * B * H * d_factor / rate_r + task_s + L * tok_s`` seconds, and a
rank pays ``N * call_tok_s`` once per depth it holds units at (one engine call
per depth): the rate is the device's live-tile throughput (``DeviceRate``,
calibrated per GPU model and cached, or seeded per known model); ``task_s`` is
the per-subproblem launch; ``tok_s`` and ``call_tok_s`` depend on how the call
reads Q/K/V (``PATH_COSTS``: the shared chunk pool, the host gather path, or
device-resident inputs) and were fitted on the 4-GPU job 14330294.

Assignment. LPT for related machines: units sorted by their cost on the slowest
rank, each placed on the rank whose predicted finish ``load_r + cost_r(unit)``
is earliest among the ranks whose memory budget admits it; ties go to the lower
rank. The function is a pure function of the unit list and the rate/budget table,
so every rank computes the same assignment from an all_gathered table (the
driver checks a digest).

Refinement. When the makespan exceeds the ideal ``sum(work)/sum(rate)`` by more
than ``tol`` (5%), the largest splittable unit on the most loaded rank is split
and LPT re-run; a split is kept only if it improves the load vector (sorted in descending order,
compared lexicographically: a lower maximum, or as high a maximum on fewer ranks). The loads carry
the split's own price -- ``c - 1`` more launches, the children's gathered tokens and the second
depth call -- so a split is kept only when the balance it buys exceeds that price: 7 equal tasks
on 4 equal ranks stay whole at itr=1 on the host-streamed path (round-robin's assignment, which
measured 0.90x faster than the split at 1M), while the device-resident backward (no gather cost)
and unequal ranks still split. Two split kinds:

  * ``"children"`` (executable today): the unit is replaced by its ``c``
    children at the next depth -- the tasks ``refine_task`` builds, whose
    retained pairs partition the parent's exactly (the engine's OOM escalation
    relies on the same identity). Under the 128-aligned layout every tile of a
    child is a whole tile of the parent, so the children's live tiles sum to the
    parent's: the split costs no extra pair work, only ``c - 1`` launches and
    ``l``x the parent's gathered tokens. The engine runs a depth as one call with
    ``task_subset``, so a rank holding units at two depths makes two calls.
  * ``"rows"`` (model only): a forward unit split into row-block spans (a
    backward unit into column-block spans) of equal live work. The CQS kernels
    require ``seqlen_q == seqlen_k`` in the group-bits mode (v11
    ``flash_api.cpp:771``, backward ``:1862``) and the wave kernel has no row
    range either, so running a span needs the kernel change the design calls the
    row-range launch parameter (dependency U1). The driver refuses it.

Memory. A unit's planned peak (``n_par`` gathered working sets of its length)
plus the rank's fixed call footprint (the accumulator, plus one more partial
output per extra depth) must not exceed the rank's budget; a unit no rank can
hold is split if it can be, otherwise the schedule is infeasible.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from typing import Any, Callable, Iterable, Sequence

import numpy as np

TILE = 128


# ---------------------------------------------------------------------------
# live-tile work model
# ---------------------------------------------------------------------------

def row_live_tiles(blk_and, causal: bool) -> np.ndarray:
    """Live 128x128 tiles per row block of one subproblem (int64 [nb]).

    ``blk_and`` is the per-block AND of the task's group bits
    (``interface.cqs_block_summaries``); a tile (i, j) is masked iff
    ``blk_and[i] & blk_and[j] != 0``. Causal: columns ``j <= i``. Computed from
    the few distinct block values (one per chunk combination), O(nb * U)."""
    a = np.asarray(blk_and, dtype=np.int64).reshape(-1)
    nb = int(a.shape[0])
    if nb == 0:
        return np.zeros((0,), dtype=np.int64)
    vals, inv = np.unique(a, return_inverse=True)
    conflict = ((vals[:, None] & vals[None, :]) != 0).astype(np.int64)     # [U, U]
    if causal:
        onehot = np.zeros((nb, vals.shape[0]), dtype=np.int64)
        onehot[np.arange(nb), inv] = 1
        cnt = np.cumsum(onehot, axis=0)                                    # blocks j <= i per value
        masked = (cnt * conflict[inv]).sum(axis=1)
        return np.arange(1, nb + 1, dtype=np.int64) - masked
    tot = np.bincount(inv, minlength=vals.shape[0]).astype(np.int64)
    return nb - conflict[inv] @ tot


def live_tiles(blk_and, causal: bool) -> int:
    return int(row_live_tiles(blk_and, causal).sum())


@dataclass(frozen=True)
class Unit:
    """One schedulable piece of work: a subproblem (``path`` at ``depth``), or a
    row-block span of one (``rows``, model only)."""
    depth: int
    path: tuple
    path_idx: int
    L: int
    tiles: float                        # live tiles per (b, h)
    rows: tuple | None = None           # [lo, hi) row blocks of the parent task (model-only split)

    @property
    def key(self) -> tuple:
        return (self.depth, self.path_idx, self.rows or ())


class WorkModel:
    """Live-tile work of the tasks of one call shape, and of their children.

    ``seg_align_for_depth(d)`` is the ``seg_align`` the engine hands
    ``build_tasks`` at depth ``d`` (the driver derives it with the engine's own
    ``task_seg_align_for``), so the modelled layout is the executed one."""

    def __init__(self, N: int, *, c: int = 7, interest_set: Sequence[int] = (0, 1, 3), causal: bool = True,
                 sorted_gather: bool = True, seg_align_for_depth: Callable[[int], int | None] | None = None):
        self.N, self.c, self.interest_set = int(N), int(c), tuple(int(x) for x in interest_set)
        self.l = len(self.interest_set)
        self.causal, self.sorted_gather = bool(causal), bool(sorted_gather)
        self._align = seg_align_for_depth or (lambda d: None)
        self._rows: dict[tuple, np.ndarray] = {}
        self._kids: dict[tuple, list] = {}
        self.base_units: list[Unit] | None = None      # the call's own task list (set by the driver)

    def _record(self, depth: int, path: tuple, path_idx: int, blk_and) -> np.ndarray:
        rows = row_live_tiles(blk_and, self.causal)
        self._rows[(depth, path_idx)] = rows
        return rows

    def unit_from_task(self, task, depth: int) -> Unit:
        """A unit from an engine task. The summaries are rebuilt at the kernel's 128-row tile: the
        task's own ``extra["blk_and"]`` is at ``CQS_BLK_SIZE`` (64) rows."""
        from .interface import cqs_block_summaries
        _, blk_and = cqs_block_summaries(task.group_bits, TILE)
        rows = self._record(depth, tuple(task.path), int(task.path_idx), blk_and.numpy())
        return Unit(depth=int(depth), path=tuple(int(x) for x in task.path), path_idx=int(task.path_idx),
                    L=int(task.local_size), tiles=float(rows.sum()))

    def units_from_tasks(self, tasks, depth: int) -> list[Unit]:
        return [self.unit_from_task(t, depth) for t in tasks]

    def unit_for_path(self, path: Sequence[int]) -> Unit | None:
        from .interface import cqs_block_summaries
        from .reference import group_bits_for_path
        from .stable_stream import task_layout
        path = tuple(int(x) for x in path)
        depth = len(path)
        align, strict = task_layout(self._align(depth))
        _, bits = group_bits_for_path(self.N, path, self.c, self.interest_set, sorted_gather=self.sorted_gather,
                                      align=align, strict_align=strict)
        L = int(bits.shape[0])
        if L == 0:
            return None
        _, blk_and = cqs_block_summaries(bits, TILE)
        idx = 0
        for p in path:
            idx = idx * self.c + p
        rows = self._record(depth, path, idx, blk_and.numpy())
        return Unit(depth=depth, path=path, path_idx=idx, L=L, tiles=float(rows.sum()))

    def children(self, u: Unit) -> list[Unit]:
        """The unit's ``c`` children at depth + 1 (non-empty ones), i.e. ``refine_task``'s."""
        if u.rows is not None:
            raise ValueError("a row span has no children")
        kids = self._kids.get(u.path)
        if kids is None:
            kids = [k for k in (self.unit_for_path(u.path + (j,)) for j in range(self.c)) if k is not None]
            self._kids[u.path] = kids
        return list(kids)

    def refinable(self, u: Unit) -> bool:
        """Children are an exact partition of the parent only when the parent can be refined under the
        aligned layout (``min_refinable_tokens``), and the group bits allow the extra level."""
        from .stable_stream import min_refinable_tokens
        return (u.rows is None and u.L >= min_refinable_tokens(self.c, 128)
                and (u.depth + 1) * (self.l - 1) <= 63)

    def row_split(self, u: Unit, parts: int = 2) -> list[Unit]:
        """Model-only: split a unit into ``parts`` row-block spans of equal live work (U1)."""
        rows = self._rows[(u.depth, u.path_idx)]
        lo, hi = u.rows or (0, int(rows.shape[0]))
        if hi - lo < 2:
            return [u]
        cum = np.cumsum(rows[lo:hi])
        tot = float(cum[-1])
        cuts = [lo] + [lo + int(np.searchsorted(cum, tot * k / parts)) + 1 for k in range(1, parts)] + [hi]
        cuts = sorted(set(min(max(x, lo), hi) for x in cuts))
        out = []
        for a, b in zip(cuts[:-1], cuts[1:]):
            if b > a:
                # a span of rows reads its K/V prefix: gathered rows ~ the span plus (causal) the columns before it
                L = min(u.L, (b - a) * TILE + (b * TILE if self.causal else u.L))
                out.append(Unit(u.depth, u.path, u.path_idx, int(L), float(rows[a:b].sum()), rows=(a, b)))
        return out


class SyntheticWork:
    """Children of synthetic units: ``c`` equal children of ``1/c`` of the parent's work (the ideal
    partition), for dry runs and the planner's equal-task model."""

    def __init__(self, c: int = 7, l: int = 3, min_L: int = 0):
        self.c, self.l, self.min_L = int(c), int(l), int(min_L)

    def children(self, u: Unit) -> list[Unit]:
        Lc = max(1, int(u.L * self.l / self.c))
        return [Unit(u.depth + 1, u.path + (j,), u.path_idx * self.c + j, Lc, u.tiles / self.c) for j in range(self.c)]

    def refinable(self, u: Unit) -> bool:
        return u.rows is None and u.L >= self.min_L

    def row_split(self, u: Unit, parts: int = 2) -> list[Unit]:
        lo, hi = u.rows or (0, 1 << 20)
        cuts = [lo + (hi - lo) * k // parts for k in range(parts + 1)]
        return [Unit(u.depth, u.path, u.path_idx, u.L, u.tiles * (b - a) / (hi - lo), rows=(a, b))
                for a, b in zip(cuts[:-1], cuts[1:]) if b > a]


def synthetic_units(works: Sequence[float], *, L: int = 1 << 18, depth: int = 1) -> list[Unit]:
    return [Unit(depth, (i,), i, int(L), float(w)) for i, w in enumerate(works)]


# ---------------------------------------------------------------------------
# device rates
# ---------------------------------------------------------------------------

# Live-tile throughput of the classic engine's forward kernel, tiles (128 x 128 x one head) per second, fp16,
# D = 64, aligned tasks. The sources say which numbers are measurements and which are seeds.
KNOWN_DEVICE_RATES: dict[str, tuple[float, str]] = {
    # v11 real pattern, L=56064 causal D64: 17.88 ms for 74,776 live tiles x 8 heads (fwd_ab.md, job 14319938)
    "a100_sxm4_80gb": (3.35e7, "measured: v11 kernel, A100-SXM4-80GB, job 14319938 (results/ultimate/fwd_ab.md)"),
    "a100_sxm4_40gb": (3.35e7, "seed: same SM count and clocks as A100-SXM4-80GB"),
    # the host-streamed rate of the 4-GPU scheduler job, fitted with the per-call and per-token overheads of
    # PATH_COSTS over its 136 per-rank observations (benchmarks/ultimate/dist_cost_refit.py; the seed was 2.9e7)
    "a100_pcie_80gb": (2.91e7, "measured: 4 x A100-80GB-PCIe job 14330294 refit (results/ultimate/dist_cost_refit.json)"),
    "a100_80gb_pcie": (2.91e7, "measured: 4 x A100-80GB-PCIe job 14330294 refit (results/ultimate/dist_cost_refit.json)"),
    # pair_rate 4.98e11 vs 7.4e11 on the shared interactive node (results/costmodel_a100_pcie40_interactive.json)
    "a100_pcie_40gb": (2.25e7, "seed: shared-node calibration pair_rate ratio 4.98e11/7.4e11 x SXM4 (noisy)"),
    # H100: not reachable from this account; FA-2 (the design this kernel derives from) runs ~1.6x an A100-SXM4 on
    # an H100-SXM5 (FA-2/FA-3 papers); PCIe / NVL / GH200 scaled by their SM count and clock.
    "h100_80gb_hbm3": (5.4e7, "spec seed (unmeasured): FA-2 on H100-SXM5 ~1.6x A100-SXM4"),
    "h100_sxm": (5.4e7, "spec seed (unmeasured): FA-2 on H100-SXM5 ~1.6x A100-SXM4"),
    "h100_nvl": (4.9e7, "spec seed (unmeasured): H100-SXM5 x 0.9"),
    "h100_pcie": (4.3e7, "spec seed (unmeasured): H100-SXM5 x 114/132 SMs x lower clock"),
    "gh200": (5.7e7, "spec seed (unmeasured): H100-SXM5 x 1.05 (higher power limit)"),
}
DEFAULT_RATE = (2.9e7, "seed: unknown GPU model, A100-PCIe-80GB value")


def gpu_key(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in str(name)).strip("_").lower()


def _cache_dir() -> str:
    return os.environ.get("CQSA_CACHE_DIR") or os.path.join(os.path.expanduser("~"), ".cache", "stream_cqsa")


def _rate_cache_path(name: str) -> str:
    return os.path.join(_cache_dir(), f"device_rate_{gpu_key(name)}.json")


@dataclass
class DeviceRate:
    model: str
    tile_rate: float               # live tiles per second (one head, D=64, fp16)
    source: str                    # "calibrated ...", "measured ...", "seed ...", "spec seed ..."

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=1)


# ---------------------------------------------------------------------------
# per-call and per-token costs of the three ways a call reads Q/K/V
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PathCost:
    """What a unit and an engine call cost beyond the kernel on one read path, at B*H*D = 512 (scaled by
    ``CostParams.hd_scale`` for other shapes)."""
    tok_s: float                   # per gathered token of a unit
    call_tok_s: float              # per token of N, once per engine call (one call per depth a rank holds)


# Fitted on the per-rank local times of the 4 x A100-80GB-PCIe scheduler job 14330294 (della-l02g13; results/
# ultimate/dist_sched_{homog,hetero,hetero_itr2}.json: 124 (row, rank) observations plus the 12 startup
# calibrations at N=262144) by benchmarks/ultimate/dist_cost_refit.py: rms residual 2.0%, slowest-rank residual
# within 4% on every arm (results/ultimate/dist_cost_refit.json). The per-call term is proportional to N on both
# host paths (a fixed per-call cost fitted to zero); the device path shows nothing beyond the launch.
PATH_COSTS: dict[str, PathCost] = {
    # host-streamed itr=1 call with the shared chunk pool: one H2D copy of the whole N per call (0.68 s at 1M),
    # in-place reads per unit (per-token term fitted to zero)
    "pool": PathCost(tok_s=0.0, call_tok_s=6.46e-7),
    # host-streamed gather path (every unit gathers its L tokens on the host; four ranks share one host)
    "host": PathCost(tok_s=5.24e-7, call_tok_s=1.85e-7),
    # device-resident inputs (the job's backward rows)
    "dev": PathCost(tok_s=0.0, call_tok_s=0.0),
}
TASK_S = 3.0e-3                    # per-subproblem launch (the single-device fit, held fixed in the refit)
BWD_FACTOR = 3.64                  # backward seconds per live tile / forward (same refit; D1 assumed 3.0)


def path_overhead_s(units: Iterable[Unit], N: int, path: str, *, hd_scale: float = 1.0,
                    task_s: float = TASK_S) -> float:
    """Seconds one engine call at depth ``path`` spends beyond the kernel on ``units`` (the launches, the
    gathered tokens and the per-call term): what a calibration subtracts to recover the kernel rate."""
    pc = PATH_COSTS[path]
    return sum(task_s + u.L * pc.tok_s * hd_scale for u in units) + N * pc.call_tok_s * hd_scale


def seed_rate(name: str) -> DeviceRate:
    key = gpu_key(name)
    # most specific pattern first (longest key contained in the device's key)
    hits = sorted((k for k in KNOWN_DEVICE_RATES if k in key), key=len, reverse=True)
    rate, src = KNOWN_DEVICE_RATES[hits[0]] if hits else DEFAULT_RATE
    return DeviceRate(model=str(name), tile_rate=float(rate), source=src)


def device_rate(name: str, *, use_cache: bool = True) -> DeviceRate:
    """The rate of a GPU model: a cached calibration (``calibrate_rate(save=True)``) if one exists, else the seed."""
    if use_cache:
        try:
            with open(_rate_cache_path(name)) as f:
                d = json.load(f)
            return DeviceRate(**{k: d[k] for k in ("model", "tile_rate", "source")})
        except (OSError, ValueError, TypeError, KeyError):
            pass
    return seed_rate(name)


def save_device_rate(r: DeviceRate) -> str:
    path = _rate_cache_path(r.model)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(r.to_json())
    return path


def calibrate_rate(device=None, *, N: int = 262144, H: int = 8, D: int = 64, c: int = 7, reps: int = 2,
                   save: bool = False, engine_kw: dict | None = None, stream_from_host: bool = False) -> DeviceRate:
    """Measure this device's live-tile rate: the classic engine forward (N=262144, c=7, itr=1, causal,
    fp16: seven aligned 112K-token subproblems, ~0.5 s on an A100), best of ``reps`` after a warm-up.
    Device-resident inputs by default (the kernel's rate); ``stream_from_host=True`` measures the
    host-streamed production path (token-major host inputs, shared chunk pool, device accumulator),
    whose effective rate also carries the host stages that do not scale with the SMs. The call's
    overheads beyond the kernel (``path_overhead_s``: the launches and, host-streamed, the chunk pool's
    H2D copy) are subtracted, so the rate is comparable to the seeds at any ``N``. Runs under whatever
    limits the process has (a green context capping the SMs, a shared GPU), which is the point for a
    per-rank measurement."""
    import torch
    from .stable_stream import build_tasks_cached, stream_cqsa_forward, task_seg_align_for
    dev = torch.device("cuda", torch.cuda.current_device()) if device is None else torch.device(device)
    name = torch.cuda.get_device_name(dev)
    g = torch.Generator(device="cpu").manual_seed(0)
    if stream_from_host:
        q, k, v = (torch.randn(1, N, H, D, generator=g, dtype=torch.float32).to(torch.float16).transpose(1, 2)
                   for _ in range(3))
    else:
        q, k, v = (torch.randn(1, H, N, D, generator=g, dtype=torch.float32).to(torch.float16).to(dev) for _ in range(3))
    kw = dict(itr=1, causal=True, c=c, allow_escalation=False, max_parallel=1)
    if stream_from_host:
        kw.update(stream_from_host=True, shared_chunks=True, out_device="acc")
    kw.update(engine_kw or {})
    tasks = build_tasks_cached(N, 1, B=1, H=H, D=D, itemsize=2, sorted_gather=True, pin=True, c=c,
                               interest_set=tuple(kw.get("interest_set", (0, 1, 3))),
                               seg_align=task_seg_align_for(kw.get("seg_align", 128), in_place=False))
    units = WorkModel(N, c=c, causal=True).units_from_tasks(tasks, 1)
    tiles = sum(u.tiles for u in units) * H
    stream_cqsa_forward(q, k, v, **kw)
    best = float("inf")
    for _ in range(max(1, reps)):
        torch.cuda.synchronize(dev)
        t0 = time.perf_counter()
        stream_cqsa_forward(q, k, v, **kw)
        torch.cuda.synchronize(dev)
        best = min(best, time.perf_counter() - t0)
    ovh = path_overhead_s(units, N, "pool" if stream_from_host else "dev", hd_scale=H * D / 512.0)
    compute = max(best - ovh, 0.5 * best)
    r = DeviceRate(model=name, tile_rate=tiles * (D / 64.0) / compute,
                   source=(f"calibrated {time.strftime('%Y-%m-%d %H:%M')}: N={N} c={c} itr=1 D={D} "
                           f"{'host-streamed' if stream_from_host else 'device-resident'}, {best * 1e3:.1f} ms "
                           f"less {ovh * 1e3:.0f} ms overheads"))
    del q, k, v
    if save:
        save_device_rate(r)
    return r


def head_dim_factor(D: int) -> float:
    """Time per tile at head dim D relative to D=64 (v11: 31.80 / 17.88 ms at D=128, fwd_ab.md)."""
    return 1.0 if D <= 64 else 1.78 * (D / 128.0)


# ---------------------------------------------------------------------------
# ranks, costs, the assignment
# ---------------------------------------------------------------------------

@dataclass
class RankSpec:
    rate: float                    # live tiles per second (D=64 units)
    budget_bytes: int = 0          # 0 = unlimited
    name: str = ""
    source: str = ""


@dataclass
class CostParams:
    BH: int = 8                    # B * H
    d_factor: float = 1.0          # head_dim_factor(D), x BWD_FACTOR for the backward
    task_s: float = TASK_S
    hd_scale: float = 1.0          # (B*H*D) / 512
    N: int = 0                     # tokens of the call (the per-call term of a depth)
    base_depth: int = 1            # the call's own depth; units deeper than it are split children
    path_base: PathCost = PATH_COSTS["dev"]      # how the units of the base depth read Q/K/V
    path_deeper: PathCost = PATH_COSTS["dev"]    # ... and the children (never the chunk pool: itr=1 only)
    # memory (bytes): the rank's fixed call footprint, one more per extra depth, per gathered token in flight
    base_bytes: int = 0
    depth_bytes: int = 0
    unit_bytes_per_tok: int = 0
    n_par: int = 1

    def path(self, depth: int) -> PathCost:
        return self.path_base if depth <= self.base_depth else self.path_deeper

    def cost(self, u: Unit, rate: float) -> float:
        return u.tiles * self.BH * self.d_factor / rate + self.task_s + u.L * self.path(u.depth).tok_s * self.hd_scale

    def call_cost(self, depth: int) -> float:
        """The engine call a rank makes for the units it holds at ``depth``, paid once per depth."""
        return self.N * self.path(depth).call_tok_s * self.hd_scale

    def unit_peak(self, u: Unit) -> int:
        return int(self.n_par * self.unit_bytes_per_tok * u.L)

    def load(self, units: Iterable[Unit], rate: float) -> float:
        """A rank's predicted local time: its units plus one call per depth."""
        units = list(units)
        return sum(self.cost(u, rate) for u in units) + sum(self.call_cost(d) for d in {u.depth for u in units})


@dataclass
class Assignment:
    mode: str
    split: str
    ranks: list
    per_rank: list                 # list[list[Unit]]
    loads: list                    # predicted seconds per rank
    ideal: float                   # sum(work)/sum(rate) on the unsplit units (+ their overheads spread evenly)
    n_units_in: int
    n_splits: int = 0
    tol: float = 0.05
    notes: list = field(default_factory=list)

    @property
    def makespan(self) -> float:
        return max(self.loads) if self.loads else 0.0

    @property
    def ratio(self) -> float:
        return self.makespan / self.ideal if self.ideal > 0 else 1.0

    def depths(self, r: int) -> list[int]:
        return sorted({u.depth for u in self.per_rank[r]})

    def subsets(self, r: int) -> dict[int, list[int]]:
        """{depth: [path_idx, ...]} of rank r (row-span units are not executable and raise)."""
        out: dict[int, list[int]] = {}
        for u in self.per_rank[r]:
            if u.rows is not None:
                raise NotImplementedError("row-span units need the kernel's row-range launch parameter (U1)")
            out.setdefault(u.depth, []).append(u.path_idx)
        return {d: sorted(v) for d, v in sorted(out.items())}

    def canonical(self) -> list:
        return [[[u.depth, u.path_idx] + (list(u.rows) if u.rows else []) for u in sorted(p, key=lambda x: x.key)]
                for p in self.per_rank]

    def digest(self) -> str:
        return hashlib.sha256(json.dumps(self.canonical(), separators=(",", ":")).encode()).hexdigest()[:16]

    def summary(self) -> dict:
        return dict(mode=self.mode, split=self.split, makespan_s=self.makespan, ideal_s=self.ideal, ratio=self.ratio,
                    loads_s=list(self.loads), n_units_in=self.n_units_in,
                    n_units=sum(len(p) for p in self.per_rank), n_splits=self.n_splits, digest=self.digest(),
                    units=[[[u.depth, u.path_idx, u.L, u.tiles] + (list(u.rows) if u.rows else []) for u in p]
                           for p in self.per_rank],
                    rates=[r.rate for r in self.ranks], budgets=[r.budget_bytes for r in self.ranks],
                    notes=list(self.notes))


def ideal_time(units: Sequence[Unit], ranks: Sequence[RankSpec], p: CostParams) -> float:
    """Fluid lower bound: the pair work at the summed rate, the per-unit overheads spread over the ranks,
    and the one engine call every rank with work makes at the units' depth."""
    work = sum(u.tiles for u in units) * p.BH * p.d_factor
    ovh = sum(p.task_s + u.L * p.path(u.depth).tok_s * p.hd_scale for u in units)
    depth = min((u.depth for u in units), default=p.base_depth)
    return work / sum(r.rate for r in ranks) + ovh / len(ranks) + p.call_cost(depth)


class _Mem:
    """Per-rank memory bookkeeping: base + depth_bytes per extra depth + the largest unit's working set."""

    def __init__(self, ranks: Sequence[RankSpec], p: CostParams):
        self.ranks, self.p = ranks, p
        self.depths = [set() for _ in ranks]
        self.peak = [0] * len(ranks)

    def need(self, r: int, u: Unit) -> int:
        d = self.depths[r] | {u.depth}
        return int(self.p.base_bytes + self.p.depth_bytes * (len(d) - 1) + max(self.peak[r], self.p.unit_peak(u)))

    def fits(self, r: int, u: Unit) -> bool:
        b = self.ranks[r].budget_bytes
        return b <= 0 or self.need(r, u) <= b

    def add(self, r: int, u: Unit) -> None:
        self.depths[r].add(u.depth)
        self.peak[r] = max(self.peak[r], self.p.unit_peak(u))


class Infeasible(ValueError):
    def __init__(self, msg: str, unit: Unit | None = None):
        super().__init__(msg)
        self.unit = unit


def _lpt(units: Sequence[Unit], ranks: Sequence[RankSpec], p: CostParams) -> tuple[list, list]:
    slow = min(r.rate for r in ranks)
    order = sorted(units, key=lambda u: (-p.cost(u, slow), u.depth, u.path_idx, u.rows or ()))
    loads = [0.0] * len(ranks)
    per = [[] for _ in ranks]
    mem = _Mem(ranks, p)
    for u in order:
        best, best_f = -1, math.inf
        for r, spec in enumerate(ranks):
            if not mem.fits(r, u):
                continue
            # a unit at a depth the rank does not hold yet brings that depth's engine call with it
            f = loads[r] + p.cost(u, spec.rate) + (0.0 if u.depth in mem.depths[r] else p.call_cost(u.depth))
            if f < best_f:
                best, best_f = r, f
        if best < 0:
            raise Infeasible(f"no rank's memory budget admits unit depth={u.depth} path={u.path} L={u.L} "
                             f"(needs {p.base_bytes + p.unit_peak(u)} B)", u)
        loads[best] = best_f
        per[best].append(u)
        mem.add(best, u)
    return per, loads


def _set_fits(spec: RankSpec, units: Sequence[Unit], p: CostParams) -> bool:
    if spec.budget_bytes <= 0 or not units:
        return True
    need = p.base_bytes + p.depth_bytes * (len({u.depth for u in units}) - 1) + max(p.unit_peak(u) for u in units)
    return need <= spec.budget_bytes


def _improve(per: list, loads: list, ranks: Sequence[RankSpec], p: CostParams, max_iters: int = 256) -> tuple[list, list]:
    """Local descent after LPT: move one unit off the most loaded rank, or swap one of its units with
    another rank's, whenever that lowers the larger of the two loads below the current maximum. LPT
    alone is limited by its last placements (49 unequal itr=2 tasks on 4 ranks: 1.042x the ideal,
    round-robin happened to reach 1.037x); the descent removes most of that. A move that empties a
    depth on one rank or opens one on the other carries that depth's call cost. Deterministic."""
    per = [list(x) for x in per]
    loads = list(loads)
    R = len(ranks)
    counts = [Counter(u.depth for u in us) for us in per]

    def taken(r: int, u: Unit) -> float:
        """Change of rank r's load when u leaves it (its depth's call goes with the last unit)."""
        return -p.cost(u, ranks[r].rate) - (p.call_cost(u.depth) if counts[r][u.depth] == 1 else 0.0)

    def given(r: int, u: Unit) -> float:
        """Change of rank r's load when u joins it (a new depth brings its call)."""
        return p.cost(u, ranks[r].rate) + (p.call_cost(u.depth) if counts[r][u.depth] == 0 else 0.0)

    def swapped(r: int, u: Unit, v: Unit) -> float:
        """Change of rank r's load when u leaves and v joins (same depth: the call stays)."""
        if u.depth == v.depth:
            return p.cost(v, ranks[r].rate) - p.cost(u, ranks[r].rate)
        return taken(r, u) + given(r, v)

    for _ in range(max_iters):
        r = max(range(R), key=lambda i: (loads[i], -i))
        top = loads[r]
        best = None                                   # (new pair max, kind, i, s, j)
        for i, u in enumerate(per[r]):
            r_less = top + taken(r, u)
            for s in range(R):
                if s == r:
                    continue
                m = max(r_less, loads[s] + given(s, u))
                if m < top * (1 - 1e-9) and (best is None or m < best[0]) and _set_fits(ranks[s], per[s] + [u], p):
                    best = (m, "move", i, s, -1)
                for j, v in enumerate(per[s]):
                    m = max(top + swapped(r, u, v), loads[s] + swapped(s, v, u))
                    if m < top * (1 - 1e-9) and (best is None or m < best[0]):
                        if _set_fits(ranks[s], per[s][:j] + per[s][j + 1:] + [u], p) and \
                                _set_fits(ranks[r], per[r][:i] + per[r][i + 1:] + [v], p):
                            best = (m, "swap", i, s, j)
        if best is None:
            break
        _, kind, i, s, j = best
        u = per[r].pop(i)
        counts[r][u.depth] -= 1
        if kind == "swap":
            v = per[s].pop(j)
            counts[s][v.depth] -= 1
            per[r].append(v)
            counts[r][v.depth] += 1
        per[s].append(u)
        counts[s][u.depth] += 1
        loads[r] = p.load(per[r], ranks[r].rate)
        loads[s] = p.load(per[s], ranks[s].rate)
    return per, loads


def _round_robin(units: Sequence[Unit], ranks: Sequence[RankSpec], p: CostParams) -> tuple[list, list]:
    """The previous driver's shard: task-list order, unit i -> rank i % world (memory not checked)."""
    per = [list(units[r::len(ranks)]) for r in range(len(ranks))]
    loads = [p.load(per[r], ranks[r].rate) for r in range(len(ranks))]
    return per, loads


def _better(a: Sequence[float], b: Sequence[float], rel: float = 1e-9) -> bool:
    """Load vector ``a`` beats ``b``: lexicographically smaller when both are sorted in descending order.
    A split that leaves the maximum where it is but empties one of the ranks at the maximum is progress
    (7 tasks on 8 ranks: every task has to be split before the maximum can fall)."""
    for x, y in zip(sorted(a, reverse=True), sorted(b, reverse=True)):
        if x < y * (1.0 - rel):
            return True
        if x > y * (1.0 + rel):
            return False
    return False


def schedule(units: Sequence[Unit], ranks: Sequence[RankSpec], *, params: CostParams | None = None,
             mode: str = "lpt", split: str = "none", work: Any = None, tol: float = 0.05,
             max_extra_depth: int = 2, max_iters: int = 64, max_units: int = 4096,
             improve: bool = True, min_gain: float = 0.02) -> Assignment:
    """
    Assign ``units`` to ``ranks``. ``mode``: ``"lpt"`` (cost-aware, memory-checked) or ``"rr"``
    (round-robin, the previous driver). ``split``: ``"none"``, ``"children"`` (refine the largest unit
    of the most loaded rank into its children, executable), ``"rows"`` (row spans, model only: U1).
    ``work`` provides ``children(u)``, ``refinable(u)`` and ``row_split(u)`` (a ``WorkModel`` for real
    task lists, ``SyntheticWork`` for dry runs). ``improve``: the move/swap descent after every LPT
    placement. Splits are kept only when the makespan they reach -- the children's launches and
    gathered tokens and the extra depth call included -- is below the unsplit one by more than
    ``min_gain`` (2%: the model's own resolution, its rms residual on job 14330294). Deterministic: a
    pure function of its arguments.
    """
    p = params or CostParams()
    ranks = list(ranks)
    if not ranks:
        raise ValueError("no ranks")
    units = list(units)
    base_depth = min((u.depth for u in units), default=0)
    ideal = ideal_time(units, ranks, p)
    if mode == "rr":
        per, loads = _round_robin(units, ranks, p)
        return Assignment("rr", "none", ranks, per, loads, ideal, len(units), tol=tol)
    if mode != "lpt":
        raise ValueError(f"mode must be 'lpt' or 'rr' (got {mode!r})")
    if split not in ("none", "children", "rows"):
        raise ValueError(f"split must be 'none', 'children' or 'rows' (got {split!r})")
    if split != "none" and work is None:
        raise ValueError("splitting needs a work model (children / row_split)")

    def splittable(u: Unit) -> bool:
        if split == "children":
            return u.depth - base_depth < max_extra_depth and work.refinable(u)
        if split == "rows":
            lo, hi = u.rows or (0, 2)
            return hi - lo >= 2
        return False

    def split_unit(u: Unit) -> list[Unit]:
        # row spans: as many as there are ranks, so one split can spread a unit over all of them
        return work.children(u) if split == "children" else work.row_split(u, max(2, len(ranks)))

    cur = list(units)
    notes: list[str] = []
    n_splits = 0
    # memory first: a unit no rank can hold is split until it fits (or the schedule is infeasible)
    def place(us):
        per_, loads_ = _lpt(us, ranks, p)
        return _improve(per_, loads_, ranks, p) if improve else (per_, loads_)

    for _ in range(max_iters):
        try:
            per, loads = place(cur)
            break
        except Infeasible as e:
            if e.unit is None or not splittable(e.unit) or len(cur) >= max_units:
                raise
            kids = split_unit(e.unit)
            cur = [x for x in cur if x.key != e.unit.key] + kids
            n_splits += 1
            notes.append(f"split depth={e.unit.depth} path_idx={e.unit.path_idx} to fit a memory budget")
    else:
        raise Infeasible("memory refinement did not converge")
    unsplit = (cur, per, loads, n_splits)
    tried: set = set()
    it = 0
    while split != "none" and max(loads) > (1.0 + tol) * ideal and it < max_iters and len(cur) < max_units:
        it += 1
        # the largest untried splittable unit of the most loaded rank that still has one
        u = None
        for r in sorted(range(len(ranks)), key=lambda i: (-loads[i], i)):
            cands = sorted((x for x in per[r] if x.key not in tried and splittable(x)),
                           key=lambda x: (-p.cost(x, ranks[r].rate), x.depth, x.path_idx, x.rows or ()))
            if cands:
                u = cands[0]
                break
        if u is None:
            break
        tried.add(u.key)
        kids = split_unit(u)
        if len(kids) <= 1:
            continue
        trial = [x for x in cur if x.key != u.key] + kids
        try:
            per2, loads2 = place(trial)
        except Infeasible:
            continue
        # an exactly equal vector is accepted too: splitting the last whole unit of 7 on 8 ranks changes
        # nothing by itself and is the only way to the finer units that do (real costs carry the per-unit
        # overhead, so a useless split there is strictly worse and rejected)
        if _better(loads2, loads) or not _better(loads, loads2, rel=1e-12):
            cur, per, loads = trial, per2, loads2
            n_splits += 1
    # The steps above may lower a rank below the maximum without lowering the maximum (a split whose
    # children land on the lightest ranks): that is a path to the splits that do, not a gain by itself,
    # and with real costs it leaves ranks paying a second depth call for nothing. Splits are kept only
    # when the makespan they reach is below the unsplit one by more than the model resolves: the gain
    # net of the launches, gathered tokens and extra calls (all in the loads). Otherwise the whole
    # units stand (7 equal tasks on 4 equal ranks at itr=1: round-robin's assignment).
    if n_splits > unsplit[3] and not max(loads) < max(unsplit[2]) * (1.0 - min_gain):
        cur, per, loads, n_splits = unsplit
        notes.append(f"splits gained < {min_gain:.0%} of the makespan net of their cost; whole units kept")
    return Assignment("lpt", split, ranks, per, loads, ideal, len(units), n_splits=n_splits, tol=tol, notes=notes)


# ---------------------------------------------------------------------------
# planner helper: equal tasks on related machines
# ---------------------------------------------------------------------------

@lru_cache(maxsize=256)
def dist_local_time(n_tasks: int, rel_rates: tuple, *, unit_s: float = 1.0, L: int = 0, N: int = 0,
                    path_base: str = "dev", path_deeper: str = "dev", task_s: float = 0.0, hd_scale: float = 1.0,
                    split: str = "children", c: int = 7, l: int = 3, tol: float = 0.05) -> float:
    """Predicted local time of the slowest rank when ``n_tasks`` equal subproblems of ``unit_s`` seconds
    each (on a rank of rate 1.0) and ``L`` gathered tokens are assigned by ``schedule`` to ranks of
    relative rates ``rel_rates``, with the per-unit and per-call costs of the read paths (``PATH_COSTS``
    names; a child of a split reads through ``path_deeper``). With the costs at zero this is the pure
    work share: ``split="none"`` on equal ranks gives round-robin's ``ceil(n / world) / n``."""
    units = synthetic_units([float(unit_s)] * int(n_tasks), L=int(L))
    ranks = [RankSpec(rate=float(r)) for r in rel_rates]
    p = CostParams(BH=1, task_s=float(task_s), hd_scale=float(hd_scale), N=int(N),
                   path_base=PATH_COSTS[path_base], path_deeper=PATH_COSTS[path_deeper])
    a = schedule(units, ranks, params=p, mode="lpt", split=split, work=SyntheticWork(c, l), tol=tol)
    return a.makespan


__all__ = [
    "TILE", "row_live_tiles", "live_tiles", "Unit", "WorkModel", "SyntheticWork", "synthetic_units",
    "KNOWN_DEVICE_RATES", "DeviceRate", "gpu_key", "seed_rate", "device_rate", "save_device_rate", "calibrate_rate",
    "head_dim_factor", "PathCost", "PATH_COSTS", "TASK_S", "BWD_FACTOR", "path_overhead_s",
    "RankSpec", "CostParams", "Assignment", "Infeasible", "ideal_time", "schedule", "dist_local_time",
]
