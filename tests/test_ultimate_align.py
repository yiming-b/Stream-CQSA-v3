"""
Block-aligned chunk layout everywhere (ultimate P0.1).

Every task the engine builds -- device-resident gathered ones included -- has
its chunk boundaries on 128-token blocks, so no kernel tile straddles a chunk
boundary (measured 1.365x -> 1.004x FA-2 at L~56K on the v11 kernel,
results/ultimate/layout_ab.json). The structural claims checked here on the
CPU are the ones the escalation relies on: children are built on the parent's
local sequence with the same alignment, so they are aligned, never padded and
never larger than their parent; refinement stops before the layout degenerates;
the legacy unaligned layout stays reachable for A/B. The engine checks need the
shared vis1 GPU (N <= 100003, < 1 GiB) and compare the aligned engine with the
unaligned one and with an fp64 reference on sampled rows.
"""
from __future__ import annotations

import os

import numpy as np
import pytest
import torch

os.environ.setdefault("CQSA_CUDA_MODULE", "cqsa_cuda_next_v11")
os.environ.setdefault("CQSA_CUDA_MODULE_NONCAUSAL", "cqsa_cuda_next_v9")

from stream_cqsa.autoconfig import QUORUM_SETS
from stream_cqsa.interface import CQS_BLK_SIZE, cqs_block_summaries
from stream_cqsa.reference import check_pair_coverage, level_align
from stream_cqsa.stable_stream import (
    TASK_SEG_ALIGN,
    build_tasks,
    max_depth_for,
    min_refinable_tokens,
    refine_task,
    stream_cqsa_backward,
    stream_cqsa_forward,
    task_layout,
    task_seg_align_for,
)

CUDA = torch.cuda.is_available()
A = TASK_SEG_ALIGN
TASK_CAP = 4096
KW = dict(B=1, H=8, D=64, itemsize=2, pin=False, sorted_gather=True)


def _configs():
    out = []
    for N in (100003, 131072, 524288):
        for c in (7, 31, 73, 133):
            for itr in (1, 2, 3):
                if c ** itr <= TASK_CAP:
                    out.append((N, c, itr))
    return out


def _assert_block_contiguous(task):
    """Every full 128-block of the local sequence is one contiguous global run,
    and the block map is exactly the first id of every block."""
    ids = task.token_ids.numpy()
    L = ids.shape[0]
    nfull = L // A
    if nfull:
        blocks = ids[: nfull * A].reshape(nfull, A)
        assert np.array_equal(blocks, blocks[:, :1] + np.arange(A)), task.path
    tail = ids[nfull * A:]
    if tail.size:
        assert np.array_equal(tail, tail[0] + np.arange(tail.size)), task.path
    sb = task.extra["seg_base"]
    assert sb is not None and sb.dtype == torch.int32, task.path
    assert np.array_equal(sb.numpy(), ids[::A].astype(np.int32)), task.path


def _assert_no_mixed_tiles(task):
    """The kernel classifies a 64x64 tile as masked, clear or mixed from the
    per-block OR/AND summaries; a block uniform in its group bits has OR == AND,
    which rules out mixed tiles in every row and column it takes part in."""
    bits = task.group_bits
    blk_or, blk_and = cqs_block_summaries(bits, CQS_BLK_SIZE)
    assert torch.equal(blk_or, task.extra["blk_or"]) and torch.equal(blk_and, task.extra["blk_and"])
    nfull = int(bits.numel()) // CQS_BLK_SIZE
    assert torch.equal(blk_or[:nfull], blk_and[:nfull]), task.path


def _mixed_tile_fraction(task):
    """Brute-force tile census, lower triangle: fraction of tiles neither
    masked ((AND & AND) != 0) nor clear ((OR & OR) == 0)."""
    o = task.extra["blk_or"].numpy().astype(np.int64)
    a = task.extra["blk_and"].numpy().astype(np.int64)
    tri = np.tril(np.ones((o.shape[0], o.shape[0]), dtype=bool))
    masked = (np.bitwise_and.outer(a, a) != 0) & tri
    clear = (np.bitwise_and.outer(o, o) == 0) & tri
    return float((tri & ~masked & ~clear).sum() / tri.sum())


# ------------------------------------------------------------ default layout --

@pytest.mark.parametrize("N,c,itr", _configs())
def test_default_tasks_are_block_aligned(N, c, itr):
    iset = QUORUM_SETS[c]
    tasks = build_tasks(N, itr, c=c, interest_set=iset, **KW)
    assert len(tasks) == c ** itr, "no task may be empty"
    # Alignment moves chunk boundaries; it never pads, so the token budget is
    # exactly that of the unaligned layout: every token in l**itr leaves.
    assert sum(int(t.local_size) for t in tasks) == len(iset) ** itr * N
    for t in tasks:
        assert int(t.local_size) == int(t.token_ids.numel()) > 0
        _assert_block_contiguous(t)
        _assert_no_mixed_tiles(t)
        ids = t.token_ids.numpy()
        assert np.all(np.diff(ids) > 0), "sorted gather: local order is global order"


@pytest.mark.parametrize("c", [7, 133])
def test_default_tasks_have_zero_mixed_tiles_by_census(c):
    """The brute-force 64x64 census agrees with the OR == AND argument."""
    tasks = build_tasks(131072, 1, c=c, interest_set=QUORUM_SETS[c], **KW)
    assert all(_mixed_tile_fraction(t) == 0.0 for t in tasks)


def test_default_layout_is_exact():
    """Pair coverage on the layout the engine builds by default (adaptive 128)."""
    for N in (2048, 4096):
        check_pair_coverage(N, 1, causal=True, sorted_gather=True, align=A, strict_align=False)
    check_pair_coverage(3000, 2, causal=False, sorted_gather=True, align=A, strict_align=False)


# --------------------------------------------------------------- escalation --

def _children_by_build(N, c, iset, parent):
    """What build_tasks produces one level deeper under this parent."""
    deeper = build_tasks(N, int(parent.itr) + 1, c=c, interest_set=iset, **KW)
    return [t for t in deeper if tuple(t.path)[:-1] == tuple(parent.path)]


@pytest.mark.parametrize("N,c,depths", [
    (131072, 7, (1, 2, 3)),
    (100003, 7, (1, 2, 3)),
    (131072, 31, (1, 2)),
    (524288, 7, (1, 2, 3)),
])
def test_escalation_never_grows_a_child(N, c, depths):
    """
    itr=1 -> 2 -> 3 by refine_task, the call the OOM recovery makes: every
    child is aligned, carries a block map, is strictly smaller than its parent,
    and is the very task build_tasks would have built at that depth (so a
    hybrid schedule and a uniform one partition the pairs the same way).
    """
    iset = QUORUM_SETS[c]
    l = len(iset)
    parents = build_tasks(N, depths[0], c=c, interest_set=iset, **KW)
    for depth in depths[1:]:
        # Refine a spread of parents (every path prefix), not just the first.
        sample = parents[:: max(1, len(parents) // 8)]
        next_level = []
        for p in sample:
            kids = refine_task(p, N, c=c, interest_set=iset, **KW)
            assert len(kids) == c
            assert sum(int(k.local_size) for k in kids) == l * int(p.local_size)
            for k in kids:
                assert int(k.itr) == depth
                assert int(k.local_size) < int(p.local_size), (p.path, k.path)
                _assert_block_contiguous(k)
                _assert_no_mixed_tiles(k)
            built = _children_by_build(N, c, iset, p)
            assert [tuple(k.path) for k in kids] == [tuple(b.path) for b in built]
            for k, b in zip(kids, built):
                assert torch.equal(k.token_ids, b.token_ids) and torch.equal(k.group_bits, b.group_bits)
            next_level += kids
        parents = next_level


@pytest.mark.parametrize("c", [7, 31, 73, 133])
def test_max_depth_under_alignment(c):
    """
    ``max_depth_for(align=128)`` is the deepest level the escalation guard
    admits: depth d is reachable while its PARENT holds c*128 tokens (mean
    L = N*(l/c)**(d-1)), i.e. depth <= log_{c/l}(N/(c*128)) + 1, never deeper
    than the token bound, and 0 below c*128 tokens. The engines refuse to
    refine a task AT the cap, so a cap one level short loses recoveries.
    """
    l = len(QUORUM_SETS[c])
    for N in (2 ** 15, 2 ** 17, 2 ** 20, 100003):
        d = max_depth_for(N, c, align=A, l=l)
        assert d <= max_depth_for(N, c)
        if N < c * A:
            assert d == 0
            continue
        assert N * (l / c) ** (d - 1) >= c * A
        assert N * (l / c) ** d < c * A or d == max_depth_for(N, c)
    assert max_depth_for(131072, 7, align=A, l=3) == 6
    assert max_depth_for(c * A - 1, c, align=A, l=l) == 0
    # The token bound is untouched: the recovery of the unaligned layout.
    assert max_depth_for(1 << 24, 7) == 8 and max_depth_for(343, 7) == 3


@pytest.mark.parametrize("N,c,cap", [(131072, 73, 2), (4096, 7, 2), (1 << 20, 133, 2)])
def test_depth_cap_admits_every_refinement_the_guard_admits(N, c, cap):
    """Regression (I0, P0.1 review): the cap was one level stricter than
    ``refine_task``'s own guard, so the OOM recovery refused an itr=1 task that
    refines cleanly (N=131072 c=73: 16K-token task -> 2K-token children)."""
    iset = QUORUM_SETS[c]
    assert max_depth_for(N, c, align=A, l=len(iset)) == cap
    parent = max(build_tasks(N, cap - 1, B=1, H=1, D=64, itemsize=2, c=c, interest_set=iset, pin=False),
                 key=lambda t: t.local_size)
    kids = refine_task(parent, N, B=1, H=1, D=64, itemsize=2, pin=False, c=c, interest_set=iset)
    assert kids and all(0 < k.local_size < parent.local_size for k in kids)


def test_forward_recovers_an_oom_one_level_below_the_old_cap():
    """Regression (I0, P0.1 review): one injected OOM on the 1.9K-token itr=1
    tasks of N=4096, c=7 is recovered by refining them (the old aligned cap of
    1 raised "itr=1 is the deepest decomposition"); the result stays exact."""
    from stream_cqsa.reference import sdpa_reference
    from stream_cqsa.stable_stream import local_stats_torch, stream_cqsa_forward
    g = torch.Generator().manual_seed(0)
    q, k, v = (torch.randn(1, 1, 4096, 32, generator=g) for _ in range(3))
    seen = []

    def inner(q_i, k_i, v_i, bits, **kw):
        seen.append(q_i.shape[1])
        if q_i.shape[1] > 1024:
            raise torch.cuda.OutOfMemoryError("injected")
        return local_stats_torch(q_i, k_i, v_i, bits, **kw)

    out, info = stream_cqsa_forward(q, k, v, itr=1, inner=inner, max_parallel=1, causal=True)
    assert info["depth_escalations"] > 0 and info["itr_max_reached"] == 2 and max(seen) > 1024
    torch.testing.assert_close(out.float(), sdpa_reference(q, k, v, causal=True), rtol=1e-4, atol=1e-4)


def test_strict_refine_refuses_a_degenerate_parent():
    """
    Under a strict alignment (the in-place modes) a parent below c*128 tokens
    cannot give every chunk a whole block; refining it could return a child
    equal to itself, so it is refused with the threshold in the message.
    """
    N, c = 2000, 7
    tasks = build_tasks(N, 1, c=c, seg_align=A, **KW)
    small = [t for t in tasks if int(t.local_size) < c * A]
    assert small, "the layout at N=2000 has parents below 896 tokens"
    with pytest.raises(ValueError, match=str(c * A)):
        refine_task(small[0], N, c=c, seg_align=A, **KW)
    big = [t for t in tasks if int(t.local_size) >= c * A]
    assert all(int(k.local_size) < int(big[0].local_size)
               for k in refine_task(big[0], N, c=c, seg_align=A, **KW))
    assert min_refinable_tokens(c, A) == c * A
    assert min_refinable_tokens(133, A) == 133 * A


def test_default_refine_stops_at_the_block_threshold_but_builds_go_on():
    """
    Refinement stops at c*128 tokens in the default layout too (design 5.1: a
    subproblem that small is never the OOM cause), so the recovery can never
    hand back a child equal to its parent. A fixed-depth build of a tiny N is
    a different matter: a level below c*128 tokens falls back to the even
    split there, so every task of the build is non-empty and attention() keeps
    working on tiny inputs; those tasks are gathered (no block map).
    """
    N, c = 2000, 7
    tasks = build_tasks(N, 1, c=c, **KW)
    assert all(t.extra["seg_base"] is not None for t in tasks)      # level 1 is aligned
    parent = min(tasks, key=lambda t: int(t.local_size))
    assert int(parent.local_size) < c * A
    with pytest.raises(ValueError, match=str(c * A)):
        refine_task(parent, N, c=c, **KW)
    assert min_refinable_tokens(c) == c * A == min_refinable_tokens(c, A)
    assert min_refinable_tokens(c, 0) == c
    deeper = build_tasks(N, 2, c=c, **KW)
    assert len(deeper) == c ** 2
    kids = [t for t in deeper if tuple(t.path)[:-1] == tuple(parent.path)]
    assert sum(int(k.local_size) for k in kids) == 3 * int(parent.local_size)
    assert all(0 < int(k.local_size) < int(parent.local_size) for k in kids)
    assert all(k.extra["seg_base"] is None for k in kids)
    check_pair_coverage(N, 2, causal=True, sorted_gather=True, align=A, strict_align=False)
    assert level_align(c * A - 1, c, A, strict=False) == 1
    assert level_align(c * A, c, A, strict=False) == A
    assert level_align(c * A - 1, c, A, strict=True) == A
    # A tiny N builds (even split at level 1) and its tasks refuse refinement.
    tiny = build_tasks(20, 1, c=c, **KW)
    assert len(tiny) == c and all(int(t.local_size) > 0 for t in tiny)
    with pytest.raises(ValueError, match=str(c * A)):
        refine_task(min(tiny, key=lambda t: int(t.local_size)), 20, c=c, **KW)


# --------------------------------------------------------------- modes / A/B --

def test_layout_modes():
    assert task_layout(None) == (A, False)
    assert task_layout(A) == (A, True)
    assert task_layout(0) == (1, True)
    assert task_seg_align_for(128, in_place=False) is None
    assert task_seg_align_for(128, in_place=True) == 128
    assert task_seg_align_for(0, in_place=True) == 0
    assert task_seg_align_for(0, in_place=False) == 0


def test_legacy_layout_is_reachable_for_ab():
    """seg_align=0 rebuilds the pre-alignment task list: even chunks, no block map."""
    N = 131072
    legacy = build_tasks(N, 1, seg_align=0, **KW)
    aligned = build_tasks(N, 1, **KW)
    Ls = [int(t.local_size) for t in legacy]
    assert max(Ls) - min(Ls) <= 3 and all(t.extra["seg_base"] is None for t in legacy)
    assert sum(Ls) == sum(int(t.local_size) for t in aligned)
    assert [tuple(t.path) for t in legacy] == [tuple(t.path) for t in aligned]
    # A legacy task straddles chunk boundaries inside its tiles; the aligned one does not.
    assert any(_mixed_tile_fraction(t) > 0 for t in legacy)


def test_distributed_shard_partitions_the_default_task_list():
    from stream_cqsa.distributed import _shard, _task_seg_align
    tasks = build_tasks(100003, 2, c=7, seg_align=_task_seg_align({}, in_place=False), **KW)
    world = 3
    shards = [_shard(tasks, r, world) for r in range(world)]
    assert sorted(sum(shards, [])) == sorted(int(t.path_idx) for t in tasks)
    assert _task_seg_align({"seg_align": 0}, in_place=False) == 0
    assert _task_seg_align({"segmented": True}, in_place=True) == 128


# ------------------------------------------------------------- engine (GPU) --

def _sample_rows(N, c, n_random=64, seed=0):
    """Rows around every chunk boundary of both layouts (where the layouts differ
    most) plus random ones; the fp64 reference is only affordable on a subset."""
    from stream_cqsa.reference import chunk_layout
    rows = set()
    for align in (1, A):
        for s in chunk_layout(N, c, align)[1]:
            for r in (s - 1, s, s + 1):
                if 0 <= r < N:
                    rows.add(int(r))
    g = np.random.default_rng(seed)
    rows |= {int(r) for r in g.integers(0, N, n_random)}
    rows |= {0, N - 1}
    return torch.tensor(sorted(rows), dtype=torch.long)


def _rel(a, b):
    return ((a.double() - b.double()).norm() / b.double().norm()).item()


def _fp64_reference(q, k, v, rows, causal, scale):
    """[H, len(rows), D] fp64 attention of the sampled rows against all keys."""
    outs = []
    for h in range(q.shape[1]):
        qh = q[0, h, rows].double()
        s = (qh @ k[0, h].double().T) * scale
        if causal:
            keep = torch.arange(q.shape[2], device=q.device)[None, :] <= rows.to(q.device)[:, None]
            s = s.masked_fill(~keep, float("-inf"))
        outs.append(torch.softmax(s, dim=-1) @ v[0, h].double())
        del s
    return torch.stack(outs)


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
@pytest.mark.parametrize("N", [65536, 100003])
@pytest.mark.parametrize("itr", [1, 2])
@pytest.mark.parametrize("causal", [False, True])
def test_engine_forward_aligned_matches_legacy_and_fp64(N, itr, causal):
    """
    Device-resident classic-engine forward on the aligned task layout versus
    the same call on the legacy layout (norm-relative <= 1e-4 at these N >= 64K;
    small N / deep itr are judged against fp64 instead, see below; the engine
    invariant as tests/test_api.py states it: a different partition of the
    same pairs changes the per-subproblem max shift and the fp16 rounding of P
    inside the kernel, which is an ulp-level per-element difference) and
    versus an fp64 reference on sampled rows, where it must stay within 1.2x
    the error of FA-2 itself (torch SDPA flash, fp16 output).
    """
    import torch.nn.functional as F
    from torch.nn.attention import sdpa_kernel, SDPBackend

    H, D, c = 4, 64, 7
    g = torch.Generator(device="cuda").manual_seed(1234 + N)
    q, k, v = (torch.randn(1, H, N, D, device="cuda", dtype=torch.float16, generator=g) for _ in range(3))
    scale = D ** -0.5
    out_new, info_new = stream_cqsa_forward(q, k, v, itr=itr, causal=causal, c=c, scale=scale)
    out_old, info_old = stream_cqsa_forward(q, k, v, itr=itr, causal=causal, c=c, scale=scale, seg_align=0)
    assert info_new["n_subproblems"] == info_old["n_subproblems"] == c ** itr
    assert info_new["untouched_tokens"] == 0
    rel = _rel(out_new, out_old)
    assert rel <= 1e-4, f"aligned vs legacy engine rel={rel:.3e}"
    lse_abs = (info_new["lse"] - info_old["lse"]).abs().max().item()
    assert lse_abs <= 1e-4, f"lse differs by {lse_abs:.3e}"

    rows = _sample_rows(N, c)
    ref = _fp64_reference(q, k, v, rows, causal, scale)                       # [H, R, D]
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        fa2 = F.scaled_dot_product_attention(q, k, v, is_causal=causal, scale=scale)
    err_new = (out_new[0, :, rows.cuda()].double() - ref).abs().max().item()
    err_fa2 = (fa2[0, :, rows.cuda()].double() - ref).abs().max().item()
    assert err_new <= 1.2 * err_fa2, f"engine fp64 error {err_new:.3e} vs FA-2 {err_fa2:.3e}"
    del q, k, v, out_new, out_old, fa2, ref
    torch.cuda.empty_cache()


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
@pytest.mark.parametrize("N,itr", [(2693, 3), (500, 2)])
def test_engine_forward_small_n_deep_itr_is_judged_against_fp64(N, itr):
    """
    The engine-vs-engine bound (rel 1e-4) is stated for N >= 64K: at tiny N and
    deep itr a subproblem is a few tiles and the partition change moves more of
    the fp16 rounding (measured aligned vs legacy 1.15e-4 at N=2693 itr=3,
    1.22e-4 at N=500 itr=2; P0.1 review). The invariant that holds everywhere is
    the fp64 one: the aligned layout within 1.2x FA-2's error and 1.2x the
    legacy layout's, on every row.
    """
    import torch.nn.functional as F
    from torch.nn.attention import sdpa_kernel, SDPBackend

    H, D = 4, 64
    g = torch.Generator(device="cuda").manual_seed(N)
    q, k, v = (torch.randn(1, H, N, D, device="cuda", dtype=torch.float16, generator=g) for _ in range(3))
    out_new, info = stream_cqsa_forward(q, k, v, itr=itr, causal=True)
    out_old, _ = stream_cqsa_forward(q, k, v, itr=itr, causal=True, seg_align=0)
    assert info["untouched_tokens"] == 0
    ref = F.scaled_dot_product_attention(q.double(), k.double(), v.double(), is_causal=True)
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        fa2 = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    err_new, err_old, err_fa2 = (_rel(o, ref) for o in (out_new, out_old, fa2))
    assert err_new <= 1.2 * err_fa2 and err_new <= 1.2 * err_old, (err_new, err_old, err_fa2)


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
@pytest.mark.parametrize("causal", [False, True])
def test_engine_backward_aligned_matches_legacy(causal):
    """
    The backward gathers, so alignment only moves its tiles: the two layouts
    are equally valid partitions of the same pairs and differ by the kernel's
    fp16 intermediates (P and dS are cast for the matmuls; measured norm-rel
    ~1.6e-4 for dq at N=8192), so each is judged against an fp64 autograd
    reference: the aligned gradients stay within 1.2x FA-2's error and within
    1.2x the legacy layout's, tests/test_backward_default.py's convention.
    """
    import torch.nn.functional as F
    from torch.nn.attention import sdpa_kernel, SDPBackend

    H, D, N, c = 4, 64, 4096, 7
    g = torch.Generator(device="cuda").manual_seed(7)
    q, k, v, do = (torch.randn(1, H, N, D, device="cuda", dtype=torch.float16, generator=g) for _ in range(4))
    out, info = stream_cqsa_forward(q, k, v, itr=1, causal=causal, c=c)
    lse = info["lse"]
    grads_new = stream_cqsa_backward(q, k, v, do, out.half(), lse, itr=1, causal=causal, c=c)
    grads_old = stream_cqsa_backward(q, k, v, do, out.half(), lse, itr=1, causal=causal, c=c, seg_align=0)
    q64, k64, v64 = (t.detach().double().requires_grad_(True) for t in (q, k, v))
    F.scaled_dot_product_attention(q64, k64, v64, is_causal=causal).backward(do.double())
    grads_64 = (q64.grad, k64.grad, v64.grad)
    q16, k16, v16 = (t.detach().clone().requires_grad_(True) for t in (q, k, v))
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        F.scaled_dot_product_attention(q16, k16, v16, is_causal=causal).backward(do)
    grads_fa2 = (q16.grad, k16.grad, v16.grad)
    for a, b, f, r, name in zip(grads_new, grads_old, grads_fa2, grads_64, ("dq", "dk", "dv")):
        err_new, err_old, err_fa2 = _rel(a, r), _rel(b, r), _rel(f, r)
        assert err_new <= 1.2 * err_fa2, f"{name}: aligned fp64 error {err_new:.3e} vs FA-2 {err_fa2:.3e}"
        assert err_new <= 1.2 * err_old, f"{name}: aligned fp64 error {err_new:.3e} vs legacy {err_old:.3e}"
        assert _rel(a, b) <= 1e-3, f"{name}: aligned vs legacy rel={_rel(a, b):.3e}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_distributed_pair_takes_one_kwargs_dict(tmp_path):
    """Regression (I0, P0.1 review): a world=1 forward/backward pair driven with the same
    kwargs (segmented=True, seg_align=None) matches the single-device engine; the backward
    used to raise TypeError on the forward-only key and ``_task_seg_align`` on None."""
    import torch.distributed as dist
    from stream_cqsa.distributed import dist_stream_cqsa_backward, dist_stream_cqsa_forward
    from stream_cqsa.stable_stream import stream_cqsa_backward, stream_cqsa_forward
    if dist.is_initialized():
        pytest.skip("a process group is already initialised")
    dist.init_process_group("nccl", store=dist.FileStore(str(tmp_path / "store"), 1), rank=0, world_size=1)
    try:
        g = torch.Generator(device="cuda").manual_seed(0)
        q, k, v, do = (torch.randn(1, 2, 4096, 64, device="cuda", dtype=torch.float16, generator=g) for _ in range(4))
        for kw in (dict(segmented=True), dict(seg_align=None)):
            out, info = dist_stream_cqsa_forward(q, k, v, itr=1, causal=True, **kw)
            ref, rinfo = stream_cqsa_forward(q, k, v, itr=1, causal=True, **kw)
            assert torch.equal(out, ref) and rinfo["task_seg_align"] > 1
            grads, _ = dist_stream_cqsa_backward(q, k, v, do, out, info["lse"], itr=1, causal=True, **kw)
            rgrads = stream_cqsa_backward(q, k, v, do, ref, rinfo["lse"], itr=1, causal=True,
                                          seg_align=kw.get("seg_align", 128))
            assert all(torch.equal(a, b) for a, b in zip(grads, rgrads))
    finally:
        dist.destroy_process_group()
