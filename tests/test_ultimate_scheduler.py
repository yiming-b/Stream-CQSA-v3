"""D1: the multi-device scheduler (stream_cqsa/scheduler.py) and the driver's reductions
(stream_cqsa/distributed.py). CPU only: the scheduler is pure; the collectives run on a 2-process gloo
group, and the full driver runs the engine's pure-torch path on CPU tensors."""
import json
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

from stream_cqsa import scheduler as S

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
P0 = S.CostParams(BH=1, task_s=0.0, tok_s=0.0)          # pure work: ratios are exact


def _ranks(rates, budgets=None):
    budgets = budgets or [0] * len(rates)
    return [S.RankSpec(rate=float(r), budget_bytes=int(b)) for r, b in zip(rates, budgets)]


# ------------------------------------------------------------------ synthetic -----

def test_seven_tasks_four_equal_ranks():
    units = S.synthetic_units([1.0] * 7)
    rr = S.schedule(units, _ranks([1] * 4), params=P0, mode="rr")
    assert rr.ratio == pytest.approx(8 / 7)                                    # 2/7 vs 1/4: 1.14x
    none = S.schedule(units, _ranks([1] * 4), params=P0, split="none")
    assert none.ratio == pytest.approx(8 / 7)                                  # whole tasks cannot do better
    for split in ("children", "rows"):                                         # where splitting is available
        a = S.schedule(units, _ranks([1] * 4), params=P0, split=split, work=S.SyntheticWork())
        assert a.ratio <= 1.05, (split, a.ratio, a.loads)
        assert sum(u.tiles for p in a.per_rank for u in p) == pytest.approx(7.0)   # work conserved


def test_heterogeneous_rates_harmonic_ideal():
    units = S.synthetic_units([1.0] * 7)
    rates = [1.0, 1.0, 0.67, 0.67]
    a = S.schedule(units, _ranks(rates), params=P0, split="children", work=S.SyntheticWork())
    ideal = 7.0 / sum(rates)
    assert a.ideal == pytest.approx(ideal)
    assert a.makespan <= 1.05 * ideal, a.loads
    rr = S.schedule(units, _ranks(rates), params=P0, mode="rr")
    assert rr.ratio > 1.4                                                      # 2 tasks on a 0.67 rank
    # the slow ranks get less work
    work = [sum(u.tiles for u in p) for p in a.per_rank]
    assert max(work[2:]) < min(work[:2])


@pytest.mark.parametrize("rates", [[1] * 3, [1] * 5, [1] * 8, [1, 0.5], [1, 1, 1, 0.5]])
def test_children_split_reaches_tolerance(rates):
    a = S.schedule(S.synthetic_units([1.0] * 7), _ranks(rates), params=P0, split="children", work=S.SyntheticWork())
    assert a.ratio <= 1.05, (rates, a.loads)


def test_memory_budgets_respected():
    units = S.synthetic_units([1.0] * 7, L=1000)
    p = S.CostParams(BH=1, task_s=0.0, tok_s=0.0, base_bytes=100, depth_bytes=50, unit_bytes_per_tok=1, n_par=1)
    # rank 3 cannot hold a whole task (100 + 1000 > 700) but can hold children (L=428: 100 + 50 + 428)
    ranks = _ranks([1] * 4, [10_000, 10_000, 10_000, 700])
    for split in ("none", "children"):
        a = S.schedule(units, ranks, params=p, split=split, work=S.SyntheticWork())
        for r, spec in enumerate(ranks):
            us = a.per_rank[r]
            if us:
                need = p.base_bytes + p.depth_bytes * (len({u.depth for u in us}) - 1) + max(p.unit_peak(u) for u in us)
                assert need <= spec.budget_bytes, (split, r, need)
        assert all(u.L < 1000 for u in a.per_rank[3])
    # nothing fits and nothing can be split -> infeasible
    with pytest.raises(S.Infeasible):
        S.schedule(units, _ranks([1] * 2, [500, 500]), params=p, split="none")


def test_memory_forces_a_split():
    """A unit no rank can hold is refined until its children fit."""
    units = S.synthetic_units([1.0] * 3, L=1000)
    p = S.CostParams(BH=1, task_s=0.0, tok_s=0.0, unit_bytes_per_tok=1, n_par=1)
    a = S.schedule(units, _ranks([1, 1], [600, 600]), params=p, split="children", work=S.SyntheticWork())
    assert all(u.depth == 2 for pr in a.per_rank for u in pr) and a.n_splits >= 3


def test_round_robin_is_the_previous_shard():
    from stream_cqsa.distributed import _shard

    class T:
        def __init__(self, i):
            self.path_idx = i
    tasks = [T(i) for i in range(49)]
    units = S.synthetic_units([1.0] * 49)
    a = S.schedule(units, _ranks([1] * 4), params=P0, mode="rr")
    for r in range(4):
        assert [u.path_idx for u in a.per_rank[r]] == _shard(tasks, r, 4)


def test_descent_beats_plain_lpt_on_unequal_tasks():
    rng = np.random.default_rng(0)
    works = list(rng.uniform(0.5, 0.9, size=49))
    rr = S.schedule(S.synthetic_units(works), _ranks([1] * 4), params=P0, mode="rr")
    lpt = S.schedule(S.synthetic_units(works), _ranks([1] * 4), params=P0, split="none", improve=False)
    best = S.schedule(S.synthetic_units(works), _ranks([1] * 4), params=P0, split="none")
    assert best.makespan <= lpt.makespan and best.makespan <= rr.makespan
    assert best.ratio <= 1.01


def test_equal_task_share():
    assert S.equal_task_share(7, (1.0,) * 4, split="none") == pytest.approx(2 / 7)
    assert S.equal_task_share(7, (1.0,) * 4) <= 1.05 / 4
    assert S.equal_task_share(49, (1.0,) * 8, split="none") == pytest.approx(7 / 49)
    assert S.equal_task_share(7, (1.0, 1.0, 0.67, 0.67)) <= 1.05 / 3.34


_DET_SCRIPT = r"""
import json, sys
from stream_cqsa import scheduler as S
from stream_cqsa.distributed import plan_distributed
out = {}
p0 = S.CostParams(BH=1, task_s=0.0, tok_s=0.0)
for rates in ([1, 1, 1, 1], [1, 1, 0.67, 0.67], [1, 0.5, 1, 0.8, 1]):
    a = S.schedule(S.synthetic_units([1.0 + 0.01 * i for i in range(7)]), [S.RankSpec(r) for r in rates],
                   params=p0, split="children", work=S.SyntheticWork())
    out[str(rates)] = a.digest()
a = plan_distributed(131072, 1, [dict(rate=3e7), dict(rate=3e7), dict(rate=2e7), dict(rate=2e7)], split="children")
out["real"] = a.digest()
print(json.dumps(out))
"""


def test_assignment_identical_across_processes():
    """Every rank computes the assignment independently: it must not depend on the process (hash seed,
    import order); the driver additionally all_gathers the digest and refuses a mismatch."""
    outs = []
    for seed in ("0", "12345"):
        env = dict(os.environ, PYTHONHASHSEED=seed, PYTHONPATH=ROOT + os.pathsep + os.environ.get("PYTHONPATH", ""))
        r = subprocess.run([sys.executable, "-P", "-c", _DET_SCRIPT], capture_output=True, text=True, env=env,
                           timeout=600)
        assert r.returncode == 0, r.stderr[-2000:]
        outs.append(json.loads(r.stdout.strip().splitlines()[-1]))
    assert outs[0] == outs[1]


# ------------------------------------------------------------------ real task lists -----

def _brute_rows(blk_and, causal):
    a = np.asarray(blk_and)
    m = (a[:, None] & a[None, :]) != 0
    tri = np.tril(np.ones((len(a), len(a)), bool)) if causal else np.ones((len(a), len(a)), bool)
    return (tri & ~m).sum(axis=1)


@pytest.mark.parametrize("N,itr,c", [(131072, 1, 7), (131072, 2, 7), (100003, 2, 7), (131072, 1, 31)])
def test_live_tiles_match_the_census_definition(N, itr, c):
    from stream_cqsa.autoconfig import QUORUM_SETS
    from stream_cqsa.interface import cqs_block_summaries
    from stream_cqsa.stable_stream import build_tasks
    tasks = build_tasks(N, itr, B=1, H=1, D=64, itemsize=2, pin=False, c=c, interest_set=QUORUM_SETS[c])
    for t in tasks[:6]:
        _, ba = cqs_block_summaries(t.group_bits, 128)
        for causal in (True, False):
            assert np.array_equal(S.row_live_tiles(ba.numpy(), causal), _brute_rows(ba.numpy(), causal))


@pytest.mark.parametrize("causal", [True, False])
def test_children_partition_parent_work_and_match_the_engine(causal):
    """The split the driver executes: the children's live tiles sum to the parent's exactly (aligned
    layout: every child tile is a whole parent tile), and they are the engine's depth+1 tasks."""
    from stream_cqsa.stable_stream import build_tasks
    N = 131072
    t1 = build_tasks(N, 1, B=1, H=1, D=64, itemsize=2, pin=False)
    t2 = {t.path_idx: t for t in build_tasks(N, 2, B=1, H=1, D=64, itemsize=2, pin=False)}
    wm = S.WorkModel(N, causal=causal)
    for u in wm.units_from_tasks(t1, 1):
        kids = wm.children(u)
        assert len(kids) == 7 and sum(k.tiles for k in kids) == u.tiles
        for k in kids:
            assert k.path_idx == u.path_idx * 7 + k.path[-1]
            assert k.L == t2[k.path_idx].local_size and tuple(t2[k.path_idx].path) == k.path
        g = wm.children(kids[0])
        assert sum(x.tiles for x in g) == kids[0].tiles


def test_strict_and_adaptive_layouts_agree_where_refinable():
    """The itr=1 call with the shared chunk pool builds its tasks strictly aligned, the children call
    adaptively: they must be the same tasks for the children to refine the executed parents."""
    from stream_cqsa.stable_stream import build_tasks
    for N in (131072, 100003):
        a = build_tasks(N, 1, B=1, H=1, D=64, itemsize=2, pin=False, seg_align=128)
        b = build_tasks(N, 1, B=1, H=1, D=64, itemsize=2, pin=False, seg_align=None)
        for x, y in zip(a, b):
            assert torch.equal(x.token_ids, y.token_ids) and torch.equal(x.group_bits, y.group_bits)


def test_real_task_lists_meet_the_targets():
    """1M tokens, c=7: itr=1 on 4 equal ranks (round-robin 1.14x), rates (1, 1, 0.67, 0.67), and the
    unequal itr=2 tasks (live tiles differ by 1.76x between tasks)."""
    from stream_cqsa.distributed import plan_distributed
    N = 1 << 20
    eq = [dict(rate=3e7)] * 4
    het = [dict(rate=3e7), dict(rate=3e7), dict(rate=2e7), dict(rate=2e7)]
    rr = plan_distributed(N, 1, eq, scheduler="rr")
    assert rr.ratio > 1.14
    for ranks in (eq, het):
        a = plan_distributed(N, 1, ranks, split="children")
        assert a.ratio <= 1.05, a.summary()
        assert a.subsets(0)                                                   # executable: {depth: [path_idx]}
    a2 = plan_distributed(N, 2, eq, split="children")
    rr2 = plan_distributed(N, 2, eq, scheduler="rr")
    assert a2.ratio <= 1.01 < rr2.ratio


def test_row_split_is_model_only():
    from stream_cqsa.distributed import _resolve_split
    a = S.schedule(S.synthetic_units([1.0] * 7), _ranks([1] * 4), params=P0, split="rows", work=S.SyntheticWork())
    with pytest.raises(NotImplementedError, match="U1"):
        a.subsets(0)
    with pytest.raises(NotImplementedError, match="U1"):
        _resolve_split("lpt", "rows")
    assert _resolve_split("lpt", "auto") == "children" and _resolve_split("rr", "auto") == "none"


def test_device_rates_seeds_and_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("CQSA_CACHE_DIR", str(tmp_path))
    sxm = S.device_rate("NVIDIA A100-SXM4-80GB")
    pcie = S.device_rate("NVIDIA A100 80GB PCIe")
    h100 = S.device_rate("NVIDIA H100 80GB HBM3")
    assert sxm.source.startswith("measured") and pcie.tile_rate < sxm.tile_rate < h100.tile_rate
    assert "spec seed" in h100.source and "seed" in pcie.source
    assert S.device_rate("NVIDIA A100-PCIE-40GB").tile_rate < pcie.tile_rate
    S.save_device_rate(S.DeviceRate(model="NVIDIA A100 80GB PCIe", tile_rate=1.23e7, source="calibrated test"))
    assert S.device_rate("NVIDIA A100 80GB PCIe").tile_rate == 1.23e7
    assert S.device_rate("NVIDIA A100 80GB PCIe", use_cache=False).tile_rate == pcie.tile_rate


# ------------------------------------------------------------------ gloo, 2 processes -----

def _gloo_worker(rank, world, store_path, result_path):
    import torch.distributed as dist
    from stream_cqsa.distributed import (merge_forward, reduce_gradients, dist_stream_cqsa_forward)
    from stream_cqsa.stable_stream import stream_cqsa_forward
    torch.set_num_threads(1)
    dist.init_process_group("gloo", store=dist.FileStore(store_path, world), rank=rank, world_size=world)
    res = {}
    try:
        B, H, N, D = 1, 2, 1001, 8
        g = torch.Generator().manual_seed(0)
        # partials: rank 0 two (a split rank), rank 1 one; -inf where a partial never touched a token
        parts_all = []
        for _ in range(3):
            o = torch.randn(B, H, N, D, generator=g)
            l = torch.randn(B, H, N, generator=g) * 3
            l[:, :, torch.rand(N, generator=g) < 0.3] = float("-inf")
            parts_all.append((o, l))
        mine = parts_all[:2] if rank == 0 else parts_all[2:]
        dev = torch.device("cpu")
        clone = lambda ps: [(o.clone(), l.clone()) for o, l in ps]           # noqa: E731
        ar, lse_ar, _ = merge_forward(clone(mine), N, B=B, H=H, D=D, dev=dev, reduce="allreduce", chunk_tokens=128)
        rs, lse_rs, st = merge_forward(clone(mine), N, B=B, H=H, D=D, dev=dev, reduce="reduce_scatter",
                                       output="replicated", chunk_tokens=128)
        sh, lse_sh, st2 = merge_forward(clone(mine), N, B=B, H=H, D=D, dev=dev, reduce="reduce_scatter",
                                        output="sharded", chunk_tokens=100)
        ar_sh, _, _ = merge_forward(clone(mine), N, B=B, H=H, D=D, dev=dev, reduce="allreduce", output="sharded",
                                    chunk_tokens=300)
        none_rs, _, _ = merge_forward([] if rank == 1 else clone(parts_all), N, B=B, H=H, D=D, dev=dev,
                                      reduce="reduce_scatter", chunk_tokens=64)
        # reference: the single-process merge of all three partials, in float64
        m = torch.stack([l for _, l in parts_all]).double().amax(0)
        w = [torch.where(torch.isfinite(l.double()), torch.exp(l.double() - m), torch.zeros_like(m)) for _, l in parts_all]
        acc = sum(torch.nan_to_num(o.double() * wi.unsqueeze(-1)) for (o, _), wi in zip(parts_all, w))
        lsum = sum(w)
        ref = acc / lsum.clamp_min(1e-300).unsqueeze(-1)
        lo, hi = st2["token_range"]
        rel = lambda a, b: ((a.double() - b.double()).norm() / b.double().norm()).item()   # noqa: E731

        def ldiff(a, b):
            """max |a - b| over finite entries; inf if the -inf patterns differ."""
            a, b = a.double(), b.double()
            if not torch.equal(torch.isfinite(a), torch.isfinite(b)):
                return float("inf")
            f = torch.isfinite(a)
            return (a[f] - b[f]).abs().max().item() if f.any() else 0.0
        res.update(ar_ref=rel(ar, ref), rs_ar=rel(rs, ar), sh_ar=rel(sh, ar[:, :, lo:hi]),
                   arsh_ar=rel(ar_sh, ar[:, :, lo:hi]), none_ar=rel(none_rs, ar),
                   lse_rs_ar=ldiff(lse_rs, lse_ar), lse_sh_ar=ldiff(lse_sh, lse_ar),
                   lse_ref=ldiff(lse_ar, torch.where(torch.isfinite(m), m + torch.log(lsum), m)),
                   token_range=[lo, hi], sh_shape=list(sh.shape))
        # gradients: rank 0 two sets (a split rank), rank 1 one set
        gsets = [tuple(torch.randn(B, H, N, D, generator=g) for _ in range(3)) for _ in range(3)]
        mine_g = gsets[:2] if rank == 0 else gsets[2:]
        cl = lambda ss: [tuple(x.clone() for x in s) for s in ss]            # noqa: E731
        gar, _ = reduce_gradients(cl(mine_g), N, B=B, H=H, D=D, dev=dev, reduce="allreduce", chunk_tokens=128)
        grs, gst = reduce_gradients(cl(mine_g), N, B=B, H=H, D=D, dev=dev, reduce="reduce_scatter", chunk_tokens=100)
        gref = [sum(s[i] for s in gsets) for i in range(3)]
        glo, ghi = gst["token_range"]
        res.update(g_ar_ref=max(rel(a, b) for a, b in zip(gar, gref)),
                   g_rs_ar=max(rel(a, b[:, :, glo:ghi]) for a, b in zip(grs, gar)))
        # the full driver on the engine's CPU path: lpt with children forced (tol=0) and rr, both reductions
        Ne = 4096
        q, k, v = (torch.randn(1, 2, Ne, 16, generator=torch.Generator().manual_seed(s)) for s in (1, 2, 3))
        ref_o, ref_i = stream_cqsa_forward(q, k, v, itr=1, causal=True, allow_escalation=False)
        for sched, red, out_mode in (("rr", "allreduce", "replicated"), ("lpt", "allreduce", "replicated"),
                                     ("lpt", "reduce_scatter", "sharded")):
            o, info = dist_stream_cqsa_forward(q, k, v, itr=1, causal=True, allow_escalation=False, scheduler=sched,
                                               reduce=red, output=out_mode, tol=0.0, chunk_tokens=1000,
                                               rate=1e3 if rank == 0 else 5e2)   # work >> the 3 ms per-unit overhead
            a, b = info["token_range"]
            r_o = ref_o if out_mode == "replicated" else ref_o[:, :, a:b]
            res[f"drv_{sched}_{red}"] = dict(rel=rel(o, r_o), lse=ldiff(info["lse"], ref_i["lse"]),
                                            units=info["units_mine"], digest=info["assignment"]["digest"],
                                            split=info["split"], depths=[c_["depth"] for c_ in info["depth_calls"]])
    finally:
        dist.destroy_process_group()
    with open(result_path.replace(".json", f"_{rank}.json"), "w") as f:
        json.dump(res, f)


def test_gloo_reductions_and_driver(tmp_path):
    import torch.multiprocessing as mp
    store = str(tmp_path / "store")
    out = str(tmp_path / "res.json")
    mp.spawn(_gloo_worker, args=(2, store, out), nprocs=2, join=True)
    rs = [json.load(open(out.replace(".json", f"_{r}.json"))) for r in range(2)]
    for r in rs:
        assert r["ar_ref"] <= 1e-6                          # fp32 merge vs float64
        assert r["rs_ar"] <= 1e-7 and r["sh_ar"] <= 1e-7 and r["arsh_ar"] <= 1e-7 and r["none_ar"] <= 1e-7
        assert r["lse_rs_ar"] <= 1e-6 and r["lse_sh_ar"] <= 1e-6 and r["lse_ref"] <= 1e-5
        assert r["g_ar_ref"] <= 1e-6 and r["g_rs_ar"] <= 1e-7
    assert rs[0]["token_range"] == [0, 500] and rs[1]["token_range"] == [500, 1001]
    assert rs[0]["sh_shape"] == [1, 2, 500, 8] and rs[1]["sh_shape"] == [1, 2, 501, 8]
    for key in ("drv_rr_allreduce", "drv_lpt_allreduce", "drv_lpt_reduce_scatter"):
        a, b = rs[0][key], rs[1][key]
        assert a["digest"] == b["digest"]
        assert a["rel"] <= 1e-5 and b["rel"] <= 1e-5 and a["lse"] <= 1e-4 and b["lse"] <= 1e-4, key
    # lpt with tol=0 on (1.0, 0.5) rates splits into children: two depths somewhere, the fast rank more work
    lpt = rs[0]["drv_lpt_allreduce"], rs[1]["drv_lpt_allreduce"]
    assert lpt[0]["split"] == "children"
    assert any(len(x["depths"]) > 1 or 2 in x["depths"] for x in lpt)


# ------------------------------------------------------------------ CUDA: 2 ranks sharing one GPU (gloo) -----

def _gpu_worker(rank, world, store_path, result_path):
    """Two ranks on one GPU (gloo carries the collectives through host memory): the CUDA kernel runs the
    split units, the host-streamed shared-chunk forward drops the pool for its depth-2 call, and the
    backward runs the same assignment."""
    import torch.distributed as dist
    import torch.nn.functional as F
    from stream_cqsa.distributed import dist_stream_cqsa_forward, dist_stream_cqsa_backward
    from stream_cqsa.stable_stream import stream_cqsa_forward, stream_cqsa_backward
    torch.cuda.set_device(0)
    dist.init_process_group("gloo", store=dist.FileStore(store_path, world), rank=rank, world_size=world)
    res = {}
    try:
        B, H, N, D = 1, 2, 32768, 64
        g = torch.Generator().manual_seed(0)
        q, k, v, do = (torch.randn(B, N, H, D, generator=g).half().transpose(1, 2) for _ in range(4))
        rel = lambda a, b: ((a.double() - b.double().to(a.device)).norm() / b.double().norm()).item()  # noqa: E731
        # float64 reference on a row subset (the last rows see every key)
        rows = torch.arange(N - 256, N)
        qd, kd, vd = (t.cuda().double() for t in (q, k, v))
        s = (qd[:, :, rows] @ kd.transpose(-1, -2)) * D ** -0.5
        s = s.masked_fill(torch.arange(N, device="cuda")[None, :] > rows.cuda()[:, None], float("-inf"))
        ref64 = (torch.softmax(s, -1) @ vd).cpu()
        del qd, kd, vd, s
        kw = dict(causal=True, allow_escalation=False, max_parallel=1)
        for name, extra, inputs in (
                ("host_shared", dict(stream_from_host=True, shared_chunks=True, out_device="acc"), (q, k, v)),
                ("device", dict(), tuple(t.cuda() for t in (q, k, v)))):
            one, i1 = stream_cqsa_forward(*inputs, itr=1, **kw, **extra)
            for red in ("allreduce", "reduce_scatter"):
                o, info = dist_stream_cqsa_forward(*inputs, itr=1, **kw, **extra, scheduler="lpt", tol=0.0,
                                                   reduce=red, rate=1e3 if rank == 0 else 5e2)
                full = o
                if red == "reduce_scatter":           # replicated by default: the whole output on every rank
                    assert o.shape[2] == N
                res[f"{name}_{red}"] = dict(rel_single=rel(full, one), err64=rel(full[:, :, rows].cpu(), ref64),
                                           err64_single=rel(one[:, :, rows].cpu(), ref64),
                                           units=info["units_mine"], depths=[c_["depth"] for c_ in info["depth_calls"]])
        # backward, device-resident, split units
        qc, kc, vc, doc = (t.cuda().contiguous() for t in (q, k, v, do))
        out1, i1 = stream_cqsa_forward(qc, kc, vc, itr=1, **kw)
        g1 = stream_cqsa_backward(qc, kc, vc, doc, out1.half(), i1["lse"], itr=1, causal=True, allow_escalation=False)
        for red in ("allreduce", "reduce_scatter"):
            gd, binfo = dist_stream_cqsa_backward(qc, kc, vc, doc, out1.half(), i1["lse"], itr=1, causal=True,
                                                  allow_escalation=False, scheduler="lpt", tol=0.0, reduce=red,
                                                  rate=1e3 if rank == 0 else 5e2)
            lo, hi = binfo["token_range"]
            sl = (lambda t: t) if red == "allreduce" else (lambda t: t[:, :, lo:hi])
            res[f"bwd_{red}"] = dict(rel=max(rel(a, sl(b)) for a, b in zip(gd, g1)), units=binfo["units_mine"],
                                     depths=[c_["depth"] for c_ in binfo["depth_calls"]])
    finally:
        dist.destroy_process_group()
    with open(result_path.replace(".json", f"_{rank}.json"), "w") as f:
        json.dump(res, f)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_two_ranks_one_gpu_split_units(tmp_path):
    import torch.multiprocessing as mp
    out = str(tmp_path / "res.json")
    mp.spawn(_gpu_worker, args=(2, str(tmp_path / "store"), out), nprocs=2, join=True)
    rs = [json.load(open(out.replace(".json", f"_{r}.json"))) for r in range(2)]
    assert any(2 in r["host_shared_allreduce"]["depths"] for r in rs)       # a split unit ran at depth 2
    for r in rs:
        for key in ("host_shared_allreduce", "host_shared_reduce_scatter", "device_allreduce", "device_reduce_scatter"):
            x = r[key]
            assert x["rel_single"] <= 2e-3, (key, x)                       # a different tile packing: fp16 P rounding
            assert x["err64"] <= 1.2 * x["err64_single"] + 1e-4, (key, x)
        assert r["bwd_allreduce"]["rel"] <= 2e-3 and r["bwd_reduce_scatter"]["rel"] <= 2e-3, r


# ------------------------------------------------------------------ planner (autoconfig) -----

def test_planner_device_rates_and_heterogeneous_worlds(monkeypatch, tmp_path):
    monkeypatch.setenv("CQSA_CACHE_DIR", str(tmp_path))
    from stream_cqsa.autoconfig import CostModel, hardware_from_dict, plan
    hw = hardware_from_dict({"cuda:0": {"mem": "40GiB", "model": "NVIDIA A100-PCIE-40GB"},
                             "cuda:1": {"mem": "80GiB", "model": "NVIDIA A100-SXM4-80GB"},
                             "cuda:2": {"mem": "80GiB", "model": "NVIDIA A100-SXM4-80GB", "rate": 3.0e7},
                             "host": "500GiB"})
    assert not hw.homogeneous()
    assert [d.tile_rate for d in hw.devices] == [S.seed_rate("NVIDIA A100-PCIE-40GB").tile_rate,
                                                 S.seed_rate("NVIDIA A100-SXM4-80GB").tile_rate, 3.0e7]
    assert hw.devices[2].rate_source == "given" and hw.devices[1].rate_source.startswith("measured")
    p = plan(N=1 << 22, hardware=hw, engines=("classic",))
    assert p.devices[0] == "cuda:1" and (p.world == 1 or p.devices[:2] == ["cuda:1", "cuda:2"])   # fastest first
    assert len(p.device_rates) == p.world
    # homogeneous "same" devices share one rate
    hw4 = hardware_from_dict({f"cuda:{i}": "80GiB" for i in range(4)} | {"host": "500GiB"})
    assert len({d.tile_rate for d in hw4.devices}) == 1 and hw4.homogeneous()
    # the multi-device time follows the driver's scheduler: splitting beats whole tasks at 7 on 4, slower
    # ranks cost time, and whole tasks on equal ranks are the previous ceil(n/world)/n model
    args = (1 << 22, 1, 8, 64, 1, True, "gpu", False, 1)
    whole, split = CostModel(dist_split="none"), CostModel()
    t1 = split.cqsa_time(*args, world=1)
    t_rr, t_lpt = whole.cqsa_time(*args, world=4), split.cqsa_time(*args, world=4)
    assert t_lpt < t_rr
    comm_fixed = t_rr - t1 * (2 / 7) * whole.dist_imbalance
    assert t_lpt - comm_fixed == pytest.approx(t1 * S.equal_task_share(7, (1.0,) * 4) * split.dist_imbalance)
    t_het = split.cqsa_time(*args, world=4, rates=(1.0, 1.0, 0.67, 0.67))
    assert t_lpt < t_het < whole.cqsa_time(*args, world=4, rates=(1.0, 1.0, 0.67, 0.67))
