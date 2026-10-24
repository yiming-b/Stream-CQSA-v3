"""v3: the accumulator's placement is the planner's frugal default, and the device-resident backward honours it.

Three claims. (1) ``stream_cqsa_backward`` on device-resident Q/K/V with ``accumulate_on_gpu=False`` keeps the
three fp32 gradient accumulators in host memory: the device peak drops by their size and the gradients are the
device-accumulated ones (before v3.0 the flag was ignored there: acc=cpu saved nothing in a device-resident
backward, E4 of results/v3). (2) The planner enumerates "Q/K/V on the device, accumulator on the host" and picks
it where its predicted cost is within the frontier tolerance (15% by default, 2% before v3): the host accumulator
from about 1M tokens up on an A100, the device accumulator below; the monolithic call whenever it fits; ``tol=0``
restores "the fastest configuration". (3) The refit cost model reproduces the measured v3 rows.
"""
from __future__ import annotations

import pytest
import torch

from stream_cqsa.autoconfig import MICRO_TOL, CostModel, PLAN_TOL_DEFAULT, frontier_pick, hardware_from_dict, plan

CUDA = torch.cuda.is_available()
K, M, GIB = 1024, 1024 ** 2, 2 ** 30
H, D = 8, 64


def _hw(gib, host="200GiB"):
    return hardware_from_dict({"cuda:0": f"{gib}GiB", "host": host})


def _rel(a, b):
    a, b = a.float().cpu(), b.float().cpu()
    return ((a - b).norm() / b.norm()).item()


# ---------------------------------------------------------------------------------------------------------------
# the planner
# ---------------------------------------------------------------------------------------------------------------
def test_default_tolerance_and_opt_out():
    assert PLAN_TOL_DEFAULT == pytest.approx(0.15)
    rows = [dict(time=1.00, peak=3.0, acc="gpu"), dict(time=1.10, peak=1.5, acc="cpu"), dict(time=1.30, peak=1.0, acc="cpu", itr=2)]
    assert frontier_pick(rows)[0]["peak"] == 1.5                 # 10% slower for half the memory: taken by default
    assert frontier_pick(rows, 0.0)[0]["peak"] == 3.0            # tol=0: the fastest
    assert frontier_pick(rows, 0.5)[0]["peak"] == 1.0
    # the micro choices within a residency group keep the 2% rule: a wave budget W is never halved for 10%
    grp = [dict(time=1.00, peak=3.0, wave_tokens=4), dict(time=1.10, peak=2.9, wave_tokens=2)]
    assert frontier_pick(grp)[0]["wave_tokens"] == 4


def test_planner_enumerates_the_host_accumulator_on_device_inputs():
    for direction in ("fwd", "bwd"):
        p = plan(N=M, H=H, D=D, hardware=_hw(72), direction=direction, allow_distributed=False, out_bytes_per_el=2,
                 allow_mono=False, quorum_sets={7: (0, 1, 3)})
        dev_cpu = [c for c in p.candidates if c["mode"] == "cqsa" and not c["stream_from_host"] and c["acc"] == "cpu"]
        dev_gpu = [c for c in p.candidates if c["mode"] == "cqsa" and not c["stream_from_host"] and c["acc"] == "gpu"]
        assert dev_cpu and dev_gpu
        # the host accumulator removes the O(N) fp32 terms from the device: at 1M the forward's 2 GiB (+ the output
        # copies), the backward's three gradients (6 GiB)
        c1 = min(c["peak"] for c in dev_cpu if c["itr"] == 1)
        g1 = min(c["peak"] for c in dev_gpu if c["itr"] == 1)
        assert g1 - c1 >= (1.5 if direction == "fwd" else 5.5), (direction, g1, c1)
        # its predicted cost at 1M is within the default tolerance, so the device-resident pick is the host accumulator
        dev = [c for c in p.candidates if c["ok"] and c["mode"] == "cqsa" and not c["stream_from_host"]]
        pick, _ = frontier_pick(dev, p.tol)
        assert pick["acc"] == "cpu", pick
        fastest = min(c["time"] for c in dev)
        assert pick["time"] <= fastest * (1 + PLAN_TOL_DEFAULT + MICRO_TOL)   # the tolerance is on the groups' representatives
        pick0, _ = frontier_pick(dev, 0.0)
        assert pick0["acc"] == "gpu"                              # the opt-out


def test_small_calls_keep_the_device_accumulator():
    # at 256K the host accumulator's fixed costs make it 1.5-2.5x slower (E0/E2): never within the tolerance
    p = plan(N=256 * K, H=H, D=D, hardware=_hw(72), allow_distributed=False, out_bytes_per_el=2, allow_mono=False,
             quorum_sets={7: (0, 1, 3)})
    assert p.acc == "gpu" and not p.stream_from_host, p.name()


def test_monolithic_whenever_it_fits():
    for N in (256 * K, M, 2 * M):
        for direction in ("fwd", "bwd"):
            p = plan(N=N, H=H, D=D, hardware=_hw(72), direction=direction, allow_distributed=False, out_bytes_per_el=2)
            assert p.mode == "mono", (N, direction, p.name())


def test_tol_reaches_the_planner_through_attention():
    import stream_cqsa
    q, k, v = (torch.randn(1, H, 4096, D, dtype=torch.float16) for _ in range(3))
    p = stream_cqsa.attention(q, k, v, is_causal=True, plan_only=True, hardware={"cuda:0": "0.5GiB", "host": "16GiB"}, tol=0.4)
    assert p.tol == pytest.approx(0.4)
    p = stream_cqsa.attention(q, k, v, is_causal=True, plan_only=True, hardware={"cuda:0": "0.5GiB", "host": "16GiB"})
    assert p.tol == pytest.approx(PLAN_TOL_DEFAULT)


def test_cost_model_defaults_reproduce_the_a100_rows():
    """Anchors from results/v3 (A100-SXM4-80GB, fp16, H=8, D=64, c=7, causal): the defaults must stay within 10%."""
    cm = CostModel()
    rows = [  # (N, itr, acc, stream_from_host, direction, measured wall s)
        (M, 1, "gpu", False, "fwd", 5.85), (4 * M, 1, "gpu", False, "fwd", 92.67), (M, 2, "gpu", False, "fwd", 6.08),
        (M, 1, "cpu", False, "fwd", 6.69), (M, 2, "cpu", False, "fwd", 8.89), (M, 1, "cpu", True, "fwd", 7.08),
        (4 * M, 1, "cpu", True, "fwd", 101.47), (M, 1, "gpu", False, "bwd", 20.88), (2 * M, 1, "cpu", True, "bwd", 95.78),
    ]
    for N, itr, acc, host, direction, meas in rows:
        t = cm.cqsa_time(N, 1, H, D, itr, True, acc, host, 1, c=7, l=3, direction=direction)
        assert abs(t / meas - 1) <= 0.10, (N, itr, acc, host, direction, meas, t)
    assert cm.mono_time(M, 1, H, True) == pytest.approx(5.95, rel=0.10)         # FA-2 at 1M


# ---------------------------------------------------------------------------------------------------------------
# the engine
# ---------------------------------------------------------------------------------------------------------------
@pytest.mark.skipif(not CUDA, reason="needs a GPU")
def test_device_resident_backward_accumulates_on_the_host():
    from stream_cqsa.stable_stream import stream_cqsa_backward, stream_cqsa_forward
    torch.manual_seed(0)
    N = 65536
    dev = torch.device("cuda")
    q, k, v, do = (torch.randn(1, H, N, D, device=dev, dtype=torch.float16) for _ in range(4))
    out32, info = stream_cqsa_forward(q, k, v, itr=1, causal=True, accumulate_on_gpu=True, out_device="acc")
    assert not info.get("monolithic") and info["n_subproblems"] == 7
    lse = info["lse"]
    peaks, grads = {}, {}
    for acc_gpu in (True, False):
        torch.cuda.synchronize(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        g = stream_cqsa_backward(q, k, v, do, out32, lse, itr=1, causal=True, accumulate_on_gpu=acc_gpu)
        torch.cuda.synchronize()
        peaks[acc_gpu] = torch.cuda.max_memory_allocated() - base
        grads[acc_gpu] = [t.clone() for t in g]
        assert all(t.device.type == ("cuda" if acc_gpu else "cpu") and t.dtype == torch.float32 for t in g)
    acc_bytes = 3 * N * H * D * 4                                                 # the three fp32 accumulators
    assert peaks[True] - peaks[False] >= 0.8 * acc_bytes, (peaks, acc_bytes)
    for a, b in zip(grads[True], grads[False]):
        assert _rel(a, b) < 1e-5                                                  # the same sums, in host memory


@pytest.mark.skipif(not CUDA, reason="needs a GPU")
def test_attention_trains_with_the_host_accumulator_on_device_inputs():
    """attention() on device Q/K/V under a budget where only the host accumulator fits next to them: plans acc=cpu
    without streaming, runs decomposed, delivers the output and the gradients on the device."""
    import stream_cqsa
    import torch.nn.functional as F
    from tests._decomp import assert_decomposed
    torch.manual_seed(0)
    N = 65536
    leaves = [torch.randn(1, H, N, D, device="cuda", dtype=torch.float16, requires_grad=True) for _ in range(3)]
    hw = {"cuda:0": "0.55GiB", "host": "16GiB"}                               # Q/K/V 0.19 GiB resident; fp32 grads 0.38 GiB do not fit
    p = stream_cqsa.attention(*leaves, is_causal=True, plan_only=True, hardware=hw, evict_inputs=False)
    assert p.mode == "cqsa" and p.acc == "cpu" and not p.stream_from_host, p.name()
    out, info = stream_cqsa.attention(*leaves, is_causal=True, hardware=hw, evict_inputs=False, return_info=True)
    assert_decomposed(info)
    assert info["accumulate_on_gpu"] is False and info["stream_from_host"] is False
    do = torch.randn_like(out)
    out.backward(do)
    got = [t.grad.clone() for t in leaves]
    assert out.is_cuda and all(g.is_cuda and g.dtype == torch.float16 for g in got)
    for t in leaves:
        t.grad = None
    ref = F.scaled_dot_product_attention(*leaves, is_causal=True)
    ref.backward(do)
    assert _rel(out, ref) < 2e-3
    for g, t in zip(got, leaves):
        assert _rel(g, t.grad) < 5e-3
