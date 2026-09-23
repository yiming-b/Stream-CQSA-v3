"""
The v11x experiment kernel (next/kernel/v11x: -DCQSA_NO_ELEMENT_PATH forward, -DCQSA_BWD_NO_CQS backward).

Its contract: on patterns whose tiles are all clear (plain, all-zero bits) it is bit-identical to v11 (out, lse,
dk, dv; dq up to the atomics' drift); on any other pattern every 128-row CTA that meets a tile needing the
per-element path poisons its rows with NaN instead of returning a silently wrong result. The helpers come from
benchmarks/ultimate/exp_build_ab.py, which imports cqsa_cuda_next_{v11,v11x,v9} at import (next/kernel/{stable,v11x}
on PYTHONPATH): everything skips when they are absent. The poison predictor test needs no GPU.
"""
import os, sys

import pytest
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
DEV = torch.device("cuda")
H, N = 8, 16384


def _harness():
    path = os.path.join(REPO, "benchmarks", "ultimate")
    if path not in sys.path:
        sys.path.insert(0, path)
    try:
        import exp_build_ab
    except Exception as e:  # noqa
        pytest.skip(f"exp_build_ab not importable (v11/v11x/v9 kernels missing?): {e}")
    return exp_build_ab


def _tasks(seg_align):
    from stream_cqsa.stable_stream import build_tasks
    return build_tasks(N, 1, B=1, H=H, D=64, itemsize=2, sorted_gather=True, pin=False,
                       c=7, interest_set=(0, 1, 3), seg_align=seg_align)


def test_poison_rows_predictor():
    ab = _harness()
    # three 128-aligned chunks: non-owner A (bit 1), the owner B, non-owner C (bit 2) with a 72-row tail
    bits = torch.tensor([1] * 256 + [0] * 256 + [2] * 200, dtype=torch.int64)
    L = bits.numel()
    rows = torch.arange(L)
    causal = ab.poison_rows(bits, L, True)
    assert torch.equal(causal, (rows < 256) | (rows >= 512)), "causal: every non-owner block meets its masked diagonal tile"
    non_causal = ab.poison_rows(bits, L, False)
    assert torch.equal(non_causal, rows >= 512), "non-causal: only the blocks sharing C's bits with the last column tile"
    # with the owner last, the non-causal kernel's single masking tile is clear for every row block
    bits = torch.tensor([1] * 256 + [2] * 200 + [0] * 256, dtype=torch.int64)
    assert not ab.poison_rows(bits, bits.numel(), False).any()


@cuda
@pytest.mark.parametrize("D", [64, 128])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("pattern", ["plain", "zero"])
def test_forward_bit_identical_on_clear_patterns(D, causal, pattern):
    ab = _harness()
    task = _tasks(seg_align=0)[0]
    L = int(task.local_size)
    pat = ab.patterns_for(task)
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, L, H, D, device=DEV, dtype=torch.float16) for _ in range(3))
    outs = {}
    for name in ("v11", "v11x"):
        with ab.use_module(name):
            outs[name] = ab.fwd(q, k, v, pattern, pat, causal)
    assert torch.equal(outs["v11x"][0], outs["v11"][0]), "out differs from v11"
    assert torch.equal(outs["v11x"][1], outs["v11"][1]), "lse differs from v11"
    assert not outs["v11x"][0].isnan().any()


@cuda
@pytest.mark.parametrize("causal", [True, False])
def test_forward_real_pattern_is_poisoned(causal):
    ab = _harness()
    # aligned: every 128-row block is uniform in chunk; a task with both poisoned and clean blocks
    for task in _tasks(seg_align=128):
        L = int(task.local_size)
        pat = ab.patterns_for(task)
        poison = ab.poison_rows(pat["real"][0], L, causal)
        if 0 < poison.float().mean() < 1:
            break
    else:
        pytest.fail("no task with both poisoned and clean row blocks")
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, L, H, 64, device=DEV, dtype=torch.float16) for _ in range(3))
    with ab.use_module("v11"):
        out_ref, lse_ref = ab.fwd(q, k, v, "real", pat, causal)
    with ab.use_module("v11x"):
        out, lse = ab.fwd(q, k, v, "real", pat, causal)
    o, l = out[0], lse[0].T
    o_ref, l_ref = out_ref[0], lse_ref[0].T
    assert o[poison].isnan().all(), "a tile needing the element path must poison its CTA's out"
    # lse is NaN too, except on a fully masked row (row_sum == 0: a leading non-owner chunk under causal), where
    # normalize_softmax_lse writes +inf whatever row_max holds, in v11 and v11x alike
    lp, lr = l[poison], l_ref[poison]
    assert (lp.isnan() | (lr.isinf() & (lp == lr))).all(), "a poisoned row's lse must be NaN (or v11's +inf)"
    assert torch.equal(o[~poison], o_ref[~poison]) and torch.equal(l[~poison], l_ref[~poison]), \
        "row blocks whose visited tiles are all clear or skipped must still equal v11"


@cuda
def test_backward_no_cqs_matches_cqs_backward_on_zero_pattern():
    ab = _harness()
    task = _tasks(seg_align=0)[0]
    L = int(task.local_size)
    pat = ab.patterns_for(task)
    torch.manual_seed(0)
    q, k, v, do = (torch.randn(1, L, H, 64, device=DEV, dtype=torch.float16) for _ in range(4))
    with ab.use_module("v11"):
        out, lse = ab.fwd(q, k, v, "zero", pat, True)
    o16, lse = out.to(torch.float16).contiguous(), lse.contiguous()
    grads = {}
    for name in ("v11", "v11x"):
        with ab.use_module(name):
            grads[name] = ab.bwd(ab.MODS[name], do, q, k, v, o16, lse, "zero", pat)
    dq_x, dk_x, dv_x = grads["v11x"]
    dq, dk, dv = grads["v11"]
    assert torch.equal(dk_x, dk) and torch.equal(dv_x, dv), "dk/dv are deterministic and must match v11 exactly"
    # dq is accumulated with fp32 atomics in both kernels, so it agrees only up to the accumulation order
    rel = ((dq_x.double() - dq.double()).norm() / dq.double().norm()).item()
    assert rel <= 1e-3, f"dq relative difference {rel:.2e}"
