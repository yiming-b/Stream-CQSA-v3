"""
Every quorum set computes exact attention when pinned through ``stream_cqsa.attention(..., c=)``.

Regression for the v2.2.3 c=73 bug (docs/ultimate/progress/BUG_v223_c73.md): ``attention(..., itr=1, c=73)``
pinned ``c`` but kept the plan's interest set ``(0, 1, 3)`` (the c=7 set), so the classic engine ran 73
subproblems of 3 chunks each -- most chunk pairs were never gathered -- and returned a result with a
norm-relative error of 1.3 against flash attention, in a third of the time. Every c other than 7 was wrong
(c=31: 0.72, c=57: 1.1, c=91: 1.4, c=133: 1.8); c=7 was correct only because its set is the default.

v3's ``attention()`` re-plans a pinned ``c`` with ``QUORUM_SETS[c]`` (commit 837e2d9), and since this test
the task builders refuse a pair that is not a perfect difference set (``reference.check_quorum_set``), so
the engine entry points cannot be handed the mismatched pair either.

Rows: every c in ``QUORUM_SETS`` x N in {65536, 131072} x itr in {1, 2}, fp16, causal, B=1 H=4 D=64, on the
classic engine (``kernel="cuda"``), forward and backward against torch SDPA (flash backend) on the same
inputs; the decomposition is asserted with ``tests/_decomp.py`` and the forward info must report
``c**itr`` subproblems and ``QUORUM_SETS[c]``. The planner caps a pinned depth at 4096 subproblems, so the
itr=2 rows of c in {73, 91, 133} are the planner's refusal (a ``ValueError``), never a silent substitute.
Shared vis1 GPU: correctness only, < 2 GiB per row.
"""
import pytest
import torch
import torch.nn.functional as F

from _decomp import attention_decomposed, assert_decomposed

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
B, H, D = 1, 4, 64
NS = (65536, 131072)
TASK_CAP = 4096                       # autoconfig.plan: a pinned (c, itr) with c**itr > 4096 tasks is not a candidate
# norm-relative error vs SDPA-flash fp16 on the same inputs: measured 1.3e-4..2.4e-4 (out) / 3.5e-4 (grads)
# for every c at these N (A100, v13 module); the v2.2.3 failure is ~1e0
TOL_OUT, TOL_GRAD = 1e-3, 1.5e-3


def _quorum_sets():
    from stream_cqsa.autoconfig import QUORUM_SETS
    return QUORUM_SETS


def _rel(a, b):
    return ((a.double() - b.double()).norm() / b.double().norm()).item()


def _inputs(N):
    g = torch.Generator(device="cuda").manual_seed(N)
    q, k, v = (torch.randn(B, H, N, D, device="cuda", dtype=torch.float16, generator=g) for _ in range(3))
    dout = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16, generator=g)
    return q, k, v, dout


def _reference(q, k, v, dout):
    from torch.nn.attention import sdpa_kernel, SDPBackend
    qr, kr, vr = (t.clone().requires_grad_(True) for t in (q, k, v))
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        out = F.scaled_dot_product_attention(qr, kr, vr, is_causal=True)
    grads = torch.autograd.grad(out, (qr, kr, vr), dout)
    return out.detach(), [g_.detach() for g_ in grads]


ROWS = [(N, c, itr) for N in NS for c in sorted(_quorum_sets()) for itr in (1, 2)]
RUNNABLE = [r for r in ROWS if r[1] ** r[2] <= TASK_CAP]
CAPPED = [r for r in ROWS if r[1] ** r[2] > TASK_CAP]


@cuda
@pytest.mark.parametrize("N,c,itr", RUNNABLE, ids=[f"N{N}-c{c}-itr{itr}" for N, c, itr in RUNNABLE])
def test_pinned_quorum_set_is_exact(N, c, itr):
    QUORUM_SETS = _quorum_sets()
    q, k, v, dout = _inputs(N)
    ref, ref_g = _reference(q, k, v, dout)
    # forward alone: the info names the subproblem count and the interest set that ran
    with torch.no_grad():
        out, info = attention_decomposed(q, k, v, is_causal=True, kernel="cuda", itr=itr, c=c, engine="classic")
    assert_decomposed(info, itr=itr, c=c, engine="classic", n_subproblems=c ** itr)
    assert tuple(info["interest_set"]) == QUORUM_SETS[c], info
    r_out = _rel(out, ref)
    assert r_out < TOL_OUT, f"N={N} c={c} itr={itr}: forward rel error {r_out:.2e} (v2.2.3's mismatched set gives ~1e0)"
    del out
    # forward + backward through autograd (the harness's path: direction "bwd")
    qg, kg, vg = (t.clone().requires_grad_(True) for t in (q, k, v))
    out, info = attention_decomposed(qg, kg, vg, is_causal=True, kernel="cuda", itr=itr, c=c, bwd_itr=itr, engine="classic")
    r_out = _rel(out, ref)
    assert r_out < TOL_OUT, f"N={N} c={c} itr={itr}: forward (grad path) rel error {r_out:.2e}"
    grads = torch.autograd.grad(out, (qg, kg, vg), dout)
    r_g = [_rel(a, b) for a, b in zip(grads, ref_g)]
    assert max(r_g) < TOL_GRAD, f"N={N} c={c} itr={itr}: gradient rel errors {['%.2e' % x for x in r_g]}"


@cuda
@pytest.mark.parametrize("N,c,itr", CAPPED, ids=[f"N{N}-c{c}-itr{itr}" for N, c, itr in CAPPED])
def test_pinned_depth_over_task_cap_is_refused(N, c, itr):
    """A pinned (c, itr) beyond the planner's task cap raises; it never runs another configuration."""
    import stream_cqsa
    q, k, v, _ = _inputs(N)
    with torch.no_grad(), pytest.raises(ValueError, match="no decomposition"):
        stream_cqsa.attention(q, k, v, is_causal=True, kernel="cuda", itr=itr, c=c)


# ---- the engine layer refuses the mismatched pair (CPU-only: the task builders run before any kernel) ----

MISMATCHED = [(73, (0, 1, 3)), (31, (0, 1, 3)), (7, (0, 1, 2)), (7, (0, 1, 3, 9)), (13, (0, 0, 3, 9)), (3, (0, 1, 3))]


@pytest.mark.parametrize("c,iset", MISMATCHED, ids=[f"c{c}-{'_'.join(map(str, s))}" for c, s in MISMATCHED])
def test_task_builders_refuse_mismatched_quorum_set(c, iset):
    from stream_cqsa.reference import check_quorum_set, is_perfect_difference_set
    from stream_cqsa.stable_stream import build_tasks, refine_task
    from stream_cqsa.native_wave import wave_tasks
    assert not is_perfect_difference_set(c, iset)
    with pytest.raises(ValueError, match="not a perfect difference set"):
        check_quorum_set(c, iset)
    with pytest.raises(ValueError, match="not a perfect difference set"):
        build_tasks(8192, 1, B=1, H=1, D=D, itemsize=2, pin=False, c=c, interest_set=iset)
    with pytest.raises(ValueError, match="not a perfect difference set"):
        wave_tasks(8192, 1, c, iset, H=1, D=D, itemsize=2)
    parent = build_tasks(8192, 1, B=1, H=1, D=D, itemsize=2, pin=False, c=7, interest_set=(0, 1, 3))[0]
    with pytest.raises(ValueError, match="not a perfect difference set"):
        refine_task(parent, 8192, B=1, H=1, D=D, itemsize=2, pin=False, c=c, interest_set=iset)


def test_every_quorum_set_passes_the_guard():
    from stream_cqsa.reference import check_quorum_set, is_perfect_difference_set, QUORUM_SETS
    from stream_cqsa import autoconfig
    assert autoconfig.QUORUM_SETS is QUORUM_SETS                     # one table, re-exported
    for c, iset in QUORUM_SETS.items():
        assert is_perfect_difference_set(c, iset) and c == len(iset) ** 2 - len(iset) + 1
        check_quorum_set(c, iset)
    with pytest.raises(ValueError, match=r"interest_set=\(0, 1, 3, 7, 15, 31, 36, 54, 63\)"):   # the message names the fix
        check_quorum_set(73, (0, 1, 3))


@cuda
def test_attention_rejects_explicit_mismatched_pair():
    """The v2.2.3 construction, written out: attention(..., c=73, interest_set=(0, 1, 3)) is an error, not a result."""
    import stream_cqsa
    q, k, v, _ = _inputs(16384)
    with torch.no_grad(), pytest.raises(ValueError, match="not a perfect difference set"):
        stream_cqsa.attention(q, k, v, is_causal=True, kernel="cuda", itr=1, c=73, interest_set=(0, 1, 3))
