"""
The v12b backward (next/kernel/v12b, module cqsa_cuda_next_v12b): census + ALIGNED run-table backward, LEGACY = v11.

Contract: with CQS on, a census kernel classifies every tile the backward visits (per column block: live / dead /
mixed, from the 64-token summaries) and writes per-column runs of live m blocks. When no tile is mixed the ALIGNED
instantiation (upstream FlashAttention-2's seqk-parallel loop over the runs) computes the call: dk and dv are
torch.equal to v11's CQS backward (same tiles, same order, dead tiles contribute exact zeros in both) and dq agrees
within the fp32 atomics' drift; otherwise LEGACY (v11's own code) does. CQS off runs the same loop without a run
table (== v11's CQS-off backward for dk/dv). Checked on the shared vis1 GPU at small L: the census against a numpy
reference, aligned real tasks (c = 7, 31, 73; itr 1-2; ragged L), zero / all-ones / plain, causal and non-causal,
D = 64 and 128, a caller-supplied dsoftmax_sum (the engine's call), automatic LEGACY routing of random / half /
unaligned patterns, forced LEGACY, deterministic=True reruns, and float64 gradients against flash_attn's error.
Helpers come from benchmarks/ultimate/bwd_ab.py; every GPU test skips when the extensions are absent
(next/kernel/{stable,v11x,v12b} on PYTHONPATH, the variant dirs last). The census-reference test needs no GPU.
"""
import functools, os, sys

import numpy as np
import pytest
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
DEV = torch.device("cuda")
H = 4


def _harness(gpu=True):
    path = os.path.join(REPO, "benchmarks", "ultimate")
    if path not in sys.path:
        sys.path.insert(0, path)
    import bwd_ab
    if gpu:
        try:
            bwd_ab.load_modules(("v11", "v12b"))
        except Exception as e:  # noqa
            pytest.skip(f"cqsa_cuda_next_v11/v12b not importable: {e}")
    return bwd_ab


def _inputs(L, D, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return [torch.randn(1, L, H, D, device=DEV, dtype=torch.float16, generator=g) for _ in range(4)]


@functools.lru_cache(maxsize=None)
def _tasks(N, c, itr, seg_align=128):
    """The c/itr decomposition of N sorted by (L, path); cached, c=73 builds 73 tasks."""
    ab = _harness(gpu=False)
    return sorted(ab.c_tasks(N, c, itr, seg_align=seg_align), key=lambda t: (int(t.local_size), t.path))


def _task(N, c, itr=1, pick=0):
    tasks = _tasks(N, c, itr)
    return tasks[pick % len(tasks)]


def _dq_tol(ref):
    """The fp32 atomics' drift band on fp16 dq: 2 fp16 ulps at the largest |dq| (one rounding flip either way)."""
    return max(2.0 ** -9 * ref.float().abs().max().item(), 1e-6)


def _compare(ab, tabs, L, D, causal, flags=0, seed=0, dsum=False):
    """v12b vs v11 on one call (inputs, o and lse from v11's forward with the same pattern). Returns (v12b, v11)."""
    q, k, v, do = _inputs(L, D, seed)
    o16, lse = ab.cqs_fwd(ab.MODS["v11"], q, k, v, tabs, causal)
    ds = None
    if dsum:
        ds = (do.float() * o16.float()).sum(-1).transpose(1, 2).contiguous()      # [1, H, L]
    ref = ab.cqs_bwd(ab.MODS["v11"], do, q, k, v, o16, lse, tabs, causal, dsum=ds)
    out = ab.cqs_bwd(ab.MODS["v12b"], do, q, k, v, o16, lse, tabs, causal, flags=flags, dsum=ds)
    return out, ref


def _assert_match(out, ref, what):
    for name, a, b in zip(("dk", "dv"), out[1:], ref[1:]):
        assert torch.equal(a, b), f"{what}: {name} differs from v11 (max |d| {(a.float() - b.float()).abs().max().item():.3e})"
    d = (out[0].float() - ref[0].float()).abs().max().item()
    assert d <= _dq_tol(ref[0]), f"{what}: dq max |d| {d:.3e} vs v11 beyond the atomics' drift ({_dq_tol(ref[0]):.3e})"
    for name, x in zip(("dq", "dk", "dv"), out):
        assert torch.isfinite(x).all(), f"{what}: {name} not finite"


# ------------------------------------------------------------------------------------------------ census
def test_census_reference_runs():
    ab = _harness(gpu=False)
    # 128-token blocks: owner (0), non-owner A (bit 1) x2, owner, non-owner A again, ragged non-owner B (bit 2)
    bits = np.array([0] * 128 + [1] * 256 + [0] * 128 + [1] * 128 + [2] * 70, dtype=np.int64)
    route, cols = ab.census_ref(bits, 128, 128, causal=True)
    assert route == 0
    assert cols[0] == [[5, 0]]                        # owner column: every row block at and below its diagonal
    assert cols[1] == [[5, 5], [3, 3]]               # A column: skips its own blocks 1, 2 (diagonal) and 4
    assert cols[2] == [[5, 5], [3, 3]]
    assert cols[3] == [[5, 3]]                       # owner column
    assert cols[4] == [[5, 5]]                       # A again: its own diagonal is dead
    assert cols[5] == []                             # B (ragged): its only visited tile is its own diagonal
    route_nc, cols_nc = ab.census_ref(bits, 128, 128, causal=False)
    assert route_nc == 0 and cols_nc[1] == [[5, 5], [3, 3], [0, 0]] and cols_nc[5] == [[4, 0]]
    # hdim128's backward tile: 64 rows x 128 columns
    route, cols = ab.census_ref(bits, 64, 128, causal=True)
    assert route == 0 and cols[1] == [[11, 10], [7, 6]]    # rows 64-row blocks 10, 11 are B (704-709 ragged)
    # an unaligned boundary makes a tile mixed; more runs than slots overflow
    route, _ = ab.census_ref(np.array([0] * 100 + [1] * 156, dtype=np.int64), 128, 128, causal=True)
    assert route & 1
    comb = np.repeat(np.arange(40) % 2, 128).astype(np.int64)   # alternate owner / non-owner blocks: 20 runs
    route, cols = ab.census_ref(comb, 128, 128, causal=False)
    assert route & 2 and len(cols[1]) == ab.RUN_SLOTS - 1


def _census_cases():
    cases = []
    for N, c, itr in ((32768, 7, 1), (131072, 7, 2), (65536, 31, 1), (131072, 73, 1), (100003, 7, 1)):
        for t in _tasks(N, c, itr)[:4]:
            cases.append((f"real c={c} itr={itr} N={N} path={t.path}", t.group_bits, True))
    L = 12000
    cases += [("zero", torch.zeros(L, dtype=torch.int64), True),
              ("allones", torch.full((L,), -1, dtype=torch.int64), True),
              ("random3bit", torch.randint(0, 8, (L,), generator=torch.Generator().manual_seed(0)), False),
              ("half", torch.cat([torch.zeros(L // 2 - 37, dtype=torch.int64), torch.ones(L - L // 2 + 37, dtype=torch.int64)]), False)]
    t = _tasks(32768, 7, 1, seg_align=0)[0]
    cases.append(("unaligned c=7 (seg_align=0)", t.group_bits, False))
    return cases


@cuda
@pytest.mark.parametrize("D", [64, 128])
@pytest.mark.parametrize("causal", [True, False])
def test_census_matches_reference(D, causal):
    ab = _harness()
    for what, bits, aligned in _census_cases():
        route, runs, tile = ab.census(ab.MODS["v12b"], ab.tables(bits), causal, D)
        assert tile == ((128, 128) if D == 64 else (64, 128)), f"A100 backward tile for D={D}: {tile}"
        ref_route, ref_cols = ab.census_ref(bits.numpy(), *tile, causal=causal)
        assert route == ref_route, f"{what}: route {route} vs reference {ref_route}"
        assert [ab.runs_prefix(r) for r in runs] == ref_cols, f"{what}: run table differs from the reference"
        if aligned:
            assert route == 0, f"{what}: a 128-aligned pattern must route to ALIGNED"
        else:
            assert route & 1, f"{what}: a mixed pattern must route to LEGACY"


# ------------------------------------------------------------------------------------------------ ALIGNED == v11
ALIGNED_CASES = [  # (N, c, itr, pick): real 128-aligned tasks, L up to ~28K; N=100003 gives ragged L
    (32768, 7, 1, 0), (32768, 7, 1, 3), (65536, 7, 1, 6), (131072, 7, 2, 0), (131072, 7, 2, 25),
    (65536, 31, 1, 0), (65536, 31, 1, 17), (131072, 73, 1, 0), (131072, 73, 1, 40), (100003, 7, 1, 0),
    (100003, 7, 1, 5),
]


@cuda
@pytest.mark.parametrize("D", [64, 128])
@pytest.mark.parametrize("causal", [True, False])
def test_aligned_real_tasks_match_v11(D, causal):
    ab = _harness()
    for N, c, itr, pick in ALIGNED_CASES:
        t = _task(N, c, itr, pick)
        L = int(t.local_size)
        tabs = ab.task_tables(t)
        assert ab.census(ab.MODS["v12b"], tabs, causal, D)[0] == 0, f"c={c} itr={itr} L={L}: census must pick ALIGNED"
        out, ref = _compare(ab, tabs, L, D, causal, seed=pick)
        _assert_match(out, ref, f"c={c} itr={itr} N={N} path={t.path} L={L} (L%128={L % 128}) D={D} causal={causal}")


@cuda
@pytest.mark.parametrize("causal", [True, False])
def test_engine_call_with_dsoftmax_sum(causal):
    """The engine passes a global dsoftmax_sum (no preprocess pass, dq_accum zeroed by the entry)."""
    ab = _harness()
    for N, c, itr, pick in ((32768, 7, 1, 3), (100003, 7, 1, 5), (131072, 7, 2, 25)):
        t = _task(N, c, itr, pick)
        out, ref = _compare(ab, ab.task_tables(t), int(t.local_size), 64, causal, seed=pick, dsum=True)
        _assert_match(out, ref, f"dsoftmax_sum given, c={c} itr={itr} path={t.path}")


@cuda
@pytest.mark.parametrize("D", [64, 128])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("pattern", ["zero", "allones"])
@pytest.mark.parametrize("L", [16384, 12345])
def test_aligned_synthetic_match_v11(pattern, L, D, causal):
    ab = _harness()
    bits = torch.zeros(L, dtype=torch.int64) if pattern == "zero" else torch.full((L,), -1, dtype=torch.int64)
    tabs = ab.tables(bits)
    assert ab.census(ab.MODS["v12b"], tabs, causal, D)[0] == 0
    out, ref = _compare(ab, tabs, L, D, causal)
    _assert_match(out, ref, f"{pattern} L={L} D={D} causal={causal}")
    if pattern == "allones":
        # every pair masked: no live tile anywhere, all gradients exactly zero
        assert all((x == 0).all() for x in out)


@cuda
@pytest.mark.parametrize("D", [64, 128])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("L", [16384, 12345])
def test_plain_matches_v11(L, D, causal):
    """CQS off: the ALIGNED loop without a run table == v11's CQS-off backward (dk/dv equal, dq in the drift)."""
    ab = _harness()
    q, k, v, do = _inputs(L, D)
    o16, lse = ab.plain_fwd(ab.MODS["v11"], q, k, v, causal)
    a = ab.cqs_bwd(ab.MODS["v12b"], do, q, k, v, o16, lse, None, causal)
    b = ab.cqs_bwd(ab.MODS["v11"], do, q, k, v, o16, lse, None, causal)
    _assert_match(a, b, f"plain L={L} D={D} causal={causal}")


# ------------------------------------------------------------------------------------------------ LEGACY
@cuda
@pytest.mark.parametrize("D", [64, 128])
@pytest.mark.parametrize("causal", [True, False])
def test_mixed_patterns_route_to_legacy(D, causal):
    ab = _harness()
    L = 12000
    unaligned = _tasks(32768, 7, 1, seg_align=0)[2]
    for what, bits in (("random3bit", torch.randint(0, 8, (L,), generator=torch.Generator().manual_seed(1))),
                       ("half", torch.cat([torch.zeros(L // 2 - 37, dtype=torch.int64), torch.ones(L - L // 2 + 37, dtype=torch.int64)])),
                       ("unaligned c=7", unaligned.group_bits)):
        tabs = ab.tables(bits)
        assert ab.census(ab.MODS["v12b"], tabs, causal, D)[0] != 0, f"{what}: census must pick LEGACY"
        out, ref = _compare(ab, tabs, bits.numel(), D, causal)
        _assert_match(out, ref, f"{what} (LEGACY) D={D} causal={causal}")


@cuda
@pytest.mark.parametrize("causal", [True, False])
def test_forced_legacy_matches_v11(causal):
    ab = _harness()
    for N, c, itr, pick in ((32768, 7, 1, 3), (100003, 7, 1, 5)):
        t = _task(N, c, itr, pick)
        out, ref = _compare(ab, ab.task_tables(t), int(t.local_size), 64, causal, flags=ab.FORCE_LEGACY, seed=pick)
        _assert_match(out, ref, f"forced LEGACY path={t.path}")


# ------------------------------------------------------------------------------------------------ deterministic
@cuda
def test_deterministic_reruns_equal():
    """deterministic=True (split dq_accum, one CTA per SM-share walking its columns): reruns torch.equal."""
    ab = _harness()
    t = _task(32768, 7, 1, 3)
    L = int(t.local_size)
    tabs = ab.task_tables(t)
    q, k, v, do = _inputs(L, 64, seed=3)
    o16, lse = ab.cqs_fwd(ab.MODS["v11"], q, k, v, tabs, True)
    runs = [ab.cqs_bwd(ab.MODS["v12b"], do, q, k, v, o16, lse, tabs, True, flags=0, deterministic=True) for _ in range(3)]
    for g in runs[1:]:
        for a, b in zip(runs[0], g):
            assert torch.equal(a, b)
    ref = ab.cqs_bwd(ab.MODS["v11"], do, q, k, v, o16, lse, tabs, True)
    _assert_match(runs[0], ref, "deterministic vs v11 atomics")


# ------------------------------------------------------------------------------------------------ float64
def _ref64(q, k, v, do, bits, causal):
    """Masked attention gradients in float64 (fully masked rows: P = 0)."""
    qd, kd, vd, dod = (x[0].double().transpose(0, 1).requires_grad_(x is not do) for x in (q, k, v, do))   # [H, L, D]
    L, D = qd.shape[1], qd.shape[2]
    s = torch.einsum("hrd,hkd->hrk", qd, kd) * D ** -0.5
    idx = torch.arange(L, device=q.device)
    masked = (bits[:, None] & bits[None, :]) != 0
    if causal:
        masked |= idx[None, :] > idx[:, None]
    s = s.masked_fill(masked[None], float("-inf"))
    dead = masked.all(dim=1)
    p = torch.softmax(s.masked_fill(dead[None, :, None], 0.0), dim=-1).masked_fill(dead[None, :, None], 0.0)
    o = torch.einsum("hrk,hkd->hrd", p, vd)
    dq, dk, dv = torch.autograd.grad(o, (qd, kd, vd), dod)
    return [x.transpose(0, 1)[None] for x in (dq, dk, dv)]


def _rel(a, b):
    return ((a.double() - b).norm() / b.norm()).item()


@cuda
@pytest.mark.parametrize("causal", [True, False])
def test_float64_error_within_fa2(causal):
    """Gradients vs float64 on an aligned real task: v12b's norm-relative error <= 1.2x flash_attn's error on the
    unmasked problem of the same inputs (the zero pattern, which flash_attn computes directly) + 1e-4, and the
    zero pattern itself compared with flash_attn head to head."""
    ab = _harness()
    try:
        from flash_attn import flash_attn_func
    except Exception as e:  # noqa
        pytest.skip(f"flash_attn not importable: {e}")
    t = _task(8192, 7, 1, 0)
    L, D = int(t.local_size), 64
    q, k, v, do = _inputs(L, D, seed=11)
    real = ab.task_tables(t)
    zero = ab.tables(torch.zeros(L, dtype=torch.int64))
    # flash_attn on the unmasked problem
    qf, kf, vf = (x.clone().requires_grad_(True) for x in (q, k, v))
    of = flash_attn_func(qf, kf, vf, causal=causal, softmax_scale=D ** -0.5)
    fa = torch.autograd.grad(of, (qf, kf, vf), do)
    ref0 = _ref64(q, k, v, do, zero[0], causal)
    fa_err = [_rel(a, b) for a, b in zip(fa, ref0)]
    for name, tabs in (("zero", zero), ("real", real)):
        o16, lse = ab.cqs_fwd(ab.MODS["v11"], q, k, v, tabs, causal)
        g = ab.cqs_bwd(ab.MODS["v12b"], do, q, k, v, o16, lse, tabs, causal, flags=0)
        ref = ref0 if name == "zero" else _ref64(q, k, v, do, real[0], causal)
        for grad, a, b, fe in zip(("dq", "dk", "dv"), g, ref, fa_err):
            e = _rel(a, b)
            assert e <= 1.2 * fe + 1e-4, f"{name} {grad}: error vs float64 {e:.3e} > 1.2 x flash_attn's {fe:.3e} + 1e-4"
