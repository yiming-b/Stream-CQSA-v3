"""
The v12f forward (next/kernel/v12f, module cqsa_cuda_next_v12f): census + ALIGNED run-table forward, LEGACY = v11.

Contract: with CQS on, a census kernel classifies every tile the forward visits (live / dead / mixed, from the
64-token summaries) and writes a per-row-block run table of live tiles. When no tile is mixed the ALIGNED
instantiation computes the call and its out (fp32 and fp16) and lse are bit-identical to v11's; otherwise LEGACY
(v11's own CQS forward) does, so every pattern stays exact. Checked here on the shared vis1 GPU at small L:
the census against a numpy reference (routes and runs), bit-identity with v11 for aligned real tasks (c = 7,
31, 73; itr 1-2; ragged L), zero, all-ones and plain calls, causal and non-causal, D = 64 and 128, both
forced row-block orders (the default is descending for causal calls), in-place segmented reads, automatic
LEGACY routing of mixed patterns, and a float64 spot check. F23 regressions (review fixes): sliding-window /
local calls (causal + window, non-causal (w, w)) route to LEGACY and match v11 and a masked float64 reference
(the blocker: they used to reach the non-causal ALIGNED kernel); run-table overflow (> 15 runs per row block)
through the kernels, and exactly 15 runs staying ALIGNED; the finite-V caveat of bit-identity pinned; every
identity check writes into NaN-pre-filled output buffers. Helpers come from benchmarks/ultimate/fwd_ab.py; everything GPU skips when the v11/v12f
extensions are absent (next/kernel/{stable,v12f} on PYTHONPATH). The census-reference test needs no GPU.
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
    import fwd_ab
    if gpu:
        try:
            fwd_ab.load_modules(("v11", "v12f"))
        except Exception as e:  # noqa
            pytest.skip(f"cqsa_cuda_next_v11/v12f not importable: {e}")
    return fwd_ab


def _qkv(L, D, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return [torch.randn(1, L, H, D, device=DEV, dtype=torch.float16, generator=g) for _ in range(3)]


@functools.lru_cache(maxsize=None)
def _tasks(N, c, itr, seg_align=128):
    """The c/itr decomposition of N sorted by (L, path); cached, c=73 builds 73 tasks."""
    ab = _harness(gpu=False)
    return sorted(ab.c_tasks(N, c, itr, seg_align=seg_align), key=lambda t: (int(t.local_size), t.path))


def _task(ab, N, c, itr=1, pick=0):
    tasks = _tasks(N, c, itr)
    return tasks[pick % len(tasks)]


def _compare(ab, tabs, L, D, causal, flags=0, seed=0, window=(-1, -1)):
    """v12f vs v11 on one call; both write into NaN-pre-filled buffers (F23), so a kernel that skips a write
    cannot pass by inheriting the freed buffer of an identical earlier call."""
    q, k, v = _qkv(L, D, seed)
    ref = ab.cqs_fwd(ab.MODS["v11"], q, k, v, tabs, causal, window=window, fill=float("nan"))
    out = ab.cqs_fwd(ab.MODS["v12f"], q, k, v, tabs, causal, flags=flags, window=window, fill=float("nan"))
    return out, ref


def _assert_equal(out, ref, what):
    for name, a, b in zip(("out32", "lse", "out16"), out, ref):
        assert torch.equal(a, b), f"{what}: {name} differs from v11 (max |d| {(a.float() - b.float()).abs().max().item():.3e})"


# ------------------------------------------------------------------------------------------------ census
def test_census_reference_runs():
    ab = _harness(gpu=False)
    # 128-token blocks: owner (0), non-owner A (bit 1), owner, non-owner A again, ragged non-owner B (bit 2)
    bits = np.array([0] * 128 + [1] * 256 + [0] * 128 + [1] * 128 + [2] * 70, dtype=np.int64)
    route, rows = ab.census_ref(bits, 128, 128, causal=True)
    assert route == 0
    assert rows[0] == [[0, 0]]                       # owner diagonal
    assert rows[1] == [[0, 0]] and rows[2] == [[0, 0]]   # non-owner A: its own blocks (and the diagonal) are dead
    assert rows[3] == [[3, 0]]                       # owner: everything to its left
    assert rows[4] == [[3, 3], [0, 0]]              # A again: skips blocks 1, 2 and its own diagonal
    assert rows[5] == [[4, 0]]                       # B (ragged): only its own diagonal is dead
    route_nc, rows_nc = ab.census_ref(bits, 128, 128, causal=False)
    assert route_nc == 0 and rows_nc[1] == [[5, 5], [3, 3], [0, 0]]
    # an unaligned boundary makes a tile mixed; more runs than slots overflow
    route, _ = ab.census_ref(np.array([0] * 100 + [1] * 156, dtype=np.int64), 128, 128, causal=True)
    assert route & 1
    comb = np.repeat(np.arange(40) % 2, 128).astype(np.int64)   # alternate owner / non-owner blocks: 20 runs
    route, rows = ab.census_ref(comb, 128, 128, causal=False)
    assert route & 2 and len(rows[1]) == ab.RUN_SLOTS - 1


def _census_cases(ab):
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
    for what, bits, aligned in _census_cases(ab):
        route, runs, tile = ab.census(ab.MODS["v12f"], ab.tables(bits), causal, D)
        assert tile == ab.ALIGNED_TILE[D]
        ref_route, ref_rows = ab.census_ref(bits.numpy(), *tile, causal=causal)
        assert route == ref_route, f"{what}: route {route} vs reference {ref_route}"
        assert [ab.runs_prefix(r) for r in runs] == ref_rows, f"{what}: run table differs from the reference"
        if aligned:
            assert route == 0, f"{what}: a 128-aligned pattern must route to ALIGNED"
        elif causal and D == 64:
            assert route & 1, f"{what}: a mixed pattern must route to LEGACY"


# ------------------------------------------------------------------------------------------------ ALIGNED == v11
ALIGNED_CASES = [  # (N, c, itr, pick): real 128-aligned tasks; N=100003 gives ragged L
    (32768, 7, 1, 0), (32768, 7, 1, 3), (32768, 7, 1, 6), (131072, 7, 2, 0), (131072, 7, 2, 25),
    (65536, 31, 1, 0), (65536, 31, 1, 17), (131072, 73, 1, 0), (131072, 73, 1, 40), (100003, 7, 1, 0),
    (100003, 7, 1, 5),
]


@cuda
@pytest.mark.parametrize("D", [64, 128])
@pytest.mark.parametrize("causal", [True, False])
def test_aligned_real_tasks_bit_identical(D, causal):
    ab = _harness()
    for N, c, itr, pick in ALIGNED_CASES:
        t = _task(ab, N, c, itr, pick)
        L = int(t.local_size)
        tabs = ab.task_tables(t)
        assert ab.census(ab.MODS["v12f"], tabs, causal, D)[0] == 0, f"c={c} itr={itr} L={L}: census must pick ALIGNED"
        out, ref = _compare(ab, tabs, L, D, causal, seed=pick)
        _assert_equal(out, ref, f"c={c} itr={itr} N={N} path={t.path} L={L} (L%128={L % 128})")


@cuda
@pytest.mark.parametrize("D", [64, 128])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("pattern", ["zero", "allones"])
@pytest.mark.parametrize("L", [16384, 12345])
def test_aligned_synthetic_bit_identical(pattern, L, D, causal):
    ab = _harness()
    bits = torch.zeros(L, dtype=torch.int64) if pattern == "zero" else torch.full((L,), -1, dtype=torch.int64)
    tabs = ab.tables(bits)
    assert ab.census(ab.MODS["v12f"], tabs, causal, D)[0] == 0
    out, ref = _compare(ab, tabs, L, D, causal)
    _assert_equal(out, ref, f"{pattern} L={L}")
    if pattern == "allones":
        # every pair masked: no live tile anywhere, v11's fully-masked-row output
        assert (out[0] == 0).all() and torch.isinf(out[1]).all() and (out[1] > 0).all()


@cuda
@pytest.mark.parametrize("D", [64, 128])
@pytest.mark.parametrize("causal", [True, False])
def test_plain_matches_v11(D, causal):
    ab = _harness()
    q, k, v = _qkv(12345, D)
    a = ab.plain_fwd(ab.MODS["v12f"], q, k, v, causal)
    b = ab.plain_fwd(ab.MODS["v11"], q, k, v, causal)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


@cuda
@pytest.mark.parametrize("order", ["REVERSE_M", "ASCENDING_M"])
@pytest.mark.parametrize("causal", [True, False])
def test_row_order_bit_identical(causal, order):
    """Both forced row-block orders (the default is descending for causal, ascending for non-causal)."""
    ab = _harness()
    for N, c, itr, pick in ((32768, 7, 1, 3), (100003, 7, 1, 5), (131072, 7, 2, 25)):
        t = _task(ab, N, c, itr, pick)
        for D in (64, 128):
            out, ref = _compare(ab, ab.task_tables(t), int(t.local_size), D, causal, flags=getattr(ab, order))
            _assert_equal(out, ref, f"{order} rows, D={D} c={c} itr={itr} path={t.path}")


@cuda
@pytest.mark.parametrize("causal", [True, False])
def test_inplace_segmented_reads(causal):
    """cqs_block_base / cqs_seg_align: ALIGNED reads the original Q/K/V in place; equal to v11 in place and to
    its own gathered call."""
    ab = _harness()
    N = 100003
    for pick in (0, 5):
        t = _task(ab, N, 7, 1, pick)
        L = int(t.local_size)
        g = torch.Generator(device="cuda").manual_seed(pick)
        qt, kt, vt = (torch.randn(1, N, H, 64, device=DEV, dtype=torch.float16, generator=g) for _ in range(3))
        views = [x.as_strided((1, L, H, 64), x.stride()) for x in (qt, kt, vt)]
        idx = t.token_ids.to(DEV)
        packed = [x.index_select(1, idx).contiguous() for x in (qt, kt, vt)]
        base = ab.to_dev(t.extra["seg_base"])
        tabs = ab.task_tables(t)
        ip = ab.cqs_fwd(ab.MODS["v12f"], *views, tabs, causal, flags=0, block_base=base)
        ip11 = ab.cqs_fwd(ab.MODS["v11"], *views, tabs, causal, block_base=base)
        gathered = ab.cqs_fwd(ab.MODS["v12f"], *packed, tabs, causal, flags=0)
        _assert_equal(ip, ip11, f"in place, path={t.path}")
        _assert_equal(ip, gathered, f"in place vs gathered, path={t.path}")


# ------------------------------------------------------------------------------------------------ LEGACY
@cuda
@pytest.mark.parametrize("causal", [True, False])
def test_mixed_patterns_route_to_legacy(causal):
    ab = _harness()
    L = 12000
    unaligned = _tasks(32768, 7, 1, seg_align=0)[2]
    for what, bits in (("random3bit", torch.randint(0, 8, (L,), generator=torch.Generator().manual_seed(1))),
                       ("half", torch.cat([torch.zeros(L // 2 - 37, dtype=torch.int64), torch.ones(L - L // 2 + 37, dtype=torch.int64)])),
                       ("unaligned c=7", unaligned.group_bits)):
        tabs = ab.tables(bits)
        assert ab.census(ab.MODS["v12f"], tabs, causal, 64)[0] != 0, f"{what}: census must pick LEGACY"
        out, ref = _compare(ab, tabs, bits.numel(), 64, causal)
        _assert_equal(out, ref, f"{what} (LEGACY)")


@cuda
def test_forced_legacy_matches_v11():
    ab = _harness()
    t = _task(ab, 32768, 7, 1, 3)
    out, ref = _compare(ab, ab.task_tables(t), int(t.local_size), 64, True, flags=ab.FORCE_LEGACY)
    _assert_equal(out, ref, "forced LEGACY")


# ------------------------------------------------------------------------------------------------ float64
@cuda
def test_float64_spot_check():
    """Sampled rows of an aligned real task against a float64 reference: fully masked rows give out 0 and
    lse +inf, the others agree to fp16-input accuracy (norm-relative)."""
    ab = _harness()
    t = _task(ab, 32768, 7, 1, 0)
    L, D = int(t.local_size), 64
    bits = t.group_bits.to(DEV)
    q, k, v = _qkv(L, D, seed=7)
    out, lse, _ = ab.cqs_fwd(ab.MODS["v12f"], q, k, v, ab.task_tables(t), True, flags=0)
    rows = torch.unique(torch.cat([torch.arange(0, L, 97), torch.tensor([0, 127, 128, L - 1])])).to(DEV)
    qd, kd, vd = (x[0].double() for x in (q, k, v))                                  # [L, H, D]
    s = torch.einsum("rhd,khd->hrk", qd[rows], kd) * D ** -0.5                        # [H, R, L]
    cols = torch.arange(L, device=DEV)
    masked = (cols[None, :] > rows[:, None]) | ((bits[rows][:, None] & bits[None, :]) != 0)
    s = s.masked_fill(masked[None], float("-inf"))
    dead = masked.all(dim=1)
    p = torch.softmax(s[:, ~dead], dim=-1)
    ref = torch.einsum("hrk,khd->rhd", p, vd)                                         # [R_live, H, D]
    got = out[0, rows[~dead]].double()
    rel = ((got - ref).norm() / ref.norm()).item()
    assert rel < 2e-3, f"norm-relative error vs float64 {rel:.2e}"
    if dead.any():
        assert (out[0, rows[dead]] == 0).all()
        assert torch.isinf(lse[0, :, rows[dead]]).all()


# ------------------------------------------------------------------------------------------------ F23 regressions
def _ref64_rows(q, k, v, bits, causal, window, rows):
    """float64 masked attention for the sampled rows: group bits, causal, sliding window (left, right; -1 = none,
    causal forces right = 0, as mha_fwd_impl does). Returns (out [R, H, D], dead [R])."""
    L, D = q.shape[1], q.shape[-1]
    left, right = window
    if causal:
        right = 0
    qd, kd, vd = (x[0].double() for x in (q, k, v))
    s = torch.einsum("rhd,khd->hrk", qd[rows], kd) * D ** -0.5
    cols = torch.arange(L, device=DEV)
    masked = (bits[rows][:, None] & bits[None, :]) != 0
    if right >= 0:
        masked |= cols[None, :] > rows[:, None] + right
    if left >= 0:
        masked |= cols[None, :] < rows[:, None] - left
    s = s.masked_fill(masked[None], float("-inf"))
    dead = masked.all(dim=1)
    p = torch.softmax(s.masked_fill(dead[None, :, None], 0.0), dim=-1).masked_fill(dead[None, :, None], 0.0)
    return torch.einsum("hrk,khd->rhd", p, vd), dead


@cuda
@pytest.mark.parametrize("D", [64, 128])
@pytest.mark.parametrize("causal,window", [(True, (256, 0)), (True, (1000, -1)), (False, (256, 128)),
                                           (False, (300, -1)), (False, (-1, 200))])
def test_window_calls_route_to_legacy(causal, window, D):
    """F23 blocker: a windowed (local) call must never reach ALIGNED. Causal + window has params.is_causal ==
    false after set_params_fprop; the old gate tested the is_causal ARGUMENT and ran it on the non-causal
    ALIGNED kernel (no window, no causal mask: max error 3.6 on vis1). v12f == v11 bit for bit (LEGACY is
    v11's code), == forced LEGACY, and matches a masked float64 reference on sampled rows."""
    ab = _harness()
    t = _task(ab, 32768, 7, 1, 3)
    L = int(t.local_size)
    for what, tabs in (("zero", ab.tables(torch.zeros(L, dtype=torch.int64))), ("real", ab.task_tables(t))):
        out, ref = _compare(ab, tabs, L, D, causal, window=window, seed=5)
        _assert_equal(out, ref, f"{what} window={window} causal={causal} D={D}")
        q, k, v = _qkv(L, D, 5)
        forced = ab.cqs_fwd(ab.MODS["v12f"], q, k, v, tabs, causal, flags=ab.FORCE_LEGACY, window=window,
                            fill=float("nan"))
        _assert_equal(out, forced, f"{what} window={window}: default vs forced LEGACY")
        rows = torch.unique(torch.cat([torch.arange(0, L, 331), torch.tensor([0, 255, 256, 1000, L - 1])])).to(DEV)
        ref64, dead = _ref64_rows(q, k, v, tabs[0], causal, window, rows)
        got = out[0][0, rows].double()
        live = ~dead
        rel = ((got[live] - ref64[live]).norm() / ref64[live].norm()).item()
        assert rel < 2e-3, f"{what} window={window} causal={causal} D={D}: norm-relative error vs float64 {rel:.2e}"


@cuda
@pytest.mark.parametrize("D", [64, 128])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("n_groups,overflow", [(30, False), (32, True), (40, True), (41, True)])
def test_run_table_overflow_through_kernels(n_groups, overflow, causal, D):
    """Alternating owner / non-owner 128-token groups: a non-owner row block has n_groups/2 isolated live runs.
    15 runs (n_groups = 30) fit the 16-slot table (route 0, ALIGNED); 16+ set the overflow bit (route 2, no
    mixed tile) and the call must run on LEGACY, end to end, equal to v11. n_groups = 41 ends on an owner group
    with a ragged 57-token tail (Is_even_MN = false)."""
    ab = _harness()
    bits = torch.repeat_interleave(torch.arange(n_groups) % 2, 128).to(torch.int64)
    if n_groups == 41:
        bits = bits[: 40 * 128 + 57]
    L = bits.numel()
    tabs = ab.tables(bits)
    route, runs, tile = ab.census(ab.MODS["v12f"], tabs, causal, D)
    ref_route, _ = ab.census_ref(bits.numpy(), *tile, causal=causal)
    assert route == ref_route
    assert not (route & 1), "a 128-aligned comb has no mixed tile"
    assert bool(route & 2) == overflow, f"n_groups={n_groups}: route {route}"   # causal: the last row blocks
    out, ref = _compare(ab, tabs, L, D, causal, seed=n_groups)
    _assert_equal(out, ref, f"comb n_groups={n_groups} route={route} causal={causal} D={D}")


@cuda
def test_nonfinite_v_in_dead_tile_caveat():
    """Pins the documented caveat of bit-identity (F23): it holds for FINITE K/V. v11 runs the fully masked
    diagonal tile of a non-owner row block as P = 0 then P @ V, so an inf in a masked key's V row turns v11's
    rows NaN; ALIGNED never loads that tile, so v12f's rows stay finite and match the masked reference. Rows
    128-383 (group 1) never attend keys 128-383; rows 384-511 (owner) attend key 200."""
    ab = _harness()
    L, D = 512, 64
    bits = torch.tensor([0] * 128 + [1] * 256 + [0] * 128, dtype=torch.int64)
    tabs = ab.tables(bits)
    assert ab.census(ab.MODS["v12f"], tabs, True, D)[0] == 0
    q, k, v = _qkv(L, D, seed=2)
    v[0, 200, 0, 3] = float("inf")
    out = ab.cqs_fwd(ab.MODS["v12f"], q, k, v, tabs, True, flags=0, fill=float("nan"))
    ref = ab.cqs_fwd(ab.MODS["v11"], q, k, v, tabs, True, fill=float("nan"))
    o12, o11 = out[0][0], ref[0][0]
    assert torch.isfinite(o12[:384]).all(), "v12f: rows that never attend key 200 must stay finite"
    fin = torch.isfinite(o11) & torch.isfinite(o12)
    assert torch.equal(o12[fin], o11[fin]), "finite elements must agree with v11"
    rows = torch.arange(0, 384, device=DEV)
    ref64, dead = _ref64_rows(q, k, v, tabs[0], True, (-1, -1), rows)
    rel = ((o12[rows][~dead].double() - ref64[~dead]).norm() / ref64[~dead].norm()).item()
    assert rel < 2e-3, f"v12f vs masked float64 reference on rows 0-383: {rel:.2e}"
