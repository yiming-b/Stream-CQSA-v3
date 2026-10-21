"""
The v13 classic CUDA module (kernels/v13, module cqsa_cuda_v13; ultimate program package U1): v12f's forward
(census -> ALIGNED run-table kernel -> LEGACY) and v12b's backward (census per column block -> ALIGNED ->
LEGACY) in ONE extension, with v11's full per-element mask path kept in both LEGACY directions, so every
pattern is exact in both directions and one module serves causal and non-causal calls, fp16 and bf16.

Contract checked here (shared vis1 GPU, small L, H=4): the two censuses against the numpy references;
ALIGNED real tasks (c in {7, 31, 73}, itr 1-2, ragged N=100003), zero and all-masked patterns, random / half /
unaligned (LEGACY) patterns, sliding windows (LEGACY), causal and non-causal, D 64 and 128, fp16 AND bf16.
fp16 identity is stated against v11 (the production module: out32 / lse / out16 torch.equal, dk / dv torch.equal,
dq within the fp32 atomics' drift); bf16 has no v11 build, so its ALIGNED results are stated against the module's
own forced-LEGACY call (= v11's code) and against float64. Every identity copy writes into NaN-pre-filled buffers
(F23). The engine-level tests run a subprocess without CQSA_CUDA_MODULE in its environment and check that
interface.py picks cqsa_cuda_v13, serves non-causal calls from it and defaults to its CUDA backward, then run
stream_cqsa.attention forward + backward on it against float64.
Helpers come from benchmarks/ultimate/{fwd_ab,bwd_ab}.py (module key "v13"); the GPU tests skip when the v11 /
v13 modules are absent (next/kernel/stable and kernels/v13 -- or a frozen copy -- on PYTHONPATH, variant dirs LAST).
bf16 tests skip on a fp16-only build (kernel set a100_fp16_hdim64_128); the `common` build runs them.
"""
import functools, importlib.util, os, subprocess, sys

import numpy as np
import pytest
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
DEV = torch.device("cuda")
H = 4
NAN = float("nan")
DTYPES = [torch.float16, torch.bfloat16]
# norm-relative error vs float64 on the FULL output / gradients: measured fp16 fwd 1.9e-4 / bwd 3.1e-4,
# bf16 1.5e-3 / 2.5e-3 at N=4096 (tests/test_bf16.py); ~5x headroom, loose enough for the masked patterns
TOL_FWD = {torch.float16: 1e-3, torch.bfloat16: 8e-3}
TOL_BWD = {torch.float16: 1.5e-3, torch.bfloat16: 1.2e-2}


def _harness():
    path = os.path.join(REPO, "benchmarks", "ultimate")
    if path not in sys.path:
        sys.path.insert(0, path)
    import fwd_ab, bwd_ab
    bwd_ab.SENTINEL_FILL[0] = NAN       # every cqs_bwd in these tests writes into NaN-filled buffers
    return fwd_ab, bwd_ab


def _mods():
    F, B = _harness()
    try:
        F.load_modules(("v11", "v13"))
        B.load_modules(("v11", "v13"))
    except Exception as e:  # noqa
        pytest.skip(f"cqsa_cuda_next_v11 / cqsa_cuda_v13 not importable: {e}")
    return F, B


@functools.lru_cache(maxsize=None)
def _bf16_built():
    """Whether the loaded v13 build carries the bf16 kernels (CQSA_KERNEL_SET=common)."""
    F, _ = _mods()
    q = torch.randn(1, 256, H, 64, device=DEV, dtype=torch.bfloat16)
    try:
        F.plain_fwd(F.MODS["v13"], q, q, q, True)
    except RuntimeError as e:
        if "bf16" in str(e).lower() or "bfloat16" in str(e).lower():
            return False
        raise
    return True


def _need(dtype):
    F, B = _mods()
    if dtype == torch.bfloat16 and not _bf16_built():
        pytest.skip("v13 build without bf16 kernels (CQSA_KERNEL_SET=common has them)")
    return F, B


def _rand(L, D, dtype, n, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return [torch.randn(1, L, H, D, device=DEV, dtype=dtype, generator=g) for _ in range(n)]


@functools.lru_cache(maxsize=None)
def _tasks(N, c, itr, seg_align=128):
    """The c/itr decomposition of N sorted by (L, path); cached (c=73 builds 73 tasks)."""
    F, _ = _harness()
    return sorted(F.c_tasks(N, c, itr, seg_align=seg_align), key=lambda t: (int(t.local_size), t.path))


def _task(N, c, itr=1, pick=0):
    tasks = _tasks(N, c, itr)
    return tasks[pick % len(tasks)]


ALIGNED_CASES = [  # (N, c, itr, pick): real 128-aligned tasks, L up to ~28K; N=100003 gives ragged L
    (32768, 7, 1, 0), (32768, 7, 1, 3), (65536, 7, 1, 6), (131072, 7, 2, 0), (131072, 7, 2, 25),
    (65536, 31, 1, 0), (65536, 31, 1, 17), (131072, 73, 1, 0), (131072, 73, 1, 40), (100003, 7, 1, 0),
    (100003, 7, 1, 5),
]


def _mixed_cases(L=12000):
    unaligned = _tasks(32768, 7, 1, seg_align=0)[2]
    return (("random3bit", torch.randint(0, 8, (L,), generator=torch.Generator().manual_seed(1))),
            ("half", torch.cat([torch.zeros(L // 2 - 37, dtype=torch.int64), torch.ones(L - L // 2 + 37, dtype=torch.int64)])),
            ("unaligned c=7", unaligned.group_bits))


def _rel(a, b):
    return ((a.double() - b.double()).norm() / b.double().norm()).item()


# ================================================================================================ forward
def _fwd_ref64_rows(q, k, v, bits, causal, rows, window=(-1, -1)):
    """float64 masked attention for the sampled rows (group bits, causal, optional window as the extension takes
    it; causal forces right = 0). Returns (out [R, H, D], dead [R]): fully masked rows have out 0 / lse +inf."""
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


def _sample_rows(L):
    return torch.unique(torch.cat([torch.arange(0, L, 97), torch.tensor([0, 127, 128, L - 1])])).to(DEV)


def _assert_fwd_ref64(out, tabs, q, k, v, causal, dtype, what, window=(-1, -1)):
    L = q.shape[1]
    rows = _sample_rows(L)
    ref, dead = _fwd_ref64_rows(q, k, v, tabs[0], causal, rows, window)
    out32, lse = out[0], out[1]
    live = rows[~dead]
    if live.numel():
        rel = _rel(out32[0, live], ref[~dead])
        assert rel <= TOL_FWD[dtype], f"{what}: norm-relative error vs float64 {rel:.2e} > {TOL_FWD[dtype]}"
    if dead.any():
        assert (out32[0, rows[dead]] == 0).all() and torch.isinf(lse[0, :, rows[dead]]).all(), f"{what}: dead rows"
    assert torch.isfinite(out32).all() and torch.isfinite(out[2]).all(), f"{what}: non-finite output"


def _fwd_pair(F, tabs, L, D, causal, dtype, seed=0, window=(-1, -1), flags=0):
    """v13 (default route) and its identity reference: v11 for fp16, v13 forced LEGACY for bf16. Both write into
    NaN-pre-filled buffers. Returns (out, ref, (q, k, v)) with out = (out32, lse, out16)."""
    q, k, v = _rand(L, D, dtype, 3, seed)
    out = F.cqs_fwd(F.MODS["v13"], q, k, v, tabs, causal, flags=flags, window=window, fill=NAN)
    if dtype == torch.float16:
        ref = F.cqs_fwd(F.MODS["v11"], q, k, v, tabs, causal, window=window, fill=NAN)
    else:
        ref = F.cqs_fwd(F.MODS["v13"], q, k, v, tabs, causal, flags=F.FORCE_LEGACY, window=window, fill=NAN)
    return out, ref, (q, k, v)


def _assert_fwd_equal(out, ref, what):
    for name, a, b in zip(("out32", "lse", "out16"), out, ref):
        assert torch.equal(a, b), f"{what}: {name} differs (max |d| {(a.float() - b.float()).abs().max().item():.3e})"


def test_fwd_census_reference_runs():
    F, _ = _harness()
    bits = np.array([0] * 128 + [1] * 256 + [0] * 128 + [1] * 128 + [2] * 70, dtype=np.int64)
    route, rows = F.census_ref(bits, 128, 128, causal=True)
    assert route == 0 and rows[3] == [[3, 0]] and rows[4] == [[3, 3], [0, 0]] and rows[5] == [[4, 0]]


@cuda
@pytest.mark.parametrize("D", [64, 128])
@pytest.mark.parametrize("causal", [True, False])
def test_fwd_census_matches_reference(D, causal):
    F, _ = _mods()
    cases = [(f"real c={c} itr={itr} N={N} path={t.path}", t.group_bits, True)
             for N, c, itr in ((32768, 7, 1), (131072, 7, 2), (65536, 31, 1), (131072, 73, 1), (100003, 7, 1))
             for t in _tasks(N, c, itr)[:3]]
    L = 12000
    cases += [("zero", torch.zeros(L, dtype=torch.int64), True), ("allones", torch.full((L,), -1, dtype=torch.int64), True)]
    cases += [(what, bits, False) for what, bits in _mixed_cases(L)]
    for what, bits, aligned in cases:
        route, runs, tile = F.census(F.MODS["v13"], F.tables(bits), causal, D)
        assert tile == F.ALIGNED_TILE[D]
        ref_route, ref_rows = F.census_ref(bits.numpy(), *tile, causal=causal)
        assert route == ref_route, f"{what}: route {route} vs reference {ref_route}"
        assert [F.runs_prefix(r) for r in runs] == ref_rows, f"{what}: run table differs from the reference"
        assert (route == 0) == aligned, f"{what}: route {route}"


@cuda
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("D", [64, 128])
@pytest.mark.parametrize("causal", [True, False])
def test_fwd_aligned_real_tasks(D, causal, dtype):
    """ALIGNED on every 128-aligned real task: route 0, bit-identical to v11 (fp16) / to the forced LEGACY call
    (bf16 = v11's code), and the ragged tasks (N=100003) go through the Is_even_MN=false instantiation."""
    F, _ = _need(dtype)
    for N, c, itr, pick in ALIGNED_CASES:
        t = _task(N, c, itr, pick)
        L, tabs = int(t.local_size), F.task_tables(t)
        assert F.census(F.MODS["v13"], tabs, causal, D)[0] == 0, f"c={c} itr={itr} L={L}: census must pick ALIGNED"
        out, ref, _ = _fwd_pair(F, tabs, L, D, causal, dtype, seed=pick)
        _assert_fwd_equal(out, ref, f"c={c} itr={itr} N={N} path={t.path} L={L} (L%128={L % 128}) D={D} causal={causal} {dtype}")


@cuda
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("causal", [True, False])
def test_fwd_aligned_vs_float64(causal, dtype):
    """One aligned real task and the zero pattern per dtype against float64 on sampled rows (both D)."""
    F, _ = _need(dtype)
    t = _task(32768, 7, 1, 3)
    L = int(t.local_size)
    for D in (64, 128):
        for what, tabs in (("real", F.task_tables(t)), ("zero", F.tables(torch.zeros(L, dtype=torch.int64)))):
            out, _, (q, k, v) = _fwd_pair(F, tabs, L, D, causal, dtype, seed=7)
            _assert_fwd_ref64(out, tabs, q, k, v, causal, dtype, f"{what} D={D} causal={causal} {dtype}")


@cuda
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("D", [64, 128])
@pytest.mark.parametrize("causal", [True, False])
def test_fwd_zero_and_all_masked(D, causal, dtype):
    for L in (16384, 12345):
        for pattern in ("zero", "allones"):
            bits = torch.zeros(L, dtype=torch.int64) if pattern == "zero" else torch.full((L,), -1, dtype=torch.int64)
            F, _ = _need(dtype)
            tabs = F.tables(bits)
            assert F.census(F.MODS["v13"], tabs, causal, D)[0] == 0
            out, ref, _ = _fwd_pair(F, tabs, L, D, causal, dtype)
            _assert_fwd_equal(out, ref, f"{pattern} L={L} D={D} causal={causal} {dtype}")
            if pattern == "allones":      # every pair masked: no live tile anywhere, v11's fully-masked-row output
                assert (out[0] == 0).all() and torch.isinf(out[1]).all() and (out[1] > 0).all()


@cuda
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("D", [64, 128])
@pytest.mark.parametrize("causal", [True, False])
def test_fwd_mixed_patterns_through_legacy(D, causal, dtype):
    """random / half / unaligned patterns: the census finds a mixed tile, LEGACY (v11's per-element path, kept in
    this build) computes the call: bit-identical to v11 (fp16), float64 on sampled rows (both dtypes). The
    non-causal D=128 rows are the mixed non-causal case the U1 plan verifies before retiring csrc_nc (v9)."""
    F, _ = _need(dtype)
    for what, bits in _mixed_cases():
        tabs = F.tables(bits)
        assert F.census(F.MODS["v13"], tabs, causal, D)[0] & 1, f"{what}: census must pick LEGACY"
        out, ref, (q, k, v) = _fwd_pair(F, tabs, bits.numel(), D, causal, dtype)
        tag = f"{what} (LEGACY) D={D} causal={causal} {dtype}"
        _assert_fwd_equal(out, ref, tag)
        _assert_fwd_ref64(out, tabs, q, k, v, causal, dtype, tag)


@cuda
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("D", [64, 128])
@pytest.mark.parametrize("causal", [True, False])
def test_fwd_plain(D, causal, dtype):
    """CQS off: upstream's loop; fp16 torch.equal to v11, both dtypes within the dtype's error of flash_attn."""
    F, _ = _need(dtype)
    q, k, v = _rand(12345, D, dtype, 3)
    a = F.plain_fwd(F.MODS["v13"], q, k, v, causal)
    if dtype == torch.float16:
        b = F.plain_fwd(F.MODS["v11"], q, k, v, causal)
        assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
    flash_attn = pytest.importorskip("flash_attn")
    fa = flash_attn.flash_attn_func(q, k, v, causal=causal, softmax_scale=D ** -0.5)
    assert _rel(a[0], fa) <= TOL_FWD[dtype], f"plain D={D} causal={causal} {dtype}: rel vs flash_attn {_rel(a[0], fa):.2e}"


@cuda
@pytest.mark.parametrize("dtype", DTYPES)
def test_fwd_row_orders_and_forced_legacy(dtype):
    """Both forced row-block orders of ALIGNED (default: descending for causal, ascending otherwise) and the forced
    LEGACY route give the same bits as the default call."""
    F, _ = _need(dtype)
    t = _task(100003, 7, 1, 5)
    L, tabs = int(t.local_size), F.task_tables(t)
    for causal in (True, False):
        for D in (64, 128):
            out, ref, _ = _fwd_pair(F, tabs, L, D, causal, dtype, seed=5)
            q, k, v = _rand(L, D, dtype, 3, 5)
            for name, flags in (("REVERSE_M", F.REVERSE_M), ("ASCENDING_M", F.ASCENDING_M), ("FORCE_LEGACY", F.FORCE_LEGACY)):
                o = F.cqs_fwd(F.MODS["v13"], q, k, v, tabs, causal, flags=flags, fill=NAN)
                _assert_fwd_equal(o, out, f"{name} vs default, D={D} causal={causal} {dtype}")
            _assert_fwd_equal(out, ref, f"default vs reference D={D} causal={causal} {dtype}")


@cuda
@pytest.mark.parametrize("causal", [True, False])
def test_fwd_inplace_segmented_reads(causal):
    """cqs_block_base / cqs_seg_align: ALIGNED reads the original Q/K/V in place (the engine's in-place mode);
    equal to v11 in place and to the gathered call."""
    F, _ = _mods()
    N = 100003
    t = _task(N, 7, 1, 5)
    L = int(t.local_size)
    tabs = F.task_tables(t)
    base = F.to_dev(t.extra["seg_base"])
    g = torch.Generator(device="cuda").manual_seed(9)
    full = [torch.randn(1, N, H, 64, device=DEV, dtype=torch.float16, generator=g) for _ in range(3)]
    views = [x.as_strided((1, L, H, 64), x.stride()) for x in full]
    idx = t.token_ids.to(DEV)
    gathered = [x.index_select(1, idx).contiguous() for x in full]
    out_g = F.cqs_fwd(F.MODS["v13"], *gathered, tabs, causal, fill=NAN)
    out_ip = F.cqs_fwd(F.MODS["v13"], *views, tabs, causal, block_base=base, fill=NAN)
    ref_ip = F.cqs_fwd(F.MODS["v11"], *views, tabs, causal, block_base=base, fill=NAN)
    _assert_fwd_equal(out_ip, ref_ip, f"in place vs v11 in place, causal={causal}")
    _assert_fwd_equal(out_ip, out_g, f"in place vs gathered, causal={causal}")


@cuda
@pytest.mark.parametrize("D", [64, 128])
@pytest.mark.parametrize("causal,window", [(True, (256, 0)), (False, (256, 128)), (False, (-1, 200))])
def test_fwd_window_calls_route_to_legacy(causal, window, D):
    """F23 blocker regression: a windowed (local) call never reaches ALIGNED (a causal call with a window has
    params.is_causal == false); equal to v11 and to the masked float64 reference."""
    F, _ = _mods()
    t = _task(32768, 7, 1, 3)
    L, tabs = int(t.local_size), F.task_tables(t)
    out, ref, (q, k, v) = _fwd_pair(F, tabs, L, D, causal, torch.float16, seed=3, window=window)
    _assert_fwd_equal(out, ref, f"window {window} causal={causal} D={D}")
    _assert_fwd_ref64(out, tabs, q, k, v, causal, torch.float16, f"window {window} causal={causal} D={D}", window=window)


# ================================================================================================ backward
def _dq_tol(ref):
    """The fp32 atomics' drift band on the output dtype's dq: 2 ulps at the largest |dq| (one rounding flip)."""
    ulp = 2.0 ** -9 if ref.dtype == torch.float16 else 2.0 ** -6
    return max(ulp * ref.float().abs().max().item(), 1e-6)


def _bwd_pair(B, tabs, L, D, causal, dtype, seed=0, dsum=False, deterministic=False, window=(-1, -1)):
    """v13 backward (default route) and its identity reference (v11 for fp16, v13 forced LEGACY for bf16) on the
    same o / lse (v11's forward for fp16, v13's for bf16). Returns (out, ref, inputs) with out = (dq, dk, dv)."""
    q, k, v, do = _rand(L, D, dtype, 4, seed)
    fwd_mod = B.MODS["v11"] if dtype == torch.float16 else B.MODS["v13"]
    o16, lse = B.cqs_fwd(fwd_mod, q, k, v, tabs, causal, window=window)
    ds = (do.float() * o16.float()).sum(-1).transpose(1, 2).contiguous() if dsum else None      # [1, H, L]
    out = B.cqs_bwd(B.MODS["v13"], do, q, k, v, o16, lse, tabs, causal, flags=0, dsum=ds, deterministic=deterministic,
                    window=window)
    if dtype == torch.float16:
        ref = B.cqs_bwd(B.MODS["v11"], do, q, k, v, o16, lse, tabs, causal, dsum=ds, deterministic=deterministic, window=window)
    else:
        ref = B.cqs_bwd(B.MODS["v13"], do, q, k, v, o16, lse, tabs, causal, flags=B.FORCE_LEGACY, dsum=ds,
                        deterministic=deterministic, window=window)
    return out, ref, (q, k, v, do)


def _assert_bwd_match(out, ref, what, dq_equal=False):
    for name, a, b in zip(("dk", "dv"), out[1:], ref[1:]):
        assert torch.equal(a, b), f"{what}: {name} differs (max |d| {(a.float() - b.float()).abs().max().item():.3e})"
    d = (out[0].float() - ref[0].float()).abs().max().item()
    if dq_equal:
        assert d == 0, f"{what}: dq differs (max |d| {d:.3e})"
    else:
        assert d <= _dq_tol(ref[0]), f"{what}: dq max |d| {d:.3e} beyond the atomics' drift ({_dq_tol(ref[0]):.3e})"
    for name, x in zip(("dq", "dk", "dv"), out):
        assert torch.isfinite(x).all(), f"{what}: {name} not finite"


def _bwd_ref64(q, k, v, do, bits, causal):
    """Masked attention gradients in float64 (fully masked rows: P = 0); [1, L, H, D] each."""
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


def test_bwd_census_reference_runs():
    _, B = _harness()
    bits = np.array([0] * 128 + [1] * 256 + [0] * 128 + [1] * 128 + [2] * 70, dtype=np.int64)
    route, cols = B.census_ref(bits, 128, 128, causal=True)
    assert route == 0 and cols[0] == [[5, 0]] and cols[1] == [[5, 5], [3, 3]] and cols[5] == []


@cuda
@pytest.mark.parametrize("D", [64, 128])
@pytest.mark.parametrize("causal", [True, False])
def test_bwd_census_matches_reference(D, causal):
    _, B = _mods()
    cases = [(f"real c={c} itr={itr} N={N} path={t.path}", t.group_bits, True)
             for N, c, itr in ((32768, 7, 1), (131072, 7, 2), (65536, 31, 1), (131072, 73, 1), (100003, 7, 1))
             for t in _tasks(N, c, itr)[:3]]
    L = 12000
    cases += [("zero", torch.zeros(L, dtype=torch.int64), True), ("allones", torch.full((L,), -1, dtype=torch.int64), True)]
    cases += [(what, bits, False) for what, bits in _mixed_cases(L)]
    for what, bits, aligned in cases:
        route, runs, tile = B.census(B.MODS["v13"], B.tables(bits), causal, D)
        assert tile == ((128, 128) if D == 64 else (64, 128)), f"A100 backward tile for D={D}: {tile}"
        ref_route, ref_cols = B.census_ref(bits.numpy(), *tile, causal=causal)
        assert route == ref_route, f"{what}: route {route} vs reference {ref_route}"
        assert [B.runs_prefix(r) for r in runs] == ref_cols, f"{what}: run table differs from the reference"
        assert (route == 0) == aligned, what


@cuda
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("D", [64, 128])
@pytest.mark.parametrize("causal", [True, False])
def test_bwd_aligned_real_tasks(D, causal, dtype):
    _, B = _need(dtype)
    for N, c, itr, pick in ALIGNED_CASES:
        t = _task(N, c, itr, pick)
        L, tabs = int(t.local_size), B.task_tables(t)
        assert B.census(B.MODS["v13"], tabs, causal, D)[0] == 0, f"c={c} itr={itr} L={L}: census must pick ALIGNED"
        out, ref, _ = _bwd_pair(B, tabs, L, D, causal, dtype, seed=pick)
        _assert_bwd_match(out, ref, f"c={c} itr={itr} N={N} path={t.path} L={L} (L%128={L % 128}) D={D} causal={causal} {dtype}")


@cuda
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("causal", [True, False])
def test_bwd_engine_call_with_dsoftmax_sum(causal, dtype):
    """The engine passes a global dsoftmax_sum (no preprocess pass, dq_accum zeroed by the entry)."""
    _, B = _need(dtype)
    for N, c, itr, pick in ((32768, 7, 1, 3), (100003, 7, 1, 5), (131072, 7, 2, 25)):
        t = _task(N, c, itr, pick)
        out, ref, _ = _bwd_pair(B, B.task_tables(t), int(t.local_size), 64, causal, dtype, seed=pick, dsum=True)
        _assert_bwd_match(out, ref, f"dsoftmax_sum given, c={c} itr={itr} path={t.path} causal={causal} {dtype}")


@cuda
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("D", [64, 128])
@pytest.mark.parametrize("causal", [True, False])
def test_bwd_zero_all_masked_and_plain(D, causal, dtype):
    _, B = _need(dtype)
    for L in (16384, 12345):
        for pattern in ("zero", "allones"):
            bits = torch.zeros(L, dtype=torch.int64) if pattern == "zero" else torch.full((L,), -1, dtype=torch.int64)
            tabs = B.tables(bits)
            assert B.census(B.MODS["v13"], tabs, causal, D)[0] == 0
            out, ref, _ = _bwd_pair(B, tabs, L, D, causal, dtype)
            _assert_bwd_match(out, ref, f"{pattern} L={L} D={D} causal={causal} {dtype}")
            if pattern == "allones":      # every pair masked: no live tile anywhere, all gradients exactly zero
                assert all((x == 0).all() for x in out)
        # CQS off: the ALIGNED loop without a run table == v11's CQS-off backward (fp16); finite and
        # consistent with the forced LEGACY route (bf16: the same loop with the run table off vs v11's code)
        q, k, v, do = _rand(L, D, dtype, 4)
        fwd_mod = B.MODS["v11"] if dtype == torch.float16 else B.MODS["v13"]
        o16, lse = B.plain_fwd(fwd_mod, q, k, v, causal)
        a = B.cqs_bwd(B.MODS["v13"], do, q, k, v, o16, lse, None, causal)
        b = (B.cqs_bwd(B.MODS["v11"], do, q, k, v, o16, lse, None, causal) if dtype == torch.float16
             else B.cqs_bwd(B.MODS["v13"], do, q, k, v, o16, lse, None, causal, flags=B.FORCE_LEGACY))
        _assert_bwd_match(a, b, f"plain L={L} D={D} causal={causal} {dtype}")


@cuda
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("D", [64, 128])
@pytest.mark.parametrize("causal", [True, False])
def test_bwd_mixed_patterns_through_legacy(D, causal, dtype):
    _, B = _need(dtype)
    for what, bits in _mixed_cases():
        tabs = B.tables(bits)
        assert B.census(B.MODS["v13"], tabs, causal, D)[0] & 1, f"{what}: census must pick LEGACY"
        out, ref, _ = _bwd_pair(B, tabs, bits.numel(), D, causal, dtype)
        _assert_bwd_match(out, ref, f"{what} (LEGACY) D={D} causal={causal} {dtype}")


@cuda
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("causal", [True, False])
def test_bwd_float64_error_within_fa2(causal, dtype):
    """Gradients vs float64 on an aligned real task (ALIGNED) and a random pattern (LEGACY): norm-relative error
    <= 1.2x flash_attn's error on the unmasked problem of the same inputs + 1e-4 (both dtypes), and the zero
    pattern head to head with flash_attn."""
    _, B = _need(dtype)
    flash_attn = pytest.importorskip("flash_attn")
    t = _task(8192, 7, 1, 0)
    L, D = int(t.local_size), 64
    q, k, v, do = _rand(L, D, dtype, 4, seed=11)
    zero = B.tables(torch.zeros(L, dtype=torch.int64))
    qf, kf, vf = (x.clone().requires_grad_(True) for x in (q, k, v))
    of = flash_attn.flash_attn_func(qf, kf, vf, causal=causal, softmax_scale=D ** -0.5)
    fa = torch.autograd.grad(of, (qf, kf, vf), do)
    ref0 = _bwd_ref64(q, k, v, do, zero[0], causal)
    fa_err = [_rel(a, b) for a, b in zip(fa, ref0)]
    patterns = [("zero", zero), ("real", B.task_tables(t)),
                ("random3bit", B.tables(torch.randint(0, 8, (L,), generator=torch.Generator().manual_seed(2))))]
    for name, tabs in patterns:
        o16, lse = B.cqs_fwd(B.MODS["v13"], q, k, v, tabs, causal)
        g = B.cqs_bwd(B.MODS["v13"], do, q, k, v, o16, lse, tabs, causal, flags=0)
        ref = ref0 if name == "zero" else _bwd_ref64(q, k, v, do, tabs[0], causal)
        for grad, a, b, fe in zip(("dq", "dk", "dv"), g, ref, fa_err):
            e = _rel(a, b)
            assert e <= 1.2 * fe + 1e-4, f"{name} {grad} {dtype}: error vs float64 {e:.3e} > 1.2 x flash_attn's {fe:.3e} + 1e-4"


@cuda
@pytest.mark.parametrize("causal", [True, False])
def test_bwd_deterministic_matches_v11_deterministic(causal):
    """deterministic=True: reruns torch.equal, and dq / dk / dv all torch.equal to v11's deterministic backward."""
    _, B = _mods()
    t = _task(100003, 7, 1, 5)
    out, ref, _ = _bwd_pair(B, B.task_tables(t), int(t.local_size), 64, causal, torch.float16, seed=5, deterministic=True)
    _assert_bwd_match(out, ref, f"deterministic vs v11 deterministic, causal={causal}", dq_equal=True)
    out2, _, _ = _bwd_pair(B, B.task_tables(t), int(t.local_size), 64, causal, torch.float16, seed=5, deterministic=True)
    assert all(torch.equal(a, b) for a, b in zip(out, out2))


@cuda
@pytest.mark.parametrize("causal,window", [(True, (256, 0)), (False, (256, 128))])
def test_bwd_window_calls_route_to_legacy(causal, window):
    _, B = _mods()
    t = _task(32768, 7, 1, 3)
    out, ref, _ = _bwd_pair(B, B.task_tables(t), int(t.local_size), 64, causal, torch.float16, seed=3, window=window)
    _assert_bwd_match(out, ref, f"window {window} causal={causal}")


# ================================================================================================ engine
def _v13_env():
    """The test process's environment without the module overrides, so interface.py makes its own choice; the
    directory the v13 module was imported from is appended to PYTHONPATH (variant dirs go last)."""
    spec = importlib.util.find_spec("cqsa_cuda_v13")
    if spec is None or not spec.origin:
        pytest.skip("cqsa_cuda_v13 not importable")
    env = {k: v for k, v in os.environ.items()
           if k not in ("CQSA_CUDA_MODULE", "CQSA_CUDA_MODULE_NONCAUSAL", "CQSA_BACKWARD", "CQSA_FORWARD")}
    env["PYTHONPATH"] = ":".join([REPO] + [p for p in env.get("PYTHONPATH", "").split(":") if p] + [os.path.dirname(spec.origin)])
    return env


def _run_py(code, env):
    r = subprocess.run([sys.executable, "-P", "-c", code], env=env, capture_output=True, text=True, cwd=REPO, timeout=1200)
    assert r.returncode == 0, f"subprocess failed:\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}"
    return r.stdout


def test_engine_default_module_is_v13():
    """Without CQSA_CUDA_MODULE in the environment the interface loads cqsa_cuda_v13, serves non-causal calls from
    the same module (CQSA_CUDA_MODULE_NONCAUSAL unnecessary) and defaults the backward to its CUDA kernel;
    CQSA_BACKWARD=triton still forces Triton."""
    env = _v13_env()
    out = _run_py("""
import stream_cqsa, stream_cqsa.interface as I
assert stream_cqsa.__file__.startswith(%r), stream_cqsa.__file__
assert I.CQSA_CUDA_MODULE == "cqsa_cuda_v13", I.CQSA_CUDA_MODULE
assert I.CQSA_CUDA_UNIFIED and I.cqsa_cuda_noncausal is I.cqsa_cuda and I.CQSA_CUDA_MODULE_NONCAUSAL == "cqsa_cuda_v13"
assert I.backward_kernel() == "cuda", I.backward_kernel()
ks = stream_cqsa.kernels_available()
assert ks["cuda_extension"] and ks["cuda_extension_noncausal"] and ks["cuda_extension_unified"], ks
print("OK", I.cqsa_cuda.__file__)
""" % REPO, env)
    assert out.startswith("OK")
    out = _run_py('import stream_cqsa.interface as I; assert I.backward_kernel() == "triton"; print("OK")', dict(env, CQSA_BACKWARD="triton"))
    assert out.startswith("OK")


@cuda
@pytest.mark.parametrize("dtype", DTYPES)
def test_engine_attention_forward_backward_on_v13(dtype):
    """stream_cqsa.attention on the v13 module (classic engine, kernel="cuda", pinned itr=1 c=7, CUDA backward by
    default): forward and gradients within 1.5x SDPA's error of the float64 reference, causal and non-causal."""
    _need(dtype)
    env = _v13_env()
    out = _run_py("""
import torch, torch.nn.functional as F, stream_cqsa, stream_cqsa.interface as I
assert I.CQSA_CUDA_MODULE == "cqsa_cuda_v13" and I.backward_kernel() == "cuda"
dt = getattr(torch, %r)
rel = lambda a, b: ((a.double() - b.double()).norm() / b.double().norm()).item()
for causal in (True, False):
    torch.manual_seed(0)
    # N=4096: the float64 SDPA reference holds the H x N x N scores (512 MiB) -- the shared vis1 GPU's share
    q, k, v = (torch.randn(1, 4, 4096, 64, device="cuda", dtype=dt, requires_grad=True) for _ in range(3))
    dout = torch.randn(1, 4, 4096, 64, device="cuda", dtype=dt)
    out, info = stream_cqsa.attention(q, k, v, is_causal=causal, kernel="cuda", itr=1, c=7, bwd_itr=1, return_info=True)
    # the backward-direction info names the engine, kernel, itr and c (not the subproblem count)
    assert (info["engine"], info["kernel"], info["itr"], info["c"]) == ("classic", "cuda", 1, 7), info
    g = torch.autograd.grad(out, (q, k, v), dout)
    q2, k2, v2 = (t.detach().double().requires_grad_(True) for t in (q, k, v))
    o2 = F.scaled_dot_product_attention(q2, k2, v2, is_causal=causal)
    g2 = torch.autograd.grad(o2, (q2, k2, v2), dout.double())
    q3, k3, v3 = (t.detach().clone().requires_grad_(True) for t in (q, k, v))
    o3 = F.scaled_dot_product_attention(q3, k3, v3, is_causal=causal)
    g3 = torch.autograd.grad(o3, (q3, k3, v3), dout)
    assert rel(out, o2) <= 1.5 * rel(o3, o2) + 1e-4, (causal, rel(out, o2), rel(o3, o2))
    for name, a, b, c in zip(("dq", "dk", "dv"), g, g2, g3):
        assert rel(a, b) <= 1.5 * rel(c, b) + 1e-4, (causal, name, rel(a, b), rel(c, b))
print("OK")
""" % dtype.__str__().split(".")[-1], env)
    assert out.strip().endswith("OK")
