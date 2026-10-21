"""
The native wave kernel v2 (kernels/native_v2, module cqsa_native_v2): the census + ALIGNED run-table design of
v12f / v12b on the uniform wave layout, W subproblems per launch, lens[W].

Contract checked here (shared vis1 GPU, small N): the wave forward of W subproblems equals the per-task v12f
forward bit for bit (fp32 out and lse; ALIGNED on every 128-aligned task, finite inputs), W in {1, 7, 73}, read
in place through the block map or gathered, ragged N (lens, the Is_even_MN=false instantiation), causal and
non-causal, head dims 64 and 128; the wave backward's dk/dv equal v12b's per task bit for bit and its dq within
the fp32 atomics' drift; a subproblem with mixed tiles routes to LEGACY inside an otherwise ALIGNED wave and
equals v11; the engine (wave_forward / wave_backward / wave_attention on cqsa_native_v2, uniform layout for
every call) against float64 and FlashAttention, in-place vs gathered, several waves, deterministic reruns, the
chunk pool and the host accumulator (the checks of next/native/test_wave.py); fp16 and bf16 where the build has
the kernels. The reference modules come from benchmarks/ultimate/{fwd_ab,bwd_ab}.py (v11, v12f, v12b on
PYTHONPATH); everything skips without a GPU or the extensions.
"""
import functools, os, sys

import numpy as np
import pytest
import torch
import torch.nn.functional as F

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
DEV = torch.device("cuda")
NAN = float("nan")
SEG_ALIGN = 128


def _harness():
    path = os.path.join(REPO, "benchmarks", "ultimate")
    if path not in sys.path:
        sys.path.insert(0, path)
    import fwd_ab, bwd_ab
    return fwd_ab, bwd_ab


@functools.lru_cache(maxsize=None)
def _ext():
    """cqsa_native_v2 (or CQSA_NATIVE_MODULE); the wave API must be the v2 one."""
    from stream_cqsa.native_wave import native_ext, wave_api
    try:
        ext = native_ext()
    except ImportError as e:
        pytest.skip(f"cqsa_native_v2 not importable: {e}")
    if wave_api(ext) < 2:
        pytest.skip("the loaded wave extension is the first-generation cqsa_native")
    return ext


def _ref_mods(*names):
    fwd_ab, bwd_ab = _harness()
    try:
        if any(n in ("v11", "v12f") for n in names):
            fwd_ab.load_modules(tuple(n for n in names if n in fwd_ab.MODULES))
        if "v12b" in names:
            bwd_ab.load_modules(("v12b",))
    except Exception as e:  # noqa
        pytest.skip(f"reference module not importable: {e}")
    return fwd_ab, bwd_ab


def _supports(dtype, D):
    from stream_cqsa.native_wave import native_supports
    if not native_supports(dtype, int(D)):
        pytest.skip(f"this cqsa_native_v2 build has no {dtype} head_dim={D} kernels")


@functools.lru_cache(maxsize=None)
def _tasks(N, c, itr=1, seg_align=SEG_ALIGN):
    fwd_ab, _ = _harness()
    return sorted(fwd_ab.c_tasks(N, c, itr, seg_align=seg_align), key=lambda t: (int(t.local_size), t.path))


def _qkv(N, H, D, dtype, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return [torch.randn(1, H, N, D, device=DEV, dtype=dtype, generator=g) for _ in range(3)]


def _gather(t, task):
    """[1, H, N, D] -> the task's packed [1, L, H, D]."""
    return t[0].index_select(1, task.token_ids.to(DEV)).transpose(0, 1).unsqueeze(0).contiguous()


def _wave(tasks, q, k, v, causal, in_place, flags=0, fill=NAN):
    """One fwd_wave launch over `tasks` (uniform layout); returns (tables, out_pack [W*S, H, D], lse [W, H, S])."""
    from stream_cqsa.native_wave import build_wave_tables, block_base_for
    ext = _ext()
    tb = build_wave_tables(list(tasks), DEV, uniform=True)
    H, D = q.shape[1], q.shape[3]
    qt, kt, vt = (x[0].transpose(0, 1) for x in (q, k, v))                     # [N, H, D] strided views
    if in_place:
        qs, ks, vs = qt, kt, vt
        bb = block_base_for(tb, lambda g: g).to(DEV)
    else:
        valid = tb.tok >= 0
        pos = torch.nonzero(valid).squeeze(1); src = tb.tok[valid]
        def gather_rows(t):
            g = torch.zeros((tb.total, H, D), device=DEV, dtype=t.dtype)
            g[pos] = t[src]
            return g
        qs, ks, vs = (gather_rows(t) for t in (qt, kt, vt)); bb = None
    out = torch.full((tb.total, H, D), fill, device=DEV, dtype=torch.float32)
    lse = torch.full((tb.W, H, tb.S), fill, device=DEV, dtype=torch.float32)
    ext.fwd_wave(qs, ks, vs, tb.lens, int(tb.S), tb.bits, tb.blk_or, tb.blk_and, bb, SEG_ALIGN if in_place else 0,
                 D ** -0.5, bool(causal), out, lse, int(flags))
    return tb, out, lse


def _per_task_fwd(mod_name, tasks, q, k, v, causal):
    """The reference module's forward per task: list of (acc [L, H, D] fp32, lse [H, L])."""
    fwd_ab, _ = _ref_mods(mod_name)
    refs = []
    for t in tasks:
        qi, ki, vi = (_gather(x, t) for x in (q, k, v))
        acc, lse, _ = fwd_ab.cqs_fwd(fwd_ab.MODS[mod_name], qi, ki, vi, fwd_ab.task_tables(t), causal, fill=NAN)
        refs.append((acc[0], lse[0]))
    return refs


def _assert_wave_equals(tb, out, lse, refs, what):
    for w, (acc, lse_t) in enumerate(refs):
        L = int(tb.lens[w])
        o = out[w * tb.S: w * tb.S + L]
        l = lse[w, :, :L]
        assert torch.equal(o, acc), f"{what}: subproblem {w} (L={L}) out differs, max |d| {(o - acc).abs().max().item():.3e}"
        assert torch.equal(l, lse_t), f"{what}: subproblem {w} (L={L}) lse differs, max |d| {(l - lse_t).abs().max().item():.3e}"


# ------------------------------------------------------------------------------------------------ census
@cuda
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("D", [64, 128])
def test_wave_census_matches_per_task(causal, D):
    """The wave census (one route word and run table per subproblem) equals v12f's census of each task."""
    from stream_cqsa.native_wave import build_wave_tables
    ext = _ext()
    fwd_ab, _ = _ref_mods("v12f")
    tasks = _tasks(100003, 7)                                # ragged N: even and ragged subproblems in one wave
    tb = build_wave_tables(tasks, DEV, uniform=True)
    route, runs, tile = ext.fwd_cqs_census(tb.bits, tb.blk_or, tb.blk_and, causal, D, tb.lens, int(tb.S))
    assert tuple(int(x) for x in tile) == fwd_ab.ALIGNED_TILE[D]
    route, runs = route.cpu().numpy(), runs.cpu().numpy()
    for w, t in enumerate(tasks):
        r_ref, runs_ref, _ = fwd_ab.census(fwd_ab.MODS["v12f"], fwd_ab.task_tables(t), causal, D)
        assert int(route[w]) == r_ref == 0, f"task {t.path}: route {route[w]} vs v12f {r_ref}"
        num_m = -(-int(t.local_size) // tile[0].item())
        for m in range(num_m):
            assert fwd_ab.runs_prefix(runs[w, m]) == fwd_ab.runs_prefix(runs_ref[m]), f"task {t.path} row block {m}"
    # the backward census per column block, against v12b's
    _, bwd_ab = _ref_mods("v12b")
    route, runs, tile = ext.bwd_cqs_census(tb.bits, tb.blk_or, tb.blk_and, causal, D, tb.lens, int(tb.S))
    route, runs = route.cpu().numpy(), runs.cpu().numpy()
    for w, t in enumerate(tasks):
        r_ref, runs_ref, tile_ref = bwd_ab.census(bwd_ab.MODS["v12b"], bwd_ab.task_tables(t), causal, D)
        assert tuple(int(x) for x in tile) == tile_ref
        assert int(route[w]) == r_ref == 0
        num_n = -(-int(t.local_size) // tile[1].item())
        for n in range(num_n):
            assert bwd_ab.runs_prefix(runs[w, n]) == bwd_ab.runs_prefix(runs_ref[n]), f"task {t.path} column block {n}"


# ------------------------------------------------------------------------------------------------ forward identity
FWD_CASES = [
    # (N, c, itr, causal, D, in_place): W = c**itr subproblems in ONE launch
    (32768, 7, 1, True, 64, True), (32768, 7, 1, True, 64, False),
    (32768, 7, 1, False, 64, True), (32768, 7, 1, False, 64, False),
    (32768, 7, 1, True, 128, True), (32768, 7, 1, False, 128, False),
    (131072, 73, 1, True, 64, True), (131072, 73, 1, False, 64, False),
    (100003, 7, 1, True, 64, True), (100003, 7, 1, False, 64, True),     # ragged lens: Is_even_MN=false
    (100003, 7, 1, True, 128, False),
    (32768, 7, 2, True, 64, True),                                        # 49 small subproblems
]


@cuda
@pytest.mark.parametrize("N,c,itr,causal,D,in_place", FWD_CASES)
def test_fwd_wave_equals_v12f_per_task(N, c, itr, causal, D, in_place):
    """One launch over every task of the decomposition == v12f's per-task ALIGNED forward, bit for bit."""
    _supports(torch.float16, D)
    tasks = _tasks(N, c, itr)
    H = 2 if c == 73 else 4
    q, k, v = _qkv(N, H, D, torch.float16)
    tb, out, lse = _wave(tasks, q, k, v, causal, in_place)
    refs = _per_task_fwd("v12f", tasks, q, k, v, causal)
    _assert_wave_equals(tb, out, lse, refs, f"N={N} c={c} itr={itr} causal={causal} D={D} in_place={in_place}")
    # every subproblem took the ALIGNED route
    route = _ext().fwd_cqs_census(tb.bits, tb.blk_or, tb.blk_and, causal, D, tb.lens, int(tb.S))[0]
    assert int(route.abs().sum().item()) == 0


@cuda
@pytest.mark.parametrize("causal", [True, False])
def test_fwd_wave_w1_is_the_classic_call(causal):
    """W=1: a single subproblem through fwd_wave == v12f's classic call == the module's own classic entry."""
    _supports(torch.float16, 64)
    fwd_ab, _ = _ref_mods("v12f")
    ext = _ext()
    t = _tasks(32768, 7)[2]
    q, k, v = _qkv(32768, 4, 64, torch.float16)
    for in_place in (True, False):
        tb, out, lse = _wave([t], q, k, v, causal, in_place)
        _assert_wave_equals(tb, out, lse, _per_task_fwd("v12f", [t], q, k, v, causal), f"W=1 in_place={in_place}")
    qi, ki, vi = (_gather(x, t) for x in (q, k, v))
    acc, lse_c, out16 = fwd_ab.cqs_fwd(ext, qi, ki, vi, fwd_ab.task_tables(t), causal, fill=NAN)
    acc_ref, lse_ref, out16_ref = fwd_ab.cqs_fwd(fwd_ab.MODS["v12f"], qi, ki, vi, fwd_ab.task_tables(t), causal, fill=NAN)
    assert torch.equal(acc, acc_ref) and torch.equal(lse_c, lse_ref) and torch.equal(out16, out16_ref)
    # lens on the device (the sync-free path) with lens_even stated == lens on the CPU
    from stream_cqsa.native_wave import build_wave_tables
    tb = build_wave_tables([t], DEV, uniform=True)
    out_d, lse_d = ext.fwd_wave(qi[0], ki[0], vi[0], tb.lens.to(DEV), int(tb.S), tb.bits, tb.blk_or, tb.blk_and, None, 0,
                                64 ** -0.5, causal, lens_even=True)
    assert torch.equal(out_d, acc[0]) and torch.equal(lse_d[0], lse_ref[0])


@cuda
def test_fwd_in_place_equals_gathered_ragged_N():
    """Reading the original tensor in place through the block map == the gathered wave, for a ragged N."""
    _supports(torch.float16, 64)
    tasks = _tasks(100003, 7)
    q, k, v = _qkv(100003, 4, 64, torch.float16)
    for causal in (True, False):
        tb, out_g, lse_g = _wave(tasks, q, k, v, causal, in_place=False)
        _, out_i, lse_i = _wave(tasks, q, k, v, causal, in_place=True)
        valid = (tb.tok >= 0)
        rows = lambda lse: lse.transpose(1, 2).reshape(-1, lse.shape[1])     # [W, H, S] -> [W*S, H]
        assert torch.equal(out_i[valid], out_g[valid]) and torch.equal(rows(lse_i)[valid], rows(lse_g)[valid])


@cuda
def test_fwd_row_block_order_flags():
    """Ascending / descending row-block orders and the default give bit-identical outputs."""
    _supports(torch.float16, 64)
    ext = _ext()
    tasks = _tasks(32768, 7)
    q, k, v = _qkv(32768, 4, 64, torch.float16)
    tb, out0, lse0 = _wave(tasks, q, k, v, True, True)
    valid = tb.tok >= 0                                 # the NaN-filled padding rows are never written
    rows = lambda lse: lse.transpose(1, 2).reshape(-1, lse.shape[1])
    for flags in (ext.CQS_FWD_REVERSE_M, ext.CQS_FWD_ASCENDING_M):
        _, out, lse = _wave(tasks, q, k, v, True, True, flags=flags)
        assert torch.equal(out[valid], out0[valid]) and torch.equal(rows(lse)[valid], rows(lse0)[valid])


# ------------------------------------------------------------------------------------------------ LEGACY routing
@cuda
@pytest.mark.parametrize("causal", [True, False])
def test_mixed_subproblem_routes_to_legacy_inside_the_wave(causal):
    """A wave of aligned tasks plus one unaligned (seg_align=0) task: the unaligned one routes to LEGACY and
    equals v11, the others stay ALIGNED and equal v12f; forced LEGACY equals v11 everywhere."""
    _supports(torch.float16, 64)
    from stream_cqsa.native_wave import build_wave_tables
    ext = _ext()
    fwd_ab, _ = _ref_mods("v11", "v12f")
    N = 32768
    tasks = _tasks(N, 7)[:3] + [_tasks(N, 7, seg_align=0)[0]]
    q, k, v = _qkv(N, 4, 64, torch.float16)
    tb, out, lse = _wave(tasks, q, k, v, causal, in_place=False)
    route = ext.fwd_cqs_census(tb.bits, tb.blk_or, tb.blk_and, causal, 64, tb.lens, int(tb.S))[0].cpu().numpy()
    assert list(route[:3]) == [0, 0, 0] and route[3] & 1
    refs = _per_task_fwd("v12f", tasks[:3], q, k, v, causal) + _per_task_fwd("v11", tasks[3:], q, k, v, causal)
    _assert_wave_equals(tb, out, lse, refs, f"mixed wave causal={causal}")
    _, out_l, lse_l = _wave(tasks, q, k, v, causal, in_place=False, flags=ext.CQS_FWD_FORCE_LEGACY)
    _assert_wave_equals(tb, out_l, lse_l, _per_task_fwd("v11", tasks, q, k, v, causal), f"forced LEGACY causal={causal}")


# ------------------------------------------------------------------------------------------------ backward identity
def _global_lse_dpsum(q, k, v, dout, causal):
    """The global lse [H, N] and rowsum(dO * O) [H, N] (fp32) of the full attention: the shared input of the
    wave backward and the per-task reference (the identity needs the same values on both sides, not exact
    ones), from flash_attn on the shared GPU (a float64 score matrix would be H * N^2 * 8 bytes)."""
    from flash_attn import flash_attn_func
    o, lse, _ = flash_attn_func(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), causal=causal,
                                softmax_scale=q.shape[-1] ** -0.5, return_attn_probs=True)
    dpsum = (o.float() * dout.transpose(1, 2).float()).sum(-1).transpose(1, 2)          # [1, H, N]
    return lse[0].contiguous(), dpsum[0].contiguous()


def _bwd_wave(tasks, q, k, v, dout, lse, dpsum, causal, flags=0):
    """One bwd_wave launch over `tasks`: (tables, dq_p, dk_p, dv_p) in the uniform layout, NaN-filled outputs."""
    from stream_cqsa.native_wave import build_wave_tables
    ext = _ext()
    tb = build_wave_tables(list(tasks), DEV, uniform=True)
    H, D = q.shape[1], q.shape[3]
    valid = tb.tok >= 0
    pos = torch.nonzero(valid).squeeze(1); src = tb.tok[valid]
    def gather_rows(t):
        g = torch.zeros((tb.total, H, D), device=DEV, dtype=t.dtype)
        g[pos] = t[0].transpose(0, 1)[src]
        return g
    def pack_rows(x):
        p = torch.zeros((H, tb.total), device=DEV, dtype=torch.float32)
        p[:, pos] = x[:, src]
        return p.view(H, tb.W, tb.S).transpose(0, 1).contiguous()
    qp, kp, vp, dop = (gather_rows(t) for t in (q, k, v, dout))
    grads = [torch.full((tb.total, H, D), NAN, device=DEV, dtype=q.dtype) for _ in range(3)]
    ext.bwd_wave(dop, qp, kp, vp, pack_rows(lse), pack_rows(dpsum), tb.lens, int(tb.S), tb.bits, tb.blk_or, tb.blk_and,
                 D ** -0.5, bool(causal), *grads, int(flags))
    return tb, grads


def _per_task_bwd(mod_name, tasks, q, k, v, dout, lse, dpsum, causal):
    _, bwd_ab = _ref_mods(mod_name if mod_name == "v12b" else "v11")
    mod = bwd_ab.MODS["v12b"] if mod_name == "v12b" else bwd_ab.load_modules(("v11",))["v11"]
    refs = []
    for t in tasks:
        idx = t.token_ids.to(DEV)
        qi, ki, vi, doi = (_gather(x, t) for x in (q, k, v, dout))
        lse_i = lse.index_select(1, idx).unsqueeze(0).contiguous()
        dps_i = dpsum.index_select(1, idx).unsqueeze(0).contiguous()
        dq, dk, dv = bwd_ab.cqs_bwd(mod, doi, qi, ki, vi, doi, lse_i, bwd_ab.task_tables(t), causal, dsum=dps_i, fill=NAN)
        refs.append((dq[0], dk[0], dv[0]))
    return refs


BWD_CASES = [
    (16384, 7, 1, True, 64), (16384, 7, 1, False, 64), (16384, 7, 1, True, 128),
    (65536, 73, 1, True, 64), (16384, 7, 2, True, 64), (12003, 7, 1, True, 64), (12003, 7, 1, False, 64),
]


@cuda
@pytest.mark.parametrize("N,c,itr,causal,D", BWD_CASES)
def test_bwd_wave_equals_v12b_per_task(N, c, itr, causal, D):
    """One backward launch over every task == v12b per task: dk/dv bit for bit, dq within the atomics' drift."""
    _supports(torch.float16, D)
    tasks = _tasks(N, c, itr)
    H = 2 if c == 73 else 4
    q, k, v = _qkv(N, H, D, torch.float16)
    dout = _qkv(N, H, D, torch.float16, seed=1)[0]
    lse, dpsum = _global_lse_dpsum(q, k, v, dout, causal)
    tb, (dq, dk, dv) = _bwd_wave(tasks, q, k, v, dout, lse, dpsum, causal)
    refs = _per_task_bwd("v12b", tasks, q, k, v, dout, lse, dpsum, causal)
    for w, (dq_r, dk_r, dv_r) in enumerate(refs):
        L = int(tb.lens[w]); sl = slice(w * tb.S, w * tb.S + L)
        assert torch.equal(dk[sl], dk_r) and torch.equal(dv[sl], dv_r), f"subproblem {w}: dk/dv differ from v12b"
        d = (dq[sl].float() - dq_r.float()).abs().max().item()
        assert torch.isfinite(dq[sl].float()).all() and d <= 2e-3 * max(1.0, dq_r.float().abs().max().item()), f"subproblem {w}: dq max |d| {d:.3e}"
    route = _ext().bwd_cqs_census(tb.bits, tb.blk_or, tb.blk_and, causal, D, tb.lens, int(tb.S))[0]
    assert int(route.abs().sum().item()) == 0


@cuda
def test_bwd_mixed_subproblem_routes_to_legacy_inside_the_wave():
    _supports(torch.float16, 64)
    ext = _ext()
    N, causal = 16384, True
    tasks = _tasks(N, 7)[:2] + [_tasks(N, 7, seg_align=0)[0]]
    q, k, v = _qkv(N, 4, 64, torch.float16)
    dout = _qkv(N, 4, 64, torch.float16, seed=1)[0]
    lse, dpsum = _global_lse_dpsum(q, k, v, dout, causal)
    tb, (dq, dk, dv) = _bwd_wave(tasks, q, k, v, dout, lse, dpsum, causal)
    route = ext.bwd_cqs_census(tb.bits, tb.blk_or, tb.blk_and, causal, 64, tb.lens, int(tb.S))[0].cpu().numpy()
    assert list(route[:2]) == [0, 0] and route[2] & 1
    refs = _per_task_bwd("v12b", tasks[:2], q, k, v, dout, lse, dpsum, causal) + _per_task_bwd("v11", tasks[2:], q, k, v, dout, lse, dpsum, causal)
    for w, (dq_r, dk_r, dv_r) in enumerate(refs):
        L = int(tb.lens[w]); sl = slice(w * tb.S, w * tb.S + L)
        assert torch.equal(dk[sl], dk_r) and torch.equal(dv[sl], dv_r), f"subproblem {w}: dk/dv differ"
        assert (dq[sl].float() - dq_r.float()).abs().max().item() <= 2e-3 * max(1.0, dq_r.float().abs().max().item())


# ------------------------------------------------------------------------------------------------ engine
def _ref64(q, k, v, causal, rows=2048):
    """float64 attention and lse of [B, H, N, D] inputs, `rows` query rows at a time (the full [H, N, N] score
    matrix would not fit next to the other users of the shared GPU)."""
    B, H, N, D = q.shape
    out = torch.empty((B, H, N, D), dtype=torch.float64, device=q.device)
    lse = torch.empty((B, H, N), dtype=torch.float64, device=q.device)
    kf, vf = k.double(), v.double()
    col = torch.arange(N, device=q.device)
    for r0 in range(0, N, rows):
        r1 = min(N, r0 + rows)
        s = q[:, :, r0:r1].double() @ kf.transpose(-1, -2) * (D ** -0.5)
        if causal:
            s = s.masked_fill((col[None, :] > torch.arange(r0, r1, device=q.device)[:, None]), float("-inf"))
        lse[:, :, r0:r1] = torch.logsumexp(s, -1)
        out[:, :, r0:r1] = torch.softmax(s, -1) @ vf
    return out, lse


def _rel(a, b):
    b = b.to(a.device)
    return ((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-30)).item()


ENGINE_CASES = [
    # (N, B, itr, c, causal, D, dtype)
    (4096, 2, 1, 7, True, 64, torch.float16), (4096, 1, 1, 7, False, 64, torch.float16),
    (8192, 1, 2, 7, True, 64, torch.float16), (8192, 1, 1, 13, True, 64, torch.float16),
    (6000, 1, 1, 7, True, 64, torch.float16), (6000, 1, 1, 7, False, 64, torch.float16),
    (16384, 1, 1, 31, False, 64, torch.float16),
    (4096, 1, 1, 7, True, 128, torch.float16), (6000, 1, 1, 7, False, 128, torch.float16),
    (4096, 1, 1, 7, True, 64, torch.bfloat16), (6000, 1, 1, 7, False, 64, torch.bfloat16),
    (4096, 1, 1, 7, True, 128, torch.bfloat16),
]


@cuda
@pytest.mark.parametrize("N,B,itr,c,causal,D,dtype", ENGINE_CASES)
def test_engine_forward_vs_float64(N, B, itr, c, causal, D, dtype):
    """wave_forward on cqsa_native_v2 (uniform layout, in place and gathered) vs float64, within FlashAttention's
    error; several waves and reruns agree bit for bit."""
    _ext(); _supports(dtype, D)
    from stream_cqsa.native_wave import wave_forward
    from stream_cqsa.autoconfig import QUORUM_SETS
    H = 4
    torch.manual_seed(0)
    q, k, v = (torch.randn(B, H, N, D, device=DEV, dtype=dtype) for _ in range(3))
    iset = QUORUM_SETS[c]
    out, info = wave_forward(q, k, v, causal=causal, itr=itr, c=c, interest_set=iset, kernel="cuda", gather=False)
    assert info["gather"] is False and info["kernel"] == "cuda"
    o64, l64 = _ref64(q, k, v, causal)
    e_w, e_fa = _rel(out, o64), _rel(F.scaled_dot_product_attention(q, k, v, is_causal=causal), o64)
    lse_err = (info["lse"].double() - l64).abs().max().item()
    assert e_w <= 1.2 * e_fa and lse_err < 1e-2, f"rel err wave {e_w:.2e} vs FA {e_fa:.2e}, lse max |d| {lse_err:.1e}"
    out_g, _ = wave_forward(q, k, v, causal=causal, itr=itr, c=c, interest_set=iset, kernel="cuda", gather=True)
    assert torch.equal(out_g, out), "gathered wave != in-place wave"
    out2, info2 = wave_forward(q, k, v, causal=causal, itr=itr, c=c, interest_set=iset, kernel="cuda", max_wave_subproblems=2)
    assert info2["n_waves"] > 1 and _rel(out2, out) < 1e-6
    out3, _ = wave_forward(q, k, v, causal=causal, itr=itr, c=c, interest_set=iset, kernel="cuda", gather=False)
    assert torch.equal(out3, out)


@cuda
def test_engine_host_resident_and_host_accumulator():
    """The chunk pool (host-resident inputs) and the host accumulator (acc=CPU, sync and async fold) reproduce
    the device-resident result; the checks of next/native/test_wave.py sections 3 and 3b."""
    _ext(); _supports(torch.float16, 64)
    from stream_cqsa.native_wave import wave_forward
    N, H, D = 8192, 4, 64
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, H, N, D, dtype=torch.float16) for _ in range(3))
    qd, kd, vd = (t.to(DEV) for t in (q, k, v))
    out_d, _ = wave_forward(qd, kd, vd, causal=True, itr=1, c=7, interest_set=(0, 1, 3), kernel="cuda")
    out_h, info_h = wave_forward(q, k, v, causal=True, itr=1, c=7, interest_set=(0, 1, 3), kernel="cuda")
    assert _rel(out_h, out_d) < 1e-6
    out_h2, _ = wave_forward(q, k, v, causal=True, itr=1, c=7, interest_set=(0, 1, 3), kernel="cuda", pool_slots=4)
    assert _rel(out_h2, out_d) < 1e-6
    for async_fold in ("1", "0"):
        os.environ["CQSA_WAVE_MERGE_ASYNC"] = async_fold
        try:
            out_c, _ = wave_forward(qd, kd, vd, causal=True, itr=1, c=7, interest_set=(0, 1, 3), kernel="cuda", accumulate_on_gpu=False)
            assert out_c.device.type == "cpu" and _rel(out_c, out_d) < 1e-6
            out_c2, info_c2 = wave_forward(qd, kd, vd, causal=True, itr=1, c=7, interest_set=(0, 1, 3), kernel="cuda",
                                           accumulate_on_gpu=False, max_wave_subproblems=2)
            assert info_c2["n_waves"] > 1 and _rel(out_c2, out_d) < 1e-6
            out_c3, _ = wave_forward(q, k, v, causal=True, itr=1, c=7, interest_set=(0, 1, 3), kernel="cuda",
                                     accumulate_on_gpu=False, pool_slots=4)
            assert _rel(out_c3, out_d) < 1e-6
        finally:
            os.environ.pop("CQSA_WAVE_MERGE_ASYNC", None)
    out_d4, _ = wave_forward(qd, kd, vd, causal=True, itr=2, c=7, interest_set=(0, 1, 3), kernel="cuda")
    out_c4, info_c4 = wave_forward(qd, kd, vd, causal=True, itr=2, c=7, interest_set=(0, 1, 3), kernel="cuda",
                                   accumulate_on_gpu=False, max_wave_subproblems=5)
    assert info_c4["n_waves"] > 1 and _rel(out_c4, out_d4) < 1e-6


@cuda
@pytest.mark.parametrize("N,itr,c,causal,D,dtype", [
    (2048, 1, 7, True, 64, torch.float16), (2048, 1, 7, False, 64, torch.float16), (4096, 2, 7, True, 64, torch.float16),
    (3000, 1, 7, True, 64, torch.float16), (2048, 1, 7, True, 128, torch.float16), (2048, 1, 7, True, 64, torch.bfloat16),
])
def test_engine_backward_vs_float64(N, itr, c, causal, D, dtype):
    """wave_attention's backward (bwd_wave on the uniform layout + wave_scatter_add) vs float64 autograd, within
    FlashAttention's gradient error; several waves agree."""
    _ext(); _supports(dtype, D)
    from stream_cqsa.native_wave import wave_attention, wave_forward, wave_backward
    from stream_cqsa.autoconfig import QUORUM_SETS
    H = 4
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, H, N, D, device=DEV, dtype=dtype, requires_grad=True) for _ in range(3))
    dout = torch.randn(1, H, N, D, device=DEV, dtype=dtype)
    iset = QUORUM_SETS[c]
    out = wave_attention(q, k, v, causal=causal, itr=itr, c=c, interest_set=iset)
    out.backward(dout.float())
    g_w = [t.grad.clone() for t in (q, k, v)]
    q64, k64, v64 = (t.detach().double().requires_grad_(True) for t in (q, k, v))
    o64, _ = _ref64(q64, k64, v64, causal)
    o64.backward(dout.double())
    g_64 = [t.grad for t in (q64, k64, v64)]
    q2, k2, v2 = (t.detach().clone().requires_grad_(True) for t in (q, k, v))
    F.scaled_dot_product_attention(q2, k2, v2, is_causal=causal).backward(dout)
    g_fa = [t.grad for t in (q2, k2, v2)]
    errs = [_rel(a, b) for a, b in zip(g_w, g_64)]
    errs_fa = [_rel(a, b) for a, b in zip(g_fa, g_64)]
    assert all(e <= 1.5 * f + 1e-4 for e, f in zip(errs, errs_fa)), f"dq/dk/dv rel err {errs} vs FA {errs_fa}"
    _, info_f = wave_forward(q.detach(), k.detach(), v.detach(), causal=causal, itr=itr, c=c, interest_set=iset, kernel="cuda")
    (dq2, dk2, dv2), info_b = wave_backward(q.detach(), k.detach(), v.detach(), out.detach(), dout, info_f["lse"],
                                            causal=causal, itr=itr, c=c, interest_set=iset, max_wave_subproblems=3)
    assert info_b["n_waves"] > 1
    assert max(_rel(a, b) for a, b in zip((dq2, dk2, dv2), g_w)) < 2e-3
    # host-resident inputs: gathered on the host into the padded layout
    (dq3, dk3, dv3), _ = wave_backward(q.detach().cpu(), k.detach().cpu(), v.detach().cpu(), out.detach().cpu(), dout.cpu(),
                                       info_f["lse"].cpu(), causal=causal, itr=itr, c=c, interest_set=iset)
    assert max(_rel(a, b) for a, b in zip((dq3, dk3, dv3), g_w)) < 2e-3


@cuda
def test_engine_in_place_possible_for_any_N():
    _ext()
    from stream_cqsa.native_wave import in_place_possible
    assert in_place_possible(100003, True) and in_place_possible(100003, False)


# ------------------------------------------------------------------------------------------------ argument checks
@cuda
def test_wave_argument_checks():
    ext = _ext(); _supports(torch.float16, 64)
    q = torch.zeros((256, 1, 64), device=DEV, dtype=torch.float16)
    bits = torch.zeros(256, dtype=torch.int64, device=DEV)
    summ = torch.zeros(4, dtype=torch.int64, device=DEV)
    with pytest.raises(RuntimeError, match="lens"):
        ext.fwd_wave(q, q, q, torch.tensor([0, 128], dtype=torch.int32), 128, bits, summ, summ, None, 0, 0.125, True)
    with pytest.raises(RuntimeError, match="lens_even"):      # device lens without lens_even
        ext.fwd_wave(q, q, q, torch.tensor([128, 128], dtype=torch.int32, device=DEV), 128, bits, summ, summ, None, 0, 0.125, True)
    with pytest.raises(RuntimeError, match="contradicts"):     # CPU lens with a wrong lens_even
        ext.fwd_wave(q, q, q, torch.tensor([128, 100], dtype=torch.int32), 128, bits, summ, summ, None, 0, 0.125, True, lens_even=True)
    with pytest.raises(RuntimeError, match="multiple of 128"):
        ext.fwd_wave(q, q, q, torch.tensor([100, 100], dtype=torch.int32), 100, bits, summ, summ, None, 0, 0.125, True)
    with pytest.raises(RuntimeError, match="W\\*S rows"):
        ext.fwd_wave(q[:128], q, q, torch.tensor([128, 128], dtype=torch.int32), 128, bits, summ, summ, None, 0, 0.125, True)
