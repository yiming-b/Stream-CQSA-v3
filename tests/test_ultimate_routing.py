"""Routing by the memory model (ultimate P0.2): the wave engine is the executor of device-resident
device-accumulator calls whenever its wave fits, with the wave budget W and gathered-vs-in-place reads
chosen by the planner; a tight budget shrinks W before the depth grows; in place only when the copies
do not fit; the classic pinned pipeline stays for host-resident inputs where it is smaller and for
host-accumulator backwards. Planner tests run on the CPU; the smoke tests need a GPU (N <= 131072)."""
import pytest, torch
import torch.nn.functional as F

from stream_cqsa.autoconfig import plan as _plan, hardware_from_dict, frontier_pick, CostModel
from stream_cqsa.native_wave import wave_token_ladder, task_tokens_estimate, estimate_wave_peak_bytes, default_wave_tokens
from _decomp import attention_decomposed          # every output comparison below asserts that a decomposition ran

E = ("wave", "classic")
N, H, D = 131072, 8, 64
GIB = 2 ** 30


def plan(**kw):
    return _plan(model=CostModel(), **kw)        # the default constants, not a cost model calibrated on this node


def _hw(gib, host="100GiB"):
    return hardware_from_dict({"cuda:0": f"{gib}GiB", "host": host})


@pytest.fixture(autouse=True)
def _no_calibration_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("CQSA_CACHE_DIR", str(tmp_path))


def test_wave_token_ladder_and_memory_model():
    lad = wave_token_ladder(N, 1, 7, 3)
    assert lad[0] == default_wave_tokens(N) and lad[-1] == task_tokens_estimate(N, 1, 7, 3)
    assert all(a > b for a, b in zip(lad, lad[1:])) and all(w == lad[0] >> i for i, w in enumerate(lad[:-1]))
    assert task_tokens_estimate(N, 1, 7, 3) >= 3 * (N // 7)                  # never below the ideal task
    kw = dict(B=1, H=H, D=D, itemsize=2, itr=1, c=7, l=3, stream_from_host=False, accumulate_on_gpu=True, out_bytes_per_el=2)
    g = [estimate_wave_peak_bytes(N, w, gather=True, **kw) for w in lad]
    i = [estimate_wave_peak_bytes(N, w, gather=False, **kw) for w in lad]
    assert all(a > b for a, b in zip(g, g[1:])) and all(x > y for x, y in zip(g, i))   # W and the copies both cost
    # the fitted a-term of the profile sweep (7.149 GiB per M tokens, itr=1 acc=GPU at H=8 D=64 fp16) within 5%
    W = 8 * N                                                              # above both floors: the difference is a*N
    a = (estimate_wave_peak_bytes(2 * N, W, gather=False, **kw) - estimate_wave_peak_bytes(N, W, gather=False, **kw)) / N * 2 ** 20 / GIB
    assert abs(a / 7.149 - 1) < 0.05
    # host-resident: the chunk pool replaces Q/K/V and never exceeds the c chunks
    hkw = dict(kw, stream_from_host=True, accumulate_on_gpu=False)
    assert estimate_wave_peak_bytes(N, lad[0], gather=False, **hkw) < estimate_wave_peak_bytes(N, lad[0], gather=False, **dict(kw, accumulate_on_gpu=False)) + 3 * N * H * D * 2


def test_planner_wave_for_device_resident_forward_and_classic_for_host_accumulator_backward():
    # a generous budget: monolithic fits, so the decomposition shows in the pinned-depth plan (allow_mono=False,
    # what attention(itr=1, plan_only=True) asks for): the wave, gathered, one wave of every subproblem
    p = plan(N=N, H=H, D=D, hardware=_hw(30), allow_distributed=False, out_bytes_per_el=2, engines=E, allow_mono=False,
             quorum_sets={7: (0, 1, 3)})
    assert p.engine == "wave" and p.gather and not p.stream_from_host and p.acc == "gpu" and p.itr == 1
    assert "wave W=" in p.name() and "gathered" in p.name()
    # no classic device-resident device-accumulator candidate is enumerated when the wave is available (W=1 == classic)
    assert not any(c.get("engine") == "classic" and not c["stream_from_host"] and c["acc"] == "gpu" for c in p.candidates)
    # ... but it is when the wave is not (kernel="cuda"/"triton", or a build without the kernel)
    pc = plan(N=N, H=H, D=D, hardware=_hw(30), allow_distributed=False, out_bytes_per_el=2, engines=("classic",), allow_mono=False)
    assert pc.engine == "classic" and pc.wave_tokens == 0 and all(c.get("engine") != "wave" for c in pc.candidates)
    with pytest.raises(ValueError):
        plan(N=N, hardware=_hw(30), engines=("gpu",))
    # a device-resident caller never streams from the host while something device-resident fits: attention()
    # applies the planner's rule to those candidates (plan_only shows it)
    from stream_cqsa.api import _prefer_device_resident
    p2 = plan(N=N, H=H, D=D, hardware=_hw(30), allow_distributed=False, out_bytes_per_el=2, engines=E, allow_mono=False)
    if p2.stream_from_host:
        assert _prefer_device_resident(p2) and "keeping them there" in p2.reason
    assert not p2.stream_from_host and p2.engine == "wave" and p2.gather
    # host-accumulator backward: the classic pipeline (the wave backward keeps fp32 gradients on the device)
    b = plan(N=1048576, H=H, D=D, hardware=_hw(6), allow_distributed=False, engines=E, direction="bwd")
    assert b.mode == "cqsa" and b.acc == "cpu" and b.engine == "classic"
    assert all(c["acc"] == "gpu" and not c["stream_from_host"] for c in b.candidates if c.get("engine") == "wave")
    # device-resident backward with room: the wave (deterministic gradients, one launch per wave)
    b2 = plan(N=N, H=H, D=D, hardware=_hw(40), allow_distributed=False, engines=E, direction="bwd", allow_mono=False)
    assert b2.engine == "wave" and b2.acc == "gpu" and not b2.stream_from_host


def test_tight_budget_shrinks_wave_before_depth_and_in_place_only_when_copies_do_not_fit():
    R = 3 * N * H * D * 2
    kw = dict(N=N, H=H, D=D, allow_distributed=False, out_bytes_per_el=2, resident_input_bytes=R, budget_excludes_inputs=False,
              evict_inputs=False, engines=E, allow_mono=False, quorum_sets={7: (0, 1, 3)})
    picks = [plan(hardware=_hw(g), **kw) for g in (4.0, 1.6, 1.35, 1.3, 1.15)]
    assert [p.engine for p in picks[:4]] == ["wave"] * 4
    assert [p.itr for p in picks] == [1, 1, 1, 1, 2]                            # the depth grows last
    assert picks[0].gather and picks[1].gather and picks[0].wave_tokens > picks[1].wave_tokens   # W halves first
    assert picks[1].wave_tokens == wave_token_ladder(N, 1, 7, 3)[-1]           # down to one subproblem per wave
    assert not picks[2].gather and not picks[3].gather                         # then the copies are given up
    # in place appears only where no gathered rung fits: the gathered floor exceeds that budget
    floor_gathered = estimate_wave_peak_bytes(N, picks[1].wave_tokens, B=1, H=H, D=D, itemsize=2, itr=1, c=7, l=3, stream_from_host=False,
                                              accumulate_on_gpu=True, gather=True, out_bytes_per_el=2)
    assert floor_gathered > 1.35 * 0.85 * GIB >= picks[2].est_peak_gib * GIB
    assert not any(c["ok"] and c.get("engine") == "wave" and c["gather"] and c["itr"] == 1 for c in picks[2].candidates)
    # the frontier rule itself: fastest, ties within 2% broken to the smaller peak
    rows = [dict(time=1.00, peak=3.0), dict(time=1.01, peak=2.0), dict(time=1.05, peak=1.0)]
    assert frontier_pick(rows)[0]["peak"] == 2.0


def test_host_resident_forward_is_decided_by_memory():
    # both engines are enumerated for host-resident inputs; the pick is on the frontier either way
    p = plan(N=1048576, H=H, D=D, hardware=_hw(12, host="200GiB"), allow_distributed=False, out_bytes_per_el=0, engines=E, allow_mono=False)
    hosts = [c for c in p.candidates if c["ok"] and c["stream_from_host"] and c["mode"] == "cqsa"]
    assert {c["engine"] for c in hosts} == {"wave", "classic"}
    assert p.stream_from_host and p.engine in ("wave", "classic")
    # the wave's chunk-pool candidate costs more device memory than the classic pipeline at the same depth (measured 6.25 vs 3.88 GiB at 1M)
    w7 = min(c["peak"] for c in hosts if c["engine"] == "wave" and c["c"] == 7 and c["itr"] == 1 and c["acc"] == "cpu")
    c7 = min(c["peak"] for c in hosts if c["engine"] == "classic" and c["c"] == 7 and c["itr"] == 1 and c["acc"] == "cpu")
    assert w7 > c7
    # host memory: the wave's acc=cpu candidates also hold the async fold's two pinned slots (<= W tokens each)
    per_tok = H * D * 4 + 8 * H + 8
    for cw in (c for c in hosts if c["engine"] == "wave" and c["acc"] == "cpu" and c["c"] == 7 and c["itr"] == 1):
        cc = next(c for c in hosts if c["engine"] == "classic" and c["acc"] == "cpu" and c["c"] == 7 and c["itr"] == 1)
        assert abs((cw["host"] - cc["host"]) * GIB - 2 * min(cw["wave_tokens"], 1048576) * per_tok) < 1e3


def test_host_resident_inputs_plan_only_streaming_candidates():
    """Regression (I0, P0.2 review BLOCKER): for host-resident inputs the planner enumerated device-resident
    wave candidates and, with CostModel defaults, planned 'wave W=32,768 gathered' for an itr=1 backward,
    which crashed in wave_backward. Only host-streaming decompositions can run on host tensors."""
    n = 8192
    kw = dict(N=n, H=H, D=D, hardware=_hw(30), allow_distributed=False, out_bytes_per_el=0, engines=E, allow_mono=False,
              quorum_sets={7: (0, 1, 3)}, min_itr=1, max_itr=1, inputs_on_host=True)
    b = plan(direction="bwd", **kw)
    assert b.stream_from_host and b.engine == "classic" and "host-resident" in b.name()
    assert all(c["stream_from_host"] for c in b.candidates if c["mode"] == "cqsa")
    assert not any(c.get("engine") == "wave" for c in b.candidates)            # the wave backward is device-resident only
    f = plan(direction="fwd", **kw)
    assert f.stream_from_host and all(c["stream_from_host"] for c in f.candidates if c["mode"] == "cqsa")
    assert {c["engine"] for c in f.candidates if c["mode"] == "cqsa"} == {"wave", "classic"}   # the chunk pool competes
    # without the flag the device-resident wave is what a device caller gets
    assert plan(direction="bwd", **dict(kw, inputs_on_host=False)).engine == "wave"


def test_nothing_fits_fallback_under_an_engine_restriction():
    """Regression (I0, P0.2 review MAJOR): plan(engines=('wave',), direction='bwd') with nothing fitting raised
    'min() arg is an empty sequence' (the fallback looked only at host-streamed acc=cpu candidates)."""
    for n, gib in ((1 << 20, 6), (2 << 20, 12)):
        p = plan(N=n, H=H, D=D, hardware=_hw(gib), engines=("wave",), direction="bwd", allow_distributed=False)
        assert p.mode == "cqsa" and p.engine == "wave" and not p.stream_from_host and "nothing fits" in p.reason
        frugal = min(c["peak"] for c in p.candidates if c["mode"] == "cqsa")
        assert p.est_peak_gib == frugal
    # the classic fallback is unchanged: everything offloaded
    p = plan(N=1 << 20, H=H, D=D, hardware=_hw(0.05), engines=E, direction="bwd", allow_distributed=False)
    assert "nothing fits" in p.reason and p.stream_from_host and p.acc == "cpu"
    # no executable decomposition at all: a clear error, not a crash in min()
    with pytest.raises(ValueError, match="device-resident only"):
        plan(N=1 << 20, H=H, D=D, hardware=_hw(6), engines=("wave",), direction="bwd", allow_distributed=False,
             inputs_on_host=True)


def test_monolithic_fits_whenever_a_device_resident_wave_does():
    """kernel="auto" with nothing pinned runs the monolithic call whenever it fits (the design's rule, and the
    fastest point); this pins down why a device-resident wave then runs only after a monolithic OOM or under a
    pinned itr: every device-resident decomposition needs more device memory than the monolithic call."""
    for n in (1 << 15, 1 << 17, 1 << 20, 100003):
        for gib in (0.3, 0.6, 1.2, 2.5, 5, 10, 20, 40):
            for direction in ("fwd", "bwd"):
                p = plan(N=n, H=H, D=D, hardware=_hw(gib), engines=E, direction=direction, allow_distributed=False,
                         out_bytes_per_el=2)
                dev_wave = [c for c in p.candidates if c.get("engine") == "wave" and not c["stream_from_host"]]
                mono = p.candidates[0]
                assert min(c["peak"] for c in dev_wave) > mono["peak"], (n, gib, direction)
                if any(c["ok"] for c in dev_wave):
                    assert mono["ok"] and p.mode == "mono", (n, gib, direction)


# ---------------------------------------------------------------------------------------------- GPU
cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def _native():
    from stream_cqsa.native_wave import native_available, native_supports
    return native_available() and native_supports(torch.float16, D)


def _qkv(n=N, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return tuple(torch.randn(1, H, n, D, dtype=torch.float16, device="cuda", generator=g) for _ in range(3))


def _rel(a, b):
    return ((a.double() - b.double()).norm() / b.double().norm()).item()


@cuda
def test_auto_routes_to_the_wave_and_matches_the_forced_paths():
    import stream_cqsa
    if not _native():
        pytest.skip("cqsa_native not built for fp16 hdim64")
    q, k, v = _qkv()
    p = stream_cqsa.attention(q, k, v, is_causal=True, itr=1, plan_only=True)
    assert p.engine == "wave" and p.gather and p.wave_tokens > 0
    out, info = attention_decomposed(q, k, v, is_causal=True, itr=1)
    assert info["engine"] == "wave" and info["kernel"] == "cuda" and info["gather"] is True
    assert info["max_wave_tokens"] == p.wave_tokens and info["n_waves"] == len(info["wave_sizes"]) and info["wave_oom_retries"] == 0
    o_cuda, i_cuda = attention_decomposed(q, k, v, is_causal=True, itr=1, kernel="cuda", engine="classic")
    o_wave, i_wave = attention_decomposed(q, k, v, is_causal=True, itr=1, kernel="wave-cuda", engine="wave")
    assert i_cuda["engine"] == "classic" and i_wave["engine"] == "wave"
    assert _rel(out, o_cuda) < 1e-4 and _rel(out, o_wave) < 1e-4
    assert torch.equal(out, o_wave)                                  # the same plan: bit-identical
    # accuracy against fp64 (the reference materialises H x N^2 doubles: a small call)
    q, k, v = _qkv(4096)
    out, info = attention_decomposed(q, k, v, is_causal=True, itr=1)
    assert info["engine"] == "wave"
    ref = F.scaled_dot_product_attention(q.double(), k.double(), v.double(), is_causal=True)
    o_s = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    assert _rel(out, ref) <= 1.2 * _rel(o_s, ref)


@cuda
def test_wave_overrides_and_unplanned_combinations():
    import stream_cqsa
    if not _native():
        pytest.skip("cqsa_native not built for fp16 hdim64")
    q, k, v = _qkv()
    base, _ = attention_decomposed(q, k, v, is_causal=True, itr=1, kernel="wave-cuda", engine="wave")
    # in place and a smaller wave budget are honoured and exact
    o_ip, i_ip = attention_decomposed(q, k, v, is_causal=True, itr=1, gather=False)
    assert i_ip["engine"] == "wave" and i_ip["gather"] is False and _rel(o_ip, base) < 1e-4
    o_w, i_w = attention_decomposed(q, k, v, is_causal=True, itr=1, max_wave_tokens=65536)
    assert i_w["engine"] == "wave" and i_w["max_wave_tokens"] == 65536 and i_w["n_waves"] == 7 and _rel(o_w, base) < 1e-4
    # a host accumulator on device-resident inputs is not a planned combination: the classic pipeline runs it
    o_c, i_c = attention_decomposed(q, k, v, is_causal=True, itr=1, accumulate_on_gpu=False, low_memory=True)
    assert i_c["engine"] == "classic" and _rel(o_c, base) < 1e-4
    # every kernel= override still selects its engine
    for kn, eng in (("cuda", "classic"), ("triton", "classic"), ("wave", "wave"), ("wave-triton", "wave")):
        o, i = attention_decomposed(q, k, v, is_causal=True, itr=1, kernel=kn)
        assert i["engine"] == eng and _rel(o, base) < 1e-3, kn
    import os
    os.environ.pop("CQSA_FORWARD", None); os.environ.pop("CQSA_BACKWARD", None)


@cuda
def test_oom_inside_a_wave_halves_the_budget_before_leaving_the_engine(monkeypatch, capsys):
    import stream_cqsa
    from stream_cqsa import native_wave
    if not _native():
        pytest.skip("cqsa_native not built for fp16 hdim64")
    q, k, v = _qkv(8192)
    real = native_wave.wave_forward
    seen = []

    def flaky(*a, max_wave_tokens=None, gather=None, **kw):
        seen.append((max_wave_tokens, gather))
        if max_wave_tokens is None or max_wave_tokens > 16384 or gather:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory (simulated)")
        return real(*a, max_wave_tokens=max_wave_tokens, gather=gather, **kw)

    monkeypatch.setattr(native_wave, "wave_forward", flaky)
    out, info = attention_decomposed(q, k, v, is_causal=True, itr=1, verbose=True)
    s = capsys.readouterr().out
    assert info["engine"] == "wave" and info["gather"] is False and info["max_wave_tokens"] <= 16384
    budgets = [w for w, _ in seen if w is not None]
    assert all(a > b for a, b in zip(budgets, budgets[1:] ) if a != b) and seen[-1][1] is False   # halved, then in place
    assert info["wave_oom_retries"] == len(seen) - 1 and "halving the wave budget" in s and "reading Q/K/V in place" in s
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    assert _rel(out, ref) < 1e-3
    # a forced wave kernel never falls back to the classic engine: the error surfaces
    monkeypatch.setattr(native_wave, "wave_forward", lambda *a, **kw: (_ for _ in ()).throw(torch.cuda.OutOfMemoryError("simulated")))
    with pytest.raises(torch.cuda.OutOfMemoryError):
        stream_cqsa.attention(q, k, v, is_causal=True, itr=1, kernel="wave-cuda")
    out2, info2 = attention_decomposed(q, k, v, is_causal=True, itr=1)   # auto: the classic pipeline
    assert info2["engine"] == "classic" and _rel(out2, ref) < 1e-3


@cuda
def test_verbose_plan_line_reports_the_wave_choice(capsys):
    import stream_cqsa
    if not _native():
        pytest.skip("cqsa_native not built for fp16 hdim64")
    q, k, v = _qkv(32768)
    stream_cqsa.attention(q, k, v, is_causal=True, itr=1, verbose=True)
    s = capsys.readouterr().out
    assert "engine wave, kernel cuda" in s and "wave budget" in s and "Q/K/V gathered" in s
    assert "Stream-CQSA ran:" in s and "waves of" in s
    stream_cqsa.attention(q, k, v, is_causal=True, itr=1, gather=False, verbose=True)
    assert "Q/K/V read in place" in capsys.readouterr().out


@cuda
def test_host_resident_backward_under_auto_is_classic_and_exact():
    """Regression (I0, P0.2 review BLOCKER), end to end with the uncalibrated default cost model (the fixture
    isolates the cache): CPU inputs with requires_grad and itr=1 plan and run the classic host-resident
    pipeline, the forward reports what it planned, and .backward() returns finite CPU gradients."""
    import stream_cqsa
    g = torch.Generator().manual_seed(0)
    q, k, v = (torch.randn(1, H, 8192, D, generator=g).half().requires_grad_() for _ in range(3))
    p = stream_cqsa.attention(q, k, v, is_causal=True, itr=1, plan_only=True)
    assert p.engine == "classic" and p.stream_from_host and "host-resident" in p.name()
    for kern in ("auto", "wave"):                   # a forced wave kernel cannot run a host-resident backward either
        for t in (q, k, v):
            t.grad = None
        out, info = attention_decomposed(q, k, v, is_causal=True, itr=1, kernel=kern)
        assert info["engine"] == "classic" and info["stream_from_host"] and out.device.type == "cpu"
        out.float().sum().backward()
        assert all(t.grad is not None and t.grad.device.type == "cpu" and torch.isfinite(t.grad).all() for t in (q, k, v))
    qd, kd, vd = (t.detach().cuda().requires_grad_() for t in (q, k, v))
    F.scaled_dot_product_attention(qd, kd, vd, is_causal=True).float().sum().backward()
    assert _rel(q.grad, qd.grad.cpu()) < 5e-3 and _rel(v.grad, vd.grad.cpu()) < 5e-3
    # host-resident forward: the plan and the run describe the same configuration
    with torch.no_grad():
        p = stream_cqsa.attention(q, k, v, is_causal=True, itr=1, plan_only=True)
        out, info = attention_decomposed(q, k, v, is_causal=True, itr=1)
    assert p.stream_from_host and info["engine"] == p.engine and info.get("host_resident", True)


@cuda
def test_oom_in_the_wave_backward_is_recovered(monkeypatch, capsys):
    """Regression (I0, P0.2 review MAJOR): an OOM raised inside the wave backward (during loss.backward(),
    outside attention()'s ladder) propagated to the caller. Now the backward halves its wave budget, then under
    kernel="auto" falls back to the classic streamed backward; a forced wave kernel still reports the OOM."""
    import stream_cqsa
    from stream_cqsa import native_wave
    if not _native():
        pytest.skip("cqsa_native not built for fp16 hdim64")
    q0, k0, v0 = _qkv(8192)
    do = torch.randn_like(q0)
    ref = tuple(t.detach().clone().requires_grad_() for t in (q0, k0, v0))
    F.scaled_dot_product_attention(*ref, is_causal=True).backward(do)
    real = native_wave.wave_backward
    budgets = []

    def run(fail_while, kernel="auto"):
        def flaky(*a, max_wave_tokens=None, **kw):
            budgets.append(max_wave_tokens)
            if fail_while(max_wave_tokens):
                raise torch.cuda.OutOfMemoryError("CUDA out of memory (simulated)")
            return real(*a, max_wave_tokens=max_wave_tokens, **kw)
        monkeypatch.setattr(native_wave, "wave_backward", flaky)
        budgets.clear()
        q, k, v = (t.detach().clone().requires_grad_() for t in (q0, k0, v0))
        out, info = attention_decomposed(q, k, v, is_causal=True, itr=1, kernel=kernel, verbose=True)
        assert info["engine"] == "wave" and info["backward"] == {"forward_gather": True}
        out.backward(do)
        return (q, k, v), info["backward"]

    calls = {"n": 0}

    def once(w):
        calls["n"] += 1
        return calls["n"] == 1
    grads, rec = run(once)                                   # one OOM: the halved budget runs
    assert rec["engine"] == "wave" and rec["wave_oom_retries"] == 1 and budgets[1] == budgets[0] // 2
    assert "halving the wave budget" in capsys.readouterr().out
    for t, r in zip(grads, ref):
        assert torch.isfinite(t.grad).all() and _rel(t.grad, r.grad) < 5e-3
    grads, rec = run(lambda w: True)                         # never fits: down the ladder, then the classic backward
    assert rec["engine"] == "classic" and rec["wave_oom_retries"] == len(budgets)
    assert all(a > b for a, b in zip(budgets, budgets[1:]))
    assert "retrying on the streaming engine" in capsys.readouterr().out
    for t, r in zip(grads, ref):
        assert t.grad.is_cuda and t.grad.dtype == torch.float16 and _rel(t.grad, r.grad) < 5e-3
    with pytest.raises(torch.cuda.OutOfMemoryError):
        run(lambda w: True, kernel="wave-cuda")


@cuda
def test_monolithic_oom_replans_among_decompositions(monkeypatch):
    """Minor (I0, P0.2 review): after the monolithic call ran out of memory the re-plan hand-set a classic
    decomposition; it now plans among the decompositions, so the device-resident call runs on the wave."""
    import stream_cqsa
    from stream_cqsa import api
    if not _native():
        pytest.skip("cqsa_native not built for fp16 hdim64")
    q, k, v = _qkv(65536)
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    monkeypatch.setattr(api, "_TRUE_SDPA", lambda *a, **kw: (_ for _ in ()).throw(torch.cuda.OutOfMemoryError("simulated")))
    out, info = attention_decomposed(q, k, v, is_causal=True)        # forced by the simulated monolithic OOM, not by a pin
    assert info["engine"] == "wave" and info["n_subproblems"] > 1 and _rel(out, ref) < 1e-3


@cuda
def test_pinned_depth_is_what_plan_only_reports_and_the_backward_reports_the_gather_that_ran():
    """Minors (I0, P0.2 review): under a pinned itr the plan (and its W) is planned at that depth, so plan_only
    describes the call that runs; a ragged causal backward reports the gathered forward it ran."""
    import stream_cqsa
    if not _native():
        pytest.skip("cqsa_native not built for fp16 hdim64")
    q, k, v = _qkv(32768)
    p = stream_cqsa.attention(q, k, v, is_causal=True, itr=2, plan_only=True)
    assert p.itr == 2 and p.engine == "wave"
    out, info = attention_decomposed(q, k, v, is_causal=True, itr=2)
    assert info["itr"] == 2 and info["max_wave_tokens"] == p.wave_tokens
    q, k, v = (t.contiguous().requires_grad_() for t in _qkv(100003))
    out, info = attention_decomposed(q, k, v, is_causal=True, itr=1, gather=False)
    from stream_cqsa.native_wave import in_place_possible
    # in place is impossible at a ragged causal N on the first wave module; cqsa_native_v2 (lens) reads any N in place
    assert info["engine"] == "wave" and info["gather"] is (not in_place_possible(q.shape[2], True))
    out.float().sum().backward()
    assert torch.isfinite(q.grad).all()


@cuda
def test_inputs_restored_under_a_memory_fraction():
    """Wart (I0): _restore_inputs reserved 10% of the PHYSICAL device, so under set_per_process_memory_fraction
    Q/K/V never came back; the reserve is now a fraction of the capacity the process may use."""
    import stream_cqsa
    n = 262144
    q, k, v = _qkv(n)
    torch.cuda.synchronize(); torch.cuda.empty_cache()
    total = torch.cuda.get_device_properties(0).total_memory
    cap = torch.cuda.memory_reserved() + int(1.5 * GIB)
    try:
        torch.cuda.set_per_process_memory_fraction(cap / total)
        # routing is incidental here (monolithic when the evicted inputs fit back under the cap, nothing pinned):
        # the subject is the restore; the SDPA check below only guards the delivered output, it is no decomposition test
        out, info = stream_cqsa.attention(q, k, v, is_causal=True, return_info=True, evict_inputs=True)
    finally:
        torch.cuda.set_per_process_memory_fraction(1.0)
    assert info["inputs_evicted"] and info["inputs_restored"] and q.is_cuda and k.is_cuda and v.is_cuda
    assert out.is_cuda                           # the output goes to the caller's device, even on the monolithic path
    assert _rel(out, F.scaled_dot_product_attention(q, k, v, is_causal=True)) < 2e-3
