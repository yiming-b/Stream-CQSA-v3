"""v3: grouped-query / multi-query attention (K/V with fewer heads than Q) on the default modes.

``attention(q, k, v)`` accepts ``k``/``v`` with ``H_kv`` heads dividing ``q``'s ``H`` (query head ``h`` attends K/V
head ``h // (H // H_kv)``, FlashAttention's convention, SDPA's ``enable_gqa=True``). The CUDA kernels handle it
natively on every default route: the monolithic call, the classic engine (device-resident and host-streamed,
chunk pool included, forward and backward, device or host accumulator) and the wave engine (gathered, in place,
chunk pool; the wave backward). The Triton kernels reject it with a message. The memory model counts K/V and
dK/dV with ``H_kv`` heads.

Every comparison pins ``itr=1`` and asserts the decomposition (tests/conftest.py). The references are torch
SDPA with ``enable_gqa=True`` (the flash backend) and the same Stream-CQSA call on K/V repeated to ``H`` heads:
the GQA path reads the same numbers through the head ratio, so the two agree far inside the kernels' error.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from stream_cqsa.autoconfig import hardware_from_dict, plan
from stream_cqsa.stable_stream import estimate_monolithic_bytes, estimate_peak_bytes, estimate_peak_bytes_bwd

CUDA = torch.cuda.is_available()
GIB = 2 ** 30
H, D = 8, 64
N = 32768


def _rel(a, b):
    a, b = a.float().cpu(), b.float().cpu()
    return ((a - b).norm() / b.norm()).item()


def _native_ok() -> bool:
    try:
        from stream_cqsa.native_wave import native_available, native_supports
        return native_available() and native_supports(torch.float16, D)
    except Exception:          # noqa: BLE001
        return False


def _inputs(Hk, *, device="cuda", grad=False, seed=0):
    torch.manual_seed(seed)
    q = torch.randn(1, H, N, D, device=device, dtype=torch.float16, requires_grad=grad)
    k = torch.randn(1, Hk, N, D, device=device, dtype=torch.float16, requires_grad=grad)
    v = torch.randn(1, Hk, N, D, device=device, dtype=torch.float16, requires_grad=grad)
    return q, k, v


def _expand(t, Hk):
    """K/V repeated to H heads in FlashAttention's grouping (head h <- h // (H // Hk))."""
    return t.detach().repeat_interleave(H // Hk, dim=1)


def _sdpa_ref(q, k, v, do=None):
    """torch SDPA with enable_gqa=True (the flash backend); gradients too when ``do`` is given."""
    qs, ks, vs = (t.detach().clone().requires_grad_(do is not None) for t in (q, k, v))
    out = F.scaled_dot_product_attention(qs, ks, vs, is_causal=True, enable_gqa=True)
    if do is None:
        return out, None
    out.backward(do)
    return out, [t.grad for t in (qs, ks, vs)]


# ---------------------------------------------------------------------------------------------------------------
# shapes and the memory model (no GPU)
# ---------------------------------------------------------------------------------------------------------------
def test_attention_accepts_gqa_shapes_and_rejects_the_rest():
    import stream_cqsa
    hw = {"cuda:0": "80GiB", "host": "64GiB"}
    q = torch.randn(1, 8, 4096, 64, dtype=torch.float16)
    for Hk in (8, 4, 2, 1):
        k = torch.randn(1, Hk, 4096, 64, dtype=torch.float16)
        p = stream_cqsa.attention(q, k, k.clone(), is_causal=True, plan_only=True, hardware=hw)
        assert p.mode == "mono"
        p = stream_cqsa.attention(q, k, k.clone(), is_causal=True, plan_only=True, hardware=hw, enable_gqa=True)
        assert p.mode == "mono"
    with pytest.raises(ValueError, match="divide"):
        k = torch.randn(1, 3, 4096, 64, dtype=torch.float16)
        stream_cqsa.attention(q, k, k.clone(), is_causal=True, plan_only=True, hardware=hw)
    with pytest.raises(ValueError, match="same shape"):
        stream_cqsa.attention(q, torch.randn(1, 4, 4096, 64, dtype=torch.float16),
                              torch.randn(1, 2, 4096, 64, dtype=torch.float16), is_causal=True, plan_only=True, hardware=hw)


def test_memory_model_counts_kv_with_their_own_heads():
    M = 1 << 20
    kw = dict(B=1, H=32, D=128, itemsize=2)
    mono = [estimate_monolithic_bytes(M, Hk=Hk, **kw) for Hk in (32, 8, 1)]
    assert mono[0] > mono[1] > mono[2]
    assert mono[1] == pytest.approx(((32 + 2 * 8 + 32) * M * 128 * 2 + M * 32 * 4) * 1.15, rel=1e-6)   # Q + K + V + O + lse
    fwd = [estimate_peak_bytes(M, 1, Hk=Hk, **kw) for Hk in (32, 8)]
    bwd = [estimate_peak_bytes_bwd(M, 1, Hk=Hk, **kw) for Hk in (32, 8)]
    assert fwd[1] < fwd[0] and bwd[1] < bwd[0]
    # host-streamed: K/V are not on the device, so the forward floor no longer depends on Hk
    assert (estimate_peak_bytes(M, 1, Hk=8, stream_from_host=True, **kw)
            < estimate_peak_bytes(M, 1, Hk=32, stream_from_host=True, **kw))   # only the transient shrinks
    hw = hardware_from_dict({"cuda:0": "80GiB", "host": "200GiB"})
    for direction in ("fwd", "bwd"):
        ps = [plan(N=M, B=1, H=32, D=128, dtype=torch.float16, causal=True, hardware=hw, direction=direction,
                   Hk=Hk, allow_distributed=False) for Hk in (32, 8)]
        m = [[c for c in p.candidates if c["mode"] == "mono"][0]["peak"] for p in ps]
        assert m[1] < m[0]


@pytest.mark.skipif(not CUDA, reason="needs a GPU")
def test_triton_kernels_reject_gqa():
    import stream_cqsa
    q, k, v = _inputs(2)
    with pytest.raises(NotImplementedError, match="grouped-query"):
        stream_cqsa.attention(q, k, v, is_causal=True, kernel="triton", itr=1)


# ---------------------------------------------------------------------------------------------------------------
# the classic engine (CUDA kernel): device-resident, host-streamed, chunk pool; forward and backward
# ---------------------------------------------------------------------------------------------------------------
@pytest.mark.skipif(not CUDA, reason="needs a GPU")
@pytest.mark.parametrize("Hk", [2, 1])
def test_classic_forward_gqa_matches_sdpa_and_the_expanded_call(Hk):
    from tests._decomp import attention_decomposed
    q, k, v = _inputs(Hk)
    out, info = attention_decomposed(q, k, v, is_causal=True, kernel="cuda", itr=1, engine="classic")
    assert out.shape == q.shape and out.dtype == q.dtype and out.is_cuda
    ref, _ = _sdpa_ref(q, k, v)
    assert _rel(out, ref) < 2e-3
    exp, _ = attention_decomposed(q, _expand(k, Hk), _expand(v, Hk), is_causal=True, kernel="cuda", itr=1, engine="classic")
    assert _rel(out, exp) < 1e-4                      # the same numbers through the head ratio


@pytest.mark.skipif(not CUDA, reason="needs a GPU")
@pytest.mark.parametrize("acc_gpu", [True, False])
def test_classic_backward_gqa_device_resident(acc_gpu):
    """dK/dV come back with H_kv heads, summed over each group of query heads, on the device accumulator and on
    the host one (the device-resident path's pinned staging and host add)."""
    from tests._decomp import attention_decomposed
    Hk = 2
    q, k, v = _inputs(Hk, grad=True)
    out, info = attention_decomposed(q, k, v, is_causal=True, kernel="cuda", itr=1, bwd_itr=1,
                                     accumulate_on_gpu=acc_gpu, engine="classic")
    do = torch.randn_like(out)
    out.backward(do)
    got = [t.grad.clone() for t in (q, k, v)]
    assert got[1].shape == k.shape and got[2].shape == v.shape and all(g.is_cuda for g in got)
    ref, rg = _sdpa_ref(q, k, v, do)
    assert _rel(out, ref) < 2e-3
    for g, r in zip(got, rg):
        assert _rel(g, r) < 5e-3


@pytest.mark.skipif(not CUDA, reason="needs a GPU")
@pytest.mark.parametrize("shared_chunks", [False, True])
def test_classic_host_streamed_forward_gqa(shared_chunks):
    """Host-resident Q/K/V streamed per subproblem (pinned staging per tensor) and through the chunk pool (slots
    with K/V's own head count); host accumulator."""
    from stream_cqsa.stable_stream import stream_cqsa_forward
    from tests._decomp import assert_decomposed
    Hk = 2
    q, k, v = _inputs(Hk, device="cpu")
    out32, info = stream_cqsa_forward(q, k, v, itr=1, causal=True, stream_from_host=True, accumulate_on_gpu=False,
                                      shared_chunks=shared_chunks, out_device="acc")
    assert_decomposed(info, n_subproblems=7)
    assert out32.device.type == "cpu" and tuple(out32.shape) == tuple(q.shape)
    if shared_chunks:
        assert info["chunk_hits"] > 0 and info["chunk_transfers"] < info["chunk_loads_requested"]   # the pool ran
    ref, _ = _sdpa_ref(q.cuda(), k.cuda(), v.cuda())
    assert _rel(out32, ref) < 2e-3


@pytest.mark.skipif(not CUDA, reason="needs a GPU")
def test_classic_host_streamed_backward_gqa():
    """attention() on host tensors (training): the host-streamed classic backward with the gradients
    accumulated in host memory; dK/dV with H_kv heads."""
    from tests._decomp import attention_decomposed
    Hk = 2
    q, k, v = _inputs(Hk, device="cpu", grad=True)
    out, info = attention_decomposed(q, k, v, is_causal=True, kernel="cuda", itr=1, bwd_itr=1, engine="classic")
    assert info["stream_from_host"] is True
    do = torch.randn_like(out)
    out.backward(do)
    got = [t.grad.clone() for t in (q, k, v)]
    assert got[1].shape == k.shape and got[2].shape == v.shape
    ref, rg = _sdpa_ref(q.cuda(), k.cuda(), v.cuda(), do.cuda())
    assert _rel(out, ref) < 2e-3
    for g, r in zip(got, rg):
        assert _rel(g, r) < 5e-3


# ---------------------------------------------------------------------------------------------------------------
# the wave engine (CUDA kernel): gathered, in place, chunk pool; the wave backward
# ---------------------------------------------------------------------------------------------------------------
@pytest.mark.skipif(not (CUDA and _native_ok()), reason="needs a GPU and the native wave extension")
@pytest.mark.parametrize("gather", [True, False])
@pytest.mark.parametrize("Hk", [2, 1])
def test_wave_forward_gqa_device_resident(Hk, gather):
    from tests._decomp import attention_decomposed
    q, k, v = _inputs(Hk)
    out, info = attention_decomposed(q, k, v, is_causal=True, kernel="wave-cuda", itr=1, gather=gather, engine="wave")
    assert info["gather"] is gather
    ref, _ = _sdpa_ref(q, k, v)
    assert _rel(out, ref) < 2e-3
    exp, _ = attention_decomposed(q, _expand(k, Hk), _expand(v, Hk), is_causal=True, kernel="wave-cuda", itr=1,
                                  gather=gather, engine="wave")
    assert _rel(out, exp) < 1e-4


@pytest.mark.skipif(not (CUDA and _native_ok()), reason="needs a GPU and the native wave extension")
def test_wave_forward_gqa_host_resident_chunk_pool():
    from stream_cqsa.native_wave import wave_forward
    from tests._decomp import assert_decomposed
    Hk = 2
    q, k, v = _inputs(Hk, device="cpu")
    out32, info = wave_forward(q, k, v, causal=True, itr=1, c=7, interest_set=(0, 1, 3), accumulate_on_gpu=False,
                               kernel="cuda", pool_slots=3)
    assert_decomposed(info, n_subproblems=7)
    assert out32.device.type == "cpu"
    ref, _ = _sdpa_ref(q.cuda(), k.cuda(), v.cuda())
    assert _rel(out32, ref) < 2e-3


@pytest.mark.skipif(not (CUDA and _native_ok()), reason="needs a GPU and the native wave extension")
def test_wave_backward_gqa():
    """The wave backward forms dK/dV per query head and sums each group inside the extension."""
    from tests._decomp import attention_decomposed
    Hk = 2
    q, k, v = _inputs(Hk, grad=True)
    out, info = attention_decomposed(q, k, v, is_causal=True, kernel="wave-cuda", itr=1, bwd_itr=1, engine="wave")
    do = torch.randn_like(out)
    out.backward(do)
    got = [t.grad.clone() for t in (q, k, v)]
    assert got[1].shape == k.shape and got[2].shape == v.shape
    assert info["backward"]["engine"] == "wave", info["backward"]
    ref, rg = _sdpa_ref(q, k, v, do)
    assert _rel(out, ref) < 2e-3
    for g, r in zip(got, rg):
        assert _rel(g, r) < 5e-3


@pytest.mark.skipif(not CUDA, reason="needs a GPU")
def test_patched_sdpa_routes_gqa_calls(monkeypatch):
    """F.scaled_dot_product_attention(..., enable_gqa=True) goes through stream_cqsa.attention once patched."""
    import stream_cqsa.api as api
    calls = []
    monkeypatch.setattr(api, "attention", lambda *a, **kw: calls.append(kw) or a[0])
    monkeypatch.setattr(api, "_PATCH_MIN_TOKENS", 1024)
    q, k, v = _inputs(2)
    api._patched_sdpa(q, k, v, is_causal=True, enable_gqa=True)
    assert calls and calls[0]["enable_gqa"] is True
    api._patched_sdpa(q, k, v, is_causal=True)                       # heads differ, no flag: routed as well
    assert len(calls) == 2
