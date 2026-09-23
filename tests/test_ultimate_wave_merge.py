"""
Wave engine, host accumulator (accumulate_on_gpu=False): the asynchronous fold (CQSA_WAVE_MERGE_ASYNC=1,
the default) must be bit-identical to the synchronous one, bound its in-flight waves, and surface a
fold exception on the caller's thread. Sized for a shared GPU: N <= 131072, well under 6 GiB.
"""
import threading
import pytest, torch, torch.nn.functional as F

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
H, D = 8, 64
WAVE_TOKENS = 1 << 17          # ~0.3 GiB of fp32 partials per wave; forces several waves at 131K


def _native():
    from stream_cqsa.native_wave import native_available, native_supports
    if not native_available():
        pytest.skip("cqsa_native not built")
    return native_supports


def _inputs(N, dtype, host, seed=0):
    g = torch.Generator().manual_seed(seed)
    q, k, v = (torch.randn(1, H, N, D, generator=g).to(dtype) for _ in range(3))
    return (q, k, v) if host else tuple(t.cuda() for t in (q, k, v))


def _run(monkeypatch, async_, src, **kw):
    from stream_cqsa.native_wave import wave_forward
    monkeypatch.setenv("CQSA_WAVE_MERGE_ASYNC", "1" if async_ else "0")
    out, info = wave_forward(*src, accumulate_on_gpu=False, max_wave_tokens=WAVE_TOKENS, **kw)
    torch.cuda.synchronize()
    assert info["merge_async"] is async_
    assert not out.is_cuda and not info["lse"].is_cuda
    return out, info


def _rel(a, b):
    return ((a.double() - b.double()).norm() / b.double().norm()).item()


@cuda
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("c,itr", [(7, 1), (7, 2), (73, 1), (73, 2)])
def test_async_fold_bit_identical_host_streamed(monkeypatch, c, itr, causal, dtype):
    """Host-resident Q/K/V streamed through the chunk pool: async == sync exactly, out and lse."""
    from stream_cqsa.autoconfig import QUORUM_SETS
    if not _native()(dtype, D):
        pytest.skip(f"cqsa_native build lacks {dtype} head_dim={D}")
    src = _inputs(131072, dtype, host=True)
    kw = dict(causal=causal, itr=itr, c=c, interest_set=QUORUM_SETS[c])
    o_sync, i_sync = _run(monkeypatch, False, src, **kw)
    o_async, i_async = _run(monkeypatch, True, src, **kw)
    assert i_sync["n_waves"] == i_async["n_waves"] >= 2
    assert torch.equal(o_sync, o_async)
    assert torch.equal(i_sync["lse"], i_async["lse"])
    assert torch.isfinite(o_async).all()


@cuda
@pytest.mark.parametrize("causal", [True, False])
def test_async_fold_bit_identical_device_resident(monkeypatch, causal):
    """Device-resident Q/K/V (read in place) with the host accumulator: the same fold, the same result."""
    from stream_cqsa.autoconfig import QUORUM_SETS
    if not _native()(torch.float16, D):
        pytest.skip("cqsa_native build lacks fp16 head_dim=64")
    src = _inputs(131072, torch.float16, host=False)
    kw = dict(causal=causal, itr=1, c=7, interest_set=QUORUM_SETS[7])
    o_sync, i_sync = _run(monkeypatch, False, src, **kw)
    o_async, i_async = _run(monkeypatch, True, src, **kw)
    assert torch.equal(o_sync, o_async) and torch.equal(i_sync["lse"], i_async["lse"])


@cuda
def test_many_waves_two_slots(monkeypatch):
    """>= 4 waves through the two pinned slots (every slot reused several times), still exact."""
    from stream_cqsa.autoconfig import QUORUM_SETS
    from stream_cqsa.native_wave import wave_forward
    if not _native()(torch.float16, D):
        pytest.skip("cqsa_native build lacks fp16 head_dim=64")
    src = _inputs(131072, torch.float16, host=True)
    kw = dict(causal=True, itr=2, c=7, interest_set=QUORUM_SETS[7])
    o_sync, i_sync = _run(monkeypatch, False, src, **kw)
    o_async, i_async = _run(monkeypatch, True, src, **kw)
    assert i_async["n_waves"] >= 4
    assert torch.equal(o_sync, o_async) and torch.equal(i_sync["lse"], i_async["lse"])
    # and the acc=GPU path agrees up to fp32 summation order
    o_gpu, _ = wave_forward(*(t.cuda() for t in src), accumulate_on_gpu=True, max_wave_tokens=WAVE_TOKENS, **kw)
    assert _rel(o_async, o_gpu.cpu()) < 1e-5


@cuda
@pytest.mark.parametrize("causal", [True, False])
def test_async_fold_accuracy_vs_fp64(monkeypatch, causal):
    """The async host-accumulator output is within 1.2x FA-2's fp64 error (small N, dense reference)."""
    from stream_cqsa.autoconfig import QUORUM_SETS
    if not _native()(torch.float16, D):
        pytest.skip("cqsa_native build lacks fp16 head_dim=64")
    q, k, v = _inputs(8192, torch.float16, host=True)
    out, _ = _run(monkeypatch, True, (q, k, v), causal=causal, itr=1, c=7, interest_set=QUORUM_SETS[7])
    qd, kd, vd = (t.cuda() for t in (q, k, v))
    ref = F.scaled_dot_product_attention(qd.double(), kd.double(), vd.double(), is_causal=causal).cpu()
    fa2 = F.scaled_dot_product_attention(qd, kd, vd, is_causal=causal).cpu()
    assert _rel(out, ref) <= 1.2 * _rel(fa2, ref) + 1e-6


@cuda
def test_fold_exception_surfaces(monkeypatch):
    """An exception inside the worker's fold is re-raised on the caller's thread (no hang, no leaked worker)."""
    from stream_cqsa import native_wave
    from stream_cqsa.autoconfig import QUORUM_SETS
    if not _native()(torch.float16, D):
        pytest.skip("cqsa_native build lacks fp16 head_dim=64")
    src = _inputs(131072, torch.float16, host=True)

    def boom(*args, **kwargs):
        raise RuntimeError("fold boom")

    monkeypatch.setattr(native_wave, "_fold_host", boom)
    monkeypatch.setenv("CQSA_WAVE_MERGE_ASYNC", "1")
    n_threads = threading.active_count()
    box = {}

    def call():
        try:
            native_wave.wave_forward(*src, causal=True, itr=1, c=7, interest_set=QUORUM_SETS[7],
                                     accumulate_on_gpu=False, max_wave_tokens=WAVE_TOKENS)
        except Exception as exc:  # noqa: BLE001 - the test inspects it
            box["exc"] = exc

    t = threading.Thread(target=call, daemon=True)
    t.start(); t.join(timeout=300)
    assert not t.is_alive(), "wave_forward hung after a fold exception"
    assert isinstance(box.get("exc"), RuntimeError) and "fold boom" in str(box["exc"])
    assert threading.active_count() == n_threads     # the worker pool was shut down


@cuda
@pytest.mark.parametrize("N", [131072, 100003])
@pytest.mark.parametrize("workers", [2, 3])
def test_fold_workers_split_by_token_range_are_bit_identical(monkeypatch, N, workers):
    """CQSA_WAVE_FOLD_WORKERS > 1: each worker folds a fixed token range of every wave, so each token is still
    folded in wave order and the result equals the synchronous fold exactly, at a ragged N too (the last
    chunk absorbs the remainder). The info reports the fold and wait seconds for the A/B."""
    from stream_cqsa.autoconfig import QUORUM_SETS
    if not _native()(torch.float16, D):
        pytest.skip("cqsa_native build lacks fp16 head_dim=64")
    src = _inputs(N, torch.float16, host=True)
    kw = dict(causal=True, itr=1, c=7, interest_set=QUORUM_SETS[7])
    o_sync, i_sync = _run(monkeypatch, False, src, **kw)
    monkeypatch.setenv("CQSA_WAVE_FOLD_WORKERS", str(workers))
    o_async, i_async = _run(monkeypatch, True, src, **kw)
    assert i_async["fold_workers"] == workers and i_sync["fold_workers"] == 0
    assert torch.equal(o_sync, o_async) and torch.equal(i_sync["lse"], i_async["lse"])
    assert i_async["merge_fold_s"] > 0 and i_async["merge_wait_s"] >= 0


@cuda
def test_async_fold_batch_of_two_matches_per_batch_calls(monkeypatch):
    """B=2 host-streamed: the accumulators are drained and reset per batch and the chunk pool reloads its chunks
    per batch; each batch equals its own B=1 call. (Regression, I0: the pool kept chunks resident by chunk id
    alone, so batch 1's first wave read batch 0's Q/K/V -- rel error 1.41 against SDPA.)"""
    from stream_cqsa.autoconfig import QUORUM_SETS
    if not _native()(torch.float16, D):
        pytest.skip("cqsa_native build lacks fp16 head_dim=64")
    g = torch.Generator().manual_seed(3)
    q, k, v = (torch.randn(2, H, 65536, D, generator=g).half() for _ in range(3))
    kw = dict(causal=True, itr=1, c=7, interest_set=QUORUM_SETS[7])
    out, _ = _run(monkeypatch, True, (q, k, v), **kw)
    for b in range(2):
        ob, _ = _run(monkeypatch, True, tuple(t[b:b + 1].contiguous() for t in (q, k, v)), **kw)
        assert torch.equal(out[b:b + 1], ob)


@cuda
def test_async_fold_on_the_triton_kernel(monkeypatch):
    """The Triton wave kernel feeds the same host fold (merge_rows_torch): async == sync exactly."""
    from stream_cqsa.autoconfig import QUORUM_SETS
    from stream_cqsa.native_wave import triton_wave_available
    if not triton_wave_available():
        pytest.skip("triton not available")
    src = _inputs(32768, torch.float16, host=True)
    kw = dict(causal=True, itr=1, c=7, interest_set=QUORUM_SETS[7], kernel="triton")
    from stream_cqsa.native_wave import wave_forward
    res = {}
    for a in (False, True):
        monkeypatch.setenv("CQSA_WAVE_MERGE_ASYNC", "1" if a else "0")
        res[a] = wave_forward(*src, accumulate_on_gpu=False, max_wave_tokens=1 << 15, **kw)
    assert res[True][1]["kernel"] == "triton" and res[True][1]["n_waves"] >= 2
    assert torch.equal(res[False][0], res[True][0]) and torch.equal(res[False][1]["lse"], res[True][1]["lse"])
