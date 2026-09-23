"""benchmarks/ultimate/kernel_rows.py: the synthetic patterns, the recomposer and the cross-layout invariant.

The harness imports the CQS CUDA extension at module import (through benchmarks/layout_ab.py), so everything
here needs a GPU and a built kernel; run with ``python -P -m pytest tests/test_ultimate_kernel_rows.py``."""
import os, sys

import pytest
import torch
import torch.nn.functional as F

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


@pytest.fixture(scope="module")
def kr(tmp_path_factory):
    os.environ["KERNEL_ROWS_OUT"] = str(tmp_path_factory.mktemp("kernel_rows"))   # never write into results/
    sys.path.insert(0, os.path.join(REPO, "benchmarks", "ultimate"))
    try:
        import kernel_rows
    except (ImportError, AssertionError) as e:
        pytest.skip(f"kernel_rows needs the CQS CUDA extension: {e}")
    kernel_rows.ITERS = 1                      # the timings are not under test
    return kernel_rows


def rel(a, b):
    return ((a.double() - b.double()).norm() / b.double().norm()).item()


@cuda
def test_synthetic_patterns(kr):
    L = 1000
    _, bo, ba = kr.tables_for(kr.allones_bits(L))
    c = kr.tile_census(bo, ba, L)
    assert c["masked"] == 1.0 and c["mixed"] == 0.0            # every pair masked, the diagonal included
    _, bo, ba = kr.tables_for(torch.zeros(L, dtype=torch.int64))
    assert kr.tile_census(bo, ba, L)["clear"] == 1.0
    r = kr.random_bits(L)
    assert torch.equal(r, kr.random_bits(L)) and int(r.min()) >= 0 and int(r.max()) < 8
    _, bo, ba = kr.tables_for(r)
    c = kr.tile_census(bo, ba, L)
    assert abs(c["masked"] + c["clear"] + c["mixed"] - 1.0) < 1e-12


@cuda
def test_recomposer_ignores_empty_rows(kr):
    N, D = 64, 8
    rec = kr.Recomposer(N, D)
    ids = torch.arange(N, device=kr.DEV)
    out = torch.ones(1, N, kr.H, D, device=kr.DEV)
    rec.add(ids, out, torch.zeros(1, kr.H, N, device=kr.DEV))
    before = rec.output().clone()
    for sentinel in (float("inf"), float("-inf"), float("nan")):   # FA-2 writes +inf for a row with no live key
        rec.add(ids, torch.full_like(out, sentinel), torch.full((1, kr.H, N), sentinel, device=kr.DEV))
    after = rec.output()
    assert torch.isfinite(after).all() and torch.equal(after, before)


@cuda
@pytest.mark.parametrize("itr", [1, 2])
def test_recomposed_layouts_agree_and_match_fp64(kr, itr):
    """The engine invariant on the harness's own merge: aligned and unaligned task sets recompose to the same
    output (rel <= 1e-4) and each is within 1.2x FA-2's fp64 error."""
    N, D = 4096, 64
    qt, kt, vt = kr.inputs(N, D)
    sk = dict(D=D, dtype=torch.float16, causal=True)
    outs = {}
    for layout, seg_align in (("aligned_128", kr.SEG_ALIGN), ("classic_align1", 0)):
        tasks, rows, outs[layout] = kr.layout_pass(N, itr, seg_align, layout, qt, kt, vt, causal=True, sdpa_key=sk,
                                                   fp64_paths=(), zero=False)
        assert len(rows) == 7 ** itr and all(r["real_over_sdpa"] > 0 for r in rows)
    q, k, v = (x.permute(0, 2, 1, 3) for x in (qt, kt, vt))            # [1,H,N,D] for SDPA
    ref = F.scaled_dot_product_attention(q.double(), k.double(), v.double(), is_causal=True)
    e_fa = rel(F.scaled_dot_product_attention(q, k, v, is_causal=True), ref)
    for layout, out in outs.items():
        assert rel(out, ref) <= 1.2 * e_fa + 1e-6, layout
    # two exact partitions differ by the fp16 rounding of different key groupings: ~1e-4 at N=4096, 2.5e-5 at N=25K
    assert rel(outs["aligned_128"], outs["classic_align1"]) <= max(1e-4, e_fa)
    x = kr.cross_layout(outs["aligned_128"], outs["classic_align1"], qt, kt, vt, causal=True)
    assert x["fp64_check_recomposed_aligned"]["max_abs"] < 1e-3 * x["fp64_check_recomposed_aligned"]["max_abs_ref"] + 1e-5
