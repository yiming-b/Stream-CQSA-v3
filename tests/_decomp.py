"""
Guards against vacuous correctness comparisons (see the rule in tests/conftest.py).

``stream_cqsa.attention`` runs the monolithic torch SDPA call whenever the planner says the call fits in
device memory and nothing is pinned: ``kernel=`` alone, ``evict_inputs=True`` alone and a ``hardware=``
budget that still holds the monolithic call do NOT decompose. A "Stream-CQSA vs SDPA / flash_attn / fp64"
comparison on that path compares SDPA with itself. The engine entry points have the same trap one level
down: ``stream_cqsa_forward(itr="auto")`` and ``itr=0`` take a monolithic flash branch (``info["monolithic"]``).

Every correctness comparison therefore forces a decomposition (``itr>=1`` pinned, a budget that cannot hold
the monolithic call, or an engine entry point with an int ``itr``) AND asserts, through these helpers, that
the decomposition happened -- so a routing change cannot make the test vacuous without failing it.
"""
from __future__ import annotations

DECOMPOSED_ENGINES = ("classic", "wave")


def assert_decomposed(info, *, itr=None, c=None, engine=None, n_subproblems=None):
    """Assert that ``info`` (``attention(..., return_info=True)`` or an engine entry point's info dict)
    describes a decomposed run: a classic / wave engine and more than one subproblem.

    The backward-direction info of ``attention()`` names engine / kernel / itr / c but not the subproblem
    count; there ``itr >= 1`` is the evidence. ``itr=``, ``c=``, ``engine=`` and ``n_subproblems=`` add
    exact expectations. Returns ``info`` so it can be used inline."""
    assert isinstance(info, dict), f"expected an info dict, got {type(info).__name__}"
    if "engine" in info:                                 # attention(): "monolithic" | "classic" | "wave"
        assert info["engine"] in DECOMPOSED_ENGINES, (
            f"the monolithic path ran (engine={info['engine']!r}): this comparison would be SDPA against itself. "
            f"Pin itr>=1 (or a hardware= budget the monolithic call cannot fit). info={info}")
    assert not info.get("monolithic", False), f"the engine took its monolithic flash branch (itr=0). info={info}"
    n = info.get("n_subproblems")
    if n is not None:
        assert n > 1, f"n_subproblems={n}: no decomposition ran. info={info}"
    else:
        it = info.get("itr")
        assert isinstance(it, int) and it >= 1, (
            f"no subproblem count and itr={it!r}: the decomposition cannot be confirmed (pin an int itr). info={info}")
    if itr is not None:
        assert info.get("itr") == itr, f"itr={info.get('itr')!r}, pinned {itr}. info={info}"
    if c is not None and "c" in info:
        assert info["c"] == c, f"c={info['c']!r}, pinned {c}. info={info}"
    if engine is not None:
        assert info.get("engine") == engine, f"engine={info.get('engine')!r}, expected {engine!r}. info={info}"
    if n_subproblems is not None:
        assert n == n_subproblems, f"n_subproblems={n}, expected {n_subproblems}. info={info}"
    return info


def assert_monolithic(info):
    """The converse, for the few tests whose subject IS the monolithic path (say so in a comment next to the
    call): ``attention()`` reports ``engine == "monolithic"``; the classic engine's own branch reports
    ``monolithic=True`` / ``n_subproblems == 0``."""
    assert isinstance(info, dict), f"expected an info dict, got {type(info).__name__}"
    ok = info.get("engine") == "monolithic" or info.get("monolithic") is True
    assert ok, f"a decomposition ran where the test's subject is the monolithic path. info={info}"
    return info


def attention_decomposed(q, k, v, *args, engine=None, **kw):
    """``stream_cqsa.attention(q, k, v, *args, return_info=True, **kw)`` asserted decomposed.

    Returns ``(out, info)`` (``(out, plan, info)`` with ``return_plan=True``). Pin ``itr=1`` (or ``c=``, or a
    ``hardware=`` budget the monolithic call cannot fit) in ``kw``; an int ``itr`` / ``c`` in ``kw`` is also
    checked against what ran. ``engine=`` pins the expected engine ("classic" or "wave")."""
    from stream_cqsa import attention
    kw["return_info"] = True
    res = attention(q, k, v, *args, **kw)
    info = res[-1]
    itr = kw.get("itr") if isinstance(kw.get("itr"), int) else None
    c = kw.get("c") if isinstance(kw.get("c"), int) else None
    assert_decomposed(info, itr=itr, c=c, engine=engine)
    return res
