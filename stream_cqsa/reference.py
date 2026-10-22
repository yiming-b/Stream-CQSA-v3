"""
Stable Python reference for Stream-CQSA exact attention.

This module is the semantic ground truth for the CUDA/streaming production path.
It is deliberately simple and slow. Nothing here is meant to be fast; everything
here is meant to be obviously correct.

It provides four layers:

1. CQS decomposition        -- chunk layout, quorum gather, path state
2. CQS mask                 -- per-token int64 group bits, dense keep mask
3. Stable local statistics  -- (local_acc, local_l, local_m) per subproblem
4. Stable recomposition     -- max-shifted merge back to global coordinates

Numerical contract
------------------
A subproblem returns shifted statistics::

    local_m[q]   = max retained score for q
    local_l[q]   = sum_k exp(score[q,k] - local_m[q])
    local_acc[q] = sum_k exp(score[q,k] - local_m[q]) * V[k]

so that the unshifted contribution is ``exp(local_m) * local_acc`` over
``exp(local_m) * local_l``. The merge never forms ``exp(local_m)`` directly,
which is what keeps it safe when scores are large.

A row with no retained pairs has ``local_m = -inf`` and contributes nothing.
"""

from __future__ import annotations

from itertools import product
from typing import Iterable, Sequence

import numpy as np
import torch

__all__ = [
    "C",
    "INTEREST_SET",
    "QUORUM_SETS",
    "is_perfect_difference_set",
    "check_quorum_set",
    "chunk_layout",
    "level_align",
    "quorum_chunks",
    "build_path_state",
    "all_paths",
    "group_bits_for_path",
    "dense_keep_mask_for_path",
    "check_pair_coverage",
    "pair_coverage_counts",
    "local_cqsa_reference",
    "merge_one_contribution",
    "merge_contribution_scatter",
    "finalize",
    "stream_cqsa_forward_reference",
    "sdpa_reference",
    "stream_cqsa_backward_reference",
]

C = 7
INTEREST_SET = (0, 1, 3)

# Perfect (Singer) difference sets: c = l^2 - l + 1 chunks, |interest_set| = l.
# A subproblem gathers l of the c chunks; the pair work per depth is (l^2/c)^itr
# of the monolithic call, the masked fraction inside a subproblem is (l-1)/l^2,
# the gathered tokens per depth are l^itr * N and there are c^itr tasks.
# The engine's int64 group bits allow itr*(l-1) <= 63. (``autoconfig`` re-exports
# this table; it lives here so the task builders can validate against it
# without importing the planner.)
QUORUM_SETS: dict[int, tuple[int, ...]] = {
    3: (0, 1), 7: (0, 1, 3), 13: (0, 1, 3, 9), 21: (0, 1, 4, 14, 16),
    31: (0, 1, 3, 8, 12, 18), 57: (0, 1, 3, 13, 32, 36, 43, 52),
    73: (0, 1, 3, 7, 15, 31, 36, 54, 63),
    # Singer set for q = 9 (from the trace of GF(q^3) over GF(q)); verified perfect
    91: (0, 20, 28, 37, 60, 70, 73, 84, 85, 89),
    # q = 11 (l = 12); verified perfect
    133: (0, 1, 3, 12, 20, 34, 38, 81, 88, 94, 104, 109),
}


def is_perfect_difference_set(c: int, s: Sequence[int]) -> bool:
    """``(c, s)`` is a perfect difference set: the members are distinct mod ``c`` and
    the differences of distinct members cover every non-zero residue mod ``c``
    exactly once -- so every pair of chunks is gathered by exactly one owner."""
    c = int(c)
    s = [int(x) % c for x in s]
    if c < 2 or len(set(s)) != len(s):
        return False
    d = sorted((a - b) % c for i, a in enumerate(s) for j, b in enumerate(s) if i != j)
    return d == list(range(1, c))


def check_quorum_set(c: int, interest_set: Sequence[int]) -> None:
    """
    Refuse a ``(c, interest_set)`` pair that is not a perfect difference set.

    The decomposition is exact only when every pair of chunks ``(i, j)`` lands in
    exactly one subproblem; with an interest set that does not match ``c`` the
    differences it covers are a subset of the residues, so most chunk pairs are
    never gathered and the engine returns a result with most of the attention
    SKIPPED -- silently, fast, and with a plausible-looking output. That is what
    v2.2.3's ``attention(..., c=73)`` did: it pinned ``c`` and kept the plan's
    ``(0, 1, 3)`` (docs/ultimate/progress/BUG_v223_c73.md). The engine entry
    points therefore validate the pair before any task is built.
    """
    if is_perfect_difference_set(c, interest_set):
        return
    c = int(c)
    iset = tuple(int(x) for x in interest_set)
    covered = {(a - b) % c for i, a in enumerate(iset) for j, b in enumerate(iset) if i != j}
    missing = (c - 1) - len(covered & set(range(1, c)))
    known = QUORUM_SETS.get(c)
    hint = (f"the quorum set for c={c} is interest_set={known}" if known is not None
            else f"the known quorum sets are c in {sorted(QUORUM_SETS)} (stream_cqsa.autoconfig.QUORUM_SETS)")
    raise ValueError(
        f"Stream-CQSA: (c={c}, interest_set={iset}) is not a perfect difference set: {missing} of the {c - 1} "
        f"non-zero chunk differences are never gathered, so the decomposition would skip those chunk pairs and "
        f"return a wrong result. Pass a matching pair: {hint}."
    )


# ---------------------------------------------------------------------------
# 1. CQS decomposition
# ---------------------------------------------------------------------------


def chunk_layout(
    n_tokens: int, c: int = C, align: int = 1
) -> tuple[list[int], list[int], list[int]]:
    """
    Return chunk sizes, starts, and ends.

    ``align = 1`` (default): the first ``remainder`` chunks get +1 token.

    ``align = A > 1``: every chunk except the last is a multiple of ``A`` tokens,
    and chunk ``c-1`` absorbs the ragged tail. This is what makes *segmented*
    input possible: a gathered subsequence is then a concatenation of runs whose
    local offsets are all multiples of ``A``, so a kernel tile of ``A`` rows
    never straddles a run boundary and the gather collapses to a per-tile base
    pointer offset.

    Exactness does not depend on chunks being equal. The coverage argument is
    combinatorial on chunk *indices* -- each ordered off-diagonal chunk pair is
    covered exactly once by the difference set, whatever the chunk sizes -- and
    the default layout already produces unequal chunks whenever ``c`` does not
    divide ``n_tokens``. Verified for the aligned layout by
    ``tests/test_reference.py::test_aligned_chunks_are_exact``.

    The tail goes to the *last* chunk on purpose: under ``sorted_gather`` the
    gathered chunks are concatenated in ascending id, so a ragged chunk c-1 is
    always last in the local sequence and cannot misalign anything after it.

    Below ``c * align`` tokens the aligned layout is *degenerate*: there are
    fewer whole blocks than chunks, so some chunks are empty and the last one
    holds the tail (everything, below one block). It is still exact, but a
    refinement of such a sequence can return the parent itself. See
    ``level_align`` for the rule the decomposition uses to avoid that.
    """
    n_tokens, c, align = int(n_tokens), int(c), int(align)
    if align <= 1:
        q, r = divmod(n_tokens, c)
        sizes = [q] * c
        for i in range(r):
            sizes[i] += 1
    else:
        n_blocks, tail = divmod(n_tokens, align)
        qb, rb = divmod(n_blocks, c)
        sizes = [qb * align] * c
        for i in range(rb):
            sizes[i] += align
        sizes[c - 1] += tail

    starts = [0] * c
    for i in range(1, c):
        starts[i] = starts[i - 1] + sizes[i - 1]
    ends = [starts[i] + sizes[i] for i in range(c)]
    return sizes, starts, ends


def level_align(n_tokens: int, c: int = C, align: int = 1, *, strict: bool = True) -> int:
    """
    The alignment one decomposition level of ``n_tokens`` local tokens is built
    with.

    A level can give every one of its ``c`` chunks a whole ``align``-token block
    only when it holds at least ``c * align`` tokens. ``strict=True`` applies
    ``align`` regardless (empty chunks below that; the caller guarantees a
    block map for every task and stops refining before this point).
    ``strict=False`` falls back to the even split for such a level: its
    chunks are then never empty and its children always shrink, at the price
    of the level -- and every task under it -- no longer being block-contiguous,
    so those tasks carry no block map and are always gathered. Below
    ``c * align`` tokens a subproblem is a few kernel tiles, so nothing is lost.
    """
    align = int(align)
    if align <= 1 or strict or int(n_tokens) >= int(c) * align:
        return align
    return 1


def quorum_chunks(owner: int, c: int = C, interest_set: Sequence[int] = INTEREST_SET) -> list[int]:
    """Chunks gathered by subsequence ``owner``: ``(owner + off) mod c``."""
    return [int((int(owner) + int(off)) % int(c)) for off in interest_set]


def build_path_state(
    N: int,
    path: Sequence[int],
    c: int = C,
    interest_set: Sequence[int] = INTEREST_SET,
    *,
    sorted_gather: bool = False,
    align: int = 1,
    strict_align: bool = True,
):
    """
    Walk a decomposition path and return the resulting local subsequence.

    ``align`` is applied afresh at every level, on the LOCAL sequence of that
    level (``chunk_layout(cur_len, c, align)``): with ``align > 1`` every level's
    chunks except the last are whole blocks, so each level concatenates runs
    that start on block boundaries and, by induction, every full block of the
    final subsequence is one contiguous global run. Children of an aligned
    task are therefore aligned by construction and never padded.
    ``strict_align`` is ``level_align``'s ``strict``: whether a level too short
    for whole-block chunks keeps the aligned layout or falls back to the even
    split. ``trace[level]["align"]`` records what each level used.

    ``sorted_gather`` controls the order in which the quorum chunks are
    concatenated:

    * ``False`` -- interest-set order ``(i+0, i+1, i+3) mod c``. For owners
      4..6 this wraps, so the local order is *not* global order.
    * ``True``  -- ascending chunk id. Each level then concatenates contiguous
      runs of an already-ascending sequence, so ``token_ids`` comes out strictly
      increasing at every level. The gathered chunk *set* per level is
      identical, so the retained pair set is still a partition (see
      ``tests/test_reference.py::test_sorted_gather_is_exact``), but the local
      permutation differs, and at ``itr >= 2`` the level-2 chunk boundaries fall
      on different tokens, so the leaf problems are genuinely different.

    The payoff of ``sorted_gather=True`` is that a *local* lower-triangular mask
    becomes exactly the *global* causal mask, which lets the stock
    FlashAttention causal path be reused with no kernel change.

    Returns
    -------
    token_ids : np.ndarray[int64], shape [L]
        Global token id of each local position, in local order.
    label_history : list[np.ndarray[int16]]
        ``label_history[level][local_pos]`` is the chunk id that local position
        belonged to at that decomposition level, reindexed into final local
        coordinates.
    trace : list[dict]
        Per-level bookkeeping (owner, gathered chunks, local ranges).
    """
    token_ids = np.arange(int(N), dtype=np.int64)
    label_history: list[np.ndarray] = []
    trace: list[dict] = []

    for owner in path:
        cur_len = int(token_ids.shape[0])
        lvl_align = level_align(cur_len, c, align, strict=strict_align)
        _, starts, ends = chunk_layout(cur_len, c, lvl_align)
        chunks = quorum_chunks(int(owner), c, interest_set)
        if sorted_gather:
            chunks = sorted(chunks)

        labels_cur = np.empty((cur_len,), dtype=np.int16)
        for chunk_id in range(int(c)):
            s, e = starts[chunk_id], ends[chunk_id]
            if e > s:
                labels_cur[s:e] = chunk_id

        gather_segments = []
        local_ranges: dict[int, tuple[int, int]] = {}
        offset = 0
        for chunk_id in chunks:
            s, e = starts[chunk_id], ends[chunk_id]
            seg = np.arange(s, e, dtype=np.int64)
            gather_segments.append(seg)
            local_ranges[int(chunk_id)] = (int(offset), int(offset + len(seg)))
            offset += len(seg)

        gather_idx = (
            np.concatenate(gather_segments) if gather_segments else np.zeros((0,), dtype=np.int64)
        )

        # Reindex previously assigned labels through the new gather.
        for t in range(len(label_history)):
            label_history[t] = label_history[t][gather_idx]
        label_history.append(labels_cur[gather_idx])
        token_ids = token_ids[gather_idx]

        trace.append(
            {
                "owner": int(owner),
                "chunks": [int(x) for x in chunks],
                "local_ranges": local_ranges,
                "align": int(lvl_align),
            }
        )

    return token_ids, label_history, trace


def all_paths(itr: int, c: int = C) -> list[tuple[int, ...]]:
    """All ``c ** itr`` leaf paths under uniform decomposition."""
    return list(product(range(int(c)), repeat=int(itr)))


# ---------------------------------------------------------------------------
# 2. CQS mask
# ---------------------------------------------------------------------------


def group_bits_for_path(
    N: int,
    path: Sequence[int],
    c: int = C,
    interest_set: Sequence[int] = INTEREST_SET,
    *,
    sorted_gather: bool = False,
    align: int = 1,
    strict_align: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build the per-token int64 CQS group bitset for one path.

    A local pair ``(row, col)`` is CQS-kept iff ``bits[row] & bits[col] == 0``.

    Each decomposition level contributes one bit per *non-owner* gathered chunk.
    The owner chunk keeps its diagonal block; every other gathered chunk has its
    diagonal block masked, because that block is owned by a different
    subsequence.
    """
    token_ids, label_history, trace = build_path_state(
        N, path, c, interest_set, sorted_gather=sorted_gather, align=align,
        strict_align=strict_align,
    )
    local_size = int(token_ids.shape[0])
    bits = np.zeros((local_size,), dtype=np.int64)

    bit_id = 0
    for level, owner in enumerate(path):
        labels = label_history[level]
        chunks = trace[level]["chunks"]
        for chunk_id in chunks:
            if int(chunk_id) == int(owner):
                continue
            idx = np.nonzero(labels == int(chunk_id))[0]
            if idx.size == 0:
                continue
            if bit_id >= 63:
                raise ValueError(
                    "group_bits int64 encoding supports at most 63 masked groups"
                )
            bits[idx] |= np.int64(1) << np.int64(bit_id)
            bit_id += 1

    return token_ids, bits


def dense_keep_mask_for_path(
    N: int,
    path: Sequence[int],
    *,
    causal: bool,
    c: int = C,
    interest_set: Sequence[int] = INTEREST_SET,
    sorted_gather: bool = False,
    align: int = 1,
    strict_align: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Dense ``[L, L]`` boolean keep mask for one path.

    The causal comparison uses *global* token ids, not local gathered positions.
    In a gathered subsequence chunk 3 can sit immediately after chunk 1, and
    those tokens are not adjacent in the original sequence.
    """
    token_ids, bits = group_bits_for_path(
        N, path, c, interest_set, sorted_gather=sorted_gather, align=align,
        strict_align=strict_align,
    )
    b = torch.from_numpy(bits)
    keep = torch.bitwise_and(b[:, None], b[None, :]).eq(0).numpy()
    if causal:
        ids = token_ids
        keep = keep & (ids[None, :] <= ids[:, None])
    return token_ids, keep


# ---------------------------------------------------------------------------
# Pair coverage
# ---------------------------------------------------------------------------


def pair_coverage_counts(
    N: int,
    itr: int,
    *,
    causal: bool,
    c: int = C,
    interest_set: Sequence[int] = INTEREST_SET,
    sorted_gather: bool = False,
    align: int = 1,
    strict_align: bool = True,
) -> np.ndarray:
    """
    Count how many times each global pair ``(q, k)`` is retained across all leaf
    subproblems. Exactness requires this to equal the target mask exactly.
    """
    counts = np.zeros((int(N), int(N)), dtype=np.int32)
    for path in all_paths(itr, c):
        token_ids, keep = dense_keep_mask_for_path(
            N, path, causal=causal, c=c, interest_set=interest_set,
            sorted_gather=sorted_gather, align=align, strict_align=strict_align,
        )
        rows, cols = np.nonzero(keep)
        np.add.at(counts, (token_ids[rows], token_ids[cols]), 1)
    return counts


def check_pair_coverage(
    N: int,
    itr: int,
    *,
    causal: bool,
    c: int = C,
    sorted_gather: bool = False,
    interest_set: Sequence[int] = INTEREST_SET,
    align: int = 1,
    strict_align: bool = True,
) -> None:
    """Assert every target pair is retained exactly once."""
    counts = pair_coverage_counts(
        N, itr, causal=causal, c=c, sorted_gather=sorted_gather,
        interest_set=interest_set, align=align, strict_align=strict_align,
    )

    target = np.ones((int(N), int(N)), dtype=np.int32)
    if causal:
        q = np.arange(int(N))[:, None]
        k = np.arange(int(N))[None, :]
        target = (k <= q).astype(np.int32)

    if not np.array_equal(counts, target):
        raise AssertionError(
            {
                "N": int(N),
                "itr": int(itr),
                "causal": bool(causal),
                "max_count": int(counts.max()),
                "missing": int(((target == 1) & (counts == 0)).sum()),
                "duplicates": int((counts > 1).sum()),
                "extra": int(((target == 0) & (counts != 0)).sum()),
            }
        )


# ---------------------------------------------------------------------------
# 3. Stable local statistics
# ---------------------------------------------------------------------------


def local_cqsa_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    token_ids: torch.Tensor,
    group_bits: torch.Tensor,
    *,
    causal: bool,
    scale: float,
):
    """
    Dense local reference attention under the CQS mask (+ optional causal mask).

    q/k/v      : [B, H, L, D]
    token_ids  : [L] global token ids
    group_bits : [L] int64 CQS group bits

    Returns ``(local_acc, local_l, local_m)`` with the shifted convention
    documented at module level. Rows with no retained pair get
    ``local_m = -inf``, ``local_l = 0``, ``local_acc = 0``.
    """
    B, H, L, D = q.shape
    if L == 0:
        # A gathered subsequence can be empty when N < c, or at deep itr.
        return (
            torch.zeros((B, H, 0, D), device=q.device, dtype=torch.float32),
            torch.zeros((B, H, 0), device=q.device, dtype=torch.float32),
            torch.zeros((B, H, 0), device=q.device, dtype=torch.float32),
        )

    scores = torch.einsum("bhld,bhmd->bhlm", q.float(), k.float()) * float(scale)

    bits = group_bits.to(device=scores.device, dtype=torch.long)
    keep = torch.bitwise_and(bits[:, None], bits[None, :]).eq(0)
    if causal:
        ids = token_ids.to(device=scores.device, dtype=torch.long)
        keep = keep & (ids[None, :] <= ids[:, None])

    keep4 = keep[None, None, :, :]
    scores = scores.masked_fill(~keep4, float("-inf"))

    local_m = scores.max(dim=-1).values           # -inf for empty rows
    finite = torch.isfinite(local_m)
    safe_m = torch.where(finite, local_m, torch.zeros_like(local_m))

    p_shift = torch.exp(scores - safe_m[..., None])
    p_shift = torch.where(keep4, p_shift, torch.zeros_like(p_shift))

    local_l = p_shift.sum(dim=-1)
    local_acc = torch.einsum("bhlm,bhmd->bhld", p_shift, v.float())

    local_m = torch.where(finite, local_m, torch.full_like(local_m, float("-inf")))
    return local_acc, local_l, local_m


# ---------------------------------------------------------------------------
# 4. Stable recomposition
# ---------------------------------------------------------------------------


def merge_one_contribution(
    global_acc: torch.Tensor,
    global_l: torch.Tensor,
    global_m: torch.Tensor,
    idx: torch.Tensor,
    local_acc: torch.Tensor,
    local_l: torch.Tensor,
    local_m: torch.Tensor,
) -> None:
    """
    Vectorised max-shifted streaming merge of one subproblem, in place.

    global_acc : [B, H, N, D]   shifted accumulator
    global_l   : [B, H, N]
    global_m   : [B, H, N]
    idx        : [L] global token ids for this subproblem

    Semantically identical to the per-token loop, but done as one scatter so it
    stays usable at benchmark sizes. Because every local position maps to a
    distinct global token within a single subproblem, ``index_copy_`` is safe.
    """
    old_m = global_m.index_select(2, idx)                     # [B, H, L]
    new_m = torch.maximum(old_m, local_m)

    old_scale = torch.where(
        torch.isfinite(old_m), torch.exp(old_m - new_m), torch.zeros_like(old_m)
    )
    new_scale = torch.where(
        torch.isfinite(local_m), torch.exp(local_m - new_m), torch.zeros_like(local_m)
    )
    # A token untouched so far and empty here keeps new_m = -inf; both scales are
    # already zero, so the accumulator stays zero.

    acc = global_acc.index_select(2, idx) * old_scale[..., None] + local_acc * new_scale[..., None]
    lsum = global_l.index_select(2, idx) * old_scale + local_l * new_scale

    global_acc.index_copy_(2, idx, acc)
    global_l.index_copy_(2, idx, lsum)
    global_m.index_copy_(2, idx, new_m)


def merge_contribution_scatter(
    global_acc: torch.Tensor,
    global_l: torch.Tensor,
    global_m_final: torch.Tensor,
    idx: torch.Tensor,
    local_acc: torch.Tensor,
    local_l: torch.Tensor,
    local_m: torch.Tensor,
) -> None:
    """
    Two-pass merge, accumulation half.

    ``global_m_final`` must already hold the final per-token max over all
    subproblems. Then this is a pure ``index_add_``, which is order-independent
    and therefore reproducible regardless of scheduling order.
    """
    ref_m = global_m_final.index_select(2, idx)
    scale = torch.where(
        torch.isfinite(local_m), torch.exp(local_m - ref_m), torch.zeros_like(local_m)
    )
    global_acc.index_add_(2, idx, local_acc * scale[..., None])
    global_l.index_add_(2, idx, local_l * scale)


def finalize(global_acc: torch.Tensor, global_l: torch.Tensor) -> torch.Tensor:
    """Normalize the shifted accumulator into the final attention output."""
    return global_acc / global_l.clamp_min(1e-20).unsqueeze(-1)


# ---------------------------------------------------------------------------
# Full forward reference
# ---------------------------------------------------------------------------


def stream_cqsa_forward_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    itr: int,
    causal: bool,
    scale: float | None = None,
    c: int = C,
    interest_set: Sequence[int] = INTEREST_SET,
    two_pass: bool = False,
    sorted_gather: bool = False,
) -> torch.Tensor:
    """
    Small-N Stream-CQSA forward reference. q/k/v: [B, H, N, D] -> O: [B, H, N, D].

    ``two_pass=True`` uses the order-independent merge (max pass, then
    accumulate pass), which is the design the production scheduler should follow
    because it does not depend on subproblem completion order.
    """
    B, H, N, D = q.shape
    device = q.device
    if scale is None:
        scale = float(D) ** -0.5

    global_acc = torch.zeros((B, H, N, D), device=device, dtype=torch.float32)
    global_l = torch.zeros((B, H, N), device=device, dtype=torch.float32)
    global_m = torch.full((B, H, N), float("-inf"), device=device, dtype=torch.float32)

    paths = all_paths(itr, c)

    def _local(path):
        token_ids_np, group_bits_np = group_bits_for_path(
            N, path, c, interest_set, sorted_gather=sorted_gather
        )
        token_ids = torch.as_tensor(token_ids_np, device=device, dtype=torch.long)
        group_bits = torch.as_tensor(group_bits_np, device=device, dtype=torch.long)
        q_i = q.index_select(dim=2, index=token_ids)
        k_i = k.index_select(dim=2, index=token_ids)
        v_i = v.index_select(dim=2, index=token_ids)
        local_acc, local_l, local_m = local_cqsa_reference(
            q_i, k_i, v_i, token_ids, group_bits, causal=causal, scale=float(scale)
        )
        return token_ids, local_acc, local_l, local_m

    if not two_pass:
        for path in paths:
            token_ids, local_acc, local_l, local_m = _local(path)
            merge_one_contribution(
                global_acc, global_l, global_m, token_ids, local_acc, local_l, local_m
            )
        return finalize(global_acc, global_l)

    # Two-pass: cache locals, reduce the max first, then accumulate.
    cached = [_local(path) for path in paths]
    for token_ids, _, _, local_m in cached:
        cur = global_m.index_select(2, token_ids)
        global_m.index_copy_(2, token_ids, torch.maximum(cur, local_m))
    for token_ids, local_acc, local_l, local_m in cached:
        merge_contribution_scatter(
            global_acc, global_l, global_m, token_ids, local_acc, local_l, local_m
        )
    return finalize(global_acc, global_l)


def sdpa_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool,
    scale: float | None = None,
) -> torch.Tensor:
    """Monolithic float32 target: what Stream-CQSA must reproduce. [B,H,N,D]."""
    D = q.shape[-1]
    if scale is None:
        scale = float(D) ** -0.5
    return torch.nn.functional.scaled_dot_product_attention(
        q.float(), k.float(), v.float(), is_causal=bool(causal), scale=float(scale)
    )


# ---------------------------------------------------------------------------
# Segmented input
# ---------------------------------------------------------------------------


def segment_block_base(
    N: int,
    path: Sequence[int],
    *,
    c: int = C,
    interest_set: Sequence[int] = INTEREST_SET,
    align: int = 128,
    strict_align: bool = True,
) -> np.ndarray | None:
    """
    Map each aligned block of the LOCAL subsequence to its global start row.

    ``global_row(r) = base[r // align] + (r % align)``, which is exact because a
    block never straddles a run boundary once chunks are ``align``-aligned (see
    ``chunk_layout``). This is what lets the kernel read the original Q/K/V in
    place instead of consuming a gathered copy.

    Returns ``None`` when the path holds a level that fell back to the even
    split (``strict_align=False`` below ``c * align`` tokens): such a
    subsequence is not block-contiguous and has no block map.
    """
    token_ids, _, trace = build_path_state(
        N, path, c, interest_set, sorted_gather=True, align=align,
        strict_align=strict_align,
    )
    if any(int(t["align"]) != int(align) for t in trace):
        return None
    L = int(token_ids.shape[0])
    n_blocks = (L + align - 1) // align
    base = token_ids[: n_blocks * align : align]
    return np.ascontiguousarray(base.astype(np.int32, copy=False))


# ---------------------------------------------------------------------------
# Backward
# ---------------------------------------------------------------------------


def stream_cqsa_backward_reference(
    q, k, v, dout, *, itr: int, causal: bool, scale: float | None = None,
    c: int = C, interest_set: Sequence[int] = INTEREST_SET,
    sorted_gather: bool = True,
):
    """
    Stream-CQSA exact attention backward. q/k/v/dout: [B, H, N, D].

    Formulated so that **no unshifted quantity is ever formed**. The shipped
    backward consumes ``dNum = dO/Den`` and ``dDen = -sum(dO*Num)/Den^2``, which
    inherits exactly the ``exp(lse)`` overflow the forward had: above
    ``lse ~= 88.7`` (fp32) ``Den`` is inf and the gradients silently become zero
    or NaN.

    Instead, give every subproblem the **global** log-sum-exp. Then

        P[q,k] = exp(s_qk - lse_global[q])  <= 1   for all q, k

    is the true global attention weight, and the standard softmax backward
    applies unchanged::

        D[q]     = sum_d dO[q,d] * O[q,d]          (global, computed once)
        dV[k]   += sum_q P[q,k] * dO[q]
        dP[q,k]  = dO[q] . V[k]
        dS[q,k]  = P[q,k] * (dP[q,k] - D[q])
        dQ[q]   += scale * sum_k dS[q,k] * K[k]
        dK[k]   += scale * sum_q dS[q,k] * Q[q]

    Every one of these is a *sum* over the retained pairs, and the CQS
    decomposition partitions those pairs exactly once, so summing the
    per-subproblem contributions is exact. Nothing exceeds 1 in magnitude
    relative to the softmax, so the whole computation is overflow-free by
    construction rather than by clamping.

    Returns ``(dq, dk, dv)``.
    """
    B, H, N, D = q.shape
    device = q.device
    if scale is None:
        scale = float(D) ** -0.5

    # Global forward statistics. lse is finite for any input magnitude; it is
    # exp(lse) that is not, and it is never formed.
    out = stream_cqsa_forward_reference(
        q, k, v, itr=itr, causal=causal, scale=scale, c=c,
        interest_set=interest_set, sorted_gather=sorted_gather,
    )
    scores_full = torch.einsum("bhnd,bhmd->bhnm", q.float(), k.float()) * scale
    if causal:
        ids = torch.arange(N, device=device)
        scores_full = scores_full.masked_fill(
            (ids[None, :] > ids[:, None])[None, None], float("-inf")
        )
    lse = torch.logsumexp(scores_full, dim=-1)          # [B, H, N]
    Dvec = (dout.float() * out).sum(dim=-1)             # [B, H, N]

    dq = torch.zeros_like(q, dtype=torch.float32)
    dk = torch.zeros_like(k, dtype=torch.float32)
    dv = torch.zeros_like(v, dtype=torch.float32)

    for path in all_paths(itr, c):
        ids_np, bits_np = group_bits_for_path(
            N, path, c, interest_set, sorted_gather=sorted_gather
        )
        idx = torch.as_tensor(ids_np, device=device, dtype=torch.long)
        bits = torch.as_tensor(bits_np, device=device, dtype=torch.long)
        if idx.numel() == 0:
            continue

        q_i = q.index_select(2, idx).float()
        k_i = k.index_select(2, idx).float()
        v_i = v.index_select(2, idx).float()
        do_i = dout.index_select(2, idx).float()
        lse_i = lse.index_select(2, idx)                 # GLOBAL lse, gathered
        D_i = Dvec.index_select(2, idx)                  # GLOBAL D, gathered

        keep = torch.bitwise_and(bits[:, None], bits[None, :]).eq(0)
        if causal:
            keep = keep & (idx[None, :] <= idx[:, None])

        s = torch.einsum("bhld,bhmd->bhlm", q_i, k_i) * scale
        p = torch.exp(s - lse_i[..., None])              # <= 1 by construction
        p = torch.where(keep[None, None], p, torch.zeros_like(p))

        dv_i = torch.einsum("bhlm,bhld->bhmd", p, do_i)
        dp = torch.einsum("bhld,bhmd->bhlm", do_i, v_i)
        ds = p * (dp - D_i[..., None])
        dq_i = torch.einsum("bhlm,bhmd->bhld", ds, k_i) * scale
        dk_i = torch.einsum("bhlm,bhld->bhmd", ds, q_i) * scale

        dq.index_add_(2, idx, dq_i)
        dk.index_add_(2, idx, dk_i)
        dv.index_add_(2, idx, dv_i)

    return dq, dk, dv
