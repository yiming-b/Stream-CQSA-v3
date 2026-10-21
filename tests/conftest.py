"""
Shared test configuration, and THE RULE for correctness comparisons.

Decomposition must be forced AND asserted. ``stream_cqsa.attention`` runs the monolithic torch SDPA call
whenever the planner says the call fits in device memory and nothing is pinned -- ``kernel=`` alone,
``evict_inputs=True`` alone and a ``hardware=`` budget that still holds the monolithic call do NOT
decompose. A "Stream-CQSA vs SDPA / flash_attn / fp64" comparison on that path is SDPA against itself and
proves nothing about the decomposition. Every correctness comparison therefore (a) forces a decomposition:
pin ``itr>=1`` (or ``c=``), give a ``hardware=`` budget the monolithic call cannot fit, or call an engine
entry point directly (``stream_cqsa_forward`` / ``wave_forward`` / ``stream_cqsa_attn`` with an int ``itr``;
``itr="auto"`` and ``itr=0`` can take the classic engine's own monolithic flash branch); and (b) asserts
that it happened, with ``tests/_decomp.py``: ``attention_decomposed(q, k, v, itr=1, ...)`` returns
``(out, info)`` and fails on ``engine == "monolithic"`` / ``n_subproblems <= 1``; ``assert_decomposed(info)``
does the same for an info dict in hand (engine entry points included). A test whose subject IS the
monolithic path says so in a comment next to the call and asserts it with ``assert_monolithic(info)``.
The audit that introduced the rule: docs/ultimate/progress/AUDIT_decomposition.md.
"""
import os
import sys

import torch

# `from _decomp import ...` works in every pytest import mode (tests/ is not a package).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# The reference path runs many very small CPU tensor ops. On a Della node torch
# defaults to one intra-op thread per core (80 here), and the OpenMP fork/join
# overhead dominates by ~400x on these shapes. Pin to a single thread so the
# reference suite runs in seconds instead of minutes.
torch.set_num_threads(1)
