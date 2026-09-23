"""CPU-only checks of the M5 telemetry harness's pure parts: the arm table, the task table the
driver shards (built with the driver's own layout rule), and the imbalance figure."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "benchmarks", "ultimate"))
import dist_telemetry as dt  # noqa: E402


def test_arms_shared_only_for_aligned_itr1():
    assert dt.arms_for(1, [0, 128]) == [(0, False), (128, False), (128, True)]
    assert dt.arms_for(2, [0, 128]) == [(0, False), (128, False)]
    assert dt.arms_for(1, [0]) == [(0, False)]
    assert dt.layout_name(0, False) == "classic_align1"
    assert dt.layout_name(128, True) == "aligned_128_shared"


@pytest.mark.parametrize("itr", [1, 2])
def test_task_table_matches_engine_and_covers_every_token(itr):
    N = 131072
    _, un, a_un = dt.task_table(N, itr, 0)
    _, al, a_al = dt.task_table(N, itr, 128)
    _, sh, a_sh = dt.task_table(N, itr, 128, shared_chunks=True)
    # The driver's layout rule: 0 = legacy, gather path = adaptive default, in-place = strict.
    assert (a_un, a_al, a_sh) == (0, None, 128)
    assert len(un) == len(al) == len(sh) == 7 ** itr
    assert [t["path_idx"] for t in un] == [t["path_idx"] for t in al] == list(range(7 ** itr))
    # Same tokens, only the chunk boundaries move (align_census: sumL/un = 1.0000 at every depth).
    assert sum(t["L"] for t in un) == sum(t["L"] for t in al) == sum(t["L"] for t in sh)
    # Aligned tasks are whole 128-blocks except the ones holding the ragged tail.
    assert sum(t["L"] % 128 != 0 for t in al) <= 3 ** itr
    assert sum(t["L"] % 128 != 0 for t in sh) <= 3 ** itr


@pytest.mark.parametrize("seg_align,shared", [(0, False), (128, False), (128, True)])
def test_shard_is_the_drivers_partition(seg_align, shared):
    from stream_cqsa.distributed import _shard
    tasks, table, _ = dt.task_table(131072, 2, seg_align, shared)
    shards = [dt.shard(tasks, table, r, 4) for r in range(4)]
    assert sorted(t["path_idx"] for s in shards for t in s) == list(range(49))
    assert [len(s) for s in shards] == [13, 12, 12, 12]
    for r, s in enumerate(shards):
        assert [t["path_idx"] for t in s] == _shard(tasks, r, 4)


def test_imbalance():
    assert dt.imbalance([1.0, 1.0, 1.0, 1.0]) == 1.0
    assert dt.imbalance([2.0, 1.0, 1.0]) == pytest.approx(1.5)
