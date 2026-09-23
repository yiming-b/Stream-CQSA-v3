"""
M5: multi-device telemetry baseline of the classic-engine distributed forward.

    python -P -m torch.distributed.run --standalone --nproc_per_node=4 \
        benchmarks/ultimate/dist_telemetry.py [N ...] --itr 1 2 --acc gpu --seg-align 0 128

The measurement is distributed/test_dist.py's (every rank runs the round-robin
distributed forward; rank 0 then runs the single-device engine on the full
problem as the reference and reports the relative error and the speedup) with
the telemetry the scheduler package needs as its baseline, per row:

  * per-rank GPU name, per-rank local compute time, per-rank merge time, the
    imbalance max/mean of the local times (what a cost-aware assignment has to
    beat: today's shard ignores that the subproblems have unequal L);
  * the tasks each rank ran with their L (path_idx, L) and their sum;
  * per-rank device peak and host RSS peak, so the next job's --mem is sized
    from data rather than copied;
  * `nvidia-smi topo -m` and the GPU names captured once in the file's meta.

Layout arms. --seg-align 0 128 runs both chunk layouts through the engine's
own ``seg_align`` kwarg (P0.1: the driver reads it from engine_kw and builds its
shard list with ``task_seg_align_for`` exactly as the engine does). seg_align=0
is the legacy unaligned layout; 128 is the default: the gather path takes the
adaptive layout (aligned wherever alignment is well defined), the shared-chunk
path the strict 128-block layout. At itr=1 a third arm, aligned +
shared_chunks (the production host-streaming path), is added; the unaligned
arm cannot use shared chunks (the pool's block map needs aligned chunks), so
at seg_align=0 the plain gather path is measured. The task table this harness
records is built with the same ``task_seg_align_for`` call as the driver's, so
the (path_idx, L) rows are the tasks each rank actually ran.

Inputs are generated token-major ([B,N,H,D] storage viewed as [B,H,N,D]), the
layout the host-streamed engine wants: head-major inputs cost a full
token-major copy per rank (3 KiB/token), which is what pushed the earlier
4-rank 2M runs to 119 GiB. The fp32 output is left where the accumulator lives
(out_device="acc"; on the device for acc=gpu), so no rank holds a 2 KiB/token
host copy of it; the earlier rows (results/dist_test_w4_itr1-2_gpu.json)
included that final D2H copy in t_single/t_dist, ~0.3 s per M tokens.

The single-device reference and every distributed arm use the same engine
kwargs; FA-2 is not measured here (the FA-2 arm of every kernel-level
measurement so far is torch SDPA's flash backend, ~5% slower than flash_attn).
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import socket
import subprocess
import sys
import time
import traceback
from datetime import timedelta

import torch

B, H, D = 1, 8, 64
C, INTEREST_SET = 7, (0, 1, 3)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_OUT = os.path.join(ROOT, "results", "ultimate", "dist_telemetry.json")


def arms_for(itr: int, seg_aligns: list[int]) -> list[tuple[int, bool]]:
    """(seg_align, shared_chunks) arms of one itr. shared_chunks is an itr=1 feature of the
    engine and needs aligned chunks, so it is added only for the aligned arm at itr=1."""
    arms = [(int(a), False) for a in seg_aligns]
    if itr == 1 and any(a for a in seg_aligns):
        arms.append((max(int(a) for a in seg_aligns), True))
    return arms


def layout_name(seg_align: int, shared_chunks: bool) -> str:
    base = f"aligned_{seg_align}" if seg_align else "classic_align1"
    return base + ("_shared" if shared_chunks else "")


def task_table(N: int, itr: int, seg_align: int, shared_chunks: bool = False):
    """The task list the driver shards for these engine kwargs, its rows as
    {path_idx, path, L}, and the builder's seg_align (None = adaptive default).
    The layout is derived exactly as the driver derives it (in-place readers =
    shared_chunks here; the host-streamed gather path never reads in place)."""
    from stream_cqsa.stable_stream import build_tasks_cached, task_seg_align_for
    task_seg_align = task_seg_align_for(int(seg_align), in_place=bool(shared_chunks))
    tasks = build_tasks_cached(N, itr, B=B, H=H, D=D, itemsize=2, sorted_gather=True, pin=False,
                               c=C, interest_set=INTEREST_SET, seg_align=task_seg_align)
    rows = [dict(path_idx=int(t.path_idx), path=list(t.path), L=int(t.local_size)) for t in tasks]
    return tasks, rows, task_seg_align


def shard(tasks, table: list[dict], rank: int, world: int) -> list[dict]:
    """The rows of `table` rank `rank` runs under the driver's round-robin shard."""
    from stream_cqsa.distributed import _shard
    mine = set(_shard(tasks, rank, world))
    return [t for t in table if t["path_idx"] in mine]


def imbalance(local_times: list[float]) -> float:
    """max / mean of the per-rank local compute times; 1.0 is a perfectly balanced shard."""
    return max(local_times) / (sum(local_times) / len(local_times))


def rss_peak_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20


def release_caches():
    """Between rows: device cache, and the pinned-host cache that otherwise keeps every
    freed staging slot (power-of-two sizes never reused by the next N) resident."""
    torch.cuda.empty_cache()
    torch._C._host_emptyCache()


def topo_snapshot() -> str:
    try:
        return subprocess.run(["nvidia-smi", "topo", "-m"], capture_output=True, text=True,
                              timeout=60).stdout
    except (OSError, subprocess.SubprocessError) as e:
        return f"nvidia-smi topo -m failed: {e}"


def make_inputs(N: int, seed: int):
    """Token-major storage viewed as [B, H, N, D]: the engine streams it without a copy."""
    torch.manual_seed(seed)
    return tuple(torch.randn(B, N, H, D, dtype=torch.float16).transpose(1, 2) for _ in range(3))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("N", type=int, nargs="*", default=[1048576, 2097152])
    ap.add_argument("--itr", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--seg-align", type=int, nargs="+", default=[0, 128])
    ap.add_argument("--acc", choices=("cpu", "gpu"), default="gpu")
    ap.add_argument("--npar", type=int, default=1)
    ap.add_argument("--no-ref", action="store_true", help="skip the single-device reference on rank 0")
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args(argv)

    import torch.distributed as dist
    from stream_cqsa.distributed import dist_stream_cqsa_forward
    from stream_cqsa.stable_stream import stream_cqsa_forward

    def _excepthook(t, v, tb):
        # torch.distributed.run hides child tracebacks; print our own to stdout.
        print(f"[rank {os.environ.get('RANK', '?')}] UNCAUGHT {t.__name__}: {v}", flush=True)
        traceback.print_exception(t, v, tb, file=sys.stdout); sys.stdout.flush()
    sys.excepthook = _excepthook

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    # Rank 0 runs the reference between collectives: minutes at 2M, beyond NCCL's default watchdog.
    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank), timeout=timedelta(hours=2))
    rank, world = dist.get_rank(), dist.get_world_size()
    gpu = torch.cuda.get_device_name()

    def log(msg):
        print(f"[rank {rank} {time.strftime('%H:%M:%S')}] {msg}", flush=True)

    log(f"device {torch.cuda.current_device()} {gpu} world={world} host={socket.gethostname()}")
    # The Track-2 recommended engine settings, identical for both arms (test_dist.py).
    kw = dict(causal=True, stream_from_host=True, low_memory=(args.acc == "cpu"), allow_escalation=False,
              max_parallel=args.npar, out_device="acc")

    # Warm-up: CUDA context, NCCL communicator, kernel module, pinned pools.
    wq, wk, wv = make_inputs(16384, 0)
    dist_stream_cqsa_forward(wq, wk, wv, itr=1, shared_chunks=True, **kw)
    if rank == 0 and not args.no_ref:
        stream_cqsa_forward(wq, wk, wv, itr=1, shared_chunks=True, **kw)
    del wq, wk, wv
    torch.cuda.synchronize(); dist.barrier()

    gpus = [None] * world
    dist.all_gather_object(gpus, gpu)
    meta = dict(job_id=os.environ.get("SLURM_JOB_ID"), host=socket.gethostname(), world=world,
                gpus=gpus, topo=topo_snapshot() if rank == 0 else None,
                torch=torch.__version__, cuda_module=os.environ.get("CQSA_CUDA_MODULE"),
                engine_kw=dict(kw), B=B, H=H, D=D,
                c=C, interest_set=list(INTEREST_SET),
                note="FA-2 not measured; every FA-2 arm so far is torch SDPA flash (~5% slower than flash_attn)")
    rows = []

    def dump():
        if rank == 0:
            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            with open(args.out, "w") as f:
                json.dump(dict(meta=meta, rows=rows), f, indent=1)

    for itr in args.itr:
        for N in args.N:
            for seg_align, shared in arms_for(itr, args.seg_align):
                layout = layout_name(seg_align, shared)
                q, k, v = make_inputs(N, 0)
                akw = dict(kw, seg_align=seg_align, shared_chunks=shared)
                tasks, table, task_seg_align = task_table(N, itr, seg_align, shared)
                mine = shard(tasks, table, rank, world)
                log(f"fwd N={N} itr={itr} {layout}: {len(mine)} tasks, sum L {sum(t['L'] for t in mine)}")
                torch.cuda.reset_peak_memory_stats()
                dist.barrier(); torch.cuda.synchronize()
                t0 = time.perf_counter()
                out_d, info = dist_stream_cqsa_forward(q, k, v, itr=itr, **akw)
                dist.barrier(); torch.cuda.synchronize()
                t_dist = time.perf_counter() - t0
                if info["tasks_mine"] != [t["path_idx"] for t in mine]:
                    raise RuntimeError(f"rank {rank}: the driver ran {info['tasks_mine']}, the table says "
                                       f"{[t['path_idx'] for t in mine]}")
                tele = dict(rank=rank, gpu=gpu, t_local_s=info["t_local_s"], t_merge_s=info["t_merge_s"],
                            tasks=[[t["path_idx"], t["L"]] for t in mine],
                            sum_L=sum(t["L"] for t in mine),
                            dev_peak_gib=torch.cuda.max_memory_allocated() / 2**30,
                            host_rss_peak_gib=rss_peak_gib())
                per_rank = [None] * world
                dist.all_gather_object(per_rank, tele)
                if rank != 0 or args.no_ref:
                    del out_d
                r = None
                if rank == 0:
                    t_loc = [p["t_local_s"] for p in per_rank]
                    r = dict(N=N, itr=itr, world=world, npar=args.npar, acc=args.acc, seg_align=seg_align,
                             shared_chunks=shared, layout=layout, gpu=gpu, gpus=gpus, out_device="acc",
                             task_seg_align=task_seg_align,
                             t_dist_s=t_dist, t_local_s=per_rank[0]["t_local_s"], t_merge_s=per_rank[0]["t_merge_s"],
                             t_local_per_rank=t_loc, t_merge_per_rank=[p["t_merge_s"] for p in per_rank],
                             t_local_max_s=max(t_loc), t_local_mean_s=sum(t_loc) / world,
                             imbalance=imbalance(t_loc),
                             tasks_per_rank=[len(p["tasks"]) for p in per_rank],
                             tasks=[p["tasks"] for p in per_rank], sum_L_per_rank=[p["sum_L"] for p in per_rank],
                             n_tasks=len(table), L_min=min(t["L"] for t in table), L_max=max(t["L"] for t in table),
                             dev_peak_gib_per_rank=[p["dev_peak_gib"] for p in per_rank],
                             host_rss_peak_gib_per_rank=[p["host_rss_peak_gib"] for p in per_rank])
                    if not args.no_ref:
                        t0 = time.perf_counter()
                        out_1, info1 = stream_cqsa_forward(q, k, v, itr=itr, **akw)
                        torch.cuda.synchronize()
                        t_single = time.perf_counter() - t0
                        rel = ((out_d.float() - out_1.float()).norm() / out_1.float().norm()).item()
                        lse_err = (info["lse"].float() - info1["lse"].float().to(info["lse"].device)).abs().max().item()
                        r.update(t_single_s=t_single, speedup=t_single / t_dist, rel_err=rel, lse_max_abs_err=lse_err,
                                 host_rss_peak_gib_rank0_after_ref=rss_peak_gib())
                        del out_1, info1, out_d
                        print(f"N={N:>8} itr={itr} {layout:<20} world={world}  single {t_single:7.2f}s  dist {t_dist:7.2f}s  "
                              f"speedup {t_single / t_dist:4.2f}x  local/rank {' '.join(f'{t:.2f}' for t in t_loc)}  "
                              f"imbalance {r['imbalance']:.3f}  merge {r['t_merge_s']:.2f}s  rel.err {rel:.2e}  "
                              f"lse.err {lse_err:.2e}  tasks/rank {r['tasks_per_rank']}", flush=True)
                    else:
                        print(f"N={N:>8} itr={itr} {layout:<20} world={world}  dist {t_dist:7.2f}s  "
                              f"local/rank {' '.join(f'{t:.2f}' for t in t_loc)}  imbalance {r['imbalance']:.3f}", flush=True)
                    rows.append(r)
                    dump()
                del q, k, v, info
                release_caches()
                dist.barrier()
    dump()
    if rank == 0:
        print("wrote", args.out, flush=True)
    dist.barrier(); dist.destroy_process_group()


if __name__ == "__main__":
    main()
