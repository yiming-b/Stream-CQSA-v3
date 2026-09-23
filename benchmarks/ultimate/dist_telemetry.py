"""
Multi-device telemetry of the distributed Stream-CQSA forward (and optionally backward): scheduler and
reduction arms, per-rank predicted vs measured times, a single-device reference in the same job.

    python -P -m torch.distributed.run --standalone --nproc_per_node=4 \\
        benchmarks/ultimate/dist_telemetry.py 1048576 2097152 --itr 1 2 --layouts prod \\
        --scheduler rr lpt --reduce allreduce reduce_scatter [--bwd] [--throttle 3:54 --rates measured]

Arms. Every (N, itr, layout) row runs the single-device reference once (rank 0; the other ranks wait),
then every arm ``scheduler x reduce`` (``--scheduler``/``--reduce``, or explicit ``--arms
sched/reduce/rates`` triples): ``rr`` is the previous round-robin shard, ``lpt`` the cost-aware
assignment of stream_cqsa/scheduler.py (live-tile work, LPT + descent + child-refinement split);
``allreduce`` returns the replicated output, ``reduce_scatter`` the sharded one (each rank its token
range; gathered to rank 0 after the timed region for the error check). ``rates``: ``seed`` = each
rank's GPU model rate (cache or seed table), ``measured`` = the per-rank ``calibrate_rate()`` taken
at startup under whatever limits the rank runs with.

Heterogeneity without a mixed node. ``--throttle RANK:SMS`` confines one rank to SMS of the GPU's SMs
through a CUDA green context (torch.cuda.green_contexts), a sleep-free rate cap: the context is made
current before anything touches CUDA streams, the process group is initialised lazily (an eager NCCL
init with device_id makes the primary context current again, measured on vis1: the cap then stops
applying), and the rank warms the engine up (which creates torch's stream pool inside the green
context) before its first collective. Measured on vis1 (shared A100-PCIe-40GB): 27 of 108 SMs ->
engine 2.7-3.0x slower, with NCCL collectives running.

Layouts. ``--layouts prod`` (default) runs the production layout per depth: aligned + shared chunk
pool at itr=1 (the host-streamed production path), aligned gather at itr>=2. ``--layouts all`` runs
the M5 set (seg_align 0 / 128 + the shared arm at itr=1, ``arms_for``).

Every row carries: per-rank GPU name, measured t_local / t_merge, the scheduler's predicted local time
(the rank table's rate), the units each rank ran (depth, path_idx, L, live tiles) and its engine calls,
the assignment digest, per-rank device peak above the arm's starting allocation and host RSS, the
imbalance max/mean of t_local, collective count/bytes, speedup vs the single-device reference and the
relative error. The meta block has ``nvidia-smi topo -m``, the rank table and the calibrations.

Inputs are generated token-major ([B,N,H,D] storage viewed as [B,H,N,D]), the layout the host-streamed
engine wants (head-major inputs cost a token-major copy per rank). The fp32 output stays where the
accumulator lives (out_device="acc"). The backward rows (``--bwd``, N <= ``--bwd-max-n``) run
device-resident with the fp32 gradients on the device (host-resident gradients at 4 ranks do not fit
the job's host memory budget), from the reference forward's output and global lse.
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


def prod_layout(itr: int) -> list[tuple[int, bool]]:
    """The production layout of one depth: aligned + shared chunk pool at itr=1, aligned gather deeper."""
    return [(128, itr == 1)]


def layout_name(seg_align: int, shared_chunks: bool) -> str:
    base = f"aligned_{seg_align}" if seg_align else "classic_align1"
    return base + ("_shared" if shared_chunks else "")


def parse_arms(arms: list[str] | None, schedulers: list[str], reduces: list[str], rates: str) -> list[dict]:
    """``sched/reduce[/rates]`` strings, or the product ``schedulers x reduces`` at ``rates``."""
    out = []
    if arms:
        for a in arms:
            parts = a.split("/")
            if len(parts) not in (2, 3):
                raise ValueError(f"arm {a!r}: expected sched/reduce[/rates]")
            out.append(dict(scheduler=parts[0], reduce=parts[1], rates=parts[2] if len(parts) == 3 else rates))
    else:
        out = [dict(scheduler=s, reduce=r, rates=rates) for s in schedulers for r in reduces]
    for a in out:
        if a["scheduler"] not in ("rr", "lpt") or a["reduce"] not in ("allreduce", "reduce_scatter") \
                or a["rates"] not in ("seed", "measured"):
            raise ValueError(f"bad arm {a}")
        a["output"] = "sharded" if a["reduce"] == "reduce_scatter" else "replicated"
        a["name"] = f"{a['scheduler']}/{a['reduce']}/{a['rates']}"
    return out


def parse_throttle(spec: str | None) -> tuple[int, int] | None:
    if not spec:
        return None
    r, s = spec.split(":")
    return int(r), int(s)


def task_table(N: int, itr: int, seg_align: int, shared_chunks: bool = False):
    """The task list the driver builds for these engine kwargs, its rows as
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
    """The rows of `table` rank `rank` runs under the round-robin shard (scheduler="rr")."""
    from stream_cqsa.distributed import _shard
    mine = set(_shard(tasks, rank, world))
    return [t for t in table if t["path_idx"] in mine]


def imbalance(local_times: list[float]) -> float:
    """max / mean of the per-rank local compute times; 1.0 is a perfectly balanced shard."""
    return max(local_times) / (sum(local_times) / len(local_times))


def measured_rates(units_per_rank: list[list], t_local: list[float]) -> list[float]:
    """Effective live-tile rate each rank achieved (sum of its units' tiles x B*H / its local time)."""
    return [(sum(u[3] for u in us) * B * H / t) if t > 0 and us else 0.0 for us, t in zip(units_per_rank, t_local)]


def rss_now_gib() -> float:
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 2**20
    except OSError:
        pass
    return float("nan")


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


def make_inputs(N: int, seed: int, n: int = 3, device="cpu"):
    """Token-major storage viewed as [B, H, N, D]: the engine streams it without a copy."""
    torch.manual_seed(seed)
    return tuple(torch.randn(B, N, H, D, dtype=torch.float16, device=device).transpose(1, 2) for _ in range(n))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("N", type=int, nargs="*", default=[1048576, 2097152])
    ap.add_argument("--itr", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--layouts", choices=("prod", "all"), default="prod")
    ap.add_argument("--seg-align", type=int, nargs="+", default=[0, 128], help="--layouts all only")
    ap.add_argument("--scheduler", nargs="+", default=["rr", "lpt"], choices=("rr", "lpt"))
    ap.add_argument("--reduce", nargs="+", default=["allreduce", "reduce_scatter"], choices=("allreduce", "reduce_scatter"))
    ap.add_argument("--rates", choices=("seed", "measured"), default="seed")
    ap.add_argument("--arms", nargs="+", default=None, help="explicit sched/reduce[/rates] arms (overrides the product)")
    ap.add_argument("--split", choices=("auto", "children", "none"), default="auto")
    ap.add_argument("--acc", choices=("cpu", "gpu"), default="gpu")
    ap.add_argument("--npar", type=int, default=1)
    ap.add_argument("--chunk-tokens", type=int, default=1 << 18)
    ap.add_argument("--throttle", default=None, help="RANK:SMS -- confine that rank to SMS SMs (green context)")
    ap.add_argument("--bwd", action="store_true", help="add backward rows (device-resident) for N <= --bwd-max-n")
    ap.add_argument("--bwd-max-n", type=int, default=1 << 20)
    ap.add_argument("--bwd-itr", type=int, nargs="+", default=[1])
    ap.add_argument("--tag", default="")
    ap.add_argument("--no-ref", action="store_true", help="skip the single-device reference on rank 0")
    ap.add_argument("--backend", choices=("nccl", "gloo"), default="nccl",
                    help="gloo: smoke tests with several ranks on one GPU (collectives through host memory)")
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args(argv)
    arms = parse_arms(args.arms, args.scheduler, args.reduce, args.rates)
    throttle = parse_throttle(args.throttle)

    def _excepthook(t, v, tb):
        # torch.distributed.run hides child tracebacks; print our own to stdout.
        print(f"[rank {os.environ.get('RANK', '?')}] UNCAUGHT {t.__name__}: {v}", flush=True)
        traceback.print_exception(t, v, tb, file=sys.stdout); sys.stdout.flush()
    sys.excepthook = _excepthook

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    grank = int(os.environ.get("RANK", 0))
    if args.backend == "gloo":
        local_rank = local_rank % torch.cuda.device_count()     # several ranks may share a GPU
    torch.cuda.set_device(local_rank)
    green = None
    n_sm = torch.cuda.get_device_properties(local_rank).multi_processor_count
    if throttle is not None and throttle[0] == grank:
        import torch.cuda.green_contexts as gc
        green = gc.GreenContext.create(int(throttle[1]), local_rank)
        green.set_context()                       # before anything creates a stream

    import torch.distributed as dist
    from stream_cqsa import scheduler as S
    from stream_cqsa.distributed import dist_stream_cqsa_forward, dist_stream_cqsa_backward
    from stream_cqsa.stable_stream import stream_cqsa_forward, stream_cqsa_backward

    # Eager NCCL init (device_id) unless a rank is throttled: the eager init makes the primary context
    # current again, and the green context would stop applying to the streams created afterwards.
    if throttle is None and args.backend == "nccl":
        dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank), timeout=timedelta(hours=2))
    else:
        dist.init_process_group(args.backend, timeout=timedelta(hours=2))
    cdev = torch.device("cuda", local_rank) if args.backend == "nccl" else torch.device("cpu")   # p2p/broadcast tensors
    rank, world = dist.get_rank(), dist.get_world_size()
    gpu = torch.cuda.get_device_name()

    def log(msg):
        print(f"[rank {rank} {time.strftime('%H:%M:%S')}] {msg}", flush=True)

    kw = dict(causal=True, stream_from_host=True, low_memory=(args.acc == "cpu"), allow_escalation=False,
              max_parallel=args.npar, out_device="acc")
    # Local warm-up BEFORE the first collective (kernel module, pinned pools, torch's stream pool --
    # inside the green context on a throttled rank), then the per-rank rate calibration.
    wq, wk, wv = make_inputs(16384, 0)
    stream_cqsa_forward(wq, wk, wv, itr=1, shared_chunks=True, **kw)
    stream_cqsa_forward(wq, wk, wv, itr=2, **kw)
    del wq, wk, wv
    t0 = time.perf_counter()
    # the rate of the path the forward rows run (host-streamed, shared chunk pool, device accumulator):
    # a device-resident calibration under a 27-of-108 SM cap fell 3.2x where the host-streamed forward
    # fell 1.7x (vis1), so the kernel rate alone would over-correct the assignment
    cal = S.calibrate_rate(reps=2, stream_from_host=True)
    seed = S.device_rate(gpu)
    log(f"device {torch.cuda.current_device()} {gpu} world={world} host={socket.gethostname()} "
        f"SMs={throttle[1] if green is not None else n_sm}/{n_sm}  calibrated rate {cal.tile_rate:.3e} tiles/s "
        f"({time.perf_counter() - t0:.1f} s; seed {seed.tile_rate:.3e}, {seed.source})")
    my_rates = dict(seed=seed.tile_rate, measured=cal.tile_rate)

    # distributed warm-up: NCCL communicator (lazy on throttled runs), both reductions
    wq, wk, wv = make_inputs(16384, 0)
    for red in ("allreduce", "reduce_scatter"):
        dist_stream_cqsa_forward(wq, wk, wv, itr=1, shared_chunks=True, reduce=red, **kw)
    if rank == 0 and not args.no_ref:
        stream_cqsa_forward(wq, wk, wv, itr=1, shared_chunks=True, **kw)
    del wq, wk, wv
    torch.cuda.synchronize(); dist.barrier()

    me = dict(rank=rank, gpu=gpu, host=socket.gethostname(), local_rank=local_rank, n_sm=n_sm,
              sms_used=throttle[1] if green is not None else n_sm, rate_seed=seed.tile_rate, rate_seed_source=seed.source,
              rate_measured=cal.tile_rate, rate_measured_source=cal.source)
    ranks_meta = [None] * world
    dist.all_gather_object(ranks_meta, me)
    gpus = [m["gpu"] for m in ranks_meta]
    meta = dict(job_id=os.environ.get("SLURM_JOB_ID"), host=socket.gethostname(), world=world, gpus=gpus,
                ranks=ranks_meta, topo=topo_snapshot() if rank == 0 else None, tag=args.tag,
                throttle=args.throttle, arms=arms, split=args.split, chunk_tokens=args.chunk_tokens,
                torch=torch.__version__, cuda_module=os.environ.get("CQSA_CUDA_MODULE"),
                engine_kw=dict(kw), B=B, H=H, D=D, c=C, interest_set=list(INTEREST_SET),
                note="FA-2 not measured here (engine-level multi-device rows; see fwd_ab/bwd_ab for kernel-level FA-2 arms)")
    rows = []

    def dump():
        if rank == 0:
            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            with open(args.out, "w") as f:
                json.dump(dict(meta=meta, rows=rows), f, indent=1)

    def gather_to_rank0(x: torch.Tensor, lo: int, hi: int, full_N: int):
        """Sharded output -> full tensor on rank 0 (point-to-point, untimed)."""
        if rank == 0:
            full = torch.empty((B, H, full_N, D), dtype=torch.float32, device=x.device)
            full[:, :, lo:hi].copy_(x)
            for r in range(1, world):
                rng = torch.empty(2, dtype=torch.int64, device=cdev)
                dist.recv(rng, src=r)
                a, b = int(rng[0]), int(rng[1])
                buf = torch.empty((B, H, b - a, D), dtype=torch.float32, device=cdev)
                dist.recv(buf, src=r)
                full[:, :, a:b].copy_(buf)
            return full
        dist.send(torch.tensor([lo, hi], dtype=torch.int64, device=cdev), dst=0)
        dist.send(x.to(cdev).contiguous(), dst=0)
        return None

    def rel(a, b):
        return ((a.float() - b.float()).norm() / b.float().norm()).item()

    def lse_err(a, b):
        f = torch.isfinite(b)
        return (a.float()[f] - b.float().to(a.device)[f]).abs().max().item()

    def run_arm(direction, arm, N, itr, layout, seg_align, shared, inputs, ref):
        """One timed arm on every rank; returns rank 0's row (None elsewhere)."""
        akw = dict(kw, seg_align=seg_align, shared_chunks=shared) if direction == "fwd" else \
            dict(causal=True, allow_escalation=False, max_parallel=args.npar, accumulate_on_gpu=True, seg_align=seg_align)
        rate = my_rates[arm["rates"]]
        release_caches()
        torch.cuda.synchronize()
        base_alloc = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        dist.barrier(); torch.cuda.synchronize()
        t0 = time.perf_counter()
        if direction == "fwd":
            q, k, v = inputs
            out_d, info = dist_stream_cqsa_forward(q, k, v, itr=itr, scheduler=arm["scheduler"], split=args.split,
                                                   reduce=arm["reduce"], output=arm["output"], rate=rate,
                                                   chunk_tokens=args.chunk_tokens, **akw)
        else:
            q, k, v, do, o_ref, lse_ref = inputs
            grads, info = dist_stream_cqsa_backward(q, k, v, do, o_ref, lse_ref, itr=itr, scheduler=arm["scheduler"],
                                                    split=args.split, reduce=arm["reduce"], rate=rate,
                                                    chunk_tokens=args.chunk_tokens, **akw)
        dist.barrier(); torch.cuda.synchronize()
        t_dist = time.perf_counter() - t0
        units = info["assignment"]["units"][rank]
        tele = dict(rank=rank, gpu=gpu, t_local_s=info["t_local_s"], t_merge_s=info["t_merge_s"],
                    t_schedule_s=info["t_schedule_s"], predicted_local_s=info["predicted_local_s"],
                    depth_calls=info["depth_calls"], units=units,
                    dev_peak_above_start_gib=(torch.cuda.max_memory_allocated() - base_alloc) / 2**30,
                    dev_start_gib=base_alloc / 2**30, host_rss_now_gib=rss_now_gib(), host_rss_peak_gib=rss_peak_gib(),
                    token_range=list(info["token_range"]), n_collectives=info["n_collectives"],
                    collective_bytes=info.get("collective_bytes"), rate_used=rate)
        per_rank = [None] * world
        dist.all_gather_object(per_rank, tele)
        # error check (untimed)
        err = {}
        if direction == "fwd":
            full = out_d
            if arm["output"] == "sharded":
                lo, hi = info["token_range"]
                full = gather_to_rank0(out_d, lo, hi, N)
            if rank == 0 and ref is not None:
                err = dict(rel_err=rel(full, ref[0]), lse_max_abs_err=lse_err(info["lse"], ref[1]))
            del full, out_d
        else:
            if rank == 0 and ref is not None:
                if arm["reduce"] == "allreduce":
                    err = dict(rel_err=max(rel(a, b) for a, b in zip(grads, ref)))
            if arm["reduce"] == "reduce_scatter" and not args.no_ref:
                # every rank checks its own range against rank 0's reference, broadcast per gradient
                lo, hi = info["token_range"]
                errs = []
                for gi in range(3):
                    refg = ref[gi].to(cdev).contiguous() if rank == 0 else \
                        torch.empty((B, H, N, D), dtype=torch.float32, device=cdev)
                    dist.broadcast(refg, src=0)
                    refg = refg[:, :, lo:hi].to(grads[gi].device)
                    errs.append([((grads[gi].float() - refg).norm() ** 2).item(), (refg.norm() ** 2).item()])
                    if rank != 0:
                        del refg
                allerr = [None] * world
                dist.all_gather_object(allerr, errs)
                if rank == 0 and ref is not None:
                    err = dict(rel_err=max((sum(e[gi][0] for e in allerr) / sum(e[gi][1] for e in allerr)) ** 0.5
                                           for gi in range(3)))
            del grads
        release_caches()
        if rank != 0:
            return None
        t_loc = [p["t_local_s"] for p in per_rank]
        pred = [p["predicted_local_s"] for p in per_rank]
        units_all = [p["units"] for p in per_rank]
        a = info["assignment"]
        return dict(direction=direction, N=N, itr=itr, world=world, tag=args.tag, throttle=args.throttle,
                    layout=layout, seg_align=seg_align, shared_chunks=shared, arm=arm["name"],
                    scheduler=arm["scheduler"], reduce=arm["reduce"], output=arm["output"], rates=arm["rates"],
                    split=info["split"], acc=args.acc, npar=args.npar, gpu=gpu, gpus=gpus,
                    t_dist_s=t_dist, t_local_per_rank=t_loc, t_merge_per_rank=[p["t_merge_s"] for p in per_rank],
                    t_schedule_s=max(p["t_schedule_s"] for p in per_rank),
                    predicted_local_per_rank=pred, predicted_over_measured=[x / y if y else None for x, y in zip(pred, t_loc)],
                    predicted_makespan_s=a["makespan_s"], predicted_ideal_s=a["ideal_s"], predicted_ratio=a["ratio"],
                    t_local_max_s=max(t_loc), t_local_mean_s=sum(t_loc) / world, imbalance=imbalance(t_loc),
                    effective_rate_per_rank=measured_rates(units_all, t_loc), rate_used_per_rank=[p["rate_used"] for p in per_rank],
                    units_per_rank=units_all, n_units_per_rank=[len(u) for u in units_all], n_splits=a["n_splits"],
                    digest=a["digest"], depth_calls_per_rank=[p["depth_calls"] for p in per_rank],
                    dev_peak_above_start_gib_per_rank=[p["dev_peak_above_start_gib"] for p in per_rank],
                    dev_start_gib_per_rank=[p["dev_start_gib"] for p in per_rank],
                    host_rss_now_gib_per_rank=[p["host_rss_now_gib"] for p in per_rank],
                    host_rss_peak_gib_per_rank=[p["host_rss_peak_gib"] for p in per_rank],
                    token_range_per_rank=[p["token_range"] for p in per_rank],
                    n_collectives=per_rank[0]["n_collectives"], collective_bytes=per_rank[0]["collective_bytes"],
                    **err)

    for itr in args.itr:
        for N in args.N:
            layouts = prod_layout(itr) if args.layouts == "prod" else arms_for(itr, args.seg_align)
            for seg_align, shared in layouts:
                layout = layout_name(seg_align, shared)
                q, k, v = make_inputs(N, 0)
                ref = None
                t_single = None
                if rank == 0 and not args.no_ref:
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    o1, i1 = stream_cqsa_forward(q, k, v, itr=itr, **dict(kw, seg_align=seg_align, shared_chunks=shared))
                    torch.cuda.synchronize()
                    t_single = time.perf_counter() - t0
                    ref = (o1, i1["lse"])
                    log(f"fwd N={N} itr={itr} {layout}: single device {t_single:.2f} s")
                dist.barrier()
                for arm in arms:
                    r = run_arm("fwd", arm, N, itr, layout, seg_align, shared, (q, k, v), ref)
                    if rank == 0:
                        if t_single is not None:
                            r.update(t_single_s=t_single, speedup=t_single / r["t_dist_s"])
                        rows.append(r)
                        dump()
                        print(f"fwd N={N:>8} itr={itr} {layout:<18} {arm['name']:<30} world={world}  "
                              + (f"single {t_single:6.2f}s  " if t_single else "")
                              + f"dist {r['t_dist_s']:6.2f}s  " + (f"speedup {r['speedup']:4.2f}x  " if t_single else "")
                              + f"local {' '.join(f'{t:.2f}' for t in r['t_local_per_rank'])}  "
                              f"pred {' '.join(f'{t:.2f}' for t in r['predicted_local_per_rank'])}  "
                              f"imb {r['imbalance']:.3f}  merge {max(r['t_merge_per_rank']):.2f}s  "
                              f"units {r['n_units_per_rank']}  peak+ {max(r['dev_peak_above_start_gib_per_rank']):.1f}GiB  "
                              + (f"rel.err {r['rel_err']:.2e} lse.err {r['lse_max_abs_err']:.1e}" if "rel_err" in r else ""),
                              flush=True)
                del q, k, v, ref
                release_caches()
                dist.barrier()

    if args.bwd:
        for itr in args.bwd_itr:
            for N in [n for n in args.N if n <= args.bwd_max_n]:
                q, k, v, do = (t.cuda() for t in make_inputs(N, 1, n=4))
                # reference forward output and global lse on every rank (the backward's inputs), then
                # rank 0's single-device backward
                o1, i1 = stream_cqsa_forward(q, k, v, itr=itr, causal=True, allow_escalation=False,
                                             max_parallel=args.npar)
                o16, lse = o1.half(), i1["lse"]
                del o1, i1
                ref = None
                t_single = None
                bkw = dict(causal=True, allow_escalation=False, max_parallel=args.npar, accumulate_on_gpu=True, seg_align=128)
                if rank == 0 and not args.no_ref:
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    ref = stream_cqsa_backward(q, k, v, do, o16, lse, itr=itr, **bkw)
                    torch.cuda.synchronize()
                    t_single = time.perf_counter() - t0
                    log(f"bwd N={N} itr={itr}: single device {t_single:.2f} s")
                dist.barrier()
                for arm in arms:
                    r = run_arm("bwd", arm, N, itr, "aligned_128_device", 128, False, (q, k, v, do, o16, lse), ref)
                    if rank == 0:
                        if t_single is not None:
                            r.update(t_single_s=t_single, speedup=t_single / r["t_dist_s"])
                        rows.append(r)
                        dump()
                        print(f"bwd N={N:>8} itr={itr} {arm['name']:<30} world={world}  "
                              + (f"single {t_single:6.2f}s  " if t_single else "")
                              + f"dist {r['t_dist_s']:6.2f}s  " + (f"speedup {r['speedup']:4.2f}x  " if t_single else "")
                              + f"local {' '.join(f'{t:.2f}' for t in r['t_local_per_rank'])}  "
                              f"pred {' '.join(f'{t:.2f}' for t in r['predicted_local_per_rank'])}  "
                              f"imb {r['imbalance']:.3f}  merge {max(r['t_merge_per_rank']):.2f}s  units {r['n_units_per_rank']}  "
                              + (f"rel.err {r['rel_err']:.2e}" if "rel_err" in r else ""), flush=True)
                del q, k, v, do, o16, lse, ref
                release_caches()
                dist.barrier()
    dump()
    if rank == 0:
        print("wrote", args.out, flush=True)
    dist.barrier()
    dist.destroy_process_group()
    if green is not None:
        # tensors freed after the green context at interpreter exit abort the process ("context is
        # destroyed"); everything is written, so leave without the teardown
        sys.stdout.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
