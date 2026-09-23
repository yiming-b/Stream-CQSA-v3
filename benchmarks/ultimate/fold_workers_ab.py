"""
A/B of the wave engine's host-accumulator fold workers (I0, P0.3 review MAJOR): does a second fold
worker help? Forward, acc=CPU, host-resident (streamed) Q/K/V, fp16 B=1 H=8 D=64 causal, N in {1M, 2M},
at the three shapes the review named: c=7 itr=1 (the P0.3 A/B's shape, 2 big waves), c=73 itr=1 (many
16K-token subproblems per wave: the fold per wave is large against the kernel per wave) and c=7 itr=2.

Arms (all the same wave_forward call; only the fold differs):
  wave_sync      CQSA_WAVE_MERGE_ASYNC=0: the fold runs on the main thread, the GPU idles through it
  wave_w1        async, one worker (the committed default), CQSA_CPU_THREADS threads
  wave_w2        async, two workers, each owning half of the token ids (bit-identical by construction)
  wave_w1_2t     async, one worker with twice the threads: separates "a second worker" from "more cores"
  classic        the classic pinned pipeline (n_par=2, shared chunks at itr=1): the planner's alternative
Rows carry the fold seconds (summed over workers) and the main thread's seconds blocked on a slot
(merge_fold_s / merge_wait_s): wait ~ 0 means the fold is hidden and a second worker cannot help.

Every (N, shape, arm) runs in its own subprocess, 1 warm-up + --reps timed runs, arms interleaved per
shape; the parent first checks at N=131072 that every wave arm is bit-identical to wave_sync and the
classic engine agrees to rel 1e-5. GPU name in every row. No FA-2 arm: this A/B compares host folds.

    python -P benchmarks/ultimate/fold_workers_ab.py run --out results/ultimate/fold_workers_ab.json
"""
import argparse, gc, json, os, resource, socket, subprocess, sys, time

os.environ.setdefault("CQSA_CUDA_MODULE", "cqsa_cuda_next_v11")
os.environ.setdefault("CQSA_CUDA_MODULE_NONCAUSAL", "cqsa_cuda_next_v9")
os.environ.setdefault("CQSA_CPU_THREADS", "8")

import torch

B, H, D = 1, 8, 64
SHAPES = ((7, 1), (73, 1), (7, 2))
ARMS = ("wave_sync", "wave_w1", "wave_w2", "wave_w1_2t", "classic")
NOTE = "host-fold A/B, no FA-2 arm; every wave arm is the same kernel work, only the host fold differs"


def gpu_info():
    info = dict(gpu=torch.cuda.get_device_name(0), torch=torch.__version__, node=socket.gethostname(),
                slurm_job=os.environ.get("SLURM_JOB_ID"), cpu_threads=int(os.environ["CQSA_CPU_THREADS"]),
                cpus=len(os.sched_getaffinity(0)), code_root=os.path.dirname(os.path.dirname(os.path.dirname(
                    os.path.abspath(__file__)))))
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name,clocks.sm,clocks.max.sm,memory.total", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=30)
        info["nvidia_smi"] = r.stdout.strip()
    except Exception as e:  # pragma: no cover
        info["nvidia_smi"] = f"unavailable: {e!r}"
    return info


def inputs(N, seed=0):
    """Host-resident, token-major (the layout both engines stream without a per-call copy)."""
    g = torch.Generator().manual_seed(seed)
    q, k, v = (torch.randn(B, N, H, D, generator=g).to(torch.float16).transpose(1, 2) for _ in range(3))
    return q, k, v


def make_runner(arm, c, itr):
    from stream_cqsa import native_wave
    from stream_cqsa.autoconfig import QUORUM_SETS
    iset = QUORUM_SETS[c]
    if arm == "classic":
        from stream_cqsa.stable_stream import stream_cqsa_forward
        # the planner's host-streamed acc=CPU configuration (Plan.engine_kwargs), n_par=2
        return lambda q, k, v: stream_cqsa_forward(q, k, v, causal=True, itr=itr, c=c, interest_set=iset, allow_escalation=False,
                                                   max_parallel=2, low_memory=True, accumulate_on_gpu=False,
                                                   stream_from_host=True, shared_chunks=(itr == 1))
    os.environ["CQSA_WAVE_MERGE_ASYNC"] = "0" if arm == "wave_sync" else "1"
    os.environ["CQSA_WAVE_FOLD_WORKERS"] = "2" if arm == "wave_w2" else "1"
    if arm == "wave_w1_2t":
        native_wave.CPU_THREADS = 2 * int(os.environ["CQSA_CPU_THREADS"])      # read by _fold_host at call time
    return lambda q, k, v: native_wave.wave_forward(q, k, v, causal=True, itr=itr, c=c, interest_set=iset,
                                                    accumulate_on_gpu=False, kernel="cuda")


def worker(cfg):
    run = make_runner(cfg["arm"], cfg["c"], cfg["itr"])
    q, k, v = inputs(cfg["N"])
    times, peaks, folds, waits, info = [], [], [], [], {}
    for i in range(cfg["warmup"] + cfg["reps"]):
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
        t0 = time.perf_counter()
        out, info = run(q, k, v)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        peaks.append(torch.cuda.max_memory_allocated())
        if i >= cfg["warmup"]:
            times.append(dt); folds.append(info.get("merge_fold_s")); waits.append(info.get("merge_wait_s"))
        info.pop("lse", None); del out
    return dict(cfg, times_s=times, mean_s=sum(times) / len(times), min_s=min(times), merge_fold_s=folds, merge_wait_s=waits,
                peak_gib=max(peaks) / 2**30, host_maxrss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
                n_waves=info.get("n_waves"), wave_sizes=info.get("wave_sizes"), fold_workers=info.get("fold_workers"),
                n_parallel=info.get("n_parallel"), **gpu_info(), note=NOTE)


def check_arms(N=131072):
    """One process: every wave arm bit-identical to the synchronous fold; classic within fp32 summation order."""
    from stream_cqsa import native_wave
    q, k, v = inputs(N)
    res = {}
    for c, itr in SHAPES:
        outs = {}
        for arm in ARMS:
            native_wave.CPU_THREADS = int(os.environ["CQSA_CPU_THREADS"])
            out, _ = make_runner(arm, c, itr)(q, k, v)
            torch.cuda.synchronize()
            outs[arm] = out.float().cpu()
        rel = ((outs["classic"].double() - outs["wave_sync"].double()).norm() / outs["wave_sync"].double().norm()).item()
        res[f"c{c}_itr{itr}"] = dict(wave_arms_bit_identical=all(torch.equal(outs[a], outs["wave_sync"]) for a in ARMS[1:4]),
                                     rel_classic_vs_wave=rel)
        assert res[f"c{c}_itr{itr}"]["wave_arms_bit_identical"] and rel < 1e-5, res
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["run", "worker"])
    ap.add_argument("cfg", nargs="?")
    ap.add_argument("--out", required=False, default="results/ultimate/fold_workers_ab.json")
    ap.add_argument("--N", type=int, nargs="+", default=[1 << 20, 1 << 21])
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--deadline-min", type=float, default=52.0, help="start no new row after this many minutes")
    a = ap.parse_args()
    if a.mode == "worker":
        print("RESULT " + json.dumps(worker(json.loads(a.cfg))), flush=True)
        return
    t_start = time.time()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    print(gpu_info(), flush=True)
    chk = check_arms()
    print("arm check:", chk, flush=True)
    doc = dict(info=gpu_info(), config=dict(B=B, H=H, D=D, dtype="fp16", causal=True, acc="cpu", residency="host",
                                            reps=a.reps, warmup=a.warmup, shapes=SHAPES, arms=ARMS),
               arm_check=chk, note=NOTE, rows=[])
    for N in a.N:                                  # 1M first, so a timeout keeps the smaller rows
        for c, itr in SHAPES:
            for arm in ARMS:
                if (time.time() - t_start) / 60 > a.deadline_min:
                    print(f"deadline: skipping N={N} c={c} itr={itr} {arm}", flush=True)
                    continue
                cfg = dict(arm=arm, N=N, c=c, itr=itr, reps=a.reps, warmup=a.warmup)
                proc = subprocess.run([sys.executable, "-P", os.path.abspath(__file__), "worker", json.dumps(cfg)],
                                      stdout=subprocess.PIPE, text=True)
                line = next((ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")), None)
                r = json.loads(line[7:]) if line else dict(cfg, status="error", rc=proc.returncode, stdout=proc.stdout[-4000:])
                doc["rows"].append(r)
                json.dump(doc, open(a.out, "w"), indent=1)
                if "times_s" in r:
                    fold = [round(x, 2) for x in r["merge_fold_s"] if x is not None]
                    wait = [round(x, 2) for x in r["merge_wait_s"] if x is not None]
                    print(f"N={N:8d} c={c:3d} itr={itr} {arm:11s}: {r['mean_s']:7.2f} s (runs {[round(t, 2) for t in r['times_s']]})"
                          f"  fold {fold} wait {wait}  peak {r['peak_gib']:.2f} GiB  host {r['host_maxrss_gib']:.1f} GiB"
                          f"  waves={r['n_waves']}", flush=True)
                else:
                    print(f"{cfg}: ERROR rc={r.get('rc')}\n{r.get('stdout')}", flush=True)
    print("\nsummary (mean s; ratios vs wave_w1):")
    for N in a.N:
        for c, itr in SHAPES:
            m = {r["arm"]: r.get("mean_s") for r in doc["rows"] if r["N"] == N and r["c"] == c and r["itr"] == itr}
            if m.get("wave_w1"):
                print(f"N={N:8d} c={c:3d} itr={itr}: " + "  ".join(
                    f"{x} {m[x]:.2f} ({m[x] / m['wave_w1']:.3f})" for x in ARMS if m.get(x)), flush=True)


if __name__ == "__main__":
    main()
