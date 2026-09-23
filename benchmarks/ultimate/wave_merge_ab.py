"""
A/B of the wave engine's host-accumulator fold (P0.3): wave synchronous fold (CQSA_WAVE_MERGE_ASYNC=0)
vs wave asynchronous fold (=1, default) vs the classic engine, forward, acc=CPU, itr=1, c=7, fp16,
B=1 H=8 D=64 causal, at N in {1M, 2M}. Host-resident Q/K/V (streamed) is the primary arm; the
device-resident rows repeat the configuration of results/profile_sweep/results.jsonl (where the wave
engine measured 9.41 s vs the classic 9.08-9.44 s at 1M and 35.4-35.8 vs 33.7-34.1 s at 2M).

Every (N, residency, arm) runs in its own subprocess (flushed GPU, released host memory), 1 warm-up +
`--reps` timed runs (wall clock around the call + synchronize), peak device memory and host max RSS
recorded, GPU name in every row. The parent first checks the three arms against each other in one
process at N=131072 (wave sync == async bit-identical; classic within rel 1e-5, fp32 order only).

There is NO FA-2 arm here. Where FA-2 ratios are quoted for this program they are torch SDPA with the
flash backend (flash_attn is not installed), about 5% slower than the flash_attn package.

    python -P benchmarks/ultimate/wave_merge_ab.py run --out results/ultimate/wave_merge_ab.json
"""
import argparse, gc, json, os, resource, socket, subprocess, sys, time

os.environ.setdefault("CQSA_CUDA_MODULE", "cqsa_cuda_next_v11")
os.environ.setdefault("CQSA_CUDA_MODULE_NONCAUSAL", "cqsa_cuda_next_v9")
os.environ.setdefault("CQSA_BACKWARD", "cuda")
os.environ.setdefault("CQSA_CPU_THREADS", "8")

import torch

B, H, D = 1, 8, 64
C, ITR = 7, 1
ARMS = ("wave_sync", "wave_async", "classic")
NOTE = ("no FA-2 arm in this A/B; FA-2 ratios elsewhere in the program are torch SDPA flash backend "
        "(flash_attn not installed), ~5% slower than the flash_attn package")


def gpu_info():
    info = dict(gpu=torch.cuda.get_device_name(0), torch=torch.__version__, node=socket.gethostname(),
                slurm_job=os.environ.get("SLURM_JOB_ID"), cpu_threads=os.environ["CQSA_CPU_THREADS"])
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name,clocks.sm,clocks.max.sm,memory.total", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=30)
        info["nvidia_smi"] = r.stdout.strip()
    except Exception as e:  # pragma: no cover
        info["nvidia_smi"] = f"unavailable: {e!r}"
    return info


def inputs(N, host, seed=0):
    g = torch.Generator().manual_seed(seed)
    with_threads = torch.get_num_threads()
    torch.set_num_threads(int(os.environ["CQSA_CPU_THREADS"]))
    q, k, v = (torch.randn(B, H, N, D, generator=g).to(torch.float16) for _ in range(3))
    torch.set_num_threads(with_threads)
    if host:
        # token-major storage, as both engines ask for host-resident inputs (else each call copies 3 x N*H*D*2 B)
        return tuple(t.transpose(1, 2).contiguous().transpose(1, 2) for t in (q, k, v))
    return tuple(t.cuda() for t in (q, k, v))


def make_runner(arm, host):
    """The forward call of one arm: host accumulator, itr=1, c=7, causal, no escalation."""
    from stream_cqsa.autoconfig import QUORUM_SETS
    iset = QUORUM_SETS[C]
    if arm.startswith("wave"):
        os.environ["CQSA_WAVE_MERGE_ASYNC"] = "1" if arm == "wave_async" else "0"
        from stream_cqsa.native_wave import wave_forward
        return lambda q, k, v: wave_forward(q, k, v, causal=True, itr=ITR, c=C, interest_set=iset,
                                            accumulate_on_gpu=False, kernel="cuda")
    from stream_cqsa.stable_stream import stream_cqsa_forward
    # the planner's host-streamed acc=CPU configuration (autoconfig.Plan.engine_kwargs), n_par=2 as profile_sweep
    return lambda q, k, v: stream_cqsa_forward(q, k, v, causal=True, itr=ITR, c=C, interest_set=iset, allow_escalation=False,
                                               max_parallel=2, low_memory=True, accumulate_on_gpu=False,
                                               stream_from_host=host, shared_chunks=host)


def worker(cfg):
    arm, N, host = cfg["arm"], cfg["N"], cfg["residency"] == "host"
    run = make_runner(arm, host)
    q, k, v = inputs(N, host)
    base = torch.cuda.memory_allocated()
    times, peaks, info = [], [], {}
    for i in range(cfg["warmup"] + cfg["reps"]):
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
        t0 = time.perf_counter()
        out, info = run(q, k, v)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        peaks.append(torch.cuda.max_memory_allocated())
        if i >= cfg["warmup"]:
            times.append(dt)
        info.pop("lse", None); del out
    return dict(cfg, times_s=times, mean_s=sum(times) / len(times), min_s=min(times),
                peak_gib=[p / 2**30 for p in peaks], inputs_gib=base / 2**30,
                host_maxrss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
                n_waves=info.get("n_waves"), wave_sizes=info.get("wave_sizes"), merge_async=info.get("merge_async"),
                n_parallel=info.get("n_parallel"), **gpu_info(), note=NOTE)


def check_arms(N=131072):
    """One process: wave sync == wave async bit-identical; classic agrees to fp32 summation order."""
    q, k, v = inputs(N, host=True)
    outs = {}
    for arm in ARMS:
        out, info = make_runner(arm, host=True)(q, k, v)
        torch.cuda.synchronize()
        outs[arm] = out.float().cpu()          # the classic engine hands the output back on the device
    rel = lambda a, b: ((a.double() - b.double()).norm() / b.double().norm()).item()
    res = dict(N=N, wave_sync_eq_async=bool(torch.equal(outs["wave_sync"], outs["wave_async"])),
               rel_classic_vs_wave=rel(outs["classic"], outs["wave_async"]))
    assert res["wave_sync_eq_async"], "async fold is not bit-identical to the synchronous fold"
    assert res["rel_classic_vs_wave"] < 1e-5, res
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["run", "worker"])
    ap.add_argument("cfg", nargs="?")
    ap.add_argument("--out", default="/scratch/gpfs/AKEY/yb2807/Stream-CQSA-ultimate/results/ultimate/wave_merge_ab.json")
    ap.add_argument("--N", type=int, nargs="+", default=[1 << 20, 1 << 21])
    ap.add_argument("--residency", nargs="+", default=["host", "device"])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    a = ap.parse_args()
    if a.mode == "worker":
        print("RESULT " + json.dumps(worker(json.loads(a.cfg))), flush=True)
        return
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    print(gpu_info(), flush=True)
    print("arm check:", check_arms(), flush=True)
    doc = dict(info=gpu_info(), config=dict(B=B, H=H, D=D, c=C, itr=ITR, dtype="fp16", causal=True, acc="cpu",
                                            reps=a.reps, warmup=a.warmup), arm_check=check_arms(), note=NOTE, rows=[])
    for residency in a.residency:               # host-resident (the target path) first, so a timeout keeps it
        for N in a.N:
            for arm in ARMS:
                cfg = dict(arm=arm, N=N, residency=residency, reps=a.reps, warmup=a.warmup)
                proc = subprocess.run([sys.executable, "-P", os.path.abspath(__file__), "worker", json.dumps(cfg)],
                                      stdout=subprocess.PIPE, text=True)
                line = next((ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")), None)
                r = json.loads(line[7:]) if line else dict(cfg, status="error", rc=proc.returncode, stdout=proc.stdout[-4000:])
                doc["rows"].append(r)
                json.dump(doc, open(a.out, "w"), indent=1)
                if "times_s" in r:
                    print(f"{residency:6s} N={N:8d} {arm:10s}: {r['mean_s']:7.2f} s (min {r['min_s']:.2f}, runs {[round(t, 2) for t in r['times_s']]})"
                          f"  peak {max(r['peak_gib']):.2f} GiB  host {r['host_maxrss_gib']:.1f} GiB  waves={r['n_waves']}", flush=True)
                else:
                    print(f"{cfg}: ERROR rc={r.get('rc')}\n{r.get('stdout')}", flush=True)
    print("\nsummary (mean s; ratio = wave_async / classic):")
    for residency in a.residency:
        for N in a.N:
            m = {r["arm"]: r.get("mean_s") for r in doc["rows"] if r["N"] == N and r["residency"] == residency}
            if all(m.get(x) for x in ARMS):
                print(f"  {residency:6s} N={N:8d}: sync {m['wave_sync']:.2f}  async {m['wave_async']:.2f}  classic {m['classic']:.2f}"
                      f"  async/classic {m['wave_async'] / m['classic']:.3f}  async/sync {m['wave_async'] / m['wave_sync']:.3f}")


if __name__ == "__main__":
    main()
