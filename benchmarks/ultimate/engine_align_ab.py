"""
A/B of the classic engine's task layout (P0.1 "alignment everywhere"): device-resident forward,
acc=GPU, itr=1, fp16, B=1 H=8 D=64 causal, seg_align=0 (the legacy even split, A/B only) vs the
default block-aligned layout (chunk boundaries on 128-token blocks; results/ultimate/layout_ab.json
measured 1.365x -> 1.004x FA-2 per subproblem on the v11 kernel), at (N, c) in {(131072, 7),
(131072, 73), (1048576, 7)}. The wave engine (one launch per wave, aligned by construction) and the
monolithic torch SDPA flash call run in the same job as the reference arms.

Every (N, c, arm) runs in its own subprocess (flushed GPU), 1 warm-up + `--reps` timed runs (wall
clock around the call + synchronize), peak device memory recorded, GPU name in every row. The parent
first checks the arms against each other in one process at N=131072 c=7 (aligned vs legacy classic
norm-rel <= 1e-4, wave vs aligned classic <= 1e-4; SDPA fp16 output recorded).

The FA-2 arm is torch SDPA with the flash backend (flash_attn is not installed), about 5-7% slower
than the flash_attn package.

    python -P benchmarks/ultimate/engine_align_ab.py run --out results/ultimate/engine_align_ab.json
"""
import argparse, gc, json, os, socket, subprocess, sys, time

os.environ.setdefault("CQSA_CUDA_MODULE", "cqsa_cuda_next_v11")
os.environ.setdefault("CQSA_CUDA_MODULE_NONCAUSAL", "cqsa_cuda_next_v9")
os.environ.setdefault("CQSA_CPU_THREADS", "8")

import torch

B, H, D = 1, 8, 64
ITR = 1
ARMS = ("classic_align0", "classic_align128", "wave", "sdpa_flash")
CONFIGS = ((131072, 7), (131072, 73), (1 << 20, 7))
NOTE = ("FA-2 arm is torch SDPA flash backend (flash_attn not installed), ~5-7% slower than the "
        "flash_attn package; classic arms: device-resident, acc=GPU, allow_escalation=False, "
        "max_parallel by the engine's planner")


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


def inputs(N, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return tuple(torch.randn(B, H, N, D, device="cuda", dtype=torch.float16, generator=g) for _ in range(3))


def make_runner(arm, c):
    """The forward call of one arm: device accumulator, itr=1, causal."""
    from stream_cqsa.autoconfig import QUORUM_SETS
    iset = QUORUM_SETS[c]
    if arm == "sdpa_flash":
        import torch.nn.functional as F
        from torch.nn.attention import sdpa_kernel, SDPBackend

        def run(q, k, v):
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                return F.scaled_dot_product_attention(q, k, v, is_causal=True), {}
        return run
    if arm == "wave":
        from stream_cqsa.native_wave import wave_forward
        return lambda q, k, v: wave_forward(q, k, v, causal=True, itr=ITR, c=c, interest_set=iset,
                                            accumulate_on_gpu=True, kernel="cuda")
    from stream_cqsa.stable_stream import stream_cqsa_forward
    seg_align = 0 if arm == "classic_align0" else 128
    return lambda q, k, v: stream_cqsa_forward(q, k, v, causal=True, itr=ITR, c=c, interest_set=iset,
                                               allow_escalation=False, accumulate_on_gpu=True, seg_align=seg_align)


def worker(cfg):
    arm, N, c = cfg["arm"], cfg["N"], cfg["c"]
    run = make_runner(arm, c)
    q, k, v = inputs(N)
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
                n_subproblems=info.get("n_subproblems"), n_parallel=info.get("n_parallel"),
                n_waves=info.get("n_waves"), wave_sizes=info.get("wave_sizes"), **gpu_info(), note=NOTE)


def check_arms(N=131072, c=7):
    """One process: aligned and legacy classic agree to fp32 summation order (a different partition
    of the same pairs); the wave engine agrees with the aligned classic; SDPA (fp16 output) recorded."""
    q, k, v = inputs(N)
    outs = {}
    for arm in ARMS:
        out, _ = make_runner(arm, c)(q, k, v)
        torch.cuda.synchronize()
        outs[arm] = out.float()
    rel = lambda a, b: ((a.double() - b.double()).norm() / b.double().norm()).item()
    res = dict(N=N, c=c, rel_align128_vs_align0=rel(outs["classic_align128"], outs["classic_align0"]),
               rel_wave_vs_align128=rel(outs["wave"], outs["classic_align128"]),
               rel_align128_vs_sdpa=rel(outs["classic_align128"], outs["sdpa_flash"]),
               rel_align0_vs_sdpa=rel(outs["classic_align0"], outs["sdpa_flash"]))
    assert res["rel_align128_vs_align0"] <= 1e-4 and res["rel_wave_vs_align128"] <= 1e-4, res
    del q, k, v, outs
    torch.cuda.empty_cache()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["run", "worker"])
    ap.add_argument("cfg", nargs="?")
    ap.add_argument("--out", default="/scratch/gpfs/AKEY/yb2807/Stream-CQSA-ultimate/results/ultimate/engine_align_ab.json")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    a = ap.parse_args()
    if a.mode == "worker":
        print("RESULT " + json.dumps(worker(json.loads(a.cfg))), flush=True)
        return
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    print(gpu_info(), flush=True)
    doc = dict(info=gpu_info(), config=dict(B=B, H=H, D=D, itr=ITR, dtype="fp16", causal=True, acc="gpu",
                                            residency="device", reps=a.reps, warmup=a.warmup),
               arm_check=check_arms(), note=NOTE, rows=[])
    print("arm check:", doc["arm_check"], flush=True)
    for N, c in CONFIGS:
        for arm in ARMS:
            cfg = dict(arm=arm, N=N, c=c, reps=a.reps, warmup=a.warmup)
            proc = subprocess.run([sys.executable, "-P", os.path.abspath(__file__), "worker", json.dumps(cfg)],
                                  stdout=subprocess.PIPE, text=True)
            line = next((ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")), None)
            r = json.loads(line[7:]) if line else dict(cfg, status="error", rc=proc.returncode, stdout=proc.stdout[-4000:])
            doc["rows"].append(r)
            json.dump(doc, open(a.out, "w"), indent=1)
            if "times_s" in r:
                print(f"N={N:8d} c={c:3d} {arm:16s}: {r['mean_s'] * 1e3:9.1f} ms (min {r['min_s'] * 1e3:.1f}, runs "
                      f"{[round(t * 1e3, 1) for t in r['times_s']]})  peak {max(r['peak_gib']):.2f} GiB  "
                      f"n_par={r['n_parallel']} waves={r['n_waves']}", flush=True)
            else:
                print(f"{cfg}: ERROR rc={r.get('rc')}\n{r.get('stdout')}", flush=True)
    print("\nsummary (mean ms; ratios to sdpa_flash):")
    for N, c in CONFIGS:
        m = {r["arm"]: r.get("mean_s") for r in doc["rows"] if r["N"] == N and r["c"] == c}
        if all(m.get(x) for x in ARMS):
            sd = m["sdpa_flash"]
            print(f"  N={N:8d} c={c:3d}: align0 {m['classic_align0'] * 1e3:8.1f} ({m['classic_align0'] / sd:.2f}x)  "
                  f"align128 {m['classic_align128'] * 1e3:8.1f} ({m['classic_align128'] / sd:.2f}x)  "
                  f"wave {m['wave'] * 1e3:8.1f} ({m['wave'] / sd:.2f}x)  sdpa {sd * 1e3:8.1f}  "
                  f"align128/align0 {m['classic_align128'] / m['classic_align0']:.3f}")


if __name__ == "__main__":
    main()
