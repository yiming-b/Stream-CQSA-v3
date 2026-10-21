"""
U1 engine-level A/B: stream_cqsa.attention() forward AND backward with device-resident inputs and a device
accumulator, the released v2.2.3 package (its stable v11 / v9 modules) against v3 with the v13 module, next to
the monolithic flash_attn and SDPA-flash calls, ONE node (A100-SXM4-80GB in slurm/ultimate/engine_ab.slurm).

Arms (every (arm, row) runs in its own subprocess with its own PYTHONPATH / environment, flushed GPU):
  v223_cuda     v2.2.3 (git archive of the tag), CQSA_CUDA_MODULE=cqsa_cuda_next_v11 / NONCAUSAL=v9, CQSA_BACKWARD=cuda,
                kernel="cuda": the classic engine with v11's CQS backward
  v223_triton   the same with CQSA_BACKWARD=triton (v2.2.3's default backward; the forward is the same v11 kernel)
  v3_cuda       v3 + cqsa_cuda_v13 (CQSA_CUDA_MODULE unset: the interface picks it), kernel="cuda": the classic
                engine, census + ALIGNED forward and backward, the CUDA backward by default
  v3_auto       the same, kernel="auto": the planner's choice (with pinned itr / c: the wave engine, which still
                runs the old cqsa_native module -- package U2 ports the census + ALIGNED design into it)
  flash_attn    monolithic flash_attn_func / torch.autograd.grad
  sdpa_flash    monolithic torch SDPA with the FLASH_ATTENTION backend
Rows: (N, c, itr, dtype) = (131072, 7, 1, fp16), (131072, 73, 1, fp16), (1048576, 7, 1, fp16), (1048576, 7, 2, fp16),
(131072, 7, 1, bf16); B=1, H=8, D=64, causal; the monolithic arms run once per (N, dtype).
Per row: forward (1 warm-up + `--reps` timed, wall clock around the call + synchronize, median and min) and
backward (torch.autograd.grad on the retained graph, same protocol), peak device memory, the engine's info
(engine, subproblem count, backward kernel), GPU name; every worker also runs the monolithic flash_attn call on
the same seeded inputs and records the norm-relative error of its output and gradients against it (a sanity check
next to the bit-level identity of tests/test_v13.py).
Design acceptance (docs/ultimate/01_design_a100.md P2.1 / P3.x rows): engine backward 1M itr=1 <= 24 s
(v2.2.x: 39.1 s CUDA / 29.0 s Triton); forward 1M c=7 device-resident <= 1.17x monolithic flash_attn.

    SNAP=$(bash slurm/ultimate/snapshot.sh) && sbatch --export=ALL,CQSA_SRC=$SNAP,V13_DIR=<frozen> slurm/ultimate/engine_ab.slurm
    python -P benchmarks/ultimate/engine_ab.py run --smoke        # vis1: N=16384 rows, correctness of the plumbing
    python -P benchmarks/ultimate/engine_ab.py report results/ultimate/engine_ab.json
"""
import argparse, gc, json, os, socket, statistics, subprocess, sys, time

B, H, D = 1, 8, 64
ROWS = ((131072, 7, 1, "fp16"), (131072, 73, 1, "fp16"), (1 << 20, 7, 1, "fp16"), (1 << 20, 7, 2, "fp16"), (131072, 7, 1, "bf16"))
SMOKE_ROWS = ((16384, 7, 1, "fp16"), (16384, 7, 2, "fp16"), (16384, 7, 1, "bf16"))
ARMS = ("v223_cuda", "v223_triton", "v3_cuda", "v3_auto", "flash_attn", "sdpa_flash")
MONO = ("flash_attn", "sdpa_flash")
KER = "/scratch/gpfs/AKEY/yb2807/Stream-CQSA-dev/next/kernel"
NATIVE = "/scratch/gpfs/AKEY/yb2807/Stream-CQSA-dev/next/native"
OUT_DEFAULT = "/scratch/gpfs/AKEY/yb2807/Stream-CQSA-v3/results/ultimate/engine_ab.json"


def arm_env(arm, v3_root, v223_root, v13_dir):
    """PYTHONPATH and kernel selection of one arm; the base is the job's environment (env_next.sh)."""
    env = {k: v for k, v in os.environ.items()
           if k not in ("CQSA_CUDA_MODULE", "CQSA_CUDA_MODULE_NONCAUSAL", "CQSA_BACKWARD", "CQSA_FORWARD")}
    rest = [p for p in os.environ.get("PYTHONPATH", "").split(":") if p and p not in (v3_root, v223_root, v13_dir)]
    if arm.startswith("v223"):
        env["PYTHONPATH"] = ":".join([v223_root, f"{KER}/stable"] + rest + [NATIVE])
        env.update(CQSA_CUDA_MODULE="cqsa_cuda_next_v11", CQSA_CUDA_MODULE_NONCAUSAL="cqsa_cuda_next_v9",
                   CQSA_BACKWARD="cuda" if arm == "v223_cuda" else "triton")
    else:
        # v3: nothing set, so interface.py picks cqsa_cuda_v13 (v13_dir last: variant dirs never shadow the package)
        env["PYTHONPATH"] = ":".join([v3_root, f"{KER}/stable"] + rest + [NATIVE, v13_dir])
    return env


def gpu_info():
    import torch
    info = dict(gpu=torch.cuda.get_device_name(0), torch=torch.__version__, node=socket.gethostname(),
                slurm_job=os.environ.get("SLURM_JOB_ID"))
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name,clocks.sm,clocks.max.sm,memory.total", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=30)
        info["nvidia_smi"] = r.stdout.strip()
    except Exception as e:  # pragma: no cover
        info["nvidia_smi"] = f"unavailable: {e!r}"
    return info


def timed(fn, reps, warmup):
    import torch
    times, peaks = [], []
    for i in range(warmup + reps):
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
        t0 = time.perf_counter()
        r = fn()
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        peaks.append(torch.cuda.max_memory_allocated() / 2**30)
        if i >= warmup:
            times.append(dt)
        if i < warmup + reps - 1:
            del r
    return r, dict(times_s=times, median_s=statistics.median(times), min_s=min(times), peak_gib=max(peaks))


def rel(a, b):
    return ((a.double() - b.double()).norm() / b.double().norm()).item()


def worker(cfg):
    import torch
    arm, N, c, itr, dtype = cfg["arm"], cfg["N"], cfg["c"], cfg["itr"], cfg["dtype"]
    dt = {"fp16": torch.float16, "bf16": torch.bfloat16}[dtype]
    g = torch.Generator(device="cuda").manual_seed(N)
    q, k, v = (torch.randn(B, H, N, D, device="cuda", dtype=dt, generator=g).requires_grad_(True) for _ in range(3))
    dout = torch.randn(B, H, N, D, device="cuda", dtype=dt, generator=g)
    r = dict(cfg, **gpu_info())

    # the monolithic flash_attn reference on the same inputs (fp16/bf16 output and gradients)
    from flash_attn import flash_attn_func
    qf, kf, vf = (t.detach().transpose(1, 2).contiguous().requires_grad_(True) for t in (q, k, v))
    of = flash_attn_func(qf, kf, vf, causal=True, softmax_scale=D ** -0.5)
    gf = torch.autograd.grad(of, (qf, kf, vf), dout.transpose(1, 2).contiguous())
    ref_out, ref_g = of.transpose(1, 2), [x.transpose(1, 2) for x in gf]
    del qf, kf, vf, of, gf
    torch.cuda.synchronize(); torch.cuda.empty_cache()

    if arm == "flash_attn":
        fwd = lambda: flash_attn_func(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), causal=True, softmax_scale=D ** -0.5).transpose(1, 2)
    elif arm == "sdpa_flash":
        import torch.nn.functional as F
        from torch.nn.attention import sdpa_kernel, SDPBackend

        def fwd():
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                return F.scaled_dot_product_attention(q, k, v, is_causal=True)
    else:
        import stream_cqsa
        from stream_cqsa import interface as I
        r.update(package=os.path.dirname(stream_cqsa.__file__), version=stream_cqsa.__version__,
                 cuda_module=I.CQSA_CUDA_MODULE, cuda_module_file=getattr(I.cqsa_cuda, "__file__", None),
                 backward_kernel=(I.backward_kernel() if hasattr(I, "backward_kernel")
                                  else os.environ.get("CQSA_BACKWARD", "auto")))
        kernel = "cuda" if arm.endswith("cuda") or arm.startswith("v223") else "auto"
        info = {}

        def fwd():
            out, inf = stream_cqsa.attention(q, k, v, is_causal=True, kernel=kernel, itr=itr, c=c, bwd_itr=itr, return_info=True)
            info.update({k_: v_ for k_, v_ in inf.items() if isinstance(v_, (int, float, str, bool, list, tuple, type(None)))})
            return out
    out, r["fwd"] = timed(fwd, cfg["reps"], cfg["warmup"])
    if arm not in MONO:
        # attention()'s backward-direction info carries engine / kernel / itr / c but not the subproblem
        # count; with a pinned itr and c either engine runs exactly c**itr subproblems
        if info.get("n_subproblems") is None:
            info["n_subproblems"] = c ** itr
        r["info"] = info
    r["rel_out_vs_flash_attn"] = rel(out, ref_out)
    bwd = lambda: torch.autograd.grad(out, (q, k, v), dout, retain_graph=True)
    grads, r["bwd"] = timed(bwd, cfg["reps"], cfg["warmup"])
    r["rel_grads_vs_flash_attn"] = [rel(a, b) for a, b in zip(grads, ref_g)]
    r["inputs_gib"] = 3 * q.numel() * q.element_size() / 2**30
    return r


def run_worker(cfg, env, root):
    proc = subprocess.run([sys.executable, "-P", os.path.join(root, "benchmarks", "ultimate", "engine_ab.py"), "worker", json.dumps(cfg)],
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env, cwd=root)
    line = next((ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")), None)
    if line:
        return json.loads(line[7:])
    return dict(cfg, status="error", rc=proc.returncode, stdout=proc.stdout[-6000:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["run", "worker", "report"])
    ap.add_argument("cfg", nargs="?")
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--smoke", action="store_true", help="small rows on a shared GPU (plumbing + correctness only)")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--v3-root", default=os.environ.get("CQSA_SRC", "/scratch/gpfs/AKEY/yb2807/Stream-CQSA-v3"))
    ap.add_argument("--v223-root", default=os.environ.get("V223_DIR", "/scratch/gpfs/AKEY/yb2807/Stream-CQSA-snapshots/v2-v2.2.3"))
    ap.add_argument("--v13-dir", default=os.environ.get("V13_DIR", "/scratch/gpfs/AKEY/yb2807/Stream-CQSA-v3/kernels/v13"))
    a = ap.parse_args()
    if a.mode == "worker":
        print("RESULT " + json.dumps(worker(json.loads(a.cfg))), flush=True)
        return
    if a.mode == "report":
        report(json.load(open(a.cfg)), a.cfg.replace(".json", ".md"))
        return
    rows = SMOKE_ROWS if a.smoke else ROWS
    arms = a.arms.split(",")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    doc = dict(info=gpu_info(), config=dict(B=B, H=H, D=D, causal=True, acc="gpu", residency="device", reps=a.reps,
                                            warmup=a.warmup, v3_root=a.v3_root, v223_root=a.v223_root, v13_dir=a.v13_dir),
               rows=[])
    print(doc["info"], flush=True)
    done_mono = set()
    for N, c, itr, dtype in rows:
        for arm in arms:
            if arm in MONO:
                if (arm, N, dtype) in done_mono:
                    continue
                done_mono.add((arm, N, dtype))
            cfg = dict(arm=arm, N=N, c=c, itr=itr, dtype=dtype, reps=a.reps, warmup=a.warmup)
            if arm.startswith("v223") and dtype == "bf16":
                # v2.2.3 runs on the stable dev v11 module, which is a fp16-only build (the 12 known bf16 suite failures)
                doc["rows"].append(dict(cfg, status="skipped", reason="v2.2.3's stable v11 module is fp16-only"))
                continue
            t0 = time.time()
            r = run_worker(cfg, arm_env(arm, a.v3_root, a.v223_root, a.v13_dir), a.v3_root)
            r["wall_s"] = time.time() - t0
            doc["rows"].append(r)
            json.dump(doc, open(a.out, "w"), indent=1)
            if "fwd" in r:
                print(f"N={N:8d} c={c:3d} itr={itr} {dtype} {arm:12s}: fwd {r['fwd']['median_s'] * 1e3:9.1f} ms  bwd "
                      f"{r['bwd']['median_s'] * 1e3:9.1f} ms  peak {r['fwd']['peak_gib']:.1f}/{r['bwd']['peak_gib']:.1f} GiB  "
                      f"rel out {r['rel_out_vs_flash_attn']:.1e} grads {max(r['rel_grads_vs_flash_attn']):.1e}  "
                      f"engine={r.get('info', {}).get('engine')} n_sub={r.get('info', {}).get('n_subproblems')} "
                      f"bwd_kernel={r.get('backward_kernel')}  ({r['wall_s']:.0f} s)", flush=True)
            else:
                print(f"{cfg}: ERROR rc={r.get('rc')}\n{r.get('stdout')}", flush=True)
    report(doc, a.out.replace(".json", ".md"))


def report(doc, path_md):
    info = doc["info"]
    cfg = doc["config"]
    lines = [f"# U1 engine-level A/B: v2.2.3 vs v3 + v13 (job {info.get('slurm_job')}, {info.get('node')}, {info.get('gpu')})", "",
             f"`benchmarks/ultimate/engine_ab.py`; B={cfg['B']} H={cfg['H']} D={cfg['D']} causal, device-resident inputs, "
             f"acc=GPU, {cfg['reps']} timed runs after {cfg['warmup']} warm-up, medians (ms). Backward = "
             "torch.autograd.grad of attention()'s output. rel = norm-relative error of the fp16/bf16 output / the worst "
             "gradient against the monolithic flash_attn call on the same inputs.", ""]
    rows = doc["rows"]
    keys = sorted({(r["N"], r["c"], r["itr"], r["dtype"]) for r in rows}, key=lambda t: (t[3], t[0], t[1], t[2]))
    lines += ["| N | c | itr | dtype | arm | engine / n_sub / bwd kernel | fwd ms | bwd ms | peak fwd / bwd GiB | rel out / grads | fwd / flash_attn | bwd / flash_attn |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for N, c, itr, dtype in keys:
        fa = next((r for r in rows if r["arm"] == "flash_attn" and r["N"] == N and r["dtype"] == dtype and "fwd" in r), None)
        for r in rows:
            if (r["N"], r["dtype"]) != (N, dtype) or (r["arm"] not in MONO and (r["c"], r["itr"]) != (c, itr)):
                continue
            if r["arm"] in MONO and (c, itr) != min((k[1], k[2]) for k in keys if (k[0], k[3]) == (N, dtype)):
                continue
            if "fwd" not in r:
                what = f"skipped: {r['reason']}" if r.get("status") == "skipped" else f"ERROR rc={r.get('rc')}"
                lines.append(f"| {N} | {c} | {itr} | {dtype} | {r['arm']} | {what} | | | | | | |")
                continue
            inf = r.get("info", {})
            eng = f"{inf.get('engine', '-')} / {inf.get('n_subproblems', '-')} / {r.get('backward_kernel', '-')}" if r["arm"] not in MONO else "monolithic"
            f_ratio = f"{r['fwd']['median_s'] / fa['fwd']['median_s']:.3f}" if fa else "-"
            b_ratio = f"{r['bwd']['median_s'] / fa['bwd']['median_s']:.3f}" if fa else "-"
            lines.append(f"| {N} | {c} | {itr} | {dtype} | {r['arm']} | {eng} | {r['fwd']['median_s'] * 1e3:.1f} | {r['bwd']['median_s'] * 1e3:.1f} | "
                         f"{r['fwd']['peak_gib']:.1f} / {r['bwd']['peak_gib']:.1f} | {r['rel_out_vs_flash_attn']:.1e} / "
                         f"{max(r['rel_grads_vs_flash_attn']):.1e} | {f_ratio} | {b_ratio} |")
    lines.append("")
    # design acceptance rows
    def get(arm, N, c, itr, dtype="fp16"):
        return next((r for r in rows if r["arm"] == arm and (r["N"], r["dtype"]) == (N, dtype) and "fwd" in r
                     and (arm in MONO or (r["c"], r["itr"]) == (c, itr))), None)
    v3, v2, v2t, fa = (get(a_, 1 << 20, 7, 1) for a_ in ("v3_cuda", "v223_cuda", "v223_triton", "flash_attn"))
    lines += ["## Design acceptance (01_design_a100.md P2.1 / P3.x)", "", "| target | measured | pass |", "|---|---|---|"]
    if v3:
        b = v3["bwd"]["median_s"]
        lines.append(f"| engine backward 1M itr=1 device-resident acc=GPU <= 24 s (v2.2.x: 39.1 s CUDA / 29.0 s Triton) | "
                     f"v3+v13 {b:.2f} s" + (f"; this job's v2.2.3: {v2['bwd']['median_s']:.2f} s CUDA" if v2 else "")
                     + (f" / {v2t['bwd']['median_s']:.2f} s Triton" if v2t else "") + f" | {'yes' if b <= 24 else 'NO'} |")
        if fa:
            fr = v3["fwd"]["median_s"] / fa["fwd"]["median_s"]
            lines.append(f"| forward 1M c=7 itr=1 device-resident <= 1.17x monolithic flash_attn | {fr:.3f}x "
                         f"({v3['fwd']['median_s'] * 1e3:.0f} vs {fa['fwd']['median_s'] * 1e3:.0f} ms)"
                         + (f"; v2.2.3 {v2['fwd']['median_s'] / fa['fwd']['median_s']:.3f}x" if v2 else "") + f" | {'yes' if fr <= 1.17 else 'NO'} |")
    lines += ["", "Note: the `v3_auto` arm is the wave engine, which still runs the old `cqsa_native` module (package U2 ports the "
              "census + ALIGNED design into it); `v3_cuda` is the classic engine on the v13 module.", ""]
    open(path_md, "w").write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
