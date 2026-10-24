"""Refit the planner's cost model on the v3 measurements (results/v3/results.jsonl, A100-SXM4-80GB rows).

The shipped CostModel defaults (and the per-GPU calibration cache) were fitted on the v2 kernels; v3's kernels
changed the two numbers that decide the planner's accumulator choice: the decomposition itself costs ~0%
instead of ~40% against FA-2 at 1M, and a further depth costs ~4% instead of ~22% per level. With the old
numbers the host accumulator's overhead (10-15% at 1M) looked like 1-5% and the planner's 2% tolerance picked
it for the wrong reason; with these it is predicted within a few percent and the 15% tolerance picks it where
it is cheap.

Fitted here, in this order (each from the rows designed to isolate it):
  pair_rate            FA-2 forward rows 2M-4M (the asymptotic pair rate)
  kernel_ratio,        classic_CUDA device-resident device-accumulator forwards at itr=1 (E1 128K-4M, E3 c=3..133
  task_overhead_s      at 1M): linear least squares in (kernel_ratio, task_overhead_s)
  depth_factor         E4 itr=2 vs itr=1 (device accumulator), residual per-level factor
  bwd_ratio            classic_CUDA fwd+bwd / modelled forward compute (E1, E3, E4 at itr=1, acc=gpu)
  host terms           E4 acc=cpu with device Q/K/V (the accumulator's unhidden cost per gathered token), E2-E1
                       (the host-resident inputs' extra: a per-token gather cost and a fixed setup cost)
  bwd_host_factor      E2-E1 backward extra / forward extra (512K-2M)
Then prints predicted / measured for every classic_CUDA and wave_CUDA row of E1-E4 and the planner's pick per N.

    python benchmarks/v3/refit_cost_model.py [--write-defaults] [--save-cache]
"""
import argparse, json, os, re, statistics, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from stream_cqsa.autoconfig import CostModel, QUORUM_SETS, plan, hardware_from_dict, _cname  # noqa: E402

K, M = 1024, 1024 ** 2


def load(path):
    rows = [json.loads(l) for l in open(path)]
    return [r for r in rows if r.get("status") == "ok" and r.get("node") == "sxm" and r.get("dtype") == "fp16"
            and r.get("D") == 64 and r.get("exp") in ("E1", "E2", "E3", "E4", "E5", "E6")]


def pairs_of(N, c, l, itr, causal=True, B=1, H=8):
    """Live score pairs over all heads (the cost model's definition), subproblem count and subproblem length."""
    L = N * (l / c) ** itr
    n = c ** itr
    masked = (l - 1) / (l * l)
    return n * B * H * L * L * (0.5 if causal else 1.0) * (1 - masked), n, L


def fit(rows, B=1, H=8, D=64):
    cm = CostModel()
    fa2 = {(r["N"], r["direction"]): r["wall_s"] for r in rows if r["arm"] == "FA-2"}
    # 1. pair rate from the large FA-2 forwards
    rates = [B * H * N * N / 2 / fa2[(N, "fwd")] for N in (2 * M, 4 * M) if (N, "fwd") in fa2]
    cm.pair_rate = statistics.mean(rates)
    # 2. kernel_ratio + task_overhead_s on itr=1 device-resident device-accumulator classic forwards
    sel = [r for r in rows if r["arm"] == "classic_CUDA" and r["direction"] == "fwd" and r["acc"] == "gpu"
           and r["qkv"] == "device" and r["itr_ran"] == 1 and r["exp"] in ("E1", "E3", "E4") and not r.get("extra")]
    A, y = [], []
    for r in sel:
        l = len(QUORUM_SETS[r["c"]])
        p, n, L = pairs_of(r["N"], r["c"], l, 1, B=B, H=H)
        ts = n * L * (B * H * D) / 512.0
        dev_terms = ts * (cm.gather_dev_s_per_tok + cm.merge_gpu_s_per_tok) * (1 - cm.overlap)
        A.append([p / cm.pair_rate, n]); y.append(r["wall_s"] - dev_terms)
    (kr, to), *_ = np.linalg.lstsq(np.array(A), np.array(y), rcond=None)
    cm.kernel_ratio, cm.task_overhead_s = float(kr), float(max(to, 0.0))
    # 3. depth factor from E4 itr=2 vs itr=1 (classic, acc=gpu)
    e4 = {(r["itr"], r["acc"]): r["wall_s"] for r in rows if r["exp"] == "E4" and r["arm"] == "classic_CUDA" and r["direction"] == "fwd"}
    cm.depth_factor = 1.0
    N = M
    t1 = cm.cqsa_time(N, B, H, D, 1, True, "gpu", False, 1, c=7, l=3)
    t2 = cm.cqsa_time(N, B, H, D, 2, True, "gpu", False, 1, c=7, l=3)
    cm.depth_factor = float(max(1.0, (e4[(2, "gpu")] / e4[(1, "gpu")]) / (t2 / t1)))
    # 4. bwd_ratio: fwd+bwd wall / modelled forward time, classic acc=gpu device itr=1
    selb = [r for r in rows if r["arm"] == "classic_CUDA" and r["direction"] == "bwd" and r["acc"] == "gpu" and r["qkv"] == "device"
            and r["itr_ran"] == 1 and r["exp"] in ("E1", "E3", "E4") and not r.get("extra") and r["N"] >= 512 * K]
    ratios = []
    for r in selb:
        l = len(QUORUM_SETS[r["c"]])
        cm.bwd_ratio = 1.0
        ratios.append(r["wall_s"] / cm.cqsa_time(r["N"], B, H, D, 1, True, "gpu", False, 1, c=r["c"], l=l, direction="fwd"))
    cm.bwd_ratio = float(statistics.median(ratios))
    # 5. host terms, jointly: per-token merge+d2h (the accumulator), per-token gather (host-resident inputs),
    #    the overlap fraction and the streamed call's setup cost, on the forward rows that carry them: E4 acc=cpu
    #    with device Q/K/V (itr 1 and 2) and E2 host-resident 128K-2M (the 4M row carries a 32 GiB pinned
    #    footprint whose cost is not per token).
    from scipy.optimize import least_squares
    e1 = {r["N"]: r["wall_s"] for r in rows if r["exp"] == "E1" and r["arm"] == "classic_CUDA" and r["direction"] == "fwd"}
    e2 = {r["N"]: r["wall_s"] for r in rows if r["exp"] == "E2" and r["arm"] == "classic_CUDA" and r["direction"] == "fwd"}
    obs = []                                                 # (ts, compute_s, with_gather, measured_extra)
    for itr in (1, 2):
        p_, n, L = pairs_of(M, 7, 3, itr, B=B, H=H)
        obs.append((n * L * (B * H * D) / 512.0, e4[(itr, "gpu")], False, e4[(itr, "cpu")] - e4[(itr, "gpu")]))
    for N in sorted(set(e1) & set(e2)):
        if N > 2 * M:
            continue
        p_, n, L = pairs_of(N, 7, 3, 1, B=B, H=H)
        obs.append((n * L * (B * H * D) / 512.0, e1[N], True, e2[N] - e1[N]))

    def resid(x):
        h_acc, h_g, ov, setup = x
        out = []
        for ts, comp, g, meas in obs:
            host = ts * (h_acc + (h_g if g else 0.0))
            pred = host * (1 - ov) + max(0.0, host * ov - comp) + (setup if g else 0.0)
            out.append((pred - meas) / max(meas, 0.05))
        return out
    x0 = [1e-6, 3e-7, 0.8, 0.2]
    fitted = least_squares(resid, x0, bounds=([1e-8, 0.0, 0.0, 0.0], [1e-4, 1e-4, 0.99, 2.0]))
    h_acc, h_g, ov, setup = fitted.x
    split = cm.merge_cpu_s_per_tok / (cm.merge_cpu_s_per_tok + cm.d2h_s_per_tok)
    cm.merge_cpu_s_per_tok, cm.d2h_s_per_tok = float(h_acc * split), float(h_acc * (1 - split))
    cm.gather_s_per_tok, cm.overlap, cm.host_stream_setup_s = float(h_g), float(ov), float(setup)
    # the wave engine's host fold (a worker under the next wave): its unhidden share relative to the classic
    # engine's, from E4 (wave cpu-gpu) / (classic cpu-gpu) at itr 1 and 2
    e4w = {(r["itr"], r["acc"]): r["wall_s"] for r in rows if r["exp"] == "E4" and r["arm"] == "wave_CUDA" and r["direction"] == "fwd"}
    rel = statistics.mean((e4w[(i_, "cpu")] - e4w[(i_, "gpu")]) / (e4[(i_, "cpu")] - e4[(i_, "gpu")]) for i_ in (1, 2))
    cm.overlap_npar2 = float(1 - rel * (1 - ov))
    # 6. the backward's host factor: (E2 - E1 bwd extra - setup) / the forward's unhidden host share, 256K-2M
    e1b = {r["N"]: r["wall_s"] for r in rows if r["exp"] == "E1" and r["arm"] == "classic_CUDA" and r["direction"] == "bwd" and r["itr_ran"] == 1}
    e2b = {r["N"]: r["wall_s"] for r in rows if r["exp"] == "E2" and r["arm"] == "classic_CUDA" and r["direction"] == "bwd" and r["itr_ran"] == 1}
    f = []
    for N in (256 * K, 512 * K, M, 2 * M):
        if N in e1b and N in e2b:
            p_, n, L = pairs_of(N, 7, 3, 1, B=B, H=H)
            ts = n * L * (B * H * D) / 512.0
            f.append(((e2b[N] - e1b[N]) - setup) / (ts * (h_acc + h_g) * (1 - ov)))
    cm.bwd_host_factor = float(statistics.median(f))
    return cm


def validate(cm, rows, B=1, H=8, D=64):
    print("\npredicted / measured (classic_CUDA and wave_CUDA rows, SXM):")
    print(f"{'exp':3} {'dir':3} {'arm':12} {'N':>5} {'c':>3} {'itr':>3} {'acc':>3} {'qkv':>6} {'eng':>7} | {'meas':>8} {'pred':>8} {'ratio':>6}")
    errs = []
    for r in sorted(rows, key=lambda r: (r["exp"], r["direction"], r["arm"], r["N"], r.get("c") or 0, r.get("itr_ran") or 0, str(r.get("acc")))):
        if r["arm"] not in ("classic_CUDA", "wave_CUDA") or r.get("extra"):
            continue
        l = len(QUORUM_SETS[r["c"]])
        host = r["qkv"] == "host"
        acc = r["acc"] if not (r["engine"] == "wave" and r["direction"] == "bwd") else "gpu"
        eng = r["engine"]
        t = cm.cqsa_time(r["N"], B, H, D, int(r["itr_ran"]), True, acc, host, 1, c=r["c"], l=l, direction=r["direction"],
                         engine=eng, wave_tokens=int(r.get("max_wave_tokens") or 0), gather=True)
        errs.append(t / r["wall_s"])
        print(f"{r['exp']:3} {r['direction']:3} {r['arm']:12} {r['N'] // K:>4}K {r['c']:>3} {r['itr_ran']:>3} {acc:>3} {r['qkv']:>6} {eng:>7} | "
              f"{r['wall_s']:8.2f} {t:8.2f} {t / r['wall_s']:6.2f}")
    print(f"median pred/meas {statistics.median(errs):.3f}; within 10%: {sum(abs(e - 1) <= 0.10 for e in errs)}/{len(errs)}")


def picks(cm, budgets=(72, 40, 24, 12), B=1, H=8, D=64):
    """What the planner takes once the monolithic call does not fit (allow_mono=False): the frontier pick over all
    decompositions (host inputs or evictable ones) and the pick among the device-resident ones (Q/K/V already on
    the device: api._prefer_device_resident), each with its predicted time over the fastest feasible one."""
    from stream_cqsa.autoconfig import frontier_pick
    print("\nplanner picks among the decompositions (c=7; the monolithic call is taken whenever it fits):")
    print("  budget  dir      N | any residency                      | Q/K/V on the device")
    for bud in budgets:
        hw = hardware_from_dict({"cuda:0": f"{bud}GiB", "host": "200GiB"})
        for direction in ("fwd", "bwd"):
            for N in (256 * K, 512 * K, M, 2 * M, 4 * M):
                try:
                    p = plan(N=N, B=B, H=H, D=D, hardware=hw, direction=direction, allow_distributed=False, out_bytes_per_el=2,
                             quorum_sets={7: (0, 1, 3)}, max_itr=3, model=cm, allow_mono=False)
                    feas = [c for c in p.candidates if c["ok"] and c["mode"] == "cqsa"]
                    best = min(c["time"] for c in feas)
                    a = f"{p.name().replace('cqsa ', ''):26s} {p.est_time_s / best:4.2f}x {p.est_peak_gib:5.1f}G"
                    dev = [c for c in feas if not c["stream_from_host"]]
                    if dev:
                        c0, _ = frontier_pick(dev, p.tol)
                        b = f"{_cname(c0):22s} {c0['time'] / best:4.2f}x {c0['peak']:5.1f}G"
                    else:
                        b = "nothing device-resident fits"
                    print(f"  {bud:>3}GiB  {direction}  {N // K:>5}K | {a} | {b}")
                except Exception as exc:                              # noqa: BLE001
                    print(f"  {bud:>3}GiB  {direction}  {N // K:>5}K | {type(exc).__name__}: {str(exc)[:70]}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(ROOT, "results", "v3", "results.jsonl"))
    ap.add_argument("--write-defaults", action="store_true", help="rewrite the CostModel defaults in stream_cqsa/autoconfig.py")
    ap.add_argument("--save-cache", action="store_true", help="save as the calibration cache for the A100-SXM4-80GB")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "v3", "cost_model_a100_sxm4_80gb.json"))
    a = ap.parse_args()
    rows = load(a.results)
    cm = fit(rows)
    print("fitted:")
    for k_ in ("pair_rate", "kernel_ratio", "task_overhead_s", "depth_factor", "bwd_ratio", "merge_cpu_s_per_tok", "d2h_s_per_tok",
               "gather_s_per_tok", "host_stream_setup_s", "bwd_host_factor", "overlap", "overlap_npar2"):
        print(f"  {k_:22s} {getattr(cm, k_):.4g}")
    validate(cm, rows)
    picks(cm)
    with open(a.out, "w") as f:
        f.write(cm.to_json())
    print(f"\nwritten {a.out}")
    if a.save_cache:
        from stream_cqsa.autoconfig import _cache_dir
        path = os.path.join(_cache_dir(), "cost_model_nvidia_a100_sxm4_80gb.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(cm.to_json())
        print(f"saved calibration cache {path}")
    if a.write_defaults:
        p = os.path.join(ROOT, "stream_cqsa", "autoconfig.py")
        s = open(p).read()
        for k_ in ("pair_rate", "kernel_ratio", "task_overhead_s", "depth_factor", "bwd_ratio", "merge_cpu_s_per_tok",
                   "d2h_s_per_tok", "gather_s_per_tok", "host_stream_setup_s", "bwd_host_factor", "overlap", "overlap_npar2"):
            v = getattr(cm, k_)
            s, n = re.subn(rf"^(    {k_}: float = )([^#\n]+?)(\s*#.*)?$", lambda m: f"{m.group(1)}{v:.4g}{m.group(3) or ''}", s, flags=re.M)
            assert n == 1, k_
        open(p, "w").write(s)
        print("CostModel defaults rewritten in stream_cqsa/autoconfig.py")


if __name__ == "__main__":
    main()
