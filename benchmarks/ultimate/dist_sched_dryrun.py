"""
Dry run of the multi-device scheduler on the real task lists (CPU only, no GPU) with the cost model refitted
from the 4-GPU job 14330294 (D2: scheduler.PATH_COSTS, the PCIe-80 rate, BWD_FACTOR; autoconfig.CostModel's
merge terms).

    python -P benchmarks/ultimate/dist_sched_dryrun.py --out results/ultimate/dist_sched_dryrun.md

Three tables:

 1. predicted vs measured for every arm of the job (dist_sched_homog / hetero / hetero_itr2.json): the slowest
    rank's local time predicted on the units it ran, the merge, and the distributed time -- the refit's
    validation (target: residual within 10%);
 2. the decision: for every (direction, N, itr, rate table) of the job, the assignment the default rule (lpt,
    cost-aware children split, reduce_scatter) picks, the measured arm that ran that assignment and the best
    measured arm -- whether the rule reproduces the measured winner;
 3. the model over rate vectors and shapes (round-robin, LPT on whole tasks, LPT + cost-aware split, the
    model-only row-span split U1) on the production read paths: the host-streamed forward (the chunk pool at
    itr=1, the host gather deeper), the device-resident forward and the device-resident backward.

Rates are tiles/s of the seeds (scheduler.KNOWN_DEVICE_RATES): A100-SXM4-80GB 3.35e7, A100-PCIe-80GB 2.91e7
(the refit), A100-PCIe-40GB 2.25e7; the job's throttled rank (54 of 108 SMs) ran at 0.573 of a full one.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter

from stream_cqsa import scheduler as S
from stream_cqsa.autoconfig import CostModel, HardwareSpec
from stream_cqsa.distributed import plan_distributed, _WORK_CACHE
from stream_cqsa.stable_stream import task_seg_align_for

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESULTS = os.path.join(ROOT, "results", "ultimate")
JOB_FILES = ("dist_sched_homog.json", "dist_sched_hetero.json", "dist_sched_hetero_itr2.json")
SXM, P80, P40 = 3.35e7, S.seed_rate("NVIDIA A100 80GB PCIe").tile_rate, 2.25e7
VECTORS = {
    "4 x PCIe-80 (equal)": [P80] * 4,
    "SXM4, PCIe-40, PCIe-40, SXM4": [SXM, P40, P40, SXM],
    "(1, 1, 0.67, 0.67) x PCIe-80": [P80, P80, 0.67 * P80, 0.67 * P80],
    "(1, 0.573, 1, 1) x PCIe-80 (the job's throttle)": [P80, 0.573 * P80, P80, P80],
    "8 x PCIe-80 (equal)": [P80] * 8,
    "2 x SXM4 + 2 x PCIe-80": [SXM, SXM, P80, P80],
}
SHAPES = [(1 << 20, 1), (1 << 20, 2), (1 << 21, 1), (1 << 21, 2), (1 << 22, 1), (1 << 22, 2)]
# the production read paths (the driver's rule): the chunk pool at itr=1 of a host-streamed forward, the host
# gather deeper; device-resident inputs read on the device
PROD = {"fwd host-streamed": lambda itr: ("pool" if itr == 1 else "host", "host"),
        "fwd device-resident": lambda itr: ("dev", "dev"),
        "bwd device-resident": lambda itr: ("dev", "dev")}


def align_for(paths):
    """The task layout per depth the driver derives for these paths (in place = the chunk pool)."""
    return lambda d: task_seg_align_for(128, in_place=(d == 1 and paths[0] == "pool"))


def row_paths(row: dict) -> tuple[str, str]:
    if row["direction"] == "bwd":
        return ("dev", "dev")
    return ("pool" if row.get("shared_chunks") else "host", "host")


def true_rates(row: dict, throttled_frac: float) -> list[float]:
    """The ranks' rates: the refit PCIe-80 rate, the throttled rank at its fitted fraction of it."""
    thr = row.get("throttle")
    t = int(thr.split(":")[0]) if thr else -1
    return [P80 * (throttled_frac if r == t else 1.0) for r in range(row["world"])]


def table_rates(row: dict, throttled_frac: float) -> list[float]:
    """The rate table the driver would assign with today: the refit rates on ``seed`` arms; on ``measured``
    arms the job's calibrated per-rank rates in their measured proportions (the D1 calibration subtracted
    only the launches, which left every rank ~20% low; today's subtracts the chunk pool's copy too)."""
    if row["rates"] != "measured":
        return true_rates(row, throttled_frac)
    thr = row.get("throttle")
    t = int(thr.split(":")[0]) if thr else -1
    used = [float(r) for r in row["rate_used_per_rank"]]
    full = [u for r, u in enumerate(used) if r != t]
    return [u * P80 / (sum(full) / len(full)) for u in used]


def load_rows() -> list[dict]:
    rows = []
    for name in JOB_FILES:
        path = os.path.join(RESULTS, name)
        if os.path.exists(path):
            d = json.load(open(path))
            for r in d["rows"]:
                rows.append(dict(r, tag=d["meta"].get("tag"), job=d["meta"].get("job_id")))
    return rows


def predicted_vs_measured(rows: list[dict], throttled_frac: float) -> list[str]:
    cm = CostModel()
    hw = HardwareSpec()                       # the nominal 25 GB/s link the measured ring's merge matches
    out = ["| tag | dir | N | itr | arm | units/rank | max t_local meas s | pred s | merge+fixed meas s | pred s | "
           "t_dist meas s | pred s | meas/pred |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    worst = 0.0
    for row in rows:
        paths = row_paths(row)
        p = S.CostParams(BH=8, d_factor=S.BWD_FACTOR if row["direction"] == "bwd" else 1.0, N=row["N"],
                         base_depth=row["itr"], path_base=S.PATH_COSTS[paths[0]], path_deeper=S.PATH_COSTS[paths[1]])
        rates = true_rates(row, throttled_frac)
        pred = [p.load([S.Unit(d, (), i, L, t) for d, i, L, t in us], r) for us, r in zip(row["units_per_rank"], rates)]
        merge_p = cm.dist_merge_s(row["N"], 1, 8, 64, row["world"], hw, direction=row["direction"], reduce=row["reduce"],
                                  output=row["output"]) + cm.dist_fixed_s
        merge_m = row["t_dist_s"] - max(row["t_local_per_rank"])
        t_pred = max(pred) + merge_p
        ratio = row["t_dist_s"] / t_pred
        worst = max(worst, abs(ratio - 1))
        out.append(f"| {row['tag']} | {row['direction']} | {row['N'] >> 20}M | {row['itr']} | {row['arm']} | "
                   f"{' '.join(str(n) for n in row['n_units_per_rank'])} | {max(row['t_local_per_rank']):.2f} | {max(pred):.2f} | "
                   f"{merge_m:.2f} | {merge_p:.2f} | {row['t_dist_s']:.2f} | {t_pred:.2f} | {ratio:.3f} |")
    out.append("")
    out.append(f"Worst |meas/pred - 1| over the {len(rows)} arms: {worst:.3f}.")
    return out


def partition(units_per_rank) -> tuple:
    """Per rank, how many units at each depth: what an assignment costs up to the pairing of equal tasks."""
    return tuple(tuple(sorted(Counter(int(u[0]) for u in us).items())) for us in units_per_rank)


def same_assignment(a: tuple, b: tuple, rates: list[float]) -> bool:
    """The same unit counts per depth on every rank, up to a permutation of ranks of the same rate (the
    itr=1 tasks are equal within 0.1%, so which of them pair up on a rank does not matter; a balanced
    itr=2 LPT differs from another only in that pairing)."""
    groups = {}
    for r, rate in enumerate(rates):
        groups.setdefault(round(rate / rates[0], 3), []).append(r)
    return all(sorted(a[r] for r in g) == sorted(b[r] for r in g) for g in groups.values())


def decisions(rows: list[dict], throttled_frac: float) -> list[str]:
    out = ["| tag | dir | N | itr | rates | rule: units/rank (splits) | pred makespan s | measured arm with that assignment | "
           "its t_dist s | best measured arm | its t_dist s | verdict |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    keys = []
    for row in rows:
        k = (row["tag"], row["direction"], row["N"], row["itr"], row["rates"])
        if k not in keys:
            keys.append(k)
    for tag, direction, N, itr, rates_kind in keys:
        group = [r for r in rows if (r["tag"], r["direction"], r["N"], r["itr"], r["rates"]) == (tag, direction, N, itr, rates_kind)]
        # the arms of one (dir, N, itr) share the single-device reference; the decision competes against every arm
        # measured at that shape, whatever rate table it used
        shape = [r for r in rows if (r["tag"], r["direction"], r["N"], r["itr"]) == (tag, direction, N, itr)]
        rates = table_rates(group[0], throttled_frac)
        paths = row_paths(group[0])
        a = plan_distributed(N, itr, [dict(rate=r) for r in rates], split="children", direction=direction,
                             seg_align_for_depth=align_for(paths) if direction == "fwd" else None, paths=paths, n_par=1)
        mine = partition([[u.depth, u.path_idx] for u in p] for p in a.per_rank)
        matched = [r for r in shape if same_assignment(mine, partition(r["units_per_rank"]), rates)]
        best = min(shape, key=lambda r: r["t_dist_s"])
        rates_s = " ".join(f"{r / P80:.2f}" for r in rates)
        units = " ".join(str(len(p)) for p in a.per_rank)
        if matched:
            m = min(matched, key=lambda r: r["t_dist_s"])
            verdict = "winner" if m["t_dist_s"] <= best["t_dist_s"] * 1.02 else f"{m['t_dist_s'] / best['t_dist_s']:.2f}x the winner"
            out.append(f"| {tag} | {direction} | {N >> 20}M | {itr} | {rates_s} | {units} ({a.n_splits}) | {a.makespan:.2f} | "
                       f"{m['arm']} | {m['t_dist_s']:.2f} | {best['arm']} | {best['t_dist_s']:.2f} | {verdict} |")
        else:
            out.append(f"| {tag} | {direction} | {N >> 20}M | {itr} | {rates_s} | {units} ({a.n_splits}) | {a.makespan:.2f} | "
                       f"not measured | - | {best['arm']} | {best['t_dist_s']:.2f} | predicted vs measured best: "
                       f"{a.makespan + (best['t_dist_s'] - max(best['t_local_per_rank'])):.2f} vs {best['t_dist_s']:.2f} |")
    return out


def model_table(max_n: int) -> list[str]:
    lines = ["| path | ranks | N | itr | rr | lpt (whole tasks) | lpt + cost-aware children | splits | units/rank | "
             "makespan s | rows (U1) | schedule s |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for pname, path_of in PROD.items():
        direction = "bwd" if pname.startswith("bwd") else "fwd"
        shapes = SHAPES if pname == "fwd host-streamed" else [s for s in SHAPES if s[1] == 1 and s[0] <= (1 << 21)]
        for name, rates in VECTORS.items():
            ranks = [dict(rate=r) for r in rates]
            for N, itr in shapes:
                if N > max_n:
                    continue
                paths = path_of(itr)
                kw = dict(direction=direction, paths=paths, n_par=1,
                          seg_align_for_depth=align_for(paths) if direction == "fwd" else None)
                res = {}
                t0 = time.perf_counter()
                for key, sched, split in (("rr", "rr", "none"), ("lpt", "lpt", "none"), ("ch", "lpt", "children")):
                    res[key] = plan_distributed(N, itr, ranks, scheduler=sched, split=split, **kw)
                dt = time.perf_counter() - t0
                # the model-only row spans on the same work model and costs
                units = sorted((u for p in res["lpt"].per_rank for u in p), key=lambda u: u.path_idx)
                wm = next(w for k, w in _WORK_CACHE.items() if k[0] == N and k[5] == itr)
                p = S.CostParams(BH=8, d_factor=S.BWD_FACTOR if direction == "bwd" else 1.0, N=N, base_depth=itr,
                                 path_base=S.PATH_COSTS[paths[0]], path_deeper=S.PATH_COSTS[paths[1]])
                rows_a = S.schedule(units, [S.RankSpec(r) for r in rates], params=p, split="rows", work=wm)
                ch = res["ch"]
                lines.append(f"| {pname} | {name} | {N >> 20}M | {itr} | {res['rr'].ratio:.3f} | {res['lpt'].ratio:.3f} | "
                             f"{ch.ratio:.3f} | {ch.n_splits} | {' '.join(str(len(x)) for x in ch.per_rank)} | "
                             f"{ch.makespan:.2f} | {rows_a.ratio:.3f} | {dt:.1f} |")
                print(lines[-1], flush=True)
    return lines


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-n", type=int, default=1 << 22)
    args = ap.parse_args(argv)
    refit = os.path.join(RESULTS, "dist_cost_refit.json")
    thr = 0.573
    if os.path.exists(refit):
        c = json.load(open(refit))["coefficients"]
        thr = c["inv_rate"] / c["inv_rate_throttled"]
    rows = load_rows()
    lines = ["# D2 scheduler dry run: the refit cost model on the real task lists (CPU; c=7, causal, H=8, D=64)", "",
             "Cost model: `scheduler.PATH_COSTS` (per gathered token and per engine call of the chunk-pool, host-gather "
             "and device paths), `TASK_S`, `BWD_FACTOR`, the PCIe-80 rate 2.91e7 tiles/s -- all fitted on the per-rank "
             "local times of job 14330294 (`benchmarks/ultimate/dist_cost_refit.py`, `results/ultimate/dist_cost_refit.json`); "
             "the merge from `CostModel.dist_merge_s` (per-token allreduce / reduce_scatter / all_gather terms of the same "
             f"job). The job's throttled rank (54/108 SMs) ran at {thr:.3f} of a full rank.", ""]
    if rows:
        lines += ["## 1. Predicted vs measured, every arm of job 14330294", "",
                  "predicted local time = the scheduler's load of the units the rank ran (depth, L, live tiles from the row); "
                  "merge+fixed measured = t_dist - max t_local.", "", *predicted_vs_measured(rows, thr), "",
                  "## 2. The default rule's decision against the measured arms", "",
                  "lpt with the cost-aware children split on the row's rate table (`seed` arms: equal refit rates, the throttled "
                  "rank at its fitted fraction; `measured` arms: the job's calibrated per-rank rates in their measured "
                  "proportions), matched to the measured arm that ran the same assignment (the same unit counts per depth on "
                  "every rank, rank permutations among equal ranks allowed) and compared with the fastest arm measured at that "
                  "shape (any reduction).", "", *decisions(rows, thr), ""]
    lines += ["## 3. The model over rate vectors and shapes", "",
              "ratio = predicted makespan / the fluid ideal (pair work at the summed rate + the per-unit overheads spread evenly "
              "+ one engine call); `rows (U1)` is the model-only row-span split (needs the kernel row-range parameter).", "",
              *model_table(args.max_n)]
    text = "\n".join(lines) + "\n"
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
        print("wrote", args.out)
    else:
        print(text)


if __name__ == "__main__":
    main()
