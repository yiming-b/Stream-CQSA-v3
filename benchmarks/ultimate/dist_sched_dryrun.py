"""
Dry run of the D1 scheduler on the real task lists (CPU only, no GPU): predicted makespan / ideal for
round-robin, LPT (+ descent), LPT + child-refinement splits, and the model-only row-span split (U1), on
homogeneous and heterogeneous rate vectors, at the shapes the multi-device rows use.

    python -P benchmarks/ultimate/dist_sched_dryrun.py --out results/ultimate/dist_sched_dryrun.md

Rates are tiles/s of the seeds (scheduler.KNOWN_DEVICE_RATES): A100-SXM4-80GB 3.35e7 (measured),
A100-PCIe-80GB 2.9e7, A100-PCIe-40GB 2.25e7. The design's heterogeneous check (01_design_a100.md 6)
is the {SXM4, PCIe-40, PCIe-40, SXM4} vector: predicted makespan <= 1.05 x sum(work)/sum(rate).
"""
from __future__ import annotations

import argparse
import time

from stream_cqsa import scheduler as S
from stream_cqsa.distributed import plan_distributed

SXM, P80, P40 = 3.35e7, 2.9e7, 2.25e7
VECTORS = {
    "4 x PCIe-80 (equal)": [P80] * 4,
    "SXM4, PCIe-40, PCIe-40, SXM4": [SXM, P40, P40, SXM],
    "(1, 1, 0.67, 0.67) x PCIe-80": [P80, P80, 0.67 * P80, 0.67 * P80],
    "(1, 0.5, 1, 1) x PCIe-80 (the job's throttle)": [P80, 0.5 * P80, P80, P80],
    "8 x PCIe-80 (equal)": [P80] * 8,
    "2 x SXM4 + 2 x PCIe-80": [SXM, SXM, P80, P80],
}
SHAPES = [(1 << 20, 1), (1 << 20, 2), (1 << 21, 1), (1 << 21, 2), (1 << 22, 1), (1 << 22, 2)]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-n", type=int, default=1 << 22)
    args = ap.parse_args(argv)
    lines = ["# D1 scheduler dry run (predicted makespan / ideal; CPU, real task lists, c=7, causal, H=8, D=64)", "",
             "ideal = sum(live-tile work) / sum(rates) + per-unit overheads spread evenly (the harmonic ideal); "
             "`rows` is the model-only row-span split (needs the kernel row-range parameter, U1).", "",
             "| ranks | N | itr | rr | lpt (whole tasks) | lpt + children | splits | units/rank | rows (U1) | schedule s |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for name, rates in VECTORS.items():
        ranks = [dict(rate=r) for r in rates]
        for N, itr in SHAPES:
            if N > args.max_n:
                continue
            res = {}
            t0 = time.perf_counter()
            for key, sched, split in (("rr", "rr", "none"), ("lpt", "lpt", "none"), ("ch", "lpt", "children")):
                res[key] = plan_distributed(N, itr, ranks, scheduler=sched, split=split)
            dt = time.perf_counter() - t0
            # model-only row spans on the same work model
            wm_units = res["lpt"].per_rank
            units = sorted((u for p in wm_units for u in p), key=lambda u: u.path_idx)
            from stream_cqsa.distributed import _WORK_CACHE
            wm = next(w for k, w in _WORK_CACHE.items() if k[0] == N and k[5] == itr)
            rows = S.schedule(units, [S.RankSpec(r) for r in rates], params=S.CostParams(BH=8), split="rows", work=wm)
            ch = res["ch"]
            lines.append(f"| {name} | {N >> 20}M | {itr} | {res['rr'].ratio:.3f} | {res['lpt'].ratio:.3f} | "
                         f"{ch.ratio:.3f} | {ch.n_splits} | {' '.join(str(len(p)) for p in ch.per_rank)} | "
                         f"{rows.ratio:.3f} | {dt:.1f} |")
            print(lines[-1], flush=True)
    text = "\n".join(lines) + "\n"
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)


if __name__ == "__main__":
    main()
