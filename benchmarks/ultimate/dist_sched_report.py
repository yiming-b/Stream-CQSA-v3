"""
Markdown report of the D1 multi-device scheduler job (dist_telemetry.py JSONs).

    python benchmarks/ultimate/dist_sched_report.py results/ultimate/dist_sched_homog.json \
        results/ultimate/dist_sched_hetero.json results/ultimate/dist_sched_hetero_itr2.json \
        --out results/ultimate/dist_sched.md
"""
from __future__ import annotations

import argparse
import json
import os


def _f(x, fmt="{:.2f}"):
    return "-" if x is None else fmt.format(x)


def _lst(xs, fmt="{:.2f}"):
    return " ".join(_f(x, fmt) for x in xs)


def rows_table(rows: list[dict]) -> list[str]:
    out = ["| dir | N | itr | layout | arm | split | single s | dist s | speedup | t_local per rank s | predicted s | "
           "imbalance | merge s (max) | units/rank | peak above start GiB (max) | rel err |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        out.append(
            f"| {r['direction']} | {r['N']} | {r['itr']} | {r['layout']} | {r['arm']} | {r['split']} | "
            f"{_f(r.get('t_single_s'))} | {_f(r['t_dist_s'])} | {_f(r.get('speedup'))} | "
            f"{_lst(r['t_local_per_rank'])} | {_lst(r['predicted_local_per_rank'])} | {r['imbalance']:.3f} | "
            f"{max(r['t_merge_per_rank']):.2f} | {' '.join(str(x) for x in r['n_units_per_rank'])} | "
            f"{max(r['dev_peak_above_start_gib_per_rank']) if r.get('dev_peak_above_start_gib_per_rank') else 0:.1f} | "
            f"{_f(r.get('rel_err'), '{:.1e}')} |")
    return out


def pair_summary(rows: list[dict]) -> list[str]:
    """lpt vs rr and reduce_scatter vs allreduce per (direction, N, itr)."""
    out = ["| dir | N | itr | rr/allreduce dist s | lpt/allreduce dist s | lpt gain | lpt/reduce_scatter dist s | "
           "merge ar -> rs s | peak+ ar -> rs GiB |", "|---|---|---|---|---|---|---|---|---|"]
    keys = sorted({(r["direction"], r["N"], r["itr"]) for r in rows})
    for d, N, itr in keys:
        by = {r["arm"].rsplit("/", 1)[0]: r for r in rows if (r["direction"], r["N"], r["itr"]) == (d, N, itr)}
        rr, la, lr = by.get("rr/allreduce"), by.get("lpt/allreduce"), by.get("lpt/reduce_scatter")
        if not (rr and la):
            continue
        pk = lambda r: max(r["dev_peak_above_start_gib_per_rank"]) if r and r.get("dev_peak_above_start_gib_per_rank") else None  # noqa: E731
        out.append(f"| {d} | {N} | {itr} | {rr['t_dist_s']:.2f} | {la['t_dist_s']:.2f} | {rr['t_dist_s'] / la['t_dist_s']:.3f}x | "
                   f"{_f(lr['t_dist_s'] if lr else None)} | {max(la['t_merge_per_rank']):.2f} -> "
                   f"{_f(max(lr['t_merge_per_rank']) if lr else None)} | {_f(pk(la), '{:.1f}')} -> {_f(pk(lr), '{:.1f}')} |")
    return out


def residual_table(rows: list[dict]) -> list[str]:
    """The measured local share of the single-device time vs the scheduler's modelled share (makespan / total
    predicted work; equal-rate rows only) -- the D1 residual the D2 refit removed (benchmarks/ultimate/
    dist_cost_refit.py) -- and what the distributed call spends outside the slowest rank's local phase
    (schedule + merge + barriers: the merge terms of CostModel.dist_merge_s)."""
    out = ["| dir | N | itr | arm | local share = max t_local / t_single | model share | residual | t_dist - max t_local s |",
           "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if not r.get("t_single_s") or len(set(r.get("rate_used_per_rank", [0]))) != 1:
            continue
        tot = sum(r["predicted_local_per_rank"])
        share_m = r["predicted_makespan_s"] / tot if tot else None
        share = max(r["t_local_per_rank"]) / r["t_single_s"]
        out.append(f"| {r['direction']} | {r['N']} | {r['itr']} | {r['arm']} | {share:.3f} | {_f(share_m, '{:.3f}')} | "
                   f"{_f(share / share_m if share_m else None, '{:.3f}')} | {r['t_dist_s'] - max(r['t_local_per_rank']):.2f} |")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    lines = ["# D1 multi-device scheduler: job results", ""]
    for path in args.inputs:
        if not os.path.exists(path):
            lines += [f"## {os.path.basename(path)}: missing", ""]
            continue
        d = json.load(open(path))
        m, rows = d["meta"], d["rows"]
        lines += [f"## {os.path.basename(path)} (tag `{m.get('tag')}`, job {m.get('job_id')}, {m.get('host')})", ""]
        lines.append(f"GPUs: {', '.join(m.get('gpus', []))}; throttle: {m.get('throttle')}; split: {m.get('split')}; "
                     f"chunk_tokens: {m.get('chunk_tokens')}; engine: {m.get('engine_kw')}")
        lines.append("")
        lines.append("| rank | GPU | SMs used | seed rate (tiles/s) | measured rate (host-streamed) |")
        lines.append("|---|---|---|---|---|")
        for rk in m.get("ranks", []):
            lines.append(f"| {rk['rank']} | {rk['gpu']} | {rk['sms_used']}/{rk['n_sm']} | {rk['rate_seed']:.3e} | "
                         f"{rk['rate_measured']:.3e} |")
        lines += ["", *rows_table(rows), "", *pair_summary(rows), "", *residual_table(rows), ""]
        if m.get("topo"):
            lines += ["<details><summary>nvidia-smi topo -m</summary>", "", "```", m["topo"].rstrip(), "```", "</details>", ""]
    text = "\n".join(lines) + "\n"
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
    print(text)


if __name__ == "__main__":
    main()
