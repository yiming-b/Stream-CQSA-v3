"""
Refit of the multi-device cost model from the per-rank telemetry of the 4-GPU scheduler job (D2).

    python -P benchmarks/ultimate/dist_cost_refit.py results/ultimate/dist_sched_homog.json \\
        results/ultimate/dist_sched_hetero.json results/ultimate/dist_sched_hetero_itr2.json \\
        [--out results/ultimate/dist_cost_refit.json]

Every (row, rank) of the job is one observation of the rank's local time, and the rows carry what the
rank ran: its units (depth, path_idx, L, live tiles) and its engine calls (one per depth). The model
fitted here is the scheduler's (``stream_cqsa/scheduler.py``, ``CostParams``):

    t_local = sum over depth calls  [ call_s + N * call_tok_s[path] ]
            + sum over units        [ tiles * B*H * f_dir / rate + task_s + L * tok_s[path] ]

``path`` is how the call reads Q/K/V -- ``pool`` (the host-streamed itr=1 call with the shared chunk
pool: one H2D copy of the whole N per call, in-place reads per unit), ``host`` (the host-streamed gather
path: every unit gathers its L tokens on the host), ``dev`` (device-resident gather) -- and ``f_dir`` is
1 for the forward, the backward factor for the backward. The unknowns are linear in the observations
(1/rate, f_bwd/rate, call_s, call_tok_s per path, task_s, tok_s per path), solved by non-negative least
squares on relative errors. A throttled rank (green context) gets its own 1/rate. Residuals per arm
(max over ranks of predicted/measured) are printed; the fitted constants with their provenance go to
``--out`` and, by hand, into ``scheduler.CostParams`` / ``autoconfig.CostModel``.
"""
from __future__ import annotations

import argparse
import json
import os
import re

import numpy as np

PATHS = ("pool", "host", "dev")


def unit_path(row: dict, depth: int) -> str:
    """How the engine read Q/K/V for the units of this depth in this row (the driver's rule:
    the shared chunk pool exists at depth 1 of a host-streamed forward only)."""
    if row["direction"] == "bwd":
        return "dev"                                   # the job's backward rows are device-resident
    if row.get("shared_chunks") and depth == 1:
        return "pool"
    return "host"


def calibration_observations(meta: dict, throttled: int | None) -> list[dict]:
    """The per-rank startup calibrations of the job (``scheduler.calibrate_rate(stream_from_host=True)``: one
    host-streamed itr=1 call with the shared chunk pool at a small N) as observations of the same model, so
    the fit holds at the shape the rate is calibrated at and the calibration can subtract the same terms."""
    from stream_cqsa import scheduler as S
    from stream_cqsa.stable_stream import build_tasks_cached, task_seg_align_for
    obs = []
    for rk in meta.get("ranks", []):
        m = re.search(r"N=(\d+) c=(\d+) itr=1 D=(\d+) host-streamed, ([0-9.]+) ms", rk.get("rate_measured_source", ""))
        if not m:
            continue
        N, c, D, ms = int(m.group(1)), int(m.group(2)), int(m.group(3)), float(m.group(4))
        tasks = build_tasks_cached(N, 1, B=1, H=8, D=D, itemsize=2, sorted_gather=True, pin=False, c=c,
                                   interest_set=(0, 1, 3), seg_align=task_seg_align_for(128, in_place=False))
        units = S.WorkModel(N, c=c, causal=True).units_from_tasks(tasks, 1)
        r = int(rk["rank"])
        rate_key = "inv_rate_throttled" if (throttled is not None and r == throttled) else "inv_rate"
        feat = {rate_key: sum(u.tiles for u in units) * 8, "task_s": float(len(units)),
                "tok_s_pool": float(sum(u.L for u in units)), "call_s": 1.0, "call_tok_s_pool": float(N)}
        row = dict(tag="calibration", direction="fwd", N=N, itr=1, arm=f"rank {r} startup calibration",
                   n_units_per_rank=[len(units)])
        obs.append(dict(feat=feat, t=ms / 1e3, row=row, rank=r))
    return obs


def observations(rows: list[dict], throttled: int | None) -> tuple[list[dict], list[str]]:
    """One observation per (row, rank): feature dict and the measured local time."""
    obs = []
    for row in rows:
        BH = 8
        for r in range(row["world"]):
            feat = {}
            rate_key = "inv_rate_throttled" if (throttled is not None and r == throttled) else "inv_rate"
            if row["direction"] == "bwd":
                rate_key = rate_key + "_bwd"
            for u in row["units_per_rank"][r]:
                depth, _, L, tiles = u
                p = unit_path(row, depth)
                feat[rate_key] = feat.get(rate_key, 0.0) + tiles * BH
                feat["task_s"] = feat.get("task_s", 0.0) + 1.0
                feat[f"tok_s_{p}"] = feat.get(f"tok_s_{p}", 0.0) + L
            for call in row["depth_calls_per_rank"][r]:
                p = unit_path(row, call["depth"])
                feat["call_s"] = feat.get("call_s", 0.0) + 1.0
                feat[f"call_tok_s_{p}"] = feat.get(f"call_tok_s_{p}", 0.0) + row["N"]
            obs.append(dict(feat=feat, t=row["t_local_per_rank"][r], row=row, rank=r))
    keys = sorted({k for o in obs for k in o["feat"]})
    return obs, keys


def fit(obs: list[dict], keys: list[str], fixed: dict[str, float]) -> dict[str, float]:
    """NNLS on relative errors (every observation weighted by 1/t), with ``fixed`` constants held."""
    from scipy.optimize import nnls
    free = [k for k in keys if k not in fixed]
    A = np.zeros((len(obs), len(free)))
    b = np.zeros(len(obs))
    for i, o in enumerate(obs):
        t = o["t"]
        known = sum(fixed.get(k, 0.0) * v for k, v in o["feat"].items() if k in fixed)
        b[i] = (t - known) / t
        for j, k in enumerate(free):
            A[i, j] = o["feat"].get(k, 0.0) / t
    x, _ = nnls(A, b)
    out = dict(fixed)
    out.update({k: float(v) for k, v in zip(free, x)})
    return out


def predict(o: dict, coef: dict[str, float]) -> float:
    return sum(coef.get(k, 0.0) * v for k, v in o["feat"].items())


def arm_table(obs: list[dict], coef: dict[str, float]) -> list[str]:
    lines = ["| tag | dir | N | itr | arm | units/rank | measured t_local | predicted t_local | pred/meas (max rank) |",
             "|---|---|---|---|---|---|---|---|---|"]
    seen = {}
    for o in obs:
        row = o["row"]
        key = (row.get("tag"), row["direction"], row["N"], row["itr"], row["arm"])
        seen.setdefault(key, []).append(o)
    worst = 0.0
    for key, os_ in seen.items():
        row = os_[0]["row"]
        meas = [x["t"] for x in os_]
        pred = [predict(x, coef) for x in os_]
        i = int(np.argmax(meas))
        ratio = pred[i] / meas[i]
        worst = max(worst, abs(ratio - 1))
        lines.append(f"| {key[0]} | {key[1]} | {key[2] >> 20}M | {key[3]} | {key[4]} | "
                     f"{' '.join(str(n) for n in row['n_units_per_rank'])} | {' '.join(f'{m:.2f}' for m in meas)} | "
                     f"{' '.join(f'{p:.2f}' for p in pred)} | {ratio:.3f} |")
    lines.append(f"\nworst |pred/meas - 1| on the slowest rank: {worst:.3f}")
    return lines


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--out", default=None)
    ap.add_argument("--task-s", type=float, default=3.0e-3, help="per-unit launch overhead held fixed (single-device fit)")
    ap.add_argument("--free-task-s", action="store_true")
    ap.add_argument("--no-calibrations", dest="calibrations", action="store_false",
                    help="leave out the ranks' startup calibrations (N=262144) as observations")
    args = ap.parse_args(argv)
    obs_all, keys_all = [], set()
    job = None
    for path in args.inputs:
        d = json.load(open(path))
        thr = d["meta"].get("throttle")
        throttled = int(thr.split(":")[0]) if thr else None
        job = job or d["meta"].get("job_id")
        obs, keys = observations(d["rows"], throttled)
        obs_all += obs
        if args.calibrations:
            obs_all += calibration_observations(d["meta"], throttled)
        keys_all |= set(keys)
    keys = sorted(keys_all)
    fixed = {} if args.free_task_s else {"task_s": args.task_s}
    coef = fit(obs_all, keys, fixed)
    rate = 1.0 / coef["inv_rate"]
    print(f"job {job}: {len(obs_all)} observations, {len(keys)} coefficients")
    for k in keys:
        print(f"  {k:<26} {coef[k]:.4e}")
    print(f"  tile rate (fwd)            {rate:.3e} tiles/s")
    if coef.get("inv_rate_bwd"):
        print(f"  backward factor            {coef['inv_rate_bwd'] / coef['inv_rate']:.3f}")
    if coef.get("inv_rate_throttled"):
        print(f"  throttled rank rate        {1.0 / coef['inv_rate_throttled']:.3e} ({coef['inv_rate'] / coef['inv_rate_throttled']:.3f} of full)")
    rel = np.array([predict(o, coef) / o["t"] - 1 for o in obs_all])
    print(f"  per-rank residual: mean {rel.mean():+.3f}, rms {np.sqrt((rel ** 2).mean()):.3f}, max |.| {np.abs(rel).max():.3f}")
    print()
    print("\n".join(arm_table(obs_all, coef)))
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(dict(job_id=job, inputs=[os.path.basename(p) for p in args.inputs], n_obs=len(obs_all),
                           coefficients=coef, tile_rate=rate,
                           bwd_factor=(coef["inv_rate_bwd"] / coef["inv_rate"]) if coef.get("inv_rate_bwd") else None,
                           residual=dict(mean=float(rel.mean()), rms=float(np.sqrt((rel ** 2).mean())),
                                         max_abs=float(np.abs(rel).max()))), f, indent=1)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
