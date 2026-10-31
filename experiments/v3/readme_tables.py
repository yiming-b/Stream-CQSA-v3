"""The README's measurement tables from results/v3/results.jsonl (one node type per table; ratios against the FA-2
baseline of the same N, D, dtype and direction). Prints the markdown; `--write` splices it into README.md between the
`<!-- v3-tables:E# -->` markers (replacing the `[E#: pending]` placeholders the first time).

    python -P experiments/v3/readme_tables.py [--write]
"""
import argparse, os, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench import (BASELINES, DEFAULT_OUT, SERIES, baseline_lookup, fmt_e, latest_by_key, n_label,   # noqa: E402
                   read_rows, summary_e0, summary_e8)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ARMS = ("classic_CUDA", "wave_CUDA", "classic_Triton", "wave_Triton")
SHORT = {"classic_CUDA": "classic CUDA", "wave_CUDA": "wave CUDA", "classic_Triton": "classic Triton", "wave_Triton": "wave Triton"}


def cell(r, base):
    """wall s (x FA-2) / device GiB of a series row; the itr that ran when it is not 1."""
    if r is None:
        return "-"
    if r.get("status") == "oom":
        return "OOM"
    if r.get("status") != "ok":
        return r.get("status", "?")
    f = base(r, "FA-2")
    rat = f" ({r['wall_s'] / f['wall_s']:.2f}x)" if (f and f.get("status") == "ok") else ""
    itr = f" itr{r['itr_ran']}" if r.get("itr_ran") not in (None, 1) else ""
    return f"{r['wall_s']:.2f} s{rat} / {r['peak_dev_gib']:.1f} GiB{itr}"


def base_cell(r):
    if r is None:
        return "-"
    if r.get("status") == "oom":
        return "OOM"
    return f"{r['wall_s']:.2f} s / {r['peak_dev_gib']:.1f} GiB" if r.get("status") == "ok" else r.get("status", "?")


def by(rows, **kw):
    return [r for r in rows if all(r.get(k) == v for k, v in kw.items())]


def one(rows, **kw):
    rs = by(rows, **kw)
    return rs[0] if rs else None


def table_scaling(exp, rows, base, node="sxm", Ns=None, arms=ARMS, host_col=False):
    ex = by(rows, exp=exp, node=node)
    Ns = Ns or sorted({r["N"] for r in ex})
    out = []
    for d in ("fwd", "bwd"):
        out += [f"*{'forward' if d == 'fwd' else 'forward + backward'}* (wall s (x FA-2) / peak device GiB; the depth that ran when it is not 1):", "",
                "| N | FA-2 | SDPA | " + " | ".join(SHORT[a] for a in arms) + (" | host GiB (classic CUDA) |" if host_col else " |"),
                "|---|---|---|" + "---|" * len(arms) + ("---|" if host_col else "")]
        for N in Ns:
            fa = base(dict(exp=exp, node=node, N=N, D=64, dtype="fp16", direction=d), "FA-2")
            sd = base(dict(exp=exp, node=node, N=N, D=64, dtype="fp16", direction=d), "SDPA")
            cells = [cell(one(ex, arm=a, N=N, direction=d), base) for a in arms]
            line = f"| {n_label(N)} | {base_cell(fa)} | {base_cell(sd)} | " + " | ".join(cells)
            if host_col:
                r = one(ex, arm="classic_CUDA", N=N, direction=d)
                line += f" | {r['peak_host_gib']:.1f} (VmHWM {r['vmhwm_gib']:.0f}) |" if r and r.get("status") == "ok" else " | - |"
            else:
                line += " |"
            out.append(line)
        out.append("")
    return out


def table_e3(rows, base):
    ex = by(rows, exp="E3")
    out = []
    for d in ("fwd", "bwd"):
        out += [f"*{'forward' if d == 'fwd' else 'forward + backward'}* at N=1M, itr=1 (wall s (x FA-2) / peak device GiB):", "",
                "| c | subproblems | " + " | ".join(SHORT[a] for a in ARMS) + " |", "|---|---|" + "---|" * len(ARMS)]
        for c in sorted({r["c"] for r in ex}):
            cells = [cell(one(ex, arm=a, c=c, direction=d), base) for a in ARMS]
            out.append(f"| {c} | {c} | " + " | ".join(cells) + " |")
        out.append("")
    return out


def table_e4(rows, base):
    ex = by(rows, exp="E4")
    out = ["N=1M, c=7, Q/K/V on the device (wall s (x FA-2) / peak device GiB / peak host GiB):", "",
           "| direction | itr | accumulator | " + " | ".join(SHORT[a] for a in ARMS) + " |", "|---|---|---|" + "---|" * len(ARMS)]
    for d in ("fwd", "bwd"):
        for itr in (1, 2):
            for acc in ("gpu", "cpu"):
                cells = []
                for a in ARMS:
                    r = one(ex, arm=a, itr=itr, acc=acc, direction=d)
                    c_ = cell(r, base)
                    if r and r.get("status") == "ok":
                        c_ += f" / {r['peak_host_gib']:.1f}"
                        if r.get("acc_ran") != acc:
                            c_ += f" (ran acc={r['acc_ran']})"
                    cells.append(c_)
                out.append(f"| {d} | {itr} | {'device' if acc == 'gpu' else 'host'} | " + " | ".join(cells) + " |")
    return out + [""]


def table_e5(rows, base):
    ex = by(rows, exp="E5")
    out = ["N=1M, c=73, itr=1 (73 subproblems), the wave budget as a multiple of the largest subproblem (wall s (x FA-2) / peak device GiB / waves):", "",
           "| direction | W | wave CUDA | wave Triton |", "|---|---|---|---|"]
    for d in ("fwd", "bwd"):
        for w in ("W=1x", "W=2x", "W=4x", "W=8x", "W=16x"):
            cells = []
            for a in ("wave_CUDA", "wave_Triton"):
                r = one(ex, arm=a, extra=w, direction=d)
                c_ = cell(r, base)
                if r and r.get("status") == "ok" and r.get("n_waves"):
                    c_ += f" / {r['n_waves']} waves"
                cells.append(c_)
            out.append(f"| {d} | {w[2:]} | " + " | ".join(cells) + " |")
    return out + [""]


def table_e6(rows, base):
    ex = by(rows, exp="E6")
    out = ["N=1M, c=7, itr=1 (wall s (x FA-2) / peak device GiB):", "",
           "| direction | dtype | D | FA-2 | " + " | ".join(SHORT[a] for a in ARMS) + " |", "|---|---|---|---|" + "---|" * len(ARMS)]
    for d in ("fwd", "bwd"):
        for dt in ("fp16", "bf16"):
            for D in (64, 128):
                fa = base(dict(exp="E6", node="sxm", N=1 << 20, D=D, dtype=dt, direction=d), "FA-2")
                cells = []
                for a in ARMS:
                    r = one(ex, arm=a, dtype=dt, D=D, direction=d)
                    cells.append(cell(r, base) if r else ("excluded" if (a.endswith("Triton") and D == 128 and d == "bwd") else "-"))
                out.append(f"| {d} | {dt} | {D} | {base_cell(fa)} | " + " | ".join(cells) + " |")
    return out + ["", "excluded: the Triton backward at D=128 does not finish at 1M (a known 128-wide key-tile pathology; the CUDA backward is the default), see `results/v3/STATUS.md`.", ""]


def table_e9(rows, base):
    ex = by(rows, exp="E9")
    out = ["Host-resident forward (Q/K/V in pinned host memory, accumulator on the host), c=7, itr=1. Classic engine: chunk pool of 3 / 4 / 7 "
           "slots (3 = the current subproblem's chunks; the pool keeps a chunk the next subproblem shares) against `flush` = no pool, every "
           "subproblem gathered on the host; wave engine: pool of 3 (one wave's chunks) against 7 (all chunks resident, what it sized itself "
           "to before v3.0). Wall s (x FA-2 where a resident FA-2 row exists) / peak device GiB; classic rows add H2D chunk transfers / hits of 21 loads.", "",
           "| N | D | classic flush | classic pool 3 | classic pool 4 | classic pool 7 | wave pool 3 | wave pool 7 |", "|---|---|---|---|---|---|---|---|"]
    for N in sorted({r["N"] for r in ex}):
        for D in sorted({r["D"] for r in ex}):
            cells = []
            for arm, pool in (("classic_CUDA", "pool=flush"), ("classic_CUDA", "pool=3"), ("classic_CUDA", "pool=4"), ("classic_CUDA", "pool=7"),
                              ("wave_CUDA", "pool=3"), ("wave_CUDA", "pool=7")):
                r = one(ex, arm=arm, N=N, D=D, extra=pool)
                c_ = cell(r, base)
                info = (r or {}).get("info") or {}
                if r and r.get("status") == "ok" and info.get("chunk_transfers") is not None:
                    c_ += f" / {info['chunk_transfers']}t {info['chunk_hits']}h"
                if r and r.get("status") == "ok" and (r.get("reps") or 1) > 1:
                    c_ += f" (median of {r['reps']})"
                cells.append(c_)
            out.append(f"| {n_label(N)} | {D} | " + " | ".join(cells) + " |")
    return out + [""]


def sections(rows):
    ok = [r for r in rows if r.get("status") == "ok"]
    base = baseline_lookup(ok)
    S = {}
    e0 = by(rows, exp="E0")
    S["E0"] = [f"{sum(r.get('status') == 'ok' for r in e0)} of {len(e0)} rows ran and are exact; the worst errors per series (the reference is FlashAttention-2's output and gradients; "
               "the fp64 columns compare both to a float64 SDPA):", ""] + summary_e0(e0)[2:]
    e0b = by(rows, exp="E0b")
    S["E0b"] = ([f"{sum(r.get('status') == 'ok' for r in e0b)} of {len(e0b)} rows ran and are exact (the reference is FlashAttention-2; fp64 = a float64 SDPA):", ""]
                + summary_e0(e0b)[2:]) if e0b else ["_no E0b rows yet_", ""]
    S["E1"] = table_scaling("E1", rows, base)
    S["E2"] = (["**A100-SXM4-80GB**, Q/K/V in pinned host memory and the accumulator in host memory (ratios against E1's resident FA-2 rows of the same N):", ""]
               + table_scaling("E2", rows, base, host_col=True)
               + ["**A100 80GB PCIe** (its own resident baselines; FA-2's fwd+bwd at 8M does not fit an 80 GB device):", ""]
               + table_scaling("E2", rows, base, node="pcie", host_col=True))
    S["E3"] = table_e3(rows, base)
    S["E4"] = table_e4(rows, base)
    S["E5"] = table_e5(rows, base)
    S["E6"] = table_e6(rows, base)
    S["E8"] = summary_e8(by(rows, exp="E8"), ok)[2:] or ["_no E8 rows yet_", ""]
    S["E9"] = table_e9(rows, base) if by(rows, exp="E9") else ["_no E9 rows yet_", ""]
    return S


def write_readme(S):
    p = os.path.join(ROOT, "README.md")
    s = open(p).read()
    for exp, lines in S.items():
        block = f"<!-- v3-tables:{exp} -->\n" + "\n".join(lines).rstrip("\n") + f"\n<!-- /v3-tables:{exp} -->"
        pat = re.compile(rf"<!-- v3-tables:{exp} -->.*?<!-- /v3-tables:{exp} -->", re.S)
        if pat.search(s):
            s = pat.sub(lambda m: block, s)
        else:
            # first time: after the section whose heading carries the placeholder (E3-E6 share one heading)
            head = {"E3": "E3-E6", "E4": None, "E5": None, "E6": None}.get(exp, exp)
            if head is None:
                continue
            m = re.search(rf"^### .*\*\*\[{re.escape(head)}: pending\]\*\*.*?$", s, re.M)
            if not m:
                m = re.search(rf"^### {re.escape(head)}\b.*$", s, re.M)
            if not m:
                print(f"no heading for {exp}", file=sys.stderr)
                continue
            end = s.find("\n### ", m.end())
            end = len(s) if end < 0 else end
            if exp == "E3":
                block = "\n\n".join(f"<!-- v3-tables:{e} -->\n**{e}**\n\n" + "\n".join(S[e]).rstrip("\n") + f"\n<!-- /v3-tables:{e} -->" for e in ("E3", "E4", "E5", "E6"))
            s = s[:end].rstrip("\n") + "\n\n" + block + "\n" + s[end:]
            s = s.replace(f" **[{head}: pending]**", "", 1)
    open(p, "w").write(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    rows = list(latest_by_key(read_rows(a.out)).values())
    S = sections(rows)
    for exp, lines in S.items():
        print(f"\n===== {exp} =====\n" + "\n".join(lines))
    if a.write:
        write_readme(S)
        print("README.md updated")


if __name__ == "__main__":
    main()
