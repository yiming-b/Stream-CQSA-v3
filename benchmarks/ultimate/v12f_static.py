"""
Static side of the P3 forward (recorded, not gated): REG / STACK / SASS instructions of the production forward
instantiations -- CQS off, v11's CQS-on (LEGACY in v12f), v11x's CQS-on without the element path, and v12f's
ALIGNED kernel -- side by side, v12f's ptxas spills and times from its -Xptxas -v build log, and the per-TU
nvcc phase times of its build. Reuses benchmarks/ultimate/cuobj_report.py's parsers.

    python -P benchmarks/ultimate/v12f_static.py --tag b1 --out results/ultimate/v12f_static.md
"""
import argparse, os, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cuobj_report as cr  # noqa: E402

KER = "/scratch/gpfs/AKEY/yb2807/Stream-CQSA-dev/next/kernel"
LOGS = "/scratch/gpfs/AKEY/yb2807/Stream-CQSA-dev/next/logs"
SO = {"v11": f"{KER}/stable/cqsa_cuda_next_v11.cpython-311-x86_64-linux-gnu.so",
      "v11x": f"{KER}/v11x/cqsa_cuda_next_v11x.cpython-311-x86_64-linux-gnu.so",
      "v12f": f"{KER}/v12f/cqsa_cuda_next_v12f.cpython-311-x86_64-linux-gnu.so"}
# the A100 tiles of the production forward (run_mha_fwd_hdim64/128 on sm80, = CqsAlignedTile)
TRAITS = ("hdim64 128x128 w4 fp16", "hdim128 128x64 w4 fp16")


def aligned_rows(so):
    """{(traits, causal, even_MN): dict(reg, stack, sass, mangled)} of v12f's flash_fwd_aligned_kernel."""
    res = cr.parse_res_usage(cr.run([cr.CUOBJDUMP, "-res-usage", so]))
    rows = {}
    for n, d in cr.demangle(n for n in res if "flash_fwd_aligned_kernel" in n).items():
        t = re.search(r"Flash_fwd_kernel_traits<(\d+), (\d+), (\d+), (\d+),.*?(half_t|bfloat16_t)", d)
        flags = re.search(r">, (true|false), (true|false)>\(", d)
        if not (t and flags):
            continue
        traits = f"hdim{t.group(1)} {t.group(2)}x{t.group(3)} w{t.group(4)} {'fp16' if t.group(5) == 'half_t' else 'bf16'}"
        rows[(traits, flags.group(1) == "true", flags.group(2) == "true")] = dict(reg=res[n][0], stack=res[n][1], mangled=n)
    sass = cr.sass_counts(so, [r["mangled"] for r in rows.values()])
    for r in rows.values():
        r["sass"] = sass.get(r["mangled"])
    return rows


def fwd_rows(so):
    """{(traits, causal, even_MN): {cqs: row}} of a module's production flash_fwd_kernel instantiations."""
    out = {}
    for (kind, traits, fs), slot in cr.collect(so).items():
        if kind != "fwd" or traits not in TRAITS:
            continue
        flags = fs.split(",")
        out[(traits, "causal" in flags, "even_MN" in flags)] = slot
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="b1", help="v12f build tag (next/logs/05_build_v12f_<tag>.log)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    mods = {m: fwd_rows(p) for m, p in SO.items()}
    aligned = aligned_rows(SO["v12f"])
    ptxas = cr.parse_ptxas_log(open(f"{LOGS}/05_build_v12f_{a.tag}.log").read())
    lines = [f"# v12f static report (build {a.tag}, sm_80)", "",
             "Cells are REG / STACK bytes / SASS instructions. CQS-on = Cqs_mode 1 (causal) or 2 (non-causal runtime flag): "
             "v11's is v12f's LEGACY; v11x's has the per-element path compiled out (poisons tiles that need it); "
             "ALIGNED = v12f's `flash_fwd_aligned_kernel` (run-table loop, no verdicts, no element path).", "",
             "| traits | causal | even_MN | v11 CQS off | v11 CQS-on | v11x CQS-on | v12f CQS off | v12f LEGACY | v12f ALIGNED | ALIGNED spills st/ld B | ALIGNED ptxas s |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for traits in TRAITS:
        for causal in (True, False):
            for even in (True, False):
                k = (traits, causal, even)
                on = "1" if causal else "2"
                al = aligned.get(k)
                p = ptxas.get(al["mangled"]) if al else None
                lines.append(f"| {traits} | {int(causal)} | {int(even)} | " + " | ".join(
                    cr.cell(mods[m].get(k, {}).get(c)) for m, c in (("v11", "0"), ("v11", on), ("v11x", on), ("v12f", "0"), ("v12f", on)))
                    + f" | {cr.cell(al)} | " + (f"{p['spill_st']}/{p['spill_ld']} | {p['ms'] / 1000:.1f}" if p else "- | -") + " |")
    csv_path = f"{LOGS}/build_v12f_{a.tag}_nvcc_time.csv"
    if os.path.exists(csv_path):
        lines += ["", f"## Per-TU nvcc phases (`{csv_path}`)", ""] + cr.time_table(csv_path)
    open(a.out, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
