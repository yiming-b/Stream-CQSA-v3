"""
Static side of the P2 backward (recorded, not gated; design 3.4): REG / STACK / SASS instructions of the production
backward instantiations -- v11's (every call; = v12b's LEGACY), v11x's NO_CQS build (upstream's loop with our
contract), v12b's LEGACY and v12b's ALIGNED kernel with and without the run table -- side by side, v12b's ptxas
spills and times from its -Xptxas -v build log, and the per-TU nvcc phase times of its build. torch's vendored FA-2
and flash_attn 2.8.3 are in results/ultimate/cuobjdump_v11x.md (STACK 88-112 B for the hdim64 128x128 bwd).
Reuses benchmarks/ultimate/cuobj_report.py's parsers.

    python -P benchmarks/ultimate/v12b_static.py --tag b1 --out results/ultimate/v12b_static.md
"""
import argparse, os, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cuobj_report as cr  # noqa: E402

KER = "/scratch/gpfs/AKEY/yb2807/Stream-CQSA-dev/next/kernel"
LOGS = "/scratch/gpfs/AKEY/yb2807/Stream-CQSA-dev/next/logs"
SO = {"v11": f"{KER}/stable/cqsa_cuda_next_v11.cpython-311-x86_64-linux-gnu.so",
      "v11x": f"{KER}/v11x/cqsa_cuda_next_v11x.cpython-311-x86_64-linux-gnu.so",
      "v12b": f"{KER}/v12b/cqsa_cuda_next_v12b.cpython-311-x86_64-linux-gnu.so"}
# the A100 tiles of the production backward (run_mha_bwd_hdim64/128 with >= 144 KB smem per block)
TRAITS = ("hdim64 128x128 w8 fp16", "hdim128 64x128 w8 fp16")


def aligned_rows(so):
    """{(traits, causal, even_MN, use_runs, dpsum_smem): dict(reg, stack, sass, mangled)} of v12b's ALIGNED kernel
    (b1/b2 builds have no dpsum_smem argument: it reads as False)."""
    res = cr.parse_res_usage(cr.run([cr.CUOBJDUMP, "-res-usage", so]))
    rows = {}
    for n, d in cr.demangle(n for n in res if "seqk_parallel_aligned_kernel" in n).items():
        t = re.search(r"Flash_bwd_kernel_traits<(\d+), (\d+), (\d+), (\d+),.*?(half_t|bfloat16_t)", d)
        flags = re.search(r">((?:, (?:true|false))+)>\(", d)
        if not (t and flags):
            continue
        fl = [x.strip() == "true" for x in flags.group(1).strip(", ").split(",")] + [False]
        traits = f"hdim{t.group(1)} {t.group(2)}x{t.group(3)} w{t.group(4)} {'fp16' if t.group(5) == 'half_t' else 'bf16'}"
        key = (traits, fl[0], fl[1], fl[2], fl[3])
        rows[key] = dict(reg=res[n][0], stack=res[n][1], mangled=n)
    sass = cr.sass_counts(so, [r["mangled"] for r in rows.values()])
    for r in rows.values():
        r["sass"] = sass.get(r["mangled"])
    return rows


def bwd_rows(so):
    """{(traits, causal, even_MN): row} of a module's production flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel."""
    out = {}
    for (kind, traits, fs), slot in cr.collect(so).items():
        if kind != "bwd" or traits not in TRAITS:
            continue
        flags = fs.split(",")
        out[(traits, "causal" in flags, "even_MN" in flags)] = slot.get("-")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="b1", help="v12b build tag (next/logs/06_build_v12b_<tag>.log)")
    ap.add_argument("--csv-tag", default=None, help="the build's nvcc -time csv tag (CSV_TAG of build.sh; default --tag)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    mods = {m: bwd_rows(p) for m, p in SO.items()}
    aligned = aligned_rows(SO["v12b"])
    ptxas = cr.parse_ptxas_log(open(f"{LOGS}/06_build_v12b_{a.tag}.log").read())
    lines = [f"# v12b static report (build {a.tag}, sm_80)", "",
             "Cells are REG / STACK bytes / SASS instructions of the production backward instantiations (fp16, even_K, "
             "no dropout / local / alibi / softcap). v11 = the CQS backward every CQS call runs today (v12b's LEGACY is the "
             "same source); v11x = `-DCQSA_BWD_NO_CQS` (upstream's loop with our contract, ignores the mask); ALIGNED = "
             "v12b's `flash_bwd_dq_dk_dv_loop_seqk_parallel_aligned_kernel` with the run table (CQS-on calls) and without "
             "(CQS-off calls). Upstream for comparison (cuobjdump_v11x.md): hdim64 128x128 causal even_MN torch-FA2 "
             "255/96/2336, flash_attn 2.8.3 STACK 96; hdim128 64x128 causal even_MN torch-FA2 255/64/2024.", "",
             "| traits | causal | even_MN | v11 | v11x NO_CQS | v12b LEGACY | ALIGNED runs | ALIGNED runs, dP_sum smem | ALIGNED no runs | ALIGNED no runs, dP_sum smem | ALIGNED runs spills st/ld B | ALIGNED runs ptxas s |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for traits in TRAITS:
        for causal in (True, False):
            for even in (True, False):
                k = (traits, causal, even)
                al = aligned.get(k + (True, False))
                p = ptxas.get(al["mangled"]) if al else None
                lines.append(f"| {traits} | {int(causal)} | {int(even)} | " + " | ".join(
                    cr.cell(mods[m].get(k)) for m in ("v11", "v11x", "v12b"))
                    + " | " + " | ".join(cr.cell(aligned.get(k + x)) for x in ((True, False), (True, True), (False, False), (False, True)))
                    + " | "
                    + (f"{p['spill_st']}/{p['spill_ld']} | {p['ms'] / 1000:.1f}" if p else "- | -") + " |")
    csv_path = f"{LOGS}/build_v12b_{a.csv_tag or a.tag}_nvcc_time.csv"
    if os.path.exists(csv_path):
        lines += ["", f"## Per-TU nvcc phases (`{csv_path}`)", ""] + cr.time_table(csv_path)
    open(a.out, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
