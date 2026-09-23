"""
cuobjdump / ptxas report for the kernel modules of the ultimate program: resource usage (REG/STACK) and SASS
instruction count of every forward / backward instantiation side by side across modules, plus, for a module
built with `-Xptxas -v` and `-time`, its spill bytes, per-instantiation ptxas time and per-TU nvcc phase times.

    python -P benchmarks/ultimate/cuobj_report.py \
        --so v11=next/kernel/stable/cqsa_cuda_next_v11.cpython-311-x86_64-linux-gnu.so \
        --so v11x=next/kernel/v11x/cqsa_cuda_next_v11x.cpython-311-x86_64-linux-gnu.so \
        --so v9=next/kernel/stable/cqsa_cuda_next_v9.cpython-311-x86_64-linux-gnu.so \
        --so torch-FA2=.../torch/lib/libtorch_cuda.so --res-usage-cache torch-FA2=<saved -res-usage dump> \
        --res-usage-log fa2.8.3=next/logs/01_cuobjdump_upstream_fa2.txt \
        --ptxas-log v11x=next/logs/04_build_v11x.log --time-csv next/logs/build_v11x_nvcc_time.csv \
        --out results/ultimate/cuobjdump_v11x.md

Rows are keyed by (kernel, traits, upstream FA-2's template flags), so one instantiation is matched across
modules; a module's `Cqs_mode` template argument (v10/v11: 0 off, 1 on, 2 runtime) becomes a sub-column, which
puts a CQS-on instantiation next to its CQS-off twin and next to upstream: torch's vendored FA-2 (namespace
pytorch_flash, the FA-2 arm of every measurement; 566 MB, so its -res-usage dump is cached) and a saved
-res-usage of the flash_attn 2.8.3 wheel (REG/STACK only, no binary at hand). Kernels without the argument
(v9, upstream) fill one sub-column. By default only the production instantiations are listed (fp16, even_K,
no dropout / local / alibi / softcap / return_softmax: none of these is ever used with CQS); --all lists every one.
"""
import argparse, csv, os, re, subprocess, sys, time
from collections import defaultdict

CUOBJDUMP = os.environ.get("CUOBJDUMP", "/usr/local/cuda-13.0/bin/cuobjdump")
ARCH = "sm_80"
KERNELS = {"flash_fwd_kernel": "fwd", "flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel": "bwd"}
# upstream FA-2's template flags after the traits type; v10/v11 append Cqs_mode to the forward's
FLAGS = {"fwd": ("dropout", "causal", "local", "alibi", "even_MN", "even_K", "softcap", "ret_softmax"),
         "bwd": ("dropout", "causal", "local", "alibi", "even_MN", "even_K", "softcap")}
EXOTIC = ("dropout", "local", "alibi", "softcap", "ret_softmax")


def run(args):
    return subprocess.run(args, capture_output=True, text=True, check=True).stdout


def demangle(names):
    names = list(names)
    return dict(zip(names, run(["c++filt"] + names).splitlines())) if names else {}


def parse_res_usage(text):
    """{mangled: (REG, STACK)} of the ARCH section(s) of a `cuobjdump -res-usage` dump."""
    cur = fn = None
    res = {}
    for line in text.splitlines():
        m = re.match(r"arch = (\S+)", line)
        if m:
            cur = m.group(1)
            continue
        m = re.match(r"\s*Function (\S+):", line)
        if m:
            fn = m.group(1)
            continue
        m = re.match(r"\s*REG:(\d+) STACK:(\d+)", line)
        if m and cur == ARCH and fn is not None:
            res[fn] = (int(m.group(1)), int(m.group(2)))
    return res


def sass_counts(so, funcs):
    """{mangled: SASS instruction count}; one `-fun a,b,...` call, so a big library is read once."""
    if not funcs:
        return {}
    p = subprocess.Popen([CUOBJDUMP, "-sass", "-arch", ARCH, "-fun", ",".join(funcs), so],
                         stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    fn, n, out = None, 0, {}
    for line in p.stdout:
        m = re.match(r"\s*Function : (\S+)", line)
        if m:
            if fn is not None:
                out[fn] = n
            fn, n = m.group(1), 0
        elif re.match(r"\s*/\*[0-9a-f]{4,}\*/", line):        # the address comment: one per instruction
            n += 1
    if fn is not None:
        out[fn] = n
    p.wait()
    return out


def parse_ptxas_log(text):
    """{mangled: dict(reg, stack, spill_st, spill_ld, ms)} from a build log made with `-Xptxas -v`."""
    out, cur = {}, None
    for line in text.splitlines():
        m = re.search(r"Compiling entry function '(\S+)' for '(\S+)'", line)
        if m:
            cur = {"fn": m.group(1)} if m.group(2) == ARCH else None
            continue
        if cur is None:
            continue
        m = re.search(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads", line)
        if m:
            cur.update(stack=int(m.group(1)), spill_st=int(m.group(2)), spill_ld=int(m.group(3)))
            continue
        m = re.search(r"Used (\d+) registers", line)
        if m:
            cur["reg"] = int(m.group(1))
            continue
        m = re.search(r"Compile time = ([\d.]+) ms", line)
        if m:
            cur["ms"] = float(m.group(1))
            out[cur.pop("fn")] = cur
            cur = None
    return out


def parse_key(demangled):
    """(kind, traits, flags, cqs) of a main-kernel instantiation, else None. `cqs` is the Cqs_mode argument
    ("0"/"1"/"2") or "-" for a kernel without one (v9, upstream)."""
    m = re.match(r"void (?:\w+::)?(\w+)<(.*)>\(", demangled)
    if not m or m.group(1) not in KERNELS:
        return None
    kind, args = KERNELS[m.group(1)], m.group(2)
    t = re.search(r"Flash_(?:fwd|bwd)_kernel_traits<(\d+), (\d+), (\d+), (\d+),", args)
    if not t:
        return None
    dtype = "fp16" if "half_t" in args else ("bf16" if "bfloat16_t" in args else "?")
    tail = [x.strip() for x in args[args.rfind(">") + 1:].strip(", ").split(",")]     # the flags after the traits
    names = FLAGS[kind]
    if len(tail) == len(names) + 1 and tail[-1].isdigit():
        cqs, tail = tail[-1], tail[:-1]
    elif len(tail) == len(names):
        cqs = "-"
    else:
        return None
    flags = {k: v == "true" for k, v in zip(names, tail)}
    return kind, f"hdim{t.group(1)} {t.group(2)}x{t.group(3)} w{t.group(4)} {dtype}", flags, cqs


def production(traits, flags):
    return traits.endswith("fp16") and flags["even_K"] and not any(flags.get(k) for k in EXOTIC)


def flag_str(kind, flags):
    return ",".join(k for k in FLAGS[kind] if flags[k]) or "-"


def collect(so, res_text=None, everything=False):
    """{(kind, traits, flags): {cqs: dict(reg, stack, sass, mangled)}} of one module. `so` None = a saved
    -res-usage dump only (no SASS)."""
    res = parse_res_usage(res_text if res_text is not None else run([CUOBJDUMP, "-res-usage", so]))
    keyed = {}
    for n, d in demangle(n for n in res if any(k in n for k in KERNELS)).items():
        key = parse_key(d)
        if key is None:
            continue
        kind, traits, flags, cqs = key
        if everything or production(traits, flags):
            keyed[n] = (kind, traits, flag_str(kind, flags), cqs)
    sass = sass_counts(so, list(keyed)) if so else {}
    rows = defaultdict(dict)
    for n, (kind, traits, fs, cqs) in keyed.items():
        slot = rows[(kind, traits, fs)]
        if cqs in slot:      # traits fields beyond the block shape (Is_Q_in_regs, ...) differ: keep both visible
            cqs = f"{cqs}'"
        slot[cqs] = dict(reg=res[n][0], stack=res[n][1], sass=sass.get(n), mangled=n)
    return dict(rows)


def cell(r):
    return "-" if r is None else f"{r['reg']}/{r['stack']}/{r['sass'] if r['sass'] is not None else '?'}"


def instantiation_table(mods):
    cols = [(m, c) for m in mods for c in sorted({c for r in mods[m].values() for c in r})]
    keys = sorted({k for r in mods.values() for k in r})
    lines = ["| kind | traits | flags | " + " | ".join(f"{m} cqs={c}" if c != "-" else m for m, c in cols) + " |",
             "|---|---|---|" + "---|" * len(cols)]
    for k in keys:
        lines.append(f"| {k[0]} | {k[1]} | {k[2]} | " + " | ".join(cell(mods[m].get(k, {}).get(c)) for m, c in cols) + " |")
    return lines


def ptxas_table(rows, ptxas):
    lines = ["| kind | traits | flags | cqs | REG | STACK B | spill stores/loads B | ptxas s |", "|---|---|---|---|---|---|---|---|"]
    for k in sorted(rows):
        for cqs, r in sorted(rows[k].items()):
            p = ptxas.get(r["mangled"])
            if p is None:
                continue
            lines.append(f"| {k[0]} | {k[1]} | {k[2]} | {cqs} | {p['reg']} | {p['stack']} | {p['spill_st']}/{p['spill_ld']} | {p['ms'] / 1000:.1f} |")
    return lines


def time_table(path):
    """Per-TU nvcc phase seconds from an `nvcc -time` csv (one row per phase and TU; the driver row has no source)."""
    per = defaultdict(lambda: defaultdict(float))
    with open(path) as fh:
        for row in csv.reader(fh):
            if len(row) < 7 or not row[0].strip() or row[0].strip() == "source file name":
                continue
            phase = row[1].strip()
            per[os.path.basename(row[0].strip())][phase if phase in ("cicc", "ptxas") else "other"] += float(row[6]) / 1000
    lines = ["| TU | cicc s | ptxas s | other s | total s |", "|---|---|---|---|---|"]
    for tu in sorted(per):
        p = per[tu]
        lines.append(f"| {tu} | {p['cicc']:.1f} | {p['ptxas']:.1f} | {p['other']:.1f} | {sum(p.values()):.1f} |")
    return lines


def named(specs):
    return dict(s.split("=", 1) for s in specs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--so", action="append", default=[], help="name=path of a module (.so, or a library) to dump")
    ap.add_argument("--res-usage-cache", action="append", default=[], help="name=path: saved `-res-usage` of that --so, used instead of re-running it")
    ap.add_argument("--res-usage-log", action="append", default=[], help="name=path: saved `-res-usage` of a module whose binary is not at hand")
    ap.add_argument("--ptxas-log", action="append", default=[], help="name=path: that module's build log made with -Xptxas -v")
    ap.add_argument("--time-csv", default=None, help="the `nvcc -time` csv of a build: per-TU phase times")
    ap.add_argument("--all", action="store_true", help="every instantiation, not only the production ones")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    caches = named(a.res_usage_cache)
    mods, sizes = {}, {}
    for name, path in named(a.so).items():
        t0 = time.time()
        mods[name] = collect(path, open(caches[name]).read() if name in caches else None, a.all)
        sizes[name] = f"{os.path.getsize(path) / 2**20:.0f} MB"
        print(f"{name}: {sum(len(r) for r in mods[name].values())} instantiations in {time.time() - t0:.0f} s", file=sys.stderr)
    for name, path in named(a.res_usage_log).items():
        mods[name] = collect(None, open(path).read(), a.all)
        sizes[name] = "res-usage log"
        print(f"{name}: {sum(len(r) for r in mods[name].values())} instantiations (REG/STACK only)", file=sys.stderr)
    lines = [f"# cuobjdump / ptxas report ({time.strftime('%Y-%m-%d %H:%M')}, {ARCH})", "",
             "Modules: " + "; ".join(f"`{n}` = `{p}` ({sizes[n]})" for n, p in {**named(a.so), **named(a.res_usage_log)}.items()), "",
             "Cells are REG/STACK bytes/SASS instructions (`?` = no binary). " + ("All instantiations." if a.all else
             "Production instantiations only (fp16, even_K, no dropout/local/alibi/softcap/return_softmax)."), ""]
    if a.time_csv:
        lines += [f"## Build phases per TU (`{a.time_csv}`)", ""] + time_table(a.time_csv) + [""]
    lines += ["## Instantiations", ""] + instantiation_table(mods) + [""]
    for name, path in named(a.ptxas_log).items():
        lines += [f"## ptxas -v: {name} (`{path}`)", ""] + ptxas_table(mods[name], parse_ptxas_log(open(path).read())) + [""]
    with open(a.out, "w") as fh:
        fh.write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
