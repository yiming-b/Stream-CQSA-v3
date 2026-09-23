"""
M0.3: stall attribution of the all-live tax on the 128-ALIGNED layout, under Nsight Compute.

Four arms, ONE kernel launch each, one process per arm (one kernel module per process, as
benchmarks/ncu_one.py), on the c=7 itr=1 aligned_128 task built exactly as benchmarks/layout_ab.py
builds it (build_tasks(..., seg_align=128), gathered inputs, H=8, D=64, fp16, causal):
    fa2    torch SDPA with the FLASH_ATTENTION backend forced (flash_attn is not installed; SDPA-flash
           is ~5% slower than the flash_attn package, results/kernel_ab_clean_v11.json) at the task's L
    plain  the v11 module with CQS off (bits=None -> Cqs_mode 0, the upstream FA-2 loop)
    zero   v11 CQS on, all-zero group bits (every tile live: the all-live tax)
    real   v11 CQS on, the task's real CQS bits (22% masked tiles skipped)

    run        python -P ncu_aligned.py run --arm zero [--N 131072] [--c 7] [--path 2] [--row-json f]
               (wrap in ncu: --nvtx --nvtx-include "measure/" -k regex:flash_fwd_kernel -c 1)
    summarize  python -P ncu_aligned.py summarize --dir results/ultimate/ncu_aligned --out ncu_aligned.md
               (reads <dir>/<arm>.ncu-rep through `ncu -i`, no GPU needed; also writes <dir>/summary.json)

SourceCounters caveat: the shipped .so files carry no -lineinfo, so the source view is SASS only
(per-instruction stall samples, no CUDA line correlation). The summary therefore reports the top
SASS instructions by stall samples and a per-opcode-class aggregation of the samples, which is what
attributes the tax to search loads / softmax / barrier / MMA / issue.
"""
import os, sys, csv, io, re, json, socket, argparse, subprocess

NCU = os.environ.get("NCU", "/usr/local/cuda-13.0/bin/ncu")
ARMS = ("fa2", "plain", "zero", "real")
H, D = 8, 64
SEG_ALIGN = 128


# ============================================================================ run: one launch per arm
def run(args):
    os.environ.setdefault("CQSA_CUDA_MODULE", "cqsa_cuda_next_v11")
    os.environ.setdefault("CQSA_CUDA_MODULE_NONCAUSAL", "cqsa_cuda_next_v9")
    import torch
    import torch.nn.functional as F
    from torch.nn.attention import sdpa_kernel, SDPBackend
    from stream_cqsa.interface import (flash_attn_func, flash_attn_func_cqs_group_bits, cqs_block_summaries,
                                       CQS_BLK_SIZE, CQSA_CUDA_MODULE, cqsa_cuda)
    from stream_cqsa.stable_stream import build_tasks

    assert cqsa_cuda is not None, f"{CQSA_CUDA_MODULE} did not import"
    dev = torch.device("cuda")
    dt = torch.float16
    scale = D ** -0.5

    tasks = build_tasks(args.N, 1, B=1, H=H, D=D, itemsize=2, sorted_gather=True, pin=False,
                        c=args.c, interest_set=(0, 1, 3), seg_align=SEG_ALIGN)
    t = tasks[args.path]
    L = int(t.local_size)
    row = dict(arm=args.arm, N=args.N, c=args.c, path=list(t.path), L=L, L_mod_128=L % 128, H=H, D=D, dtype="float16",
               causal=True, layout="aligned_128", seg_align=SEG_ALIGN, inputs="gathered",
               cuda_module=CQSA_CUDA_MODULE, torch=torch.__version__, node=socket.gethostname(),
               slurm_job=os.environ.get("SLURM_JOB_ID"), gpu=torch.cuda.get_device_name(0))
    try:
        row["nvidia_smi"] = subprocess.run(["nvidia-smi", "--query-gpu=name,clocks.sm,clocks.max.sm", "--format=csv,noheader"],
                                           capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as e:  # pragma: no cover
        row["nvidia_smi"] = f"unavailable: {e!r}"

    if args.arm == "fa2":
        torch.manual_seed(1)
        q, k, v = (torch.randn(1, H, L, D, device=dev, dtype=dt) for _ in range(3))
        ctx = sdpa_kernel(SDPBackend.FLASH_ATTENTION)
        fn = lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=scale)
    else:
        torch.manual_seed(0)
        qt, kt, vt = (torch.randn(1, args.N, H, D, device=dev, dtype=dt) for _ in range(3))   # token-major, as layout_ab
        idx = t.token_ids.to(dev)
        q, k, v = (x.index_select(1, idx).contiguous() for x in (qt, kt, vt))
        del qt, kt, vt
        import contextlib
        ctx = contextlib.nullcontext()
        if args.arm == "plain":
            fn = lambda: flash_attn_func(q, k, v, softmax_scale=scale, causal=True)
        else:
            if args.arm == "real":
                bits = t.group_bits.to(dev); bo = t.extra["blk_or"].to(dev); ba = t.extra["blk_and"].to(dev)
            else:
                bits = torch.zeros(L, dtype=torch.int64, device=dev)
                bo, ba = (x.to(dev) for x in cqs_block_summaries(bits, CQS_BLK_SIZE))
            # 64x64 lower-triangular tile census from the summaries (same definition as layout_ab.tile_census)
            import numpy as np
            o = bo.cpu().numpy().astype(np.int64); a = ba.cpu().numpy().astype(np.int64)
            nb = (L + CQS_BLK_SIZE - 1) // CQS_BLK_SIZE
            tri = np.tril(np.ones((nb, nb), dtype=bool))
            masked = (np.bitwise_and.outer(a[:nb], a[:nb]) != 0) & tri
            clear = (np.bitwise_and.outer(o[:nb], o[:nb]) == 0) & tri
            n = int(tri.sum())
            row["census"] = dict(n_tiles64=n, masked=int(masked.sum()) / n, clear=int(clear.sum()) / n,
                                 mixed=int((tri & ~masked & ~clear).sum()) / n)
            fn = lambda: flash_attn_func_cqs_group_bits(q, k, v, bits, softmax_scale=scale, causal=True,
                                                        cqs_blk_or=bo, cqs_blk_and=ba, cqs_blk_size=CQS_BLK_SIZE,
                                                        fp32_out=True)
    with ctx:
        fn(); torch.cuda.synchronize()                       # warm-up (module load, allocator), outside the range
        torch.cuda.nvtx.range_push("measure")
        fn(); torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()
    row["device_mem_peak_GiB"] = torch.cuda.max_memory_allocated() / 2**30
    print(json.dumps(row), flush=True)
    if args.row_json:
        with open(args.row_json, "w") as f:
            json.dump(row, f, indent=1)


# ============================================================================ summarize: report -> tables
def _num(s):
    try:
        return float(str(s).replace(",", ""))
    except ValueError:
        return None


def _ncu_csv(rep, *page_args):
    out = subprocess.run([NCU, "-i", rep, "--csv", "--print-units", "base", *page_args], capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"ncu -i {rep} {' '.join(page_args)} failed:\n{out.stderr[-2000:]}")
    return out.stdout


def raw_metrics(rep):
    """{metric_name: value} for the (single) kernel result of the report, from the raw page."""
    text = _ncu_csv(rep, "--page", "raw")
    lines = [l for l in text.splitlines() if l.strip()]
    start = next(i for i, l in enumerate(lines) if l.startswith('"ID"'))
    rows = list(csv.reader(io.StringIO("\n".join(lines[start:]))))
    names, units, vals = rows[0], rows[1], rows[2]
    metrics = {}
    for n, u, v in zip(names, units, vals):
        x = _num(v)
        metrics[n] = x if x is not None else v
    return metrics


def sass_table(rep):
    """List of dict rows (one per SASS instruction) from the source page, SASS view."""
    text = _ncu_csv(rep, "--page", "source", "--print-source", "sass")
    lines = text.splitlines()
    start = next((i for i, l in enumerate(lines) if l.startswith('"# Address"') or l.startswith('"Address"')), None)
    if start is None:
        return [], text[:2000]
    rows = list(csv.DictReader(io.StringIO("\n".join(lines[start:]))))
    return rows, None


# Opcode classes used to attribute stall samples (SASS mnemonics, sm80).
OPCLASS = [
    ("mma",       re.compile(r"^HMMA")),
    ("smem_ld",   re.compile(r"^(LDSM|LDS)\b")),
    ("smem_st",   re.compile(r"^STS\b")),
    ("cp_async",  re.compile(r"^(LDGSTS|LDGDEPBAR|DEPBAR)\b")),
    ("global_ld", re.compile(r"^LDG\b")),
    ("global_st", re.compile(r"^(STG|RED|ATOM)\b")),
    ("local",     re.compile(r"^(LDL|STL)\b")),
    ("barrier",   re.compile(r"^(BAR|MEMBAR|ERRBAR)\b")),
    ("branch",    re.compile(r"^(BRA|BRX|BSSY|BSYNC|EXIT|RET|CALL|JMP|WARPSYNC)\b")),
    ("exp_mufu",  re.compile(r"^MUFU")),
    ("fp32",      re.compile(r"^(FADD|FMUL|FFMA|FMNMX|FSEL|FSETP|FMIN|FMAX|FRND|FCHK)\b")),
    ("convert",   re.compile(r"^(F2F|F2FP|I2F|F2I|I2I|PRMT|SHF|SHFL)\b")),
    ("int",       re.compile(r"^(IADD3|IMAD|LOP3|ISETP|LEA|SEL|IABS|IMNMX|POPC|FLO|BFE|VOTE|PLOP3|P2R|R2P|MOV|CS2R|S2R|ULDC|UIADD3|UMOV|ULOP3|USEL|UISETP|ULEA|R2UR|S2UR|UFLO|UPOPC|USHF|UPRMT|UBFE|UPLOP3|VOTEU)\b")),
    ("nop",       re.compile(r"^NOP\b")),
]


def opclass(src):
    m = re.match(r"\s*(?:@!?U?P\w+\s+)?([A-Z0-9_.]+)", src)
    if not m:
        return "other"
    op = m.group(1)
    for name, rx in OPCLASS:
        if rx.match(op):
            return name
    return "other"


STALL_RX = re.compile(r"^smsp__average_warps_issue_stalled_(\w+)_per_issue_active\.ratio$")
PCSAMP_RX = re.compile(r"^smsp__pcsamp_warps_issue_stalled_(\w+)$")


def pick(metrics, *names):
    for n in names:
        if n in metrics and isinstance(metrics[n], float):
            return metrics[n]
    return None


def summarize_report(rep):
    m = raw_metrics(rep)
    s = dict(report=rep)
    s["kernel"] = m.get("Kernel Name")
    s["device"] = m.get("Device")
    s["duration_ms"] = (pick(m, "gpu__time_duration.sum") or 0) / 1e6
    s["sm_freq_GHz"] = (pick(m, "sm__cycles_elapsed.avg.per_second") or 0) / 1e9
    s["elapsed_cycles"] = pick(m, "gpc__cycles_elapsed.max", "sm__cycles_elapsed.max")
    s["sm_active_cycles"] = pick(m, "sm__cycles_active.avg")
    s["inst_executed"] = pick(m, "smsp__inst_executed.sum", "sm__inst_executed.sum")
    s["inst_issued"] = pick(m, "smsp__inst_issued.sum")
    s["warp_cycles_per_issued_inst"] = pick(m, "smsp__average_warp_latency_per_inst_issued.ratio")
    s["warp_cycles_per_executed_inst"] = pick(m, "smsp__average_warps_active_per_inst_executed.ratio")
    s["eligible_warps_per_scheduler"] = pick(m, "smsp__warps_eligible.avg.per_cycle_active")
    s["active_warps_per_scheduler"] = pick(m, "smsp__warps_active.avg.per_cycle_active")
    s["issued_per_scheduler_per_cycle"] = pick(m, "smsp__issue_active.avg.per_cycle_active")
    s["issue_slot_util_pct"] = pick(m, "smsp__issue_active.avg.pct_of_peak_sustained_active")
    s["tensor_pipe_active_pct"] = pick(m, "sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active",
                                       "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active",
                                       "sm__inst_executed_pipe_tensor_op_hmma.avg.pct_of_peak_sustained_active")
    s["sm_throughput_pct"] = pick(m, "sm__throughput.avg.pct_of_peak_sustained_elapsed")
    s["registers_per_thread"] = pick(m, "launch__registers_per_thread")
    s["stack_size"] = pick(m, "launch__stack_size")
    s["smem_dynamic_per_block"] = pick(m, "launch__shared_mem_per_block_dynamic")
    s["grid_size"] = pick(m, "launch__grid_size")
    s["waves_per_sm"] = pick(m, "launch__waves_per_multiprocessor")
    s["occupancy_theoretical_pct"] = pick(m, "sm__maximum_warps_per_active_cycle_pct")
    s["occupancy_achieved_pct"] = pick(m, "sm__warps_active.avg.pct_of_peak_sustained_active")
    s["occupancy_limit_registers_blocks"] = pick(m, "launch__occupancy_limit_registers")
    s["occupancy_limit_smem_blocks"] = pick(m, "launch__occupancy_limit_shared_mem")
    s["l1_hit_pct"] = pick(m, "l1tex__t_sector_hit_rate.pct")
    s["l2_hit_pct"] = pick(m, "lts__t_sector_hit_rate.pct")
    s["dram_throughput_pct"] = pick(m, "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
                                    "dram__throughput.avg.pct_of_peak_sustained_elapsed")
    s["l1_throughput_pct"] = pick(m, "l1tex__throughput.avg.pct_of_peak_sustained_active")
    s["l2_throughput_pct"] = pick(m, "lts__throughput.avg.pct_of_peak_sustained_elapsed")
    s["local_ld_bytes"] = pick(m, "l1tex__t_bytes_pipe_lsu_mem_local_op_ld.sum")
    s["local_st_bytes"] = pick(m, "l1tex__t_bytes_pipe_lsu_mem_local_op_st.sum")
    s["global_ld_bytes"] = pick(m, "l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum")
    s["global_ld_requests"] = pick(m, "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum")
    s["branch_uniform_pct"] = pick(m, "smsp__sass_average_branch_targets_threads_uniform.pct")
    s["threads_not_predicated_off_per_warp"] = pick(m, "smsp__thread_inst_executed_per_inst_executed.ratio")
    # stall breakdown: warp cycles per issued instruction split by reason (sums to warp_cycles_per_issued_inst)
    stalls = {STALL_RX.match(k).group(1): v for k, v in m.items() if STALL_RX.match(k) and isinstance(v, float)}
    s["stall_cycles_per_issue"] = dict(sorted(stalls.items(), key=lambda kv: -kv[1]))
    pcs = {PCSAMP_RX.match(k).group(1): v for k, v in m.items() if PCSAMP_RX.match(k) and isinstance(v, float)}
    s["pcsamp_by_reason"] = dict(sorted(pcs.items(), key=lambda kv: -kv[1]))
    pipes = {k: v for k, v in m.items() if k.startswith("sm__inst_executed_pipe_") and k.endswith(".avg.pct_of_peak_sustained_active") and isinstance(v, float)}
    s["pipe_util_pct"] = dict(sorted(pipes.items(), key=lambda kv: -kv[1])[:8])
    # SASS source view
    rows, err = sass_table(rep)
    s["sass_error"] = err
    if rows:
        cols = rows[0].keys()
        col_all = next((c for c in cols if "Stall Sampling (All" in c), None)
        col_ni = next((c for c in cols if "Not-issued" in c or "Not Issued" in c), None)
        col_ie = next((c for c in cols if c.startswith("Instructions Executed")), None)
        col_src = next((c for c in cols if c == "Source"), None)
        col_addr = next((c for c in cols if "Address" in c), None)
        reason_cols = [c for c in cols if c.startswith("stall_") or ("Stall" in c and c not in (col_all, col_ni))]
        s["sass_columns"] = list(cols)
        s["sass_n_instructions"] = len(rows)
        tot_all = sum((_num(r.get(col_all)) or 0) for r in rows) if col_all else 0
        tot_ni = sum((_num(r.get(col_ni)) or 0) for r in rows) if col_ni else 0
        tot_ie = sum((_num(r.get(col_ie)) or 0) for r in rows) if col_ie else 0
        s["sass_samples_all"], s["sass_samples_not_issued"], s["sass_inst_executed"] = tot_all, tot_ni, tot_ie
        byclass = {}
        for r in rows:
            c = opclass(r.get(col_src, ""))
            d = byclass.setdefault(c, dict(n_static=0, samples_all=0.0, samples_not_issued=0.0, inst_executed=0.0))
            d["n_static"] += 1
            d["samples_all"] += _num(r.get(col_all)) or 0
            d["samples_not_issued"] += _num(r.get(col_ni)) or 0
            d["inst_executed"] += _num(r.get(col_ie)) or 0
        for d in byclass.values():
            d["samples_all_pct"] = 100 * d["samples_all"] / tot_all if tot_all else None
            d["inst_executed_pct"] = 100 * d["inst_executed"] / tot_ie if tot_ie else None
        s["sass_by_opclass"] = dict(sorted(byclass.items(), key=lambda kv: -kv[1]["samples_all"]))
        top = sorted(rows, key=lambda r: -(_num(r.get(col_all)) or 0))[:10]
        s["sass_top10"] = [dict(address=r.get(col_addr), source=r.get(col_src),
                                samples_all=_num(r.get(col_all)), samples_all_pct=100 * (_num(r.get(col_all)) or 0) / tot_all if tot_all else None,
                                samples_not_issued=_num(r.get(col_ni)) if col_ni else None,
                                inst_executed=_num(r.get(col_ie)) if col_ie else None,
                                reasons={c: _num(r.get(c)) for c in reason_cols if _num(r.get(c))})
                           for r in top]
    return s


def fmt(x, nd=2):
    if x is None:
        return "n/a"
    if isinstance(x, float):
        return f"{x:,.{nd}f}" if abs(x) < 1e6 else f"{x:,.0f}"
    return str(x)


def summarize(args):
    rows = {}
    for arm in ARMS:
        rep = os.path.join(args.dir, f"{arm}.ncu-rep")
        if not os.path.exists(rep):
            rows[arm] = dict(error=f"{rep} missing")
            continue
        try:
            rows[arm] = summarize_report(rep)
        except Exception as e:  # noqa
            rows[arm] = dict(error=f"{type(e).__name__}: {e}")
        rj = os.path.join(args.dir, f"{arm}.row.json")
        if os.path.exists(rj):
            rows[arm]["run_row"] = json.load(open(rj))
    with open(os.path.join(args.dir, "summary.json"), "w") as f:
        json.dump(rows, f, indent=1)
    write_md(rows, args.out, args.dir)
    print(f"wrote {args.out}")


def write_md(rows, out, d):
    ok = [a for a in ARMS if "error" not in rows[a]]
    L = []
    L.append("# M0.3: ncu stall attribution of the all-live tax on the aligned layout\n")
    r0 = rows[ok[0]].get("run_row", {}) if ok else {}
    L.append(f"Task: c={r0.get('c')} itr=1 path={r0.get('path')} N={r0.get('N')} L={r0.get('L')} (L%128={r0.get('L_mod_128')}), "
             f"aligned_128 layout, gathered inputs, H={H} D={D} fp16 causal. One kernel launch per arm, one process per arm, "
             f"`ncu --set full --section SourceCounters --section WarpStateStats --clock-control base`.\n")
    L.append(f"GPU: {r0.get('gpu')} ({r0.get('nvidia_smi')}), node {r0.get('node')}, slurm job {r0.get('slurm_job')}, "
             f"torch {r0.get('torch')}, module {r0.get('cuda_module')}. PCIe node: relative metrics only, never compare "
             f"durations with SXM timings.\n")
    L.append("FA-2 arm = torch SDPA FLASH_ATTENTION backend (flash_attn not installed); SDPA-flash is ~5% slower than the "
             "flash_attn package. Arms: plain = v11 module with CQS off (bits=None, upstream loop); zero = CQS on, all-zero "
             "bits (all tiles live); real = CQS on, task bits (22% masked tiles skipped).\n")
    for a in ARMS:
        if "error" in rows[a]:
            L.append(f"- **{a}: {rows[a]['error']}**")
    L.append("\n## Per-arm metrics\n")
    keys = [("kernel", "kernel"), ("duration_ms", "duration ms (PCIe, base clock)"), ("sm_freq_GHz", "SM clock GHz"),
            ("elapsed_cycles", "elapsed cycles"), ("inst_executed", "executed instructions"), ("inst_issued", "issued instructions"),
            ("warp_cycles_per_issued_inst", "warp cycles / issued inst"), ("eligible_warps_per_scheduler", "eligible warps / scheduler"),
            ("active_warps_per_scheduler", "active warps / scheduler"), ("issued_per_scheduler_per_cycle", "issued / scheduler / cycle"),
            ("issue_slot_util_pct", "issue slots busy %"), ("tensor_pipe_active_pct", "tensor pipe active %"),
            ("sm_throughput_pct", "SM throughput %"), ("registers_per_thread", "registers / thread"), ("stack_size", "stack B"),
            ("smem_dynamic_per_block", "dyn smem / block B"), ("grid_size", "grid"), ("waves_per_sm", "waves / SM"),
            ("occupancy_theoretical_pct", "occupancy theoretical %"), ("occupancy_achieved_pct", "occupancy achieved %"),
            ("l1_hit_pct", "L1 hit %"), ("l2_hit_pct", "L2 hit %"), ("dram_throughput_pct", "DRAM throughput %"),
            ("global_ld_bytes", "global load bytes"), ("global_ld_requests", "global load requests"),
            ("local_ld_bytes", "local load bytes"), ("local_st_bytes", "local store bytes"),
            ("branch_uniform_pct", "branch targets uniform %"), ("threads_not_predicated_off_per_warp", "not-pred-off threads / warp")]
    L.append("| metric | " + " | ".join(ok) + " |")
    L.append("|---|" + "---|" * len(ok))
    for k, label in keys:
        vals = [rows[a].get(k) for a in ok]
        if k == "kernel":
            vals = [str(v)[:90] + ("..." if v and len(str(v)) > 90 else "") for v in vals]
        L.append(f"| {label} | " + " | ".join(fmt(v) for v in vals) + " |")
    L.append("\n## Stall breakdown (warp cycles per issued instruction, by reason; sums to the row above)\n")
    reasons = sorted({r for a in ok for r in rows[a]["stall_cycles_per_issue"]},
                     key=lambda r: -max(rows[a]["stall_cycles_per_issue"].get(r, 0) for a in ok))
    L.append("| reason | " + " | ".join(f"{a} cyc (% of total)" for a in ok) + " |")
    L.append("|---|" + "---|" * len(ok))
    for r in reasons:
        cells = []
        for a in ok:
            v = rows[a]["stall_cycles_per_issue"].get(r, 0.0)
            tot = rows[a]["warp_cycles_per_issued_inst"] or sum(rows[a]["stall_cycles_per_issue"].values())
            cells.append(f"{v:.2f} ({100 * v / tot:.1f}%)" if tot else f"{v:.2f}")
        L.append(f"| {r} | " + " | ".join(cells) + " |")
    L.append("\n## Pipe utilisation (% of peak sustained active, top pipes)\n")
    pipes = sorted({p for a in ok for p in rows[a]["pipe_util_pct"]})
    L.append("| pipe | " + " | ".join(ok) + " |")
    L.append("|---|" + "---|" * len(ok))
    for p in pipes:
        L.append(f"| {p.replace('sm__inst_executed_pipe_', '').replace('.avg.pct_of_peak_sustained_active', '')} | "
                 + " | ".join(fmt(rows[a]["pipe_util_pct"].get(p)) for a in ok) + " |")
    L.append("\n## SASS stall samples by opcode class (SourceCounters, no -lineinfo: SASS view only)\n")
    L.append("Columns: % of all warp-stall samples in the kernel, then executed warp-instructions of that class in % of the kernel's total.\n")
    classes = sorted({c for a in ok for c in rows[a].get("sass_by_opclass", {})},
                     key=lambda c: -max(rows[a].get("sass_by_opclass", {}).get(c, {}).get("samples_all", 0) for a in ok))
    L.append("| class | " + " | ".join(f"{a} samples% / inst%" for a in ok) + " |")
    L.append("|---|" + "---|" * len(ok))
    for c in classes:
        cells = []
        for a in ok:
            d = rows[a].get("sass_by_opclass", {}).get(c)
            cells.append(f"{d['samples_all_pct']:.1f} / {d['inst_executed_pct']:.1f}" if d and d["samples_all_pct"] is not None else "n/a")
        L.append(f"| {c} | " + " | ".join(cells) + " |")
    L.append("| static SASS instructions | " + " | ".join(fmt(rows[a].get("sass_n_instructions")) for a in ok) + " |")
    L.append("| total stall samples | " + " | ".join(fmt(rows[a].get("sass_samples_all")) for a in ok) + " |")
    for a in ok:
        if rows[a].get("sass_error"):
            L.append(f"\n{a}: source page unavailable: {rows[a]['sass_error'][:500]}")
    for a in ok:
        if not rows[a].get("sass_top10"):
            continue
        L.append(f"\n## Top 10 SASS instructions by stall samples: {a}\n")
        L.append("| address | SASS | samples (% of all) | not-issued samples | executed | top reasons |")
        L.append("|---|---|---|---|---|---|")
        for r in rows[a]["sass_top10"]:
            reasons_s = ", ".join(f"{k}={fmt(v, 0)}" for k, v in sorted(r["reasons"].items(), key=lambda kv: -kv[1])[:3])
            L.append(f"| {r['address']} | `{r['source']}` | {fmt(r['samples_all'], 0)} ({fmt(r['samples_all_pct'], 1)}%) | "
                     f"{fmt(r['samples_not_issued'], 0)} | {fmt(r['inst_executed'], 0)} | {reasons_s} |")
    L.append("\n## Verdict\n")
    L.append("(written by hand after reading the tables; see the bottom of this file)\n")
    with open(out, "w") as f:
        f.write("\n".join(L) + "\n")


# ============================================================================ main
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    pr = sub.add_parser("run")
    pr.add_argument("--arm", choices=ARMS, required=True)
    pr.add_argument("--N", type=int, default=131072)
    pr.add_argument("--c", type=int, default=7)
    pr.add_argument("--path", type=int, default=2, help="task index (path) among the c itr=1 tasks; 2 -> L=56064 at N=131072")
    pr.add_argument("--row-json", default=None)
    pr.set_defaults(fn=run)
    ps = sub.add_parser("summarize")
    ps.add_argument("--dir", required=True, help="directory holding <arm>.ncu-rep (and optional <arm>.row.json)")
    ps.add_argument("--out", required=True, help="markdown output")
    ps.set_defaults(fn=summarize)
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
