"""CPU-only checks of benchmarks/ultimate/ncu_aligned.py's report parsing (raw page, SASS source page, opcode classes, markdown)."""
import importlib.util, json, os, pathlib, pytest

HERE = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("ncu_aligned", HERE.parent / "benchmarks" / "ultimate" / "ncu_aligned.py")
na = importlib.util.module_from_spec(spec)
spec.loader.exec_module(na)

RAW = '''==PROF== some banner line
"ID","Process ID","Process Name","Host Name","Kernel Name","Context","Stream","Block Size","Grid Size","Device","CC","gpu__time_duration.sum","smsp__inst_executed.sum","smsp__average_warp_latency_per_inst_issued.ratio","smsp__warps_eligible.avg.per_cycle_active","smsp__average_warps_issue_stalled_wait_per_issue_active.ratio","smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio","launch__registers_per_thread","sm__inst_executed_pipe_tensor_op_hmma.avg.pct_of_peak_sustained_active"
"","","","","","","","","","","","ns","inst","ratio","warp","ratio","ratio","register/thread","%"
"0","1","python","node","void flash_fwd_kernel<...>(Flash_fwd_params)","1","7","(128, 1, 1)","(438, 1, 8)","0","8.0","25,000,000","4,026,326,888","6.58","0.35","2.80","0.90","255","33.3"
'''

SASS = '''"# Address","Source","Warp Stall Sampling (All Samples)","Warp Stall Sampling (Not-issued Samples)","Instructions Executed","stall_wait","stall_barrier"
"0x000","        IADD3 R1, R1, -0x100, RZ","10","5","3,512","10","0"
"0x010","        HMMA.16816.F32 R4, R8, R12, R4","400","300","1,000,000","0","0"
"0x020","  @P0   LDG.E.64 R2, [R6.64]","150","140","500,000","150","0"
"0x030","        BAR.SYNC.DEFER_BLOCKING 0x0","90","80","200,000","0","90"
"0x040","        MUFU.EX2 R3, R3","50","10","800,000","50","0"
'''


def test_opclass():
    assert na.opclass("        HMMA.16816.F32 R4, R8, R12, R4") == "mma"
    assert na.opclass("  @P0   LDG.E.64 R2, [R6.64]") == "global_ld"
    assert na.opclass("  @!UP1 LDGSTS.E.BYPASS.128 [R5], [R2.64]") == "cp_async"
    assert na.opclass("        LDSM.16.M88.4 R20, [R31]") == "smem_ld"
    assert na.opclass("        BAR.SYNC.DEFER_BLOCKING 0x0") == "barrier"
    assert na.opclass("        MUFU.EX2 R3, R3") == "exp_mufu"
    assert na.opclass("        FMNMX R5, R5, R7, !PT") == "fp32"
    assert na.opclass("        ISETP.NE.AND P0, PT, R2, RZ, PT") == "int"
    assert na.opclass("        BRA 0x1230") == "branch"
    assert na.opclass("        STL [R1+0x40], R9") == "local"
    assert na.opclass("        FOOBAR R1") == "other"


def test_summarize_report_and_markdown(monkeypatch, tmp_path):
    def fake_csv(rep, *page_args):
        return SASS if "source" in page_args else RAW
    monkeypatch.setattr(na, "_ncu_csv", fake_csv)
    s = na.summarize_report(str(tmp_path / "zero.ncu-rep"))
    assert s["inst_executed"] == 4026326888 and s["duration_ms"] == 25.0
    assert s["warp_cycles_per_issued_inst"] == 6.58 and s["registers_per_thread"] == 255
    assert s["stall_cycles_per_issue"] == {"wait": 2.8, "barrier": 0.9}          # sorted by size
    assert s["tensor_pipe_active_pct"] == 33.3
    assert s["sass_n_instructions"] == 5 and s["sass_samples_all"] == 700
    by = s["sass_by_opclass"]
    assert list(by)[0] == "mma" and by["global_ld"]["samples_all"] == 150
    assert abs(by["barrier"]["samples_all_pct"] - 100 * 90 / 700) < 1e-9
    top = s["sass_top10"]
    assert top[0]["source"].lstrip().startswith("HMMA") and top[1]["reasons"] == {"stall_wait": 150.0}
    # markdown: every arm shares the same fake report, missing arms are reported not raised
    rows = {a: dict(s, run_row=dict(N=131072, c=7, path=[2], L=56064, L_mod_128=0)) for a in na.ARMS}
    rows["real"] = dict(error="real.ncu-rep missing")
    out = tmp_path / "ncu.md"
    na.write_md(rows, str(out), str(tmp_path))
    text = out.read_text()
    assert "| executed instructions | 4,026,326,888 | 4,026,326,888 | 4,026,326,888 |" in text
    assert "| wait | 2.80 (42.6%)" in text and "real: real.ncu-rep missing" in text
    assert "Top 10 SASS instructions by stall samples: zero" in text and "HMMA.16816.F32" in text


def test_sass_table_without_source_page(monkeypatch):
    monkeypatch.setattr(na, "_ncu_csv", lambda rep, *a: "==ERROR== no source data" if "source" in a else RAW)
    s = na.summarize_report("x.ncu-rep")
    assert s["sass_error"].startswith("==ERROR==") and "sass_top10" not in s
