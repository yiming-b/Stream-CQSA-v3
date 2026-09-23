"""benchmarks/ultimate/cuobj_report.py: the parsers behind the cuobjdump / ptxas report (CPU only, no binaries)."""
import os, sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "benchmarks", "ultimate"))
import cuobj_report as R  # noqa: E402

V11_FWD = ("void flash::flash_fwd_kernel<Flash_fwd_kernel_traits<64, 128, 128, 4, false, false, cutlass::half_t, "
           "Flash_kernel_traits<64, 128, 128, 4, cutlass::half_t> >, false, true, false, false, true, true, false, false, 1>"
           "(flash::Flash_fwd_params)")
V9_FWD = ("void flash::flash_fwd_kernel<Flash_fwd_kernel_traits<128, 128, 64, 4, false, false, cutlass::half_t, "
          "Flash_kernel_traits<128, 128, 64, 4, cutlass::half_t> >, false, false, false, false, false, true, false, false>"
          "(flash::Flash_fwd_params)")
TORCH_BWD = ("void pytorch_flash::flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<pytorch_flash::Flash_bwd_kernel_traits<64, 128, 128, 8, "
             "2, 4, 4, false, false, cutlass::bfloat16_t, pytorch_flash::Flash_kernel_traits<64, 128, 128, 8, cutlass::bfloat16_t> >, "
             "true, true, false, false, true, false, false>(pytorch_flash::Flash_bwd_params)")


def test_parse_key_keeps_cqs_mode_apart_from_upstream_flags():
    kind, traits, flags, cqs = R.parse_key(V11_FWD)
    assert (kind, traits, cqs) == ("fwd", "hdim64 128x128 w4 fp16", "1")
    assert flags == dict(dropout=False, causal=True, local=False, alibi=False, even_MN=True, even_K=True, softcap=False, ret_softmax=False)
    assert R.flag_str(kind, flags) == "causal,even_MN,even_K" and R.production(traits, flags)
    kind, traits, flags, cqs = R.parse_key(V9_FWD)          # no Cqs_mode argument: one sub-column
    assert (kind, traits, cqs) == ("fwd", "hdim128 128x64 w4 fp16", "-") and not flags["causal"]
    kind, traits, flags, cqs = R.parse_key(TORCH_BWD)       # other namespace, 7 backward flags, bf16 is not production
    assert (kind, traits, cqs) == ("bwd", "hdim64 128x128 w8 bf16", "-") and flags["dropout"]
    assert not R.production(traits, flags)
    assert R.parse_key("void flash::flash_fwd_splitkv_kernel<Flash_fwd_kernel_traits<64, 128, 128, 4, false, false, cutlass::half_t, "
                       "Flash_kernel_traits<64, 128, 128, 4, cutlass::half_t> >, false, true, false, false, true, true, false, false, 1>"
                       "(flash::Flash_fwd_params)") is None


def test_parse_res_usage_takes_only_the_sm80_section():
    text = "\n".join(["Fatbin elf code:", "arch = sm_90", " Function _Za:", "  REG:100 STACK:8 SHARED:0",
                      "Fatbin elf code:", "arch = sm_80", " Function _Za:", "  REG:255 STACK:640 SHARED:0",
                      " Function _Zb:", "  REG:24 STACK:0 SHARED:0", "arch = sm_80", " Common:", "  GLOBAL:28"])
    assert R.parse_res_usage(text) == {"_Za": (255, 640), "_Zb": (24, 0)}


def test_parse_ptxas_log_pairs_each_entry_with_its_report():
    text = "\n".join(["ptxas info    : Compiling entry function '_Za' for 'sm_80'", "ptxas info    : Function properties for _Za",
                      "    640 bytes stack frame, 768 bytes spill stores, 820 bytes spill loads",
                      "ptxas info    : Used 255 registers, used 1 barriers, 1016 bytes cmem[0]", "ptxas info    : Compile time = 1834.5 ms",
                      "ptxas info    : Compiling entry function '_Zb' for 'sm_90'", "    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads",
                      "ptxas info    : Used 24 registers, used 0 barriers", "ptxas info    : Compile time = 12.6 ms"])
    assert R.parse_ptxas_log(text) == {"_Za": dict(stack=640, spill_st=768, spill_ld=820, reg=255, ms=1834.5)}


def test_time_table_sums_phases_per_tu(tmp_path):
    csv = tmp_path / "t.csv"
    csv.write_text("source file name , phase name , phase input files , phase output file , arch , tool, metric , unit\n"
                   "/x/flash_fwd_hdim64_fp16_sm80.cu , cicc ,  a , b , compute_80 , nvcc , 123228.4 , ms\n"
                   "/x/flash_fwd_hdim64_fp16_sm80.cu , ptxas , b , c , sm_80 , nvcc , 61469.7 , ms\n"
                   "/x/flash_fwd_hdim64_fp16_sm80.cu , gcc (compiling) ,   , d , compute_80 , nvcc , 5700.0 , ms\n"
                   " , nvcc (driver) ,  , d ,  , nvcc , 0.36 , ms\n")
    lines = R.time_table(str(csv))
    assert lines[2] == "| flash_fwd_hdim64_fp16_sm80.cu | 123.2 | 61.5 | 5.7 | 190.4 |" and len(lines) == 3
