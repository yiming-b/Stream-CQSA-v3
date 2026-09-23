# v12b static report (build b1, sm_80)

Cells are REG / STACK bytes / SASS instructions of the production backward instantiations (fp16, even_K, no dropout / local / alibi / softcap). v11 = the CQS backward every CQS call runs today (v12b's LEGACY is the same source); v11x = `-DCQSA_BWD_NO_CQS` (upstream's loop with our contract, ignores the mask); ALIGNED = v12b's `flash_bwd_dq_dk_dv_loop_seqk_parallel_aligned_kernel` with the run table (CQS-on calls) and without (CQS-off calls). Upstream for comparison (cuobjdump_v11x.md): hdim64 128x128 causal even_MN torch-FA2 255/96/2336, flash_attn 2.8.3 STACK 96; hdim128 64x128 causal even_MN torch-FA2 255/64/2024.

| traits | causal | even_MN | v11 | v11x NO_CQS | v12b LEGACY | v12b ALIGNED runs | v12b ALIGNED no runs | ALIGNED runs spills st/ld B | ALIGNED runs ptxas s |
|---|---|---|---|---|---|---|---|---|---|
| hdim64 128x128 w8 fp16 | 1 | 1 | 255/224/5624 | 255/112/2152 | 255/224/5632 | 255/72/2176 | 255/64/2024 | 68/72 | 2.2 |
| hdim64 128x128 w8 fp16 | 1 | 0 | 255/248/6152 | 255/120/2672 | 255/248/6160 | 255/72/2608 | 255/64/2536 | 68/72 | 2.3 |
| hdim64 128x128 w8 fp16 | 0 | 1 | 255/200/3896 | 255/112/1968 | 255/200/3904 | 255/48/1984 | 255/56/1840 | 44/48 | 2.0 |
| hdim64 128x128 w8 fp16 | 0 | 0 | 255/184/4960 | 255/128/2712 | 255/184/4968 | 255/48/2624 | 255/64/2568 | 48/52 | 2.6 |
| hdim128 64x128 w8 fp16 | 1 | 1 | 255/96/4168 | 255/88/1968 | 255/96/4176 | 255/48/2040 | 255/120/1912 | 40/68 | 2.2 |
| hdim128 64x128 w8 fp16 | 1 | 0 | 255/112/4736 | 255/72/2480 | 255/112/4744 | 255/48/2448 | 255/112/2416 | 44/72 | 2.2 |
| hdim128 64x128 w8 fp16 | 0 | 1 | 255/88/3128 | 255/72/1848 | 255/88/3136 | 255/40/1936 | 255/120/1800 | 40/68 | 2.0 |
| hdim128 64x128 w8 fp16 | 0 | 0 | 255/88/4120 | 255/88/2496 | 255/88/4128 | 255/48/2448 | 255/120/2408 | 44/72 | 2.2 |

## Per-TU nvcc phases (`/scratch/gpfs/AKEY/yb2807/Stream-CQSA-dev/next/logs/build_v12b_b1_nvcc_time.csv`)

| TU | cicc s | ptxas s | other s | total s |
|---|---|---|---|---|
| cqs_census_bwd.cu | 12.9 | 0.3 | 38.6 | 51.8 |
| cqsa_kernel.cu | 60.6 | 0.5 | 164.9 | 226.0 |
| flash_bwd_hdim128_fp16_legacy_sm80.cu | 1337.9 | 78.2 | 66.4 | 1482.5 |
| flash_bwd_hdim128_fp16_sm80.cu | 495.0 | 31.9 | 71.6 | 598.5 |
| flash_bwd_hdim64_fp16_legacy_sm80.cu | 1365.4 | 100.4 | 65.4 | 1531.2 |
| flash_bwd_hdim64_fp16_sm80.cu | 490.3 | 32.9 | 73.4 | 596.6 |
| flash_fwd_hdim128_fp16_causal_sm80.cu | 1031.2 | 183.7 | 66.6 | 1281.5 |
| flash_fwd_hdim128_fp16_sm80.cu | 1298.8 | 278.6 | 66.8 | 1644.2 |
| flash_fwd_hdim64_fp16_causal_sm80.cu | 737.7 | 152.6 | 64.5 | 954.8 |
| flash_fwd_hdim64_fp16_sm80.cu | 1039.3 | 240.1 | 63.3 | 1342.7 |
