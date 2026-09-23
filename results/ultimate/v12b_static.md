# v12b static report (build b4, sm_80)

Cells are REG / STACK bytes / SASS instructions of the production backward instantiations (fp16, even_K, no dropout / local / alibi / softcap). v11 = the CQS backward every CQS call runs today (v12b's LEGACY is the same source); v11x = `-DCQSA_BWD_NO_CQS` (upstream's loop with our contract, ignores the mask); ALIGNED = v12b's `flash_bwd_dq_dk_dv_loop_seqk_parallel_aligned_kernel` with the run table (CQS-on calls) and without (CQS-off calls). Upstream for comparison (cuobjdump_v11x.md): hdim64 128x128 causal even_MN torch-FA2 255/96/2336, flash_attn 2.8.3 STACK 96; hdim128 64x128 causal even_MN torch-FA2 255/64/2024.

| traits | causal | even_MN | v11 | v11x NO_CQS | v12b LEGACY | ALIGNED runs | ALIGNED runs, dP_sum smem | ALIGNED no runs | ALIGNED no runs, dP_sum smem | ALIGNED runs spills st/ld B | ALIGNED runs ptxas s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| hdim64 128x128 w8 fp16 | 1 | 1 | 255/224/5624 | 255/112/2152 | 255/224/5632 | 255/112/2312 | 255/104/2320 | 255/96/2104 | 255/80/2136 | 108/136 | 1.2 |
| hdim64 128x128 w8 fp16 | 1 | 0 | 255/248/6152 | 255/120/2672 | 255/248/6160 | 255/104/2776 | 255/96/2800 | 255/96/2624 | 255/80/2640 | 100/104 | 1.4 |
| hdim64 128x128 w8 fp16 | 0 | 1 | 255/200/3896 | 255/112/1968 | 255/200/3904 | 255/96/2088 | 255/96/2120 | 255/96/1920 | 255/88/1944 | 92/96 | 1.1 |
| hdim64 128x128 w8 fp16 | 0 | 0 | 255/184/4960 | 255/128/2712 | 255/184/4968 | 255/104/2776 | 255/88/2808 | 255/136/2496 | 255/128/2504 | 100/104 | 1.4 |
| hdim128 64x128 w8 fp16 | 1 | 1 | 255/96/4168 | 255/88/1968 | 255/96/4176 | 255/56/2104 | 255/72/2160 | 255/64/1912 | 255/48/1920 | 52/76 | 1.1 |
| hdim128 64x128 w8 fp16 | 1 | 0 | 255/112/4736 | 255/72/2480 | 255/112/4744 | 255/32/2536 | 255/64/2592 | 255/72/2432 | 255/48/2448 | 32/32 | 1.2 |
| hdim128 64x128 w8 fp16 | 0 | 1 | 255/88/3128 | 255/72/1848 | 255/88/3136 | 255/48/1992 | 255/64/2024 | 255/64/1800 | 255/48/1816 | 48/48 | 1.1 |
| hdim128 64x128 w8 fp16 | 0 | 0 | 255/88/4120 | 255/88/2496 | 255/88/4128 | 255/40/2528 | 255/32/2568 | 255/56/2440 | 255/40/2464 | 40/40 | 1.2 |

## Per-TU nvcc phases (`/scratch/gpfs/AKEY/yb2807/Stream-CQSA-dev/next/logs/build_v12b_b2_nvcc_time.csv`; this csv accumulates the b2, b3a, b3 and b4 builds of the main units, so their rows are sums; b1: `build_v12b_b1_nvcc_time.csv`)

| TU | cicc s | ptxas s | other s | total s |
|---|---|---|---|---|
| cqs_census_bwd.cu | 7.1 | 0.2 | 21.7 | 29.0 |
| cqsa_kernel.cu | 30.5 | 0.2 | 96.5 | 127.2 |
| flash_bwd_hdim128_fp16_legacy_sm80.cu | 656.5 | 43.9 | 38.2 | 738.7 |
| flash_bwd_hdim128_fp16_sm80.cu | 774.2 | 91.5 | 94.7 | 960.4 |
| flash_bwd_hdim64_fp16_legacy_sm80.cu | 677.9 | 54.7 | 36.0 | 768.6 |
| flash_bwd_hdim64_fp16_sm80.cu | 769.1 | 100.1 | 96.1 | 965.3 |
| flash_fwd_hdim128_fp16_causal_sm80.cu | 479.3 | 101.9 | 39.6 | 620.8 |
| flash_fwd_hdim128_fp16_sm80.cu | 646.8 | 146.8 | 36.6 | 830.2 |
| flash_fwd_hdim64_fp16_causal_sm80.cu | 343.6 | 82.0 | 34.1 | 459.7 |
| flash_fwd_hdim64_fp16_sm80.cu | 489.9 | 127.6 | 34.0 | 651.5 |
