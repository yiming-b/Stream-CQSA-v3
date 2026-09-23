# v12f static report (build b1, sm_80)

Cells are REG / STACK bytes / SASS instructions. CQS-on = Cqs_mode 1 (causal) or 2 (non-causal runtime flag): v11's is v12f's LEGACY; v11x's has the per-element path compiled out (poisons tiles that need it); ALIGNED = v12f's `flash_fwd_aligned_kernel` (run-table loop, no verdicts, no element path).

| traits | causal | even_MN | v11 CQS off | v11 CQS-on | v11x CQS-on | v12f CQS off | v12f LEGACY | v12f ALIGNED | ALIGNED spills st/ld B | ALIGNED ptxas s |
|---|---|---|---|---|---|---|---|---|---|---|
| hdim64 128x128 w4 fp16 | 1 | 1 | 255/216/4032 | 255/264/72776 | 255/112/5464 | 255/216/4032 | 255/264/72784 | 255/224/4112 | 292/280 | 1.1 |
| hdim64 128x128 w4 fp16 | 1 | 0 | 255/304/6728 | 255/400/74264 | 255/144/7688 | 255/304/6728 | 255/400/74280 | 255/288/6600 | 520/516 | 1.7 |
| hdim64 128x128 w4 fp16 | 0 | 1 | 255/224/3656 | 255/640/69624 | 255/96/5424 | 255/224/3664 | 255/312/66992 | 255/264/3792 | 336/328 | 0.6 |
| hdim64 128x128 w4 fp16 | 0 | 0 | 255/192/4344 | 255/680/71304 | 255/120/6216 | 255/192/4344 | 255/328/68304 | 255/256/4336 | 296/304 | 0.6 |
| hdim128 128x64 w4 fp16 | 1 | 1 | 255/160/4984 | 255/256/57592 | 255/272/6680 | 255/160/4984 | 255/256/57600 | 255/152/5288 | 248/236 | 1.3 |
| hdim128 128x64 w4 fp16 | 1 | 0 | 255/104/6536 | 255/296/40792 | 255/288/8360 | 255/104/6536 | 255/296/40800 | 255/168/7176 | 320/340 | 1.6 |
| hdim128 128x64 w4 fp16 | 0 | 1 | 255/144/3304 | 255/632/38224 | 255/48/5048 | 255/144/3304 | 255/248/36664 | 255/168/3632 | 228/180 | 1.2 |
| hdim128 128x64 w4 fp16 | 0 | 0 | 255/104/3920 | 255/576/39592 | 255/56/5672 | 255/104/3920 | 255/272/37784 | 255/104/3976 | 152/120 | 1.2 |

## Per-TU nvcc phases (`/scratch/gpfs/AKEY/yb2807/Stream-CQSA-dev/next/logs/build_v12f_b1_nvcc_time.csv`)

| TU | cicc s | ptxas s | other s | total s |
|---|---|---|---|---|
| cqs_census.cu | 4.0 | 0.1 | 9.5 | 13.6 |
| cqsa_kernel.cu | 19.5 | 0.2 | 50.6 | 70.3 |
| flash_bwd_hdim128_fp16_sm80.cu | 397.7 | 15.7 | 17.1 | 430.5 |
| flash_bwd_hdim64_fp16_sm80.cu | 399.1 | 19.1 | 17.9 | 436.2 |
| flash_fwd_hdim128_fp16_causal_legacy_sm80.cu | 530.3 | 154.8 | 18.0 | 703.1 |
| flash_fwd_hdim128_fp16_causal_sm80.cu | 192.0 | 44.4 | 42.6 | 279.0 |
| flash_fwd_hdim128_fp16_legacy_sm80.cu | 607.0 | 140.0 | 17.9 | 764.9 |
| flash_fwd_hdim128_fp16_sm80.cu | 235.6 | 65.1 | 24.1 | 324.8 |
| flash_fwd_hdim64_fp16_causal_legacy_sm80.cu | 1026.6 | 289.1 | 19.1 | 1334.7 |
| flash_fwd_hdim64_fp16_causal_sm80.cu | 123.6 | 32.3 | 28.2 | 184.1 |
| flash_fwd_hdim64_fp16_legacy_sm80.cu | 1329.8 | 315.1 | 18.1 | 1663.0 |
| flash_fwd_hdim64_fp16_sm80.cu | 136.8 | 32.3 | 16.1 | 185.2 |
