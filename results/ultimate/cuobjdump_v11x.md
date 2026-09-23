# cuobjdump / ptxas report (2026-09-23 08:13, sm_80)

Modules: `v11` = `/scratch/gpfs/AKEY/yb2807/Stream-CQSA-dev/next/kernel/stable/cqsa_cuda_next_v11.cpython-311-x86_64-linux-gnu.so` (44 MB); `v11x` = `/scratch/gpfs/AKEY/yb2807/Stream-CQSA-dev/next/kernel/v11x/cqsa_cuda_next_v11x.cpython-311-x86_64-linux-gnu.so` (37 MB); `v9` = `/scratch/gpfs/AKEY/yb2807/Stream-CQSA-dev/next/kernel/stable/cqsa_cuda_next_v9.cpython-311-x86_64-linux-gnu.so` (37 MB); `torch-FA2` = `/scratch/gpfs/AKEY/yb2807/VENVs/CQSA311/lib/python3.11/site-packages/torch/lib/libtorch_cuda.so` (540 MB); `fa2.8.3` = `/scratch/gpfs/AKEY/yb2807/Stream-CQSA-dev/next/logs/01_cuobjdump_upstream_fa2.txt` (res-usage log)

Cells are REG/STACK bytes/SASS instructions (`?` = no binary). Production instantiations only (fp16, even_K, no dropout/local/alibi/softcap/return_softmax).

## Build phases per TU (`/scratch/gpfs/AKEY/yb2807/Stream-CQSA-dev/next/logs/build_v11x_nvcc_time.csv`)

| TU | cicc s | ptxas s | other s | total s |
|---|---|---|---|---|
| cqsa_kernel.cu | 11.0 | 0.1 | 27.0 | 38.1 |
| flash_bwd_hdim128_fp16_sm80.cu | 240.8 | 22.7 | 14.3 | 277.8 |
| flash_bwd_hdim64_fp16_sm80.cu | 236.9 | 24.0 | 14.7 | 275.6 |
| flash_fwd_hdim128_fp16_causal_sm80.cu | 176.7 | 46.5 | 15.1 | 238.2 |
| flash_fwd_hdim128_fp16_sm80.cu | 250.3 | 75.1 | 16.0 | 341.5 |
| flash_fwd_hdim64_fp16_causal_sm80.cu | 123.2 | 33.7 | 13.8 | 170.7 |
| flash_fwd_hdim64_fp16_sm80.cu | 196.4 | 61.5 | 15.4 | 273.3 |

## Instantiations

| kind | traits | flags | v11 | v11 cqs=0 | v11 cqs=1 | v11 cqs=2 | v11x | v11x cqs=0 | v11x cqs=1 | v11x cqs=2 | v9 | torch-FA2 | torch-FA2 cqs=-' | fa2.8.3 | fa2.8.3 cqs=-' |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bwd | hdim128 64x128 w8 fp16 | causal,even_K | 255/112/4736 | - | - | - | 255/72/2480 | - | - | - | 255/112/4736 | 255/64/2528 | - | 255/72/? | - |
| bwd | hdim128 64x128 w8 fp16 | causal,even_MN,even_K | 255/96/4168 | - | - | - | 255/88/1968 | - | - | - | 255/96/4168 | 255/64/2024 | - | 255/64/? | - |
| bwd | hdim128 64x128 w8 fp16 | even_K | 255/88/4120 | - | - | - | 255/88/2496 | - | - | - | 255/88/4120 | 255/64/2536 | - | 255/56/? | - |
| bwd | hdim128 64x128 w8 fp16 | even_MN,even_K | 255/88/3128 | - | - | - | 255/72/1848 | - | - | - | 255/88/3128 | 255/64/1912 | - | 255/64/? | - |
| bwd | hdim128 64x64 w8 fp16 | causal,even_K | 255/0/2920 | - | - | - | 255/40/1584 | - | - | - | 255/0/2920 | 255/32/1608 | - | 255/32/? | - |
| bwd | hdim128 64x64 w8 fp16 | causal,even_MN,even_K | 252/0/2600 | - | - | - | 255/40/1288 | - | - | - | 252/0/2600 | 255/0/1328 | - | 255/32/? | - |
| bwd | hdim128 64x64 w8 fp16 | even_K | 255/0/2624 | - | - | - | 255/40/1568 | - | - | - | 255/0/2624 | 255/24/1584 | - | 255/24/? | - |
| bwd | hdim128 64x64 w8 fp16 | even_MN,even_K | 254/0/2256 | - | - | - | 255/24/1208 | - | - | - | 254/0/2256 | 255/16/1248 | - | 255/16/? | - |
| bwd | hdim160 64x64 w8 fp16 | causal,even_K | - | - | - | - | - | - | - | - | - | 218/0/1544 | 254/0/1592 | - | - |
| bwd | hdim160 64x64 w8 fp16 | even_K | - | - | - | - | - | - | - | - | - | 214/0/1520 | 254/0/1560 | - | - |
| bwd | hdim192 64x64 w8 fp16 | causal,even_K | - | - | - | - | - | - | - | - | - | 255/216/2016 | 255/16/1864 | 255/216/? | 255/16/? |
| bwd | hdim192 64x64 w8 fp16 | even_K | - | - | - | - | - | - | - | - | - | 255/208/2032 | 255/24/1848 | 255/208/? | 255/24/? |
| bwd | hdim256 64x32 w8 fp16 | causal,even_K | - | - | - | - | - | - | - | - | - | 255/24/1600 | - | 255/24/? | - |
| bwd | hdim256 64x32 w8 fp16 | even_K | - | - | - | - | - | - | - | - | - | 255/24/1584 | - | 255/24/? | - |
| bwd | hdim256 64x64 w8 fp16 | causal,even_K | - | - | - | - | - | - | - | - | - | 255/0/2152 | 255/88/2384 | 255/0/? | 255/88/? |
| bwd | hdim256 64x64 w8 fp16 | even_K | - | - | - | - | - | - | - | - | - | 254/0/2152 | 255/88/2400 | 255/0/? | 255/88/? |
| bwd | hdim32 128x128 w8 fp16 | causal,even_K | - | - | - | - | - | - | - | - | - | 255/0/2208 | - | 255/64/? | - |
| bwd | hdim32 128x128 w8 fp16 | causal,even_MN,even_K | - | - | - | - | - | - | - | - | - | 255/0/1880 | - | 255/0/? | - |
| bwd | hdim32 128x128 w8 fp16 | even_K | - | - | - | - | - | - | - | - | - | 255/0/2240 | - | 255/40/? | - |
| bwd | hdim32 128x128 w8 fp16 | even_MN,even_K | - | - | - | - | - | - | - | - | - | 255/0/1712 | - | 255/0/? | - |
| bwd | hdim64 128x128 w8 fp16 | causal,even_K | 255/248/6152 | - | - | - | 255/120/2672 | - | - | - | 255/248/6152 | 255/96/2856 | - | 255/96/? | - |
| bwd | hdim64 128x128 w8 fp16 | causal,even_MN,even_K | 255/224/5624 | - | - | - | 255/112/2152 | - | - | - | 255/224/5624 | 255/96/2336 | - | 255/96/? | - |
| bwd | hdim64 128x128 w8 fp16 | even_K | 255/184/4960 | - | - | - | 255/128/2712 | - | - | - | 255/184/4960 | 255/88/2888 | - | 255/96/? | - |
| bwd | hdim64 128x128 w8 fp16 | even_MN,even_K | 255/200/3896 | - | - | - | 255/112/1968 | - | - | - | 255/200/3896 | 255/112/2168 | - | 255/96/? | - |
| bwd | hdim64 64x128 w8 fp16 | causal,even_K | 255/0/3872 | - | - | - | 255/8/1816 | - | - | - | 255/0/3872 | 255/0/1896 | - | 255/16/? | - |
| bwd | hdim64 64x128 w8 fp16 | causal,even_MN,even_K | 249/0/3424 | - | - | - | 252/0/1368 | - | - | - | 249/0/3424 | 255/0/1528 | - | 255/0/? | - |
| bwd | hdim64 64x128 w8 fp16 | even_K | 252/0/3312 | - | - | - | 255/0/1776 | - | - | - | 252/0/3312 | 255/0/1864 | - | 255/0/? | - |
| bwd | hdim64 64x128 w8 fp16 | even_MN,even_K | 241/0/2688 | - | - | - | 255/0/1320 | - | - | - | 241/0/2688 | 255/0/1376 | - | 255/0/? | - |
| bwd | hdim96 64x128 w8 fp16 | causal,even_K | - | - | - | - | - | - | - | - | - | 255/72/2144 | - | 255/64/? | - |
| bwd | hdim96 64x128 w8 fp16 | causal,even_MN,even_K | - | - | - | - | - | - | - | - | - | 255/72/1800 | - | 255/80/? | - |
| bwd | hdim96 64x128 w8 fp16 | even_K | - | - | - | - | - | - | - | - | - | 255/56/2160 | - | 255/64/? | - |
| bwd | hdim96 64x128 w8 fp16 | even_MN,even_K | - | - | - | - | - | - | - | - | - | 255/64/1696 | - | 255/48/? | - |
| fwd | hdim128 128x32 w4 fp16 | even_K | - | 255/0/2824 | - | 255/152/21160 | - | 255/0/2824 | - | 248/0/3736 | 255/48/20848 | 255/48/2600 | - | 255/24/? | - |
| fwd | hdim128 128x32 w4 fp16 | even_MN,even_K | - | 255/0/2296 | - | 255/128/20168 | - | 255/0/2296 | - | 249/0/3160 | 255/32/19976 | 255/8/2112 | - | 255/16/? | - |
| fwd | hdim128 128x64 w4 fp16 | causal,even_K | - | 255/104/6536 | 255/296/40792 | - | - | 255/104/6536 | 255/288/8360 | - | 255/344/38936 | 255/296/7384 | - | 255/344/? | - |
| fwd | hdim128 128x64 w4 fp16 | causal,even_MN,even_K | - | 255/160/4984 | 255/256/57592 | - | - | 255/160/4984 | 255/272/6680 | - | 255/336/38224 | 255/168/5552 | - | 255/264/? | - |
| fwd | hdim128 128x64 w4 fp16 | even_K | - | 255/104/3920 | - | 255/576/39592 | - | 255/104/3920 | - | 255/56/5672 | 255/288/38824 | 255/208/4040 | - | 255/144/? | - |
| fwd | hdim128 128x64 w4 fp16 | even_MN,even_K | - | 255/144/3304 | - | 255/632/38224 | - | 255/144/3304 | - | 255/48/5048 | 255/264/37976 | 255/56/3056 | - | 255/128/? | - |
| fwd | hdim128 64x64 w4 fp16 | causal,even_K | - | 180/0/3000 | 232/0/30184 | - | - | 180/0/3000 | 190/0/4560 | - | 255/40/30592 | 176/0/2936 | - | 210/0/? | - |
| fwd | hdim128 64x64 w4 fp16 | causal,even_MN,even_K | - | 180/0/2136 | 226/0/20560 | - | - | 180/0/2136 | 170/0/3544 | - | 217/0/20152 | 184/0/2000 | - | 182/0/? | - |
| fwd | hdim160 128x32 w4 fp16 | causal,even_K | - | - | - | - | - | - | - | - | - | 255/152/6272 | - | - | - |
| fwd | hdim160 128x32 w4 fp16 | even_K | - | - | - | - | - | - | - | - | - | 255/104/2632 | - | - | - |
| fwd | hdim160 128x64 w8 fp16 | even_K | - | - | - | - | - | - | - | - | - | 176/0/2152 | - | - | - |
| fwd | hdim160 64x64 w4 fp16 | causal,even_K | - | - | - | - | - | - | - | - | - | 188/0/3024 | - | - | - |
| fwd | hdim192 128x64 w8 fp16 | causal,even_K | - | - | - | - | - | - | - | - | - | 230/0/4144 | - | 242/0/? | - |
| fwd | hdim192 128x64 w8 fp16 | even_K | - | - | - | - | - | - | - | - | - | 226/0/2440 | - | 242/0/? | - |
| fwd | hdim256 128x64 w8 fp16 | causal,even_K | - | - | - | - | - | - | - | - | - | 254/0/4840 | - | 255/88/? | - |
| fwd | hdim256 128x64 w8 fp16 | even_K | - | - | - | - | - | - | - | - | - | 254/0/2848 | - | 255/0/? | - |
| fwd | hdim256 64x64 w4 fp16 | causal,even_K | - | - | - | - | - | - | - | - | - | 254/0/3992 | - | 255/32/? | - |
| fwd | hdim256 64x64 w4 fp16 | even_K | - | - | - | - | - | - | - | - | - | 254/0/2944 | - | 255/8/? | - |
| fwd | hdim32 128x128 w4 fp16 | causal,even_K | - | - | - | - | - | - | - | - | - | 255/184/5968 | - | 255/360/? | - |
| fwd | hdim32 128x128 w4 fp16 | causal,even_MN,even_K | - | - | - | - | - | - | - | - | - | 255/96/3864 | - | 255/240/? | - |
| fwd | hdim32 128x128 w4 fp16 | even_K | - | - | - | - | - | - | - | - | - | 255/120/3952 | - | 255/160/? | - |
| fwd | hdim32 128x128 w4 fp16 | even_MN,even_K | - | - | - | - | - | - | - | - | - | 255/0/3352 | - | 255/88/? | - |
| fwd | hdim64 128x128 w4 fp16 | causal,even_K | - | 255/304/6728 | 255/400/74264 | - | - | 255/304/6728 | 255/144/7688 | - | 255/440/69816 | 255/168/6896 | - | 255/472/? | - |
| fwd | hdim64 128x128 w4 fp16 | causal,even_MN,even_K | - | 255/216/4032 | 255/264/72776 | - | - | 255/216/4032 | 255/112/5464 | - | 255/328/71392 | 255/88/4400 | - | 255/384/? | - |
| fwd | hdim64 128x128 w4 fp16 | even_K | - | 255/192/4344 | - | 255/680/71304 | - | 255/192/4344 | - | 255/120/6216 | 255/352/70488 | 255/112/4760 | - | 255/384/? | - |
| fwd | hdim64 128x128 w4 fp16 | even_MN,even_K | - | 255/224/3656 | - | 255/640/69624 | - | 255/224/3656 | - | 255/96/5424 | 255/344/69464 | 255/128/3944 | - | 255/264/? | - |
| fwd | hdim96 128x64 w4 fp16 | causal,even_K | - | - | - | - | - | - | - | - | - | 255/104/5648 | - | 255/24/? | - |
| fwd | hdim96 128x64 w4 fp16 | causal,even_MN,even_K | - | - | - | - | - | - | - | - | - | 255/64/4232 | - | 255/88/? | - |
| fwd | hdim96 128x64 w4 fp16 | even_K | - | - | - | - | - | - | - | - | - | 255/32/3080 | - | 255/72/? | - |
| fwd | hdim96 128x64 w4 fp16 | even_MN,even_K | - | - | - | - | - | - | - | - | - | 252/0/2640 | - | 255/40/? | - |
| fwd | hdim96 64x64 w4 fp16 | causal,even_K | - | - | - | - | - | - | - | - | - | 165/0/2552 | - | 194/0/? | - |
| fwd | hdim96 64x64 w4 fp16 | causal,even_MN,even_K | - | - | - | - | - | - | - | - | - | 166/0/1776 | - | 168/0/? | - |

## ptxas -v: v11x (`/scratch/gpfs/AKEY/yb2807/Stream-CQSA-dev/next/logs/04_build_v11x.log`)

| kind | traits | flags | cqs | REG | STACK B | spill stores/loads B | ptxas s |
|---|---|---|---|---|---|---|---|
| bwd | hdim128 64x128 w8 fp16 | causal,even_K | - | 255 | 72 | 68/88 | 0.4 |
| bwd | hdim128 64x128 w8 fp16 | causal,even_MN,even_K | - | 255 | 88 | 84/108 | 0.3 |
| bwd | hdim128 64x128 w8 fp16 | even_K | - | 255 | 88 | 88/116 | 0.4 |
| bwd | hdim128 64x128 w8 fp16 | even_MN,even_K | - | 255 | 72 | 68/92 | 0.3 |
| bwd | hdim128 64x64 w8 fp16 | causal,even_K | - | 255 | 40 | 36/52 | 0.3 |
| bwd | hdim128 64x64 w8 fp16 | causal,even_MN,even_K | - | 255 | 40 | 36/52 | 0.2 |
| bwd | hdim128 64x64 w8 fp16 | even_K | - | 255 | 40 | 36/64 | 0.3 |
| bwd | hdim128 64x64 w8 fp16 | even_MN,even_K | - | 255 | 24 | 28/36 | 0.2 |
| bwd | hdim64 128x128 w8 fp16 | causal,even_K | - | 255 | 120 | 112/132 | 0.4 |
| bwd | hdim64 128x128 w8 fp16 | causal,even_MN,even_K | - | 255 | 112 | 108/132 | 0.3 |
| bwd | hdim64 128x128 w8 fp16 | even_K | - | 255 | 128 | 124/128 | 0.4 |
| bwd | hdim64 128x128 w8 fp16 | even_MN,even_K | - | 255 | 112 | 108/116 | 0.3 |
| bwd | hdim64 64x128 w8 fp16 | causal,even_K | - | 255 | 8 | 4/4 | 0.3 |
| bwd | hdim64 64x128 w8 fp16 | causal,even_MN,even_K | - | 252 | 0 | 0/0 | 0.2 |
| bwd | hdim64 64x128 w8 fp16 | even_K | - | 255 | 0 | 0/0 | 0.2 |
| bwd | hdim64 64x128 w8 fp16 | even_MN,even_K | - | 255 | 0 | 0/0 | 0.2 |
| fwd | hdim128 128x32 w4 fp16 | even_K | 0 | 255 | 0 | 0/0 | 0.4 |
| fwd | hdim128 128x32 w4 fp16 | even_K | 2 | 248 | 0 | 0/0 | 0.3 |
| fwd | hdim128 128x32 w4 fp16 | even_MN,even_K | 0 | 255 | 0 | 0/0 | 0.3 |
| fwd | hdim128 128x32 w4 fp16 | even_MN,even_K | 2 | 249 | 0 | 0/0 | 0.3 |
| fwd | hdim128 128x64 w4 fp16 | causal,even_K | 0 | 255 | 104 | 184/240 | 1.1 |
| fwd | hdim128 128x64 w4 fp16 | causal,even_K | 1 | 255 | 288 | 344/552 | 0.8 |
| fwd | hdim128 128x64 w4 fp16 | causal,even_MN,even_K | 0 | 255 | 160 | 220/236 | 0.8 |
| fwd | hdim128 128x64 w4 fp16 | causal,even_MN,even_K | 1 | 255 | 272 | 340/428 | 0.6 |
| fwd | hdim128 128x64 w4 fp16 | even_K | 0 | 255 | 104 | 128/128 | 0.6 |
| fwd | hdim128 128x64 w4 fp16 | even_K | 2 | 255 | 56 | 52/72 | 0.5 |
| fwd | hdim128 128x64 w4 fp16 | even_MN,even_K | 0 | 255 | 144 | 164/168 | 0.5 |
| fwd | hdim128 128x64 w4 fp16 | even_MN,even_K | 2 | 255 | 48 | 48/64 | 0.4 |
| fwd | hdim128 64x64 w4 fp16 | causal,even_K | 0 | 180 | 0 | 0/0 | 0.4 |
| fwd | hdim128 64x64 w4 fp16 | causal,even_K | 1 | 190 | 0 | 0/0 | 0.4 |
| fwd | hdim128 64x64 w4 fp16 | causal,even_MN,even_K | 0 | 180 | 0 | 0/0 | 0.3 |
| fwd | hdim128 64x64 w4 fp16 | causal,even_MN,even_K | 1 | 170 | 0 | 0/0 | 0.3 |
| fwd | hdim64 128x128 w4 fp16 | causal,even_K | 0 | 255 | 304 | 548/608 | 1.0 |
| fwd | hdim64 128x128 w4 fp16 | causal,even_K | 1 | 255 | 144 | 212/332 | 0.8 |
| fwd | hdim64 128x128 w4 fp16 | causal,even_MN,even_K | 0 | 255 | 216 | 300/308 | 0.6 |
| fwd | hdim64 128x128 w4 fp16 | causal,even_MN,even_K | 1 | 255 | 112 | 112/160 | 0.5 |
| fwd | hdim64 128x128 w4 fp16 | even_K | 0 | 255 | 192 | 284/300 | 0.7 |
| fwd | hdim64 128x128 w4 fp16 | even_K | 2 | 255 | 120 | 168/212 | 0.6 |
| fwd | hdim64 128x128 w4 fp16 | even_MN,even_K | 0 | 255 | 224 | 324/336 | 0.6 |
| fwd | hdim64 128x128 w4 fp16 | even_MN,even_K | 2 | 255 | 96 | 120/156 | 0.5 |

## Verdict (P1.1, 2026-09-23)

Build: `next/kernel/v11x` = v11 + `-DCQSA_NO_ELEMENT_PATH` (forward per-element mask path compiled out; a tile that
would need it poisons its CTA's rows with NaN) + `-DCQSA_BWD_NO_CQS` (backward CQS compiled out: upstream's loop
with our contract) + `-lineinfo -Xptxas -v -time`, kernel set `a100_fp16_hdim64_128`, della-vis1, nvcc 13.0,
`MAX_JOBS=8 NVCC_THREADS=4`, watchdog `LIMIT=45` (never fired). Log `next/logs/04_build_v11x.log`, phase csv
`next/logs/build_v11x_nvcc_time.csv`. Whole build **343 s, rc=0**; v11's identical set on the same node and settings
took **2410 s** (`next/logs/03_build_v11.log`, 2026-09-12).

Per-TU wall time from the two builds' `.ninja_log` (8 TUs in flight, so each is inflated by contention the same way):

| TU | v11 s | v11x s | v11x cicc / ptxas s (`-time`) | speed-up |
|---|---|---|---|---|
| flash_fwd_hdim64_fp16_sm80 (non-causal) | 2406 | 271 | 196 / 61.5 | 8.9x |
| flash_fwd_hdim64_fp16_causal_sm80 | 1783 | 169 | 123 / 33.7 | 10.6x |
| flash_fwd_hdim128_fp16_sm80 (non-causal) | 1208 | 340 | 250 / 75.1 | 3.6x |
| flash_fwd_hdim128_fp16_causal_sm80 | 1026 | 236 | 177 / 46.5 | 4.3x |
| flash_bwd_hdim64_fp16_sm80 | 558 | 274 | 237 / 24.0 | 2.0x |
| flash_bwd_hdim128_fp16_sm80 | 529 | 276 | 241 / 22.7 | 1.9x |

1. **Every TU finished, ptxas is 23-75 s per TU** (the largest single instantiation 3.4 s), against the design's
   30-min gate (target 15). With the per-element path gone, cicc (template instantiation, 123-250 s) is the
   dominant phase, not ptxas. The non-causal hdim64 and both hdim128 CQS-on instantiations, the ones that
   expose the pathology (F15), are among the fastest.
2. **Code size is the trigger (design 4.4, "both are tested").** The CQS-on production instantiations shrink
   from 69.6-74.3K SASS instructions (v11; v9 69.5-70.5K) to 5.4-7.7K, 1.3-1.5x their CQS-off twins
   (3.7-6.7K) and torch's FA-2 (3.9-6.9K); the hdim128 causal CQS-on one from 57.6K to 6.7K. The non-causal
   CQS-on instantiation (`cqs=2`, the runtime flag) now compiles to 5424 instructions with STACK 96 B and 120/156 B
   of spills, where v11's has STACK 640 B (the mis-compile signature of F15). The TU split (K2) is therefore
   not needed for ptxas time; the same-job A/B (`exp_build_ab.json`) says whether the 3x runtime pathology of
   the non-causal CQS-on kernel goes with it.
3. **What the live loop costs in code**: the remaining +1432 / +960 instructions (causal, even_MN / ragged)
   over the CQS-off twin are the verdict loads, the search loop and the poison flag: the "in-loop verdict
   loads + integer/branch bookkeeping" that `ncu_aligned.md` attributes the tax to. REG stays 255 in every
   production instantiation of every module (ptxas -v: 255 registers, 0.5-1.1 s each).
4. **Backward with CQS compiled out is upstream's kernel**: 2152 SASS / STACK 112 B / 108-132 B spills for the
   hdim64 128x128 causal even_MN instantiation vs torch-FA2 2336 / 96 B and flash_attn 2.8.3 STACK 96 B; v11's
   CQS backward is 5624 / 224 B (2.6x the instructions, 2x the stack), the static side of F6's "the 2x is the
   live-tile path". hdim128 64x128: 1968 / 88 B vs torch 2024 / 64 B; v11 4168 / 96 B.
5. Recorded, not gated (3.4): CQS-off instantiations spill more in v11x than torch's (300/308 B vs STACK 88 B:
   a different nvcc), and v11x's CQS-on instantiations spill less than its CQS-off ones (112/160 vs 300/308 B).

Correctness of the experiment build (della-vis1, A100-PCIe-40GB, `tests/test_ultimate_v11x.py`, 16 passed, and
`exp_build_ab.py --n-fwd 65536`): plain and zero patterns bit-identical to v11 (out and lse) for causal and
non-causal, hdim64 and hdim128, L=28032, and v9 bit-identical to v11 on all three non-causal patterns; the real
pattern poisons exactly the predicted row blocks (causal: every non-owner block, 66.7% of rows; non-causal: the
blocks sharing the last column tile's chunk, 33.3%) with NaN out and NaN lse (+inf where v11's is +inf), the other
rows equal v11; NO_CQS backward dk/dv `torch.equal` to v11's on the zero pattern, dq within the atomics' drift.
