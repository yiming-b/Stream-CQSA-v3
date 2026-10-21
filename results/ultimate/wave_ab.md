# U2 wave A/B: cqsa_native_v2 vs cqsa_native vs per-task v12f / v12b vs monolithic FlashAttention

GPU NVIDIA A100-SXM4-80GB; node della-l09g3; job 15958597; torch 2.10.0+cu130; flash_attn 2.8.3 /scratch/gpfs/AKEY/yb2807/VENVs/CQSA311/lib/python3.11/site-packages/flash_attn/__init__.py; nvidia-smi `NVIDIA A100-SXM4-80GB, 825 MHz, 1410 MHz, 81920 MiB`. 1 warm-up + 3 timed runs (2 at N=1M, 1 for its engine rows), CUDA events, median (min also stored). fp16, H=8, D=64, causal; every arm on prepared inputs (launches only) except the engine rows. Modules: {"v11": "/scratch/gpfs/AKEY/yb2807/Stream-CQSA-dev/next/kernel/stable/cqsa_cuda_next_v11.cpython-311-x86_64-linux-gnu.so", "v12f": "/scratch/gpfs/AKEY/yb2807/Stream-CQSA-dev/next/kernel/v12f/frozen/b3/cqsa_cuda_next_v12f.cpython-311-x86_64-linux-gnu.so", "v12b": "/scratch/gpfs/AKEY/yb2807/Stream-CQSA-dev/next/kernel/v12b/frozen/b5/cqsa_cuda_next_v12b.cpython-311-x86_64-linux-gnu.so", "v2": "/scratch/gpfs/AKEY/yb2807/Stream-CQSA-dev/next/native_v2/frozen/b2/cqsa_native_v2.cpython-311-x86_64-linux-gnu.so", "old": "/scratch/gpfs/AKEY/yb2807/Stream-CQSA-dev/next/native/cqsa_native.cpython-311-x86_64-linux-gnu.so"}.

## Forward (ms; one launch per wave for the wave arms, one launch per task for the loops)

| N | c | W | waves | L | v2 wave | v2 in place | old wave | old in place | v12f loop | v11 loop | flash_attn mono | SDPA mono | v2 / flash_attn | v2 / v12f loop | v2 / old | W=1 v2 / v12f | equal v12f | equal old | routes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 131072 | 7 | 7 | [7] | 56064-56320 | 91.07 | 92.39 | 122.60 | 126.83 | 90.59 | 125.92 | 90.85 | 95.82 | **1.002** | 1.005 | 0.743 | **1.017** (13.12 / 12.89 ms at L=56064) | True | True | [0] |
| 131072 | 73 | 73 | [32, 32, 9] | 16128-16384 | 93.45 | 94.90 | 134.56 | 138.86 | 106.16 | 162.21 | 91.32 | 95.95 | **1.023** | 0.880 | 0.695 | **1.018** (1.52 / 1.49 ms at L=16128) | True | True | [0] |
| 1048576 | 7 | 7 | [2, 2, 2, 1] | 449280-449536 | 5886.64 | 5959.60 | 7668.68 | 7891.11 | 5736.93 | nan | 6052.03 | 6343.02 | **0.973** | 1.026 | 0.768 | **1.026** (840.96 / 819.93 ms at L=449280) | True | True | [0] |

Device us per kernel kind of one v2 wave call (census / aligned / legacy no-op / memset): N=131072 c=7: {"other": 1.2, "census": 16.1, "aligned": 90228.3, "legacy": 42.8}; N=131072 c=73: {"other": 3.5, "census": 25.9, "aligned": 93391.6, "legacy": 128.4}; N=1048576 c=7: {"other": 4.6, "census": 482.8, "aligned": 5884317.6, "legacy": 338.8}

## Backward (ms; global lse and rowsum(dO*O) supplied; one launch per wave / per task)

| N | c | W | waves | v2 bwd wave | old bwd wave | v12b loop | v11 loop | flash_attn mono bwd | SDPA mono bwd | v2 / flash_attn | v2 / v12b loop | old / v2 | W=1 v2 / v12b | dk/dv equal v12b | dq max d | dk/dv equal old | routes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 131072 | 7 | 7 | [7] | 236.15 | 607.14 | 239.57 | 594.20 | 237.36 | 256.75 | **0.995** | 0.986 | **2.57x** | **0.998** (34.26 / 34.32 ms at L=56064) | True | 2.4e-04 | True | [0] |
| 131072 | 73 | 73 | [32, 32, 9] | 240.99 | 584.33 | 251.08 | 592.48 | 237.43 | 256.92 | **1.015** | 0.960 | **2.42x** | **0.996** (3.53 / 3.54 ms at L=16128) | True | 2.4e-04 | True | [0] |
| 1048576 | 7 | 7 | [2, 2, 2, 1] | 14947.57 | 39099.39 | 15001.72 | nan | 15099.30 | 16329.78 | **0.990** | 0.996 | **2.62x** | **1.001** (2135.37 / 2134.08 ms at L=449280) | True | 6.1e-05 | True | [0] |

## Engine end to end (wave_forward / wave_backward: tables, gathers or block maps, merges, scatter-adds)

| N | c | v2 fwd gathered | v2 fwd in place | old fwd gathered | old fwd in place | v2 bwd | old bwd | old / v2 bwd | v2 == old fwd | in place == gathered |
|---|---|---|---|---|---|---|---|---|---|---|
| 131072 | 7 | 118.0 | 107.0 | 147.1 | 139.1 | 269.2 | 630.1 | 2.34 | True | True |
| 131072 | 73 | 155.6 | 117.4 | 195.9 | 163.0 | 320.0 | 635.1 | 1.98 | True | True |
| 1048576 | 7 | 6042.9 | 6037.4 | 7845.4 | 7978.5 | 15188.9 | 39287.3 | 2.59 | True | True |

## Targets (docs/ultimate/03_kernels_report.md section 9, package U2)

- forward N=131072 c=7: W=1 v2 / v12f = 1.017 (target within 1%: NOT met); one-launch wave / monolithic flash_attn = 1.002
- forward N=131072 c=73: W=1 v2 / v12f = 1.018 (target within 1%: NOT met); one-launch wave / monolithic flash_attn = 1.023 (target <= 1.10 at c=73: met)
- forward N=1048576 c=7: W=1 v2 / v12f = 1.026 (target within 1%: NOT met); one-launch wave / monolithic flash_attn = 0.973
- backward N=131072 c=7: W=1 v2 / v12b = 0.998 (target within 1%: met); old bwd_wave / v2 = 2.57x (target >= 2x: met); v2 / monolithic flash_attn bwd = 0.995
- backward N=131072 c=73: W=1 v2 / v12b = 0.996 (target within 1%: met); old bwd_wave / v2 = 2.42x (target >= 2x: met); v2 / monolithic flash_attn bwd = 1.015
- backward N=1048576 c=7: W=1 v2 / v12b = 1.001 (target within 1%: met); old bwd_wave / v2 = 2.62x (target >= 2x: met); v2 / monolithic flash_attn bwd = 0.990
