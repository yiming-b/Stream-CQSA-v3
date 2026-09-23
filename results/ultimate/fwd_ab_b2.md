# Forward A/B (P3): v12f vs v11 / v11x / v9 / flash_attn / SDPA-flash

Job **14324080**, della-l08g2, **NVIDIA A100-SXM4-80GB**, torch 2.10.0+cu130, flash_attn 2.8.3, fp16, H=8, 1 warm-up + 5 timed runs, CUDA events, median (min also stored). Harness `benchmarks/ultimate/fwd_ab.py`, raw rows `fwd_ab.json`. Kernels: flash_attn ['void flash::flash_fwd_kernel<Flash_fwd_kernel_traits<64, 128, 128, 4, false, false, cutlass::half_t, Flash_kernel_traits<64, 128, 128, 4, cutlass::half_t> >, false, true, false, false, true, true, false, false>(flash::Flash_fwd_params)']; SDPA-flash ['void pytorch_flash::flash_fwd_kernel<Flash_fwd_kernel_traits<64, 128, 128, 4, false, false, cutlass::half_t, Flash_kernel_traits<64, 128, 128, 4, cutlass::half_t> >, false, true, false, false, true, true, false, false>(pytorch_flash::Flash_fwd_params)'].

## aligned c=7 task, N=131072 (L=56064), ms (ratio vs flash_attn / vs SDPA-flash)

v12f kernel columns: device us of one profiled call (census / ALIGNED / LEGACY; the losing kernel's CTAs return at once). The census is inside every timed v12f CQS row.

| causal | D | flash_attn | SDPA | module | pattern | flags | ms | /flash_attn | /SDPA | equal v11 | route | census us | ALIGNED us | LEGACY us |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 64 | 18.22 | 17.83 | v11 | plain | - | 17.29 | 0.949 | 0.969 | - | - | - | - | - |
| 1 | 64 | 18.22 | 17.83 | v11 | zero | - | 22.59 | 1.240 | 1.267 | - | - | - | - | - |
| 1 | 64 | 18.22 | 17.83 | v11 | real | - | 17.88 | 0.982 | 1.003 | - | - | - | - | - |
| 1 | 64 | 18.22 | 17.83 | v11x | plain | - | 17.31 | 0.950 | 0.971 | True | - | - | - | - |
| 1 | 64 | 18.22 | 17.83 | v11x | zero | - | 19.44 | 1.067 | 1.090 | True | - | - | - | - |
| 1 | 64 | 18.22 | 17.83 | v12f | plain | - | 17.30 | 0.949 | 0.970 | True | - | - | - | - |
| 1 | 64 | 18.22 | 17.83 | v12f | zero | 0 | 16.23 | 0.891 | 0.910 | True | 0 | 10.6 | 16156.7 | 8.4 |
| 1 | 64 | 18.22 | 17.83 | v12f | real | 0 | 12.65 | 0.694 | 0.709 | True | 0 | 10.7 | 12606.1 | 8.4 |
| 1 | 64 | 18.22 | 17.83 | v12f | zero | 4 | 17.05 | 0.936 | 0.956 | True | 0 | 10.4 | 16974.8 | 8.5 |
| 1 | 64 | 18.22 | 17.83 | v12f | real | 4 | 13.02 | 0.715 | 0.730 | True | 0 | 10.4 | 12958.8 | 8.6 |
| 1 | 64 | 18.22 | 17.83 | v12f | zero | 2 | 16.32 | 0.896 | 0.915 | True | 0 | 10.5 | 16182.2 | 7.7 |
| 1 | 64 | 18.22 | 17.83 | v12f | real | 2 | 12.65 | 0.695 | 0.710 | True | 0 | 10.5 | 12676.2 | 8.0 |
| 1 | 128 | 29.71 | 32.07 | v11 | plain | - | 29.69 | 1.000 | 0.926 | - | - | - | - | - |
| 1 | 128 | 29.71 | 32.07 | v11 | zero | - | 40.51 | 1.364 | 1.263 | - | - | - | - | - |
| 1 | 128 | 29.71 | 32.07 | v11 | real | - | 31.83 | 1.072 | 0.993 | - | - | - | - | - |
| 1 | 128 | 29.71 | 32.07 | v11x | plain | - | 29.58 | 0.996 | 0.922 | True | - | - | - | - |
| 1 | 128 | 29.71 | 32.07 | v11x | zero | - | 33.91 | 1.141 | 1.057 | True | - | - | - | - |
| 1 | 128 | 29.71 | 32.07 | v12f | plain | - | 29.90 | 1.007 | 0.932 | True | - | - | - | - |
| 1 | 128 | 29.71 | 32.07 | v12f | zero | 0 | 28.18 | 0.949 | 0.879 | True | 0 | 14.8 | 28083.9 | 8.3 |
| 1 | 128 | 29.71 | 32.07 | v12f | real | 0 | 21.91 | 0.737 | 0.683 | True | 0 | 14.8 | 21817.9 | 8.3 |
| 1 | 128 | 29.71 | 32.07 | v12f | zero | 4 | 29.81 | 1.003 | 0.930 | True | 0 | 15.1 | 29908.8 | 8.4 |
| 1 | 128 | 29.71 | 32.07 | v12f | real | 4 | 22.58 | 0.760 | 0.704 | True | 0 | 14.7 | 22516.3 | 8.6 |
| 1 | 128 | 29.71 | 32.07 | v12f | zero | 2 | 28.14 | 0.947 | 0.878 | True | 0 | 14.8 | 28083.0 | 8.2 |
| 1 | 128 | 29.71 | 32.07 | v12f | real | 2 | 21.90 | 0.737 | 0.683 | True | 0 | 15.0 | 21849.0 | 8.2 |
| 0 | 64 | 33.07 | 34.16 | v11 | plain | - | 33.51 | 1.013 | 0.981 | - | - | - | - | - |
| 0 | 64 | 33.07 | 34.16 | v11 | zero | - | 204.73 | 6.192 | 5.993 | - | - | - | - | - |
| 0 | 64 | 33.07 | 34.16 | v11 | real | - | 157.49 | 4.763 | 4.610 | - | - | - | - | - |
| 0 | 64 | 33.07 | 34.16 | v11x | plain | - | 33.28 | 1.006 | 0.974 | True | - | - | - | - |
| 0 | 64 | 33.07 | 34.16 | v11x | zero | - | 34.73 | 1.050 | 1.017 | True | - | - | - | - |
| 0 | 64 | 33.07 | 34.16 | v9 | plain | - | 42.42 | 1.283 | 1.242 | True | - | - | - | - |
| 0 | 64 | 33.07 | 34.16 | v9 | zero | - | 47.42 | 1.434 | 1.388 | True | - | - | - | - |
| 0 | 64 | 33.07 | 34.16 | v9 | real | - | 38.14 | 1.153 | 1.116 | True | - | - | - | - |
| 0 | 64 | 33.07 | 34.16 | v12f | plain | - | 33.50 | 1.013 | 0.981 | True | - | - | - | - |
| 0 | 64 | 33.07 | 34.16 | v12f | zero | 0 | 32.44 | 0.981 | 0.950 | True | 0 | 12.0 | 32486.4 | 8.5 |
| 0 | 64 | 33.07 | 34.16 | v12f | real | 0 | 25.36 | 0.767 | 0.742 | True | 0 | 11.8 | 25244.6 | 8.5 |
| 0 | 64 | 33.07 | 34.16 | v12f | zero | 4 | 32.39 | 0.979 | 0.948 | True | 0 | 11.9 | 32339.3 | 8.4 |
| 0 | 64 | 33.07 | 34.16 | v12f | real | 4 | 25.33 | 0.766 | 0.741 | True | 0 | 11.8 | 25260.3 | 8.4 |
| 0 | 64 | 33.07 | 34.16 | v12f | zero | 2 | 32.40 | 0.980 | 0.949 | True | 0 | 12.0 | 32536.0 | 8.5 |
| 0 | 64 | 33.07 | 34.16 | v12f | real | 2 | 25.30 | 0.765 | 0.741 | True | 0 | 11.8 | 25240.5 | 8.3 |
| 0 | 128 | 56.75 | 59.71 | v11 | plain | - | 56.74 | 1.000 | 0.950 | - | - | - | - | - |
| 0 | 128 | 56.75 | 59.71 | v11 | zero | - | 256.37 | 4.517 | 4.294 | - | - | - | - | - |
| 0 | 128 | 56.75 | 59.71 | v11 | real | - | 201.52 | 3.551 | 3.375 | - | - | - | - | - |
| 0 | 128 | 56.75 | 59.71 | v11x | plain | - | 56.80 | 1.001 | 0.951 | True | - | - | - | - |
| 0 | 128 | 56.75 | 59.71 | v11x | zero | - | 64.33 | 1.133 | 1.077 | True | - | - | - | - |
| 0 | 128 | 56.75 | 59.71 | v9 | plain | - | 82.62 | 1.456 | 1.384 | True | - | - | - | - |
| 0 | 128 | 56.75 | 59.71 | v9 | zero | - | 99.41 | 1.752 | 1.665 | True | - | - | - | - |
| 0 | 128 | 56.75 | 59.71 | v9 | real | - | 76.09 | 1.341 | 1.274 | True | - | - | - | - |
| 0 | 128 | 56.75 | 59.71 | v12f | plain | - | 56.76 | 1.000 | 0.951 | True | - | - | - | - |
| 0 | 128 | 56.75 | 59.71 | v12f | zero | 0 | 56.70 | 0.999 | 0.950 | True | 0 | 17.7 | 56568.0 | 8.3 |
| 0 | 128 | 56.75 | 59.71 | v12f | real | 0 | 44.34 | 0.781 | 0.743 | True | 0 | 17.6 | 44284.2 | 8.4 |
| 0 | 128 | 56.75 | 59.71 | v12f | zero | 4 | 56.93 | 1.003 | 0.953 | True | 0 | 17.8 | 56861.6 | 8.5 |
| 0 | 128 | 56.75 | 59.71 | v12f | real | 4 | 44.36 | 0.782 | 0.743 | True | 0 | 17.6 | 44136.5 | 8.3 |
| 0 | 128 | 56.75 | 59.71 | v12f | zero | 2 | 56.92 | 1.003 | 0.953 | True | 0 | 17.7 | 56744.5 | 8.3 |
| 0 | 128 | 56.75 | 59.71 | v12f | real | 2 | 44.36 | 0.782 | 0.743 | True | 0 | 17.6 | 44312.1 | 8.4 |

## aligned c=7 task, N=524288 (L=224640), ms (ratio vs flash_attn / vs SDPA-flash)

v12f kernel columns: device us of one profiled call (census / ALIGNED / LEGACY; the losing kernel's CTAs return at once). The census is inside every timed v12f CQS row.

| causal | D | flash_attn | SDPA | module | pattern | flags | ms | /flash_attn | /SDPA | equal v11 | route | census us | ALIGNED us | LEGACY us |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 64 | 267.46 | 282.47 | v11 | plain | - | 279.93 | 1.047 | 0.991 | - | - | - | - | - |
| 1 | 64 | 267.46 | 282.47 | v11 | zero | - | 343.16 | 1.283 | 1.215 | - | - | - | - | - |
| 1 | 64 | 267.46 | 282.47 | v11 | real | - | 269.91 | 1.009 | 0.956 | - | - | - | - | - |
| 1 | 64 | 267.46 | 282.47 | v11x | plain | - | 280.29 | 1.048 | 0.992 | True | - | - | - | - |
| 1 | 64 | 267.46 | 282.47 | v11x | zero | - | 300.52 | 1.124 | 1.064 | True | - | - | - | - |
| 1 | 64 | 267.46 | 282.47 | v12f | plain | - | 281.47 | 1.052 | 0.996 | True | - | - | - | - |
| 1 | 64 | 267.46 | 282.47 | v12f | zero | 0 | 260.20 | 0.973 | 0.921 | True | 0 | 30.0 | 259784.6 | 25.6 |
| 1 | 64 | 267.46 | 282.47 | v12f | real | 0 | 203.30 | 0.760 | 0.720 | True | 0 | 30.2 | 202154.6 | 25.2 |
| 1 | 64 | 267.46 | 282.47 | v12f | zero | 4 | 275.18 | 1.029 | 0.974 | True | 0 | 31.3 | 275005.6 | 27.0 |
| 1 | 64 | 267.46 | 282.47 | v12f | real | 4 | 207.82 | 0.777 | 0.736 | True | 0 | 30.6 | 207829.1 | 26.1 |
| 1 | 64 | 267.46 | 282.47 | v12f | zero | 2 | 261.58 | 0.978 | 0.926 | True | 0 | 29.8 | 259916.3 | 25.9 |
| 1 | 64 | 267.46 | 282.47 | v12f | real | 2 | 202.88 | 0.759 | 0.718 | True | 0 | 29.9 | 203015.1 | 25.7 |

## aligned c=7 task, N=16384 (L=6912), ms (ratio vs flash_attn / vs SDPA-flash)

v12f kernel columns: device us of one profiled call (census / ALIGNED / LEGACY; the losing kernel's CTAs return at once). The census is inside every timed v12f CQS row.

| causal | D | flash_attn | SDPA | module | pattern | flags | ms | /flash_attn | /SDPA | equal v11 | route | census us | ALIGNED us | LEGACY us |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 64 | 0.44 | 0.42 | v11 | plain | - | 0.42 | 0.962 | 1.002 | - | - | - | - | - |
| 1 | 64 | 0.44 | 0.42 | v11 | zero | - | 0.66 | 1.504 | 1.565 | - | - | - | - | - |
| 1 | 64 | 0.44 | 0.42 | v11 | real | - | 0.57 | 1.287 | 1.339 | - | - | - | - | - |
| 1 | 64 | 0.44 | 0.42 | v11x | plain | - | 0.42 | 0.956 | 0.995 | True | - | - | - | - |
| 1 | 64 | 0.44 | 0.42 | v11x | zero | - | 0.51 | 1.170 | 1.218 | True | - | - | - | - |
| 1 | 64 | 0.44 | 0.42 | v12f | plain | - | 0.43 | 0.967 | 1.007 | True | - | - | - | - |
| 1 | 64 | 0.44 | 0.42 | v12f | zero | 0 | 0.43 | 0.979 | 1.019 | True | 0 | 4.3 | 373.9 | 3.8 |
| 1 | 64 | 0.44 | 0.42 | v12f | real | 0 | 0.35 | 0.806 | 0.838 | True | 0 | 4.2 | 304.8 | 3.7 |
| 1 | 64 | 0.44 | 0.42 | v12f | zero | 4 | 0.46 | 1.047 | 1.089 | True | 0 | 4.0 | 410.1 | 3.8 |
| 1 | 64 | 0.44 | 0.42 | v12f | real | 4 | 0.37 | 0.832 | 0.866 | True | 0 | 4.2 | 318.0 | 3.5 |
| 1 | 64 | 0.44 | 0.42 | v12f | zero | 2 | 0.43 | 0.978 | 1.018 | True | 0 | 4.1 | 385.8 | 3.6 |
| 1 | 64 | 0.44 | 0.42 | v12f | real | 2 | 0.36 | 0.809 | 0.842 | True | 0 | 4.2 | 305.6 | 3.5 |

## c=73, N=131072: 73 tasks, L 16128-16384, causal D=64, per-task medians summed

| arm | sum ms | /flash_attn sum | /monolithic flash_attn |
|---|---|---|---|
| v11_plain | 126.3 | 1.022 | 1.395 |
| v11_real | 164.6 | 1.333 | 1.818 |
| v12f_plain | 125.9 | 1.020 | 1.391 |
| v12f_real | 108.8 | 0.881 | 1.202 |
| v12f_real_asc | 114.5 | 0.927 | 1.264 |
| v12f_real_rev | 108.8 | 0.881 | 1.201 |
| flash_attn | 123.5 | 1.000 | 1.364 |
| sdpa_flash | 129.2 | 1.046 | 1.427 |

Monolithic flash_attn at N=131072: 90.5 ms (SDPA-flash 95.8). Census routes over the tasks: [0]; v12f real equal to v11 on every task: True. v12f real (flags 0) device time summed over the tasks by kernel (us): {'other': 82.5639999999994, 'census': 383.7150000000003, 'aligned': 105466.64500000003, 'legacy': 326.9139999999984}.

## Mixed-tile patterns (LEGACY), N=131072 unaligned c=7 task (L=56173), D=64

| causal | pattern | module | ms | /flash_attn | equal v11 | route | census us | ALIGNED us | LEGACY us |
|---|---|---|---|---|---|---|---|---|---|
| 1 | unaligned | v11 | 24.58 | 1.439 | - | - | - | - | - |
| 1 | unaligned | v12f | 24.55 | 1.436 | True | 1 | 11.3 | 8.1 | 24643.6 |
| 1 | random3bit | v11 | 158.72 | 9.288 | - | - | - | - | - |
| 1 | random3bit | v12f | 157.76 | 9.232 | True | 1 | 11.2 | 8.1 | 157938.9 |
| 0 | unaligned | v11 | 194.89 | 5.720 | - | - | - | - | - |
| 0 | unaligned | v9 | 48.81 | 1.432 | True | - | - | - | - |
| 0 | unaligned | v12f | 46.74 | 1.372 | True | 1 | 11.6 | 8.1 | 46813.0 |
| 0 | random3bit | v11 | 515.54 | 15.131 | - | - | - | - | - |
| 0 | random3bit | v9 | 347.53 | 10.200 | True | - | - | - | - |
| 0 | random3bit | v12f | 346.74 | 10.176 | True | 1 | 11.9 | 8.3 | 346868.4 |

## Targets

- causal=1 D=64: v12f zero / v11 plain = **0.939** (target <= 1.05); v12f zero / v12f plain = 0.938; v12f real / flash_attn = **0.694** (target <= 0.82)
- causal=1 D=128: v12f zero / v11 plain = **0.949** (target <= 1.05); v12f zero / v12f plain = 0.942; v12f real / flash_attn = **0.737**
- causal=0 D=64: v12f zero / v11 plain = **0.968** (target <= 1.05); v12f zero / v12f plain = 0.968; v12f real / flash_attn = **0.767**
- causal=0 D=128: v12f zero / v11 plain = **0.999** (target <= 1.05); v12f zero / v12f plain = 0.999; v12f real / flash_attn = **0.781**

## Verdict (build b2 = b1 + descending row-block order as the causal default; job 14324080, della-l08g2)

b2 changes only `flash_api.cpp` (flags 0 = descending for causal, ascending for non-causal; 2 / 4 force either);
the ALIGNED and census SASS are identical to b1's (cuobjdump -sass of the 9 kernels, `cmp` equal). Every v12f row is
`torch.equal` to v11 (fp32 out, lse, fp16 out), mixed patterns included; every aligned pattern routes to ALIGNED.
Job MaxRSS 2.1 GB, 76 s of harness.

1. **The default flip is confirmed:** flags 0 = flags 2 for causal (56K D=64 zero 16.23 / 16.32 ms, real 12.65 / 12.65;
   D=128 28.18 / 28.14, 21.91 / 21.90; 225K 260.20 / 261.58, 203.30 / 202.88; c=73 real sum 108.8 / 108.8) and
   = flags 4 for non-causal (32.44 / 32.39, 25.36 / 25.33; D=128 56.70 / 56.93, 44.34 / 44.36). Ascending (flags 4)
   reproduces b1's flags-0 rows (17.05 vs 17.00, 13.02 vs 12.99, 275.18 vs 274.69 ms).
2. **Targets with the production default (b2):** causal D=64 L=56064 zero / v11 plain **0.939**, real / flash_attn
   **0.694** (vs SDPA 0.709); causal D=128 zero / v11 plain 0.949, real / flash_attn 0.737; non-causal zero / plain
   0.968 (D=64), 0.999 (D=128); L=224640 zero / v11 plain 0.930, real / flash_attn 0.760; c=73 N=131072 real summed
   108.8 ms = **0.881x** the flash_attn per-task sum (123.5), 0.661x v11 real (164.6), 1.202x monolithic flash_attn (90.5).
3. **Small L (c=7, N=16384, aligned L=6912, causal D=64):** v12f zero 0.43 ms = 0.979x flash_attn (1.019x SDPA; v11
   1.504x / 1.565x); real 0.806x flash_attn (0.838x SDPA; v11 1.287x). Design 7.1 P3.1's small-L gates (zero/F <= 1.40,
   real <= 1.6 at L=7023, align=1 then) are met on the aligned layout by a wide margin.
4. **Mixed tiles (LEGACY) and the end of v9.** Unaligned c=7 task (seg_align=0, L=56173) and a random 3-bit pattern route
   to LEGACY (route 1; census 11-12 us, the ALIGNED no-op 8 us). Causal: LEGACY = v11 (24.55 vs 24.58 ms unaligned,
   157.76 vs 158.72 random). **Non-causal: v12f LEGACY 46.74 ms vs v11 194.89 vs v9 48.81 (unaligned); 346.74 vs 515.54
   vs 347.53 (random).** v11's non-causal CQS-on code in its own translation unit (no CQS-off instantiation beside it;
   STACK 312-328 B vs 640-680 B in v11, `v12f_static.md`) is not mis-compiled: with ALIGNED 1.46-1.75x faster than v9
   on aligned tasks, v12f is at or below v9 on every non-causal pattern measured (D=64), so the second extension
   (`CQSA_CUDA_MODULE_NONCAUSAL` = v9) can retire once D=128 mixed non-causal is checked. F15's trigger is therefore
   co-instantiation (v11x shows code size alone also removes it): either fix suffices.
5. Census + dual launch per call: 10.4-17.8 us census and 7.7-8.6 us LEGACY no-op at L=56064; 30 us / 25-27 us at
   224640; 4 us / 3.5-3.8 us at 6912; over the 73 c=73 tasks 0.38 + 0.33 + 0.08 (memset) = 0.79 ms of 108.8 ms (0.73%).
