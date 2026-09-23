# Forward A/B (P3): v12f vs v11 / v11x / v9 / flash_attn / SDPA-flash

Job **14319938**, della-l07g6, **NVIDIA A100-SXM4-80GB**, torch 2.10.0+cu130, flash_attn 2.8.3, fp16, H=8, 1 warm-up + 5 timed runs, CUDA events, median (min also stored). Harness `benchmarks/ultimate/fwd_ab.py`, raw rows `fwd_ab.json`. Kernels: flash_attn ['void flash::flash_fwd_kernel<Flash_fwd_kernel_traits<64, 128, 128, 4, false, false, cutlass::half_t, Flash_kernel_traits<64, 128, 128, 4, cutlass::half_t> >, false, true, false, false, true, true, false, false>(flash::Flash_fwd_params)']; SDPA-flash ['void pytorch_flash::flash_fwd_kernel<Flash_fwd_kernel_traits<64, 128, 128, 4, false, false, cutlass::half_t, Flash_kernel_traits<64, 128, 128, 4, cutlass::half_t> >, false, true, false, false, true, true, false, false>(pytorch_flash::Flash_fwd_params)'].

## aligned c=7 task, N=131072 (L=56064), ms (ratio vs flash_attn / vs SDPA-flash)

v12f kernel columns: device us of one profiled call (census / ALIGNED / LEGACY; the losing kernel's CTAs return at once). The census is inside every timed v12f CQS row.

| causal | D | flash_attn | SDPA | module | pattern | flags | ms | /flash_attn | /SDPA | equal v11 | route | census us | ALIGNED us | LEGACY us |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 64 | 18.23 | 19.64 | v11 | plain | - | 17.30 | 0.949 | 0.881 | - | - | - | - | - |
| 1 | 64 | 18.23 | 19.64 | v11 | zero | - | 22.51 | 1.235 | 1.146 | - | - | - | - | - |
| 1 | 64 | 18.23 | 19.64 | v11 | real | - | 17.88 | 0.981 | 0.910 | - | - | - | - | - |
| 1 | 64 | 18.23 | 19.64 | v11x | plain | - | 17.24 | 0.946 | 0.878 | True | - | - | - | - |
| 1 | 64 | 18.23 | 19.64 | v11x | zero | - | 19.43 | 1.066 | 0.989 | True | - | - | - | - |
| 1 | 64 | 18.23 | 19.64 | v12f | plain | - | 17.23 | 0.945 | 0.877 | True | - | - | - | - |
| 1 | 64 | 18.23 | 19.64 | v12f | zero | 0 | 17.00 | 0.933 | 0.865 | True | 0 | 10.4 | 16948.5 | 8.6 |
| 1 | 64 | 18.23 | 19.64 | v12f | real | 0 | 12.99 | 0.713 | 0.661 | True | 0 | 10.7 | 12917.9 | 8.4 |
| 1 | 64 | 18.23 | 19.64 | v12f | zero | 2 | 16.19 | 0.888 | 0.824 | True | 0 | 10.6 | 16127.8 | 8.4 |
| 1 | 64 | 18.23 | 19.64 | v12f | real | 2 | 12.62 | 0.692 | 0.642 | True | 0 | 10.4 | 12559.8 | 8.4 |
| 1 | 128 | 29.90 | 32.41 | v11 | plain | - | 29.67 | 0.992 | 0.916 | - | - | - | - | - |
| 1 | 128 | 29.90 | 32.41 | v11 | zero | - | 40.62 | 1.359 | 1.253 | - | - | - | - | - |
| 1 | 128 | 29.90 | 32.41 | v11 | real | - | 31.80 | 1.064 | 0.981 | - | - | - | - | - |
| 1 | 128 | 29.90 | 32.41 | v11x | plain | - | 29.58 | 0.989 | 0.913 | True | - | - | - | - |
| 1 | 128 | 29.90 | 32.41 | v11x | zero | - | 34.03 | 1.138 | 1.050 | True | - | - | - | - |
| 1 | 128 | 29.90 | 32.41 | v12f | plain | - | 29.36 | 0.982 | 0.906 | True | - | - | - | - |
| 1 | 128 | 29.90 | 32.41 | v12f | zero | 0 | 29.92 | 1.001 | 0.923 | True | 0 | 15.3 | 29895.7 | 8.4 |
| 1 | 128 | 29.90 | 32.41 | v12f | real | 0 | 22.51 | 0.753 | 0.695 | True | 0 | 14.9 | 22448.0 | 8.7 |
| 1 | 128 | 29.90 | 32.41 | v12f | zero | 2 | 28.09 | 0.940 | 0.867 | True | 0 | 14.8 | 28020.5 | 8.1 |
| 1 | 128 | 29.90 | 32.41 | v12f | real | 2 | 21.86 | 0.731 | 0.675 | True | 0 | 14.8 | 21784.7 | 8.2 |
| 0 | 64 | 33.05 | 34.16 | v11 | plain | - | 33.49 | 1.013 | 0.981 | - | - | - | - | - |
| 0 | 64 | 33.05 | 34.16 | v11 | zero | - | 204.05 | 6.175 | 5.974 | - | - | - | - | - |
| 0 | 64 | 33.05 | 34.16 | v11 | real | - | 157.33 | 4.761 | 4.606 | - | - | - | - | - |
| 0 | 64 | 33.05 | 34.16 | v11x | plain | - | 33.14 | 1.003 | 0.970 | True | - | - | - | - |
| 0 | 64 | 33.05 | 34.16 | v11x | zero | - | 34.73 | 1.051 | 1.017 | True | - | - | - | - |
| 0 | 64 | 33.05 | 34.16 | v9 | plain | - | 42.38 | 1.282 | 1.241 | True | - | - | - | - |
| 0 | 64 | 33.05 | 34.16 | v9 | zero | - | 47.28 | 1.431 | 1.384 | True | - | - | - | - |
| 0 | 64 | 33.05 | 34.16 | v9 | real | - | 38.17 | 1.155 | 1.117 | True | - | - | - | - |
| 0 | 64 | 33.05 | 34.16 | v12f | plain | - | 33.49 | 1.013 | 0.980 | True | - | - | - | - |
| 0 | 64 | 33.05 | 34.16 | v12f | zero | 0 | 32.29 | 0.977 | 0.945 | True | 0 | 11.9 | 32237.5 | 8.6 |
| 0 | 64 | 33.05 | 34.16 | v12f | real | 0 | 25.19 | 0.762 | 0.737 | True | 0 | 11.8 | 25130.4 | 8.5 |
| 0 | 64 | 33.05 | 34.16 | v12f | zero | 2 | 32.60 | 0.986 | 0.954 | True | 0 | 12.0 | 32412.7 | 8.4 |
| 0 | 64 | 33.05 | 34.16 | v12f | real | 2 | 25.36 | 0.768 | 0.743 | True | 0 | 12.2 | 25325.6 | 8.5 |
| 0 | 128 | 56.94 | 59.72 | v11 | plain | - | 57.27 | 1.006 | 0.959 | - | - | - | - | - |
| 0 | 128 | 56.94 | 59.72 | v11 | zero | - | 257.15 | 4.516 | 4.306 | - | - | - | - | - |
| 0 | 128 | 56.94 | 59.72 | v11 | real | - | 200.96 | 3.529 | 3.365 | - | - | - | - | - |
| 0 | 128 | 56.94 | 59.72 | v11x | plain | - | 56.66 | 0.995 | 0.949 | True | - | - | - | - |
| 0 | 128 | 56.94 | 59.72 | v11x | zero | - | 64.34 | 1.130 | 1.077 | True | - | - | - | - |
| 0 | 128 | 56.94 | 59.72 | v9 | plain | - | 82.97 | 1.457 | 1.389 | True | - | - | - | - |
| 0 | 128 | 56.94 | 59.72 | v9 | zero | - | 99.31 | 1.744 | 1.663 | True | - | - | - | - |
| 0 | 128 | 56.94 | 59.72 | v9 | real | - | 75.99 | 1.335 | 1.273 | True | - | - | - | - |
| 0 | 128 | 56.94 | 59.72 | v12f | plain | - | 56.85 | 0.999 | 0.952 | True | - | - | - | - |
| 0 | 128 | 56.94 | 59.72 | v12f | zero | 0 | 56.91 | 1.000 | 0.953 | True | 0 | 17.7 | 56544.7 | 8.4 |
| 0 | 128 | 56.94 | 59.72 | v12f | real | 0 | 44.35 | 0.779 | 0.743 | True | 0 | 17.8 | 44216.5 | 8.5 |
| 0 | 128 | 56.94 | 59.72 | v12f | zero | 2 | 56.91 | 1.000 | 0.953 | True | 0 | 17.7 | 56845.7 | 8.0 |
| 0 | 128 | 56.94 | 59.72 | v12f | real | 2 | 44.37 | 0.779 | 0.743 | True | 0 | 17.6 | 44291.7 | 8.4 |

## aligned c=7 task, N=524288 (L=224640), ms (ratio vs flash_attn / vs SDPA-flash)

v12f kernel columns: device us of one profiled call (census / ALIGNED / LEGACY; the losing kernel's CTAs return at once). The census is inside every timed v12f CQS row.

| causal | D | flash_attn | SDPA | module | pattern | flags | ms | /flash_attn | /SDPA | equal v11 | route | census us | ALIGNED us | LEGACY us |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 64 | 267.36 | 282.14 | v11 | plain | - | 279.63 | 1.046 | 0.991 | - | - | - | - | - |
| 1 | 64 | 267.36 | 282.14 | v11 | zero | - | 342.49 | 1.281 | 1.214 | - | - | - | - | - |
| 1 | 64 | 267.36 | 282.14 | v11 | real | - | 269.66 | 1.009 | 0.956 | - | - | - | - | - |
| 1 | 64 | 267.36 | 282.14 | v11x | plain | - | 280.42 | 1.049 | 0.994 | True | - | - | - | - |
| 1 | 64 | 267.36 | 282.14 | v11x | zero | - | 299.83 | 1.121 | 1.063 | True | - | - | - | - |
| 1 | 64 | 267.36 | 282.14 | v12f | plain | - | 280.22 | 1.048 | 0.993 | True | - | - | - | - |
| 1 | 64 | 267.36 | 282.14 | v12f | zero | 0 | 274.69 | 1.027 | 0.974 | True | 0 | 31.3 | 274207.1 | 26.3 |
| 1 | 64 | 267.36 | 282.14 | v12f | real | 0 | 206.59 | 0.773 | 0.732 | True | 0 | 30.6 | 207198.0 | 25.7 |
| 1 | 64 | 267.36 | 282.14 | v12f | zero | 2 | 261.00 | 0.976 | 0.925 | True | 0 | 30.4 | 261347.9 | 25.8 |
| 1 | 64 | 267.36 | 282.14 | v12f | real | 2 | 203.50 | 0.761 | 0.721 | True | 0 | 30.0 | 203127.6 | 26.0 |

## c=73, N=131072: 73 tasks, L 16128-16384, causal D=64, per-task medians summed

| arm | sum ms | /flash_attn sum | /monolithic flash_attn |
|---|---|---|---|
| v11_plain | 125.9 | 1.024 | 1.393 |
| v11_real | 164.7 | 1.340 | 1.823 |
| v12f_plain | 126.3 | 1.027 | 1.398 |
| v12f_real | 113.9 | 0.927 | 1.261 |
| v12f_real_rev | 108.4 | 0.882 | 1.200 |
| flash_attn | 122.9 | 1.000 | 1.361 |
| sdpa_flash | 129.2 | 1.051 | 1.430 |

Monolithic flash_attn at N=131072: 90.3 ms (SDPA-flash 95.8). Census routes over the tasks: [0]; v12f real equal to v11 on every task: True. v12f real (flags 0) device time summed over the tasks by kernel (us): {'other': 82.69, 'census': 383.2320000000003, 'aligned': 110465.63999999998, 'legacy': 332.57099999999946}.

## Targets

- causal=1 D=64: v12f zero / v11 plain = **0.983** (target <= 1.05); v12f zero / v12f plain = 0.987; v12f real / flash_attn = **0.713** (target <= 0.82)
- causal=1 D=128: v12f zero / v11 plain = **1.008** (target <= 1.05); v12f zero / v12f plain = 1.019; v12f real / flash_attn = **0.753**
- causal=0 D=64: v12f zero / v11 plain = **0.964** (target <= 1.05); v12f zero / v12f plain = 0.964; v12f real / flash_attn = **0.762**
- causal=0 D=128: v12f zero / v11 plain = **0.994** (target <= 1.05); v12f zero / v12f plain = 1.001; v12f real / flash_attn = **0.779**
