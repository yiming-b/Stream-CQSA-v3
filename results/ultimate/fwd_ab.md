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

## Verdict (P3, written from `fwd_ab.json` of job 14319938; v12f build b1, `next/kernel/v12f/frozen/b1`)

In build b1 `flags 0` = ascending row-block order everywhere, `flags 2` = descending. Every v12f row is `torch.equal`
to v11 (fp32 out, lse, fp16 out) and the census routed every aligned pattern to ALIGNED (route 0), all 73 c=73 tasks included.

| target (package P3) | measured | verdict |
|---|---|---|
| causal D=64 L=56064: v12f zero <= 1.05x v11 plain | 17.00 / 17.30 = **0.983** (descending 16.19: 0.936) | met |
| causal D=64 L=56064: v12f real <= 0.82x flash_attn | 12.99 / 18.23 = **0.713** (descending 0.692; vs SDPA 0.661 / 0.642) | met |
| non-causal v12f zero <= 1.05x plain (retires v9) | D=64 32.29 / 33.49 = **0.964**; D=128 56.91 / 56.85 = **1.001** | met |
| causal D=128 L=56064 | zero / v11 plain 1.008 (descending 0.947); real / flash_attn 0.753 (0.731) | recorded |
| causal D=64 L=224640 | zero / v11 plain 0.982 (descending 0.933); real / flash_attn 0.773 (0.761) | recorded |
| c=73 N=131072, 73 tasks summed, causal D=64 | real 113.9 ms (descending 108.4) = 0.927 (0.882) x flash_attn per-task sum 122.9, 0.692 (0.658) x v11 real 164.7, 1.261 (1.200) x monolithic flash_attn 90.3 | recorded |

1. **The all-live tax is gone.** v11's zero/plain is 1.301 at 56K causal D=64 (22.51 / 17.30; v11x without the element
   path 1.127); v12f's ALIGNED on the same all-live pattern is 0.987x its own CQS-off kernel (0.940 descending):
   22.51 -> 17.00 ms. D=128: v11 1.369 -> v12f 1.019 (0.957 descending); 225K: v11 1.225 -> 0.980.
   ALIGNED visits exactly the CQS-off kernel's tiles on the zero pattern and runs them as fast (3.8-7.2K SASS vs
   3.3-6.7K, `v12f_static.md`).
2. **Dead-tile skipping now pays in full.** v12f real / v11 real = 0.727 at 56K causal D=64 (12.99 vs 17.88 ms),
   0.708 D=128, 0.766 at 225K: with 22.2% of the causal pair work masked on the c=7 aligned task the forward runs
   at 0.764x (56K) and 0.752x (225K) its all-live time, close to the work ratio.
3. **Non-causal: one kernel, v9 retired.** v11's non-causal CQS-on instantiation is 6.1x plain (204 ms, the F15
   mis-compile); v12f routes aligned non-causal calls to ALIGNED: zero 32.29 ms vs v9 47.28 (1.46x faster), real
   25.19 vs 38.17 (1.52x); D=128 zero 56.91 vs 99.31 (1.75x), real 44.35 vs 75.99 (1.71x). Non-causal MIXED
   patterns (unaligned / random) still run on LEGACY = v11's non-causal CQS-on code, F15-slow in v11 and not
   re-measured in v12f (whose LEGACY TU has STACK 312-328 B vs v11's 640-680 B, `v12f_static.md`): v9 retires for
   aligned tasks now, for mixed non-causal patterns only once that row is measured.
4. **Census + dual launch cost** (profiled, inside every timed row): census 10.4-17.8 us at L=56064 (<= 0.08% of
   the call), 30-31 us at 224640; the LEGACY no-op launch (all CTAs return on route 0) 8-9 us and 26 us; the route
   memset ~1 us. Over the 73 c=73 tasks: census 0.38 ms + LEGACY no-op 0.33 ms + memset 0.08 ms = 0.80 ms of
   113.9 ms (0.7%).
5. **Descending row-block order (flags 2) helps causal, not non-causal.** Causal: zero -4.8% / real -2.8% (D=64),
   -6.1% / -2.9% (D=128), -5.0% / -1.5% (225K), c=73 real -4.8%; non-causal +1.0% / +0.7% (D=64), 0.0% (D=128).
   Kept for causal only: build b2 makes descending the default for causal calls (flags 0), ascending for
   non-causal; flag 4 forces ascending (flash_api.cpp only, the ALIGNED device code is unchanged).
6. Cross-job reproducibility with exp_build_ab (job 14308401, della-l07g4): v11 plain 17.30 = 17.30 ms, v11 zero
   22.51 vs 22.55, v11x zero 19.43 vs 19.35, SDPA 19.64 vs 19.70. In-job flash_attn factor over SDPA-flash (design
   3.3): 1.077 causal D=64, 1.084 D=128, 1.034 / 1.049 non-causal, 1.055 at 225K.

## F23 corrections (2026-09-23, job 14330595 and the kernel reviews; see docs/ultimate/03_kernels_report.md 4.2)

- The flash_attn arm at causal D=64 L=56064 (18.22-18.23 ms) was this job's first timed arm, without a clock warm-up.
  Warmed up (job 14330595) the package takes 16.84 ms against the in-tree plain kernel's 17.59: it is ~4% FASTER than
  our build of the same kernel. Re-based, v12f real / flash_attn = 0.766 (target <= 0.82 still met); the in-tree ratio
  (real / v11 plain 0.731-0.733) is the stable figure. The SDPA arm was bimodal here; the flash_attn/SDPA factor 1.077
  and the "SDPA 19.64 vs 19.70" reproducibility claim are withdrawn.
- 0.939 (zero / v11 plain) includes a 4.8% descending-order scheduling gain; with the same (ascending) order on both
  sides the tax figure is 0.986 (0.985 in 14330595; 0.982 at H=32). The non-causal order penalty was noise (neutral).
- c=73: the end-to-end figure is 1.202x monolithic flash_attn; 0.881x the per-task flash_attn sum is a same-layout
  dense (sparsity) reference.
- Bit-identity with v11 holds for finite K/V (a non-finite V in a dead masking-step tile makes v11's rows NaN, v12f's
  stay finite and match the masked reference).
