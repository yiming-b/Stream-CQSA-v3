# F23 kernel A/B: review-fixed builds v12f b3 / v12b b5

Job **14330595**, della-l08g1, **NVIDIA A100-SXM4-80GB** (`NVIDIA A100-SXM4-80GB, 210 MHz, 1410 MHz, 81920 MiB`), torch 2.10.0+cu130, flash_attn 2.8.3, fp16. our arms 2 warm-up + 7 timed runs, FA arms 3 + 10; CUDA events, median (min stored). Harness `benchmarks/ultimate/kernels_f23.py`, raw rows `kernels_f23.json`. Modules: v11 = `/scratch/gpfs/AKEY/yb2807/Stream-CQSA-dev/next/kernel/stable/cqsa_cuda_next_v11.cpython-311-x86_64-linux-gnu.so`, v11x = `/scratch/gpfs/AKEY/yb2807/Stream-CQSA-dev/next/kernel/v11x/cqsa_cuda_next_v11x.cpython-311-x86_64-linux-gnu.so`, v12f = `/scratch/gpfs/AKEY/yb2807/Stream-CQSA-dev/next/kernel/v12f/frozen/b3/cqsa_cuda_next_v12f.cpython-311-x86_64-linux-gnu.so`, v12b = `/scratch/gpfs/AKEY/yb2807/Stream-CQSA-dev/next/kernel/v12b/frozen/b5/cqsa_cuda_next_v12b.cpython-311-x86_64-linux-gnu.so`.

## Forward, causal D=64, c=7 aligned task (headline re-time, H=8 and H=32)

| L | L%128 | H | arm | flags | ms (median) | min | / flash_attn | / v11 plain (in-tree) | == v11 | route | device us (census / aligned / legacy / memset) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 56064 | 0 | 8 | flash_attn | - | 16.84 | 16.76 | 1 | 0.957 | - | - | - |
| 56064 | 0 | 8 | SDPA-flash | - | 17.84 | 17.74 | 1.060 | 1.014 | - | - | - |
| 56064 | 0 | 8 | v11 plain | None | 17.59 | 17.48 | 1.045 | 1.000 | - | - | - |
| 56064 | 0 | 8 | v11 zero | None | 22.55 | 22.53 | 1.339 | 1.282 | - | - | - |
| 56064 | 0 | 8 | v11 real | None | 17.85 | 17.80 | 1.060 | 1.015 | - | - | - |
| 56064 | 0 | 8 | v12f plain | None | 17.57 | 17.44 | 1.044 | 0.999 | True | - | - |
| 56064 | 0 | 8 | v12f zero | 0 | 16.49 | 16.45 | 0.979 | 0.937 | True | 0 | 10.8 / 16394.4 / 8.5 / 1.2 |
| 56064 | 0 | 8 | v12f real | 0 | 12.90 | 12.87 | 0.766 | 0.733 | True | 0 | 10.8 / 12829.8 / 8.5 / 1.1 |
| 56064 | 0 | 8 | v12f zero | 2 | 16.54 | 16.53 | 0.982 | 0.940 | True | 0 | 10.8 / 16473.5 / 8.5 / 1.2 |
| 56064 | 0 | 8 | v12f zero | 4 | 17.33 | 17.26 | 1.030 | 0.985 | True | 0 | 10.8 / 17342.9 / 8.3 / 1.2 |
| 56064 | 0 | 8 | v12f real | 2 | 12.90 | 12.87 | 0.766 | 0.733 | True | 0 | 10.8 / 12837.3 / 8.4 / 1.2 |
| 56064 | 0 | 8 | v12f real | 4 | 13.27 | 13.23 | 0.788 | 0.754 | True | 0 | 10.7 / 13193.2 / 8.6 / 1.1 |
| 56064 | 0 | 8 | v11 plain (again) | None | 17.56 | 17.42 | 1.043 | 0.998 | True | - | - |
| 56064 | 0 | 32 | flash_attn | - | 64.81 | 64.76 | 1 | 0.948 | - | - | - |
| 56064 | 0 | 32 | SDPA-flash | - | 69.15 | 69.11 | 1.067 | 1.012 | - | - | - |
| 56064 | 0 | 32 | v11 plain | None | 68.34 | 68.09 | 1.054 | 1.000 | - | - | - |
| 56064 | 0 | 32 | v11 zero | None | 86.35 | 86.29 | 1.332 | 1.264 | - | - | - |
| 56064 | 0 | 32 | v11 real | None | 68.89 | 68.88 | 1.063 | 1.008 | - | - | - |
| 56064 | 0 | 32 | v12f plain | None | 68.28 | 68.17 | 1.054 | 0.999 | True | - | - |
| 56064 | 0 | 32 | v12f zero | 0 | 65.88 | 65.87 | 1.017 | 0.964 | True | 0 | 11.1 / 65643.9 / 25.5 / 1.2 |
| 56064 | 0 | 32 | v12f real | 0 | 51.28 | 51.28 | 0.791 | 0.750 | True | 0 | 10.9 / 51203.2 / 24.9 / 1.2 |
| 56064 | 0 | 32 | v12f zero | 2 | 65.95 | 65.93 | 1.018 | 0.965 | True | 0 | 10.7 / 65860.2 / 25.5 / 1.2 |
| 56064 | 0 | 32 | v12f zero | 4 | 67.13 | 67.02 | 1.036 | 0.982 | True | 0 | 10.8 / 67040.4 / 25.4 / 1.2 |
| 56064 | 0 | 32 | v12f real | 2 | 51.33 | 51.32 | 0.792 | 0.751 | True | 0 | 10.8 / 51245.0 / 25.7 / 1.2 |
| 56064 | 0 | 32 | v12f real | 4 | 51.91 | 51.89 | 0.801 | 0.760 | True | 0 | 10.8 / 51840.9 / 26.0 / 1.2 |
| 56064 | 0 | 32 | v11 plain (again) | None | 68.28 | 68.19 | 1.053 | 0.999 | True | - | - |

## Forward, causal D=64, N=100003 c=7: even vs ragged (Is_even_MN=false) task

| L | L%128 | H | arm | flags | ms (median) | min | / flash_attn | / v11 plain (in-tree) | == v11 | route | device us (census / aligned / legacy / memset) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 42752 | 0 | 8 | flash_attn | - | 10.16 | 10.10 | 1 | 0.961 | - | - | - |
| 42752 | 0 | 8 | SDPA-flash | - | 10.75 | 10.68 | 1.058 | 1.017 | - | - | - |
| 42752 | 0 | 8 | v11 plain | None | 10.57 | 10.50 | 1.040 | 1.000 | - | - | - |
| 42752 | 0 | 8 | v11 zero | None | 13.56 | 13.46 | 1.335 | 1.284 | - | - | - |
| 42752 | 0 | 8 | v11 real | None | 10.69 | 10.66 | 1.053 | 1.012 | - | - | - |
| 42752 | 0 | 8 | v12f plain | None | 10.62 | 10.60 | 1.045 | 1.005 | True | - | - |
| 42752 | 0 | 8 | v12f zero | 0 | 9.87 | 9.85 | 0.972 | 0.934 | True | 0 | 9.1 / 9835.1 / 7.4 / 1.2 |
| 42752 | 0 | 8 | v12f real | 0 | 7.78 | 7.78 | 0.766 | 0.737 | True | 0 | 9.5 / 7750.0 / 7.2 / 1.2 |
| 42787 | 35 | 8 | flash_attn | - | 10.28 | 10.21 | 1 | 0.987 | - | - | - |
| 42787 | 35 | 8 | SDPA-flash | - | 11.01 | 10.84 | 1.071 | 1.057 | - | - | - |
| 42787 | 35 | 8 | v11 plain | None | 10.42 | 10.39 | 1.014 | 1.000 | - | - | - |
| 42787 | 35 | 8 | v11 zero | None | 14.14 | 14.11 | 1.376 | 1.357 | - | - | - |
| 42787 | 35 | 8 | v11 real | None | 14.36 | 14.20 | 1.398 | 1.379 | - | - | - |
| 42787 | 35 | 8 | v12f plain | None | 10.49 | 10.38 | 1.021 | 1.007 | True | - | - |
| 42787 | 35 | 8 | v12f zero | 0 | 10.25 | 10.21 | 0.997 | 0.984 | True | 0 | 10.0 / 10201.2 / 7.5 / 1.2 |
| 42787 | 35 | 8 | v12f real | 0 | 7.93 | 7.92 | 0.772 | 0.762 | True | 0 | 10.2 / 7865.0 / 7.5 / 1.2 |

- H=8: all-live tax, same (ascending) order both sides: v12f zero asc / v11 plain = 0.985; descending gain on its own: zero desc / zero asc = 0.954, real desc / real asc = 0.972; default (flags 0) real / v11 plain (in-tree) = 0.733, / flash_attn = 0.766; default zero / v11 plain = 0.937.
- H=32: all-live tax, same (ascending) order both sides: v12f zero asc / v11 plain = 0.982; descending gain on its own: zero desc / zero asc = 0.982, real desc / real asc = 0.989; default (flags 0) real / v11 plain (in-tree) = 0.750, / flash_attn = 0.791; default zero / v11 plain = 0.964.
- H=8: FA arms re-timed at the end of the group: flash_attn 16.84 -> 16.82 ms (min 16.69), SDPA-flash 17.84 -> 17.84 ms (min 17.81).
- H=32: FA arms re-timed at the end of the group: flash_attn 64.81 -> 65.21 ms (min 65.15), SDPA-flash 69.15 -> 69.24 ms (min 69.09).
- Ragged ALIGNED (Is_even_MN=false), L=42787 vs even L=42752: real / v11 plain 0.762 vs 0.737; zero / v11 plain 0.984 vs 0.934; real / flash_attn 0.772 vs 0.766.

## fwd_small_c73_itr2: 73 of 5329 tasks (L 1664-2176, sum L 145280), causal D=64 H=8, one timed region per arm

| arm | ms | min | / flash_attn loop | / in-tree plain loop | device us: census / aligned / legacy (= flash_fwd_kernel or the dq_dk_dv kernel, also the CQS-off forward) / memset / other (launches) |
|---|---|---|---|---|---|
| flash_attn_direct | 5.30 | 5.29 | 1.000 | 0.962 | - |
| v11_plain | 5.51 | 5.50 | 1.040 | 1.000 | - |
| v11_real | 9.55 | 9.09 | 1.804 | 1.735 | - (0) / - (0) / 8990 (73) / - (0) / - (0) |
| v12f_plain | 4.98 | 4.98 | 0.940 | 0.904 | - (0) / - (0) / 4906 (73) / - (0) / - (0) |
| v12f_zero | 6.00 | 6.00 | 1.133 | 1.090 | 236 (73) / 5293 (73) / 174 (73) / 99 (73) / - (0) |
| v12f_real | 5.28 | 5.27 | 0.996 | 0.958 | 255 (73) / 4549 (73) / 179 (73) / 98 (73) / - (0) |
| v12f_legacy_real | 9.16 | 9.16 | 1.729 | 1.663 | - |

Routes [0]; identity on every task: v12f real == v11 True.

## bwd_small_c13_itr3: 64 of 2197 tasks (L 3584-3968, sum L 244096), causal D=64 H=8, one timed region per arm

| arm | ms | min | / flash_attn loop | / in-tree plain loop | device us: census / aligned / legacy (= flash_fwd_kernel or the dq_dk_dv kernel, also the CQS-off forward) / memset / other (launches) |
|---|---|---|---|---|---|
| flash_attn_direct | 22.40 | 22.33 | 1.000 | 1.010 | - |
| v11x_nocqs | 22.18 | 22.17 | 0.990 | 1.000 | - |
| v12b_plain | 22.18 | 22.17 | 0.990 | 1.000 | - (0) / 20777 (64) / - (0) / - (0) / 1299 (128) |
| v11_real | 35.37 | 35.33 | 1.579 | 1.595 | - (0) / - (0) / 33911 (64) / - (0) / 1320 (128) |
| v12b_real | 13.35 | 13.34 | 0.596 | 0.602 | 231 (64) / 11318 (64) / 184 (64) / 97 (64) / 1293 (128) |
| v12b_zero | 22.64 | 22.63 | 1.011 | 1.021 | 223 (64) / 20665 (64) / 211 (64) / 96 (64) / 1283 (128) |
| v12b_legacy_real | 35.34 | 35.26 | 1.577 | 1.594 | - |

Routes [0]; identity on every task: v12b dk/dv == v11 True, dq max |d| 2.4e-04.

## bwd_small_c31_itr2: 64 of 961 tasks (L 4608-5120, sum L 313856), causal D=64 H=8, one timed region per arm

| arm | ms | min | / flash_attn loop | / in-tree plain loop | device us: census / aligned / legacy (= flash_fwd_kernel or the dq_dk_dv kernel, also the CQS-off forward) / memset / other (launches) |
|---|---|---|---|---|---|
| flash_attn_direct | 32.14 | 32.13 | 1.000 | 1.007 | - |
| v11x_nocqs | 31.90 | 31.89 | 0.993 | 1.000 | - |
| v12b_plain | 31.87 | 31.87 | 0.992 | 0.999 | - (0) / 30255 (64) / - (0) / - (0) / 1576 (128) |
| v11_real | 61.61 | 61.48 | 1.917 | 1.931 | - (0) / - (0) / 59865 (64) / - (0) / 1607 (128) |
| v12b_real | 25.71 | 25.70 | 0.800 | 0.806 | 258 (64) / 23417 (64) / 217 (64) / 91 (64) / 1558 (128) |
| v12b_zero | 32.23 | 32.23 | 1.003 | 1.010 | 255 (64) / 29975 (64) / 226 (64) / 93 (64) / 1531 (128) |
| v12b_legacy_real | 61.57 | 61.55 | 1.916 | 1.930 | - |

Routes [0]; identity on every task: v12b dk/dv == v11 True, dq max |d| 1.2e-04.

## Backward, causal D=64, c=7 aligned task L=56064 (headline re-time)

| L | L%128 | arm | ms (median) | min | / flash_attn (autograd) | / flash_attn direct | / v11 real (first) | dk/dv == v11 | dq max vs v11 | route | device us (census / aligned / legacy) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 56064 | 0 | flash_attn autograd / direct / SDPA | 43.80 / 43.69 / 47.48 | 43.79 / 43.68 / 47.47 | - | - | - | - | - | - | - |
| 56064 | 0 | v11_real | 85.26 | 85.23 | 1.947 | 1.952 | 1.000 | - | - | - | - |
| 56064 | 0 | v11x_nocqs | 43.65 | 43.64 | 0.997 | 0.999 | 0.512 | - | - | - | - |
| 56064 | 0 | v12b_plain | 43.30 | 43.29 | 0.989 | 0.991 | 0.508 | - | - | - | - |
| 56064 | 0 | v12b_real | 34.34 | 34.33 | 0.784 | 0.786 | 0.403 | True | 1.2e-04 | 0 | 11.4 / 34009.7 / 14.0 |
| 56064 | 0 | v12b_zero | 43.19 | 43.19 | 0.986 | 0.989 | 0.507 | - | - | 0 | 11.5 / 42862.5 / 14.0 |
| 56064 | 0 | v11_zero | 94.29 | 94.26 | 2.153 | 2.158 | 1.106 | - | - | - | - |
| 56064 | 0 | v11x_nocqs_again | 43.65 | 43.64 | 0.997 | 0.999 | 0.512 | - | - | - | - |

## Backward, causal D=64, N=100003 c=7: even vs ragged task

| L | L%128 | arm | ms (median) | min | / flash_attn (autograd) | / flash_attn direct | / v11 real (first) | dk/dv == v11 | dq max vs v11 | route | device us (census / aligned / legacy) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 42752 | 0 | flash_attn autograd / direct / SDPA | 25.69 / 25.60 / 27.87 | 25.68 / 25.59 / 27.87 | - | - | - | - | - | - | - |
| 42752 | 0 | v11_real | 49.66 | 49.59 | 1.933 | 1.940 | 1.000 | - | - | - | - |
| 42752 | 0 | v11x_nocqs | 25.56 | 25.55 | 0.995 | 0.998 | 0.515 | - | - | - | - |
| 42752 | 0 | v12b_plain | 25.36 | 25.36 | 0.987 | 0.991 | 0.511 | - | - | - | - |
| 42752 | 0 | v12b_real | 20.25 | 20.25 | 0.788 | 0.791 | 0.408 | True | 3.1e-05 | 0 | 9.6 / 19992.4 / 11.2 |
| 42752 | 0 | v12b_zero | 25.31 | 25.31 | 0.985 | 0.989 | 0.510 | - | - | 0 | 9.5 / 25061.9 / 11.3 |
| 42752 | 0 | v11_zero | 55.24 | 55.21 | 2.150 | 2.158 | 1.112 | - | - | - | - |
| 42787 | 35 | flash_attn autograd / direct / SDPA | 25.66 / 25.57 / 27.65 | 25.65 / 25.56 / 27.64 | - | - | - | - | - | - | - |
| 42787 | 35 | v11_real | 51.91 | 51.86 | 2.023 | 2.030 | 1.000 | - | - | - | - |
| 42787 | 35 | v11x_nocqs | 26.14 | 26.14 | 1.019 | 1.023 | 0.504 | - | - | - | - |
| 42787 | 35 | v12b_plain | 25.58 | 25.57 | 0.997 | 1.001 | 0.493 | - | - | - | - |
| 42787 | 35 | v12b_real | 20.29 | 20.28 | 0.791 | 0.793 | 0.391 | True | 6.1e-05 | 0 | 10.3 / 20031.8 / 10.9 |
| 42787 | 35 | v12b_zero | 25.37 | 25.36 | 0.989 | 0.992 | 0.489 | - | - | 0 | 10.4 / 25116.6 / 11.3 |
| 42787 | 35 | v11_zero | 56.36 | 56.31 | 2.196 | 2.204 | 1.086 | - | - | - | - |

## Backward, AUTOMATIC LEGACY route (unaligned seg_align=0 c=7 tasks)

| L | L%128 | arm | ms (median) | min | / flash_attn (autograd) | / flash_attn direct | / v11 real (first) | dk/dv == v11 | dq max vs v11 | route | device us (census / aligned / legacy) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 14043 | 91 | v11_real | 6.25 | 6.24 | - | - | 1.000 | - | - | - | - |
| 14043 | 91 | v12b_auto | 6.28 | 6.27 | - | - | 1.004 | True | 1.2e-04 | 1 | 5.9 / 5.4 / 6167.5 |
| 14043 | 91 | v12b_forced_legacy | 6.26 | 6.25 | - | - | 1.002 | True | 1.2e-04 | 1 | - / - / 6159.3 |
| 14043 | 91 | v11_real_again | 6.27 | 6.24 | - | - | 1.002 | True | 1.2e-04 | - | - |
| 56173 | 109 | v11_real | 88.79 | 88.74 | - | - | 1.000 | - | - | - | - |
| 56173 | 109 | v12b_auto | 88.85 | 88.69 | - | - | 1.001 | True | 1.2e-04 | 1 | 12.0 / 13.7 / 88446.5 |
| 56173 | 109 | v12b_forced_legacy | 88.85 | 88.74 | - | - | 1.001 | True | 1.2e-04 | 1 | - / - / 88501.5 |
| 56173 | 109 | v11_real_again | 88.82 | 88.71 | - | - | 1.000 | True | 6.1e-05 | - | - |

## Backward, forced LEGACY vs v11 on the aligned N=32768 task

| L | L%128 | arm | ms (median) | min | / flash_attn (autograd) | / flash_attn direct | / v11 real (first) | dk/dv == v11 | dq max vs v11 | route | device us (census / aligned / legacy) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 13952 | 0 | v11_real | 5.93 | 5.92 | - | - | 1.000 | - | - | - | - |
| 13952 | 0 | v12b_forced_legacy | 5.93 | 5.92 | - | - | 0.999 | True | 6.1e-05 | 0 | - / - / 5816.8 |
| 13952 | 0 | v11_real_again | 5.93 | 5.92 | - | - | 0.999 | True | 1.2e-04 | - | - |
| 13952 | 0 | v12b_forced_legacy_again | 5.93 | 5.92 | - | - | 0.999 | True | 1.2e-04 | 0 | - / - / 5824.2 |

## deterministic=True, L=56064 causal real D=64, H=8, b=1, 108 SMs

atomics 34.33 ms (peak extra 277 MiB), deterministic 65.36 ms (1.904x, peak extra 1701 MiB).


## Verdict

Written up in docs/ultimate/03_kernels_report.md sections 4.2, 4.3 and 5.2. In short: the review-fixed builds reproduce
the measured ones (their whole-module SASS is identical to v12f b2 / v12b b4); the ragged ALIGNED kernels, the
automatic LEGACY route and the small-L regimes are inside the targets; warmed up, flash_attn is ~4% faster than the
in-tree build of the same forward kernel, which re-bases the forward headline to 0.766x flash_attn.
