# Backward A/B (P2): v12b vs v11 / v11x / flash_attn / SDPA-flash

Job **14323074**, della-l09g6, **NVIDIA A100-SXM4-80GB** (`NVIDIA A100-SXM4-80GB, 210 MHz, 1410 MHz, 81920 MiB`), torch 2.10.0+cu130, flash_attn 2.8.3, fp16, H=8, 1 warm-up + 5 timed runs, CUDA events, median (min also stored). Harness `benchmarks/ultimate/bwd_ab.py` (`slurm/ultimate/bwd_ab.slurm`), raw rows `bwd_ab.json`. Every CQS arm runs its own dsoftmax_sum preprocess (as flash_attn does); v12b's census is inside its timed region. Modules: v11 = `/scratch/gpfs/AKEY/yb2807/Stream-CQSA-dev/next/kernel/stable/cqsa_cuda_next_v11.cpython-311-x86_64-linux-gnu.so`, v11x = `/scratch/gpfs/AKEY/yb2807/Stream-CQSA-dev/next/kernel/v11x/cqsa_cuda_next_v11x.cpython-311-x86_64-linux-gnu.so`, v12b = `/scratch/gpfs/AKEY/yb2807/Stream-CQSA-dev/next/kernel/v12b/frozen/b2/cqsa_cuda_next_v12b.cpython-311-x86_64-linux-gnu.so`.

## Per-config rows (ms, median; ratio vs flash_attn bwd / vs SDPA-flash bwd)

flash_attn = the package's backward through autograd (as the SDPA arm); flash_attn direct = its extension's `bwd` called like our arms (recorded in the json as `flash_attn_direct_bwd_ms`).

| config | L | D | causal | flash_attn | SDPA | v11x NO_CQS | v12b plain | v12b_s plain | v11 real | v12b real | v12b_s real | v12b LEGACY real | v11 zero | v12b zero | v12b_s zero | v12b route (real) | dk/dv == v11 (real, zero, s real, s zero) | dq max vs v11 (real) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| c7_N32768_causal_d64 | 13952 | 64 | 1 | 3.26 | 3.50 | 3.14 (0.964 / 0.898) | 2.95 (0.906 / 0.844) | - | 5.93 (1.821 / 1.697) | 2.63 (0.809 / 0.753) | - | 5.94 (1.823 / 1.698) | 6.27 (1.925 / 1.793) | 3.05 (0.937 / 0.873) | - | 0 | True, True, -, - | 1.2e-04 |
| c7_N65536_causal_d64 | 28032 | 64 | 1 | 11.19 | 12.15 | 11.09 (0.991 / 0.913) | 11.40 (1.018 / 0.938) | - | 21.78 (1.946 / 1.793) | 9.46 (0.845 / 0.778) | - | 21.80 (1.948 / 1.794) | 24.20 (2.162 / 1.991) | 11.76 (1.051 / 0.968) | - | 0 | True, True, -, - | 6.1e-05 |
| c7_N131072_causal_d64 | 56064 | 64 | 1 | 43.85 | 47.58 | 43.74 (0.997 / 0.919) | 45.08 (1.028 / 0.947) | - | 85.30 (1.945 / 1.793) | 36.78 (0.839 / 0.773) | - | 85.29 (1.945 / 1.792) | 94.30 (2.150 / 1.982) | 46.45 (1.059 / 0.976) | - | 0 | True, True, -, - | 1.2e-04 |
| c7_N131072_causal_d128 | 56064 | 128 | 1 | 80.08 | 83.61 | 79.88 (0.997 / 0.955) | 83.37 (1.041 / 0.997) | - | 124.52 (1.555 / 1.489) | 65.88 (0.823 / 0.788) | - | 124.54 (1.555 / 1.490) | 136.94 (1.710 / 1.638) | 82.95 (1.036 / 0.992) | - | 0 | True, True, -, - | 1.2e-04 |
| c7_N131072_noncausal_d64 | 56064 | 64 | 0 | 86.35 | 92.89 | 87.25 (1.010 / 0.939) | 91.70 (1.062 / 0.987) | - | 131.00 (1.517 / 1.410) | 70.04 (0.811 / 0.754) | - | 131.12 (1.518 / 1.412) | 140.29 (1.625 / 1.510) | 90.06 (1.043 / 0.970) | - | 0 | True, True, -, - | 3.1e-05 |

Drift check (the first arms re-timed after all others of the config): c7_N32768_causal_d64: v11x NO_CQS first 3.14 / last 2.85 ms, v12b plain first 2.95 / last 2.94 ms; c7_N65536_causal_d64: v11x NO_CQS first 11.09 / last 11.09 ms, v12b plain first 11.40 / last 11.39 ms; c7_N131072_causal_d64: v11x NO_CQS first 43.74 / last 43.72 ms, v12b plain first 45.08 / last 45.05 ms; c7_N131072_causal_d128: v11x NO_CQS first 79.88 / last 79.87 ms, v12b plain first 83.37 / last 83.39 ms; c7_N131072_noncausal_d64: v11x NO_CQS first 87.25 / last 87.25 ms, v12b plain first 91.70 / last 91.70 ms.

## Targets

| target | value | pass |
|---|---|---|
| c7_N32768_causal_d64: v12b zero / v11x NO_CQS <= 1.05 | 1.071 | False |
| c7_N32768_causal_d64: v12b_plain / v11x NO_CQS (recorded) | 1.035 | - |
| c7_N65536_causal_d64: v12b zero / v11x NO_CQS <= 1.05 | 1.060 | False |
| c7_N65536_causal_d64: v12b_plain / v11x NO_CQS (recorded) | 1.028 | - |
| c7_N131072_causal_d64: v12b zero / v11x NO_CQS <= 1.05 | 1.062 | False |
| c7_N131072_causal_d64: v12b_plain / v11x NO_CQS (recorded) | 1.031 | - |
| c7_N131072_causal_d128: v12b zero / v11x NO_CQS <= 1.05 | 1.039 | True |
| c7_N131072_causal_d128: v12b_plain / v11x NO_CQS (recorded) | 1.044 | - |
| c7_N131072_noncausal_d64: v12b zero / v11x NO_CQS <= 1.05 | 1.032 | True |
| c7_N131072_noncausal_d64: v12b_plain / v11x NO_CQS (recorded) | 1.051 | - |
| c7_N131072_causal_d64 [v12b]: v12b real / flash_attn bwd <= 0.85 | 0.839 | True |
| c7_N131072_causal_d64 [v12b]: v12b real / SDPA bwd <= 1.30 (design P2.1 gate) | 0.773 | True |
| c7_N131072_causal_d128 [v12b]: v12b real / flash_attn bwd (recorded) | 0.823 | - |
| c7_N131072_noncausal_d64 [v12b]: non-causal v12b real / SDPA bwd <= 1.10 (design P2.1 gate) | 0.754 | True |
| c7_N131072_noncausal_d64 [v12b]: non-causal v12b real / flash_attn bwd (recorded) | 0.811 | - |
| c=73 N=131072 (L 16128-16384): v12b sum / SDPA sum <= 1.45 (design: L=16159 causal) | 0.846 | True |
| c=73: v12b loop / flash_attn loop (recorded) | 0.942 | - |

## c=73, N=131072, causal D=64, real: 73 tasks (L 16128-16384) in one timed region per arm

| arm | ms | / flash_attn loop | / flash_attn per-task sum | / SDPA per-task sum |
|---|---|---|---|---|
| v11 | 593.3 | 2.082 | 2.008 | 1.869 |
| v12b | 268.4 | 0.942 | 0.909 | 0.846 |
| v12b_legacy | 593.3 | 2.082 | 2.008 | 1.869 |
| v11x_nocqs | 284.1 | 0.997 | 0.962 | 0.895 |
| flash_attn_loop | 284.9 | 1.000 | 0.964 | 0.898 |
| flash_attn per task (autograd), summed | 295.4 | - | 1 | - |
| SDPA-flash per task (autograd), summed | 317.4 | - | - | 1 |
| monolithic flash_attn at N=131072 | 237.0 | - | - |

Routes [0]; v12b dk/dv torch.equal to v11 on every task: True; dq max |d| vs v11 1.2e-04.

## deterministic=True (item 3), L=56064 causal real D=64, H=8, b=1, 108 SMs

atomics 36.74 ms (3 reruns torch.equal: False, dq drift 1.2e-04); deterministic 70.52 ms (1.919x; 3 reruns torch.equal: True); deterministic vs atomics max |d| dq/dk/dv 6.1e-05, 0.0e+00, 0.0e+00. Grid x = ceil(108 / (b*H)) = 14 CTAs per head, each walking its column blocks serially: free only when W*H is close to a multiple of the SM count (design 4.7 item 5).

## Kernels recorded by the profiler

- flash_attn_bwd: `void flash::flash_bwd_convert_dq_kernel<Flash_bwd_kernel_traits<64, 128, 128, 8, 4, 4, 4, false, false, cutlass::half_t, Flash_kernel_traits`; `void flash::flash_bwd_dot_do_o_kernel<true, Flash_bwd_kernel_traits<64, 128, 128, 8, 4, 4, 4, false, false, cutlass::half_t, Flash_kernel_tr`; `void flash::flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<Flash_bwd_kernel_traits<64, 128, 128, 8, 4, 4, 4, false, false, cutlass::half_t, Fl`
- sdpa_flash_bwd: `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&):`; `void pytorch_flash::flash_bwd_convert_dq_kernel<Flash_bwd_kernel_traits<64, 128, 128, 8, 4, 4, 4, false, false, cutlass::half_t, Flash_kerne`; `void pytorch_flash::flash_bwd_dot_do_o_kernel<true, Flash_bwd_kernel_traits<64, 128, 128, 8, 4, 4, 4, false, false, cutlass::half_t, Flash_k`; `void pytorch_flash::flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<Flash_bwd_kernel_traits<64, 128, 128, 8, 4, 4, 4, false, false, cutlass::ha`
- v12b_real_D64_causal1: `Memset (Device)`; `flash::(anonymous namespace)::cqs_bwd_census_kernel(long const*, long const*, long const*, int, int, int, bool, int2*, int*)`; `void flash::flash_bwd_convert_dq_kernel<Flash_bwd_kernel_traits<64, 128, 128, 8, 4, 4, 4, false, false, cutlass::half_t, Flash_kernel_traits`; `void flash::flash_bwd_dot_do_o_kernel<true, Flash_bwd_kernel_traits<64, 128, 128, 8, 4, 4, 4, false, false, cutlass::half_t, Flash_kernel_tr`; `void flash::flash_bwd_dq_dk_dv_loop_seqk_parallel_aligned_kernel<Flash_bwd_kernel_traits<64, 128, 128, 8, 4, 4, 4, false, false, cutlass::ha`; `void flash::flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<Flash_bwd_kernel_traits<64, 128, 128, 8, 4, 4, 4, false, false, cutlass::half_t, Fl`
- v11_real_D64_causal1: `void flash::flash_bwd_convert_dq_kernel<Flash_bwd_kernel_traits<64, 128, 128, 8, 4, 4, 4, false, false, cutlass::half_t, Flash_kernel_traits`; `void flash::flash_bwd_dot_do_o_kernel<true, Flash_bwd_kernel_traits<64, 128, 128, 8, 4, 4, 4, false, false, cutlass::half_t, Flash_kernel_tr`; `void flash::flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<Flash_bwd_kernel_traits<64, 128, 128, 8, 4, 4, 4, false, false, cutlass::half_t, Fl`
- v12b_real_D128_causal1: `Memset (Device)`; `flash::(anonymous namespace)::cqs_bwd_census_kernel(long const*, long const*, long const*, int, int, int, bool, int2*, int*)`; `void flash::flash_bwd_convert_dq_kernel<Flash_bwd_kernel_traits<128, 64, 128, 8, 2, 4, 2, false, false, cutlass::half_t, Flash_kernel_traits`; `void flash::flash_bwd_dot_do_o_kernel<true, Flash_bwd_kernel_traits<128, 64, 128, 8, 2, 4, 2, false, false, cutlass::half_t, Flash_kernel_tr`; `void flash::flash_bwd_dq_dk_dv_loop_seqk_parallel_aligned_kernel<Flash_bwd_kernel_traits<128, 64, 128, 8, 2, 4, 2, false, false, cutlass::ha`; `void flash::flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<Flash_bwd_kernel_traits<128, 64, 128, 8, 2, 4, 2, false, false, cutlass::half_t, Fl`
- v11_real_D128_causal1: `void flash::flash_bwd_convert_dq_kernel<Flash_bwd_kernel_traits<128, 64, 128, 8, 2, 4, 2, false, false, cutlass::half_t, Flash_kernel_traits`; `void flash::flash_bwd_dot_do_o_kernel<true, Flash_bwd_kernel_traits<128, 64, 128, 8, 2, 4, 2, false, false, cutlass::half_t, Flash_kernel_tr`; `void flash::flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<Flash_bwd_kernel_traits<128, 64, 128, 8, 2, 4, 2, false, false, cutlass::half_t, Fl`
- v12b_real_D64_causal0: `Memset (Device)`; `flash::(anonymous namespace)::cqs_bwd_census_kernel(long const*, long const*, long const*, int, int, int, bool, int2*, int*)`; `void flash::flash_bwd_convert_dq_kernel<Flash_bwd_kernel_traits<64, 128, 128, 8, 4, 4, 4, false, false, cutlass::half_t, Flash_kernel_traits`; `void flash::flash_bwd_dot_do_o_kernel<true, Flash_bwd_kernel_traits<64, 128, 128, 8, 4, 4, 4, false, false, cutlass::half_t, Flash_kernel_tr`; `void flash::flash_bwd_dq_dk_dv_loop_seqk_parallel_aligned_kernel<Flash_bwd_kernel_traits<64, 128, 128, 8, 4, 4, 4, false, false, cutlass::ha`; `void flash::flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<Flash_bwd_kernel_traits<64, 128, 128, 8, 4, 4, 4, false, false, cutlass::half_t, Fl`
- v11_real_D64_causal0: `void flash::flash_bwd_convert_dq_kernel<Flash_bwd_kernel_traits<64, 128, 128, 8, 4, 4, 4, false, false, cutlass::half_t, Flash_kernel_traits`; `void flash::flash_bwd_dot_do_o_kernel<true, Flash_bwd_kernel_traits<64, 128, 128, 8, 4, 4, 4, false, false, cutlass::half_t, Flash_kernel_tr`; `void flash::flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<Flash_bwd_kernel_traits<64, 128, 128, 8, 4, 4, 4, false, false, cutlass::half_t, Fl`

