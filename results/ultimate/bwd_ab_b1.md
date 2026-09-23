# Backward A/B (P2): v12b vs v11 / v11x / flash_attn / SDPA-flash

Job **14322302**, della-l07g4, **NVIDIA A100-SXM4-80GB** (`NVIDIA A100-SXM4-80GB, 705 MHz, 1410 MHz, 81920 MiB`), torch 2.10.0+cu130, flash_attn 2.8.3, fp16, H=8, 1 warm-up + 5 timed runs, CUDA events, median (min also stored). Harness `benchmarks/ultimate/bwd_ab.py` (`slurm/ultimate/bwd_ab.slurm`), raw rows `bwd_ab.json`. Every CQS arm runs its own dsoftmax_sum preprocess (as flash_attn does); v12b's census is inside its timed region. Modules: v11 = `/scratch/gpfs/AKEY/yb2807/Stream-CQSA-dev/next/kernel/stable/cqsa_cuda_next_v11.cpython-311-x86_64-linux-gnu.so`, v11x = `/scratch/gpfs/AKEY/yb2807/Stream-CQSA-dev/next/kernel/v11x/cqsa_cuda_next_v11x.cpython-311-x86_64-linux-gnu.so`, v12b = `/scratch/gpfs/AKEY/yb2807/Stream-CQSA-dev/next/kernel/v12b/frozen/b1/cqsa_cuda_next_v12b.cpython-311-x86_64-linux-gnu.so`.

## Per-config rows (ms, median; ratio vs flash_attn bwd / vs SDPA-flash bwd)

flash_attn = the package's backward through autograd (as the SDPA arm); flash_attn direct = its extension's `bwd` called like our arms (recorded in the json as `flash_attn_direct_bwd_ms`).

| config | L | D | causal | flash_attn | SDPA | v11x NO_CQS | v12b plain | v12b_s plain | v11 real | v12b real | v12b_s real | v12b LEGACY real | v11 zero | v12b zero | v12b_s zero | v12b route (real) | dk/dv == v11 (real, zero, s real, s zero) | dq max vs v11 (real) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| c7_N32768_causal_d64 | 13952 | 64 | 1 | 3.27 | 3.50 | 3.14 (0.959 / 0.898) | 3.01 (0.918 / 0.859) | - | 5.94 (1.815 / 1.700) | 2.65 (0.811 / 0.759) | - | 5.94 (1.814 / 1.699) | 6.29 (1.920 / 1.798) | 3.11 (0.949 / 0.888) | - | 0 | True, True, -, - | 1.2e-04 |
| c7_N65536_causal_d64 | 28032 | 64 | 1 | 11.20 | 12.16 | 11.11 (0.992 / 0.913) | 11.60 (1.036 / 0.954) | - | 21.83 (1.948 / 1.795) | 9.61 (0.858 / 0.790) | - | 21.84 (1.949 / 1.795) | 24.26 (2.165 / 1.995) | 11.95 (1.067 / 0.982) | - | 0 | True, True, -, - | 6.1e-05 |
| c7_N131072_causal_d64 | 56064 | 64 | 1 | 43.90 | 47.65 | 43.82 (0.998 / 0.920) | 45.78 (1.043 / 0.961) | - | 85.43 (1.946 / 1.793) | 37.31 (0.850 / 0.783) | - | 85.43 (1.946 / 1.793) | 94.51 (2.153 / 1.984) | 47.10 (1.073 / 0.989) | - | 0 | True, True, -, - | 6.1e-05 |
| c7_N131072_causal_d128 | 56064 | 128 | 1 | 80.28 | 83.82 | 79.74 (0.993 / 0.951) | 82.83 (1.032 / 0.988) | - | 124.79 (1.554 / 1.489) | 65.26 (0.813 / 0.779) | - | 124.77 (1.554 / 1.489) | 137.31 (1.710 / 1.638) | 82.03 (1.022 / 0.979) | - | 0 | True, True, -, - | 1.2e-04 |
| c7_N131072_noncausal_d64 | 56064 | 64 | 0 | 86.37 | 92.97 | 87.16 (1.009 / 0.937) | 89.72 (1.039 / 0.965) | - | 131.23 (1.519 / 1.411) | 72.69 (0.842 / 0.782) | - | 131.14 (1.518 / 1.411) | 140.50 (1.627 / 1.511) | 93.56 (1.083 / 1.006) | - | 0 | True, True, -, - | 3.1e-05 |

## Targets

| target | value | pass |
|---|---|---|
| c7_N32768_causal_d64: v12b zero / v11x NO_CQS <= 1.05 | 0.989 | True |
| c7_N32768_causal_d64: v12b_plain / v11x NO_CQS (recorded) | 0.957 | - |
| c7_N65536_causal_d64: v12b zero / v11x NO_CQS <= 1.05 | 1.076 | False |
| c7_N65536_causal_d64: v12b_plain / v11x NO_CQS (recorded) | 1.044 | - |
| c7_N131072_causal_d64: v12b zero / v11x NO_CQS <= 1.05 | 1.075 | False |
| c7_N131072_causal_d64: v12b_plain / v11x NO_CQS (recorded) | 1.045 | - |
| c7_N131072_causal_d128: v12b zero / v11x NO_CQS <= 1.05 | 1.029 | True |
| c7_N131072_causal_d128: v12b_plain / v11x NO_CQS (recorded) | 1.039 | - |
| c7_N131072_noncausal_d64: v12b zero / v11x NO_CQS <= 1.05 | 1.073 | False |
| c7_N131072_noncausal_d64: v12b_plain / v11x NO_CQS (recorded) | 1.029 | - |
| c7_N131072_causal_d64 [v12b]: v12b real / flash_attn bwd <= 0.85 | 0.850 | True |
| c7_N131072_causal_d64 [v12b]: v12b real / SDPA bwd <= 1.30 (design P2.1 gate) | 0.783 | True |
| c7_N131072_causal_d128 [v12b]: v12b real / flash_attn bwd (recorded) | 0.813 | - |
| c7_N131072_noncausal_d64 [v12b]: non-causal v12b real / SDPA bwd <= 1.10 (design P2.1 gate) | 0.782 | True |
| c7_N131072_noncausal_d64 [v12b]: non-causal v12b real / flash_attn bwd (recorded) | 0.842 | - |
| c=73 N=131072 (L 16128-16384): v12b sum / SDPA sum <= 1.45 (design: L=16159 causal) | 0.859 | True |
| c=73: v12b loop / flash_attn loop (recorded) | 0.956 | - |

## c=73, N=131072, causal D=64, real: 73 tasks (L 16128-16384) in one timed region per arm

| arm | ms | / flash_attn loop | / flash_attn per-task sum | / SDPA per-task sum |
|---|---|---|---|---|
| v11 | 594.0 | 2.082 | 2.010 | 1.870 |
| v12b | 272.9 | 0.956 | 0.924 | 0.859 |
| v12b_legacy | 594.2 | 2.082 | 2.011 | 1.871 |
| v11x_nocqs | 284.4 | 0.997 | 0.963 | 0.896 |
| flash_attn_loop | 285.4 | 1.000 | 0.966 | 0.898 |
| flash_attn per task (autograd), summed | 295.5 | - | 1 | - |
| SDPA-flash per task (autograd), summed | 317.6 | - | - | 1 |
| monolithic flash_attn at N=131072 | 237.3 | - | - |

Routes [0]; v12b dk/dv torch.equal to v11 on every task: True; dq max |d| vs v11 1.2e-04.

## deterministic=True (item 3), L=56064 causal real D=64, H=8, b=1, 108 SMs

atomics 37.28 ms (3 reruns torch.equal: False, dq drift 1.2e-04); deterministic 72.39 ms (1.942x; 3 reruns torch.equal: True); deterministic vs atomics max |d| dq/dk/dv 6.1e-05, 0.0e+00, 0.0e+00. Grid x = ceil(108 / (b*H)) = 14 CTAs per head, each walking its column blocks serially: free only when W*H is close to a multiple of the SM count (design 4.7 item 5).

## Kernels recorded by the profiler

- flash_attn_bwd: `void flash::flash_bwd_convert_dq_kernel<Flash_bwd_kernel_traits<64, 128, 128, 8, 4, 4, 4, false, false, cutlass::half_t, Flash_kernel_traits`; `void flash::flash_bwd_dot_do_o_kernel<true, Flash_bwd_kernel_traits<64, 128, 128, 8, 4, 4, 4, false, false, cutlass::half_t, Flash_kernel_tr`; `void flash::flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<Flash_bwd_kernel_traits<64, 128, 128, 8, 4, 4, 4, false, false, cutlass::half_t, Fl`
- sdpa_flash_bwd: `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&):`; `void pytorch_flash::flash_bwd_convert_dq_kernel<Flash_bwd_kernel_traits<64, 128, 128, 8, 4, 4, 4, false, false, cutlass::half_t, Flash_kerne`; `void pytorch_flash::flash_bwd_dot_do_o_kernel<true, Flash_bwd_kernel_traits<64, 128, 128, 8, 4, 4, 4, false, false, cutlass::half_t, Flash_k`; `void pytorch_flash::flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<Flash_bwd_kernel_traits<64, 128, 128, 8, 4, 4, 4, false, false, cutlass::ha`
- v12b_real_D64_causal1: `Memset (Device)`; `flash::(anonymous namespace)::cqs_bwd_census_kernel(long const*, long const*, long const*, int, int, int, bool, int2*, int*)`; `void flash::flash_bwd_convert_dq_kernel<Flash_bwd_kernel_traits<64, 128, 128, 8, 4, 4, 4, false, false, cutlass::half_t, Flash_kernel_traits`; `void flash::flash_bwd_dot_do_o_kernel<true, Flash_bwd_kernel_traits<64, 128, 128, 8, 4, 4, 4, false, false, cutlass::half_t, Flash_kernel_tr`; `void flash::flash_bwd_dq_dk_dv_loop_seqk_parallel_aligned_kernel<Flash_bwd_kernel_traits<64, 128, 128, 8, 4, 4, 4, false, false, cutlass::ha`; `void flash::flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<Flash_bwd_kernel_traits<64, 128, 128, 8, 4, 4, 4, false, false, cutlass::half_t, Fl`
- v11_real_D64_causal1: `void flash::flash_bwd_convert_dq_kernel<Flash_bwd_kernel_traits<64, 128, 128, 8, 4, 4, 4, false, false, cutlass::half_t, Flash_kernel_traits`; `void flash::flash_bwd_dot_do_o_kernel<true, Flash_bwd_kernel_traits<64, 128, 128, 8, 4, 4, 4, false, false, cutlass::half_t, Flash_kernel_tr`; `void flash::flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<Flash_bwd_kernel_traits<64, 128, 128, 8, 4, 4, 4, false, false, cutlass::half_t, Fl`
- v12b_real_D128_causal1: `Memset (Device)`; `flash::(anonymous namespace)::cqs_bwd_census_kernel(long const*, long const*, long const*, int, int, int, bool, int2*, int*)`; `void flash::flash_bwd_convert_dq_kernel<Flash_bwd_kernel_traits<128, 64, 128, 8, 2, 4, 2, false, false, cutlass::half_t, Flash_kernel_traits`; `void flash::flash_bwd_dot_do_o_kernel<true, Flash_bwd_kernel_traits<128, 64, 128, 8, 2, 4, 2, false, false, cutlass::half_t, Flash_kernel_tr`; `void flash::flash_bwd_dq_dk_dv_loop_seqk_parallel_aligned_kernel<Flash_bwd_kernel_traits<128, 64, 128, 8, 2, 4, 2, false, false, cutlass::ha`; `void flash::flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<Flash_bwd_kernel_traits<128, 64, 128, 8, 2, 4, 2, false, false, cutlass::half_t, Fl`
- v11_real_D128_causal1: `void flash::flash_bwd_convert_dq_kernel<Flash_bwd_kernel_traits<128, 64, 128, 8, 2, 4, 2, false, false, cutlass::half_t, Flash_kernel_traits`; `void flash::flash_bwd_dot_do_o_kernel<true, Flash_bwd_kernel_traits<128, 64, 128, 8, 2, 4, 2, false, false, cutlass::half_t, Flash_kernel_tr`; `void flash::flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<Flash_bwd_kernel_traits<128, 64, 128, 8, 2, 4, 2, false, false, cutlass::half_t, Fl`
- v12b_real_D64_causal0: `Memset (Device)`; `flash::(anonymous namespace)::cqs_bwd_census_kernel(long const*, long const*, long const*, int, int, int, bool, int2*, int*)`; `void flash::flash_bwd_convert_dq_kernel<Flash_bwd_kernel_traits<64, 128, 128, 8, 4, 4, 4, false, false, cutlass::half_t, Flash_kernel_traits`; `void flash::flash_bwd_dot_do_o_kernel<true, Flash_bwd_kernel_traits<64, 128, 128, 8, 4, 4, 4, false, false, cutlass::half_t, Flash_kernel_tr`; `void flash::flash_bwd_dq_dk_dv_loop_seqk_parallel_aligned_kernel<Flash_bwd_kernel_traits<64, 128, 128, 8, 4, 4, 4, false, false, cutlass::ha`; `void flash::flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<Flash_bwd_kernel_traits<64, 128, 128, 8, 4, 4, 4, false, false, cutlass::half_t, Fl`
- v11_real_D64_causal0: `void flash::flash_bwd_convert_dq_kernel<Flash_bwd_kernel_traits<64, 128, 128, 8, 4, 4, 4, false, false, cutlass::half_t, Flash_kernel_traits`; `void flash::flash_bwd_dot_do_o_kernel<true, Flash_bwd_kernel_traits<64, 128, 128, 8, 4, 4, 4, false, false, cutlass::half_t, Flash_kernel_tr`; `void flash::flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<Flash_bwd_kernel_traits<64, 128, 128, 8, 4, 4, 4, false, false, cutlass::half_t, Fl`

