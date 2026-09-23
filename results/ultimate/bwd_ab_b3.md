# Backward A/B (P2): v12b vs v11 / v11x / flash_attn / SDPA-flash

Job **14323878**, della-l07g5, **NVIDIA A100-SXM4-80GB** (`NVIDIA A100-SXM4-80GB, 825 MHz, 1410 MHz, 81920 MiB`), torch 2.10.0+cu130, flash_attn 2.8.3, fp16, H=8, 1 warm-up + 5 timed runs, CUDA events, median (min also stored). Harness `benchmarks/ultimate/bwd_ab.py` (`slurm/ultimate/bwd_ab.slurm`), raw rows `bwd_ab.json`. Every CQS arm runs its own dsoftmax_sum preprocess (as flash_attn does); v12b's census is inside its timed region. Modules: v11 = `/scratch/gpfs/AKEY/yb2807/Stream-CQSA-dev/next/kernel/stable/cqsa_cuda_next_v11.cpython-311-x86_64-linux-gnu.so`, v11x = `/scratch/gpfs/AKEY/yb2807/Stream-CQSA-dev/next/kernel/v11x/cqsa_cuda_next_v11x.cpython-311-x86_64-linux-gnu.so`, v12b = `/scratch/gpfs/AKEY/yb2807/Stream-CQSA-dev/next/kernel/v12b/frozen/b3/cqsa_cuda_next_v12b.cpython-311-x86_64-linux-gnu.so`.

## Per-config rows (ms, median; ratio vs flash_attn bwd / vs SDPA-flash bwd)

flash_attn = the package's backward through autograd (as the SDPA arm); flash_attn direct = its extension's `bwd` called like our arms (recorded in the json as `flash_attn_direct_bwd_ms`).

| config | L | D | causal | flash_attn | SDPA | v11x NO_CQS | v12b plain | v12b_s plain | v11 real | v12b real | v12b_s real | v12b LEGACY real | v11 zero | v12b zero | v12b_s zero | v12b route (real) | dk/dv == v11 (real, zero, s real, s zero) | dq max vs v11 (real) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| c7_N32768_causal_d64 | 13952 | 64 | 1 | 3.27 | 3.51 | 2.87 (0.876 / 0.817) | 2.85 (0.871 / 0.812) | 2.84 (0.870 / 0.811) | 5.90 (1.803 / 1.681) | 2.49 (0.762 / 0.710) | 2.48 (0.759 / 0.707) | 5.96 (1.821 / 1.698) | 6.28 (1.921 / 1.791) | 2.88 (0.880 / 0.820) | 2.85 (0.872 / 0.813) | 0 | True, True, True, True | 1.2e-04 |
| c7_N65536_causal_d64 | 28032 | 64 | 1 | 11.24 | 12.20 | 11.14 (0.992 / 0.914) | 11.04 (0.983 / 0.906) | 11.04 (0.983 / 0.905) | 21.86 (1.945 / 1.792) | 8.88 (0.790 / 0.728) | 8.89 (0.791 / 0.729) | 21.85 (1.944 / 1.791) | 24.27 (2.159 / 1.990) | 11.04 (0.982 / 0.905) | 11.05 (0.983 / 0.906) | 0 | True, True, True, True | 6.1e-05 |
| c7_N131072_causal_d64 | 56064 | 64 | 1 | 43.95 | 47.68 | 43.87 (0.998 / 0.920) | 43.47 (0.989 / 0.912) | 43.46 (0.989 / 0.911) | 85.43 (1.944 / 1.792) | 34.40 (0.783 / 0.721) | 34.41 (0.783 / 0.722) | 85.42 (1.943 / 1.791) | 94.48 (2.150 / 1.981) | 43.42 (0.988 / 0.911) | 43.43 (0.988 / 0.911) | 0 | True, True, True, True | 6.1e-05 |
| c7_N131072_causal_d128 | 56064 | 128 | 1 | 80.37 | 83.95 | 80.17 (0.998 / 0.955) | 80.17 (0.998 / 0.955) | 82.43 (1.026 / 0.982) | 124.86 (1.553 / 1.487) | 63.08 (0.785 / 0.751) | 61.82 (0.769 / 0.736) | 124.78 (1.553 / 1.486) | 137.32 (1.708 / 1.636) | 79.69 (0.991 / 0.949) | 78.02 (0.971 / 0.929) | 0 | True, True, True, True | 1.2e-04 |
| c7_N131072_noncausal_d64 | 56064 | 64 | 0 | 86.42 | 92.98 | 87.36 (1.011 / 0.940) | 86.32 (0.999 / 0.928) | 86.42 (1.000 / 0.929) | 131.11 (1.517 / 1.410) | 68.60 (0.794 / 0.738) | 67.47 (0.781 / 0.726) | 131.16 (1.518 / 1.411) | 140.49 (1.626 / 1.511) | 87.96 (1.018 / 0.946) | 86.57 (1.002 / 0.931) | 0 | True, True, True, True | 3.1e-05 |

Drift check (the first arms re-timed after all others of the config): c7_N32768_causal_d64: v11x NO_CQS first 2.87 / last 2.86 ms, v12b plain first 2.85 / last 2.86 ms; c7_N65536_causal_d64: v11x NO_CQS first 11.14 / last 11.14 ms, v12b plain first 11.04 / last 11.04 ms; c7_N131072_causal_d64: v11x NO_CQS first 43.87 / last 43.87 ms, v12b plain first 43.47 / last 43.48 ms; c7_N131072_causal_d128: v11x NO_CQS first 80.17 / last 80.08 ms, v12b plain first 80.17 / last 80.12 ms; c7_N131072_noncausal_d64: v11x NO_CQS first 87.36 / last 87.21 ms, v12b plain first 86.32 / last 86.31 ms.

## Targets

| target | value | pass |
|---|---|---|
| c7_N32768_causal_d64: v12b zero / v11x NO_CQS <= 1.05 | 1.005 | True |
| c7_N32768_causal_d64: v12b_s zero / v11x NO_CQS <= 1.05 | 0.996 | True |
| c7_N32768_causal_d64: v12b_plain / v11x NO_CQS (recorded) | 0.994 | - |
| c7_N32768_causal_d64: v12b_plain_s / v11x NO_CQS (recorded) | 0.993 | - |
| c7_N65536_causal_d64: v12b zero / v11x NO_CQS <= 1.05 | 0.991 | True |
| c7_N65536_causal_d64: v12b_s zero / v11x NO_CQS <= 1.05 | 0.992 | True |
| c7_N65536_causal_d64: v12b_plain / v11x NO_CQS (recorded) | 0.991 | - |
| c7_N65536_causal_d64: v12b_plain_s / v11x NO_CQS (recorded) | 0.991 | - |
| c7_N131072_causal_d64: v12b zero / v11x NO_CQS <= 1.05 | 0.990 | True |
| c7_N131072_causal_d64: v12b_s zero / v11x NO_CQS <= 1.05 | 0.990 | True |
| c7_N131072_causal_d64: v12b_plain / v11x NO_CQS (recorded) | 0.991 | - |
| c7_N131072_causal_d64: v12b_plain_s / v11x NO_CQS (recorded) | 0.991 | - |
| c7_N131072_causal_d128: v12b zero / v11x NO_CQS <= 1.05 | 0.995 | True |
| c7_N131072_causal_d128: v12b_s zero / v11x NO_CQS <= 1.05 | 0.974 | True |
| c7_N131072_causal_d128: v12b_plain / v11x NO_CQS (recorded) | 1.001 | - |
| c7_N131072_causal_d128: v12b_plain_s / v11x NO_CQS (recorded) | 1.029 | - |
| c7_N131072_noncausal_d64: v12b zero / v11x NO_CQS <= 1.05 | 1.009 | True |
| c7_N131072_noncausal_d64: v12b_s zero / v11x NO_CQS <= 1.05 | 0.993 | True |
| c7_N131072_noncausal_d64: v12b_plain / v11x NO_CQS (recorded) | 0.990 | - |
| c7_N131072_noncausal_d64: v12b_plain_s / v11x NO_CQS (recorded) | 0.991 | - |
| c7_N131072_causal_d64 [v12b]: v12b real / flash_attn bwd <= 0.85 | 0.783 | True |
| c7_N131072_causal_d64 [v12b]: v12b real / SDPA bwd <= 1.30 (design P2.1 gate) | 0.721 | True |
| c7_N131072_causal_d64 [v12b_s]: v12b real / flash_attn bwd <= 0.85 | 0.783 | True |
| c7_N131072_causal_d64 [v12b_s]: v12b real / SDPA bwd <= 1.30 (design P2.1 gate) | 0.722 | True |
| c7_N131072_causal_d128 [v12b]: v12b real / flash_attn bwd (recorded) | 0.785 | - |
| c7_N131072_causal_d128 [v12b_s]: v12b real / flash_attn bwd (recorded) | 0.769 | - |
| c7_N131072_noncausal_d64 [v12b]: non-causal v12b real / SDPA bwd <= 1.10 (design P2.1 gate) | 0.738 | True |
| c7_N131072_noncausal_d64 [v12b]: non-causal v12b real / flash_attn bwd (recorded) | 0.794 | - |
| c7_N131072_noncausal_d64 [v12b_s]: non-causal v12b real / SDPA bwd <= 1.10 (design P2.1 gate) | 0.726 | True |
| c7_N131072_noncausal_d64 [v12b_s]: non-causal v12b real / flash_attn bwd (recorded) | 0.781 | - |
| c=73 N=131072 (L 16128-16384): v12b sum / SDPA sum <= 1.45 (design: L=16159 causal) | 0.795 | True |
| c=73: v12b loop / flash_attn loop (recorded) | 0.885 | - |

## c=73, N=131072, causal D=64, real: 73 tasks (L 16128-16384) in one timed region per arm

| arm | ms | / flash_attn loop | / flash_attn per-task sum | / SDPA per-task sum |
|---|---|---|---|---|
| v11 | 594.3 | 2.082 | 2.011 | 1.870 |
| v12b | 252.7 | 0.885 | 0.855 | 0.795 |
| v12b_s | 252.5 | 0.885 | 0.855 | 0.795 |
| v12b_legacy | 594.3 | 2.082 | 2.011 | 1.870 |
| v11x_nocqs | 284.6 | 0.997 | 0.963 | 0.895 |
| flash_attn_loop | 285.4 | 1.000 | 0.966 | 0.898 |
| flash_attn per task (autograd), summed | 295.5 | - | 1 | - |
| SDPA-flash per task (autograd), summed | 317.8 | - | - | 1 |
| monolithic flash_attn at N=131072 | 237.4 | - | - |

Routes [0]; v12b dk/dv torch.equal to v11 on every task: True; dq max |d| vs v11 1.2e-04.

## deterministic=True (item 3), L=56064 causal real D=64, H=8, b=1, 108 SMs

atomics 34.40 ms (3 reruns torch.equal: False, dq drift 6.1e-05); deterministic 66.60 ms (1.936x; 3 reruns torch.equal: True); deterministic vs atomics max |d| dq/dk/dv 6.1e-05, 0.0e+00, 0.0e+00. Grid x = ceil(108 / (b*H)) = 14 CTAs per head, each walking its column blocks serially: free only when W*H is close to a multiple of the SM count (design 4.7 item 5).

## Kernels recorded by the profiler

- flash_attn_bwd: `void flash::flash_bwd_convert_dq_kernel<Flash_bwd_kernel_traits<64, 128, 128, 8, 4, 4, 4, false, false, cutlass::half_t, Flash_kernel_traits`; `void flash::flash_bwd_dot_do_o_kernel<true, Flash_bwd_kernel_traits<64, 128, 128, 8, 4, 4, 4, false, false, cutlass::half_t, Flash_kernel_tr`; `void flash::flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<Flash_bwd_kernel_traits<64, 128, 128, 8, 4, 4, 4, false, false, cutlass::half_t, Fl`
- sdpa_flash_bwd: `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&):`; `void pytorch_flash::flash_bwd_convert_dq_kernel<Flash_bwd_kernel_traits<64, 128, 128, 8, 4, 4, 4, false, false, cutlass::half_t, Flash_kerne`; `void pytorch_flash::flash_bwd_dot_do_o_kernel<true, Flash_bwd_kernel_traits<64, 128, 128, 8, 4, 4, 4, false, false, cutlass::half_t, Flash_k`; `void pytorch_flash::flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<Flash_bwd_kernel_traits<64, 128, 128, 8, 4, 4, 4, false, false, cutlass::ha`
- v12b_real_D64_causal1: `Memset (Device)`; `flash::(anonymous namespace)::cqs_bwd_census_kernel(long const*, long const*, long const*, int, int, int, bool, int2*, int*)`; `void flash::flash_bwd_convert_dq_kernel<Flash_bwd_kernel_traits<64, 128, 128, 8, 4, 4, 4, false, false, cutlass::half_t, Flash_kernel_traits`; `void flash::flash_bwd_dot_do_o_kernel<true, Flash_bwd_kernel_traits<64, 128, 128, 8, 4, 4, 4, false, false, cutlass::half_t, Flash_kernel_tr`; `void flash::flash_bwd_dq_dk_dv_loop_seqk_parallel_aligned_kernel<Flash_bwd_kernel_traits<64, 128, 128, 8, 4, 4, 4, false, false, cutlass::ha`; `void flash::flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<Flash_bwd_kernel_traits<64, 128, 128, 8, 4, 4, 4, false, false, cutlass::half_t, Fl`
- v11_real_D64_causal1: `void flash::flash_bwd_convert_dq_kernel<Flash_bwd_kernel_traits<64, 128, 128, 8, 4, 4, 4, false, false, cutlass::half_t, Flash_kernel_traits`; `void flash::flash_bwd_dot_do_o_kernel<true, Flash_bwd_kernel_traits<64, 128, 128, 8, 4, 4, 4, false, false, cutlass::half_t, Flash_kernel_tr`; `void flash::flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<Flash_bwd_kernel_traits<64, 128, 128, 8, 4, 4, 4, false, false, cutlass::half_t, Fl`
- v12b_real_D128_causal1: `Memset (Device)`; `flash::(anonymous namespace)::cqs_bwd_census_kernel(long const*, long const*, long const*, int, int, int, bool, int2*, int*)`; `void flash::flash_bwd_convert_dq_kernel<Flash_bwd_kernel_traits<128, 64, 128, 8, 2, 4, 2, false, false, cutlass::half_t, Flash_kernel_traits`; `void flash::flash_bwd_dot_do_o_kernel<true, Flash_bwd_kernel_traits<128, 64, 128, 8, 2, 4, 2, false, false, cutlass::half_t, Flash_kernel_tr`; `void flash::flash_bwd_dq_dk_dv_loop_seqk_parallel_aligned_kernel<Flash_bwd_kernel_traits<128, 64, 128, 8, 2, 4, 2, false, false, cutlass::ha`; `void flash::flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<Flash_bwd_kernel_traits<128, 64, 128, 8, 2, 4, 2, false, false, cutlass::half_t, Fl`
- v11_real_D128_causal1: `void flash::flash_bwd_convert_dq_kernel<Flash_bwd_kernel_traits<128, 64, 128, 8, 2, 4, 2, false, false, cutlass::half_t, Flash_kernel_traits`; `void flash::flash_bwd_dot_do_o_kernel<true, Flash_bwd_kernel_traits<128, 64, 128, 8, 2, 4, 2, false, false, cutlass::half_t, Flash_kernel_tr`; `void flash::flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<Flash_bwd_kernel_traits<128, 64, 128, 8, 2, 4, 2, false, false, cutlass::half_t, Fl`
- v12b_real_D64_causal0: `Memset (Device)`; `flash::(anonymous namespace)::cqs_bwd_census_kernel(long const*, long const*, long const*, int, int, int, bool, int2*, int*)`; `void flash::flash_bwd_convert_dq_kernel<Flash_bwd_kernel_traits<64, 128, 128, 8, 4, 4, 4, false, false, cutlass::half_t, Flash_kernel_traits`; `void flash::flash_bwd_dot_do_o_kernel<true, Flash_bwd_kernel_traits<64, 128, 128, 8, 4, 4, 4, false, false, cutlass::half_t, Flash_kernel_tr`; `void flash::flash_bwd_dq_dk_dv_loop_seqk_parallel_aligned_kernel<Flash_bwd_kernel_traits<64, 128, 128, 8, 4, 4, 4, false, false, cutlass::ha`; `void flash::flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<Flash_bwd_kernel_traits<64, 128, 128, 8, 4, 4, 4, false, false, cutlass::half_t, Fl`
- v11_real_D64_causal0: `void flash::flash_bwd_convert_dq_kernel<Flash_bwd_kernel_traits<64, 128, 128, 8, 4, 4, 4, false, false, cutlass::half_t, Flash_kernel_traits`; `void flash::flash_bwd_dot_do_o_kernel<true, Flash_bwd_kernel_traits<64, 128, 128, 8, 4, 4, 4, false, false, cutlass::half_t, Flash_kernel_tr`; `void flash::flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<Flash_bwd_kernel_traits<64, 128, 128, 8, 4, 4, 4, false, false, cutlass::half_t, Fl`

