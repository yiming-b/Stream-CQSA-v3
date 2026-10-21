# U1 engine-level A/B: v2.2.3 vs v3 + v13 (job 15958597, della-l09g3, NVIDIA A100-SXM4-80GB)

`benchmarks/ultimate/engine_ab.py`; B=1 H=8 D=64 causal, device-resident inputs, acc=GPU, 2 timed runs after 1 warm-up, medians (ms). Backward = torch.autograd.grad of attention()'s output. rel = norm-relative error of the fp16/bf16 output / the worst gradient against the monolithic flash_attn call on the same inputs.

| N | c | itr | dtype | arm | engine that ran (itr) / n_sub / bwd kernel | fwd ms | bwd ms | peak fwd / bwd GiB | rel out / grads | fwd / flash_attn | bwd / flash_attn |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 131072 | 7 | 1 | bf16 | v3_cuda | classic itr=1 / 7 / cuda | 109.4 | 267.9 | 2.2 / 3.3 | 1.0e-03 / 2.8e-03 | 1.240 | 1.124 |
| 131072 | 7 | 1 | bf16 | flash_attn | monolithic / 1 / - | 88.2 | 238.5 | 1.5 / 2.1 | 0.0e+00 / 1.9e-05 | 1.000 | 1.000 |
| 131072 | 7 | 1 | bf16 | v223_cuda | skipped: v2.2.3's stable v11 module is fp16-only | | | | | | |
| 131072 | 7 | 1 | bf16 | v223_triton | skipped: v2.2.3's stable v11 module is fp16-only | | | | | | |
| 131072 | 7 | 1 | bf16 | v3_wave | wave itr=1 / 7 / cuda | 151.5 | 629.7 | 2.6 / 3.9 | 1.0e-03 / 3.0e-03 | 1.718 | 2.640 |
| 131072 | 7 | 1 | bf16 | v3_auto | monolithic itr=0 / 1 / sdpa | 96.9 | 257.6 | 1.5 / 2.4 | 6.3e-05 / 1.8e-04 | 1.098 | 1.080 |
| 131072 | 7 | 1 | bf16 | sdpa_flash | monolithic / 1 / - | 94.9 | 257.7 | 1.5 / 2.4 | 6.3e-05 / 1.8e-04 | 1.076 | 1.081 |
| 131072 | 7 | 1 | fp16 | v3_cuda | classic itr=1 / 7 / cuda | 110.0 | 268.0 | 2.2 / 3.3 | 1.3e-04 / 3.6e-04 | 1.224 | 1.125 |
| 131072 | 7 | 1 | fp16 | flash_attn | monolithic / 1 / - | 89.8 | 238.3 | 1.5 / 2.1 | 0.0e+00 / 7.1e-06 | 1.000 | 1.000 |
| 131072 | 7 | 1 | fp16 | v223_cuda | classic itr=1 / 7 / cuda | 151.7 | 642.6 | 2.2 / 3.3 | 1.4e-04 / 3.6e-04 | 1.688 | 2.697 |
| 131072 | 7 | 1 | fp16 | v223_triton | classic itr=1 / 7 / triton | 151.6 | 502.8 | 2.2 / 3.3 | 1.4e-04 / 3.8e-04 | 1.687 | 2.110 |
| 131072 | 7 | 1 | fp16 | v3_wave | wave itr=1 / 7 / cuda | 151.4 | 631.4 | 2.6 / 3.9 | 1.3e-04 / 3.7e-04 | 1.685 | 2.650 |
| 131072 | 7 | 1 | fp16 | v3_auto | monolithic itr=0 / 1 / sdpa | 97.4 | 257.7 | 1.5 / 2.4 | 2.0e-05 / 4.8e-05 | 1.084 | 1.082 |
| 131072 | 7 | 1 | fp16 | sdpa_flash | monolithic / 1 / - | 95.4 | 257.9 | 1.5 / 2.4 | 2.0e-05 / 4.8e-05 | 1.062 | 1.082 |
| 131072 | 73 | 1 | fp16 | v3_cuda | classic itr=1 / 73 / cuda | 167.8 | 340.5 | 1.9 / 2.9 | 2.1e-04 / 3.5e-04 | 1.868 | 1.429 |
| 131072 | 73 | 1 | fp16 | v223_cuda | classic itr=1 / 73 / cuda | 66.3 | 133.5 | 1.9 / 2.9 | 1.3e+00 / 1.4e+00 | 0.738 | 0.560 |
| 131072 | 73 | 1 | fp16 | v223_triton | classic itr=1 / 73 / triton | 67.8 | 261.6 | 1.9 / 2.9 | 1.3e+00 / 1.4e+00 | 0.754 | 1.098 |
| 131072 | 73 | 1 | fp16 | v3_wave | wave itr=1 / 73 / cuda | 194.9 | 628.9 | 3.6 / 5.1 | 2.1e-04 / 3.6e-04 | 2.170 | 2.639 |
| 131072 | 73 | 1 | fp16 | v3_auto | monolithic itr=0 / 1 / sdpa | 97.1 | 257.8 | 1.5 / 2.4 | 2.0e-05 / 4.8e-05 | 1.080 | 1.082 |
| 1048576 | 7 | 1 | fp16 | v3_cuda | classic itr=1 / 7 / cuda | 5843.7 | 15207.4 | 17.5 / 26.1 | 2.7e-04 / 4.3e-04 | 0.975 | 1.007 |
| 1048576 | 7 | 1 | fp16 | flash_attn | monolithic / 1 / - | 5991.0 | 15107.3 | 12.0 / 17.1 | 0.0e+00 / 9.9e-06 | 1.000 | 1.000 |
| 1048576 | 7 | 1 | fp16 | v223_cuda | classic itr=1 / 7 / cuda | 7789.4 | 39170.8 | 17.5 / 26.1 | 2.7e-04 / 4.3e-04 | 1.300 | 2.593 |
| 1048576 | 7 | 1 | fp16 | v223_triton | classic itr=1 / 7 / triton | 7784.8 | 28994.8 | 17.5 / 26.1 | 2.7e-04 / 1.2e-03 | 1.299 | 1.919 |
| 1048576 | 7 | 1 | fp16 | v3_wave | wave itr=1 / 7 / cuda | 7855.9 | 39300.7 | 18.4 / 27.0 | 2.7e-04 / 4.4e-04 | 1.311 | 2.601 |
| 1048576 | 7 | 1 | fp16 | v3_auto | monolithic itr=0 / 1 / sdpa | 6346.8 | 16321.2 | 12.0 / 19.1 | 2.6e-05 / 5.6e-05 | 1.059 | 1.080 |
| 1048576 | 7 | 1 | fp16 | sdpa_flash | monolithic / 1 / - | 6339.9 | 16349.7 | 12.0 / 19.1 | 2.6e-05 / 5.6e-05 | 1.058 | 1.082 |
| 1048576 | 7 | 2 | fp16 | v3_cuda | classic itr=2 / 49 / cuda | 6031.7 | 15635.3 | 8.2 / 16.0 | - (itr=2: not re-checked) | 1.007 | 1.035 |
| 1048576 | 7 | 2 | fp16 | v223_cuda | classic itr=2 / 49 / cuda | 8310.2 | 45049.3 | 8.2 / 16.0 | - (itr=2: not re-checked) | 1.387 | 2.982 |
| 1048576 | 7 | 2 | fp16 | v223_triton | classic itr=2 / 49 / triton | 8311.0 | 29499.5 | 8.2 / 16.0 | - (itr=2: not re-checked) | 1.387 | 1.953 |
| 1048576 | 7 | 2 | fp16 | v3_wave | wave itr=2 / 49 / cuda | 8338.4 | 44925.3 | 10.7 / 19.1 | - (itr=2: not re-checked) | 1.392 | 2.974 |
| 1048576 | 7 | 2 | fp16 | v3_auto | skipped: budget: 1442 s elapsed + 167 s estimated > 1500 s | | | | | | |

## Design acceptance (01_design_a100.md P2.1 / P3.x)

| target | measured | pass |
|---|---|---|
| engine backward 1M itr=1 device-resident acc=GPU <= 24 s (v2.2.x: 39.1 s CUDA / 29.0 s Triton) | v3+v13 15.21 s; this job's v2.2.3: 39.17 s CUDA / 28.99 s Triton | yes |
| forward 1M c=7 itr=1 device-resident <= 1.17x monolithic flash_attn | 0.975x (5844 vs 5991 ms); v2.2.3 1.300x | yes |

Arms: `v3_cuda` = the classic engine on the v13 module (itr, c pinned); `v3_wave` = the wave engine, which still runs the old `cqsa_native` module (package U2 ports the census + ALIGNED design into it; itr, c pinned); `v3_auto` = `attention()` with nothing pinned, i.e. what the user gets for this call on this device (the engine column says what ran: at these N the monolithic SDPA call fits, so that row is not a measurement of the new kernels); `v223_*` = the released v2.2.3 package, classic engine, pinned. The engine arms' rows report the engine that actually ran, its depth and the subproblem count.
