# D2 scheduler dry run: the refit cost model on the real task lists (CPU; c=7, causal, H=8, D=64)

Cost model: `scheduler.PATH_COSTS` (per gathered token and per engine call of the chunk-pool, host-gather and device paths), `TASK_S`, `BWD_FACTOR`, the PCIe-80 rate 2.91e7 tiles/s -- all fitted on the per-rank local times of job 14330294 (`benchmarks/ultimate/dist_cost_refit.py`, `results/ultimate/dist_cost_refit.json`); the merge from `CostModel.dist_merge_s` (per-token allreduce / reduce_scatter / all_gather terms of the same job). The job's throttled rank (54/108 SMs) ran at 0.573 of a full rank.

## 1. Predicted vs measured, every arm of job 14330294

predicted local time = the scheduler's load of the units the rank ran (depth, L, live tiles from the row); merge+fixed measured = t_dist - max t_local.

| tag | dir | N | itr | arm | units/rank | max t_local meas s | pred s | merge+fixed meas s | pred s | t_dist meas s | pred s | meas/pred |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| homog | fwd | 1M | 1 | rr/allreduce/seed | 2 2 2 1 | 3.42 | 3.32 | 0.13 | 0.13 | 3.55 | 3.45 | 1.030 |
| homog | fwd | 1M | 1 | rr/reduce_scatter/seed | 2 2 2 1 | 3.42 | 3.32 | 0.09 | 0.11 | 3.51 | 3.43 | 1.022 |
| homog | fwd | 1M | 1 | lpt/allreduce/seed | 6 6 6 7 | 3.83 | 3.80 | 0.12 | 0.13 | 3.95 | 3.93 | 1.006 |
| homog | fwd | 1M | 1 | lpt/reduce_scatter/seed | 6 6 6 7 | 3.84 | 3.80 | 0.10 | 0.11 | 3.94 | 3.91 | 1.007 |
| homog | fwd | 2M | 1 | rr/allreduce/seed | 2 2 2 1 | 12.20 | 11.90 | 0.21 | 0.23 | 12.41 | 12.13 | 1.023 |
| homog | fwd | 2M | 1 | rr/reduce_scatter/seed | 2 2 2 1 | 12.12 | 11.90 | 0.17 | 0.20 | 12.29 | 12.10 | 1.016 |
| homog | fwd | 2M | 1 | lpt/allreduce/seed | 7 6 6 6 | 12.23 | 12.20 | 0.23 | 0.23 | 12.47 | 12.42 | 1.003 |
| homog | fwd | 2M | 1 | lpt/reduce_scatter/seed | 7 6 6 6 | 12.26 | 12.20 | 0.20 | 0.20 | 12.46 | 12.39 | 1.005 |
| homog | fwd | 1M | 2 | rr/allreduce/seed | 13 12 12 12 | 4.00 | 3.93 | 0.11 | 0.13 | 4.11 | 4.06 | 1.010 |
| homog | fwd | 1M | 2 | rr/reduce_scatter/seed | 13 12 12 12 | 3.96 | 3.93 | 0.09 | 0.11 | 4.05 | 4.05 | 0.999 |
| homog | fwd | 1M | 2 | lpt/allreduce/seed | 12 12 12 13 | 3.85 | 3.85 | 0.11 | 0.13 | 3.96 | 3.98 | 0.996 |
| homog | fwd | 1M | 2 | lpt/reduce_scatter/seed | 12 12 12 13 | 3.86 | 3.85 | 0.09 | 0.11 | 3.94 | 3.96 | 0.996 |
| homog | fwd | 2M | 2 | rr/allreduce/seed | 13 12 12 12 | 12.57 | 12.61 | 0.21 | 0.23 | 12.79 | 12.84 | 0.996 |
| homog | fwd | 2M | 2 | rr/reduce_scatter/seed | 13 12 12 12 | 12.58 | 12.61 | 0.17 | 0.20 | 12.76 | 12.81 | 0.996 |
| homog | fwd | 2M | 2 | lpt/allreduce/seed | 13 12 12 12 | 12.41 | 12.27 | 0.21 | 0.23 | 12.62 | 12.50 | 1.010 |
| homog | fwd | 2M | 2 | lpt/reduce_scatter/seed | 13 12 12 12 | 12.30 | 12.27 | 0.17 | 0.20 | 12.48 | 12.47 | 1.001 |
| homog | bwd | 1M | 1 | rr/allreduce/seed | 2 2 2 1 | 9.99 | 9.60 | 0.27 | 0.33 | 10.26 | 9.93 | 1.033 |
| homog | bwd | 1M | 1 | rr/reduce_scatter/seed | 2 2 2 1 | 9.63 | 9.60 | 0.22 | 0.28 | 9.85 | 9.89 | 0.996 |
| homog | bwd | 1M | 1 | lpt/allreduce/seed | 6 6 6 7 | 8.25 | 8.42 | 0.28 | 0.33 | 8.53 | 8.75 | 0.975 |
| homog | bwd | 1M | 1 | lpt/reduce_scatter/seed | 6 6 6 7 | 8.38 | 8.42 | 0.23 | 0.28 | 8.61 | 8.70 | 0.990 |
| hetero_sm54 | fwd | 1M | 1 | rr/allreduce/seed | 2 2 2 1 | 5.35 | 5.28 | 0.13 | 0.13 | 5.48 | 5.41 | 1.013 |
| hetero_sm54 | fwd | 1M | 1 | lpt/allreduce/seed | 6 6 6 7 | 5.44 | 5.42 | 0.12 | 0.13 | 5.56 | 5.55 | 1.002 |
| hetero_sm54 | fwd | 1M | 1 | lpt/allreduce/measured | 2 1 2 2 | 3.45 | 3.32 | 0.11 | 0.13 | 3.55 | 3.45 | 1.030 |
| hetero_sm54 | fwd | 1M | 1 | lpt/reduce_scatter/measured | 2 1 2 2 | 3.43 | 3.32 | 0.09 | 0.11 | 3.52 | 3.43 | 1.024 |
| hetero_sm54 | fwd | 2M | 1 | rr/allreduce/seed | 2 2 2 1 | 19.69 | 19.75 | 0.21 | 0.23 | 19.90 | 19.98 | 0.996 |
| hetero_sm54 | fwd | 2M | 1 | lpt/allreduce/seed | 7 6 6 6 | 18.59 | 18.86 | 0.24 | 0.23 | 18.83 | 19.08 | 0.987 |
| hetero_sm54 | fwd | 2M | 1 | lpt/allreduce/measured | 2 1 2 2 | 12.08 | 11.91 | 0.21 | 0.23 | 12.30 | 12.13 | 1.013 |
| hetero_sm54 | fwd | 2M | 1 | lpt/reduce_scatter/measured | 2 1 2 2 | 12.11 | 11.91 | 0.18 | 0.20 | 12.28 | 12.10 | 1.015 |
| hetero_sm54_itr2 | fwd | 1M | 2 | rr/allreduce/seed | 13 12 12 12 | 5.34 | 5.32 | 0.14 | 0.13 | 5.47 | 5.45 | 1.004 |
| hetero_sm54_itr2 | fwd | 1M | 2 | lpt/allreduce/seed | 12 12 12 13 | 5.46 | 5.47 | 0.11 | 0.13 | 5.57 | 5.60 | 0.994 |
| hetero_sm54_itr2 | fwd | 1M | 2 | lpt/allreduce/measured | 13 9 14 13 | 4.21 | 4.20 | 0.11 | 0.13 | 4.32 | 4.33 | 0.998 |

Worst |meas/pred - 1| over the 31 arms: 0.033.

## 2. The default rule's decision against the measured arms

lpt with the cost-aware children split on the row's rate table (`seed` arms: equal refit rates, the throttled rank at its fitted fraction; `measured` arms: the job's calibrated per-rank rates in their measured proportions), matched to the measured arm that ran the same assignment (the same unit counts per depth on every rank, rank permutations among equal ranks allowed) and compared with the fastest arm measured at that shape (any reduction).

| tag | dir | N | itr | rates | rule: units/rank (splits) | pred makespan s | measured arm with that assignment | its t_dist s | best measured arm | its t_dist s | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| homog | fwd | 1M | 1 | 1.00 1.00 1.00 1.00 | 1 2 2 2 (0) | 3.32 | rr/reduce_scatter/seed | 3.51 | rr/reduce_scatter/seed | 3.51 | winner |
| homog | fwd | 2M | 1 | 1.00 1.00 1.00 1.00 | 1 2 2 2 (0) | 11.90 | rr/reduce_scatter/seed | 12.29 | rr/reduce_scatter/seed | 12.29 | winner |
| homog | fwd | 1M | 2 | 1.00 1.00 1.00 1.00 | 12 12 12 13 (0) | 3.77 | lpt/reduce_scatter/seed | 3.94 | lpt/reduce_scatter/seed | 3.94 | winner |
| homog | fwd | 2M | 2 | 1.00 1.00 1.00 1.00 | 13 12 12 12 (0) | 12.13 | lpt/reduce_scatter/seed | 12.48 | lpt/reduce_scatter/seed | 12.48 | winner |
| homog | bwd | 1M | 1 | 1.00 1.00 1.00 1.00 | 6 6 6 7 (3) | 8.42 | lpt/allreduce/seed | 8.53 | lpt/allreduce/seed | 8.53 | winner |
| hetero_sm54 | fwd | 1M | 1 | 1.00 0.57 1.00 1.00 | 2 1 2 2 (0) | 3.32 | lpt/reduce_scatter/measured | 3.52 | lpt/reduce_scatter/measured | 3.52 | winner |
| hetero_sm54 | fwd | 1M | 1 | 1.00 0.60 0.99 1.01 | 2 1 2 2 (0) | 3.33 | lpt/reduce_scatter/measured | 3.52 | lpt/reduce_scatter/measured | 3.52 | winner |
| hetero_sm54 | fwd | 2M | 1 | 1.00 0.57 1.00 1.00 | 2 1 2 2 (0) | 11.90 | lpt/reduce_scatter/measured | 12.28 | lpt/reduce_scatter/measured | 12.28 | winner |
| hetero_sm54 | fwd | 2M | 1 | 1.00 0.60 0.99 1.01 | 2 1 2 2 (0) | 11.96 | lpt/reduce_scatter/measured | 12.28 | lpt/reduce_scatter/measured | 12.28 | winner |
| hetero_sm54_itr2 | fwd | 1M | 2 | 1.00 0.57 1.00 1.00 | 14 9 13 13 (0) | 4.09 | lpt/allreduce/measured | 4.32 | lpt/allreduce/measured | 4.32 | winner |
| hetero_sm54_itr2 | fwd | 1M | 2 | 0.98 0.63 1.01 1.01 | 13 10 13 13 (0) | 4.03 | not measured | - | lpt/allreduce/measured | 4.32 | predicted vs measured best: 4.14 vs 4.32 |

## 3. The model over rate vectors and shapes

ratio = predicted makespan / the fluid ideal (pair work at the summed rate + the per-unit overheads spread evenly + one engine call); `rows (U1)` is the model-only row-span split (needs the kernel row-range parameter).

| path | ranks | N | itr | rr | lpt (whole tasks) | lpt + cost-aware children | splits | units/rank | makespan s | rows (U1) | schedule s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| fwd host-streamed | 4 x PCIe-80 (equal) | 1M | 1 | 1.111 | 1.110 | 1.110 | 0 | 1 2 2 2 | 3.32 | 1.002 | 0.0 |
| fwd host-streamed | 4 x PCIe-80 (equal) | 1M | 2 | 1.043 | 1.000 | 1.000 | 0 | 12 12 12 13 | 3.77 | 1.000 | 0.0 |
| fwd host-streamed | 4 x PCIe-80 (equal) | 2M | 1 | 1.125 | 1.125 | 1.125 | 0 | 1 2 2 2 | 11.90 | 1.001 | 0.0 |
| fwd host-streamed | 4 x PCIe-80 (equal) | 2M | 2 | 1.040 | 1.000 | 1.000 | 0 | 13 12 12 12 | 12.13 | 1.000 | 0.0 |
| fwd host-streamed | 4 x PCIe-80 (equal) | 4M | 1 | 1.133 | 1.133 | 1.075 | 4 | 6 7 7 11 | 42.59 | 1.000 | 31.1 |
| fwd host-streamed | 4 x PCIe-80 (equal) | 4M | 2 | 1.038 | 1.000 | 1.000 | 0 | 13 12 12 12 | 42.66 | 1.000 | 13.1 |
| fwd host-streamed | SXM4, PCIe-40, PCIe-40, SXM4 | 1M | 1 | 1.329 | 1.328 | 1.130 | 1 | 3 3 4 3 | 3.48 | 1.016 | 0.9 |
| fwd host-streamed | SXM4, PCIe-40, PCIe-40, SXM4 | 1M | 2 | 1.151 | 1.009 | 1.009 | 0 | 14 10 11 14 | 3.90 | 1.009 | 0.1 |
| fwd host-streamed | SXM4, PCIe-40, PCIe-40, SXM4 | 2M | 1 | 1.370 | 1.370 | 1.073 | 1 | 3 3 4 3 | 11.75 | 1.014 | 3.0 |
| fwd host-streamed | SXM4, PCIe-40, PCIe-40, SXM4 | 2M | 2 | 1.191 | 1.007 | 1.007 | 0 | 15 10 10 14 | 12.58 | 1.007 | 0.1 |
| fwd host-streamed | SXM4, PCIe-40, PCIe-40, SXM4 | 4M | 1 | 1.394 | 1.394 | 1.054 | 1 | 3 4 3 3 | 43.27 | 1.014 | 0.0 |
| fwd host-streamed | SXM4, PCIe-40, PCIe-40, SXM4 | 4M | 2 | 1.219 | 1.005 | 1.005 | 0 | 14 10 10 15 | 44.31 | 1.005 | 0.0 |
| fwd host-streamed | (1, 1, 0.67, 0.67) x PCIe-80 | 1M | 1 | 1.340 | 1.340 | 1.113 | 1 | 3 3 3 4 | 3.83 | 1.015 | 0.0 |
| fwd host-streamed | (1, 1, 0.67, 0.67) x PCIe-80 | 1M | 2 | 1.161 | 1.009 | 1.009 | 0 | 14 14 10 11 | 4.27 | 1.009 | 0.0 |
| fwd host-streamed | (1, 1, 0.67, 0.67) x PCIe-80 | 2M | 1 | 1.378 | 1.378 | 1.067 | 1 | 3 3 3 4 | 13.23 | 1.013 | 0.1 |
| fwd host-streamed | (1, 1, 0.67, 0.67) x PCIe-80 | 2M | 2 | 1.199 | 1.007 | 1.007 | 0 | 15 14 10 10 | 14.04 | 1.007 | 0.0 |
| fwd host-streamed | (1, 1, 0.67, 0.67) x PCIe-80 | 4M | 1 | 1.400 | 1.400 | 1.049 | 1 | 3 3 4 3 | 49.21 | 1.013 | 0.0 |
| fwd host-streamed | (1, 1, 0.67, 0.67) x PCIe-80 | 4M | 2 | 1.225 | 1.004 | 1.004 | 0 | 15 14 10 10 | 50.16 | 1.004 | 0.1 |
| fwd host-streamed | (1, 0.573, 1, 1) x PCIe-80 (the job's throttle) | 1M | 1 | 1.619 | 1.017 | 1.017 | 0 | 2 1 2 2 | 3.32 | 1.017 | 0.0 |
| fwd host-streamed | (1, 0.573, 1, 1) x PCIe-80 (the job's throttle) | 1M | 2 | 1.315 | 1.010 | 1.010 | 0 | 14 9 13 13 | 4.09 | 1.010 | 0.0 |
| fwd host-streamed | (1, 0.573, 1, 1) x PCIe-80 (the job's throttle) | 2M | 1 | 1.691 | 1.018 | 1.018 | 0 | 2 1 2 2 | 11.90 | 1.018 | 0.0 |
| fwd host-streamed | (1, 0.573, 1, 1) x PCIe-80 (the job's throttle) | 2M | 2 | 1.389 | 1.008 | 1.008 | 0 | 14 8 14 13 | 13.33 | 1.008 | 0.1 |
| fwd host-streamed | (1, 0.573, 1, 1) x PCIe-80 (the job's throttle) | 4M | 1 | 1.733 | 1.020 | 1.020 | 0 | 2 1 2 2 | 44.89 | 1.020 | 0.0 |
| fwd host-streamed | (1, 0.573, 1, 1) x PCIe-80 (the job's throttle) | 4M | 2 | 1.440 | 1.005 | 1.005 | 0 | 13 8 14 14 | 47.29 | 1.005 | 0.0 |
| fwd host-streamed | 8 x PCIe-80 (equal) | 1M | 1 | 1.091 | 1.091 | 1.091 | 0 | 1 1 1 1 1 1 1 0 | 2.00 | 1.010 | 7.4 |
| fwd host-streamed | 8 x PCIe-80 (equal) | 1M | 2 | 1.142 | 1.001 | 1.001 | 0 | 7 6 6 6 6 6 6 6 | 1.98 | 1.001 | 0.1 |
| fwd host-streamed | 8 x PCIe-80 (equal) | 2M | 1 | 1.111 | 1.111 | 1.048 | 7 | 7 6 6 6 6 6 6 6 | 6.26 | 1.003 | 0.0 |
| fwd host-streamed | 8 x PCIe-80 (equal) | 2M | 2 | 1.151 | 1.001 | 1.001 | 0 | 7 6 6 6 6 6 6 6 | 6.26 | 1.001 | 0.1 |
| fwd host-streamed | 8 x PCIe-80 (equal) | 4M | 1 | 1.125 | 1.125 | 1.026 | 7 | 6 7 6 6 6 6 6 6 | 21.72 | 1.001 | 0.0 |
| fwd host-streamed | 8 x PCIe-80 (equal) | 4M | 2 | 1.158 | 1.000 | 1.000 | 0 | 6 7 6 6 6 6 6 6 | 21.72 | 1.000 | 0.0 |
| fwd host-streamed | 2 x SXM4 + 2 x PCIe-80 | 1M | 1 | 1.174 | 1.174 | 1.174 | 0 | 2 2 2 1 | 3.32 | 1.052 | 0.1 |
| fwd host-streamed | 2 x SXM4 + 2 x PCIe-80 | 1M | 2 | 1.043 | 1.002 | 1.002 | 0 | 13 13 11 12 | 3.62 | 1.002 | 0.0 |
| fwd host-streamed | 2 x SXM4 + 2 x PCIe-80 | 2M | 1 | 1.198 | 1.198 | 1.059 | 1 | 2 2 5 4 | 10.52 | 1.059 | 0.1 |
| fwd host-streamed | 2 x SXM4 + 2 x PCIe-80 | 2M | 2 | 1.058 | 1.001 | 1.001 | 0 | 13 13 12 11 | 11.49 | 1.001 | 0.0 |
| fwd host-streamed | 2 x SXM4 + 2 x PCIe-80 | 4M | 1 | 1.212 | 1.212 | 1.063 | 2 | 7 2 5 5 | 39.35 | 1.063 | 0.1 |
| fwd host-streamed | 2 x SXM4 + 2 x PCIe-80 | 4M | 2 | 1.069 | 1.001 | 1.001 | 0 | 13 13 11 12 | 40.10 | 1.001 | 0.0 |
| fwd device-resident | 4 x PCIe-80 (equal) | 1M | 1 | 1.143 | 1.143 | 1.006 | 3 | 6 6 6 7 | 2.33 | 1.003 | 0.0 |
| fwd device-resident | 4 x PCIe-80 (equal) | 2M | 1 | 1.143 | 1.143 | 1.002 | 3 | 7 6 6 6 | 9.25 | 1.001 | 1.5 |
| fwd device-resident | SXM4, PCIe-40, PCIe-40, SXM4 | 1M | 1 | 1.422 | 1.421 | 1.022 | 1 | 3 3 4 3 | 2.45 | 1.021 | 0.0 |
| fwd device-resident | SXM4, PCIe-40, PCIe-40, SXM4 | 2M | 1 | 1.422 | 1.422 | 1.022 | 1 | 3 3 4 3 | 9.81 | 1.016 | 0.0 |
| fwd device-resident | (1, 1, 0.67, 0.67) x PCIe-80 | 1M | 1 | 1.423 | 1.423 | 1.021 | 1 | 3 3 3 4 | 2.82 | 1.019 | 0.0 |
| fwd device-resident | (1, 1, 0.67, 0.67) x PCIe-80 | 2M | 1 | 1.424 | 1.424 | 1.021 | 1 | 3 3 3 4 | 11.29 | 1.015 | 0.0 |
| fwd device-resident | (1, 0.573, 1, 1) x PCIe-80 (the job's throttle) | 1M | 1 | 1.781 | 1.021 | 1.021 | 0 | 2 1 2 2 | 2.64 | 1.021 | 0.0 |
| fwd device-resident | (1, 0.573, 1, 1) x PCIe-80 (the job's throttle) | 2M | 1 | 1.781 | 1.021 | 1.021 | 0 | 2 1 2 2 | 10.55 | 1.021 | 0.0 |
| fwd device-resident | 8 x PCIe-80 (equal) | 1M | 1 | 1.144 | 1.144 | 1.016 | 7 | 7 6 6 6 6 6 6 6 | 1.17 | 1.016 | 0.5 |
| fwd device-resident | 8 x PCIe-80 (equal) | 2M | 1 | 1.143 | 1.143 | 1.006 | 7 | 7 6 6 6 6 6 6 6 | 4.64 | 1.004 | 1.3 |
| fwd device-resident | 2 x SXM4 + 2 x PCIe-80 | 1M | 1 | 1.229 | 1.228 | 1.008 | 3 | 7 7 6 5 | 2.17 | 1.068 | 0.0 |
| fwd device-resident | 2 x SXM4 + 2 x PCIe-80 | 2M | 1 | 1.229 | 1.229 | 1.003 | 3 | 7 7 6 5 | 8.61 | 1.068 | 0.0 |
| bwd device-resident | 4 x PCIe-80 (equal) | 1M | 1 | 1.143 | 1.143 | 1.002 | 3 | 6 6 6 7 | 8.42 | 1.001 | 0.0 |
| bwd device-resident | 4 x PCIe-80 (equal) | 2M | 1 | 1.143 | 1.143 | 1.001 | 3 | 7 6 6 6 | 33.63 | 1.000 | 0.0 |
| bwd device-resident | SXM4, PCIe-40, PCIe-40, SXM4 | 1M | 1 | 1.422 | 1.421 | 1.021 | 1 | 3 3 4 3 | 8.91 | 1.016 | 0.0 |
| bwd device-resident | SXM4, PCIe-40, PCIe-40, SXM4 | 2M | 1 | 1.422 | 1.422 | 1.022 | 1 | 3 3 4 3 | 35.67 | 1.015 | 0.0 |
| bwd device-resident | (1, 1, 0.67, 0.67) x PCIe-80 | 1M | 1 | 1.424 | 1.423 | 1.020 | 1 | 3 3 3 4 | 10.26 | 1.015 | 0.0 |
| bwd device-resident | (1, 1, 0.67, 0.67) x PCIe-80 | 2M | 1 | 1.424 | 1.424 | 1.021 | 1 | 3 3 3 4 | 41.06 | 1.014 | 0.0 |
| bwd device-resident | (1, 0.573, 1, 1) x PCIe-80 (the job's throttle) | 1M | 1 | 1.781 | 1.021 | 1.021 | 0 | 2 1 2 2 | 9.60 | 1.021 | 0.0 |
| bwd device-resident | (1, 0.573, 1, 1) x PCIe-80 (the job's throttle) | 2M | 1 | 1.781 | 1.021 | 1.021 | 0 | 2 1 2 2 | 38.38 | 1.021 | 0.0 |
| bwd device-resident | 8 x PCIe-80 (equal) | 1M | 1 | 1.144 | 1.144 | 1.006 | 7 | 7 6 6 6 6 6 6 6 | 4.22 | 1.004 | 0.0 |
| bwd device-resident | 8 x PCIe-80 (equal) | 2M | 1 | 1.143 | 1.143 | 1.003 | 7 | 7 6 6 6 6 6 6 6 | 16.85 | 1.001 | 0.1 |
| bwd device-resident | 2 x SXM4 + 2 x PCIe-80 | 1M | 1 | 1.229 | 1.229 | 1.003 | 3 | 7 7 6 5 | 7.83 | 1.068 | 0.0 |
| bwd device-resident | 2 x SXM4 + 2 x PCIe-80 | 2M | 1 | 1.229 | 1.229 | 1.002 | 3 | 7 7 6 5 | 31.28 | 1.068 | 0.0 |
