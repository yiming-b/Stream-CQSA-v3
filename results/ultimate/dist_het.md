# D2 heterogeneous job 15949746: 2x SXM4-80 + 2x PCIe-80

## dist_het.json (tag `het_sxm2_pcie2`, job 15949746, della-l07g3)

GPUs: NVIDIA A100-SXM4-80GB, NVIDIA A100-SXM4-80GB, NVIDIA A100 80GB PCIe, NVIDIA A100 80GB PCIe; throttle: None; split: auto; chunk_tokens: 262144; engine: {'causal': True, 'stream_from_host': True, 'low_memory': False, 'allow_escalation': False, 'max_parallel': 1, 'out_device': 'acc'}

| rank | GPU | SMs used | seed rate (tiles/s) | measured rate (host-streamed) |
|---|---|---|---|---|
| 0 | NVIDIA A100-SXM4-80GB | 108/108 | 3.350e+07 | 3.872e+07 |
| 1 | NVIDIA A100-SXM4-80GB | 108/108 | 3.350e+07 | 3.869e+07 |
| 2 | NVIDIA A100 80GB PCIe | 108/108 | 2.910e+07 | 3.099e+07 |
| 3 | NVIDIA A100 80GB PCIe | 108/108 | 2.910e+07 | 3.257e+07 |

| dir | N | itr | layout | arm | split | single s | dist s | speedup | t_local per rank s | predicted s | imbalance | merge s (max) | units/rank | peak above start GiB (max) | rel err |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fwd | 1048576 | 1 | aligned_128_shared | rr/allreduce/seed | none | 9.24 | 3.72 | 2.49 | 3.25 3.27 3.34 1.88 | 2.97 2.97 3.32 2.00 | 1.138 | 1.84 | 2 2 2 1 | 7.8 | 3.3e-08 |
| fwd | 1048576 | 1 | aligned_128_shared | lpt/allreduce/seed | children | 9.24 | 3.65 | 2.53 | 3.29 3.28 3.31 1.89 | 2.97 2.97 3.32 2.00 | 1.126 | 1.76 | 2 2 2 1 | 7.8 | 3.7e-08 |
| fwd | 1048576 | 1 | aligned_128_shared | lpt/allreduce/measured | children | 9.24 | 3.62 | 2.55 | 3.28 3.28 1.93 3.23 | 2.67 2.67 1.92 3.04 | 1.120 | 1.69 | 2 2 1 2 | 7.8 | 3.5e-08 |
| fwd | 1048576 | 1 | aligned_128_shared | lpt/reduce_scatter/measured | children | 9.24 | 3.64 | 2.54 | 3.29 3.29 1.93 3.24 | 2.67 2.67 1.92 3.04 | 1.120 | 1.71 | 2 2 1 2 | 7.8 | 3.4e-08 |
| fwd | 1048576 | 1 | aligned_128_shared | lpt/reduce_scatter/measured/rep | children | 9.24 | 3.93 | 2.35 | 3.24 3.27 1.93 3.24 | 2.67 2.67 1.92 3.04 | 1.119 | 2.00 | 2 2 1 2 | 7.8 | 3.4e-08 |
| fwd | 2097152 | 1 | aligned_128_shared | rr/allreduce/seed | none | 34.21 | 12.45 | 2.75 | 11.07 11.05 11.77 6.27 | 10.52 10.52 11.90 6.63 | 1.172 | 6.17 | 2 2 2 1 | 15.5 | 3.3e-08 |
| fwd | 2097152 | 1 | aligned_128_shared | lpt/allreduce/seed | children | 34.21 | 11.75 | 2.91 | 11.05 11.03 10.19 9.87 | 10.52 10.52 10.42 10.32 | 1.049 | 1.87 | 2 2 5 4 | 15.5 | 6.4e-05 |
| fwd | 2097152 | 1 | aligned_128_shared | lpt/allreduce/measured | children | 34.21 | 11.76 | 2.91 | 11.06 11.07 10.02 10.15 | 9.29 9.29 9.70 9.70 | 1.047 | 1.74 | 2 2 4 5 | 15.5 | 6.4e-05 |
| fwd | 2097152 | 1 | aligned_128_shared | lpt/reduce_scatter/measured | children | 34.21 | 11.77 | 2.91 | 11.06 11.07 10.02 10.16 | 9.29 9.29 9.70 9.70 | 1.046 | 1.75 | 2 2 4 5 | 15.5 | 6.4e-05 |
| fwd | 2097152 | 1 | aligned_128_shared | lpt/reduce_scatter/measured/rep | children | 34.21 | 12.40 | 2.76 | 11.05 11.06 10.03 10.26 | 9.29 9.29 9.70 9.70 | 1.043 | 2.36 | 2 2 4 5 | 15.5 | 6.4e-05 |
| bwd | 1048576 | 1 | aligned_128_device | rr/allreduce/seed | none | 32.03 | 11.12 | 2.88 | 8.16 8.72 10.17 5.51 | 8.34 8.34 9.60 4.80 | 1.249 | 5.60 | 2 2 2 1 | 12.0 | 0.0e+00 |
| bwd | 1048576 | 1 | aligned_128_device | lpt/allreduce/seed | children | 32.03 | 8.69 | 3.69 | 7.71 7.70 7.59 7.34 | 7.83 7.83 7.80 7.83 | 1.017 | 1.34 | 7 7 6 5 | 14.6 | 1.0e-04 |
| bwd | 1048576 | 1 | aligned_128_device | lpt/allreduce/measured | children | 32.03 | 9.15 | 3.50 | 8.14 8.18 6.68 7.39 | 7.22 7.22 6.40 6.80 | 1.077 | 2.46 | 2 2 4 5 | 14.6 | 7.8e-05 |
| bwd | 1048576 | 1 | aligned_128_device | lpt/reduce_scatter/measured | children | 32.03 | 9.13 | 3.51 | 8.15 8.18 6.71 7.43 | 7.22 7.22 6.40 6.80 | 1.073 | 2.42 | 2 2 4 5 | 14.6 | 7.8e-05 |
| bwd | 1048576 | 1 | aligned_128_device | lpt/reduce_scatter/measured/rep | children | 32.03 | 10.09 | 3.17 | 8.15 8.18 6.72 7.45 | 7.22 7.22 6.40 6.80 | 1.073 | 3.37 | 2 2 4 5 | 14.6 | 7.8e-05 |

| dir | N | itr | rr/allreduce dist s | lpt/allreduce dist s | lpt gain | lpt/reduce_scatter dist s | merge ar -> rs s | peak+ ar -> rs GiB |
|---|---|---|---|---|---|---|---|---|
| bwd | 1048576 | 1 | 11.12 | 9.15 | 1.215x | 9.13 | 2.46 -> 2.42 | 14.6 -> 14.6 |
| fwd | 1048576 | 1 | 3.72 | 3.62 | 1.026x | 3.64 | 1.69 -> 1.71 | 7.8 -> 7.8 |
| fwd | 2097152 | 1 | 12.45 | 11.76 | 1.058x | 11.77 | 1.74 -> 1.75 | 15.5 -> 15.5 |

| dir | N | itr | arm | local share = max t_local / t_single | model share | residual | t_dist - max t_local s |
|---|---|---|---|---|---|---|---|

<details><summary>nvidia-smi topo -m</summary>

```
	[4mGPU0	GPU1	NIC0	CPU Affinity	NUMA Affinity	GPU NUMA ID[0m
GPU0	 X 	NV4	NODE	0-3	0		N/A
GPU1	NV4	 X 	SYS	24-27	1		N/A
NIC0	NODE	SYS	 X 				

Legend:

  X    = Self
  SYS  = Connection traversing PCIe as well as the SMP interconnect between NUMA nodes (e.g., QPI/UPI)
  NODE = Connection traversing PCIe as well as the interconnect between PCIe Host Bridges within a NUMA node
  PHB  = Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)
  PXB  = Connection traversing multiple PCIe bridges (without traversing the PCIe Host Bridge)
  PIX  = Connection traversing at most a single PCIe bridge
  NV#  = Connection traversing a bonded set of # NVLinks

NIC Legend:

  NIC0: mlx5_0
```
</details>

## dist_het_itr2.json (tag `het_sxm2_pcie2_itr2`, job 15949746, della-l07g3)

GPUs: NVIDIA A100-SXM4-80GB, NVIDIA A100-SXM4-80GB, NVIDIA A100 80GB PCIe, NVIDIA A100 80GB PCIe; throttle: None; split: auto; chunk_tokens: 262144; engine: {'causal': True, 'stream_from_host': True, 'low_memory': False, 'allow_escalation': False, 'max_parallel': 1, 'out_device': 'acc'}

| rank | GPU | SMs used | seed rate (tiles/s) | measured rate (host-streamed) |
|---|---|---|---|---|
| 0 | NVIDIA A100-SXM4-80GB | 108/108 | 3.350e+07 | 3.884e+07 |
| 1 | NVIDIA A100-SXM4-80GB | 108/108 | 3.350e+07 | 3.865e+07 |
| 2 | NVIDIA A100 80GB PCIe | 108/108 | 2.910e+07 | 3.089e+07 |
| 3 | NVIDIA A100 80GB PCIe | 108/108 | 2.910e+07 | 3.207e+07 |

| dir | N | itr | layout | arm | split | single s | dist s | speedup | t_local per rank s | predicted s | imbalance | merge s (max) | units/rank | peak above start GiB (max) | rel err |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fwd | 1048576 | 2 | aligned_128 | rr/allreduce/seed | none | 11.71 | 4.11 | 2.85 | 3.37 3.19 3.73 3.63 | 3.62 3.37 3.77 3.73 | 1.071 | 0.92 | 13 12 12 12 | 4.2 | 8.5e-08 |
| fwd | 1048576 | 2 | aligned_128 | lpt/allreduce/seed | children | 11.71 | 3.91 | 2.99 | 3.37 3.40 3.57 3.53 | 3.62 3.62 3.62 3.62 | 1.029 | 0.54 | 13 13 11 12 | 4.2 | 8.3e-08 |
| fwd | 1048576 | 2 | aligned_128 | lpt/allreduce/measured | children | 11.71 | 3.84 | 3.05 | 3.46 3.45 3.47 3.50 | 3.39 3.39 3.39 3.39 | 1.009 | 0.39 | 13 13 11 12 | 4.2 | 8.2e-08 |
| fwd | 1048576 | 2 | aligned_128 | lpt/reduce_scatter/measured | children | 11.71 | 3.93 | 2.98 | 3.45 3.46 3.46 3.50 | 3.39 3.39 3.39 3.39 | 1.010 | 0.48 | 13 13 11 12 | 4.2 | 8.3e-08 |
| fwd | 2097152 | 2 | aligned_128 | rr/allreduce/seed | none | 38.52 | 12.52 | 3.08 | 10.69 10.03 11.83 11.51 | 11.36 10.58 12.14 11.99 | 1.074 | 2.49 | 13 12 12 12 | 8.3 | 8.5e-08 |
| fwd | 2097152 | 2 | aligned_128 | lpt/allreduce/seed | children | 38.52 | 11.91 | 3.24 | 10.82 10.88 11.22 10.96 | 11.49 11.49 11.49 11.48 | 1.023 | 1.09 | 13 13 12 11 | 8.3 | 8.0e-08 |
| fwd | 2097152 | 2 | aligned_128 | lpt/allreduce/measured | children | 38.52 | 11.76 | 3.28 | 11.05 11.07 10.82 10.94 | 10.56 10.56 10.56 10.56 | 1.009 | 0.93 | 13 13 11 12 | 8.3 | 8.1e-08 |
| fwd | 2097152 | 2 | aligned_128 | lpt/reduce_scatter/measured | children | 38.52 | 11.72 | 3.29 | 11.02 11.05 10.83 10.95 | 10.56 10.56 10.56 10.56 | 1.008 | 0.89 | 13 13 11 12 | 8.3 | 8.1e-08 |

| dir | N | itr | rr/allreduce dist s | lpt/allreduce dist s | lpt gain | lpt/reduce_scatter dist s | merge ar -> rs s | peak+ ar -> rs GiB |
|---|---|---|---|---|---|---|---|---|
| fwd | 1048576 | 2 | 4.11 | 3.84 | 1.070x | 3.93 | 0.39 -> 0.48 | 4.2 -> 4.2 |
| fwd | 2097152 | 2 | 12.52 | 11.76 | 1.065x | 11.72 | 0.93 -> 0.89 | 8.3 -> 8.3 |

| dir | N | itr | arm | local share = max t_local / t_single | model share | residual | t_dist - max t_local s |
|---|---|---|---|---|---|---|---|

<details><summary>nvidia-smi topo -m</summary>

```
	[4mGPU0	GPU1	NIC0	CPU Affinity	NUMA Affinity	GPU NUMA ID[0m
GPU0	 X 	NV4	NODE	0-3	0		N/A
GPU1	NV4	 X 	SYS	24-27	1		N/A
NIC0	NODE	SYS	 X 				

Legend:

  X    = Self
  SYS  = Connection traversing PCIe as well as the SMP interconnect between NUMA nodes (e.g., QPI/UPI)
  NODE = Connection traversing PCIe as well as the interconnect between PCIe Host Bridges within a NUMA node
  PHB  = Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)
  PXB  = Connection traversing multiple PCIe bridges (without traversing the PCIe Host Bridge)
  PIX  = Connection traversing at most a single PCIe bridge
  NV#  = Connection traversing a bonded set of # NVLinks

NIC Legend:

  NIC0: mlx5_0
```
</details>

