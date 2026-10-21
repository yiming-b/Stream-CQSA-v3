# D1 multi-device scheduler: job results

## dist_sched_homog.json (tag `homog`, job 14330294, della-l02g13)

GPUs: NVIDIA A100 80GB PCIe, NVIDIA A100 80GB PCIe, NVIDIA A100 80GB PCIe, NVIDIA A100 80GB PCIe; throttle: None; split: auto; chunk_tokens: 262144; engine: {'causal': True, 'stream_from_host': True, 'low_memory': False, 'allow_escalation': False, 'max_parallel': 1, 'out_device': 'acc'}

| rank | GPU | SMs used | seed rate (tiles/s) | measured rate (host-streamed) |
|---|---|---|---|---|
| 0 | NVIDIA A100 80GB PCIe | 108/108 | 2.900e+07 | 2.369e+07 |
| 1 | NVIDIA A100 80GB PCIe | 108/108 | 2.900e+07 | 2.365e+07 |
| 2 | NVIDIA A100 80GB PCIe | 108/108 | 2.900e+07 | 2.339e+07 |
| 3 | NVIDIA A100 80GB PCIe | 108/108 | 2.900e+07 | 2.335e+07 |

| dir | N | itr | layout | arm | split | single s | dist s | speedup | t_local per rank s | predicted s | imbalance | merge s (max) | units/rank | peak above start GiB (max) | rel err |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fwd | 1048576 | 1 | aligned_128_shared | rr/allreduce/seed | none | 10.29 | 3.55 | 2.90 | 3.40 3.37 3.42 2.01 | 2.67 2.67 2.67 1.34 | 1.122 | 1.54 | 2 2 2 1 | 7.8 | 3.3e-08 |
| fwd | 1048576 | 1 | aligned_128_shared | rr/reduce_scatter/seed | none | 10.29 | 3.51 | 2.93 | 3.40 3.37 3.42 2.01 | 2.67 2.67 2.67 1.34 | 1.123 | 1.50 | 2 2 2 1 | 7.8 | 3.3e-08 |
| fwd | 1048576 | 1 | aligned_128_shared | lpt/allreduce/seed | children | 10.29 | 3.95 | 2.61 | 3.73 3.68 3.71 3.83 | 2.37 2.37 2.37 2.37 | 1.025 | 0.27 | 6 6 6 7 | 7.8 | 6.0e-05 |
| fwd | 1048576 | 1 | aligned_128_shared | lpt/reduce_scatter/seed | children | 10.29 | 3.94 | 2.62 | 3.74 3.68 3.70 3.84 | 2.37 2.37 2.37 2.37 | 1.025 | 0.25 | 6 6 6 7 | 7.8 | 6.0e-05 |
| fwd | 2097152 | 1 | aligned_128_shared | rr/allreduce/seed | none | 38.91 | 12.41 | 3.14 | 12.20 11.81 11.91 6.52 | 10.63 10.63 10.63 5.31 | 1.150 | 5.89 | 2 2 2 1 | 15.5 | 3.2e-08 |
| fwd | 2097152 | 1 | aligned_128_shared | rr/reduce_scatter/seed | none | 38.91 | 12.29 | 3.17 | 12.12 11.90 12.01 6.59 | 10.63 10.63 10.63 5.31 | 1.138 | 5.71 | 2 2 2 1 | 15.5 | 3.3e-08 |
| fwd | 2097152 | 1 | aligned_128_shared | lpt/allreduce/seed | children | 38.91 | 12.47 | 3.12 | 12.23 11.78 12.03 11.92 | 9.35 9.34 9.35 9.34 | 1.020 | 0.68 | 7 6 6 6 | 15.5 | 1.2e-04 |
| fwd | 2097152 | 1 | aligned_128_shared | lpt/reduce_scatter/seed | children | 38.91 | 12.46 | 3.12 | 12.26 11.81 12.07 11.95 | 9.35 9.34 9.35 9.34 | 1.020 | 0.64 | 7 6 6 6 | 15.5 | 1.2e-04 |
| fwd | 1048576 | 2 | aligned_128 | rr/allreduce/seed | none | 14.19 | 4.11 | 3.46 | 4.00 3.60 3.76 3.74 | 2.50 2.33 2.43 2.39 | 1.059 | 0.50 | 13 12 12 12 | 4.2 | 8.5e-08 |
| fwd | 1048576 | 2 | aligned_128 | rr/reduce_scatter/seed | none | 14.19 | 4.05 | 3.51 | 3.96 3.64 3.78 3.77 | 2.50 2.33 2.43 2.39 | 1.046 | 0.40 | 13 12 12 12 | 4.2 | 8.5e-08 |
| fwd | 1048576 | 2 | aligned_128 | lpt/allreduce/seed | children | 14.19 | 3.96 | 3.58 | 3.82 3.69 3.71 3.85 | 2.41 2.41 2.41 2.41 | 1.022 | 0.27 | 12 12 12 13 | 4.2 | 8.0e-08 |
| fwd | 1048576 | 2 | aligned_128 | lpt/reduce_scatter/seed | children | 14.19 | 3.94 | 3.60 | 3.77 3.73 3.74 3.86 | 2.41 2.41 2.41 2.41 | 1.022 | 0.21 | 12 12 12 13 | 4.2 | 8.0e-08 |
| fwd | 2097152 | 2 | aligned_128 | rr/allreduce/seed | none | 45.93 | 12.79 | 3.59 | 12.57 11.38 11.90 11.84 | 9.76 9.08 9.48 9.33 | 1.055 | 1.40 | 13 12 12 12 | 8.3 | 8.5e-08 |
| fwd | 2097152 | 2 | aligned_128 | rr/reduce_scatter/seed | none | 45.93 | 12.76 | 3.60 | 12.58 11.47 11.97 11.92 | 9.76 9.08 9.48 9.33 | 1.050 | 1.28 | 13 12 12 12 | 8.3 | 8.5e-08 |
| fwd | 2097152 | 2 | aligned_128 | lpt/allreduce/seed | children | 45.93 | 12.62 | 3.64 | 12.41 11.72 11.85 11.95 | 9.42 9.42 9.40 9.41 | 1.036 | 0.90 | 13 12 12 12 | 8.3 | 8.2e-08 |
| fwd | 2097152 | 2 | aligned_128 | lpt/reduce_scatter/seed | children | 45.93 | 12.48 | 3.68 | 12.30 11.84 11.92 11.97 | 9.42 9.42 9.40 9.41 | 1.025 | 0.64 | 13 12 12 12 | 8.3 | 8.2e-08 |
| bwd | 1048576 | 1 | aligned_128_device | rr/allreduce/seed | none | 33.43 | 10.26 | 3.26 | 9.60 9.99 9.97 5.37 | 7.97 7.96 7.96 3.98 | 1.144 | 4.89 | 2 2 2 1 | 12.0 | 0.0e+00 |
| bwd | 1048576 | 1 | aligned_128_device | rr/reduce_scatter/seed | none | 33.43 | 9.85 | 3.39 | 9.63 9.43 9.48 4.83 | 7.97 7.96 7.96 3.98 | 1.154 | 5.02 | 2 2 2 1 | 12.0 | 0.0e+00 |
| bwd | 1048576 | 1 | aligned_128_device | lpt/allreduce/seed | children | 33.43 | 8.53 | 3.92 | 8.25 8.12 8.20 8.21 | 7.00 7.00 7.00 7.00 | 1.007 | 0.41 | 6 6 6 7 | 14.6 | 1.0e-04 |
| bwd | 1048576 | 1 | aligned_128_device | lpt/reduce_scatter/seed | children | 33.43 | 8.61 | 3.88 | 8.38 8.26 8.33 8.34 | 7.00 7.00 7.00 7.00 | 1.006 | 0.35 | 6 6 6 7 | 14.6 | 1.0e-04 |

| dir | N | itr | rr/allreduce dist s | lpt/allreduce dist s | lpt gain | lpt/reduce_scatter dist s | merge ar -> rs s | peak+ ar -> rs GiB |
|---|---|---|---|---|---|---|---|---|
| bwd | 1048576 | 1 | 10.26 | 8.53 | 1.202x | 8.61 | 0.41 -> 0.35 | 14.6 -> 14.6 |
| fwd | 1048576 | 1 | 3.55 | 3.95 | 0.900x | 3.94 | 0.27 -> 0.25 | 7.8 -> 7.8 |
| fwd | 1048576 | 2 | 4.11 | 3.96 | 1.037x | 3.94 | 0.27 -> 0.21 | 4.2 -> 4.2 |
| fwd | 2097152 | 1 | 12.41 | 12.47 | 0.996x | 12.46 | 0.68 -> 0.64 | 15.5 -> 15.5 |
| fwd | 2097152 | 2 | 12.79 | 12.62 | 1.013x | 12.48 | 0.90 -> 0.64 | 8.3 -> 8.3 |

| dir | N | itr | arm | local share = max t_local / t_single | model share | residual | t_dist - max t_local s |
|---|---|---|---|---|---|---|---|
| fwd | 1048576 | 1 | rr/allreduce/seed | 0.332 | 0.286 | 1.163 | 0.13 |
| fwd | 1048576 | 1 | rr/reduce_scatter/seed | 0.332 | 0.286 | 1.163 | 0.09 |
| fwd | 1048576 | 1 | lpt/allreduce/seed | 0.372 | 0.250 | 1.486 | 0.12 |
| fwd | 1048576 | 1 | lpt/reduce_scatter/seed | 0.373 | 0.250 | 1.488 | 0.10 |
| fwd | 2097152 | 1 | rr/allreduce/seed | 0.313 | 0.286 | 1.097 | 0.21 |
| fwd | 2097152 | 1 | rr/reduce_scatter/seed | 0.311 | 0.286 | 1.090 | 0.17 |
| fwd | 2097152 | 1 | lpt/allreduce/seed | 0.314 | 0.250 | 1.256 | 0.23 |
| fwd | 2097152 | 1 | lpt/reduce_scatter/seed | 0.315 | 0.250 | 1.259 | 0.20 |
| fwd | 1048576 | 2 | rr/allreduce/seed | 0.282 | 0.259 | 1.086 | 0.11 |
| fwd | 1048576 | 2 | rr/reduce_scatter/seed | 0.279 | 0.259 | 1.075 | 0.09 |
| fwd | 1048576 | 2 | lpt/allreduce/seed | 0.271 | 0.250 | 1.086 | 0.11 |
| fwd | 1048576 | 2 | lpt/reduce_scatter/seed | 0.272 | 0.250 | 1.086 | 0.09 |
| fwd | 2097152 | 2 | rr/allreduce/seed | 0.274 | 0.259 | 1.056 | 0.21 |
| fwd | 2097152 | 2 | rr/reduce_scatter/seed | 0.274 | 0.259 | 1.057 | 0.17 |
| fwd | 2097152 | 2 | lpt/allreduce/seed | 0.270 | 0.250 | 1.080 | 0.21 |
| fwd | 2097152 | 2 | lpt/reduce_scatter/seed | 0.268 | 0.250 | 1.071 | 0.17 |
| bwd | 1048576 | 1 | rr/allreduce/seed | 0.299 | 0.286 | 1.045 | 0.27 |
| bwd | 1048576 | 1 | rr/reduce_scatter/seed | 0.288 | 0.286 | 1.008 | 0.22 |
| bwd | 1048576 | 1 | lpt/allreduce/seed | 0.247 | 0.250 | 0.987 | 0.28 |
| bwd | 1048576 | 1 | lpt/reduce_scatter/seed | 0.251 | 0.250 | 1.002 | 0.23 |

<details><summary>nvidia-smi topo -m</summary>

```
	[4mGPU0	GPU1	GPU2	GPU3	NIC0	NIC1	CPU Affinity	NUMA Affinity	GPU NUMA ID[0m
GPU0	 X 	NV12	SYS	SYS	NODE	SYS	0,2,4,6,8,10	0		N/A
GPU1	NV12	 X 	SYS	SYS	NODE	SYS	0,2,4,6,8,10	0		N/A
GPU2	SYS	SYS	 X 	NV12	SYS	NODE	1,3,5,7,9,11	1		N/A
GPU3	SYS	SYS	NV12	 X 	SYS	NODE	1,3,5,7,9,11	1		N/A
NIC0	NODE	NODE	SYS	SYS	 X 	SYS				
NIC1	SYS	SYS	NODE	NODE	SYS	 X 				

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
  NIC1: mlx5_1
```
</details>

## dist_sched_hetero.json (tag `hetero_sm54`, job 14330294, della-l02g13)

GPUs: NVIDIA A100 80GB PCIe, NVIDIA A100 80GB PCIe, NVIDIA A100 80GB PCIe, NVIDIA A100 80GB PCIe; throttle: 1:54; split: auto; chunk_tokens: 262144; engine: {'causal': True, 'stream_from_host': True, 'low_memory': False, 'allow_escalation': False, 'max_parallel': 1, 'out_device': 'acc'}

| rank | GPU | SMs used | seed rate (tiles/s) | measured rate (host-streamed) |
|---|---|---|---|---|
| 0 | NVIDIA A100 80GB PCIe | 108/108 | 2.900e+07 | 2.350e+07 |
| 1 | NVIDIA A100 80GB PCIe | 54/108 | 2.900e+07 | 1.409e+07 |
| 2 | NVIDIA A100 80GB PCIe | 108/108 | 2.900e+07 | 2.342e+07 |
| 3 | NVIDIA A100 80GB PCIe | 108/108 | 2.900e+07 | 2.377e+07 |

| dir | N | itr | layout | arm | split | single s | dist s | speedup | t_local per rank s | predicted s | imbalance | merge s (max) | units/rank | peak above start GiB (max) | rel err |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fwd | 1048576 | 1 | aligned_128_shared | rr/allreduce/seed | none | 10.31 | 5.48 | 1.88 | 3.42 5.35 3.40 1.99 | 2.67 2.67 2.67 1.34 | 1.511 | 3.49 | 2 2 2 1 | 7.8 | 3.3e-08 |
| fwd | 1048576 | 1 | aligned_128_shared | lpt/allreduce/seed | children | 10.31 | 5.56 | 1.86 | 3.74 5.44 3.67 3.83 | 2.37 2.37 2.37 2.37 | 1.304 | 1.89 | 6 6 6 7 | 7.8 | 6.0e-05 |
| fwd | 1048576 | 1 | aligned_128_shared | lpt/allreduce/measured | children | 10.31 | 3.55 | 2.90 | 3.45 3.00 3.40 3.40 | 3.29 2.74 3.30 3.26 | 1.041 | 0.55 | 2 1 2 2 | 7.8 | 3.5e-08 |
| fwd | 1048576 | 1 | aligned_128_shared | lpt/reduce_scatter/measured | children | 10.31 | 3.52 | 2.93 | 3.43 2.98 3.38 3.39 | 3.29 2.74 3.30 3.26 | 1.040 | 0.54 | 2 1 2 2 | 7.8 | 3.4e-08 |
| fwd | 2097152 | 1 | aligned_128_shared | rr/allreduce/seed | none | 38.97 | 19.90 | 1.96 | 12.21 19.69 11.93 6.54 | 10.63 10.63 10.63 5.31 | 1.564 | 13.36 | 2 2 2 1 | 15.5 | 3.2e-08 |
| fwd | 2097152 | 1 | aligned_128_shared | lpt/allreduce/seed | children | 38.97 | 18.83 | 2.07 | 12.20 18.59 11.90 11.85 | 9.35 9.34 9.35 9.34 | 1.363 | 6.97 | 7 6 6 6 | 15.5 | 1.2e-04 |
| fwd | 2097152 | 1 | aligned_128_shared | lpt/allreduce/measured | children | 38.97 | 12.30 | 3.17 | 12.08 10.46 12.05 12.01 | 13.11 10.92 13.15 12.96 | 1.037 | 1.83 | 2 1 2 2 | 15.5 | 4.1e-08 |
| fwd | 2097152 | 1 | aligned_128_shared | lpt/reduce_scatter/measured | children | 38.97 | 12.28 | 3.17 | 12.10 10.44 12.11 12.03 | 13.11 10.92 13.15 12.96 | 1.038 | 1.85 | 2 1 2 2 | 15.5 | 4.1e-08 |

| dir | N | itr | rr/allreduce dist s | lpt/allreduce dist s | lpt gain | lpt/reduce_scatter dist s | merge ar -> rs s | peak+ ar -> rs GiB |
|---|---|---|---|---|---|---|---|---|
| fwd | 1048576 | 1 | 5.48 | 3.55 | 1.542x | 3.52 | 0.55 -> 0.54 | 7.8 -> 7.8 |
| fwd | 2097152 | 1 | 19.90 | 12.30 | 1.618x | 12.28 | 1.83 -> 1.85 | 15.5 -> 15.5 |

| dir | N | itr | arm | local share = max t_local / t_single | model share | residual | t_dist - max t_local s |
|---|---|---|---|---|---|---|---|
| fwd | 1048576 | 1 | rr/allreduce/seed | 0.519 | 0.286 | 1.814 | 0.13 |
| fwd | 1048576 | 1 | lpt/allreduce/seed | 0.527 | 0.250 | 2.106 | 0.12 |
| fwd | 2097152 | 1 | rr/allreduce/seed | 0.505 | 0.286 | 1.768 | 0.21 |
| fwd | 2097152 | 1 | lpt/allreduce/seed | 0.477 | 0.250 | 1.907 | 0.24 |

<details><summary>nvidia-smi topo -m</summary>

```
	[4mGPU0	GPU1	GPU2	GPU3	NIC0	NIC1	CPU Affinity	NUMA Affinity	GPU NUMA ID[0m
GPU0	 X 	NV12	SYS	SYS	NODE	SYS	0,2,4,6,8,10	0		N/A
GPU1	NV12	 X 	SYS	SYS	NODE	SYS	0,2,4,6,8,10	0		N/A
GPU2	SYS	SYS	 X 	NV12	SYS	NODE	1,3,5,7,9,11	1		N/A
GPU3	SYS	SYS	NV12	 X 	SYS	NODE	1,3,5,7,9,11	1		N/A
NIC0	NODE	NODE	SYS	SYS	 X 	SYS				
NIC1	SYS	SYS	NODE	NODE	SYS	 X 				

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
  NIC1: mlx5_1
```
</details>

## dist_sched_hetero_itr2.json (tag `hetero_sm54_itr2`, job 14330294, della-l02g13)

GPUs: NVIDIA A100 80GB PCIe, NVIDIA A100 80GB PCIe, NVIDIA A100 80GB PCIe, NVIDIA A100 80GB PCIe; throttle: 1:54; split: auto; chunk_tokens: 262144; engine: {'causal': True, 'stream_from_host': True, 'low_memory': False, 'allow_escalation': False, 'max_parallel': 1, 'out_device': 'acc'}

| rank | GPU | SMs used | seed rate (tiles/s) | measured rate (host-streamed) |
|---|---|---|---|---|
| 0 | NVIDIA A100 80GB PCIe | 108/108 | 2.900e+07 | 2.260e+07 |
| 1 | NVIDIA A100 80GB PCIe | 54/108 | 2.900e+07 | 1.449e+07 |
| 2 | NVIDIA A100 80GB PCIe | 108/108 | 2.900e+07 | 2.313e+07 |
| 3 | NVIDIA A100 80GB PCIe | 108/108 | 2.900e+07 | 2.315e+07 |

| dir | N | itr | layout | arm | split | single s | dist s | speedup | t_local per rank s | predicted s | imbalance | merge s (max) | units/rank | peak above start GiB (max) | rel err |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fwd | 1048576 | 2 | aligned_128 | rr/allreduce/seed | none | 14.08 | 5.47 | 2.57 | 4.00 5.34 3.80 3.82 | 2.50 2.33 2.43 2.39 | 1.259 | 1.67 | 13 12 12 12 | 4.2 | 8.5e-08 |
| fwd | 1048576 | 2 | aligned_128 | lpt/allreduce/seed | children | 14.08 | 5.57 | 2.53 | 3.78 5.46 3.78 3.88 | 2.41 2.41 2.41 2.41 | 1.292 | 1.79 | 12 12 12 13 | 4.2 | 8.0e-08 |
| fwd | 1048576 | 2 | aligned_128 | lpt/allreduce/measured | children | 14.08 | 4.32 | 3.26 | 4.05 4.00 4.21 4.14 | 3.32 3.32 3.32 3.32 | 1.027 | 0.32 | 13 9 14 13 | 4.2 | 8.1e-08 |

| dir | N | itr | rr/allreduce dist s | lpt/allreduce dist s | lpt gain | lpt/reduce_scatter dist s | merge ar -> rs s | peak+ ar -> rs GiB |
|---|---|---|---|---|---|---|---|---|
| fwd | 1048576 | 2 | 5.47 | 4.32 | 1.267x | - | 0.32 -> - | 4.2 -> - |

| dir | N | itr | arm | local share = max t_local / t_single | model share | residual | t_dist - max t_local s |
|---|---|---|---|---|---|---|---|
| fwd | 1048576 | 2 | rr/allreduce/seed | 0.379 | 0.259 | 1.461 | 0.14 |
| fwd | 1048576 | 2 | lpt/allreduce/seed | 0.388 | 0.250 | 1.550 | 0.11 |

<details><summary>nvidia-smi topo -m</summary>

```
	[4mGPU0	GPU1	GPU2	GPU3	NIC0	NIC1	CPU Affinity	NUMA Affinity	GPU NUMA ID[0m
GPU0	 X 	NV12	SYS	SYS	NODE	SYS	0,2,4,6,8,10	0		N/A
GPU1	NV12	 X 	SYS	SYS	NODE	SYS	0,2,4,6,8,10	0		N/A
GPU2	SYS	SYS	 X 	NV12	SYS	NODE	1,3,5,7,9,11	1		N/A
GPU3	SYS	SYS	NV12	 X 	SYS	NODE	1,3,5,7,9,11	1		N/A
NIC0	NODE	NODE	SYS	SYS	 X 	SYS				
NIC1	SYS	SYS	NODE	NODE	SYS	 X 				

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
  NIC1: mlx5_1
```
</details>

