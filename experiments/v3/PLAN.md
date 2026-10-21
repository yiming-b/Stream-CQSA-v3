# Stream-CQSA v3 experiment plan

One plan, one harness, one results directory. Every measurement of v3 lives under `results/v3/` as one
JSONL row per (experiment, configuration, arm) plus the tables and figures generated from it; the plan is
the table below. Written 2026-10-21; status column kept current by the harness's report step.

## Parameters and their values

| parameter | values supported | value fixed in the sweeps | why |
|---|---|---|---|
| N (tokens) | 128K, 256K, 512K, 1M, 2M, 4M, 8M | varies in E1/E2; **1M** elsewhere | 1M is the largest size at which every arm (incl. the resident baselines and the host-resident 8M rows' references) runs in one gpu-test job |
| precision | fp16, bf16 | **fp16**; E6 proves bf16 == fp16 behaviour | bf16 costs the same (engine_ab: 109.4 vs 110.0 ms at 131K) |
| head dim | 64, 128 | **64**; E6 covers 128 | 128 = 2x the pair FLOPs, nothing else changes |
| baseline | SDPA (flash backend), FA-2 (flash_attn 2.8.3) | **both** in every row that fits | FA-2 package is ~5% faster than SDPA's copy of the same kernel; both are reported |
| c (quorum size) | 3, 7, 13, 21, 31, 57, 73, 91, 133 | **7**; varies in E3 | interest sets: `stream_cqsa.autoconfig.QUORUM_SETS` (perfect difference sets, verified) |
| itr (depth) | 1, 2 | **1**; both in E1/E2/E4 | depth 3 only through escalation |
| acc (accumulator) | gpu, cpu | **gpu**; both in E2/E4 | cpu = the host-resident recovery configuration |
| series (engine x kernel) | classic_CUDA, classic_Triton, wave_CUDA, wave_Triton | **all four** in every experiment | the four production paths |
| direction | forward, forward+backward | **both** | backward = autograd of attention()'s output; "fwd+bwd" times include the forward |
| layout | 128-aligned (v3 default) | fixed | the v2 unaligned layout is a kernel A/B question, settled (`results/ultimate/layout_ab.json`) |

Fixed everywhere: B=1, H=8, causal, device-resident Q/K/V unless the row says host-resident, one process per
configuration (allocator state cannot leak between rows), 1 warm-up + 3 timed runs, median, `nvidia-smi` GPU
name in every row. Node type: **A100-SXM4-80GB** (`--qos=gpu-test --constraint="sxm&gpu80"`, 59-min jobs) for
E1-E6; the 8M rows and the 4M fwd+bwd rows exceed a 59-min slot and run on **A100-80GB-PCIe** (`gpu-short`,
`gpu80&pcie`) as E7 with their own baselines (never compare across node types).

## Metrics (every row)

1. `wall_s`: wall-clock of the call (forward) or of forward + backward (fwd+bwd), median of 3; `wall_min_s`.
2. `peak_dev_gib`: `torch.cuda.max_memory_allocated` over the call; `workspace_gib` = peak minus the inputs.
3. `peak_host_gib`: the process's VmHWM delta over the call (what the host-resident rows cost in RAM).
4. `rel_err`: norm-relative error of the fp16 output vs the FA-2 reference; `fp64_err` on 128 sampled rows
   (Stream-CQSA rows must be <= 1.2x FA-2's own fp64 error); backward: worst of dq/dk/dv vs FA-2's grads.
5. `engine`, `n_subproblems`, `n_waves`, `kernel`: what actually ran (an unpinned call that ran monolithically
   is labelled so; every Stream-CQSA row in E1-E7 pins itr and asserts the decomposition, per the rule in
   `tests/conftest.py`).
6. Derived: `ratio_fa2 = wall_s / wall_s(FA-2 resident)` and `ratio_sdpa`; `tokens_per_s`; for the OOM
   boundary, `largest_N_completed` per arm and budget.

## Experiments (each row of this table = one slurm part, each part ~20-50 min of GPU time)

| id | question | varies | fixed | arms | rows | GPU-min (est.) |
|---|---|---|---|---|---|---|
| E1 | wall and peak memory vs N, device-resident | N in {128K..4M} x itr {1,2} x direction | c=7, acc=gpu, fp16, D=64 | 4 series + SDPA + FA-2 | 7x2x2 x 6 = 168 | 2 parts: fwd ~25, fwd+bwd ~45 |
| E2 | the host-resident recovery configuration vs N | N in {128K..4M} x itr {1,2} x direction | c=7, acc=cpu, host-resident Q/K/V | 4 series (+ resident baselines from E1) | 7x2x2 x 4 = 112 | 2 parts: ~30 + ~50 |
| E3 | how c shapes time and memory | c in {3..133} x direction | N=1M, itr=1, acc=gpu | 4 series | 9x2 x 4 = 72 | 2 parts: ~30 + ~40 |
| E4 | itr x acc at one size | itr {1,2} x acc {gpu,cpu} x direction | N=1M, c=7 | 4 series | 2x2x2 x 4 = 32 | 1 part: ~35 |
| E5 | wave budget W (subproblems per launch) | max_wave_tokens ladder | N=1M, c=73, itr=1, acc=gpu | wave_CUDA, wave_Triton | 5 x 2 x 2 dir = 20 | 1 part: ~20 |
| E6 | precision and head dim | {fp16,bf16} x {64,128} x direction | N=1M, c=7, itr=1, acc=gpu | 4 series + SDPA + FA-2 | 4x2 x 6 = 48 | 1 part: ~40 |
| E7 | beyond the device: the boundary | N in {4M, 8M} x direction x acc | c=7, itr=1 | 4 series + resident baselines where they fit | 2x2x2 x 6 = 48 | gpu-short PCIe, 2 parts: ~3 h each |
| E8 | multi-device scaling | world {1,2,4} x N {1M,2M} x itr {1,2} | c=7, acc=gpu, scheduler lpt+rs | classic_CUDA, wave_CUDA | 24 | gpu-short 4x PCIe, 1 part ~1 h |
| E9 | automatic mode: what the user gets | budget {10,20,40,80 GiB} x N {512K..4M} x direction | `attention(kernel="auto")`, nothing pinned | auto + the OOC rivals (reuse `experiments/ooc_rivals`) | from the OOC grid | re-run of the OOC harness on v3: gpu-short PCIe |

E1-E6 and the decomposition audit's notebook regeneration go into gpu-test SXM jobs (several parts per job,
each part time-boxed, the job re-submittable with `--only` to finish what a time-box cut). E7-E9 are
gpu-short PCIe jobs.

## Readers of the results

- `results/v3/results.jsonl`: every row (appended; the report takes the last row per key).
- `results/v3/report.md`: generated tables per experiment (time / peak / ratio vs FA-2 / error), the fits
  (quadratic in N for time, linear for memory, per series), and the figures `results/v3/E*.png`.
- `results/v3/STATUS.md`: which rows exist, which are missing, which failed (OOM / error), with job ids.
- The README's measurement sections are regenerated from report.md (the v2 README's "Headline" and
  "One parameter at a time" sections become E1 / E3 of v3).

## Reuse

The harness `experiments/v3/bench.py` generalises `benchmarks/profile_sweep.py` (its worker-per-configuration
and JSONL conventions) and `benchmarks/ultimate/engine_ab.py` (engine labels, decomposition assertion, FA-2 /
SDPA arms, backward via autograd, fp64 spot check); `experiments/ooc_rivals/bench.py` is reused as is for E9;
`benchmarks/ultimate/dist_telemetry.py` for E8.
