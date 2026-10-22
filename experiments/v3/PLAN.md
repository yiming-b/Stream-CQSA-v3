# Stream-CQSA v3 experiment plan

One plan, one harness, one results directory. Every measurement of v3 lives under `results/v3/` as one
JSONL row per (experiment, configuration, arm) plus the tables and figures generated from it; the plan is
the table below. Written 2026-10-21; status column kept current by the harness's report step.

## Parameters and their values

| parameter | values supported | value fixed in the sweeps | why |
|---|---|---|---|
| N (tokens) | 128K, 256K, 512K, 1M, 2M, 4M, 8M | varies in E1/E2; **1M** elsewhere; 8M only host-resident (E2, PCIe) | 1M is the largest size at which every arm runs comfortably in a 59-min gpu-test part |
| precision | fp16, bf16 | **fp16**; E6 proves bf16 == fp16 behaviour | bf16 costs the same (engine_ab: 109.4 vs 110.0 ms at 131K) |
| head dim | 64, 128 | **64**; E6 covers 128 | 128 = 2x the pair FLOPs, nothing else changes |
| baseline | SDPA (flash backend), FA-2 (flash_attn 2.8.3) | **both** in every row that fits | FA-2 package is ~5% faster than SDPA's copy of the same kernel; both are reported |
| c (quorum size) | 3, 7, 13, 21, 31, 57, 73, 91, 133 | **7**; varies in E3 | interest sets: `stream_cqsa.autoconfig.QUORUM_SETS` (perfect difference sets, verified) |
| itr (depth) | 1, 2 (3+ when the memory needs it) | **1**; E1/E2: the smallest depth >= 1 that fits; E4: both | min(itr) = 1 always: itr=0 would be the monolithic call |
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

## Experiments

Three questions, in the order the paper asked them: **(1) correctness**, **(2) scaling to very long
sequences with reasonable overhead**, **(3) how the parameters change the performance**. Each row of the
table is one slurm part (~20-50 min of GPU time). The earlier draft's E8 (multi-device) is dropped (the
scheduler's measurements already exist under `results/ultimate/dist_sched*.md`); E4/E5 are folded into E3.

| id | question | varies | fixed | arms | rows | GPU-min (est.) |
|---|---|---|---|---|---|---|
| **E0 correctness** | every series and every (c, itr, acc) path is exact: output and gradients vs FA-2 and vs fp64 | series x c in {3, 7, 31, 133} x itr {1, 2} x acc {gpu, cpu} x direction (fwd, fwd+bwd), plus bf16 and D=128 at c=7 | N=256K | 4 series, FA-2 and fp64 references | 4x4x2x2x2 + 4x2x2 = 144 | 1 part: ~30 (small rows, one timed run each; the errors are the result) |
| **E1 scaling** | wall and peak memory vs N, device-resident, with the depth the memory allows | N in {128K, 256K, 512K, 1M, 2M, 4M} x direction | c=7, acc=gpu, fp16, D=64; **itr = the smallest depth >= 1 whose plan fits** (the harness asks the planner with `plan_only`; a row that OOMs at that depth retries at itr+1 and records both) | 4 series + SDPA + FA-2 (the baselines up to where they fit) | 6x2 x 6 = 72 | 2 parts: fwd ~25, fwd+bwd ~45 |
| **E2 scaling, host-resident** | the recovery configuration: Q/K/V and the accumulator in host memory, same N ladder as E1 plus 8M | N in {128K..4M} (SXM) and {4M, 8M} (PCIe) x direction | c=7, acc=cpu, itr as in E1 (min 1) | 4 series (+ E1's resident baselines) | 6x2 x 4 + 2x2 x 4 = 64 | SXM: 2 parts ~30 + ~50; PCIe (8M): 1 part ~3 h |
| **E3 parameters: c** | how the quorum size shapes time and memory | c in {3, 7, 13, 21, 31, 57, 73, 91, 133} x direction | N=1M, itr=1, acc=gpu | 4 series | 9x2 x 4 = 72 | 2 parts: ~30 + ~40 |
| **E4 parameters: itr x acc** | the depth and the accumulator placement at one size | itr {1, 2} x acc {gpu, cpu} x direction | N=1M, c=7 | 4 series | 2x2x2 x 4 = 32 | 1 part: ~35 |
| **E5 parameters: wave budget** | subproblems per launch | max_wave_tokens ladder (5 rungs) x direction | N=1M, c=73, itr=1, acc=gpu | wave_CUDA, wave_Triton | 5x2 x 2 = 20 | 1 part: ~20 |
| **E6 parameters: precision, head dim** | bf16 and D=128 cost what fp16 / D=64 cost (performance; E0 has their correctness) | {fp16, bf16} x {64, 128} x direction | N=1M, c=7, itr=1, acc=gpu | 4 series + SDPA + FA-2 | 4x2 x 6 = 48 | 1 part: ~40 |
| **E7 automatic mode** | what the user gets without pinning anything, under device budgets, vs the out-of-core rivals | budget {10, 20, 40, 80 GiB} x N {256K..4M} x direction | `attention(kernel="auto")`, nothing pinned (the engine column records what ran) | auto + FA-2 resident + the two rectangular OOC rivals | the OOC grid (`experiments/ooc_rivals`) re-run on v3 | gpu-short PCIe: ~8 x 1-7 h |

Depth rule for E1/E2 (the user's instruction): `itr` is never 0 (that would be the monolithic call) and is
raised one level at a time until the plan fits the device; the row records the depth that ran and, if a
lower depth was tried and OOMed, that attempt too. Correctness (E0) is a separate experiment so that the
scaling rows can time one run per configuration without the reference computations.

E0-E6 run on **A100-SXM4-80GB** in one resubmittable 59-min gpu-test job (`slurm/v3/E_sxm.slurm`; parts in
the order E0, E1:fwd, E3:fwd, E4, E1:bwd, E6, E3:bwd, E5, E2:fwd, E2:bwd; every part `--only-missing`,
so the job is resubmitted until STATUS.md shows no missing rows). E2's 8M rows and E7 run on
**A100-80GB-PCIe** (`gpu-short`): `slurm/v3/E_pcie.slurm` and the OOC harness.

## Readers of the results

- `results/v3/results.jsonl`: every row (appended; the report takes the last row per key).
- `results/v3/report.md`: generated tables per experiment (time / peak / ratio vs FA-2 / error), the fits
  (quadratic in N for time, linear for memory, per series), and the figures `results/v3/E*.png`.
- `results/v3/STATUS.md`: which rows exist, which are missing, which failed (OOM / error), with job ids.
- The README's measurement sections are regenerated from report.md (the v2 README's "Headline" and
  "One parameter at a time" sections become E1 / E3 of v3; the paper's accuracy table becomes E0).

## Reuse

The harness `experiments/v3/bench.py` generalises `benchmarks/profile_sweep.py` (its worker-per-configuration
and JSONL conventions) and `benchmarks/ultimate/engine_ab.py` (engine labels, decomposition assertion, FA-2 /
SDPA arms, backward via autograd, fp64 spot check); `experiments/ooc_rivals/bench.py` is reused as is for E7.
