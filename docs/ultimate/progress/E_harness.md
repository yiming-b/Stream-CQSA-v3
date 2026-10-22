# Package E-HARNESS: the v3 experiment harness and job scripts (experiments/v3/PLAN.md)

2026-10-21. Deliverables: `experiments/v3/bench.py` (modes run / worker / report / status / plan), `slurm/v3/parts.sh`
(the part runner), `slurm/v3/E_sxm.slurm` (E0-E6, gpu-test SXM, resubmittable), `slurm/v3/E_pcie.slurm` (E2's 4M / 8M
host-resident rows, gpu-short PCIe), PLAN.md updated (E7 recipe, corrected rows and estimates, status pointer).
Everything below from vis1 (shared A100-PCIe-40GB) is a PLUMBING CHECK, not a measurement.

## Design decisions (what the harness does and why)

- One JSONL row per (experiment, arm, N, c, itr, acc, qkv residency, dtype, D, direction, extra, node class); the
  last row per key wins; `run` skips present rows (ok and oom are final; `--retry-errors` re-runs error rows,
  `--redo` everything). Every row runs in a fresh `worker` subprocess (`python -P bench.py worker '<json>'`,
  RESULT line parsed by the loop; a crash without RESULT is an error row with the stdout tail; a host-OOM kill is
  an error row naming rc=-9).
- Budget: `--budget-s` defers rows whose estimated wall (a quadratic model anchored at 6 s / forward at 1M on SXM,
  3.6x fwd+bwd, 1.15x / 1.8x Triton fwd / fwd+bwd, 1.3x host-resident, x calls per row + startup, overridden by the measured wall of the
  same configuration when larger, x1.25 safety) would overrun; it never kills a row half-way. The slurm part runner
  gives every part min(28 min, what remains of the job) and a `timeout` 5 min beyond that as a backstop; the job
  ends with `report` and `status`, whose last line `MISSING <n>` says whether to resubmit.
- Ordering within a part: (node, N, D, dtype, direction, FA-2, SDPA, classic_CUDA, wave_CUDA, classic_Triton,
  wave_Triton, c, itr, acc, W): cheapest first, both baselines before the series at every N.
- Pinning + assertion: every Stream-CQSA row calls `attention(kernel=<series>, itr=, c=, interest_set=QUORUM_SETS[c],
  accumulate_on_gpu=, low_memory=, [max_wave_tokens=], [bwd_itr=itr], return_info=True)` and runs
  tests/_decomp.py's `assert_decomposed(info, itr=, c=)` on the info (imported by path from the snapshot's tests/; an
  inline replica if absent). A failed assertion is an error row. Forward rows pass `allow_escalation=False`; for
  fwd+bwd rows the harness wraps `stream_cqsa.native_autograd.stream_cqsa_forward / _backward` so the classic
  autograd op reports its forward info (n_subproblems, oom_retries, depth_escalations) and its backward's bwd_info
  (itr_reached, depth_escalations) and runs with `allow_escalation=False` too: a pinned depth that does not fit is
  an OOM row, never a silently different configuration. The wave backward's record comes from attention()'s
  `info["backward"]`.
- Depth rule (E1/E2, `itr="min"` in the key): `smallest_depth_that_fits` asks `attention(..., plan_only=True)` at
  itr=1, 2, 3, 4 with the series' kernel, the pinned c and the row's residency / accumulator, and takes the first
  depth with an `ok` candidate of that (itr, c, acc, residency); the row records itr_planned, plan_name,
  plan_reason, then runs; an OOM retries at itr+1 (two retries), each attempt recorded in `itr_attempts`;
  `itr_ran` is what ran.
- References: the fp64 spot check (128 sampled query rows incl. 0 and N-1: an online-softmax fp64 forward over all
  keys in 16K-key chunks on the device (~0.4 GiB of fp64 scratch), and the fp64 dQ of the same rows for fwd+bwd) costs O(128 N) and runs for
  every row, baselines included (so FA-2's own fp64 error is known at every N: `fp64_err` of the FA-2 row). Under
  ref="fa2" (E0, E3-E6) the worker also runs FA-2 (output + grads) on token-major copies of the same inputs, moves
  the results to the host, and reports `rel_err` / `rel_err_grads` (chunked fp64 norms over the full tensors) plus
  FA-2's fp64 error on the same sampled rows and the ratio (`fp64_ratio_fa2`, `fp64_dq_ratio_fa2`; PLAN.md's
  <= 1.2x criterion). An FA-2 reference that does not fit is `ref_fa2 = "oom"` and the row keeps the spot check.
- Inputs: seeded per (N, D, dtype), generated on the device in 1M-token chunks so device- and host-resident rows
  hold identical values; host rows: pinned token-major Q/K/V ([B, N, H, D] storage viewed as [B, H, N, D], what the
  streaming engines gather) and a pinned head-major dO; the FA-2 arm gets token-major device storage (its call is a
  view, no transpose in the timed region). One tiny untimed call (N=16K) of the arm precedes everything (Triton JIT,
  module loads, streams), so E0's single run and E1/E2's single timed run are not compile-time measurements.
- Memory: `peak_dev_gib` = max_memory_allocated over the call (reset per run), `workspace_gib` = peak minus the
  allocation at the start of the run (inputs + dO; the host-side references do not count), `peak_host_gib` = the
  VmHWM delta over the timed runs, `vmhwm_gib` the worker's absolute high-water mark (for --mem sizing).
- Report: per experiment a table (wall / min / peak / workspace / host / ratios vs FA-2 and SDPA of the same (node,
  N, D, dtype, direction), the experiment's own baselines first and any experiment's otherwise / rel vs FA-2 / fp64
  with the ratio to FA-2's / status, with the engine that ran and `[*]` where it is not the series' nominal one),
  E1/E2 fits (time quadratic in N, peak linear in N, per series and direction, >= 3 points, as profile_sweep),
  figures E0 (errors vs c), E1/E2 (wall and peak vs N, log-log, with E1's baselines drawn into E2), E3 (vs c),
  E4 / E6 (grouped bars), E5 (vs W), and STATUS.md (planned / ok / oom / error / missing per part, job ids, the
  failed rows with their messages).

## Facts found while building (also written into PLAN.md)

- The planner caps a pinned decomposition at 4096 subproblems: c=133 and c=73 at itr=2 raise "no decomposition of
  N can run", so E0's itr=2 rows exist for c in {3, 7, 31} only (128 rows, not 144).
- Routing: `kernel="wave-triton"` fwd+bwd runs the classic engine with the Triton kernel (the wave backward is CUDA
  only); a host-resident backward runs the classic engine whatever kernel= says; the wave backward keeps its
  accumulator on the device (`wave_attention` has no accumulate_on_gpu), so wave_CUDA acc=cpu fwd+bwd rows (E4) run
  with a device accumulator in the backward; the rows record `engine_expected`, `engine_as_expected`, `acc_ran`.
- `attention(kernel="triton")` sets CQSA_FORWARD=triton in the process environment for good: a reason for one
  process per row (the harness strips CQSA_FORWARD from every worker's environment).
- E5's ladder: at N=1M, c=73 the largest subproblem is 129,664 tokens; W = 1/2/4/8/16 x that gives 73 / 37 / 19 /
  10 / 5 waves (the planner's own halving ladder {2M, 1M, 512K, 256K, floor} forms the same waves).
- Cost: with the plan's protocol the SXM experiments are ~770 GPU-min (E0 ~70, E1 ~180, E2 ~165, E3 ~135, E4 ~65,
  E5 ~40, E6 ~120) -- ~14 gpu-test submissions; the PCIe parts ~14 h (two submissions). `E_ARGS="--reps 1"` on the
  E3-E6 parts would roughly halve them.
- Host memory of the host-resident fwd+bwd rows is dominated by the classic streamed backward's fp32 host gradients
  (3 x N x 2 KiB) plus their fp16 casts: ~71 GiB at 4M (E_sxm --mem=80G), ~137 GiB at 8M (E_pcie --mem=150G); the
  arithmetic is in the job headers.

## Smoke on vis1 (shared A100-PCIe-40GB; PLUMBING CHECKS, NOT MEASUREMENTS)

`bench.py run --smoke` (N=16K / 32K, no warm-up, one run, output under `.smoke/v3/`, git-ignored) for E1:fwd (all
6 arms, both N), E1:bwd (FA-2, classic_CUDA, wave_CUDA, wave_Triton), E0 (c=3 x both acc x both directions,
c=133 host-resident fwd+bwd, the bf16 and D=128 rows fwd+bwd), E3:fwd (c=3, 133), E4 (fwd all 16; bwd
wave_CUDA / classic_Triton acc=cpu), E5 (fwd, 5 rungs x 2), E6 (fwd, all 4 combos x 6 arms), E2:fwd (8
host-resident rows), E2:bwd (classic_CUDA, wave_CUDA host-resident): 122 rows, 122 ok, 0 errors; every series
row's decomposition asserted; `report` renders 7 tables, the E0 worst-case table, fits (none on the 2-point smoke
set: >= 3 points), figures E0-E6 (E2 with E1's baselines drawn in),
STATUS.md; `status` prints `MISSING <n>`; `--only-missing` re-runs nothing (12 already present). The fit path was exercised on three points (16K / 32K smoke rows + the 131K capped-run row: a quadratic with R2 1.000 and a linear memory fit, figure with the fitted curve drawn).

| experiment | rows ok | engines observed (series: engine/kernel) | fp64_err | fp64 ratio vs FA-2 (out / dQ) | rel vs FA-2 (out / grads) |
|---|---|---|---|---|---|
| E0 | 34 | classic/cuda, wave/cuda, classic/triton, wave/triton; wave_Triton fwd+bwd -> classic/triton; host-resident wave_CUDA bwd -> classic/cuda | 1.1e-4 (fp16 D128) .. 7.9e-4 (bf16) | 0.99-1.00 / 0.96-1.05 | 9.4e-5 .. 2.2e-3 (bf16) / 3.5e-4 .. 3.0e-3 (bf16) |
| E1 | 16 | all four + FA-2 / SDPA; depth rule planned itr=1 | 1.3e-4 .. 1.5e-4 (spot only) | - | - (ref="spot") |
| E2 | 10 | host-resident: classic chunk pipeline, wave chunk pool; host bwd -> classic | 1.3e-4 .. 1.5e-4 | - | - |
| E3 | 8 | c=3: 3 subproblems; c=133: 133 (4 waves) | 1.3e-4 | 0.99-1.00 | 9.4e-5 .. 2.8e-4 |
| E4 | 20 | itr=2: 49 subproblems (2 waves); acc=cpu host +0.2-0.4 GiB | 1.3e-4 | 1.00 / 0.97-1.05 | 1.4e-4 .. 2.8e-4 / 3.6e-4 |
| E5 | 10 | W=1x..16x -> 73 / 37 / 19 / 9 / 5 waves at N=32K | 1.3e-4 | 0.99 | 2.1e-4 .. 2.8e-4 |
| E6 | 24 | all four at fp16/bf16 x D=64/128 + baselines | 1.1e-4 .. 8.8e-4 | 1.00 | 1.4e-4 .. 2.3e-3 |

OOM paths (allocator capped with `--mem-cap-gib`, output `.smoke/oom/`): an E4 row pinned at itr=1 under a 0.95 GiB
cap at N=131072 is an `oom` row with `itr_attempts=[(1, oom)]` and the engine's message ("a subproblem at itr=1
does not fit and escalation is disabled"); an E1 row under the same cap planned and ran itr=4 (2401 subproblems,
peak 0.53 GiB; the planner honours the cap, so the planned depth fit and the OOM-then-retry branch was not
exercised end to end -- it is the same try/except around `run_timed` with `itr + 1`); an OOM before the arm (in the
fp64 spot reference under a 0.25 GiB cap) is an `oom` row without attempts.

Observation (not a bug): the Triton series' `rel_err` vs FA-2 is ~2x the CUDA series' at every dtype / D
(e.g. 2.8e-4 vs 1.4e-4 fp16) while their fp64 errors are identical; checked on every row of an N=2048 call
(scratch script): fp64 error FA-2 2.53e-4, CUDA kernel 2.52e-4, Triton 2.51e-4, uniform over rows -- the Triton
kernel rounds differently from FA-2 (the CUDA kernel is FA-2's code), so its distance to FA-2 is larger without
being less accurate. `fp64_err` (and its ratio to FA-2's) is the precision metric; `rel_err` is agreement with FA-2.

The part runner was exercised end to end from the frozen snapshot of commit a5da2e8 (`slurm/v3/parts.sh` with
CQSA_SRC / CQSA_V13_SO / CQSA_NATIVE_V2_SO set, CQSA_PARTS="E1:fwd E5", a 200 s part budget, E_ARGS=--smoke ...,
results into a scratch directory): module check and md5sums printed, 9 smoke rows run, report + STATUS written,
`MISSING <n>` last; the rows record the snapshot as code_root, commit a5da2e8 (taken from the snapshot's name:
a git archive has no .git) and the frozen .so files as cuda_module_file / native_module_file.

## Open issues

- The E1/E2 OOM-then-retry branch of the depth rule is untested end to end (see above); the first real job tells.
- The budget model is a guess calibrated on engine_ab's 1M numbers; the measured `row_wall_s` of the same
  configuration overrides it from the second job on, and a wrong estimate only defers rows, never loses them.
- `--mem=80G` (SXM) / `150G` (PCIe) are sized by arithmetic on the host-resident fwd+bwd rows; `vmhwm_gib` in
  the rows is the measurement to tighten them with after the first job (E_sxm's header shows the 40G variant).
- E2's SXM 4M fwd+bwd host-resident rows (~15 min each) and the PCIe 8M rows (~1.4 h each) are the rows the
  28-min / 12-h budgets were set for; a slower node defers them to the next submission rather than cutting them.
- E6's D=128 wave rows at 1M: the wave engine's packed-partials term doubles with D (peak ~1.4 GiB at 32K vs 0.7
  at D=64 in the smoke); within 80 GB at 1M, but worth reading off the rows.
