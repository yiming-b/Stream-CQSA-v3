# Stream-CQSA ultimate A100 program: Phase 0 + Phase 1 report (02)

Written 2026-09-23 by package I0 (integrate Phase 0), branch `ultimate` of the worktree
`/scratch/gpfs/AKEY/yb2807/Stream-CQSA-ultimate`. Plan of record: `docs/ultimate/01_design_a100.md`.
Every number below names its GPU, node and slurm job. Timings are A100-SXM4-80GB unless stated; the one
PCIe row (ncu) is relative metrics only. "FA-2" in Phase 0/1 rows is torch SDPA with the FLASH_ATTENTION
backend forced (flash_attn was not installed then; the package is ~5% faster, design 3.3). flash_attn 2.8.3
is now installed; new harnesses carry both arms.

## 1. Status in one paragraph

Phase 0 (Python only) landed: 128-aligned chunk layouts for every task (P0.1), routing by the memory model
with the wave engine as the device-resident executor (P0.2), an asynchronous host fold for the wave engine's
host accumulator (P0.3). Their adversarial reviews found 1 blocker, 4 majors and 12 minors; I0 fixed the
blocker, every major and the minors (section 4), found and fixed three further bugs on the way (a wrong
result for B>1 host-streamed wave forwards, a host tensor returned to a device caller after eviction, 13
escalation tests broken since P0.1), and added a regression test per finding. Phase 1's experiment build
(v11x) settled the two kernel questions: three fifths of the forward's all-live tax is the per-element
path's *presence* (code size), and the backward's 2x over FA-2 is the CQS code's presence in the fused
kernel, so P2.1 fixes the fused kernel (NO_CQS backward = 0.92-0.95x FA-2). **Full suite (job 14328407,
A100-SXM4-80GB, snapshot ultimate-82236e3): 537 passed, 4 skipped, 12 failed -- all 12 the known bf16 cases on
the fp16-only dev v11 module; nothing else fails** (section 5).

## 2. What landed

| commit | package | files | what |
|---|---|---|---|
| 5fe426a | M0 | `benchmarks/align_census.py` | alignment census (CPU) |
| d4ced17 | M0 | `benchmarks/layout_ab.py` | same-job layout A/B harness (job 14249868) |
| 1dd1506, 62afdfc | M0.3 | `benchmarks/ultimate/ncu_aligned.py`, `slurm/ultimate/ncu_aligned.slurm`, `tests/test_ultimate_ncu_aligned.py` | ncu stall attribution |
| 3d3be92, 50abbfe | M0.1b | `benchmarks/ultimate/kernel_rows.py`, results | kernel rows, wave merge A/B, ncu results |
| 74db835 | P0.3 | `stream_cqsa/native_wave.py`, `tests/test_ultimate_wave_merge.py` | async host fold (`_HostMerge`) |
| 4448052 | design | `docs/ultimate/00_map.md`, `01_design_a100.md` | program design |
| 73dc023 | M5 | `benchmarks/ultimate/dist_telemetry.py`, `slurm/ultimate/dist4.slurm` | 4-GPU telemetry harness |
| fae7d1e | P0.1 | `stream_cqsa/{stable_stream,reference,distributed}.py`, `tests/test_ultimate_align.py`, `benchmarks/ultimate/engine_align_ab.py` | aligned layout everywhere |
| d02e261 | P0.1 | results | engine_align_ab (job 14308145) |
| 03e995c, 472d8e3 | P1.1 | `benchmarks/ultimate/{exp_build_ab,cuobj_report}.py`, results | experiment build A/B, cuobjdump report |
| 837e2d9 | P0.2 | `stream_cqsa/{api,autoconfig,native_wave}.py`, `tests/test_ultimate_routing.py`, CHANGELOG, README | routing by memory model |
| 528162d, 68d0721 | infra | `slurm/ultimate/{dist4.slurm,snapshot.sh}` | jobs run from a frozen `git archive` snapshot |
| 4ae02ea | I0 | `stream_cqsa/{stable_stream,distributed}.py`, `tests/test_ultimate_align.py`, `tests/test_escalation_autograd.py` | P0.1 review fixes |
| edf0fb2 | I0 | `stream_cqsa/{api,autoconfig,native_wave,stable_stream}.py`, `tests/test_ultimate_routing.py` | P0.2 review fixes (blocker + majors) |
| aeb401b | I0 | `stream_cqsa/{native_wave,autoconfig}.py`, `tests/test_ultimate_{wave_merge,routing}.py`, `benchmarks/ultimate/fold_workers_ab.py`, `slurm/ultimate/{fold_workers_ab,test_suite}.slurm` | P0.3 review fixes, B>1 pool bug, fold A/B |
| 867a207 | I0 | `tests/test_devkit_autoconfig.py` | cost-model cache isolation |
| 82236e3 | I0 | `tests/test_bwd_host_stream.py`, `slurm/ultimate/test_suite.slurm` | suite fixes (section 5) |
| 25dfb5e | I0 | `stream_cqsa/native_wave.py`, `tests/test_ultimate_wave_merge.py`, `benchmarks/ultimate/fold_workers_ab.py`, results | fold-worker decision: one worker (3.8) |

Kernel variants (dev tree, not git): `next/kernel/v11` (production), `next/kernel/v11x` (P1.1 experiment
build: v11 + `-DCQSA_NO_ELEMENT_PATH` + `-DCQSA_BWD_NO_CQS`).

## 3. Measurements

### 3.1 Layout A/B (job 14249868, A100-SXM4-80GB, della-l07g7; `results/ultimate/layout_ab.json`)

Forward, causal fp16 H=8 D=64, c=7 itr=1, per-task kernel time, mean over the 7 tasks (ms):

| N | layout | v11 real | v11 zero | SDPA-flash | real/F | zero/F | mixed % |
|---|---|---|---|---|---|---|---|
| 131072 | legacy (align 1) | 25.12 | 23.35 | 18.40 | 1.365 | 1.269 | 0.21 |
| 131072 | aligned 128 | 18.00 | 22.65 | 17.93 | **1.004** | 1.264 | 0.00 |
| 524288 | legacy | 298.26 | 350.39 | 287.88 | 1.036 | 1.217 | 0.05 |
| 524288 | aligned 128 | 270.86 | 345.69 | 286.04 | **0.947** | 1.209 | 0.00 |

Aligned arms: in-place 19.07 / 287.39 ms (+6% / +6%), wave W=1 gathered 18.11 / 272.49, wave W=1 in place
18.66 / 281.20; in-place, wave packed and wave in-place all bit-identical to v11 gathered (maxdiff 0.0).
Backward (legacy path (0,), causal): CUDA bwd real/F 1.93 / 1.91 / 1.89 and zero/F 1.99 / 2.04 / 2.04 at
L = 14044 / 28088 / 56175; dead-tile skipping saves 0.18-7.16 ms, the live path costs 3.2-48.9 ms over F.
Small L: c=73 N=131072 (73 tasks of ~16.2K) sum real 229.2 / zero 188.6 / F 136.4 ms, one wave of 73:
136.3 ms (gathered) / 141.0 (in place), monolithic F 96.3 ms; c=7 N=16384 L=7023 real/F 2.83.

### 3.2 Alignment census (CPU; `results/ultimate/align_census.json`, `benchmarks/align_census.py`)

128x128 tiles, causal, N in {131072, 524288, 1048576}, c in {7, 31, 73, 133}, itr 1-3 (42 configurations):
**mixed tiles 0.000% for every aligned configuration**; legacy layouts 0.05-7.56% mixed (worst c=31 itr=2
at 131K). Live work over the monolithic triangle: aligned 1.0001-1.0010, legacy up to 1.092. Masked tile
fraction matches (l-1)/l^2 per level (22.2% c=7 itr=1, 39.5% itr=2, 53% itr=3). Aligned task lengths
spread by at most a few blocks (e.g. c=133 itr=1 at 131K: 11392-12288).

### 3.3 Kernel rows (job 14292580, A100-SXM4-80GB, della-l08g3; `results/ultimate/kernel_rows.json`)

| row | layout | real/F | zero/F |
|---|---|---|---|
| hdim128 causal v11, L~56K, 7 tasks | aligned | 0.972-1.084 (median task 0.976) | 1.24 |
| hdim128 causal v11 | legacy | 1.151 | 1.324 |
| non-causal hdim64 v9, L~56K | aligned | 1.091-1.108 | 1.37 |
| non-causal hdim64 v9 | legacy | 1.342 | 1.307 |
| bf16 hdim64 causal, variant-dir v11 build (fp16+bf16) | aligned | 1.104 (fp16 same binary 0.999) | 1.258 |
| ragged N=100003, sum of 7 tasks | aligned / legacy | 1.100 / 1.413 | 1.277 / 1.285 |
| itr=2 N=131072, sum of 49 tasks | aligned / legacy | 0.861 / 1.681 | 1.289 / 1.340 |

bf16 on the recipe module (`next/kernel/stable/cqsa_cuda_next_v11`) is NOT RUN: that build is fp16-only.
Patterns on one aligned task (L=56320): real 1.000, zero 1.263, allones 0.139, random3bit 8.78 (100% mixed:
the per-element path). One wave of 49 at itr=2: 139.0 ms packed / 143.3 in place (bit-identical to v11 per
task) against monolithic F 96.0 ms. Ragged tasks with L%128=35 run 1.30x F (the !Is_even_MN tail).

### 3.4 ncu stall attribution (job 14292338, A100-80GB-**PCIe**, della-l02g14, base clocks; `results/ultimate/ncu_aligned.md`)

Aligned c=7 task L=56064, causal D=64, one launch per arm. Relative metrics only:

| metric | FA-2 | v11 plain | v11 zero | v11 real |
|---|---|---|---|---|
| executed instructions (G) | 5.06 | 3.66 | 4.13 | 3.29 |
| warp cycles / issued instruction | 4.03 | 5.35 | 6.23 | 6.26 |
| tensor pipe active % | 61.0 | 63.0 | 48.1 | 46.8 |
| int instructions % | 3.6 | 5.8 | 12.5 | 13.4 |
| static SASS instructions | 4400 | 4032 | 72776 | 72776 |

Verdict (derived from the tables; the file's own verdict section was left unwritten): zero vs plain is +13%
instructions and +16% cycles per instruction; the added stalls are short/long scoreboard (0.24 -> 1.14
cycles/inst) and no_instruction (0.04 -> 0.21), i.e. the in-loop verdict loads and the integer/branch
bookkeeping of the CQS live loop, with an instruction cache under pressure from an 18x larger kernel. It is
not tensor-pipe bound (48% vs 63%). Phase 1 (3.7) split this tax between code size and the loop itself.

### 3.5 Engine alignment A/B (job 14308145, A100-SXM4-80GB, della-l07g3; `results/ultimate/engine_align_ab.json`)

Classic engine end to end, forward causal fp16 H=8 D=64, device-resident, n_par=2 (mean of 3, ms):

| N, c | legacy | aligned | wave | SDPA-flash | aligned/legacy |
|---|---|---|---|---|---|
| 131072, c=7 | 149.9 | 135.1 | 139.7 | 95.8 | 0.901 |
| 131072, c=73 | 222.2 | 179.4 | 162.8 | 96.3 | 0.808 |
| 1048576, c=7 | 7759.0 | 7576.0 | 7958.7 | 6273.6 | 0.976 |

Arm check at 131K: aligned vs legacy rel 2.1e-5, wave vs aligned 2.8e-8, both 2.2e-4 vs SDPA.

### 3.6 Async wave host fold A/B (job 14292616, A100-SXM4-80GB, della-l08g3; `results/ultimate/wave_merge_ab.json`)

Forward acc=CPU, c=7 itr=1 (s, mean of 3): host-streamed 1M sync 9.87 / async 9.03 / classic 9.32; 2M 35.74
/ 34.22 / 34.63 (async/classic 0.970 / 0.988: the P0.3 acceptance holds). Device-resident 1M 9.61 / 8.75 /
8.59, 2M 35.24 / 33.54 / 32.20 (async 2-4% slower than classic, but the planner never enumerates acc=CPU
with device-resident inputs). Sync == async bit-identical; classic vs wave rel 2.8e-8. Host max RSS of the
async 2M arm 29.2 GiB. Provenance caveat (review minor): the log records HEAD 3d3be92, i.e. the measured code
was the uncommitted worktree later committed as 74db835; the fold A/B of 3.8 re-measures from a snapshot (wave_w1 = async, wave_sync = sync; the same 1M/2M c=7 itr=1 cells agree within 2.5%).

### 3.7 Phase 1 experiment build (v11x; `results/ultimate/exp_build_ab.md`, `cuobjdump_v11x.md`)

Build (della-vis1, nvcc 13.0, `MAX_JOBS=8 NVCC_THREADS=4`, watchdog 45 min never fired): **343 s total vs
2410 s for v11's identical kernel set**. ptxas per TU 23-75 s (largest instantiation 3.4 s); cicc
(123-250 s) is now the dominant phase:

| TU | v11 s | v11x s | v11x cicc / ptxas s |
|---|---|---|---|
| fwd hdim64 non-causal | 2406 | 271 | 196 / 61.5 |
| fwd hdim64 causal | 1783 | 169 | 123 / 33.7 |
| fwd hdim128 non-causal | 1208 | 340 | 250 / 75.1 |
| fwd hdim128 causal | 1026 | 236 | 177 / 46.5 |
| bwd hdim64 | 558 | 274 | 237 / 24.0 |
| bwd hdim128 | 529 | 276 | 241 / 22.7 |

SASS: CQS-on forward instantiations 69.6-74.3K instructions (v11) -> 5.4-7.7K (v11x), 1.3-1.5x their CQS-off
twins and torch FA-2 (3.9-6.9K); REG 255 everywhere; v11's non-causal CQS-on instantiation has STACK 640 B
(the mis-compile signature), v11x's 96 B. Backward hdim64 128x128 causal even_MN: v11 5624 SASS / STACK 224 B,
v11x NO_CQS 2152 / 112 B, torch FA-2 2336 / 96 B.

Same-job A/B (job **14308401**, A100-SXM4-80GB, della-l07g4), aligned c=7 task L=56064, ms (ratio to SDPA-flash):

| config | SDPA | v11 plain | v11 zero | v11 real | v11x plain | v11x zero |
|---|---|---|---|---|---|---|
| causal D=64 | 19.70 | 17.30 (0.878) | 22.55 (1.144) | 17.85 (0.906) | 17.38 (0.882) | 19.35 (0.982) |
| non-causal D=64 | 34.17 | 33.16 (0.970) | 195.82 (5.731) | 151.17 (4.424) | 33.51 (0.981) | 34.73 (1.016) |
| causal D=128 | 32.38 | 29.71 (0.917) | 40.52 (1.251) | 31.84 (0.983) | 29.82 (0.921) | 34.08 (1.053) |
| non-causal D=128 | 59.71 | 56.66 (0.949) | 248.16 (4.156) | 194.37 (3.255) | 57.26 (0.959) | 64.34 (1.078) |

**Forward tax split** (all-live tax = zero - plain): causal D=64 5.25 ms = 3.20 ms (61%) presence of the
per-element path + 1.97 ms (39%) the live loop's own verdict loads, next-live search and flags (v11x
zero/plain 1.113); D=128 60/40 (v11x zero/plain 1.143). Non-causal: v11's CQS-on instantiation is
mis-compiled purely from code size (5.9x plain); v11x non-causal zero is 1.036x plain, 1.016x SDPA and 1.36x
faster than the v9 kernel production uses (47.28 ms). v11x plain/zero bit-identical to v11 in all four configs.

**Backward route** (classic layout, causal D=64, ratio to SDPA-flash backward):

| L | v11 plain | v11 zero | v11 real | v11x NO_CQS |
|---|---|---|---|---|
| 14044 | 1.840 | 2.015 | 1.942 | 0.923 |
| 28088 | 1.874 | 2.051 | 1.914 | 0.940 |
| 56175 | 1.863 | 2.044 | 1.889 | 0.947 |

At 56175, zero/F = 0.947 (upstream loop with our contract) x 1.967 (presence of the CQS code, switched off at
runtime) x 1.097 (verdicts on an all-clear pattern); skipping dead tiles nets -7.6%. **Route decided: fix the
fused kernel** (P2.1); the Triton two-kernel port is not taken.

### 3.8 Fold workers A/B (I0; job 14313570, A100-SXM4-80GB, della-l09g2, snapshot ultimate-aeb401b; `results/ultimate/fold_workers_ab.json`)

Forward, acc=CPU, host-resident (streamed) Q/K/V, fp16 B=1 H=8 D=64 causal, CQSA_CPU_THREADS=8, 16 cores.
Mean of 2 timed runs after 1 warm-up, each (N, shape, arm) in its own subprocess, arms interleaved per shape.
Arms: `wave_sync` (fold on the main thread), `wave_w1` (one async worker, the default), `wave_w2` (two
workers, each owning half of the token ids), `wave_w1_2t` (one worker, 16 CPU threads), `classic` (the
planner's classic host pipeline, n_par=2). Seconds (ratio to wave_w1):

| N | c, itr | waves | wave_sync | wave_w1 | wave_w2 | wave_w1_2t | classic | w1 fold / wait s |
|---|---|---|---|---|---|---|---|---|
| 1M | 7, 1 | 4 | 9.63 (1.082) | **8.90** | 9.02 (1.013) | 8.94 (1.004) | 9.22 (1.036) | 1.05 / 1.33 |
| 1M | 73, 1 | 10 | 11.71 (1.211) | **9.67** | 10.36 (1.071) | 10.07 (1.042) | 11.40 (1.179) | 2.54 / 0.27 |
| 1M | 7, 2 | 10 | 11.04 (1.174) | **9.40** | 9.59 (1.020) | 9.46 (1.006) | 10.32 (1.098) | 1.96 / 0.94 |
| 2M | 7, 1 | 4 | 35.24 (1.043) | **33.78** | 33.97 (1.006) | 33.91 (1.004) | 34.23 (1.013) | 2.03 / 4.94 |
| 2M | 73, 1 | 10 | 38.90 (1.127) | **34.51** | 35.67 (1.034) | 35.34 (1.024) | 38.70 (1.121) | 5.20 / 0.61 |
| 2M | 7, 2 | 10 | 37.83 (1.093) | **34.62** | 34.96 (1.010) | 34.73 (1.003) | 36.03 (1.041) | 3.88 / 3.33 |

Arm check at N=131072: every wave arm bit-identical to wave_sync; classic vs wave rel 2.8e-8 / 5.3e-8 / 4.4e-8.
**Verdict: one worker, fastest in all 6 cells.** The main thread's wait for a slot (`merge_wait_s`) is the
same for one and two workers within 0.05 s in every cell (e.g. 2M c=7: 4.94 / 4.95 s): a fold's future also covers the
wait for its wave's D2H event, so where the wait is large it is the GPU wave, not the fold, and where the
fold is largest against the kernel (c=73) the wait is already 0.27-0.61 s. Splitting the fold roughly
doubles its summed CPU time (1M c=73: 2.54 -> 4.63 s; per-worker searchsorted + slicing and two OpenMP
teams on 16 cores) and costs 0.6-7.1% wall time; doubling one worker's threads costs 0.3-4.2%. The
multi-worker option was removed (25dfb5e); `merge_fold_s`/`merge_wait_s` stay in `info`. Against the
alternatives the async fold holds: sync is 4.3-21.1% slower and the classic host pipeline 1.3-17.9% slower
(largest at c=73, where the classic engine pays one launch + merge per 16K-token subproblem). Peak device
6.2-6.5 GiB (1M) / 12.4-13.1 GiB (2M) for every wave arm; host max RSS of the async 2M arms 28.3-29.4 GiB
(classic 17.9-28.5 GiB).

### 3.9 Multi-device telemetry (job 14312486, 4 x A100-PCIe-80GB; `results/ultimate/dist_telemetry.json`)

**PENDING at the time of writing.** Job 14312486 (`slurm/ultimate/dist4.slurm`, 4 x A100-PCIe-80GB,
gpu-short, 75G host, harness `benchmarks/ultimate/dist_telemetry.py` from 73dc023) was submitted
2026-09-23 10:18 and is still queued (Priority; slurm's estimate at 16:00 was 2026-09-23 23:57 on
della-l02g4). It records, per rank, the GPU name, local compute and merge time, the imbalance max/mean of the
round-robin shard, the (path_idx, L) of every task and device/host peaks, for seg_align 0 and 128 (and
aligned + shared chunks at itr=1). It is the baseline of the scheduler package (D1, which now owns
`stream_cqsa/distributed.py` and `autoconfig.py`); its results go to `results/ultimate/dist_telemetry.json`
and are reported there, not here.

## 4. Review findings and their resolution

Source: workflow wf_bfb744ca-a43 (adversarial reviews of P0.1, P0.2, P0.3). Each fix has a regression test
that fails on the pre-fix code.

| # | sev | finding | resolution | test |
|---|---|---|---|---|
| P0.2-1 | BLOCKER | host-resident inputs: planner enumerated device-resident wave candidates; itr=1 autograd planned "wave W=32,768 gathered" and `.backward()` crashed (device mismatch; reproduced on vis1) | `plan(inputs_on_host=True)` enumerates only host-streaming decompositions (monolithic still copies to the device); `attention()` passes it; a host-resident or evicted backward is the classic engine's whatever `kernel=` says (design 5.2/5.4) (edf0fb2) | `test_host_resident_inputs_plan_only_streaming_candidates`, `test_host_resident_backward_under_auto_is_classic_and_exact` |
| P0.1-1 | MAJOR | `max_depth_for(align=128)` one level stricter than `refine_task`'s guard: OOM recovery refused refinable tasks (N=131072 c=73 itr=1) | cap counts the level whose parent holds c*128 tokens; aligned recovery refuses by the exact per-task token guard (the mean-length cap is still short where aligned tasks round up: N=2048 c=7 has 1024-token itr=1 tasks vs mean 877); cap stays exact on the legacy layout (4ae02ea). Design 5.1's `depth <= floor(log_{c/l}(N/(c*128)))` should read `+ 1` | `test_max_depth_under_alignment` (131072/7 -> 6), `test_depth_cap_admits_every_refinement_the_guard_admits`, `test_forward_recovers_an_oom_one_level_below_the_old_cap`; `test_device_backward_escalates_and_stays_exact` now passes aligned |
| P0.2-2 | MAJOR | nothing-fits fallback: `min()` of an empty list for `engines=('wave',)`, `direction='bwd'` | falls back to the most frugal candidate the restriction allows; a clear `ValueError` when none can run (edf0fb2) | `test_nothing_fits_fallback_under_an_engine_restriction` |
| P0.2-3 | MAJOR | OOM inside the wave backward (in `loss.backward()`) not recovered | `wave_backward_recovering`: halve the backward wave budget to one subproblem per wave, then (kernel="auto") the classic streamed backward from the saved out/lse with host gradients, escalating depth itself; forced wave kernels re-raise; `info["backward"]` records engine, retries, budget (edf0fb2) | `test_oom_in_the_wave_backward_is_recovered` |
| P0.3-1 | MAJOR | single fold worker serialises; c=73 / itr=2 never measured | measured (3.8, job 14313570): token-range workers (aeb401b, opt-in) were 0.6-7.1% slower than one worker in all 6 cells, with the same slot wait; the option was removed (25dfb5e), one worker stays, `merge_fold_s`/`merge_wait_s` telemetry kept | `test_async_fold_is_bit_identical_and_reports_telemetry` (N=131072 and ragged 100003) |
| wart | - | `_restore_inputs` reserved 10% of the physical device: never restored under a memory fraction or `hardware=` budget | room = effective free - 10% of the effective capacity (`effective_total_bytes`), within what a `hardware=` budget leaves after the output (edf0fb2) | `test_inputs_restored_under_a_memory_fraction` |
| doc | - | monolithic estimate always fits before the wave floor | confirmed intended: every device-resident decomposition needs more device memory than the monolithic call (checked over N x budget x direction), so under `kernel="auto"` with nothing pinned a device-resident wave runs only after a monolithic OOM or a pinned itr; documented in `attention()` | `test_monolithic_fits_whenever_a_device_resident_wave_does` |
| P0.1-2 | minor | engine rel 1e-4 does not hold at tiny N / deep itr | invariant stated as enforced: rel 1e-4 for N >= 64K; everywhere fp64 error <= 1.2x FA-2 and <= 1.2x legacy | `test_engine_forward_small_n_deep_itr_is_judged_against_fp64` (N=2693 itr=3, N=500 itr=2) |
| P0.1-3 | minor | `seg_align=None` = legacy at the engine level, TypeError in distributed | None = default at every level; distributed passes it through; forward info records `task_seg_align` | `test_distributed_pair_takes_one_kwargs_dict` |
| P0.1-4 | minor (pre-existing) | `dist_stream_cqsa_backward` forwarded `segmented`/`shared_chunks` to the backward | forward-only keys dropped | same (world=1 NCCL, bit-identical to single device) |
| P0.2-4 | minor | monolithic-OOM fallback hand-set classic itr=1 | re-plan with `allow_mono=False` (the wave runs) | `test_monolithic_oom_replans_among_decompositions` |
| P0.2-5 | minor | pinned `itr=` re-plan chose its own depth; plan_only/W described another call | `plan(min_itr=, max_itr=)` plans the pinned depth | `test_pinned_depth_is_what_plan_only_reports_...` |
| P0.2-6 | minor | forward OOM ladder assumed the running budget was N | starts from `default_wave_tokens(N)`; the backward ladder computes the budget it runs (`bwd_wave_budget`) | covered by the OOM tests |
| P0.2-7 | minor | frontier tie rule never yields "one wave of 7" | accepted: the rule picks the smallest rung within 2% of the fastest (W=262,144, 2 waves at 131K, default model), ~1 ms at 131K (kernel_rows: 0.33 ms/launch); the P0.2 acceptance text "W=7" is superseded | existing ladder tests |
| P0.2-8 | minor | a ragged causal backward reported `gather=False` | reports the forward's actual gather | `test_pinned_depth_is_what_plan_only_reports_and_the_backward_reports_the_gather_that_ran` |
| P0.3-2 | minor | fold tests aligned N and B=1 only | ragged N=100003, B=2, Triton kernel cases added -- **the B=2 case found a real bug** (below) | `test_async_fold_batch_of_two_matches_per_batch_calls`, `test_async_fold_on_the_triton_kernel` |
| P0.3-3 | minor | slots via `.pin_memory()` (pageable intermediate) | allocated with `pin_memory=True` | - |
| P0.3-4 | minor | wave_merge_ab provenance (dirty worktree) | every I0 job runs from a `git archive` snapshot and records its code root | - |
| P0.3-5 | minor | pinned slots not in the host memory model | planner's host estimate for wave acc=cpu adds 2 x min(W, N) x (4HD + 8H + 8) B | `test_host_resident_forward_is_decided_by_memory` |

Further bugs found by I0 and fixed:
- **ChunkPool reused chunks across batches** (aeb401b): a host-streamed wave forward with B>1 returned batch 0's
  rows for batch b>0 (rel error 1.41 vs SDPA on batch 1, any fold mode); the pool now reloads per batch. The
  classic host pipeline was correct.
- **Evicted inputs + monolithic plan returned a host tensor** to a device caller (edf0fb2): the output now goes
  to the caller's device.
- **13 tests in `tests/test_escalation_autograd.py` failed since fae7d1e**: tiny-N (343, 169, 2401) refinement
  mechanics under the aligned default, which does not refine below c*128 tokens by design; they run on the
  legacy layout whose sizes they quote (4ae02ea); the two backward-escalation tests at N=2048 pass on the
  aligned layout after the P0.1-1 fix.
- `tests/test_devkit_autoconfig.py::test_planner_uses_devices_when_faster` read whatever calibrated cost model
  sat in `~/.cache/stream_cqsa` (fails at 68d0721 on vis1); the cache is isolated (867a207).

## 5. Test suite

`slurm/ultimate/test_suite.slurm`: the whole `tests/` directory in one process on one A100-SXM4-80GB
(gpu-test, 48G host), code from a `git archive` snapshot, production kernels `cqsa_cuda_next_v11` / `_v9` from
`next/kernel/stable` (fp16-only), native wave module from `next/native`, and (since 82236e3) the kernel-variant
modules from a frozen copy `next/kernel/frozen_suite/2026-09-23` (v11x; v12f b2; v12b b3; md5 printed in the log).

| job | snapshot | node | passed | skipped | failed | time |
|---|---|---|---|---|---|---|
| 14327651 | ultimate-25dfb5e | della-l08g7 | 439 | 101 | 13 | 222 s |
| **14328407** | **ultimate-82236e3** | della-l07g6 | **537** | **4** | **12** | 182 s |

**Known environmental failures (12, both runs)** -- `dtype1` = bfloat16, every one `RuntimeError: this CQSA
build only supports fp16` raised by the dev v11 module:
- `tests/test_backward_default.py::test_default_backward_matches_sdpa[{True,False}-{64,128}-dtype1]` (4)
- `tests/test_bf16.py::test_dtype_fwd_bwd[{True,False}-{64,128}-dtype1]` (4)
- `tests/test_bwd_host_stream.py::test_host_matches_device[{True,False}-{1,2}-dtype1]` (4)

**Skips in 14328407 (4)**: `tests/test_wave_shapes.py` hdim128 fp16/bf16 x 2 -- the `next/native` build has no
head_dim=128 instantiation (build scope, not a failure).

**Fixed between the two runs** (82236e3):
- `test_bwd_host_stream.py::test_host_uses_less_device_memory` failed in 14327651 only (host 26.4 MB vs device
  49.6 MB, bound 24.8 MB). It compared absolute `max_memory_allocated()`: in suite order ~21 MB of device tensors
  kept alive by the earlier bf16 failures' tracebacks were charged to both paths (alone: 5.5 vs 28.7 MB).
  Reproduced on vis1 with the suite's order; both peaks are now counted above the allocation at test start.
  Not a regression of the streamed backward.
- The 101 skips of 14327651 were `test_ultimate_v11x/v12b/v12f.py` (modules not on the job's PYTHONPATH);
  14328407 runs them all against the frozen copies (they also pass on vis1: 99 passed).

Local checks on vis1 (shared A100-PCIe-40GB, ~6 GiB free) of the tests I0 changed: `test_ultimate_wave_merge`,
`_routing`, `_align`, `test_escalation_autograd`: 128 passed; 3 failures all environmental and all passing in
14328407 (2 x fp64 SDPA reference OOM in `test_async_fold_accuracy_vs_fp64`; one run of
`test_auto_routes_to_the_wave_and_matches_the_forced_paths` where another process moved the free memory
between `plan_only` and the call, see section 6; it passed on re-run).

## 6. Open issues

- **4-GPU telemetry not yet run** (3.9, job 14312486): the distributed rows of this report are missing until
  it runs; D1 owns the follow-up.
- **Design 5.1 depth-cap formula** should read `floor(log_{c/l}(N/(c*128))) + 1` (P0.1-1); the code is right,
  the design text is not yet edited (01_design_a100.md is the plan of record, owned by the orchestrator).
- **Frontier tie rule** (P0.2-7) accepted as is: the P0.2 acceptance text "one wave of 7" is superseded by the
  smallest-rung-within-2% rule; no code change.
- **Runtime clamp of the planned wave budget**: `wave_forward` caps the plan's `wave_tokens` by the free device
  memory at run time (`_budget_tokens`), so on a shared GPU the executed budget can be smaller than
  `plan_only` reports (seen once on vis1: planned 262,144, ran 131,072; passes on a dedicated GPU). Intended
  (safety), but `test_auto_routes_to_the_wave_and_matches_the_forced_paths` is sensitive to it on shared GPUs.
- **bf16 on the recipe module**: the dev v11 module (`next/kernel/stable/cqsa_cuda_next_v11`) is fp16-only, so
  bf16 production paths are only exercised with the variant-dir build (3.3) and fail in the suite (section 5).
- **v12f / v12b are single-direction builds** (each with the other direction disabled): the production module
  that merges them (and replaces v11/v9) is the next integration step; the Phase 0 engine code is ready for it
  (aligned layouts, routing, wave engine), but nothing in `stream_cqsa/` imports v12* yet.

## 7. Pending jobs

| job | what | partition / GPU | state at writing |
|---|---|---|---|
| 14312486 | 4-GPU classic distributed telemetry (3.9) | gpu-short, 4 x A100-PCIe-80GB | PENDING (Priority) |
| 14328407 | full suite (section 5) | gpu-test, A100-SXM4-80GB | COMPLETED (12 known bf16 failures) |
