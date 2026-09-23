# P2 progress log: backward kernel v12b (ALIGNED run-table backward + LEGACY v11 backward)

Package goal (task text, 2026-09-23): a CQS backward that on 128-aligned layouts runs at upstream speed on
live tiles and skips dead tiles for free, exact everywhere. Variant tree:
`/scratch/gpfs/AKEY/yb2807/Stream-CQSA-dev/next/kernel/v12b` (not git; module `cqsa_cuda_next_v12b`, cloned from
v11x, forward keeps `-DCQSA_NO_ELEMENT_PATH`, backward WITHOUT `CQSA_BWD_NO_CQS`); diff against `next/kernel/v11`
committed as `docs/ultimate/kernel_diffs/v12b.diff`.

## Relayed status question (2026-09-23 13:12, answered here, package continues)

The user asked (relayed) to check the slurm jobs and report v2.2.3 and ultimate-kernel progress. squeue at
13:12: only 14312486 (cqsa_dist4_tele, pending Priority). sacct today: v2.2.3 side (main's unreleased
24a4ff7/5ba4c90/c4bc39b; the 12 formerly held jobs) -- paper mids 13908223/13908224/13915115/13915470 COMPLETED
(08:31-12:47), OOC forward 13952419/21/23/25 COMPLETED, OOC backward 13952418/20/22/24 OUT_OF_MEMORY (host
OOM, the known 4M-bwd-at-96G issue; the OOC harness fix is another package's). Ultimate program: P1.1 done
(v11x, job 14308401), I0 fixes landed (4ae02ea, edf0fb2, aeb401b) with fold_workers_ab 14313570 COMPLETED,
flash-attn 2.8.3 built (14240707), P3 forward v12f built (1926 s, 11:10) and its fwd_ab job not yet submitted,
P2 (this package) started 13:12.

## State (update at every milestone)

| step | status | notes |
|---|---|---|
| read design 1/3/4/7, exp_build_ab.md, cuobjdump_v11x.md, P3 log + census, v11/upstream bwd | done 13:10 | |
| clone v11x -> v12b, code written | done 13:33 | files below; standalone compile of the hdim64 main unit rc=0 (ALIGNED 128x128: REG 255, STACK 48-72 B vs v11x NO_CQS 112-128 B) |
| build b1 (clean, detached, vis1) | done 14:02, rc=0, 1697 s | log `next/logs/06_build_v12b_b1.log`, csv `next/logs/build_v12b_b1_nvcc_time.csv` (vis1 shared with the v12f b2 build and a v2 wheel build: LEGACY bwd units 1480-1530 s, main bwd units ~600 s, ptxas <= 100 s per unit); frozen copy `next/kernel/v12b/frozen/b1/` (md5 1bd3f31f...) |
| harness / test / slurm written | done | `benchmarks/ultimate/bwd_ab.py`, `tests/test_ultimate_v12b.py`, `slurm/ultimate/bwd_ab.slurm`; CPU census-reference test passes |
| tests/test_ultimate_v12b.py on vis1 | done 14:06: 44 passed | first build, no fix needed; profiler confirms the ALIGNED kernel does the work (L=28032: census 8 us, LEGACY launch 9 us, ALIGNED 31.9 ms vs v11 65.3 ms on the shared PCIe GPU) |
| bwd_ab --smoke on vis1 | done | every arm and check runs; timings on the shared, 100%-busy PCIe GPU are noise (not used) |
| static report | done | `results/ultimate/v12b_static.md`: ALIGNED hdim64 128x128 causal even: 255/72 B/2176 SASS vs v11 255/224/5624, v11x NO_CQS 255/112/2152, torch-FA2 255/96/2336 |
| SXM job bwd_ab (b1) | done 14:11, 57 s | job 14322302, snapshot `ultimate-50e64d2`, frozen/b1 -> `results/ultimate/bwd_ab_b1.{json,md}`. dk/dv torch.equal to v11 in every row; real 0.850x flash_attn at 56K causal D64 (gate 0.85: borderline), 0.813 D128, 0.842 non-causal; c=73 0.956x flash_attn loop, 2.18x faster than v11; BUT v12b zero / v11x NO_CQS = 0.989 / 1.076 / 1.075 (14K/28K/56K causal D64), 1.029 D128, 1.073 non-causal: the ALIGNED loop itself (plain) is 1.03-1.045x v11x's |
| diagnosis b1 | done 14:15 | SASS (hdim64 128x128 causal even): same loop shape, fewer instructions than v11x (1235 vs 1306), but ptxas sank the four dP_sum global loads (issued after the top barrier in upstream's source) to ~27 instructions before their use in dS (v11x: ~70, behind the dP GEMM): global-load latency exposed on every tile |
| b2: dP_sum loaded before the top barrier; next-m computed at the Q prefetch; run slot + sQ buffer in one register | built 14:30 (836 s: a new csv path changed the nvcc command line, so ninja rebuilt every unit), 44 tests pass, frozen/b2 | SASS: loads now 115 HMMAs ahead of use, but 17 STL + 17 LDL per tile (4 more live registers across the S GEMM) |
| SXM job bwd_ab (b2) | done 14:3x | job 14323074, snapshot `ultimate-d5c7f0e`, frozen/b2 -> `results/ultimate/bwd_ab_b2.{json,md}`, `bwd_ab_b2_14323074.out`. Drift check: every arm re-timed at the end within 0.1% except the job's very first timed arm (v11x at 14K: 3.14 first, 2.85 last = GPU clock ramp; b1's 14K ratios are flattered by it). b2 vs v11x (last): plain 1.028-1.051, zero 1.032-1.071; real 0.839x flash_attn (56K causal D64), 0.754x SDPA non-causal; c=73 0.942x flash_attn loop: b2 did not close the gap |
| b3a: dP_sum staged in smem (hand-specialised body) | built 14:35 (249 s), NOT measured | superseded by b3 before its job; frozen/b3a keeps .so + header |
| b3 (final kernel iteration): the ALIGNED function = the TEXT of v11x's compute_dq_dk_dv_1colblock (CQSA_BWD_NO_CQS side), + Use_runs changes folded away by `if constexpr` when off, + Dpsum_smem template (env CQSA_V12B_DPSUM_SMEM, default 0 = upstream global reads) | built 14:44 (235 s, main units only), 44 tests pass in both dP_sum modes, frozen/b3 (md5 942f97c3...) | SASS: even the text-identical CQS-off instantiation schedules the dP_sum LDGs 1-2 instructions before use (v11x: ~70), so the TU context, not the source, decides it; Dpsum_smem removes the LDGs from the loop. One job measures both dP_sum sources |
| SXM job bwd_ab (b3) | done 14:47, 58 s | job 14323878 (della-l07g5), snapshot `ultimate-35d5562`, frozen/b3 -> `results/ultimate/bwd_ab.{json,md}` (canonical; copy `bwd_ab_b3.*`), `bwd_ab_b3_14323878.out`. ALL TARGETS MET (below) |
| b4: host-side default of the dP_sum source per path (CQS-on smem, CQS-off global) from job 14323878 | built 14:51 (207 s), cuobjdump -sass of every ALIGNED / LEGACY / census kernel identical to b3, 48 tests pass (both sources), frozen/b4 (md5 fc6fde7b...) | no new measurement needed: the b4 default runs exactly the kernels measured as `v12b_s` / `v12b_plain` |
| results, diff, static report, commit | done 14:5x | `docs/ultimate/kernel_diffs/v12b.diff` (vs v11), `results/ultimate/v12b_static.md` (b4) |

## Verdict (job 14323878, A100-SXM4-80GB; b4 default = the `v12b_s` rows for CQS-on, `v12b_plain` for CQS-off)

| target | measured | pass |
|---|---|---|
| zero <= 1.05x v11x NO_CQS (causal D64 14K/28K/56K; D128 56K; non-causal D64 56K) | 0.996 / 0.992 / 0.990; 0.974; 0.993 | yes |
| real <= 0.85x flash_attn bwd, causal D64 56K | 0.783 (34.41 vs 43.95 ms) | yes |
| design P2.1: real <= 1.30x SDPA causal 56K, non-causal <= 1.10x | 0.722, 0.726 | yes |
| design P2.1: c=73 N=131072 <= 1.45x SDPA | 0.795x SDPA per-task sum; 0.885x flash_attn over the same 73 calls; 1.064x one monolithic flash_attn bwd at N | yes |
| CQS-off at upstream speed | 0.990-1.001x v11x, 0.983-0.999x flash_attn (L >= 28K) | yes |
| dk/dv torch.equal v11; dq drift | every row / every c=73 task; dq max 1.2e-4 (= v11's own rerun drift) | yes |
| float64 within 1.2x FA-2's error (+1e-4) | tests/test_ultimate_v12b.py::test_float64_error_within_fa2 (vis1, c=7 N=8192, zero + real, causal + non-causal) | yes |
| LEGACY bit-identical to v11 | forced LEGACY dk/dv equal in every row, times within 0.1% of v11; random/half/unaligned route to LEGACY (tests) | yes |

vs v11 (today's CQS backward): 2.48x faster real causal D64 56K, 2.02x D128, 1.94x non-causal, 2.35x c=73.
Static (b4): ALIGNED runs+smem hdim64 128x128 causal even 255 REG / 104 B STACK / 2320 SASS vs v11 224 B / 5624,
v11x 112 B / 2152, torch-FA2 96 B / 2336 (in upstream's 96-144 B band). deterministic=True (item 3): 66.60 vs 34.40 ms
at L=56064 H=8 b=1 (1.94x, reruns torch.equal) -- W*H ~ k x 108 caveat confirmed; exposed through `bwd`'s existing
`deterministic` argument (the interface still passes False).

Lessons: (1) b1/b2 were hand-specialised copies of upstream's loop; the same algorithm with fewer instructions ran
3-5% slower because ptxas scheduled it differently (dP_sum loads next to their use, spills). Deriving the ALIGNED
function from the exact text of v11x's loop (`if constexpr` / constant `m_step` for everything new) gave upstream
speed. (2) Even the text-identical CQS-off instantiation schedules the dP_sum loads late in this TU, yet measures at
v11x speed: the static load-use distance was not the cause. (3) A new nvcc `-time` csv path changes the command line
and makes ninja rebuild every unit: build.sh now takes CSV_TAG. (4) Never edit a build script while a detached build
runs it (bash reads the file by offset); the b2-era edit was reverted before its `wait` returned.

Open (not in this package): engine integration (interface.py backward default is Triton; `bwd_wave` in native/ still
has v11's loop), in-place wave backward, engine-level acceptance (1M itr=1 backward <= 24 s, wave 131K <= 450 ms /
3 GiB), bf16 / `common` build, P3 merge (both censuses in one binary), exposing `deterministic` in interface.py.

## Design as built (refinements of the task's sketch)

- **Census kernel** (`src/cqs_census_bwd.{h,cu}`, own TU): v12f's census transposed. One warp per COLUMN block of the
  backward's tile (A100: hdim64 128x128, hdim128 64x128 = `run_mha_bwd_hdim64/128`'s >= 144 KB choice; the census
  is launched from the templated launcher with `Kernel_traits::kBlockM/kBlockN`, so it cannot disagree with the
  kernel). Lanes classify the m tiles m in [m_min, m_max) downward (m_min = n*kBlockN/kBlockM under causal, the
  kernel's m_block_min); live iff row OR & col OR == 0, dead iff row AND & col AND != 0, else mixed; the ragged
  last 64-token summary is recomputed from the group bits (exact verdict for the ragged tile). Writes
  `route` (0 ALIGNED; bit 1 mixed, bit 2 run overflow) and per column up to 15 runs `{hi, lo}` of live m blocks
  in descending m + `{-1,-1}` (`kCqsBwdRunSlots = 16`, O(column blocks) memory). Constants carry a `Bwd` infix so
  the eventual merge with v12f's `cqs_census.h` does not collide.
- **Routing without a host sync**: `mha_bwd` allocates the census (int32) only for eligible calls (group bits +
  summaries, not numden, no dropout/alibi/softcap/local, head 64/128, not forced LEGACY: `cqs_bwd_flags & 1`).
  `run_mha_bwd_` (main TU) = preprocess -> [census -> ALIGNED(Use_runs) -> LEGACY] -> convert_dQ on one stream;
  ALIGNED CTAs return if route != 0, LEGACY CTAs return if route == 0 (gate in the LEGACY `__global__` wrapper,
  the only change to v11's kernel code path). Ineligible CQS calls: LEGACY alone (route null). CQS-off calls:
  ALIGNED<Use_runs=false> (upstream's loop, no table) when eligible, else LEGACY.
- **ALIGNED kernel** (`src/flash_bwd_aligned.h`; b3/b4: the TEXT of v11x's `compute_dq_dk_dv_1colblock` with the
  CQSA_BWD_NO_CQS sides selected, called with Is_first = Is_last = false, Seq_parallel = true, no dropout/local/
  alibi/softcap, Is_even_K, plus template flags Use_runs and Dpsum_smem whose changes fold away when off; b1/b2 were a
  hand-specialised copy, same algorithm): upstream FA-2's seqk-parallel loop; the
  m loop walks the runs: `m_next = m > run_lo ? m - 1 : next run's hi` (the table is re-read only at a run
  boundary, index in a register, base in constant memory); Q / dO / LSE / dPsum / dQaccum pointers jump by
  `m_step = m - m_next` rows (64-bit arithmetic); the sQ double buffer toggles per VISITED tile (`sQ_cur`), not by
  m parity; offsets are taken at the first visited m. Masking = upstream's `apply_mask` (ragged column) /
  `apply_mask_causal` (diagonal + ragged), the same -inf positions v11's `Mask<>` writes. dS = p * (dP - dPsum)
  (the numden runtime branch is not in this instantiation). A column with no live tile writes dK = dV = 0.
  Template: `<Kernel_traits, Is_causal, Is_even_MN, Use_runs>` (8 instantiations per traits).
- **TU split**: `flash_bwd_hdim*_{fp16,bf16}_sm80.cu` = the main path (`run_mha_bwd_`, includes
  `flash_bwd_aligned.h`); `flash_bwd_hdim*_{fp16,bf16}_legacy_sm80.cu` = v11's dq/dk/dv kernel with every feature
  switch (`run_mha_bwd_legacy_`, declared in flash.h). `flash_bwd_launch_template.h`: v11's
  `run_flash_bwd_seqk_parallel` split into `run_flash_bwd_preprocess` / `run_flash_bwd_legacy_kernel` /
  `run_flash_bwd_convert_dq` + `bwd_grid_dimx`; `BwdPart {Main, Legacy}` threaded through `run_mha_bwd_hdim*`.
- Forward = v11x's (`-DCQSA_NO_ELEMENT_PATH`): only plain/zero forwards are valid in v12b; the harness and tests
  take o/lse from v11's forward. setup.py: kernel sets `a100_fp16_hdim64_128` and `common` only; refuses
  `CQSA_BWD_NO_CQS`. `build.sh`: watchdog scoped to this build's own ptxas (the shared watchdog.sh kills ANY ptxas,
  including other agents' builds).
- Test entry `bwd_cqs_census(bits, blk_or, blk_and, is_causal, head_size)` -> (route, runs [num_n, 16, 2], tile);
  module attrs `CQS_BWD_FORCE_LEGACY`, `CQS_BWD_RUN_SLOTS`. `deterministic` was already an argument of `bwd`
  (upstream's split dq_accum); ALIGNED keeps its `blockIdx.x * dq_accum_split_stride` offset and grid.

Files changed in v12b vs v11: `setup.py`, `csrc/flash_attn/flash_api.cpp`, `src/flash.h`,
`src/flash_bwd_launch_template.h`, `src/flash_bwd_hdim{64,128}_{fp16,bf16}_sm80.cu`, `src/flash_fwd_kernel.h`
(v11x's NO_ELEMENT_PATH, unchanged from v11x); new: `src/cqs_census_bwd.{h,cu}`, `src/flash_bwd_aligned.h`,
`src/flash_bwd_hdim*_legacy_sm80.cu` (4), `build.sh`. `src/flash_bwd_kernel.h` is v11's, unchanged.

## Job ids / builds

- b2 (full rebuild, csv path changed) 14:16-14:30; b3a 14:31-14:35 (not measured); b3 14:40-14:44; b4 14:48-14:51.
- 14323074: bwd_ab b2 (snapshot ultimate-d5c7f0e, frozen/b2) -> bwd_ab_b2.*.
- 14323878: bwd_ab b3 (snapshot ultimate-35d5562, frozen/b3) -> bwd_ab.{json,md} (canonical), bwd_ab_b3.*.

- b1: clean build 2026-09-23 13:34-14:02 on vis1 (`TAG=b1 CLEAN=1 nohup setsid bash next/kernel/v12b/build.sh`), rc=0.
- 14322302: bwd_ab on A100-SXM4-80GB (snapshot ultimate-50e64d2, frozen/b1) -> results/ultimate/bwd_ab.{json,md}, bwd_ab_14322302.out.

## Next step for a resumed agent (package complete; kept for reference)

Check `next/logs/06_build_v12b_b1.log` ("finished rc="). Then on vis1 with
PYTHONPATH = worktree:next/kernel/stable:$PYTHONPATH:next/native:next/kernel/v11x:next/kernel/v12b (variant dirs LAST):
`python -P -m pytest -q tests/test_ultimate_v12b.py`, `python -P benchmarks/ultimate/bwd_ab.py --smoke`. Then freeze
the .so (`next/kernel/v12b/frozen/<tag>/`), commit, `SNAP=$(bash slurm/ultimate/snapshot.sh) && sbatch
--export=ALL,CQSA_SRC=$SNAP,V12B_DIR=<frozen> slurm/ultimate/bwd_ab.slurm` (check squeue: <= 1 gpu-test job of mine);
`python -P benchmarks/ultimate/bwd_ab.py --report results/ultimate/bwd_ab.json` writes bwd_ab.md.
