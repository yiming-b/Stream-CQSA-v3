# P3 progress log: forward kernel v12f (ALIGNED run-table loop + LEGACY v11 fallback)

Package goal (task text, 2026-09-23): a CQS forward whose CQS-on instantiation on 128-aligned layouts is
upstream's loop plus a run table (no in-loop verdict loads, no next-live search, no per-element mask path),
exact everywhere. Variant tree: `/scratch/gpfs/AKEY/yb2807/Stream-CQSA-dev/next/kernel/v12f` (not git; module
`cqsa_cuda_next_v12f`); diff against `next/kernel/v11` committed as `docs/ultimate/kernel_diffs/v12f.diff`.

## State (update at every milestone)

| step | status | notes |
|---|---|---|
| read design 1/3/4/7, exp_build_ab.md, cuobjdump_v11x.md, v11 kernel | done 2026-09-23 | |
| clone v11 -> v12f, code written | done 2026-09-23 10:38 | files listed below |
| build b1 (clean, detached, vis1) | done rc=0 in 1926 s (11:10) | log `next/logs/05_build_v12f_b1.log`, csv `next/logs/build_v12f_b1_nvcc_time.csv` |
| harness / test / slurm written | done | `benchmarks/ultimate/fwd_ab.py`, `tests/test_ultimate_v12f.py`, `slurm/ultimate/fwd_ab.slurm` |
| correctness on vis1 (tests/test_ultimate_v12f.py) | done 13:20: **37 passed** (8 s) | profiler check: aligned real -> census 7 us + ALIGNED 6.3 ms + LEGACY 6 us (no-op); mixed random -> ALIGNED 4 us (no-op) + LEGACY; forced LEGACY -> LEGACY alone. `fwd_ab.py --smoke` (vis1, N=16K/32K/64K): every v12f row equal to v11, route 0 on every aligned task incl. all 73 c=73 tasks |
| static report b1 | done | `results/ultimate/v12f_static.md` (`benchmarks/ultimate/v12f_static.py`): ALIGNED 3.8-7.2K SASS vs CQS-off 3.3-6.7K, LEGACY = v11's CQS-on (37-74K); v12f LEGACY non-causal STACK 312/328 B vs v11's 640/680 (TU split) |
| SXM job fwd_ab | - | |
| results fwd_ab.json/md, diff, commit | - | |

## Design as built (refinements of the task's sketch)

- **Census kernel** (`src/cqs_census.{h,cu}`, own TU): one warp per row block of the ALIGNED tile shape
  (hdim64 128x128, hdim128 128x64 = v11's A100 tiles, `CqsAlignedTile`). A visited tile ([0, n_block_max) with the
  causal limit) is live iff row OR & col OR == 0, dead iff row AND & col AND != 0, else mixed; the ragged last
  64-token summary block is recomputed from the group bits so a ragged tile gets an exact verdict. Writes
  `route` (0 ALIGNED; bit 1 mixed, bit 2 run overflow) and the run table.
- **Run table instead of a per-tile CSR list**: per row block up to 15 runs `{hi, lo}` in descending n plus a
  `{-1,-1}` sentinel (`kCqsRunSlots = 16`). Memory O(row blocks), not O(live tiles) (a CSR list is ~L^2/8192 B:
  0.5 GB at L=2M). In the loop `n_next = n > run_lo ? n - 1 : next run's hi`; the table is read only at a run
  boundary, the load issued before the QK GEMM. Aligned tasks have 1 run per causal row block at itr=1.
- **Routing without a host sync**: flash_api allocates the census (int32 buffer) only for eligible calls
  (group bits + summaries, no dropout/local/alibi/softcap/return_softmax, head 64/128, not forced LEGACY).
  `run_mha_fwd` then launches census -> ALIGNED -> LEGACY on the stream; ALIGNED CTAs return if route != 0,
  LEGACY CTAs return if route == 0 (`compute_attn` gate, the only change to v11's kernel code). Ineligible calls:
  LEGACY alone, no census. The Python API is unchanged (drop-in `CQSA_CUDA_MODULE=cqsa_cuda_next_v12f`); new
  trailing `cqs_fwd_flags` (1 = force LEGACY, 2 = reversed row-block order in ALIGNED) and `fwd_cqs_census`
  (test entry: route, run table, tile shape).
- **ALIGNED kernel** (`src/flash_fwd_aligned.h`): v11's compute_attn_1rowblock minus dropout/local/alibi/softcap/
  return_softmax, Is_even_K forced; softmax state initialised to (-inf, 0, 0) and ONE rescale instantiation
  (Is_first=false, Check_inf=true) for every tile (softmax.h checked: max(-inf,x)=x via fmaxf, Check_inf maps
  -inf row max to 0 so the scale is exp2(-inf)=+0, row_sum 0*0+s, acc_o 0*0; = Is_first bit for bit). Masking
  steps = v11's (the top n_masking_steps tiles of the row; live ones take the masked body, predicated V load only
  for the first visited tile, which is the only one that can be ragged); steady loop = upstream body without a
  mask call. No-live-tile row blocks skip the prologue (no Q load) and write out 0 / lse +inf like v11. In-place
  segmented reads (cqs_block_base) and the fp32 out contract copied from v11. `cqs_reverse_m` selects
  m_block = gridDim.x-1-blockIdx.x. No sCqsBits smem (kSmemSize only).
- **TU split**: `flash_fwd_hdim*_sm80.cu` = CQS-off (mode 0) + ALIGNED (2 instantiations: Is_even_MN);
  `flash_fwd_hdim*_legacy_sm80.cu` = v11's CQS-on (modes 1/2, full switch). `run_flash_fwd<.., Cqs_on>` replaces
  the runtime CQS_MODE_SWITCH. ALIGNED/census live in headers only the main TUs include, so an ALIGNED-only
  change rebuilds only the main TUs (the build script is incremental unless CLEAN=1).
- setup.py: only `a100_fp16_hdim64_128` and `common` kernel sets; `-lineinfo -Xptxas -v -time` via
  `CQSA_NVCC_TIME_CSV`; build with `CQSA_BWD_NO_CQS=1` (backward irrelevant here, NOT correct for CQS bwd).

Files changed in v12f vs v11: `setup.py`, `csrc/flash_attn/flash_api.cpp`, `src/flash.h`,
`src/flash_fwd_kernel.h` (gate), `src/flash_fwd_launch_template.h`, `src/flash_fwd_hdim{64,128}_{fp16,bf16}{,_causal}_sm80.cu`;
new: `src/cqs_census.{h,cu}`, `src/flash_fwd_aligned.h`, `src/flash_fwd_hdim*_legacy_sm80.cu` (8), `build.sh`.

## Job ids / builds

- b1 .so frozen at `next/kernel/v12f/frozen/b1/` (md5 22b93a49...); `fwd_ab.slurm` takes `V12F_DIR` so a rebuild cannot change a queued job.
- `fwd_ab.py` now records per-kernel device us of one profiled call per v12f CQS row (census / ALIGNED / LEGACY) and the c=73 sums: smoke says census 4-6 us + LEGACY no-op 4-5 us + memset 1 us per call (~0.7 ms of 37 ms over the 73 tasks at N=64K on vis1).

- b1: clean build started 2026-09-23 10:38:39 on vis1 (`TAG=b1 CLEAN=1 nohup setsid bash next/kernel/v12f/build.sh`).

## Next step for a resumed agent

Check `next/logs/05_build_v12f_b1.log` ("finished rc="). Then on vis1:
`python -P -m pytest -q tests/test_ultimate_v12f.py` with PYTHONPATH = worktree:next/kernel/stable:$PYTHONPATH:next/native:next/kernel/v12f
(v12f LAST: it carries a stub stream_cqsa). Then commit, `SNAP=$(bash slurm/ultimate/snapshot.sh) && sbatch
--export=ALL,CQSA_SRC=$SNAP slurm/ultimate/fwd_ab.slurm`; `python -P benchmarks/ultimate/fwd_ab.py --report
results/ultimate/fwd_ab.json` writes fwd_ab.md.

## Status snapshot (relayed user status question, 2026-09-23 13:15)

Slurm (sacct since 09-21): the v2.2.3 work (main's unreleased 24a4ff7/5ba4c90/c4bc39b; not tagged, `__version__`
still 2.2.2) has all 12 released jobs finished: paper supplements 13908223/13908224/13915115/13915470 COMPLETED
(1.6-4.2 h), OOC forward grid 13952419/21/23/25 COMPLETED, OOC backward grid 13952418/20/22/24 OUT_OF_MEMORY
(the known 4M-bwd host OOM at 96G, owed by the OOC-harness package). Ultimate program: dist4 telemetry 14312486
still PENDING (Priority, gpu-short), fold-workers A/B 14313570 COMPLETED (I0), no gpu-test job of this package
queued. P3 itself: v12f build b1 finished rc=0 in 1926 s; next = vis1 correctness, then the SXM fwd_ab job.
