# Stream-CQSA ultimate A100 program: the kernels report (03) -- Phase 2/3 kernels, review fixes, Phase 4 plan

Written 2026-09-23 by package F23, branch `ultimate` of `/scratch/gpfs/AKEY/yb2807/Stream-CQSA-ultimate`. Plan of
record: `docs/ultimate/01_design_a100.md`; the Phase 0/1 report is `02_phase01_report.md`. Kernel variant trees are
under `/scratch/gpfs/AKEY/yb2807/Stream-CQSA-dev/next/kernel/<variant>` (not git); their diffs against v11 are
committed as `docs/ultimate/kernel_diffs/{v12f,v12b}.diff`. Every timing below names its GPU, node and slurm job; all
timings are **A100-SXM4-80GB** (gpu-test, `--constraint="sxm&gpu80"`), fp16, B=1, H=8 unless stated, one process
per job, CUDA events, median of 5 (F23 job: 7) timed runs after warm-up. The shared vis1 GPU (A100-PCIe-40GB) was used
for correctness only.

Names used throughout: **plain** = CQS off (the extension's `fwd`/`bwd`, upstream's loop); **zero** = CQS on with
all-zero group bits (every tile live: the "all-live tax"); **real** = CQS on with the task's own group bits;
**c=7 task** = the smallest 128-aligned task of the c=7, itr=1 decomposition (N=131072 -> L=56064).
**flash_attn** = the FlashAttention-2 package 2.8.3 (built in job 14240707); **SDPA-flash** = torch SDPA with the
FLASH_ATTENTION backend forced; **in-tree plain** = our own build of upstream's loop (v11/v12f `fwd`, v11x NO_CQS
`bwd`), the same template instantiation as flash_attn's kernel.

## 1. Summary

| | forward v12f (P3) | backward v12b (P2) |
|---|---|---|
| design | census kernel -> ALIGNED run-table kernel -> LEGACY (= v11's CQS forward), one stream, no host sync | the same per column block: census -> ALIGNED (upstream's seqk-parallel loop over runs) -> LEGACY (= v11's CQS backward) |
| causal D64 L=56064 real | 12.65 ms = **0.731x in-tree plain**, 0.707x v11 (job 14324080); **0.766x flash_attn** with a warmed-up FA arm (job 14330595; the 0.694 of 14324080 was a first-arm clock artifact, 4.2) | 34.41 ms = **0.783x flash_attn**, 0.784x in-tree NO_CQS, 2.48x faster than v11 (job 14323878) |
| all-live tax (zero / in-tree plain) | 0.939 with the default descending order; **0.985-0.986 with the same (ascending) order on both sides** (H=8; 0.982 at H=32) | 0.990 (v12b zero / v11x NO_CQS) |
| non-causal | ALIGNED 1.46x (D64) / 1.75x (D128) faster than v9 on zero; LEGACY <= v9 on mixed patterns: the second extension can retire | real 0.781x flash_attn, 1.94x faster than v11 |
| c=73 N=131072 (73 tasks, L 16128-16384) | 108.8 ms summed = **1.202x monolithic flash_attn** (90.5); 0.881x the per-task flash_attn sum | 252.5 ms = **1.064x monolithic flash_attn bwd** (237.4); 0.885x the same 73 flash_attn calls |
| identity | out32, lse, out16 `torch.equal` v11 on every row, every task (finite K/V; section 6.3) | dk, dv `torch.equal` v11 on every row, every task; dq within the fp32 atomics' drift (<= 1.2e-4) |
| review | 1 blocker (windowed causal call -> non-causal ALIGNED, silently wrong) **fixed** in b3; 8 minors resolved | 0 blockers; 9 minors resolved or recorded |
| build shipped by F23 | `next/kernel/v12f/frozen/b3` | `next/kernel/v12b/frozen/b5` (whole-module SASS identical to the measured b4) |

F23's combined SXM job **14330595** (della-l08g1, A100-SXM4-80GB, builds v12f b3 + v12b b5, 63 s) re-timed the
headline rows on the fixed builds (forward real / in-tree plain 0.733, backward real / flash_attn 0.784: unchanged)
and measured what the reviews found missing: the ragged (`Is_even_MN=false`) ALIGNED kernels (forward zero 0.984x
in-tree plain, real 0.762x; backward real 0.791x flash_attn, no cost over the even task), the automatic LEGACY route
of the backward (+0.1-0.4% over v11), the small-L regime of itr=2 (forward L~2K: real 0.996x the flash_attn loop,
fixed per-call cost ~10% of the call; backward L~4-5K: real 0.60-0.80x the flash_attn loop, fixed cost ~2%), an H=32
row (descending order still helps: -1.8%), and a warmed-up FA arm that **corrects the forward headline against
flash_attn from 0.694 to 0.766** (the package is 4.3% *faster* than our in-tree build of the same kernel once the
clocks are up; the target <= 0.82 still holds).

Both variants are **experiment builds** of one direction each: v12f is built with `CQSA_BWD_NO_CQS=1` (its CQS
backward ignores the mask) and v12b with `-DCQSA_NO_ELEMENT_PATH` (its CQS forward returns NaN rows for any real,
unaligned or mixed pattern). Neither is usable for production alone; package U1 builds the one module (section 9).

## 2. Forward kernel v12f as built

Source: `next/kernel/v12f` (module `cqsa_cuda_next_v12f`), diff `kernel_diffs/v12f.diff`; progress log
`progress/P3_forward_kernel.md`.

- **Census** (`src/cqs_census.{h,cu}`, own TU): one warp per row block of the ALIGNED tile (A100: hdim64 128x128,
  hdim128 128x64 = v11's tiles, `CqsAlignedTile`). For every visited tile (`[0, n_block_max)` with the causal limit)
  it decides from the 64-token OR/AND summaries: **live** iff row OR & col OR == 0, **dead** iff row AND & col AND != 0,
  else **mixed**; the ragged last summary block is recomputed from the group bits so a ragged tile gets an exact
  verdict. It writes `route` (0 = ALIGNED; bit 1 = a mixed tile; bit 2 = run-table overflow) and a **run table**:
  per row block up to 15 runs `{hi, lo}` of live column tiles in descending n plus a `{-1,-1}` sentinel
  (`kCqsRunSlots = 16`). Memory is O(row blocks), not O(live tiles) (a CSR tile list would be ~L^2/8192 B: 0.5 GB at
  L=2M). An aligned itr=1 task has 1 run per causal row block.
- **Routing without a host sync**: `mha_fwd_impl` allocates the census buffer (int32) only for eligible calls (group
  bits + summaries, no dropout / local window / alibi / softcap / returned softmax, head 64 or 128, not forced
  LEGACY). `run_mha_fwd` then launches **memset(route) -> census -> ALIGNED -> LEGACY** on the stream; ALIGNED CTAs
  return if route != 0, LEGACY CTAs return if route == 0 (the `compute_attn` gate is the only change to v11's
  kernel code). Ineligible calls run LEGACY alone. The Python API is unchanged (drop-in `CQSA_CUDA_MODULE`); new
  trailing `cqs_fwd_flags` (1 = force LEGACY, 2 = force descending row-block order, 4 = force ascending) and a
  `fwd_cqs_census` test entry.
- **ALIGNED kernel** (`src/flash_fwd_aligned.h`): v11's `compute_attn_1rowblock` minus dropout / local / alibi /
  softcap / return_softmax, `Is_even_K` forced; the KV loop walks the run table (`n_next = n > run_lo ? n - 1 :
  next run's hi`; the table is read only at a run boundary, the load issued before the QK GEMM). The online softmax
  starts at (-inf, 0, 0) and every tile takes the non-`Is_first`, `Check_inf` rescale, which reproduces `Is_first`
  bit for bit (max(-inf, x) = x via fmaxf; the Check_inf guard maps an all -inf row to scale exp2(-inf) = 0). The
  masking steps are v11's (the top `n_masking_steps` tiles take the masked body; only the first visited tile can be
  ragged, so only it takes the predicated V load); the steady loop is upstream's body without a mask call. Row blocks
  with no live tile skip the prologue and write out 0 / lse +inf like v11. In-place segmented reads
  (`cqs_block_base`) and the fp32 out contract are v11's. `cqs_reverse_m` selects `m_block = gridDim.x-1-blockIdx.x`.
- **Row-block order**: descending is the **causal default** since build b2 (heaviest row blocks first, no drain tail;
  flash_api.cpp only, ALIGNED SASS identical to b1); ascending for non-causal (neutral there, section 4.1).
- **TU split**: `flash_fwd_hdim*_sm80.cu` = CQS-off + ALIGNED (2 instantiations: `Is_even_MN`);
  `flash_fwd_hdim*_legacy_sm80.cu` = v11's CQS-on code (modes 1/2, full switch) in its own TU. `run_flash_fwd<..,
  Cqs_on>` replaces the runtime `CQS_MODE_SWITCH`. The TU split is also what keeps v11's non-causal CQS-on code from
  being mis-compiled (F15; section 4.1 mixed rows).
- **F23 changes (b3)**: the census gate reads the post-`set_params_fprop` parameters (`!cqs_is_local`, upstream's
  `Is_local == false`) and requires `blk_and`; `run_mha_fwd` checks that a route implies its summaries; the
  non-`CQSA_MINIMAL_FWD_KERNELS` `#else` dispatch is an `#error`; the finite-V caveat is stated in the kernel header;
  `build.sh` uses a watchdog scoped to its own process tree. `cuobjdump -sass` of the whole b3 module (387 functions)
  is identical to the measured b2's: every b2 timing stands for b3.

## 3. Backward kernel v12b as built

Source: `next/kernel/v12b` (module `cqsa_cuda_next_v12b`, cloned from v11x), diff `kernel_diffs/v12b.diff`; log
`progress/P2_backward_kernel.md`.

- **Census** (`src/cqs_census_bwd.{h,cu}`, own TU): v12f's census transposed: one warp per **column block** of the
  backward tile (A100: hdim64 128x128, hdim128 64x128 = `run_mha_bwd_hdim64/128`'s >= 144 KB smem choice; launched
  from the templated launcher with `Kernel_traits::kBlockM/kBlockN`, so it cannot disagree with the kernel). Lanes
  classify m tiles in `[m_min, m_max)` downward (`m_min = n*kBlockN/kBlockM` under causal = the kernel's
  `m_block_min`); same live / dead / mixed rule and exact ragged tail; per column up to 15 runs of live m blocks in
  descending m + sentinel (`kCqsBwdRunSlots = 16`). Constants carry a `Bwd` infix so the U1 merge with v12f's
  census does not collide.
- **Routing**: `mha_bwd` allocates the census only for eligible calls (group bits + both summaries, not numden, no
  dropout / alibi / softcap / local, head 64/128, not forced LEGACY via `cqs_bwd_flags & 1`). `run_flash_bwd_main`
  (main TU) = preprocess -> [census -> ALIGNED(Use_runs) -> LEGACY] -> convert_dQ on one stream; ALIGNED CTAs return
  if route != 0, LEGACY CTAs return if route == 0 (a gate in the LEGACY `__global__` wrapper, the only change to
  v11's kernel path). Ineligible CQS calls: LEGACY alone. **CQS-off calls**: ALIGNED<Use_runs=false> (upstream's
  loop, no table) when eligible, else LEGACY.
- **ALIGNED kernel** (`src/flash_bwd_aligned.h`, build b3 on): the **text** of v11x's
  `compute_dq_dk_dv_1colblock` (the `CQSA_BWD_NO_CQS` side selected; `Is_first = Is_last = false`,
  `Seq_parallel`, no dropout / local / alibi / softcap, `Is_even_K`) plus template flags `Use_runs` and `Dpsum_smem`
  whose changes fold away by `if constexpr` when off. The m loop walks the runs (`m_next = m > run_lo ? m - 1 : next
  run's hi`); Q / dO / LSE / dPsum / dQaccum pointers jump by `m_step` rows (64-bit); the sQ double buffer toggles per
  **visited** tile; masking is upstream's `apply_mask` / `apply_mask_causal` (the same -inf positions v11's `Mask<>`
  writes); a column with no live tile writes dK = dV = 0. dP_sum source: staged in smem by cp.async next to the Q
  prefetch (default for CQS-on since b4) or upstream's global reads (default for CQS-off); env
  `CQSA_V12B_DPSUM_SMEM=0|1` overrides for A/B.
- Lesson of b1 -> b3 (recorded in P2): hand-specialised copies of upstream's loop with *fewer* instructions ran 3-5%
  slower (ptxas sank the dP_sum global loads next to their use); deriving the function from the exact text of
  upstream's loop gave upstream speed. The TU context, not the source, decides such schedules.
- **F23 changes (b5)**: the host census gate reads `params.is_causal` / `params.window_size_*` (the launcher's
  `cqs_bwd_aligned_eligible` did already, so a windowed causal call was correct but allocated a census the launcher
  discarded); `CQSA_V12B_DPSUM_SMEM` accepts only "0" and "1" (TORCH_CHECK). `cuobjdump -sass` of the whole b5 module
  (443 functions) is identical to b4's: every b3/b4 timing stands for b5.

## 4. Forward measurements

### 4.1 Headline rows (job 14324080, della-l08g2, A100-SXM4-80GB, build b2; b1 = job 14319938, della-l07g6)

Aligned c=7 task N=131072 (L=56064), ms. flash_attn / SDPA columns are this job's FA arms at the same L.

| causal | D | flash_attn | SDPA | v11 plain | v11 zero | v11 real | v11x zero | v9 zero / real | v12f plain | v12f zero (default) | v12f real (default) | v12f zero / real ascending (flags 4) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 64 | 18.22 | 17.83 | 17.29 | 22.59 | 17.88 | 19.44 | - | 17.30 | 16.23 | **12.65** | 17.05 / 13.02 |
| 1 | 128 | 29.71 | 32.07 | 29.69 | 40.51 | 31.83 | 33.91 | - | 29.90 | 28.18 | **21.91** | 29.81 / 22.58 |
| 0 | 64 | 33.07 | 34.16 | 33.51 | 204.73 | 157.49 | 34.73 | 47.42 / 38.14 | 33.50 | 32.44 | **25.36** | (default) |
| 0 | 128 | 56.75 | 59.71 | 56.74 | 256.37 | 201.52 | 64.33 | 99.41 / 76.09 | 56.76 | 56.70 | **44.34** | (default) |
| 1 | 64, L=224640 | 267.46 | 282.47 | 279.93 | 343.16 | 269.91 | 300.52 | - | 281.47 | 260.20 | **203.30** | 275.18 / 207.82 |
| 1 | 64, L=6912 | 0.44 | 0.42 | 0.42 | 0.66 | 0.57 | 0.51 | - | 0.43 | 0.43 | **0.35** | 0.46 / 0.37 |

Ratios (same job; "in-tree" = v11 plain, the same upstream loop compiled in our tree):

| config | real / flash_attn | real / in-tree plain | real / v11 real | zero / in-tree plain (default order) | zero / in-tree plain, both ascending (the tax alone) | descending gain (zero desc / zero asc) |
|---|---|---|---|---|---|---|
| causal D64 56K | **0.694** | 0.731 | 0.707 | 0.939 | **0.986** | 0.952 |
| causal D128 56K | 0.737 | 0.738 | 0.688 | 0.949 | 1.004 | 0.945 |
| non-causal D64 56K | 0.767 | 0.757 | 0.161 | 0.968 | 0.968 | 1.000 (neutral: 32.39 vs 32.40 ms) |
| non-causal D128 56K | 0.781 | 0.781 | 0.220 | 0.999 | 0.999 | 1.000 |
| causal D64 225K | 0.760 | 0.726 | 0.753 | 0.930 | 0.983 | 0.946 |
| causal D64 6912 | 0.806 | 0.837 | 0.626 | 1.018 | 1.088 | 0.936 |

Targets (package P3): zero <= 1.05x v11 plain: met everywhere, including with the ascending order on both sides
(0.968-1.004); real <= 0.82x flash_attn at causal D64 56K: 0.694 in this job, **0.766 with a warmed-up FA arm** (job 14330595,
section 4.2); 0.731 against the in-tree kernel. Non-causal zero / plain 0.968 / 0.999 retires v9 for aligned tasks (v9 zero is 1.46x / 1.75x slower).

What the columns say:

1. **The all-live tax is gone.** v11's zero / plain is 1.307 at causal D64 56K (v11x without the element path
   1.123); v12f's ALIGNED on the same all-live pattern runs at 0.986x the in-tree CQS-off kernel with the same row
   order, 0.939x with its default descending order. The 0.939 therefore contains a scheduling gain of 4.8% (heaviest
   causal row blocks first) that the baseline arms could also take; it is reported separately above (review
   finding M3).
2. **Dead-tile skipping pays in full.** real / zero = 0.779 at causal D64 56K; the c=7 aligned task masks 22.2% of
   the causal pair work.
3. **Non-causal: one kernel.** v11's non-causal CQS-on instantiation is 6.1x plain (F15, a code-size mis-compile);
   v12f routes aligned non-causal calls to ALIGNED (zero 1.46x faster than v9 at D64, 1.75x at D128) and mixed
   non-causal calls to LEGACY in its own TU, which is not mis-compiled (next table).
4. **Small L (6912):** real 0.806x flash_attn, zero 1.02x in-tree plain (the fixed per-call cost is visible here;
   section 4.3 measures L ~ 2K).

Mixed-tile patterns (LEGACY), N=131072 unaligned c=7 task (seg_align=0, L=56173), D=64, job 14324080:

| causal | pattern | v11 | v9 | v12f (route 1) | flash_attn at L |
|---|---|---|---|---|---|
| 1 | unaligned | 24.58 | - | 24.55 | 17.09 |
| 1 | random 3-bit | 158.72 | - | 157.76 | 17.09 |
| 0 | unaligned | 194.89 | 48.81 | **46.74** | 34.07 |
| 0 | random 3-bit | 515.54 | 347.53 | **346.74** | 34.07 |

v12f is at or below v9 on every non-causal D64 pattern measured; D128 mixed non-causal is not measured yet (U1 item).

c=73, N=131072 (73 tasks, L 16128-16384), causal D64, per-task medians summed (job 14324080):

| arm | sum ms | / monolithic flash_attn (90.5 ms at N=131072) | / flash_attn per-task sum (123.5) |
|---|---|---|---|
| v12f real (default) | **108.8** | **1.202** | 0.881 |
| v12f real ascending | 114.5 | 1.264 | 0.927 |
| v11 real | 164.6 | 1.818 | 1.333 |
| v12f plain / v11 plain | 125.9 / 126.3 | 1.391 / 1.395 | 1.020 / 1.022 |
| SDPA-flash per task | 129.2 | 1.427 | 1.046 |

The end-to-end comparison is the **monolithic** one: computing the same attention as 73 decomposed subproblems costs
1.202x one flash_attn call over all N tokens, before the gather, the (m,l,acc) merge and the host pipeline. The
per-task flash_attn sum (dense attention over the packed layouts, dead tiles included) is a same-layout dense
reference, i.e. a sparsity figure, not a speed comparison with FA-2 (review finding M5). Census + LEGACY no-op +
memset over the 73 tasks: 0.38 + 0.33 + 0.08 = 0.79 ms (0.73%).

### 4.2 Package-vs-in-tree offset and the FA-arm caveats

The flash_attn package runs the same template instantiation as our in-tree plain kernel
(`Flash_fwd_kernel_traits<64,128,128,4,...>`), but its measured time differs by an L-dependent amount: at causal D64
L=56064 18.22 ms vs 17.29-17.30 (package +5.3%), at L=56173 later in the same job 17.09 ms (package -1.2%), at
L=224640 267.46 vs 279.93 (package -4.5%), c=73 per-task 123.5 vs 125.9. Ratios against flash_attn therefore carry
an offset that has nothing to do with v12f; section 4.1 reports real / in-tree plain next to real / flash_attn.
The SDPA-flash arm was bimodal in job 14319938 (median 19.64 ms, min 17.95) and 17.83 in 14324080, so the design-3.3
flash_attn/SDPA factor (1.077) and the "SDPA 19.64 vs 19.70" cross-job reproducibility claim of `fwd_ab.md` do not
hold; the F23 job gives the FA arms 3 warm-ups and 10 runs, a 5 s GEMM clock warm-up before the first row, and
re-times them after every other arm (section 4.3).

**Resolution (job 14330595):** with the clocks warmed up, flash_attn at causal D64 L=56064 H=8 takes 16.84 ms
(re-timed at the end of the group: 16.82) against the in-tree plain kernel's 17.59 (17.56 at the end): the package is
**4.3% faster**, not 5.3% slower. The 18.22 ms of jobs 14319938 / 14324080 was the job's first timed arm (no clock
warm-up; the SM clock idles at 210 MHz, `nvidia-smi` in the job log) -- consistent with the same package kernel
taking 17.09 ms at L=56173 later in 14324080 and 267.46 vs 279.93 ms (-4.5%) at L=224640. At H=32 the package is
5.2% faster (64.81 vs 68.34). Consequences: (a) every forward ratio "vs flash_attn" from jobs 14319938 / 14324080 is
flattered by ~5-9% at 56K; the in-tree ratios are the stable ones; (b) re-based, **v12f real / flash_attn = 0.766**
at causal D64 56K (target <= 0.82: met), 0.791 at H=32, 0.772 on the ragged L=42787 task; (c) the backward's FA arm
was not affected (43.80 ms here vs 43.95 there; the backward rows ran after a forward in the same process).
SDPA-flash here: 17.84 ms (1.060x flash_attn, 1.014x in-tree plain), 69.15 at H=32 (1.067x).

### 4.3 F23 job: re-timed headline, ragged ALIGNED, H=32, small L (itr=2)

Job **14330595** (della-l08g1, A100-SXM4-80GB, v12f b3 = `frozen/b3`), `results/ultimate/kernels_f23.{json,md}`.
Every v12f row `torch.equal` to v11 (NaN-pre-filled outputs); every aligned pattern route 0.

Headline, causal D64, c=7 task L=56064 (ms; in-tree = v11 plain of the same group):

| H | flash_attn | v11 plain | v12f plain | v12f zero desc (default) / asc | v12f real desc (default) / asc | zero asc / in-tree (tax) | descending gain zero / real | real / in-tree | real / flash_attn |
|---|---|---|---|---|---|---|---|---|---|
| 8 | 16.84 | 17.59 | 17.57 | 16.49 / 17.33 | 12.90 / 13.27 | **0.985** | 0.954 / 0.972 | **0.733** | **0.766** |
| 32 | 64.81 | 68.34 | 68.28 | 65.88 / 67.13 | 51.28 / 51.91 | **0.982** | 0.982 / 0.989 | 0.750 | 0.791 |

The b3 fixes are host-only and the headline reproduces b2 in in-tree terms (0.733 vs 0.731; zero default 0.937 vs
0.939). At B*H = 32 the descending gain shrinks (tail effects scale with 108 SMs / (B*H*num_m)) but stays a gain:
descending remains the causal default (review M3's question answered).

Ragged ALIGNED (`Is_even_MN=false`), N=100003 c=7, causal D64 H=8:

| task | L | flash_attn | v11 plain | v11 zero | v11 real | v12f zero | v12f real | zero / in-tree | real / in-tree | real / flash_attn |
|---|---|---|---|---|---|---|---|---|---|---|
| even | 42752 | 10.16 | 10.57 | 13.56 | 10.69 | 9.87 | 7.78 | 0.934 | 0.737 | 0.766 |
| ragged (L%128 = 35) | 42787 | 10.28 | 10.42 | 14.14 | 14.36 | 10.25 | 7.93 | **0.984** | **0.762** | 0.772 |

The ragged instantiation costs ~4% on the all-live pattern (zero 10.25 vs 9.87 ms for +0.08% tokens; its first
visited tile per row block takes the predicated loads and two masking steps) and ~2% on the real pattern; both stay
inside the targets. v11's ragged real task is 1.38x plain (its ragged tail runs the element path); v12f is 1.81x
faster there.

Small L, c=73 itr=2, N=131072: 73 of the 5329 tasks (evenly spaced, L 1664-2176), one timed region per arm
(host launches included), causal D64 H=8:

| arm | ms | / flash_attn loop | / in-tree plain loop | device us: census / ALIGNED / LEGACY no-op / memset |
|---|---|---|---|---|
| flash_attn (extension called directly) | 5.30 | 1 | 0.962 | - |
| v11 plain | 5.51 | 1.040 | 1 | - |
| v11 real | 9.55 | 1.804 | 1.735 | (v11 kernel 8990) |
| v12f plain | 4.98 | 0.940 | 0.904 | (plain kernel 4906) |
| v12f zero | 6.00 | 1.133 | 1.090 | 236 / 5293 / 174 / 99 |
| v12f real | **5.28** | **0.996** | 0.958 | 255 / 4549 / 179 / 98 |
| v12f forced LEGACY real | 9.16 | 1.729 | 1.663 | - |

At L ~ 2K the fixed per-call cost (census 3.5 us + LEGACY no-op 2.5 us + memset 1.3 us per call = 532 us over 73
calls) is **10.1% of v12f real's time** (review M4 estimated 15-25%), and v12f real still matches the flash_attn loop
(0.996x) and runs 1.81x faster than v11. On the all-live pattern ALIGNED's device time is 1.08x the plain kernel's
(5293 vs 4906 us; at these L a row block has 13-17 tiles, so the per-row prologue weighs more). This is the regime
where U2's one-launch wave (one census and one no-op per wave, not per task) and folding the memset into the census
pay: tasks this small come from itr=2 (or large-c) decompositions, which the wave engine batches.

## 5. Backward measurements

### 5.1 Headline rows (job 14323878, della-l07g5, A100-SXM4-80GB, build b3 = b4 = b5 SASS)

ms; flash_attn = the package's backward through autograd (direct extension call within 0.1 ms at L >= 28K);
v12b CQS-on arms with the dP_sum smem stage (the b4 default).

| config | L | flash_attn | SDPA | v11x NO_CQS | v12b plain | v11 real | v12b real | v12b zero | v12b LEGACY (forced) | v11 zero |
|---|---|---|---|---|---|---|---|---|---|---|
| causal D64 | 13952 | 3.27 | 3.51 | 2.87 | 2.84 | 5.90 | 2.48 | 2.85 | 5.96 | 6.28 |
| causal D64 | 28032 | 11.24 | 12.20 | 11.14 | 11.04 | 21.86 | 8.89 | 11.05 | 21.85 | 24.27 |
| causal D64 | 56064 | 43.95 | 47.68 | 43.87 | 43.46 | 85.43 | **34.41** | 43.43 | 85.42 | 94.48 |
| causal D128 | 56064 | 80.37 | 83.95 | 80.17 | 80.17 | 124.86 | **61.82** | 78.02 | 124.78 | 137.32 |
| non-causal D64 | 56064 | 86.42 | 92.98 | 87.36 | 86.32 | 131.11 | **67.47** | 86.57 | 131.16 | 140.49 |

| config | real / flash_attn | real / in-tree NO_CQS | real / SDPA | zero / NO_CQS | vs v11 real |
|---|---|---|---|---|---|
| causal D64 56K | **0.783** | 0.784 | 0.722 | 0.990 | 2.48x faster |
| causal D128 56K | 0.769 | 0.771 | 0.736 | 0.974 | 2.02x |
| non-causal D64 56K | 0.781 | 0.772 | 0.726 | 0.993 | 1.94x |
| causal D64 28K / 14K | 0.791 / 0.759 | 0.798 / 0.864 | 0.729 / 0.707 | 0.992 / 0.996 | 2.46x / 2.38x |

For the backward the package and the in-tree loop agree (43.95 vs 43.87 ms at 56K), so the flash_attn ratios need no
offset. real / zero = 0.792 at 56K and the live-tile fraction of that task is 0.777: ALIGNED runs at upstream speed
and the gain is exactly the skipped tiles. **SDPA ratios are flattered**: SDPA-flash's backward is 8.5% slower than
flash_attn's on the same work, so the 0.72x "vs SDPA" figures are a historical gate basis (design P2.1), not a
kernel speed-up (review finding B4).

c=73, N=131072, causal D64 real (73 tasks in one timed region per arm):

| arm | ms | / flash_attn over the same 73 calls (285.4) | / monolithic flash_attn bwd at N (237.4) |
|---|---|---|---|
| v12b | **252.5** | **0.885** | **1.064** |
| v11 | 594.3 | 2.082 | 2.503 |
| v11x NO_CQS | 284.6 | 0.997 | 1.199 |
| flash_attn per task (autograd), summed | 295.5 | 1.035 (sum of individually timed calls) | 1.245 |

deterministic=True, L=56064 H=8 b=1: 66.60 vs 34.40 ms (1.94x, reruns torch.equal). The cause is the grid, not a
general caveat: `gridDimx = ceil(108 / (b*H)) = 14` CTAs per head at 1 CTA/SM (255 regs x 256 threads) = 112 CTAs,
108 in the first round and 4 in a second round, each walking ~31 of the 438 column blocks serially -> predicted
2 x 31.3 / (438 x 8 / 108) = 1.93x (review finding B1). A floor grid (13 per head, 104 CTAs, one round) would be
~1.04x; not changed here (a Phase 4 / U1 item; deterministic is not the engine's default).

### 5.2 F23 job: re-timed headline, ragged ALIGNED, automatic LEGACY route, small L

Job **14330595** (v12b b5 = `frozen/b5`). Every v12b row: dk/dv `torch.equal` to v11 (NaN-pre-filled
outputs), all 128 small-L tasks included; dq max |d| vs v11 1.2e-4 (2.4e-4 once, over the 64 c=13 itr=3 tasks:
fp32 atomics order at |dq| up to ~0.06, within the 2-ulp drift band the tests use).

| config | L | flash_attn (autograd / direct) | v11x NO_CQS | v12b plain | v11 real | v12b real | v12b zero | real / flash_attn | zero / NO_CQS |
|---|---|---|---|---|---|---|---|---|---|
| headline c=7 N=131072 | 56064 | 43.80 / 43.69 | 43.65 | 43.30 | 85.26 | **34.34** | 43.19 | **0.784** | 0.989 |
| N=100003 even | 42752 | 25.69 / 25.60 | 25.56 | 25.36 | 49.66 | 20.25 | 25.31 | 0.788 | 0.990 |
| N=100003 ragged (L%128 = 35) | 42787 | 25.66 / 25.57 | 26.14 | 25.58 | 51.91 | 20.29 | 25.37 | **0.791** | 0.971 |

The ragged `Is_even_MN=false` backward costs nothing measurable (real 20.29 vs 20.25 ms); v12b is 2.56x faster than
v11 on the ragged task.

Automatic LEGACY route (review B2): unaligned seg_align=0 c=7 tasks, causal D64, real pattern (route 1):

| L | v11 | v12b auto (memset + census + ALIGNED no-op + LEGACY) | v12b forced LEGACY | v11 again | auto / v11 | census / ALIGNED no-op us |
|---|---|---|---|---|---|---|
| 14043 | 6.25 | 6.28 | 6.26 | 6.27 | **1.004** | 5.9 / 5.4 |
| 56173 | 88.79 | 88.85 | 88.85 | 88.82 | **1.001** | 12.0 / 13.7 |

Forced LEGACY on the aligned L=13952 task: 5.93 = v11 5.93 ms (0.999; re-timed 0.999): the +1.0% of job 14323878 at
that L was noise; "LEGACY = v11 within 0.1%" now holds on every row measured.

Small L (review B3), N=131072, 64 of the tasks evenly spaced, one timed region per arm, causal D64 H=8:

| decomposition | L | flash_attn loop | v11x NO_CQS | v12b plain | v11 real | v12b real | v12b zero | real / flash_attn loop | zero / NO_CQS | fixed per-call cost (census + no-op + memset) |
|---|---|---|---|---|---|---|---|---|---|---|
| c=13 itr=3 | 3584-3968 | 22.40 | 22.18 | 22.18 | 35.37 | **13.35** | 22.64 | **0.596** | 1.021 | 3.6 + 2.9 + 1.5 us per call = 2.3% of zero |
| c=31 itr=2 | 4608-5120 | 32.14 | 31.90 | 31.87 | 61.61 | **25.71** | 32.23 | **0.800** | 1.010 | 4.0 + 3.4 + 1.4 us = 1.6% |

deterministic=True at L=56064 H=8 (review B1): 65.36 vs 34.33 ms (1.904x), peak extra device memory 1701 MiB vs
277 MiB for the atomics path (14 dq_accum splits of 115 MB). The ceil grid explanation of 5.1 stands; the floor
grid is a U1 item.

## 6. Correctness evidence

### 6.1 Test suites (vis1, A100-PCIe-40GB, shared; correctness only)

| suite | build | result |
|---|---|---|
| `tests/test_ultimate_v12f.py` (census vs numpy reference, ALIGNED real tasks c=7/31/73 itr 1-2 incl. ragged L, zero / all-ones / plain, both forced row orders, in-place segmented reads, automatic LEGACY on random / half / unaligned, forced LEGACY, float64 spot check; **F23**: windowed calls -> LEGACY vs v11 / forced LEGACY / masked float64 reference, run-table overflow through the kernels, finite-V caveat pinned, NaN-pre-filled outputs in every identity check) | b3 | **66 passed** (39 before F23 + 27 new); against the old b2 the 4 causal-window tests fail (max \|d\| 3.07: the blocker reproduced) |
| `tests/test_ultimate_v12b.py` (census reference, ALIGNED real tasks, caller dsoftmax_sum, zero / all-ones / plain, LEGACY routing, forced LEGACY, deterministic reruns, float64 error <= 1.2x flash_attn's + 1e-4, both dP_sum sources; **F23**: NaN-pre-filled dq/dk/dv in every call, overflow end to end (15 / 16 / 19-20 runs, ragged tail), B=2 GQA 4:2 and 4:1, deterministic dq/dk/dv == v11 deterministic, strided packed-qkv and prefix views, CQS-off Lq != Lk, windowed calls, env parse) | b5 | **87 passed** (48 before F23 + 39 new); against the old b4 only the 5 env-parse tests fail, as expected |

### 6.2 In-job identity (every SXM job)

Every v12f row of jobs 14319938 / 14324080 is `torch.equal` to v11 (fp32 out, lse, fp16 out), all 73 c=73 tasks and
the mixed-tile rows included; every aligned pattern routes to ALIGNED. Every v12b row of 14322302 / 14323074 /
14323878 has dk/dv `torch.equal` to v11 (real, zero, plain; both dP_sum sources; all 73 tasks); dq max |d| 1.2e-4 =
v11's own rerun drift. Since F23 the identity copies in `fwd_ab.py` / `bwd_ab.py` / `kernels_f23.py` are written into
NaN-pre-filled buffers (`torch.equal(NaN, NaN)` is False), so a kernel that skips a write cannot pass by inheriting
the freed buffer of an identical v11 call from the caching allocator (review finding B5).

### 6.3 The finite-V caveat of bit-identity (forward)

Bit-identity with v11 holds for **finite K/V**. v11 runs the fully masked tiles of the causal masking window (e.g.
the diagonal tile of a non-owner row block) as P = 0 followed by P @ V, so an inf or NaN in those V rows turns v11's
rows NaN (0 * inf); ALIGNED never loads such a tile, so its rows stay finite and match the masked math reference.
Reproduced on vis1 (L=512, bits `[0]*128+[1]*256+[0]*128`, V[200] = inf): v11 NaN in rows 128-255, v12f none; with
NaN: v11 rows 128-511, v12f only rows 384-511 (the rows that genuinely attend key 200). All finite elements agree.
v12f's behaviour is the correct one; `test_nonfinite_v_in_dead_tile_caveat` pins it. The backward has the analogous
property for dead tiles (v11 guards their GEMMs off or computes P == 0), and its dk/dv identity is likewise stated for
finite inputs.

### 6.4 Reviewers' adversarial runs (vis1)

- v12f: 119 random fuzz cases bit-identical to v11 (L 1-3000 incl. every ragged edge, 64/128-token block patterns,
  bit 63, B 1-2, GQA, strided qkv views, flags 0/2/4, D 64/128, both causal modes); 14/15 runs -> ALIGNED, 16/17 ->
  LEGACY, exact; 24 cases with live runs crossing the census's 32-tile chunk boundaries; 20 back-to-back calls on a
  side stream; `torch.cuda.set_sync_debug_mode('error')` around the production call: no host sync, exactly 4 kernels
  per call (memset, census, ALIGNED, LEGACY); census sweep over c in {3..91}, itr 1-2, N 131072 / 1048576: every
  sampled production task routes to ALIGNED with no overflow.
- v12b: 160 random 64/128/256-block patterns, B=2/3, GQA, deterministic, ragged L=42915, strided prefix views,
  overflow combs, exactly 15 runs, three patterns on three streams, CQS-off Lq != Lk incl. window (256,128): dk/dv
  `torch.equal` v11 everywhere; NaN-prefilled outputs on 64 real tasks + 200 edge configs: no unwritten element;
  census sweep over all 9 quorum sets, itr 1-3: every 128-aligned task routes ALIGNED with <= 2 runs per column,
  every seg_align=1 task routes LEGACY.

## 7. Review findings and resolutions

Four reviews (workflow wf_10003d64-e1a): v12f exactness (E-F), v12f methodology (M), v12b exactness (E-B), v12b
methodology (B).

| id | severity | finding | resolution |
|---|---|---|---|
| E-F1 | **blocker** | v12f census gate tested the `is_causal` argument; a causal call with a sliding window has `params.is_causal == false`, reached the non-causal ALIGNED kernel (no window, no causal mask): max error 3.6 (D64) / 2.7 (D128) on vis1. Production passes window (-1,-1), so no measured job was affected; `interface.py` exposes `window_size` | **fixed (b3)**: gate on `!cqs_is_local` (= `params.is_causal` or no window at all); every local call is LEGACY; regression `test_window_calls_route_to_legacy` (5 windows x D64/128, zero + real patterns, vs v11, forced LEGACY and a masked float64 reference) |
| E-F2 | minor | bit-identity holds only for finite V | **documented** (kernel header, this report 6.3), pinned by a test |
| E-F3 | minor (latent) | the non-MINIMAL `#else` dispatch routes CQS calls through the ALIGNED launcher with null census buffers | **fixed (b3)**: `#error` in that branch + a TORCH_CHECK that a route implies its summaries |
| M1 | minor | flash_attn package 5.3% slower than the in-tree kernel at causal D64 56K; offset is L-dependent | **resolved** (4.2): a first-arm clock artifact; warmed up, the package is 4.3% *faster* than in-tree; headline re-based to 0.766x flash_attn; in-tree ratios reported everywhere |
| M2 | minor | SDPA arm bimodal (19.64 median vs 17.95 min); flash_attn/SDPA factor 1.077 and "19.64 vs 19.70" do not hold | **withdrawn** in this report (4.2); F23 harness: 3 warm-ups + 10 runs, mins stored, FA arms re-timed at the end |
| M3 | minor | 0.939 mixes the tax removal with a 4.8% descending-order gain; the non-causal penalty was noise; order chosen on H=8 only | **reported separately** (4.1: tax 0.986, gain 0.952; non-causal neutral); H=32 (4.3): tax 0.982, gain 0.982 / 0.989: descending stays the causal default |
| M4 | minor | per-call overhead measured only down to L=6912; itr=2 tasks (L 1.3-2.9K) route ALIGNED | **measured** (job 14330595, 4.3): 7.3 us per call = 10.1% of v12f real at L~2K (estimate was 15-25%); real still 0.996x the flash_attn loop |
| M5 | minor | c=73 headline should be 1.202x monolithic flash_attn, not 0.881x per-task | **done** (1 and 4.1) |
| M6 | minor | Is_even_MN=false ALIGNED never timed | **measured** (4.3): zero 0.984x / real 0.762x in-tree plain on L=42787 vs 0.934 / 0.737 on the even L=42752 |
| E-B1 | minor | v12b tests miss overflow end to end, B>1, GQA, deterministic vs v11, strided views, 15 runs, CQS-off Lq != Lk | **added** (39 new tests) |
| E-B2 | minor | v12b host census gate tests the `is_causal` argument (the launcher re-checks, so the result was correct; a census was allocated and discarded) | **fixed (b5)** |
| E-B3 | minor | v12b's forward returns NaN for any real pattern (`-DCQSA_NO_ELEMENT_PATH`), nothing refuses the call | **recorded**: U1 builds the shipping module without the flag; a refusing TORCH_CHECK would need a host read of the pattern (a sync); v12b is never configured as `CQSA_CUDA_MODULE` |
| E-B4 | minor | `CQSA_V12B_DPSUM_SMEM` parse: any value not starting with '1' meant "global" | **fixed (b5)**: only "0"/"1", else TORCH_CHECK; tested |
| B1 | minor | deterministic 1.94x comes from the ceil grid (112 CTAs, 2 rounds), not the W*H caveat; memory not recorded | **explained** (5.1); peak extra memory 1701 vs 277 MiB (5.2); the floor grid is a U1 item |
| B2 | minor | "LEGACY = v11 within 0.1% everywhere" is false at L=13952 (+1.0%); the automatic LEGACY route (memset + census + ALIGNED no-op + LEGACY) never timed | **measured** (5.2): auto route 1.004x / 1.001x v11 at L=14043 / 56173; forced LEGACY at 13952 = v11 (0.999) |
| B3 | minor | fixed per-call cost only covered down to L=13952; engine tasks reach L ~ 4K | **measured** (5.2): 1.6-2.3% at L 3.6-5.1K; zero 1.010-1.021x NO_CQS, real 0.60-0.80x the flash_attn loop |
| B4 | minor | SDPA-based ratios flattered by 8-12% | **reported**: flash_attn and flash_attn-loop ratios are the headline (1, 5.1) |
| B5 | minor | dk/dv equality on `torch.empty_like` buffers can pass a kernel that skips writes | **fixed**: NaN-pre-filled buffers in harnesses and tests (6.2) |

## 8. Build, ptxas and SASS numbers

Builds on della-vis1 (nvcc 13.0, 80 cores shared with other agents' builds), kernel set `a100_fp16_hdim64_128`,
`MAX_JOBS=8`, `NVCC_THREADS=4`, `-lineinfo -Xptxas -v -time`:

| build | what | wall | notes |
|---|---|---|---|
| v12f b1 | clean | 1926 s | LEGACY units dominate: hdim64 causal/non-causal legacy 1335 / 1663 s (cicc 1027 / 1330 s, ptxas 289 / 315 s); main units 184-325 s |
| v12f b2 | flash_api.cpp change | 4763 s | the TAG-dependent `-time` csv path changed every nvcc command: full rebuild (build.sh now uses one csv path) |
| v12f b3 (F23) | flash_api.cpp + header comment | 3928 s (vis1 load 55-65) | the first build after the csv-path change rebuilds every unit once |
| v12b b1 | clean | 1697 s | LEGACY bwd units 1480-1530 s, main bwd ~600 s (vis1 shared) |
| v12b b3 / b4 / b5 | main units only | 235 / 207 / 407 s | b5 = F23 (flash_api.cpp + host code in `flash_bwd_aligned.h`) |

Static numbers (REG / STACK B / SASS instructions, sm_80, fp16, `results/ultimate/v12f_static.md`, `v12b_static.md`,
`cuobjdump_v11x.md`):

| instantiation | v11 CQS-on (= LEGACY) | v11x | new ALIGNED | CQS off / upstream |
|---|---|---|---|---|
| fwd hdim64 128x128 causal even_MN | 255/264/72776 | 255/112/5464 | **255/224/4112** | 255/216/4032 |
| fwd hdim64 128x128 non-causal even_MN | 255/640/69624 (v12f LEGACY own TU: 255/312/66992) | 255/96/5424 | **255/264/3792** | 255/224/3656 |
| fwd hdim128 128x64 causal even_MN | 255/256/57592 | 255/272/6680 | **255/152/5288** | 255/160/4984 |
| bwd hdim64 128x128 causal even_MN | 255/224/5624 | 255/112/2152 (NO_CQS) | **255/104/2320** (runs + dP_sum smem) | torch-FA2 255/96/2336 |
| bwd hdim128 64x128 causal even_MN | 255/96/4168 | 255/88/1968 | **255/72/2160** | torch-FA2 255/64/2024 |

ALIGNED ptxas time 0.6-1.7 s per instantiation (forward), 1.1-1.4 s (backward), against minutes for the LEGACY
units: the per-element path is 88-93% of the forward's CQS-on SASS. Both ALIGNED kernels sit at upstream's size
(forward 3.8-7.2K vs CQS-off 3.3-6.7K SASS; backward 2.1-2.8K vs torch-FA2 2.3K), with spills in upstream's band.

## 9. Phase 4 plan (packages U1, U2; D1 runs alongside)

**U1 -- one classic CUDA module with both kernels, wired into the engine** (log `progress/U1.md`).
- Build `next/kernel/v13` = v12f's forward (census + ALIGNED + LEGACY in separate TUs, descending causal default) +
  v12b's backward (census + ALIGNED + LEGACY, dP_sum smem default for CQS-on), with **neither**
  `-DCQSA_NO_ELEMENT_PATH` **nor** `-DCQSA_BWD_NO_CQS` (LEGACY keeps v11's element path for mixed tiles); module
  `cqsa_cuda_next_v13`. Kernel set first `a100_fp16_hdim64_128`, then `common` (fp16+bf16, hdim 64+128, causal and
  non-causal): the common build **retires the second extension** (`csrc_nc` = v9) once D128 mixed non-causal is
  verified. Keep forward and backward census constants distinct (the `Bwd` infix). Carry the F23 fixes (both gates,
  the `#error`, the env parse). Candidate small items: the floor grid for deterministic mode (B1), folding the route
  memset into the census and skipping the no-op LEGACY launch when the host knows the layout is aligned (if the
  F23 small-L rows show the per-call floor matters).
- Wire it on branch `ultimate`: `stream_cqsa/interface.py` (`CQSA_CUDA_MODULE` default -> v13 when importable,
  fallback `cqsa_cuda`; `CQSA_CUDA_MODULE_NONCAUSAL` unneeded; backward default switches from Triton to the CUDA
  backward when v13 is present), `native_autograd.py` if needed, the release `setup.py` compiling csrc from v13's
  sources (`csrc_nc` out of the build).
- Correctness: `tests/test_ultimate_v13.py` (both directions; aligned, legacy, random, ragged; causal / non-causal;
  D64/128; fp16 AND bf16; windows; NaN-prefilled outputs) + the full existing suite on vis1 + one gpu-test suite job
  (the 12 bf16 failures of the fp16-only dev module must pass). Performance: one SXM job `engine_ab.py` (engine-level
  forward and backward at N=131072 c=7/c=73, N=1048576 c=7 itr 1-2, bf16 at 131K; v2.2.3 vs ultimate+v13 vs
  monolithic flash_attn / SDPA); design acceptance: engine backward 1M itr=1 <= 24 s (today 39.1 s CUDA / 29.0 s
  Triton).

**U2 -- the census + ALIGNED design in the native wave kernel** (log `progress/U2.md`).
- Dev copy `next/native_v2` (module `cqsa_native_v2`) of `native/csrc/flash_attn` (v11 + wave modes: Cqs_mode 3
  causal uniform layout, grid (S/128, W, H); mode 2 varlen non-causal; `bwd_wave`; `wave_merge` /
  `wave_scatter_add`). Census per (subproblem, row block) forward and per (subproblem, column block) backward, LEGACY
  per subproblem when it has mixed tiles, uniform layout + in-place block-map deltas per run step, so the classic call
  is a W=1 wave at no penalty (design 4.5); hdim 64 and 128 (the hdim128 wave modes never compiled before because of
  the element path's size: now they should), causal and non-causal (mode 4), fp16 and bf16.
- Correctness `tests/test_ultimate_wave_v2.py` (wave vs v12f/v12b per task bit-identical where the design says so,
  W in {1, 7, 73}, in-place vs gathered, ragged N, both causal modes, D64/128, fp16/bf16, vs fp64). One SXM job
  `wave_ab.py`: targets W=1 within 1% of v12f/v12b; c=73 one-launch forward <= 1.10x monolithic flash_attn (the
  classic per-task loop is 1.202x today, section 4.1); wave backward >= 2x faster than today's `bwd_wave`. Then plug
  into `native_wave.py` (module switch with fallback to `cqsa_native`).
- What F23 hands U2: the run-table and census designs above, the per-call fixed cost at small L (F23 job), the
  finite-V caveat (wave vs per-task identity must be stated for finite inputs), and the review's lesson that a new
  loop must be derived from upstream's exact text (P2) and gated on post-`set_params` values (E-F1).

**D1** (multi-device scheduler, running now) needs a row-span (forward) / column-span (backward) launch of one task
to split the largest task; neither v12f nor v12b has it (design 4.2's `[m_lo, m_hi)` row-range launch is not built).
The census already works per row / column block, so a span is a grid restriction plus a lse/acc merge of the parts:
a U1/U2 follow-up if D1's refinement needs it.

**Open items carried forward**: D128 mixed non-causal row (retire v9 everywhere); bf16 and the `common` kernel set
(never built for v12f/v12b); engine integration and engine-level acceptance (U1); the wave port (U2); the row-range
launch (design 4.2); 128x64 tiles (K3); deterministic floor grid (B1).
