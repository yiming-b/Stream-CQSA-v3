# Experiment-build A/B (P1.1): v11 vs v11x vs v9 vs SDPA-flash

Job **14308401**, della-l07g4, **NVIDIA A100-SXM4-80GB**, torch 2.10.0+cu130, fp16, H=8, 1 warm-up + 5 timed runs
(median; the min is within 1% of the median in every row), one process, 25 s. FA-2 arm = torch SDPA with the
FLASH_ATTENTION backend forced (`pytorch_flash::flash_fwd_kernel<...>` recorded by the profiler; flash_attn not
installed, ~5% slower than the package). Harness `benchmarks/ultimate/exp_build_ab.py` (`slurm/ultimate/exp_build_ab.slurm`),
raw rows `exp_build_ab.json`, log `exp_build_ab_14308401.out`. v11x = `next/kernel/v11x` = v11 + `-DCQSA_NO_ELEMENT_PATH`
(forward per-element mask path compiled out, poison on any tile that needs it) + `-DCQSA_BWD_NO_CQS` (backward CQS
compiled out); its build and static side are in `cuobjdump_v11x.md`. MaxRSS of the job: 1.27 GB (24G requested).

## Forward: aligned c=7 task, L=56064 (N=131072, `seg_align=128`), ms (ratio vs SDPA-flash)

| config | SDPA | v11 plain | v11 zero | v11 real | v11x plain | v11x zero | v9 plain | v9 zero | v9 real |
|---|---|---|---|---|---|---|---|---|---|
| causal D=64 | 19.70 | 17.30 (0.878) | 22.55 (1.144) | 17.85 (0.906) | 17.38 (0.882) | 19.35 (0.982) | - | - | - |
| non-causal D=64 | 34.17 | 33.16 (0.970) | 195.82 (5.731) | 151.17 (4.424) | 33.51 (0.981) | 34.73 (1.016) | 42.42 (1.242) | 47.28 (1.384) | 38.15 (1.116) |
| causal D=128 | 32.38 | 29.71 (0.917) | 40.52 (1.251) | 31.84 (0.983) | 29.82 (0.921) | 34.08 (1.053) | - | - | - |
| non-causal D=128 | 59.71 | 56.66 (0.949) | 248.16 (4.156) | 194.37 (3.255) | 57.26 (0.959) | 64.34 (1.078) | 82.70 (1.385) | 99.46 (1.666) | 75.91 (1.271) |

Identity: v11x plain and zero are bit-identical to v11 (out and lse, maxdiff 0.0) in all four configs; v9 is
bit-identical to v11 on plain, zero and real, non-causal, both head dims. v11x real (not timed): NaN on exactly
the predicted rows (causal: every non-owner row block, 66.7%; non-causal: the blocks sharing the last column
tile's chunk, 33.3%), the other rows equal v11 (`poison_ok` true in all four configs).

### The forward tax split (all-live tax = zero / plain of the same module)

| config | v11 zero/plain | v11 tax ms | per-element path's presence (v11 zero - v11x zero) | live loop's own loads + bookkeeping (v11x zero - v11x plain) | v11x zero/plain |
|---|---|---|---|---|---|
| causal D=64 | 1.303 | 5.25 | 3.20 ms (61%) | 1.97 ms (39%) | 1.113 |
| causal D=128 | 1.364 | 10.81 | 6.44 ms (60%) | 4.26 ms (40%) | 1.143 |
| non-causal D=64 | 5.905 (mis-compile, F15) | - | - | 1.22 ms | 1.036 |
| non-causal D=128 | 4.380 (mis-compile) | - | - | 7.08 ms | 1.124 |

1. **Three fifths of the all-live tax is the per-element path's presence, two fifths the live loop itself.**
   Compiling the path out takes v11's zero pattern from 22.55 to 19.35 ms at L=56064 causal (x0.858; hdim128
   40.52 to 34.08, x0.841) with nothing else changed, bit-identically; what remains over plain is 1.11x (hdim64) /
   1.14x (hdim128): the verdict loads, the `while` search and the flag, i.e. the "in-loop verdict loads +
   integer/branch bookkeeping" of `ncu_aligned.md`. Applied to layout_ab's basis (zero/F 1.264 at 56K), the
   outlined colmask alone (K1's "per-element path leaves the hot loop", design 4.3) is worth ~1.085 and the run
   table (4.2) the rest of the way to P3.1's gate (zero/F <= 1.15).
2. **The non-causal pathology is code size too.** v11's non-causal CQS-on instantiation runs at 5.9x plain
   (195.8 ms; hdim128 4.4x); v11x's, same source minus the element path, at 1.036x plain (34.73 ms) and 1.016x
   SDPA. It is 1.36x faster than v9's production non-causal kernel on the zero pattern (47.28 ms) and 1.24x on
   plain (v9's CQS-off path is itself 1.24x SDPA); hdim128 1.55x (64.34 vs 99.46). With `cuobjdump_v11x.md`
   (ptxas 23-75 s per TU, 5.4K vs 69.6K SASS), the TU split (K2) is not what retires F15; the second extension
   (`csrc_nc` = v9) can go as soon as the unified kernel keeps the element path out of the CQS-on instantiations.
3. On the unaligned L=56175 rows of the backward part (fwd columns): v11 plain/zero/real 0.936/1.269/1.340 vs
   v11x plain/zero 0.935/1.051.

## Backward: classic layout path (0,), causal, D=64, ms (ratio vs SDPA-flash backward)

| L | SDPA bwd | v11 plain | v11 zero | v11 real | v11x plain (NO_CQS) | v11x zero | v11x real (timing only) |
|---|---|---|---|---|---|---|---|
| 14044 | 3.22 | 5.93 (1.840) | 6.49 (2.015) | 6.26 (1.942) | 2.97 (0.923) | 2.98 (0.926) | 2.98 (0.926) |
| 28088 | 12.08 | 22.64 (1.874) | 24.78 (2.051) | 23.12 (1.914) | 11.36 (0.940) | 11.37 (0.941) | 11.39 (0.943) |
| 56175 | 47.03 | 87.61 (1.863) | 96.14 (2.044) | 88.84 (1.889) | 44.53 (0.947) | 44.55 (0.947) | 44.55 (0.947) |

Identity: v11x dk/dv `torch.equal` to v11's on plain and zero at every L; dq differs by 1.2e-4 (max abs, the
fp32 atomics' order). Rerun drift of every arm: dq 1.2e-4 (6.1e-5 once), dk/dv 0.0 (F6b). v11's zero/real
ratios reproduce layout_ab's F6 (1.99-2.04 / 1.89-1.93).

Decomposition at L=56175, zero/F = 2.044 = (NO_CQS/F) x (plain/NO_CQS) x (zero/plain) = **0.947 x 1.967 x 1.097**;
real/zero = 0.924. The whole 2x is the *presence* of the CQS code in the fused kernel: with CQS merely switched
off at runtime (v11 plain) the kernel is already 1.86x, the verdicts on an all-clear pattern add +9.7%, and
skipping dead tiles nets -7.6%. Statically (`cuobjdump_v11x.md`): 5624 vs 2152 SASS instructions and STACK 224 vs
112 B for the same instantiation, both at REG 255.

### Route decision for P2.1 (design 4.7 item 7)

NO_CQS backward / SDPA-flash backward = **0.923, 0.940, 0.947** at L = 14044, 28088, 56175, all <= 1.25:
**fix the fused kernel.** Upstream's 5-GEMM loop with our contract (global lse in, caller-supplied dsoftmax_sum,
fp32 dq_accum) is 5% faster than SDPA-flash's backward, so the 2x is entirely inside our additions to it (the
runtime verdicts, the guard structure, the pointer bookkeeping, the numden branch: 4.7 items 1-4); the Triton
two-kernel port is not taken.

## Caveats

- Cross-job: the SDPA arm ran 19.70 ms at L=56064 causal here against 18.36 ms at L=56175 in layout_ab (job
  14249868, della-l07g7), so this job's vs-SDPA ratios sit ~7% below layout_ab's for the same kernels (v11 zero
  22.55 ms here vs ~23.2 there). The split and the route rest on in-job kernel-vs-kernel ratios and do not move
  with it; the flash_attn re-base (design 3.3) applies when job 14240707 lands.
- NO_CQS is a measurement build: its real-pattern backward ignores the mask (dk/dv differ from v11 by 0.1-0.3),
  and the forward poisons real patterns by design.
