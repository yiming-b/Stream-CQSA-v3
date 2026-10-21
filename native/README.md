# native/ -- the multi-subproblem CQS kernel ("wave" kernel), v2

The wave kernel runs W >= 1 CQS subproblems in ONE launch. Version 2 (ultimate program package U2,
`docs/ultimate/progress/U2.md`; design `docs/ultimate/01_design_a100.md` 4.5) is the classic module's
census + ALIGNED run-table design (`kernels/v12f`, `kernels/v12b`, `docs/ultimate/03_kernels_report.md`)
on the uniform wave layout: every subproblem padded to S = ceil(max L / 128) * 128 rows, grid
(S/128, W, H), per-subproblem tables at `bidb * stride`, real lengths in `lens[W]`. These sources are
`kernels/native_v2/` (the development tree with its own build); the release `setup.py` builds them as
`cqsa_native` with `CQSA_BUILD_NATIVE=1` (or alone as the companion wheel, `CQSA_ONLY_NATIVE=1`).

    csrc/flash_attn/
      src/flash.h                 Flash_fwd_params: cqs_wave_* strides, cqs_wave_lens, the census route / run tables
      src/cqs_census.cu           forward census: one warp per (subproblem, row block) -> route[w], runs of live tiles
      src/cqs_census_bwd.cu       backward census: per (subproblem, column block)
      src/flash_fwd_aligned.h     ALIGNED wave forward (v12f's loop; per-step in-place block-map delta; fp32-only epilogue)
      src/flash_bwd_aligned.h     ALIGNED wave backward (v12b's loop)
      src/flash_fwd_kernel.h      LEGACY forward (v11's loop with the per-subproblem offsets) for subproblems with mixed tiles
      src/flash_bwd_kernel.h      LEGACY backward
      src/flash_*_sm80.cu         one main unit (plain + ALIGNED) and one *_legacy unit per (dtype, hdim, causal)
      src/wave_kernels.cu         wave_merge (deterministic batched merge), wave_scatter_add
      flash_api.cpp               fwd_wave / bwd_wave (uniform layout, lens on the CPU or on the device with lens_even),
                                  the classic entries (a W=1 wave), fwd_cqs_census / bwd_cqs_census for tests
    setup.py                      standalone build of this tree (module cqsa_native_v2; CQSA_KERNEL_SET common | native_v2_dev)
    build.sh                      detached build on della-vis1 with a watchdog scoped to its own ptxas processes
    build.slurm                   the same on a CPU node

Engine: `stream_cqsa/native_wave.py` (wave_forward, wave_backward, wave_attention, ChunkPool); it loads
`cqsa_native_v2` when importable, else the first wave module `cqsa_native` (WAVE_API 1: v11's loop, uniform
layout for causal and varlen for non-causal, hdim64 forwards only), whose sources live in
`/scratch/gpfs/AKEY/yb2807/Stream-CQSA-dev/next/native`.

Tests: `tests/test_wave_v2.py` (the wave vs the per-task v12f / v12b modules bit for bit, LEGACY routing, the
engine vs float64, the chunk pool and the host accumulator). Benchmark: `benchmarks/ultimate/wave_ab.py`
(`slurm/ultimate/parts/u2.sh` in the combined ultimate job; `results/ultimate/wave_ab.md`).
