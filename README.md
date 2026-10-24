# Stream-CQSA v3

Exact out-of-memory recovery for attention. When an attention call does not fit
in device memory, Stream-CQSA decomposes its pair set over a cyclic quorum set
into independent subproblems, runs them one at a time (or a few at a time, or
on several devices), and recomposes the exact result. It defines no attention
rule of its own: it reproduces whatever kernel it wraps.

Paper (v1): [arXiv:2604.20819](https://arxiv.org/abs/2604.20819) ·
v1 repo: [yiming-b/Stream-CQSA](https://github.com/yiming-b/Stream-CQSA) ·
v2 repo (released, PyPI `stream-cqsa` 2.2.3): [yiming-b/Stream-CQSA-v2](https://github.com/yiming-b/Stream-CQSA-v2)

**v3 is in development** (version `3.0.0.dev0`, not on PyPI yet). It is v2.2.3 plus the kernel and engine
work of the "ultimate A100 kernel" program (`docs/ultimate/`): new forward and backward kernels that remove
the decomposition's overhead on the live tiles, a wave kernel with the same design, 128-aligned chunk
layouts, routing by a memory model, a multi-device scheduler for mixed GPUs, and one experiment plan
(`experiments/v3/PLAN.md`) whose results are the measurement sections below (E0-E6, E8 measured on
A100-SXM4-80GB in jobs 15966097-16119471, E2's 4M/8M rows on A100 80GB PCIe; `results/v3/STATUS.md` says which
rows exist; E7 is still **[E7: pending]**).

## How it works

![Stream-CQSA: data movement in the forward and backward](docs/stream_cqsa_demo.gif)

Q/K/V live in host memory in 7 chunks. Each of the 7 subproblems gathers 3 chunks
(its owner chunk and two quorum partners) to the device, runs the CQS kernel on
them, and hands back a partial result: `(out_i, lse_i)` in the forward, which are
merged into a host accumulator with the same max-shifted arithmetic FlashAttention
uses inside one kernel; `(dq_i, dk_i, dv_i)` in the backward, computed against
the *global* log-sum-exp and scatter-added into host gradient buffers. Only one
subproblem's inputs and partial result are on the device at a time, and every
kept query–key pair is counted exactly once, so the result is exact.

## What is new in v3

| area | v3 | measured (A100-SXM4-80GB, fp16, B=1 H=8 D=64, causal; `results/ultimate/`) |
|---|---|---|
| **forward kernel** | a census kernel classifies every 128x128 tile on the device, then an ALIGNED run-table kernel (upstream FlashAttention-2's loop walking the live tiles, no per-tile verdicts) runs tasks without mixed tiles and v11's code stays as the LEGACY path; one stream, no host sync (`csrc/`, `docs/ultimate/03_kernels_report.md`) | 56K-token task: **0.77x flash_attn** (v2's kernel: 1.04x); the all-live tax 1.30x -> 0.99x; non-causal 1.46-1.75x faster than v2's separate kernel; bit-identical to v2's kernel |
| **backward kernel** | the same design per column block; dP_sum staged in shared memory; deterministic dq available | 56K-token task: **0.78x flash_attn**, **2.48x faster than v2** (dk/dv bit-identical, dq within the atomics' drift) |
| **one extension** | causal and non-causal, fp16 and bf16, head dims 64 and 128 in one module (`cqsa_cuda_v13`); the CUDA backward is the default again; v2's second non-causal extension is gone | engine, 1M tokens, 7 subproblems: forward **0.98x** a single flash_attn call (v2.2.3: 1.30x), forward+backward **15.2 s vs 15.1 s** (v2.2.3: 39.2 s CUDA / 29.0 s Triton) |
| **wave kernel** | the census + ALIGNED design on the uniform wave layout (several subproblems per launch, `native/`), head dim 128 now compiles | 73 subproblems in one launch at 131K: **1.02x** one flash_attn call forward, 1.02x backward (v2's wave: 1.42x / 2.4-2.6x slower) |
| **128-aligned chunks** | every task's chunk boundaries fall on 128-token blocks, at every depth, so no tile straddles a chunk (v2: 1.365x -> 1.004x flash_attn at 56K with the SAME kernel) | engine 131K c=7: 150 -> 135 ms; c=73: 222 -> 179 ms |
| **routing by a memory model** | the planner models the wave's peak (`a*N + b*W`) and chooses the engine, the wave budget W and gathered-vs-in-place reads; the subproblem-count rule is gone | — |
| **async host fold** | the wave engine's host accumulator folds on a worker under the next wave | host-streamed 1M: 9.0 s (classic 9.3 s); 2M: 34.2 vs 34.6 s |
| **multi-device scheduler** | per-GPU rates (calibrated per model), exact live-tile work per task, LPT with a cost-aware split, reduce_scatter merge; identical schedule on every rank; heterogeneous devices supported (`stream_cqsa/scheduler.py`) | model within 2% rms of the measured 4x A100 job; the measured winner reproduced on every shape (`results/ultimate/dist_sched.md`); on 2 SXM + 2 PCIe A100s (job 15949746, `results/ultimate/dist_het.md`) LPT reaches 2.91x one GPU forward and 3.5-3.7x forward+backward at 2M where round-robin reaches 2.75x / 2.88x |
| **correctness discipline** | every comparison against SDPA / FA-2 / fp64 forces a decomposition and asserts it ran (`tests/_decomp.py`; an unpinned call that fits is SDPA itself) | 745 tests |
| **fixes** | a pinned `c=` without `interest_set=` silently used the c=7 quorum set in v2.1-v2.2.3 (wrong results for every c != 7; v3 re-plans the set and the engine refuses a non-perfect-difference-set pair); the autograd forward moved an 8 GiB fp32 host result to the device at 4M tokens (the 10 GiB-budget backward OOM in the OOC study) | `docs/ultimate/progress/BUG_v223_c73.md` |

### What v2 brought (released as 2.2.x)

| area | v2 |
|---|---|
| **native kernel** | second-generation forward kernel: 19% faster than v1 on the real workload (L=131K causal, A100), and with CQS off it now equals FlashAttention-2 (`docs/kernel_technical_note.md`) |
| **engine** | pipelined host accumulator (d2h + merge overlap the next kernel), contiguous-chunk DMA (`shared_chunks`), measured concurrency policy; 1.44–1.86x faster end to end than v1 on one A100 |
| **multi-device** | `stream_cqsa.distributed`: exact sharding of the subproblems over a `torch.distributed` group; 3.2–3.5x on 4 GPUs, 5.1x on 8 GPUs / 2 nodes |
| **automatic configuration** | `stream_cqsa.autoconfig`: monolithic vs decomposed, depth `itr`, quorum set `(c, interest_set)`, accumulator placement, host residency, concurrency and device count chosen from a hardware description and a calibratable cost model |
| **independent fwd/bwd** | the backward plans its own decomposition depth (`bwd_itr="auto"`) |
| **developer kit** | `stream_cqsa.devkit`: plug any inner kernel into the framework and get exactness (bit-identical / within rounding / not) and performance against its monolithic call; `quick_bench` sweeps |
| **Triton kernels (no build)** | `stream_cqsa.triton_kernel`: forward AND backward CQS kernels in Triton with the CUDA kernels' contract, selected automatically when no extension is compiled (`CQSA_BACKWARD=triton` forces the backward). Forward: 0.75–0.91x the CUDA kernel at L=56K (faster), within 13% at 899K; non-causal it beats FlashAttention-2 itself on the same L. Backward: 0.77x the CUDA CQS backward at L=56K, engine backward at 1M 29 s vs 39 s. Engine forward at 1M: 1.15x the CUDA path (`docs/LOG.md`, Phase 4) |
| **native wave kernel** (`native/`, `stream_cqsa.native_wave`) | a CQS kernel that runs *several subproblems per launch*: the wave is one batched launch with per-subproblem group bits, tile summaries and a block map that reads the original Q/K/V in place (no gather); one deterministic merge kernel recomposes the wave. 9-30% faster than the stream-based engine at N=131K (more subproblems, more gain), bit-identical to the v11 kernel per subproblem; forward + backward (`docs/LOG.md`, Phase 6) |
| **adapters** | `stream_cqsa.adapters`: automatic conversion of FlexAttention-expressible kernels (ALiBi, windows, soft-cap, document masks) into inner kernels, with global-position remapping |

## Install

**v3 (this repo, development):** source only until 3.0.0 is released. In a checkout:
`pip install -e . --no-build-isolation` compiles the classic module (`csrc/`, one extension; `CQSA_KERNEL_SET=common`
= fp16+bf16, head dims 64+128, causal + non-causal; 30-75 min of nvcc) and `CQSA_BUILD_NATIVE=1` adds the wave kernel
(`native/`); `CQSA_SKIP_EXT=1` installs the Triton-only build with no compilation. **[release: pending]** -- the
3.0.0 wheels (PyPI pure wheel; CUDA wheels per python / torch / CUDA / GPU on the release page) follow the
experiment results.

**v2.2.3 (released):** two implementations of the kernels ship, and where you get them differs:

| | kernels | where | install |
|---|---|---|---|
| **Triton build** | CQS forward and backward in Triton, compiled on your machine for your GPU; no compiler, no torch/CUDA matching | PyPI | `pip install stream-cqsa` |
| **CUDA build** | the classic CUDA extensions (FlashAttention-2-derived CQS forward, CUDA backward) plus the native wave kernel; the fastest forward | GitHub release page (prebuilt wheels per python / torch / CUDA / GPU) or a source build | see below |

The package picks at run time: CUDA extensions and the wave kernel when they are importable,
Triton otherwise. `stream-cqsa-doctor` shows which kernels loaded. `stream_cqsa.place_inputs(q, k, v)` moves host tensors to the GPU only if they fit and keeps the rest pinned
in host memory (a guardrail for shared GPUs); `verbose=True` prints every automatic choice. Since v2.2.0 the wave engine
also runs on the Triton kernel (`attention(..., kernel="wave-triton")`; `"wave-cuda"` pins the
CUDA one), supports the host accumulator (`accumulate_on_gpu=False`) on both, and the quorum-set
register includes c=3 and c=133; the automatic planner enumerates every registered set. In v2.2.x the Triton backward was the
default in both builds (it measured faster than v2's CUDA backward at every length);
in v3 the CUDA backward of the new module is the default and `CQSA_BACKWARD=triton` selects Triton.

### Triton build (PyPI)

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126     # match your CUDA
pip install stream-cqsa                                                  # Triton kernels, no compilation
python -m stream_cqsa.doctor                                             # what this machine can run, and how far
```

Triton is installed with the package (`triton` on Linux, `triton-windows` on Windows, where the official
package has no wheels). `pip install flash-attn` is optional and
gives the monolithic fast path its own kernel. The same build installs from a checkout without
compiling: `CQSA_SKIP_EXT=1 pip install --no-build-isolation git+https://github.com/yiming-b/Stream-CQSA-v2.git`.

### CUDA build (GitHub release page)

The [release page](https://github.com/yiming-b/Stream-CQSA-v2/releases/tag/v2.2.3) carries, for
every python 3.10/3.11/3.12 x torch 2.5.1/2.6.0 x CUDA 12.4 x GPU architecture (sm80 = A100 and
other 8.x cards, sm90 = H100) cell, two wheels: `stream_cqsa` with the classic CUDA extensions and
the companion `stream_cqsa_native` with the wave kernel. Install both; they replace the PyPI
package with the same version plus the extensions:

```bash
V=v2.2.3; T=1cu124torch2.6sm80; PY=cp311     # pick your torch / CUDA / GPU / python
pip install https://github.com/yiming-b/Stream-CQSA-v2/releases/download/$V/stream_cqsa-2.2.3-$T-$PY-$PY-linux_x86_64.whl
pip install https://github.com/yiming-b/Stream-CQSA-v2/releases/download/$V/stream_cqsa_native-2.2.3-$T-$PY-$PY-linux_x86_64.whl
```

Also on the release page: one wheel with both extensions for torch 2.10 / CUDA 13 / python 3.11
(`stream_cqsa-2.2.3-1cu130torch210sm8090-...whl`, sm80 + sm90), and the pure wheel and sdist that
PyPI serves. The CUDA wheels are too large for PyPI (100 MB per file), which is why they live on
GitHub, as PyTorch's own CUDA wheels do. They are built by `.github/workflows/wheels.yml` on every
tag; publishing to PyPI is the `pypi` job, enabled by the repository variable `PYPI_PUBLISH=true`
with trusted publishing.

From source, for any GPU: `pip install -e . --no-build-isolation` in a checkout compiles the
classic extension for the card present (40-75 min of nvcc; `CQSA_KERNEL_SET=common` covers
fp16/bf16 and head dims 64/128); `CQSA_BUILD_NATIVE=1` adds the wave kernel.

## Use

One function, with the signature of `torch.nn.functional.scaled_dot_product_attention`:

```python
import torch, stream_cqsa

q, k, v = (torch.randn(1, 8, 4_000_000, 64, dtype=torch.float16) for _ in range(3))   # host or device tensors
out = stream_cqsa.attention(q, k, v, is_causal=True)          # exact, always fits
```

Below the memory boundary the call *is* the monolithic kernel; above it the planner chooses the
decomposition from the free memory and the call runs exactly, on the best kernel installed, recovering
from out-of-memory on its own. Gradients flow when the inputs require them. There is nothing else to learn;
the rest is optional:

| I want to... | how |
|---|---|
| see what it decided and what ran | `attention(..., verbose=True)`, or `return_plan=True` / `return_info=True` |
| know before running a huge call | `stream_cqsa.estimate(N=16_777_216)` (memory, configuration, time; nothing is allocated) |
| be safe on a shared or small GPU | create inputs in host memory, or `q, k, v = stream_cqsa.place_inputs(q, k, v)` (moves what fits) |
| run inside a model where Q/K/V fill the GPU | nothing: `attention` releases Q/K/V to pinned host memory when nothing fits next to them, puts the output on the GPU first, and brings Q/K/V back if room remains (`evict_inputs=`, `restore_inputs=` = `"auto"`｜`True`｜`False`; `del` a fused QKV tensor first so the memory can be freed) |
| describe the budget instead of detecting it | `attention(..., hardware={"cuda:0": "10GiB", "host": "64GiB"})` |
| route an existing model without touching it | `stream_cqsa.patch_sdpa()` or `with stream_cqsa.patched_sdpa(): model(...)` |
| pin a setting | keywords of `attention`: `itr=`, `c=`/`interest_set=`, `accumulate_on_gpu=True`｜`False`, `max_parallel=`, `bwd_itr=`, `kernel="cuda"｜"triton"｜"wave-cuda"｜"wave-triton"` |
| trade differently between time and device memory | the planner takes the configuration with the smallest device peak among those predicted within 15% of the fastest (so from about 1M tokens on an A100 the fp32 accumulator -- the forward's output, the backward's gradients -- lives in host memory: 10-15% slower, 40-70% less device memory); `attention(..., tol=0)` or `CQSA_PLAN_TOL=0` takes the fastest, `accumulate_on_gpu=True` pins the device accumulator, `tol=0.3` leans further toward memory |
| make the planner's times accurate for my GPU | `stream_cqsa.calibrate()` once (about a minute; cached in `~/.cache/stream_cqsa`) |
| check an installation | `stream-cqsa-doctor` |

Everything else in the package is the engine layer that `attention` is built on, for kernel and systems
developers: `stream_cqsa_forward` / `stream_cqsa_backward` / `stream_cqsa_attn` (the streaming engine),
`native_wave` (the wave engine), `autoconfig.plan` / `autotune`, `devkit.compare_kernels(inner_fn, mono_fn)`,
`adapters.flex_inner(score_mod, extra_mask_mod)`, and `distributed.dist_stream_cqsa_forward/_backward` under
`torch.distributed`. `attention_oom_safe`, `stream_cqsa_auto` and `auto_attention` are legacy entry points kept
for old code; they emit a `DeprecationWarning` pointing to `attention`.

`sbatch slurm/quickstart.slurm` runs all of the above on one GPU (`benchmarks/quickstart.py`;
output in `results/quickstart/`): the dry run, the monolithic path, the decomposed path under
a memory cap with progress, host-resident inputs, autograd, and a patched module.

Notebooks (executed, outputs included): `notebooks/stream_cqsa_v2_demo.ipynb` shows the one
function (Part A) and the engine layer under it (Part B) on one GPU; `notebooks/oom_boundary_demo.ipynb` sweeps N explicitly under a memory cap
and shows the baseline matching Stream-CQSA below the boundary and OOM-ing above it while
Stream-CQSA continues, exact.

## Layout

    stream_cqsa/      the package (engine, planner, devkit, adapters, distributed, autograd)
    csrc/             the classic CUDA kernel (FlashAttention-2 + CQS: a census routes every call to the ALIGNED
                      run-table kernel or the LEGACY per-element path, forward and backward, causal and non-causal);
                      = kernels/v13. csrc_nc/ (v2's separate non-causal build) is retired and no longer built
    native/           the multi-subproblem "wave" kernel (cqsa_native): sources, build, tests, benchmark
    benchmarks/profile_sweep.py   one-parameter profiles (c; N x {itr, acc}) with quadratic/linear fits: results/profile_sweep/
    tests/            pytest suite (engine, planner, devkit, adapters)
    benchmarks/       kernel A/B, pipeline A/B, end-to-end, quick bench, ncu driver, quorum axis
    distributed/      multi-device tests (2/4/8 GPUs)
    slurm/            job templates, incl. slurm/paper/ (the paper's experiments re-run with v2)
    notebooks/        the demo notebook and its generator
    docs/             kernel technical note, design notes (kernel suite / automatic conversion),
                      the full engineering log (LOG.md), per-step kernel diffs
    kernels/          the kernel variant sources of the program (v12f forward, v12b backward, v11x experiment)
    experiments/v3/   PLAN.md (the experiment plan) and bench.py (the one harness: run / report / status)
    slurm/v3/         the resubmittable experiment jobs (E_sxm.slurm, E_pcie.slurm)
    docs/ultimate/    the kernel program: 00_map (design basis), 01_design, 02 and 03 reports, progress logs
    results/v3/       every v3 measurement: results.jsonl, report.md, STATUS.md, figures
    results/          v2-era measurements referenced in the docs (JSON + slurm logs), incl. results/ultimate/

## Measurements

All v3 numbers come from one plan, `experiments/v3/PLAN.md`, one harness (`experiments/v3/bench.py`) and one
results directory, `results/v3/` (`results.jsonl` = every row; `report.md` = the tables and fits; `STATUS.md` =
which rows exist; `E*.png` = the figures). Every row records the GPU, the engine and kernel that ran, the
subproblem count, wall-clock (median of 3 after a warm-up), peak device memory, peak host memory and the
error against FlashAttention-2 and against float64. Every Stream-CQSA row pins the depth and asserts that
the decomposition ran. Node types are never mixed within a table: E0-E6 are **A100-SXM4-80GB**, the 8M rows
and E7 are **A100 80GB PCIe**.

### Headline (job 15958597, A100-SXM4-80GB, fp16, B=1 H=8 D=64, causal, device-resident Q/K/V, 7 subproblems)

| N | direction | flash_attn (one call) | v2.2.3 | v3 | v3 / flash_attn |
|---|---|---|---|---|---|
| 1M | forward | 5.99 s | 7.79 s (1.30x) | **5.84 s** | **0.98x** |
| 1M | forward + backward | 15.1 s | 39.2 s CUDA / 29.0 s Triton | **15.2 s** | **1.01x** |
| 131K | forward | 90 ms | 152 ms (1.69x) | 110 ms | 1.22x |
| 131K | forward + backward | 238 ms | 643 ms CUDA / 503 ms Triton | 268 ms | 1.13x |

The decomposition does no extra attention work (perfect-difference-set quorums cover every chunk pair
exactly once), so at 1M the launches, gathers and the merge are the whole overhead and vanish next to seven
0.8-s kernels; at 131K they are a visible share. Full rows (bf16, c=73, itr=2, the wave engine, SDPA):
`results/ultimate/engine_ab.md`; the kernel-level numbers: `docs/ultimate/03_kernels_report.md`.

### E0 correctness (job 15966097, A100-SXM4-80GB)

Every series (classic CUDA / Triton, wave CUDA / Triton) x c in {3, 7, 31, 133} x itr {1, 2} x accumulator
{GPU, host} x {forward, forward+backward} at N=256K, plus bf16 and head dim 128: output and gradient errors
against FlashAttention-2 and against float64. Every row decomposed (asserted) and every row is exact: the
output's error against float64 is 0.91-1.00x FlashAttention-2's own, the gradients' at most 1.24x (fp16
rounding of the gradient accumulation at c=133, itr=2), bf16 ~8e-4 as the format allows.

<!-- v3-tables:E0 -->
128 of 128 rows ran and are exact; the worst errors per series (the reference is FlashAttention-2's output and gradients; the fp64 columns compare both to a float64 SDPA):

| series | rows | max rel vs FA-2 (output) | max rel vs FA-2 (grads) | max fp64 err (output) | max fp64 err / FA-2's | max fp64 err (dQ) | max fp64 dQ err / FA-2's |
|---|---|---|---|---|---|---|---|
| classic_CUDA | 32 | 1.0e-03 | 2.8e-03 | 2.2e-04 | 1.00 | 2.6e-03 | 1.24 |
| wave_CUDA | 32 | 1.0e-03 | 3.0e-03 | 2.2e-04 | 1.00 | 2.6e-03 | 1.24 |
| classic_Triton | 32 | 2.4e-03 | 2.9e-03 | 2.2e-04 | 1.00 | 2.6e-03 | 1.24 |
| wave_Triton | 32 | 2.4e-03 | 2.9e-03 | 2.2e-04 | 1.00 | 2.6e-03 | 1.24 |
<!-- /v3-tables:E0 -->

### E1 scaling, device-resident

Wall-clock and peak device memory vs N in {128K, ..., 4M}, c=7, the smallest depth that fits (never 0),
forward and forward+backward, four series against SDPA and FA-2 (all fit on the 80 GB device at these sizes).
From 512K up the decomposition costs nothing against FA-2: the classic CUDA series is 0.95-1.00x FA-2's
forward and 0.99-1.01x its forward+backward at 1M-4M (1.27x / 1.14x at 128K, where seven launches and gathers
are a visible share of a 90 ms call); the wave series is 1-3% behind it (the W=1 wave's grid tail); the
Triton series 1.45-2.0x. Peak device memory is 2.0-2.4x FA-2's (the fp32 accumulator and output copies that
the host accumulator removes, E8); at 4M the forward+backward needed depth 2 (classic) / 4 (wave) to fit.

<!-- v3-tables:E1 -->
*forward* (wall s (x FA-2) / peak device GiB; the depth that ran when it is not 1):

| N | FA-2 | SDPA | classic CUDA | wave CUDA | classic Triton | wave Triton |
|---|---|---|---|---|---|---|
| 128K | 0.09 s / 0.5 GiB | 0.10 s / 0.5 GiB | 0.11 s (1.27x) / 1.2 GiB | 0.12 s (1.32x) / 2.2 GiB | 0.17 s (1.89x) / 1.2 GiB | 0.18 s (2.03x) / 3.6 GiB |
| 256K | 0.36 s / 1.0 GiB | 0.38 s / 1.0 GiB | 0.38 s (1.06x) / 2.4 GiB | 0.41 s (1.13x) / 2.6 GiB | 0.60 s (1.66x) / 2.4 GiB | 0.64 s (1.76x) / 3.1 GiB |
| 512K | 1.48 s / 2.0 GiB | 1.56 s / 2.0 GiB | 1.48 s (1.00x) / 4.8 GiB | 1.54 s (1.04x) / 5.2 GiB | 2.28 s (1.54x) / 4.8 GiB | 2.36 s (1.60x) / 6.2 GiB |
| 1M | 6.03 s / 4.0 GiB | 6.32 s / 4.0 GiB | 5.85 s (0.97x) / 9.5 GiB | 6.03 s (1.00x) / 10.4 GiB | 8.93 s (1.48x) / 9.5 GiB | 9.16 s (1.52x) / 11.0 GiB |
| 2M | 24.40 s / 8.1 GiB | 25.53 s / 8.1 GiB | 23.30 s (0.95x) / 19.0 GiB | 23.98 s (0.98x) / 20.8 GiB | 35.49 s (1.45x) / 19.0 GiB | 36.29 s (1.49x) / 19.8 GiB |
| 4M | 97.67 s / 16.1 GiB | 101.88 s / 16.1 GiB | 92.67 s (0.95x) / 37.9 GiB | 96.16 s (0.98x) / 41.7 GiB | 142.20 s (1.46x) / 37.9 GiB | 145.57 s (1.49x) / 37.7 GiB |

*forward + backward* (wall s (x FA-2) / peak device GiB; the depth that ran when it is not 1):

| N | FA-2 | SDPA | classic CUDA | wave CUDA | classic Triton | wave Triton |
|---|---|---|---|---|---|---|
| 128K | 0.33 s / 1.3 GiB | 0.35 s / 1.5 GiB | 0.37 s (1.14x) / 2.4 GiB | 0.38 s (1.17x) / 4.0 GiB | 0.65 s (2.00x) / 2.4 GiB | 0.65 s (1.99x) / 2.4 GiB |
| 256K | 1.30 s / 2.5 GiB | 1.40 s / 3.0 GiB | 1.38 s (1.06x) / 4.8 GiB | 1.42 s (1.09x) / 5.0 GiB | 2.40 s (1.84x) / 4.8 GiB | 2.40 s (1.84x) / 4.8 GiB |
| 512K | 5.24 s / 5.0 GiB | 5.62 s / 6.0 GiB | 5.31 s (1.01x) / 9.5 GiB | 5.40 s (1.03x) / 10.0 GiB | 9.31 s (1.78x) / 9.5 GiB | 9.31 s (1.78x) / 9.5 GiB |
| 1M | 21.03 s / 10.1 GiB | 22.58 s / 12.1 GiB | 20.88 s (0.99x) / 19.1 GiB | 21.13 s (1.00x) / 20.0 GiB | 36.97 s (1.76x) / 19.1 GiB | 37.00 s (1.76x) / 19.1 GiB |
| 2M | 84.12 s / 20.1 GiB | 90.15 s / 24.1 GiB | 83.08 s (0.99x) / 38.1 GiB | 83.85 s (1.00x) / 40.0 GiB | 147.69 s (1.76x) / 38.1 GiB | 148.17 s (1.76x) / 38.1 GiB |
| 4M | 337.80 s / 40.3 GiB | 363.53 s / 48.3 GiB | 340.95 s (1.01x) / 64.1 GiB itr2 | 379.99 s (1.12x) / 76.3 GiB itr4 | 592.77 s (1.75x) / 64.1 GiB itr2 | 592.69 s (1.75x) / 64.1 GiB itr2 |
<!-- /v3-tables:E1 -->

### E2 scaling, host-resident (the recovery configuration)

The same N ladder with Q/K/V in pinned host memory and the accumulator in host memory (the configuration a
small device falls back to), plus 4M and 8M on the PCIe node type. The overhead against the resident FA-2 call
falls as the per-subproblem transfers amortise: forward 1.40x at 512K, 1.17x at 1M, 1.06x at 2M, 1.04x at 4M;
forward+backward 1.41x / 1.29x / 1.14x / 1.07x; below 512K the fixed costs dominate (2.5-5.5x at 128K-256K).
Device memory is 3-3.6x less than the device-resident decomposition's and 0.55-0.65x FA-2's (4M forward: 10.4
GiB against FA-2's 16.1 and E1's 37.9); the price is host RAM, about 8 GiB per million tokens forward and 17
forward+backward (VmHWM 12 / 22). On the PCIe node the 8M forward -- which no monolithic kernel can hold
with its output on an 80 GB card -- runs at 0.97x FA-2's time in 21 GiB of device memory and 91 GiB of host
RAM; the 8M forward+backward rows are not run (1.5-3 h each).

<!-- v3-tables:E2 -->
**A100-SXM4-80GB**, Q/K/V in pinned host memory and the accumulator in host memory (ratios against E1's resident FA-2 rows of the same N):

*forward* (wall s (x FA-2) / peak device GiB; the depth that ran when it is not 1):

| N | FA-2 | SDPA | classic CUDA | wave CUDA | classic Triton | wave Triton | host GiB (classic CUDA) |
|---|---|---|---|---|---|---|---|
| 128K | 0.09 s / 0.5 GiB | 0.10 s / 0.5 GiB | 0.50 s (5.55x) / 0.5 GiB | 0.34 s (3.80x) / 0.8 GiB | 0.50 s (5.58x) / 0.3 GiB | 0.37 s (4.07x) / 1.9 GiB | 1.4 (VmHWM 3) |
| 256K | 0.36 s / 1.0 GiB | 0.38 s / 1.0 GiB | 0.93 s (2.56x) / 0.7 GiB | 0.74 s (2.03x) / 1.2 GiB | 1.28 s (3.53x) / 0.6 GiB | 1.04 s (2.87x) / 2.3 GiB | 2.0 (VmHWM 4) |
| 512K | 1.48 s / 2.0 GiB | 1.56 s / 2.0 GiB | 2.07 s (1.40x) / 1.3 GiB | 2.18 s (1.47x) / 2.4 GiB | 3.57 s (2.41x) / 1.1 GiB | 3.09 s (2.09x) / 4.6 GiB | 3.9 (VmHWM 7) |
| 1M | 6.03 s / 4.0 GiB | 6.32 s / 4.0 GiB | 7.08 s (1.17x) / 2.6 GiB | 7.30 s (1.21x) / 4.8 GiB | 11.47 s (1.90x) / 2.2 GiB | 10.56 s (1.75x) / 7.8 GiB | 7.5 (VmHWM 12) |
| 2M | 24.40 s / 8.1 GiB | 25.53 s / 8.1 GiB | 25.86 s (1.06x) / 7.8 GiB | 26.69 s (1.09x) / 9.6 GiB | 38.30 s (1.57x) / 4.3 GiB | 38.90 s (1.59x) / 13.4 GiB | 20.8 (VmHWM 29) |
| 4M | 97.67 s / 16.1 GiB | 101.88 s / 16.1 GiB | 101.47 s (1.04x) / 10.4 GiB | 102.67 s (1.05x) / 19.1 GiB | 151.36 s (1.55x) / 8.7 GiB | 150.10 s (1.54x) / 24.7 GiB | 32.0 (VmHWM 46) |

*forward + backward* (wall s (x FA-2) / peak device GiB; the depth that ran when it is not 1):

| N | FA-2 | SDPA | classic CUDA | wave CUDA | classic Triton | wave Triton | host GiB (classic CUDA) |
|---|---|---|---|---|---|---|---|
| 128K | 0.33 s / 1.3 GiB | 0.35 s / 1.5 GiB | 1.23 s (3.77x) / 0.7 GiB | 1.24 s (3.79x) / 0.7 GiB | 1.49 s (4.56x) / 0.8 GiB | 1.52 s (4.64x) / 0.8 GiB | 2.2 (VmHWM 4) |
| 256K | 1.30 s / 2.5 GiB | 1.40 s / 3.0 GiB | 2.95 s (2.26x) / 1.4 GiB | 2.98 s (2.29x) / 1.4 GiB | 4.09 s (3.13x) / 1.5 GiB | 4.04 s (3.10x) / 1.5 GiB | 4.4 (VmHWM 7) |
| 512K | 5.24 s / 5.0 GiB | 5.62 s / 6.0 GiB | 7.38 s (1.41x) / 2.8 GiB | 7.50 s (1.43x) / 2.8 GiB | 10.62 s (2.03x) / 3.0 GiB | 10.69 s (2.04x) / 3.0 GiB | 11.5 (VmHWM 15) |
| 1M | 21.03 s / 10.1 GiB | 22.58 s / 12.1 GiB | 27.10 s (1.29x) / 5.6 GiB | 26.98 s (1.28x) / 5.6 GiB | 43.21 s (2.06x) / 6.0 GiB | 43.30 s (2.06x) / 6.0 GiB | 17.2 (VmHWM 22) |
| 2M | 84.12 s / 20.1 GiB | 90.15 s / 24.1 GiB | 95.78 s (1.14x) / 11.2 GiB | 95.54 s (1.14x) / 11.2 GiB | 160.41 s (1.91x) / 12.0 GiB | 160.62 s (1.91x) / 12.0 GiB | 33.6 (VmHWM 44) |
| 4M | 337.80 s / 40.3 GiB | 363.53 s / 48.3 GiB | 360.01 s (1.07x) / 22.4 GiB | 358.57 s (1.06x) / 22.4 GiB | 617.40 s (1.83x) / 24.1 GiB | 617.64 s (1.83x) / 24.1 GiB | 67.7 (VmHWM 86) |

**A100 80GB PCIe** (its own resident baselines; FA-2's fwd+bwd at 8M does not fit an 80 GB device):

*forward* (wall s (x FA-2) / peak device GiB; the depth that ran when it is not 1):

| N | FA-2 | SDPA | classic CUDA | wave CUDA | classic Triton | wave Triton | host GiB (classic CUDA) |
|---|---|---|---|---|---|---|---|
| 4M | 136.08 s / 16.1 GiB | 131.02 s / 16.1 GiB | 125.95 s (0.93x) / 10.4 GiB | 129.93 s (0.95x) / 19.1 GiB | 182.46 s (1.34x) / 8.7 GiB | 180.61 s (1.33x) / 24.7 GiB | 32.0 (VmHWM 46) |
| 8M | 544.97 s / 32.3 GiB | 507.86 s / 32.3 GiB | 526.60 s (0.97x) / 20.8 GiB | 535.63 s (0.98x) / 34.8 GiB | 708.50 s (1.30x) / 17.3 GiB | 716.09 s (1.31x) / 47.2 GiB | 64.8 (VmHWM 91) |

*forward + backward* (wall s (x FA-2) / peak device GiB; the depth that ran when it is not 1):

| N | FA-2 | SDPA | classic CUDA | wave CUDA | classic Triton | wave Triton | host GiB (classic CUDA) |
|---|---|---|---|---|---|---|---|
| 4M | - | - | - | - | - | - | - |
| 8M | - | - | - | - | - | - | - |
<!-- /v3-tables:E2 -->

### E3-E6 parameters

**E3, the quorum size c** (N=1M, itr=1): time is nearly flat in c -- classic CUDA 0.97x FA-2 at c=3 to 1.03x
at c=133 forward, 0.99x to 1.05x forward+backward (133 launches instead of 3) -- while the peak device memory
falls from 11.9 to 7.2 GiB forward and 22.4 to 16.0 GiB forward+backward as the subproblems shrink: c is the
memory knob. **E4, depth x accumulator** (N=1M, c=7, Q/K/V on the device): a second level costs 4% (classic)
to 5% (wave) and saves 24% of the device memory; the host accumulator costs 10-14% forward (classic 9.5 -> 5.6
GiB, -41%) and 6-10% forward+backward (19.1 -> 11.4 GiB, -40%), at depth 2 the classic engine's host fold costs
46% where the wave engine's per-wave merge costs 20%. These two rows are why the planner's default became the
host accumulator from about 1M tokens (E8). **E5, the wave budget** (c=73): 5 to 73 waves per call change the
time by 1% and the memory by 2.2x (9.5 to 21 GiB forward). **E6, precision and head dim**: bf16 costs what fp16
costs; D=128 costs what FA-2's D=128 costs (0.93-0.99x) on the CUDA series, 2.2-2.3x on the Triton forward,
and the Triton backward at D=128 does not finish (a known 128-wide key-tile pathology; the CUDA backward is the
default and is unaffected).

<!-- v3-tables:E3 -->
*forward* at N=1M, itr=1 (wall s (x FA-2) / peak device GiB):

| c | subproblems | classic CUDA | wave CUDA | classic Triton | wave Triton |
|---|---|---|---|---|---|
| 3 | 3 | 5.84 s (0.97x) / 11.9 GiB | 6.04 s (1.00x) / 11.8 GiB | 8.91 s (1.48x) / 11.9 GiB | 9.15 s (1.52x) / 11.9 GiB |
| 7 | 7 | 5.86 s (0.97x) / 9.5 GiB | 6.06 s (1.01x) / 10.4 GiB | 8.92 s (1.48x) / 9.5 GiB | 9.16 s (1.52x) / 11.0 GiB |
| 13 | 13 | 5.90 s (0.98x) / 8.2 GiB | 6.10 s (1.01x) / 10.1 GiB | 8.94 s (1.48x) / 8.2 GiB | 9.22 s (1.53x) / 10.5 GiB |
| 21 | 21 | 5.93 s (0.98x) / 7.5 GiB | 6.16 s (1.02x) / 9.9 GiB | 9.04 s (1.50x) / 7.5 GiB | 9.32 s (1.55x) / 10.1 GiB |
| 31 | 31 | 5.97 s (0.99x) / 7.2 GiB | 6.21 s (1.03x) / 9.7 GiB | 9.14 s (1.51x) / 7.2 GiB | 9.43 s (1.56x) / 9.5 GiB |
| 57 | 57 | 6.04 s (1.00x) / 7.2 GiB | 6.31 s (1.05x) / 9.6 GiB | 9.32 s (1.55x) / 7.2 GiB | 9.65 s (1.60x) / 9.1 GiB |
| 73 | 73 | 6.08 s (1.01x) / 7.2 GiB | 6.36 s (1.06x) / 9.5 GiB | 9.49 s (1.57x) / 7.2 GiB | 9.78 s (1.62x) / 9.1 GiB |
| 91 | 91 | 6.13 s (1.02x) / 7.2 GiB | 6.42 s (1.07x) / 9.5 GiB | 9.62 s (1.59x) / 7.2 GiB | 9.89 s (1.64x) / 9.1 GiB |
| 133 | 133 | 6.20 s (1.03x) / 7.2 GiB | 6.55 s (1.09x) / 9.4 GiB | 9.89 s (1.64x) / 7.2 GiB | 10.12 s (1.68x) / 9.1 GiB |

*forward + backward* at N=1M, itr=1 (wall s (x FA-2) / peak device GiB):

| c | subproblems | classic CUDA | wave CUDA | classic Triton | wave Triton |
|---|---|---|---|---|---|
| 3 | 3 | 20.86 s (0.99x) / 22.4 GiB | 21.27 s (1.01x) / 22.2 GiB | 37.11 s (1.77x) / 22.4 GiB | 37.21 s (1.77x) / 22.4 GiB |
| 7 | 7 | 20.94 s (1.00x) / 19.1 GiB | 21.30 s (1.01x) / 20.0 GiB | 37.07 s (1.76x) / 19.1 GiB | 37.21 s (1.77x) / 19.1 GiB |
| 13 | 13 | 21.03 s (1.00x) / 17.4 GiB | 21.39 s (1.02x) / 19.1 GiB | 37.02 s (1.76x) / 17.4 GiB | 37.14 s (1.77x) / 17.4 GiB |
| 21 | 21 | 21.14 s (1.01x) / 16.4 GiB | 21.51 s (1.02x) / 19.1 GiB | 37.09 s (1.76x) / 16.4 GiB | 37.20 s (1.77x) / 16.4 GiB |
| 31 | 31 | 21.24 s (1.01x) / 16.0 GiB | 21.65 s (1.03x) / 19.1 GiB | 37.36 s (1.78x) / 16.0 GiB | 37.30 s (1.77x) / 16.0 GiB |
| 57 | 57 | 21.45 s (1.02x) / 16.0 GiB | 21.82 s (1.04x) / 19.1 GiB | 37.82 s (1.80x) / 16.0 GiB | 37.79 s (1.80x) / 16.0 GiB |
| 73 | 73 | 21.67 s (1.03x) / 16.0 GiB | 21.97 s (1.05x) / 19.1 GiB | 38.23 s (1.82x) / 16.0 GiB | 38.18 s (1.82x) / 16.0 GiB |
| 91 | 91 | 21.78 s (1.04x) / 16.0 GiB | 22.12 s (1.05x) / 19.1 GiB | 38.51 s (1.83x) / 16.0 GiB | 38.49 s (1.83x) / 16.0 GiB |
| 133 | 133 | 22.01 s (1.05x) / 16.0 GiB | 22.43 s (1.07x) / 19.1 GiB | 39.36 s (1.87x) / 16.0 GiB | 39.30 s (1.87x) / 16.0 GiB |
<!-- /v3-tables:E3 -->

<!-- v3-tables:E4 -->
N=1M, c=7, Q/K/V on the device (wall s (x FA-2) / peak device GiB / peak host GiB):

| direction | itr | accumulator | classic CUDA | wave CUDA | classic Triton | wave Triton |
|---|---|---|---|---|---|---|
| fwd | 1 | device | 5.85 s (0.97x) / 9.5 GiB / 0.1 | 6.05 s (1.00x) / 10.4 GiB / 0.1 | 8.92 s (1.48x) / 9.5 GiB / 0.1 | 9.15 s (1.52x) / 11.0 GiB / 0.1 |
| fwd | 1 | host | 6.69 s (1.11x) / 5.6 GiB / 6.8 | 6.92 s (1.15x) / 6.1 GiB / 8.4 | 9.86 s (1.64x) / 5.2 GiB / 6.8 | 10.09 s (1.67x) / 7.8 GiB / 8.4 |
| fwd | 2 | device | 6.08 s (1.01x) / 7.2 GiB itr2 / 0.2 | 6.38 s (1.06x) / 9.7 GiB itr2 / 0.2 | 9.29 s (1.54x) / 7.2 GiB itr2 / 0.2 | 9.66 s (1.60x) / 9.4 GiB itr2 / 0.2 |
| fwd | 2 | host | 8.89 s (1.47x) / 5.0 GiB itr2 / 4.9 | 7.64 s (1.27x) / 4.3 GiB itr2 / 7.4 | 10.85 s (1.80x) / 5.0 GiB itr2 / 5.0 | 11.05 s (1.83x) / 5.7 GiB itr2 / 7.4 |
| bwd | 1 | device | 21.10 s (1.00x) / 19.1 GiB / 0.1 | 21.31 s (1.01x) / 20.0 GiB / 0.1 | 36.97 s (1.76x) / 19.1 GiB / 0.1 | 36.98 s (1.76x) / 19.1 GiB / 0.1 |
| bwd | 1 | host | 23.09 s (1.10x) / 11.4 GiB / 14.2 | 21.32 s (1.01x) / 20.0 GiB / 0.1 (ran acc=gpu) | 39.14 s (1.86x) / 11.8 GiB / 14.2 | 39.06 s (1.86x) / 11.8 GiB / 14.2 |
| bwd | 2 | device | 21.71 s (1.03x) / 16.0 GiB itr2 / 0.2 | 22.12 s (1.05x) / 19.1 GiB itr2 / 0.2 | 37.74 s (1.79x) / 16.0 GiB itr2 / 0.2 | 37.74 s (1.79x) / 16.0 GiB itr2 / 0.2 |
| bwd | 2 | host | 25.72 s (1.22x) / 10.0 GiB itr2 / 11.3 | 22.12 s (1.05x) / 19.1 GiB itr2 / 0.2 (ran acc=gpu) | 40.29 s (1.92x) / 10.0 GiB itr2 / 11.3 | 40.40 s (1.92x) / 10.0 GiB itr2 / 11.4 |
<!-- /v3-tables:E4 -->

<!-- v3-tables:E5 -->
N=1M, c=73, itr=1 (73 subproblems), the wave budget as a multiple of the largest subproblem (wall s (x FA-2) / peak device GiB / waves):

| direction | W | wave CUDA | wave Triton |
|---|---|---|---|
| fwd | 1x | 6.37 s (1.06x) / 9.5 GiB / 73 waves | 9.80 s (1.62x) / 9.1 GiB / 73 waves |
| fwd | 2x | 6.34 s (1.05x) / 9.5 GiB / 37 waves | 9.80 s (1.62x) / 10.5 GiB / 37 waves |
| fwd | 4x | 6.34 s (1.05x) / 10.6 GiB / 19 waves | 9.76 s (1.62x) / 12.8 GiB / 19 waves |
| fwd | 8x | 6.32 s (1.05x) / 14.1 GiB / 10 waves | 9.71 s (1.61x) / 16.4 GiB / 10 waves |
| fwd | 16x | 6.30 s (1.04x) / 21.0 GiB / 5 waves | 9.62 s (1.60x) / 21.5 GiB / 5 waves |
| bwd | 1x | 22.10 s (1.05x) / 19.1 GiB / 73 waves | 38.08 s (1.81x) / 16.0 GiB |
| bwd | 2x | 22.03 s (1.05x) / 19.1 GiB / 37 waves | 38.09 s (1.81x) / 16.0 GiB |
| bwd | 4x | 21.98 s (1.05x) / 20.6 GiB / 19 waves | 38.08 s (1.81x) / 16.0 GiB |
| bwd | 8x | 21.93 s (1.04x) / 25.1 GiB / 10 waves | 38.08 s (1.81x) / 16.0 GiB |
| bwd | 16x | 21.80 s (1.04x) / 34.1 GiB / 5 waves | 38.09 s (1.81x) / 16.0 GiB |
<!-- /v3-tables:E5 -->

<!-- v3-tables:E6 -->
N=1M, c=7, itr=1 (wall s (x FA-2) / peak device GiB):

| direction | dtype | D | FA-2 | classic CUDA | wave CUDA | classic Triton | wave Triton |
|---|---|---|---|---|---|---|---|
| fwd | fp16 | 64 | 5.95 s / 4.0 GiB | 5.75 s (0.97x) / 9.5 GiB | 5.98 s (1.01x) / 10.4 GiB | 8.90 s (1.50x) / 9.5 GiB | 9.14 s (1.54x) / 11.0 GiB |
| fwd | fp16 | 128 | 10.87 s / 8.0 GiB | 10.12 s (0.93x) / 18.8 GiB | 10.32 s (0.95x) / 20.7 GiB | 24.08 s (2.22x) / 18.8 GiB | 24.52 s (2.26x) / 21.7 GiB |
| fwd | bf16 | 64 | 5.83 s / 4.0 GiB | 5.71 s (0.98x) / 9.5 GiB | 5.89 s (1.01x) / 10.4 GiB | 8.89 s (1.53x) / 9.5 GiB | 9.14 s (1.57x) / 11.0 GiB |
| fwd | bf16 | 128 | 10.50 s / 8.0 GiB | 10.04 s (0.96x) / 18.8 GiB | 10.24 s (0.98x) / 20.7 GiB | 24.05 s (2.29x) / 18.8 GiB | 24.48 s (2.33x) / 21.7 GiB |
| bwd | fp16 | 64 | 21.08 s / 10.1 GiB | 20.96 s (0.99x) / 19.1 GiB | 21.20 s (1.01x) / 20.0 GiB | 37.05 s (1.76x) / 19.1 GiB | 37.36 s (1.77x) / 19.1 GiB |
| bwd | fp16 | 128 | 38.38 s / 20.1 GiB | 37.39 s (0.97x) / 38.1 GiB | 37.88 s (0.99x) / 39.8 GiB | excluded | excluded |
| bwd | bf16 | 64 | 20.97 s / 10.1 GiB | 20.89 s (1.00x) / 19.1 GiB | 21.12 s (1.01x) / 20.0 GiB | 36.93 s (1.76x) / 19.1 GiB | 36.92 s (1.76x) / 19.1 GiB |
| bwd | bf16 | 128 | 38.39 s / 20.1 GiB | 37.62 s (0.98x) / 38.1 GiB | 37.88 s (0.99x) / 39.8 GiB | excluded | excluded |

excluded: the Triton backward at D=128 does not finish at 1M (a known 128-wide key-tile pathology; the CUDA backward is the default), see `results/v3/STATUS.md`.
<!-- /v3-tables:E6 -->

### E8 the default's trade: the accumulator in host memory

What `attention()` picks from about 1M tokens up when the monolithic call does not fit: Q/K/V on the device,
the fp32 accumulator (the forward's output, the backward's three gradients) in host memory, against E1's
device accumulator at the same N. The time cost shrinks with N while the device saving stays at 40%: the
classic CUDA forward costs 1.30x / 1.16x / 1.06x / 1.03x of the device-accumulator run at 512K / 1M / 2M /
4M, the forward+backward 1.19x / 1.11x / 1.06x / 1.01x, for 2.8 / 5.6 / 11.2 / 22.4 GiB of device memory
instead of 4.8 / 9.5 / 19.0 / 37.9 (forward) and 5.7 / 11.4 / 22.7 / 45.4 instead of 9.5 / 19.1 / 38.1 / 64.1
(forward+backward; E1's 4M row needed depth 2). Against FA-2 itself that is 1.0-1.1x the time in 0.7x
(forward) to 1.1x (forward+backward) of its device memory, plus host RAM of about 7 GiB per million tokens
forward and 14 forward+backward. The wave backward keeps its accumulator on the device (its rows ran with
acc=gpu in the backward, as recorded).

<!-- v3-tables:E8 -->
| arm | N | dir | E1 wall s (itr) | E8 wall s (itr) | E8/E1 | E1 dev GiB | E8 dev GiB | saved | E8 host GiB (VmHWM) | E8 acc ran |
|---|---|---|---|---|---|---|---|---|---|---|
| classic_CUDA | 512K | bwd | 5.31 (1) | 6.34 (1) | 1.193 | 9.54 | 5.69 | 40% | 6.8 (8.0) | cpu |
| classic_CUDA | 1M | bwd | 20.88 (1) | 23.08 (1) | 1.105 | 19.06 | 11.37 | 40% | 14.3 (15.5) | cpu |
| classic_CUDA | 2M | bwd | 83.08 (1) | 87.92 (1) | 1.058 | 38.12 | 22.71 | 40% | 28.4 (29.5) | cpu |
| classic_CUDA | 4M | bwd | 340.95 (2) | 344.53 (1) | 1.011 | 64.13 | 45.42 | 29% | 52.5 (53.7) | cpu |
| wave_CUDA | 512K | bwd | 5.40 (1) | 5.39 (1) | 0.997 | 10.00 | 10.00 | 0% | 0.0 (1.3) | gpu |
| wave_CUDA | 1M | bwd | 21.13 (1) | 21.16 (1) | 1.001 | 19.99 | 19.99 | 0% | 0.1 (1.3) | gpu |
| wave_CUDA | 2M | bwd | 83.85 (1) | 84.21 (1) | 1.004 | 39.95 | 39.95 | 0% | 0.2 (1.4) | gpu |
| wave_CUDA | 4M | bwd | 379.99 (4) | 380.07 (4) | 1.000 | 76.26 | 76.26 | 0% | 5.2 (6.4) | gpu |
| classic_CUDA | 512K | fwd | 1.48 (1) | 1.92 (1) | 1.300 | 4.75 | 2.81 | 41% | 3.4 (4.5) | cpu |
| classic_CUDA | 1M | fwd | 5.85 (1) | 6.76 (1) | 1.155 | 9.49 | 5.61 | 41% | 7.8 (8.9) | cpu |
| classic_CUDA | 2M | fwd | 23.30 (1) | 24.72 (1) | 1.061 | 18.97 | 11.20 | 41% | 13.6 (14.7) | cpu |
| classic_CUDA | 4M | fwd | 92.67 (1) | 95.49 (1) | 1.031 | 37.93 | 22.38 | 41% | 26.8 (27.9) | cpu |
| wave_CUDA | 512K | fwd | 1.54 (1) | 1.99 (1) | 1.291 | 5.22 | 3.05 | 42% | 4.3 (5.4) | cpu |
| wave_CUDA | 1M | fwd | 6.03 (1) | 6.85 (1) | 1.136 | 10.43 | 6.08 | 42% | 8.4 (9.5) | cpu |
| wave_CUDA | 2M | fwd | 23.98 (1) | 25.50 (1) | 1.063 | 20.84 | 12.14 | 42% | 16.7 (17.8) | cpu |
| wave_CUDA | 4M | fwd | 96.16 (1) | 98.87 (1) | 1.028 | 41.68 | 24.28 | 42% | 32.8 (33.9) | cpu |
<!-- /v3-tables:E8 -->

### E7 automatic mode under device budgets **[E7: pending]**

`attention()` with nothing pinned against FA-2 resident and two rectangular out-of-core rivals at budgets of
10/20/40/80 GiB (the v2.2.3 run of this grid: `results/ooc_rivals/report.md`; on the 1M backward at 10 GiB
v2.2.3 was 1.7x behind the rivals because of its backward kernel, which v3 replaces).

### v2.2.3 reference: one parameter at a time (`results/profile_sweep/`, forward, A100-SXM4-80GB)

The v2.2.3 kernels, for the record (E3 above is the v3 measurement). Vary c at N=512K, itr=1, acc=GPU (time in s / peak device memory in GiB):

| c | 3 | 7 | 13 | 21 | 31 | 57 | 73 | 91 | 133 |
|---|---|---|---|---|---|---|---|---|---|
| classic, CUDA | 1.98 / 5.97 | 1.99 / 4.74 | 2.01 / 4.12 | 2.04 / 3.76 | 2.08 / 3.58 | 2.16 / 3.58 | 2.19 / 3.58 | 2.24 / 3.58 | 2.32 / 3.58 |
| classic, Triton | 2.23 / 5.97 | 2.24 / 4.74 | 2.26 / 4.12 | 2.29 / 3.76 | 2.35 / 3.58 | 2.44 / 3.58 | 2.50 / 3.58 | 2.59 / 3.58 | 2.79 / 3.58 |
| wave, CUDA | 2.01 / 5.61 | 2.02 / 6.64 | 2.00 / 7.68 | 2.02 / 7.48 | 2.04 / 7.56 | 2.08 / 7.61 | 2.09 / 7.64 | 2.08 / 7.64 | 2.13 / 7.67 |
| wave, Triton | 2.32 / 10.8 | 2.34 / 13.9 | 2.37 / 17.0 | 2.42 / 17.6 | 2.45 / 18.8 | 2.50 / 18.9 | 2.55 / 19.0 | 2.57 / 20.0 | 2.63 / 20.1 |

The classic engine pays per launch (17% from c=3 to c=133 on CUDA, 25% on Triton); the wave engine
is nearly flat (6% and 13%) and wins from c=21 up on CUDA and from c=91 on Triton. The classic
engine's memory falls with c to the floor of Q/K/V plus the fp32 output; the wave engine packs a
whole wave (up to 2M tokens) of partial outputs, so its peak is higher and set by the wave budget.

Vary N at c=7 (x = N / 1e6; quadratic fit of the time over 64K..2M, R² >= 0.998 throughout):

| series | classic, CUDA | classic, Triton | wave, CUDA | wave, Triton |
|---|---|---|---|---|
| itr=1, acc=GPU | 7.22x² − 0.15x + 0.06 | 8.05x² − 0.04x + 0.03 | 7.43x² − 0.13x + 0.03 | 8.40x² − 0.09x + 0.05 |
| itr=1, acc=CPU | 7.16x² + 1.02x + 0.25 | 8.13x² + 0.48x + 0.20 | 7.68x² + 0.86x + 0.00 | 8.65x² + 1.41x + 0.02 |
| itr=2, acc=GPU | 7.21x² + 0.40x + 0.06 | 7.89x² + 0.31x + 0.07 | 7.36x² + 0.03x + 0.05 | 8.26x² + 0.43x + 0.02 |
| itr=2, acc=CPU | 6.12x² + 3.09x + 1.10 | 7.58x² + 0.76x + 1.11 | 7.95x² + 1.92x − 0.11 | 8.91x² + 2.78x − 0.08 |

The quadratic coefficient is the kernel: Triton is 11-16% behind CUDA at this head dim (the known
ratio of Triton's FlashAttention-2 forward), and the wave engine is within 3-4% of the classic
engine on the same kernel. The linear and constant terms are the engine overheads: acc=CPU adds
about 1 s per M tokens; itr=2 adds the launches, which the wave engine removes (its itr=2, acc=GPU
linear term is 0.03 against 0.40). At 2M tokens with acc=CPU the classic engine is 5-7% faster
than the wave engine in this sweep because its per-subproblem host transfers overlap with compute while
the wave engine then merged each wave on the host synchronously (the fold now runs on a worker under
the next wave and the wave is 1-3% ahead, `results/ultimate/wave_merge_ab.json`). `attention()` now
routes by the memory model (`autoconfig.plan`, engines axis): the wave engine runs every device-resident,
device-accumulator call whenever its wave fits (a W=1 wave costs what a classic subproblem costs), with
the wave budget W halved down to one subproblem per wave before the depth grows and gathered Q/K/V copies
(3-6% faster than in-place reads) given up only when they do not fit; host-resident inputs are enumerated
on both engines and the smaller device footprint wins a tie; host-accumulator backwards stay on the classic
pipeline. Peak device memory is linear in N for the classic engine
(9.0 GiB per M tokens at itr=1 acc=GPU, 5.3 at acc=CPU, 6.8 / 4.8 at itr=2). The wave engine's is
not: it holds a whole wave of packed partial outputs, and the wave is capped by a token budget (2M
packed tokens by default), so the fit is `peak = a*N + b*W_wave + c` with W_wave the largest wave the
planner forms: wave CUDA 6.98*N + 1.92*W (R² 0.9998; Q/K/V + fp32 output + accumulator per resident
token, one fp32 partial output per packed token), wave Triton 4.84*N + 7.79*W (R² 0.978; the wave is
gathered into packed fp16 Q/K/V copies as well; both pooled over the c sweep and the N series, per-series
fits in `fits.json`). W_wave rises with N and with c until it hits the
budget (1.6M packed tokens at 512K, 1.8M from 1M on), which is the plateau in the plots. The sweep ran with
the v2.2.0 default budget of 2M; since v2.2.1 the default is min(2M, max(512K, N)), which at 512K, itr=1
gives 4 waves and a 4.8 GiB peak instead of 6.6 at a time cost of at most 1.3%. Within a
wave the subproblems run in one batched launch, so the GPU schedules all of them together; waves run
one after another.

![profile sweep](results/profile_sweep/profile_sweep.png)

## v2.2.3 paper re-run (A100 80GB PCIe, `results/paper_rerun/`)

The paper's single-GPU experiments re-run with the v2.2.3 engine (E1/E2 of the v3 plan supersede this table) (B=1 H=8 D=64, causal, fp16 / bf16, 262K-16.8M
tokens, automatic depth with a floor of 1, every measurement in its own process). Baselines run on
device-resident Q/K/V; Stream-CQSA streams Q/K/V from host memory (acc=GPU / acc=CPU = where the fp32
accumulator lives). Native CUDA kernel, fp16 (time / peak device memory; ratio = Stream-CQSA's faster arm
over FlashAttention-2):

| N | FA-2 fwd | Stream-CQSA fwd | ratio | FA-2 fwd+bwd | Stream-CQSA fwd+bwd | ratio |
|---|---|---|---|---|---|---|
| 1.0M | 7.7 s / 5.0 GiB | 11.2 s / 6.5 GiB | 1.45x | 24.9 s / 10.1 GiB | 52.1 s / 13.6 GiB | 2.10x |
| 4.2M | 128.4 s / 20.1 GiB | 153.2 s / 25.9 GiB | 1.19x | 405.1 s / 40.3 GiB | 799.2 s / 46.5 GiB | 1.97x |
| 8.4M | 534.2 s / 40.3 GiB | 595.6 s / 51.8 GiB | 1.12x | not run (> 79.3 GiB) | 3189.8 s / 44.8 GiB (acc=CPU) | only Stream-CQSA |
| 16.8M | not run (> 79.3 GiB) | 2536.2 s / 32.0 GiB (acc=CPU) | only Stream-CQSA | not run | 14353.0 s / 38.4 GiB (acc=CPU) | only Stream-CQSA |

Full tables (both dtypes, SDPA mem-efficient, accuracy against float64, the Triton kernel set as its own
table, not-run reasons): `results/paper_rerun/tables_native.md`, `tables_triton.md`; regenerate with
`python results/paper_rerun/make_tables.py`. The forward+backward cost (~2x) is the v11 backward kernel
this release ships; the Triton backward is already faster (732 s at 4.2M).

## License

BSD-3-Clause (see `LICENSE`); vendored FlashAttention-2 and CUTLASS notices in `third_party/`.
