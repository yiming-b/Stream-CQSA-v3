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
(`experiments/v3/PLAN.md`) whose results fill the placeholders marked **[E#: pending]** below as the
jobs finish (`results/v3/STATUS.md` says which rows exist).

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
| **multi-device scheduler** | per-GPU rates (calibrated per model), exact live-tile work per task, LPT with a cost-aware split, reduce_scatter merge; identical schedule on every rank; heterogeneous devices supported (`stream_cqsa/scheduler.py`) | model within 2% rms of the measured 4x A100 job; the measured winner reproduced on every shape (`results/ultimate/dist_sched.md`); **[mixed-GPU job pending]** |
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
| pin a setting | keywords of `attention`: `itr=`, `c=`/`interest_set=`, `accumulate_on_gpu=False`, `max_parallel=`, `bwd_itr=`, `kernel="cuda"｜"triton"｜"wave-cuda"｜"wave-triton"` |
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

### E0 correctness **[E0: pending]**

Every series (classic CUDA / Triton, wave CUDA / Triton) x c in {3, 7, 31, 133} x itr {1, 2} x accumulator
{GPU, host} x {forward, forward+backward} at N=256K, plus bf16 and head dim 128: output and gradient errors
against FlashAttention-2 and float64. Expected from the smoke rows: fp16 fp64-error 1.1-1.5e-4 (= FA-2's own,
ratio 0.99-1.00), bf16 ~8e-4.

### E1 scaling, device-resident **[E1: pending]**

Wall-clock and peak memory vs N in {128K, 256K, 512K, 1M, 2M, 4M}, c=7, the smallest depth that fits
(never 0), forward and forward+backward, four series against SDPA and FA-2 up to where they fit; quadratic
fits of time and linear fits of memory per series. Replaces v2's "Headline" table and the N sweep.

### E2 scaling, host-resident (the recovery configuration) **[E2: pending]**

The same N ladder with Q/K/V and the accumulator in host memory, plus 4M and 8M on the PCIe node type.

### E3-E6 parameters **[E3-E6: pending]**

E3: c in {3, 7, 13, 21, 31, 57, 73, 91, 133} at N=1M (replaces v2's c table below); E4: itr x accumulator at
1M; E5: the wave budget (subproblems per launch) at c=73; E6: {fp16, bf16} x {head dim 64, 128} at 1M.

### E7 automatic mode under device budgets **[E7: pending]**

`attention()` with nothing pinned against FA-2 resident and two rectangular out-of-core rivals at budgets of
10/20/40/80 GiB (the v2.2.3 run of this grid: `results/ooc_rivals/report.md`; on the 1M backward at 10 GiB
v2.2.3 was 1.7x behind the rivals because of its backward kernel, which v3 replaces).

### v2.2.3 reference: one parameter at a time (`results/profile_sweep/`, forward, A100-SXM4-80GB)

Kept until E3 replaces it. Vary c at N=512K, itr=1, acc=GPU (time in s / peak device memory in GiB):

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
