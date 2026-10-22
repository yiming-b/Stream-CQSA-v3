# Stream-CQSA v3 on CITA (H100, no network): the offline bundle

One tarball, `stream_cqsa_v3_cita_<date>.tar.gz`, built on Della by `dist_cita/make_bundle.sh` from the v3
repository's HEAD. It unpacks to one directory and runs with no network access:

```
stream_cqsa_v3_cita_<date>/
  README.md            this file
  setup_cita.sh        installs everything offline, builds the CUDA modules when an nvcc exists (three tiers)
  run_cita.py          executes run_cita.ipynb with nbclient; saves run_cita_executed.ipynb + the results tarball
  run_cita.ipynb       the experiment notebook (environment, correctness, E0, E1 fwd, E1 fwd+bwd, E3, E4, E6, report)
  wheels/              54 python-3.12 wheels (pytest, nbformat, nbclient, nbconvert, ipykernel and every dependency)
                       + flash_attn-2.8.3.tar.gz (the source distribution; optional, see below)
  cuda_min/            a relocatable CUDA 12.9.86 compiler (nvcc, cicc, ptxas, cudafe++, fatbinary, nvlink, nvvm,
                       libdevice, the crt / cudart / CCCL headers, libcudart): the conda-forge packages of
                       cuda_min_pkgs.txt unpacked into one prefix; used by setup_cita.sh when CITA has no CUDA 12.x nvcc
  src/                 the repository at commit GIT_COMMIT: stream_cqsa/, csrc/ (incl. csrc/cutlass), native/,
                       experiments/v3/, tests/, setup.py, pyproject.toml, third_party/, docs/, benchmarks/
  GIT_COMMIT  MANIFEST.txt  SHA256SUMS  wheels.txt
```

What comes back: `cita_results_<host>_<date>.tar.gz` (printed as the last line of `python run_cita.py`): the
results directory, the executed notebook with every cell's output, the setup / build logs and the doctor output.

## Step by step

1. Copy the tarball (~130 MB; 3000 files, 320 MB unpacked) to CITA, e.g. into `$HOME` or a scratch directory with ~2 GB free (the build adds ~1 GB).
2. Unpack and enter it:
   ```
   tar xzf stream_cqsa_v3_cita_<date>.tar.gz
   cd stream_cqsa_v3_cita_<date>
   sha256sum -c SHA256SUMS --quiet && echo "bundle intact"      # optional
   ```
3. Activate the venv that has torch 2.13.0+cu129 (python 3.12.7, triton 3.7.1):
   ```
   source <venv>/bin/activate
   ```
4. Install (offline). Everything it does is logged next to the script; it never stops on a failed tier:
   ```
   bash setup_cita.sh 2>&1 | tee setup.log
   ```
   | tier | what | how long | needs |
   |---|---|---|---|
   | wheels | `pip install --no-index --find-links wheels pytest nbformat nbclient nbconvert ipykernel` | < 1 min | nothing |
   | (a) triton | `CQSA_SKIP_EXT=1 pip install --no-index --no-build-isolation --no-deps -e src/` + a `.pth` file that puts `src/` on `sys.path` (the compiled modules of (b) / (c) land there) | < 1 min | nothing: the pure-Triton kernels (`stream_cqsa/triton_kernel.py`) serve the `classic_Triton` and `wave_Triton` series |
   | (b) classic_cuda | `cd src && CQSA_KERNEL_SET=common FLASH_ATTN_CUDA_ARCHS=90 TORCH_CUDA_ARCH_LIST=9.0 MAX_JOBS=<min(nproc/2,16)> python setup.py build_ext --inplace` -> `src/cqsa_cuda.*.so`, log `build_classic.log` | 20-60 min | an nvcc of CUDA **12.x** (a system one, else the bundled `cuda_min/`; see below) and `g++` |
   | (c) wave_cuda | the same with `CQSA_ONLY_NATIVE=1` -> `src/cqsa_native.*.so`, log `build_native.log` | 20-60 min | the same |
   | doctor | `python -m stream_cqsa.doctor` -> `doctor.txt`: which modules loaded, what fits | < 1 min | |

   The last lines (`tiers.txt`) say per tier `OK` / `SKIPPED(reason)` / `FAILED(see log)`. **sm90 only**: the modules are
   compiled for the H100 (`-gencode arch=compute_90,code=sm_90`) and for no other GPU.

   **About nvcc.** The venv's pip packages do not provide one: the wheel `nvidia-cuda-nvcc-cu12 12.9.86` (what
   `cuda-toolkit 12.9.1` pulls in) contains only `ptxas`, `nvvm/` and the `crt/` headers, not the `nvcc` driver,
   `cicc` or `cudafe++`. The script looks for `$CUDA_HOME/bin/nvcc`, `nvcc` on `PATH`, `/usr/local/cuda*`,
   `/opt/cuda*` and the NVIDIA HPC SDK, and accepts a release whose major version equals torch's CUDA (12),
   preferring the one closest to 12.9 (a CUDA 13 nvcc is refused by torch's extension builder for a cu129 torch,
   and a far newer minor can fail on torch's own headers). If none is found it uses the bundled `cuda_min/`
   (nvcc 12.9.86, the conda-forge build, relocatable, no installation) and links the cuBLAS / cuSPARSE / cuSOLVER
   headers of the venv's `nvidia-*-cu12` wheels into it (torch's ATen headers include them). `CITA_USE_BUNDLED_NVCC=1`
   forces the bundled one; `CUDA_HOME=/path/to/cuda-12.x` or `module load cuda/12.x` before step 4 selects a system
   one. nvcc needs a host `g++` (<= 14 for nvcc 12.9; for a newer one set `NVCC_APPEND_FLAGS=-allow-unsupported-compiler`).
   Tested on Della: `cuda_min`'s nvcc compiles the classic module's units for sm90 through the exact command torch's
   builder generates. If every route fails, tiers (b) and (c) are `SKIPPED` / `FAILED` and the run is Triton-only,
   which is a complete run of the two Triton series (exact; on the A100 they run at 0.65-0.7x the CUDA modules'
   speed).

5. First pass (~20 min of GPU time): the tests, the inline correctness check, E0 at a reduced grid, E1 forward
   up to N = 1M:
   ```
   CITA_QUICK=1 python run_cita.py
   ```
   Full run (bounded by ~2 h 30 min with the default per-cell budgets; rows that did not fit are deferred, and
   running the same command again continues with only the missing rows):
   ```
   python run_cita.py
   ```
   Both print one line per experiment row as it finishes; the per-row logs are in `results/v3_cita/logs/`
   (`tail -f results/v3_cita/logs/E1_bwd.log` to watch a long cell). `run_cita_executed.ipynb` is saved after
   every cell, so an interrupted run keeps what it had. The notebook way, if preferred:
   ```
   jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 --output run_cita_executed.ipynb run_cita.ipynb
   ```
   (`run_cita.py` additionally refreshes the results tarball with the complete executed notebook; with nbconvert,
   pack `results/v3_cita/`, `run_cita_executed.ipynb`, `setup.log`, `build_*.log`, `doctor.txt` by hand.)
   If ipykernel cannot start for some reason, `python run_cita.py --inproc` runs the cells in one process and
   still saves their outputs into the notebook.

6. Bring back `cita_results_<host>_<date>.tar.gz` (the path and size are the last line of `python run_cita.py`).
   It is 5-30 MB.

## Knobs

| variable | default | meaning |
|---|---|---|
| `CITA_QUICK=1` | off | the ~20-minute first pass (pytest, inline correctness, E0 at c in {7, 31} itr 1, E1 forward up to 1M) |
| `CITA_MAX_N` | 4194304 | the largest N of the E1 ladder; set `2097152` after an OOM at 4M or to save time |
| `CITA_BUDGET_SCALE` | 1 | multiplies every cell's time budget (e.g. 4 for an overnight run); `CITA_FULL=1` removes the budgets |
| `CITA_SERIES` | detected | comma list of the series to run, e.g. `classic_Triton,wave_Triton` |
| `CITA_OUT` | `results/v3_cita` | the results directory (`results.jsonl` is append-only; every run resumes it) |
| `CITA_MAX_JOBS` | min(nproc/2, 16) | parallel nvcc jobs of `setup_cita.sh` (each needs ~2-4 GB of host RAM) |
| `CUDA_VISIBLE_DEVICES` | `0` | the notebook sets it to one GPU when unset; with several H100s pick one explicitly |

## What the notebook does, and how long (A100-SXM4-80GB reference; the H100 will differ)

The times are what `results/v3/report.md` measured on Della's A100-SXM4-80GB (jobs 15966097 / 16003186), for the
whole cell including the untimed references; no H100/A100 factor is assumed. Each experiment cell runs its rows
cheapest first under a budget (`--budget-s`), so a budget cuts the largest configurations, never a row half-way.

| cell | content | A100 wall | default budget |
|---|---|---|---|
| 0 environment | host, GPUs, driver, versions, modules loaded, doctor, git commit, tiers | < 1 min | - |
| 1a pytest | `tests/test_v3_quorum_sizes.py` (every c x N in {64K, 128K} x itr in {1, 2}, fwd + bwd vs SDPA-flash, decomposition asserted), `test_api.py`, `test_evict_inputs.py`, `test_ultimate_routing.py` | ~2-5 min | - |
| 1b inline correctness | `attention()` vs SDPA-flash, itr=1 pinned, N in {64K, 256K}, every available series, output + gradients | ~1 min | - |
| 2 E0 | correctness: every series x c in {3, 7, 31, 133} x itr in {1, 2} x acc in {gpu, cpu} x {fwd, bwd} at 256K, + bf16 and D=128 rows (128 rows; quick: 48) | 21 min measured | 30 min (quick 15) |
| 3 E1 fwd | wall / peak memory vs N in {128K ... 4M}, every series + baselines, reps 2 | 32 min measured at reps 1 | 30 min |
| 4 E1 fwd+bwd | the same with the backward | not measured; ~2.5 h estimated | 30 min |
| 5 E3 fwd | c in {3 ... 133} at 1M, every series, reps 2 | 25 min measured at reps 3 | 15 min |
| 6 E4 | itr in {1, 2} x acc in {gpu, cpu} at 1M, fwd and bwd, reps 2 | 45 min measured (24/32 rows, reps 3) | 15 min |
| 7 E6 | {fp16, bf16} x D in {64, 128} at 1M, fwd and bwd, + baselines, reps 2 | not measured; ~1 h estimated | 15 min |
| 8 report | `bench.py report`: `report.md`, `STATUS.md`, `fits.json`, `E*.png` shown inline | < 1 min | - |
| 9 tarball | `cita_results_<host>_<date>.tar.gz` | < 1 min | - |

Every Stream-CQSA row pins `itr` and `c` and asserts the decomposition it reports (`src/tests/_decomp.py`); a
row whose engine took the monolithic path is recorded as an error, never as a measurement.

## The reference arm: FA-2 or SDPA-flash

`flash_attn` is not installed on CITA and no binary for torch 2.13 / cu129 / python 3.12 could be built on
Della. The harness therefore uses **torch SDPA on its flash backend** (`torch.nn.attention.sdpa_kernel(
SDPBackend.FLASH_ATTENTION)`) as the reference for the error columns and as the monolithic baseline; every row
records `ref_arm = "SDPA-flash"`, the report's `ref` column and headers say so, the `/FA-2` ratio column stays
empty and the `FA-2` arm is dropped from the plan with a printed notice. It is never labelled FA-2.

Optional: `wheels/flash_attn-2.8.3.tar.gz` is the source distribution. With a CUDA 12.x nvcc (the one
`setup.log` says it used: `export CUDA_HOME=$PWD/cuda_min` for the bundled one, after `setup_cita.sh` has linked the
headers), a 20-60 min compile (`MAX_JOBS` nvcc jobs, ~3 GB RAM each) adds the real FA-2 arm and reference:
```
export CUDA_HOME=$PWD/cuda_min          # or the system toolkit setup_cita.sh chose
FLASH_ATTENTION_FORCE_BUILD=TRUE FLASH_ATTN_CUDA_ARCHS=90 MAX_JOBS=8 \
  pip install --no-index --no-build-isolation --no-deps wheels/flash_attn-2.8.3.tar.gz
python -c "import flash_attn; print(flash_attn.__version__)"
```
Then `python run_cita.py` picks it up (the environment cell prints `flash_attn 2.8.3`, the baselines include
`FA-2`, the reference arm is FA-2). flash-attn 2.8.3 predates torch 2.13; if its build fails, send
the pip output along and stay with SDPA-flash. Rows already recorded with `ref_arm = SDPA-flash` keep that label
(delete `results/v3_cita/results.jsonl` to redo them against FA-2).

## Troubleshooting

- **`tiers.txt` says `classic_cuda SKIPPED no nvcc of CUDA 12.x found`** (the bundled `cuda_min/bin/nvcc` did not
  run either: `setup.log` says so): the Triton-only run is a complete, exact run (series `classic_Triton`,
  `wave_Triton`). To get the CUDA series, find a CUDA 12.x toolkit on CITA (`module avail cuda`,
  `ls /usr/local/ | grep cuda`, `ls /opt`), then `CUDA_HOME=<root> bash setup_cita.sh`. Set `CITA_REBUILD=1` to
  force a rebuild of a module that already imports.
- **`FAILED` tier**: send back `build_classic.log` / `build_native.log` (they are in the results tarball too).
  The usual causes: a g++ older than 9 or newer than nvcc 12.9 supports (14; `NVCC_APPEND_FLAGS=-allow-unsupported-compiler`,
  or `CXX=<older g++>`), a missing `cublas_v2.h` / `cusparse.h` / `cusolverDn.h` (the venv lacks the
  `nvidia-cublas-cu12` / `-cusparse-cu12` / `-cusolver-cu12` wheels that torch depends on; `setup.log` warns),
  running out of host RAM (lower `CITA_MAX_JOBS`). A system CUDA 12.x can be tried instead with `CUDA_HOME=<root>`.
- **Out of memory in an experiment row**: the row is recorded as `oom` and the run continues. If the 4M rows of
  E1 are all `oom` on an 80 GB H100 something else holds the GPU: check `nvidia-smi`, set
  `CUDA_VISIBLE_DEVICES` to a free GPU, or `CITA_MAX_N=2097152`.
- **Several H100s**: the notebook uses one (`CUDA_VISIBLE_DEVICES=0` unless set). Do not run two copies on the same GPU.
- **The run was interrupted**: `python run_cita.py` again. Rows in `results/v3_cita/results.jsonl` are kept
  (`ok` and `oom` rows are final; `error` rows are re-run only with `--retry-errors`, i.e. by hand:
  `cd src && python -P experiments/v3/bench.py run --exp E1 --part fwd --out ../results/v3_cita --retry-errors`).
- **`run_cita.py` cannot start the kernel** (traceback mentioning jupyter_client / zmq): `python run_cita.py --inproc`.
- **A Triton compile error** (in `tier_a.log`, the doctor's check, the pytest cell or a `classic_Triton` /
  `wave_Triton` row): the Triton kernels were validated with triton 3.6 (Della, A100); triton 3.7.1 on an H100 is
  untested here. Send the traceback; the CUDA series (tiers b / c) do not depend on Triton.
- **Time**: `CITA_QUICK=1` first. The default run is bounded by ~2.5 h; `CITA_BUDGET_SCALE=4` for an overnight
  run; `CITA_FULL=1` for no budgets (6-8 h on the A100 scale).
- **Running bench.py by hand** (from `src/`): `python -P experiments/v3/bench.py plan --exp E1 --part fwd`
  lists the rows and their estimates; `run ... --dry-run` the same with the budget; `report --out <dir>` the tables.

## Reproducing the bundle on Della

`bash dist_cita/make_bundle.sh` (`--download` to refresh `dist_cita/wheels/` from PyPI first). The wheel set's
closure was verified on Della in a clean python-3.12.7 conda env (`/scratch/gpfs/AKEY/yb2807/VENVs/py312_cita_closure`,
no torch) holding CITA's relevant packages at CITA's versions (numpy 2.3.5, matplotlib 3.11.2, pandas 3.0.6,
setuptools 80.10.2, psutil, einops, packaging, pyzmq, jsonschema, referencing, markupsafe, python-dateutil,
tabulate, jinja2, typing_extensions, ninja):
```
pip install --dry-run --no-index --find-links wheels pytest nbformat nbclient nbconvert ipykernel   # "Would install ..." 39 packages, no ERROR
pip install --no-index --find-links wheels pytest nbformat nbclient nbconvert ipykernel             # Successfully installed ... (39)
python -c "import pytest, nbformat, nbclient, nbconvert, ipykernel, jupyter_client, IPython"       # ok
```
and a one-cell notebook executed with nbclient in that env. The first attempt found two missing wheels
(`nest-asyncio2`, `ipython_pygments_lexers`); the set is closed since. The notebook's plumbing was proven end to
end on Della's shared A100 (vis1) with `CITA_SMOKE=1` in a python-3.11 / torch 2.10 scratch venv with
`flash_attn` blocked (correctness and plumbing only; that executed notebook is not in the bundle).
