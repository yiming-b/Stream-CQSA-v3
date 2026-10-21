#!/bin/bash
# U1 part of the combined ultimate job (one A100-SXM4-80GB): the v13 classic module's acceptance.
#   step 1  the full pytest suite on cqsa_cuda_v13 (CQSA_CUDA_MODULE=cqsa_cuda_v13: one module for both causal
#           modes, CUDA backward by default); the 12 bf16 cases that failed on the fp16-only dev v11 module
#           (test_backward_default / test_bf16 / test_bwd_host_stream, dtype1) must pass on the `common` build.
#           Expected ~25-35 min of GPU time (the vis1 subset of 400 tests took 447 s under load; the full suite
#           adds the v11x / v12f / v12b / v13 kernel tests and the wave tests); host memory 48G (the suite holds
#           several host tensors and forks workers; 24G was OOM-killed in job 14231448).
#   step 2  benchmarks/ultimate/engine_ab.py: engine-level attention() forward + backward, device-resident
#           inputs, acc=GPU: v2.2.3 (classic engine, CUDA and Triton backward) vs v3 + v13 (kernel="cuda" and
#           "auto") vs monolithic flash_attn / SDPA-flash at N=131072 (c=7, 73), 1048576 (c=7, itr 1 and 2) and
#           a bf16 row at 131K; medians of 3 after 1 warm-up. Expected ~25-35 min of GPU time (the 1M backward
#           rows: ~20-40 s per run x 4 runs x 4 engine arms x 2 rows, plus the flash_attn reference in every
#           worker); host memory <= 24G (everything lives on the device; the host holds the pinned task lists,
#           a few hundred MiB at 1M, and the interpreter: engine_align_ab's 1M job ran within 24G).
# Environment (set by the combined job):
#   CQSA_SRC          the frozen code root (git archive of one Stream-CQSA-v3 commit; slurm/ultimate/snapshot.sh)
#   CQSA_V13_SO       the directory holding the frozen cqsa_cuda_v13 .so (next/kernel/v13/frozen/<tag>)
#   CQSA_SRC_RESULTS  the live repo's results/ultimate/ (default: Stream-CQSA-v3/results/ultimate)
# Outputs in $CQSA_SRC_RESULTS: u1_test_suite_<job>.out (the pytest log), engine_ab.json + engine_ab.md (+ the
# job's stdout). Each step runs under `set +e`; its exit code is printed and the next step runs regardless.
# Knobs: U1_PYTEST_ARGS (extra pytest arguments, e.g. `-k test_v13` for a smoke), ENGINE_AB_ARGS (e.g. `--smoke`
# on a shared GPU: small rows, plumbing only), ENGINE_AB_NAME (result file stem, default engine_ab).
set +e
set -u
ROOT=${CQSA_SRC:?set CQSA_SRC to the frozen code root}
V13=${CQSA_V13_SO:?set CQSA_V13_SO to the frozen cqsa_cuda_v13 directory}
RES=${CQSA_SRC_RESULTS:-/scratch/gpfs/AKEY/yb2807/Stream-CQSA-v3/results/ultimate}
JOB=${SLURM_JOB_ID:-nojob}
KER=/scratch/gpfs/AKEY/yb2807/Stream-CQSA-dev/next/kernel
SUITE_KERNELS=${SUITE_KERNELS:-$KER/frozen_suite/2026-09-23}      # v11x / v12f / v12b modules for their own tests
V223_DIR=${V223_DIR:-/scratch/gpfs/AKEY/yb2807/Stream-CQSA-snapshots/v2-v2.2.3}
mkdir -p "$RES"
echo "== U1 part: $(date)  job $JOB  host $(hostname)"
nvidia-smi --query-gpu=name,clocks.sm,clocks.max.sm,memory.total --format=csv,noheader
echo "code: $ROOT   v13 module: $V13   results: $RES"
ls -la --time-style=full-iso "$V13"/cqsa_cuda_v13*.so; md5sum "$V13"/cqsa_cuda_v13*.so
cd "$ROOT" || { echo "U1: cannot cd to $ROOT"; exit 1; }
source /scratch/gpfs/AKEY/yb2807/Stream-CQSA-dev/next/env_next.sh
# the variant dirs carry a stub stream_cqsa/__init__.py for their setup.py, so they go LAST
export PYTHONPATH=$ROOT:$KER/stable:${PYTHONPATH:-}:/scratch/gpfs/AKEY/yb2807/Stream-CQSA-dev/next/native:$SUITE_KERNELS:$V13
export CQSA_CPU_THREADS=${CQSA_CPU_THREADS:-4} OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
unset CQSA_CUDA_MODULE CQSA_CUDA_MODULE_NONCAUSAL CQSA_BACKWARD CQSA_FORWARD
which python

# ---------------------------------------------------------------- step 1: the full suite on v13
echo "== U1 step 1: pytest suite on cqsa_cuda_v13  $(date +%T)"
SUITE_OUT=$RES/u1_test_suite_$JOB.out
(
  export CQSA_CUDA_MODULE=cqsa_cuda_v13
  python -P -c "import torch, stream_cqsa, stream_cqsa.interface as I; print(stream_cqsa.__file__, torch.__version__, I.CQSA_CUDA_MODULE, I.CQSA_CUDA_MODULE_NONCAUSAL, I.cqsa_cuda.__file__, 'backward', I.backward_kernel()); print(stream_cqsa.kernels_available())"
  python -P -c "import torch, cqsa_cuda_next_v11 as a, cqsa_cuda_next_v9 as b, cqsa_cuda_next_v11x as c, cqsa_cuda_next_v12f as d, cqsa_cuda_next_v12b as e, cqsa_cuda_v13 as f; print(*(m.__file__ for m in (a, b, c, d, e, f)), sep='\n')"
  timeout -s INT 3600 python -P -X faulthandler -m pytest tests -q -rfs --tb=line -p no:cacheprovider -o faulthandler_timeout=600 ${U1_PYTEST_ARGS:-} 2>&1 \
      | grep -v "^$\|site-packages\|frozen runpy" | cut -c1-300
  exit ${PIPESTATUS[0]}
) > "$SUITE_OUT" 2>&1
RC1=$?
echo "U1 step 1 (suite) rc=$RC1  $(date +%T)  log $SUITE_OUT"
tail -3 "$SUITE_OUT"; grep -c "^FAILED" "$SUITE_OUT" | sed 's/^/failed tests: /'

# ---------------------------------------------------------------- step 2: engine-level A/B
echo "== U1 step 2: engine_ab  $(date +%T)"
if [ ! -f "$V223_DIR/.complete" ]; then
  mkdir -p "$V223_DIR" && git -C /scratch/gpfs/AKEY/yb2807/Stream-CQSA-v2 archive v2.2.3 | tar -x -C "$V223_DIR" && touch "$V223_DIR/.complete"
fi
PYTHONPATH=$V223_DIR:$KER/stable:${PYTHONPATH:-} CQSA_CUDA_MODULE=cqsa_cuda_next_v11 CQSA_CUDA_MODULE_NONCAUSAL=cqsa_cuda_next_v9 \
  python -P -c "import stream_cqsa, stream_cqsa.interface as I; print('v2.2.3 arm:', stream_cqsa.__file__, stream_cqsa.__version__, I.cqsa_cuda.__file__)"
export CQSA_SRC=$ROOT V223_DIR V13_DIR=$V13
AB=${ENGINE_AB_NAME:-engine_ab}
python -P -u "$ROOT/benchmarks/ultimate/engine_ab.py" run --out "$RES/$AB.json" --reps 3 --warmup 1 ${ENGINE_AB_ARGS:-}
RC2=$?
echo "U1 step 2 (engine_ab) rc=$RC2  $(date +%T)  results $RES/$AB.json / $AB.md"

echo "== U1 part done: suite rc=$RC1, engine_ab rc=$RC2  $(date)"
