#!/bin/bash
# U1 part of the combined ultimate job (one A100-SXM4-80GB, gpu-test: the whole job must fit in 59 min, this part
# gets ~25 min of it): the v13 classic module's engine-level acceptance.
#   step 2  benchmarks/ultimate/engine_ab.py: engine-level attention() forward + backward, device-resident inputs,
#           acc=GPU: v3 + v13 classic (kernel="cuda", itr pinned) and wave (kernel="wave", itr pinned, old
#           cqsa_native) and the unpinned kernel="auto" row (what the user gets: monolithic at these N), v2.2.3
#           (classic engine, CUDA and Triton backward, pinned), monolithic flash_attn / SDPA-flash, at N=1048576 c=7
#           itr=1 (the acceptance row, first), 131072 (c=7, 73, and a bf16 row), 1048576 c=7 itr=2 (last); every
#           row records the engine that ran (monolithic / classic / wave), itr and the subproblem count; 2 timed runs
#           after 1 warm-up. Expected <= 25 min of GPU
#           time: the driver carries a wall-clock budget (U1_BUDGET_S, default 1500 s) and skips the workers that
#           would exceed it (estimate = the longest worker seen at that N), so the 1M itr=2 row is what a slow node
#           loses. Host memory <= 24G (everything lives on the device; the host holds the pinned task lists, a few
#           hundred MiB at 1M, and the interpreter: engine_align_ab's 1M job ran within 24G).
#   step 1  (OFF by default; U1_RUN_SUITE=1 enables it) the full pytest suite on cqsa_cuda_v13 -- run on vis1 by the
#           package instead (correctness only; results in docs/ultimate/progress/U1.md). If enabled: ~25-35 min of
#           GPU time, host 48G (the suite holds several host tensors and forks workers; 24G was OOM-killed in job
#           14231448).
# Environment (set by the combined job):
#   CQSA_SRC          the frozen code root (git archive of one Stream-CQSA-v3 commit; slurm/ultimate/snapshot.sh)
#   CQSA_V13_SO       the directory holding the frozen cqsa_cuda_v13 .so (next/kernel/v13/frozen/<tag>)
#   CQSA_SRC_RESULTS  the live repo's results/ultimate/ (default: Stream-CQSA-v3/results/ultimate)
# Outputs in $CQSA_SRC_RESULTS: engine_ab.json + engine_ab.md (and u1_test_suite_<job>.out when the suite runs).
# Each step runs under `set +e`; its exit code is printed and the next step runs regardless.
# Knobs: U1_RUN_SUITE=1, U1_PYTEST_ARGS (extra pytest arguments), U1_BUDGET_S (engine_ab wall-clock budget, s),
# ENGINE_AB_ARGS (e.g. `--smoke` on a shared GPU: small rows, plumbing only), ENGINE_AB_NAME (result stem).
set +e
set -u
ROOT=${CQSA_SRC:?set CQSA_SRC to the frozen code root}
V13=${CQSA_V13_SO:?set CQSA_V13_SO to the frozen cqsa_cuda_v13 directory}
RES=${CQSA_SRC_RESULTS:-/scratch/gpfs/AKEY/yb2807/Stream-CQSA-v3/results/ultimate}
JOB=${SLURM_JOB_ID:-nojob}
KER=/scratch/gpfs/AKEY/yb2807/Stream-CQSA-dev/next/kernel
SUITE_KERNELS=${SUITE_KERNELS:-$KER/frozen_suite/2026-10-21}      # v11x / v12f / v12b modules for their own tests
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

# ---------------------------------------------------------------- step 1: the full suite on v13 (opt-in)
RC1=skipped
if [ "${U1_RUN_SUITE:-0}" = 1 ]; then
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
else
echo "== U1 step 1 (suite) skipped: run on vis1 by the package (U1_RUN_SUITE=1 enables it here)"
fi

# ---------------------------------------------------------------- step 2: engine-level A/B
echo "== U1 step 2: engine_ab  $(date +%T)"
if [ ! -f "$V223_DIR/.complete" ]; then
  mkdir -p "$V223_DIR" && git -C /scratch/gpfs/AKEY/yb2807/Stream-CQSA-v2 archive v2.2.3 | tar -x -C "$V223_DIR" && touch "$V223_DIR/.complete"
fi
PYTHONPATH=$V223_DIR:$KER/stable:${PYTHONPATH:-} CQSA_CUDA_MODULE=cqsa_cuda_next_v11 CQSA_CUDA_MODULE_NONCAUSAL=cqsa_cuda_next_v9 \
  python -P -c "import stream_cqsa, stream_cqsa.interface as I; print('v2.2.3 arm:', stream_cqsa.__file__, stream_cqsa.__version__, I.cqsa_cuda.__file__)"
export CQSA_SRC=$ROOT V223_DIR V13_DIR=$V13
AB=${ENGINE_AB_NAME:-engine_ab}
python -P -u "$ROOT/benchmarks/ultimate/engine_ab.py" run --out "$RES/$AB.json" --reps 2 --warmup 1 --budget-s "${U1_BUDGET_S:-1500}" ${ENGINE_AB_ARGS:-}
RC2=$?
echo "U1 step 2 (engine_ab) rc=$RC2  $(date +%T)  results $RES/$AB.json / $AB.md"

echo "== U1 part done: suite rc=$RC1, engine_ab rc=$RC2  $(date)"
