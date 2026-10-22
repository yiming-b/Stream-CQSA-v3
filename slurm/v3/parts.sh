#!/bin/bash
# The part runner of the v3 experiment jobs (sourced by slurm/v3/E_sxm.slurm and E_pcie.slurm): runs the parts in
# CQSA_PARTS ("E1:fwd", "E4", ...) one after the other, each as
#     bench.py run --exp <E> [--part <p>] --only-missing --budget-s <budget> --out $CQSA_V3_RESULTS
# under `timeout`, from the frozen snapshot CQSA_SRC with the frozen modules CQSA_V13_SO / CQSA_NATIVE_V2_SO, results
# into the live repo's results/v3/. Resumable: every part skips the rows already recorded, defers the rows its budget
# cannot hold, and the job ends with the report and the count of missing rows (resubmit until it is 0).
#
# Environment (required): CQSA_SRC, CQSA_V13_SO, CQSA_NATIVE_V2_SO, CQSA_JOB_TOTAL_S (the job's time limit in s).
# Optional: CQSA_PARTS (the parts, in order), CQSA_V3_RESULTS (default <live repo>/results/v3),
# CQSA_PART_BUDGET_S (per-part budget; 0 = whatever remains of the job), CQSA_RESERVE_S (kept for the report; 180),
# E_ARGS (extra bench.py arguments, e.g. "--retry-errors" or "--reps 1").
set -u
ROOT=${CQSA_SRC:?CQSA_SRC (frozen code root, slurm/ultimate/snapshot.sh) is required}
V13=${CQSA_V13_SO:?CQSA_V13_SO (frozen cqsa_cuda_v13 directory) is required}
NV2=${CQSA_NATIVE_V2_SO:?CQSA_NATIVE_V2_SO (frozen cqsa_native_v2 directory) is required}
TOTAL=${CQSA_JOB_TOTAL_S:?CQSA_JOB_TOTAL_S (job time limit, s) is required}
job_limit_s() {      # "59:00" | "12:00:00" | "1-00:00:00" -> seconds
  local l=$1 d=0 h=0 m=0 s=0 a b c
  [[ $l == *-* ]] && { d=${l%%-*}; l=${l#*-}; }
  IFS=: read -r a b c <<<"$l"
  if [ -n "${c:-}" ]; then h=$a; m=$b; s=$c; elif [ -n "${b:-}" ]; then m=$a; s=$b; else s=$a; fi
  echo $((10#$d * 86400 + 10#$h * 3600 + 10#$m * 60 + 10#$s))
}
if [ -n "${SLURM_JOB_ID:-}" ]; then      # the real limit (an sbatch --time on the command line overrides the script's)
  lim=$(squeue -h -j "$SLURM_JOB_ID" -o %l 2>/dev/null | head -1)
  if [ -n "$lim" ] && [ "$lim" != "UNLIMITED" ]; then t=$(job_limit_s "$lim"); [ "$t" -gt 0 ] && TOTAL=$t; fi
fi
RES=${CQSA_V3_RESULTS:-/scratch/gpfs/AKEY/yb2807/Stream-CQSA-v3/results/v3}
RESERVE=${CQSA_RESERVE_S:-180}
PART_BUDGET=${CQSA_PART_BUDGET_S:-0}
DEV=/scratch/gpfs/AKEY/yb2807/Stream-CQSA-dev
KER=$DEV/next/kernel
mkdir -p "$RES/logs"
echo "== v3 job ${SLURM_JOB_ID:-nojob} on $(hostname)  $(date)"
nvidia-smi --query-gpu=name,clocks.sm,clocks.max.sm,memory.total --format=csv,noheader
echo "code $ROOT | v13 $V13 | native_v2 $NV2 | results $RES | parts: ${CQSA_PARTS:-?} | part budget ${PART_BUDGET}s | total ${TOTAL}s"
source $DEV/next/env_next.sh
# the variant dirs carry a stub stream_cqsa/__init__.py for their setup.py, so they go LAST
export PYTHONPATH=$ROOT:$KER/stable:${PYTHONPATH:-}:$DEV/next/native:$V13:$NV2
export CQSA_CPU_THREADS=${CQSA_CPU_THREADS:-4} OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export CQSA_CACHE_DIR=$(mktemp -d)       # no calibration cache from another node type: the memory model only
unset CQSA_CUDA_MODULE CQSA_CUDA_MODULE_NONCAUSAL CQSA_BACKWARD CQSA_FORWARD
cd "$ROOT" || { echo "cannot cd to $ROOT"; exit 1; }
which python
python -P -c "import torch, flash_attn, stream_cqsa, stream_cqsa.interface as I, stream_cqsa.native_wave as W; e = W.native_ext(); print(stream_cqsa.__file__, torch.__version__, 'flash_attn', flash_attn.__version__); print('classic module', I.CQSA_CUDA_MODULE, I.cqsa_cuda.__file__, 'backward', I.backward_kernel()); print('wave module', e.__name__, e.__file__, 'WAVE_API', W.wave_api(e))" \
  || { echo "module import failed"; exit 1; }
md5sum "$V13"/cqsa_cuda_v13*.so "$NV2"/cqsa_native_v2*.so
for part in ${CQSA_PARTS:?CQSA_PARTS is required}; do
  exp=${part%%:*}; sub=${part#*:}; [ "$sub" = "$part" ] && sub=""
  remaining=$((TOTAL - SECONDS - RESERVE))
  if [ "$remaining" -lt 300 ]; then echo "== part $part: skipped, only $remaining s of the job left  $(date +%T)"; continue; fi
  budget=$remaining
  if [ "$PART_BUDGET" -gt 0 ] && [ "$PART_BUDGET" -lt "$budget" ]; then budget=$PART_BUDGET; fi
  tmo=$((budget + 300)); hard=$((TOTAL - SECONDS - 60)); [ "$tmo" -gt "$hard" ] && tmo=$hard
  log=$RES/logs/${SLURM_JOB_ID:-nojob}_${exp}${sub:+_$sub}.log
  echo "== part $part: budget ${budget}s, timeout ${tmo}s, starts $(date +%T) -> $log"
  timeout -s INT "$tmo" python -P -u "$ROOT/experiments/v3/bench.py" run --exp "$exp" ${sub:+--part "$sub"} --only-missing \
      --budget-s "$budget" --out "$RES" ${E_ARGS:-} > "$log" 2>&1
  rc=$?
  echo "== part $part: exit $rc (124 = timeout)  $(date +%T)"
  grep -E "^== done|OOM:|ERROR:|deferred" "$log" | tail -6
done
echo "== report  $(date +%T)"
python -P "$ROOT/experiments/v3/bench.py" report --out "$RES" 2>&1 | tail -3
python -P "$ROOT/experiments/v3/bench.py" status --out "$RES" | tail -1      # "MISSING <n>": resubmit until 0
sacct -j "${SLURM_JOB_ID:-0}" -o JobID,MaxRSS,Elapsed -P 2>/dev/null | tail -3
echo "== v3 job done  $(date)"
