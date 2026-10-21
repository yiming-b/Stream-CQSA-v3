#!/bin/bash
# Build the v13 module (ultimate program U1: v12f forward + v12b backward, full LEGACY mask paths).
#   TAG=a1 bash kernels/v13/build.sh                              # incremental (a100_fp16_hdim64_128)
#   CLEAN=1 TAG=c1 CQSA_KERNEL_SET=common bash kernels/v13/build.sh
# Detached on della-vis1 (nvcc 13.0):
#   TAG=a1 nohup setsid bash kernels/v13/build.sh > /dev/null 2>&1 < /dev/null &
# The LEGACY units carry v11's per-element path and dominate the build (20-65 min per unit
# depending on the node's load); ninja resumes finished objects, so an ALIGNED-only change
# rebuilds only the main units. Logs: kernels/v13/logs/ (git-ignored), nvcc -time csv under build/.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TAG=${TAG:-a1}
source /scratch/gpfs/AKEY/yb2807/Stream-CQSA-dev/env.sh
export CUDA_HOME=/usr/local/cuda-13.0
export PATH=$CUDA_HOME/bin:$PATH
export FLASH_ATTN_CUDA_ARCHS=80
export TORCH_CUDA_ARCH_LIST="8.0"
export CQSA_KERNEL_SET=${CQSA_KERNEL_SET:-a100_fp16_hdim64_128}
# Forced, not defaulted: the session environment carries MAX_JOBS=4.
export MAX_JOBS=${MAX_JOBS_V13:-8}
export NVCC_THREADS=4
unset CQSA_BWD_NO_CQS CQSA_NO_ELEMENT_PATH
# One csv path for every build of a kernel set: it is part of each nvcc command line, so a
# TAG-dependent path would make ninja rebuild every unit on a new TAG (P3 lesson).
mkdir -p "$HERE/logs" "$HERE/build"
export CQSA_NVCC_TIME_CSV=$HERE/build/nvcc_time_${CQSA_KERNEL_SET}.csv
TAG_CSV=$HERE/logs/nvcc_time_${TAG}.csv
LOG=$HERE/logs/build_${TAG}.log
WATCH_LOG=$HERE/logs/watchdog_${TAG}.log
cd "$HERE"
[ "${CLEAN:-0}" = 1 ] && rm -rf build && mkdir -p build
rm -f "$CQSA_NVCC_TIME_CSV" cqsa_cuda_v13*.so
{
  echo "build v13 $TAG: kernel_set=$CQSA_KERNEL_SET MAX_JOBS=$MAX_JOBS NVCC_THREADS=$NVCC_THREADS clean=${CLEAN:-0} host=$(hostname) load=$(cut -d' ' -f1-3 /proc/loadavg)  $(date)"
  echo "flags: full LEGACY mask paths (no CQSA_BWD_NO_CQS / CQSA_NO_ELEMENT_PATH) -lineinfo -Xptxas -v -time $CQSA_NVCC_TIME_CSV (copied to $TAG_CSV)"
  echo "git: $(git -C "$HERE" rev-parse --short HEAD 2>/dev/null) $(git -C "$HERE" status --short -- "$HERE" | wc -l) uncommitted paths under kernels/v13"
  nvcc --version | tail -1
} > "$LOG"
# Scoped watchdog (v12b's pattern): only ptxas processes descended from THIS build are touched
# (other agents build on vis1 at the same time); 75 min flags a hang, not a slow unit (v11's
# LEGACY ptxas took up to ~35 min unloaded, the F23 rebuild under load 55-65 ran ~65 min units).
t0=$(date +%s)
python setup.py build_ext --inplace >> "$LOG" 2>&1 &
BUILD=$!
(
  while kill -0 $BUILD 2>/dev/null; do
    for pid in $(pgrep -x ptxas); do
      p=$pid; mine=0
      while [ -n "$p" ] && [ "$p" -gt 1 ]; do
        [ "$p" = "$BUILD" ] && { mine=1; break; }
        p=$(ps -o ppid= -p "$p" 2>/dev/null | tr -d ' ')
      done
      [ $mine = 1 ] || continue
      mins=$(ps -o etimes= -p "$pid" 2>/dev/null | awk '{print int($1/60)}')
      if [ -n "$mins" ] && [ "$mins" -ge 75 ]; then
        echo "$(date) killing ptxas $pid after $mins min"; kill -9 "$pid"
      fi
    done
    sleep 60
  done
) > "$WATCH_LOG" 2>&1 &
WD=$!
wait $BUILD
rc=$?
echo "build v13 $TAG finished rc=$rc in $(( $(date +%s) - t0 )) s  $(date)" | tee -a "$LOG"
kill $WD 2>/dev/null
cp -f "$CQSA_NVCC_TIME_CSV" "$TAG_CSV" 2>/dev/null
ls -la "$HERE"/cqsa_cuda_v13*.so 2>&1 | tee -a "$LOG"
exit $rc
