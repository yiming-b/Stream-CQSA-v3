#!/bin/bash
# Build v12b (ultimate program P2: ALIGNED run-table backward + LEGACY v11 backward).
#   TAG=b1 bash next/kernel/v12b/build.sh          # incremental: reuses up-to-date objects
#   CLEAN=1 TAG=b1 bash next/kernel/v12b/build.sh  # from scratch
# Detached on della-vis1 (nvcc 13.0):
#   TAG=b1 CLEAN=1 nohup setsid bash next/kernel/v12b/build.sh > /dev/null 2>&1 &
# The forward is v11x's (-DCQSA_NO_ELEMENT_PATH: plain/zero patterns only, a real
# pattern's forward comes back NaN) -- this package measures the backward only,
# and it keeps the forward units short. The backward is NOT built with
# CQSA_BWD_NO_CQS: its CQS-on path is the census + ALIGNED kernel + LEGACY (v11).
set -uo pipefail
DEV=/scratch/gpfs/AKEY/yb2807/Stream-CQSA-dev
TAG=${TAG:-b1}
source $DEV/env.sh
export CUDA_HOME=/usr/local/cuda-13.0
export PATH=$CUDA_HOME/bin:$PATH
export FLASH_ATTN_CUDA_ARCHS=80
export TORCH_CUDA_ARCH_LIST="8.0"
export CQSA_KERNEL_SET=${CQSA_KERNEL_SET:-a100_fp16_hdim64_128}
# Forced, not defaulted: the session environment carries MAX_JOBS=4.
export MAX_JOBS=${MAX_JOBS_V12B:-8}
export NVCC_THREADS=4
unset CQSA_BWD_NO_CQS
export CQSA_NO_ELEMENT_PATH=1
# The csv path is part of the nvcc command line, so a new path makes ninja rebuild EVERY unit;
# CSV_TAG keeps an earlier build's path for an incremental rebuild (the csv is appended to, not reset).
CSV_TAG=${CSV_TAG:-$TAG}
export CQSA_NVCC_TIME_CSV=$DEV/next/logs/build_v12b_${CSV_TAG}_nvcc_time.csv
LOG=$DEV/next/logs/06_build_v12b_${TAG}.log
WATCH_LOG=$DEV/next/logs/watchdog_v12b_${TAG}.log
[ "$CSV_TAG" = "$TAG" ] && rm -f "$CQSA_NVCC_TIME_CSV"
cd $DEV/next/kernel/v12b
[ "${CLEAN:-0}" = 1 ] && rm -rf build
rm -f cqsa_cuda_next_v12b*.so
{
  echo "build v12b $TAG: kernel_set=$CQSA_KERNEL_SET MAX_JOBS=$MAX_JOBS NVCC_THREADS=$NVCC_THREADS clean=${CLEAN:-0} host=$(hostname) load=$(cut -d' ' -f1-3 /proc/loadavg)  $(date)"
  echo "flags: CQSA_NO_ELEMENT_PATH=1 (fwd) -lineinfo -Xptxas -v -time $CQSA_NVCC_TIME_CSV"
  nvcc --version | tail -1
} > "$LOG"
# LEGACY's backward took ~9 min per unit in v11's build; 45 min flags a hang, not a slow unit.
# Unlike next/kernel/watchdog.sh this one only touches ptxas processes descended from THIS build
# (other agents build on vis1 at the same time: the v12f build, the v2 wheel build).
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
      if [ -n "$mins" ] && [ "$mins" -ge 45 ]; then
        echo "$(date) killing ptxas $pid after $mins min"; kill -9 "$pid"
      fi
    done
    sleep 60
  done
) > "$WATCH_LOG" 2>&1 &
WD=$!
wait $BUILD
rc=$?
echo "build v12b $TAG finished rc=$rc in $(( $(date +%s) - t0 )) s  $(date)" | tee -a "$LOG"
kill $WD 2>/dev/null
ls -la $DEV/next/kernel/v12b/cqsa_cuda_next_v12b*.so 2>&1 | tee -a "$LOG"
exit $rc
