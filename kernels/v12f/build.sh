#!/bin/bash
# Build v12f (ultimate program P3: ALIGNED run-table forward + LEGACY v11 forward).
#   TAG=b1 bash next/kernel/v12f/build.sh          # incremental: reuses up-to-date objects
#   CLEAN=1 TAG=b1 bash next/kernel/v12f/build.sh  # from scratch
# Detached on della-vis1 (nvcc 13.0):
#   nohup setsid bash next/kernel/v12f/build.sh > /dev/null 2>&1 &
# The backward is compiled with CQSA_BWD_NO_CQS (not correct for CQS patterns; this
# package measures the forward only, and it keeps the build short). The LEGACY units
# carry v11's per-element path and dominate the build (v11: 17-40 min per unit); ninja
# resumes finished objects, so an ALIGNED-only change rebuilds only the main units.
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
export MAX_JOBS=8
export NVCC_THREADS=4
export CQSA_BWD_NO_CQS=1
# One csv path for every build: it is part of each nvcc command (-time), so a TAG-dependent path made
# ninja rebuild every unit on a new TAG (b2 rebuilt all 12 in 79 min for a flash_api.cpp-only change).
# Units rebuilt by this run append to it; it is copied to the TAG's name at the end.
export CQSA_NVCC_TIME_CSV=$DEV/next/logs/build_v12f_nvcc_time.csv
TAG_CSV=$DEV/next/logs/build_v12f_${TAG}_nvcc_time.csv
LOG=$DEV/next/logs/05_build_v12f_${TAG}.log
WATCH_LOG=$DEV/next/logs/watchdog_v12f_${TAG}.log
rm -f "$CQSA_NVCC_TIME_CSV"
cd $DEV/next/kernel/v12f
[ "${CLEAN:-0}" = 1 ] && rm -rf build
rm -f cqsa_cuda_next_v12f*.so
{
  echo "build v12f $TAG: kernel_set=$CQSA_KERNEL_SET MAX_JOBS=$MAX_JOBS NVCC_THREADS=$NVCC_THREADS clean=${CLEAN:-0} host=$(hostname) load=$(cut -d' ' -f1-3 /proc/loadavg)  $(date)"
  echo "flags: CQSA_BWD_NO_CQS=1 -lineinfo -Xptxas -v -time $CQSA_NVCC_TIME_CSV (copied to $TAG_CSV)"
  nvcc --version | tail -1
} > "$LOG"
# LEGACY's ptxas took up to ~35 min in v11's build; 60 min flags a hang, not a slow unit.
# F23: scoped watchdog (v12b's pattern) -- only ptxas processes descended from THIS build are
# touched; next/kernel/watchdog.sh kills ANY old ptxas, including other agents' builds on vis1.
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
      if [ -n "$mins" ] && [ "$mins" -ge 60 ]; then
        echo "$(date) killing ptxas $pid after $mins min"; kill -9 "$pid"
      fi
    done
    sleep 60
  done
) > "$WATCH_LOG" 2>&1 &
WD=$!
wait $BUILD
rc=$?
echo "build v12f $TAG finished rc=$rc in $(( $(date +%s) - t0 )) s  $(date)" | tee -a "$LOG"
kill $WD 2>/dev/null
cp -f "$CQSA_NVCC_TIME_CSV" "$TAG_CSV" 2>/dev/null
ls -la $DEV/next/kernel/v12f/cqsa_cuda_next_v12f*.so 2>&1 | tee -a "$LOG"
exit $rc
