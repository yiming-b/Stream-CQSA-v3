#!/bin/bash
# Build the native wave extension cqsa_native_v2 (ultimate program package U2) in place.
#   TAG=b1 bash kernels/native_v2/build.sh                         # the `common` set (fp16+bf16, hdim 64+128)
#   TAG=d1 CQSA_KERNEL_SET=native_v2_dev bash kernels/native_v2/build.sh   # hdim64 fp16 only -> cqsa_native_v2_dev
#   CLEAN=1 ... from scratch; otherwise ninja reuses up-to-date objects (its own build dir per kernel set).
# Detached on della-vis1 (nvcc 13.0):
#   TAG=b1 nohup setsid bash kernels/native_v2/build.sh > /dev/null 2>&1 < /dev/null &
# The LEGACY units carry v11's per-element path and dominate the build (v12f/v12b: 20-30 min per
# unit); the main units (every new line of this package) come first and take minutes.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
DEV=/scratch/gpfs/AKEY/yb2807/Stream-CQSA-dev
TAG=${TAG:-b1}
source $DEV/env.sh
export CUDA_HOME=/usr/local/cuda-13.0
export PATH=$CUDA_HOME/bin:$PATH
export FLASH_ATTN_CUDA_ARCHS=80
export TORCH_CUDA_ARCH_LIST="8.0"
export CQSA_KERNEL_SET=${CQSA_KERNEL_SET:-common}
# Forced, not defaulted: the session environment carries MAX_JOBS=4.
export MAX_JOBS=${MAX_JOBS_NATIVE_V2:-8}
export NVCC_THREADS=4
if [ "$CQSA_KERNEL_SET" = native_v2_dev ]; then
  export CQSA_NATIVE_V2_MODULE=cqsa_native_v2_dev
else
  export CQSA_NATIVE_V2_MODULE=cqsa_native_v2
fi
BUILD_DIR=$HERE/build_$CQSA_KERNEL_SET
# One csv path per kernel set: it is part of every nvcc command (-time), so a TAG-dependent
# path would make ninja rebuild every unit on a new TAG (v12f b2 lost 79 min that way).
export CQSA_NVCC_TIME_CSV=$DEV/next/logs/build_native_v2_${CQSA_KERNEL_SET}_nvcc_time.csv
TAG_CSV=$DEV/next/logs/build_native_v2_${TAG}_nvcc_time.csv
LOG=$DEV/next/logs/07_build_native_v2_${TAG}.log
WATCH_LOG=$DEV/next/logs/watchdog_native_v2_${TAG}.log
cd "$HERE"
[ "${CLEAN:-0}" = 1 ] && rm -rf "$BUILD_DIR"
rm -f "$HERE"/${CQSA_NATIVE_V2_MODULE}.*.so
{
  echo "build native_v2 $TAG: kernel_set=$CQSA_KERNEL_SET module=$CQSA_NATIVE_V2_MODULE MAX_JOBS=$MAX_JOBS NVCC_THREADS=$NVCC_THREADS clean=${CLEAN:-0} host=$(hostname) load=$(cut -d' ' -f1-3 /proc/loadavg)  $(date)"
  echo "flags: -lineinfo -Xptxas -v -time $CQSA_NVCC_TIME_CSV (copied to $TAG_CSV)"
  nvcc --version | tail -1
} > "$LOG"
t0=$(date +%s)
python setup.py build_ext --inplace --build-temp "$BUILD_DIR" >> "$LOG" 2>&1 &
BUILD=$!
# Scoped watchdog: only ptxas processes descended from THIS build are touched (other agents build
# on vis1 too; kernels/watchdog.sh kills ANY old ptxas). LEGACY's ptxas took up to ~35 min in v11's
# build; 60 min flags a hang, not a slow unit.
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
echo "build native_v2 $TAG finished rc=$rc in $(( $(date +%s) - t0 )) s  $(date)" | tee -a "$LOG"
kill $WD 2>/dev/null
cp -f "$CQSA_NVCC_TIME_CSV" "$TAG_CSV" 2>/dev/null
ls -la "$HERE"/${CQSA_NATIVE_V2_MODULE}.*.so 2>&1 | tee -a "$LOG"
exit $rc
