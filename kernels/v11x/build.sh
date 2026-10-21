#!/bin/bash
# Experiment build v11x (Phase 1 of the ultimate-kernel program): v11 with
#   -DCQSA_NO_ELEMENT_PATH   forward per-element mask path compiled out
#   -DCQSA_BWD_NO_CQS        backward CQS compiled out (upstream loop, our contract)
#   -lineinfo -Xptxas -v     source lines in SASS, per-instantiation ptxas report
#   -time <csv>              per-phase nvcc times, one row per phase and TU
# Runs on della-vis1 (nvcc 13.0) under watchdog.sh with LIMIT=45: a TU whose
# ptxas needs more than 45 min is itself a result and is recorded in the log.
#   bash next/kernel/v11x/build.sh
set -uo pipefail
DEV=/scratch/gpfs/AKEY/yb2807/Stream-CQSA-dev
source $DEV/env.sh
export CUDA_HOME=/usr/local/cuda-13.0
export PATH=$CUDA_HOME/bin:$PATH
export FLASH_ATTN_CUDA_ARCHS=80
export TORCH_CUDA_ARCH_LIST="8.0"
export CQSA_KERNEL_SET=${CQSA_KERNEL_SET:-a100_fp16_hdim64_128}
# Forced, not defaulted: the session environment carries MAX_JOBS=4, which
# serialises the four forward TUs into two waves.
export MAX_JOBS=8
export NVCC_THREADS=4
export CQSA_BWD_NO_CQS=1
export CQSA_NO_ELEMENT_PATH=1
export CQSA_NVCC_TIME_CSV=$DEV/next/logs/build_v11x_nvcc_time.csv
LOG=$DEV/next/logs/04_build_v11x.log
WATCH_LOG=$DEV/next/logs/watchdog_v11x.log
rm -f "$CQSA_NVCC_TIME_CSV"
cd $DEV/next/kernel/v11x
rm -rf build cqsa_cuda_next_v11x*.so
{
  echo "build v11x: kernel_set=$CQSA_KERNEL_SET MAX_JOBS=$MAX_JOBS NVCC_THREADS=$NVCC_THREADS host=$(hostname) load=$(cut -d' ' -f1-3 /proc/loadavg)  $(date)"
  echo "flags: CQSA_BWD_NO_CQS=1 CQSA_NO_ELEMENT_PATH=1 -lineinfo -Xptxas -v -time $CQSA_NVCC_TIME_CSV"
  nvcc --version | tail -1
} > "$LOG"
bash $DEV/next/kernel/watchdog.sh 45 > "$WATCH_LOG" 2>&1 &
WD=$!
t0=$(date +%s)
python setup.py build_ext --inplace >> "$LOG" 2>&1
rc=$?
echo "build v11x finished rc=$rc in $(( $(date +%s) - t0 )) s  $(date)" | tee -a "$LOG"
kill $WD 2>/dev/null
ls -la $DEV/next/kernel/v11x/cqsa_cuda_next_v11x*.so 2>&1 | tee -a "$LOG"
exit $rc
