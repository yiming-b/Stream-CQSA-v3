#!/bin/bash
# v3.0.0 wheel, then the v2.2.4 wheel (both torch 2.10 / CUDA 13 / py3.11, sm80+sm90), detached on della-vis1.
LOG=/scratch/gpfs/AKEY/yb2807/Stream-CQSA-v3/results/v3/logs/wheel_chain.log
{
  echo "== chain start $(date)"
  MAX_JOBS=8 NVCC_THREADS=4 bash /scratch/gpfs/AKEY/yb2807/Stream-CQSA-v3/native/wheel_build.sh
  echo "== v3 wheel rc=$? $(date)"
  MAX_JOBS=8 NVCC_THREADS=4 bash /scratch/gpfs/AKEY/yb2807/Stream-CQSA-v2/native/wheel_build.sh
  echo "== v2 wheel rc=$? $(date)"
  echo "== chain end $(date)"
} >> "$LOG" 2>&1
