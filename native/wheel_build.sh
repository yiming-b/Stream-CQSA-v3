#!/bin/bash
# Build the release wheel of Stream-CQSA-v3 on della: the classic CUDA module (csrc/, the v13 design) and the
# native wave kernel (native/), torch 2.10 / CUDA 13, sm80 + sm90.
#   sbatch slurm/v3/wheel_build.slurm            (a CPU node; 3-5 h on 32 cores)
#   nohup bash native/wheel_build.sh &           (della-vis1; shares the login node's cores)
# build/ is kept between runs: finished objects are reused when the sources and flags are unchanged.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source /scratch/gpfs/AKEY/yb2807/Stream-CQSA-dev/env.sh
export CUDA_HOME=/usr/local/cuda-13.0
export PATH=$CUDA_HOME/bin:$PATH
export FLASH_ATTN_CUDA_ARCHS="80;90" TORCH_CUDA_ARCH_LIST="8.0;9.0"
export CQSA_KERNEL_SET=common CQSA_BUILD_NATIVE=1
export MAX_JOBS=${MAX_JOBS:-8} NVCC_THREADS=${NVCC_THREADS:-4}
cd "$HERE"
rm -rf dist && mkdir -p dist
echo "wheel build start $(date) host=$(hostname) MAX_JOBS=$MAX_JOBS NVCC_THREADS=$NVCC_THREADS"; t0=$(date +%s)
nvcc --version | tail -1
python setup.py bdist_wheel --dist-dir dist 2>&1 | grep -v "^\s*$" | grep -iv "warning" | tail -40
rc=${PIPESTATUS[0]}
python -m wheel tags --build "1cu130torch210sm8090" --remove dist/*.whl
echo "wheel build finished rc=$rc in $(( $(date +%s) - t0 )) s  $(date)"; ls -la dist
exit $rc
