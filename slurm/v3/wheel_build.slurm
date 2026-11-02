#!/bin/bash
#SBATCH --job-name=cqsa_v3_wheel
#SBATCH --account=akey
#SBATCH --partition=cpu
#SBATCH --qos=short
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=32 --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/gpfs/AKEY/yb2807/Stream-CQSA-v3/results/v3/logs/wheel_%j.out
# The release wheel (torch 2.10 / CUDA 13 / py3.11, sm80 + sm90, classic module + wave kernel) on a CPU node:
#   sbatch slurm/v3/wheel_build.slurm      -> dist/stream_cqsa-3.0.0-1cu130torch210sm8090-cp311-cp311-linux_x86_64.whl
# --mem: eight nvcc processes at 4-6 GB each (the v13 and native builds used 64G on 32 cores).
set -uo pipefail
cd /scratch/gpfs/AKEY/yb2807/Stream-CQSA-v3
MAX_JOBS=8 NVCC_THREADS=4 bash native/wheel_build.sh
