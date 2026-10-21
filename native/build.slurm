#!/bin/bash
#SBATCH --job-name=native_v2_build
#SBATCH --account=akey
#SBATCH --partition=cpu
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/gpfs/AKEY/yb2807/Stream-CQSA-dev/next/logs/build_%x_%j.out
# Build the wave kernel v2 of this tree on a CPU node (the common set: fp16 + bf16, head dims 64 + 128;
# ~60-80 min on 32 cores, the hdim64 forward LEGACY units dominate).   sbatch native/build.slurm
set -euo pipefail
cd "$(dirname "$(realpath "$0")")/.."
TAG=${TAG:-slurm_$SLURM_JOB_ID} MAX_JOBS_NATIVE_V2=8 bash native/build.sh
