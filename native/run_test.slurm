#!/bin/bash
# The wave kernel v2's exactness tests on one A100-80GB (correctness only; the A/B is slurm/ultimate/parts/u2.sh).
#   sbatch native/run_test.slurm   (from the repo root; the reference modules v11 / v12f / v12b come from next/kernel)
#SBATCH --job-name=native_v2_test
#SBATCH --account=akey
#SBATCH --qos=gpu-test
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=16G
#SBATCH --gres=gpu:a100:1 --constraint=gpu80
#SBATCH --time=00:30:00
#SBATCH --output=results/native/%x_%j.out
set -uo pipefail
source /scratch/gpfs/AKEY/yb2807/Stream-CQSA-dev/next/env_next.sh
KER=/scratch/gpfs/AKEY/yb2807/Stream-CQSA-dev/next/kernel
export PYTHONPATH="$PWD:$KER/stable:${PYTHONPATH:-}:/scratch/gpfs/AKEY/yb2807/Stream-CQSA-dev/next/native:$KER/v12f/frozen/b3:$KER/v12b/frozen/b5:$PWD/native"
export CQSA_CUDA_MODULE=cqsa_cuda_next_v11 CQSA_CUDA_MODULE_NONCAUSAL=cqsa_cuda_next_v9 CQSA_CPU_THREADS=4 OMP_NUM_THREADS=4
nvidia-smi --query-gpu=name --format=csv,noheader
python -P -m pytest -q tests/test_wave_v2.py tests/test_wave_shapes.py tests/test_ultimate_wave_merge.py
