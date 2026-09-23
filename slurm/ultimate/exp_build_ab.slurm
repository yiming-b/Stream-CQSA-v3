#!/bin/bash
#SBATCH --job-name=cqsa_exp_build_ab
#SBATCH --account=akey
#SBATCH --qos=gpu-test
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=8G
#SBATCH --gres=gpu:a100:1 --constraint="sxm&gpu80"
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/gpfs/AKEY/yb2807/Stream-CQSA-ultimate/results/ultimate/exp_build_ab_%j.out
# Phase 1 experiment-build A/B: v11 vs v11x (NO_ELEMENT_PATH fwd, NO_CQS bwd) vs v9 vs SDPA-flash.
# Host tensors are < 1 GiB; job 14308401 peaked at 1.27 GB RSS (sacct MaxRSS), so 8G is ample.
set -uo pipefail
source /scratch/gpfs/AKEY/yb2807/Stream-CQSA-dev/next/env_next.sh
# the variant dirs carry a stub stream_cqsa/__init__.py for their setup.py, so they go LAST
export PYTHONPATH=/scratch/gpfs/AKEY/yb2807/Stream-CQSA-ultimate:/scratch/gpfs/AKEY/yb2807/Stream-CQSA-dev/next/kernel/stable:$PYTHONPATH:/scratch/gpfs/AKEY/yb2807/Stream-CQSA-dev/next/native:/scratch/gpfs/AKEY/yb2807/Stream-CQSA-dev/next/kernel/v11x
export CQSA_CUDA_MODULE=cqsa_cuda_next_v11 CQSA_CUDA_MODULE_NONCAUSAL=cqsa_cuda_next_v9 CQSA_BACKWARD=cuda CQSA_CPU_THREADS=4 OMP_NUM_THREADS=4
which python; hostname
nvidia-smi --query-gpu=name,clocks.sm,clocks.max.sm,memory.total --format=csv,noheader
python -P -c "import stream_cqsa, torch; print(stream_cqsa.__file__, torch.__version__)"
python -P -u /scratch/gpfs/AKEY/yb2807/Stream-CQSA-ultimate/benchmarks/ultimate/exp_build_ab.py
