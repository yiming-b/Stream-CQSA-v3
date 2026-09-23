#!/bin/bash
#SBATCH --job-name=cqsa_kernel_rows
#SBATCH --account=akey
#SBATCH --qos=gpu-test
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=24G
#SBATCH --gres=gpu:a100:1 --constraint="sxm&gpu80"
#SBATCH --time=00:45:00
#SBATCH --output=/scratch/gpfs/AKEY/yb2807/Stream-CQSA-ultimate/results/ultimate/kernel_rows_%j.out
# M0.1b kernel rows (hdim128, non-causal v9, bf16, synthetic patterns, ragged N, itr=2 + wave) on one A100-SXM4.
# Host tensors < 3 GiB (49 itr=2 task tables + token-major inputs); 24G covers torch itself (~6G) plus that.
set -uo pipefail
ULT=/scratch/gpfs/AKEY/yb2807/Stream-CQSA-ultimate
DEV=/scratch/gpfs/AKEY/yb2807/Stream-CQSA-dev
source $DEV/next/env_next.sh
# the ultimate checkout first (never next/pkg), then the stable kernel copies, cqsa_native last (its build stub must not shadow the package)
export PYTHONPATH="$ULT:$DEV/next/kernel/stable:$PYTHONPATH:$DEV/next/native"
export CQSA_CUDA_MODULE=cqsa_cuda_next_v11 CQSA_CUDA_MODULE_NONCAUSAL=cqsa_cuda_next_v9 CQSA_CPU_THREADS=8 OMP_NUM_THREADS=8
which python; hostname
nvidia-smi --query-gpu=name,clocks.sm,clocks.max.sm,memory.total --format=csv,noheader
python -P -c "import stream_cqsa, torch; print(stream_cqsa.__file__, torch.__version__)"
python -P -u $ULT/benchmarks/ultimate/kernel_rows.py
