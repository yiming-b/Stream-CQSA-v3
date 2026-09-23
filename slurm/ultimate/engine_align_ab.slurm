#!/bin/bash
#SBATCH --job-name=cqsa_engine_align_ab
#SBATCH --account=akey
#SBATCH --qos=gpu-test
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=24G
#SBATCH --gres=gpu:a100:1 --constraint="sxm&gpu80"
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/gpfs/AKEY/yb2807/Stream-CQSA-ultimate/results/ultimate/engine_align_ab_%j.out
# P0.1 A/B: classic engine device-resident forward, legacy vs block-aligned task layout, with the
# wave engine and monolithic SDPA-flash arms, at 131K (c=7, 73) and 1M (c=7), acc=GPU.
# --mem: everything lives on the device (Q/K/V 3 GiB + fp32 out 2 GiB + gathered copies at 1M);
# the host holds the pinned task lists (a few hundred MiB at 1M) and the torch/CUDA runtime.
set -uo pipefail
cd /scratch/gpfs/AKEY/yb2807/Stream-CQSA-ultimate
source /scratch/gpfs/AKEY/yb2807/Stream-CQSA-dev/next/env_next.sh
export PYTHONPATH=/scratch/gpfs/AKEY/yb2807/Stream-CQSA-ultimate:/scratch/gpfs/AKEY/yb2807/Stream-CQSA-dev/next/kernel/stable:${PYTHONPATH:-}:/scratch/gpfs/AKEY/yb2807/Stream-CQSA-dev/next/native
export CQSA_CUDA_MODULE=cqsa_cuda_next_v11 CQSA_CUDA_MODULE_NONCAUSAL=cqsa_cuda_next_v9 CQSA_CPU_THREADS=8 OMP_NUM_THREADS=8
which python; hostname
nvidia-smi --query-gpu=name,clocks.sm,clocks.max.sm,memory.total --format=csv,noheader
python -P -c "import stream_cqsa, torch; print(stream_cqsa.__file__, torch.__version__)"
git -C /scratch/gpfs/AKEY/yb2807/Stream-CQSA-ultimate log --oneline -1
python -P -u benchmarks/ultimate/engine_align_ab.py run --out results/ultimate/engine_align_ab.json --reps 3 --warmup 1
