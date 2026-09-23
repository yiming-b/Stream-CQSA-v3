#!/bin/bash
#SBATCH --job-name=cqsa_wave_merge_ab
#SBATCH --account=akey
#SBATCH --qos=gpu-test
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=32G
#SBATCH --gres=gpu:a100:1 --constraint="sxm&gpu80"
#SBATCH --time=00:40:00
#SBATCH --output=/scratch/gpfs/AKEY/yb2807/Stream-CQSA-ultimate/results/ultimate/wave_merge_ab_%j.out
# P0.3 A/B: wave engine host-accumulator fold sync vs async vs the classic engine, acc=CPU, 1M and 2M.
# --mem sized from the largest arm (host-resident, N=2M fp16 H=8 D=64, token-major inputs so no
# per-call copy): Q/K/V 6 GiB + fp32 out 4 GiB + fp32 accumulator 4 GiB + two pinned fold slots 6 GiB
# (1.5M touched tokens x 2120 B each) = 20 GiB, plus the torch/CUDA runtime; one arm per subprocess.
set -uo pipefail
cd /scratch/gpfs/AKEY/yb2807/Stream-CQSA-ultimate
source /scratch/gpfs/AKEY/yb2807/Stream-CQSA-dev/next/env_next.sh
export PYTHONPATH=/scratch/gpfs/AKEY/yb2807/Stream-CQSA-ultimate:/scratch/gpfs/AKEY/yb2807/Stream-CQSA-dev/next/kernel/stable:${PYTHONPATH:-}:/scratch/gpfs/AKEY/yb2807/Stream-CQSA-dev/next/native
export CQSA_CUDA_MODULE=cqsa_cuda_next_v11 CQSA_CUDA_MODULE_NONCAUSAL=cqsa_cuda_next_v9 CQSA_BACKWARD=cuda CQSA_CPU_THREADS=8 OMP_NUM_THREADS=8
which python; hostname
nvidia-smi --query-gpu=name,clocks.sm,clocks.max.sm,memory.total --format=csv,noheader
python -P -c "import stream_cqsa, torch; print(stream_cqsa.__file__, torch.__version__)"
git -C /scratch/gpfs/AKEY/yb2807/Stream-CQSA-ultimate log --oneline -1
python -P -u benchmarks/ultimate/wave_merge_ab.py run --out results/ultimate/wave_merge_ab.json --reps 3 --warmup 1
