#!/bin/bash
#SBATCH --job-name=cqsa_fold_workers
#SBATCH --account=akey
#SBATCH --qos=gpu-test
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=40G
#SBATCH --gres=gpu:a100:1 --constraint="sxm&gpu80"
#SBATCH --time=00:59:00
#SBATCH --output=/scratch/gpfs/AKEY/yb2807/Stream-CQSA-ultimate/results/ultimate/fold_workers_ab_%j.out
# I0 (P0.3 review MAJOR): wave acc=CPU host fold, one worker vs two token-range workers vs one worker
# with twice the threads vs synchronous vs the classic engine; host-streamed forward at 1M/2M, shapes
# c=7 itr=1, c=73 itr=1, c=7 itr=2. 16 cpus: two workers x CQSA_CPU_THREADS=8 (and the 2x-thread arm).
# --mem from the largest arm (N=2M fp16 H=8 D=64, one arm per subprocess): token-major Q/K/V 6 GiB +
# fp32 out 4 GiB + fp32 accumulator 4 GiB + two pinned fold slots <= 2 x 2M x 2120 B = 8.5 GiB, plus a
# 4 GiB fp32 transient while the inputs are drawn and the runtime; the P0.3 job measured 29.2 GiB
# max RSS for its async 2M arm -> 40G.
set -uo pipefail
OUT=/scratch/gpfs/AKEY/yb2807/Stream-CQSA-ultimate
ROOT=${CQSA_SRC:-$OUT}          # frozen snapshot (slurm/ultimate/snapshot.sh); results go to the worktree
cd "$ROOT"
echo "code: $ROOT"
source /scratch/gpfs/AKEY/yb2807/Stream-CQSA-dev/next/env_next.sh
export PYTHONPATH=$ROOT:/scratch/gpfs/AKEY/yb2807/Stream-CQSA-dev/next/kernel/stable:${PYTHONPATH:-}:/scratch/gpfs/AKEY/yb2807/Stream-CQSA-dev/next/native
export CQSA_CUDA_MODULE=cqsa_cuda_next_v11 CQSA_CUDA_MODULE_NONCAUSAL=cqsa_cuda_next_v9 CQSA_CPU_THREADS=8 OMP_NUM_THREADS=8
hostname; which python
nvidia-smi --query-gpu=name,clocks.sm,clocks.max.sm,memory.total --format=csv,noheader
python -P -c "import stream_cqsa, torch; print(stream_cqsa.__file__, torch.__version__)"
python -P -u benchmarks/ultimate/fold_workers_ab.py run --out "$OUT/results/ultimate/fold_workers_ab.json" --reps 2 --warmup 1
echo "exit $?  $(date +%T)"
sacct -j "$SLURM_JOB_ID" -o JobID,MaxRSS,Elapsed -P 2>/dev/null | tail -3
