#!/bin/bash
#SBATCH --job-name=cqsa_fwd_ab
#SBATCH --account=akey
#SBATCH --qos=gpu-test
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=12G
#SBATCH --gres=gpu:a100:1 --constraint="sxm&gpu80"
#SBATCH --time=00:45:00
#SBATCH --output=/scratch/gpfs/AKEY/yb2807/Stream-CQSA-ultimate/results/ultimate/fwd_ab_%j.out
# P3 forward A/B: v12f (census + ALIGNED run-table forward, LEGACY = v11) vs v11 / v11x / v9 / flash_attn /
# SDPA-flash, one process on one A100-SXM4-80GB. Every tensor lives on the device (random Q/K/V generated
# there, the largest set N=524288 x 8 x 64 fp16 plus the gathered L~225K copies and a fp32 output: < 4 GiB);
# the host holds the task tables and the interpreter (exp_build_ab's comparable job peaked at 1.27 GB RSS).
#   SNAP=$(bash slurm/ultimate/snapshot.sh) && sbatch --export=ALL,CQSA_SRC=$SNAP slurm/ultimate/fwd_ab.slurm
set -uo pipefail
# Code comes from a frozen snapshot (CQSA_SRC, a `git archive` of one commit) so edits to the live
# worktree while this job waits in the queue cannot change what it runs; results go to the worktree.
# The kernel modules are the built trees under next/kernel (not in git): do not rebuild v12f while
# this job is queued or running.
OUT=/scratch/gpfs/AKEY/yb2807/Stream-CQSA-ultimate
ROOT=${CQSA_SRC:-$OUT}
KER=/scratch/gpfs/AKEY/yb2807/Stream-CQSA-dev/next/kernel
cd "$ROOT"
echo "code: $ROOT"
source /scratch/gpfs/AKEY/yb2807/Stream-CQSA-dev/next/env_next.sh
# the variant dirs carry a stub stream_cqsa/__init__.py for their setup.py, so they go LAST
export PYTHONPATH=$ROOT:$KER/stable:${PYTHONPATH:-}:/scratch/gpfs/AKEY/yb2807/Stream-CQSA-dev/next/native:$KER/v11x:$KER/v12f
export CQSA_CUDA_MODULE=cqsa_cuda_next_v11 CQSA_CUDA_MODULE_NONCAUSAL=cqsa_cuda_next_v9 CQSA_CPU_THREADS=4 OMP_NUM_THREADS=4
which python; hostname
nvidia-smi --query-gpu=name,clocks.sm,clocks.max.sm,memory.total --format=csv,noheader
python -P -c "import stream_cqsa, torch, flash_attn, cqsa_cuda_next_v12f as m; print(stream_cqsa.__file__, torch.__version__, flash_attn.__version__, m.__file__)"
ls -la --time-style=full-iso $KER/v12f/cqsa_cuda_next_v12f*.so
python -P -u "$ROOT/benchmarks/ultimate/fwd_ab.py" --out "$OUT/results/ultimate/fwd_ab.json"
echo "exit $?  $(date +%T)"
sacct -j "$SLURM_JOB_ID" -o JobID,MaxRSS,Elapsed -P 2>/dev/null | tail -3
