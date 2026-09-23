#!/bin/bash
#SBATCH --job-name=cqsa_bwd_ab
#SBATCH --account=akey
#SBATCH --qos=gpu-test
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=12G
#SBATCH --gres=gpu:a100:1 --constraint="sxm&gpu80"
#SBATCH --time=00:45:00
#SBATCH --output=/scratch/gpfs/AKEY/yb2807/Stream-CQSA-ultimate/results/ultimate/bwd_ab_%j.out
# P2 backward A/B: v12b (census + ALIGNED run-table backward, LEGACY = v11) vs v11 / v11x (NO_CQS) / flash_attn /
# SDPA-flash, one process on one A100-SXM4-80GB. Every tensor lives on the device (the largest set: L=56064 x 8 x 128
# fp16 q/k/v/do/o plus gradients, < 2 GiB; the c=73 part keeps 73 tasks' o/lse views of one L~16K set, and the
# monolithic flash_attn reference at N=131072); the host holds the task tables and the interpreter
# (exp_build_ab's comparable job peaked at 1.27 GB RSS).
#   SNAP=$(bash slurm/ultimate/snapshot.sh) && sbatch --export=ALL,CQSA_SRC=$SNAP,V12B_DIR=<frozen dir> slurm/ultimate/bwd_ab.slurm
set -uo pipefail
# Code comes from a frozen snapshot (CQSA_SRC, a `git archive` of one commit) so edits to the live
# worktree while this job waits in the queue cannot change what it runs; results go to the worktree.
# V12B_DIR selects a frozen copy of one v12b build (next/kernel/v12b/frozen/<tag>, the .so alone) so a rebuild
# of the variant tree while this job waits cannot change what it loads; default = the variant tree itself.
OUT=/scratch/gpfs/AKEY/yb2807/Stream-CQSA-ultimate
ROOT=${CQSA_SRC:-$OUT}
KER=/scratch/gpfs/AKEY/yb2807/Stream-CQSA-dev/next/kernel
cd "$ROOT"
echo "code: $ROOT"
source /scratch/gpfs/AKEY/yb2807/Stream-CQSA-dev/next/env_next.sh
# the variant dirs carry a stub stream_cqsa/__init__.py for their setup.py, so they go LAST
V12B_DIR=${V12B_DIR:-$KER/v12b}
export PYTHONPATH=$ROOT:$KER/stable:${PYTHONPATH:-}:/scratch/gpfs/AKEY/yb2807/Stream-CQSA-dev/next/native:$KER/v11x:$V12B_DIR
export CQSA_CUDA_MODULE=cqsa_cuda_next_v11 CQSA_CUDA_MODULE_NONCAUSAL=cqsa_cuda_next_v9 CQSA_BACKWARD=cuda CQSA_CPU_THREADS=4 OMP_NUM_THREADS=4
which python; hostname
nvidia-smi --query-gpu=name,clocks.sm,clocks.max.sm,memory.total --format=csv,noheader
python -P -c "import stream_cqsa, torch, flash_attn, cqsa_cuda_next_v12b as m, cqsa_cuda_next_v11x as x; print(stream_cqsa.__file__, torch.__version__, flash_attn.__version__, m.__file__, x.__file__)"
ls -la --time-style=full-iso $V12B_DIR/cqsa_cuda_next_v12b*.so; md5sum $V12B_DIR/cqsa_cuda_next_v12b*.so
python -P -u "$ROOT/benchmarks/ultimate/bwd_ab.py" --out "$OUT/results/ultimate/bwd_ab.json" ${BWD_AB_ARGS:-}
echo "exit $?  $(date +%T)"
sacct -j "$SLURM_JOB_ID" -o JobID,MaxRSS,Elapsed -P 2>/dev/null | tail -3
