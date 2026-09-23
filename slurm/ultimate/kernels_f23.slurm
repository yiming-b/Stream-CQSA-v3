#!/bin/bash
#SBATCH --job-name=cqsa_kernels_f23
#SBATCH --account=akey
#SBATCH --qos=gpu-test
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=16G
#SBATCH --gres=gpu:a100:1 --constraint="sxm&gpu80"
#SBATCH --time=00:50:00
#SBATCH --output=/scratch/gpfs/AKEY/yb2807/Stream-CQSA-ultimate/results/ultimate/kernels_f23_%j.out
# F23 combined kernel A/B (benchmarks/ultimate/kernels_f23.py): v12f b3 + v12b b5 (the review-fixed builds) vs v11 /
# v11x / flash_attn / SDPA-flash, one process on one A100-SXM4-80GB. Every tensor lives on the device: the largest set
# is the H=32 forward at N=131072 (three [1,131072,32,64] fp16 = 1.5 GiB transient, gathered L=56064 copies 0.7 GiB);
# backward L=56064 H=8 q/k/v/do/o + grads + deterministic split dq_accum (~1.6 GB) < 4 GiB. Host: task tables
# (c=73 itr=2 builds 5329 tasks) and the interpreter; the comparable fwd_ab/bwd_ab jobs peaked at 2.1 GB RSS.
#   SNAP=$(bash slurm/ultimate/snapshot.sh) && sbatch --export=ALL,CQSA_SRC=$SNAP,V12F_DIR=<frozen>,V12B_DIR=<frozen> slurm/ultimate/kernels_f23.slurm
set -uo pipefail
OUT=/scratch/gpfs/AKEY/yb2807/Stream-CQSA-ultimate
ROOT=${CQSA_SRC:-$OUT}
KER=/scratch/gpfs/AKEY/yb2807/Stream-CQSA-dev/next/kernel
cd "$ROOT"
echo "code: $ROOT"
source /scratch/gpfs/AKEY/yb2807/Stream-CQSA-dev/next/env_next.sh
V12F_DIR=${V12F_DIR:-$KER/v12f}
V12B_DIR=${V12B_DIR:-$KER/v12b}
# the variant dirs carry a stub stream_cqsa/__init__.py for their setup.py, so they go LAST
export PYTHONPATH=$ROOT:$KER/stable:${PYTHONPATH:-}:/scratch/gpfs/AKEY/yb2807/Stream-CQSA-dev/next/native:$KER/v11x:$V12F_DIR:$V12B_DIR
export CQSA_CUDA_MODULE=cqsa_cuda_next_v11 CQSA_CUDA_MODULE_NONCAUSAL=cqsa_cuda_next_v9 CQSA_BACKWARD=cuda CQSA_CPU_THREADS=4 OMP_NUM_THREADS=4
unset CQSA_V12B_DPSUM_SMEM
which python; hostname
nvidia-smi --query-gpu=name,clocks.sm,clocks.max.sm,memory.total --format=csv,noheader
python -P -c "import stream_cqsa, torch, flash_attn, cqsa_cuda_next_v12f as f, cqsa_cuda_next_v12b as b, cqsa_cuda_next_v11x as x; print(stream_cqsa.__file__, torch.__version__, flash_attn.__version__, f.__file__, b.__file__, x.__file__)"
md5sum $V12F_DIR/cqsa_cuda_next_v12f*.so $V12B_DIR/cqsa_cuda_next_v12b*.so
python -P -u "$ROOT/benchmarks/ultimate/kernels_f23.py" --out "$OUT/results/ultimate/kernels_f23.json" ${F23_ARGS:-}
echo "exit $?  $(date +%T)"
sacct -j "$SLURM_JOB_ID" -o JobID,MaxRSS,Elapsed -P 2>/dev/null | tail -3
