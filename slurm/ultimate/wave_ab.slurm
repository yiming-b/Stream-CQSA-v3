#!/bin/bash
#SBATCH --job-name=cqsa_wave_ab
#SBATCH --account=akey
#SBATCH --qos=gpu-test
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=20G
#SBATCH --gres=gpu:a100:1 --constraint="sxm&gpu80"
#SBATCH --time=00:55:00
#SBATCH --output=/scratch/gpfs/AKEY/yb2807/Stream-CQSA-v3/results/ultimate/wave_ab_%j.out
# U2 wave A/B: cqsa_native_v2 (census + ALIGNED wave kernels, uniform layout) vs cqsa_native vs the per-task v12f /
# v12b loop vs monolithic flash_attn / SDPA-flash, one process on one A100-SXM4-80GB. Every tensor lives on the
# device (N=1M x 8 x 64 fp16 Q/K/V/dO, the per-task copies of 7 tasks of L~450K, the wave's packed copies and
# fp32 partials, the fp32 dq_accum: ~25 GiB peak); the host holds the task tables and the interpreter (the 1M
# task lists and wave tables: a few hundred MB; 20G leaves headroom for the pinned backward gathers).
#   SNAP=$(bash slurm/ultimate/snapshot.sh) && sbatch --export=ALL,CQSA_SRC=$SNAP,NATIVE_V2_DIR=<frozen dir> slurm/ultimate/wave_ab.slurm
set -uo pipefail
# Code comes from a frozen snapshot (CQSA_SRC, a `git archive` of one commit) so edits to the live worktree while
# this job waits in the queue cannot change what it runs; results go to the worktree. The kernel modules are the
# built trees under next/kernel (not in git) and NATIVE_V2_DIR, a frozen copy of one cqsa_native_v2 build (the
# .so alone) so a rebuild of kernels/native_v2 while this job waits cannot change what it loads.
OUT=/scratch/gpfs/AKEY/yb2807/Stream-CQSA-v3
ROOT=${CQSA_SRC:-$OUT}
KER=/scratch/gpfs/AKEY/yb2807/Stream-CQSA-dev/next/kernel
cd "$ROOT"
echo "code: $ROOT"
source /scratch/gpfs/AKEY/yb2807/Stream-CQSA-dev/next/env_next.sh
# the variant dirs carry a stub stream_cqsa/__init__.py for their setup.py, so they go LAST
V12F_DIR=${V12F_DIR:-$KER/v12f/frozen/b3}
V12B_DIR=${V12B_DIR:-$KER/v12b/frozen/b5}
NATIVE_V2_DIR=${NATIVE_V2_DIR:-$OUT/kernels/native_v2}
export PYTHONPATH=$ROOT:$KER/stable:${PYTHONPATH:-}:/scratch/gpfs/AKEY/yb2807/Stream-CQSA-dev/next/native:$V12F_DIR:$V12B_DIR:$NATIVE_V2_DIR
export CQSA_CUDA_MODULE=cqsa_cuda_next_v11 CQSA_CUDA_MODULE_NONCAUSAL=cqsa_cuda_next_v9 CQSA_CPU_THREADS=4 OMP_NUM_THREADS=4
which python; hostname
nvidia-smi --query-gpu=name,clocks.sm,clocks.max.sm,memory.total --format=csv,noheader
python -P -c "import stream_cqsa, torch, flash_attn, cqsa_native_v2 as a, cqsa_native as b, cqsa_cuda_next_v12f as c, cqsa_cuda_next_v12b as d; print(stream_cqsa.__file__, torch.__version__, flash_attn.__version__); print(a.__file__); print(b.__file__); print(c.__file__); print(d.__file__)"
ls -la --time-style=full-iso $NATIVE_V2_DIR/cqsa_native_v2*.so; md5sum $NATIVE_V2_DIR/cqsa_native_v2*.so
# WAVE_AB_NAME names the result files (wave_ab_b2.json/.md ...) so a later build does not overwrite earlier rows.
python -P -u "$ROOT/benchmarks/ultimate/wave_ab.py" --out "$OUT/results/ultimate/${WAVE_AB_NAME:-wave_ab}.json"
echo "exit $?  $(date +%T)"
sacct -j "$SLURM_JOB_ID" -o JobID,MaxRSS,Elapsed -P 2>/dev/null | tail -3
