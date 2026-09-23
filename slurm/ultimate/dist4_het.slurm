#!/bin/bash
#SBATCH --job-name=cqsa_dist4_het
#SBATCH --account=akey
#SBATCH --time=00:40:00
#SBATCH --output=/scratch/gpfs/AKEY/yb2807/Stream-CQSA-ultimate/results/ultimate/dist_het_%j.out
#SBATCH --nodes=1 --ntasks-per-node=2 --cpus-per-task=4 --mem=40G --gres=gpu:a100:2 --constraint="sxm&gpu80"
#SBATCH hetjob
#SBATCH --nodes=1 --ntasks-per-node=2 --cpus-per-task=4 --mem=40G --gres=gpu:a100:2 --constraint="gpu80&pcie"
# D1, REAL heterogeneous devices (NOT submitted by the D1 package, whose GPU budget was one 4-GPU job): a Slurm
# heterogeneous job, 2 x A100-SXM4-80GB (component 0) + 2 x A100-PCIe-80GB (component 1), one process per GPU,
# NCCL across the two nodes. `sbatch --test-only` on 2026-09-23 16:55 placed it on della-l08g1 (SXM) +
# della-l04g15 (PCIe) with an estimated start 30 min out, so a mixed-GPU job IS possible on this account.
# Arms: rr, lpt with the per-model seed rates (SXM 3.35e7 vs PCIe-80 2.9e7 tiles/s), lpt with the per-rank
# rates measured at startup; 1M and 2M at itr=1, 1M at itr=2; single-GPU reference on rank 0 (an SXM GPU).
# Host memory per node: 2 ranks x (6 GiB Q/K/V at 2M + <= 4.5 GiB pinned staging + 1.5 GiB context) ~ 24 GiB -> 40G.
#   SNAP=$(bash slurm/ultimate/snapshot.sh) && sbatch --export=ALL,CQSA_SRC=$SNAP slurm/ultimate/dist4_het.slurm
set -uo pipefail
OUT=/scratch/gpfs/AKEY/yb2807/Stream-CQSA-ultimate
ROOT=${CQSA_SRC:-$OUT}
KER=/scratch/gpfs/AKEY/yb2807/Stream-CQSA-dev/next/kernel
cd "$ROOT"
echo "code: $ROOT"
source /scratch/gpfs/AKEY/yb2807/Stream-CQSA-dev/next/env_next.sh
export PYTHONPATH=$ROOT:$KER/stable:${PYTHONPATH:-}:/scratch/gpfs/AKEY/yb2807/Stream-CQSA-dev/next/native
export CQSA_CUDA_MODULE=cqsa_cuda_next_v11 CQSA_CUDA_MODULE_NONCAUSAL=cqsa_cuda_next_v9 CQSA_CPU_THREADS=4 OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CQSA_CACHE_DIR=$(mktemp -d)
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST_HET_GROUP_0" | head -1) MASTER_PORT=29517
echo "master $MASTER_ADDR; nodes $SLURM_JOB_NODELIST_HET_GROUP_0 + $SLURM_JOB_NODELIST_HET_GROUP_1"; date
md5sum $KER/stable/*.so
run() {   # one step over both components: global ranks 0-1 on the SXM node, 2-3 on the PCIe node
  srun --het-group=0,1 --kill-on-bad-exit=1 bash -c '
    export RANK=$SLURM_PROCID WORLD_SIZE=4 LOCAL_RANK=$SLURM_LOCALID
    nvidia-smi --query-gpu=index,name --format=csv,noheader | sed "s/^/[$(hostname) rank $RANK] /"
    exec python -P -u '"$ROOT"'/benchmarks/ultimate/dist_telemetry.py "$@"' _ "$@"
}
run 1048576 2097152 --itr 1 --layouts prod --arms rr/allreduce/seed lpt/allreduce/seed lpt/allreduce/measured \
    lpt/reduce_scatter/measured --tag het_sxm2_pcie2 --out "$OUT/results/ultimate/dist_het.json"
echo "part 1 exit $?  $(date +%T)"
run 1048576 --itr 2 --layouts prod --arms rr/allreduce/seed lpt/allreduce/seed lpt/allreduce/measured \
    --tag het_sxm2_pcie2_itr2 --out "$OUT/results/ultimate/dist_het_itr2.json"
echo "part 2 exit $?  $(date +%T)"
sacct -j "$SLURM_JOB_ID" -o JobID,MaxRSS,Elapsed -P 2>/dev/null | tail -6
