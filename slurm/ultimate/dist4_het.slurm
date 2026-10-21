#!/bin/bash
#SBATCH --job-name=cqsa_dist4_het
#SBATCH --account=akey
#SBATCH --time=00:50:00
#SBATCH --output=/scratch/gpfs/AKEY/yb2807/Stream-CQSA-v3/results/ultimate/dist_het_%j.out
#SBATCH --nodes=1 --ntasks-per-node=2 --cpus-per-task=4 --mem=40G --gres=gpu:a100:2 --constraint="sxm&gpu80"
#SBATCH hetjob
#SBATCH --nodes=1 --ntasks-per-node=2 --cpus-per-task=4 --mem=40G --gres=gpu:a100:2 --constraint="gpu80&pcie"
# D2 (written by D1, submitted by D2 from a snapshot): REAL heterogeneous devices, a Slurm heterogeneous job of
# 2 x A100-SXM4-80GB (component 0) + 2 x A100-PCIe-80GB (component 1), one process per GPU, NCCL across the two
# nodes -- the program's only mixed-GPU measurement. `sbatch --test-only` on 2026-09-23 placed it on della-l08g1
# (SXM) + della-l04g15 (PCIe) 30 min out; on 2026-10-21 12:54 on della-l03g12 + della-l09g4 3 h out.
# Arms: rr; lpt with the per-model seed rates (SXM4 3.35e7 vs PCIe-80 2.91e7 tiles/s, the D2 refit) and the
# cost-aware split; lpt with the per-rank rates measured at startup (host-streamed calibration less the refit's
# overheads), allreduce and reduce_scatter (sharded, and replicated = the driver's default, unmeasured so far);
# 1M and 2M at itr=1 and itr=2; the device-resident backward at 1M itr=1; single-GPU reference on rank 0 (an
# SXM GPU: speedups are against the single fastest GPU). Part 1 (itr=1 + bwd) ~12 min, part 2 (itr=2) ~8 min on
# the PCIe node (job 14330294); cross-node NCCL merges are slower.
# Host memory per node: 2 ranks x (6 GiB Q/K/V at 2M + <= 4.5 GiB pinned staging + 1.5 GiB context) ~ 24 GiB -> 40G.
#   SNAP=$(bash slurm/ultimate/snapshot.sh) && sbatch --export=ALL,CQSA_SRC=$SNAP slurm/ultimate/dist4_het.slurm
set -uo pipefail
OUT=/scratch/gpfs/AKEY/yb2807/Stream-CQSA-v3
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
    lpt/reduce_scatter/measured lpt/reduce_scatter/measured/rep --bwd --bwd-max-n 1048576 \
    --tag het_sxm2_pcie2 --out "$OUT/results/ultimate/dist_het.json"
echo "part 1 exit $?  $(date +%T)"
run 1048576 2097152 --itr 2 --layouts prod --arms rr/allreduce/seed lpt/allreduce/seed lpt/allreduce/measured \
    lpt/reduce_scatter/measured --tag het_sxm2_pcie2_itr2 --out "$OUT/results/ultimate/dist_het_itr2.json"
echo "part 2 exit $?  $(date +%T)"
sacct -j "$SLURM_JOB_ID" -o JobID,MaxRSS,Elapsed -P 2>/dev/null | tail -6
