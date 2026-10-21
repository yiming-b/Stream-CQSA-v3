#!/bin/bash
#SBATCH --job-name=cqsa_dist4_sched
#SBATCH --account=akey
#SBATCH --qos=gpu-short
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=75G
#SBATCH --gres=gpu:a100:4 --constraint="gpu80&pcie"
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/gpfs/AKEY/yb2807/Stream-CQSA-v3/results/ultimate/dist_sched_%j.out
# D1: the multi-device scheduler on 4 x A100-PCIe-80GB (one node, one node type).
#  part 1 (homogeneous): N in {1M, 2M} x itr in {1, 2}, production layouts (itr=1 aligned + shared chunk
#    pool, itr=2 aligned gather), arms {rr, lpt} x {allreduce (replicated), reduce_scatter (sharded)},
#    single-GPU reference on rank 0 in the same job; backward rows at 1M itr=1 (device-resident).
#  part 2 (simulated heterogeneity): rank 1 confined to 54 of its 108 SMs by a CUDA green context (a
#    sleep-free rate cap; rank 1 because round-robin gives it 2 of the 7 itr=1 tasks -- rank 3 gets one,
#    which a half-speed rank absorbs by accident); arms rr, lpt with the seed rates (= equal ranks),
#    lpt / lpt+reduce_scatter with the per-rank rates measured at startup (host-streamed calibration);
#    1M, 2M at itr=1 and 1M at itr=2.
# Host memory: each rank holds fp16 Q/K/V token-major (3 KiB/token: 6 GiB at 2M) plus pinned staging
# (<= 3 GiB/slot at itr=1, one slot, x1.5 rounding) and ~1.5 GiB interpreter/CUDA context; outputs, the
# reference and the backward's gradients stay on the devices. 4 ranks ~50 GiB at 2M -> 75G.
#   SNAP=$(bash slurm/ultimate/snapshot.sh) && sbatch --export=ALL,CQSA_SRC=$SNAP slurm/ultimate/dist4_sched.slurm
set -uo pipefail
OUT=/scratch/gpfs/AKEY/yb2807/Stream-CQSA-v3
ROOT=${CQSA_SRC:-$OUT}
KER=/scratch/gpfs/AKEY/yb2807/Stream-CQSA-dev/next/kernel
cd "$ROOT"
echo "code: $ROOT"
source /scratch/gpfs/AKEY/yb2807/Stream-CQSA-dev/next/env_next.sh
# the kernel modules: next/kernel/stable (the frozen v11/v9 copies of 2026-09-12), first on the path
export PYTHONPATH=$ROOT:$KER/stable:${PYTHONPATH:-}:/scratch/gpfs/AKEY/yb2807/Stream-CQSA-dev/next/native
export CQSA_CUDA_MODULE=cqsa_cuda_next_v11 CQSA_CUDA_MODULE_NONCAUSAL=cqsa_cuda_next_v9 CQSA_CPU_THREADS=4 OMP_NUM_THREADS=4
export PYTHONSAFEPATH=1          # -P for the rank processes torch.distributed.run spawns
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CQSA_CACHE_DIR=$(mktemp -d)   # no cached calibration from another node: seeds + in-job measurements only
hostname; which python; date
nvidia-smi --query-gpu=index,name,memory.total,clocks.max.sm --format=csv,noheader
nvidia-smi topo -m
md5sum $KER/stable/*.so
python -P -c "import stream_cqsa, torch, cqsa_cuda_next_v11 as m; print(stream_cqsa.__file__, torch.__version__, m.__file__)"
# The torchrun on PATH belongs to another env; launch the module of this one.
python -P -m torch.distributed.run --standalone --nproc_per_node=4 \
    "$ROOT/benchmarks/ultimate/dist_telemetry.py" 1048576 2097152 --itr 1 2 --layouts prod \
    --scheduler rr lpt --reduce allreduce reduce_scatter --rates seed --bwd --bwd-max-n 1048576 \
    --tag homog --out "$OUT/results/ultimate/dist_sched_homog.json"
echo "part 1 exit $?  $(date +%T)"
python -P -m torch.distributed.run --standalone --nproc_per_node=4 \
    "$ROOT/benchmarks/ultimate/dist_telemetry.py" 1048576 2097152 --itr 1 --layouts prod --throttle 1:54 \
    --arms rr/allreduce/seed lpt/allreduce/seed lpt/allreduce/measured lpt/reduce_scatter/measured \
    --tag hetero_sm54 --out "$OUT/results/ultimate/dist_sched_hetero.json"
echo "part 2a exit $?  $(date +%T)"
python -P -m torch.distributed.run --standalone --nproc_per_node=4 \
    "$ROOT/benchmarks/ultimate/dist_telemetry.py" 1048576 --itr 2 --layouts prod --throttle 1:54 \
    --arms rr/allreduce/seed lpt/allreduce/seed lpt/allreduce/measured \
    --tag hetero_sm54_itr2 --out "$OUT/results/ultimate/dist_sched_hetero_itr2.json"
echo "part 2b exit $?  $(date +%T)"
sacct -j "$SLURM_JOB_ID" -o JobID,MaxRSS,Elapsed -P 2>/dev/null | tail -3
