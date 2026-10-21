#!/bin/bash
#SBATCH --job-name=cqsa_dist4_tele
#SBATCH --account=akey
#SBATCH --qos=gpu-short
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=75G
#SBATCH --gres=gpu:a100:4 --constraint="gpu80&pcie"
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/gpfs/AKEY/yb2807/Stream-CQSA-v3/results/ultimate/dist_telemetry_%j.out
# M5: multi-device telemetry baseline (4 x A100-PCIe-80GB, one node), N in {1M, 2M}, itr in {1, 2},
# acc=gpu, chunk layouts seg_align 0 / 128 (+ aligned shared-chunk arm at itr=1), single-GPU
# reference on rank 0 in the same job. Host memory: each rank holds fp16 Q/K/V (3 KiB/token, 6 GiB at
# 2M, token-major so the engine makes no copy) plus its pinned staging slots (~3 GiB at itr=1) and the
# interpreter/CUDA context (~1.5 GiB); the fp32 output stays on the device (out_device="acc"). Four
# ranks ~42 GiB, x1.5 for the pinned allocator's power-of-two rounding -> 64 GiB; 75G.
# The PCIe node type is measured throughout; never compare these times with SXM rows.
set -uo pipefail
# Code comes from a frozen snapshot (CQSA_SRC, a `git archive` of one commit) so edits to the live
# worktree while this job waits in the queue cannot change what it runs; results go to the worktree.
OUT=/scratch/gpfs/AKEY/yb2807/Stream-CQSA-v3
ROOT=${CQSA_SRC:-$OUT}
cd "$ROOT"
echo "code: $ROOT"
source /scratch/gpfs/AKEY/yb2807/Stream-CQSA-dev/next/env_next.sh
export PYTHONPATH=$ROOT:/scratch/gpfs/AKEY/yb2807/Stream-CQSA-dev/next/kernel/stable:${PYTHONPATH:-}:/scratch/gpfs/AKEY/yb2807/Stream-CQSA-dev/next/native
export CQSA_CUDA_MODULE=cqsa_cuda_next_v11 CQSA_CUDA_MODULE_NONCAUSAL=cqsa_cuda_next_v9 CQSA_CPU_THREADS=4 OMP_NUM_THREADS=4
export PYTHONSAFEPATH=1          # -P for the rank processes torch.distributed.run spawns
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
hostname; which python
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
python -P -c "import stream_cqsa, torch; print(stream_cqsa.__file__, torch.__version__)"
# The torchrun on PATH belongs to another env; launch the module of this one.
python -P -m torch.distributed.run --standalone --nproc_per_node=4 \
    "$ROOT/benchmarks/ultimate/dist_telemetry.py" 1048576 2097152 --itr 1 2 --acc gpu --seg-align 0 128 \
    --out "$OUT/results/ultimate/dist_telemetry.json"
echo "exit $?  $(date +%T)"
sacct -j "$SLURM_JOB_ID" -o JobID,MaxRSS,Elapsed -P 2>/dev/null | tail -3
