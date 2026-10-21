#!/bin/bash
#SBATCH --job-name=cqsa_engine_ab
#SBATCH --account=akey
#SBATCH --qos=gpu-test
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=24G
#SBATCH --gres=gpu:a100:1 --constraint="sxm&gpu80"
#SBATCH --time=00:59:00
#SBATCH --output=/scratch/gpfs/AKEY/yb2807/Stream-CQSA-v3/results/ultimate/engine_ab_%j.out
# U1 engine-level A/B (benchmarks/ultimate/engine_ab.py): v2.2.3 (classic engine, v11 CUDA / Triton backward) vs
# v3 + v13 (kernel="cuda" and "auto") vs monolithic flash_attn / SDPA-flash, forward and backward, device-resident,
# acc=GPU, at 131K (c=7, 73), 1M (c=7, itr 1 and 2) and a bf16 row at 131K; one A100-SXM4-80GB.
# --mem: everything lives on the device (Q/K/V/dout 4 GiB at 1M + the engine's gathered copies and fp32
# accumulators + the flash_attn reference's output / gradients); the host holds the pinned task lists (a few
# hundred MiB at 1M), the interpreter and the driver (engine_align_ab's 1M job ran within 24G).
#   SNAP=$(bash slurm/ultimate/snapshot.sh) && sbatch --export=ALL,CQSA_SRC=$SNAP,V13_DIR=<frozen dir> slurm/ultimate/engine_ab.slurm
set -uo pipefail
OUT=/scratch/gpfs/AKEY/yb2807/Stream-CQSA-v3
ROOT=${CQSA_SRC:-$OUT}            # frozen snapshot of one v3 commit; results go to the live repo
V2=/scratch/gpfs/AKEY/yb2807/Stream-CQSA-v2
V223_DIR=${V223_DIR:-/scratch/gpfs/AKEY/yb2807/Stream-CQSA-snapshots/v2-v2.2.3}
# V13_DIR: a frozen copy of one v13 build (next/kernel/v13/frozen/<tag>, the .so alone), so a rebuild of
# kernels/v13 while this job waits cannot change what it loads; default = the build tree itself.
V13_DIR=${V13_DIR:-$OUT/kernels/v13}
if [ ! -f "$V223_DIR/.complete" ]; then
  mkdir -p "$V223_DIR" && git -C "$V2" archive v2.2.3 | tar -x -C "$V223_DIR" && touch "$V223_DIR/.complete"
fi
cd "$ROOT"
echo "code: $ROOT   v2.2.3: $V223_DIR   v13: $V13_DIR"
source /scratch/gpfs/AKEY/yb2807/Stream-CQSA-dev/next/env_next.sh
export CQSA_CPU_THREADS=8 OMP_NUM_THREADS=8
unset CQSA_CUDA_MODULE CQSA_CUDA_MODULE_NONCAUSAL CQSA_BACKWARD CQSA_FORWARD
which python; hostname
nvidia-smi --query-gpu=name,clocks.sm,clocks.max.sm,memory.total --format=csv,noheader
ls -la --time-style=full-iso "$V13_DIR"/cqsa_cuda_v13*.so; md5sum "$V13_DIR"/cqsa_cuda_v13*.so
PYTHONPATH=$ROOT:${PYTHONPATH:-}:$V13_DIR python -P -c "import stream_cqsa, torch, flash_attn, stream_cqsa.interface as I; print(stream_cqsa.__file__, stream_cqsa.__version__, torch.__version__, flash_attn.__version__, I.CQSA_CUDA_MODULE, I.cqsa_cuda.__file__, I.backward_kernel())"
PYTHONPATH=$V223_DIR:/scratch/gpfs/AKEY/yb2807/Stream-CQSA-dev/next/kernel/stable:${PYTHONPATH:-} CQSA_CUDA_MODULE=cqsa_cuda_next_v11 CQSA_CUDA_MODULE_NONCAUSAL=cqsa_cuda_next_v9 python -P -c "import stream_cqsa, stream_cqsa.interface as I; print(stream_cqsa.__file__, stream_cqsa.__version__, I.CQSA_CUDA_MODULE, I.cqsa_cuda.__file__)"
export CQSA_SRC=$ROOT V223_DIR V13_DIR
python -P -u "$ROOT/benchmarks/ultimate/engine_ab.py" run --out "$OUT/results/ultimate/${ENGINE_AB_NAME:-engine_ab}.json" --reps 3 --warmup 1 ${ENGINE_AB_ARGS:-}
echo "exit $?  $(date +%T)"
sacct -j "$SLURM_JOB_ID" -o JobID,MaxRSS,Elapsed -P 2>/dev/null | tail -3
