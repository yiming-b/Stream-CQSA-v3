#!/bin/bash
#SBATCH --job-name=cqsa_v3_notebook
#SBATCH --account=akey
#SBATCH --qos=gpu-test
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=20G
#SBATCH --gres=gpu:a100:1 --constraint="sxm&gpu80"
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/gpfs/AKEY/yb2807/Stream-CQSA-v3/results/v3/logs/notebook_%j.out
# Execute the v3 tutorial notebook on one A100-SXM4-80GB with the frozen modules (the classic v13 module and the
# wave module), writing notebooks/stream_cqsa_v3_tutorial.ipynb in the live repo (outputs included).
#   sbatch slurm/v3/notebook.slurm
# --mem: 3 GiB of host Q/K/V at 1M, their pinned staging, the fp16 references and the interpreter: < 12 GiB.
set -uo pipefail
OUT=/scratch/gpfs/AKEY/yb2807/Stream-CQSA-v3
cd "$OUT"
source /scratch/gpfs/AKEY/yb2807/Stream-CQSA-dev/next/env_next.sh
V13=${V13_DIR:-/scratch/gpfs/AKEY/yb2807/Stream-CQSA-dev/next/kernel/v13/frozen/c2}
NV2=${NATIVE_V2_DIR:-/scratch/gpfs/AKEY/yb2807/Stream-CQSA-dev/next/native_v2/frozen/gqa1}
export PYTHONPATH=$OUT:${PYTHONPATH:-}:$V13:$NV2
export CQSA_CUDA_MODULE=cqsa_cuda_v13 CQSA_CPU_THREADS=8 OMP_NUM_THREADS=8
unset CQSA_CUDA_MODULE_NONCAUSAL CQSA_BACKWARD CQSA_FORWARD CQSA_VERBOSE
hostname; nvidia-smi --query-gpu=name --format=csv,noheader
python -P -c "import torch, cqsa_cuda_v13 as c, cqsa_native_v2 as n; print(c.__file__); print(n.__file__, n.WAVE_API)"
python -P notebooks/make_v3_tutorial.py --out "$OUT/notebooks/stream_cqsa_v3_tutorial.ipynb" 2>&1 | grep -v IPKernelApp
echo "exit ${PIPESTATUS[0]}  $(date +%T)"
