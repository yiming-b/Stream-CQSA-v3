#!/bin/bash
#SBATCH --job-name=cqsa_ult_suite
#SBATCH --account=akey
#SBATCH --qos=gpu-test
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=48G
#SBATCH --gres=gpu:a100:1 --constraint="sxm&gpu80"
#SBATCH --time=00:59:00
#SBATCH --output=/scratch/gpfs/AKEY/yb2807/Stream-CQSA-ultimate/results/ultimate/test_suite_%j.out
# The full pytest suite of branch `ultimate`, run from a frozen snapshot:
#   SNAP=$(bash slurm/ultimate/snapshot.sh) && sbatch --export=ALL,CQSA_SRC=$SNAP slurm/ultimate/test_suite.slurm
# The suite holds several host tensors and forks workers: 24G was OOM-killed (job 14231448); 48G is what
# the earlier verify jobs used. The dev v11 module is fp16-only, so the bf16 cases are expected to fail
# (12 in job 14327651: test_backward_default / test_bf16 / test_bwd_host_stream, dtype1 = bf16).
# Kernel variants: the v11x / v12f / v12b tests skip unless their modules import; SUITE_KERNELS is a frozen
# copy of the three .so (next/kernel/frozen_suite/<date>, README there), placed LAST on PYTHONPATH so the
# production v11 / v9 modules come from next/kernel/stable. (Job 14327651 lacked it: 101 of those skipped.)
set -uo pipefail
OUT=/scratch/gpfs/AKEY/yb2807/Stream-CQSA-ultimate
ROOT=${CQSA_SRC:-$OUT}          # frozen snapshot (a `git archive` of one commit); the log goes to the worktree
cd "$ROOT"
echo "code: $ROOT"
source /scratch/gpfs/AKEY/yb2807/Stream-CQSA-dev/next/env_next.sh
SUITE_KERNELS=${SUITE_KERNELS:-/scratch/gpfs/AKEY/yb2807/Stream-CQSA-dev/next/kernel/frozen_suite/2026-09-23}
export PYTHONPATH=$ROOT:/scratch/gpfs/AKEY/yb2807/Stream-CQSA-dev/next/kernel/stable:${PYTHONPATH:-}:/scratch/gpfs/AKEY/yb2807/Stream-CQSA-dev/next/native:$SUITE_KERNELS
export CQSA_CUDA_MODULE=cqsa_cuda_next_v11 CQSA_CUDA_MODULE_NONCAUSAL=cqsa_cuda_next_v9 CQSA_CPU_THREADS=4 OMP_NUM_THREADS=4
hostname; which python
nvidia-smi --query-gpu=name --format=csv,noheader
python -P -c "import stream_cqsa, torch; print(stream_cqsa.__file__, torch.__version__); print(stream_cqsa.kernels_available())"
md5sum "$SUITE_KERNELS"/*.so
python -P -c "import torch, cqsa_cuda_next_v11 as a, cqsa_cuda_next_v9 as b, cqsa_cuda_next_v11x as c, cqsa_cuda_next_v12f as d, cqsa_cuda_next_v12b as e; print(*(m.__file__ for m in (a, b, c, d, e)), sep='\n')"
timeout -s INT 3300 python -P -X faulthandler -m pytest tests -q -rfs --tb=line -p no:cacheprovider -o faulthandler_timeout=600 2>&1 \
    | grep -v "^$\|site-packages\|frozen runpy" | tail -200 | cut -c1-300
echo "exit ${PIPESTATUS[0]}  $(date +%T)"
sacct -j "$SLURM_JOB_ID" -o JobID,MaxRSS,Elapsed -P 2>/dev/null | tail -3
