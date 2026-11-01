#!/bin/bash
#SBATCH --job-name=cqsa_ult_suite
#SBATCH --account=akey
#SBATCH --qos=gpu-test
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=48G
#SBATCH --gres=gpu:a100:1 --constraint="sxm&gpu80"
#SBATCH --time=00:59:00
#SBATCH --output=/scratch/gpfs/AKEY/yb2807/Stream-CQSA-v3/results/ultimate/test_suite_%j.out
# The full pytest suite of Stream-CQSA-v3, run from a frozen snapshot, on the v13 module (U1):
#   SNAP=$(bash slurm/ultimate/snapshot.sh) && sbatch --export=ALL,CQSA_SRC=$SNAP,V13_DIR=<frozen dir> slurm/ultimate/test_suite.slurm
# The suite holds several host tensors and forks workers: 24G was OOM-killed (job 14231448); 48G is what
# the earlier verify jobs used. CQSA_CUDA_MODULE=cqsa_cuda_v13 (the `common` build: fp16 + bf16, one module for
# both causal modes, CUDA backward by default); the 12 bf16 cases that failed on the fp16-only dev v11 module
# (job 14327651: test_backward_default / test_bf16 / test_bwd_host_stream) must pass here. The v11x / v12f /
# v12b tests still load their modules from SUITE_KERNELS (a frozen copy, LAST on PYTHONPATH with V13_DIR, so
# the production v11 / v9 modules of tests that pin them come from next/kernel/stable).
set -uo pipefail
OUT=/scratch/gpfs/AKEY/yb2807/Stream-CQSA-v3
ROOT=${CQSA_SRC:-$OUT}          # frozen snapshot (a `git archive` of one commit); the log goes to the worktree
cd "$ROOT"
echo "code: $ROOT"
source /scratch/gpfs/AKEY/yb2807/Stream-CQSA-dev/next/env_next.sh
SUITE_KERNELS=${SUITE_KERNELS:-/scratch/gpfs/AKEY/yb2807/Stream-CQSA-dev/next/kernel/frozen_suite/2026-10-21}
V13_DIR=${V13_DIR:-$OUT/kernels/v13}
# the wave extension cqsa_native_v2 (U2): a frozen copy (next/native_v2/frozen/<build>) or the build tree
NATIVE_V2_DIR=${NATIVE_V2_DIR:-$OUT/kernels/native_v2}
export PYTHONPATH=$ROOT:/scratch/gpfs/AKEY/yb2807/Stream-CQSA-dev/next/kernel/stable:${PYTHONPATH:-}:/scratch/gpfs/AKEY/yb2807/Stream-CQSA-dev/next/native:$SUITE_KERNELS:$V13_DIR:$NATIVE_V2_DIR
export CQSA_CUDA_MODULE=cqsa_cuda_v13 CQSA_CPU_THREADS=4 OMP_NUM_THREADS=4
unset CQSA_CUDA_MODULE_NONCAUSAL CQSA_BACKWARD CQSA_FORWARD
hostname; which python
nvidia-smi --query-gpu=name --format=csv,noheader
python -P -c "import stream_cqsa, torch, stream_cqsa.interface as I; print(stream_cqsa.__file__, torch.__version__); print(stream_cqsa.kernels_available(), I.CQSA_CUDA_MODULE, I.CQSA_CUDA_MODULE_NONCAUSAL, I.backward_kernel())"
md5sum "$SUITE_KERNELS"/*.so "$V13_DIR"/cqsa_cuda_v13*.so "$NATIVE_V2_DIR"/cqsa_native_v2*.so
python -P -c "import torch, cqsa_native_v2 as m; print(m.__file__, 'WAVE_API', m.WAVE_API)"
python -P -c "import torch, cqsa_cuda_next_v11 as a, cqsa_cuda_next_v9 as b, cqsa_cuda_next_v11x as c, cqsa_cuda_next_v12f as d, cqsa_cuda_next_v12b as e, cqsa_cuda_v13 as f; print(*(m.__file__ for m in (a, b, c, d, e, f)), sep='\n')"
timeout -s INT 3300 python -P -X faulthandler -m pytest tests -q -rfs --tb=line -p no:cacheprovider -o faulthandler_timeout=600 2>&1 \
    | grep -v "^$\|site-packages\|frozen runpy" | tail -200 | cut -c1-300
echo "exit ${PIPESTATUS[0]}  $(date +%T)"
sacct -j "$SLURM_JOB_ID" -o JobID,MaxRSS,Elapsed -P 2>/dev/null | tail -3
