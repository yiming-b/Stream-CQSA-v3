#!/bin/bash
#SBATCH --job-name=cqsa_v3_combined
#SBATCH --account=akey
#SBATCH --qos=gpu-test
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=24G
#SBATCH --gres=gpu:a100:1 --constraint="sxm&gpu80"
#SBATCH --time=00:59:00
#SBATCH --output=/scratch/gpfs/AKEY/yb2807/Stream-CQSA-v3/results/ultimate/v3_combined_%j.out
# The ONE GPU job of the v3 kernel step (the user's rule: queues are long, so one large job instead of
# several short ones). A100-SXM4 nodes are reachable only through gpu-test (59 min cap; a 2.5 h gpu-short
# PCIe job was 13 days out on 2026-10-21), so the job is 59 minutes on one SXM GPU, the test suite runs on
# vis1 (correctness only) and the two timing parts are trimmed to ~25 min each (ENGINE_AB_ARGS, WAVE_AB_ITERS);
# each part has a hard timeout so an overrun cannot starve the other. A failing part does not stop the next.
#   slurm/ultimate/parts/u1.sh  engine-level A/B: v2.2.3 vs v3 + v13 vs flash_attn / SDPA-flash   ~25 min
#   slurm/ultimate/parts/u2.sh  wave A/B: cqsa_native_v2 vs the first wave module vs v12f/v12b     ~25 min
# --mem is the maximum over the parts (24G / 20G); the suite's 48G is not needed here.
# Everything is frozen: code = CQSA_SRC (slurm/ultimate/snapshot.sh), modules = the frozen directories.
#   SNAP=$(bash slurm/ultimate/snapshot.sh) && sbatch --export=ALL,CQSA_SRC=$SNAP,CQSA_V13_SO=<dir>,CQSA_NATIVE_V2_SO=<dir> slurm/ultimate/v3_combined.slurm
set -u
export CQSA_SRC=${CQSA_SRC:?CQSA_SRC (frozen code root) is required}
export CQSA_V13_SO=${CQSA_V13_SO:?CQSA_V13_SO (frozen cqsa_cuda_v13 directory) is required}
export CQSA_NATIVE_V2_SO=${CQSA_NATIVE_V2_SO:?CQSA_NATIVE_V2_SO (frozen cqsa_native_v2 directory) is required}
export CQSA_SRC_RESULTS=${CQSA_SRC_RESULTS:-/scratch/gpfs/AKEY/yb2807/Stream-CQSA-v3/results/ultimate}
mkdir -p "$CQSA_SRC_RESULTS"
echo "== combined v3 job $SLURM_JOB_ID on $(hostname)  $(date)"
nvidia-smi --query-gpu=name,clocks.sm,clocks.max.sm,memory.total --format=csv,noheader
echo "code $CQSA_SRC | v13 $CQSA_V13_SO | native_v2 $CQSA_NATIVE_V2_SO | results $CQSA_SRC_RESULTS"
export ENGINE_AB_ARGS=${ENGINE_AB_ARGS:-"--reps 2 --warmup 1"}     # medians of 2 (min was within 1% of the median in every earlier job)
export WAVE_AB_ITERS=${WAVE_AB_ITERS:-3}
export U1_RUN_SUITE=${U1_RUN_SUITE:-0}                              # the suite runs on vis1
for part in ${CQSA_PARTS:-u1 u2}; do
  log=$CQSA_SRC_RESULTS/v3_combined_${SLURM_JOB_ID}_$part.log
  echo "== part $part starts $(date +%T) -> $log"
  timeout -s INT ${CQSA_PART_TIMEOUT:-27m} bash "$CQSA_SRC/slurm/ultimate/parts/$part.sh" > "$log" 2>&1
  echo "== part $part exit $? (124 = timeout)  $(date +%T)"
  grep -iE "passed|failed|error|Traceback|target|verdict" "$log" | tail -8
done
sacct -j "$SLURM_JOB_ID" -o JobID,MaxRSS,Elapsed -P 2>/dev/null | tail -3
