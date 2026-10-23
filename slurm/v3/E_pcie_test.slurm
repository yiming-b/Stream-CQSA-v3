#!/bin/bash
#SBATCH --job-name=cqsa_v3_E_pcie_t
#SBATCH --account=akey
#SBATCH --qos=gpu-test
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=60G
#SBATCH --gres=gpu:a100:1 --constraint="gpu80&pcie"
#SBATCH --time=00:59:00
#SBATCH --output=/scratch/gpfs/AKEY/yb2807/Stream-CQSA-v3/results/v3/logs/E_pcie_t_%j.out
# The gpu-test variant of E_pcie.slurm: E2's host-resident 4M / 8M FORWARD rows (part E2:pcie_fwd) on the same
# A100-80GB-PCIe node type, in 59-minute jobs that chain themselves while rows are missing (the gpu-short job waited
# two days in the queue). Each 8M row is 17-28 min (bench.py plan), so a job runs 2-3 rows and the chain needs ~5 hops.
# The 8M fwd+bwd rows (E2:pcie_bwd, ~1.5-3 h each) do not fit an hour and stay with E_pcie.slurm on gpu-short.
# --mem 60G: the forward-only host-resident 8M row = pinned Q/K/V 24 GiB + fp32 host accumulator 16 + fp16 copy 8
# + staging ~6 + runtime ~3 = ~57 GiB (E_pcie.slurm's arithmetic).
#   SNAP=... && sbatch --export=ALL,CQSA_SRC=$SNAP,CQSA_V13_SO=<dir>,CQSA_NATIVE_V2_SO=<dir>,CQSA_CHAIN=0 slurm/v3/E_pcie_test.slurm
set -u
export CQSA_SRC=${CQSA_SRC:?CQSA_SRC (frozen code root) is required}
export CQSA_V13_SO=${CQSA_V13_SO:?CQSA_V13_SO (frozen cqsa_cuda_v13 directory) is required}
export CQSA_NATIVE_V2_SO=${CQSA_NATIVE_V2_SO:?CQSA_NATIVE_V2_SO (frozen cqsa_native_v2 directory) is required}
export CQSA_PARTS=${CQSA_PARTS:-"E2:pcie_fwd"}
export CQSA_PART_BUDGET_S=${CQSA_PART_BUDGET_S:-3300}
export CQSA_JOB_TOTAL_S=$((59 * 60))
export CQSA_V3_RESULTS=${CQSA_V3_RESULTS:-/scratch/gpfs/AKEY/yb2807/Stream-CQSA-v3/results/v3}
bash "$CQSA_SRC/slurm/v3/parts.sh"
# Self-resubmission while the part's rows are missing (see E_sxm.slurm); the MISSING line counts every
# experiment, so this chain stops on its own part's count: rows of E2:pcie_fwd still absent.
CHAIN=${CQSA_CHAIN:-0}; MAXCHAIN=${CQSA_MAX_CHAIN:-8}
LEFT=$(grep -E "^\| E2 \| pcie_fwd \|" "$CQSA_V3_RESULTS/STATUS.md" 2>/dev/null | awk -F'|' '{gsub(/ /,"",$8); print $8}')
if [ -n "$LEFT" ] && [ "$LEFT" != "0" ] && [ "$CHAIN" -lt "$MAXCHAIN" ]; then
  NEXT=$(cd /scratch/gpfs/AKEY/yb2807/Stream-CQSA-v3 && sbatch --parsable --export=ALL,CQSA_CHAIN=$((CHAIN + 1)) slurm/v3/E_pcie_test.slurm)
  echo "== E2:pcie_fwd: $LEFT rows missing: resubmitted as job $NEXT (chain hop $((CHAIN + 1)) of $MAXCHAIN)"
else
  echo "== chain ends: E2:pcie_fwd missing=${LEFT:-?} hop=$CHAIN"
fi
