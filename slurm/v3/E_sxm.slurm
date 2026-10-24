#!/bin/bash
#SBATCH --job-name=cqsa_v3_E_sxm
#SBATCH --account=akey
#SBATCH --qos=gpu-test
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=80G
#SBATCH --gres=gpu:a100:1 --constraint="sxm&gpu80"
#SBATCH --time=00:59:00
#SBATCH --output=/scratch/gpfs/AKEY/yb2807/Stream-CQSA-v3/results/v3/logs/E_sxm_%j.out
# The ONE gpu-test job of the v3 experiments E0-E6 (experiments/v3/PLAN.md) on one A100-SXM4-80GB, RESUBMITTABLE:
# every part runs `bench.py run ... --only-missing --budget-s <budget>` (slurm/v3/parts.sh), which skips the rows
# already in results/v3/results.jsonl and defers the rows its budget cannot hold, so the same command is submitted
# again until the job's last line says "MISSING 0". The parts run in PLAN.md's order (CQSA_PARTS; each gets at most
# CQSA_PART_BUDGET_S = 28 min of the job, so every submission advances several experiments; a part whose rows are
# all recorded returns in seconds and hands its time to the next). Estimated total (bench.py plan): E0 ~70 min,
# E1 ~180, E3 ~135, E4 ~65, E6 ~120, E5 ~40, E2 (SXM parts) ~150 min: ~12 h of SXM time, i.e. ~13 submissions.
#
# --mem: the host memory of the largest host-resident row of these parts, E2:bwd at N=4M (acc=cpu, host-resident
# Q/K/V, fwd+bwd on the classic streamed backward): pinned Q/K/V 3 x 4M x 8 x 64 x 2 B = 12 GiB + pinned dO 4 GiB
# + the fp32 host accumulator (output) 8 GiB + its fp16 copy 4 GiB + the backward's fp32 host gradients 3 x 8 = 24 GiB
# + their fp16 casts 12 GiB + pinned task staging ~4 GiB + interpreter / CUDA context ~3 GiB = ~71 GiB -> 80G.
# Without E2:bwd the largest is E2:fwd at 4M (12 + 4 + 8 + 4 + 3 = ~31 GiB -> 40G) and the device-resident parts
# need < 16 GiB (task lists, the fp64 spot reference, the FA-2 reference output on the host at 1M): submit those with
#   sbatch --mem=40G --export=ALL,...,CQSA_PARTS="E0 E1:fwd E3:fwd E4 E1:bwd E6 E3:bwd E5 E2:fwd" slurm/v3/E_sxm.slurm
#
# Everything is frozen: code = CQSA_SRC (slurm/ultimate/snapshot.sh), modules = the frozen directories.
#   SNAP=$(bash slurm/ultimate/snapshot.sh) && sbatch --export=ALL,CQSA_SRC=$SNAP,CQSA_V13_SO=<dir>,CQSA_NATIVE_V2_SO=<dir> slurm/v3/E_sxm.slurm
# Knobs: CQSA_PARTS, CQSA_PART_BUDGET_S (default 1700), E_ARGS (extra bench.py arguments: "--retry-errors", "--reps 1", ...).
set -u
export CQSA_SRC=${CQSA_SRC:?CQSA_SRC (frozen code root) is required}
export CQSA_V13_SO=${CQSA_V13_SO:?CQSA_V13_SO (frozen cqsa_cuda_v13 directory) is required}
export CQSA_NATIVE_V2_SO=${CQSA_NATIVE_V2_SO:?CQSA_NATIVE_V2_SO (frozen cqsa_native_v2 directory) is required}
export CQSA_PARTS=${CQSA_PARTS:-"E0 E1:fwd E3:fwd E4 E1:bwd E6 E3:bwd E5 E2:fwd E2:bwd"}
export CQSA_PART_BUDGET_S=${CQSA_PART_BUDGET_S:-1700}
export CQSA_JOB_TOTAL_S=$((59 * 60))
export CQSA_V3_RESULTS=${CQSA_V3_RESULTS:-/scratch/gpfs/AKEY/yb2807/Stream-CQSA-v3/results/v3}
bash "$CQSA_SRC/slurm/v3/parts.sh"
# Self-resubmission: when rows are still missing, queue the next copy of this job from the live repo's script
# (same frozen code and modules), up to CQSA_MAX_CHAIN hops. A watcher in an interactive session dies with that
# session; the chain does not. Stop it with: scancel <the queued job>, or CQSA_MAX_CHAIN=0.
CHAIN=${CQSA_CHAIN:-0}; MAXCHAIN=${CQSA_MAX_CHAIN:-20}
MISSING=$(grep -o "MISSING [0-9]*" "$CQSA_V3_RESULTS/logs/E_sxm_${SLURM_JOB_ID}.out" 2>/dev/null | tail -1 | awk '{print $2}')
# only this job's own parts count: the plan-wide MISSING also holds the PCIe parts, which an SXM hop can never run
# (hop 3 of the E2:bwd chain on 2026-10-23 was queued for them). A part reports "N deferred by the budget" for the
# rows it skipped and, with --retry-errors, the rows that errored are re-run on the next hop.
DEFERRED=$(grep -o "[0-9]* deferred by the budget" "$CQSA_V3_RESULTS/logs/E_sxm_${SLURM_JOB_ID}.out" 2>/dev/null | awk '{s+=$1} END {print s+0}')
TIMEDOUT=$(grep -c "exit 124 " "$CQSA_V3_RESULTS/logs/E_sxm_${SLURM_JOB_ID}.out" 2>/dev/null); TIMEDOUT=${TIMEDOUT:-0}   # grep -c prints 0 itself (an || echo 0 doubled it: two no-op hops on 2026-10-24)
if [ -n "$MISSING" ] && [ "$MISSING" != "0" ] && { [ "$DEFERRED" != "0" ] || [ "$TIMEDOUT" != "0" ]; } && [ "$CHAIN" -lt "$MAXCHAIN" ]; then
  # carry this job's own --mem into the successor (a flag is not part of the exported environment: four 80G hops
  # after a 120G one were OOM-killed on the host-resident 4M backward rows on 2026-10-23)
  MEM=$(scontrol show job "$SLURM_JOB_ID" | grep -o "MinMemoryNode=[0-9]*[GM]" | head -1 | cut -d= -f2)
  NEXT=$(cd /scratch/gpfs/AKEY/yb2807/Stream-CQSA-v3 && sbatch --parsable ${MEM:+--mem=$MEM} --export=ALL,CQSA_CHAIN=$((CHAIN + 1)) slurm/v3/E_sxm.slurm)
  echo "== $MISSING rows missing ($DEFERRED deferred by this job's parts): resubmitted as job $NEXT (chain hop $((CHAIN + 1)) of $MAXCHAIN)"
else
  echo "== chain ends: missing=${MISSING:-?} deferred=${DEFERRED:-?} hop=$CHAIN"
fi
