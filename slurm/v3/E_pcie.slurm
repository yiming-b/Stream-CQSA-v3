#!/bin/bash
#SBATCH --job-name=cqsa_v3_E_pcie
#SBATCH --account=akey
#SBATCH --qos=gpu-short
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=150G
#SBATCH --gres=gpu:a100:1 --constraint="gpu80&pcie"
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/gpfs/AKEY/yb2807/Stream-CQSA-v3/results/v3/logs/E_pcie_%j.out
# The gpu-short PCIe job of the v3 experiments: E2's host-resident 4M and 8M rows (parts E2:pcie_fwd, E2:pcie_bwd;
# experiments/v3/PLAN.md) with their own resident baselines on one A100-80GB-PCIe (never compared with the SXM
# rows). Resumable like E_sxm.slurm (slurm/v3/parts.sh): every part skips recorded rows and defers what its budget
# cannot hold; resubmit until the last line says "MISSING 0". Estimated (bench.py plan, PCIe 1.25x SXM): pcie_fwd
# ~3.5 h, pcie_bwd ~10.5 h (the 8M fwd+bwd host-resident rows are ~1.4 h each; the 8M FA-2 / SDPA fwd+bwd rows
# are expected OOM rows), so two submissions.
#
# --mem: the largest host-resident row, E2:pcie_bwd at N=8M (acc=cpu, host-resident Q/K/V, fwd+bwd on the classic
# streamed backward with host gradients): pinned Q/K/V 3 x 8M x 1 KiB/token = 24 GiB + pinned dO 8 GiB + the fp32
# host accumulator (output) 16 GiB + its fp16 copy 8 GiB + the backward's fp32 host gradients 3 x 16 = 48 GiB +
# their fp16 casts 24 GiB + pinned task staging ~6 GiB + interpreter / CUDA context ~3 GiB = ~137 GiB -> 150G (one
# of the node's 4 GPU shares of 1 TB). The forward-only part needs 24 + 16 + 8 + ~6 + 3 = ~57 GiB: submit it alone with
#   sbatch --mem=60G --time=04:00:00 --export=ALL,...,CQSA_PARTS="E2:pcie_fwd" slurm/v3/E_pcie.slurm
#
#   SNAP=$(bash slurm/ultimate/snapshot.sh) && sbatch --export=ALL,CQSA_SRC=$SNAP,CQSA_V13_SO=<dir>,CQSA_NATIVE_V2_SO=<dir> slurm/v3/E_pcie.slurm
# Knobs: CQSA_PARTS, CQSA_PART_BUDGET_S (default 0 = whatever remains of the job), E_ARGS.
set -u
export CQSA_SRC=${CQSA_SRC:?CQSA_SRC (frozen code root) is required}
export CQSA_V13_SO=${CQSA_V13_SO:?CQSA_V13_SO (frozen cqsa_cuda_v13 directory) is required}
export CQSA_NATIVE_V2_SO=${CQSA_NATIVE_V2_SO:?CQSA_NATIVE_V2_SO (frozen cqsa_native_v2 directory) is required}
export CQSA_PARTS=${CQSA_PARTS:-"E2:pcie_fwd E2:pcie_bwd"}
export CQSA_PART_BUDGET_S=${CQSA_PART_BUDGET_S:-0}
export CQSA_JOB_TOTAL_S=${CQSA_JOB_TOTAL_S:-$((12 * 3600))}
export CQSA_V3_RESULTS=${CQSA_V3_RESULTS:-/scratch/gpfs/AKEY/yb2807/Stream-CQSA-v3/results/v3}
bash "$CQSA_SRC/slurm/v3/parts.sh"
