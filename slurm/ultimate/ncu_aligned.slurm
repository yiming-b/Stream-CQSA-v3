#!/bin/bash
#SBATCH --job-name=cqsa_ncu_aligned
#SBATCH --account=akey
#SBATCH --qos=gpu-short
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=16G
#SBATCH --gres=gpu:a100:1 --constraint="gpu80&pcie"
#SBATCH --time=00:50:00
#SBATCH --output=/scratch/gpfs/AKEY/yb2807/Stream-CQSA-ultimate/results/ultimate/ncu_aligned_%j.out
# M0.3: ncu stall attribution of the all-live tax on the ALIGNED c=7 itr=1 task (N=131072, L=56064).
# Four arms (fa2 = SDPA flash, plain = CQS off, zero = all-live CQS, real = task bits), one kernel launch
# each, one process per arm (one kernel module per process), --set full + SourceCounters + WarpStateStats,
# SM clock locked to base by the profiler. PCIe node: relative metrics only, never compare its durations
# with SXM timings. Host tensors are < 1 GiB (L=56064 fp16 q/k/v + fp32 out); 16G covers torch + ncu.
set -uo pipefail
ROOT=/scratch/gpfs/AKEY/yb2807/Stream-CQSA-ultimate
OUT=$ROOT/results/ultimate/ncu_aligned
NCU=/usr/local/cuda-13.0/bin/ncu
mkdir -p "$OUT"
cd "$ROOT"
source /scratch/gpfs/AKEY/yb2807/Stream-CQSA-dev/next/env_next.sh
export PYTHONPATH=$ROOT:/scratch/gpfs/AKEY/yb2807/Stream-CQSA-dev/next/kernel/stable:${PYTHONPATH:-}:/scratch/gpfs/AKEY/yb2807/Stream-CQSA-dev/next/native
export CQSA_CUDA_MODULE=cqsa_cuda_next_v11 CQSA_CUDA_MODULE_NONCAUSAL=cqsa_cuda_next_v9 CQSA_CPU_THREADS=4 OMP_NUM_THREADS=4
hostname; which python; ls -la "$NCU"; "$NCU" --version | tail -1
nvidia-smi --query-gpu=name,clocks.sm,clocks.max.sm,memory.total --format=csv,noheader
python -P -c "import stream_cqsa, torch; print(stream_cqsa.__file__, torch.__version__)"

ARMS=(${ARMS:-fa2 plain zero real})
for arm in "${ARMS[@]}"; do
  echo "=============== arm=$arm  $(date +%T)"
  # -c 1 inside the NVTX "measure" range: the warm-up launch before the range is not profiled.
  timeout 12m "$NCU" --nvtx --nvtx-include "measure/" -k regex:flash_fwd_kernel -c 1 \
      --set full --section SourceCounters --section WarpStateStats \
      --clock-control base --import-source on -f -o "$OUT/$arm" \
      python -P benchmarks/ultimate/ncu_aligned.py run --arm "$arm" --row-json "$OUT/$arm.row.json" 2>&1 | grep -v '^==PROF=='
  echo "ncu exit: ${PIPESTATUS[0]}  $(date +%T)"
  if [ -f "$OUT/$arm.ncu-rep" ]; then
    # Human-readable and machine-readable dumps, so the summary can be rebuilt on the login node without a GPU.
    "$NCU" -i "$OUT/$arm.ncu-rep" --page details --print-details all > "$OUT/$arm.details.txt" 2>&1
    "$NCU" -i "$OUT/$arm.ncu-rep" --page raw --csv --print-units base > "$OUT/$arm.raw.csv" 2>&1
    "$NCU" -i "$OUT/$arm.ncu-rep" --page source --print-source sass --csv --print-units base > "$OUT/$arm.sass.csv" 2>&1
    ls -la "$OUT/$arm".*
    grep -E 'Duration|SM Frequency|Executed Instructions|Warp Cycles Per Issued|Eligible Warps|Registers Per Thread|Achieved Occupancy|L1/TEX Hit|L2 Hit|Stall' "$OUT/$arm.details.txt" | head -40
  fi
done

echo "=============== summarize  $(date +%T)"
python -P benchmarks/ultimate/ncu_aligned.py summarize --dir "$OUT" --out "$ROOT/results/ultimate/ncu_aligned.md"
echo "done $(date +%T)"
