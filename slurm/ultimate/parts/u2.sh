#!/bin/bash
# Package U2's part of the combined ultimate job: the wave A/B (benchmarks/ultimate/wave_ab.py) on ONE GPU.
#
# Environment (set by the combined job):
#   CQSA_SRC           frozen code root (slurm/ultimate/snapshot.sh), the stream_cqsa package and the harness
#   CQSA_NATIVE_V2_SO  directory holding the frozen cqsa_native_v2 module (.so): the native wave kernel v2
#   CQSA_SRC_RESULTS   the live repo's results/ultimate/ (wave_ab.json + wave_ab.md land there)
#   CQSA_V13_SO        (unused here) U1's classic module: the per-task arms of this part are the frozen v12f b3 /
#                      v12b b5 modules the kernels report measured (V12F_DIR / V12B_DIR below), so the wave is
#                      compared with exactly those kernels
#   WAVE_AB_ITERS      timed runs per arm (default 5; 1 warm-up each)
#
# Arms (fp16, H=8, D=64, causal; A100-SXM4-80GB expected): N=131072 c=7 (one launch of 7 subproblems) and c=73
# (one launch of 73), N=1048576 c=7 (the engine's default-budget waves, 2 subproblems per launch): cqsa_native_v2
# fwd_wave / bwd_wave (gathered and in place) vs the first wave module cqsa_native vs the per-task v12f / v12b
# loops vs monolithic flash_attn and SDPA-flash; W=1 rows (the smallest task as a wave vs the classic call);
# engine rows (wave_forward / wave_backward end to end on both modules). Identity in every part. A failing arm
# records its error in the row and the later arms run; a failing part records a traceback in `errors`.
#
# Expected: ~25-30 min of GPU time on an A100-SXM4-80GB (the 1M backward arms dominate: ~12 min), device peak
# ~30 GiB (1M: Q/K/V/dO, the per-task copies of 7 tasks of L~450K, one wave's packed inputs and fp32 dq_accum,
# the flash_attn reference), host RSS ~4 GB (the interpreter, the 1M task tables and wave tables); --mem=20G.
#
# Read the results: $CQSA_SRC_RESULTS/wave_ab.md, sections Forward / Backward / Engine and "Targets": W=1 v2 /
# v12f (v12b) within 1% (the classic call pays nothing for the wave layout), c=73 one-launch forward <= 1.10x
# monolithic flash_attn, old bwd_wave / v2 >= 2x; `equal v12f`, `equal old`, `dk/dv equal v12b` must all be
# True and routes [0]; "Failed arms" / "Errors" sections must be absent.
set -uo pipefail
ROOT=${CQSA_SRC:?CQSA_SRC (frozen code root) is required}
OUT=${CQSA_SRC_RESULTS:?CQSA_SRC_RESULTS (results directory) is required}
NATIVE_V2_DIR=${CQSA_NATIVE_V2_SO:?CQSA_NATIVE_V2_SO (frozen cqsa_native_v2 directory) is required}
DEV=/scratch/gpfs/AKEY/yb2807/Stream-CQSA-dev
KER=$DEV/next/kernel
V12F_DIR=${V12F_DIR:-$KER/v12f/frozen/b3}
V12B_DIR=${V12B_DIR:-$KER/v12b/frozen/b5}
echo "== U2 part: wave A/B  $(date)"
nvidia-smi --query-gpu=name,clocks.sm,clocks.max.sm,memory.total --format=csv,noheader
hostname
source $DEV/next/env_next.sh
# the variant dirs carry a stub stream_cqsa/__init__.py for their setup.py, so they go LAST
export PYTHONPATH=$ROOT:$KER/stable:${PYTHONPATH:-}:$DEV/next/native:$V12F_DIR:$V12B_DIR:$NATIVE_V2_DIR
export CQSA_CUDA_MODULE=cqsa_cuda_next_v11 CQSA_CUDA_MODULE_NONCAUSAL=cqsa_cuda_next_v9 CQSA_CPU_THREADS=4 OMP_NUM_THREADS=4
cd "$ROOT"
echo "code: $ROOT"
python -P -c "import stream_cqsa, torch, flash_attn, cqsa_native_v2 as a, cqsa_native as b, cqsa_cuda_next_v12f as c, cqsa_cuda_next_v12b as d; print(stream_cqsa.__file__, torch.__version__, flash_attn.__version__); print(a.__file__, 'WAVE_API', a.WAVE_API); print(b.__file__); print(c.__file__); print(d.__file__)" || { echo "U2 part: module import failed"; exit 1; }
ls -la --time-style=full-iso "$NATIVE_V2_DIR"/cqsa_native_v2*.so; md5sum "$NATIVE_V2_DIR"/cqsa_native_v2*.so
mkdir -p "$OUT"
python -P -u "$ROOT/benchmarks/ultimate/wave_ab.py" --iters "${WAVE_AB_ITERS:-5}" --out "$OUT/wave_ab.json"
rc=$?
echo "== U2 part: exit $rc  $(date)"
exit $rc
