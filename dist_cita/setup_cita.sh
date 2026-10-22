#!/bin/bash
# Stream-CQSA v3 on CITA (H100, no network). Run inside the ACTIVATED venv, from the bundle directory:
#
#     source <venv>/bin/activate
#     bash setup_cita.sh 2>&1 | tee setup.log
#
# What it does, in order (every step logs to a file next to this script and never stops the next step):
#   0. the wheels CITA lacks (pytest, nbformat, nbclient, nbconvert, ipykernel + deps), offline from wheels/
#   a. stream_cqsa from src/ with the Triton kernels only (CQSA_SKIP_EXT=1, no compiler): ALWAYS works
#   b. the classic CUDA module (csrc/ = v13: census + ALIGNED kernels, fp16 + bf16, D 64 / 128, causal and
#      non-causal, forward + backward) built in place for sm90 (H100)         -> build_classic.log
#   c. the wave CUDA module (native/ = wave kernel v2) built in place for sm90   -> build_native.log
#   d. python -m stream_cqsa.doctor                                             -> doctor.txt
# and prints one line per tier: OK / SKIPPED(reason) / FAILED(see log). The tiers file tiers.txt is read by the
# notebook. (b) and (c) need a CUDA 12.x nvcc: the pip wheel nvidia-cuda-nvcc-cu12 contains only ptxas and nvvm,
# NOT nvcc, so a system / module CUDA 12.x is used when found, else the bundled cuda_min/ (nvcc 12.9.86 from
# conda-forge, relocatable; the cuBLAS / cuSPARSE / cuSOLVER headers come from the venv's pip wheels). Without any,
# the run is Triton-only, which is a complete run (the classic_Triton and wave_Triton series; exact kernels).
#
# Knobs: CUDA_HOME (a CUDA 12.x toolkit root), CITA_USE_BUNDLED_NVCC=1 (cuda_min even if a system nvcc exists),
#        CITA_MAX_JOBS (parallel nvcc jobs; default min(nproc/2, 16)), CITA_SKIP_BUILD=1 (tier a only),
#        CITA_REBUILD=1 (rebuild even if a module imports already), NVCC_APPEND_FLAGS (e.g. -allow-unsupported-compiler).
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$HERE/src"
PY="${PYTHON:-python}"
TIERS="$HERE/tiers.txt"
: > "$TIERS"

tier() {   # tier <name> <status> <detail...>
    printf '%-14s %-10s %s\n' "$1" "$2" "${*:3}" | tee -a "$TIERS"
}
say() { echo; echo "==== $*"; }

say "setup_cita.sh in $HERE ($(date))"
echo "python: $(command -v "$PY")"
"$PY" -V || { echo "no python on PATH: activate the venv first"; exit 1; }
if [ -z "${VIRTUAL_ENV:-}" ] && [ -z "${CONDA_PREFIX:-}" ]; then
    echo "WARNING: no venv seems active (VIRTUAL_ENV / CONDA_PREFIX unset); packages go wherever '$PY' installs"
fi
"$PY" - <<'EOF' || { echo "torch is not importable in this python: activate the venv that has torch 2.13"; exit 1; }
import torch, sys
print(f"torch {torch.__version__}  cuda {torch.version.cuda}  cuda available {torch.cuda.is_available()}  python {sys.version.split()[0]}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {p.name}  sm_{p.major}{p.minor}  {p.total_memory / 2**30:.0f} GiB")
try:
    import triton; print(f"triton {triton.__version__}")
except Exception as e:
    print(f"triton: NOT importable ({e!r}); the Triton tier needs it")
EOF

# ---------------------------------------------------------------------------------------------------------------
say "0. wheels (offline, from $HERE/wheels)"
if "$PY" -m pip install --no-index --find-links "$HERE/wheels" pytest nbformat nbclient nbconvert ipykernel > "$HERE/pip_wheels.log" 2>&1; then
    "$PY" -c "import pytest, nbformat, nbclient, nbconvert, ipykernel; print('pytest', pytest.__version__, 'nbclient', nbclient.__version__, 'ipykernel', ipykernel.__version__)"
    tier wheels OK "pytest nbformat nbclient nbconvert ipykernel (pip_wheels.log)"
else
    tail -20 "$HERE/pip_wheels.log"
    tier wheels FAILED "see pip_wheels.log (run_cita.py falls back to in-process execution without nbclient)"
fi

# ---------------------------------------------------------------------------------------------------------------
say "a. stream_cqsa, Triton kernels only (CQSA_SKIP_EXT=1 pip install -e src/)"
if CQSA_SKIP_EXT=1 "$PY" -m pip install --no-index --no-build-isolation --no-deps -e "$SRC" > "$HERE/pip_src.log" 2>&1; then
    echo "editable install ok (pip_src.log)"
else
    tail -20 "$HERE/pip_src.log"
    echo "pip install -e failed; falling back to a .pth file (the package is importable from src/ all the same)"
fi
# the editable install resolves the stream_cqsa package only; the compiled modules of tiers (b) / (c) land next to it
# in src/ as top-level .so files, so src/ itself goes on sys.path through a .pth file (also the fallback above)
SITE="$("$PY" -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")"
echo "$SRC" > "$SITE/stream_cqsa_cita.pth" && echo "sys.path entry: $SITE/stream_cqsa_cita.pth -> $SRC"
if "$PY" -P - <<'EOF' > "$HERE/tier_a.log" 2>&1
import stream_cqsa, stream_cqsa.triton_kernel, os
print("stream_cqsa", stream_cqsa.__version__, "from", os.path.dirname(stream_cqsa.__file__))
ks = stream_cqsa.kernels_available()
print("kernels_available:", ks)
assert ks["triton"], "the Triton kernels are not available (triton not importable?)"
EOF
then
    cat "$HERE/tier_a.log"
    tier triton OK "$(head -1 "$HERE/tier_a.log")"
else
    cat "$HERE/tier_a.log"
    tier triton FAILED "see tier_a.log"
fi

# ---------------------------------------------------------------------------------------------------------------
say "nvcc: looking for a CUDA toolkit of torch's major version"
NVCC=""
candidates=()
[ -n "${CUDA_HOME:-}" ] && candidates+=("$CUDA_HOME/bin/nvcc")
[ -n "${CUDA_PATH:-}" ] && candidates+=("$CUDA_PATH/bin/nvcc")
w="$(command -v nvcc 2>/dev/null || true)"; [ -n "$w" ] && candidates+=("$w")
for d in /usr/local/cuda-* /usr/local/cuda /opt/cuda-* /opt/cuda /usr/lib/cuda /opt/nvidia/hpc_sdk/Linux_x86_64/*/cuda/* /usr/lib/nvidia-cuda-toolkit; do
    [ -x "$d/bin/nvcc" ] && candidates+=("$d/bin/nvcc")
done
TORCH_CUDA_VER="$("$PY" -c "import torch; print(torch.version.cuda or '0.0')")"
TORCH_CUDA_MAJOR="${TORCH_CUDA_VER%%.*}"
TORCH_CUDA_MINOR="$(echo "$TORCH_CUDA_VER" | cut -d. -f2)"
echo "  torch was built with CUDA $TORCH_CUDA_VER: an nvcc $TORCH_CUDA_MAJOR.x is required, $TORCH_CUDA_VER preferred"
# every candidate once; the same major as torch's CUDA is required (torch's extension builder refuses another major);
# among those the release closest to torch's minor is preferred (a far newer minor can break on torch's own headers)
best_score=999
seen=""
for c in "${candidates[@]}"; do
    [ -x "$c" ] || continue
    real="$(readlink -f "$c")"
    case " $seen " in *" $real "*) continue;; esac
    seen="$seen $real"
    ver="$("$c" --version 2>/dev/null | sed -n 's/.*release \([0-9]*\)\.\([0-9]*\).*/\1.\2/p' | head -1)"
    echo "  candidate $c: release ${ver:-?}"
    [ "${ver%%.*}" = "$TORCH_CUDA_MAJOR" ] || continue
    minor="$(echo "$ver" | cut -d. -f2)"
    score=$(( minor - TORCH_CUDA_MINOR )); [ $score -lt 0 ] && score=$(( -score ))
    if [ $score -lt $best_score ]; then
        best_score=$score; NVCC="$c"; NVCC_VER="$ver"
    fi
done
PIP_NVCC="$("$PY" -c "import nvidia.cuda_nvcc as m; print(list(m.__path__)[0])" 2>/dev/null || true)"
if [ -n "$PIP_NVCC" ]; then
    if [ -x "$PIP_NVCC/bin/nvcc" ]; then
        ver="$("$PIP_NVCC/bin/nvcc" --version 2>/dev/null | sed -n 's/.*release \([0-9]*\)\.\([0-9]*\).*/\1.\2/p' | head -1)"
        echo "  pip nvidia-cuda-nvcc: $PIP_NVCC/bin/nvcc release ${ver:-?}"
        [ -z "$NVCC" ] && [ "${ver%%.*}" = "$TORCH_CUDA_MAJOR" ] && { NVCC="$PIP_NVCC/bin/nvcc"; NVCC_VER="$ver"; }
    else
        echo "  pip nvidia-cuda-nvcc at $PIP_NVCC has: $(ls "$PIP_NVCC/bin" 2>/dev/null | tr '\n' ' ') -- no nvcc (the wheel ships ptxas + nvvm only)"
    fi
fi
# the bundled toolchain: cuda_min/ (nvcc 12.9.86, cicc, ptxas, cudafe++, fatbinary, nvlink, nvvm + libdevice, the crt / cudart /
# CCCL headers, libcudart; the conda-forge build of CUDA 12.9, relocatable). Used when no system nvcc of torch's major version
# exists (or with CITA_USE_BUNDLED_NVCC=1). The cuBLAS / cuSPARSE / cuSOLVER headers that torch's ATen headers include come
# from the venv's pip wheels (nvidia/*/include) and are linked into it here.
BUNDLED="$HERE/cuda_min"
if [ -x "$BUNDLED/bin/nvcc" ] && { [ -z "$NVCC" ] || [ "${CITA_USE_BUNDLED_NVCC:-}" = "1" ]; }; then
    bver="$("$BUNDLED/bin/nvcc" --version 2>/dev/null | sed -n 's/.*release \([0-9]*\)\.\([0-9]*\).*/\1.\2/p' | head -1)"
    echo "  bundled cuda_min: $BUNDLED/bin/nvcc release ${bver:-? (does not run on this machine)}"
    if [ "${bver%%.*}" = "$TORCH_CUDA_MAJOR" ]; then
        inc="$BUNDLED/targets/x86_64-linux/include"
        n_link=0
        for d in "$SITE"/nvidia/*/include; do
            [ -d "$d" ] || continue
            for f in "$d"/*; do
                b="$(basename "$f")"
                case "$b" in __*) continue;; esac                                  # __init__.py / __pycache__ of the wheel
                [ -e "$inc/$b" ] || { ln -s "$f" "$inc/$b" && n_link=$(( n_link + 1 )); }
            done
        done
        echo "  linked $n_link header entries from $SITE/nvidia/*/include into $inc"
        for h in cublas_v2.h cublasLt.h cusparse.h cusolverDn.h; do
            [ -e "$inc/$h" ] || echo "  WARNING: $h not found in the venv's nvidia/*/include: torch's ATen/cuda/CUDAContext.h needs it (pip package nvidia-cublas-cu12 / nvidia-cusparse-cu12 / nvidia-cusolver-cu12)"
        done
        NVCC="$BUNDLED/bin/nvcc"; NVCC_VER="$bver"
    fi
fi
NVCC_REASON=""
if [ -z "$NVCC" ]; then
    NVCC_REASON="no nvcc of CUDA $TORCH_CUDA_MAJOR.x found (torch is cu$("$PY" -c "import torch; print((torch.version.cuda or '').replace('.', ''))")); set CUDA_HOME=<toolkit root> or 'module load cuda/12.x' and re-run"
    echo "  $NVCC_REASON"
else
    export CUDA_HOME="$(cd "$(dirname "$NVCC")/.." && pwd)"
    echo "  using $NVCC (release $NVCC_VER), CUDA_HOME=$CUDA_HOME"
    [ -f "$CUDA_HOME/include/cuda_runtime.h" ] || echo "  WARNING: $CUDA_HOME/include/cuda_runtime.h not found; the build will probably fail (incomplete toolkit?)"
    GXX="$(command -v g++ || true)"; [ -n "$GXX" ] && echo "  host compiler: $GXX ($("$GXX" --version | head -1))" || echo "  WARNING: no g++ on PATH (nvcc needs one; set CXX / NVCC_APPEND_FLAGS=-allow-unsupported-compiler for a newer gcc)"
fi

NPROC="$(nproc 2>/dev/null || echo 8)"
JOBS="${CITA_MAX_JOBS:-$(( NPROC / 2 ))}"
[ "$JOBS" -gt 16 ] && JOBS=16
[ "$JOBS" -lt 1 ] && JOBS=1

build_tier() {    # build_tier <label> <module> <log> <env assignments...>
    local label="$1" mod="$2" log="$3"; shift 3
    if [ "${CITA_SKIP_BUILD:-}" = "1" ]; then tier "$label" SKIPPED "CITA_SKIP_BUILD=1"; return; fi
    if [ "${CITA_REBUILD:-}" != "1" ] && "$PY" -P -c "import $mod" > /dev/null 2>&1; then
        tier "$label" OK "already importable ($("$PY" -P -c "import $mod; print($mod.__file__)")); CITA_REBUILD=1 to rebuild"; return
    fi
    if [ -z "$NVCC" ]; then tier "$label" SKIPPED "$NVCC_REASON"; return; fi
    echo "building $mod for sm90 with $JOBS jobs; log: $log (this takes 20-60 min)"
    local t0=$SECONDS
    ( cd "$SRC" && env "$@" CUDA_HOME="$CUDA_HOME" FLASH_ATTN_CUDA_ARCHS=90 TORCH_CUDA_ARCH_LIST=9.0 MAX_JOBS="$JOBS" NVCC_THREADS=2 \
        "$PY" setup.py build_ext --inplace ) > "$log" 2>&1
    local rc=$? dt=$(( SECONDS - t0 ))
    if [ $rc -ne 0 ]; then
        grep -m 5 -n -i "error" "$log" | cut -c1-200
        tier "$label" FAILED "setup.py rc=$rc after $(( dt / 60 )) min; see $(basename "$log")"
    elif "$PY" -P -c "import $mod" >> "$log" 2>&1; then
        tier "$label" OK "built in $(( dt / 60 )) min: $("$PY" -P -c "import $mod; print($mod.__file__)")"
    else
        tail -5 "$log"
        tier "$label" FAILED "built in $(( dt / 60 )) min but 'import $mod' fails (ABI / load error at the end of $(basename "$log"))"
    fi
}

say "b. classic CUDA module (csrc/, CQSA_KERNEL_SET=common, sm90)"
build_tier classic_cuda cqsa_cuda "$HERE/build_classic.log" CQSA_KERNEL_SET=common

say "c. wave CUDA module (native/, sm90)"
# CQSA_ONLY_NATIVE=1 builds just cqsa_native (the same sources CQSA_BUILD_NATIVE=1 adds to a full build), so a
# failed classic build does not stop this tier
build_tier wave_cuda cqsa_native "$HERE/build_native.log" CQSA_ONLY_NATIVE=1 CQSA_KERNEL_SET=common

# ---------------------------------------------------------------------------------------------------------------
say "d. python -m stream_cqsa.doctor"
"$PY" -P -m stream_cqsa.doctor > "$HERE/doctor.txt" 2>&1; rc=$?
cat "$HERE/doctor.txt"
if [ $rc -eq 0 ]; then tier doctor OK "doctor.txt"; else tier doctor FAILED "exit code $rc (doctor.txt)"; fi
"$PY" -P -c "
import stream_cqsa, json
ks = stream_cqsa.kernels_available()
series = ['classic_Triton', 'wave_Triton'] + (['classic_CUDA'] if ks['cuda_extension'] else []) + (['wave_CUDA'] if ks['native_wave'] else [])
print('series the notebook will run:', ', '.join(series), '| FA-2 arm:', 'flash_attn' if ks.get('flash_attn') else 'absent (SDPA-flash is the reference)')
" 2>/dev/null || true

say "summary ($TIERS)"
cat "$TIERS"
echo
echo "next: python run_cita.py            (CITA_QUICK=1 python run_cita.py for the ~20-minute first pass)"
