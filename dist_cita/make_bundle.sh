#!/bin/bash
# Rebuild the CITA offline bundle from the v3 repo's HEAD plus dist_cita/wheels/.
#
#   bash dist_cita/make_bundle.sh                 -> /scratch/gpfs/AKEY/yb2807/cita_bundle/stream_cqsa_v3_cita_<date>.tar.gz
#   CITA_BUNDLE_DIR=<dir> bash dist_cita/make_bundle.sh
#   bash dist_cita/make_bundle.sh --download      (re-download the wheels of wheels.txt first; needs network)
#
# The tarball unpacks to one directory stream_cqsa_v3_cita_<date>/ with
#   README.md  setup_cita.sh  run_cita.py  run_cita.ipynb  wheels/  src/  GIT_COMMIT
# src/ is `git archive HEAD` of the repository (every tracked file) minus results/, kernels/, csrc_nc/,
# docs/ultimate/kernel_diffs/, slurm/, .github/ and the notebooks' outputs; it carries a GIT_COMMIT file
# (bench.py records it in every row). Nothing is built here: no .so, no build/ directory.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
OUT_DIR="${CITA_BUNDLE_DIR:-/scratch/gpfs/AKEY/yb2807/cita_bundle}"
PY="${PYTHON:-python}"
DATE="$(date +%Y%m%d)"
NAME="stream_cqsa_v3_cita_${DATE}"
COMMIT="$(git -C "$REPO" rev-parse --short HEAD)"
COMMIT_LONG="$(git -C "$REPO" rev-parse HEAD)"

if [ "${1:-}" = "--download" ]; then
    echo "== downloading the wheels of wheels.txt (python 3.12, manylinux2014 x86_64)"
    export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/scratch/gpfs/AKEY/yb2807/VENVs/.cache/pip}"
    "$PY" -m pip download --python-version 3.12 --platform manylinux2014_x86_64 --only-binary=:all: --no-deps \
        -d "$HERE/wheels" -r "$HERE/wheels.txt"
    # the flash-attn sdist (optional on CITA; pip download cannot prepare its metadata without torch, so fetch the file)
    if [ ! -f "$HERE/wheels/flash_attn-2.8.3.tar.gz" ]; then
        curl -sL -o "$HERE/wheels/flash_attn-2.8.3.tar.gz" \
            https://files.pythonhosted.org/packages/3b/b2/8d76c41ad7974ee264754709c22963447f7f8134613fd9ce80984ed0dab7/flash_attn-2.8.3.tar.gz
    fi
fi

n_wheels="$(ls "$HERE"/wheels/*.whl 2>/dev/null | wc -l)"
if [ "$n_wheels" -lt 40 ]; then
    echo "dist_cita/wheels/ has $n_wheels wheels; run with --download first" >&2
    exit 1
fi
if [ -n "$(git -C "$REPO" status --porcelain -- stream_cqsa csrc native experiments tests setup.py pyproject.toml dist_cita/setup_cita.sh dist_cita/run_cita.py dist_cita/run_cita.ipynb dist_cita/README.md)" ]; then
    echo "WARNING: uncommitted changes in the files that go into the bundle; src/ is HEAD ($COMMIT), the dist_cita files are the working tree" >&2
fi

STAGE="$(mktemp -d "${TMPDIR:-/tmp}/cita_bundle.XXXXXX")"
trap 'rm -rf "$STAGE"' EXIT
B="$STAGE/$NAME"
mkdir -p "$B/src" "$OUT_DIR"

echo "== src/: git archive $COMMIT"
git -C "$REPO" archive --format=tar HEAD | tar -x -C "$B/src"
rm -rf "$B/src/results" "$B/src/kernels" "$B/src/csrc_nc" "$B/src/docs/ultimate/kernel_diffs" "$B/src/slurm" \
       "$B/src/.github" "$B/src/dist_cita" "$B/src/Stream_CQSA_Auto_Mode_OOC_Rivals_Experiment_Prompt.md"
find "$B/src" -name "*.so" -delete
# the notebooks without their executed outputs
"$PY" - "$B/src/notebooks" <<'EOF'
import glob, json, os, sys
d = sys.argv[1]
for p in glob.glob(os.path.join(d, "*.ipynb")):
    nb = json.load(open(p))
    for c in nb.get("cells", []):
        if c.get("cell_type") == "code":
            c["outputs"] = []
            c["execution_count"] = None
    json.dump(nb, open(p, "w"), indent=1)
EOF
echo "$COMMIT_LONG $(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$B/src/GIT_COMMIT"
cp "$B/src/GIT_COMMIT" "$B/GIT_COMMIT"
test -f "$B/src/csrc/cutlass/include/cutlass/cutlass.h" || { echo "csrc/cutlass is incomplete in HEAD" >&2; exit 1; }
test -f "$B/src/csrc/cutlass/include/cute/tensor.hpp" || { echo "csrc/cutlass/include/cute is incomplete in HEAD" >&2; exit 1; }

echo "== dist files"
cp "$HERE/README.md" "$HERE/setup_cita.sh" "$HERE/run_cita.py" "$HERE/run_cita.ipynb" "$HERE/wheels.txt" "$B/"
mkdir -p "$B/wheels"
cp "$HERE"/wheels/*.whl "$B/wheels/"
[ -f "$HERE/wheels/flash_attn-2.8.3.tar.gz" ] && cp "$HERE/wheels/flash_attn-2.8.3.tar.gz" "$B/wheels/"
chmod +x "$B/setup_cita.sh"

( cd "$B" && find . -type f ! -name MANIFEST.txt ! -name SHA256SUMS | sort | sed 's#^\./##' > MANIFEST.txt \
  && find . -type f ! -name MANIFEST.txt ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS )

TAR="$OUT_DIR/$NAME.tar.gz"
echo "== $TAR"
tar -C "$STAGE" -czf "$TAR" "$NAME"
ls -la "$TAR"
echo "files: $(wc -l < "$B/MANIFEST.txt")  src commit: $COMMIT  wheels: $n_wheels"
