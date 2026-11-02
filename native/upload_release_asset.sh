#!/bin/bash
# Attach a locally built wheel to a GitHub release (the della torch-2.10/CUDA-13 wheels that the hosted
# runners cannot build). Needs a token with "contents: write" on the repo in GITHUB_TOKEN; it is sent only
# to api.github.com / uploads.github.com and never printed.
#
#   GITHUB_TOKEN=... bash native/upload_release_asset.sh yiming-b/Stream-CQSA-v3 v3.0.0 dist/stream_cqsa-3.0.0-*.whl
#   GITHUB_TOKEN=... bash native/upload_release_asset.sh --notes docs/releases/v3.0.0.md yiming-b/Stream-CQSA-v3 v3.0.0
# --notes FILE sets the release body (the workflow's release job leaves it empty); files are optional.
set -euo pipefail
NOTES=""
if [ "${1:-}" = "--notes" ]; then NOTES=$2; shift 2; fi
REPO=$1; TAG=$2; shift 2
[ -n "${GITHUB_TOKEN:-}" ] || { echo "GITHUB_TOKEN is not set" >&2; exit 2; }
AUTH=(-H "Authorization: Bearer $GITHUB_TOKEN" -H "Accept: application/vnd.github+json")
REL=$(curl -sS "${AUTH[@]}" "https://api.github.com/repos/$REPO/releases/tags/$TAG")
RID=$(python3 -c "import json,sys; d=json.loads(sys.argv[1]); print(d.get('id',''))" "$REL")
[ -n "$RID" ] || { echo "no release for tag $TAG in $REPO (the workflow's release job creates it after the CUDA matrix)" >&2; exit 3; }
if [ -n "$NOTES" ]; then
  echo "setting the release body from $NOTES"
  python3 -c "import json,sys; print(json.dumps({'body': open(sys.argv[1]).read()}))" "$NOTES" \
    | curl -sS "${AUTH[@]}" -X PATCH -H "Content-Type: application/json" --data-binary @- \
        "https://api.github.com/repos/$REPO/releases/$RID" \
    | python3 -c "import json,sys; d=json.load(sys.stdin); print('  ->', d.get('html_url') or d, len(d.get('body') or ''), 'chars')"
fi
for F in "$@"; do
  NAME=$(basename "$F")
  # replace an asset of the same name (a re-upload after a failed transfer)
  AID=$(python3 -c "import json,sys; d=json.loads(sys.argv[1]); print(next((a['id'] for a in d.get('assets',[]) if a['name']==sys.argv[2]), ''))" "$REL" "$NAME")
  if [ -n "$AID" ]; then
    echo "deleting the existing asset $NAME (id $AID)"
    curl -sS -o /dev/null "${AUTH[@]}" -X DELETE "https://api.github.com/repos/$REPO/releases/assets/$AID"
  fi
  echo "uploading $NAME ($(du -h "$F" | cut -f1)) to $REPO $TAG"
  curl -sS "${AUTH[@]}" -H "Content-Type: application/octet-stream" \
    --data-binary @"$F" "https://uploads.github.com/repos/$REPO/releases/$RID/assets?name=$NAME" \
    | python3 -c "import json,sys; d=json.load(sys.stdin); print('  ->', d.get('state'), d.get('size'), d.get('browser_download_url') or d)"
done
echo "assets now:"
curl -sS "https://api.github.com/repos/$REPO/releases/tags/$TAG" | python3 -c "import json,sys; [print(' ', a['name'], a['size']) for a in json.load(sys.stdin).get('assets',[])]"
