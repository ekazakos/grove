#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <links.txt> <output_dir>"
  exit 1
fi

LINKS="$1"
OUTDIR="$2"

[[ -f "$LINKS" ]] || { echo "Links file not found: $LINKS" >&2; exit 1; }
mkdir -p "$OUTDIR"

# Download one URL with wget, using a clean output name (basename before '?')
download_one() {
  local url="$1"

  # skip blanks/comments
  [[ -z "${url// }" || "$url" =~ ^# ]] && return 0

  # trim trailing whitespace and a stray trailing colon (common copy/paste artefact)
  url="${url%%[[:space:]]*}"
  url="${url%:}"

  # filename = last path segment without query string
  local clean="${url%%\?*}"
  local name
  name="$(basename "$clean")"
  if [[ -z "$name" || "$name" == "/" ]]; then
    echo "Skip (no basename): $url" >&2
    return 0
  fi

  echo "→ $name"
  # -O sets the exact filename; --continue resumes if partial exists.
  # Explicit GET; don't rely on Content-Disposition (S3 may not send one).
  wget \
    --method=GET \
    --continue \
    --tries=5 \
    --timeout=60 \
    --no-verbose \
    --output-document="$OUTDIR/$name" \
    "$url"
}

# sequential (safe). If you want parallel, see note below.
while IFS= read -r line || [[ -n "$line" ]]; do
  download_one "$line"
done < "$LINKS"

echo "Done."
