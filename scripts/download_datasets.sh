#!/usr/bin/env bash
set -euo pipefail

#########################
# USER CONFIGURATION
#########################

BASE_URL="https://data.ciirc.cvut.cz"

# IMPORTANT: This is the actual remote path on the server
REMOTE_DATASET_PATH="/datasets/how-to-100m-small"

# Dataset auth token (recommended: export TOKEN=... instead of hardcoding)
TOKEN="${TOKEN:-}"

LINES_PER_PART=100000

# Aria2c parallelism configuration
MAX_CONCURRENT_DOWNLOADS=32

#########################
# HELPER FUNCTIONS
#########################

check_aria2c() {
  if command -v aria2c >/dev/null 2>&1; then
    return
  fi

  echo "ERROR: 'aria2c' command not found." >&2
  echo >&2
  echo "Please install aria2 (the multi-connection download tool) using your package manager, e.g.:" >&2
  echo "  Debian/Ubuntu:   sudo apt-get update && sudo apt-get install aria2" >&2
  echo "  CentOS/RHEL:     sudo yum install aria2" >&2
  echo "  Fedora:          sudo dnf install aria2" >&2
  echo "  Arch Linux:      sudo pacman -S aria2" >&2
  echo "  macOS (Homebrew): brew install aria2" >&2
  echo >&2
  echo "After installing aria2, run this script again." >&2
  exit 1
}

need_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "ERROR: required command not found: $1" >&2
    exit 1
  fi
}

usage() {
  cat >&2 <<EOF
Usage:
  $0 <DATASET: howto100m|howtoground1m|iground> <target_download_directory> [annotations_pkl...]

Meaning:
  - DATASET controls behavior:

    howto100m:
      - downloads EVERYTHING from the remote manifest
      - user provides NO PKL files

    howtoground1m:
      - expects 1 PKL (train)
      - PKL formats supported:
          A) {"annotations": { "VIDEOID": ... }}
          B) { "VIDEOID_0": ..., "VIDEOID_1": ... }
      - extracts keys, strips trailing "_<digits>", deduplicates
      - filters manifest to matching video ids

    iground:
      - expects 3 PKLs (train+val+test)
      - same PKL formats + filtering behavior as above

Examples:
  # Download everything
  $0 howto100m /data/howto100m

  # Filtered download (one file)
  $0 howtoground1m /data/howtoground1m annotations_train.pkl

  # Filtered download (three files)
  $0 iground /data/iground ann_train.pkl ann_val.pkl ann_test.pkl

Token:
  Recommended: export TOKEN="YOUR_BEARER_TOKEN"
EOF
  exit 1
}

#########################
# ARGUMENTS / SETUP
#########################

if [ "$#" -lt 2 ]; then
  usage
fi

DATASET="$1"
TARGET_DIR="$2"
shift 2
PKL_FILES=("$@")

case "$DATASET" in
  howto100m)
    if [ "${#PKL_FILES[@]}" -ne 0 ]; then
      echo "ERROR: DATASET=howto100m requires NO PKL files." >&2
      usage
    fi
    ;;
  howtoground1m)
    if [ "${#PKL_FILES[@]}" -ne 1 ]; then
      echo "ERROR: DATASET=howtoground1m requires exactly 1 PKL file (train)." >&2
      usage
    fi
    ;;
  iground)
    if [ "${#PKL_FILES[@]}" -ne 3 ]; then
      echo "ERROR: DATASET=iground requires exactly 3 PKL files (train+val+test)." >&2
      usage
    fi
    ;;
  *)
    echo "ERROR: Unknown DATASET='$DATASET'. Use: howto100m | howtoground1m | iground" >&2
    usage
    ;;
esac

if [ -z "${TOKEN}" ]; then
  echo "ERROR: TOKEN is empty." >&2
  echo "Either export it like:" >&2
  echo "  export TOKEN='YOUR_TOKEN_HERE'" >&2
  exit 1
fi

# Remove accidental newlines from copied tokens
TOKEN="$(printf "%s" "${TOKEN}" | tr -d '\r\n')"

check_aria2c
need_cmd wget
need_cmd split
need_cmd python3
need_cmd wc
need_cmd head
need_cmd awk

echo "==> Creating and entering directory: ${TARGET_DIR}"
mkdir -p "${TARGET_DIR}"
cd "${TARGET_DIR}"

MANIFEST_URL="${BASE_URL}${REMOTE_DATASET_PATH}/manifest.txt"

#########################
# DOWNLOAD MANIFEST
#########################

echo "==> Downloading manifest from: ${MANIFEST_URL}"
wget --header="Authorization: Bearer ${TOKEN}" \
  "${MANIFEST_URL}" -O manifest.txt

#########################
# SELECT / PREPARE FINAL MANIFEST
#########################

FINAL_MANIFEST="manifest.final.txt"
rm -f "${FINAL_MANIFEST}" || true

if [ "${DATASET}" = "howto100m" ]; then
  echo "==> DATASET=howto100m: using full manifest (no filtering)."
  cp manifest.txt "${FINAL_MANIFEST}"

else
  #########################
  # EXTRACT UNIQUE VIDEO IDS FROM PKL(S)
  #########################

  echo "==> Extracting unique video_ids from PKL file(s)..."
  rm -f video_ids.txt manifest.filtered.txt || true

  python3 - "${PKL_FILES[@]}" <<'PY' > video_ids.txt
import pickle
import sys
import re
from pathlib import Path

pkl_files = sys.argv[1:]
video_ids = set()

suffix_re = re.compile(r"^(.*)_\d+$")  # strip trailing _0, _1, _23, ...

def normalize_id(k: str) -> str:
    k = str(k)
    m = suffix_re.match(k)
    if m:
        return m.group(1)
    return k

for p in pkl_files:
    path = Path(p)
    if not path.exists():
        raise FileNotFoundError(f"PKL not found: {p}")

    with path.open("rb") as f:
        obj = pickle.load(f)

    if not isinstance(obj, dict):
        raise TypeError(f"{p}: expected dict, got {type(obj)}")

    # Format A: {"annotations": {...}}
    if "annotations" in obj and isinstance(obj["annotations"], dict):
        keys = obj["annotations"].keys()
    else:
        # Format B: dict itself is keyed by ids (like VIDEOID_0)
        keys = obj.keys()

    for k in keys:
        if k is None:
            continue
        video_ids.add(normalize_id(k))

for vid in sorted(video_ids):
    print(vid)
PY

  TOTAL_IDS=$(wc -l < video_ids.txt || echo 0)
  echo "==> Found ${TOTAL_IDS} unique video_ids (after stripping _<digits>)"

  if [ "${TOTAL_IDS}" -eq 0 ]; then
    echo "ERROR: No video_ids found in the provided PKL files." >&2
    exit 1
  fi

  #########################
  # FILTER MANIFEST (FAST O(N), ROBUST)
  #########################

  echo "==> Filtering manifest.txt using extracted video_ids (fast O(N), robust)..."
  rm -f manifest.filtered.txt || true

  python3 - <<'PY'
import sys
import re

# Load ids into a hash set
ids = set()
with open("video_ids.txt", "r", encoding="utf-8") as f:
    for line in f:
        v = line.strip()
        if v:
            ids.add(v)

# Robust: search for YouTube-style IDs anywhere in the line
yt_re = re.compile(r"(?<![A-Za-z0-9_-])([A-Za-z0-9_-]{11})(?![A-Za-z0-9_-])")

kept = 0
total = 0

with open("manifest.txt", "r", encoding="utf-8", errors="ignore") as f_in, \
     open("manifest.filtered.txt", "w", encoding="utf-8") as f_out:
    for line in f_in:
        total += 1
        cands = yt_re.findall(line)
        if not cands:
            continue
        if any(c in ids for c in cands):
            f_out.write(line)
            kept += 1

print(f"==> Done. Kept {kept} / {total} lines.", file=sys.stderr)
PY

  FILTERED_COUNT=$(wc -l < manifest.filtered.txt || echo 0)
  echo "==> Matching manifest lines:     ${FILTERED_COUNT}"

  if [ "${FILTERED_COUNT}" -eq 0 ]; then
    echo "ERROR: Filtering produced 0 lines." >&2
    echo
    echo "Debug tips:" >&2
    echo "  head -n 5 video_ids.txt" >&2
    echo "  head -n 5 manifest.txt" >&2
    exit 1
  fi

  cp manifest.filtered.txt "${FINAL_MANIFEST}"
fi

#########################
# ENSURE FINAL MANIFEST HAS ABSOLUTE URLS
#########################

if head -n 1 "${FINAL_MANIFEST}" | grep -q '^/'; then
  echo "==> Final manifest lines look like relative paths. Converting to absolute URLs..."
  awk -v base="${BASE_URL}" '{print base $0}' "${FINAL_MANIFEST}" > "${FINAL_MANIFEST}.abs"
  mv "${FINAL_MANIFEST}.abs" "${FINAL_MANIFEST}"
fi

#########################
# COUNT UNIQUE VIDEOS (FOR DISPLAY)
#########################

VIDEOS_TO_DOWNLOAD=$(
  python3 - <<'PY'
import re

yt_re = re.compile(r"(?<![A-Za-z0-9_-])([A-Za-z0-9_-]{11})(?![A-Za-z0-9_-])")
matched = set()

with open("manifest.final.txt", "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        for c in yt_re.findall(line):
            matched.add(c)

print(len(matched))
PY
)

echo
echo "============================================================"
echo "==> ${VIDEOS_TO_DOWNLOAD} videos are going to be downloaded."
echo "==> Press SPACE if you wish to proceed."
echo "============================================================"
echo

KEY=""
IFS= read -r -n 1 -s KEY
if [[ "${KEY}" != " " ]]; then
  echo "Aborted. (You did not press SPACE.)"
  exit 1
fi

echo "Proceeding with download..."

#########################
# SPLIT FINAL MANIFEST
#########################

echo "==> Splitting final manifest into parts of ${LINES_PER_PART} lines..."
rm -f part_* || true
split -l "${LINES_PER_PART}" "${FINAL_MANIFEST}" part_

#########################
# DOWNLOAD WITH ARIA2C
#########################

echo "==> Starting parallel downloads with aria2c..."
for f in part_*; do
  echo "--> Processing ${f}"
  aria2c --header="Authorization: Bearer ${TOKEN}" \
    -i "${f}" \
    --continue=true \
    --max-concurrent-downloads="${MAX_CONCURRENT_DOWNLOADS}" \
    -x1 -s1 \
    --min-split-size=20M \
    --file-allocation=none \
    --retry-wait=5 \
    --max-tries=0
done

echo "==> All aria2c jobs finished."
echo "==> Files are in: ${TARGET_DIR}"
echo "==> Used final manifest: ${TARGET_DIR}/${FINAL_MANIFEST}"