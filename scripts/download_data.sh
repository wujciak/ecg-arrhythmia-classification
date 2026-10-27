#!/usr/bin/env bash
# Downloads the MIT-BIH Arrhythmia Database (PhysioNet, ODC-By 1.0) into data/mitdb/.
# Uses PhysioNet's AWS Open Data mirror, which is much faster than physionet.org,
# and checks every file against PhysioNet's published SHA-256 sums.
set -euo pipefail

BASE=https://physionet-open.s3.amazonaws.com/mitdb/1.0.0
OUT=data/mitdb
mkdir -p "$OUT"
cd "$OUT"

curl -fsSL -o SHA256SUMS.txt "$BASE/SHA256SUMS.txt"
curl -fsSL -o RECORDS "$BASE/RECORDS"
while read -r rec; do
  for ext in hea dat atr; do
    [ -f "$rec.$ext" ] || curl -fsSL -o "$rec.$ext" "$BASE/$rec.$ext"
  done
done < RECORDS

grep -E ' [0-9]+\.(hea|dat|atr)$' SHA256SUMS.txt | sha256sum --check --quiet
echo "MIT-BIH: $(wc -l < RECORDS) records in $OUT, checksums OK"
