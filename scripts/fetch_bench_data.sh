#!/usr/bin/env bash
# Download the plate-solve bench frames and the density-tier star catalog.
#
#   scripts/fetch_bench_data.sh <dest>
#
# Leaves <dest>/bench/*.fits|*.fit|*.xisf and <dest>/catalogs/tier_*/stars.smac,
# ready for:
#   SOLVEMYASTRO_BENCH_DIR=<dest>/bench SOLVEMYASTRO_TIER_DIR=<dest>/catalogs \
#     cargo test --release --test corpus_bench -- --ignored corpus_layered_tiers --nocapture
#
# Re-running skips whatever is already in place. About 1.5 GB of frames and
# 7 GB of catalog are downloaded; unpacked they take about 9.5 GB.
set -euo pipefail

BENCH_URL="${SOLVEMYASTRO_BENCH_URL:-https://artfrom.space/builds/platesolve_bench/platesolve_bench.zip}"
CATALOG_URL="${SOLVEMYASTRO_CATALOG_URL:-https://artfrom.space/catalogs/}"
TIERS=(500 2000 5000 8000)

dest="${1:?usage: $0 <dest>}"
mkdir -p "$dest/bench" "$dest/catalogs"

if [ -z "$(ls -A "$dest/bench")" ]; then
  echo "bench frames <- $BENCH_URL"
  curl -fSL -C - --retry 4 -o "$dest/bench.zip" "$BENCH_URL"
  unzip -q -o "$dest/bench.zip" -x '__MACOSX/*' -d "$dest/bench"
  rm "$dest/bench.zip"
fi
echo "bench frames: $(ls "$dest/bench" | wc -l) files in $dest/bench"

for t in "${TIERS[@]}"; do
  dir="$dest/catalogs/tier_$t"
  if [ -f "$dir/stars.smac" ]; then
    echo "tier_$t: present"
    continue
  fi
  zip="$dest/catalogs/tier_$t.zip"
  echo "tier_$t <- ${CATALOG_URL}tier_$t.zip"
  curl -fSL -C - --retry 4 -o "$zip" "${CATALOG_URL}tier_$t.zip"
  want="$(curl -fsSL "${CATALOG_URL}tier_$t.zip.sha256" | awk '{print $1}')"
  got="$(sha256sum "$zip" | awk '{print $1}')"
  if [ "$want" != "$got" ]; then
    echo "tier_$t: SHA-256 mismatch (want $want, got $got)" >&2
    exit 1
  fi
  tmp="$dest/catalogs/.x_$t"
  rm -rf "$tmp" && mkdir -p "$tmp" "$dir"
  unzip -q "$zip" -d "$tmp"
  mv "$(find "$tmp" -name stars.smac | head -1)" "$dir/stars.smac"
  rm -rf "$tmp" "$zip"
  echo "tier_$t: ok"
done
