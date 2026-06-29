#!/usr/bin/env bash
# Materialize the flat dataset layout the harness expects, on top of the
# in-repo per-instance layout under data/<instance_id>/.
#
# This script is idempotent — re-run any time. It only creates symlinks;
# nothing under data/<instance_id>/ is modified.
#
# Created flat helpers (siblings of the per-instance dirs, all gitignored):
#   data/input/<stem>.json       -> ../<instance_id>/input.json
#   data/prompts/<stem>.txt      -> ../<instance_id>/prompt.txt
#   data/testbench/<stem>/       -> ../<instance_id>/testbench
#   data/references/<stem>.{cu,py} -> ../<instance_id>/reference.{cu,py}
#
# Per-testbench reference_sources/ symlinks (Class 1 cutlass + thunderkittens
# build dependencies) point at the bundled CUTLASS / ThunderKittens headers
# under data/_external/ via absolute paths (so testbench can be copied to
# arbitrary workdirs without breaking).
#
# Usage:
#   bash scripts/prepare_benchmark.sh           # operates on this repo's data/
set -euo pipefail

CODE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$CODE_ROOT/data"

[ -f "$CODE_ROOT/manifest.json" ] || {
  echo "ERROR: $CODE_ROOT/manifest.json not found." >&2
  echo "       This script must be run from within the cuda-debugger-bench code repo." >&2
  exit 1
}
[ -d "$DATA_DIR" ] || {
  echo "ERROR: $DATA_DIR/ not found — dataset is missing from this repo." >&2
  exit 1
}

INPUT_DIR="$DATA_DIR/input"
PROMPTS_DIR="$DATA_DIR/prompts"
TESTBENCH_DIR="$DATA_DIR/testbench"
REFERENCES_DIR="$DATA_DIR/references"

mkdir -p "$INPUT_DIR" "$PROMPTS_DIR" "$TESTBENCH_DIR" "$REFERENCES_DIR"

n=0
for inst in "$DATA_DIR"/*/; do
  inst_id="$(basename "$inst")"
  case "$inst_id" in
    _external|input|prompts|testbench|references) continue ;;
  esac
  stem="${inst_id%%__*}"

  ln -sf "../$inst_id/input.json" "$INPUT_DIR/$stem.json"

  if [ -f "$inst/prompt.txt" ]; then
    ln -sf "../$inst_id/prompt.txt" "$PROMPTS_DIR/$stem.txt"
  fi

  if [ -d "$inst/testbench" ]; then
    ln -sfn "../$inst_id/testbench" "$TESTBENCH_DIR/$stem"
  fi

  for ref in "$inst"/reference.*; do
    [ -f "$ref" ] || continue
    ext="${ref##*.}"
    ln -sf "../$inst_id/reference.$ext" "$REFERENCES_DIR/$stem.$ext"
  done

  # External library headers (CUTLASS / ThunderKittens) for Class 1 Makefile-
  # driven testbenches. Use absolute symlink targets so testbench can be
  # copied to a workdir at any depth without the relative target breaking.
  if [ -d "$DATA_DIR/_external" ] && [ -d "$inst/testbench" ]; then
    mkdir -p "$inst/testbench/reference_sources"
    for ext_lib in cutlass ThunderKittens; do
      target="$DATA_DIR/_external/$ext_lib"
      link="$inst/testbench/reference_sources/$ext_lib"
      if [ -d "$target" ]; then
        rm -f "$link"
        ln -s "$target" "$link"
      fi
    done
  fi

  n=$((n+1))
done

echo "Created flat layout for $n instances under:"
echo "  $INPUT_DIR/"
echo "  $PROMPTS_DIR/"
echo "  $TESTBENCH_DIR/"
echo "  $REFERENCES_DIR/"
echo
echo "Note: collisions on stem (e.g., cuda_123 appearing in both v2 and v3) will overwrite -- only one link per stem."
echo "$n instances mapped to ~$(ls "$INPUT_DIR" | wc -l) unique stems."
