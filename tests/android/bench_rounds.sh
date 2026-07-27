#!/usr/bin/env bash
# Multi-round A/B benchmark for ONE device: N rounds of {base, new}, order
# rotated per round, cooldown between every run — so per-round distributions
# are measurable and thermal drift can't masquerade as a build effect.
# CSV rows: DEV,<device>,<build>,r<round>,BENCH,<model>,...
#
# Usage: BASE_BIN=<path> ./tests/android/bench_rounds.sh <serial> <model> [...]
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
NEW_BIN="$ROOT/tests/android/build/kv_cache_bench"
BASE_BIN="${BASE_BIN:?set BASE_BIN to the merge-base kv_cache_bench binary}"
MODELS_DIR="$ROOT/tests/models"
DEV_DIR="/data/local/tmp/rnllama_bench"
NGL="${RNLLAMA_NGL:-99}"
ROUNDS="${BENCH_ROUNDS:-4}"
COOLDOWN="${BENCH_COOLDOWN_S:-45}"

dev=$1; shift
models=("$@")

adb -s "$dev" shell "mkdir -p $DEV_DIR" >/dev/null 2>&1
adb -s "$dev" push "$NEW_BIN"  "$DEV_DIR/bench_new"  >/dev/null
adb -s "$dev" push "$BASE_BIN" "$DEV_DIR/bench_base" >/dev/null
adb -s "$dev" shell "chmod +x $DEV_DIR/bench_new $DEV_DIR/bench_base" >/dev/null 2>&1
adb -s "$dev" push "$ROOT/bin/arm64-v8a/libOpenCL.so" "$DEV_DIR/" >/dev/null 2>&1

run_one() { # bin_name model round_tag
  local bin=$1 key=$2 tag=$3 out
  out=$(adb -s "$dev" shell \
    "cd $DEV_DIR && LD_LIBRARY_PATH=/vendor/lib64:/system/vendor/lib64:$DEV_DIR MODELS_DIR=$DEV_DIR RNLLAMA_NGL=$NGL timeout 1200 ./$bin $key 2>/dev/null" \
    | tr -d '\r' | grep -a "^BENCH,")
  if [ -z "$out" ]; then
    out=$(adb -s "$dev" shell \
      "cd $DEV_DIR && LD_LIBRARY_PATH=$DEV_DIR:/vendor/lib64:/system/vendor/lib64 MODELS_DIR=$DEV_DIR RNLLAMA_NGL=$NGL timeout 1200 ./$bin $key 2>/dev/null" \
      | tr -d '\r' | grep -a "^BENCH,")
  fi
  while IFS= read -r line; do
    [ -n "$line" ] && echo "DEV,$dev,${bin#bench_},$tag,$line"
  done <<< "$out"
}

for key in "${models[@]}"; do
  gguf="$MODELS_DIR/${key}.gguf"
  [ -f "$gguf" ] || continue
  echo "  [$dev] push $key" >&2
  adb -s "$dev" push "$gguf" "$DEV_DIR/${key}.gguf" >/dev/null
  for ((r = 1; r <= ROUNDS; r++)); do
    if (( r % 2 == 1 )); then order=(base new); else order=(new base); fi
    for b in "${order[@]}"; do
      run_one "bench_$b" "$key" "r$r"
      sleep "$COOLDOWN"
    done
  done
  adb -s "$dev" shell "rm -f $DEV_DIR/${key}.gguf" >/dev/null 2>&1
done
echo "  [$dev] done" >&2
