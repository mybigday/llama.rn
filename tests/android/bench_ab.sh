#!/usr/bin/env bash
# A/B/C benchmark across connected devices: baseline (merge-base) vs branch vs
# branch-with-cache-off. Prints tagged CSV: DEV,<device>,<build>,<BENCH line>.
#
# Usage: BASE_BIN=<path> ./tests/android/bench_ab.sh [serial ...]
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
NEW_BIN="$ROOT/tests/android/build/kv_cache_bench"
BASE_BIN="${BASE_BIN:?set BASE_BIN to the merge-base kv_cache_bench binary}"
MODELS="$ROOT/tests/models"
DEV_DIR="/data/local/tmp/rnllama_bench"
NGL="${RNLLAMA_NGL:-99}"
TURNS="${BENCH_TURNS:-8}"
GEN="${BENCH_GEN:-32}"
COOLDOWN="${BENCH_COOLDOWN_S:-20}"

[ -x "$NEW_BIN" ] && [ -x "$BASE_BIN" ] || { echo "missing bench binaries"; exit 1; }

# model : min device RAM GiB
SPEC=( "smollm2:2" "lfm2:2" "granite4:3" "qwen35:6" "gemma4:6" )

devices=("$@")
if [ ${#devices[@]} -eq 0 ]; then
  mapfile -t devices < <(adb devices | awk 'NR>1 && $2=="device"{print $1}')
fi

run_one() { # dev bin_name model extra_env
  local dev=$1 bin=$2 key=$3 extra=$4
  adb -s "$dev" shell \
    "cd $DEV_DIR && LD_LIBRARY_PATH=/vendor/lib64:/system/vendor/lib64:$DEV_DIR MODELS_DIR=$DEV_DIR RNLLAMA_NGL=$NGL BENCH_TURNS=$TURNS BENCH_GEN=$GEN $extra timeout 900 ./$bin $key 2>/dev/null" \
    | tr -d '\r' | grep -a "^BENCH,"
}

for dev in "${devices[@]}"; do
  name=$(adb -s "$dev" shell getprop ro.product.model 2>/dev/null | tr -d '\r')
  ram=$(adb -s "$dev" shell cat /proc/meminfo 2>/dev/null | awk '/MemTotal/{printf "%d", $2/1024/1024}')
  echo "### $dev ($name, ${ram}GiB) NGL=$NGL turns=$TURNS gen=$GEN" >&2
  adb -s "$dev" shell "mkdir -p $DEV_DIR" >/dev/null 2>&1
  adb -s "$dev" push "$NEW_BIN"  "$DEV_DIR/bench_new"  >/dev/null
  adb -s "$dev" push "$BASE_BIN" "$DEV_DIR/bench_base" >/dev/null
  adb -s "$dev" shell "chmod +x $DEV_DIR/bench_new $DEV_DIR/bench_base" >/dev/null 2>&1
  # Old vendor OpenCL (OnePlus 6) can't link the binary; stage the ICD stub too.
  adb -s "$dev" push "$ROOT/bin/arm64-v8a/libOpenCL.so" "$DEV_DIR/" >/dev/null 2>&1

  for spec in "${SPEC[@]}"; do
    key="${spec%%:*}"; minram="${spec#*:}"
    gguf="$MODELS/${key}.gguf"
    [ -f "$gguf" ] || continue
    [ "$ram" -ge "$minram" ] || continue
    echo "  [push] $key" >&2
    adb -s "$dev" push "$gguf" "$DEV_DIR/${key}.gguf" >/dev/null

    for build in base new new-off; do
      case $build in
        base)    bin=bench_base; extra="" ;;
        new)     bin=bench_new;  extra="" ;;
        new-off) bin=bench_new;  extra="BENCH_BUDGET_MB=0" ;;
      esac
      out=$(run_one "$dev" "$bin" "$key" "$extra")
      if [ -z "$out" ]; then
        # linker failure on old vendor drivers -> retry with the stub first
        out=$(adb -s "$dev" shell \
          "cd $DEV_DIR && LD_LIBRARY_PATH=$DEV_DIR:/vendor/lib64:/system/vendor/lib64 MODELS_DIR=$DEV_DIR RNLLAMA_NGL=$NGL BENCH_TURNS=$TURNS BENCH_GEN=$GEN $extra timeout 900 ./$bin $key 2>/dev/null" \
          | tr -d '\r' | grep -a "^BENCH,")
      fi
      while IFS= read -r line; do
        [ -n "$line" ] && echo "DEV,$dev,$build,$line"
      done <<< "$out"
      sleep "$COOLDOWN"
    done
    adb -s "$dev" shell "rm -f $DEV_DIR/${key}.gguf" >/dev/null 2>&1
  done
done
