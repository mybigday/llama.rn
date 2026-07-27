#!/usr/bin/env bash
#
# Run the KV-cache-reuse harness on every connected Android device, on the GPU
# (OpenCL / Adreno). Picks, per device, only the models that fit its RAM, pushes
# them one at a time (cleaning up after each to save storage), and collects a
# pass/fail matrix.
#
# Prereqs:
#   ./scripts/build-opencl.sh                          # builds bin/arm64-v8a/libOpenCL.so
#   ./tests/models/download.sh all                     # models + images (host side)
#   cmake -S tests/android -B tests/android/build \
#     -DCMAKE_TOOLCHAIN_FILE=$ANDROID_HOME/ndk/<ver>/build/cmake/android.toolchain.cmake \
#     -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-28 -DENABLE_OPENCL=ON
#   cmake --build tests/android/build -j
#
# Usage:
#   ./tests/android/run_on_devices.sh                  # all devices, GPU
#   RNLLAMA_NGL=0 ./tests/android/run_on_devices.sh    # CPU baseline
#   ./tests/android/run_on_devices.sh <serial> [...]   # specific devices

set -uo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BIN="$ROOT/tests/android/build/kv_cache_reuse_test"
LIBCL="$ROOT/bin/arm64-v8a/libOpenCL.so"
MODELS="$ROOT/tests/models"
DEV_DIR="/data/local/tmp/rnllama_test"
NGL="${RNLLAMA_NGL:-99}"          # 99 = all layers on GPU; 0 = CPU baseline

[ -x "$BIN" ] || { echo "missing $BIN — build tests/android first"; exit 1; }

# model_key : min device RAM (GiB) : extra files (mmproj, space separated)
MODELS_SPEC=(
  "smollm2:2:"
  "mamba:2:"
  "lfm2:2:"
  "granite4:3:"
  "lfm2vl:4:mmproj-LFM2.5-VL-450m-Q8_0.gguf"
  "qwen35:6:"
  "gemma4:6:mmproj-google_gemma-4-E2B-it-f16.gguf"
)

devices=("$@")
if [ ${#devices[@]} -eq 0 ]; then
  mapfile -t devices < <(adb devices | awk 'NR>1 && $2=="device"{print $1}')
fi
[ ${#devices[@]} -gt 0 ] || { echo "no devices connected"; exit 1; }

echo "Devices: ${devices[*]}"
echo "GPU offload: RNLLAMA_NGL=$NGL"
declare -A RESULT

for dev in "${devices[@]}"; do
  model_name=$(adb -s "$dev" shell getprop ro.product.model 2>/dev/null | tr -d '\r')
  ram_kb=$(adb -s "$dev" shell cat /proc/meminfo 2>/dev/null | awk '/MemTotal/{print $2}')
  ram_gb=$(( ram_kb / 1024 / 1024 ))
  soc=$(adb -s "$dev" shell getprop ro.soc.model 2>/dev/null | tr -d '\r')
  echo ""
  echo "================================================================"
  echo "Device $dev  ($model_name, ${ram_gb}GiB RAM, SoC=$soc)"
  echo "================================================================"

  adb -s "$dev" shell "mkdir -p $DEV_DIR" >/dev/null 2>&1
  adb -s "$dev" push "$BIN" "$DEV_DIR/" >/dev/null
  adb -s "$dev" shell "chmod +x $DEV_DIR/kv_cache_reuse_test" >/dev/null 2>&1
  # We use the device's own OpenCL driver (/vendor/lib64/libOpenCL.so on Adreno);
  # the bin/ stub is only needed at link time. Non-Adreno GPUs (Mali/Tensor) may
  # not run the Adreno kernels and will error under GPU — reported as such.
  # test images (small)
  for img in test_dog.jpg test_cat.jpg; do
    [ -f "$MODELS/$img" ] && adb -s "$dev" push "$MODELS/$img" "$DEV_DIR/" >/dev/null 2>&1
  done

  for spec in "${MODELS_SPEC[@]}"; do
    key="${spec%%:*}"; rest="${spec#*:}"; min_ram="${rest%%:*}"; extras="${rest#*:}"
    gguf="$MODELS/${key}.gguf"
    [ -f "$gguf" ] || { echo "  [skip] $key (not downloaded)"; continue; }
    if [ "$ram_gb" -lt "$min_ram" ]; then
      echo "  [skip] $key (needs ${min_ram}GiB, device has ${ram_gb}GiB)"
      RESULT["$dev/$key"]="skip(ram)"
      continue
    fi
    echo "  [push] $key ..."
    adb -s "$dev" push "$gguf" "$DEV_DIR/${key}.gguf" >/dev/null
    for ef in $extras; do
      [ -n "$ef" ] && [ -f "$MODELS/${key}.${ef}" ] && \
        adb -s "$dev" push "$MODELS/${key}.${ef}" "$DEV_DIR/${key}.${ef}" >/dev/null
    done

    echo "  [run ] $key (NGL=$NGL) ..."
    # Keep stderr (2>&1): llama.cpp logs its per-backend weight allocation there,
    # which is how we confirm the model actually landed on the GPU (below).
    # Vendor driver first so real GPUs (Adreno) resolve their own libOpenCL.
    out=$(adb -s "$dev" shell \
      "cd $DEV_DIR && LD_LIBRARY_PATH=/vendor/lib64:/system/vendor/lib64:$DEV_DIR MODELS_DIR=$DEV_DIR RNLLAMA_NGL=$NGL SKIP_MTP=1 timeout 600 ./kv_cache_reuse_test $key 2>&1" \
      | tr -d '\r')
    # Old OpenCL-2.0 vendor drivers (e.g. OnePlus 6 / Adreno 630) lack the 3.0
    # symbols the binary links (clCreateBufferWithProperties) and the loader
    # refuses to start. Retry with our ICD stub first — the run then proceeds
    # (on CPU; the GPU check below still reports honestly).
    if echo "$out" | grep -aq "CANNOT LINK EXECUTABLE"; then
      echo "  [retry] vendor libOpenCL too old, using bundled stub (CPU expected)"
      adb -s "$dev" push "$LIBCL" "$DEV_DIR/" >/dev/null 2>&1
      out=$(adb -s "$dev" shell \
        "cd $DEV_DIR && LD_LIBRARY_PATH=$DEV_DIR:/vendor/lib64:/system/vendor/lib64 MODELS_DIR=$DEV_DIR RNLLAMA_NGL=$NGL SKIP_MTP=1 timeout 600 ./kv_cache_reuse_test $key 2>&1" \
        | tr -d '\r')
    fi
    summary=$(echo "$out" | grep -aE "checks passed" | tail -1)
    fails=$(echo "$out" | grep -acE "^  FAIL")

    # Confirm GPU offload from the resident-weight allocation, not the NGL request.
    # At load, llama.cpp prints one line per backend buffer, e.g.:
    #   load_tensors:      OpenCL model buffer size =  2145.00 MiB   (on GPU)
    #   load_tensors:         CPU model buffer size =   112.50 MiB   (on CPU)
    # The backend only accepts Adreno/Qualcomm and Intel devices (Mali/Tensor
    # are dropped as unsupported and run on CPU), so a CPU-only run shows ZERO
    # OpenCL lines and is reported honestly as CPU-FALLBACK.
    # Take the largest single allocation of each (a full model load).
    cl_mib=$(echo "$out"  | grep -aE "OpenCL model buffer size" | grep -oE "[0-9]+\.[0-9]+" | sort -gr | head -1)
    cpu_mib=$(echo "$out" | grep -aE "CPU[_A-Za-z]* model buffer size" | grep -oE "[0-9]+\.[0-9]+" | sort -gr | head -1)
    cl_mib=${cl_mib:-0}; cpu_mib=${cpu_mib:-0}
    if [ "$NGL" -gt 0 ]; then
      if awk "BEGIN{exit !($cl_mib > 0 && $cl_mib >= $cpu_mib)}"; then
        backend="GPU(OpenCL ${cl_mib} >= CPU ${cpu_mib} MiB)"
      elif awk "BEGIN{exit !($cl_mib > 0)}"; then
        backend="GPU-partial(OpenCL ${cl_mib}, CPU ${cpu_mib} MiB)"
      else
        backend="!! CPU-FALLBACK (no OpenCL weights; NGL=$NGL had no effect)"
      fi
    else
      backend="CPU (NGL=0 baseline; OpenCL ${cl_mib} MiB)"
    fi
    echo "      $summary   (FAIL lines: $fails)  [$backend]"
    RESULT["$dev/$key"]="${summary:-no-output} fails=$fails | $backend"

    # free device storage
    adb -s "$dev" shell "rm -f $DEV_DIR/${key}.gguf $DEV_DIR/${key}.*.gguf" >/dev/null 2>&1
  done
done

echo ""
echo "==================== MATRIX ===================="
for dev in "${devices[@]}"; do
  echo "--- $dev ---"
  for spec in "${MODELS_SPEC[@]}"; do
    key="${spec%%:*}"
    printf "  %-9s %s\n" "$key" "${RESULT[$dev/$key]:-not-run}"
  done
done
