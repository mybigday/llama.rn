# KV-cache reuse — Android device benchmark harness

Perf A/B harness for the prompt state-cache feature. This is **not** a unit test —
the correctness suite is `tests/kv_cache_reuse_test.cpp`. This measures per-turn
time-to-first-token and RSS, baseline vs branch, on a real device.

Device-generic: the serial is a script argument and the A/B just needs a
merge-base binary — nothing local is hardcoded, so it runs on any device.

## Build

```sh
NDK=$ANDROID_HOME/ndk/<version>
cmake -S tests/android -B tests/android/build \
  -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-28 -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_OPENCL=ON                          # Adreno GPU
# or: -DENABLE_HEXAGON=ON -DENABLE_OPENCL=OFF  # Hexagon NPU (needs the Hexagon SDK
#                                                and bin/arm64-v8a/libggml-htp-v*.so)
cmake --build tests/android/build --target kv_cache_bench -j
```

## Run — single device

```sh
MODELS_DIR=<dir with <model>.gguf> RNLLAMA_NGL=99 ./kv_cache_bench <model> [<model> ...]
```
Emits CSV rows: `BENCH,<model>,<phase>,<prompt_tokens>,<reused>,<ttft_ms>,<gen_tps>,<rss_mb>,<hwm_mb>`.
Phases: `append-t1..N`, `regenerate`, `new-session`.

Env: `RNLLAMA_NGL` (99 = offload all layers to GPU/NPU, 0 = CPU), `BENCH_BUDGET_MB=0`
disables the cache (baseline arm), `BENCH_TURNS` (default 8), `BENCH_GEN` (default 32).

## Run — A/B sweep across a device

```sh
BASE_BIN=<merge-base kv_cache_bench> BENCH_ROUNDS=3 RNLLAMA_NGL=99 \
  ./bench_rounds.sh <adb-serial> <model> [<model> ...]
```
Builds base-vs-HEAD, N rounds, order-rotated, cooldown between runs; prepends
`DEV,<serial>,<build>,r<round>,` to each CSV row.

**Hexagon NPU**: push `bin/arm64-v8a/libggml-htp-v<arch>.so` alongside the binary,
set `ADSP_LIBRARY_PATH` to its dir, and keep the device's `/vendor/lib64/libcdsprpc.so`
ahead of the SDK stub in `LD_LIBRARY_PATH`.

## Model-free R4 attention regression

`opencl_r4_attention_test` compares identical synthetic f16 keys and f32 queries
on CPU and OpenCL. It does not load a model or use a scheduler that could silently
fall back to CPU. It covers the Qwen3-VL 4B decode shape, the R4 admission boundary,
larger/non-power-of-two row counts and two shapes outside the specialization.
Nonfinite output or normalized squared error above `1e-5` fails the test.

Configure as above with `ENABLE_OPENCL=ON` and `ENABLE_HEXAGON=OFF`, then build
only this target (no Hexagon SDK is required):

```sh
free -h
cmake --build tests/android/build --target opencl_r4_attention_test -j1
adb push tests/android/build/opencl_r4_attention_test /data/local/tmp/
adb shell 'LD_LIBRARY_PATH=/system/lib64:/vendor/lib64 /data/local/tmp/opencl_r4_attention_test'
```

If the NDK configuration uses the shared C++ runtime, also stage its matching
`libc++_shared.so` beside the executable and prepend `/data/local/tmp` to the
library path. Use the phone's real OpenCL driver, never the link-time stub.
Specify `adb -s SERIAL` when multiple devices are connected. Run on a device with
an available OpenCL backend; a missing backend is a failure, not a skipped pass.
Leave `LM_GGML_OPENCL_MM_KQ_GQA_R4_IMG` unset to test the default selection.

On Adreno 730 / E031.38.11.14, the unmodified 0.13.0-rc.2 backend fails the
first case at NMSE `0.509330598492`. The compatibility guard routes the affected
driver series to the existing OpenCL fallback; it is not a switch to CPU and
does not establish the internal cause of the faulty R4 kernel.
The standalone target built from the patched source passes all six cases on
that device, with NMSE between `3.04e-8` and `3.60e-8`. Other driver families
are not covered by this device result.

The matching patch in `scripts/patches/ggml-opencl-r4-attention.cpp.patch` is
automatically applied by the existing patch loop in `scripts/bootstrap.sh`,
preserving the guard across llama.cpp source synchronization.
