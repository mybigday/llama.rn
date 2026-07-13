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
