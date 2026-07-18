# C++ Tests for llama.rn

This directory contains integration tests for the llama.rn C++ implementation.

## Building and Running

### Quick Start (Recommended)

```bash
cd tests

# Build both test executables
./build_and_test.sh

# Run all tests
./run_tests.sh
```

### Using CMake Directly

```bash
cd tests
mkdir -p build
cd build

# Configure
cmake ..

# Build
make

# Run basic tests
./rnllama_tests

# Run parallel decoding tests
./parallel_decoding_test

# Run chat parse UTF-8 robustness tests (no model needed)
./chat_parse_utf8_test

# Run all
./rnllama_tests && ./parallel_decoding_test && ./chat_parse_utf8_test
```

### Build Scripts

**`build_and_test.sh`**
- Builds `rnllama_tests`, `parallel_decoding_test` and `chat_parse_utf8_test`
- Uses CMake with Release configuration
- Parallel compilation with `-j4`

**`run_tests.sh`**
- Runs all test suites
- Reports pass/fail for each suite
- Returns appropriate exit code for CI/CD

### Requirements

- CMake 3.16 or higher
- C++17 compatible compiler
- Test model: `tiny-random-llama.gguf` in tests directory

### Test Model

The tests use a tiny random LLAMA model (`tiny-random-llama.gguf`) for quick testing without requiring large model downloads.

## KV-cache-reuse tests (`kv_cache_reuse_test`)

These exercise multi-turn prompt-cache reuse across memory architectures
(pure-attention, recurrent, and hybrid) through the `rn-completion` API. Unlike
the basic tests they need real GGUF models, so they are opt-in.

```bash
cd tests

# 1. Download the models (small "core" set: smollm2, mamba, lfm2, granite4)
./models/download.sh
# ...or everything, including the 2B qwen35 and the gemma4 vision model:
./models/download.sh all

# 2. Build and run
./build_and_test.sh                 # builds kv_cache_reuse_test too
./build/kv_cache_reuse_test         # runs every downloaded model
./build/kv_cache_reuse_test qwen35  # or a specific one
```

Each model is driven through several scenarios that check both **reuse** (how
many prompt tokens were reused vs. reprocessed per turn) and **correctness**:

- **append-only** and **resend-last** — reuse mechanics.
- **edit mid-conversation** — replace an earlier user turn ("my name is Ali" →
  "my hobby is cycling") and continue; the model must recall the new fact and
  must NOT surface the edited-out one.
- **new session without clearing the cache** — reuses the shared system prefix
  but must not recall the previous session's facts.
- **vision** (gemma4, lfm2vl + dog/cat images) — a dog image is called a dog,
  swapping to a cat is called a cat with no dog leak, removing the image surfaces
  neither; the image is reused (not re-encoded) across turns.
- **MTP** (qwen35) and a **config** check (`state_cache_budget_mb=0` disables).

Answer-correctness is asserted on the capable 2 B models (qwen35, gemma4) and the
vision models; no-leak invariants on every instruct model. Env knobs:
`SKIP_MTP`, `SKIP_VISION`, `MTP_ONLY`, `VISION_ONLY`, `RNLLAMA_NGL` (GPU offload).

`run_tests.sh` runs this suite automatically when `tests/models/*.gguf` exists.

### On-device (Android GPU)

To verify the checkpoint reuse on a real GPU (OpenCL / Adreno), build the harness
for arm64 and run it across connected devices — see `tests/android/`:

```bash
./scripts/build-opencl.sh                       # bin/arm64-v8a/libOpenCL.so
./tests/models/download.sh all                  # models + images
NDK=$ANDROID_HOME/ndk/27.3.13750724
cmake -S tests/android -B tests/android/build \
  -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-28 -DENABLE_OPENCL=ON
cmake --build tests/android/build -j

./tests/android/run_on_devices.sh               # GPU, all devices, RAM-fitted
RNLLAMA_NGL=0 ./tests/android/run_on_devices.sh # CPU baseline
```

It reads each device's RAM, pushes only the models that fit, runs on the GPU
(using the device's own `/vendor/lib64` OpenCL driver), and prints a pass/fail
matrix.
