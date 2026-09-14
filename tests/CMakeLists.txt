cmake_minimum_required(VERSION 3.16)
project(rnllama_tests VERSION 1.0.0 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Dependencies and compile options
add_definitions(
    -DNDEBUG
    -DO3
    -DGGML_USE_CPU
)

# Platform-specific defines
if(APPLE)
    add_definitions(
        -DGGML_USE_ACCELERATE
        # Metal disabled for tests to avoid shader file dependencies
        # -DGGML_USE_METAL
    )
elseif(UNIX AND NOT APPLE)
    # Linux-specific defines for CPU affinity and scheduling
    add_definitions(-D_GNU_SOURCE)
endif()

# Force generic CPU optimizations to avoid missing assembly functions
add_definitions(-DGGML_CPU_GENERIC)

include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/rnllama-sources.cmake)

# Shared sources build in generic mode (-DGGML_CPU_GENERIC above), so only
# the portable ggml-cpu kernels are compiled — matching the library/main build.
# The per-arch SIMD kernels (arch/x86) are added ONLY to the probe targets,
# which undefine GGML_CPU_GENERIC + build -march=native for accurate perf.
# Compiling arch/x86 into a generic-mode target double-defines the public
# kernels and leaves the *_generic fallbacks unresolved.
if(APPLE)
    # Probe targets keep GGML_CPU_GENERIC on Apple (no -U override below),
    # so the per-arch SIMD sources must be left out — mixing them with the
    # generic build double-defines the public kernels / drops the *_generic
    # fallbacks and the link fails.
    set(GGML_CPU_ARCH_FILES "")
else()
    file(GLOB GGML_CPU_ARCH_FILES
        ${RNLLAMA_GGML_CPU_ARCH_DIR}/x86/*.c
        ${RNLLAMA_GGML_CPU_ARCH_DIR}/x86/*.cpp)
endif()

# Every test links the whole library plus codec.cpp's WAV/NPY helpers
# (included as "utils/wav_io.h", hence the examples/ include dir).
set(TEST_SOURCES ${RNLLAMA_CORE_SOURCES} ${RNLLAMA_CODEC_UTILS_SOURCES})
set(TEST_INCLUDE_DIRS ${RNLLAMA_INCLUDE_DIRS} ${RNLLAMA_CODEC_CPP_DIR}/examples)

# rnllama_test(<target> <main.cpp> [NATIVE_SIMD] [SOURCES <extra>...])
# NATIVE_SIMD: probe targets that measure performance; on Linux they get the
# per-arch kernels and -march=native instead of the forced-generic build.
function(rnllama_test name main)
    cmake_parse_arguments(ARG "NATIVE_SIMD" "" "SOURCES" ${ARGN})

    set(arch_sources "")
    if(ARG_NATIVE_SIMD)
        set(arch_sources ${GGML_CPU_ARCH_FILES})
    endif()

    add_executable(${name} ${main} ${TEST_SOURCES} ${arch_sources} ${ARG_SOURCES})
    target_include_directories(${name} PRIVATE ${TEST_INCLUDE_DIRS})

    if(APPLE)
        target_link_libraries(${name} PRIVATE
            "-framework Accelerate"
            "-framework Foundation"
        )
    elseif(UNIX AND NOT APPLE)
        find_package(Threads REQUIRED)
        target_link_libraries(${name} PRIVATE Threads::Threads m dl)
        if(ARG_NATIVE_SIMD)
            target_compile_options(${name} PRIVATE -march=native -U GGML_CPU_GENERIC)
        endif()
    endif()
endfunction()

# Basic integration tests
rnllama_test(rnllama_tests simple_test.cpp)

# Chat parse UTF-8 robustness tests
rnllama_test(chat_parse_utf8_test chat_parse_utf8_test.cpp)

# Parallel decoding tests
rnllama_test(parallel_decoding_test parallel_decoding_test.cpp)

# BlueMagpie CLI probe: measures per-step latency of the continuous_embd
# completion-loop hook on host, so we can pin down the bottleneck without
# fighting Pixel's OOM killer.
rnllama_test(bluemagpie_probe bluemagpie_probe.cpp NATIVE_SIMD)

# -DTESTS_METAL=ON (Apple only): compile the Metal backend into tts_probe so
# host runs exercise the same GPU path the app uses (backbone via --ngl,
# codec via --codec-gpu). Target-scoped: other test targets stay CPU-only.
if(APPLE AND TESTS_METAL)
    enable_language(ASM)
    set(TTS_PROBE_METAL_FILES ${RNLLAMA_GGML_METAL_SOURCES})
else()
    set(TTS_PROBE_METAL_FILES "")
endif()

rnllama_test(tts_probe tts_probe.cpp NATIVE_SIMD SOURCES ${TTS_PROBE_METAL_FILES})
if(APPLE AND TESTS_METAL)
    target_compile_definitions(tts_probe PRIVATE GGML_USE_METAL GGML_METAL_EMBED_LIBRARY=1 GGMLMetalClass=RNLlamaGGMLMetalClass)
    target_link_libraries(tts_probe PRIVATE
        "-framework Metal"
        "-framework MetalKit"
    )
endif()

rnllama_test(barbet_hidden_dump barbet_hidden_dump.cpp NATIVE_SIMD)

# KV-cache-reuse harness
rnllama_test(kv_cache_reuse_test kv_cache_reuse_test.cpp)

# A/B benchmark for the prompt state cache (also builds at the merge-base;
# KV_BENCH_HAS_STATE_CACHE gates branch-only knobs).
rnllama_test(kv_cache_bench kv_cache_bench.cpp)
if(KV_BENCH_HAS_STATE_CACHE)
    target_compile_definitions(kv_cache_bench PRIVATE KV_BENCH_HAS_STATE_CACHE)
endif()
