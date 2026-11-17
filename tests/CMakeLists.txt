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

include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/rnllama-compiler-cache.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/rnllama-sources.cmake)

# Shared sources build in generic mode (-DGGML_CPU_GENERIC above), so only
# the portable ggml-cpu kernels are compiled — matching the library/main build.
# The per-arch SIMD kernels (arch/x86) are added ONLY to the native-SIMD core,
# which undefines GGML_CPU_GENERIC + builds -march=native for accurate perf.
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
# (included as "utils/wav_io.h" from common/; npy_io from the examples/ include dir).
set(TEST_SOURCES ${RNLLAMA_CORE_SOURCES} ${RNLLAMA_CODEC_UTILS_SOURCES})
set(TEST_INCLUDE_DIRS ${RNLLAMA_INCLUDE_DIRS} ${RNLLAMA_CODEC_CPP_DIR}/examples)

# The library is ~450 translation units, so it is compiled once per flavor and
# shared by every test using that flavor, instead of once per executable. An
# OBJECT library (rather than a static one) keeps the link identical to listing
# the sources on the executable: no object is dropped for being unreferenced.
# Usage requirements are PUBLIC so each test's own main file is compiled with
# the same flags as the library it links.
#
# rnllama_test_core(<target> [SOURCES <extra>...])
function(rnllama_test_core name)
    cmake_parse_arguments(ARG "" "" "SOURCES" ${ARGN})

    add_library(${name} OBJECT ${TEST_SOURCES} ${ARG_SOURCES})
    target_include_directories(${name} PUBLIC ${TEST_INCLUDE_DIRS})

    if(APPLE)
        target_link_libraries(${name} PUBLIC
            "-framework Accelerate"
            "-framework Foundation"
            ${RNLLAMA_VIDEO_LIBS}
        )
    elseif(UNIX AND NOT APPLE)
        find_package(Threads REQUIRED)
        target_link_libraries(${name} PUBLIC Threads::Threads m dl)
    endif()
endfunction()

# Generic CPU kernels: the tests.
rnllama_test_core(rnllama_test_core)

# Native SIMD: probe targets that measure performance. On Linux they get the
# per-arch kernels and -march=native instead of the forced-generic build; on
# Apple they use the generic core (see GGML_CPU_ARCH_FILES above).
if(UNIX AND NOT APPLE)
    rnllama_test_core(rnllama_test_core_native SOURCES ${GGML_CPU_ARCH_FILES})
    target_compile_options(rnllama_test_core_native PUBLIC -march=native -U GGML_CPU_GENERIC)
endif()

# rnllama_test(<target> <main.cpp> [NATIVE_SIMD] [CORE <core target>])
# NATIVE_SIMD: link the native-SIMD core, where there is one.
# CORE: link a specific core library (overrides NATIVE_SIMD).
function(rnllama_test name main)
    cmake_parse_arguments(ARG "NATIVE_SIMD" "CORE" "" ${ARGN})

    set(core rnllama_test_core)
    if(ARG_CORE)
        set(core ${ARG_CORE})
    elseif(ARG_NATIVE_SIMD AND TARGET rnllama_test_core_native)
        set(core rnllama_test_core_native)
    endif()

    add_executable(${name} ${main})
    target_link_libraries(${name} PRIVATE ${core})
endfunction()

# Basic integration tests
rnllama_test(rnllama_tests simple_test.cpp)

# Chat parse UTF-8 robustness tests
rnllama_test(chat_parse_utf8_test chat_parse_utf8_test.cpp)

# Parallel decoding tests
rnllama_test(parallel_decoding_test parallel_decoding_test.cpp)

# Decision model probe: prints the /v1/systemone response of a request, to diff
# against llama-server
rnllama_test(decision_probe decision_probe.cpp NATIVE_SIMD)

# -DTESTS_METAL=ON (Apple only): compile the Metal backend into tts_probe so
# host runs exercise the same GPU path the app uses (backbone via --ngl,
# codec via --codec-gpu). It gets its own core library, since GGML_USE_METAL
# changes how the library itself is compiled: other test targets stay CPU-only.
if(APPLE AND TESTS_METAL)
    enable_language(ASM)
    rnllama_test_core(rnllama_test_core_metal SOURCES ${RNLLAMA_GGML_METAL_SOURCES})
    target_compile_definitions(rnllama_test_core_metal PUBLIC
        GGML_USE_METAL GGML_METAL_EMBED_LIBRARY=1 GGMLMetalClass=RNLlamaGGMLMetalClass)
    target_link_libraries(rnllama_test_core_metal PUBLIC
        "-framework Metal"
        "-framework MetalKit"
    )
    rnllama_test(tts_probe tts_probe.cpp CORE rnllama_test_core_metal)
else()
    rnllama_test(tts_probe tts_probe.cpp NATIVE_SIMD)
endif()

rnllama_test(barbet_hidden_dump barbet_hidden_dump.cpp NATIVE_SIMD)

# read_gguf_file_info() array-length reporting
rnllama_test(model_info_test model_info_test.cpp)

# Platform video decoder behind video_url (cpp/rn-video.h)
rnllama_test(video_test video_test.cpp)

# KV-cache-reuse harness
rnllama_test(kv_cache_reuse_test kv_cache_reuse_test.cpp)

# A/B benchmark for the prompt state cache (also builds at the merge-base;
# KV_BENCH_HAS_STATE_CACHE gates branch-only knobs).
rnllama_test(kv_cache_bench kv_cache_bench.cpp)
if(KV_BENCH_HAS_STATE_CACHE)
    target_compile_definitions(kv_cache_bench PRIVATE KV_BENCH_HAS_STATE_CACHE)
endif()
