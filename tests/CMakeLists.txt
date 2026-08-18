cmake_minimum_required(VERSION 3.16)
project(rnllama_tests VERSION 1.0.0 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Dependencies and compile options
add_definitions(
    -DNDEBUG
    -DO3
    -DLM_GGML_USE_CPU
)

# Platform-specific defines
if(APPLE)
    add_definitions(
        -DLM_GGML_USE_ACCELERATE
        # Metal disabled for tests to avoid shader file dependencies
        # -DLM_GGML_USE_METAL
    )
elseif(UNIX AND NOT APPLE)
    # Linux-specific defines for CPU affinity and scheduling
    add_definitions(-D_GNU_SOURCE)
endif()

# Force generic CPU optimizations to avoid missing assembly functions
add_definitions(-DLM_GGML_CPU_GENERIC)

set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../cpp)
include_directories(${SOURCE_DIR}/hash)

# Collect source files using glob patterns
file(GLOB MODEL_FILES ${SOURCE_DIR}/models/*.cpp)
# Shared sources build in generic mode (-DLM_GGML_CPU_GENERIC below), so only
# the portable ggml-cpu kernels are compiled — matching the library/main build.
# The per-arch SIMD kernels (arch/x86) are added ONLY to the probe targets,
# which undefine LM_GGML_CPU_GENERIC + build -march=native for accurate perf.
# Compiling arch/x86 into a generic-mode target double-defines the public
# kernels and leaves the *_generic fallbacks unresolved.
file(GLOB GGML_CPU_C_FILES ${SOURCE_DIR}/ggml-cpu/*.c)
file(GLOB GGML_CPU_CPP_FILES ${SOURCE_DIR}/ggml-cpu/*.cpp)
if(APPLE)
    # Probe targets keep LM_GGML_CPU_GENERIC on Apple (no -U override below),
    # so the per-arch SIMD sources must be left out — mixing them with the
    # generic build double-defines the public kernels / drops the *_generic
    # fallbacks and the link fails.
    set(GGML_CPU_ARCH_FILES "")
else()
    file(GLOB GGML_CPU_ARCH_FILES ${SOURCE_DIR}/ggml-cpu/arch/x86/*.c ${SOURCE_DIR}/ggml-cpu/arch/x86/*.cpp)
endif()
file(GLOB GGML_CPU_AMX_FILES ${SOURCE_DIR}/ggml-cpu/amx/*.cpp)
file(GLOB LLAMA_FILES ${SOURCE_DIR}/llama*.cpp)
file(GLOB MTMD_FILES ${SOURCE_DIR}/tools/mtmd/*.cpp)
file(GLOB MTMD_MODEL_FILES ${SOURCE_DIR}/tools/mtmd/models/*.cpp)
file(GLOB COMMON_FILES ${SOURCE_DIR}/common/*.cpp)
file(GLOB JINJA_FILES ${SOURCE_DIR}/common/jinja/*.cpp)
file(GLOB CODEC_FILES
    ${SOURCE_DIR}/codec/src/*.cpp
    ${SOURCE_DIR}/codec/src/batch/*.cpp
    ${SOURCE_DIR}/codec/src/lm/*.cpp
    ${SOURCE_DIR}/codec/src/models/*.cpp
    ${SOURCE_DIR}/codec/src/ops/*.cpp
    ${SOURCE_DIR}/codec/src/runtime/*.cpp
    ${SOURCE_DIR}/codec/common/audio_lm.cpp
    ${SOURCE_DIR}/codec/common/utils/*.cpp
)
# NB: tts_runner{,_flow}.cpp are excluded — they are the codec.cpp reference
# host loop that pulls in llama.cpp's `common_*` symbols under their un-prefixed
# names, which conflict with the LM_-prefixed common/ in this project.
# rn-tts drives its own AR loop through the low-level codec_lm_* + codec_common
# APIs, so we only need audio_lm.cpp from codec/common.

# Platform-specific source files
set(SOURCE_FILES_ARCH "")
set(PLATFORM_SOURCES "")

if(APPLE)
    # Metal disabled for tests, so no Metal sources needed
    # set(PLATFORM_SOURCES ${SOURCE_DIR}/ggml-metal.m)
endif()

# Shared source files for all test executables
set(RNLLAMA_COMMON_SOURCES
    # Core GGML files
    ${SOURCE_DIR}/ggml.c
    ${SOURCE_DIR}/ggml-alloc.c
    ${SOURCE_DIR}/ggml-backend.cpp
    ${SOURCE_DIR}/ggml-backend-dl.cpp
    ${SOURCE_DIR}/ggml-backend-meta.cpp
    ${SOURCE_DIR}/ggml-backend-reg.cpp
    ${SOURCE_DIR}/ggml-opt.cpp
    ${SOURCE_DIR}/ggml-threading.cpp
    ${SOURCE_DIR}/ggml-quants.c
    ${SOURCE_DIR}/gguf.cpp

    # GGML CPU files (globbed)
    ${GGML_CPU_C_FILES}
    ${GGML_CPU_CPP_FILES}
    ${GGML_CPU_AMX_FILES}

    # Platform-specific sources
    ${PLATFORM_SOURCES}

    # Llama files (globbed)
    ${LLAMA_FILES}
    ${SOURCE_DIR}/unicode-data.cpp
    ${SOURCE_DIR}/unicode.cpp

    # Common utilities
    ${COMMON_FILES}

    # Jinja template engine
    ${JINJA_FILES}

    ${SOURCE_DIR}/anyascii.c
    ${SOURCE_DIR}/hash/hash.cpp
    ${SOURCE_DIR}/hash/sha256/sha256.c

    # Multimodal support (globbed)
    ${MTMD_MODEL_FILES}
    ${MTMD_FILES}

    # React Native llama APIs
    ${SOURCE_DIR}/rn-llama.cpp
    ${SOURCE_DIR}/rn-completion.cpp
    ${SOURCE_DIR}/rn-tts.cpp
    ${SOURCE_DIR}/rn-slot.cpp
    ${SOURCE_DIR}/rn-slot-manager.cpp

    # Model implementations (globbed)
    ${MODEL_FILES}

    # Architecture-specific files
    ${SOURCE_FILES_ARCH}

    # codec.cpp sources
    ${CODEC_FILES}
)

# Create test executable
add_executable(rnllama_tests
    simple_test.cpp
    ${RNLLAMA_COMMON_SOURCES}
)

# Setup include directories
target_include_directories(rnllama_tests
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../cpp/common
        ${CMAKE_CURRENT_SOURCE_DIR}/../cpp/common/jinja
        ${CMAKE_CURRENT_SOURCE_DIR}/../cpp/ggml-cpu
        ${CMAKE_CURRENT_SOURCE_DIR}/../cpp/tools/mtmd
        ${CMAKE_CURRENT_SOURCE_DIR}/../cpp/codec/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../cpp/codec/common
        ${CMAKE_CURRENT_SOURCE_DIR}/../cpp/codec/common/utils
)

# Platform-specific linking
if(APPLE)
    # Link required frameworks for macOS
    target_link_libraries(rnllama_tests PRIVATE
        "-framework Accelerate"
        "-framework Foundation"
    )
elseif(UNIX AND NOT APPLE)
    # Link required libraries for Linux
    find_package(Threads REQUIRED)
    target_link_libraries(rnllama_tests PRIVATE
        Threads::Threads
        m  # Math library
        dl # Dynamic loading library
    )
endif()

# Create chat parse UTF-8 robustness test executable
add_executable(chat_parse_utf8_test
    chat_parse_utf8_test.cpp
    ${RNLLAMA_COMMON_SOURCES}
)

# Setup include directories
target_include_directories(chat_parse_utf8_test
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../cpp/common
        ${CMAKE_CURRENT_SOURCE_DIR}/../cpp/common/jinja
        ${CMAKE_CURRENT_SOURCE_DIR}/../cpp/ggml-cpu
        ${CMAKE_CURRENT_SOURCE_DIR}/../cpp/tools/mtmd
        ${CMAKE_CURRENT_SOURCE_DIR}/../cpp/codec/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../cpp/codec/common
        ${CMAKE_CURRENT_SOURCE_DIR}/../cpp/codec/common/utils
)

# Platform-specific linking
if(APPLE)
    target_link_libraries(chat_parse_utf8_test PRIVATE
        "-framework Accelerate"
        "-framework Foundation"
    )
elseif(UNIX AND NOT APPLE)
    find_package(Threads REQUIRED)
    target_link_libraries(chat_parse_utf8_test PRIVATE
        Threads::Threads
        m
        dl
    )
endif()

# Create parallel decoding test executable
add_executable(parallel_decoding_test
    parallel_decoding_test.cpp
    ${RNLLAMA_COMMON_SOURCES}
)

# Setup include directories for parallel decoding test
target_include_directories(parallel_decoding_test
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../cpp/common
        ${CMAKE_CURRENT_SOURCE_DIR}/../cpp/common/jinja
        ${CMAKE_CURRENT_SOURCE_DIR}/../cpp/ggml-cpu
        ${CMAKE_CURRENT_SOURCE_DIR}/../cpp/tools/mtmd
        ${CMAKE_CURRENT_SOURCE_DIR}/../cpp/codec/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../cpp/codec/common
        ${CMAKE_CURRENT_SOURCE_DIR}/../cpp/codec/common/utils
)

# Platform-specific linking for parallel decoding test
if(APPLE)
    target_link_libraries(parallel_decoding_test PRIVATE
        "-framework Accelerate"
        "-framework Foundation"
    )
elseif(UNIX AND NOT APPLE)
    find_package(Threads REQUIRED)
    target_link_libraries(parallel_decoding_test PRIVATE
        Threads::Threads
        m
        dl
    )
endif()

# BlueMagpie CLI probe: measures per-step latency of the continuous_embd
# completion-loop hook on host, so we can pin down the bottleneck without
# fighting Pixel's OOM killer.
add_executable(bluemagpie_probe
    bluemagpie_probe.cpp
    ${RNLLAMA_COMMON_SOURCES}
    ${GGML_CPU_ARCH_FILES}
)
target_include_directories(bluemagpie_probe
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../cpp/common
        ${CMAKE_CURRENT_SOURCE_DIR}/../cpp/common/jinja
        ${CMAKE_CURRENT_SOURCE_DIR}/../cpp/ggml-cpu
        ${CMAKE_CURRENT_SOURCE_DIR}/../cpp/tools/mtmd
        ${CMAKE_CURRENT_SOURCE_DIR}/../cpp/codec/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../cpp/codec/common
        ${CMAKE_CURRENT_SOURCE_DIR}/../cpp/codec/common/utils
)
if(APPLE)
    target_link_libraries(bluemagpie_probe PRIVATE
        "-framework Accelerate"
        "-framework Foundation"
    )
elseif(UNIX AND NOT APPLE)
    find_package(Threads REQUIRED)
    target_link_libraries(bluemagpie_probe PRIVATE Threads::Threads m dl)
    # Native SIMD for accurate perf measurement (skip the forced-generic
    # define; the ggml-cpu arch sources will pick native ARM/x86 kernels).
    target_compile_options(bluemagpie_probe PRIVATE -march=native -U LM_GGML_CPU_GENERIC)
endif()

# -DTESTS_METAL=ON (Apple only): compile the Metal backend into tts_probe so
# host runs exercise the same GPU path the app uses (backbone via --ngl,
# codec via --codec-gpu). Target-scoped: other test targets stay CPU-only.
if(APPLE AND TESTS_METAL)
    enable_language(ASM)
    file(GLOB TTS_PROBE_METAL_FILES
        ${SOURCE_DIR}/ggml-metal/*.cpp
        ${SOURCE_DIR}/ggml-metal/*.m
        ${SOURCE_DIR}/ggml-metal/*.s)
    set_source_files_properties(${SOURCE_DIR}/ggml-metal/ggml-metal-embed.s PROPERTIES LANGUAGE ASM)
else()
    set(TTS_PROBE_METAL_FILES "")
endif()

add_executable(tts_probe
    tts_probe.cpp
    ${RNLLAMA_COMMON_SOURCES}
    ${GGML_CPU_ARCH_FILES}
    ${TTS_PROBE_METAL_FILES}
)
if(APPLE AND TESTS_METAL)
    target_compile_definitions(tts_probe PRIVATE LM_GGML_USE_METAL LM_GGML_METAL_EMBED_LIBRARY=1)
endif()
target_include_directories(tts_probe
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../cpp/common
        ${CMAKE_CURRENT_SOURCE_DIR}/../cpp/common/jinja
        ${CMAKE_CURRENT_SOURCE_DIR}/../cpp/ggml-cpu
        ${CMAKE_CURRENT_SOURCE_DIR}/../cpp/tools/mtmd
        ${CMAKE_CURRENT_SOURCE_DIR}/../cpp/codec/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../cpp/codec/common
        ${CMAKE_CURRENT_SOURCE_DIR}/../cpp/codec/common/utils
)
if(APPLE)
    target_link_libraries(tts_probe PRIVATE
        "-framework Accelerate"
        "-framework Foundation"
    )
    if(TESTS_METAL)
        target_link_libraries(tts_probe PRIVATE
            "-framework Metal"
            "-framework MetalKit"
        )
    endif()
elseif(UNIX AND NOT APPLE)
    find_package(Threads REQUIRED)
    target_link_libraries(tts_probe PRIVATE Threads::Threads m dl)
    target_compile_options(tts_probe PRIVATE -march=native -U LM_GGML_CPU_GENERIC)
endif()

add_executable(barbet_hidden_dump
    barbet_hidden_dump.cpp
    ${RNLLAMA_COMMON_SOURCES}
    ${GGML_CPU_ARCH_FILES}
)
target_include_directories(barbet_hidden_dump
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../cpp/common
        ${CMAKE_CURRENT_SOURCE_DIR}/../cpp/common/jinja
        ${CMAKE_CURRENT_SOURCE_DIR}/../cpp/ggml-cpu
        ${CMAKE_CURRENT_SOURCE_DIR}/../cpp/tools/mtmd
        ${CMAKE_CURRENT_SOURCE_DIR}/../cpp/codec/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../cpp/codec/common
        ${CMAKE_CURRENT_SOURCE_DIR}/../cpp/codec/common/utils
)
if(APPLE)
    target_link_libraries(barbet_hidden_dump PRIVATE
        "-framework Accelerate"
        "-framework Foundation"
    )
elseif(UNIX AND NOT APPLE)
    find_package(Threads REQUIRED)
    target_link_libraries(barbet_hidden_dump PRIVATE Threads::Threads m dl)
    target_compile_options(barbet_hidden_dump PRIVATE -march=native -U LM_GGML_CPU_GENERIC)
endif()

# Create KV-cache-reuse harness executable
add_executable(kv_cache_reuse_test
    kv_cache_reuse_test.cpp
    ${RNLLAMA_COMMON_SOURCES}
)
target_include_directories(kv_cache_reuse_test
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../cpp/common
        ${CMAKE_CURRENT_SOURCE_DIR}/../cpp/common/jinja
        ${CMAKE_CURRENT_SOURCE_DIR}/../cpp/ggml-cpu
        ${CMAKE_CURRENT_SOURCE_DIR}/../cpp/tools/mtmd
        ${CMAKE_CURRENT_SOURCE_DIR}/../cpp/codec/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../cpp/codec/common
        ${CMAKE_CURRENT_SOURCE_DIR}/../cpp/codec/common/utils
)
if(APPLE)
    target_link_libraries(kv_cache_reuse_test PRIVATE
        "-framework Accelerate"
        "-framework Foundation"
    )
elseif(UNIX AND NOT APPLE)
    find_package(Threads REQUIRED)
    target_link_libraries(kv_cache_reuse_test PRIVATE
        Threads::Threads
        m
        dl
    )
endif()

# A/B benchmark for the prompt state cache (also builds at the merge-base;
# KV_BENCH_HAS_STATE_CACHE gates branch-only knobs).
add_executable(kv_cache_bench
    kv_cache_bench.cpp
    ${RNLLAMA_COMMON_SOURCES}
)
target_include_directories(kv_cache_bench
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../cpp/common
        ${CMAKE_CURRENT_SOURCE_DIR}/../cpp/common/jinja
        ${CMAKE_CURRENT_SOURCE_DIR}/../cpp/ggml-cpu
        ${CMAKE_CURRENT_SOURCE_DIR}/../cpp/tools/mtmd
        ${CMAKE_CURRENT_SOURCE_DIR}/../cpp/codec/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../cpp/codec/common
        ${CMAKE_CURRENT_SOURCE_DIR}/../cpp/codec/common/utils
)
if(KV_BENCH_HAS_STATE_CACHE)
    target_compile_definitions(kv_cache_bench PRIVATE KV_BENCH_HAS_STATE_CACHE)
endif()
if(APPLE)
    target_link_libraries(kv_cache_bench PRIVATE
        "-framework Accelerate"
        "-framework Foundation"
    )
elseif(UNIX AND NOT APPLE)
    find_package(Threads REQUIRED)
    target_link_libraries(kv_cache_bench PRIVATE
        Threads::Threads
        m
        dl
    )
endif()
