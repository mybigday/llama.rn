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

# Collect source files using glob patterns
file(GLOB MODEL_FILES ${SOURCE_DIR}/models/*.cpp)
file(GLOB GGML_CPU_C_FILES ${SOURCE_DIR}/ggml-cpu/*.c)
file(GLOB GGML_CPU_CPP_FILES ${SOURCE_DIR}/ggml-cpu/*.cpp)
file(GLOB GGML_CPU_AMX_FILES ${SOURCE_DIR}/ggml-cpu/amx/*.cpp)
file(GLOB LLAMA_FILES ${SOURCE_DIR}/llama*.cpp)
file(GLOB MTMD_FILES ${SOURCE_DIR}/tools/mtmd/*.cpp)
file(GLOB MTMD_MODEL_FILES ${SOURCE_DIR}/tools/mtmd/models/*.cpp)
file(GLOB COMMON_FILES ${SOURCE_DIR}/common/*.cpp)
file(GLOB JINJA_FILES ${SOURCE_DIR}/common/jinja/*.cpp)

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
