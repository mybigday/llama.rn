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

# Platform-specific source files
set(SOURCE_FILES_ARCH "")
set(PLATFORM_SOURCES "")

if(APPLE)
    # Metal disabled for tests, so no Metal sources needed
    # set(PLATFORM_SOURCES ${SOURCE_DIR}/ggml-metal.m)
endif()

# Create test executable
add_executable(rnllama_tests
    # Test files
    simple_test.cpp

    # Core rnllama cpp files
    ${SOURCE_DIR}/rn-llama.cpp
    ${SOURCE_DIR}/rn-completion.cpp
    ${SOURCE_DIR}/rn-tts.cpp
    ${SOURCE_DIR}/rn-slot.cpp
    ${SOURCE_DIR}/rn-slot-manager.cpp

    # Core llama.cpp files
    ${SOURCE_DIR}/ggml.c
    ${SOURCE_DIR}/ggml-alloc.c
    ${SOURCE_DIR}/ggml-backend.cpp
    ${SOURCE_DIR}/ggml-backend-reg.cpp
    ${SOURCE_DIR}/ggml-cpu/amx/amx.cpp
    ${SOURCE_DIR}/ggml-cpu/amx/mmq.cpp
    ${SOURCE_DIR}/ggml-cpu/ggml-cpu.c
    ${SOURCE_DIR}/ggml-cpu/ggml-cpu.cpp
    ${SOURCE_DIR}/ggml-cpu/quants.c
    ${SOURCE_DIR}/ggml-cpu/traits.cpp
    ${SOURCE_DIR}/ggml-cpu/repack.cpp
    ${SOURCE_DIR}/ggml-cpu/unary-ops.cpp
    ${SOURCE_DIR}/ggml-cpu/binary-ops.cpp
    ${SOURCE_DIR}/ggml-cpu/vec.cpp
    ${SOURCE_DIR}/ggml-cpu/ops.cpp
    ${PLATFORM_SOURCES}
    ${SOURCE_DIR}/ggml-opt.cpp
    ${SOURCE_DIR}/ggml-threading.cpp
    ${SOURCE_DIR}/ggml-quants.c
    ${SOURCE_DIR}/gguf.cpp
    ${SOURCE_DIR}/log.cpp
    ${SOURCE_DIR}/llama-impl.cpp
    ${SOURCE_DIR}/llama-grammar.cpp
    ${SOURCE_DIR}/llama-sampling.cpp
    ${SOURCE_DIR}/llama-vocab.cpp
    ${SOURCE_DIR}/llama-adapter.cpp
    ${SOURCE_DIR}/llama-chat.cpp
    ${SOURCE_DIR}/llama-context.cpp
    ${SOURCE_DIR}/llama-arch.cpp
    ${SOURCE_DIR}/llama-batch.cpp
    ${SOURCE_DIR}/llama-cparams.cpp
    ${SOURCE_DIR}/llama-hparams.cpp
    ${SOURCE_DIR}/llama.cpp
    ${SOURCE_DIR}/llama-model.cpp
    ${SOURCE_DIR}/llama-model-loader.cpp
    ${SOURCE_DIR}/llama-model-saver.cpp
    ${SOURCE_DIR}/llama-mmap.cpp
    ${SOURCE_DIR}/llama-kv-cache.cpp
    ${SOURCE_DIR}/llama-kv-cache-iswa.cpp
    ${SOURCE_DIR}/llama-memory-hybrid.cpp
    ${SOURCE_DIR}/llama-memory-recurrent.cpp
    ${SOURCE_DIR}/llama-memory.cpp
    ${SOURCE_DIR}/llama-io.cpp
    ${SOURCE_DIR}/llama-graph.cpp
    ${SOURCE_DIR}/sampling.cpp
    ${SOURCE_DIR}/unicode-data.cpp
    ${SOURCE_DIR}/unicode.cpp
    ${SOURCE_DIR}/common.cpp
    ${SOURCE_DIR}/chat.cpp
    ${SOURCE_DIR}/json-schema-to-grammar.cpp
    ${SOURCE_DIR}/minja/minja.hpp
    ${SOURCE_DIR}/minja/chat-template.hpp
    ${SOURCE_DIR}/nlohmann/json.hpp
    ${SOURCE_DIR}/nlohmann/json_fwd.hpp
    ${SOURCE_DIR}/chat-parser.cpp
    ${SOURCE_DIR}/json-partial.cpp
    ${SOURCE_DIR}/regex-partial.cpp
    # Multimodal support
    ${SOURCE_DIR}/tools/mtmd/mtmd.cpp
    ${SOURCE_DIR}/tools/mtmd/mtmd-audio.cpp
    ${SOURCE_DIR}/tools/mtmd/clip.cpp
    ${SOURCE_DIR}/tools/mtmd/mtmd-helper.cpp
    ${SOURCE_DIR}/anyascii.c
    ${SOURCE_FILES_ARCH}
)

# Setup include directories
target_include_directories(rnllama_tests
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../cpp/ggml-cpu
        ${CMAKE_CURRENT_SOURCE_DIR}/../cpp/tools/mtmd
        ${CMAKE_CURRENT_SOURCE_DIR}/../cpp/minja
        ${CMAKE_CURRENT_SOURCE_DIR}/../cpp/nlohmann
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

# Create parallel decoding test executable
add_executable(parallel_decoding_test
    # Test file
    parallel_decoding_test.cpp

    # Core rnllama cpp files
    ${SOURCE_DIR}/rn-llama.cpp
    ${SOURCE_DIR}/rn-completion.cpp
    ${SOURCE_DIR}/rn-tts.cpp
    ${SOURCE_DIR}/rn-slot.cpp
    ${SOURCE_DIR}/rn-slot-manager.cpp

    # Core llama.cpp files
    ${SOURCE_DIR}/ggml.c
    ${SOURCE_DIR}/ggml-alloc.c
    ${SOURCE_DIR}/ggml-backend.cpp
    ${SOURCE_DIR}/ggml-backend-reg.cpp
    ${SOURCE_DIR}/ggml-cpu/amx/amx.cpp
    ${SOURCE_DIR}/ggml-cpu/amx/mmq.cpp
    ${SOURCE_DIR}/ggml-cpu/ggml-cpu.c
    ${SOURCE_DIR}/ggml-cpu/ggml-cpu.cpp
    ${SOURCE_DIR}/ggml-cpu/quants.c
    ${SOURCE_DIR}/ggml-cpu/traits.cpp
    ${SOURCE_DIR}/ggml-cpu/repack.cpp
    ${SOURCE_DIR}/ggml-cpu/unary-ops.cpp
    ${SOURCE_DIR}/ggml-cpu/binary-ops.cpp
    ${SOURCE_DIR}/ggml-cpu/vec.cpp
    ${SOURCE_DIR}/ggml-cpu/ops.cpp
    ${PLATFORM_SOURCES}
    ${SOURCE_DIR}/ggml-opt.cpp
    ${SOURCE_DIR}/ggml-threading.cpp
    ${SOURCE_DIR}/ggml-quants.c
    ${SOURCE_DIR}/gguf.cpp
    ${SOURCE_DIR}/log.cpp
    ${SOURCE_DIR}/llama-impl.cpp
    ${SOURCE_DIR}/llama-grammar.cpp
    ${SOURCE_DIR}/llama-sampling.cpp
    ${SOURCE_DIR}/llama-vocab.cpp
    ${SOURCE_DIR}/llama-adapter.cpp
    ${SOURCE_DIR}/llama-chat.cpp
    ${SOURCE_DIR}/llama-context.cpp
    ${SOURCE_DIR}/llama-arch.cpp
    ${SOURCE_DIR}/llama-batch.cpp
    ${SOURCE_DIR}/llama-cparams.cpp
    ${SOURCE_DIR}/llama-hparams.cpp
    ${SOURCE_DIR}/llama.cpp
    ${SOURCE_DIR}/llama-model.cpp
    ${SOURCE_DIR}/llama-model-loader.cpp
    ${SOURCE_DIR}/llama-model-saver.cpp
    ${SOURCE_DIR}/llama-mmap.cpp
    ${SOURCE_DIR}/llama-kv-cache.cpp
    ${SOURCE_DIR}/llama-kv-cache-iswa.cpp
    ${SOURCE_DIR}/llama-memory-hybrid.cpp
    ${SOURCE_DIR}/llama-memory-recurrent.cpp
    ${SOURCE_DIR}/llama-memory.cpp
    ${SOURCE_DIR}/llama-io.cpp
    ${SOURCE_DIR}/llama-graph.cpp
    ${SOURCE_DIR}/sampling.cpp
    ${SOURCE_DIR}/unicode-data.cpp
    ${SOURCE_DIR}/unicode.cpp
    ${SOURCE_DIR}/common.cpp
    ${SOURCE_DIR}/chat.cpp
    ${SOURCE_DIR}/json-schema-to-grammar.cpp
    ${SOURCE_DIR}/minja/minja.hpp
    ${SOURCE_DIR}/minja/chat-template.hpp
    ${SOURCE_DIR}/nlohmann/json.hpp
    ${SOURCE_DIR}/nlohmann/json_fwd.hpp
    ${SOURCE_DIR}/chat-parser.cpp
    ${SOURCE_DIR}/json-partial.cpp
    ${SOURCE_DIR}/regex-partial.cpp
    # Multimodal support
    ${SOURCE_DIR}/tools/mtmd/mtmd.cpp
    ${SOURCE_DIR}/tools/mtmd/mtmd-audio.cpp
    ${SOURCE_DIR}/tools/mtmd/clip.cpp
    ${SOURCE_DIR}/tools/mtmd/mtmd-helper.cpp
    ${SOURCE_DIR}/anyascii.c
    ${SOURCE_FILES_ARCH}
)

# Setup include directories for parallel decoding test
target_include_directories(parallel_decoding_test
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../cpp/ggml-cpu
        ${CMAKE_CURRENT_SOURCE_DIR}/../cpp/tools/mtmd
        ${CMAKE_CURRENT_SOURCE_DIR}/../cpp/minja
        ${CMAKE_CURRENT_SOURCE_DIR}/../cpp/nlohmann
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
