cmake_minimum_required(VERSION 3.16)
project(rnllama VERSION 1.0.0 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# iOS specific settings
set(CMAKE_OSX_DEPLOYMENT_TARGET 13.0)
set(CMAKE_XCODE_ATTRIBUTE_ENABLE_BITCODE NO)

# Dependencies and compile options
add_definitions(
    -DNDEBUG
    -DO3
    -DLM_GGML_USE_CPU
    -DLM_GGML_USE_ACCELERATE
    -DLM_GGML_USE_BLAS
    -DLM_GGML_BLAS_USE_ACCELERATE
    -DLM_GGML_USE_METAL
)

if (CMAKE_OSX_ARCHITECTURES STREQUAL "arm64;x86_64")
    add_definitions(-DLM_GGML_CPU_GENERIC)
endif ()

set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../cpp)

# Collect source files using glob patterns
file(GLOB MODEL_FILES ${SOURCE_DIR}/models/*.cpp)
file(GLOB GGML_CPU_C_FILES ${SOURCE_DIR}/ggml-cpu/*.c)
file(GLOB GGML_CPU_CPP_FILES ${SOURCE_DIR}/ggml-cpu/*.cpp)
file(GLOB GGML_CPU_AMX_FILES ${SOURCE_DIR}/ggml-cpu/amx/*.cpp)
file(GLOB GGML_METAL_CPP_FILES ${SOURCE_DIR}/ggml-metal/*.cpp)
file(GLOB GGML_METAL_M_FILES ${SOURCE_DIR}/ggml-metal/*.m)
file(GLOB GGML_BLAS_FILES ${SOURCE_DIR}/ggml-blas/*.cpp)
file(GLOB LLAMA_FILES ${SOURCE_DIR}/llama*.cpp)
file(GLOB MTMD_FILES ${SOURCE_DIR}/tools/mtmd/*.cpp)
file(GLOB COMMON_FILES ${SOURCE_DIR}/common/*.cpp)

if (CMAKE_OSX_ARCHITECTURES STREQUAL "arm64")
    set(SOURCE_FILES_ARCH
        ${SOURCE_DIR}/ggml-cpu/arch/arm/quants.c
        ${SOURCE_DIR}/ggml-cpu/arch/arm/repack.cpp
    )
endif ()

# Define public headers
set(PUBLIC_HEADERS
    ${SOURCE_DIR}/rn-llama.h
    ${SOURCE_DIR}/rn-completion.h
    ${SOURCE_DIR}/rn-slot.h
    ${SOURCE_DIR}/rn-slot-manager.h
    ${SOURCE_DIR}/rn-tts.h
    ${SOURCE_DIR}/llama.h
    ${SOURCE_DIR}/llama-impl.h
    ${SOURCE_DIR}/ggml.h
)

# Create library target
add_library(rnllama SHARED
    # Core GGML files
    ${SOURCE_DIR}/ggml.c
    ${SOURCE_DIR}/ggml-alloc.c
    ${SOURCE_DIR}/ggml-backend.cpp
    ${SOURCE_DIR}/ggml-backend-reg.cpp
    ${SOURCE_DIR}/ggml-opt.cpp
    ${SOURCE_DIR}/ggml-threading.cpp
    ${SOURCE_DIR}/ggml-quants.c
    ${SOURCE_DIR}/gguf.cpp

    # GGML CPU files (globbed)
    ${GGML_CPU_C_FILES}
    ${GGML_CPU_CPP_FILES}
    ${GGML_CPU_AMX_FILES}

    # GGML Metal files (globbed)
    ${GGML_METAL_CPP_FILES}
    ${GGML_METAL_M_FILES}

    # GGML BLAS files (globbed)
    ${GGML_BLAS_FILES}

    # Llama files (globbed)
    ${LLAMA_FILES}
    ${SOURCE_DIR}/unicode-data.cpp
    ${SOURCE_DIR}/unicode.cpp

    # Common utilities
    ${COMMON_FILES}

    ${SOURCE_DIR}/anyascii.c

    # Header-only libraries (needed for build)
    ${SOURCE_DIR}/minja/minja.hpp
    ${SOURCE_DIR}/minja/chat-template.hpp
    ${SOURCE_DIR}/nlohmann/json.hpp
    ${SOURCE_DIR}/nlohmann/json_fwd.hpp
    ${SOURCE_DIR}/common/chat.h
    ${SOURCE_DIR}/common/common.h

    # Multimodal support (globbed)
    ${MTMD_FILES}

    # React Native llama APIs
    ${SOURCE_DIR}/rn-llama.cpp
    ${SOURCE_DIR}/rn-completion.cpp
    ${SOURCE_DIR}/rn-slot.cpp
    ${SOURCE_DIR}/rn-slot-manager.cpp
    ${SOURCE_DIR}/rn-tts.cpp

    # Model implementations (globbed)
    ${MODEL_FILES}

    # Architecture-specific files
    ${SOURCE_FILES_ARCH}
)

# Setup include directories
target_include_directories(rnllama
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../cpp>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../cpp/common>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../cpp/ggml-cpu>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../cpp/tools/mtmd>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../cpp/minja>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../cpp/nlohmann>
        $<INSTALL_INTERFACE:include>
)

# Link required frameworks
target_link_libraries(rnllama PRIVATE
    "-framework Accelerate"
    "-framework Foundation"
    "-framework Metal"
    "-framework MetalKit"
)

# Set properties for framework
set_target_properties(rnllama PROPERTIES
    MACOSX_FRAMEWORK_IDENTIFIER "com.rnllama"
    MACOSX_FRAMEWORK_BUNDLE_VERSION 1.0.0
    MACOSX_FRAMEWORK_SHORT_VERSION_STRING 1.0.0
    FRAMEWORK TRUE
    FRAMEWORK_VERSION 1.0.0
    VERSION 1.0.0
    PUBLIC_HEADER "${PUBLIC_HEADERS}"
    XCODE_ATTRIBUTE_CLANG_ENABLE_OBJC_ARC NO
)
