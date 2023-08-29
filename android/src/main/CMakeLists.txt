cmake_minimum_required(VERSION 3.10)

project(llama.rn)

set(CMAKE_CXX_STANDARD 11)
set(RNLLAMA_LIB_DIR ${CMAKE_SOURCE_DIR}/../../../cpp)

set(
    SOURCE_FILES
    ${RNLLAMA_LIB_DIR}/ggml-alloc.c
    ${RNLLAMA_LIB_DIR}/ggml.c
    ${RNLLAMA_LIB_DIR}/k_quants.c
    ${RNLLAMA_LIB_DIR}/common.cpp
    ${RNLLAMA_LIB_DIR}/grammar-parser.cpp
    ${RNLLAMA_LIB_DIR}/llama.cpp
    ${RNLLAMA_LIB_DIR}/rn-llama.hpp
    ${CMAKE_SOURCE_DIR}/jni.cpp
)

if (${ANDROID_ABI} STREQUAL "arm64-v8a")
    set(RNLLAMA_LIBRARY_NAME rnllama_arm64)
endif ()

add_library(
    ${RNLLAMA_LIBRARY_NAME}
    SHARED
    ${SOURCE_FILES}
)

if (${ANDROID_ABI} STREQUAL "arm64-v8a")
    # target_compile_options(${RNLLAMA_LIBRARY_NAME} PRIVATE -mcpu=native)
endif ()

target_link_libraries(${RNLLAMA_LIBRARY_NAME} log android)
include_directories(${RNLLAMA_LIB_DIR})

target_compile_options(${RNLLAMA_LIBRARY_NAME} PRIVATE -DLM_GGML_USE_K_QUANTS -pthread)

# NOTE: If you want to debug the native code, you can uncomment if and endif
# if (NOT ${CMAKE_BUILD_TYPE} STREQUAL "Debug")

target_compile_options(${RNLLAMA_LIBRARY_NAME} PRIVATE -Ofast -DNDEBUG)
target_compile_options(${RNLLAMA_LIBRARY_NAME} PRIVATE -fvisibility=hidden -fvisibility-inlines-hidden)
target_compile_options(${RNLLAMA_LIBRARY_NAME} PRIVATE -ffunction-sections -fdata-sections)

target_link_options(${RNLLAMA_LIBRARY_NAME} PRIVATE -Wl,--gc-sections)
target_link_options(${RNLLAMA_LIBRARY_NAME} PRIVATE -Wl,--exclude-libs,ALL)
target_link_options(${RNLLAMA_LIBRARY_NAME} PRIVATE -flto)

# endif ()

# target_compile_features(${RNLLAMA_LIBRARY_NAME} PRIVATE exceptions)