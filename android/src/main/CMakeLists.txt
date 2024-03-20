cmake_minimum_required(VERSION 3.10)

project(llama.rn)

set(CMAKE_CXX_STANDARD 11)
set(RNLLAMA_LIB_DIR ${CMAKE_SOURCE_DIR}/../../../cpp)

include_directories(${RNLLAMA_LIB_DIR})

set(
    SOURCE_FILES
    ${RNLLAMA_LIB_DIR}/ggml-alloc.c
    ${RNLLAMA_LIB_DIR}/ggml-backend.c
    ${RNLLAMA_LIB_DIR}/ggml.c
    ${RNLLAMA_LIB_DIR}/ggml-quants.c
    ${RNLLAMA_LIB_DIR}/common.cpp
    ${RNLLAMA_LIB_DIR}/grammar-parser.cpp
    ${RNLLAMA_LIB_DIR}/sampling.cpp
    ${RNLLAMA_LIB_DIR}/unicode.cpp
    ${RNLLAMA_LIB_DIR}/llama.cpp
    ${RNLLAMA_LIB_DIR}/rn-llama.hpp
    ${CMAKE_SOURCE_DIR}/jni.cpp
)

find_library(LOG_LIB log)

function(build_library target_name)
    add_library(
        ${target_name}
        SHARED
        ${SOURCE_FILES}
    )

    target_link_libraries(${target_name} ${LOG_LIB} android)

    target_compile_options(${target_name} PRIVATE -pthread)

    if (${target_name} STREQUAL "rnllama_v8fp16_va")
        target_compile_options(${target_name} PRIVATE -march=armv8.4-a+fp16+dotprod)
    endif ()

    if (${CMAKE_BUILD_TYPE} STREQUAL "Debug")
        target_compile_options(${target_name} PRIVATE -DRNLLAMA_ANDROID_ENABLE_LOGGING)
    endif ()

    # NOTE: If you want to debug the native code, you can uncomment if and endif
    # if (NOT ${CMAKE_BUILD_TYPE} STREQUAL "Debug")

    target_compile_options(${target_name} PRIVATE -O3 -DNDEBUG)
    target_compile_options(${target_name} PRIVATE -fvisibility=hidden -fvisibility-inlines-hidden)
    target_compile_options(${target_name} PRIVATE -ffunction-sections -fdata-sections)

    target_link_options(${target_name} PRIVATE -Wl,--gc-sections)
    target_link_options(${target_name} PRIVATE -Wl,--exclude-libs,ALL)
    target_link_options(${target_name} PRIVATE -flto)

    # endif ()
endfunction()

build_library("rnllama") # Default target

if (${ANDROID_ABI} STREQUAL "arm64-v8a")
    build_library("rnllama_v8fp16_va")
endif ()
