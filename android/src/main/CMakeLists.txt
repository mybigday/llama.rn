cmake_minimum_required(VERSION 3.10)

project(llama.rn)

set(CMAKE_CXX_STANDARD 17)
set(RNLLAMA_LIB_DIR ${CMAKE_SOURCE_DIR}/../../../cpp)

include_directories(${RNLLAMA_LIB_DIR})

set(
    SOURCE_FILES
    ${RNLLAMA_LIB_DIR}/ggml.c
    ${RNLLAMA_LIB_DIR}/ggml-alloc.c
    ${RNLLAMA_LIB_DIR}/ggml-backend.cpp
    ${RNLLAMA_LIB_DIR}/ggml-backend-reg.cpp
    ${RNLLAMA_LIB_DIR}/ggml-cpu.c
    ${RNLLAMA_LIB_DIR}/ggml-cpu.cpp
    ${RNLLAMA_LIB_DIR}/ggml-cpu-aarch64.cpp
    ${RNLLAMA_LIB_DIR}/ggml-cpu-quants.c
    ${RNLLAMA_LIB_DIR}/ggml-cpu-traits.cpp
    ${RNLLAMA_LIB_DIR}/ggml-opt.cpp
    ${RNLLAMA_LIB_DIR}/ggml-threading.cpp
    ${RNLLAMA_LIB_DIR}/ggml-quants.c
    ${RNLLAMA_LIB_DIR}/log.cpp
    ${RNLLAMA_LIB_DIR}/llama-grammar.cpp
    ${RNLLAMA_LIB_DIR}/llama-sampling.cpp
    ${RNLLAMA_LIB_DIR}/llama-vocab.cpp
    ${RNLLAMA_LIB_DIR}/llama.cpp
    ${RNLLAMA_LIB_DIR}/sampling.cpp
    ${RNLLAMA_LIB_DIR}/unicode-data.cpp
    ${RNLLAMA_LIB_DIR}/unicode.cpp
    ${RNLLAMA_LIB_DIR}/sgemm.cpp
    ${RNLLAMA_LIB_DIR}/common.cpp
    ${RNLLAMA_LIB_DIR}/rn-llama.hpp
    ${RNLLAMA_LIB_DIR}/amx/amx.cpp
    ${RNLLAMA_LIB_DIR}/amx/mmq.cpp
    ${CMAKE_SOURCE_DIR}/jni-utils.h
    ${CMAKE_SOURCE_DIR}/jni.cpp
)

find_library(LOG_LIB log)

function(build_library target_name cpu_flags)
    add_library(
        ${target_name}
        SHARED
        ${SOURCE_FILES}
    )

    target_link_libraries(${target_name} ${LOG_LIB} android)

    target_compile_options(${target_name} PRIVATE -DLM_GGML_USE_CPU -DLM_GGML_USE_CPU_AARCH64 -pthread ${cpu_flags})

    if (${CMAKE_BUILD_TYPE} STREQUAL "Debug")
        target_compile_options(${target_name} PRIVATE -DRNLLAMA_ANDROID_ENABLE_LOGGING)
    endif ()

    # NOTE: If you want to debug the native code, you can uncomment if and endif
    # Note that it will be extremely slow
    # if (NOT ${CMAKE_BUILD_TYPE} STREQUAL "Debug")
        target_compile_options(${target_name} PRIVATE -O3 -DNDEBUG)
        target_compile_options(${target_name} PRIVATE -fvisibility=hidden -fvisibility-inlines-hidden)
        target_compile_options(${target_name} PRIVATE -ffunction-sections -fdata-sections)

        target_link_options(${target_name} PRIVATE -Wl,--gc-sections)
        target_link_options(${target_name} PRIVATE -Wl,--exclude-libs,ALL)
        target_link_options(${target_name} PRIVATE -flto)
    # endif ()
endfunction()

# Default target (no specific CPU features)
build_library("rnllama" "")

if (${ANDROID_ABI} STREQUAL "arm64-v8a")
    # ARM64 targets
    build_library("rnllama_v8_4_fp16_dotprod_sve" "-march=armv8.4-a+fp16+dotprod+sve")
    build_library("rnllama_v8_4_fp16_dotprod_i8mm_sve" "-march=armv8.4-a+fp16+dotprod+i8mm+sve")
    build_library("rnllama_v8_4_fp16_dotprod_i8mm" "-march=armv8.4-a+fp16+dotprod+i8mm")
    build_library("rnllama_v8_4_fp16_dotprod" "-march=armv8.4-a+fp16+dotprod")
    build_library("rnllama_v8_2_fp16_dotprod" "-march=armv8.2-a+fp16+dotprod")
    build_library("rnllama_v8_2_fp16" "-march=armv8.2-a+fp16")
    build_library("rnllama_v8" "-march=armv8-a")

    # https://github.com/ggerganov/llama.cpp/blob/master/docs/android.md#cross-compile-using-android-ndk
    # llama.cpp will deal with the cpu features
    # build_library("rnllama_v8_7" "-march=armv8.7-a")
    # TODO: Add support runtime check for cpu features
    # At the moment runtime check is failing.

elseif (${ANDROID_ABI} STREQUAL "x86_64")
    # x86_64 target
    build_library("rnllama_x86_64" "-march=x86-64" "-mtune=intel" "-msse4.2" "-mpopcnt")

endif ()
