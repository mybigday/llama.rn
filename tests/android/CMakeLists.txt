cmake_minimum_required(VERSION 3.16)
project(rnllama_android_tests VERSION 1.0.0 LANGUAGES CXX C)

# On-device (Android arm64) build of the KV-cache-reuse harness, with the OpenCL
# (Adreno GPU) backend so we can verify prompt-state-cache reuse on real GPUs.
# Configure with the NDK toolchain; enable GPU with -DENABLE_OPENCL=ON.
#
#   cmake -S tests/android -B tests/android/build \
#     -DCMAKE_TOOLCHAIN_FILE=$ANDROID_HOME/ndk/<ver>/build/cmake/android.toolchain.cmake \
#     -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-28 -DENABLE_OPENCL=ON
#   cmake --build tests/android/build -j
#
# See tests/android/run_on_devices.sh which drives this end-to-end.

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(ENABLE_OPENCL "Build with the OpenCL (Adreno) backend" ON)
option(ENABLE_HEXAGON "Build with the Hexagon (HTP/NPU) backend" OFF)

# GENERIC CPU kernels (top-level quants.c provides the vec_dot/quantize impls);
# the heavy compute goes to the GPU via RNLLAMA_NGL anyway.
add_definitions(-DNDEBUG -DO3 -DLM_GGML_USE_CPU -DLM_GGML_CPU_GENERIC -D_GNU_SOURCE)

set(REPO_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../..)
set(SOURCE_DIR ${REPO_ROOT}/cpp)

file(GLOB MODEL_FILES        ${SOURCE_DIR}/models/*.cpp)
file(GLOB GGML_CPU_C_FILES   ${SOURCE_DIR}/ggml-cpu/*.c)
file(GLOB GGML_CPU_CPP_FILES ${SOURCE_DIR}/ggml-cpu/*.cpp)
file(GLOB GGML_CPU_AMX_FILES ${SOURCE_DIR}/ggml-cpu/amx/*.cpp)
file(GLOB LLAMA_FILES        ${SOURCE_DIR}/llama*.cpp)
file(GLOB MTMD_FILES         ${SOURCE_DIR}/tools/mtmd/*.cpp)
file(GLOB MTMD_MODEL_FILES   ${SOURCE_DIR}/tools/mtmd/models/*.cpp)
file(GLOB COMMON_FILES       ${SOURCE_DIR}/common/*.cpp)
file(GLOB JINJA_FILES        ${SOURCE_DIR}/common/jinja/*.cpp)

set(RNLLAMA_COMMON_SOURCES
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
    ${GGML_CPU_C_FILES}
    ${GGML_CPU_CPP_FILES}
    ${GGML_CPU_AMX_FILES}
    ${LLAMA_FILES}
    ${SOURCE_DIR}/unicode-data.cpp
    ${SOURCE_DIR}/unicode.cpp
    ${COMMON_FILES}
    ${JINJA_FILES}
    ${SOURCE_DIR}/anyascii.c
    ${MTMD_MODEL_FILES}
    ${MTMD_FILES}
    ${SOURCE_DIR}/rn-llama.cpp
    ${SOURCE_DIR}/rn-completion.cpp
    ${SOURCE_DIR}/rn-tts.cpp
    ${SOURCE_DIR}/rn-slot.cpp
    ${SOURCE_DIR}/rn-slot-manager.cpp
    ${MODEL_FILES}
)

add_executable(kv_cache_reuse_test
    ${CMAKE_CURRENT_SOURCE_DIR}/../kv_cache_reuse_test.cpp
    ${RNLLAMA_COMMON_SOURCES}
)

target_include_directories(kv_cache_reuse_test PRIVATE
    ${SOURCE_DIR}
    ${SOURCE_DIR}/common
    ${SOURCE_DIR}/common/jinja
    ${SOURCE_DIR}/ggml-cpu
    ${SOURCE_DIR}/tools/mtmd
)

# --- OpenCL (Adreno GPU) backend, mirroring android/src/main/rnllama ----------
if(ENABLE_OPENCL)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    target_include_directories(kv_cache_reuse_test PRIVATE
        ${REPO_ROOT}/third_party/OpenCL-Headers)

    # Embed every kernel: ggml-opencl.cpp #includes the ones it needs; the rest
    # are harmless. Generated .cl.h land in the build dir.
    file(GLOB GGML_OPENCL_KERNEL_FILES ${SOURCE_DIR}/ggml-opencl/kernels/*.cl)
    set(GGML_OPENCL_KERNEL_HEADERS "")
    foreach(cl ${GGML_OPENCL_KERNEL_FILES})
        get_filename_component(kname ${cl} NAME_WE)
        # Skip macOS AppleDouble junk (._foo.cl) which isn't valid kernel source.
        if(kname MATCHES "^\\._" OR kname STREQUAL "")
            continue()
        endif()
        set(out ${CMAKE_CURRENT_BINARY_DIR}/${kname}.cl.h)
        add_custom_command(
            OUTPUT ${out}
            COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${SOURCE_DIR}/ggml-opencl/kernels
                    ${Python3_EXECUTABLE} ${SOURCE_DIR}/ggml-opencl/kernels/embed_kernel.py
                    ${cl} ${out}
            DEPENDS ${cl} ${SOURCE_DIR}/ggml-opencl/kernels/embed_kernel.py
            COMMENT "Embedding OpenCL kernel: ${kname}.cl")
        list(APPEND GGML_OPENCL_KERNEL_HEADERS ${out})
    endforeach()

    target_sources(kv_cache_reuse_test PRIVATE
        ${SOURCE_DIR}/ggml-opencl/ggml-opencl.cpp
        ${GGML_OPENCL_KERNEL_HEADERS})
    target_include_directories(kv_cache_reuse_test PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
    target_compile_definitions(kv_cache_reuse_test PRIVATE
        LM_GGML_USE_OPENCL
        LM_GGML_OPENCL_USE_ADRENO_KERNELS
        LM_GGML_OPENCL_EMBED_KERNELS
        LM_GGML_OPENCL_SOA_Q)

    set(OPENCL_STUB_DIR ${REPO_ROOT}/bin/${ANDROID_ABI})
    if(EXISTS ${OPENCL_STUB_DIR}/libOpenCL.so)
        target_link_directories(kv_cache_reuse_test PRIVATE ${OPENCL_STUB_DIR})
        target_link_libraries(kv_cache_reuse_test PRIVATE OpenCL)
    else()
        message(FATAL_ERROR "libOpenCL.so not found in ${OPENCL_STUB_DIR}; run ./scripts/build-opencl.sh")
    endif()
endif()

# --- Hexagon (HTP / NPU) backend, mirroring android/src/main/rnllama ----------
# Host-side sources + QAIC v73 stub; DSP libs (libggml-htp-v*.so) come from
# scripts/build-hexagon-htp.sh and are pushed to the device alongside the binary.
if(ENABLE_HEXAGON)
    set(HEXAGON_SDK_ROOT "$ENV{HEXAGON_SDK_ROOT}")
    if(NOT HEXAGON_SDK_ROOT)
        set(HEXAGON_SDK_ROOT "$ENV{HOME}/.hexagon-sdk/6.4.0.2")
    endif()
    set(CDSPRPC_LIB "${HEXAGON_SDK_ROOT}/ipc/fastrpc/remote/ship/android_aarch64/libcdsprpc.so")
    set(HTP_STUB_DIR "${SOURCE_DIR}/ggml-hexagon/htp/v73")
    if(NOT EXISTS ${CDSPRPC_LIB} OR NOT EXISTS ${HTP_STUB_DIR}/htp_iface_stub.c)
        message(FATAL_ERROR "Hexagon requested but SDK / QAIC stubs missing (run scripts/build-hexagon-htp.sh)")
    endif()
    target_sources(kv_cache_reuse_test PRIVATE
        ${SOURCE_DIR}/ggml-hexagon/ggml-hexagon.cpp
        ${SOURCE_DIR}/ggml-hexagon/htp-drv.cpp
        ${HTP_STUB_DIR}/htp_iface_stub.c)
    target_include_directories(kv_cache_reuse_test PRIVATE
        ${HEXAGON_SDK_ROOT}/incs
        ${HEXAGON_SDK_ROOT}/incs/stddef
        ${HEXAGON_SDK_ROOT}/ipc/fastrpc/rpcmem/inc
        ${HEXAGON_SDK_ROOT}/utils/examples
        ${SOURCE_DIR}/ggml-hexagon
        ${SOURCE_DIR}/ggml-hexagon/htp
        ${HTP_STUB_DIR})
    target_compile_definitions(kv_cache_reuse_test PRIVATE LM_GGML_USE_HEXAGON)
    target_link_libraries(kv_cache_reuse_test PRIVATE ${CDSPRPC_LIB})
    message(STATUS "Hexagon backend enabled for kv_cache_reuse_test")
endif()

find_library(LOG_LIB log)
target_link_libraries(kv_cache_reuse_test PRIVATE ${LOG_LIB} m dl)

# A/B benchmark binary (same library + backend; kernel headers are generated by
# the harness target above).
add_executable(kv_cache_bench
    ${CMAKE_CURRENT_SOURCE_DIR}/../kv_cache_bench.cpp
    ${RNLLAMA_COMMON_SOURCES}
)
target_include_directories(kv_cache_bench PRIVATE
    ${SOURCE_DIR}
    ${SOURCE_DIR}/common
    ${SOURCE_DIR}/common/jinja
    ${SOURCE_DIR}/ggml-cpu
    ${SOURCE_DIR}/tools/mtmd
)
target_compile_definitions(kv_cache_bench PRIVATE KV_BENCH_HAS_STATE_CACHE)
if(ENABLE_OPENCL)
    target_sources(kv_cache_bench PRIVATE ${SOURCE_DIR}/ggml-opencl/ggml-opencl.cpp)
    target_include_directories(kv_cache_bench PRIVATE
        ${REPO_ROOT}/third_party/OpenCL-Headers ${CMAKE_CURRENT_BINARY_DIR})
    target_compile_definitions(kv_cache_bench PRIVATE
        LM_GGML_USE_OPENCL
        LM_GGML_OPENCL_USE_ADRENO_KERNELS
        LM_GGML_OPENCL_EMBED_KERNELS
        LM_GGML_OPENCL_SOA_Q)
    target_link_directories(kv_cache_bench PRIVATE ${OPENCL_STUB_DIR})
    target_link_libraries(kv_cache_bench PRIVATE OpenCL)
    add_dependencies(kv_cache_bench kv_cache_reuse_test)
endif()
if(ENABLE_HEXAGON)
    target_sources(kv_cache_bench PRIVATE
        ${SOURCE_DIR}/ggml-hexagon/ggml-hexagon.cpp
        ${SOURCE_DIR}/ggml-hexagon/htp-drv.cpp
        ${HTP_STUB_DIR}/htp_iface_stub.c)
    target_include_directories(kv_cache_bench PRIVATE
        ${HEXAGON_SDK_ROOT}/incs
        ${HEXAGON_SDK_ROOT}/incs/stddef
        ${HEXAGON_SDK_ROOT}/ipc/fastrpc/rpcmem/inc
        ${HEXAGON_SDK_ROOT}/utils/examples
        ${SOURCE_DIR}/ggml-hexagon
        ${SOURCE_DIR}/ggml-hexagon/htp
        ${HTP_STUB_DIR})
    target_compile_definitions(kv_cache_bench PRIVATE LM_GGML_USE_HEXAGON)
    target_link_libraries(kv_cache_bench PRIVATE ${CDSPRPC_LIB})
    message(STATUS "Hexagon backend enabled for kv_cache_bench")
endif()
target_link_libraries(kv_cache_bench PRIVATE ${LOG_LIB} m dl)
