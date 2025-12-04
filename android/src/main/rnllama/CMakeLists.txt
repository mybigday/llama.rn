cmake_minimum_required(VERSION 3.10)

project(rnllama)

option(HEXAGON_SDK_ROOT "Hexagon SDK root directory" OFF)
option(HEXAGON_TOOLS_ROOT "Hexagon SDK toolchain directory" OFF)

find_package(Python3 REQUIRED)
set(CMAKE_CXX_STANDARD 17)
# Use CMAKE_CURRENT_SOURCE_DIR to work correctly both standalone and as subdirectory
set(RNLLAMA_LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../../cpp)

include_directories(
    ${RNLLAMA_LIB_DIR}
    ${RNLLAMA_LIB_DIR}/common
    ${RNLLAMA_LIB_DIR}/ggml-cpu
    ${RNLLAMA_LIB_DIR}/tools/mtmd
)

# Collect source files using glob patterns
file(GLOB MODEL_FILES ${RNLLAMA_LIB_DIR}/models/*.cpp)
file(GLOB GGML_CPU_C_FILES ${RNLLAMA_LIB_DIR}/ggml-cpu/*.c)
file(GLOB GGML_CPU_CPP_FILES ${RNLLAMA_LIB_DIR}/ggml-cpu/*.cpp)
file(GLOB GGML_CPU_AMX_FILES ${RNLLAMA_LIB_DIR}/ggml-cpu/amx/*.cpp)
file(GLOB LLAMA_FILES ${RNLLAMA_LIB_DIR}/llama*.cpp)
file(GLOB MTMD_FILES ${RNLLAMA_LIB_DIR}/tools/mtmd/*.cpp)
file(GLOB COMMON_FILES ${RNLLAMA_LIB_DIR}/common/*.cpp)

# Source files for rnllama library (excluding jni.cpp and jni-utils)
set(
    RNLLAMA_SOURCE_FILES
    # Core GGML files
    ${RNLLAMA_LIB_DIR}/ggml.c
    ${RNLLAMA_LIB_DIR}/ggml-alloc.c
    ${RNLLAMA_LIB_DIR}/ggml-backend.cpp
    ${RNLLAMA_LIB_DIR}/ggml-backend-reg.cpp
    ${RNLLAMA_LIB_DIR}/ggml-opt.cpp
    ${RNLLAMA_LIB_DIR}/ggml-threading.cpp
    ${RNLLAMA_LIB_DIR}/ggml-quants.c
    ${RNLLAMA_LIB_DIR}/gguf.cpp

    # GGML CPU files (globbed)
    ${GGML_CPU_C_FILES}
    ${GGML_CPU_CPP_FILES}
    ${GGML_CPU_AMX_FILES}

    # Llama files (globbed)
    ${LLAMA_FILES}
    ${RNLLAMA_LIB_DIR}/unicode-data.cpp
    ${RNLLAMA_LIB_DIR}/unicode.cpp

    # Common utilities
    ${COMMON_FILES}

    ${RNLLAMA_LIB_DIR}/anyascii.c

    # Header-only libraries (needed for build)
    ${RNLLAMA_LIB_DIR}/nlohmann/json.hpp
    ${RNLLAMA_LIB_DIR}/nlohmann/json_fwd.hpp
    ${RNLLAMA_LIB_DIR}/minja/minja.hpp
    ${RNLLAMA_LIB_DIR}/minja/chat-template.hpp

    # Multimodal support (globbed)
    ${MTMD_FILES}

    # React Native llama APIs
    ${RNLLAMA_LIB_DIR}/rn-llama.cpp
    ${RNLLAMA_LIB_DIR}/rn-completion.cpp
    ${RNLLAMA_LIB_DIR}/rn-tts.cpp
    ${RNLLAMA_LIB_DIR}/rn-slot.cpp
    ${RNLLAMA_LIB_DIR}/rn-slot-manager.cpp

    # Model implementations (globbed)
    ${MODEL_FILES}
)

find_library(LOG_LIB log)

# Check if building standalone (for prebuilts) or as subdirectory (for source build)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(RNLLAMA_STANDALONE_BUILD ON)
    message(STATUS "Building rnllama as standalone (for prebuilt distribution)")
else()
    set(RNLLAMA_STANDALONE_BUILD OFF)
    message(STATUS "Building rnllama as subdirectory (will be linked to JNI wrapper)")
endif()

function(build_rnllama_library target_name arch cpu_flags)
    set(ENABLE_OPENCL OFF)
    if (${target_name} MATCHES ".*_opencl$")
        set(ENABLE_OPENCL ON)
    endif ()

    set(ENABLE_HEXAGON OFF)
    if (${target_name} MATCHES ".*_hexagon.*")
        set(ENABLE_HEXAGON ON)
    endif ()

    if (NOT ${arch} STREQUAL "generic")
        set(SOURCE_FILES_ARCH
            ${RNLLAMA_LIB_DIR}/ggml-cpu/arch/${arch}/quants.c
            ${RNLLAMA_LIB_DIR}/ggml-cpu/arch/${arch}/repack.cpp
        )
    endif ()

    # Build as shared library for prebuilt distribution
    add_library(
        ${target_name}
        SHARED
        ${RNLLAMA_SOURCE_FILES}
        ${SOURCE_FILES_ARCH}
    )

    target_link_libraries(${target_name} PRIVATE ${LOG_LIB} android)

    if (${arch} STREQUAL "generic")
        target_compile_options(${target_name} PRIVATE -DLM_GGML_CPU_GENERIC)
    endif ()

    target_compile_options(${target_name} PRIVATE -DLM_GGML_USE_CPU -DLM_GGML_USE_CPU_REPACK -pthread ${cpu_flags} -fvectorize -ffp-model=fast -fno-finite-math-only -flto -D_GNU_SOURCE)

    if (CMAKE_BUILD_TYPE AND CMAKE_BUILD_TYPE STREQUAL "Debug")
        target_compile_options(${target_name} PRIVATE -DRNLLAMA_ANDROID_ENABLE_LOGGING)
    endif ()

    # Hexagon backend support
    # Note: Hexagon requires building DSP libraries with Hexagon SDK toolchain
    # This is a complex process that needs to be done separately
    if (ENABLE_HEXAGON)
        # Hexagon SDK is ALWAYS required to link cdsprpc, even for prebuilt distribution
        set(HEXAGON_SDK_AVAILABLE OFF)
        if (HEXAGON_SDK_ROOT AND HEXAGON_TOOLS_ROOT)
            # Verify libcdsprpc.so exists
            set(CDSPRPC_LIB "${HEXAGON_SDK_ROOT}/ipc/fastrpc/remote/ship/android_aarch64/libcdsprpc.so")
            if (EXISTS ${CDSPRPC_LIB})
                set(HEXAGON_SDK_AVAILABLE ON)
                message(STATUS "Hexagon SDK verified and available")
            else()
                message(FATAL_ERROR
                    "Hexagon SDK found but libcdsprpc.so is missing at ${CDSPRPC_LIB}.\n"
                    "The SDK installation may be incomplete. Run 'npm run bootstrap' to reinstall.")
            endif()
        else()
            message(FATAL_ERROR
                "Hexagon SDK is required to build ${target_name} but not found.")
        endif()

        if (HEXAGON_SDK_AVAILABLE)
            # Add host-side hexagon sources
            target_sources(${target_name} PRIVATE
                ${RNLLAMA_LIB_DIR}/ggml-hexagon/ggml-hexagon.cpp
                ${RNLLAMA_LIB_DIR}/ggml-hexagon/htp-utils.c
            )

            # Use pre-built stub
            set(HEXAGON_PREBUILT_STUB "${RNLLAMA_LIB_DIR}/ggml-hexagon/htp/v73/htp_iface_stub.c")
            if (EXISTS ${HEXAGON_PREBUILT_STUB})
                target_sources(${target_name} PRIVATE ${HEXAGON_PREBUILT_STUB})
            else()
                message(WARNING "Pre-built HTP stub not found: ${HEXAGON_PREBUILT_STUB}")
            endif()

            # Use pre-built IDL headers from build-hexagon-htp directory
            set(HEXAGON_PREBUILT_HTP_DIR "${RNLLAMA_LIB_DIR}/ggml-hexagon/htp/v73/")

            target_include_directories(${target_name} PRIVATE
                ${HEXAGON_SDK_ROOT}/incs
                ${HEXAGON_SDK_ROOT}/incs/stddef
                ${HEXAGON_SDK_ROOT}/ipc/fastrpc/rpcmem/inc
                ${HEXAGON_SDK_ROOT}/utils/examples
                ${RNLLAMA_LIB_DIR}/ggml-hexagon
                ${RNLLAMA_LIB_DIR}/ggml-hexagon/htp
                ${HEXAGON_PREBUILT_HTP_DIR}
                ${CMAKE_CURRENT_BINARY_DIR}
            )

            # Link Hexagon SDK libraries
            # These are stub libraries that communicate with the DSP
            # libcdsprpc.so contains all needed symbols: remote_handle64_*, dspqueue_*, rpcmem_*
            # Note: CDSPRPC_LIB is already verified to exist above
            target_link_libraries(${target_name} PRIVATE ${CDSPRPC_LIB})
            message(STATUS "Linking Hexagon cdsprpc library: ${CDSPRPC_LIB}")

            target_compile_options(${target_name} PRIVATE
                -DLM_GGML_USE_HEXAGON
            )

            message(STATUS "Hexagon backend enabled for ${target_name}")
            if (NOT RNLLAMA_STANDALONE_BUILD)
                message(STATUS "  Note: DSP libraries (libggml-htp-*.so) must be built separately")
                message(STATUS "  Run: scripts/build-hexagon-htp.sh to build DSP libraries")
            endif()
        else()
            # Only show warning when building from source
            if (NOT RNLLAMA_STANDALONE_BUILD)
                message(WARNING
                    "Hexagon SDK not found. Hexagon backend will not be built.")
            endif()
        endif()
    endif ()

    # OpenCL only for the special target
    if (ENABLE_OPENCL)
        include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../../../third_party/OpenCL-Headers)

        set(GGML_OPENCL_KERNELS
            add
            add_id
            argsort
            clamp
            cpy
            cvt
            diag_mask_inf
            div
            gelu
            gemv_noshuffle_general
            gemv_noshuffle
            get_rows
            glu
            group_norm
            im2col_f32
            im2col_f16
            mean
            mul_mat_Ab_Bi_8x4
            mul_mv_f16_f16
            mul_mv_f16_f32_1row
            mul_mv_f16_f32_l4
            mul_mv_f16_f32
            mul_mv_f32_f32
            mul_mv_q4_0_f32
            mul_mv_q4_0_f32_v
            mul_mv_q4_0_f32_8x_flat
            mul_mv_q4_0_f32_1d_8x_flat
            mul_mv_q4_0_f32_1d_16x_flat
            mul_mv_q6_k
            mul_mv_q8_0_f32
            mul_mv_q8_0_f32_flat
            mul_mv_mxfp4_f32
            mul_mv_mxfp4_f32_flat
            mul_mv_id_q4_0_f32_8x_flat
            mul_mv_id_q8_0_f32
            mul_mv_id_q8_0_f32_flat
            mul_mv_id_mxfp4_f32
            mul_mv_id_mxfp4_f32_flat
            gemm_moe_mxfp4_f32
            gemv_moe_mxfp4_f32
            mul_mm_f32_f32_l4_lm
            mul_mm_f16_f32_l4_lm
            mul_mm_q8_0_f32_l4_lm
            mul
            norm
            relu
            rms_norm
            rope
            scale
            set_rows
            sigmoid
            silu
            softmax_4_f32
            softmax_4_f16
            softmax_f32
            softmax_f16
            sqr
            sqrt
            ssm_conv
            sub
            sum_rows
            transpose
            concat
            tsembd
            upscale
            tanh
            pad
            repeat
            mul_mat_f16_f32
            mul_mm_f16_f32_kq_kqv
            conv2d
            conv2d_f16_f32
            flash_attn_f32_f16
            flash_attn_f16
            flash_attn_f32
        )
        set(GGML_OPENCL_KERNEL_DIR ${RNLLAMA_LIB_DIR}/ggml-opencl/kernels)
        set(GGML_OPENCL_KERNEL_HEADERS "")

        foreach(kernel ${GGML_OPENCL_KERNELS})
            set(input_file ${GGML_OPENCL_KERNEL_DIR}/${kernel}.cl)
            set(output_file ${CMAKE_CURRENT_BINARY_DIR}/${kernel}.cl.h)
            message(STATUS "opencl: embedding kernel ${kernel}")
            add_custom_command(
                OUTPUT ${output_file}
                COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/..
                        ${Python3_EXECUTABLE} ${RNLLAMA_LIB_DIR}/ggml-opencl/kernels/embed_kernel.py
                        ${input_file} ${output_file}
                DEPENDS ${input_file} ${RNLLAMA_LIB_DIR}/ggml-opencl/kernels/embed_kernel.py
                COMMENT "Embedding OpenCL kernel: ${kernel}.cl"
            )

            list(APPEND GGML_OPENCL_KERNEL_HEADERS ${output_file})
        endforeach()

        set(OPENCL_STUB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../../bin/${ANDROID_ABI})
        set(OPENCL_STUB     ${OPENCL_STUB_DIR}/libOpenCL.so)

        if (EXISTS ${OPENCL_STUB})
            # Keep using -lOpenCL so we don't create a packagable .so target.
            target_link_directories(${target_name} PRIVATE ${OPENCL_STUB_DIR})
            target_link_libraries(${target_name} PRIVATE OpenCL)
        else()
            message(WARNING
                "OpenCL library not found. Please build with ./scripts/build-opencl.sh")
        endif()

        target_include_directories(${target_name} PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
        target_sources(${target_name} PRIVATE
            ${RNLLAMA_LIB_DIR}/ggml-opencl/ggml-opencl.cpp
            ${GGML_OPENCL_KERNEL_HEADERS}
        )
        target_compile_options(${target_name} PRIVATE
            -DLM_GGML_USE_OPENCL
            -DLM_GGML_OPENCL_USE_ADRENO_KERNELS
            -DLM_GGML_OPENCL_EMBED_KERNELS
            -DLM_GGML_OPENCL_SOA_Q
        )
    endif ()

    # Optimize for size and performance
    target_compile_options(${target_name} PRIVATE -O3 -DNDEBUG)
    target_compile_options(${target_name} PRIVATE -ffunction-sections -fdata-sections)

    target_link_options(${target_name} PRIVATE -Wl,--gc-sections)
    target_link_options(${target_name} PRIVATE -flto)
endfunction()

# Default target (no specific CPU features)
build_rnllama_library("rnllama" "generic" "")

if (ANDROID_ABI AND ANDROID_ABI STREQUAL "arm64-v8a")
    # ARM64 targets
    build_rnllama_library("rnllama_v8" "arm" "-march=armv8-a")
    build_rnllama_library("rnllama_v8_2" "arm" "-march=armv8.2-a")
    build_rnllama_library("rnllama_v8_2_dotprod" "arm" "-march=armv8.2-a+dotprod")
    build_rnllama_library("rnllama_v8_2_i8mm" "arm" "-march=armv8.2-a+i8mm")
    build_rnllama_library("rnllama_v8_2_dotprod_i8mm" "arm" "-march=armv8.2-a+dotprod+i8mm")
    build_rnllama_library("rnllama_v8_2_dotprod_i8mm_hexagon_opencl" "arm" "-march=armv8.2-a+dotprod+i8mm")

elseif (ANDROID_ABI AND ANDROID_ABI STREQUAL "x86_64")
    # x86_64 target
    build_rnllama_library("rnllama_x86_64" "x86" "-march=x86-64;-mtune=generic;-msse4.2;-mpopcnt")

endif ()
