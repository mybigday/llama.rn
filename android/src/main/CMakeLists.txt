cmake_minimum_required(VERSION 3.10)

project(llama.rn)

set(CMAKE_ANDROID_STL_TYPE c++_shared)

find_package(ReactAndroid CONFIG REQUIRED)
# Try to find fbjni separately if not in ReactAndroid
find_package(fbjni CONFIG REQUIRED)

set(CMAKE_CXX_STANDARD 17)

set(RNLLAMA_LIB_DIR ${CMAKE_SOURCE_DIR}/../../../cpp)

# Include directories for rnllama headers (needed by JNI wrapper)
include_directories(
    ${RNLLAMA_LIB_DIR}
    ${RNLLAMA_LIB_DIR}/common
)

find_library(LOG_LIB log)

# If building from source, include the rnllama library build
option(RNLLAMA_BUILD_FROM_SOURCE "Build rnllama libraries from source" OFF)

if (RNLLAMA_BUILD_FROM_SOURCE)
    message(STATUS "Building rnllama libraries from source")
    add_subdirectory(rnllama)
endif()

# JNI wrapper source files (always build from source)
set(JNI_SOURCE_FILES
    ${CMAKE_SOURCE_DIR}/RNLlamaJSI.cpp
    ${CMAKE_SOURCE_DIR}/../../../cpp/jsi/RNLlamaJSI.cpp
    ${CMAKE_SOURCE_DIR}/../../../cpp/jsi/JSIContext.cpp
    ${CMAKE_SOURCE_DIR}/../../../cpp/jsi/JSIUtils.cpp
    ${CMAKE_SOURCE_DIR}/../../../cpp/jsi/JSIParams.cpp
    ${CMAKE_SOURCE_DIR}/../../../cpp/jsi/JSITaskManager.cpp
    ${CMAKE_SOURCE_DIR}/../../../cpp/jsi/ThreadPool.cpp
)

function(build_rnllama_jni jni_name rnllama_name arch cpu_flags)
    set(ENABLE_OPENCL OFF)
    if (${rnllama_name} MATCHES ".*_opencl$")
        set(ENABLE_OPENCL ON)
    endif ()

    set(ENABLE_HEXAGON OFF)
    if (${rnllama_name} MATCHES ".*_hexagon.*")
        set(ENABLE_HEXAGON ON)
    endif ()

    # Create the shared library with JNI wrapper
    add_library(
        ${jni_name}
        SHARED
        ${JNI_SOURCE_FILES}
    )

    # Link JSI libraries
    if(${REACT_NATIVE_MINOR_VERSION} GREATER_EQUAL 76)
        target_link_libraries(${jni_name} PRIVATE
            ${LOG_LIB}
            android
            fbjni::fbjni
            ReactAndroid::jsi
            ReactAndroid::reactnative
        )
    else ()
        target_link_libraries(${jni_name} PRIVATE
            ${LOG_LIB}
            android
            fbjni::fbjni
            ReactAndroid::jsi
            ReactAndroid::turbomodulejsijni
            ReactAndroid::react_nativemodule_core
        )
    endif ()

    # Ensure CallInvoker.h path is available even if prefab targets are missing usage includes
    if(RN_PREFAB_REACTNATIVE_INCLUDE)
        target_include_directories(${jni_name} PRIVATE ${RN_PREFAB_REACTNATIVE_INCLUDE})
    endif()

    if (CMAKE_BUILD_TYPE AND CMAKE_BUILD_TYPE STREQUAL "Debug")

        target_compile_options(${jni_name} PRIVATE -DRNLLAMA_ANDROID_ENABLE_LOGGING)
    endif ()

    # Link to rnllama library
    if (RNLLAMA_BUILD_FROM_SOURCE)
        # Link to the shared library we just built in rnllama subdirectory
        target_link_libraries(${jni_name} PRIVATE ${rnllama_name})
    else()
        # Link to prebuilt shared library from jniLibs
        # The prebuilt .so files will be loaded at runtime via System.loadLibrary()
        # We need to tell CMake where to find them for linking
        set(PREBUILT_LIB_PATH ${CMAKE_SOURCE_DIR}/jniLibs/${ANDROID_ABI}/lib${rnllama_name}.so)
        if (EXISTS ${PREBUILT_LIB_PATH})
            target_link_libraries(${jni_name} PRIVATE ${PREBUILT_LIB_PATH})
        else()
            message(FATAL_ERROR "Prebuilt library not found: ${PREBUILT_LIB_PATH}")
        endif()
    endif()

    target_compile_options(${jni_name} PRIVATE -pthread ${cpu_flags})

    # Special handling for OpenCL variant when building from source
    if (RNLLAMA_BUILD_FROM_SOURCE AND ENABLE_OPENCL)
        set(OPENCL_STUB_DIR ${CMAKE_SOURCE_DIR}/../../../bin/${ANDROID_ABI})
        set(OPENCL_STUB ${OPENCL_STUB_DIR}/libOpenCL.so)

        if (EXISTS ${OPENCL_STUB})
            target_link_directories(${jni_name} PRIVATE ${OPENCL_STUB_DIR})
            target_link_libraries(${jni_name} PRIVATE OpenCL)
        else()
            message(WARNING "OpenCL library not found for JNI wrapper. Please build with ./scripts/build-opencl.sh")
        endif()
    endif()
    if (ENABLE_OPENCL)
        target_compile_options(${jni_name} PRIVATE -DLM_GGML_USE_OPENCL)
    endif()

    if (ENABLE_HEXAGON)
        target_compile_options(${jni_name} PRIVATE -DLM_GGML_USE_HEXAGON)
    endif()


    # Optimize JNI wrapper
    target_compile_options(${jni_name} PRIVATE -O3 -DNDEBUG)
    target_compile_options(${jni_name} PRIVATE -fvisibility=hidden -fvisibility-inlines-hidden)
    target_compile_options(${jni_name} PRIVATE -ffunction-sections -fdata-sections)

    target_link_options(${jni_name} PRIVATE -Wl,--gc-sections)
    target_link_options(${jni_name} PRIVATE -Wl,--exclude-libs,ALL)
    target_link_options(${jni_name} PRIVATE -flto)
endfunction()

# Default target (no specific CPU features)
build_rnllama_jni("rnllama_jni" "rnllama" "generic" "")

if (ANDROID_ABI AND ANDROID_ABI STREQUAL "arm64-v8a")
    # ARM64 targets
    # Removing fp16 for now as it leads to issues with some models like deepseek r1 distills
    # https://github.com/mybigday/llama.rn/pull/110#issuecomment-2609918310
    build_rnllama_jni("rnllama_jni_v8" "rnllama_v8" "arm" "-march=armv8-a")
    build_rnllama_jni("rnllama_jni_v8_2" "rnllama_v8_2" "arm" "-march=armv8.2-a")
    build_rnllama_jni("rnllama_jni_v8_2_dotprod" "rnllama_v8_2_dotprod" "arm" "-march=armv8.2-a+dotprod")
    build_rnllama_jni("rnllama_jni_v8_2_i8mm" "rnllama_v8_2_i8mm" "arm" "-march=armv8.2-a+i8mm")
    build_rnllama_jni("rnllama_jni_v8_2_dotprod_i8mm" "rnllama_v8_2_dotprod_i8mm" "arm" "-march=armv8.2-a+dotprod+i8mm")
    build_rnllama_jni("rnllama_jni_v8_2_dotprod_i8mm_hexagon_opencl" "rnllama_v8_2_dotprod_i8mm_hexagon_opencl" "arm" "-march=armv8.2-a+dotprod+i8mm")

    # https://github.com/ggerganov/llama.cpp/blob/master/docs/android.md#cross-compile-using-android-ndk
    # llama.cpp will deal with the cpu features
    # build_rnllama_jni("rnllama_jni_v8_7" "rnllama_v8_7" "arm" "-march=armv8.7-a")
    # TODO: Add support runtime check for cpu features
    # At the moment runtime check is failing.

elseif (ANDROID_ABI AND ANDROID_ABI STREQUAL "x86_64")
    # x86_64 target
    build_rnllama_jni("rnllama_jni_x86_64" "rnllama_x86_64" "x86" "-march=x86-64;-mtune=generic;-msse4.2;-mpopcnt")

endif ()
