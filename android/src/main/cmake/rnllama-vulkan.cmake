# ggml-vulkan for the experimental Android Vulkan variant (RNLLAMA_VULKAN).
#
# Mirrors vendor/llama.cpp/ggml/src/ggml-vulkan/CMakeLists.txt minus the parts
# that assume a desktop build (ggml_add_backend_library, find_package(Vulkan)
# and find_package(SPIRV-Headers)): the NDK has libvulkan.so but an old
# vulkan_core.h and no vulkan.hpp, so the headers come from vendor/.
#
# rnllama_vulkan_setup() generates the SPIR-V shaders once and sets, in the
# caller's scope:
#   RNLLAMA_VULKAN_SOURCES       ggml-vulkan sources + generated shader sources
#   RNLLAMA_VULKAN_INCLUDE_DIRS  generated header dir + vendored Vulkan/SPIR-V headers
#   RNLLAMA_VULKAN_DEFINITIONS   GGML_USE_VULKAN + glslc feature-probe results
#   RNLLAMA_VULKAN_LIBRARY       libvulkan.so to link against
include_guard(GLOBAL)

function(_rnllama_vulkan_find_glslc out)
    # Prefer a Vulkan SDK / system glslc: the NDK's shaderc (v2022.3 in r27)
    # predates several extensions ggml-vulkan probes for.
    find_program(RNLLAMA_GLSLC NAMES glslc
        HINTS $ENV{VULKAN_SDK}/bin
        NO_CMAKE_FIND_ROOT_PATH)
    if (NOT RNLLAMA_GLSLC AND ANDROID_NDK AND ANDROID_HOST_TAG)
        set(_ndk_glslc "${ANDROID_NDK}/shader-tools/${ANDROID_HOST_TAG}/glslc")
        if (EXISTS "${_ndk_glslc}")
            set(RNLLAMA_GLSLC "${_ndk_glslc}" CACHE FILEPATH "glslc used for ggml-vulkan shaders" FORCE)
        endif()
    endif()
    if (NOT RNLLAMA_GLSLC)
        message(FATAL_ERROR
            "RNLLAMA_VULKAN needs glslc. Install the Vulkan SDK (or your distro's glslc) "
            "or set -DRNLLAMA_GLSLC=/path/to/glslc.")
    endif()
    set(${out} "${RNLLAMA_GLSLC}" PARENT_SCOPE)
endfunction()

function(rnllama_vulkan_setup)
    set(_vk_dir     "${RNLLAMA_GGML_VULKAN_DIR}")
    set(_shader_dir "${_vk_dir}/vulkan-shaders")
    set(_out_dir    "${CMAKE_CURRENT_BINARY_DIR}/ggml-vulkan")

    _rnllama_vulkan_find_glslc(_glslc)
    message(STATUS "vulkan: glslc ${_glslc}")

    # Same probes as upstream: shaders using an extension are only generated
    # (and only referenced by ggml-vulkan.cpp) when glslc supports it.
    set(_defs GGML_USE_VULKAN)
    set(_gen_args "")
    foreach(_probe
            "GL_KHR_cooperative_matrix|coopmat.comp|GGML_VULKAN_COOPMAT_GLSLC_SUPPORT"
            "GL_NV_cooperative_matrix2|coopmat2.comp|GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT"
            "GL_NV_cooperative_matrix_decode_vector|coopmat2_decode_vector.comp|GGML_VULKAN_COOPMAT2_DECODE_VECTOR_GLSLC_SUPPORT"
            "GL_EXT_integer_dot_product|integer_dot.comp|GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT"
            "GL_EXT_bfloat16|bfloat16.comp|GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT"
            "GL_EXT_float_e2m1|float_e2m1.comp|GGML_VULKAN_FLOAT_E2M1_GLSLC_SUPPORT"
            "GL_EXT_float_e4m3|float_e4m3.comp|GGML_VULKAN_FLOAT_E4M3_GLSLC_SUPPORT")
        string(REPLACE "|" ";" _probe "${_probe}")
        list(GET _probe 0 _ext)
        list(GET _probe 1 _file)
        list(GET _probe 2 _def)
        execute_process(
            COMMAND ${_glslc} -o - -fshader-stage=compute --target-env=vulkan1.3
                    "${_shader_dir}/feature-tests/${_file}"
            OUTPUT_QUIET
            ERROR_VARIABLE _err)
        if (_err MATCHES "extension not supported: ${_ext}")
            message(STATUS "vulkan: ${_ext} not supported by glslc")
        else()
            message(STATUS "vulkan: ${_ext} supported by glslc")
            list(APPEND _defs ${_def})
            list(APPEND _gen_args -D${_def}=ON)
        endif()
    endforeach()

    # vulkan-shaders-gen is a host tool: build it with the host compiler.
    # Same preference as upstream; on macOS gcc is Apple clang anyway.
    find_program(RNLLAMA_HOST_C_COMPILER   NAMES gcc clang   NO_CMAKE_FIND_ROOT_PATH)
    find_program(RNLLAMA_HOST_CXX_COMPILER NAMES g++ clang++ NO_CMAKE_FIND_ROOT_PATH)
    if (NOT RNLLAMA_HOST_C_COMPILER OR NOT RNLLAMA_HOST_CXX_COMPILER)
        message(FATAL_ERROR "RNLLAMA_VULKAN needs a host C/C++ compiler to build vulkan-shaders-gen")
    endif()
    set(HOST_C_COMPILER   "${RNLLAMA_HOST_C_COMPILER}")
    set(HOST_CXX_COMPILER "${RNLLAMA_HOST_CXX_COMPILER}")
    set(_host_toolchain "${_out_dir}/host-toolchain.cmake")
    configure_file("${_vk_dir}/cmake/host-toolchain.cmake.in" "${_host_toolchain}" @ONLY)

    set(_gen_prefix "${_out_dir}/host")
    set(_gen_cmd    "${_gen_prefix}/vulkan-shaders-gen${CMAKE_HOST_EXECUTABLE_SUFFIX}")
    include(ExternalProject)
    ExternalProject_Add(
        rnllama-vulkan-shaders-gen
        SOURCE_DIR ${_shader_dir}
        BINARY_DIR ${_out_dir}/host-build
        CMAKE_ARGS -DCMAKE_TOOLCHAIN_FILE=${_host_toolchain}
                   -DCMAKE_INSTALL_PREFIX=${_gen_prefix}
                   -DCMAKE_INSTALL_BINDIR=.
                   -DCMAKE_BUILD_TYPE=Release
                   ${_gen_args}
        BUILD_ALWAYS TRUE
        BUILD_BYPRODUCTS ${_gen_cmd}
        INSTALL_COMMAND ${CMAKE_COMMAND} -E env --unset=DESTDIR
                        ${CMAKE_COMMAND} --install . --config Release
    )

    set(_spv_dir "${_out_dir}/spv")
    set(_header  "${_out_dir}/ggml-vulkan-shaders.hpp")
    file(GLOB _gen_sources CONFIGURE_DEPENDS "${_shader_dir}/*.cpp" "${_shader_dir}/*.h")
    file(GLOB _shaders     CONFIGURE_DEPENDS "${_shader_dir}/*.comp")

    add_custom_command(
        OUTPUT  ${_header}
        COMMAND ${_gen_cmd} --output-dir ${_spv_dir} --target-hpp ${_header}
        DEPENDS ${_gen_sources} rnllama-vulkan-shaders-gen
        COMMENT "Generate vulkan shaders header")

    set(_sources
        ${_vk_dir}/ggml-vulkan.cpp
        ${_vk_dir}/ggml-vulkan-buffers.cpp
        ${_vk_dir}/ggml-vulkan-debug.cpp
        ${_header})
    foreach(_shader ${_shaders})
        get_filename_component(_name ${_shader} NAME)
        set(_cpp "${_out_dir}/${_name}.cpp")
        add_custom_command(
            OUTPUT  ${_cpp}
            DEPFILE ${_cpp}.d
            COMMAND ${_gen_cmd}
                --glslc      ${_glslc}
                --source     ${_shader}
                --output-dir ${_spv_dir}
                --target-hpp ${_header}
                --target-cpp ${_cpp}
            DEPENDS ${_shader} ${_gen_sources} rnllama-vulkan-shaders-gen
            COMMENT "Generate vulkan shaders for ${_name}")
        list(APPEND _sources ${_cpp})
    endforeach()

    # ggml-vulkan calls Vulkan 1.1 entry points directly, which libvulkan.so
    # exports from API 28; the library's minSdk is lower, so link the API 28
    # stub explicitly. On older devices the variant fails to load and the Java
    # loader falls back to a CPU variant (it also only tries it on API 28+).
    if (ANDROID_PLATFORM_LEVEL AND ANDROID_PLATFORM_LEVEL LESS 28)
        set(_libvulkan "${CMAKE_SYSROOT}/usr/lib/${CMAKE_LIBRARY_ARCHITECTURE}/28/libvulkan.so")
    else()
        set(_libvulkan vulkan)
    endif()

    set(RNLLAMA_VULKAN_SOURCES ${_sources} PARENT_SCOPE)
    # Vendored headers first so they shadow the sysroot's older vulkan_core.h.
    set(RNLLAMA_VULKAN_INCLUDE_DIRS
        ${_out_dir}
        ${RNLLAMA_VULKAN_HEADERS_DIR}/include
        ${RNLLAMA_SPIRV_HEADERS_DIR}/include
        PARENT_SCOPE)
    set(RNLLAMA_VULKAN_DEFINITIONS ${_defs} PARENT_SCOPE)
    set(RNLLAMA_VULKAN_LIBRARY ${_libvulkan} PARENT_SCOPE)
endfunction()
