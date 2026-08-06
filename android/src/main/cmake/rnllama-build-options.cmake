# Build knobs shared by the two Android CMake entry points:
#   android/src/main/CMakeLists.txt          (AGP / build-from-source)
#   android/src/main/rnllama/CMakeLists.txt  (standalone, scripts/build-android.sh)
include_guard(GLOBAL)

# --- ccache ------------------------------------------------------------------
# Each arm64 build compiles the whole llama.cpp tree once per CPU-feature
# variant, so a warm compiler cache is worth a lot on CI and on rebuilds.
option(RNLLAMA_CCACHE "Use ccache to speed up recompilation" ON)

if (RNLLAMA_CCACHE AND NOT CMAKE_C_COMPILER_LAUNCHER)
    find_program(RNLLAMA_CCACHE_BIN NAMES ccache sccache)
    if (RNLLAMA_CCACHE_BIN)
        # include() runs in the calling scope, so these reach the targets defined
        # there and any add_subdirectory() below it.
        set(CMAKE_C_COMPILER_LAUNCHER   ${RNLLAMA_CCACHE_BIN})
        set(CMAKE_CXX_COMPILER_LAUNCHER ${RNLLAMA_CCACHE_BIN})
        message(STATUS "rnllama: using compiler cache ${RNLLAMA_CCACHE_BIN}")
    else()
        message(STATUS "rnllama: ccache not found, compiling without a compiler cache")
    endif()
endif()

# --- variant selection -------------------------------------------------------
# Every variant is a full copy of the source tree built with different -march
# flags, so building all of them costs ~4 min each on a 4-core machine. An empty
# value (the default) builds them all; CI narrows this down to the variants that
# actually exercise distinct code paths.
set(RNLLAMA_ANDROID_VARIANTS "" CACHE STRING
    "Comma/semicolon-separated subset of rnllama library variants to build (empty = all)")

if (RNLLAMA_ANDROID_VARIANTS)
    message(STATUS "rnllama: restricted to variants ${RNLLAMA_ANDROID_VARIANTS}")
endif()

# `name` is the library variant name (e.g. rnllama_v8_2_dotprod), which the JNI
# wrappers are keyed off as well.
function(rnllama_variant_enabled name result)
    if (RNLLAMA_ANDROID_VARIANTS STREQUAL "")
        set(${result} TRUE PARENT_SCOPE)
        return()
    endif()

    string(REPLACE "," ";" wanted "${RNLLAMA_ANDROID_VARIANTS}")
    if ("${name}" IN_LIST wanted)
        set(${result} TRUE PARENT_SCOPE)
    else()
        set(${result} FALSE PARENT_SCOPE)
    endif()
endfunction()
