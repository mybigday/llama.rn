# Build knobs shared by the two Android CMake entry points:
#   android/src/main/CMakeLists.txt          (AGP / build-from-source)
#   android/src/main/rnllama/CMakeLists.txt  (standalone, scripts/build-android.sh)
include_guard(GLOBAL)

# --- ccache ------------------------------------------------------------------
include(${CMAKE_CURRENT_LIST_DIR}/../../../../cmake/rnllama-compiler-cache.cmake)

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
