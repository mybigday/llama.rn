# Compiler cache (ccache / sccache) for every CMake build of llama.rn.
#
# The library is ~450 translation units and most of them rarely change, so a
# warm compiler cache is worth a lot on CI and on rebuilds. Include this before
# the first target is created: the launcher is picked up when a target is
# defined. Only the Makefile and Ninja generators honor it; the Xcode generator
# ignores compiler launchers.
include_guard(GLOBAL)

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
