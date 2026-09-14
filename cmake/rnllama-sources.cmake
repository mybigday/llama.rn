# Source and include lists shared by every native build of llama.rn:
#   ios/CMakeLists.txt                       (xcframework)
#   android/src/main/rnllama/CMakeLists.txt  (per-CPU-variant .so)
#   android/src/main/CMakeLists.txt          (JSI glue; include dirs only)
#   tests/CMakeLists.txt, tests/android/CMakeLists.txt
#
# vendor/llama.cpp and vendor/codec.cpp keep their upstream layout,
# so the paths below match upstream's own CMake targets (see vendor/README.md).
# Backends that only some builds enable (Metal, BLAS, OpenCL, Hexagon) are
# exposed as separate variables for the caller to add.
include_guard(GLOBAL)

get_filename_component(RNLLAMA_ROOT_DIR "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
set(RNLLAMA_CPP_DIR            "${RNLLAMA_ROOT_DIR}/cpp")
set(RNLLAMA_LLAMA_CPP_DIR      "${RNLLAMA_ROOT_DIR}/vendor/llama.cpp")
set(RNLLAMA_CODEC_CPP_DIR      "${RNLLAMA_ROOT_DIR}/vendor/codec.cpp")
set(RNLLAMA_OPENCL_HEADERS_DIR "${RNLLAMA_ROOT_DIR}/vendor/OpenCL-Headers")

set(_ggml   "${RNLLAMA_LLAMA_CPP_DIR}/ggml/src")
set(_llama  "${RNLLAMA_LLAMA_CPP_DIR}/src")
set(_common "${RNLLAMA_LLAMA_CPP_DIR}/common")
set(_mtmd   "${RNLLAMA_LLAMA_CPP_DIR}/tools/mtmd")
set(_vendor "${RNLLAMA_LLAMA_CPP_DIR}/vendor")
set(_codec  "${RNLLAMA_CODEC_CPP_DIR}")

# Order matters where basenames collide:
#   common/ before src/               "unicode.h" in common/jinja -> common/unicode.h
#   common/ before ggml/src/ggml-cpu  "common.h"  in rn-*.cpp     -> common/common.h
set(RNLLAMA_INCLUDE_DIRS
    ${RNLLAMA_CPP_DIR}
    ${RNLLAMA_LLAMA_CPP_DIR}                    # "tools/mtmd/mtmd.h"-style includes
    ${RNLLAMA_LLAMA_CPP_DIR}/include
    ${RNLLAMA_LLAMA_CPP_DIR}/ggml/include
    ${_ggml}                                    # ggml-impl.h, ggml-metal/ggml-metal-device.h
    ${_common}
    ${_llama}                                   # llama-impl.h, llama-model.h
    ${_ggml}/ggml-cpu                           # ggml-cpu/amx includes its siblings bare
    ${_mtmd}
    ${_vendor}                                  # nlohmann/, hash/, miniaudio/, stb/
    ${_vendor}/hash                             # sha256.c includes rotate-bits/ bare
    ${_codec}/include
    ${_codec}/common
)

# --- ggml ---------------------------------------------------------------------
file(GLOB RNLLAMA_GGML_CPU_SOURCES CONFIGURE_DEPENDS
    ${_ggml}/ggml-cpu/*.c
    ${_ggml}/ggml-cpu/*.cpp
    ${_ggml}/ggml-cpu/amx/*.cpp
)
# Intel HBM allocator (GGML_USE_CPU_HBM); never enabled here.
list(FILTER RNLLAMA_GGML_CPU_SOURCES EXCLUDE REGEX "/hbm\\.cpp$")
set(RNLLAMA_GGML_SOURCES
    ${_ggml}/ggml.c
    ${_ggml}/ggml-alloc.c
    ${_ggml}/ggml-backend.cpp
    ${_ggml}/ggml-backend-dl.cpp
    ${_ggml}/ggml-backend-meta.cpp
    ${_ggml}/ggml-backend-reg.cpp
    ${_ggml}/ggml-opt.cpp
    ${_ggml}/ggml-threading.cpp
    ${_ggml}/ggml-quants.c
    ${_ggml}/gguf.cpp
    ${RNLLAMA_GGML_CPU_SOURCES}
)
# Per-arch SIMD kernels; callers add ${RNLLAMA_GGML_CPU_ARCH_DIR}/<arm|x86>/{quants.c,repack.cpp}
# when not building with GGML_CPU_GENERIC.
set(RNLLAMA_GGML_CPU_ARCH_DIR "${_ggml}/ggml-cpu/arch")

# Metal kernels are embedded via the ggml-metal-embed-*.s files that
# scripts/bootstrap.sh generates next to the kernels.
file(GLOB RNLLAMA_GGML_METAL_SOURCES CONFIGURE_DEPENDS
    ${_ggml}/ggml-metal/*.cpp
    ${_ggml}/ggml-metal/*.m
    ${_ggml}/ggml-metal/*.s
)
file(GLOB _metal_asm CONFIGURE_DEPENDS ${_ggml}/ggml-metal/*.s)
set_source_files_properties(${_metal_asm} PROPERTIES LANGUAGE ASM)
file(GLOB RNLLAMA_GGML_BLAS_SOURCES CONFIGURE_DEPENDS ${_ggml}/ggml-blas/*.cpp)
set(RNLLAMA_GGML_OPENCL_DIR  "${_ggml}/ggml-opencl")
set(RNLLAMA_GGML_HEXAGON_DIR "${_ggml}/ggml-hexagon")

# --- llama, common, mtmd ------------------------------------------------------
# vendor/llama.cpp also carries what upstream's CMake project needs for
# llama.node (see vendor/README.md); the filters below leave out the parts
# no llama.rn build uses. Keep them in sync with llama-rn.podspec.
file(GLOB RNLLAMA_LLAMA_SOURCES  CONFIGURE_DEPENDS ${_llama}/*.cpp ${_llama}/models/*.cpp)
# Model quantization is not exposed by llama.rn.
list(FILTER RNLLAMA_LLAMA_SOURCES EXCLUDE REGEX "/llama-quant\\.cpp$")
file(GLOB RNLLAMA_COMMON_SOURCES CONFIGURE_DEPENDS ${_common}/*.cpp ${_common}/jinja/*.cpp)
# CLI argument parsing, model download (cpp-httplib), console and subprocess
# helpers are only used by upstream's tools.
list(FILTER RNLLAMA_COMMON_SOURCES EXCLUDE REGEX
    "/(arg|console|debug|download|hf-cache|imatrix-loader|llguidance|preset|subproc)\\.cpp$")
# tools/mtmd/debug is the mtmd debug CLI.
file(GLOB RNLLAMA_MTMD_SOURCES   CONFIGURE_DEPENDS ${_mtmd}/*.cpp ${_mtmd}/models/*.cpp)
# mtmd hashes media inputs with the vendored SHA-256 helper.
set(RNLLAMA_VENDOR_SOURCES
    ${_vendor}/hash/hash.cpp
    ${_vendor}/hash/sha256/sha256.c
)

# --- codec.cpp ----------------------------------------------------------------
file(GLOB RNLLAMA_CODEC_SOURCES CONFIGURE_DEPENDS
    ${_codec}/src/*.cpp
    ${_codec}/src/batch/*.cpp
    ${_codec}/src/lm/*.cpp
    ${_codec}/src/models/*.cpp
    ${_codec}/src/ops/*.cpp
    ${_codec}/src/runtime/*.cpp
    # codec_common API only: tts_runner{,_flow}.cpp are codec.cpp's reference
    # host loop (needs examples/utils and un-prefixed common_* symbols);
    # rn-tts drives its own AR loop via codec_lm_*.
    ${_codec}/common/audio_lm.cpp
)
# WAV/NPY helpers for the host test probes (include dir: ${RNLLAMA_CODEC_CPP_DIR}/examples)
file(GLOB RNLLAMA_CODEC_UTILS_SOURCES CONFIGURE_DEPENDS ${_codec}/examples/utils/*.cpp)

# --- llama.rn -----------------------------------------------------------------
set(RNLLAMA_RN_SOURCES
    ${RNLLAMA_CPP_DIR}/anyascii.c
    ${RNLLAMA_CPP_DIR}/rn-llama.cpp
    ${RNLLAMA_CPP_DIR}/rn-completion.cpp
    ${RNLLAMA_CPP_DIR}/rn-slot.cpp
    ${RNLLAMA_CPP_DIR}/rn-slot-manager.cpp
    ${RNLLAMA_CPP_DIR}/rn-tts.cpp
)

# Everything a CPU-only build of the library needs.
set(RNLLAMA_CORE_SOURCES
    ${RNLLAMA_GGML_SOURCES}
    ${RNLLAMA_LLAMA_SOURCES}
    ${RNLLAMA_COMMON_SOURCES}
    ${RNLLAMA_MTMD_SOURCES}
    ${RNLLAMA_VENDOR_SOURCES}
    ${RNLLAMA_CODEC_SOURCES}
    ${RNLLAMA_RN_SOURCES}
)

unset(_ggml)
unset(_llama)
unset(_common)
unset(_mtmd)
unset(_vendor)
unset(_codec)
unset(_metal_asm)
