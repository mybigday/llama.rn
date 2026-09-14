cmake_minimum_required(VERSION 3.16)
project(rnllama VERSION 1.0.0 LANGUAGES CXX C ASM)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# iOS specific settings
set(CMAKE_OSX_DEPLOYMENT_TARGET 15.0)
set(CMAKE_XCODE_ATTRIBUTE_ENABLE_BITCODE NO)

# Dependencies and compile options
add_definitions(
    -DNDEBUG
    -DO3
    -DGGML_USE_CPU
    -DGGML_USE_ACCELERATE
    -DGGML_USE_BLAS
    -DGGML_BLAS_USE_ACCELERATE
    -DGGML_USE_METAL
    -DGGML_METAL_EMBED_LIBRARY=1
    # ggml-metal only defines its bundle-lookup Objective-C class when the
    # Metal library is not embedded. Objective-C class names are process-global,
    # so keep it unique per library in case that configuration is ever used.
    -DGGMLMetalClass=RNLlamaGGMLMetalClass
)

if (CMAKE_OSX_ARCHITECTURES STREQUAL "arm64;x86_64")
    add_definitions(-DGGML_CPU_GENERIC)
endif ()

include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/rnllama-sources.cmake)

if (CMAKE_OSX_ARCHITECTURES STREQUAL "arm64")
    set(SOURCE_FILES_ARCH
        ${RNLLAMA_GGML_CPU_ARCH_DIR}/arm/quants.c
        ${RNLLAMA_GGML_CPU_ARCH_DIR}/arm/repack.cpp
    )
endif ()

# Define public headers
set(PUBLIC_HEADERS
    ${RNLLAMA_CPP_DIR}/rn-llama.h
    ${RNLLAMA_CPP_DIR}/rn-completion.h
    ${RNLLAMA_CPP_DIR}/rn-slot.h
    ${RNLLAMA_CPP_DIR}/rn-slot-manager.h
    ${RNLLAMA_CPP_DIR}/rn-tts.h
    ${RNLLAMA_LLAMA_CPP_DIR}/include/llama.h
    ${RNLLAMA_LLAMA_CPP_DIR}/src/llama-impl.h
    ${RNLLAMA_LLAMA_CPP_DIR}/common/speculative.h
    ${RNLLAMA_LLAMA_CPP_DIR}/ggml/include/ggml.h
)

# Create library target
add_library(rnllama SHARED
    ${RNLLAMA_CORE_SOURCES}
    ${RNLLAMA_GGML_METAL_SOURCES}
    ${RNLLAMA_GGML_BLAS_SOURCES}
    ${SOURCE_FILES_ARCH}
)

# Setup include directories
target_include_directories(rnllama
    PUBLIC
        "$<BUILD_INTERFACE:${RNLLAMA_INCLUDE_DIRS}>"
        $<INSTALL_INTERFACE:include>
)

# Keep ggml/gguf out of the framework's export table so a second ggml copy in
# the app (e.g. whisper.rn) can never be resolved through this image. Hidden
# inline visibility also stops dyld from coalescing header-inline functions
# across the two frameworks.
target_compile_options(rnllama PRIVATE $<$<COMPILE_LANGUAGE:CXX,OBJCXX>:-fvisibility-inlines-hidden>)
target_link_options(rnllama PRIVATE
    "LINKER:-unexported_symbols_list,${CMAKE_CURRENT_SOURCE_DIR}/unexported-symbols.txt")

# Link required frameworks
target_link_libraries(rnllama PRIVATE
    "-framework Accelerate"
    "-framework Foundation"
    "-framework Metal"
    "-framework MetalKit"
)

# Set properties for framework
set_target_properties(rnllama PROPERTIES
    MACOSX_FRAMEWORK_IDENTIFIER "com.rnllama"
    MACOSX_FRAMEWORK_BUNDLE_VERSION 1.0.0
    MACOSX_FRAMEWORK_SHORT_VERSION_STRING 1.0.0
    FRAMEWORK TRUE
    FRAMEWORK_VERSION 1.0.0
    VERSION 1.0.0
    PUBLIC_HEADER "${PUBLIC_HEADERS}"
    XCODE_ATTRIBUTE_CLANG_ENABLE_OBJC_ARC NO
    # Generate dSYMs so packaged xcframework slices can be symbolicated in App Store crash logs.
    XCODE_ATTRIBUTE_DEBUG_INFORMATION_FORMAT "dwarf-with-dsym"
    XCODE_ATTRIBUTE_GCC_GENERATE_DEBUGGING_SYMBOLS YES
)
