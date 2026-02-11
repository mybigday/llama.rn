#!/bin/bash -e

ROOT_DIR=$(pwd)
OS=$(uname)

LLAMA_DIR=third_party/llama.cpp
CPP_DIR="$ROOT_DIR/cpp"
SRC_DIR="$ROOT_DIR/src"

git submodule init "$LLAMA_DIR"
git submodule update --recursive "$LLAMA_DIR"

# Hexagon SDK setup for Android builds
echo ""
echo "=========================================="
echo "Hexagon SDK Setup"
echo "=========================================="
echo ""

# Check if Docker is available and recommend it
if command -v docker &> /dev/null && docker info &> /dev/null 2>&1; then
  echo "✓ Docker is available!"
  echo ""
  echo "For Hexagon builds, we recommend using Docker for consistent builds."
  echo "Docker provides a pre-configured environment with all dependencies."
  echo ""
  echo "Build commands:"
  echo "  ./scripts/build-android-docker.sh    - Build everything with Docker"
  echo "  ./scripts/build-hexagon-htp.sh       - Build HTP libraries (auto-detects Docker)"
  echo ""

  # Pull Docker image in background
  DOCKER_IMAGE="ghcr.io/snapdragon-toolchain/arm64-android:v0.3"
  if ! docker image inspect "$DOCKER_IMAGE" &> /dev/null; then
    echo "Pulling Docker image in background..."
    echo "  Image: $DOCKER_IMAGE"
    docker pull "$DOCKER_IMAGE" &
    DOCKER_PULL_PID=$!
    echo "  (Pull process running in background, PID: $DOCKER_PULL_PID)"
  else
    echo "✓ Docker image already present: $DOCKER_IMAGE"
  fi
  echo ""
else
  echo "Docker not available. You can:"
  echo "  1. Install Docker for consistent builds (recommended)"
  echo "  2. Install Hexagon SDK manually for native Linux builds"
  echo ""
fi

# Download and setup Hexagon SDK (for all platforms)
# On macOS: Needed for libcdsprpc.so linking when building Android libraries
# On Linux: Can be used for native builds without Docker
HEXAGON_SDK_VERSION="6.4.0.2"
HEXAGON_TOOLS_VERSION="19.0.04"
HEXAGON_INSTALL_DIR="${HEXAGON_INSTALL_DIR:-$HOME/.hexagon-sdk}"

if [ ! -d "$HEXAGON_INSTALL_DIR/$HEXAGON_SDK_VERSION" ]; then
  echo "Downloading Hexagon SDK v${HEXAGON_SDK_VERSION}..."
  echo ""

  if [ "$OS" = "Darwin" ]; then
    echo "Note: SDK tools won't run on macOS, but libcdsprpc.so is needed for linking"
  fi
  echo ""

  TEMP_DIR=$(mktemp -d)
  cd "$TEMP_DIR"

  curl -L -o hex-sdk.tar.gz \
    "https://github.com/snapdragon-toolchain/hexagon-sdk/releases/download/v${HEXAGON_SDK_VERSION}/hexagon-sdk-v${HEXAGON_SDK_VERSION}-amd64-lnx.tar.xz"

  echo "Extracting Hexagon SDK..."
  mkdir -p "$HEXAGON_INSTALL_DIR"
  tar -xaf hex-sdk.tar.gz -C "$HEXAGON_INSTALL_DIR"

  cd "$ROOT_DIR"
  rm -rf "$TEMP_DIR"

  echo "Hexagon SDK installed to: $HEXAGON_INSTALL_DIR/$HEXAGON_SDK_VERSION"
  echo ""
  echo "The build scripts will automatically detect and use the SDK."
  echo ""
  echo "To build with Docker (recommended):"
  echo "  ./scripts/build-android-docker.sh"
  echo ""
  if [ "$OS" != "Darwin" ]; then
    echo "Or build natively on Linux:"
    echo "  USE_DOCKER=no ./scripts/build-hexagon-htp.sh"
    echo "  npm run build:android-libs"
    echo ""
  fi
else
  echo "✓ Hexagon SDK installed: $HEXAGON_INSTALL_DIR/$HEXAGON_SDK_VERSION"
  echo ""
fi

echo "=========================================="
echo ""

# ggml api
cp ./$LLAMA_DIR/ggml/include/ggml.h ./cpp/ggml.h
cp ./$LLAMA_DIR/ggml/include/ggml-alloc.h ./cpp/ggml-alloc.h
cp ./$LLAMA_DIR/ggml/include/ggml-backend.h ./cpp/ggml-backend.h
cp ./$LLAMA_DIR/ggml/include/ggml-cpu.h ./cpp/ggml-cpu.h
cp ./$LLAMA_DIR/ggml/include/ggml-cpp.h ./cpp/ggml-cpp.h
cp ./$LLAMA_DIR/ggml/include/ggml-opt.h ./cpp/ggml-opt.h
cp ./$LLAMA_DIR/ggml/include/ggml-metal.h ./cpp/ggml-metal.h
cp ./$LLAMA_DIR/ggml/include/ggml-opencl.h ./cpp/ggml-opencl.h
cp ./$LLAMA_DIR/ggml/include/ggml-blas.h ./cpp/ggml-blas.h
cp ./$LLAMA_DIR/ggml/include/ggml-hexagon.h ./cpp/ggml-hexagon.h
cp ./$LLAMA_DIR/ggml/include/gguf.h ./cpp/gguf.h

cp -r ./$LLAMA_DIR/ggml/src/ggml-metal ./cpp/
rm ./cpp/ggml-metal/CMakeLists.txt
# Keep ggml-metal.metal for runtime compilation
# rm ./cpp/ggml-metal/ggml-metal.metal

# Embed headers into ggml-metal.metal for runtime compilation
# This allows the .metal file to be compiled at runtime without needing external header files
echo "Embedding headers into ggml-metal.metal..."
METAL_SOURCE="./cpp/ggml-metal/ggml-metal.metal"
METAL_TMP="./cpp/ggml-metal/ggml-metal.metal.tmp1"
COMMON_HEADER="./$LLAMA_DIR/ggml/src/ggml-common.h"
IMPL_HEADER="./cpp/ggml-metal/ggml-metal-impl.h"

# Step 1: Replace the entire conditional block with just the embedded header content
# Find the line with #if defined(GGML_METAL_EMBED_LIBRARY), replace __embed__ placeholder with header,
# and remove everything from #else to #endif (inclusive)
awk '
/^#if defined\(GGML_METAL_EMBED_LIBRARY\)/ { skip=1; next }
/__embed_ggml-common.h__/ {
    system("cat '"$COMMON_HEADER"'")
    next
}
/^#else/ && skip { skip_else=1; next }
/^#endif/ && skip_else { skip=0; skip_else=0; next }
!skip { print }
' < "$METAL_SOURCE" > "$METAL_TMP"

# Step 2: Embed ggml-metal-impl.h by replacing the #include with file contents
sed -e '/#include "ggml-metal-impl.h"/r '"$IMPL_HEADER" -e '/#include "ggml-metal-impl.h"/d' < "$METAL_TMP" > "$METAL_SOURCE"

rm -f "$METAL_TMP"
echo "Headers embedded successfully"

cp -r ./$LLAMA_DIR/ggml/src/ggml-blas ./cpp/
rm ./cpp/ggml-blas/CMakeLists.txt

cp -r ./$LLAMA_DIR/ggml/src/ggml-opencl ./cpp/
rm ./cpp/ggml-opencl/CMakeLists.txt

cp -r ./$LLAMA_DIR/ggml/src/ggml-hexagon ./cpp/
# Keep CMakeLists.txt for hexagon as it's needed for building HTP components

cp ./$LLAMA_DIR/ggml/src/ggml-cpu/ggml-cpu.c ./cpp/ggml-cpu/ggml-cpu.c
cp ./$LLAMA_DIR/ggml/src/ggml-cpu/ggml-cpu.cpp ./cpp/ggml-cpu/ggml-cpu.cpp
cp ./$LLAMA_DIR/ggml/src/ggml-cpu/ggml-cpu-impl.h ./cpp/ggml-cpu/ggml-cpu-impl.h
cp ./$LLAMA_DIR/ggml/src/ggml-cpu/quants.h ./cpp/ggml-cpu/quants.h
cp ./$LLAMA_DIR/ggml/src/ggml-cpu/quants.c ./cpp/ggml-cpu/quants.c
cp ./$LLAMA_DIR/ggml/src/ggml-cpu/arch-fallback.h ./cpp/ggml-cpu/arch-fallback.h
cp ./$LLAMA_DIR/ggml/src/ggml-cpu/repack.cpp ./cpp/ggml-cpu/repack.cpp
cp ./$LLAMA_DIR/ggml/src/ggml-cpu/repack.h ./cpp/ggml-cpu/repack.h
cp ./$LLAMA_DIR/ggml/src/ggml-cpu/traits.h ./cpp/ggml-cpu/traits.h
cp ./$LLAMA_DIR/ggml/src/ggml-cpu/traits.cpp ./cpp/ggml-cpu/traits.cpp
cp ./$LLAMA_DIR/ggml/src/ggml-cpu/common.h ./cpp/ggml-cpu/common.h

cp ./$LLAMA_DIR/ggml/src/ggml-cpu/unary-ops.h ./cpp/ggml-cpu/unary-ops.h
cp ./$LLAMA_DIR/ggml/src/ggml-cpu/unary-ops.cpp ./cpp/ggml-cpu/unary-ops.cpp
cp ./$LLAMA_DIR/ggml/src/ggml-cpu/binary-ops.h ./cpp/ggml-cpu/binary-ops.h
cp ./$LLAMA_DIR/ggml/src/ggml-cpu/binary-ops.cpp ./cpp/ggml-cpu/binary-ops.cpp
cp ./$LLAMA_DIR/ggml/src/ggml-cpu/vec.h ./cpp/ggml-cpu/vec.h
cp ./$LLAMA_DIR/ggml/src/ggml-cpu/vec.cpp ./cpp/ggml-cpu/vec.cpp
cp ./$LLAMA_DIR/ggml/src/ggml-cpu/simd-mappings.h ./cpp/ggml-cpu/simd-mappings.h
cp ./$LLAMA_DIR/ggml/src/ggml-cpu/ops.h ./cpp/ggml-cpu/ops.h
cp ./$LLAMA_DIR/ggml/src/ggml-cpu/ops.cpp ./cpp/ggml-cpu/ops.cpp

cp -r ./$LLAMA_DIR/ggml/src/ggml-cpu/amx ./cpp/ggml-cpu/
mkdir -p ./cpp/ggml-cpu/arch
cp -r ./$LLAMA_DIR/ggml/src/ggml-cpu/arch/arm ./cpp/ggml-cpu/arch/
cp -r ./$LLAMA_DIR/ggml/src/ggml-cpu/arch/x86 ./cpp/ggml-cpu/arch/

cp ./$LLAMA_DIR/ggml/src/ggml.c ./cpp/ggml.c
cp ./$LLAMA_DIR/ggml/src/ggml-impl.h ./cpp/ggml-impl.h
cp ./$LLAMA_DIR/ggml/src/ggml-alloc.c ./cpp/ggml-alloc.c
cp ./$LLAMA_DIR/ggml/src/ggml-backend.cpp ./cpp/ggml-backend.cpp
cp ./$LLAMA_DIR/ggml/src/ggml-backend-impl.h ./cpp/ggml-backend-impl.h
cp ./$LLAMA_DIR/ggml/src/ggml-backend-reg.cpp ./cpp/ggml-backend-reg.cpp
cp ./$LLAMA_DIR/ggml/src/ggml-backend-dl.h ./cpp/ggml-backend-dl.h
cp ./$LLAMA_DIR/ggml/src/ggml-backend-dl.cpp ./cpp/ggml-backend-dl.cpp
cp ./$LLAMA_DIR/ggml/src/ggml-common.h ./cpp/ggml-common.h
cp ./$LLAMA_DIR/ggml/src/ggml-opt.cpp ./cpp/ggml-opt.cpp
cp ./$LLAMA_DIR/ggml/src/ggml-quants.h ./cpp/ggml-quants.h
cp ./$LLAMA_DIR/ggml/src/ggml-quants.c ./cpp/ggml-quants.c
cp ./$LLAMA_DIR/ggml/src/ggml-threading.cpp ./cpp/ggml-threading.cpp
cp ./$LLAMA_DIR/ggml/src/ggml-threading.h ./cpp/ggml-threading.h
cp ./$LLAMA_DIR/ggml/src/gguf.cpp ./cpp/gguf.cpp

# llama api
cp ./$LLAMA_DIR/include/llama.h ./cpp/llama.h
cp ./$LLAMA_DIR/include/llama-cpp.h ./cpp/llama-cpp.h
rm -rf ./cpp/models
cp -r ./$LLAMA_DIR/src/models ./cpp/models
cp ./$LLAMA_DIR/src/llama.cpp ./cpp/llama.cpp
cp ./$LLAMA_DIR/src/llama-chat.h ./cpp/llama-chat.h
cp ./$LLAMA_DIR/src/llama-chat.cpp ./cpp/llama-chat.cpp
cp ./$LLAMA_DIR/src/llama-context.h ./cpp/llama-context.h
cp ./$LLAMA_DIR/src/llama-context.cpp ./cpp/llama-context.cpp
cp ./$LLAMA_DIR/src/llama-mmap.h ./cpp/llama-mmap.h
cp ./$LLAMA_DIR/src/llama-mmap.cpp ./cpp/llama-mmap.cpp
cp ./$LLAMA_DIR/src/llama-model-loader.h ./cpp/llama-model-loader.h
cp ./$LLAMA_DIR/src/llama-model-loader.cpp ./cpp/llama-model-loader.cpp
cp ./$LLAMA_DIR/src/llama-model-saver.h ./cpp/llama-model-saver.h
cp ./$LLAMA_DIR/src/llama-model-saver.cpp ./cpp/llama-model-saver.cpp
cp ./$LLAMA_DIR/src/llama-model.h ./cpp/llama-model.h
cp ./$LLAMA_DIR/src/llama-model.cpp ./cpp/llama-model.cpp
cp ./$LLAMA_DIR/src/llama-kv-cells.h ./cpp/llama-kv-cells.h
cp ./$LLAMA_DIR/src/llama-kv-cache.h ./cpp/llama-kv-cache.h
cp ./$LLAMA_DIR/src/llama-kv-cache.cpp ./cpp/llama-kv-cache.cpp
cp ./$LLAMA_DIR/src/llama-kv-cache-iswa.h ./cpp/llama-kv-cache-iswa.h
cp ./$LLAMA_DIR/src/llama-kv-cache-iswa.cpp ./cpp/llama-kv-cache-iswa.cpp
cp ./$LLAMA_DIR/src/llama-memory-hybrid.h ./cpp/llama-memory-hybrid.h
cp ./$LLAMA_DIR/src/llama-memory-hybrid.cpp ./cpp/llama-memory-hybrid.cpp
cp ./$LLAMA_DIR/src/llama-memory-hybrid-iswa.h ./cpp/llama-memory-hybrid-iswa.h
cp ./$LLAMA_DIR/src/llama-memory-hybrid-iswa.cpp ./cpp/llama-memory-hybrid-iswa.cpp
cp ./$LLAMA_DIR/src/llama-memory-recurrent.h ./cpp/llama-memory-recurrent.h
cp ./$LLAMA_DIR/src/llama-memory-recurrent.cpp ./cpp/llama-memory-recurrent.cpp
cp ./$LLAMA_DIR/src/llama-adapter.h ./cpp/llama-adapter.h
cp ./$LLAMA_DIR/src/llama-adapter.cpp ./cpp/llama-adapter.cpp
cp ./$LLAMA_DIR/src/llama-arch.h ./cpp/llama-arch.h
cp ./$LLAMA_DIR/src/llama-arch.cpp ./cpp/llama-arch.cpp
cp ./$LLAMA_DIR/src/llama-batch.h ./cpp/llama-batch.h
cp ./$LLAMA_DIR/src/llama-batch.cpp ./cpp/llama-batch.cpp
cp ./$LLAMA_DIR/src/llama-cparams.h ./cpp/llama-cparams.h
cp ./$LLAMA_DIR/src/llama-cparams.cpp ./cpp/llama-cparams.cpp
cp ./$LLAMA_DIR/src/llama-hparams.h ./cpp/llama-hparams.h
cp ./$LLAMA_DIR/src/llama-hparams.cpp ./cpp/llama-hparams.cpp
cp ./$LLAMA_DIR/src/llama-impl.h ./cpp/llama-impl.h
cp ./$LLAMA_DIR/src/llama-impl.cpp ./cpp/llama-impl.cpp

cp ./$LLAMA_DIR/src/llama-vocab.h ./cpp/llama-vocab.h
cp ./$LLAMA_DIR/src/llama-vocab.cpp ./cpp/llama-vocab.cpp
cp ./$LLAMA_DIR/src/llama-grammar.h ./cpp/llama-grammar.h
cp ./$LLAMA_DIR/src/llama-grammar.cpp ./cpp/llama-grammar.cpp
cp ./$LLAMA_DIR/src/llama-sampler.h ./cpp/llama-sampler.h
cp ./$LLAMA_DIR/src/llama-sampler.cpp ./cpp/llama-sampler.cpp

cp ./$LLAMA_DIR/src/unicode.h ./cpp/unicode.h
cp ./$LLAMA_DIR/src/unicode.cpp ./cpp/unicode.cpp
cp ./$LLAMA_DIR/src/unicode-data.h ./cpp/unicode-data.h
cp ./$LLAMA_DIR/src/unicode-data.cpp ./cpp/unicode-data.cpp

cp ./$LLAMA_DIR/src/llama-graph.h ./cpp/llama-graph.h
cp ./$LLAMA_DIR/src/llama-graph.cpp ./cpp/llama-graph.cpp
cp ./$LLAMA_DIR/src/llama-io.h ./cpp/llama-io.h
cp ./$LLAMA_DIR/src/llama-io.cpp ./cpp/llama-io.cpp
cp ./$LLAMA_DIR/src/llama-memory.h ./cpp/llama-memory.h
cp ./$LLAMA_DIR/src/llama-memory.cpp ./cpp/llama-memory.cpp

mkdir -p ./cpp/common
cp ./$LLAMA_DIR/common/log.h ./cpp/common/log.h
cp ./$LLAMA_DIR/common/log.cpp ./cpp/common/log.cpp
cp ./$LLAMA_DIR/common/common.h ./cpp/common/common.h
cp ./$LLAMA_DIR/common/common.cpp ./cpp/common/common.cpp
cp ./$LLAMA_DIR/common/sampling.h ./cpp/common/sampling.h
cp ./$LLAMA_DIR/common/sampling.cpp ./cpp/common/sampling.cpp
cp ./$LLAMA_DIR/common/json-schema-to-grammar.h ./cpp/common/json-schema-to-grammar.h
cp ./$LLAMA_DIR/common/json-schema-to-grammar.cpp ./cpp/common/json-schema-to-grammar.cpp
cp ./$LLAMA_DIR/common/json-partial.h ./cpp/common/json-partial.h
cp ./$LLAMA_DIR/common/json-partial.cpp ./cpp/common/json-partial.cpp
cp ./$LLAMA_DIR/common/regex-partial.h ./cpp/common/regex-partial.h
cp ./$LLAMA_DIR/common/regex-partial.cpp ./cpp/common/regex-partial.cpp
cp ./$LLAMA_DIR/common/chat.h ./cpp/common/chat.h
cp ./$LLAMA_DIR/common/chat.cpp ./cpp/common/chat.cpp
cp ./$LLAMA_DIR/common/chat-parser.h ./cpp/common/chat-parser.h
cp ./$LLAMA_DIR/common/chat-parser.cpp ./cpp/common/chat-parser.cpp
cp ./$LLAMA_DIR/common/chat-parser-xml-toolcall.h ./cpp/common/chat-parser-xml-toolcall.h
cp ./$LLAMA_DIR/common/chat-parser-xml-toolcall.cpp ./cpp/common/chat-parser-xml-toolcall.cpp
cp ./$LLAMA_DIR/common/chat-peg-parser.h ./cpp/common/chat-peg-parser.h
cp ./$LLAMA_DIR/common/chat-peg-parser.cpp ./cpp/common/chat-peg-parser.cpp
cp ./$LLAMA_DIR/common/peg-parser.h ./cpp/common/peg-parser.h
cp ./$LLAMA_DIR/common/peg-parser.cpp ./cpp/common/peg-parser.cpp
cp ./$LLAMA_DIR/common/unicode.h ./cpp/common/unicode.h
cp ./$LLAMA_DIR/common/unicode.cpp ./cpp/common/unicode.cpp

# Copy multimodal files from tools/mtmd
rm -rf ./cpp/tools/mtmd
mkdir -p ./cpp/tools/mtmd
cp -r ./$LLAMA_DIR/tools/mtmd/models ./cpp/tools/mtmd/models
cp ./$LLAMA_DIR/tools/mtmd/mtmd.h ./cpp/tools/mtmd/mtmd.h
cp ./$LLAMA_DIR/tools/mtmd/mtmd.cpp ./cpp/tools/mtmd/mtmd.cpp
cp ./$LLAMA_DIR/tools/mtmd/clip.h ./cpp/tools/mtmd/clip.h
cp ./$LLAMA_DIR/tools/mtmd/clip.cpp ./cpp/tools/mtmd/clip.cpp
cp ./$LLAMA_DIR/tools/mtmd/clip-impl.h ./cpp/tools/mtmd/clip-impl.h
cp ./$LLAMA_DIR/tools/mtmd/clip-model.h ./cpp/tools/mtmd/clip-model.h
cp ./$LLAMA_DIR/tools/mtmd/clip-graph.h ./cpp/tools/mtmd/clip-graph.h
cp ./$LLAMA_DIR/tools/mtmd/mtmd-helper.cpp ./cpp/tools/mtmd/mtmd-helper.cpp
cp ./$LLAMA_DIR/tools/mtmd/mtmd-helper.h ./cpp/tools/mtmd/mtmd-helper.h
cp ./$LLAMA_DIR/tools/mtmd/mtmd-audio.h ./cpp/tools/mtmd/mtmd-audio.h
cp ./$LLAMA_DIR/tools/mtmd/mtmd-audio.cpp ./cpp/tools/mtmd/mtmd-audio.cpp

rm -rf ./cpp/common/jinja
cp -r ./$LLAMA_DIR/common/jinja ./cpp/common/jinja

# Rename jinja/string.h to avoid conflict with system <string.h>
mv ./cpp/common/jinja/string.h ./cpp/common/jinja/jinja-string.h
# Update includes in jinja files
if [ "$OS" = "Darwin" ]; then
  sed -i '' 's|#include "string.h"|#include "jinja-string.h"|g' ./cpp/common/jinja/value.h
  sed -i '' 's|#include "jinja/string.h"|#include "jinja/jinja-string.h"|g' ./cpp/common/jinja/string.cpp
else
  sed -i 's|#include "string.h"|#include "jinja-string.h"|g' ./cpp/common/jinja/value.h
  sed -i 's|#include "jinja/string.h"|#include "jinja/jinja-string.h"|g' ./cpp/common/jinja/string.cpp
fi

rm -rf ./cpp/nlohmann
cp -r ./$LLAMA_DIR/vendor/nlohmann ./cpp/nlohmann
rm -rf ./cpp/tools/mtmd/miniaudio
rm -rf ./cpp/tools/mtmd/stb
cp -r ./$LLAMA_DIR/vendor/miniaudio ./cpp/tools/mtmd/miniaudio
cp -r ./$LLAMA_DIR/vendor/stb ./cpp/tools/mtmd/stb

# List of files to process
files_add_lm_prefix=(
  # ggml api
  ./cpp/ggml-metal/*.cpp
  ./cpp/ggml-metal/*.h
  ./cpp/ggml-metal/*.m
  ./cpp/ggml-metal/*.metal

  ./cpp/ggml-blas/*.cpp

  ./cpp/ggml-opencl/*.cpp

  ./cpp/ggml-hexagon/*.cpp
  ./cpp/ggml-hexagon/*.h
  ./cpp/ggml-hexagon/htp/*.c
  ./cpp/ggml-hexagon/htp/*.h

  ./cpp/ggml-cpu/*.h
  ./cpp/ggml-cpu/*.c
  ./cpp/ggml-cpu/*.cpp
  ./cpp/ggml-cpu/amx/*.h
  ./cpp/ggml-cpu/amx/*.cpp
  ./cpp/ggml-cpu/arch/arm/*.c
  ./cpp/ggml-cpu/arch/arm/*.cpp
  ./cpp/ggml-cpu/arch/x86/*.c
  ./cpp/ggml-cpu/arch/x86/*.cpp

  # Model definitions
  ./cpp/models/*.h
  ./cpp/models/*.cpp

  # Multimodal files
  ./cpp/tools/mtmd/*.h
  ./cpp/tools/mtmd/*.cpp
  ./cpp/tools/mtmd/models/*.h
  ./cpp/tools/mtmd/models/*.cpp

  # llama api
  ./cpp/*.h
  ./cpp/*.cpp
  ./cpp/*.c

  ./cpp/common/*.h
  ./cpp/common/*.cpp
)

# Loop through each file and run the sed commands
for file in "${files_add_lm_prefix[@]}"; do
  # Skip cpp/rn-* files
  if [[ $file == *"/cpp/rn-"* ]]; then
    continue
  fi

  # Add prefix to avoid redefinition with other libraries using ggml like whisper.rn
  if [ "$OS" = "Darwin" ]; then
    sed -i '' 's/GGML_/LM_GGML_/g' $file
    sed -i '' 's/ggml_/lm_ggml_/g' $file
    sed -i '' 's/GGUF_/LM_GGUF_/g' $file
    sed -i '' 's/gguf_/lm_gguf_/g' $file
    sed -i '' 's/GGMLMetalClass/LMGGMLMetalClass/g' $file

    # <nlohmann/json.hpp> -> "nlohmann/json.hpp"
    sed -i '' 's/<nlohmann\/json.hpp>/"nlohmann\/json.hpp"/g' $file

    # <nlohmann/json_fwd.hpp> -> "nlohmann/json_fwd.hpp"
    sed -i '' 's/<nlohmann\/json_fwd.hpp>/"nlohmann\/json_fwd.hpp"/g' $file
  else
    sed -i 's/GGML_/LM_GGML_/g' $file
    sed -i 's/ggml_/lm_ggml_/g' $file
    sed -i 's/GGUF_/LM_GGUF_/g' $file
    sed -i 's/gguf_/lm_gguf_/g' $file
    sed -i 's/GGMLMetalClass/LMGGMLMetalClass/g' $file

    # <nlohmann/json.hpp> -> "nlohmann/json.hpp"
    sed -i 's/<nlohmann\/json.hpp>/"nlohmann\/json.hpp"/g' $file

    # <nlohmann/json_fwd.hpp> -> "nlohmann/json_fwd.hpp"
    sed -i 's/<nlohmann\/json_fwd.hpp>/"nlohmann\/json_fwd.hpp"/g' $file
  fi
done

files_iq_add_lm_prefix=(
  "./cpp/ggml-quants.h"
  "./cpp/ggml-quants.c"
  "./cpp/ggml.c"
)

for file in "${files_iq_add_lm_prefix[@]}"; do
  # Add prefix to avoid redefinition with other libraries using ggml like whisper.rn
  if [ "$OS" = "Darwin" ]; then
    sed -i '' 's/iq2xs_init_impl/lm_iq2xs_init_impl/g' $file
    sed -i '' 's/iq2xs_free_impl/lm_iq2xs_free_impl/g' $file
    sed -i '' 's/iq3xs_init_impl/lm_iq3xs_init_impl/g' $file
    sed -i '' 's/iq3xs_free_impl/lm_iq3xs_free_impl/g' $file
  else
    sed -i 's/iq2xs_init_impl/lm_iq2xs_init_impl/g' $file
    sed -i 's/iq2xs_free_impl/lm_iq2xs_free_impl/g' $file
    sed -i 's/iq3xs_init_impl/lm_iq3xs_init_impl/g' $file
    sed -i 's/iq3xs_free_impl/lm_iq3xs_free_impl/g' $file
  fi
done

echo "Replacement completed successfully!"

cd example && npm install && cd ..

# Apply patch
# List ./scripts/patches/ and patch it
for patch_file in ./scripts/patches/*.patch; do
  patch -p0 -d ./cpp < "$patch_file"
done

rm -rf ./cpp/*.orig
rm -rf ./cpp/**/*.orig

if [ "$OS" = "Darwin" ]; then
  # Generate .xcode.env.local in iOS example
  cd example/ios
  echo export NODE_BINARY=$(command -v node) > .xcode.env.local
  cd -
fi

# Get version info
cd "$LLAMA_DIR"
BUILD_NUMBER=$(git rev-list --count HEAD)
BUILD_COMMIT=$(git rev-parse --short=7 HEAD)

# Put to ../version.ts
# clean up version.ts
rm -f "$SRC_DIR/version.ts"

echo "export const BUILD_NUMBER = '$BUILD_NUMBER'" > "$SRC_DIR/version.ts"
echo "export const BUILD_COMMIT = '$BUILD_COMMIT'" >> "$SRC_DIR/version.ts"

cd "$ROOT_DIR"
