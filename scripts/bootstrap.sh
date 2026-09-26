#!/bin/bash -e
#
# Developer environment setup:
#   - Hexagon SDK (Android HTP builds)
#   - example app dependencies (+ CocoaPods on macOS)
#   - build artifacts that are gitignored: embedded Metal kernel sources
#
# The llama.cpp / codec.cpp sources under vendor/ are committed as-is
# (vendored + patched by scripts/sync-vendor.sh); this script never
# modifies them, so every build compiles exactly the checked-in tree.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OS=$(uname)
LLAMA_CPP_DIR="$ROOT_DIR/vendor/llama.cpp"

cd "$ROOT_DIR"

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

cd example && npm install && cd ..

# llama.cpp splits Metal kernels into per-op sources. Flatten each source with
# the headers it needs, then encode it in an assembly file so Apple frameworks
# carry every source without distributing runtime .metal resources (see #348).
# The sources are the committed (already patched) vendored tree; the .s files
# are gitignored, so regenerate them on every bootstrap.
echo "Generating embedded Metal kernel sources..."
METAL_DIR="$LLAMA_CPP_DIR/ggml/src/ggml-metal"
METAL_KERNEL_DIR="$METAL_DIR/kernels"
METAL_COMMON="$LLAMA_CPP_DIR/ggml/src/ggml-common.h"
METAL_IMPL="$METAL_DIR/ggml-metal-impl.h"
rm -f "$METAL_DIR"/ggml-metal-embed*.s

for METAL_KERNEL in "$METAL_KERNEL_DIR"/*.metal; do
  METAL_KIND=$(basename "$METAL_KERNEL" .metal)
  # These are included by the per-type Flash Attention kernels, not libraries.
  case "$METAL_KIND" in
    fa_common|fa_vec_common) continue ;;
  esac
  METAL_KIND_SYMBOL=${METAL_KIND//-/_}
  METAL_TMP1="$METAL_DIR/.ggml-metal-embed-$METAL_KIND.tmp1"
  METAL_TMP2="$METAL_DIR/.ggml-metal-embed-$METAL_KIND.tmp2"
  METAL_TMP3="$METAL_DIR/.ggml-metal-embed-$METAL_KIND.tmp3"
  METAL_FLAT="$METAL_DIR/.ggml-metal-embed-$METAL_KIND.metal"
  EMBED_ASM="$METAL_DIR/ggml-metal-embed-$METAL_KIND.s"

  {
    cat "$METAL_KERNEL_DIR/common.h"
    if grep -qF '#include "dequantize.h"' "$METAL_KERNEL"; then
      cat "$METAL_KERNEL_DIR/dequantize.h"
    fi
    if grep -qF '#include "quantize.h"' "$METAL_KERNEL"; then
      cat "$METAL_KERNEL_DIR/quantize.h"
    fi
    for METAL_SHARED in fa_common.metal fa_vec_common.metal; do
      if grep -qF "#include \"$METAL_SHARED\"" "$METAL_KERNEL"; then
        cat "$METAL_KERNEL_DIR/$METAL_SHARED"
      fi
    done
    cat "$METAL_KERNEL"
  } > "$METAL_TMP1"

  sed -e '/#include "common.h"/d' \
      -e '/#include "dequantize.h"/d' \
      -e '/#include "quantize.h"/d' \
      -e '/#include "fa_common.metal"/d' \
      -e '/#include "fa_vec_common.metal"/d' \
      -e '/#pragma once/d' \
      < "$METAL_TMP1" > "$METAL_TMP2"
  sed -e "/__embed_ggml-common.h__/r $METAL_COMMON" \
      -e '/__embed_ggml-common.h__/d' \
      < "$METAL_TMP2" > "$METAL_TMP3"
  sed -e "/#include \"ggml-metal-impl.h\"/r $METAL_IMPL" \
      -e '/#include "ggml-metal-impl.h"/d' \
      < "$METAL_TMP3" > "$METAL_FLAT"

  {
    # Mach-O section names are limited to 16 characters. Keep the shared
    # upstream section name and vary only the exported per-kernel symbols.
    echo '.section __DATA,__ggml_metallib'
    echo ".globl _ggml_metallib_${METAL_KIND_SYMBOL}_start"
    echo "_ggml_metallib_${METAL_KIND_SYMBOL}_start:"
    # 16 bytes per line; compatible with both BSD and GNU od.
    od -An -vtx1 "$METAL_FLAT" | awk 'NF>0 {
      printf ".byte 0x%s", $1
      for (i=2; i<=NF; i++) printf ",0x%s", $i
      printf "\n"
    }'
    echo ".globl _ggml_metallib_${METAL_KIND_SYMBOL}_end"
    echo "_ggml_metallib_${METAL_KIND_SYMBOL}_end:"
  } > "$EMBED_ASM"

  rm -f "$METAL_TMP1" "$METAL_TMP2" "$METAL_TMP3" "$METAL_FLAT"
  echo "  $METAL_KIND ($(wc -l < "$EMBED_ASM") assembly lines)"
done

if [ "$OS" = "Darwin" ]; then
  # Refresh Pods after source list changes so the example target picks up
  # renamed/added/removed native files from the updated llama.cpp snapshot.
  cd example
  npm run pods
  cd ..

  # Generate .xcode.env.local in iOS example
  cd example/ios
  echo export NODE_BINARY=$(command -v node) > .xcode.env.local
  cd -
fi
