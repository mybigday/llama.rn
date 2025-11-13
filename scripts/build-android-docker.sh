#!/bin/bash -e
# Build Android libraries fully in Docker
# HTP libraries are built in Docker (needs Hexagon SDK)
# Android libraries are also built in Docker (needs Hexagon SDK for cdsprpc linking)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "Building Android Libraries in Docker"
echo "=========================================="
echo ""

# Step 1: Build HTP libraries in Docker
echo "Step 1/2: Building HTP (DSP) libraries in Docker..."
echo ""
"$SCRIPT_DIR/build-hexagon-htp.sh"

echo ""
echo "=========================================="
echo ""

# Step 2: Build Android libraries in Docker
echo "Step 2/2: Building Android native libraries in Docker..."
echo ""
echo "Note: Building in Docker to access Hexagon SDK for cdsprpc linking"
echo ""

# Run build-android.sh in Docker (where Hexagon SDK and Android NDK are available)
"$SCRIPT_DIR/docker-build-wrapper.sh" "./scripts/build-android.sh"

echo ""
echo "=========================================="
echo "Build Complete!"
echo "=========================================="
echo ""
echo "Built libraries:"
echo "  • HTP libraries: hexagon-prebuilt/libggml-htp-*.so"
echo "  • Android libs:  android/src/main/jniLibs/"
echo ""
echo "Verify Hexagon library has cdsprpc:"
echo "  readelf -d android/src/main/jniLibs/arm64-v8a/librnllama_v8_5_i8mm_hexagon.so | grep libcdsprpc"
echo ""
