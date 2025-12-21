#!/bin/bash -e
# Build Hexagon HTP (DSP) libraries
# These libraries run on the Qualcomm Hexagon DSP processor

ROOT_DIR=$(pwd)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check if running inside Docker
if [ -f "/.dockerenv" ] || grep -q docker /proc/1/cgroup 2>/dev/null; then
    IN_DOCKER=true
    echo "Running inside Docker container"
else
    IN_DOCKER=false
fi

# If not in Docker and USE_DOCKER is set, re-run in Docker
if [ "$IN_DOCKER" = false ] && [ "${USE_DOCKER:-auto}" != "no" ]; then
    # Auto-detect: Use Docker on macOS, allow native on Linux
    SHOULD_USE_DOCKER=false

    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "macOS detected - Docker is required for Hexagon builds"
        SHOULD_USE_DOCKER=true
    elif [ "${USE_DOCKER}" = "yes" ] || [ "${USE_DOCKER}" = "auto" ]; then
        # On Linux, offer Docker as option
        if command -v docker &> /dev/null; then
            echo "Docker available - using Docker for consistent builds"
            SHOULD_USE_DOCKER=true
        fi
    fi

    if [ "$SHOULD_USE_DOCKER" = true ]; then
        echo "Launching Docker build..."
        exec "$SCRIPT_DIR/docker-build-wrapper.sh" "./scripts/build-hexagon-htp.sh"
    fi
fi

HTP_SOURCE_DIR="${ROOT_DIR}/cpp/ggml-hexagon/htp"
HTP_BUILD_DIR="${ROOT_DIR}/build-hexagon-htp"
HTP_OUTPUT_DIR="${ROOT_DIR}/bin/arm64-v8a"

# Hexagon SDK configuration
HEXAGON_SDK_VERSION="6.4.0.2"
HEXAGON_TOOLS_VERSION="19.0.04"

# Check environment (Docker has these pre-configured)
if [ "$IN_DOCKER" = true ]; then
    # Docker environment has SDK pre-installed
    export HEXAGON_SDK_ROOT="${HEXAGON_SDK_ROOT:-/opt/hexagon/6.4.0.2}"
    export HEXAGON_TOOLS_ROOT="${HEXAGON_TOOLS_ROOT:-$HEXAGON_SDK_ROOT/tools}"
    export DEFAULT_HLOS_ARCH="${DEFAULT_HLOS_ARCH:-64}"
    export DEFAULT_TOOLS_VARIANT="${DEFAULT_TOOLS_VARIANT:-toolv19}"
    export DEFAULT_NO_QURT_INC="${DEFAULT_NO_QURT_INC:-0}"
else
    # Native build: auto-detect or use existing environment
    if [ -z "$HEXAGON_SDK_ROOT" ]; then
        # Try to auto-detect SDK installation
        HEXAGON_INSTALL_DIR="${HEXAGON_INSTALL_DIR:-$HOME/.hexagon-sdk}"

        if [ -d "$HEXAGON_INSTALL_DIR/$HEXAGON_SDK_VERSION" ]; then
            echo "Auto-detected Hexagon SDK at $HEXAGON_INSTALL_DIR/$HEXAGON_SDK_VERSION"
            export HEXAGON_SDK_ROOT="$HEXAGON_INSTALL_DIR/$HEXAGON_SDK_VERSION"
            export HEXAGON_TOOLS_ROOT="$HEXAGON_INSTALL_DIR/$HEXAGON_SDK_VERSION/tools/HEXAGON_Tools/$HEXAGON_TOOLS_VERSION"
        else
            echo "Error: Hexagon SDK not found"
            echo ""
            echo "SDK not found at: $HEXAGON_INSTALL_DIR/$HEXAGON_SDK_VERSION"
            echo ""
            echo "Solutions:"
            echo "  1. Run 'npm run bootstrap' to download and install the SDK"
            echo "  2. Or set HEXAGON_INSTALL_DIR to your SDK location"
            echo "  3. Or use Docker: ./scripts/build-android-docker.sh"
            exit 1
        fi
    fi

    # Set default build variables if not already set
    export DEFAULT_HLOS_ARCH="${DEFAULT_HLOS_ARCH:-64}"
    export DEFAULT_TOOLS_VARIANT="${DEFAULT_TOOLS_VARIANT:-toolv19}"
    export DEFAULT_NO_QURT_INC="${DEFAULT_NO_QURT_INC:-0}"
fi

if [ ! -d "$HEXAGON_SDK_ROOT" ]; then
    echo "Error: HEXAGON_SDK_ROOT directory not found: $HEXAGON_SDK_ROOT"
    exit 1
fi

if [ ! -d "$HEXAGON_TOOLS_ROOT" ]; then
    echo "Error: HEXAGON_TOOLS_ROOT directory not found: $HEXAGON_TOOLS_ROOT"
    exit 1
fi

echo "=========================================="
echo "Building Hexagon HTP Libraries"
echo "=========================================="
if [ "$IN_DOCKER" = true ]; then
    echo "Environment: Docker container"
else
    echo "Environment: Native Linux"
fi
echo "SDK Root:   $HEXAGON_SDK_ROOT"
echo "Tools Root: $HEXAGON_TOOLS_ROOT"
echo "Source:     $HTP_SOURCE_DIR"
echo "Output:     $HTP_OUTPUT_DIR"
echo ""

# Create output directory
mkdir -p "$HTP_OUTPUT_DIR"

# DSP versions to build
DSP_VERSIONS=("v69" "v73" "v75" "v79" "v81")
PREBUILT_DIRS=("toolv19_v69" "toolv19_v73" "toolv19_v75" "toolv19_v79" "toolv19_v81")

# Function to build for a specific DSP version
build_htp_version() {
    local dsp_version=$1
    local prebuilt_dir=$2

    echo "Building HTP library for DSP ${dsp_version}..."

    local build_dir="${HTP_BUILD_DIR}/${dsp_version}"

    # Clean build directory if it exists (to avoid generator mismatch)
    if [ -d "$build_dir" ]; then
        rm -rf "$build_dir"
    fi

    mkdir -p "$build_dir"
    cd "$build_dir"

    # Determine generator based on environment
    CMAKE_GENERATOR=""
    if command -v ninja &> /dev/null; then
        CMAKE_GENERATOR="-GNinja"
        echo "Using Ninja build system"
    elif command -v make &> /dev/null; then
        CMAKE_GENERATOR="-G Unix Makefiles"
        echo "Using Make build system"
    else
        echo "Warning: Neither ninja nor make found, using default generator"
    fi

    # Configure with Hexagon toolchain
    cmake "${HTP_SOURCE_DIR}" \
        $CMAKE_GENERATOR \
        -DCMAKE_TOOLCHAIN_FILE="${HTP_SOURCE_DIR}/cmake-toolchain.cmake" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_LIBDIR="${HTP_OUTPUT_DIR}" \
        -DHEXAGON_SDK_ROOT="${HEXAGON_SDK_ROOT}" \
        -DHEXAGON_TOOLS_ROOT="${HEXAGON_TOOLS_ROOT}" \
        -DDSP_VERSION="${dsp_version}" \
        -DPREBUILT_LIB_DIR="${prebuilt_dir}" \
        -DHEXAGON_HTP_DEBUG=OFF \
        -DGGML_HEXAGON_FP32_QUANTIZE_GROUP_SIZE=128

    # Build
    cmake --build . --config Release

    # Install (copy to output directory)
    cmake --install .

    echo "✓ Built libggml-htp-${dsp_version}.so"

    cd "$ROOT_DIR"
}

# Build all DSP versions
for i in "${!DSP_VERSIONS[@]}"; do
    build_htp_version "${DSP_VERSIONS[$i]}" "${PREBUILT_DIRS[$i]}"
done

echo ""
echo "=========================================="
echo "Build Complete!"
echo "=========================================="
echo "HTP libraries built and installed to: $HTP_OUTPUT_DIR"
echo ""
ls -lh "$HTP_OUTPUT_DIR"/libggml-htp-*.so 2>/dev/null || echo "Warning: Some libraries may not have been built"
echo ""


mkdir -p cpp/ggml-hexagon/htp/v73/
cp build-hexagon-htp/v73/htp_iface_stub.c cpp/ggml-hexagon/htp/v73/htp_iface_stub.c
cp build-hexagon-htp/v73/htp_iface.h cpp/ggml-hexagon/htp/v73/htp_iface.h
echo "✓ Copied htp_iface_stub.c to cpp/ggml-hexagon/htp/v73/htp_iface_stub.c"
echo "✓ Copied htp_iface.h to cpp/ggml-hexagon/htp/v73/htp_iface.h"

rm -rf build-hexagon-htp
