#!/bin/bash -e
# Build Hexagon HTP (DSP) libraries
# These libraries run on the Qualcomm Hexagon DSP processor

ROOT_DIR=$(pwd)
HTP_SOURCE_DIR="${ROOT_DIR}/cpp/ggml-hexagon/htp"
HTP_BUILD_DIR="${ROOT_DIR}/build-hexagon-htp"
HTP_OUTPUT_DIR="${ROOT_DIR}/hexagon-prebuilt"

# Check environment
if [ -z "$HEXAGON_SDK_ROOT" ] || [ -z "$HEXAGON_TOOLS_ROOT" ]; then
    echo "Error: Hexagon SDK environment not configured"
    echo "Please run: source scripts/setup-hexagon-env.sh"
    exit 1
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
echo "SDK Root:   $HEXAGON_SDK_ROOT"
echo "Tools Root: $HEXAGON_TOOLS_ROOT"
echo "Source:     $HTP_SOURCE_DIR"
echo "Output:     $HTP_OUTPUT_DIR"
echo ""

# Set additional environment variables needed for build
export DEFAULT_HLOS_ARCH=${DEFAULT_HLOS_ARCH:-64}
export DEFAULT_TOOLS_VARIANT=${DEFAULT_TOOLS_VARIANT:-toolv19}
export DEFAULT_NO_QURT_INC=${DEFAULT_NO_QURT_INC:-0}

# Create output directory
mkdir -p "$HTP_OUTPUT_DIR"

# DSP versions to build
DSP_VERSIONS=("v73" "v75" "v79" "v81")
PREBUILT_DIRS=("toolv19_v73" "toolv19_v75" "toolv19_v79" "toolv19_v81")

# Function to build for a specific DSP version
build_htp_version() {
    local dsp_version=$1
    local prebuilt_dir=$2

    echo "Building HTP library for DSP ${dsp_version}..."

    local build_dir="${HTP_BUILD_DIR}/${dsp_version}"
    mkdir -p "$build_dir"

    cd "$build_dir"

    # Configure with Hexagon toolchain
    cmake "${HTP_SOURCE_DIR}" \
        -DCMAKE_TOOLCHAIN_FILE="${HTP_SOURCE_DIR}/cmake-toolchain.cmake" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_LIBDIR="${HTP_OUTPUT_DIR}" \
        -DHEXAGON_SDK_ROOT="${HEXAGON_SDK_ROOT}" \
        -DHEXAGON_TOOLS_ROOT="${HEXAGON_TOOLS_ROOT}" \
        -DDSP_VERSION="${dsp_version}" \
        -DPREBUILT_LIB_DIR="${prebuilt_dir}" \
        -DHEXAGON_HTP_DEBUG=OFF

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

# Copy HTP libraries to Android jniLibs directories
echo "Copying HTP libraries to Android jniLibs directories..."
echo ""

# Destinations
LIBRARY_JNILIBS="${ROOT_DIR}/android/src/main/jniLibs/arm64-v8a"
EXAMPLE_JNILIBS="${ROOT_DIR}/example/android/app/src/main/jniLibs/arm64-v8a"

# Create directories if they don't exist
mkdir -p "$LIBRARY_JNILIBS"
mkdir -p "$EXAMPLE_JNILIBS"

# Copy to library jniLibs
echo "→ Copying to library package: $LIBRARY_JNILIBS"
for lib in "$HTP_OUTPUT_DIR"/libggml-htp-*.so; do
    if [ -f "$lib" ]; then
        cp "$lib" "$LIBRARY_JNILIBS/"
        echo "  ✓ $(basename "$lib")"
    fi
done

# Copy to example app jniLibs
echo ""
echo "→ Copying to example app: $EXAMPLE_JNILIBS"
for lib in "$HTP_OUTPUT_DIR"/libggml-htp-*.so; do
    if [ -f "$lib" ]; then
        cp "$lib" "$EXAMPLE_JNILIBS/"
        echo "  ✓ $(basename "$lib")"
    fi
done

echo ""
echo "=========================================="
echo "HTP Libraries Installed Successfully!"
echo "=========================================="
echo "Libraries are now available in:"
echo "  • Library package: $LIBRARY_JNILIBS"
echo "  • Example app:     $EXAMPLE_JNILIBS"
echo ""
echo "The Hexagon backend will automatically detect and load"
echo "the appropriate library at runtime based on device DSP version."
echo ""
