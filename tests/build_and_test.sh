#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"

echo "=== Building llama.rn C++ Tests ==="

# Create build directory
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Get the correct SDK path for current Xcode version
if [[ "$OSTYPE" == "darwin"* ]]; then
    SDK_PATH=$(xcrun --show-sdk-path)
    echo "Using macOS SDK: $SDK_PATH"
    CMAKE_OSX_SYSROOT="-DCMAKE_OSX_SYSROOT=$SDK_PATH"
else
    CMAKE_OSX_SYSROOT=""
fi

# Configure
echo "Configuring with CMake..."
cmake "$SCRIPT_DIR" -DCMAKE_BUILD_TYPE=Release $CMAKE_OSX_SYSROOT

# The test executables share one compiled copy of the library (see
# rnllama_test_core in CMakeLists.txt), so build them in a single invocation.
TEST_TARGETS=(
    rnllama_tests
    parallel_decoding_test
    chat_parse_utf8_test
    model_info_test
    video_test
)

if [[ "$OSTYPE" == "darwin"* ]]; then
    JOBS=$(sysctl -n hw.logicalcpu)
else
    JOBS=$(nproc)
fi

echo ""
echo "Building test executables (${TEST_TARGETS[*]}) with $JOBS jobs..."
cmake --build . --target "${TEST_TARGETS[@]}" -j "$JOBS"

for target in "${TEST_TARGETS[@]}"; do
    if [ ! -f "$target" ]; then
        echo "Error: Failed to build $target"
        exit 1
    fi
    echo "✓ $target built successfully"
done

echo ""
echo "=== Build Successful ==="
echo ""
echo "Built executables:"
echo "  - rnllama_tests (basic integration tests)"
echo "  - parallel_decoding_test (parallel decoding tests)"
echo "  - chat_parse_utf8_test (chat parse UTF-8 robustness tests)"
echo "  - model_info_test (GGUF model info tests)"
echo "  - video_test (platform video decoder tests)"
echo ""
echo "To run the tests:"
echo "  cd tests/build"
echo "  ./rnllama_tests           # Run basic tests"
echo "  ./parallel_decoding_test  # Run parallel decoding tests"
echo "  ./chat_parse_utf8_test    # Run chat parse UTF-8 tests"
echo "  ./model_info_test         # Run model info tests"
echo "  ./video_test              # Run video decoder tests"
echo ""
echo "Or run all:"
echo "  ./rnllama_tests && ./parallel_decoding_test && ./chat_parse_utf8_test && ./model_info_test && ./video_test"
echo ""
