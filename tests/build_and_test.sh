#!/bin/bash
set -e

echo "=== Building llama.rn C++ Tests ==="

# Create build directory
mkdir -p build
cd build

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
cmake .. -DCMAKE_BUILD_TYPE=Release $CMAKE_OSX_SYSROOT

# Build both test executables
echo ""
echo "Building test executables..."
echo "Building rnllama_tests..."
make rnllama_tests -j4
if [ ! -f "rnllama_tests" ]; then
    echo "Error: Failed to build rnllama_tests"
    exit 1
fi
echo "✓ rnllama_tests built successfully"

echo "Building parallel_decoding_test..."
make parallel_decoding_test -j4
if [ ! -f "parallel_decoding_test" ]; then
    echo "Error: Failed to build parallel_decoding_test"
    exit 1
fi
echo "✓ parallel_decoding_test built successfully"

echo ""
echo "=== Build Successful ==="
echo ""
echo "Built executables:"
echo "  - rnllama_tests (basic integration tests)"
echo "  - parallel_decoding_test (parallel decoding tests)"
echo ""
echo "To run the tests:"
echo "  cd tests/build"
echo "  ./rnllama_tests           # Run basic tests"
echo "  ./parallel_decoding_test  # Run parallel decoding tests"
echo ""
echo "Or run both:"
echo "  ./rnllama_tests && ./parallel_decoding_test"
echo ""
