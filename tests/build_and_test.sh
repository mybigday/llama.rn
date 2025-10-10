#!/bin/bash
set -e

echo "=== Building llama.rn C++ Tests ==="

# Create build directory
mkdir -p build
cd build

# Configure
echo "Configuring with CMake..."
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build both test executables
echo ""
echo "Building test executables..."
make rnllama_tests -j4
make parallel_decoding_test -j4

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
