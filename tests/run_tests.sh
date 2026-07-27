#!/bin/bash
set -e

cd "$(dirname "$0")"

echo "=== Running llama.rn C++ Tests ==="
echo ""

# Check if build directory exists
if [ ! -d "build" ]; then
    echo "Error: build directory not found"
    echo "Please run ./build_and_test.sh first"
    exit 1
fi

cd build

# Check if test executables exist
if [ ! -f "rnllama_tests" ]; then
    echo "Error: rnllama_tests executable not found"
    echo "Please run ./build_and_test.sh first"
    exit 1
fi

if [ ! -f "parallel_decoding_test" ]; then
    echo "Error: parallel_decoding_test executable not found"
    echo "Please run ./build_and_test.sh first"
    exit 1
fi

if [ ! -f "chat_parse_utf8_test" ]; then
    echo "Error: chat_parse_utf8_test executable not found"
    echo "Please run ./build_and_test.sh first"
    exit 1
fi

echo "Found all test executables"

TESTS_PASSED=0
TESTS_FAILED=0

# Run basic tests
echo "--- Running Basic Integration Tests ---"
if ./rnllama_tests; then
    echo "✓ Basic integration tests passed"
    TESTS_PASSED=$((TESTS_PASSED + 1))
else
    echo "✗ Basic integration tests failed"
    TESTS_FAILED=$((TESTS_FAILED + 1))
fi

echo ""

# Run parallel decoding tests
echo "--- Running Parallel Decoding Tests ---"
if ./parallel_decoding_test; then
    echo "✓ Parallel decoding tests passed"
    TESTS_PASSED=$((TESTS_PASSED + 1))
else
    echo "✗ Parallel decoding tests failed"
    TESTS_FAILED=$((TESTS_FAILED + 1))
fi

echo ""

# Run chat parse UTF-8 robustness tests
echo "--- Running Chat Parse UTF-8 Tests ---"
if ./chat_parse_utf8_test; then
    echo "✓ Chat parse UTF-8 tests passed"
    TESTS_PASSED=$((TESTS_PASSED + 1))
else
    echo "✗ Chat parse UTF-8 tests failed"
    TESTS_FAILED=$((TESTS_FAILED + 1))
fi

echo ""

# Run KV-cache-reuse tests (only if the GGUF models have been downloaded)
TOTAL_SUITES=3
if [ -f "kv_cache_reuse_test" ] && ls ../models/*.gguf >/dev/null 2>&1; then
    TOTAL_SUITES=4
    echo "--- Running KV-cache-reuse Tests ---"
    if ./kv_cache_reuse_test; then
        echo "✓ KV-cache-reuse tests passed"
        TESTS_PASSED=$((TESTS_PASSED + 1))
    else
        echo "✗ KV-cache-reuse tests failed"
        TESTS_FAILED=$((TESTS_FAILED + 1))
    fi
    echo ""
else
    echo "--- Skipping KV-cache-reuse Tests (no models in tests/models) ---"
    echo "    Run ./models/download.sh to enable them."
    echo ""
fi

echo "=== Test Summary ==="
echo "Passed: $TESTS_PASSED/$TOTAL_SUITES"
echo "Failed: $TESTS_FAILED/$TOTAL_SUITES"

if [ $TESTS_FAILED -eq 0 ]; then
    echo "✓ All test suites passed!"
    exit 0
else
    echo "✗ Some tests failed"
    exit 1
fi
