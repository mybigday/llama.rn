# C++ Tests for llama.rn

This directory contains integration tests for the llama.rn C++ implementation.

## Building and Running

### Quick Start (Recommended)

```bash
cd tests

# Build both test executables
./build_and_test.sh

# Run all tests
./run_tests.sh
```

### Using CMake Directly

```bash
cd tests
mkdir -p build
cd build

# Configure
cmake ..

# Build
make

# Run basic tests
./rnllama_tests

# Run parallel decoding tests
./parallel_decoding_test

# Run chat parse UTF-8 robustness tests (no model needed)
./chat_parse_utf8_test

# Run all
./rnllama_tests && ./parallel_decoding_test && ./chat_parse_utf8_test
```

### Build Scripts

**`build_and_test.sh`**
- Builds `rnllama_tests`, `parallel_decoding_test` and `chat_parse_utf8_test`
- Uses CMake with Release configuration
- Parallel compilation with `-j4`

**`run_tests.sh`**
- Runs all test suites
- Reports pass/fail for each suite
- Returns appropriate exit code for CI/CD

### Requirements

- CMake 3.16 or higher
- C++17 compatible compiler
- Test model: `tiny-random-llama.gguf` in tests directory

### Test Model

The tests use a tiny random LLAMA model (`tiny-random-llama.gguf`) for quick testing without requiring large model downloads.
