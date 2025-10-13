# llama.rn - Development Guidelines

## Project Overview

**llama.rn** is a React Native binding for [llama.cpp](https://github.com/ggerganov/llama.cpp), enabling on-device LLM inference for iOS and Android. The library provides a JavaScript/TypeScript API that wraps native implementations using React Native's TurboModule architecture.

## General

- Pay attention to code readability.
- Add comments appropriately, no need to explain the obvious.
- Apply first-principles thinking when appropriate.

## Architecture

### Three-Layer Architecture

1. **TypeScript Layer** (`src/`)
   - Main API in `src/index.ts` exports `LlamaContext` class and helper functions
   - `initLlama()` creates context with optional `n_parallel` parameter
   - `completion()` runs synchronous inference, `queueCompletion()` for parallel requests
   - `enableParallelMode()` configures concurrent request processing
   - Handles event-based streaming via React Native's event emitters
   - `src/grammar.ts` provides JSON schema to GBNF grammar conversion

2. **Native Bridge Layer** (`ios/`, `android/`)
   - **iOS**: `ios/RNLlama.mm` and `ios/RNLlamaContext.mm` bridge Objective-C++ to C++
   - **Android**: Java/Kotlin files in `android/src/main/java/com/rnllama/`
   - Both platforms use React Native's TurboModules/new architecture support

3. **C++ Core** (`cpp/`)
   - Contains llama.cpp source with `LM_` prefix to avoid symbol conflicts
   - Custom wrappers: `rn-llama.cpp`, `rn-completion.cpp`, `rn-mtmd.hpp` (multimodal), `rn-tts.cpp`
   - Parallel processing: `rn-slot.cpp`, `rn-slot-manager.cpp` for concurrent request handling
   - Files are copied from `third_party/llama.cpp` submodule during bootstrap

### Core Features

- **Chat & Text Completion**: Message-based chat and raw text/prompt completion
- **Streaming Responses**: Token-by-token streaming callbacks for real-time output
- **Parallel Decoding**: Slot-based concurrent request processing with automatic queue management
- **Multimodal (Vision & Audio)**: Image and audio processing via mmproj projector models
- **Tool Calling & MCP**: Function calling with Model Context Protocol (MCP) server integration
- **Embeddings & Reranking**: Vector embeddings and document reranking for semantic search (supports async queueing)
- **Text-to-Speech (TTS)**: Audio generation using vocoder models (OuteTTS)
- **Grammar Sampling**: GBNF/JSON schema support for structured output
- **Session Management**: Save/load conversation state to files
- **Benchmarking**: Performance testing (tokens/sec, prompt processing speed)

## Build System

### Bootstrap Process (`scripts/bootstrap.sh`)

The bootstrap script is critical for setup:
1. Updates llama.cpp submodule (`third_party/llama.cpp`)
2. Copies source files to `cpp/` directory
3. Renames symbols with `LM_` prefix (e.g., `ggml_*` → `lm_ggml_*`) to prevent conflicts with other native modules
4. Applies patches from `scripts/patches/`
5. Builds Metal shader libraries (`.metallib`) for iOS GPU acceleration
6. Generates version info from llama.cpp git history

**Always run `npm run bootstrap` after updating the llama.cpp submodule.**

### Platform Builds

**iOS:**
- By default uses pre-built `ios/rnllama.xcframework`
- Set `RNLLAMA_BUILD_FROM_SOURCE=1` in Podfile to build from source
- `scripts/build-ios.sh` builds frameworks for iOS/simulator
- Metal (GPU) support enabled by default unless `RNLLAMA_DISABLE_METAL=1`

**Android:**
- By default uses pre-built `.so` libraries
- Set `rnllamaBuildFromSource=true` in `android/gradle.properties` to build from source
- `scripts/build-android.sh` compiles native libraries for arm64-v8a and x86_64
- `scripts/build-opencl.sh` handles OpenCL backend for GPU (Qualcomm Adreno only)
- CMake configuration in `android/build.gradle` and `ios/CMakeLists.txt`

## Common Development Commands

### Setup and Bootstrap
```bash
npm install
npm run bootstrap              # Required after cloning or updating llama.cpp submodule
```

### Type Checking and Linting
```bash
npm run typecheck             # TypeScript type checking
npm run lint                  # Run ESLint
npm run lint -- --fix        # Fix ESLint errors
```

### Testing
```bash
npm test                      # Run Jest unit tests
```

### Building Native Libraries
```bash
npm run build:ios-frameworks  # Build iOS frameworks
npm run build:android-libs    # Build Android libraries (includes OpenCL)
```

### Documentation
```bash
npm run docgen               # Generate API docs from TypeScript
```

### Working with Example App
```bash
npm run example start        # Start Metro bundler
npm run example run ios      # Run iOS example
npm run example run android  # Run Android example
```

### Build Example Apps Directly
```bash
npm run build:ios            # Build iOS example app
npm run build:android        # Build Android example app
```

## Development Workflow

### Modifying TypeScript API
- Edit files in `src/`
- TypeScript changes reflect immediately in example app (hot reload)
- Run `npm run typecheck` and `npm run lint` before committing

### Modifying Native Code

**C++ Core:**
1. Edit files in `cpp/` (these are copied from llama.cpp during bootstrap)
2. If updating llama.cpp itself, modify in `third_party/llama.cpp`, then run `npm run bootstrap`
3. Rebuild native libraries or example app to test changes

**iOS Bridge:**
1. Edit `ios/RNLlama.mm` or `ios/RNLlamaContext.mm`
2. Open `example/ios/RNLlamaExample.xcworkspace` in Xcode
3. Find source at `Pods > Development Pods > llama.rn`
4. Rebuild to test

**Android Bridge:**
1. Edit Java/Kotlin files in `android/src/main/java/com/rnllama/`
2. Native C++ bridge code typically uses JNI
3. Run `npm run build:android` or `./gradlew assembleDebug` in `example/android`

### Adding Patches
If llama.cpp source needs modifications:
1. Make changes to files in `cpp/`
2. Create patch: `diff -u original.cpp modified.cpp > scripts/patches/filename.patch`
3. Add patch application to `scripts/bootstrap.sh`

## Important Conventions

### Symbol Prefixing
All llama.cpp/ggml symbols are prefixed with `LM_` or `lm_` (by script) to avoid conflicts:
- `GGML_*` → `LM_GGML_*`
- `ggml_*()` → `lm_ggml_*()`
- `GGUF_*` → `LM_GGUF_*`

This allows coexistence with other native modules (e.g., whisper.rn).

### Commit Messages
Follow conventional commits:
- `feat:` - new features
- `fix:` - bug fixes
- `docs:` - documentation
- `refactor:` - code refactoring
- `test:` - tests
- `chore:` - tooling/config

## Testing Strategy

- **TypeScript/JavaScript tests**: Unit tests in `src/__tests__/` using Jest
  - Mock implementation in `jest/mock.js` for testing
  - Run with `npm test`

- **C++ unit tests**: Located in `tests/` directory
  - Build and run with `tests/build_and_test.sh`
  - Alternatively run with `tests/run_tests.sh` after building
  - Tests core C++ functionality including slot manager

- **Integration testing**: Example app in `example/` serves as integration test
  - Test on real devices for GPU/performance validation
  - Verify parallel decoding with concurrent requests

## Key Files Reference

- `src/index.ts` - Main TypeScript API
- `src/NativeRNLlama.ts` - TypeScript definitions for native module
- `ios/RNLlamaContext.mm` - iOS context implementation
- `android/src/main/java/com/rnllama/LlamaContext.java` - Android context
- `cpp/rn-llama.cpp` - Core C++ wrapper and context management
- `cpp/rn-completion.cpp` - Completion/inference logic
- `cpp/rn-slot.cpp` - Individual slot implementation for parallel processing
- `cpp/rn-slot-manager.cpp` - Slot manager and processing loop
- `cpp/rn-mtmd.hpp` - Multimodal support (vision/audio)
- `cpp/rn-tts.cpp` - Text-to-speech implementation
- `llama-rn.podspec` - iOS CocoaPods spec
- `android/build.gradle` - Android build configuration
- `tests/` - C++ unit tests for core functionality
