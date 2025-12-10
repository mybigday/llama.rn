# llama.rn - Development Guidelines

## Project Overview

**llama.rn** is a React Native binding for [llama.cpp](https://github.com/ggerganov/llama.cpp), enabling on-device LLM inference for iOS and Android. The library now uses a JSI-first bridge to expose llama.cpp APIs directly to JavaScript.

## General

- Pay attention to code readability.
- Add comments appropriately, no need to explain the obvious.
- Apply first-principles thinking when appropriate.

## Architecture

1. **TypeScript API (`src/`)**
   - `src/index.ts` provides the public API and binds to JSI globals installed by `installJsi()`
   - `src/jsi.ts` handles global function exports/imports and type safety
   - `src/types.ts` and `src/grammar.ts` house shared types and grammar helpers
   - `src/NativeRNLlama.ts` is a minimal TurboModule exposing only `install()` to trigger JSI setup
   - Streaming callbacks flow directly over JSI (no React Native event emitters)

2. **JSI Bridge (`cpp/jsi/` + platform glue)**
   - Core bindings in `cpp/jsi/RNLlamaJSI.cpp` with helpers (`JSIParams`, `JSICompletion`, `JSISession`, `JSIRequestManager`, `ThreadPool`, etc.)
   - iOS install path: `ios/RNLlama.mm` + `ios/RNLlamaJSI.mm` register bindings on the JS runtime
   - Android install path: `android/src/main/java/com/rnllama/RNLlama.java` (native lib loader + HTP extraction), `android/src/main/java/com/rnllama/RNLlamaModuleShared.java`, and `android/src/main/RNLlamaJSI.cpp`

3. **C++ Core (`cpp/`)**
   - llama.cpp sources are copied from `third_party/llama.cpp` with `LM_`/`lm_` prefixes
   - Custom wrappers: `rn-llama.cpp`, `rn-completion.cpp`, `rn-slot.cpp`, `rn-slot-manager.cpp`, `rn-mtmd.hpp`, `rn-tts.cpp`
   - Parallel decoding relies on the slot manager and request queues

## Core Features

- Chat & text completion with token streaming (JSI callbacks)
- Parallel decoding with slot-based queueing
- Multimodal vision/audio via mmproj projector models
- Tool calling & MCP support
- Embeddings & reranking (sync + queued)
- Text-to-speech (OuteTTS)
- Grammar sampling (GBNF/JSON schema)
- Session save/load
- Benchmarking (tokens/sec, prompt speed)

## Build System

### Bootstrap Process (`scripts/bootstrap.sh`)

1. Updates llama.cpp submodule (`third_party/llama.cpp`)
2. Copies source files to `cpp/` directory
3. Renames symbols with `LM_` prefix to prevent conflicts
4. Applies patches from `scripts/patches/`
5. Builds Metal shader libraries (`.metallib`) for iOS GPU acceleration
6. Generates version info from llama.cpp git history

**Always run `npm run bootstrap` after updating the llama.cpp submodule.**

### Platform Builds

- **iOS:** Uses pre-built `ios/rnllama.xcframework` by default. Set `RNLLAMA_BUILD_FROM_SOURCE=1` in the Podfile to build from source. `scripts/build-ios.sh` builds device/simulator frameworks. Metal is enabled unless `RNLLAMA_DISABLE_METAL=1`.
- **Android:** Uses pre-built `.so` libraries by default. Set `rnllamaBuildFromSource=true` in `android/gradle.properties` to build from source. `scripts/build-android.sh` builds native libs (OpenCL via `scripts/build-opencl.sh`). CMake config lives in `android/build.gradle` and `ios/CMakeLists.txt`.

## Common Development Commands

```bash
npm install
npm run bootstrap              # Required after cloning or updating llama.cpp submodule
npm run typecheck              # TypeScript type checking
npm run lint                   # Run ESLint
npm run lint -- --fix          # Fix ESLint errors
npm test                       # Run Jest unit tests
npm run build:ios-frameworks   # Build iOS frameworks
npm run build:android-libs     # Build Android libraries (includes OpenCL)
npm run docgen                 # Generate API docs from TypeScript
npm run example start          # Start Metro bundler
npm run example run ios        # Run iOS example
npm run example run android    # Run Android example
npm run build:ios              # Build iOS example app
npm run build:android          # Build Android example app
```

## Development Workflow

- **TypeScript layer:** Edit `src/index.ts`, `src/types.ts`, `src/jsi.ts`, and `src/grammar.ts`. `NativeRNLlama.install()` only installs JSI; all APIs are invoked via JSI bindings. Run `npm run typecheck` and `npm run lint` before committing.
- **C++ core:** Edit files in `cpp/`. If you update llama.cpp itself, change `third_party/llama.cpp` and rerun `npm run bootstrap`. Rebuild native libs or the example app to validate changes.
- **JSI bridge/platform glue:** Implement binding logic in `cpp/jsi/*`. iOS installs live in `ios/RNLlama.mm` + `ios/RNLlamaJSI.mm`; Android uses `android/src/main/java/com/rnllama/RNLlama.java` (native loader), `android/src/main/java/com/rnllama/RNLlamaModuleShared.java`, and `android/src/main/RNLlamaJSI.cpp`.

### Adding Patches

If llama.cpp sources need modifications:
1. Edit files in `cpp/`
2. Create patch: `diff -u original.cpp modified.cpp > scripts/patches/filename.patch`
3. Add patch application to `scripts/bootstrap.sh`

## Important Conventions

- All llama.cpp/ggml symbols are prefixed with `LM_`/`lm_` to avoid conflicts (e.g., `ggml_*` → `lm_ggml_*`).
- Follow conventional commits (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`).

## Testing Strategy

- **TypeScript/Javascript:** Jest unit tests in `src/__tests__/` (JSI mocked in `jest/mock.js`). Run with `npm test`.
- **C++:** Unit tests in `tests/` (`tests/build_and_test.sh` or `tests/run_tests.sh` after building).
- **Integration:** Example app under `example/` for device validation, GPU/performance checks, and parallel decoding stress.

## Key Files Reference

- `src/index.ts`, `src/jsi.ts`, `src/types.ts`, `src/grammar.ts`, `src/NativeRNLlama.ts`
- `cpp/jsi/RNLlamaJSI.cpp` (+ helpers: `JSIParams.h/.cpp`, `JSICompletion.h`, `JSISession.h`, `JSIRequestManager.h`, `ThreadPool.*`)
- `cpp/rn-llama.cpp`, `cpp/rn-completion.cpp`, `cpp/rn-slot.cpp`, `cpp/rn-slot-manager.cpp`, `cpp/rn-mtmd.hpp`, `cpp/rn-tts.cpp`
- `ios/RNLlama.mm`, `ios/RNLlamaJSI.mm`
- `android/src/main/java/com/rnllama/RNLlama.java`
- `android/src/main/java/com/rnllama/RNLlamaModuleShared.java`
- `android/src/main/RNLlamaJSI.cpp`
- `llama-rn.podspec`, `android/build.gradle`, `tests/`
