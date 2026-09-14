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
   - `src/types.ts` houses shared types
   - `src/NativeRNLlama.ts` is a minimal TurboModule exposing only `install()` to trigger JSI setup
   - Streaming callbacks flow directly over JSI (no React Native event emitters)

2. **JSI Bridge (`cpp/jsi/` + platform glue)**
   - Core bindings in `cpp/jsi/RNLlamaJSI.cpp` with helpers (`JSIParams`, `JSICompletion`, `JSISession`, `JSIRequestManager`, `ThreadPool`, etc.)
   - iOS install path: `ios/RNLlama.mm` registers bindings on the JS runtime via `rnllama_jsi::installJSIBindings`
   - Android install path: `android/src/main/java/com/rnllama/RNLlama.java` (native lib loader + HTP extraction), `android/src/main/java/com/rnllama/RNLlamaModule.java` (TurboModule entry point), and `android/src/main/RNLlamaJSI.cpp`

3. **C++ Core (`cpp/` + `vendor/`)**
   - `cpp/` holds only llama.rn's own code: `rn-llama.cpp`, `rn-completion.cpp`, `rn-slot.cpp`, `rn-slot-manager.cpp`, `rn-mtmd.hpp`, `rn-tts.cpp`, `anyascii.*`, and `jsi/`
   - llama.cpp and codec.cpp are vendored under `vendor/` in their upstream layout (`vendor/llama.cpp/{include,src,ggml,common,tools/mtmd,vendor}`), pinned by `vendor/VERSIONS`, with llama.rn changes kept as `-p1` patches in `scripts/patches/<dep>/`. No git submodules, no symbol renaming. See `vendor/README.md`.
   - `vendor/llama.cpp` is also consumed by llama.node via upstream's CMake project, so it carries upstream's build files, all of `common/` and the CUDA/Vulkan/WebGPU backends. llama.rn's builds never compile those; `cmake/rnllama-sources.cmake` and `llama-rn.podspec` filter them out (keep the two in sync).
   - `cmake/rnllama-sources.cmake` is the single source/include list every CMake build uses; `llama-rn.podspec` mirrors it for CocoaPods
   - Parallel decoding relies on the slot manager and request queues

## Core Features

- Chat & text completion with token streaming (JSI callbacks)
- Parallel decoding with slot-based queueing
- Multimodal vision/audio via mmproj projector models
- Tool calling & MCP support
- Embeddings & reranking (sync + queued)
- Text-to-speech
- Grammar sampling (GBNF/JSON schema)
- Session save/load
- Benchmarking (tokens/sec, prompt speed)

## Build System

### Vendored sources (`scripts/sync-vendor.sh`)

1. Clones/fetches each upstream pinned in `vendor/VERSIONS` into `~/.cache/llama.rn/`
2. Exports the subset llama.rn builds into `vendor/<dep>/`, unchanged and in upstream layout
3. Applies `scripts/patches/<dep>/*.patch`
4. Regenerates `src/version.ts` and the version headers/`build-info.cpp` upstream would generate at build time

Its output is committed. Run it after changing `vendor/VERSIONS` or a patch; running it on a clean tree must produce no diff. To change upstream code, edit the vendored file in place and regenerate its patch with `scripts/update-patch.sh <dep> <path> [name]`.

### Bootstrap (`scripts/bootstrap.sh`)

Developer environment only; it never touches `vendor/` contents:

1. Hexagon SDK download (Android HTP builds)
2. `example/` dependencies and, on macOS, CocoaPods
3. Flattens each split Metal kernel with `ggml-common.h` / `ggml-metal-impl.h` and emits per-kernel `ggml-metal-embed-*.s` files next to the kernels (gitignored) so the sources are embedded into the framework binary (avoids `.metallib` distribution and runtime `.metal` file loading; see #348)

**Run `npm run bootstrap` after cloning; run `npm run sync:vendor` after editing `vendor/VERSIONS` or `scripts/patches/`.**

### Platform Builds

- **iOS:** Library consumers use pre-built `ios/rnllama.xcframework` by default. The example app (`example/ios/Podfile`) sets `RNLLAMA_BUILD_FROM_SOURCE=1` to build from source, so C++ changes take effect when building the example. `scripts/build-ios.sh` builds device/simulator frameworks for release. Metal is enabled unless `RNLLAMA_DISABLE_METAL=1`.
- **Android:** Library consumers use pre-built `.so` libraries by default. The example app (`example/android/gradle.properties`) sets `rnllamaBuildFromSource=true` to build from source. `scripts/build-android.sh` builds native libs (OpenCL via `scripts/build-opencl.sh`). CMake config lives in `android/build.gradle` and `ios/CMakeLists.txt`.
- **For library users:** Pre-built frameworks/libs are recommended (the default). Only enable build-from-source if you need to modify C++ code or apply custom patches.

## Common Development Commands

```bash
npm install
npm run bootstrap              # Required after cloning (env setup + Metal embeds)
npm run sync:vendor       # Re-vendor vendor/ from VERSIONS + patches
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

- **TypeScript layer:** Edit `src/index.ts`, `src/types.ts`, and `src/jsi.ts`. `NativeRNLlama.install()` only installs JSI; all APIs are invoked via JSI bindings. Run `npm run typecheck` and `npm run lint` before committing.
- **C++ core:** llama.rn code is in `cpp/`; llama.cpp/codec.cpp code is in `vendor/` (edit in place, then regenerate the patch, see below). The example app builds from source, so `npm run build:ios` / `npm run build:android` will compile your C++ changes directly. To move to a newer llama.cpp, edit `LLAMA_CPP_REF` in `vendor/VERSIONS` and run `npm run sync:vendor`. For releasing pre-built frameworks/libs, run `npm run build:ios-frameworks` / `npm run build:android-libs`.
- **JSI bridge/platform glue:** Implement binding logic in `cpp/jsi/*`. iOS installs live in `ios/RNLlama.mm`; Android uses `android/src/main/java/com/rnllama/RNLlama.java` (native loader), `android/src/main/java/com/rnllama/RNLlamaModule.java`, and `android/src/main/RNLlamaJSI.cpp`.

### Patching llama.cpp / codec.cpp

1. Edit the vendored file in place, e.g. `vendor/llama.cpp/common/chat.cpp`
2. Regenerate its patch: `scripts/update-patch.sh llama.cpp common/chat.cpp chat.cpp` writes `scripts/patches/llama.cpp/chat.cpp.patch` (a file upstream lacks becomes a file-creating patch; text above the first `---` line survives regeneration, use it to say why)
3. `npm run sync:vendor` must leave `git status` clean

## Important Conventions

- llama.cpp/ggml symbols keep their upstream names (`ggml_*`, `llama_*`). Coexistence with other ggml-based libraries (e.g. whisper.rn) relies on each library being its own dynamic image: two-level namespace on Apple platforms, `RTLD_LOCAL` plus `-Bsymbolic` on Android. ggml-metal's Objective-C `GGMLMetalClass` (only compiled when the Metal library is not embedded) is renamed per library via a `-D` define, since Objective-C class names are process-global.
- Follow conventional commits (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`).

## Testing Strategy

- **TypeScript/Javascript:** Jest unit tests in `src/__tests__/` (JSI mocked in `jest/mock.js`). Run with `npm test`.
- **C++:** Unit tests in `tests/` (`tests/build_and_test.sh` or `tests/run_tests.sh` after building).
- **Integration:** Example app under `example/` for device validation, GPU/performance checks, and parallel decoding stress.

## Key Files Reference

- `src/index.ts`, `src/jsi.ts`, `src/types.ts`, `src/NativeRNLlama.ts`
- `cpp/jsi/RNLlamaJSI.cpp` (+ helpers: `JSIParams.h/.cpp`, `JSICompletion.h`, `JSISession.h`, `JSIRequestManager.h`, `ThreadPool.*`)
- `cpp/rn-llama.cpp`, `cpp/rn-completion.cpp`, `cpp/rn-slot.cpp`, `cpp/rn-slot-manager.cpp`, `cpp/rn-mtmd.hpp`, `cpp/rn-tts.cpp`
- `vendor/VERSIONS`, `vendor/README.md`, `scripts/sync-vendor.sh`, `scripts/update-patch.sh`, `scripts/patches/`, `cmake/rnllama-sources.cmake`
- `ios/RNLlama.mm`
- `android/src/main/java/com/rnllama/RNLlama.java`
- `android/src/main/java/com/rnllama/RNLlamaModule.java`
- `android/src/main/RNLlamaJSI.cpp`
- `llama-rn.podspec`, `android/build.gradle`, `tests/`
