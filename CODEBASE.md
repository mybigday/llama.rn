# Inferra LLaMA.rn Codebase Explanation

## Overview

**inferra-llama.rn** is a React Native library that provides JavaScript/TypeScript bindings for running Large Language Models (LLMs) locally on mobile devices using the llama.cpp engine. This library enables developers to build AI-powered mobile applications that work offline with full privacy.

## Project Structure (ASCII Tree)

```
inferra-llama.rn/
├── 📁 src/                           # TypeScript source code (React Native layer)
│   ├── 📄 index.ts                   # Main API exports and LlamaContext class
│   ├── 📄 NativeRNLlama.ts          # Native module interface definitions
│   ├── 📄 grammar.ts                # JSON schema to grammar conversion
│   └── 📁 __tests__/                # Unit tests
├── 📁 cpp/                          # C++ native implementation (Core engine)
│   ├── 📄 rn-llama.h               # Main C++ header for RN bridge
│   ├── 📄 rn-llama.cpp             # Main C++ implementation
│   ├── 📄 sampling.h/.cpp           # Token sampling algorithms
│   ├── 📄 llama*.h/.cpp             # llama.cpp core files
│   ├── 📄 ggml*.h/.c/.cpp           # GGML tensor library
│   ├── 📄 common*.h/.cpp            # Common utilities
│   ├── 📄 chat*.h/.cpp              # Chat template handling
│   ├── 📄 json-schema-to-grammar.*  # Schema validation
│   ├── 📁 ggml-cpu/                 # CPU backend implementation
│   ├── 📁 minja/                    # Jinja template engine
│   └── 📁 tools/                    # Build and utility tools
├── 📁 ios/                          # iOS platform implementation
│   ├── 📄 RNLlama.h/.mm             # iOS bridge module
│   ├── 📄 RNLlamaContext.h/.mm      # iOS context wrapper
│   ├── 📄 CMakeLists.txt            # iOS build configuration
│   ├── 📁 rnllama.xcframework/      # Pre-built iOS framework
│   └── 📁 RNLlama.xcodeproj/        # Xcode project
├── 📁 android/                      # Android platform implementation
│   ├── 📄 build.gradle              # Android build configuration
│   ├── 📁 src/main/
│   │   ├── 📄 jni.cpp               # JNI bridge implementation
│   │   ├── 📄 jni-utils.h           # JNI utilities
│   │   ├── 📄 CMakeLists.txt        # Android CMake build
│   │   ├── 📁 java/com/rnllama/
│   │   │   ├── 📄 RNLlama.java      # Android module class
│   │   │   ├── 📄 LlamaContext.java # Android context wrapper
│   │   │   └── 📄 RNLlamaPackage.java # React Native package
│   │   └── 📁 jniLibs/              # Pre-built native libraries
├── 📁 scripts/                      # Build and setup scripts
│   ├── 📄 bootstrap.sh              # Project setup script
│   ├── 📄 build-ios.sh              # iOS framework builder
│   ├── 📄 build-android.sh          # Android library builder
│   └── 📁 patches/                  # Source code patches
├── 📁 example/                      # Example React Native app
│   ├── 📄 package.json              # Example app dependencies
│   ├── 📁 src/                      # Example app source
│   ├── 📁 ios/                      # Example iOS project
│   └── 📁 android/                  # Example Android project
├── 📁 docs/                         # Documentation
│   └── 📁 API/                      # Auto-generated API docs
├── 📁 llama.cpp/                    # Git submodule (upstream llama.cpp)
├── 📄 package.json                  # NPM package configuration
├── 📄 README.md                     # Project documentation
├── 📄 llama-rn.podspec              # iOS CocoaPods specification
└── 📄 tsconfig.json                 # TypeScript configuration
```

## Detailed Component Analysis

### 1. TypeScript Layer (`src/`)

#### `src/index.ts` - Main API Interface
This is the primary entry point that developers interact with. It provides:

**Key Classes:**
- **`LlamaContext`**: Main context class for managing LLM instances
  - `id`: Unique context identifier
  - `gpu`: GPU usage status
  - `model`: Model metadata and capabilities
  - `loadSession()`: Load saved conversation state
  - `saveSession()`: Save conversation state
  - `completion()`: Generate text completions
  - `tokenize()`/`detokenize()`: Token conversion utilities
  - `embedding()`: Generate text embeddings
  - `bench()`: Performance benchmarking
  - `applyLoraAdapters()`/`removeLoraAdapters()`: LoRA model management
  - `initMultimodal()`: Initialize vision/audio capabilities

**Key Functions:**
- `initLlama()`: Initialize new LLM context
- `releaseAllLlama()`: Clean up all contexts
- `toggleNativeLog()`: Control native logging
- `addNativeLogListener()`: Monitor native logs
- `loadLlamaModelInfo()`: Get model metadata

**Event Handling:**
- Token streaming via callbacks
- Progress monitoring during model loading
- Native log forwarding to JavaScript

#### `src/NativeRNLlama.ts` - Native Bridge Definitions
Defines the TypeScript interface for the native module using React Native's TurboModule system:

**Key Types:**
- `NativeContextParams`: Model initialization parameters
- `NativeCompletionParams`: Text generation settings
- `NativeCompletionResult`: Generation results with timing data
- `NativeTokenizeResult`: Tokenization results with media support
- `NativeEmbeddingResult`: Vector embeddings
- `NativeLlamaContext`: Context metadata and capabilities

**Native Interface (`Spec`):**
- `initContext()`: Create new LLM context
- `completion()`: Generate text
- `tokenize()`: Convert text to tokens
- `embedding()`: Generate embeddings
- `loadSession()`/`saveSession()`: Session management
- `bench()`: Performance testing
- Multimodal methods for vision/audio

#### `src/grammar.ts` - Structured Output Support
Implements JSON schema to grammar conversion for constrained generation:

**Core Class: `SchemaGrammarConverter`**
- Converts JSON schemas to llama.cpp grammar format
- Ensures LLM outputs conform to specific structures
- Supports primitive types, objects, arrays, enums
- Pattern matching for strings, dates, times
- Reference resolution for complex schemas

**Key Features:**
- `buildRepetition()`: Handle array/object repetitions
- `formatLiteral()`: Escape special characters
- `convertJsonSchemaToGrammar()`: Main conversion function
- Built-in rules for common data types

### 2. C++ Native Layer (`cpp/`)

#### `cpp/rn-llama.h/.cpp` - React Native Bridge
Core C++ implementation that bridges React Native with llama.cpp:

**Main Class: `llama_rn_context`**
```cpp
struct llama_rn_context {
    // State management
    bool is_predicting = false;
    bool is_interrupted = false;
    std::string generated_text;
    
    // Model and context
    llama_model *model = nullptr;
    llama_context *ctx = nullptr;
    common_sampler *ctx_sampling = nullptr;
    
    // Core methods
    bool loadModel(common_params &params_);
    completion_token_output doCompletion();
    std::vector<float> getEmbedding();
    llama_rn_tokenize_result tokenize();
    
    // Multimodal support
    bool initMultimodal(const std::string &mmproj_path, bool use_gpu);
    void processMedia(const std::string &prompt, const std::vector<std::string> &media_paths);
}
```

**Key Responsibilities:**
- Memory management for LLM contexts
- Token generation and sampling
- Multimodal processing (images/audio)
- LoRA adapter management
- Session state persistence

#### `cpp/sampling.h/.cpp` - Token Sampling
Implements various sampling algorithms for text generation:
- Top-k sampling
- Top-p (nucleus) sampling
- Temperature scaling
- Repetition penalties
- Grammar-guided sampling

#### Core llama.cpp Files
The library includes the complete llama.cpp implementation:
- **`llama.h/.cpp`**: Main LLM inference engine
- **`llama-model.*`**: Model loading and management
- **`llama-context.*`**: Inference context handling
- **`llama-kv-cache.*`**: Key-value cache for attention
- **`llama-sampling.*`**: Advanced sampling algorithms
- **`ggml.*`**: Low-level tensor operations
- **`ggml-metal.m`**: GPU acceleration for iOS (Metal)
- **`ggml-cpu/`**: CPU-optimized implementations

### 3. iOS Implementation (`ios/`)

#### `ios/RNLlama.h/.mm` - iOS Bridge Module
Objective-C++ bridge that implements the React Native TurboModule interface:

```objc
@interface RNLlama : RCTEventEmitter <RCTBridgeModule>
- (void)initContext:(double)contextId 
             params:(NSDictionary *)params 
           resolver:(RCTPromiseResolveBlock)resolve 
           rejecter:(RCTPromiseRejectBlock)reject;
- (void)completion:(double)contextId 
            params:(NSDictionary *)params 
          resolver:(RCTPromiseResolveBlock)resolve 
          rejecter:(RCTPromiseRejectBlock)reject;
@end
```

#### `ios/RNLlamaContext.h/.mm` - iOS Context Wrapper
Manages C++ context instances and provides Objective-C interface:
- Converts between Objective-C types and C++ types
- Handles memory management across language boundaries
- Provides thread-safe access to C++ objects
- Implements progress callbacks for model loading

#### `ios/CMakeLists.txt` - iOS Build Configuration
Configures the iOS build process:
- Links Metal framework for GPU acceleration
- Sets up C++ standard library
- Configures architecture-specific optimizations
- Includes necessary system frameworks

### 4. Android Implementation (`android/`)

#### `android/src/main/java/com/rnllama/RNLlama.java`
Main Android module that implements React Native's TurboModule interface:

```java
public class RNLlama extends NativeRNLlamaSpec {
    @Override
    public void initContext(double contextId, ReadableMap params, Promise promise) {
        // Initialize native context via JNI
    }
    
    @Override
    public void completion(double contextId, ReadableMap params, Promise promise) {
        // Perform text completion via JNI
    }
}
```

#### `android/src/main/java/com/rnllama/LlamaContext.java`
Android context wrapper that manages native resources:
- Thread management for background processing
- Memory management and cleanup
- Event emission to React Native
- Progress callback handling

#### `android/src/main/jni.cpp` - JNI Bridge
C++ implementation of Java Native Interface:
- Converts between Java types and C++ types
- Provides thread-safe access to native contexts
- Handles exceptions across JNI boundary
- Manages native library lifecycle

### 5. Build System (`scripts/`)

#### `scripts/bootstrap.sh` - Project Setup
Comprehensive setup script that:
- Initializes git submodules (llama.cpp)
- Applies necessary patches
- Downloads pre-built binaries
- Configures platform-specific settings
- Validates development environment

#### `scripts/build-ios.sh` - iOS Framework Builder
Builds the iOS xcframework:
- Compiles for multiple architectures (arm64, x86_64)
- Creates universal framework
- Optimizes for release builds
- Includes Metal shaders compilation

#### `scripts/build-android.sh` - Android Library Builder
Builds Android native libraries:
- Cross-compiles for ARM64 and x86_64
- Configures NDK toolchain
- Optimizes for mobile performance
- Creates architecture-specific libraries

### 6. Example Application (`example/`)

Demonstrates library usage with a complete React Native app:
- Model loading and initialization
- Text completion with streaming
- Chat interface implementation
- Multimodal capabilities (if supported)
- Performance benchmarking
- Session management

## Key Features Explained

### 1. **Multimodal Support**
- **Vision**: Process images alongside text prompts
- **Audio**: Handle audio inputs for speech-to-text scenarios
- **Integration**: Seamlessly combine media with text generation

### 2. **Memory Management**
- **Context Limits**: Configurable context size limits
- **Memory Mapping**: Efficient model loading using mmap
- **Cache Management**: KV-cache optimization for performance

### 3. **Platform Optimization**
- **iOS Metal**: GPU acceleration using Apple's Metal framework
- **Android CPU**: Optimized CPU implementations for ARM and x86
- **SIMD**: Advanced SIMD optimizations for mathematical operations

### 4. **Streaming and Real-time Processing**
- **Token Streaming**: Real-time token generation with callbacks
- **Progress Tracking**: Model loading progress notifications
- **Interruption**: Ability to stop generation mid-process

### 5. **Advanced Features**
- **LoRA Adapters**: Support for Low-Rank Adaptation models
- **Grammar Constraints**: Force outputs to follow specific formats
- **Session Persistence**: Save and restore conversation states
- **Benchmarking**: Performance testing and optimization

## Development Workflow

1. **Setup**: Run `scripts/bootstrap.sh` to initialize the project
2. **Build**: Use platform-specific build scripts for native libraries
3. **Test**: Run the example app to validate functionality
4. **Package**: Use `npm pack` to create distributable package

## Architecture Principles

1. **Layered Architecture**: Clear separation between JS, native bridge, and C++ core
2. **Platform Abstraction**: Unified API across iOS and Android
3. **Memory Safety**: Careful resource management across language boundaries
4. **Performance Focus**: Optimized for mobile device constraints
5. **Modularity**: Extensible design for future enhancements

This codebase represents a sophisticated integration of multiple technologies to bring powerful LLM capabilities to mobile React Native applications while maintaining performance, safety, and cross-platform compatibility. 