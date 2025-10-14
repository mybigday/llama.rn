# llama.rn example

This is an example project to show how to use the llama.rn library.

For iPhone/iPad/Mac, you can try it by downloading our test app from [TestFlight](https://testflight.apple.com/join/MmzGSneU).

For Android, you can try it by downloading our test app from [Releases](https://github.com/mybigday/llama.rn/releases) (example-app-release.apk).

## Examples

The example app demonstrates various local LLM capabilities:

- **💬 Simple Chat** - Basic chat interface with text generation ([SimpleChatScreen.tsx](src/screens/SimpleChatScreen.tsx))
- **👁️ Multimodal** - Image/audio analysis and visual/audio question answering ([MultimodalScreen.tsx](src/screens/MultimodalScreen.tsx))
- **🛠️ Tool Calling & MCP** - Advanced function calling capabilities with custom tools (mock responses) and MCP integration ([ToolCallsScreen.tsx](src/screens/ToolCallsScreen.tsx))
- **📊 Embedding** - Vector embeddings and semantic search in memory & Rerank ([EmbeddingScreen.tsx](src/screens/EmbeddingScreen.tsx))
- **⚡ Parallel Decoding** - Concurrent request processing with multiple parallel slots ([ParallelDecodingScreen.tsx](src/screens/ParallelDecodingScreen.tsx))
- **🔊 Text-to-Speech** - Local voice synthesis with OuteTTS ([TTSScreen.tsx](src/screens/TTSScreen.tsx))
- **📊 Model Info** - Model diagnostics and system information ([ModelInfoScreen.tsx](src/screens/ModelInfoScreen.tsx))

Used models are listed in [src/utils/constants.ts](src/utils/constants.ts).

## Requirements

Please back to the root directory and run the following command:

```bash
npm install && npm run bootstrap
```

## iOS

1. Install pods

```bash
npm run pods
```

2. Run the example

```bash
npm run ios
# Use device
npm run ios -- --device "<device name>"
# With release mode
npm run ios -- --mode Release
```

## Android

Run the example:
```bash
npm run android
# With release mode
npm run android -- --mode release
```

## Build with frameworks/libs

This example is build llama.rn from source code by default, you can also build with frameworks/libs.

```bash
# Build iOS frameworks
npm run build:ios-frameworks
# Build Android libs
npm run build:android-libs
```

Then you can setup the environment variable / properties in your project:

iOS:
```bash
RNLLAMA_BUILD_FROM_SOURCE=0 npm run pods
```

Android: Edit `android/gradle.properties` and set `rnllamaBuildFromSource` to `false`.

## Roadmap

The following features are planned for future updates:

- [x] **🔧 Add custom model modal** - Interface for loading and managing custom models
- [x] **📊 Add embedding example** - Demonstrate text embedding and similarity search capabilities
- [x] **🛠️ ToolCallsScreen: Support MCP** - Integrate Model Context Protocol via [`mcp-sdk-client-ssejs`](https://github.com/mybigday/mcp-sdk-client-ssejs)
- [x] **🔍 Add reranker example** - Show document reranking for improved search relevance
- [ ] **⚙️ Check hardware requirement for model** - Validate device capabilities before model loading
