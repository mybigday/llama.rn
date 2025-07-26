# llama.rn example

This is an example of how to use the llama.rn library.

## Examples

The example app demonstrates various local LLM capabilities:

- **💬 Simple Chat** - Basic chat interface with text generation ([SimpleChatScreen.tsx](src/screens/SimpleChatScreen.tsx))
- **👁️ Vision/Multimodal** - Image analysis and visual question answering ([MultimodalScreen.tsx](src/screens/MultimodalScreen.tsx))
- **🛠️ Tool Calling** - Advanced function calling capabilities ([ToolCallsScreen.tsx](src/screens/ToolCallsScreen.tsx))
- **🔊 Text-to-Speech** - Local voice synthesis with OuteTTS ([TTSScreen.tsx](src/screens/TTSScreen.tsx))
- **📊 Model Info** - Model diagnostics and system information ([ModelInfoScreen.tsx](src/screens/ModelInfoScreen.tsx))

## Requirements

Please back to the root directory and run the following command:

```bash
yarn && yarn bootstrap
```

## iOS

1. Install pods

```bash
yarn pods
```

2. Run the example

```bash
yarn ios
# Use device
yarn ios --device "<device name>"
# With release mode
yarn ios --mode Release
```

## Android

Run the example:
```bash
yarn android
# With release mode
yarn android --mode release
```

## Build with frameworks/libs

This example is build llama.rn from source code by default, you can also build with frameworks/libs.

```bash
# Build iOS frameworks
yarn build:ios-frameworks
# Build Android libs
yarn build:android-libs
```

Then you can setup the environment variable / properties in your project:

iOS:
```bash
RNLLAMA_BUILD_FROM_SOURCE=0 yarn pods
```

Android: Edit `android/gradle.properties` and set `rnllamaBuildFromSource` to `false`.
