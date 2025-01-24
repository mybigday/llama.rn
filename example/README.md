# llama.rn example

This is an example of how to use the llama.rn library.

This example used [react-native-document-picker](https://github.com/rnmods/react-native-document-picker) for select model.

- iOS: You can move the model to iOS Simulator, or iCloud for real device.
- Android: Selected file will be copied or downloaded to cache directory so it may be slow.

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
