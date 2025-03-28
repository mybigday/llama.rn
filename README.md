# llama.rn

[![Actions Status](https://github.com/sbhjt-gr/ragionare-llama.rn/workflows/CI/badge.svg)](https://github.com/sbhjt-gr/ragionare-llama.rn/actions)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![npm](https://img.shields.io/npm/v/llama.rn.svg)](https://www.npmjs.com/package/ragionare-llama.rn/)

React Native binding of [llama.cpp](https://github.com/ggerganov/llama.cpp) for Ragionare.

[llama.cpp](https://github.com/ggerganov/llama.cpp): Inference of [LLaMA](https://arxiv.org/abs/2302.13971) model in pure C/C++

#### iOS

Please re-run `npx pod-install` again.

By default, `llama.rn` will use pre-built `rnllama.xcframework` for iOS. If you want to build from source, please set `RNLLAMA_BUILD_FROM_SOURCE` to `1` in your Podfile.

#### Android

Add proguard rule if it's enabled in project (android/app/proguard-rules.pro):

```proguard
# llama.rn
-keep class com.rnllama.** { *; }
```

By default, `llama.rn` will use pre-built libraries for Android. If you want to build from source, please set `rnllamaBuildFromSource` to `true` in `android/gradle.properties`.

## NOTE

iOS:

- The [Extended Virtual Addressing](https://developer.apple.com/documentation/bundleresources/entitlements/com_apple_developer_kernel_extended-virtual-addressing) capability is recommended to enable on iOS project.
- Metal:
  - We have tested to know some devices is not able to use Metal (GPU) due to llama.cpp used SIMD-scoped operation, you can check if your device is supported in [Metal feature set tables](https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf), Apple7 GPU will be the minimum requirement.
  - It's also not supported in iOS simulator due to [this limitation](https://developer.apple.com/documentation/metal/developing_metal_apps_that_run_in_simulator#3241609), we used constant buffers more than 14.

Android:

- Currently only supported arm64-v8a / x86_64 platform, this means you can't initialize a context on another platforms. The 64-bit platform are recommended because it can allocate more memory for the model.
- No integrated any GPU backend yet.

## Contributing

See the [contributing guide](CONTRIBUTING.md) to learn how to contribute to the repository and the development workflow.

## License

MIT

---

Made with [create-react-native-library](https://github.com/callstack/react-native-builder-bob)

---

<p align="center">
  <a href="https://bricks.tools">
    <img width="90px" src="https://avatars.githubusercontent.com/u/17320237?s=200&v=4">
  </a>
  <p align="center">
    Built and maintained by <a href="https://bricks.tools">BRICKS</a>.
  </p>
</p>
