# llama.rn

React Native binding of [llama.cpp](https://github.com/ggerganov/llama.cpp). Currently only supported iOS.

⚠️ Currently this library is not recommended for production use. In our cases, we only use it on device like M1 ~ M2 iPad/Mac for the time being, with Llama-2-7b-chat q2_k ~ q4_k models. ⚠️

## Installation

```sh
npm install llama.rn
```

For iOS, please re-run `npx pod-install` again.

## Usage

Documentation is WIP. Currently you can visit the [example](example) to see how to use it.

Run the example:
```bash
yarn && yarn bootstrap

yarn example ios
# Use devoce
yarn example ios --device "<device name>"
# With release mode
yarn example ios --mode Release
```

## TODO

- [ ] Expose tokenize / embedding functions as util
- [ ] Android support

## NOTE

- The [Extended Virtual Addressing](https://developer.apple.com/documentation/bundleresources/entitlements/com_apple_developer_kernel_extended-virtual-addressing) capability is recommended to enable on iOS project.
- Currently we got some iOS devices crash by enable Metal ('options.gpuLayers > 1'), to avoid this problem, we're recommended to check [Metal 3 supported devices](https://support.apple.com/en-us/HT205073). But currently the cause is still unclear and we are giving this issue a low priority.
- We can use the ggml tensor allocor (See [llama.cpp#2411](https://github.com/ggerganov/llama.cpp/pull/2411)) by use `RNLLAMA_DISABLE_METAL=1` env on pod install, which reduces the memory usage. If you only want to use CPU, this is very useful.

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
