# llama.rn

[![Actions Status](https://github.com/mybigday/llama.rn/workflows/CI/badge.svg)](https://github.com/mybigday/llama.rn/actions)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![npm](https://img.shields.io/npm/v/llama.rn.svg)](https://www.npmjs.com/package/llama.rn/)

React Native binding of [llama.cpp](https://github.com/ggerganov/llama.cpp). Currently only supported iOS.

⚠️ Currently this library is not recommended for production use. In our cases, we only use it on device like M1 ~ M2 iPad/Mac for the time being, with Llama-2-7b-chat q2_k ~ q4_k models. ⚠️

## Installation

```sh
npm install llama.rn
```

For iOS, please re-run `npx pod-install` again.

## Obtain the model

Download the Llama 2 model
1. Request access from [here](https://ai.meta.com/llama)
2. Download the model from HuggingFace [here](https://huggingface.co/meta-llama/Llama-2-7b-chat) (`Llama-2-7b-chat`)

Convert the model to ggml format
```bash
# Start with submodule in this repo (or you can clone the repo https://github.com/ggerganov/llama.cpp.git)
yarn && yarn bootstrap
cd llama.cpp

# install Python dependencies
python3 -m pip install -r requirements.txt

# Move the Llama model weights to the models folder
mv <path to Llama-2-7b-chat> ./models/7B

# convert the 7B model to ggml FP16 format
python3 convert.py models/7B/ --outtype f16

# Build the quantize tool
make quantize

# quantize the model to 2-bits (using q2_k method)
./quantize ./models/7B/ggml-model-f16.gguf ./models/7B/ggml-model-q2_k.gguf q2_k

# quantize the model to 4-bits (using q4_0 method)
./quantize ./models/7B/ggml-model-f16.gguf ./models/7B/ggml-model-q4_0.gguf q4_0
```

## Usage

Documentation is WIP. Currently you can visit the [example](example) to see how to use it.

Run the example:
```bash
yarn && yarn bootstrap

yarn example ios
# Use device
yarn example ios --device "<device name>"
# With release mode
yarn example ios --mode Release
```

this example used [react-native-document-picker](https://github.com/rnmods/react-native-document-picker) to select model. You can move the model to iOS Simulator, or iCloud for real device.

## Mock `llama.rn`

We have provided a mock version of `llama.rn` for testing purpose you can use on Jest:

```js
jest.mock('llama.rn', () => require('llama.rn/jest/mock'))
```

## NOTE

- The [Extended Virtual Addressing](https://developer.apple.com/documentation/bundleresources/entitlements/com_apple_developer_kernel_extended-virtual-addressing) capability is recommended to enable on iOS project.
- Metal:
  - We have tested to know some devices is not able to use Metal ('params.n_gpu_layers > 0') due to llama.cpp used SIMD-scoped operation, you can check if your device is supported in [Metal feature set tables](https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf), Apple7 GPU will be the minimum requirement.
  - It's also not supported in iOS simulator due to [this limitation](https://developer.apple.com/documentation/metal/developing_metal_apps_that_run_in_simulator#3241609), we used constant buffers more than 14.

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
