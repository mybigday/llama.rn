# llama.rn

[![Actions Status](https://github.com/mybigday/llama.rn/workflows/CI/badge.svg)](https://github.com/mybigday/llama.rn/actions)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![npm](https://img.shields.io/npm/v/llama.rn.svg)](https://www.npmjs.com/package/llama.rn/)

React Native binding of [llama.cpp](https://github.com/ggerganov/llama.cpp).

[llama.cpp](https://github.com/ggerganov/llama.cpp): Inference of [LLaMA](https://arxiv.org/abs/2302.13971) model in pure C/C++

## Installation

```sh
npm install llama.rn
```

#### iOS

Please re-run `npx pod-install` again.

#### Android

Add proguard rule if it's enabled in project (android/app/proguard-rules.pro):

```proguard
# llama.rn
-keep class com.rnllama.** { *; }
```

## Obtain the model

You can search HuggingFace for available models (Keyword: [`GGUF`](https://huggingface.co/search/full-text?q=GGUF&type=model)).

For create a GGUF model manually, for example in Llama 2:

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

```js
import { initLlama } from 'llama.rn'

// Initial a Llama context with the model (may take a while)
const context = await initLlama({
  model: 'file://<path to gguf model>',
  use_mlock: true,
  n_ctx: 2048,
  n_gpu_layers: 1, // > 0: enable Metal on iOS
  // embedding: true, // use embedding
})

// Do completion
const { text, timings } = await context.completion(
  {
    prompt: 'This is a conversation between user and llama, a friendly chatbot. respond in simple markdown.\n\nUser: Hello!\nLlama:',
    n_predict: 100,
    stop: ['</s>', 'Llama:', 'User:'],
    // n_threads: 4,
  },
  (data) => {
    // This is a partial completion callback
    const { token } = data
  },
)
console.log('Result:', text)
console.log('Timings:', timings)
```

The binding’s deisgn inspired by [server.cpp](https://github.com/ggerganov/llama.cpp/tree/master/examples/server) example in llama.cpp, so you can map its API to LlamaContext:

- `/completion`: `context.completion(params, partialCompletionCallback)`
- `/tokenize`: `context.tokenize(content)`
- `/detokenize`: `context.detokenize(tokens)`
- `/embedding`: `context.embedding(content)`
- Other methods
  - `context.loadSession(path)`
  - `context.saveSession(path)`
  - `context.stopCompletion()`
  - `context.release()`

Please visit the [Documentation](docs/API) for more details.

You can also visit the [example](example) to see how to use it.

Run the example:
```bash
yarn && yarn bootstrap

# iOS
yarn example ios
# Use device
yarn example ios --device "<device name>"
# With release mode
yarn example ios --mode Release

# Android
yarn example android
# With release mode
yarn example android --mode release
```

This example used [react-native-document-picker](https://github.com/rnmods/react-native-document-picker) for select model.

- iOS: You can move the model to iOS Simulator, or iCloud for real device.
- Android: Selected file will be copied or downloaded to cache directory so it may be slow.

## Grammar Sampling

GBNF (GGML BNF) is a format for defining [formal grammars](https://en.wikipedia.org/wiki/Formal_grammar) to constrain model outputs in `llama.cpp`. For example, you can use it to force the model to generate valid JSON, or speak only in emojis.

You can see [GBNF Guide](https://github.com/ggerganov/llama.cpp/tree/master/grammars) for more details.

`llama.rn` provided a built-in function to convert JSON Schema to GBNF:

```js
import { initLlama, convertJsonSchemaToGrammar } from 'llama.rn'

const schema = { /* JSON Schema, see below */ }

const context = await initLlama({
  model: 'file://<path to gguf model>',
  use_mlock: true,
  n_ctx: 2048,
  n_gpu_layers: 1, // > 0: enable Metal on iOS
  // embedding: true, // use embedding
  grammar: convertJsonSchemaToGrammar({
    schema,
    propOrder: { function: 0, arguments: 1 },
  })
})

const { text } = await context.completion({
  prompt: 'Schedule a birthday party on Aug 14th 2023 at 8pm.',
})
console.log('Result:', text)
// Example output:
// {"function": "create_event","arguments":{"date": "Aug 14th 2023", "time": "8pm", "title": "Birthday Party"}}
```

<details>
<summary>JSON Schema example (Define function get_current_weather / create_event / image_search)</summary>

```json5
{
  oneOf: [
    {
      type: "object",
      name: "get_current_weather",
      description: "Get the current weather in a given location",
      properties: {
        function: {
          const: "get_current_weather",
        },
        arguments: {
          type: "object",
          properties: {
            location: {
              type: "string",
              description: "The city and state, e.g. San Francisco, CA",
            },
            unit: {
              type: "string",
              enum: ["celsius", "fahrenheit"],
            },
          },
          required: ["location"],
        },
      },
    },
    {
      type: "object",
      name: "create_event",
      description: "Create a calendar event",
      properties: {
        function: {
          const: "create_event",
        },
        arguments: {
          type: "object",
          properties: {
            title: {
              type: "string",
              description: "The title of the event",
            },
            date: {
              type: "string",
              description: "The date of the event",
            },
            time: {
              type: "string",
              description: "The time of the event",
            },
          },
          required: ["title", "date", "time"],
        },
      },
    },
    {
      type: "object",
      name: "image_search",
      description: "Search for an image",
      properties: {
        function: {
          const: "image_search",
        },
        arguments: {
          type: "object",
          properties: {
            query: {
              type: "string",
              description: "The search query",
            },
          },
          required: ["query"],
        },
      },
    },
  ],
}
```
</details>

<details>
<summary>Converted GBNF looks like</summary>

```bnf
space ::= " "?
0-function ::= "\"get_current_weather\""
string ::=  "\"" (
        [^"\\] |
        "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F])
      )* "\"" space
0-arguments-unit ::= "\"celsius\"" | "\"fahrenheit\""
0-arguments ::= "{" space "\"location\"" space ":" space string "," space "\"unit\"" space ":" space 0-arguments-unit "}" space
0 ::= "{" space "\"function\"" space ":" space 0-function "," space "\"arguments\"" space ":" space 0-arguments "}" space
1-function ::= "\"create_event\""
1-arguments ::= "{" space "\"date\"" space ":" space string "," space "\"time\"" space ":" space string "," space "\"title\"" space ":" space string "}" space
1 ::= "{" space "\"function\"" space ":" space 1-function "," space "\"arguments\"" space ":" space 1-arguments "}" space
2-function ::= "\"image_search\""
2-arguments ::= "{" space "\"query\"" space ":" space string "}" space
2 ::= "{" space "\"function\"" space ":" space 2-function "," space "\"arguments\"" space ":" space 2-arguments "}" space
root ::= 0 | 1 | 2
```
</details>

## Mock `llama.rn`

We have provided a mock version of `llama.rn` for testing purpose you can use on Jest:

```js
jest.mock('llama.rn', () => require('llama.rn/jest/mock'))
```

## NOTE

iOS:
- The [Extended Virtual Addressing](https://developer.apple.com/documentation/bundleresources/entitlements/com_apple_developer_kernel_extended-virtual-addressing) capability is recommended to enable on iOS project.
- Metal:
  - We have tested to know some devices is not able to use Metal ('params.n_gpu_layers > 0') due to llama.cpp used SIMD-scoped operation, you can check if your device is supported in [Metal feature set tables](https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf), Apple7 GPU will be the minimum requirement.
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
