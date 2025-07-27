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

By default, `llama.rn` will use pre-built `rnllama.xcframework` for iOS. If you want to build from source, please set `RNLLAMA_BUILD_FROM_SOURCE` to `1` in your Podfile.

#### Android

Add proguard rule if it's enabled in project (android/app/proguard-rules.pro):

```proguard
# llama.rn
-keep class com.rnllama.** { *; }
```

By default, `llama.rn` will use pre-built libraries for Android. If you want to build from source, please set `rnllamaBuildFromSource` to `true` in `android/gradle.properties`.

## Obtain the model

You can search HuggingFace for available models (Keyword: [`GGUF`](https://huggingface.co/search/full-text?q=GGUF&type=model)).

For get a GGUF model or quantize manually, see [`Prepare and Quantize`](https://github.com/ggerganov/llama.cpp?tab=readme-ov-file#prepare-and-quantize) section in llama.cpp.

## Usage

> **💡 New!** `llama.rn` now supports **multimodal models** with vision and audio capabilities! See the [Multimodal section](#multimodal-vision--audio) for details.

Load model info only:

```js
import { loadLlamaModelInfo } from 'llama.rn'

const modelPath = 'file://<path to gguf model>'
console.log('Model Info:', await loadLlamaModelInfo(modelPath))
```

Initialize a Llama context & do completion:

```js
import { initLlama } from 'llama.rn'

// Initial a Llama context with the model (may take a while)
const context = await initLlama({
  model: modelPath,
  use_mlock: true,
  n_ctx: 2048,
  n_gpu_layers: 99, // number of layers to store in VRAM (Currently only for iOS)
  // embedding: true, // use embedding
})

const stopWords = ['</s>', '<|end|>', '<|eot_id|>', '<|end_of_text|>', '<|im_end|>', '<|EOT|>', '<|END_OF_TURN_TOKEN|>', '<|end_of_turn|>', '<|endoftext|>']

// Do chat completion
const msgResult = await context.completion(
  {
    messages: [
      {
        role: 'system',
        content: 'This is a conversation between user and assistant, a friendly chatbot.',
      },
      {
        role: 'user',
        content: 'Hello!',
      },
    ],
    n_predict: 100,
    stop: stopWords,
    // ...other params
  },
  (data) => {
    // This is a partial completion callback
    const { token } = data
  },
)
console.log('Result:', msgResult.text)
console.log('Timings:', msgResult.timings)

// Or do text completion
const textResult = await context.completion(
  {
    prompt: 'This is a conversation between user and llama, a friendly chatbot. respond in simple markdown.\n\nUser: Hello!\nLlama:',
    n_predict: 100,
    stop: [...stopWords, 'Llama:', 'User:'],
    // ...other params
  },
  (data) => {
    // This is a partial completion callback
    const { token } = data
  },
)
console.log('Result:', textResult.text)
console.log('Timings:', textResult.timings)
```

The binding's deisgn inspired by [server.cpp](https://github.com/ggerganov/llama.cpp/tree/master/examples/server) example in llama.cpp:

- `/completion` and `/chat/completions`: `context.completion(params, partialCompletionCallback)`
- `/tokenize`: `context.tokenize(content)`
- `/detokenize`: `context.detokenize(tokens)`
- `/embedding`: `context.embedding(content)`
- `/rerank`: `context.rerank(query, documents, params)`
- ... Other methods

Please visit the [Documentation](docs/API) for more details.

You can also visit the [example](example) to see how to use it.

## Multimodal (Vision & Audio)

`llama.rn` supports multimodal capabilities including vision (images) and audio processing. This allows you to interact with models that can understand both text and media content.

### Supported Media Formats

**Images (Vision):**
- JPEG, PNG, BMP, GIF, TGA, HDR, PIC, PNM
- Base64 encoded images (data URLs)
- Local file paths
- \* Not supported HTTP URLs yet

**Audio:**
- WAV, MP3 formats
- Base64 encoded audio (data URLs)
- Local file paths
- \* Not supported HTTP URLs yet

### Setup

First, you need a multimodal model and its corresponding multimodal projector (mmproj) file, see [how to obtain mmproj](https://github.com/ggml-org/llama.cpp/tree/master/tools/mtmd#how-to-obtain-mmproj) for more details.

### Initialize Multimodal Support

```js
import { initLlama } from 'llama.rn'

// First initialize the model context
const context = await initLlama({
  model: 'path/to/your/multimodal-model.gguf',
  n_ctx: 4096,
  n_gpu_layers: 99, // Recommended for multimodal models
  // Important: Disable context shifting for multimodal
  ctx_shift: false,
})

// Initialize multimodal support with mmproj file
const success = await context.initMultimodal({
  path: 'path/to/your/mmproj-model.gguf',
  use_gpu: true, // Recommended for better performance
})

// Check if multimodal is enabled
console.log('Multimodal enabled:', await context.isMultimodalEnabled())

if (success) {
  console.log('Multimodal support initialized!')

  // Check what modalities are supported
  const support = await context.getMultimodalSupport()
  console.log('Vision support:', support.vision)
  console.log('Audio support:', support.audio)
} else {
  console.log('Failed to initialize multimodal support')
}

// Release multimodal context
await context.releaseMultimodal()
```

### Usage Examples

#### Vision (Image Processing)

```js
const result = await context.completion({
  messages: [
    {
      role: 'user',
      content: [
        {
          type: 'text',
          text: 'What do you see in this image?',
        },
        {
          type: 'image_url',
          image_url: {
            url: 'file:///path/to/image.jpg',
            // or base64: 'data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEAYABgAAD...'
          },
        },
      ],
    },
  ],
  n_predict: 100,
  temperature: 0.1,
})

console.log('AI Response:', result.text)
```

#### Audio Processing

```js
// Method 1: Using structured message content (Recommended)
const result = await context.completion({
  messages: [
    {
      role: 'user',
      content: [
        {
          type: 'text',
          text: 'Transcribe or describe this audio:',
        },
        {
          type: 'input_audio',
          input_audio: {
            data: 'data:audio/wav;base64,UklGRiQAAABXQVZFZm10...',
            // or url: 'file:///path/to/audio.wav',
            format: 'wav', // or 'mp3'
          },
        },
      ],
    },
  ],
  n_predict: 200,
})

console.log('Transcription:', result.text)
```

### Tokenization with Media

```js
// Tokenize text with media
const tokenizeResult = await context.tokenize(
  'Describe this image: <__media__>',
  {
    media_paths: ['file:///path/to/image.jpg']
  }
)

console.log('Tokens:', tokenizeResult.tokens)
console.log('Has media:', tokenizeResult.has_media)
console.log('Media positions:', tokenizeResult.chunk_pos_media)
```

### Notes

- **Context Shifting**: Multimodal models require `ctx_shift: false` to maintain media token positioning
- **Memory**: Multimodal models require more memory; use adequate `n_ctx` and consider GPU offloading
- **Media Markers**: The system automatically handles `<__media__>` markers in prompts. When using structured message content, media items are automatically replaced with this marker
- **Model Compatibility**: Ensure your model supports the media type you're trying to process

## Tool Calling

`llama.rn` has universal tool call support by using [minja](https://github.com/google/minja) (as Jinja template parser) and [chat.cpp](https://github.com/ggerganov/llama.cpp/blob/master/common/chat.cpp) in llama.cpp.

Example:

```js
import { initLlama } from 'llama.rn'

const context = await initLlama({
  // ...params
})

const { text, tool_calls } = await context.completion({
  // ...params
  jinja: true, // Enable Jinja template parser
  tool_choice: 'auto',
  tools: [
    {
      type: 'function',
      function: {
        name: 'ipython',
        description:
          'Runs code in an ipython interpreter and returns the result of the execution after 60 seconds.',
        parameters: {
          type: 'object',
          properties: {
            code: {
              type: 'string',
              description: 'The code to run in the ipython interpreter.',
            },
          },
          required: ['code'],
        },
      },
    },
  ],
  messages: [
    {
      role: 'system',
      content: 'You are a helpful assistant that can answer questions and help with tasks.',
    },
    {
      role: 'user',
      content: 'Test',
    },
  ],
})
console.log('Result:', text)
// If tool_calls is not empty, it means the model has called the tool
if (tool_calls) console.log('Tool Calls:', tool_calls)
```

You can check [chat.cpp](https://github.com/ggerganov/llama.cpp/blob/6eecde3cc8fda44da7794042e3668de4af3c32c6/common/chat.cpp#L7-L23) for models has native tool calling support, or it will fallback to `GENERIC` type tool call.

The generic tool call will be always JSON object as output, the output will be like `{"response": "..."}` when it not decided to use tool call.

## Grammar Sampling

GBNF (GGML BNF) is a format for defining [formal grammars](https://en.wikipedia.org/wiki/Formal_grammar) to constrain model outputs in `llama.cpp`. For example, you can use it to force the model to generate valid JSON, or speak only in emojis.

You can see [GBNF Guide](https://github.com/ggerganov/llama.cpp/tree/master/grammars) for more details.

`llama.rn` provided a built-in function to convert JSON Schema to GBNF:

Example gbnf grammar:
```bnf
root   ::= object
value  ::= object | array | string | number | ("true" | "false" | "null") ws

object ::=
  "{" ws (
            string ":" ws value
    ("," ws string ":" ws value)*
  )? "}" ws

array  ::=
  "[" ws (
            value
    ("," ws value)*
  )? "]" ws

string ::=
  "\"" (
    [^"\\\x7F\x00-\x1F] |
    "\\" (["\\bfnrt] | "u" [0-9a-fA-F]{4}) # escapes
  )* "\"" ws

number ::= ("-"? ([0-9] | [1-9] [0-9]{0,15})) ("." [0-9]+)? ([eE] [-+]? [0-9] [1-9]{0,15})? ws

# Optional space: by convention, applied in this grammar after literal chars when allowed
ws ::= | " " | "\n" [ \t]{0,20}
```

```js
import { initLlama } from 'llama.rn'

const gbnf = '...'

const context = await initLlama({
  // ...params
  grammar: gbnf,
})

const { text } = await context.completion({
  // ...params
  messages: [
    {
      role: 'system',
      content: 'You are a helpful assistant that can answer questions and help with tasks.',
    },
    {
      role: 'user',
      content: 'Test',
    },
  ],
})
console.log('Result:', text)
```

Also, this is how `json_schema` works in `response_format` during completion, it converts the json_schema to gbnf grammar.

## Session (State)

The session file is a binary file that contains the state of the context, it can saves time of prompt processing.

```js
const context = await initLlama({ ...params })

// After prompt processing or completion ...

// Save the session
await context.saveSession('<path to save session>')

// Load the session
await context.loadSession('<path to load session>')
```

### Notes

- \* Session is currently not supported save state from multimodal context, so it only stores the text chunk before the first media chunk.

## Embedding

The embedding API is used to get the embedding of a text.

```js
const context = await initLlama({
  ...params,
  embedding: true,
})

const { embedding } = await context.embedding('Hello, world!')
```

- You can use model like [nomic-ai/nomic-embed-text-v1.5-GGUF](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF) for better embedding quality.
- You can use DB like [op-sqlite](https://github.com/OP-Engineering/op-sqlite) with sqlite-vec support to store and search embeddings.

## Rerank

The rerank API is used to rank documents based on their relevance to a query. This is particularly useful for improving search results and implementing retrieval-augmented generation (RAG) systems.

```js
const context = await initLlama({
  ...params,
  embedding: true, // Required for reranking
  pooling_type: 'rank', // Use rank pooling for rerank models
})

// Rerank documents based on relevance to query
const results = await context.rerank(
  'What is artificial intelligence?', // query
  [
    'AI is a branch of computer science.',
    'The weather is nice today.',
    'Machine learning is a subset of AI.',
    'I like pizza.',
  ], // documents to rank
  {
    normalize: 1, // Optional: normalize scores (default: from model config)
  }
)

// Results are automatically sorted by score (highest first)
results.forEach((result, index) => {
  console.log(`Rank ${index + 1}:`, {
    score: result.score,
    document: result.document,
    originalIndex: result.index,
  })
})
```

### Notes

- **Model Requirements**: Reranking requires models with `RANK` pooling type (e.g., reranker models)
- **Embedding Enabled**: The context must have `embedding: true` to use rerank functionality
- **Automatic Sorting**: Results are returned sorted by relevance score in descending order
- **Document Access**: Each result includes the original document text and its index in the input array
- **Score Interpretation**: Higher scores indicate higher relevance to the query

### Recommended Models

- [jinaai - jina-reranker-v2-base-multilingual-GGUF](https://huggingface.co/gpustack/jina-reranker-v2-base-multilingual-GGUF)
- [BAAI - bge-reranker-v2-m3-GGUF](https://huggingface.co/gpustack/bge-reranker-v2-m3-GGUF)
- Other models with "rerank" or "reranker" in their name and GGUF format

## Mock `llama.rn`

We have provided a mock version of `llama.rn` for testing purpose you can use on Jest:

```js
jest.mock('llama.rn', () => require('llama.rn/jest/mock'))
```

## NOTE

iOS:

- The [Extended Virtual Addressing](https://developer.apple.com/documentation/bundleresources/entitlements/com_apple_developer_kernel_extended-virtual-addressing) and [Increased Memory Limit](https://developer.apple.com/documentation/bundleresources/entitlements/com.apple.developer.kernel.increased-memory-limit?language=objc) capabilities are recommended to enable on iOS project.
- Metal:
  - We have tested to know some devices is not able to use Metal (GPU) due to llama.cpp used SIMD-scoped operation, you can check if your device is supported in [Metal feature set tables](https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf), Apple7 GPU will be the minimum requirement.
  - It's also not supported in iOS simulator due to [this limitation](https://developer.apple.com/documentation/metal/developing_metal_apps_that_run_in_simulator#3241609), we used constant buffers more than 14.

Android:

- Currently only supported arm64-v8a / x86_64 platform, this means you can't initialize a context on another platforms. The 64-bit platform are recommended because it can allocate more memory for the model.
- No integrated any GPU backend yet.

## Contributing

See the [contributing guide](CONTRIBUTING.md) to learn how to contribute to the repository and the development workflow.

## Apps using `llama.rn`

- [BRICKS](https://bricks.tools): Our product for building interactive signage in simple way. We provide LLM functions as Generator LLM/Assistant.
- [ChatterUI](https://github.com/Vali-98/ChatterUI): Simple frontend for LLMs built in react-native.
- [PocketPal AI](https://github.com/a-ghorbani/pocketpal-ai): An app that brings language models directly to your phone.

## Node.js binding

- [llama.node](https://github.com/mybigday/llama.node): An another Node.js binding of `llama.cpp` but made API same as `llama.rn`.

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
