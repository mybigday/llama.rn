[llama.rn](../README.md) / TTSCompletionParams

# Interface: TTSCompletionParams

The `completion()` parameters a text-to-speech request's format needs, as
`getFormattedAudioCompletion` reports them. Sampling (the backbone GGUF's
`general.sampling.*` unless set), `n_predict` and `n_min_predict` are the
caller's.

## Table of contents

### Properties

- [embedding](TTSCompletionParams.md#embedding)
- [grammar](TTSCompletionParams.md#grammar)
- [logit\_bias](TTSCompletionParams.md#logit_bias)
- [prompt](TTSCompletionParams.md#prompt)

## Properties

### embedding

• **embedding**: `boolean`

The model reads the backbone's hidden states (the context returns them).

#### Defined in

[tts.ts:88](https://github.com/mybigday/llama.rn/blob/a5944414/src/tts.ts#L88)

___

### grammar

• `Optional` **grammar**: `string`

The tokens the model may emit and its minimum length.

#### Defined in

[tts.ts:90](https://github.com/mybigday/llama.rn/blob/a5944414/src/tts.ts#L90)

___

### logit\_bias

• `Optional` **logit\_bias**: `Float32Array`

One bias per vocabulary token (`-Infinity` for tokens the model never samples), when the model restricts tokens without a grammar.

#### Defined in

[tts.ts:92](https://github.com/mybigday/llama.rn/blob/a5944414/src/tts.ts#L92)

___

### prompt

• **prompt**: `string`

The armed text; `completion()` must be given this prompt.

#### Defined in

[tts.ts:86](https://github.com/mybigday/llama.rn/blob/a5944414/src/tts.ts#L86)
