[llama.rn](../README.md) / TTSCompletionParams

# Interface: TTSCompletionParams

The `completion()` parameters a text-to-speech request's format needs, as
`getFormattedAudioCompletion` reports them. Sampling (the backbone GGUF's
`general.sampling.*` unless set) and `n_predict` are the caller's.

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

[tts.ts:105](https://github.com/mybigday/llama.rn/blob/e7ab17bd/src/tts.ts#L105)

___

### grammar

• `Optional` **grammar**: `string`

The tokens the model may emit and its minimum length.

#### Defined in

[tts.ts:107](https://github.com/mybigday/llama.rn/blob/e7ab17bd/src/tts.ts#L107)

___

### logit\_bias

• `Optional` **logit\_bias**: `Float32Array`

One bias per vocabulary token (`-Infinity` for tokens the model never samples), when the model restricts tokens without a grammar.

#### Defined in

[tts.ts:109](https://github.com/mybigday/llama.rn/blob/e7ab17bd/src/tts.ts#L109)

___

### prompt

• **prompt**: `string`

The armed text; `completion()` must be given this prompt.

#### Defined in

[tts.ts:103](https://github.com/mybigday/llama.rn/blob/e7ab17bd/src/tts.ts#L103)
