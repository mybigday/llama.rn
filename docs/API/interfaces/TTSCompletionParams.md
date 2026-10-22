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

[tts.ts:98](https://github.com/mybigday/llama.rn/blob/1035d0aa/src/tts.ts#L98)

___

### grammar

• `Optional` **grammar**: `string`

The tokens the model may emit and its minimum length.

#### Defined in

[tts.ts:100](https://github.com/mybigday/llama.rn/blob/1035d0aa/src/tts.ts#L100)

___

### logit\_bias

• `Optional` **logit\_bias**: `Float32Array`

One bias per vocabulary token (`-Infinity` for tokens the model never samples), when the model restricts tokens without a grammar.

#### Defined in

[tts.ts:102](https://github.com/mybigday/llama.rn/blob/1035d0aa/src/tts.ts#L102)

___

### prompt

• **prompt**: `string`

The armed text; `completion()` must be given this prompt.

#### Defined in

[tts.ts:96](https://github.com/mybigday/llama.rn/blob/1035d0aa/src/tts.ts#L96)
