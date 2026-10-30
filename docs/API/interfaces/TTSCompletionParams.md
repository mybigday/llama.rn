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
- [n\_min\_predict](TTSCompletionParams.md#n_min_predict)
- [prompt](TTSCompletionParams.md#prompt)

## Properties

### embedding

• **embedding**: `boolean`

The model reads the backbone's hidden states (the context returns them).

#### Defined in

[tts.ts:87](https://github.com/mybigday/llama.rn/blob/5c3e8141/src/tts.ts#L87)

___

### grammar

• `Optional` **grammar**: `string`

The tokens the model may emit and its minimum length.

#### Defined in

[tts.ts:89](https://github.com/mybigday/llama.rn/blob/5c3e8141/src/tts.ts#L89)

___

### logit\_bias

• `Optional` **logit\_bias**: `Float32Array`

One bias per vocabulary token (`-Infinity` for tokens the model never samples), when the model restricts tokens without a grammar.

#### Defined in

[tts.ts:91](https://github.com/mybigday/llama.rn/blob/5c3e8141/src/tts.ts#L91)

___

### n\_min\_predict

• `Optional` **n\_min\_predict**: `number`

The model's shortest generation for this text, in steps (audio frames), so it does not end too early.
Models whose `grammar` holds the minimum (NeuTTS, Soprano) keep the grammar's when it is changed.

#### Defined in

[tts.ts:96](https://github.com/mybigday/llama.rn/blob/5c3e8141/src/tts.ts#L96)

___

### prompt

• **prompt**: `string`

The armed text; `completion()` must be given this prompt.

#### Defined in

[tts.ts:85](https://github.com/mybigday/llama.rn/blob/5c3e8141/src/tts.ts#L85)
