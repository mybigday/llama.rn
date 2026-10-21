[llama.rn](../README.md) / TTSCompletionParams

# Interface: TTSCompletionParams

The `completion()` parameters a text-to-speech request runs with, as
`getFormattedAudioCompletion` reports them: the model's backbone sampling
and constraints. Pass them to `completion()` (and override any of them);
nothing is applied implicitly.

## Table of contents

### Properties

- [allowed\_tokens](TTSCompletionParams.md#allowed_tokens)
- [dry\_allowed\_length](TTSCompletionParams.md#dry_allowed_length)
- [dry\_base](TTSCompletionParams.md#dry_base)
- [dry\_multiplier](TTSCompletionParams.md#dry_multiplier)
- [dry\_penalty\_last\_n](TTSCompletionParams.md#dry_penalty_last_n)
- [embedding](TTSCompletionParams.md#embedding)
- [grammar](TTSCompletionParams.md#grammar)
- [min\_p](TTSCompletionParams.md#min_p)
- [n\_predict](TTSCompletionParams.md#n_predict)
- [penalize\_prompt](TTSCompletionParams.md#penalize_prompt)
- [penalty\_freq](TTSCompletionParams.md#penalty_freq)
- [penalty\_last\_n](TTSCompletionParams.md#penalty_last_n)
- [penalty\_present](TTSCompletionParams.md#penalty_present)
- [penalty\_repeat](TTSCompletionParams.md#penalty_repeat)
- [prompt](TTSCompletionParams.md#prompt)
- [samplers](TTSCompletionParams.md#samplers)
- [seed](TTSCompletionParams.md#seed)
- [temperature](TTSCompletionParams.md#temperature)
- [top\_k](TTSCompletionParams.md#top_k)
- [top\_p](TTSCompletionParams.md#top_p)

## Properties

### allowed\_tokens

• `Optional` **allowed\_tokens**: [`number`, `number`][]

#### Defined in

[tts.ts:143](https://github.com/mybigday/llama.rn/blob/4ae85a1e/src/tts.ts#L143)

___

### dry\_allowed\_length

• **dry\_allowed\_length**: `number`

#### Defined in

[tts.ts:138](https://github.com/mybigday/llama.rn/blob/4ae85a1e/src/tts.ts#L138)

___

### dry\_base

• **dry\_base**: `number`

#### Defined in

[tts.ts:137](https://github.com/mybigday/llama.rn/blob/4ae85a1e/src/tts.ts#L137)

___

### dry\_multiplier

• **dry\_multiplier**: `number`

#### Defined in

[tts.ts:136](https://github.com/mybigday/llama.rn/blob/4ae85a1e/src/tts.ts#L136)

___

### dry\_penalty\_last\_n

• **dry\_penalty\_last\_n**: `number`

#### Defined in

[tts.ts:139](https://github.com/mybigday/llama.rn/blob/4ae85a1e/src/tts.ts#L139)

___

### embedding

• **embedding**: `boolean`

The model reads the backbone's hidden states (the context returns them).

#### Defined in

[tts.ts:123](https://github.com/mybigday/llama.rn/blob/4ae85a1e/src/tts.ts#L123)

___

### grammar

• `Optional` **grammar**: `string`

#### Defined in

[tts.ts:142](https://github.com/mybigday/llama.rn/blob/4ae85a1e/src/tts.ts#L142)

___

### min\_p

• **min\_p**: `number`

#### Defined in

[tts.ts:130](https://github.com/mybigday/llama.rn/blob/4ae85a1e/src/tts.ts#L130)

___

### n\_predict

• **n\_predict**: `number`

Step budget: the model's frame limit, or `-1`.

#### Defined in

[tts.ts:125](https://github.com/mybigday/llama.rn/blob/4ae85a1e/src/tts.ts#L125)

___

### penalize\_prompt

• **penalize\_prompt**: `boolean`

#### Defined in

[tts.ts:135](https://github.com/mybigday/llama.rn/blob/4ae85a1e/src/tts.ts#L135)

___

### penalty\_freq

• **penalty\_freq**: `number`

#### Defined in

[tts.ts:133](https://github.com/mybigday/llama.rn/blob/4ae85a1e/src/tts.ts#L133)

___

### penalty\_last\_n

• **penalty\_last\_n**: `number`

#### Defined in

[tts.ts:131](https://github.com/mybigday/llama.rn/blob/4ae85a1e/src/tts.ts#L131)

___

### penalty\_present

• **penalty\_present**: `number`

#### Defined in

[tts.ts:134](https://github.com/mybigday/llama.rn/blob/4ae85a1e/src/tts.ts#L134)

___

### penalty\_repeat

• **penalty\_repeat**: `number`

#### Defined in

[tts.ts:132](https://github.com/mybigday/llama.rn/blob/4ae85a1e/src/tts.ts#L132)

___

### prompt

• **prompt**: `string`

The armed text; `completion()` must be given this prompt.

#### Defined in

[tts.ts:121](https://github.com/mybigday/llama.rn/blob/4ae85a1e/src/tts.ts#L121)

___

### samplers

• **samplers**: `string`

Sampler order (llama.cpp `--samplers` names).

#### Defined in

[tts.ts:141](https://github.com/mybigday/llama.rn/blob/4ae85a1e/src/tts.ts#L141)

___

### seed

• **seed**: `number`

#### Defined in

[tts.ts:126](https://github.com/mybigday/llama.rn/blob/4ae85a1e/src/tts.ts#L126)

___

### temperature

• **temperature**: `number`

#### Defined in

[tts.ts:127](https://github.com/mybigday/llama.rn/blob/4ae85a1e/src/tts.ts#L127)

___

### top\_k

• **top\_k**: `number`

#### Defined in

[tts.ts:128](https://github.com/mybigday/llama.rn/blob/4ae85a1e/src/tts.ts#L128)

___

### top\_p

• **top\_p**: `number`

#### Defined in

[tts.ts:129](https://github.com/mybigday/llama.rn/blob/4ae85a1e/src/tts.ts#L129)
