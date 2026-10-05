[llama.rn](../README.md) / LlamaContext

# Class: LlamaContext

## Table of contents

### Constructors

- [constructor](LlamaContext.md#constructor)

### Properties

- [androidLib](LlamaContext.md#androidlib)
- [devices](LlamaContext.md#devices)
- [gpu](LlamaContext.md#gpu)
- [id](LlamaContext.md#id)
- [model](LlamaContext.md#model)
- [parallel](LlamaContext.md#parallel)
- [reasonNoGPU](LlamaContext.md#reasonnogpu)
- [systemInfo](LlamaContext.md#systeminfo)

### Methods

- [applyLoraAdapters](LlamaContext.md#applyloraadapters)
- [bench](LlamaContext.md#bench)
- [clearCache](LlamaContext.md#clearcache)
- [completion](LlamaContext.md#completion)
- [createSpeaker](LlamaContext.md#createspeaker)
- [decide](LlamaContext.md#decide)
- [decodeAudioEmbeddings](LlamaContext.md#decodeaudioembeddings)
- [decodeAudioTokens](LlamaContext.md#decodeaudiotokens)
- [detokenize](LlamaContext.md#detokenize)
- [embedding](LlamaContext.md#embedding)
- [generateAudioCodes](LlamaContext.md#generateaudiocodes)
- [getAudioSampleRate](LlamaContext.md#getaudiosamplerate)
- [getFormattedAudioCompletion](LlamaContext.md#getformattedaudiocompletion)
- [getFormattedChat](LlamaContext.md#getformattedchat)
- [getLoadedLoraAdapters](LlamaContext.md#getloadedloraadapters)
- [getMultimodalSupport](LlamaContext.md#getmultimodalsupport)
- [getTTSCapabilities](LlamaContext.md#getttscapabilities)
- [initMultimodal](LlamaContext.md#initmultimodal)
- [initVocoder](LlamaContext.md#initvocoder)
- [isJinjaSupported](LlamaContext.md#isjinjasupported)
- [isLlamaChatSupported](LlamaContext.md#isllamachatsupported)
- [isMultimodalEnabled](LlamaContext.md#ismultimodalenabled)
- [isVocoderEnabled](LlamaContext.md#isvocoderenabled)
- [loadSession](LlamaContext.md#loadsession)
- [release](LlamaContext.md#release)
- [releaseMultimodal](LlamaContext.md#releasemultimodal)
- [releaseVocoder](LlamaContext.md#releasevocoder)
- [removeLoraAdapters](LlamaContext.md#removeloraadapters)
- [rerank](LlamaContext.md#rerank)
- [saveSession](LlamaContext.md#savesession)
- [stopCompletion](LlamaContext.md#stopcompletion)
- [tokenize](LlamaContext.md#tokenize)

## Constructors

### constructor

• **new LlamaContext**(`«destructured»`)

#### Parameters

| Name | Type |
| :------ | :------ |
| `«destructured»` | [`NativeLlamaContext`](../README.md#nativellamacontext) |

#### Defined in

[index.ts:754](https://github.com/mybigday/llama.rn/blob/3ee457d5/src/index.ts#L754)

## Properties

### androidLib

• **androidLib**: `undefined` \| `string`

#### Defined in

[index.ts:443](https://github.com/mybigday/llama.rn/blob/3ee457d5/src/index.ts#L443)

___

### devices

• **devices**: `undefined` \| `string`[]

#### Defined in

[index.ts:439](https://github.com/mybigday/llama.rn/blob/3ee457d5/src/index.ts#L439)

___

### gpu

• **gpu**: `boolean` = `false`

#### Defined in

[index.ts:435](https://github.com/mybigday/llama.rn/blob/3ee457d5/src/index.ts#L435)

___

### id

• **id**: `number`

#### Defined in

[index.ts:433](https://github.com/mybigday/llama.rn/blob/3ee457d5/src/index.ts#L433)

___

### model

• **model**: `Object`

#### Type declaration

| Name | Type | Description |
| :------ | :------ | :------ |
| `chatTemplates` | { `jinja`: { `default`: `boolean` ; `defaultCaps`: { `parallelToolCalls`: `boolean` ; `systemRole`: `boolean` ; `toolCalls`: `boolean` ; `tools`: `boolean`  } ; `toolUse`: `boolean` ; `toolUseCaps?`: { `parallelToolCalls`: `boolean` ; `systemRole`: `boolean` ; `toolCalls`: `boolean` ; `tools`: `boolean`  }  } ; `llamaChat`: `boolean`  } | - |
| `chatTemplates.jinja` | { `default`: `boolean` ; `defaultCaps`: { `parallelToolCalls`: `boolean` ; `systemRole`: `boolean` ; `toolCalls`: `boolean` ; `tools`: `boolean`  } ; `toolUse`: `boolean` ; `toolUseCaps?`: { `parallelToolCalls`: `boolean` ; `systemRole`: `boolean` ; `toolCalls`: `boolean` ; `tools`: `boolean`  }  } | - |
| `chatTemplates.jinja.default` | `boolean` | - |
| `chatTemplates.jinja.defaultCaps` | { `parallelToolCalls`: `boolean` ; `systemRole`: `boolean` ; `toolCalls`: `boolean` ; `tools`: `boolean`  } | - |
| `chatTemplates.jinja.defaultCaps.parallelToolCalls` | `boolean` | - |
| `chatTemplates.jinja.defaultCaps.systemRole` | `boolean` | - |
| `chatTemplates.jinja.defaultCaps.toolCalls` | `boolean` | - |
| `chatTemplates.jinja.defaultCaps.tools` | `boolean` | - |
| `chatTemplates.jinja.toolUse` | `boolean` | - |
| `chatTemplates.jinja.toolUseCaps?` | { `parallelToolCalls`: `boolean` ; `systemRole`: `boolean` ; `toolCalls`: `boolean` ; `tools`: `boolean`  } | - |
| `chatTemplates.jinja.toolUseCaps.parallelToolCalls` | `boolean` | - |
| `chatTemplates.jinja.toolUseCaps.systemRole` | `boolean` | - |
| `chatTemplates.jinja.toolUseCaps.toolCalls` | `boolean` | - |
| `chatTemplates.jinja.toolUseCaps.tools` | `boolean` | - |
| `chatTemplates.llamaChat` | `boolean` | - |
| `decision?` | { `error?`: `string` ; `imageInput`: `boolean` ; `nOptionsMax`: `number` ; `readout?`: [`DecisionReadout`](../README.md#decisionreadout) ; `textGeneration`: `boolean` ; `type`: [`DecisionModelType`](../README.md#decisionmodeltype)  } | Set if the model is a typed decision model, see `LlamaContext.decide()` |
| `decision.error?` | `string` | Why the model cannot be used, when `type` is `unknown` |
| `decision.imageInput` | `boolean` | The prompt has a place for images (multimodal still has to be initialized) |
| `decision.nOptionsMax` | `number` | Most options a `choice` question can have |
| `decision.readout?` | [`DecisionReadout`](../README.md#decisionreadout) | Only for `system_one` models |
| `decision.textGeneration` | `boolean` | If false, `completion()` rejects: the model only answers decisions (clef, for one) |
| `decision.type` | [`DecisionModelType`](../README.md#decisionmodeltype) | - |
| `desc` | `string` | - |
| `isChatTemplateSupported` | `boolean` | - |
| `is_hybrid` | `boolean` | - |
| `is_recurrent` | `boolean` | - |
| `metadata` | `Object` | - |
| `nEmbd` | `number` | - |
| `nParams` | `number` | - |
| `size` | `number` | - |

#### Defined in

[index.ts:441](https://github.com/mybigday/llama.rn/blob/3ee457d5/src/index.ts#L441)

___

### parallel

• **parallel**: `Object`

Parallel processing namespace for non-blocking queue operations

#### Type declaration

| Name | Type |
| :------ | :------ |
| `completion` | (`params`: [`ParallelCompletionParams`](../README.md#parallelcompletionparams), `onToken?`: (`requestId`: `number`, `data`: [`TokenData`](../README.md#tokendata)) => `void`) => `Promise`<{ `promise`: `Promise`<[`NativeCompletionResult`](../README.md#nativecompletionresult)\> ; `requestId`: `number` ; `stop`: () => `Promise`<`void`\>  }\> |
| `configure` | (`config`: { `n_batch?`: `number` ; `n_parallel?`: `number`  }) => `Promise`<`boolean`\> |
| `decide` | <Q\>(`request`: [`DecisionRequest`](../README.md#decisionrequest)<`Q`\>) => `Promise`<{ `promise`: `Promise`<[`DecisionResult`](../README.md#decisionresult)<`Q`\>\> ; `requestId`: `number`  }\> |
| `disable` | () => `Promise`<`boolean`\> |
| `embedding` | (`text`: `string`, `params?`: [`NativeEmbeddingParams`](../README.md#nativeembeddingparams)) => `Promise`<{ `promise`: `Promise`<[`NativeEmbeddingResult`](../README.md#nativeembeddingresult)\> ; `requestId`: `number`  }\> |
| `enable` | (`config?`: { `n_batch?`: `number` ; `n_parallel?`: `number`  }) => `Promise`<`boolean`\> |
| `getStatus` | () => `Promise`<[`ParallelStatus`](../README.md#parallelstatus)\> |
| `rerank` | (`query`: `string`, `documents`: `string`[], `params?`: [`RerankParams`](../README.md#rerankparams)) => `Promise`<{ `promise`: `Promise`<[`RerankResult`](../README.md#rerankresult)[]\> ; `requestId`: `number`  }\> |
| `subscribeToStatus` | (`callback`: (`status`: [`ParallelStatus`](../README.md#parallelstatus)) => `void`) => `Promise`<{ `remove`: () => `void`  }\> |

#### Defined in

[index.ts:450](https://github.com/mybigday/llama.rn/blob/3ee457d5/src/index.ts#L450)

___

### reasonNoGPU

• **reasonNoGPU**: `string` = `''`

#### Defined in

[index.ts:437](https://github.com/mybigday/llama.rn/blob/3ee457d5/src/index.ts#L437)

___

### systemInfo

• **systemInfo**: `string`

#### Defined in

[index.ts:445](https://github.com/mybigday/llama.rn/blob/3ee457d5/src/index.ts#L445)

## Methods

### applyLoraAdapters

▸ **applyLoraAdapters**(`loraList`): `Promise`<`void`\>

#### Parameters

| Name | Type |
| :------ | :------ |
| `loraList` | { `path`: `string` ; `scaled?`: `number`  }[] |

#### Returns

`Promise`<`void`\>

#### Defined in

[index.ts:1102](https://github.com/mybigday/llama.rn/blob/3ee457d5/src/index.ts#L1102)

___

### bench

▸ **bench**(`pp`, `tg`, `pl`, `nr`): `Promise`<[`BenchResult`](../README.md#benchresult)\>

#### Parameters

| Name | Type |
| :------ | :------ |
| `pp` | `number` |
| `tg` | `number` |
| `pl` | `number` |
| `nr` | `number` |

#### Returns

`Promise`<[`BenchResult`](../README.md#benchresult)\>

#### Defined in

[index.ts:1072](https://github.com/mybigday/llama.rn/blob/3ee457d5/src/index.ts#L1072)

___

### clearCache

▸ **clearCache**(`clearData?`): `Promise`<`void`\>

Clear the KV cache and reset conversation state

#### Parameters

| Name | Type | Default value | Description |
| :------ | :------ | :------ | :------ |
| `clearData` | `boolean` | `false` | If true, clears both metadata and tensor data buffers (slower). If false, only clears metadata (faster). |

#### Returns

`Promise`<`void`\>

Promise that resolves when cache is cleared

Call this method between different conversations to prevent cache contamination.
Without clearing, the model may use cached context from previous conversations,
leading to incorrect or unexpected responses.

For hybrid architecture models (e.g., LFM2), this is essential as they
use recurrent state that cannot be partially removed - only fully cleared.

#### Defined in

[index.ts:1421](https://github.com/mybigday/llama.rn/blob/3ee457d5/src/index.ts#L1421)

___

### completion

▸ **completion**(`params`, `callback?`): `Promise`<[`NativeCompletionResult`](../README.md#nativecompletionresult)\>

#### Parameters

| Name | Type |
| :------ | :------ |
| `params` | `Omit`<[`NativeCompletionParams`](../README.md#nativecompletionparams), ``"emit_partial_completion"`` \| ``"prompt"``\> & [`CompletionBaseParams`](../README.md#completionbaseparams) & { `speaker?`: [`LlamaSpeaker`](LlamaSpeaker.md)  } |
| `callback?` | (`data`: [`TokenData`](../README.md#tokendata)) => `void` |

#### Returns

`Promise`<[`NativeCompletionResult`](../README.md#nativecompletionresult)\>

#### Defined in

[index.ts:910](https://github.com/mybigday/llama.rn/blob/3ee457d5/src/index.ts#L910)

___

### createSpeaker

▸ **createSpeaker**(`config`): `Promise`<[`LlamaSpeaker`](LlamaSpeaker.md)\>

#### Parameters

| Name | Type |
| :------ | :------ |
| `config` | `Object` |
| `config.bake?` | `boolean` |
| `config.emotion?` | `number` |
| `config.refAudio` | `number`[] \| `Float32Array` |
| `config.refAudioSampleRate` | `number` |
| `config.refText?` | `string` |

#### Returns

`Promise`<[`LlamaSpeaker`](LlamaSpeaker.md)\>

#### Defined in

[index.ts:1366](https://github.com/mybigday/llama.rn/blob/3ee457d5/src/index.ts#L1366)

___

### decide

▸ **decide**<`Q`\>(`request`): `Promise`<[`DecisionResult`](../README.md#decisionresult)<`Q`\>\>

Answer typed questions about a state with a decision model
(see `model.decision`), in the shape of the TypeSafe `/v1/systemone` API.
Each answer is read from one forward pass, no token is generated.
It runs on the sequence of `completion()`, whose cached prompt is
dropped: the next completion evaluates its prompt from the start.

Rejects if the request is invalid, if the model is not a decision
model of a supported type, or if parallel mode is enabled (use
`parallel.decide()` then).

#### Type parameters

| Name | Type |
| :------ | :------ |
| `Q` | extends [`DecisionQuestions`](../README.md#decisionquestions) |

#### Parameters

| Name | Type |
| :------ | :------ |
| `request` | [`DecisionRequest`](../README.md#decisionrequest)<`Q`\> |

#### Returns

`Promise`<[`DecisionResult`](../README.md#decisionresult)<`Q`\>\>

#### Defined in

[index.ts:1065](https://github.com/mybigday/llama.rn/blob/3ee457d5/src/index.ts#L1065)

___

### decodeAudioEmbeddings

▸ **decodeAudioEmbeddings**(`embeddings`, `embeddingDim`): `Promise`<`number`[]\>

#### Parameters

| Name | Type |
| :------ | :------ |
| `embeddings` | `number`[] \| `Float32Array` |
| `embeddingDim` | `number` |

#### Returns

`Promise`<`number`[]\>

#### Defined in

[index.ts:1384](https://github.com/mybigday/llama.rn/blob/3ee457d5/src/index.ts#L1384)

___

### decodeAudioTokens

▸ **decodeAudioTokens**(`tokens`): `Promise`<`number`[]\>

#### Parameters

| Name | Type |
| :------ | :------ |
| `tokens` | `number`[] \| `Int32Array` |

#### Returns

`Promise`<`number`[]\>

#### Defined in

[index.ts:1313](https://github.com/mybigday/llama.rn/blob/3ee457d5/src/index.ts#L1313)

___

### detokenize

▸ **detokenize**(`tokens`): `Promise`<`string`\>

#### Parameters

| Name | Type |
| :------ | :------ |
| `tokens` | `number`[] |

#### Returns

`Promise`<`string`\>

#### Defined in

[index.ts:1022](https://github.com/mybigday/llama.rn/blob/3ee457d5/src/index.ts#L1022)

___

### embedding

▸ **embedding**(`text`, `params?`): `Promise`<[`NativeEmbeddingResult`](../README.md#nativeembeddingresult)\>

#### Parameters

| Name | Type |
| :------ | :------ |
| `text` | `string` |
| `params?` | [`NativeEmbeddingParams`](../README.md#nativeembeddingparams) |

#### Returns

`Promise`<[`NativeEmbeddingResult`](../README.md#nativeembeddingresult)\>

#### Defined in

[index.ts:1027](https://github.com/mybigday/llama.rn/blob/3ee457d5/src/index.ts#L1027)

___

### generateAudioCodes

▸ **generateAudioCodes**(`options`): `Promise`<{ `aborted`: `boolean` ; `codes`: `number`[] ; `nCodebook`: `number` ; `nFrames`: `number` ; `stoppedOnEos`: `boolean`  }\>

DEPRECATED: source-compat wrapper for codec_lm-AR TTS.

As of the "one completion API" refactor, codec_lm-AR models (CSM /
Qwen3-TTS / MOSS-TTSD / MOSS-TTS-Realtime / Chatterbox) run through
the standard `completion` loop with `flow = 'tokens'` and
`embedding = true`.  The per-step codec_lm state machine that used
to live inside this call is now a hook on the completion loop
(`tryCodecLmAudioStep`); the codes get appended to
`result.audio_tokens` the same way OuteTTS / Soprano / NeuTTS do.

This method still works — internally it just primes params +
speaker prefix, runs `completion`, and drains `audio_tokens` — but
new callers should skip it and use `completion()` +
`decodeAudioTokens` directly.

`onFrame` (optional) fires after each AR step with that frame's
codes for streaming UIs. It is fire-and-forget — its return value
isn't read.

#### Parameters

| Name | Type |
| :------ | :------ |
| `options` | `Object` |
| `options.maxFrames?` | `number` |
| `options.onFrame?` | (`step`: `number`, `codes`: `number`[]) => `void` |
| `options.prompt` | `string` |
| `options.seed?` | `number` |
| `options.temperature?` | `number` |
| `options.topK?` | `number` |
| `options.topP?` | `number` |

#### Returns

`Promise`<{ `aborted`: `boolean` ; `codes`: `number`[] ; `nCodebook`: `number` ; `nFrames`: `number` ; `stoppedOnEos`: `boolean`  }\>

#### Defined in

[index.ts:1346](https://github.com/mybigday/llama.rn/blob/3ee457d5/src/index.ts#L1346)

___

### getAudioSampleRate

▸ **getAudioSampleRate**(): `Promise`<`number`\>

#### Returns

`Promise`<`number`\>

#### Defined in

[index.ts:1399](https://github.com/mybigday/llama.rn/blob/3ee457d5/src/index.ts#L1399)

___

### getFormattedAudioCompletion

▸ **getFormattedAudioCompletion**(`options`): `Promise`<{ `embedding`: `boolean` ; `flow`: ``""`` \| ``"tokens"`` \| ``"continuous_embd"`` ; `grammar?`: `string` ; `prompt`: `string`  }\>

Build a formatted prompt for the loaded TTS model.

Breaking change: takes an options object — the previous `(speaker, text)`
positional signature has been removed.

- `prompt` — text to speak. Phonemized if `phonemizer` is supplied.
- `speaker` — built-in voice name (string), a structured speaker object
  (shape depends on the model family — see `OuteTTSSpeaker` /
  `NeuTTSSpeaker`), or `undefined` to fall back to the family default.
- `phonemizer` — optional `(text, language) => string | Promise<string>`.
  When set, `prompt` and `speaker.ref_text` (if missing `ref_phones`) go
  through it. Models that need phonemes (NeuTTS) get off-distribution
  text otherwise — caller's call.
- `language` — phonemizer hook hint; defaults to capabilities.defaultLanguage.

#### Parameters

| Name | Type |
| :------ | :------ |
| `options` | `Object` |
| `options.language?` | `string` |
| `options.phonemizer?` | (`text`: `string`, `language`: `string`) => `string` \| `Promise`<`string`\> |
| `options.prompt` | `string` |
| `options.speaker?` | `string` \| [`SpeakerPayload`](../README.md#speakerpayload) \| [`LlamaSpeaker`](LlamaSpeaker.md) |

#### Returns

`Promise`<{ `embedding`: `boolean` ; `flow`: ``""`` \| ``"tokens"`` \| ``"continuous_embd"`` ; `grammar?`: `string` ; `prompt`: `string`  }\>

#### Defined in

[index.ts:1225](https://github.com/mybigday/llama.rn/blob/3ee457d5/src/index.ts#L1225)

___

### getFormattedChat

▸ **getFormattedChat**(`messages`, `template?`, `params?`): `Promise`<[`FormattedChatResult`](../README.md#formattedchatresult) \| [`JinjaFormattedChatResult`](../README.md#jinjaformattedchatresult)\>

#### Parameters

| Name | Type |
| :------ | :------ |
| `messages` | [`RNLlamaOAICompatibleMessage`](../README.md#rnllamaoaicompatiblemessage)[] |
| `template?` | ``null`` \| `string` |
| `params?` | `Object` |
| `params.add_generation_prompt?` | `boolean` |
| `params.chat_template_kwargs?` | [`ChatTemplateKwargs`](../README.md#chattemplatekwargs) |
| `params.enable_thinking?` | `boolean` |
| `params.force_pure_content?` | `boolean` |
| `params.jinja?` | `boolean` |
| `params.now?` | `string` \| `number` |
| `params.parallel_tool_calls?` | `boolean` |
| `params.reasoning_format?` | ``"none"`` \| ``"auto"`` \| ``"deepseek"`` |
| `params.response_format?` | [`CompletionResponseFormat`](../README.md#completionresponseformat) |
| `params.tool_choice?` | `string` |
| `params.tools?` | `object` |

#### Returns

`Promise`<[`FormattedChatResult`](../README.md#formattedchatresult) \| [`JinjaFormattedChatResult`](../README.md#jinjaformattedchatresult)\>

#### Defined in

[index.ts:798](https://github.com/mybigday/llama.rn/blob/3ee457d5/src/index.ts#L798)

___

### getLoadedLoraAdapters

▸ **getLoadedLoraAdapters**(): `Promise`<{ `path`: `string` ; `scaled?`: `number`  }[]\>

#### Returns

`Promise`<{ `path`: `string` ; `scaled?`: `number`  }[]\>

#### Defined in

[index.ts:1115](https://github.com/mybigday/llama.rn/blob/3ee457d5/src/index.ts#L1115)

___

### getMultimodalSupport

▸ **getMultimodalSupport**(): `Promise`<{ `audio`: `boolean` ; `vision`: `boolean`  }\>

#### Returns

`Promise`<{ `audio`: `boolean` ; `vision`: `boolean`  }\>

#### Defined in

[index.ts:1157](https://github.com/mybigday/llama.rn/blob/3ee457d5/src/index.ts#L1157)

___

### getTTSCapabilities

▸ **getTTSCapabilities**(): `Promise`<[`TTSCapabilities`](../interfaces/TTSCapabilities.md)\>

#### Returns

`Promise`<[`TTSCapabilities`](../interfaces/TTSCapabilities.md)\>

#### Defined in

[index.ts:1204](https://github.com/mybigday/llama.rn/blob/3ee457d5/src/index.ts#L1204)

___

### initMultimodal

▸ **initMultimodal**(`«destructured»`): `Promise`<`boolean`\>

Initialize multimodal support (vision/audio) with a projector model.

#### Parameters

| Name | Type |
| :------ | :------ |
| `«destructured»` | `Object` |
| › `image_max_tokens?` | `number` |
| › `image_min_tokens?` | `number` |
| › `path` | `string` |
| › `use_gpu?` | `boolean` |

#### Returns

`Promise`<`boolean`\>

#### Defined in

[index.ts:1131](https://github.com/mybigday/llama.rn/blob/3ee457d5/src/index.ts#L1131)

___

### initVocoder

▸ **initVocoder**(`«destructured»`): `Promise`<`boolean`\>

Attach a codec / vocoder GGUF to this context, enabling the TTS API.

**Experimental:** the TTS API may change without a major version bump, and
output quality varies by model family and backend. See the "Tested models"
table in the README.

#### Parameters

| Name | Type |
| :------ | :------ |
| `«destructured»` | `Object` |
| › `n_batch?` | `number` |
| › `path` | `string` |
| › `use_gpu?` | `boolean` |

#### Returns

`Promise`<`boolean`\>

#### Defined in

[index.ts:1177](https://github.com/mybigday/llama.rn/blob/3ee457d5/src/index.ts#L1177)

___

### isJinjaSupported

▸ **isJinjaSupported**(): `boolean`

#### Returns

`boolean`

#### Defined in

[index.ts:793](https://github.com/mybigday/llama.rn/blob/3ee457d5/src/index.ts#L793)

___

### isLlamaChatSupported

▸ **isLlamaChatSupported**(): `boolean`

#### Returns

`boolean`

#### Defined in

[index.ts:789](https://github.com/mybigday/llama.rn/blob/3ee457d5/src/index.ts#L789)

___

### isMultimodalEnabled

▸ **isMultimodalEnabled**(): `Promise`<`boolean`\>

#### Returns

`Promise`<`boolean`\>

#### Defined in

[index.ts:1152](https://github.com/mybigday/llama.rn/blob/3ee457d5/src/index.ts#L1152)

___

### isVocoderEnabled

▸ **isVocoderEnabled**(): `Promise`<`boolean`\>

#### Returns

`Promise`<`boolean`\>

#### Defined in

[index.ts:1199](https://github.com/mybigday/llama.rn/blob/3ee457d5/src/index.ts#L1199)

___

### loadSession

▸ **loadSession**(`filepath`): `Promise`<[`NativeSessionLoadResult`](../README.md#nativesessionloadresult)\>

#### Parameters

| Name | Type |
| :------ | :------ |
| `filepath` | `string` |

#### Returns

`Promise`<[`NativeSessionLoadResult`](../README.md#nativesessionloadresult)\>

#### Defined in

[index.ts:772](https://github.com/mybigday/llama.rn/blob/3ee457d5/src/index.ts#L772)

___

### release

▸ **release**(): `Promise`<`void`\>

#### Returns

`Promise`<`void`\>

#### Defined in

[index.ts:1426](https://github.com/mybigday/llama.rn/blob/3ee457d5/src/index.ts#L1426)

___

### releaseMultimodal

▸ **releaseMultimodal**(): `Promise`<`void`\>

#### Returns

`Promise`<`void`\>

#### Defined in

[index.ts:1165](https://github.com/mybigday/llama.rn/blob/3ee457d5/src/index.ts#L1165)

___

### releaseVocoder

▸ **releaseVocoder**(): `Promise`<`void`\>

#### Returns

`Promise`<`void`\>

#### Defined in

[index.ts:1404](https://github.com/mybigday/llama.rn/blob/3ee457d5/src/index.ts#L1404)

___

### removeLoraAdapters

▸ **removeLoraAdapters**(): `Promise`<`void`\>

#### Returns

`Promise`<`void`\>

#### Defined in

[index.ts:1110](https://github.com/mybigday/llama.rn/blob/3ee457d5/src/index.ts#L1110)

___

### rerank

▸ **rerank**(`query`, `documents`, `params?`): `Promise`<[`RerankResult`](../README.md#rerankresult)[]\>

#### Parameters

| Name | Type |
| :------ | :------ |
| `query` | `string` |
| `documents` | `string`[] |
| `params?` | [`RerankParams`](../README.md#rerankparams) |

#### Returns

`Promise`<[`RerankResult`](../README.md#rerankresult)[]\>

#### Defined in

[index.ts:1038](https://github.com/mybigday/llama.rn/blob/3ee457d5/src/index.ts#L1038)

___

### saveSession

▸ **saveSession**(`filepath`, `options?`): `Promise`<`number`\>

#### Parameters

| Name | Type |
| :------ | :------ |
| `filepath` | `string` |
| `options?` | `Object` |
| `options.tokenSize` | `number` |

#### Returns

`Promise`<`number`\>

#### Defined in

[index.ts:779](https://github.com/mybigday/llama.rn/blob/3ee457d5/src/index.ts#L779)

___

### stopCompletion

▸ **stopCompletion**(): `Promise`<`void`\>

#### Returns

`Promise`<`void`\>

#### Defined in

[index.ts:1005](https://github.com/mybigday/llama.rn/blob/3ee457d5/src/index.ts#L1005)

___

### tokenize

▸ **tokenize**(`text`, `«destructured»?`): `Promise`<[`NativeTokenizeResult`](../README.md#nativetokenizeresult)\>

#### Parameters

| Name | Type |
| :------ | :------ |
| `text` | `string` |
| `«destructured»` | `Object` |
| › `media_paths?` | `string`[] |

#### Returns

`Promise`<[`NativeTokenizeResult`](../README.md#nativetokenizeresult)\>

#### Defined in

[index.ts:1010](https://github.com/mybigday/llama.rn/blob/3ee457d5/src/index.ts#L1010)
