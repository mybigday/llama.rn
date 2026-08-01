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

[index.ts:677](https://github.com/mybigday/llama.rn/blob/a8c8191/src/index.ts#L677)

## Properties

### androidLib

• **androidLib**: `undefined` \| `string`

#### Defined in

[index.ts:406](https://github.com/mybigday/llama.rn/blob/a8c8191/src/index.ts#L406)

___

### devices

• **devices**: `undefined` \| `string`[]

#### Defined in

[index.ts:402](https://github.com/mybigday/llama.rn/blob/a8c8191/src/index.ts#L402)

___

### gpu

• **gpu**: `boolean` = `false`

#### Defined in

[index.ts:398](https://github.com/mybigday/llama.rn/blob/a8c8191/src/index.ts#L398)

___

### id

• **id**: `number`

#### Defined in

[index.ts:396](https://github.com/mybigday/llama.rn/blob/a8c8191/src/index.ts#L396)

___

### model

• **model**: `Object`

#### Type declaration

| Name | Type |
| :------ | :------ |
| `chatTemplates` | { `jinja`: { `default`: `boolean` ; `defaultCaps`: { `parallelToolCalls`: `boolean` ; `systemRole`: `boolean` ; `toolCalls`: `boolean` ; `tools`: `boolean`  } ; `toolUse`: `boolean` ; `toolUseCaps?`: { `parallelToolCalls`: `boolean` ; `systemRole`: `boolean` ; `toolCalls`: `boolean` ; `tools`: `boolean`  }  } ; `llamaChat`: `boolean`  } |
| `chatTemplates.jinja` | { `default`: `boolean` ; `defaultCaps`: { `parallelToolCalls`: `boolean` ; `systemRole`: `boolean` ; `toolCalls`: `boolean` ; `tools`: `boolean`  } ; `toolUse`: `boolean` ; `toolUseCaps?`: { `parallelToolCalls`: `boolean` ; `systemRole`: `boolean` ; `toolCalls`: `boolean` ; `tools`: `boolean`  }  } |
| `chatTemplates.jinja.default` | `boolean` |
| `chatTemplates.jinja.defaultCaps` | { `parallelToolCalls`: `boolean` ; `systemRole`: `boolean` ; `toolCalls`: `boolean` ; `tools`: `boolean`  } |
| `chatTemplates.jinja.defaultCaps.parallelToolCalls` | `boolean` |
| `chatTemplates.jinja.defaultCaps.systemRole` | `boolean` |
| `chatTemplates.jinja.defaultCaps.toolCalls` | `boolean` |
| `chatTemplates.jinja.defaultCaps.tools` | `boolean` |
| `chatTemplates.jinja.toolUse` | `boolean` |
| `chatTemplates.jinja.toolUseCaps?` | { `parallelToolCalls`: `boolean` ; `systemRole`: `boolean` ; `toolCalls`: `boolean` ; `tools`: `boolean`  } |
| `chatTemplates.jinja.toolUseCaps.parallelToolCalls` | `boolean` |
| `chatTemplates.jinja.toolUseCaps.systemRole` | `boolean` |
| `chatTemplates.jinja.toolUseCaps.toolCalls` | `boolean` |
| `chatTemplates.jinja.toolUseCaps.tools` | `boolean` |
| `chatTemplates.llamaChat` | `boolean` |
| `desc` | `string` |
| `isChatTemplateSupported` | `boolean` |
| `is_hybrid` | `boolean` |
| `is_recurrent` | `boolean` |
| `metadata` | `Object` |
| `nEmbd` | `number` |
| `nParams` | `number` |
| `size` | `number` |

#### Defined in

[index.ts:404](https://github.com/mybigday/llama.rn/blob/a8c8191/src/index.ts#L404)

___

### parallel

• **parallel**: `Object`

Parallel processing namespace for non-blocking queue operations

#### Type declaration

| Name | Type |
| :------ | :------ |
| `completion` | (`params`: [`ParallelCompletionParams`](../README.md#parallelcompletionparams), `onToken?`: (`requestId`: `number`, `data`: [`TokenData`](../README.md#tokendata)) => `void`) => `Promise`<{ `promise`: `Promise`<[`NativeCompletionResult`](../README.md#nativecompletionresult)\> ; `requestId`: `number` ; `stop`: () => `Promise`<`void`\>  }\> |
| `configure` | (`config`: { `n_batch?`: `number` ; `n_parallel?`: `number`  }) => `Promise`<`boolean`\> |
| `disable` | () => `Promise`<`boolean`\> |
| `embedding` | (`text`: `string`, `params?`: [`NativeEmbeddingParams`](../README.md#nativeembeddingparams)) => `Promise`<{ `promise`: `Promise`<[`NativeEmbeddingResult`](../README.md#nativeembeddingresult)\> ; `requestId`: `number`  }\> |
| `enable` | (`config?`: { `n_batch?`: `number` ; `n_parallel?`: `number`  }) => `Promise`<`boolean`\> |
| `getStatus` | () => `Promise`<[`ParallelStatus`](../README.md#parallelstatus)\> |
| `rerank` | (`query`: `string`, `documents`: `string`[], `params?`: [`RerankParams`](../README.md#rerankparams)) => `Promise`<{ `promise`: `Promise`<[`RerankResult`](../README.md#rerankresult)[]\> ; `requestId`: `number`  }\> |
| `subscribeToStatus` | (`callback`: (`status`: [`ParallelStatus`](../README.md#parallelstatus)) => `void`) => `Promise`<{ `remove`: () => `void`  }\> |

#### Defined in

[index.ts:413](https://github.com/mybigday/llama.rn/blob/a8c8191/src/index.ts#L413)

___

### reasonNoGPU

• **reasonNoGPU**: `string` = `''`

#### Defined in

[index.ts:400](https://github.com/mybigday/llama.rn/blob/a8c8191/src/index.ts#L400)

___

### systemInfo

• **systemInfo**: `string`

#### Defined in

[index.ts:408](https://github.com/mybigday/llama.rn/blob/a8c8191/src/index.ts#L408)

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

[index.ts:1010](https://github.com/mybigday/llama.rn/blob/a8c8191/src/index.ts#L1010)

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

[index.ts:979](https://github.com/mybigday/llama.rn/blob/a8c8191/src/index.ts#L979)

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

[index.ts:1300](https://github.com/mybigday/llama.rn/blob/a8c8191/src/index.ts#L1300)

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

[index.ts:838](https://github.com/mybigday/llama.rn/blob/a8c8191/src/index.ts#L838)

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

[index.ts:1247](https://github.com/mybigday/llama.rn/blob/a8c8191/src/index.ts#L1247)

___

### decodeAudioEmbeddings

▸ **decodeAudioEmbeddings**(`embeddings`, `embeddingDim`): `Promise`<`number`[]\>

#### Parameters

| Name | Type |
| :------ | :------ |
| `embeddings` | `number`[] |
| `embeddingDim` | `number` |

#### Returns

`Promise`<`number`[]\>

#### Defined in

[index.ts:1270](https://github.com/mybigday/llama.rn/blob/a8c8191/src/index.ts#L1270)

___

### decodeAudioTokens

▸ **decodeAudioTokens**(`tokens`): `Promise`<`number`[]\>

#### Parameters

| Name | Type |
| :------ | :------ |
| `tokens` | `number`[] |

#### Returns

`Promise`<`number`[]\>

#### Defined in

[index.ts:1201](https://github.com/mybigday/llama.rn/blob/a8c8191/src/index.ts#L1201)

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

[index.ts:950](https://github.com/mybigday/llama.rn/blob/a8c8191/src/index.ts#L950)

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

[index.ts:955](https://github.com/mybigday/llama.rn/blob/a8c8191/src/index.ts#L955)

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

[index.ts:1226](https://github.com/mybigday/llama.rn/blob/a8c8191/src/index.ts#L1226)

___

### getAudioSampleRate

▸ **getAudioSampleRate**(): `Promise`<`number`\>

#### Returns

`Promise`<`number`\>

#### Defined in

[index.ts:1278](https://github.com/mybigday/llama.rn/blob/a8c8191/src/index.ts#L1278)

___

### getFormattedAudioCompletion

▸ **getFormattedAudioCompletion**(`options`): `Promise`<{ `embedding`: `boolean` ; `flow`: ``""`` \| ``"tokens"`` \| ``"codec_lm_ar"`` \| ``"continuous_embd"`` ; `grammar?`: `string` ; `prompt`: `string`  }\>

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
| `options.speaker?` | `string` \| [`LlamaSpeaker`](LlamaSpeaker.md) |

#### Returns

`Promise`<{ `embedding`: `boolean` ; `flow`: ``""`` \| ``"tokens"`` \| ``"codec_lm_ar"`` \| ``"continuous_embd"`` ; `grammar?`: `string` ; `prompt`: `string`  }\>

#### Defined in

[index.ts:1126](https://github.com/mybigday/llama.rn/blob/a8c8191/src/index.ts#L1126)

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
| `params.parallel_tool_calls?` | `object` |
| `params.reasoning_format?` | ``"none"`` \| ``"auto"`` \| ``"deepseek"`` |
| `params.response_format?` | [`CompletionResponseFormat`](../README.md#completionresponseformat) |
| `params.tool_choice?` | `string` |
| `params.tools?` | `object` |

#### Returns

`Promise`<[`FormattedChatResult`](../README.md#formattedchatresult) \| [`JinjaFormattedChatResult`](../README.md#jinjaformattedchatresult)\>

#### Defined in

[index.ts:719](https://github.com/mybigday/llama.rn/blob/a8c8191/src/index.ts#L719)

___

### getLoadedLoraAdapters

▸ **getLoadedLoraAdapters**(): `Promise`<{ `path`: `string` ; `scaled?`: `number`  }[]\>

#### Returns

`Promise`<{ `path`: `string` ; `scaled?`: `number`  }[]\>

#### Defined in

[index.ts:1023](https://github.com/mybigday/llama.rn/blob/a8c8191/src/index.ts#L1023)

___

### getMultimodalSupport

▸ **getMultimodalSupport**(): `Promise`<{ `audio`: `boolean` ; `vision`: `boolean`  }\>

#### Returns

`Promise`<{ `audio`: `boolean` ; `vision`: `boolean`  }\>

#### Defined in

[index.ts:1065](https://github.com/mybigday/llama.rn/blob/a8c8191/src/index.ts#L1065)

___

### getTTSCapabilities

▸ **getTTSCapabilities**(): `Promise`<[`TTSCapabilities`](../interfaces/TTSCapabilities.md)\>

#### Returns

`Promise`<[`TTSCapabilities`](../interfaces/TTSCapabilities.md)\>

#### Defined in

[index.ts:1105](https://github.com/mybigday/llama.rn/blob/a8c8191/src/index.ts#L1105)

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

[index.ts:1039](https://github.com/mybigday/llama.rn/blob/a8c8191/src/index.ts#L1039)

___

### initVocoder

▸ **initVocoder**(`«destructured»`): `Promise`<`boolean`\>

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

[index.ts:1078](https://github.com/mybigday/llama.rn/blob/a8c8191/src/index.ts#L1078)

___

### isJinjaSupported

▸ **isJinjaSupported**(): `boolean`

#### Returns

`boolean`

#### Defined in

[index.ts:714](https://github.com/mybigday/llama.rn/blob/a8c8191/src/index.ts#L714)

___

### isLlamaChatSupported

▸ **isLlamaChatSupported**(): `boolean`

#### Returns

`boolean`

#### Defined in

[index.ts:710](https://github.com/mybigday/llama.rn/blob/a8c8191/src/index.ts#L710)

___

### isMultimodalEnabled

▸ **isMultimodalEnabled**(): `Promise`<`boolean`\>

#### Returns

`Promise`<`boolean`\>

#### Defined in

[index.ts:1060](https://github.com/mybigday/llama.rn/blob/a8c8191/src/index.ts#L1060)

___

### isVocoderEnabled

▸ **isVocoderEnabled**(): `Promise`<`boolean`\>

#### Returns

`Promise`<`boolean`\>

#### Defined in

[index.ts:1100](https://github.com/mybigday/llama.rn/blob/a8c8191/src/index.ts#L1100)

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

[index.ts:695](https://github.com/mybigday/llama.rn/blob/a8c8191/src/index.ts#L695)

___

### release

▸ **release**(): `Promise`<`void`\>

#### Returns

`Promise`<`void`\>

#### Defined in

[index.ts:1305](https://github.com/mybigday/llama.rn/blob/a8c8191/src/index.ts#L1305)

___

### releaseMultimodal

▸ **releaseMultimodal**(): `Promise`<`void`\>

#### Returns

`Promise`<`void`\>

#### Defined in

[index.ts:1073](https://github.com/mybigday/llama.rn/blob/a8c8191/src/index.ts#L1073)

___

### releaseVocoder

▸ **releaseVocoder**(): `Promise`<`void`\>

#### Returns

`Promise`<`void`\>

#### Defined in

[index.ts:1283](https://github.com/mybigday/llama.rn/blob/a8c8191/src/index.ts#L1283)

___

### removeLoraAdapters

▸ **removeLoraAdapters**(): `Promise`<`void`\>

#### Returns

`Promise`<`void`\>

#### Defined in

[index.ts:1018](https://github.com/mybigday/llama.rn/blob/a8c8191/src/index.ts#L1018)

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

[index.ts:963](https://github.com/mybigday/llama.rn/blob/a8c8191/src/index.ts#L963)

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

[index.ts:702](https://github.com/mybigday/llama.rn/blob/a8c8191/src/index.ts#L702)

___

### stopCompletion

▸ **stopCompletion**(): `Promise`<`void`\>

#### Returns

`Promise`<`void`\>

#### Defined in

[index.ts:933](https://github.com/mybigday/llama.rn/blob/a8c8191/src/index.ts#L933)

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

[index.ts:938](https://github.com/mybigday/llama.rn/blob/a8c8191/src/index.ts#L938)
