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
- [listSpeakers](LlamaContext.md#listspeakers)
- [loadSession](LlamaContext.md#loadsession)
- [loadSpeaker](LlamaContext.md#loadspeaker)
- [release](LlamaContext.md#release)
- [releaseMultimodal](LlamaContext.md#releasemultimodal)
- [releaseVocoder](LlamaContext.md#releasevocoder)
- [removeLoraAdapters](LlamaContext.md#removeloraadapters)
- [rerank](LlamaContext.md#rerank)
- [saveSession](LlamaContext.md#savesession)
- [saveSpeaker](LlamaContext.md#savespeaker)
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

[index.ts:748](https://github.com/mybigday/llama.rn/blob/897c824b/src/index.ts#L748)

## Properties

### androidLib

• **androidLib**: `undefined` \| `string`

#### Defined in

[index.ts:437](https://github.com/mybigday/llama.rn/blob/897c824b/src/index.ts#L437)

___

### devices

• **devices**: `undefined` \| `string`[]

#### Defined in

[index.ts:433](https://github.com/mybigday/llama.rn/blob/897c824b/src/index.ts#L433)

___

### gpu

• **gpu**: `boolean` = `false`

#### Defined in

[index.ts:429](https://github.com/mybigday/llama.rn/blob/897c824b/src/index.ts#L429)

___

### id

• **id**: `number`

#### Defined in

[index.ts:427](https://github.com/mybigday/llama.rn/blob/897c824b/src/index.ts#L427)

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
| `decision?` | { `audioInput`: `boolean` ; `error?`: `string` ; `imageInput`: `boolean` ; `nOptionsMax`: `number` ; `readout?`: [`DecisionReadout`](../README.md#decisionreadout) ; `textGeneration`: `boolean` ; `type`: [`DecisionModelType`](../README.md#decisionmodeltype)  } | Set if the model is a typed decision model, see `LlamaContext.decide()` |
| `decision.audioInput` | `boolean` | The prompt has a place for an audio clip (multimodal still has to be initialized) |
| `decision.error?` | `string` | Why the model cannot be used, when `type` is `unknown` |
| `decision.imageInput` | `boolean` | The prompt has a place for images (multimodal still has to be initialized). For a model read from its label logits (openjev, lev, nimble, ...) this follows from its template: a template that writes `images` takes them. |
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

[index.ts:435](https://github.com/mybigday/llama.rn/blob/897c824b/src/index.ts#L435)

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

[index.ts:444](https://github.com/mybigday/llama.rn/blob/897c824b/src/index.ts#L444)

___

### reasonNoGPU

• **reasonNoGPU**: `string` = `''`

#### Defined in

[index.ts:431](https://github.com/mybigday/llama.rn/blob/897c824b/src/index.ts#L431)

___

### systemInfo

• **systemInfo**: `string`

#### Defined in

[index.ts:439](https://github.com/mybigday/llama.rn/blob/897c824b/src/index.ts#L439)

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

[index.ts:1100](https://github.com/mybigday/llama.rn/blob/897c824b/src/index.ts#L1100)

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

[index.ts:1070](https://github.com/mybigday/llama.rn/blob/897c824b/src/index.ts#L1070)

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

[index.ts:1481](https://github.com/mybigday/llama.rn/blob/897c824b/src/index.ts#L1481)

___

### completion

▸ **completion**(`params`, `callback?`): `Promise`<[`NativeCompletionResult`](../README.md#nativecompletionresult)\>

#### Parameters

| Name | Type |
| :------ | :------ |
| `params` | `Omit`<[`NativeCompletionParams`](../README.md#nativecompletionparams), ``"emit_partial_completion"`` \| ``"prompt"``\> & [`CompletionBaseParams`](../README.md#completionbaseparams) & { `speaker?`: `string` \| [`LlamaSpeaker`](LlamaSpeaker.md)  } |
| `callback?` | (`data`: [`TokenData`](../README.md#tokendata)) => `void` |

#### Returns

`Promise`<[`NativeCompletionResult`](../README.md#nativecompletionresult)\>

#### Defined in

[index.ts:904](https://github.com/mybigday/llama.rn/blob/897c824b/src/index.ts#L904)

___

### createSpeaker

▸ **createSpeaker**(`config`): `Promise`<[`LlamaSpeaker`](LlamaSpeaker.md)\>

Clone a voice from a reference clip (mono, any sample rate) and its
transcript `refText`, for models with `canSpeakerFromAudio`. All of the
voice's conditioning is computed here, once; `emotion` and `bake` are
not used.

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

[index.ts:1383](https://github.com/mybigday/llama.rn/blob/897c824b/src/index.ts#L1383)

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

[index.ts:1063](https://github.com/mybigday/llama.rn/blob/897c824b/src/index.ts#L1063)

___

### decodeAudioEmbeddings

▸ **decodeAudioEmbeddings**(`embeddings`, `embeddingDim`): `Promise`<`number`[]\>

PCM of a text-to-speech completion's `embeddings` / `embedding_dim`
(latent models), like `decodeAudioTokens`.

#### Parameters

| Name | Type |
| :------ | :------ |
| `embeddings` | `number`[] \| `Float32Array` |
| `embeddingDim` | `number` |

#### Returns

`Promise`<`number`[]\>

#### Defined in

[index.ts:1444](https://github.com/mybigday/llama.rn/blob/897c824b/src/index.ts#L1444)

___

### decodeAudioTokens

▸ **decodeAudioTokens**(`tokens`): `Promise`<`number`[]\>

PCM (at `getAudioSampleRate()`) of a text-to-speech completion's
`audio_tokens`, with the voice of the last text-to-speech completion. Any
earlier completion's tokens decode, in any order; tokens generated with
another voice decode with this one.

#### Parameters

| Name | Type |
| :------ | :------ |
| `tokens` | `number`[] \| `Int32Array` |

#### Returns

`Promise`<`number`[]\>

#### Defined in

[index.ts:1314](https://github.com/mybigday/llama.rn/blob/897c824b/src/index.ts#L1314)

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

[index.ts:1020](https://github.com/mybigday/llama.rn/blob/897c824b/src/index.ts#L1020)

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

[index.ts:1025](https://github.com/mybigday/llama.rn/blob/897c824b/src/index.ts#L1025)

___

### generateAudioCodes

▸ **generateAudioCodes**(`options`): `Promise`<{ `aborted`: `boolean` ; `codes`: `number`[] ; `nCodebook`: `number` ; `nFrames`: `number` ; `stoppedOnEos`: `boolean`  }\>

DEPRECATED: `getFormattedAudioCompletion` + `completion()` with its
`completionParams`; `maxFrames` is `n_predict`, `temperature`, `topP`,
`topK` and `seed` are the completion's. `onFrame` fires per completion
step with an empty `codes`; the codes come in the result.

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

[index.ts:1333](https://github.com/mybigday/llama.rn/blob/897c824b/src/index.ts#L1333)

___

### getAudioSampleRate

▸ **getAudioSampleRate**(): `Promise`<`number`\>

#### Returns

`Promise`<`number`\>

#### Defined in

[index.ts:1459](https://github.com/mybigday/llama.rn/blob/897c824b/src/index.ts#L1459)

___

### getFormattedAudioCompletion

▸ **getFormattedAudioCompletion**(`options`): `Promise`<{ `completionParams`: [`TTSCompletionParams`](../interfaces/TTSCompletionParams.md) ; `embedding`: `boolean` ; `flow`: ``""`` \| ``"tokens"`` \| ``"continuous_embd"`` ; `grammar?`: `string` ; `prompt`: `string`  }\>

Arm a text-to-speech request and report how to run it.

The flow is `getFormattedAudioCompletion` → `completion()` →
`decodeAudioTokens` / `decodeAudioEmbeddings`. Like `getFormattedChat`,
this returns what the request's format needs, as `completionParams`: the
prompt, the grammar or `logit_bias` token mask, `embedding` and the
model's minimum length (`n_min_predict`). Sampling (the backbone GGUF's
`general.sampling.*` unless set) and `n_predict` are the `completion()`'s,
which runs with
exactly the params it gets; its `prompt` must be
`completionParams.prompt`. The completion samples with its own sampler
(codec.cpp's own samplers use the same settings) and hands every step to
codec.cpp, which decides what the backbone reads next and when the
speech ends; the result carries the decoder input:
`audio_tokens` (codec codes, `audio_tokens_width` per frame) or
`embeddings` + `embedding_dim` (latents, e.g. Soprano / BlueMagpie).
Long text is the caller's to split: one request per piece.

- `prompt` — text to speak. Phonemized if `phonemizer` is supplied.
- `speaker` — a builtin voice name of the codec (`builtinVoices`), a
  `LlamaSpeaker`, or `undefined` for the model's default voice. A voice
  written as text (OuteTTS / NeuTTS references) goes into the prompt;
  an embedding voice becomes a `<__voice__>` marker, filled by the
  `speaker` given to `completion()` — pass the same one there.
- `phonemizer` — optional `(text, language) => string | Promise<string>`,
  for models with `requiresPhonemes`.
- `language` — phonemizer hint; defaults to capabilities.defaultLanguage.
- `instruct` — how to speak (e.g. "Speak cheerfully"), for models with
  `canTakeInstruct`; others reject it.

#### Parameters

| Name | Type |
| :------ | :------ |
| `options` | `Object` |
| `options.instruct?` | `string` |
| `options.language?` | `string` |
| `options.phonemizer?` | (`text`: `string`, `language`: `string`) => `string` \| `Promise`<`string`\> |
| `options.prompt` | `string` |
| `options.speaker?` | `string` \| [`LlamaSpeaker`](LlamaSpeaker.md) |

#### Returns

`Promise`<{ `completionParams`: [`TTSCompletionParams`](../interfaces/TTSCompletionParams.md) ; `embedding`: `boolean` ; `flow`: ``""`` \| ``"tokens"`` \| ``"continuous_embd"`` ; `grammar?`: `string` ; `prompt`: `string`  }\>

#### Defined in

[index.ts:1244](https://github.com/mybigday/llama.rn/blob/897c824b/src/index.ts#L1244)

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

[index.ts:792](https://github.com/mybigday/llama.rn/blob/897c824b/src/index.ts#L792)

___

### getLoadedLoraAdapters

▸ **getLoadedLoraAdapters**(): `Promise`<{ `path`: `string` ; `scaled?`: `number`  }[]\>

#### Returns

`Promise`<{ `path`: `string` ; `scaled?`: `number`  }[]\>

#### Defined in

[index.ts:1113](https://github.com/mybigday/llama.rn/blob/897c824b/src/index.ts#L1113)

___

### getMultimodalSupport

▸ **getMultimodalSupport**(): `Promise`<{ `audio`: `boolean` ; `vision`: `boolean`  }\>

#### Returns

`Promise`<{ `audio`: `boolean` ; `vision`: `boolean`  }\>

#### Defined in

[index.ts:1155](https://github.com/mybigday/llama.rn/blob/897c824b/src/index.ts#L1155)

___

### getTTSCapabilities

▸ **getTTSCapabilities**(): `Promise`<[`TTSCapabilities`](../interfaces/TTSCapabilities.md)\>

#### Returns

`Promise`<[`TTSCapabilities`](../interfaces/TTSCapabilities.md)\>

#### Defined in

[index.ts:1208](https://github.com/mybigday/llama.rn/blob/897c824b/src/index.ts#L1208)

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

[index.ts:1129](https://github.com/mybigday/llama.rn/blob/897c824b/src/index.ts#L1129)

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
| › `kv_overrides?` | `string`[] |
| › `n_batch?` | `number` |
| › `path` | `string` |
| › `use_gpu?` | `boolean` |

#### Returns

`Promise`<`boolean`\>

#### Defined in

[index.ts:1175](https://github.com/mybigday/llama.rn/blob/897c824b/src/index.ts#L1175)

___

### isJinjaSupported

▸ **isJinjaSupported**(): `boolean`

#### Returns

`boolean`

#### Defined in

[index.ts:787](https://github.com/mybigday/llama.rn/blob/897c824b/src/index.ts#L787)

___

### isLlamaChatSupported

▸ **isLlamaChatSupported**(): `boolean`

#### Returns

`boolean`

#### Defined in

[index.ts:783](https://github.com/mybigday/llama.rn/blob/897c824b/src/index.ts#L783)

___

### isMultimodalEnabled

▸ **isMultimodalEnabled**(): `Promise`<`boolean`\>

#### Returns

`Promise`<`boolean`\>

#### Defined in

[index.ts:1150](https://github.com/mybigday/llama.rn/blob/897c824b/src/index.ts#L1150)

___

### isVocoderEnabled

▸ **isVocoderEnabled**(): `Promise`<`boolean`\>

#### Returns

`Promise`<`boolean`\>

#### Defined in

[index.ts:1203](https://github.com/mybigday/llama.rn/blob/897c824b/src/index.ts#L1203)

___

### listSpeakers

▸ **listSpeakers**(`path`): `Promise`<[`TTSSpeakerInfo`](../interfaces/TTSSpeakerInfo.md)[]\>

Speakers stored in a speaker file.

#### Parameters

| Name | Type |
| :------ | :------ |
| `path` | `string` |

#### Returns

`Promise`<[`TTSSpeakerInfo`](../interfaces/TTSSpeakerInfo.md)[]\>

#### Defined in

[index.ts:1434](https://github.com/mybigday/llama.rn/blob/897c824b/src/index.ts#L1434)

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

[index.ts:766](https://github.com/mybigday/llama.rn/blob/897c824b/src/index.ts#L766)

___

### loadSpeaker

▸ **loadSpeaker**(`path`, `options?`): `Promise`<[`LlamaSpeaker`](LlamaSpeaker.md)\>

Load a speaker from a speaker file (written by `saveSpeaker` or codec.cpp's
`tts-cli --save-speaker`). `options.name` picks a speaker inside the file
(default: the file's default speaker). Files of another model are
refused.

#### Parameters

| Name | Type |
| :------ | :------ |
| `path` | `string` |
| `options?` | `Object` |
| `options.name?` | `string` |

#### Returns

`Promise`<[`LlamaSpeaker`](LlamaSpeaker.md)\>

#### Defined in

[index.ts:1423](https://github.com/mybigday/llama.rn/blob/897c824b/src/index.ts#L1423)

___

### release

▸ **release**(): `Promise`<`void`\>

#### Returns

`Promise`<`void`\>

#### Defined in

[index.ts:1486](https://github.com/mybigday/llama.rn/blob/897c824b/src/index.ts#L1486)

___

### releaseMultimodal

▸ **releaseMultimodal**(): `Promise`<`void`\>

#### Returns

`Promise`<`void`\>

#### Defined in

[index.ts:1163](https://github.com/mybigday/llama.rn/blob/897c824b/src/index.ts#L1163)

___

### releaseVocoder

▸ **releaseVocoder**(): `Promise`<`void`\>

#### Returns

`Promise`<`void`\>

#### Defined in

[index.ts:1464](https://github.com/mybigday/llama.rn/blob/897c824b/src/index.ts#L1464)

___

### removeLoraAdapters

▸ **removeLoraAdapters**(): `Promise`<`void`\>

#### Returns

`Promise`<`void`\>

#### Defined in

[index.ts:1108](https://github.com/mybigday/llama.rn/blob/897c824b/src/index.ts#L1108)

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

[index.ts:1036](https://github.com/mybigday/llama.rn/blob/897c824b/src/index.ts#L1036)

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

[index.ts:773](https://github.com/mybigday/llama.rn/blob/897c824b/src/index.ts#L773)

___

### saveSpeaker

▸ **saveSpeaker**(`speaker`, `path`, `options?`): `Promise`<`void`\>

Write a speaker's voice (all of its conditioning, a codec_speaker) to a
speaker file of this model. `options.name` names it inside the file
(default: codec.cpp's name). Reload it with `loadSpeaker`; the reloaded
voice is identical.

#### Parameters

| Name | Type |
| :------ | :------ |
| `speaker` | [`LlamaSpeaker`](LlamaSpeaker.md) |
| `path` | `string` |
| `options?` | `Object` |
| `options.name?` | `string` |

#### Returns

`Promise`<`void`\>

#### Defined in

[index.ts:1407](https://github.com/mybigday/llama.rn/blob/897c824b/src/index.ts#L1407)

___

### stopCompletion

▸ **stopCompletion**(): `Promise`<`void`\>

#### Returns

`Promise`<`void`\>

#### Defined in

[index.ts:1003](https://github.com/mybigday/llama.rn/blob/897c824b/src/index.ts#L1003)

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

[index.ts:1008](https://github.com/mybigday/llama.rn/blob/897c824b/src/index.ts#L1008)
