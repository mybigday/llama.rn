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
- [decodeAudioTokens](LlamaContext.md#decodeaudiotokens)
- [detokenize](LlamaContext.md#detokenize)
- [embedding](LlamaContext.md#embedding)
- [getAudioCompletionGuideTokens](LlamaContext.md#getaudiocompletionguidetokens)
- [getFormattedAudioCompletion](LlamaContext.md#getformattedaudiocompletion)
- [getFormattedChat](LlamaContext.md#getformattedchat)
- [getLoadedLoraAdapters](LlamaContext.md#getloadedloraadapters)
- [getMultimodalSupport](LlamaContext.md#getmultimodalsupport)
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

[index.ts:624](https://github.com/mybigday/llama.rn/blob/6d85189e/src/index.ts#L624)

## Properties

### androidLib

• **androidLib**: `undefined` \| `string`

#### Defined in

[index.ts:355](https://github.com/mybigday/llama.rn/blob/6d85189e/src/index.ts#L355)

___

### devices

• **devices**: `undefined` \| `string`[]

#### Defined in

[index.ts:351](https://github.com/mybigday/llama.rn/blob/6d85189e/src/index.ts#L351)

___

### gpu

• **gpu**: `boolean` = `false`

#### Defined in

[index.ts:347](https://github.com/mybigday/llama.rn/blob/6d85189e/src/index.ts#L347)

___

### id

• **id**: `number`

#### Defined in

[index.ts:345](https://github.com/mybigday/llama.rn/blob/6d85189e/src/index.ts#L345)

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

[index.ts:353](https://github.com/mybigday/llama.rn/blob/6d85189e/src/index.ts#L353)

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

[index.ts:362](https://github.com/mybigday/llama.rn/blob/6d85189e/src/index.ts#L362)

___

### reasonNoGPU

• **reasonNoGPU**: `string` = `''`

#### Defined in

[index.ts:349](https://github.com/mybigday/llama.rn/blob/6d85189e/src/index.ts#L349)

___

### systemInfo

• **systemInfo**: `string`

#### Defined in

[index.ts:357](https://github.com/mybigday/llama.rn/blob/6d85189e/src/index.ts#L357)

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

[index.ts:946](https://github.com/mybigday/llama.rn/blob/6d85189e/src/index.ts#L946)

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

[index.ts:915](https://github.com/mybigday/llama.rn/blob/6d85189e/src/index.ts#L915)

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

[index.ts:1075](https://github.com/mybigday/llama.rn/blob/6d85189e/src/index.ts#L1075)

___

### completion

▸ **completion**(`params`, `callback?`): `Promise`<[`NativeCompletionResult`](../README.md#nativecompletionresult)\>

#### Parameters

| Name | Type |
| :------ | :------ |
| `params` | [`CompletionParams`](../README.md#completionparams) |
| `callback?` | (`data`: [`TokenData`](../README.md#tokendata)) => `void` |

#### Returns

`Promise`<[`NativeCompletionResult`](../README.md#nativecompletionresult)\>

#### Defined in

[index.ts:785](https://github.com/mybigday/llama.rn/blob/6d85189e/src/index.ts#L785)

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

[index.ts:1053](https://github.com/mybigday/llama.rn/blob/6d85189e/src/index.ts#L1053)

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

[index.ts:886](https://github.com/mybigday/llama.rn/blob/6d85189e/src/index.ts#L886)

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

[index.ts:891](https://github.com/mybigday/llama.rn/blob/6d85189e/src/index.ts#L891)

___

### getAudioCompletionGuideTokens

▸ **getAudioCompletionGuideTokens**(`textToSpeak`): `Promise`<`number`[]\>

#### Parameters

| Name | Type |
| :------ | :------ |
| `textToSpeak` | `string` |

#### Returns

`Promise`<`number`[]\>

#### Defined in

[index.ts:1046](https://github.com/mybigday/llama.rn/blob/6d85189e/src/index.ts#L1046)

___

### getFormattedAudioCompletion

▸ **getFormattedAudioCompletion**(`speaker`, `textToSpeak`): `Promise`<{ `grammar?`: `string` ; `prompt`: `string`  }\>

#### Parameters

| Name | Type |
| :------ | :------ |
| `speaker` | ``null`` \| `object` |
| `textToSpeak` | `string` |

#### Returns

`Promise`<{ `grammar?`: `string` ; `prompt`: `string`  }\>

#### Defined in

[index.ts:1031](https://github.com/mybigday/llama.rn/blob/6d85189e/src/index.ts#L1031)

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

[index.ts:666](https://github.com/mybigday/llama.rn/blob/6d85189e/src/index.ts#L666)

___

### getLoadedLoraAdapters

▸ **getLoadedLoraAdapters**(): `Promise`<{ `path`: `string` ; `scaled?`: `number`  }[]\>

#### Returns

`Promise`<{ `path`: `string` ; `scaled?`: `number`  }[]\>

#### Defined in

[index.ts:959](https://github.com/mybigday/llama.rn/blob/6d85189e/src/index.ts#L959)

___

### getMultimodalSupport

▸ **getMultimodalSupport**(): `Promise`<{ `audio`: `boolean` ; `vision`: `boolean`  }\>

#### Returns

`Promise`<{ `audio`: `boolean` ; `vision`: `boolean`  }\>

#### Defined in

[index.ts:1001](https://github.com/mybigday/llama.rn/blob/6d85189e/src/index.ts#L1001)

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

[index.ts:975](https://github.com/mybigday/llama.rn/blob/6d85189e/src/index.ts#L975)

___

### initVocoder

▸ **initVocoder**(`«destructured»`): `Promise`<`boolean`\>

#### Parameters

| Name | Type |
| :------ | :------ |
| `«destructured»` | `Object` |
| › `n_batch?` | `number` |
| › `path` | `string` |

#### Returns

`Promise`<`boolean`\>

#### Defined in

[index.ts:1014](https://github.com/mybigday/llama.rn/blob/6d85189e/src/index.ts#L1014)

___

### isJinjaSupported

▸ **isJinjaSupported**(): `boolean`

#### Returns

`boolean`

#### Defined in

[index.ts:661](https://github.com/mybigday/llama.rn/blob/6d85189e/src/index.ts#L661)

___

### isLlamaChatSupported

▸ **isLlamaChatSupported**(): `boolean`

#### Returns

`boolean`

#### Defined in

[index.ts:657](https://github.com/mybigday/llama.rn/blob/6d85189e/src/index.ts#L657)

___

### isMultimodalEnabled

▸ **isMultimodalEnabled**(): `Promise`<`boolean`\>

#### Returns

`Promise`<`boolean`\>

#### Defined in

[index.ts:996](https://github.com/mybigday/llama.rn/blob/6d85189e/src/index.ts#L996)

___

### isVocoderEnabled

▸ **isVocoderEnabled**(): `Promise`<`boolean`\>

#### Returns

`Promise`<`boolean`\>

#### Defined in

[index.ts:1026](https://github.com/mybigday/llama.rn/blob/6d85189e/src/index.ts#L1026)

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

[index.ts:642](https://github.com/mybigday/llama.rn/blob/6d85189e/src/index.ts#L642)

___

### release

▸ **release**(): `Promise`<`void`\>

#### Returns

`Promise`<`void`\>

#### Defined in

[index.ts:1080](https://github.com/mybigday/llama.rn/blob/6d85189e/src/index.ts#L1080)

___

### releaseMultimodal

▸ **releaseMultimodal**(): `Promise`<`void`\>

#### Returns

`Promise`<`void`\>

#### Defined in

[index.ts:1009](https://github.com/mybigday/llama.rn/blob/6d85189e/src/index.ts#L1009)

___

### releaseVocoder

▸ **releaseVocoder**(): `Promise`<`void`\>

#### Returns

`Promise`<`void`\>

#### Defined in

[index.ts:1058](https://github.com/mybigday/llama.rn/blob/6d85189e/src/index.ts#L1058)

___

### removeLoraAdapters

▸ **removeLoraAdapters**(): `Promise`<`void`\>

#### Returns

`Promise`<`void`\>

#### Defined in

[index.ts:954](https://github.com/mybigday/llama.rn/blob/6d85189e/src/index.ts#L954)

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

[index.ts:899](https://github.com/mybigday/llama.rn/blob/6d85189e/src/index.ts#L899)

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

[index.ts:649](https://github.com/mybigday/llama.rn/blob/6d85189e/src/index.ts#L649)

___

### stopCompletion

▸ **stopCompletion**(): `Promise`<`void`\>

#### Returns

`Promise`<`void`\>

#### Defined in

[index.ts:869](https://github.com/mybigday/llama.rn/blob/6d85189e/src/index.ts#L869)

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

[index.ts:874](https://github.com/mybigday/llama.rn/blob/6d85189e/src/index.ts#L874)
