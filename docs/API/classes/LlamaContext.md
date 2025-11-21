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

[index.ts:559](https://github.com/mybigday/llama.rn/blob/8dd6e05/src/index.ts#L559)

## Properties

### androidLib

• **androidLib**: `undefined` \| `string`

#### Defined in

[index.ts:274](https://github.com/mybigday/llama.rn/blob/8dd6e05/src/index.ts#L274)

___

### devices

• **devices**: `undefined` \| `string`[]

#### Defined in

[index.ts:270](https://github.com/mybigday/llama.rn/blob/8dd6e05/src/index.ts#L270)

___

### gpu

• **gpu**: `boolean` = `false`

#### Defined in

[index.ts:266](https://github.com/mybigday/llama.rn/blob/8dd6e05/src/index.ts#L266)

___

### id

• **id**: `number`

#### Defined in

[index.ts:264](https://github.com/mybigday/llama.rn/blob/8dd6e05/src/index.ts#L264)

___

### model

• **model**: `Object`

#### Type declaration

| Name | Type |
| :------ | :------ |
| `chatTemplates` | { `llamaChat`: `boolean` ; `minja`: { `default`: `boolean` ; `defaultCaps`: { `parallelToolCalls`: `boolean` ; `systemRole`: `boolean` ; `toolCallId`: `boolean` ; `toolCalls`: `boolean` ; `toolResponses`: `boolean` ; `tools`: `boolean`  } ; `toolUse`: `boolean` ; `toolUseCaps`: { `parallelToolCalls`: `boolean` ; `systemRole`: `boolean` ; `toolCallId`: `boolean` ; `toolCalls`: `boolean` ; `toolResponses`: `boolean` ; `tools`: `boolean`  }  }  } |
| `chatTemplates.llamaChat` | `boolean` |
| `chatTemplates.minja` | { `default`: `boolean` ; `defaultCaps`: { `parallelToolCalls`: `boolean` ; `systemRole`: `boolean` ; `toolCallId`: `boolean` ; `toolCalls`: `boolean` ; `toolResponses`: `boolean` ; `tools`: `boolean`  } ; `toolUse`: `boolean` ; `toolUseCaps`: { `parallelToolCalls`: `boolean` ; `systemRole`: `boolean` ; `toolCallId`: `boolean` ; `toolCalls`: `boolean` ; `toolResponses`: `boolean` ; `tools`: `boolean`  }  } |
| `chatTemplates.minja.default` | `boolean` |
| `chatTemplates.minja.defaultCaps` | { `parallelToolCalls`: `boolean` ; `systemRole`: `boolean` ; `toolCallId`: `boolean` ; `toolCalls`: `boolean` ; `toolResponses`: `boolean` ; `tools`: `boolean`  } |
| `chatTemplates.minja.defaultCaps.parallelToolCalls` | `boolean` |
| `chatTemplates.minja.defaultCaps.systemRole` | `boolean` |
| `chatTemplates.minja.defaultCaps.toolCallId` | `boolean` |
| `chatTemplates.minja.defaultCaps.toolCalls` | `boolean` |
| `chatTemplates.minja.defaultCaps.toolResponses` | `boolean` |
| `chatTemplates.minja.defaultCaps.tools` | `boolean` |
| `chatTemplates.minja.toolUse` | `boolean` |
| `chatTemplates.minja.toolUseCaps` | { `parallelToolCalls`: `boolean` ; `systemRole`: `boolean` ; `toolCallId`: `boolean` ; `toolCalls`: `boolean` ; `toolResponses`: `boolean` ; `tools`: `boolean`  } |
| `chatTemplates.minja.toolUseCaps.parallelToolCalls` | `boolean` |
| `chatTemplates.minja.toolUseCaps.systemRole` | `boolean` |
| `chatTemplates.minja.toolUseCaps.toolCallId` | `boolean` |
| `chatTemplates.minja.toolUseCaps.toolCalls` | `boolean` |
| `chatTemplates.minja.toolUseCaps.toolResponses` | `boolean` |
| `chatTemplates.minja.toolUseCaps.tools` | `boolean` |
| `desc` | `string` |
| `isChatTemplateSupported` | `boolean` |
| `metadata` | `Object` |
| `nEmbd` | `number` |
| `nParams` | `number` |
| `size` | `number` |

#### Defined in

[index.ts:272](https://github.com/mybigday/llama.rn/blob/8dd6e05/src/index.ts#L272)

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
| `rerank` | (`query`: `string`, `documents`: `string`[], `params?`: [`RerankParams`](../README.md#rerankparams)) => `Promise`<{ `promise`: `Promise`<[`RerankResult`](../README.md#rerankresult)[]\> ; `requestId`: `number`  }\> |

#### Defined in

[index.ts:281](https://github.com/mybigday/llama.rn/blob/8dd6e05/src/index.ts#L281)

___

### reasonNoGPU

• **reasonNoGPU**: `string` = `''`

#### Defined in

[index.ts:268](https://github.com/mybigday/llama.rn/blob/8dd6e05/src/index.ts#L268)

___

### systemInfo

• **systemInfo**: `string`

#### Defined in

[index.ts:276](https://github.com/mybigday/llama.rn/blob/8dd6e05/src/index.ts#L276)

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

[index.ts:902](https://github.com/mybigday/llama.rn/blob/8dd6e05/src/index.ts#L902)

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

[index.ts:872](https://github.com/mybigday/llama.rn/blob/8dd6e05/src/index.ts#L872)

___

### completion

▸ **completion**(`params`, `callback?`): `Promise`<[`NativeCompletionResult`](../README.md#nativecompletionresult)\>

Generate a completion based on the provided parameters

#### Parameters

| Name | Type | Description |
| :------ | :------ | :------ |
| `params` | [`CompletionParams`](../README.md#completionparams) | Completion parameters including prompt or messages |
| `callback?` | (`data`: [`TokenData`](../README.md#tokendata)) => `void` | Optional callback for token-by-token streaming |

#### Returns

`Promise`<[`NativeCompletionResult`](../README.md#nativecompletionresult)\>

Promise resolving to the completion result

Note: For multimodal support, you can include an media_paths parameter.
This will process the images and add them to the context before generating text.
Multimodal support must be enabled via initMultimodal() first.

#### Defined in

[index.ts:722](https://github.com/mybigday/llama.rn/blob/8dd6e05/src/index.ts#L722)

___

### decodeAudioTokens

▸ **decodeAudioTokens**(`tokens`): `Promise`<`number`[]\>

Decode audio tokens

#### Parameters

| Name | Type | Description |
| :------ | :------ | :------ |
| `tokens` | `number`[] | Array of audio tokens |

#### Returns

`Promise`<`number`[]\>

Promise resolving to the decoded audio tokens

#### Defined in

[index.ts:1034](https://github.com/mybigday/llama.rn/blob/8dd6e05/src/index.ts#L1034)

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

[index.ts:833](https://github.com/mybigday/llama.rn/blob/8dd6e05/src/index.ts#L833)

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

[index.ts:837](https://github.com/mybigday/llama.rn/blob/8dd6e05/src/index.ts#L837)

___

### getAudioCompletionGuideTokens

▸ **getAudioCompletionGuideTokens**(`textToSpeak`): `Promise`<`number`[]\>

Get guide tokens for audio completion

#### Parameters

| Name | Type | Description |
| :------ | :------ | :------ |
| `textToSpeak` | `string` | Text to speak |

#### Returns

`Promise`<`number`[]\>

Promise resolving to the guide tokens

#### Defined in

[index.ts:1023](https://github.com/mybigday/llama.rn/blob/8dd6e05/src/index.ts#L1023)

___

### getFormattedAudioCompletion

▸ **getFormattedAudioCompletion**(`speaker`, `textToSpeak`): `Promise`<{ `grammar?`: `string` ; `prompt`: `string`  }\>

Get a formatted audio completion prompt

#### Parameters

| Name | Type | Description |
| :------ | :------ | :------ |
| `speaker` | ``null`` \| `object` | - |
| `textToSpeak` | `string` | Text to speak |

#### Returns

`Promise`<{ `grammar?`: `string` ; `prompt`: `string`  }\>

Promise resolving to the formatted audio completion result with prompt and grammar

#### Defined in

[index.ts:1004](https://github.com/mybigday/llama.rn/blob/8dd6e05/src/index.ts#L1004)

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
| `params.chat_template_kwargs?` | `Record`<`string`, `string`\> |
| `params.enable_thinking?` | `boolean` |
| `params.jinja?` | `boolean` |
| `params.now?` | `string` \| `number` |
| `params.parallel_tool_calls?` | `object` |
| `params.response_format?` | [`CompletionResponseFormat`](../README.md#completionresponseformat) |
| `params.tool_choice?` | `string` |
| `params.tools?` | `object` |

#### Returns

`Promise`<[`FormattedChatResult`](../README.md#formattedchatresult) \| [`JinjaFormattedChatResult`](../README.md#jinjaformattedchatresult)\>

#### Defined in

[index.ts:605](https://github.com/mybigday/llama.rn/blob/8dd6e05/src/index.ts#L605)

___

### getLoadedLoraAdapters

▸ **getLoadedLoraAdapters**(): `Promise`<{ `path`: `string` ; `scaled?`: `number`  }[]\>

#### Returns

`Promise`<{ `path`: `string` ; `scaled?`: `number`  }[]\>

#### Defined in

[index.ts:918](https://github.com/mybigday/llama.rn/blob/8dd6e05/src/index.ts#L918)

___

### getMultimodalSupport

▸ **getMultimodalSupport**(): `Promise`<{ `audio`: `boolean` ; `vision`: `boolean`  }\>

Check multimodal support

#### Returns

`Promise`<{ `audio`: `boolean` ; `vision`: `boolean`  }\>

Promise resolving to an object with vision and audio support

#### Defined in

[index.ts:957](https://github.com/mybigday/llama.rn/blob/8dd6e05/src/index.ts#L957)

___

### initMultimodal

▸ **initMultimodal**(`params`): `Promise`<`boolean`\>

Initialize multimodal support with a mmproj file

#### Parameters

| Name | Type | Description |
| :------ | :------ | :------ |
| `params` | `Object` | Parameters for multimodal support |
| `params.path` | `string` | Path to the multimodal projector file |
| `params.use_gpu?` | `boolean` | Whether to use GPU |

#### Returns

`Promise`<`boolean`\>

Promise resolving to true if initialization was successful

#### Defined in

[index.ts:931](https://github.com/mybigday/llama.rn/blob/8dd6e05/src/index.ts#L931)

___

### initVocoder

▸ **initVocoder**(`params`): `Promise`<`boolean`\>

Initialize TTS support with a vocoder model

#### Parameters

| Name | Type | Description |
| :------ | :------ | :------ |
| `params` | `Object` | Parameters for TTS support |
| `params.n_batch?` | `number` | Batch size for the vocoder model |
| `params.path` | `string` | Path to the vocoder model |

#### Returns

`Promise`<`boolean`\>

Promise resolving to true if initialization was successful

#### Defined in

[index.ts:979](https://github.com/mybigday/llama.rn/blob/8dd6e05/src/index.ts#L979)

___

### isJinjaSupported

▸ **isJinjaSupported**(): `boolean`

#### Returns

`boolean`

#### Defined in

[index.ts:600](https://github.com/mybigday/llama.rn/blob/8dd6e05/src/index.ts#L600)

___

### isLlamaChatSupported

▸ **isLlamaChatSupported**(): `boolean`

#### Returns

`boolean`

#### Defined in

[index.ts:596](https://github.com/mybigday/llama.rn/blob/8dd6e05/src/index.ts#L596)

___

### isMultimodalEnabled

▸ **isMultimodalEnabled**(): `Promise`<`boolean`\>

Check if multimodal support is enabled

#### Returns

`Promise`<`boolean`\>

Promise resolving to true if multimodal is enabled

#### Defined in

[index.ts:949](https://github.com/mybigday/llama.rn/blob/8dd6e05/src/index.ts#L949)

___

### isVocoderEnabled

▸ **isVocoderEnabled**(): `Promise`<`boolean`\>

Check if TTS support is enabled

#### Returns

`Promise`<`boolean`\>

Promise resolving to true if TTS is enabled

#### Defined in

[index.ts:994](https://github.com/mybigday/llama.rn/blob/8dd6e05/src/index.ts#L994)

___

### loadSession

▸ **loadSession**(`filepath`): `Promise`<[`NativeSessionLoadResult`](../README.md#nativesessionloadresult)\>

Load cached prompt & completion state from a file.

#### Parameters

| Name | Type |
| :------ | :------ |
| `filepath` | `string` |

#### Returns

`Promise`<[`NativeSessionLoadResult`](../README.md#nativesessionloadresult)\>

#### Defined in

[index.ts:580](https://github.com/mybigday/llama.rn/blob/8dd6e05/src/index.ts#L580)

___

### release

▸ **release**(): `Promise`<`void`\>

#### Returns

`Promise`<`void`\>

#### Defined in

[index.ts:1046](https://github.com/mybigday/llama.rn/blob/8dd6e05/src/index.ts#L1046)

___

### releaseMultimodal

▸ **releaseMultimodal**(): `Promise`<`void`\>

Release multimodal support

#### Returns

`Promise`<`void`\>

Promise resolving to void

#### Defined in

[index.ts:968](https://github.com/mybigday/llama.rn/blob/8dd6e05/src/index.ts#L968)

___

### releaseVocoder

▸ **releaseVocoder**(): `Promise`<`void`\>

Release TTS support

#### Returns

`Promise`<`void`\>

Promise resolving to void

#### Defined in

[index.ts:1042](https://github.com/mybigday/llama.rn/blob/8dd6e05/src/index.ts#L1042)

___

### removeLoraAdapters

▸ **removeLoraAdapters**(): `Promise`<`void`\>

#### Returns

`Promise`<`void`\>

#### Defined in

[index.ts:914](https://github.com/mybigday/llama.rn/blob/8dd6e05/src/index.ts#L914)

___

### rerank

▸ **rerank**(`query`, `documents`, `params?`): `Promise`<[`RerankResult`](../README.md#rerankresult)[]\>

Rerank documents based on relevance to a query

#### Parameters

| Name | Type | Description |
| :------ | :------ | :------ |
| `query` | `string` | The query text to rank documents against |
| `documents` | `string`[] | Array of document texts to rank |
| `params?` | [`RerankParams`](../README.md#rerankparams) | Optional reranking parameters |

#### Returns

`Promise`<[`RerankResult`](../README.md#rerankresult)[]\>

Promise resolving to an array of ranking results with scores and indices

#### Defined in

[index.ts:851](https://github.com/mybigday/llama.rn/blob/8dd6e05/src/index.ts#L851)

___

### saveSession

▸ **saveSession**(`filepath`, `options?`): `Promise`<`number`\>

Save current cached prompt & completion state to a file.

#### Parameters

| Name | Type |
| :------ | :------ |
| `filepath` | `string` |
| `options?` | `Object` |
| `options.tokenSize` | `number` |

#### Returns

`Promise`<`number`\>

#### Defined in

[index.ts:589](https://github.com/mybigday/llama.rn/blob/8dd6e05/src/index.ts#L589)

___

### stopCompletion

▸ **stopCompletion**(): `Promise`<`void`\>

#### Returns

`Promise`<`void`\>

#### Defined in

[index.ts:812](https://github.com/mybigday/llama.rn/blob/8dd6e05/src/index.ts#L812)

___

### tokenize

▸ **tokenize**(`text`, `«destructured»?`): `Promise`<[`NativeTokenizeResult`](../README.md#nativetokenizeresult)\>

Tokenize text or text with images

#### Parameters

| Name | Type | Description |
| :------ | :------ | :------ |
| `text` | `string` | Text to tokenize |
| `«destructured»` | `Object` | - |
| › `media_paths?` | `string`[] | - |

#### Returns

`Promise`<[`NativeTokenizeResult`](../README.md#nativetokenizeresult)\>

Promise resolving to the tokenize result

#### Defined in

[index.ts:822](https://github.com/mybigday/llama.rn/blob/8dd6e05/src/index.ts#L822)
