[llama.rn](../README.md) / LlamaContext

# Class: LlamaContext

## Table of contents

### Constructors

- [constructor](LlamaContext.md#constructor)

### Properties

- [androidLib](LlamaContext.md#androidlib)
- [gpu](LlamaContext.md#gpu)
- [gpuDevice](LlamaContext.md#gpudevice)
- [id](LlamaContext.md#id)
- [model](LlamaContext.md#model)
- [reasonNoGPU](LlamaContext.md#reasonnogpu)

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

[index.ts:247](https://github.com/mybigday/llama.rn/blob/abf6288/src/index.ts#L247)

## Properties

### androidLib

• **androidLib**: `undefined` \| `string`

#### Defined in

[index.ts:245](https://github.com/mybigday/llama.rn/blob/abf6288/src/index.ts#L245)

___

### gpu

• **gpu**: `boolean` = `false`

#### Defined in

[index.ts:237](https://github.com/mybigday/llama.rn/blob/abf6288/src/index.ts#L237)

___

### gpuDevice

• **gpuDevice**: `undefined` \| `string`

#### Defined in

[index.ts:239](https://github.com/mybigday/llama.rn/blob/abf6288/src/index.ts#L239)

___

### id

• **id**: `number`

#### Defined in

[index.ts:235](https://github.com/mybigday/llama.rn/blob/abf6288/src/index.ts#L235)

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

[index.ts:243](https://github.com/mybigday/llama.rn/blob/abf6288/src/index.ts#L243)

___

### reasonNoGPU

• **reasonNoGPU**: `string` = `''`

#### Defined in

[index.ts:241](https://github.com/mybigday/llama.rn/blob/abf6288/src/index.ts#L241)

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

[index.ts:560](https://github.com/mybigday/llama.rn/blob/abf6288/src/index.ts#L560)

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

[index.ts:540](https://github.com/mybigday/llama.rn/blob/abf6288/src/index.ts#L540)

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

[index.ts:395](https://github.com/mybigday/llama.rn/blob/abf6288/src/index.ts#L395)

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

[index.ts:686](https://github.com/mybigday/llama.rn/blob/abf6288/src/index.ts#L686)

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

[index.ts:506](https://github.com/mybigday/llama.rn/blob/abf6288/src/index.ts#L506)

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

[index.ts:510](https://github.com/mybigday/llama.rn/blob/abf6288/src/index.ts#L510)

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

[index.ts:675](https://github.com/mybigday/llama.rn/blob/abf6288/src/index.ts#L675)

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

[index.ts:656](https://github.com/mybigday/llama.rn/blob/abf6288/src/index.ts#L656)

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

[index.ts:284](https://github.com/mybigday/llama.rn/blob/abf6288/src/index.ts#L284)

___

### getLoadedLoraAdapters

▸ **getLoadedLoraAdapters**(): `Promise`<{ `path`: `string` ; `scaled?`: `number`  }[]\>

#### Returns

`Promise`<{ `path`: `string` ; `scaled?`: `number`  }[]\>

#### Defined in

[index.ts:576](https://github.com/mybigday/llama.rn/blob/abf6288/src/index.ts#L576)

___

### getMultimodalSupport

▸ **getMultimodalSupport**(): `Promise`<{ `audio`: `boolean` ; `vision`: `boolean`  }\>

Check multimodal support

#### Returns

`Promise`<{ `audio`: `boolean` ; `vision`: `boolean`  }\>

Promise resolving to an object with vision and audio support

#### Defined in

[index.ts:615](https://github.com/mybigday/llama.rn/blob/abf6288/src/index.ts#L615)

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

[index.ts:589](https://github.com/mybigday/llama.rn/blob/abf6288/src/index.ts#L589)

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

[index.ts:637](https://github.com/mybigday/llama.rn/blob/abf6288/src/index.ts#L637)

___

### isJinjaSupported

▸ **isJinjaSupported**(): `boolean`

#### Returns

`boolean`

#### Defined in

[index.ts:279](https://github.com/mybigday/llama.rn/blob/abf6288/src/index.ts#L279)

___

### isLlamaChatSupported

▸ **isLlamaChatSupported**(): `boolean`

#### Returns

`boolean`

#### Defined in

[index.ts:275](https://github.com/mybigday/llama.rn/blob/abf6288/src/index.ts#L275)

___

### isMultimodalEnabled

▸ **isMultimodalEnabled**(): `Promise`<`boolean`\>

Check if multimodal support is enabled

#### Returns

`Promise`<`boolean`\>

Promise resolving to true if multimodal is enabled

#### Defined in

[index.ts:607](https://github.com/mybigday/llama.rn/blob/abf6288/src/index.ts#L607)

___

### isVocoderEnabled

▸ **isVocoderEnabled**(): `Promise`<`boolean`\>

Check if TTS support is enabled

#### Returns

`Promise`<`boolean`\>

Promise resolving to true if TTS is enabled

#### Defined in

[index.ts:646](https://github.com/mybigday/llama.rn/blob/abf6288/src/index.ts#L646)

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

[index.ts:259](https://github.com/mybigday/llama.rn/blob/abf6288/src/index.ts#L259)

___

### release

▸ **release**(): `Promise`<`void`\>

#### Returns

`Promise`<`void`\>

#### Defined in

[index.ts:698](https://github.com/mybigday/llama.rn/blob/abf6288/src/index.ts#L698)

___

### releaseMultimodal

▸ **releaseMultimodal**(): `Promise`<`void`\>

Release multimodal support

#### Returns

`Promise`<`void`\>

Promise resolving to void

#### Defined in

[index.ts:626](https://github.com/mybigday/llama.rn/blob/abf6288/src/index.ts#L626)

___

### releaseVocoder

▸ **releaseVocoder**(): `Promise`<`void`\>

Release TTS support

#### Returns

`Promise`<`void`\>

Promise resolving to void

#### Defined in

[index.ts:694](https://github.com/mybigday/llama.rn/blob/abf6288/src/index.ts#L694)

___

### removeLoraAdapters

▸ **removeLoraAdapters**(): `Promise`<`void`\>

#### Returns

`Promise`<`void`\>

#### Defined in

[index.ts:572](https://github.com/mybigday/llama.rn/blob/abf6288/src/index.ts#L572)

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

[index.ts:524](https://github.com/mybigday/llama.rn/blob/abf6288/src/index.ts#L524)

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

[index.ts:268](https://github.com/mybigday/llama.rn/blob/abf6288/src/index.ts#L268)

___

### stopCompletion

▸ **stopCompletion**(): `Promise`<`void`\>

#### Returns

`Promise`<`void`\>

#### Defined in

[index.ts:485](https://github.com/mybigday/llama.rn/blob/abf6288/src/index.ts#L485)

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

[index.ts:495](https://github.com/mybigday/llama.rn/blob/abf6288/src/index.ts#L495)
