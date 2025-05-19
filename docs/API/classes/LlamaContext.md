[llama.rn](../README.md) / LlamaContext

# Class: LlamaContext

## Table of contents

### Constructors

- [constructor](LlamaContext.md#constructor)

### Properties

- [gpu](LlamaContext.md#gpu)
- [id](LlamaContext.md#id)
- [model](LlamaContext.md#model)
- [reasonNoGPU](LlamaContext.md#reasonnogpu)

### Methods

- [applyLoraAdapters](LlamaContext.md#applyloraadapters)
- [bench](LlamaContext.md#bench)
- [completion](LlamaContext.md#completion)
- [detokenize](LlamaContext.md#detokenize)
- [embedding](LlamaContext.md#embedding)
- [getFormattedChat](LlamaContext.md#getformattedchat)
- [getLoadedLoraAdapters](LlamaContext.md#getloadedloraadapters)
- [initMultimodal](LlamaContext.md#initmultimodal)
- [isJinjaSupported](LlamaContext.md#isjinjasupported)
- [isLlamaChatSupported](LlamaContext.md#isllamachatsupported)
- [isMultimodalEnabled](LlamaContext.md#ismultimodalenabled)
- [loadSession](LlamaContext.md#loadsession)
- [release](LlamaContext.md#release)
- [releaseMultimodal](LlamaContext.md#releasemultimodal)
- [removeLoraAdapters](LlamaContext.md#removeloraadapters)
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

[index.ts:173](https://github.com/mybigday/llama.rn/blob/67ab8f4/src/index.ts#L173)

## Properties

### gpu

• **gpu**: `boolean` = `false`

#### Defined in

[index.ts:167](https://github.com/mybigday/llama.rn/blob/67ab8f4/src/index.ts#L167)

___

### id

• **id**: `number`

#### Defined in

[index.ts:165](https://github.com/mybigday/llama.rn/blob/67ab8f4/src/index.ts#L165)

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

[index.ts:171](https://github.com/mybigday/llama.rn/blob/67ab8f4/src/index.ts#L171)

___

### reasonNoGPU

• **reasonNoGPU**: `string` = `''`

#### Defined in

[index.ts:169](https://github.com/mybigday/llama.rn/blob/67ab8f4/src/index.ts#L169)

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

[index.ts:416](https://github.com/mybigday/llama.rn/blob/67ab8f4/src/index.ts#L416)

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

[index.ts:396](https://github.com/mybigday/llama.rn/blob/67ab8f4/src/index.ts#L396)

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

Note: For multimodal support, you can include an image_paths parameter.
This will process the images and add them to the context before generating text.
Multimodal support must be enabled via initMultimodal() first.

#### Defined in

[index.ts:290](https://github.com/mybigday/llama.rn/blob/67ab8f4/src/index.ts#L290)

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

[index.ts:385](https://github.com/mybigday/llama.rn/blob/67ab8f4/src/index.ts#L385)

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

[index.ts:389](https://github.com/mybigday/llama.rn/blob/67ab8f4/src/index.ts#L389)

___

### getFormattedChat

▸ **getFormattedChat**(`messages`, `template?`, `params?`): `Promise`<[`FormattedChatResult`](../README.md#formattedchatresult)\>

#### Parameters

| Name | Type |
| :------ | :------ |
| `messages` | [`RNLlamaOAICompatibleMessage`](../README.md#rnllamaoaicompatiblemessage)[] |
| `template?` | ``null`` \| `string` |
| `params?` | `Object` |
| `params.jinja?` | `boolean` |
| `params.parallel_tool_calls?` | `object` |
| `params.response_format?` | [`CompletionResponseFormat`](../README.md#completionresponseformat) |
| `params.tool_choice?` | `string` |
| `params.tools?` | `object` |

#### Returns

`Promise`<[`FormattedChatResult`](../README.md#formattedchatresult)\>

#### Defined in

[index.ts:208](https://github.com/mybigday/llama.rn/blob/67ab8f4/src/index.ts#L208)

___

### getLoadedLoraAdapters

▸ **getLoadedLoraAdapters**(): `Promise`<{ `path`: `string` ; `scaled?`: `number`  }[]\>

#### Returns

`Promise`<{ `path`: `string` ; `scaled?`: `number`  }[]\>

#### Defined in

[index.ts:432](https://github.com/mybigday/llama.rn/blob/67ab8f4/src/index.ts#L432)

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

[index.ts:445](https://github.com/mybigday/llama.rn/blob/67ab8f4/src/index.ts#L445)

___

### isJinjaSupported

▸ **isJinjaSupported**(): `boolean`

#### Returns

`boolean`

#### Defined in

[index.ts:203](https://github.com/mybigday/llama.rn/blob/67ab8f4/src/index.ts#L203)

___

### isLlamaChatSupported

▸ **isLlamaChatSupported**(): `boolean`

#### Returns

`boolean`

#### Defined in

[index.ts:199](https://github.com/mybigday/llama.rn/blob/67ab8f4/src/index.ts#L199)

___

### isMultimodalEnabled

▸ **isMultimodalEnabled**(): `Promise`<`boolean`\>

Check if multimodal support is enabled

#### Returns

`Promise`<`boolean`\>

Promise resolving to true if multimodal is enabled

#### Defined in

[index.ts:459](https://github.com/mybigday/llama.rn/blob/67ab8f4/src/index.ts#L459)

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

[index.ts:183](https://github.com/mybigday/llama.rn/blob/67ab8f4/src/index.ts#L183)

___

### release

▸ **release**(): `Promise`<`void`\>

#### Returns

`Promise`<`void`\>

#### Defined in

[index.ts:471](https://github.com/mybigday/llama.rn/blob/67ab8f4/src/index.ts#L471)

___

### releaseMultimodal

▸ **releaseMultimodal**(): `Promise`<`void`\>

Release multimodal support

#### Returns

`Promise`<`void`\>

Promise resolving to void

#### Defined in

[index.ts:467](https://github.com/mybigday/llama.rn/blob/67ab8f4/src/index.ts#L467)

___

### removeLoraAdapters

▸ **removeLoraAdapters**(): `Promise`<`void`\>

#### Returns

`Promise`<`void`\>

#### Defined in

[index.ts:428](https://github.com/mybigday/llama.rn/blob/67ab8f4/src/index.ts#L428)

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

[index.ts:192](https://github.com/mybigday/llama.rn/blob/67ab8f4/src/index.ts#L192)

___

### stopCompletion

▸ **stopCompletion**(): `Promise`<`void`\>

#### Returns

`Promise`<`void`\>

#### Defined in

[index.ts:377](https://github.com/mybigday/llama.rn/blob/67ab8f4/src/index.ts#L377)

___

### tokenize

▸ **tokenize**(`text`): `Promise`<[`NativeTokenizeResult`](../README.md#nativetokenizeresult)\>

#### Parameters

| Name | Type |
| :------ | :------ |
| `text` | `string` |

#### Returns

`Promise`<[`NativeTokenizeResult`](../README.md#nativetokenizeresult)\>

#### Defined in

[index.ts:381](https://github.com/mybigday/llama.rn/blob/67ab8f4/src/index.ts#L381)
