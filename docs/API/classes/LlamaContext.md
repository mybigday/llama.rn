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
- [isJinjaSupported](LlamaContext.md#isjinjasupported)
- [isLlamaChatSupported](LlamaContext.md#isllamachatsupported)
- [loadSession](LlamaContext.md#loadsession)
- [release](LlamaContext.md#release)
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

[index.ts:167](https://github.com/mybigday/llama.rn/blob/64b02c4/src/index.ts#L167)

## Properties

### gpu

• **gpu**: `boolean` = `false`

#### Defined in

[index.ts:161](https://github.com/mybigday/llama.rn/blob/64b02c4/src/index.ts#L161)

___

### id

• **id**: `number`

#### Defined in

[index.ts:159](https://github.com/mybigday/llama.rn/blob/64b02c4/src/index.ts#L159)

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

[index.ts:165](https://github.com/mybigday/llama.rn/blob/64b02c4/src/index.ts#L165)

___

### reasonNoGPU

• **reasonNoGPU**: `string` = `''`

#### Defined in

[index.ts:163](https://github.com/mybigday/llama.rn/blob/64b02c4/src/index.ts#L163)

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

[index.ts:341](https://github.com/mybigday/llama.rn/blob/64b02c4/src/index.ts#L341)

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

[index.ts:321](https://github.com/mybigday/llama.rn/blob/64b02c4/src/index.ts#L321)

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

[index.ts:229](https://github.com/mybigday/llama.rn/blob/64b02c4/src/index.ts#L229)

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

[index.ts:310](https://github.com/mybigday/llama.rn/blob/64b02c4/src/index.ts#L310)

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

[index.ts:314](https://github.com/mybigday/llama.rn/blob/64b02c4/src/index.ts#L314)

___

### getFormattedChat

▸ **getFormattedChat**(`messages`, `template?`, `params?`): `Promise`<`string` \| [`JinjaFormattedChatResult`](../README.md#jinjaformattedchatresult)\>

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

`Promise`<`string` \| [`JinjaFormattedChatResult`](../README.md#jinjaformattedchatresult)\>

#### Defined in

[index.ts:202](https://github.com/mybigday/llama.rn/blob/64b02c4/src/index.ts#L202)

___

### getLoadedLoraAdapters

▸ **getLoadedLoraAdapters**(): `Promise`<{ `path`: `string` ; `scaled?`: `number`  }[]\>

#### Returns

`Promise`<{ `path`: `string` ; `scaled?`: `number`  }[]\>

#### Defined in

[index.ts:357](https://github.com/mybigday/llama.rn/blob/64b02c4/src/index.ts#L357)

___

### isJinjaSupported

▸ **isJinjaSupported**(): `boolean`

#### Returns

`boolean`

#### Defined in

[index.ts:197](https://github.com/mybigday/llama.rn/blob/64b02c4/src/index.ts#L197)

___

### isLlamaChatSupported

▸ **isLlamaChatSupported**(): `boolean`

#### Returns

`boolean`

#### Defined in

[index.ts:193](https://github.com/mybigday/llama.rn/blob/64b02c4/src/index.ts#L193)

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

[index.ts:177](https://github.com/mybigday/llama.rn/blob/64b02c4/src/index.ts#L177)

___

### release

▸ **release**(): `Promise`<`void`\>

#### Returns

`Promise`<`void`\>

#### Defined in

[index.ts:363](https://github.com/mybigday/llama.rn/blob/64b02c4/src/index.ts#L363)

___

### removeLoraAdapters

▸ **removeLoraAdapters**(): `Promise`<`void`\>

#### Returns

`Promise`<`void`\>

#### Defined in

[index.ts:353](https://github.com/mybigday/llama.rn/blob/64b02c4/src/index.ts#L353)

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

[index.ts:186](https://github.com/mybigday/llama.rn/blob/64b02c4/src/index.ts#L186)

___

### stopCompletion

▸ **stopCompletion**(): `Promise`<`void`\>

#### Returns

`Promise`<`void`\>

#### Defined in

[index.ts:302](https://github.com/mybigday/llama.rn/blob/64b02c4/src/index.ts#L302)

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

[index.ts:306](https://github.com/mybigday/llama.rn/blob/64b02c4/src/index.ts#L306)
