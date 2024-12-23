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

[index.ts:124](https://github.com/mybigday/llama.rn/blob/402a590/src/index.ts#L124)

## Properties

### gpu

• **gpu**: `boolean` = `false`

#### Defined in

[index.ts:116](https://github.com/mybigday/llama.rn/blob/402a590/src/index.ts#L116)

___

### id

• **id**: `number`

#### Defined in

[index.ts:114](https://github.com/mybigday/llama.rn/blob/402a590/src/index.ts#L114)

___

### model

• **model**: `Object` = `{}`

#### Type declaration

| Name | Type |
| :------ | :------ |
| `isChatTemplateSupported?` | `boolean` |

#### Defined in

[index.ts:120](https://github.com/mybigday/llama.rn/blob/402a590/src/index.ts#L120)

___

### reasonNoGPU

• **reasonNoGPU**: `string` = `''`

#### Defined in

[index.ts:118](https://github.com/mybigday/llama.rn/blob/402a590/src/index.ts#L118)

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

[index.ts:239](https://github.com/mybigday/llama.rn/blob/402a590/src/index.ts#L239)

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

[index.ts:219](https://github.com/mybigday/llama.rn/blob/402a590/src/index.ts#L219)

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

[index.ts:160](https://github.com/mybigday/llama.rn/blob/402a590/src/index.ts#L160)

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

[index.ts:208](https://github.com/mybigday/llama.rn/blob/402a590/src/index.ts#L208)

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

[index.ts:212](https://github.com/mybigday/llama.rn/blob/402a590/src/index.ts#L212)

___

### getFormattedChat

▸ **getFormattedChat**(`messages`, `template?`): `Promise`<`string`\>

#### Parameters

| Name | Type |
| :------ | :------ |
| `messages` | [`RNLlamaOAICompatibleMessage`](../README.md#rnllamaoaicompatiblemessage)[] |
| `template?` | `string` |

#### Returns

`Promise`<`string`\>

#### Defined in

[index.ts:150](https://github.com/mybigday/llama.rn/blob/402a590/src/index.ts#L150)

___

### getLoadedLoraAdapters

▸ **getLoadedLoraAdapters**(): `Promise`<{ `path`: `string` ; `scaled?`: `number`  }[]\>

#### Returns

`Promise`<{ `path`: `string` ; `scaled?`: `number`  }[]\>

#### Defined in

[index.ts:255](https://github.com/mybigday/llama.rn/blob/402a590/src/index.ts#L255)

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

[index.ts:134](https://github.com/mybigday/llama.rn/blob/402a590/src/index.ts#L134)

___

### release

▸ **release**(): `Promise`<`void`\>

#### Returns

`Promise`<`void`\>

#### Defined in

[index.ts:261](https://github.com/mybigday/llama.rn/blob/402a590/src/index.ts#L261)

___

### removeLoraAdapters

▸ **removeLoraAdapters**(): `Promise`<`void`\>

#### Returns

`Promise`<`void`\>

#### Defined in

[index.ts:251](https://github.com/mybigday/llama.rn/blob/402a590/src/index.ts#L251)

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

[index.ts:143](https://github.com/mybigday/llama.rn/blob/402a590/src/index.ts#L143)

___

### stopCompletion

▸ **stopCompletion**(): `Promise`<`void`\>

#### Returns

`Promise`<`void`\>

#### Defined in

[index.ts:200](https://github.com/mybigday/llama.rn/blob/402a590/src/index.ts#L200)

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

[index.ts:204](https://github.com/mybigday/llama.rn/blob/402a590/src/index.ts#L204)
