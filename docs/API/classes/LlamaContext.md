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

- [bench](LlamaContext.md#bench)
- [completion](LlamaContext.md#completion)
- [detokenize](LlamaContext.md#detokenize)
- [embedding](LlamaContext.md#embedding)
- [getFormattedChat](LlamaContext.md#getformattedchat)
- [loadSession](LlamaContext.md#loadsession)
- [release](LlamaContext.md#release)
- [saveSession](LlamaContext.md#savesession)
- [stopCompletion](LlamaContext.md#stopcompletion)
- [tokenize](LlamaContext.md#tokenize)

## Constructors

### constructor

• **new LlamaContext**(`«destructured»`)

#### Parameters

| Name | Type |
| :------ | :------ |
| `«destructured»` | `NativeLlamaContext` |

#### Defined in

[index.ts:79](https://github.com/mybigday/llama.rn/blob/38fa660/src/index.ts#L79)

## Properties

### gpu

• **gpu**: `boolean` = `false`

#### Defined in

[index.ts:71](https://github.com/mybigday/llama.rn/blob/38fa660/src/index.ts#L71)

___

### id

• **id**: `number`

#### Defined in

[index.ts:69](https://github.com/mybigday/llama.rn/blob/38fa660/src/index.ts#L69)

___

### model

• **model**: `Object` = `{}`

#### Type declaration

| Name | Type |
| :------ | :------ |
| `isChatTemplateSupported?` | `boolean` |

#### Defined in

[index.ts:75](https://github.com/mybigday/llama.rn/blob/38fa660/src/index.ts#L75)

___

### reasonNoGPU

• **reasonNoGPU**: `string` = `''`

#### Defined in

[index.ts:73](https://github.com/mybigday/llama.rn/blob/38fa660/src/index.ts#L73)

## Methods

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

[index.ts:171](https://github.com/mybigday/llama.rn/blob/38fa660/src/index.ts#L171)

___

### completion

▸ **completion**(`params`, `callback?`): `Promise`<`NativeCompletionResult`\>

#### Parameters

| Name | Type |
| :------ | :------ |
| `params` | [`CompletionParams`](../README.md#completionparams) |
| `callback?` | (`data`: [`TokenData`](../README.md#tokendata)) => `void` |

#### Returns

`Promise`<`NativeCompletionResult`\>

#### Defined in

[index.ts:115](https://github.com/mybigday/llama.rn/blob/38fa660/src/index.ts#L115)

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

[index.ts:160](https://github.com/mybigday/llama.rn/blob/38fa660/src/index.ts#L160)

___

### embedding

▸ **embedding**(`text`, `params?`): `Promise`<`NativeEmbeddingResult`\>

#### Parameters

| Name | Type |
| :------ | :------ |
| `text` | `string` |
| `params?` | `NativeEmbeddingParams` |

#### Returns

`Promise`<`NativeEmbeddingResult`\>

#### Defined in

[index.ts:164](https://github.com/mybigday/llama.rn/blob/38fa660/src/index.ts#L164)

___

### getFormattedChat

▸ **getFormattedChat**(`messages`, `template?`): `Promise`<`string`\>

#### Parameters

| Name | Type |
| :------ | :------ |
| `messages` | `RNLlamaOAICompatibleMessage`[] |
| `template?` | `string` |

#### Returns

`Promise`<`string`\>

#### Defined in

[index.ts:105](https://github.com/mybigday/llama.rn/blob/38fa660/src/index.ts#L105)

___

### loadSession

▸ **loadSession**(`filepath`): `Promise`<`NativeSessionLoadResult`\>

Load cached prompt & completion state from a file.

#### Parameters

| Name | Type |
| :------ | :------ |
| `filepath` | `string` |

#### Returns

`Promise`<`NativeSessionLoadResult`\>

#### Defined in

[index.ts:89](https://github.com/mybigday/llama.rn/blob/38fa660/src/index.ts#L89)

___

### release

▸ **release**(): `Promise`<`void`\>

#### Returns

`Promise`<`void`\>

#### Defined in

[index.ts:191](https://github.com/mybigday/llama.rn/blob/38fa660/src/index.ts#L191)

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

[index.ts:98](https://github.com/mybigday/llama.rn/blob/38fa660/src/index.ts#L98)

___

### stopCompletion

▸ **stopCompletion**(): `Promise`<`void`\>

#### Returns

`Promise`<`void`\>

#### Defined in

[index.ts:152](https://github.com/mybigday/llama.rn/blob/38fa660/src/index.ts#L152)

___

### tokenize

▸ **tokenize**(`text`): `Promise`<`NativeTokenizeResult`\>

#### Parameters

| Name | Type |
| :------ | :------ |
| `text` | `string` |

#### Returns

`Promise`<`NativeTokenizeResult`\>

#### Defined in

[index.ts:156](https://github.com/mybigday/llama.rn/blob/38fa660/src/index.ts#L156)
