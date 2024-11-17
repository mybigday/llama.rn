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

[index.ts:84](https://github.com/mybigday/llama.rn/blob/8e54cbb/src/index.ts#L84)

## Properties

### gpu

• **gpu**: `boolean` = `false`

#### Defined in

[index.ts:76](https://github.com/mybigday/llama.rn/blob/8e54cbb/src/index.ts#L76)

___

### id

• **id**: `number`

#### Defined in

[index.ts:74](https://github.com/mybigday/llama.rn/blob/8e54cbb/src/index.ts#L74)

___

### model

• **model**: `Object` = `{}`

#### Type declaration

| Name | Type |
| :------ | :------ |
| `isChatTemplateSupported?` | `boolean` |

#### Defined in

[index.ts:80](https://github.com/mybigday/llama.rn/blob/8e54cbb/src/index.ts#L80)

___

### reasonNoGPU

• **reasonNoGPU**: `string` = `''`

#### Defined in

[index.ts:78](https://github.com/mybigday/llama.rn/blob/8e54cbb/src/index.ts#L78)

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

[index.ts:176](https://github.com/mybigday/llama.rn/blob/8e54cbb/src/index.ts#L176)

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

[index.ts:120](https://github.com/mybigday/llama.rn/blob/8e54cbb/src/index.ts#L120)

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

[index.ts:165](https://github.com/mybigday/llama.rn/blob/8e54cbb/src/index.ts#L165)

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

[index.ts:169](https://github.com/mybigday/llama.rn/blob/8e54cbb/src/index.ts#L169)

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

[index.ts:110](https://github.com/mybigday/llama.rn/blob/8e54cbb/src/index.ts#L110)

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

[index.ts:94](https://github.com/mybigday/llama.rn/blob/8e54cbb/src/index.ts#L94)

___

### release

▸ **release**(): `Promise`<`void`\>

#### Returns

`Promise`<`void`\>

#### Defined in

[index.ts:196](https://github.com/mybigday/llama.rn/blob/8e54cbb/src/index.ts#L196)

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

[index.ts:103](https://github.com/mybigday/llama.rn/blob/8e54cbb/src/index.ts#L103)

___

### stopCompletion

▸ **stopCompletion**(): `Promise`<`void`\>

#### Returns

`Promise`<`void`\>

#### Defined in

[index.ts:157](https://github.com/mybigday/llama.rn/blob/8e54cbb/src/index.ts#L157)

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

[index.ts:161](https://github.com/mybigday/llama.rn/blob/8e54cbb/src/index.ts#L161)
