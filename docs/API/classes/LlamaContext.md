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

[index.ts:72](https://github.com/mybigday/llama.rn/blob/ad7e0a5/src/index.ts#L72)

## Properties

### gpu

• **gpu**: `boolean` = `false`

#### Defined in

[index.ts:64](https://github.com/mybigday/llama.rn/blob/ad7e0a5/src/index.ts#L64)

___

### id

• **id**: `number`

#### Defined in

[index.ts:62](https://github.com/mybigday/llama.rn/blob/ad7e0a5/src/index.ts#L62)

___

### model

• **model**: `Object` = `{}`

#### Type declaration

| Name | Type |
| :------ | :------ |
| `isChatTemplateSupported?` | `boolean` |

#### Defined in

[index.ts:68](https://github.com/mybigday/llama.rn/blob/ad7e0a5/src/index.ts#L68)

___

### reasonNoGPU

• **reasonNoGPU**: `string` = `''`

#### Defined in

[index.ts:66](https://github.com/mybigday/llama.rn/blob/ad7e0a5/src/index.ts#L66)

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

[index.ts:162](https://github.com/mybigday/llama.rn/blob/ad7e0a5/src/index.ts#L162)

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

[index.ts:109](https://github.com/mybigday/llama.rn/blob/ad7e0a5/src/index.ts#L109)

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

[index.ts:154](https://github.com/mybigday/llama.rn/blob/ad7e0a5/src/index.ts#L154)

___

### embedding

▸ **embedding**(`text`): `Promise`<`NativeEmbeddingResult`\>

#### Parameters

| Name | Type |
| :------ | :------ |
| `text` | `string` |

#### Returns

`Promise`<`NativeEmbeddingResult`\>

#### Defined in

[index.ts:158](https://github.com/mybigday/llama.rn/blob/ad7e0a5/src/index.ts#L158)

___

### getFormattedChat

▸ **getFormattedChat**(`messages`): `Promise`<`string`\>

#### Parameters

| Name | Type |
| :------ | :------ |
| `messages` | `RNLlamaOAICompatibleMessage`[] |

#### Returns

`Promise`<`string`\>

#### Defined in

[index.ts:98](https://github.com/mybigday/llama.rn/blob/ad7e0a5/src/index.ts#L98)

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

[index.ts:82](https://github.com/mybigday/llama.rn/blob/ad7e0a5/src/index.ts#L82)

___

### release

▸ **release**(): `Promise`<`void`\>

#### Returns

`Promise`<`void`\>

#### Defined in

[index.ts:182](https://github.com/mybigday/llama.rn/blob/ad7e0a5/src/index.ts#L182)

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

[index.ts:91](https://github.com/mybigday/llama.rn/blob/ad7e0a5/src/index.ts#L91)

___

### stopCompletion

▸ **stopCompletion**(): `Promise`<`void`\>

#### Returns

`Promise`<`void`\>

#### Defined in

[index.ts:146](https://github.com/mybigday/llama.rn/blob/ad7e0a5/src/index.ts#L146)

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

[index.ts:150](https://github.com/mybigday/llama.rn/blob/ad7e0a5/src/index.ts#L150)
