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

[index.ts:73](https://github.com/mybigday/llama.rn/blob/cb19020/src/index.ts#L73)

## Properties

### gpu

• **gpu**: `boolean` = `false`

#### Defined in

[index.ts:65](https://github.com/mybigday/llama.rn/blob/cb19020/src/index.ts#L65)

___

### id

• **id**: `number`

#### Defined in

[index.ts:63](https://github.com/mybigday/llama.rn/blob/cb19020/src/index.ts#L63)

___

### model

• **model**: `Object` = `{}`

#### Type declaration

| Name | Type |
| :------ | :------ |
| `isChatTemplateSupported?` | `boolean` |

#### Defined in

[index.ts:69](https://github.com/mybigday/llama.rn/blob/cb19020/src/index.ts#L69)

___

### reasonNoGPU

• **reasonNoGPU**: `string` = `''`

#### Defined in

[index.ts:67](https://github.com/mybigday/llama.rn/blob/cb19020/src/index.ts#L67)

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

[index.ts:163](https://github.com/mybigday/llama.rn/blob/cb19020/src/index.ts#L163)

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

[index.ts:110](https://github.com/mybigday/llama.rn/blob/cb19020/src/index.ts#L110)

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

[index.ts:155](https://github.com/mybigday/llama.rn/blob/cb19020/src/index.ts#L155)

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

[index.ts:159](https://github.com/mybigday/llama.rn/blob/cb19020/src/index.ts#L159)

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

[index.ts:99](https://github.com/mybigday/llama.rn/blob/cb19020/src/index.ts#L99)

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

[index.ts:83](https://github.com/mybigday/llama.rn/blob/cb19020/src/index.ts#L83)

___

### release

▸ **release**(): `Promise`<`void`\>

#### Returns

`Promise`<`void`\>

#### Defined in

[index.ts:183](https://github.com/mybigday/llama.rn/blob/cb19020/src/index.ts#L183)

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

[index.ts:92](https://github.com/mybigday/llama.rn/blob/cb19020/src/index.ts#L92)

___

### stopCompletion

▸ **stopCompletion**(): `Promise`<`void`\>

#### Returns

`Promise`<`void`\>

#### Defined in

[index.ts:147](https://github.com/mybigday/llama.rn/blob/cb19020/src/index.ts#L147)

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

[index.ts:151](https://github.com/mybigday/llama.rn/blob/cb19020/src/index.ts#L151)
