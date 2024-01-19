[llama.rn](../README.md) / LlamaContext

# Class: LlamaContext

## Table of contents

### Constructors

- [constructor](LlamaContext.md#constructor)

### Properties

- [gpu](LlamaContext.md#gpu)
- [id](LlamaContext.md#id)
- [reasonNoGPU](LlamaContext.md#reasonnogpu)

### Methods

- [bench](LlamaContext.md#bench)
- [completion](LlamaContext.md#completion)
- [detokenize](LlamaContext.md#detokenize)
- [embedding](LlamaContext.md#embedding)
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

[index.ts:60](https://github.com/mybigday/llama.rn/blob/e3e9f86/src/index.ts#L60)

## Properties

### gpu

• **gpu**: `boolean` = `false`

#### Defined in

[index.ts:56](https://github.com/mybigday/llama.rn/blob/e3e9f86/src/index.ts#L56)

___

### id

• **id**: `number`

#### Defined in

[index.ts:54](https://github.com/mybigday/llama.rn/blob/e3e9f86/src/index.ts#L54)

___

### reasonNoGPU

• **reasonNoGPU**: `string` = `''`

#### Defined in

[index.ts:58](https://github.com/mybigday/llama.rn/blob/e3e9f86/src/index.ts#L58)

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

[index.ts:129](https://github.com/mybigday/llama.rn/blob/e3e9f86/src/index.ts#L129)

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

[index.ts:84](https://github.com/mybigday/llama.rn/blob/e3e9f86/src/index.ts#L84)

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

[index.ts:121](https://github.com/mybigday/llama.rn/blob/e3e9f86/src/index.ts#L121)

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

[index.ts:125](https://github.com/mybigday/llama.rn/blob/e3e9f86/src/index.ts#L125)

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

[index.ts:73](https://github.com/mybigday/llama.rn/blob/e3e9f86/src/index.ts#L73)

___

### release

▸ **release**(): `Promise`<`void`\>

#### Returns

`Promise`<`void`\>

#### Defined in

[index.ts:151](https://github.com/mybigday/llama.rn/blob/e3e9f86/src/index.ts#L151)

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

[index.ts:80](https://github.com/mybigday/llama.rn/blob/e3e9f86/src/index.ts#L80)

___

### stopCompletion

▸ **stopCompletion**(): `Promise`<`void`\>

#### Returns

`Promise`<`void`\>

#### Defined in

[index.ts:113](https://github.com/mybigday/llama.rn/blob/e3e9f86/src/index.ts#L113)

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

[index.ts:117](https://github.com/mybigday/llama.rn/blob/e3e9f86/src/index.ts#L117)
