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

[index.ts:49](https://github.com/mybigday/llama.rn/blob/8738c99/src/index.ts#L49)

## Properties

### gpu

• **gpu**: `boolean` = `false`

#### Defined in

[index.ts:45](https://github.com/mybigday/llama.rn/blob/8738c99/src/index.ts#L45)

___

### id

• **id**: `number`

#### Defined in

[index.ts:43](https://github.com/mybigday/llama.rn/blob/8738c99/src/index.ts#L43)

___

### reasonNoGPU

• **reasonNoGPU**: `string` = `''`

#### Defined in

[index.ts:47](https://github.com/mybigday/llama.rn/blob/8738c99/src/index.ts#L47)

## Methods

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

[index.ts:73](https://github.com/mybigday/llama.rn/blob/8738c99/src/index.ts#L73)

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

[index.ts:110](https://github.com/mybigday/llama.rn/blob/8738c99/src/index.ts#L110)

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

[index.ts:114](https://github.com/mybigday/llama.rn/blob/8738c99/src/index.ts#L114)

___

### loadSession

▸ **loadSession**(`filepath`): `Promise`<`number`\>

Load cached prompt & completion state from a file.

#### Parameters

| Name | Type |
| :------ | :------ |
| `filepath` | `string` |

#### Returns

`Promise`<`number`\>

#### Defined in

[index.ts:62](https://github.com/mybigday/llama.rn/blob/8738c99/src/index.ts#L62)

___

### release

▸ **release**(): `Promise`<`void`\>

#### Returns

`Promise`<`void`\>

#### Defined in

[index.ts:118](https://github.com/mybigday/llama.rn/blob/8738c99/src/index.ts#L118)

___

### saveSession

▸ **saveSession**(`filepath`): `Promise`<`number`\>

Save current cached prompt & completion state to a file.

#### Parameters

| Name | Type |
| :------ | :------ |
| `filepath` | `string` |

#### Returns

`Promise`<`number`\>

#### Defined in

[index.ts:69](https://github.com/mybigday/llama.rn/blob/8738c99/src/index.ts#L69)

___

### stopCompletion

▸ **stopCompletion**(): `Promise`<`void`\>

#### Returns

`Promise`<`void`\>

#### Defined in

[index.ts:102](https://github.com/mybigday/llama.rn/blob/8738c99/src/index.ts#L102)

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

[index.ts:106](https://github.com/mybigday/llama.rn/blob/8738c99/src/index.ts#L106)
