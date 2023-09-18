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
- [release](LlamaContext.md#release)
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

[index.ts:49](https://github.com/mybigday/llama.rn/blob/50235c2/src/index.ts#L49)

## Properties

### gpu

• **gpu**: `boolean` = `false`

#### Defined in

[index.ts:45](https://github.com/mybigday/llama.rn/blob/50235c2/src/index.ts#L45)

___

### id

• **id**: `number`

#### Defined in

[index.ts:43](https://github.com/mybigday/llama.rn/blob/50235c2/src/index.ts#L43)

___

### reasonNoGPU

• **reasonNoGPU**: `string` = `''`

#### Defined in

[index.ts:47](https://github.com/mybigday/llama.rn/blob/50235c2/src/index.ts#L47)

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

[index.ts:59](https://github.com/mybigday/llama.rn/blob/50235c2/src/index.ts#L59)

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

[index.ts:96](https://github.com/mybigday/llama.rn/blob/50235c2/src/index.ts#L96)

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

[index.ts:100](https://github.com/mybigday/llama.rn/blob/50235c2/src/index.ts#L100)

___

### release

▸ **release**(): `Promise`<`void`\>

#### Returns

`Promise`<`void`\>

#### Defined in

[index.ts:104](https://github.com/mybigday/llama.rn/blob/50235c2/src/index.ts#L104)

___

### stopCompletion

▸ **stopCompletion**(): `Promise`<`void`\>

#### Returns

`Promise`<`void`\>

#### Defined in

[index.ts:88](https://github.com/mybigday/llama.rn/blob/50235c2/src/index.ts#L88)

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

[index.ts:92](https://github.com/mybigday/llama.rn/blob/50235c2/src/index.ts#L92)
