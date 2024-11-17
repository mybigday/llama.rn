llama.rn

# llama.rn

## Table of contents

### Classes

- [LlamaContext](classes/LlamaContext.md)
- [SchemaGrammarConverter](classes/SchemaGrammarConverter.md)

### Type Aliases

- [BenchResult](README.md#benchresult)
- [CompletionParams](README.md#completionparams)
- [ContextParams](README.md#contextparams)
- [EmbeddingParams](README.md#embeddingparams)
- [TokenData](README.md#tokendata)

### Functions

- [convertJsonSchemaToGrammar](README.md#convertjsonschematogrammar)
- [initLlama](README.md#initllama)
- [loadLlamaModelInfo](README.md#loadllamamodelinfo)
- [releaseAllLlama](README.md#releaseallllama)
- [setContextLimit](README.md#setcontextlimit)

## Type Aliases

### BenchResult

Ƭ **BenchResult**: `Object`

#### Type declaration

| Name | Type |
| :------ | :------ |
| `modelDesc` | `string` |
| `modelNParams` | `number` |
| `modelSize` | `number` |
| `ppAvg` | `number` |
| `ppStd` | `number` |
| `tgAvg` | `number` |
| `tgStd` | `number` |

#### Defined in

[index.ts:63](https://github.com/mybigday/llama.rn/blob/8e54cbb/src/index.ts#L63)

___

### CompletionParams

Ƭ **CompletionParams**: `Omit`<`NativeCompletionParams`, ``"emit_partial_completion"`` \| ``"prompt"``\> & { `chatTemplate?`: `string` ; `messages?`: `RNLlamaOAICompatibleMessage`[] ; `prompt?`: `string`  }

#### Defined in

[index.ts:54](https://github.com/mybigday/llama.rn/blob/8e54cbb/src/index.ts#L54)

___

### ContextParams

Ƭ **ContextParams**: `Omit`<`NativeContextParams`, ``"cache_type_k"`` \| ``"cache_type_v"`` \| ``"pooling_type"``\> & { `cache_type_k?`: ``"f16"`` \| ``"f32"`` \| ``"q8_0"`` \| ``"q4_0"`` \| ``"q4_1"`` \| ``"iq4_nl"`` \| ``"q5_0"`` \| ``"q5_1"`` ; `cache_type_v?`: ``"f16"`` \| ``"f32"`` \| ``"q8_0"`` \| ``"q4_0"`` \| ``"q4_1"`` \| ``"iq4_nl"`` \| ``"q5_0"`` \| ``"q5_1"`` ; `pooling_type?`: ``"none"`` \| ``"mean"`` \| ``"cls"`` \| ``"last"`` \| ``"rank"``  }

#### Defined in

[index.ts:43](https://github.com/mybigday/llama.rn/blob/8e54cbb/src/index.ts#L43)

___

### EmbeddingParams

Ƭ **EmbeddingParams**: `NativeEmbeddingParams`

#### Defined in

[index.ts:52](https://github.com/mybigday/llama.rn/blob/8e54cbb/src/index.ts#L52)

___

### TokenData

Ƭ **TokenData**: `Object`

#### Type declaration

| Name | Type |
| :------ | :------ |
| `completion_probabilities?` | `NativeCompletionTokenProb`[] |
| `token` | `string` |

#### Defined in

[index.ts:33](https://github.com/mybigday/llama.rn/blob/8e54cbb/src/index.ts#L33)

## Functions

### convertJsonSchemaToGrammar

▸ **convertJsonSchemaToGrammar**(`«destructured»`): `string` \| `Promise`<`string`\>

#### Parameters

| Name | Type |
| :------ | :------ |
| `«destructured»` | `Object` |
| › `allowFetch?` | `boolean` |
| › `dotall?` | `boolean` |
| › `propOrder?` | `PropOrder` |
| › `schema` | `any` |

#### Returns

`string` \| `Promise`<`string`\>

#### Defined in

[grammar.ts:824](https://github.com/mybigday/llama.rn/blob/8e54cbb/src/grammar.ts#L824)

___

### initLlama

▸ **initLlama**(`«destructured»`, `onProgress?`): `Promise`<[`LlamaContext`](classes/LlamaContext.md)\>

#### Parameters

| Name | Type |
| :------ | :------ |
| `«destructured»` | [`ContextParams`](README.md#contextparams) |
| `onProgress?` | (`progress`: `number`) => `void` |

#### Returns

`Promise`<[`LlamaContext`](classes/LlamaContext.md)\>

#### Defined in

[index.ts:230](https://github.com/mybigday/llama.rn/blob/8e54cbb/src/index.ts#L230)

___

### loadLlamaModelInfo

▸ **loadLlamaModelInfo**(`model`): `Promise`<`Object`\>

#### Parameters

| Name | Type |
| :------ | :------ |
| `model` | `string` |

#### Returns

`Promise`<`Object`\>

#### Defined in

[index.ts:215](https://github.com/mybigday/llama.rn/blob/8e54cbb/src/index.ts#L215)

___

### releaseAllLlama

▸ **releaseAllLlama**(): `Promise`<`void`\>

#### Returns

`Promise`<`void`\>

#### Defined in

[index.ts:280](https://github.com/mybigday/llama.rn/blob/8e54cbb/src/index.ts#L280)

___

### setContextLimit

▸ **setContextLimit**(`limit`): `Promise`<`void`\>

#### Parameters

| Name | Type |
| :------ | :------ |
| `limit` | `number` |

#### Returns

`Promise`<`void`\>

#### Defined in

[index.ts:201](https://github.com/mybigday/llama.rn/blob/8e54cbb/src/index.ts#L201)
