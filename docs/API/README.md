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

[index.ts:57](https://github.com/mybigday/llama.rn/blob/20a1819/src/index.ts#L57)

___

### CompletionParams

Ƭ **CompletionParams**: `Omit`<`NativeCompletionParams`, ``"emit_partial_completion"`` \| ``"prompt"``\> & { `messages?`: `RNLlamaOAICompatibleMessage`[] ; `prompt?`: `string`  }

#### Defined in

[index.ts:49](https://github.com/mybigday/llama.rn/blob/20a1819/src/index.ts#L49)

___

### ContextParams

Ƭ **ContextParams**: `Omit`<`NativeContextParams`, ``"pooling_type"``\> & { `pooling_type?`: ``"none"`` \| ``"mean"`` \| ``"cls"`` \| ``"last"`` \| ``"rank"``  }

#### Defined in

[index.ts:43](https://github.com/mybigday/llama.rn/blob/20a1819/src/index.ts#L43)

___

### EmbeddingParams

Ƭ **EmbeddingParams**: `NativeEmbeddingParams`

#### Defined in

[index.ts:47](https://github.com/mybigday/llama.rn/blob/20a1819/src/index.ts#L47)

___

### TokenData

Ƭ **TokenData**: `Object`

#### Type declaration

| Name | Type |
| :------ | :------ |
| `completion_probabilities?` | `NativeCompletionTokenProb`[] |
| `token` | `string` |

#### Defined in

[index.ts:33](https://github.com/mybigday/llama.rn/blob/20a1819/src/index.ts#L33)

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

[grammar.ts:824](https://github.com/mybigday/llama.rn/blob/20a1819/src/grammar.ts#L824)

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

[index.ts:225](https://github.com/mybigday/llama.rn/blob/20a1819/src/index.ts#L225)

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

[index.ts:210](https://github.com/mybigday/llama.rn/blob/20a1819/src/index.ts#L210)

___

### releaseAllLlama

▸ **releaseAllLlama**(): `Promise`<`void`\>

#### Returns

`Promise`<`void`\>

#### Defined in

[index.ts:269](https://github.com/mybigday/llama.rn/blob/20a1819/src/index.ts#L269)

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

[index.ts:196](https://github.com/mybigday/llama.rn/blob/20a1819/src/index.ts#L196)
