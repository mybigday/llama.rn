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
- [TokenData](README.md#tokendata)

### Functions

- [convertJsonSchemaToGrammar](README.md#convertjsonschematogrammar)
- [initLlama](README.md#initllama)
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

[index.ts:51](https://github.com/mybigday/llama.rn/blob/ad7e0a5/src/index.ts#L51)

___

### CompletionParams

Ƭ **CompletionParams**: `Omit`<`NativeCompletionParams`, ``"emit_partial_completion"`` \| ``"prompt"``\> & { `messages?`: `RNLlamaOAICompatibleMessage`[] ; `prompt?`: `string`  }

#### Defined in

[index.ts:43](https://github.com/mybigday/llama.rn/blob/ad7e0a5/src/index.ts#L43)

___

### ContextParams

Ƭ **ContextParams**: `NativeContextParams`

#### Defined in

[index.ts:41](https://github.com/mybigday/llama.rn/blob/ad7e0a5/src/index.ts#L41)

___

### TokenData

Ƭ **TokenData**: `Object`

#### Type declaration

| Name | Type |
| :------ | :------ |
| `completion_probabilities?` | `NativeCompletionTokenProb`[] |
| `token` | `string` |

#### Defined in

[index.ts:31](https://github.com/mybigday/llama.rn/blob/ad7e0a5/src/index.ts#L31)

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

[grammar.ts:824](https://github.com/mybigday/llama.rn/blob/ad7e0a5/src/grammar.ts#L824)

___

### initLlama

▸ **initLlama**(`«destructured»`): `Promise`<[`LlamaContext`](classes/LlamaContext.md)\>

#### Parameters

| Name | Type |
| :------ | :------ |
| `«destructured»` | `NativeContextParams` |

#### Returns

`Promise`<[`LlamaContext`](classes/LlamaContext.md)\>

#### Defined in

[index.ts:191](https://github.com/mybigday/llama.rn/blob/ad7e0a5/src/index.ts#L191)

___

### releaseAllLlama

▸ **releaseAllLlama**(): `Promise`<`void`\>

#### Returns

`Promise`<`void`\>

#### Defined in

[index.ts:211](https://github.com/mybigday/llama.rn/blob/ad7e0a5/src/index.ts#L211)

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

[index.ts:187](https://github.com/mybigday/llama.rn/blob/ad7e0a5/src/index.ts#L187)
