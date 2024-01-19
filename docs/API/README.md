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

[index.ts:43](https://github.com/mybigday/llama.rn/blob/e3e9f86/src/index.ts#L43)

___

### CompletionParams

Ƭ **CompletionParams**: `Omit`<`NativeCompletionParams`, ``"emit_partial_completion"``\>

#### Defined in

[index.ts:41](https://github.com/mybigday/llama.rn/blob/e3e9f86/src/index.ts#L41)

___

### ContextParams

Ƭ **ContextParams**: `NativeContextParams`

#### Defined in

[index.ts:39](https://github.com/mybigday/llama.rn/blob/e3e9f86/src/index.ts#L39)

___

### TokenData

Ƭ **TokenData**: `Object`

#### Type declaration

| Name | Type |
| :------ | :------ |
| `completion_probabilities?` | `NativeCompletionTokenProb`[] |
| `token` | `string` |

#### Defined in

[index.ts:29](https://github.com/mybigday/llama.rn/blob/e3e9f86/src/index.ts#L29)

## Functions

### convertJsonSchemaToGrammar

▸ **convertJsonSchemaToGrammar**(`«destructured»`): `string`

#### Parameters

| Name | Type |
| :------ | :------ |
| `«destructured»` | `Object` |
| › `propOrder?` | `PropOrder` |
| › `schema` | `any` |

#### Returns

`string`

#### Defined in

[grammar.ts:134](https://github.com/mybigday/llama.rn/blob/e3e9f86/src/grammar.ts#L134)

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

[index.ts:160](https://github.com/mybigday/llama.rn/blob/e3e9f86/src/index.ts#L160)

___

### releaseAllLlama

▸ **releaseAllLlama**(): `Promise`<`void`\>

#### Returns

`Promise`<`void`\>

#### Defined in

[index.ts:176](https://github.com/mybigday/llama.rn/blob/e3e9f86/src/index.ts#L176)

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

[index.ts:156](https://github.com/mybigday/llama.rn/blob/e3e9f86/src/index.ts#L156)
