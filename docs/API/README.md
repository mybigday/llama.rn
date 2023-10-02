llama.rn

# llama.rn

## Table of contents

### Classes

- [LlamaContext](classes/LlamaContext.md)
- [SchemaGrammarConverter](classes/SchemaGrammarConverter.md)

### Type Aliases

- [CompletionParams](README.md#completionparams)
- [ContextParams](README.md#contextparams)
- [TokenData](README.md#tokendata)

### Functions

- [convertJsonSchemaToGrammar](README.md#convertjsonschematogrammar)
- [initLlama](README.md#initllama)
- [releaseAllLlama](README.md#releaseallllama)
- [setContextLimit](README.md#setcontextlimit)

## Type Aliases

### CompletionParams

Ƭ **CompletionParams**: `Omit`<`NativeCompletionParams`, ``"emit_partial_completion"``\>

#### Defined in

[index.ts:40](https://github.com/mybigday/llama.rn/blob/acfc7ab/src/index.ts#L40)

___

### ContextParams

Ƭ **ContextParams**: `NativeContextParams`

#### Defined in

[index.ts:38](https://github.com/mybigday/llama.rn/blob/acfc7ab/src/index.ts#L38)

___

### TokenData

Ƭ **TokenData**: `Object`

#### Type declaration

| Name | Type |
| :------ | :------ |
| `completion_probabilities?` | `NativeCompletionTokenProb`[] |
| `token` | `string` |

#### Defined in

[index.ts:28](https://github.com/mybigday/llama.rn/blob/acfc7ab/src/index.ts#L28)

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

[grammar.ts:134](https://github.com/mybigday/llama.rn/blob/acfc7ab/src/grammar.ts#L134)

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

[index.ts:113](https://github.com/mybigday/llama.rn/blob/acfc7ab/src/index.ts#L113)

___

### releaseAllLlama

▸ **releaseAllLlama**(): `Promise`<`void`\>

#### Returns

`Promise`<`void`\>

#### Defined in

[index.ts:129](https://github.com/mybigday/llama.rn/blob/acfc7ab/src/index.ts#L129)

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

[index.ts:109](https://github.com/mybigday/llama.rn/blob/acfc7ab/src/index.ts#L109)
