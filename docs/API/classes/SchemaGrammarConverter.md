[llama.rn](../README.md) / SchemaGrammarConverter

# Class: SchemaGrammarConverter

## Table of contents

### Constructors

- [constructor](SchemaGrammarConverter.md#constructor)

### Properties

- [\_propOrder](SchemaGrammarConverter.md#_proporder)
- [\_rules](SchemaGrammarConverter.md#_rules)

### Methods

- [addRule](SchemaGrammarConverter.md#addrule)
- [formatGrammar](SchemaGrammarConverter.md#formatgrammar)
- [visit](SchemaGrammarConverter.md#visit)

## Constructors

### constructor

• **new SchemaGrammarConverter**(`propOrder?`)

#### Parameters

| Name | Type |
| :------ | :------ |
| `propOrder?` | `PropOrder` |

#### Defined in

[grammar.ts:39](https://github.com/mybigday/llama.rn/blob/e3e9f86/src/grammar.ts#L39)

## Properties

### \_propOrder

• `Private` **\_propOrder**: `PropOrder`

#### Defined in

[grammar.ts:35](https://github.com/mybigday/llama.rn/blob/e3e9f86/src/grammar.ts#L35)

___

### \_rules

• `Private` **\_rules**: `Map`<`string`, `string`\>

#### Defined in

[grammar.ts:37](https://github.com/mybigday/llama.rn/blob/e3e9f86/src/grammar.ts#L37)

## Methods

### addRule

▸ `Private` **addRule**(`name`, `rule`): `string`

#### Parameters

| Name | Type |
| :------ | :------ |
| `name` | `string` |
| `rule` | `string` |

#### Returns

`string`

#### Defined in

[grammar.ts:45](https://github.com/mybigday/llama.rn/blob/e3e9f86/src/grammar.ts#L45)

___

### formatGrammar

▸ **formatGrammar**(): `string`

#### Returns

`string`

#### Defined in

[grammar.ts:125](https://github.com/mybigday/llama.rn/blob/e3e9f86/src/grammar.ts#L125)

___

### visit

▸ **visit**(`schema`, `name?`): `string`

#### Parameters

| Name | Type |
| :------ | :------ |
| `schema` | `any` |
| `name?` | `string` |

#### Returns

`string`

#### Defined in

[grammar.ts:65](https://github.com/mybigday/llama.rn/blob/e3e9f86/src/grammar.ts#L65)
