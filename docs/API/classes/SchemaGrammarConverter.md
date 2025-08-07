[llama.rn](../README.md) / SchemaGrammarConverter

# Class: SchemaGrammarConverter

## Table of contents

### Constructors

- [constructor](SchemaGrammarConverter.md#constructor)

### Properties

- [\_allowFetch](SchemaGrammarConverter.md#_allowfetch)
- [\_dotall](SchemaGrammarConverter.md#_dotall)
- [\_propOrder](SchemaGrammarConverter.md#_proporder)
- [\_refs](SchemaGrammarConverter.md#_refs)
- [\_refsBeingResolved](SchemaGrammarConverter.md#_refsbeingresolved)
- [\_rules](SchemaGrammarConverter.md#_rules)

### Methods

- [\_addPrimitive](SchemaGrammarConverter.md#_addprimitive)
- [\_addRule](SchemaGrammarConverter.md#_addrule)
- [\_buildObjectRule](SchemaGrammarConverter.md#_buildobjectrule)
- [\_generateUnionRule](SchemaGrammarConverter.md#_generateunionrule)
- [\_resolveRef](SchemaGrammarConverter.md#_resolveref)
- [\_visitPattern](SchemaGrammarConverter.md#_visitpattern)
- [formatGrammar](SchemaGrammarConverter.md#formatgrammar)
- [resolveRefs](SchemaGrammarConverter.md#resolverefs)
- [visit](SchemaGrammarConverter.md#visit)

## Constructors

### constructor

• **new SchemaGrammarConverter**(`options`)

#### Parameters

| Name | Type |
| :------ | :------ |
| `options` | `Object` |
| `options.allow_fetch?` | `boolean` |
| `options.dotall?` | `boolean` |
| `options.prop_order?` | [`SchemaGrammarConverterPropOrder`](../interfaces/SchemaGrammarConverterPropOrder.md) |

#### Defined in

[grammar.ts:216](https://github.com/mybigday/llama.rn/blob/1571b49/src/grammar.ts#L216)

## Properties

### \_allowFetch

• `Private` **\_allowFetch**: `boolean`

#### Defined in

[grammar.ts:206](https://github.com/mybigday/llama.rn/blob/1571b49/src/grammar.ts#L206)

___

### \_dotall

• `Private` **\_dotall**: `boolean`

#### Defined in

[grammar.ts:208](https://github.com/mybigday/llama.rn/blob/1571b49/src/grammar.ts#L208)

___

### \_propOrder

• `Private` **\_propOrder**: [`SchemaGrammarConverterPropOrder`](../interfaces/SchemaGrammarConverterPropOrder.md)

#### Defined in

[grammar.ts:204](https://github.com/mybigday/llama.rn/blob/1571b49/src/grammar.ts#L204)

___

### \_refs

• `Private` **\_refs**: `Object`

#### Index signature

▪ [key: `string`]: `any`

#### Defined in

[grammar.ts:212](https://github.com/mybigday/llama.rn/blob/1571b49/src/grammar.ts#L212)

___

### \_refsBeingResolved

• `Private` **\_refsBeingResolved**: `Set`<`string`\>

#### Defined in

[grammar.ts:214](https://github.com/mybigday/llama.rn/blob/1571b49/src/grammar.ts#L214)

___

### \_rules

• `Private` **\_rules**: `Object`

#### Index signature

▪ [key: `string`]: `string`

#### Defined in

[grammar.ts:210](https://github.com/mybigday/llama.rn/blob/1571b49/src/grammar.ts#L210)

## Methods

### \_addPrimitive

▸ **_addPrimitive**(`name`, `rule`): `string`

#### Parameters

| Name | Type |
| :------ | :------ |
| `name` | `string` |
| `rule` | `undefined` \| [`SchemaGrammarConverterBuiltinRule`](SchemaGrammarConverterBuiltinRule.md) |

#### Returns

`string`

#### Defined in

[grammar.ts:698](https://github.com/mybigday/llama.rn/blob/1571b49/src/grammar.ts#L698)

___

### \_addRule

▸ **_addRule**(`name`, `rule`): `string`

#### Parameters

| Name | Type |
| :------ | :------ |
| `name` | `string` |
| `rule` | `string` |

#### Returns

`string`

#### Defined in

[grammar.ts:229](https://github.com/mybigday/llama.rn/blob/1571b49/src/grammar.ts#L229)

___

### \_buildObjectRule

▸ **_buildObjectRule**(`properties`, `required`, `name`, `additionalProperties`): `string`

#### Parameters

| Name | Type |
| :------ | :------ |
| `properties` | `any`[] |
| `required` | `Set`<`string`\> |
| `name` | `string` |
| `additionalProperties` | `any` |

#### Returns

`string`

#### Defined in

[grammar.ts:715](https://github.com/mybigday/llama.rn/blob/1571b49/src/grammar.ts#L715)

___

### \_generateUnionRule

▸ **_generateUnionRule**(`name`, `altSchemas`): `string`

#### Parameters

| Name | Type |
| :------ | :------ |
| `name` | `string` |
| `altSchemas` | `any`[] |

#### Returns

`string`

#### Defined in

[grammar.ts:317](https://github.com/mybigday/llama.rn/blob/1571b49/src/grammar.ts#L317)

___

### \_resolveRef

▸ **_resolveRef**(`ref`): `string`

#### Parameters

| Name | Type |
| :------ | :------ |
| `ref` | `string` |

#### Returns

`string`

#### Defined in

[grammar.ts:523](https://github.com/mybigday/llama.rn/blob/1571b49/src/grammar.ts#L523)

___

### \_visitPattern

▸ **_visitPattern**(`pattern`, `name`): `string`

#### Parameters

| Name | Type |
| :------ | :------ |
| `pattern` | `string` |
| `name` | `string` |

#### Returns

`string`

#### Defined in

[grammar.ts:328](https://github.com/mybigday/llama.rn/blob/1571b49/src/grammar.ts#L328)

___

### formatGrammar

▸ **formatGrammar**(): `string`

#### Returns

`string`

#### Defined in

[grammar.ts:818](https://github.com/mybigday/llama.rn/blob/1571b49/src/grammar.ts#L818)

___

### resolveRefs

▸ **resolveRefs**(`schema`, `url`): `Promise`<`any`\>

#### Parameters

| Name | Type |
| :------ | :------ |
| `schema` | `any` |
| `url` | `string` |

#### Returns

`Promise`<`any`\>

#### Defined in

[grammar.ts:252](https://github.com/mybigday/llama.rn/blob/1571b49/src/grammar.ts#L252)

___

### visit

▸ **visit**(`schema`, `name`): `string`

#### Parameters

| Name | Type |
| :------ | :------ |
| `schema` | `any` |
| `name` | `string` |

#### Returns

`string`

#### Defined in

[grammar.ts:534](https://github.com/mybigday/llama.rn/blob/1571b49/src/grammar.ts#L534)
