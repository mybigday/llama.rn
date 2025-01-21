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

[grammar.ts:213](https://github.com/mybigday/llama.rn/blob/402a590/src/grammar.ts#L213)

## Properties

### \_allowFetch

• `Private` **\_allowFetch**: `boolean`

#### Defined in

[grammar.ts:203](https://github.com/mybigday/llama.rn/blob/402a590/src/grammar.ts#L203)

___

### \_dotall

• `Private` **\_dotall**: `boolean`

#### Defined in

[grammar.ts:205](https://github.com/mybigday/llama.rn/blob/402a590/src/grammar.ts#L205)

___

### \_propOrder

• `Private` **\_propOrder**: [`SchemaGrammarConverterPropOrder`](../interfaces/SchemaGrammarConverterPropOrder.md)

#### Defined in

[grammar.ts:201](https://github.com/mybigday/llama.rn/blob/402a590/src/grammar.ts#L201)

___

### \_refs

• `Private` **\_refs**: `Object`

#### Index signature

▪ [key: `string`]: `any`

#### Defined in

[grammar.ts:209](https://github.com/mybigday/llama.rn/blob/402a590/src/grammar.ts#L209)

___

### \_refsBeingResolved

• `Private` **\_refsBeingResolved**: `Set`<`string`\>

#### Defined in

[grammar.ts:211](https://github.com/mybigday/llama.rn/blob/402a590/src/grammar.ts#L211)

___

### \_rules

• `Private` **\_rules**: `Object`

#### Index signature

▪ [key: `string`]: `string`

#### Defined in

[grammar.ts:207](https://github.com/mybigday/llama.rn/blob/402a590/src/grammar.ts#L207)

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

[grammar.ts:695](https://github.com/mybigday/llama.rn/blob/402a590/src/grammar.ts#L695)

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

[grammar.ts:226](https://github.com/mybigday/llama.rn/blob/402a590/src/grammar.ts#L226)

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

[grammar.ts:712](https://github.com/mybigday/llama.rn/blob/402a590/src/grammar.ts#L712)

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

[grammar.ts:314](https://github.com/mybigday/llama.rn/blob/402a590/src/grammar.ts#L314)

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

[grammar.ts:520](https://github.com/mybigday/llama.rn/blob/402a590/src/grammar.ts#L520)

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

[grammar.ts:325](https://github.com/mybigday/llama.rn/blob/402a590/src/grammar.ts#L325)

___

### formatGrammar

▸ **formatGrammar**(): `string`

#### Returns

`string`

#### Defined in

[grammar.ts:815](https://github.com/mybigday/llama.rn/blob/402a590/src/grammar.ts#L815)

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

[grammar.ts:249](https://github.com/mybigday/llama.rn/blob/402a590/src/grammar.ts#L249)

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

[grammar.ts:531](https://github.com/mybigday/llama.rn/blob/402a590/src/grammar.ts#L531)
