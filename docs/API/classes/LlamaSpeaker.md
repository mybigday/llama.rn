[llama.rn](../README.md) / LlamaSpeaker

# Class: LlamaSpeaker

## Table of contents

### Constructors

- [constructor](LlamaSpeaker.md#constructor)

### Properties

- [baked](LlamaSpeaker.md#baked)
- [ctxId](LlamaSpeaker.md#ctxid)
- [family](LlamaSpeaker.md#family)
- [id](LlamaSpeaker.md#id)
- [rows](LlamaSpeaker.md#rows)

### Methods

- [bake](LlamaSpeaker.md#bake)
- [release](LlamaSpeaker.md#release)

## Constructors

### constructor

• **new LlamaSpeaker**(`ctxId`, `h`)

#### Parameters

| Name | Type |
| :------ | :------ |
| `ctxId` | `number` |
| `h` | `Object` |
| `h.baked` | `boolean` |
| `h.family` | `string` |
| `h.id` | `number` |
| `h.rows` | `number` |

#### Defined in

[index.ts:411](https://github.com/mybigday/llama.rn/blob/3ee457d5/src/index.ts#L411)

## Properties

### baked

• **baked**: `boolean`

#### Defined in

[index.ts:407](https://github.com/mybigday/llama.rn/blob/3ee457d5/src/index.ts#L407)

___

### ctxId

• `Private` **ctxId**: `number`

#### Defined in

[index.ts:409](https://github.com/mybigday/llama.rn/blob/3ee457d5/src/index.ts#L409)

___

### family

• `Readonly` **family**: `string`

#### Defined in

[index.ts:403](https://github.com/mybigday/llama.rn/blob/3ee457d5/src/index.ts#L403)

___

### id

• `Readonly` **id**: `number`

#### Defined in

[index.ts:401](https://github.com/mybigday/llama.rn/blob/3ee457d5/src/index.ts#L401)

___

### rows

• **rows**: `number`

#### Defined in

[index.ts:405](https://github.com/mybigday/llama.rn/blob/3ee457d5/src/index.ts#L405)

## Methods

### bake

▸ **bake**(): `Promise`<`void`\>

#### Returns

`Promise`<`void`\>

#### Defined in

[index.ts:419](https://github.com/mybigday/llama.rn/blob/3ee457d5/src/index.ts#L419)

___

### release

▸ **release**(): `Promise`<`void`\>

#### Returns

`Promise`<`void`\>

#### Defined in

[index.ts:426](https://github.com/mybigday/llama.rn/blob/3ee457d5/src/index.ts#L426)
