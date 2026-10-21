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

[index.ts:414](https://github.com/mybigday/llama.rn/blob/6f75ba4d/src/index.ts#L414)

## Properties

### baked

• **baked**: `boolean`

#### Defined in

[index.ts:410](https://github.com/mybigday/llama.rn/blob/6f75ba4d/src/index.ts#L410)

___

### ctxId

• `Private` **ctxId**: `number`

#### Defined in

[index.ts:412](https://github.com/mybigday/llama.rn/blob/6f75ba4d/src/index.ts#L412)

___

### family

• `Readonly` **family**: `string`

#### Defined in

[index.ts:406](https://github.com/mybigday/llama.rn/blob/6f75ba4d/src/index.ts#L406)

___

### id

• `Readonly` **id**: `number`

#### Defined in

[index.ts:404](https://github.com/mybigday/llama.rn/blob/6f75ba4d/src/index.ts#L404)

___

### rows

• **rows**: `number`

#### Defined in

[index.ts:408](https://github.com/mybigday/llama.rn/blob/6f75ba4d/src/index.ts#L408)

## Methods

### bake

▸ **bake**(): `Promise`<`void`\>

#### Returns

`Promise`<`void`\>

#### Defined in

[index.ts:422](https://github.com/mybigday/llama.rn/blob/6f75ba4d/src/index.ts#L422)

___

### release

▸ **release**(): `Promise`<`void`\>

#### Returns

`Promise`<`void`\>

#### Defined in

[index.ts:429](https://github.com/mybigday/llama.rn/blob/6f75ba4d/src/index.ts#L429)
