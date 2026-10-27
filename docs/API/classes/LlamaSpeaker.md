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

[index.ts:405](https://github.com/mybigday/llama.rn/blob/7ce96e4b/src/index.ts#L405)

## Properties

### baked

• **baked**: `boolean`

#### Defined in

[index.ts:401](https://github.com/mybigday/llama.rn/blob/7ce96e4b/src/index.ts#L401)

___

### ctxId

• `Private` **ctxId**: `number`

#### Defined in

[index.ts:403](https://github.com/mybigday/llama.rn/blob/7ce96e4b/src/index.ts#L403)

___

### family

• `Readonly` **family**: `string`

#### Defined in

[index.ts:397](https://github.com/mybigday/llama.rn/blob/7ce96e4b/src/index.ts#L397)

___

### id

• `Readonly` **id**: `number`

#### Defined in

[index.ts:395](https://github.com/mybigday/llama.rn/blob/7ce96e4b/src/index.ts#L395)

___

### rows

• **rows**: `number`

#### Defined in

[index.ts:399](https://github.com/mybigday/llama.rn/blob/7ce96e4b/src/index.ts#L399)

## Methods

### bake

▸ **bake**(): `Promise`<`void`\>

#### Returns

`Promise`<`void`\>

#### Defined in

[index.ts:413](https://github.com/mybigday/llama.rn/blob/7ce96e4b/src/index.ts#L413)

___

### release

▸ **release**(): `Promise`<`void`\>

#### Returns

`Promise`<`void`\>

#### Defined in

[index.ts:420](https://github.com/mybigday/llama.rn/blob/7ce96e4b/src/index.ts#L420)
