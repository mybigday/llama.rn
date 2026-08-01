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

[index.ts:374](https://github.com/mybigday/llama.rn/blob/a8c8191/src/index.ts#L374)

## Properties

### baked

• **baked**: `boolean`

#### Defined in

[index.ts:370](https://github.com/mybigday/llama.rn/blob/a8c8191/src/index.ts#L370)

___

### ctxId

• `Private` **ctxId**: `number`

#### Defined in

[index.ts:372](https://github.com/mybigday/llama.rn/blob/a8c8191/src/index.ts#L372)

___

### family

• `Readonly` **family**: `string`

#### Defined in

[index.ts:366](https://github.com/mybigday/llama.rn/blob/a8c8191/src/index.ts#L366)

___

### id

• `Readonly` **id**: `number`

#### Defined in

[index.ts:364](https://github.com/mybigday/llama.rn/blob/a8c8191/src/index.ts#L364)

___

### rows

• **rows**: `number`

#### Defined in

[index.ts:368](https://github.com/mybigday/llama.rn/blob/a8c8191/src/index.ts#L368)

## Methods

### bake

▸ **bake**(): `Promise`<`void`\>

#### Returns

`Promise`<`void`\>

#### Defined in

[index.ts:382](https://github.com/mybigday/llama.rn/blob/a8c8191/src/index.ts#L382)

___

### release

▸ **release**(): `Promise`<`void`\>

#### Returns

`Promise`<`void`\>

#### Defined in

[index.ts:389](https://github.com/mybigday/llama.rn/blob/a8c8191/src/index.ts#L389)
