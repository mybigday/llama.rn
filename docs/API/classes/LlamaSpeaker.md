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

[index.ts:375](https://github.com/mybigday/llama.rn/blob/b3bb9620/src/index.ts#L375)

## Properties

### baked

• **baked**: `boolean`

#### Defined in

[index.ts:371](https://github.com/mybigday/llama.rn/blob/b3bb9620/src/index.ts#L371)

___

### ctxId

• `Private` **ctxId**: `number`

#### Defined in

[index.ts:373](https://github.com/mybigday/llama.rn/blob/b3bb9620/src/index.ts#L373)

___

### family

• `Readonly` **family**: `string`

#### Defined in

[index.ts:367](https://github.com/mybigday/llama.rn/blob/b3bb9620/src/index.ts#L367)

___

### id

• `Readonly` **id**: `number`

#### Defined in

[index.ts:365](https://github.com/mybigday/llama.rn/blob/b3bb9620/src/index.ts#L365)

___

### rows

• **rows**: `number`

#### Defined in

[index.ts:369](https://github.com/mybigday/llama.rn/blob/b3bb9620/src/index.ts#L369)

## Methods

### bake

▸ **bake**(): `Promise`<`void`\>

#### Returns

`Promise`<`void`\>

#### Defined in

[index.ts:383](https://github.com/mybigday/llama.rn/blob/b3bb9620/src/index.ts#L383)

___

### release

▸ **release**(): `Promise`<`void`\>

#### Returns

`Promise`<`void`\>

#### Defined in

[index.ts:390](https://github.com/mybigday/llama.rn/blob/b3bb9620/src/index.ts#L390)
