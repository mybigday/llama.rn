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
- [save](LlamaSpeaker.md#save)

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

[index.ts:414](https://github.com/mybigday/llama.rn/blob/b65aefa5/src/index.ts#L414)

## Properties

### baked

• **baked**: `boolean`

#### Defined in

[index.ts:410](https://github.com/mybigday/llama.rn/blob/b65aefa5/src/index.ts#L410)

___

### ctxId

• `Private` **ctxId**: `number`

#### Defined in

[index.ts:412](https://github.com/mybigday/llama.rn/blob/b65aefa5/src/index.ts#L412)

___

### family

• `Readonly` **family**: `string`

#### Defined in

[index.ts:406](https://github.com/mybigday/llama.rn/blob/b65aefa5/src/index.ts#L406)

___

### id

• `Readonly` **id**: `number`

#### Defined in

[index.ts:404](https://github.com/mybigday/llama.rn/blob/b65aefa5/src/index.ts#L404)

___

### rows

• **rows**: `number`

#### Defined in

[index.ts:408](https://github.com/mybigday/llama.rn/blob/b65aefa5/src/index.ts#L408)

## Methods

### bake

▸ **bake**(): `Promise`<`void`\>

#### Returns

`Promise`<`void`\>

#### Defined in

[index.ts:422](https://github.com/mybigday/llama.rn/blob/b65aefa5/src/index.ts#L422)

___

### release

▸ **release**(): `Promise`<`void`\>

#### Returns

`Promise`<`void`\>

#### Defined in

[index.ts:440](https://github.com/mybigday/llama.rn/blob/b65aefa5/src/index.ts#L440)

___

### save

▸ **save**(`path`, `name?`): `Promise`<`void`\>

Write this speaker's decoder conditioning (codec_speaker) to a speaker
file, for codecs with `canSpeakerFromAudio` (Chatterbox S3G, CosyVoice).
Reload it with `LlamaContext.loadSpeakerFile`.

#### Parameters

| Name | Type |
| :------ | :------ |
| `path` | `string` |
| `name?` | `string` |

#### Returns

`Promise`<`void`\>

#### Defined in

[index.ts:434](https://github.com/mybigday/llama.rn/blob/b65aefa5/src/index.ts#L434)
