[llama.rn](../README.md) / TTSCapabilities

# Interface: TTSCapabilities

## Table of contents

### Properties

- [defaultLanguage](TTSCapabilities.md#defaultlanguage)
- [family](TTSCapabilities.md#family)
- [promptKind](TTSCapabilities.md#promptkind)
- [requiresPhonemes](TTSCapabilities.md#requiresphonemes)
- [type](TTSCapabilities.md#type)

## Properties

### defaultLanguage

• **defaultLanguage**: `string`

Suggested language for the phonemizer hook ("en-us" today).

#### Defined in

[tts.ts:37](https://github.com/mybigday/llama.rn/blob/a8c8191/src/tts.ts#L37)

___

### family

• **family**: ``""`` \| ``"soprano"`` \| ``"neutts"`` \| ``"csm"`` \| ``"qwen3_tts"`` \| ``"moss_ttsd"`` \| ``"chatterbox"`` \| ``"bluemagpie"`` \| ``"outetts"`` \| ``"moss_tts"``

#### Defined in

[tts.ts:23](https://github.com/mybigday/llama.rn/blob/a8c8191/src/tts.ts#L23)

___

### promptKind

• **promptKind**: ``""`` \| ``"outetts_legacy"`` \| ``"outetts_v0_3"`` \| ``"outetts_v1_0"`` \| ``"soprano"`` \| ``"neutts"`` \| ``"csm"`` \| ``"qwen3_tts"`` \| ``"moss_tts_realtime"`` \| ``"moss_ttsd"`` \| ``"chatterbox"`` \| ``"chatterbox_multilingual"`` \| ``"bluemagpie"``

Prompt assembly family — drives default-voice selection on the JS side.

#### Defined in

[tts.ts:9](https://github.com/mybigday/llama.rn/blob/a8c8191/src/tts.ts#L9)

___

### requiresPhonemes

• **requiresPhonemes**: `boolean`

True when the model was trained on phonemes — caller should provide a phonemizer hook.

#### Defined in

[tts.ts:35](https://github.com/mybigday/llama.rn/blob/a8c8191/src/tts.ts#L35)

___

### type

• **type**: `number`

Numeric tts_type enum value (matches cpp/rn-tts.h).

#### Defined in

[tts.ts:7](https://github.com/mybigday/llama.rn/blob/a8c8191/src/tts.ts#L7)
