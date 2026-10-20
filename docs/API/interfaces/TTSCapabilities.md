[llama.rn](../README.md) / TTSCapabilities

# Interface: TTSCapabilities

## Table of contents

### Properties

- [builtinVoices](TTSCapabilities.md#builtinvoices)
- [canSpeakerFromAudio](TTSCapabilities.md#canspeakerfromaudio)
- [defaultLanguage](TTSCapabilities.md#defaultlanguage)
- [family](TTSCapabilities.md#family)
- [promptKind](TTSCapabilities.md#promptkind)
- [requiresPhonemes](TTSCapabilities.md#requiresphonemes)
- [samplingDefaults](TTSCapabilities.md#samplingdefaults)
- [type](TTSCapabilities.md#type)

## Properties

### builtinVoices

• **builtinVoices**: [`TTSSpeakerInfo`](TTSSpeakerInfo.md)[]

Builtin decoder voices of the codec GGUF (Chatterbox S3G, CosyVoice).
Pass a voice's `name` as `speaker` to `getFormattedAudioCompletion`.

#### Defined in

[tts.ts:53](https://github.com/mybigday/llama.rn/blob/2f663506/src/tts.ts#L53)

___

### canSpeakerFromAudio

• **canSpeakerFromAudio**: `boolean`

True when a `LlamaSpeaker`'s reference audio also conditions the decoder
(voice cloning), which also lets `saveSpeaker` write it to a file.

#### Defined in

[tts.ts:58](https://github.com/mybigday/llama.rn/blob/2f663506/src/tts.ts#L58)

___

### defaultLanguage

• **defaultLanguage**: `string`

Suggested language for the phonemizer hook ("en-us" today).

#### Defined in

[tts.ts:48](https://github.com/mybigday/llama.rn/blob/2f663506/src/tts.ts#L48)

___

### family

• **family**: ``""`` \| ``"soprano"`` \| ``"neutts"`` \| ``"csm"`` \| ``"qwen3_tts"`` \| ``"moss_ttsd"`` \| ``"chatterbox"`` \| ``"bluemagpie"`` \| ``"cosyvoice"`` \| ``"outetts"`` \| ``"moss_tts"``

#### Defined in

[tts.ts:33](https://github.com/mybigday/llama.rn/blob/2f663506/src/tts.ts#L33)

___

### promptKind

• **promptKind**: ``""`` \| ``"outetts_legacy"`` \| ``"outetts_v0_3"`` \| ``"outetts_v1_0"`` \| ``"soprano"`` \| ``"neutts"`` \| ``"csm"`` \| ``"qwen3_tts"`` \| ``"moss_tts_realtime"`` \| ``"moss_ttsd"`` \| ``"chatterbox"`` \| ``"chatterbox_multilingual"`` \| ``"chatterbox_turbo"`` \| ``"bluemagpie"`` \| ``"cosyvoice"``

Prompt assembly family — drives default-voice selection on the JS side.

#### Defined in

[tts.ts:17](https://github.com/mybigday/llama.rn/blob/2f663506/src/tts.ts#L17)

___

### requiresPhonemes

• **requiresPhonemes**: `boolean`

True when the model was trained on phonemes — caller should provide a phonemizer hook.

#### Defined in

[tts.ts:46](https://github.com/mybigday/llama.rn/blob/2f663506/src/tts.ts#L46)

___

### samplingDefaults

• **samplingDefaults**: `Object`

Always empty: a text-to-speech completion samples with the model's own
settings (codec.cpp).

**`Deprecated`**

#### Type declaration

| Name | Type |
| :------ | :------ |
| `min_p?` | `number` |
| `penalty_repeat?` | `number` |
| `temperature?` | `number` |
| `top_k?` | `number` |
| `top_p?` | `number` |

#### Defined in

[tts.ts:64](https://github.com/mybigday/llama.rn/blob/2f663506/src/tts.ts#L64)

___

### type

• **type**: `number`

Numeric tts_type enum value (matches cpp/rn-tts.h).

#### Defined in

[tts.ts:15](https://github.com/mybigday/llama.rn/blob/2f663506/src/tts.ts#L15)
