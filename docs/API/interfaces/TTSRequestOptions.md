[llama.rn](../README.md) / TTSRequestOptions

# Interface: TTSRequestOptions

codec.cpp's request overrides for `getFormattedAudioCompletion`. Unset
fields keep the model's defaults. The sampling fields apply to the samplers
codec.cpp runs itself (codec-side heads) and to the backbone sampling it
reports in `completionParams`.

## Table of contents

### Properties

- [flowGuidance](TTSRequestOptions.md#flowguidance)
- [flowSteps](TTSRequestOptions.md#flowsteps)
- [lmGuidance](TTSRequestOptions.md#lmguidance)
- [maxFrames](TTSRequestOptions.md#maxframes)
- [minFrames](TTSRequestOptions.md#minframes)
- [minP](TTSRequestOptions.md#minp)
- [preset](TTSRequestOptions.md#preset)
- [repetitionPenalty](TTSRequestOptions.md#repetitionpenalty)
- [seed](TTSRequestOptions.md#seed)
- [temperature](TTSRequestOptions.md#temperature)
- [topK](TTSRequestOptions.md#topk)
- [topP](TTSRequestOptions.md#topp)

## Properties

### flowGuidance

• `Optional` **flowGuidance**: `number`

Guidance of a flow / diffusion decoder.

#### Defined in

[tts.ts:106](https://github.com/mybigday/llama.rn/blob/a1b27a3e/src/tts.ts#L106)

___

### flowSteps

• `Optional` **flowSteps**: `number`

Steps of a flow / diffusion decoder.

#### Defined in

[tts.ts:108](https://github.com/mybigday/llama.rn/blob/a1b27a3e/src/tts.ts#L108)

___

### lmGuidance

• `Optional` **lmGuidance**: `number`

Classifier-free guidance weight of the backbone pair; `0` turns it off (one sequence).

#### Defined in

[tts.ts:104](https://github.com/mybigday/llama.rn/blob/a1b27a3e/src/tts.ts#L104)

___

### maxFrames

• `Optional` **maxFrames**: `number`

Longest generation, in audio frames (`completionParams.n_predict`).

#### Defined in

[tts.ts:102](https://github.com/mybigday/llama.rn/blob/a1b27a3e/src/tts.ts#L102)

___

### minFrames

• `Optional` **minFrames**: `number`

Shortest generation, in audio frames, before the stop token may end it.

#### Defined in

[tts.ts:100](https://github.com/mybigday/llama.rn/blob/a1b27a3e/src/tts.ts#L100)

___

### minP

• `Optional` **minP**: `number`

#### Defined in

[tts.ts:97](https://github.com/mybigday/llama.rn/blob/a1b27a3e/src/tts.ts#L97)

___

### preset

• `Optional` **preset**: `string`

A sampling preset the model defines.

#### Defined in

[tts.ts:110](https://github.com/mybigday/llama.rn/blob/a1b27a3e/src/tts.ts#L110)

___

### repetitionPenalty

• `Optional` **repetitionPenalty**: `number`

#### Defined in

[tts.ts:98](https://github.com/mybigday/llama.rn/blob/a1b27a3e/src/tts.ts#L98)

___

### seed

• `Optional` **seed**: `number`

Seed of codec.cpp's samplers (and `completionParams.seed`).

#### Defined in

[tts.ts:93](https://github.com/mybigday/llama.rn/blob/a1b27a3e/src/tts.ts#L93)

___

### temperature

• `Optional` **temperature**: `number`

#### Defined in

[tts.ts:94](https://github.com/mybigday/llama.rn/blob/a1b27a3e/src/tts.ts#L94)

___

### topK

• `Optional` **topK**: `number`

#### Defined in

[tts.ts:96](https://github.com/mybigday/llama.rn/blob/a1b27a3e/src/tts.ts#L96)

___

### topP

• `Optional` **topP**: `number`

#### Defined in

[tts.ts:95](https://github.com/mybigday/llama.rn/blob/a1b27a3e/src/tts.ts#L95)
