[llama.rn](../README.md) / TTSRequestOptions

# Interface: TTSRequestOptions

codec.cpp's request overrides for `getFormattedAudioCompletion`. Sampling
is the `completion()`'s (llama.rn hands it to codec.cpp's own samplers too);
model graph settings (guidance, flow steps) are `initVocoder`
`kv_overrides`.

## Table of contents

### Properties

- [minFrames](TTSRequestOptions.md#minframes)

## Properties

### minFrames

• `Optional` **minFrames**: `number`

Shortest generation, in audio frames, before the end token may end it (in the grammar). Default: the model's.

#### Defined in

[tts.ts:93](https://github.com/mybigday/llama.rn/blob/83447e50/src/tts.ts#L93)
