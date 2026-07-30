# TTS Speaker API redesign — native-backed speaker via a `<__speaker__>` tag

Status: approved design (brainstorm), ready for implementation planning
Date: 2026-07-28
Branch: `codec` (part of PR #300 — pre-release, clean break allowed)

## Problem

The current voice-clone / speaker-conditioning API makes the caller thread a
raw embedding matrix and its shape through three calls:

- `encodeSpeaker(...)` returns `{ speakerEmb, speakerNRows, speakerHiddenDim, … }`.
- `getFormattedAudioCompletion(...)` lifts that into `speakerEmbPrefix` /
  `speakerEmbRows` / `speakerEmbHiddenDim`.
- the caller passes those back into `generateAudioCodes(...)` (or `completion`).

Three concrete problems:

1. **`speakerEmbHiddenDim` is redundant.** The consumer requires
   `hidden_dim == llama_model_n_embd(model)` (`cpp/rn-completion.cpp`), and the
   value is *produced* natively by `codec_lm_speaker_encode` and *consumed*
   natively — JS is only a courier. `rows` is likewise `data.length / n_embd`.
2. **`speakerEmbPrefix` is misnamed.** `encodeSpeaker`'s own docstring says the
   matrix is consumed "per its arch's convention (prefix concat, additive
   overlay, cross-attn KV, …)" — i.e. *not always a prefix*. The neutral source
   field is `speakerEmb`; the `Prefix` suffix is added at the lifting step.
3. **The big float bag is JSON-marshaled every call.** The embedding
   (`rows × n_embd` floats) is stringified through the JSI bridge on each
   `getFormattedAudioCompletion` / `generateAudioCodes` — the dominant cost.

## Goals

- Speaker conditioning becomes a first-class **input to completion**, resolved
  in C++ via a prompt tag — the same *pattern* as multimodal media
  (`<__media__>` + resolve-in-C++ + inject embedding rows at the marker).
- The embedding/PCM lives **natively**; JS holds an opaque handle. No
  per-completion copy, no JSON of the big array. C++ owns the shape.
- Clean break: remove the old speaker surface (nothing external depends on it
  yet).

## Non-goals

- Persisting a baked voice across app launches (no `export()/import()` for now;
  re-encode from the persisted ref audio if needed — YAGNI).
- Changing the rest of the TTS API (`initVocoder`, `getTTSCapabilities`,
  `decodeAudioTokens`, `getAudioSampleRate`, the `tokens` /
  `codec_lm_ar` / `continuous_embd` flows) — those stay as documented on PR #300.
- Reusing the literal mtmd bitmap/projector pipeline. The speaker encoder is
  `codec_lm_speaker_encode` (part of the codec), a *different* encoder than the
  mtmd clip/audio projector, so this is a dedicated resolver that mirrors the
  pattern only.

## Decisions (resolved during brainstorm)

- **Encode timing: hybrid.** `createSpeaker` stores raw ref audio; the embedding
  is computed either eagerly (`bake()`) for cheap reuse, or lazily at
  tag-resolution time, cached by an audio hash so a repeated voice isn't
  re-encoded.
- **Entry point: `speaker` param + auto-placed tag.** `getFormattedAudioCompletion`
  inserts the `<__speaker__>` marker at the model-correct position; the caller
  never writes the tag.
- **Back-compat: clean break.** Remove `encodeSpeaker`, the `speakerEmb*` return
  fields, and `generateAudioCodes`' speaker params.
- **Handle mechanism: native id-registry, not `jsi::HostObject`.** The codebase
  has no HostObject precedent; every native resource is an id in a registry
  (`g_llamaContexts` + `getContextOrThrow`). The speaker mirrors that (avoids
  GC-finalizer lifetime pitfalls).
- **Persistence: deferred** (see non-goals).

## Design

### 1. `LlamaSpeaker` — native-backed handle (`src/index.ts`)

```ts
class LlamaSpeaker {           // thin JS handle: an id + cached read-only meta
  readonly id: number
  readonly family: string      // from TTS capabilities
  readonly rows: number        // 0 until baked
  readonly baked: boolean
  bake(): Promise<void>        // run codec_lm_speaker_encode in-place (native)
  release(): void              // free the registry entry
}

ctx.createSpeaker(config: {
  refAudio: Float32Array       // mono ref audio; ArrayBuffer-backed to skip number[] boxing
  refAudioSampleRate: number
  refText?: string             // reference transcript (models that need it)
  emotion?: number             // [0,1]; only codecs that declare needs_emotion_scalar
  bake?: boolean               // eagerly encode now (default false → lazy at resolve time)
}): Promise<LlamaSpeaker>
```

- `createSpeaker` copies the ref audio across the bridge **once** (unavoidable —
  it originates in JS), stores it in a native `g_speakers` registry keyed by a
  new speaker id, and returns the handle. `refAudio` is a `Float32Array` (read
  via ArrayBuffer) so even this one ingest avoids `number[]`/JSON.
- **Precondition:** `initVocoder` must have run first — `codec_lm_speaker_encode`
  lives in the loaded codec, and `family` comes from `getTTSCapabilities`.
  `createSpeaker` only stores raw PCM, but `bake()` and lazy tag-resolution both
  need the codec.
- Nothing large moves again: `bake()` encodes in place; `completion` /
  `getFormattedAudioCompletion` pass only `speaker.id`.

### 2. The `<__speaker__>` tag (`src/index.ts` + `cpp`)

- New marker constant `RNLLAMA_SPEAKER_MARKER = '<__speaker__>'`, **separate**
  from `<__media__>` — different resolver, so the two coexist and never collide.
- `getFormattedAudioCompletion({ prompt, speaker, phonemizer?, language? })`:
  - `speaker` is now `string | LlamaSpeaker`:
    - **`string`** → built-in named voice → resolved by the JS `tts-voices`
      tables into **prompt text** (existing behavior; no tag, `tokens` flow).
    - **`LlamaSpeaker`** → voice-clone → native inserts `<__speaker__>` at the
      model-correct spot (today the start-of-sequence speaker section for
      Chatterbox / Qwen3-TTS / MOSS; the mechanism allows any position).
  - Return type drops `speakerEmbPrefix` / `speakerEmbRows` / `speakerEmbHiddenDim`.
    The returned `prompt` carries the marker (or none when no speaker section).
- `completion({ prompt, speaker, ... })` accepts the `LlamaSpeaker` and resolves
  the marker in the ingest path.

### 3. C++ resolution (completion ingest path)

When the prompt tokenizer reaches `<__speaker__>` (speaker id available from the
completion params):

1. registry entry has a **baked embedding** → inject its `rows × n_embd` float
   rows as an embd-batch at the marker position;
2. else it has **raw PCM** → run `codec_lm_speaker_encode` **once, cached by an
   audio hash** (same idea as mtmd `bitmap_past_hashes`), then inject;
3. **C++ owns the shape**: `rows = data.length / n_embd`, hidden dim *is*
   `llama_model_n_embd` (validated, never caller-supplied).

This replaces the current `pending_speaker_emb_*` "always at position 0" block
with "at the marker position." The exact stitch into the existing
talker / chatterbox / realtime prefill paths (which build multi-row prefixes of
text+speaker) is worked out in the implementation plan; the tag owns the
**speaker-embedding** rows specifically, the text/talker prompt stays as tokens.

### 4. Clean-break migration

- Remove JS: `encodeSpeaker`; `speakerEmb{Prefix,Rows,HiddenDim}` from
  `getFormattedAudioCompletion`; the speaker params on `generateAudioCodes`
  (which stays as a deprecated `completion` wrapper, now speaker-free).
- Remove the matching JSI extraction (`RNLlamaJSI.cpp` `speakerEmbPrefix` /
  `speakerEmbRows` / `speakerEmbHiddenDim`); add `createSpeaker` / `bakeSpeaker` /
  `releaseSpeaker` JSI bindings + the `g_speakers` registry.
- Update `example/src/screens/TTSScreen.tsx`:
  `const spk = await ctx.createSpeaker({ refAudio, refAudioSampleRate })` →
  `getFormattedAudioCompletion({ prompt, speaker: spk })` →
  `completion({ prompt, speaker: spk })` → `spk.release()`.
- Refresh the PR #300 API section written earlier.

### 5. Lifetime

- Explicit `speaker.release()`. All of a context's speakers are freed when the
  context is released. Do not release a speaker while a completion using it is
  in flight (same rule as a context).

## Testing

- **JS surface:** `typecheck`, `lint`, `jest` (mock the new JSI fns in
  `jest/mock.js`).
- **Native parity:** a host probe on a voice-clone GGUF (Chatterbox or
  Qwen3-TTS) confirming the tag-injected speaker matches the old prefix path —
  compare the backbone's post-prefix hidden state, and/or an ASR round-trip on
  decoded audio.
- **Example device check:** one voice-clone model end-to-end on device.

## Risks / open items

- **§3 prefill stitch is the riskiest part.** The talker / chatterbox / realtime
  prefill paths in `rn-completion.cpp` are intricate and **not covered by the
  host C++ tests** — they need the native-parity probe above.
- Introduces the first native id-registry beyond contexts (`g_speakers`) and the
  first ArrayBuffer/TypedArray read path in the JSI completion layer.
- Real JS + C++ refactor; reasonable to land in #300 before it ships, but not
  small.
