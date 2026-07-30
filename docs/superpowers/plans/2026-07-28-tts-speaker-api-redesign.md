# TTS Speaker API Redesign — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the caller-threaded speaker-embedding API with a native-backed `LlamaSpeaker` handle passed as a `speaker` param and resolved in C++ via an auto-placed `<__speaker__>` prompt tag.

**Architecture:** The speaker's PCM/embedding lives in a native id-registry (`g_speakers`, mirroring `g_llamaContexts`); JS holds an opaque `LlamaSpeaker` handle. `getFormattedAudioCompletion` inserts a `<__speaker__>` marker; the completion ingest path resolves it — bake-if-needed via `codec_lm_speaker_encode` (audio-hash cached), then inject `rows × n_embd` rows at the marker. Clean break on the old surface.

**Tech Stack:** TypeScript (`src/`), Jest (`src/__tests__`, `jest/mock.js`), JSI/C++ (`cpp/jsi/`), rn-tts/rn-completion C++ (`cpp/`), codec.cpp speaker encoder.

**Spec:** `docs/superpowers/specs/2026-07-28-tts-speaker-api-redesign-design.md`

## Global Constraints

- Clean break: no back-compat shims. Remove `encodeSpeaker`, `getFormattedAudioCompletion`'s `speakerEmb{Prefix,Rows,HiddenDim}` return fields, and `generateAudioCodes`' speaker params.
- Native id-registry pattern, NOT `jsi::HostObject` (no HostObject precedent in this codebase).
- C++ owns the embedding shape: `rows = data.length / n_embd`; hidden dim **is** `llama_model_n_embd(model)` (validate, never accept from JS).
- Marker constant: `RNLLAMA_SPEAKER_MARKER = '<__speaker__>'`, distinct from `<__media__>`.
- All llama.cpp/ggml symbols are `LM_`/`lm_`-prefixed. Conventional commits.
- Host C++ tests do NOT compile `cpp/jsi/` and cannot run TTS models — native tasks verify by `tests/build_and_test.sh` (compile) + a voice-clone GGUF probe, not unit tests.
- Run `npm run typecheck && npm run lint && npm test` after every JS task.

## File structure

- `src/index.ts` — remove `encodeSpeaker`; add `LlamaSpeaker` class + `ctx.createSpeaker`; rework `getFormattedAudioCompletion` (`speaker: string | LlamaSpeaker`, drop `speakerEmb*` return); strip speaker params from `generateAudioCodes`; export `RNLLAMA_SPEAKER_MARKER`.
- `src/jsi.ts` — declare `llamaCreateSpeaker` / `llamaBakeSpeaker` / `llamaReleaseSpeaker`; remove `llamaEncodeSpeaker`; drop speaker fields from `llamaGenerateAudioCodes`.
- `jest/mock.js` — mock the three new JSI fns; drop `llamaEncodeSpeaker`.
- `src/__tests__/ttsSpeaker.test.ts` — new; unit tests for `createSpeaker`/`LlamaSpeaker`/`getFormattedAudioCompletion` speaker dispatch.
- `cpp/jsi/RNLlamaJSI.cpp` — `g_speakers` registry; `createSpeaker`/`bakeSpeaker`/`releaseSpeaker` bindings; thread `speakerId` into completion + `getFormattedAudioCompletion`; remove `encodeSpeaker` + `generateAudioCodes` speaker extraction.
- `cpp/jsi/JSICompletion.h` — carry `speaker_id` in completion params.
- `cpp/rn-tts.h` / `cpp/rn-tts.cpp` — native speaker struct (raw PCM + optional baked embedding + audio hash); `speakerEncodeInto` / registry storage; reuse existing `codec_lm_speaker_encode` call (rn-tts.cpp ~3384).
- `cpp/rn-llama.cpp` (or the native `getFormattedAudioCompletion` builder) — insert `<__speaker__>` at the model-correct position when a speaker id is present.
- `cpp/rn-completion.cpp` — replace the `pending_speaker_emb_*` position-0 block (~1035-1072) with marker-position injection; wire into talker/chatterbox/realtime prefills.
- `example/src/screens/TTSScreen.tsx` — migrate to `createSpeaker` + `speaker` param.
- PR #300 body — refresh the API section.

---

### Task 1: JS — `LlamaSpeaker` handle + `ctx.createSpeaker`

**Files:**
- Modify: `src/index.ts` (add class + method + marker export)
- Modify: `src/jsi.ts` (add 3 fn decls)
- Modify: `jest/mock.js` (add 3 mocks)
- Test: `src/__tests__/ttsSpeaker.test.ts` (create)

**Interfaces:**
- Consumes: `getJsi()` (existing), `this.id` (context id).
- Produces:
  - `class LlamaSpeaker { readonly id:number; readonly family:string; readonly rows:number; readonly baked:boolean; bake():Promise<void>; release():Promise<void> }`
  - `LlamaContext.createSpeaker(config:{ refAudio:Float32Array|number[]; refAudioSampleRate:number; refText?:string; emotion?:number; bake?:boolean }):Promise<LlamaSpeaker>`
  - JSI: `llamaCreateSpeaker(ctxId, optsJson)=>{ id, family, rows, baked }`, `llamaBakeSpeaker(ctxId, speakerId)=>{ rows, baked }`, `llamaReleaseSpeaker(ctxId, speakerId)=>void`
  - `export const RNLLAMA_SPEAKER_MARKER = '<__speaker__>'`

- [ ] **Step 1: Add the JSI declarations** in `src/jsi.ts` next to the existing `llamaInitVocoder`/`llamaEncodeSpeaker` decls (match their `(contextId: number, ...) => Promise<...>` style):

```ts
llamaCreateSpeaker: (
  contextId: number,
  optsJson: string,
) => Promise<{ id: number; family: string; rows: number; baked: boolean }>
llamaBakeSpeaker: (
  contextId: number,
  speakerId: number,
) => Promise<{ rows: number; baked: boolean }>
llamaReleaseSpeaker: (contextId: number, speakerId: number) => Promise<void>
```

- [ ] **Step 2: Add jest mocks** in `jest/mock.js` (mirror the existing `jest.fn(async () => ...)` TTS mocks). Use a monotonic id so tests can assert distinctness:

```js
llamaCreateSpeaker: jest.fn(async () => ({ id: 1, family: 'chatterbox', rows: 0, baked: false })),
llamaBakeSpeaker: jest.fn(async () => ({ rows: 34, baked: true })),
llamaReleaseSpeaker: jest.fn(async () => {}),
```

- [ ] **Step 3: Write the failing test** `src/__tests__/ttsSpeaker.test.ts`:

```ts
import { initLlama, LlamaSpeaker, RNLLAMA_SPEAKER_MARKER } from '..'

it('createSpeaker returns a LlamaSpeaker handle backed by the native id', async () => {
  const ctx = await initLlama({ model: 'x.gguf' })
  const spk = await ctx.createSpeaker({
    refAudio: new Float32Array([0, 0.1, -0.1]),
    refAudioSampleRate: 24000,
  })
  expect(spk).toBeInstanceOf(LlamaSpeaker)
  expect(spk.id).toBe(1)
  expect(spk.baked).toBe(false)
  await spk.bake()
  expect(spk.baked).toBe(true)
  expect(spk.rows).toBe(34)
  await spk.release()
})

it('exports the speaker marker constant', () => {
  expect(RNLLAMA_SPEAKER_MARKER).toBe('<__speaker__>')
})
```

- [ ] **Step 4: Run it, verify it fails**

Run: `npm test -- ttsSpeaker`
Expected: FAIL — `LlamaSpeaker`/`RNLLAMA_SPEAKER_MARKER` not exported, `createSpeaker` undefined.

- [ ] **Step 5: Implement in `src/index.ts`.** Add the export near the existing `RNLLAMA_MTMD_DEFAULT_MEDIA_MARKER` (line ~77):

```ts
export const RNLLAMA_SPEAKER_MARKER = '<__speaker__>'
```

Add the class (top-level, exported) and the method on `LlamaContext`:

```ts
export class LlamaSpeaker {
  id: number
  family: string
  rows: number
  baked: boolean
  private ctxId: number
  constructor(ctxId: number, h: { id: number; family: string; rows: number; baked: boolean }) {
    this.ctxId = ctxId
    this.id = h.id
    this.family = h.family
    this.rows = h.rows
    this.baked = h.baked
  }
  async bake(): Promise<void> {
    const { llamaBakeSpeaker } = getJsi()
    const r = await llamaBakeSpeaker(this.ctxId, this.id)
    this.rows = r.rows
    this.baked = r.baked
  }
  async release(): Promise<void> {
    const { llamaReleaseSpeaker } = getJsi()
    await llamaReleaseSpeaker(this.ctxId, this.id)
  }
}

// on LlamaContext:
async createSpeaker(config: {
  refAudio: Float32Array | number[]
  refAudioSampleRate: number
  refText?: string
  emotion?: number
  bake?: boolean
}): Promise<LlamaSpeaker> {
  const { llamaCreateSpeaker } = getJsi()
  const pcm =
    config.refAudio instanceof Float32Array
      ? Array.from(config.refAudio)
      : config.refAudio
  const optsJson = JSON.stringify({
    pcm,
    inputSampleRate: config.refAudioSampleRate,
    refText: config.refText ?? '',
    bake: config.bake ?? false,
    ...(config.emotion !== undefined ? { emotion: config.emotion } : {}),
  })
  const h = await llamaCreateSpeaker(this.id, optsJson)
  return new LlamaSpeaker(this.id, h)
}
```

> NOTE: `pcm` is `Array.from(Float32Array)` here to match the existing JSON path; the ArrayBuffer/typed-array optimization from the spec is a follow-up (see Task 7 notes) and does not block this task.

- [ ] **Step 6: Run tests, verify pass**

Run: `npm test -- ttsSpeaker`
Expected: PASS (2 tests).

- [ ] **Step 7: typecheck + lint + commit**

```bash
npm run typecheck && npm run lint
git add src/index.ts src/jsi.ts jest/mock.js src/__tests__/ttsSpeaker.test.ts
git commit -m "feat(tts): LlamaSpeaker handle + ctx.createSpeaker (JS surface)"
```

---

### Task 2: JS — `getFormattedAudioCompletion` speaker dispatch + remove `encodeSpeaker`

**Files:**
- Modify: `src/index.ts` (rework `getFormattedAudioCompletion`; delete `encodeSpeaker`; strip `generateAudioCodes` speaker params)
- Modify: `src/jsi.ts` (remove `llamaEncodeSpeaker`; drop speaker fields from `llamaGenerateAudioCodes` opts doc)
- Modify: `jest/mock.js` (remove `llamaEncodeSpeaker`)
- Test: `src/__tests__/ttsSpeaker.test.ts` (extend)

**Interfaces:**
- Consumes: `LlamaSpeaker` (Task 1), `llamaGetFormattedAudioCompletion(ctxId, speakerStr, text)` (existing).
- Produces: `getFormattedAudioCompletion({ prompt, speaker?: string | LlamaSpeaker, phonemizer?, language? }) => { prompt, grammar?, embedding, flow }` (NO `speakerEmb*`). When `speaker` is a `LlamaSpeaker`, its id is forwarded to native (which places the `<__speaker__>` marker).

- [ ] **Step 1: Write the failing test** (append to `ttsSpeaker.test.ts`):

```ts
it('getFormattedAudioCompletion forwards a LlamaSpeaker id and drops speakerEmb fields', async () => {
  const ctx = await initLlama({ model: 'x.gguf' })
  const spk = await ctx.createSpeaker({ refAudio: new Float32Array([0]), refAudioSampleRate: 24000 })
  const r = await ctx.getFormattedAudioCompletion({ prompt: 'hi', speaker: spk })
  expect(r).not.toHaveProperty('speakerEmbPrefix')
  expect(r).not.toHaveProperty('speakerEmbHiddenDim')
  const { llamaGetFormattedAudioCompletion } = require('../jsi').getJsi()
  // speaker id is forwarded (exact wire shape asserted against the impl below)
  expect(llamaGetFormattedAudioCompletion).toHaveBeenCalled()
})

it('no longer exposes encodeSpeaker', async () => {
  const ctx = await initLlama({ model: 'x.gguf' })
  expect((ctx as any).encodeSpeaker).toBeUndefined()
})
```

Update the `llamaGetFormattedAudioCompletion` mock in `jest/mock.js` to return `{ prompt: 'hi', grammar: '', embedding: true, flow: 'tokens' }` (no speakerEmb fields).

- [ ] **Step 2: Run it, verify it fails**

Run: `npm test -- ttsSpeaker`
Expected: FAIL — `encodeSpeaker` still defined; result still carries `speakerEmb*`.

- [ ] **Step 3: Implement.** In `getFormattedAudioCompletion` (src/index.ts ~1079): change the signature `speaker?: string | object` → `speaker?: string | LlamaSpeaker`; drop `speakerEmbPrefix/Rows/HiddenDim` from the return type; in the body, replace the "lift speakerEmb out of speaker JSON" block (lines ~1170-1181) with: if `speaker instanceof LlamaSpeaker`, pass its id to native and skip JS voice-table resolution; else keep the string→`lookupVoice` prompt-text path. Forward the id via the JSI call — extend `llamaGetFormattedAudioCompletion`'s third arg or add a 4th `speakerId` arg (mirror the `contextId, speakerStr, inputText` pattern; add `speakerId?: number`). Delete the `encodeSpeaker` method (lines ~1258-1289). In `generateAudioCodes`, delete `speakerEmbPrefix/Rows/HiddenDim` from the options type + the JSON it sends.

```ts
// sketch of the resolver branch replacing lines ~1170-1181
if (options.speaker instanceof LlamaSpeaker) {
  const result = await llamaGetFormattedAudioCompletion(
    this.id, '', inputText, options.speaker.id,
  )
  return result // marker placed natively; no speakerEmb* fields
}
// ...existing string/voice-table path (produces speakerStr) unchanged, speakerId omitted...
```

- [ ] **Step 4: Update `src/jsi.ts` + `jest/mock.js`.** Remove `llamaEncodeSpeaker` from both; add the optional `speakerId?: number` 4th param to `llamaGetFormattedAudioCompletion`'s decl.

- [ ] **Step 5: Run tests, verify pass**

Run: `npm test -- ttsSpeaker`
Expected: PASS.

- [ ] **Step 6: typecheck + lint + commit**

```bash
npm run typecheck && npm run lint
git add src/index.ts src/jsi.ts jest/mock.js src/__tests__/ttsSpeaker.test.ts
git commit -m "feat(tts): speaker param on getFormattedAudioCompletion; drop encodeSpeaker/speakerEmb*"
```

---

### Task 3: Native — `g_speakers` registry + createSpeaker/bakeSpeaker/releaseSpeaker bindings

**Files:**
- Modify: `cpp/rn-tts.h` (native speaker struct), `cpp/rn-tts.cpp` (store/encode/free helpers)
- Modify: `cpp/jsi/RNLlamaJSI.cpp` (registry + 3 bindings + install)

**Interfaces:**
- Consumes: `getContextOrThrow(ctxId)`, `createPromiseTask`, `jsi::Function::createFromHostFunction` (existing patterns, RNLlamaJSI.cpp ~452/516); the existing `codec_lm_speaker_encode` call site (rn-tts.cpp ~3384, sets `speaker_n_rows`/`speaker_hidden_dim`).
- Produces (native):
  - `struct rn_speaker { std::vector<float> pcm; int sample_rate; std::string ref_text; float emotion; std::vector<float> emb; int rows; int hidden_dim; bool baked; uint64_t audio_hash; };`
  - `int llama_rn_context_tts::createSpeaker(pcm, sr, refText, emotion, bake) -> speakerId` (stores in a per-context `std::unordered_map<int, rn_speaker>`; returns new id).
  - `bool llama_rn_context_tts::bakeSpeaker(int id)`; `const rn_speaker* getSpeaker(int id)`; `void releaseSpeaker(int id)`.
  - JSI: `llamaCreateSpeaker`, `llamaBakeSpeaker`, `llamaReleaseSpeaker` (shapes per Task 1).

- [ ] **Step 1: Add `rn_speaker` + the per-context speaker map** to `cpp/rn-tts.h` (near the existing `pending_speaker_emb_*` fields ~256) and the create/bake/get/release methods on `llama_rn_context_tts`. `createSpeaker` allocates a monotonically-increasing id; `bakeSpeaker` runs the same encode as the current `encodeSpeaker` path (factor the rn-tts.cpp ~3384 `codec_lm_speaker_encode` block into `encodeInto(rn_speaker&)`), fills `emb`/`rows`/`hidden_dim`/`baked`; `audio_hash` = a cheap 64-bit hash of the PCM for cache reuse.

- [ ] **Step 2: Implement the helpers** in `cpp/rn-tts.cpp`, reusing the existing `codec_lm_speaker_encode` invocation (do NOT duplicate it — extract the current block into `encodeInto`). Validate `hidden_dim == llama_model_n_embd(model)` here and log+skip if not.

- [ ] **Step 3: Add the `g_speakers` access + 3 JSI bindings** in `cpp/jsi/RNLlamaJSI.cpp`, mirroring an existing binding (e.g. `initVocoder`). `createSpeaker` parses `optsJson` (pcm array, inputSampleRate, refText, emotion, bake) → `ctx->tts_wrapper->createSpeaker(...)` (+ `bakeSpeaker` if `bake`) → resolve `{ id, family, rows, baked }` (`family` from `getTTSCapabilities`). `bakeSpeaker`/`releaseSpeaker` look up by id.

- [ ] **Step 4: Register the 3 functions** where the other `llama*` host functions are set on the runtime global (same block as `llamaInitVocoder`).

- [ ] **Step 5: Build (compile-only gate)**

Run: `cd tests && ./build_and_test.sh`
Expected: `rnllama_tests` + `parallel_decoding_test` + `chat_parse_utf8_test` build (these link `rn-tts.cpp`; the JSI file isn't host-compiled, so this only verifies rn-tts.cpp compiles). Then verify the JSI compiles via the iOS/Android build in Task 7 / CI.

- [ ] **Step 6: Commit**

```bash
git add cpp/rn-tts.h cpp/rn-tts.cpp cpp/jsi/RNLlamaJSI.cpp
git commit -m "feat(tts): native g_speakers registry + createSpeaker/bakeSpeaker/releaseSpeaker JSI"
```

---

### Task 4: Native — thread `speakerId` through params + place the `<__speaker__>` marker

**Files:**
- Modify: `cpp/jsi/JSICompletion.h` (carry `speaker_id`), `cpp/jsi/RNLlamaJSI.cpp` (extract `speakerId` in completion + getFormattedAudioCompletion)
- Modify: `cpp/rn-llama.cpp` (or the native `getFormattedAudioCompletion` prompt builder)

**Interfaces:**
- Consumes: `getSpeaker(id)` (Task 3), the native `getFormattedAudioCompletion` builder, `RNLLAMA_SPEAKER_MARKER` string (define a C++ constant `RN_SPEAKER_MARKER = "<__speaker__>"`).
- Produces: completion params carry `int speaker_id = -1`; native `getFormattedAudioCompletion(speakerStr, text, speakerId)` inserts `<__speaker__>` at the model-correct position (the same spot the current per-model speaker/talker prefix targets) when `speakerId >= 0`.

- [ ] **Step 1:** Add `int speaker_id = -1;` to the completion params struct (`JSICompletion.h`) and extract `speakerId` from the completion options JSON in `RNLlamaJSI.cpp` (next to the removed `speakerEmbPrefix` extraction ~2150).
- [ ] **Step 2:** Add the optional 4th `speakerId` arg to the `llamaGetFormattedAudioCompletion` binding; pass it to the native builder.
- [ ] **Step 3:** In the native builder, when `speakerId >= 0`, insert `RN_SPEAKER_MARKER` at each model's speaker-section position (Chatterbox/Qwen3-TTS/MOSS: start-of-sequence prefix — where the current code targets position 0). Define `RN_SPEAKER_MARKER` in a shared header.
- [ ] **Step 4: Build**

Run: `cd tests && ./build_and_test.sh`
Expected: builds clean.

- [ ] **Step 5: Commit**

```bash
git add cpp/jsi/JSICompletion.h cpp/jsi/RNLlamaJSI.cpp cpp/rn-llama.cpp
git commit -m "feat(tts): thread speaker_id through params + place <__speaker__> marker"
```

---

### Task 5: Native — resolve `<__speaker__>` in the completion ingest (the core stitch)

**Files:**
- Modify: `cpp/rn-completion.cpp` (replace the `pending_speaker_emb_*` position-0 block ~1035-1072 with marker-position injection)
- Modify: `cpp/rn-tts.cpp`/`.h` if the prefill paths need the resolved rows

**Interfaces:**
- Consumes: `getSpeaker(id)` (Task 3), `params.speaker_id` (Task 4), the marker token position from the tokenized prompt, `llama_model_n_embd`.
- Produces: at the `<__speaker__>` marker position, an embd-batch of `rows × n_embd` speaker rows is decoded into the sequence; if the speaker isn't baked, `bakeSpeaker` runs first (hash-cached). The old "always at position 0" behavior is removed.

- [ ] **Step 1:** Locate the marker token(s) in the ingested prompt (the tokenizer emits the special token for `<__speaker__>`; find its position like the media path finds `<__media__>` — see `cpp/rn-mtmd.hpp` split logic ~290-361 for the pattern).
- [ ] **Step 2:** Replace the current `if (is_codec_lm_ar_tts && !pending_speaker_emb_prefix.empty() ...)` block (rn-completion.cpp ~1035-1072): resolve `getSpeaker(params.speaker_id)`; `if (!baked) bakeSpeaker(id)`; inject the `rows × n_embd` rows as an embd-batch at the marker's `n_past` position (reuse the exact `llama_batch_init`/`memcpy`/`llama_decode` shape already in that block); advance `n_past` by `rows`.
- [ ] **Step 3:** Reconcile with the talker/chatterbox/realtime prefills (~1085-1246): those build multi-row prefixes that currently include the speaker section; the tag now owns the **speaker rows**, so drop the speaker portion from those prefills and let the tag inject it, keeping the text/talker rows. Keep each prefill's early-return + `embd.resize(n_past)` invariant.
- [ ] **Step 4: Build + host-probe parity check**

Run: `cd tests && ./build_and_test.sh` (compile gate), then build a voice-clone probe (`bluemagpie_probe` has no speaker section; use/extend a Chatterbox or Qwen3-TTS probe under `.scratch/gguf/`). Compare the backbone's first post-marker hidden state (and/or ASR round-trip on decoded audio) against a pre-change build — must match.
Expected: identical audio / hidden-state parity vs the old prefix path.

- [ ] **Step 5: Commit**

```bash
git add cpp/rn-completion.cpp cpp/rn-tts.cpp cpp/rn-tts.h
git commit -m "feat(tts): resolve <__speaker__> at marker position; drop position-0 prefix block"
```

---

### Task 6: Remove old JSI speaker surface + migrate example + docs

**Files:**
- Modify: `cpp/jsi/RNLlamaJSI.cpp` (delete `encodeSpeaker` binding + `generateAudioCodes` speaker extraction ~2150-2158/2213-2269)
- Modify: `example/src/screens/TTSScreen.tsx`
- Modify: PR #300 body (via `gh api PATCH`)

**Interfaces:**
- Consumes: Tasks 1–5 (full new path).

- [ ] **Step 1:** Delete the `llamaEncodeSpeaker` JSI binding and the `speakerEmbPrefix/Rows/HiddenDim` extraction from the `generateAudioCodes` binding in `RNLlamaJSI.cpp`.
- [ ] **Step 2:** Migrate `example/src/screens/TTSScreen.tsx`: replace the `encodeSpeaker(...)` call (~203) + the `speakerEmbPrefix/...` threading (~413-466) with `const spk = await context.createSpeaker({ refAudio, refAudioSampleRate })` and `getFormattedAudioCompletion({ prompt, speaker: spk })` / `completion({ prompt, speaker: spk })`; call `spk.release()` after generation.
- [ ] **Step 3:** `npm run typecheck && npm run lint && npm test` — all green.
- [ ] **Step 4:** Refresh the PR #300 body's API section to the `createSpeaker` + tag flow (use `gh api -X PATCH repos/mybigday/llama.rn/pulls/300 --input <json>` — `gh pr edit` is broken by the Projects-classic deprecation).
- [ ] **Step 5: Commit**

```bash
git add cpp/jsi/RNLlamaJSI.cpp example/src/screens/TTSScreen.tsx
git commit -m "refactor(tts): remove legacy encodeSpeaker/speakerEmb JSI; migrate example to createSpeaker"
```

---

### Task 7: Verification + optional ArrayBuffer ingest

**Files:** (verification; optional follow-up in `src/index.ts` + JSI)

- [ ] **Step 1: Full local CI parity** — `npm run typecheck`, `npm run lint`, `npm test`, `cd tests && ./build_and_test.sh && ./run_tests.sh` all green.
- [ ] **Step 2: Device check** — one voice-clone model (Chatterbox or Qwen3-TTS) end-to-end via the `device-test` skill; confirm intelligible cloned audio.
- [ ] **Step 3: Push + watch CI** — iOS/Android builds compile `cpp/jsi/` (host tests don't); confirm all 6 jobs green (`gh run watch`).
- [ ] **Step 4 (optional, follow-up):** Replace the `Array.from(Float32Array)` ingest in `createSpeaker` with an ArrayBuffer/`getTypedArray` path in the JSI binding to drop the one remaining `number[]`/JSON copy. Guard behind its own commit; not required for correctness.

---

## Self-review

- **Spec coverage:** §1 handle → Task 1/3; §2 tag → Task 4; §3 resolution → Task 5; §4 clean break + migration → Task 2/6; §5 testing → Task 5/7; precondition (initVocoder) → enforced in Task 3 Step 3. ✎ Covered.
- **Placeholders:** Native tasks (3–5) intentionally specify algorithm + anchors + verification rather than speculative line-level C++ (the spec deferred the stitch; host tests can't unit-test it). This is a deliberate altitude choice, not a TODO — flagged in Global Constraints.
- **Type consistency:** `LlamaSpeaker { id, family, rows, baked, bake(), release() }` and the JSI shapes `{ id, family, rows, baked }` are used identically in Tasks 1→2→3. `speaker_id` (native) ↔ `speaker.id` (JS) consistent across Tasks 3→4→5.
