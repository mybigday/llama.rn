#ifndef CODEC_LM_H
#define CODEC_LM_H

#include "codec.h"

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// =====================================================================
// codec_lm — adaptor between a host LLM (running in llama.cpp) and a
// codec (running in codec.cpp).  The host LLM is a standard transformer
// llama.cpp can load; this layer turns its hidden state into the audio
// codebook tokens that the codec consumes, and turns codes back into the
// audio embedding the caller has to feed into the host LLM at the next
// AR step.
//
// Scope is intentionally narrow.  codec_lm does:
//   * audio-codebook lookup `(cb_idx, code) -> [hidden_dim]` row
//   * sum-of-codebook compose: `codes[N] -> [hidden_dim]`
//   * per-step state machine: hidden -> N codebook logits, with intra-
//     step conditioning when the kind requires it (residual_depth_ar)
//
// codec_lm does NOT do:
//   * load any GGUF (it borrows everything from a `codec_model` that
//     was already loaded by codec.cpp's regular loader)
//   * sample (caller picks logits up and runs whatever sampler — e.g.
//     `llama_sampler_apply` on a constructed `llama_token_data_array` —
//     and pushes the chosen code back in)
//   * touch the text path — text token embeddings live in the host
//     LLM's `tok_embd`, accessed via llama.cpp's normal `b.token` flow
//     (or, for models that mix text + audio at the same position, the
//     caller extracts the text embedding table out-of-band)
//   * decide which positions are text vs audio — that's per-model glue
//     in the caller, not a generic library concern
//
// All weights live in the same GGUF the codec is loaded from, under
// the `lm.*` tensor namespace, plus a few `codec.lm.*` metadata keys.
// =====================================================================

// Adaptor kinds.  GGUF stores `codec.lm.kind` as a string for forward
// compatibility (old runtimes can fail gracefully on unknown kind names
// from newer GGUFs).  The loader maps string -> enum at codec_lm_create
// time; an unrecognised string yields CODEC_LM_KIND_UNKNOWN and
// codec_lm_create returns NULL.
//
// CODEC_LM_KIND_PARALLEL_HEADS_DELAY  — N parallel `Linear(hidden, vocab_i)`
//                                       heads off a backbone hidden state
//                                       the caller hands in, no intra-step
//                                       dependency.  Optional per-cb
//                                       `delay[N]` shift register.
//                                       Compose-input is
//                                       `sum_i audio_embd_i[codes[i]]`.
//                                       Models: MOSS-TTSD.
//
// CODEC_LM_KIND_RESIDUAL_DEPTH_AR     — Backbone (in llama.cpp) emits c0
//                                       from a linear head off the backbone
//                                       hidden the caller hands in;
//                                       codec_lm runs a small AR depth
//                                       transformer (4-6 Llama-style
//                                       layers) with its own KV cache
//                                       (reset every backbone step) and
//                                       emits c1..c_{N-1} sequentially,
//                                       conditioned on the backbone hidden
//                                       + previously sampled cb embeddings.
//                                       Models: CSM, Qwen3-TTS,
//                                       Qwen3-Omni-MoE Talker, Moshi,
//                                       LFM2-Audio.

// CODEC_LM_KIND_CONTINUOUS_LATENT_CFM — VoxCPM / BlueMagpie family.  The
//                                       backbone (Barbet) hands in a hidden
//                                       state; this adaptor runs the whole
//                                       continuous-latent generation step
//                                       (tslm_adapter + FSQ + RALM + LocDiT
//                                       CFM diffusion) and emits ONE latent
//                                       patch (not codebook codes) plus a stop
//                                       flag, then turns that patch back into
//                                       the embedding the backbone consumes
//                                       next (LocEnc).  No logits, no sampling:
//                                       uses the step_generate /
//                                       step_feedback_embd entry points instead
//                                       of the codebook step machine.
//                                       Models: BlueMagpie-TTS, VoxCPM2.
enum codec_lm_kind {
    CODEC_LM_KIND_UNKNOWN               = 0,
    CODEC_LM_KIND_PARALLEL_HEADS_DELAY  = 1,
    CODEC_LM_KIND_RESIDUAL_DEPTH_AR     = 2,
    CODEC_LM_KIND_CONTINUOUS_LATENT_CFM = 3,
};

// Returns the canonical GGUF-string name of the kind ("parallel_heads_delay"
// / "residual_depth_ar" / "unknown").  Useful for logging.
const char * codec_lm_kind_name(enum codec_lm_kind kind);

struct codec_model;       // from codec.h
struct codec_lm;
struct codec_lm_state;

struct codec_lm_info {
    enum codec_lm_kind kind;

    // Hidden dimension expected from the host LLM and emitted by the
    // audio embedding.  Caller is responsible for verifying this
    // matches `llama_model_n_embd(host_model)`.
    int32_t         hidden_dim;

    // Audio embedding dimension (the dim of vectors returned by
    // codec_lm_audio_embd).  Used to size the depth-side audio embed
    // rows; equals hidden_dim for shared / in_proj-on-prefix models
    // (CSM / Qwen3-TTS) and depth_hidden for in_proj_per_pos models
    // (Moshi / LFM2).
    int32_t         audio_embed_dim;

    // Output dimension of `codec_lm_compose_audio_embd`.  When the
    // model has a backbone-side compose table separate from the depth-
    // side audio embeds (LFM2-Audio's `audio_embedding`), this differs
    // from audio_embed_dim.  Zero when compose isn't supported (Moshi:
    // caller composes via the backbone's own embed tables).  For
    // models without a separate compose table this equals
    // audio_embed_dim.
    int32_t         compose_audio_embed_dim;

    // Number of audio codebooks the model emits per AR step.
    int32_t         n_codebook;

    // Per-codebook vocabulary size; pointer valid for the lifetime of
    // the codec_lm.  For some models the first codebook is a text
    // vocabulary (MOSS-TTSD channel 0 = text, channels 1..7 = audio);
    // codec_lm itself doesn't care, it just sizes the logits arrays.
    const int32_t * codebook_sizes;     // [n_codebook]

    // Per-codebook delay offset in AR steps; zeros if not applicable.
    // Caller is expected to apply the delay externally when assembling
    // the prompt; codec_lm only uses this to size internal state.
    const int32_t * delay_pattern;      // [n_codebook]

    // Informational: the host LLM architecture name written by the
    // converter (e.g., "llama", "qwen3", "lfm2").  codec_lm does not
    // enforce a match — caller is expected to load the right backbone.
    const char *    host_arch;

    // Continuous-latent kinds (CONTINUOUS_LATENT_CFM) only.  For these the
    // audio representation is a continuous latent patch, not codebook codes:
    // n_codebook / codebook_sizes are 0/NULL.  `patch_size` latent frames of
    // `latent_dim` each are produced per AR step (via codec_lm_step_generate),
    // accumulated, and decoded by codec_decode_quantized_representation.
    bool            is_continuous;
    int32_t         patch_size;
    int32_t         latent_dim;
};

// Return CODEC_STATUS_NOT_SUPPORTED via NULL when the codec_model has
// no LM adaptor section (the `codec.lm.has_adaptor` GGUF metadata key
// is absent or false).  Borrows everything from `codec`: backend, mmap,
// weights lm_ggml_context, eval scheduler.  Does not duplicate weights.
struct codec_lm * codec_lm_create(struct codec_model * codec);
void              codec_lm_free  (struct codec_lm * lm);

// Returned struct lives for the lifetime of the codec_lm.
const struct codec_lm_info * codec_lm_get_info(const struct codec_lm * lm);

// Returns a pointer to the last-error string (empty when no error has
// been recorded).  For diagnostic use; the buffer lives as long as
// `lm` does.
const char * codec_lm_get_last_error(const struct codec_lm * lm);
const char * codec_lm_state_get_last_error(const struct codec_lm_state * st);

// Returns the most recent `codec_lm_create` failure reason on the
// current thread, or an empty string if no recent failure was recorded.
// Useful for surfacing the gate / init error when create returned NULL
// and the caller has no lm handle to query.  Lives in thread-local
// storage; subsequent successful or failed creates overwrite it.
const char * codec_lm_get_create_error(void);

// Per-generation state.  Holds:
//   * residual_depth_ar — KV cache for the depth decoder (reset every
//     `codec_lm_step_begin` since CSM/Moshi/etc. reset the depth
//     decoder cache per AR step) and intermediate hidden states for
//     intra-step conditioning.
//   * parallel_heads_delay — delay shift register (`prev_codes[i]`
//     buffered for `delay[i]` steps before they're visible at the
//     codebook-i output position).
// Multiple states can coexist on the same codec_lm for parallel
// generations, each with their own caches.
struct codec_lm_state * codec_lm_state_new (struct codec_lm * lm);
void                    codec_lm_state_free(struct codec_lm_state * st);
void                    codec_lm_state_reset(struct codec_lm_state * st);

// For models with `c0_input_modality="text"` (Moshi), the caller must
// stash the text token (sampled from the backbone's text head) before
// calling `codec_lm_step_begin`.  The token is consumed at depth
// position 0.  Always succeeds for text-modality models; a no-op for
// audio-modality models (the value is recorded but unused).
enum codec_status codec_lm_state_set_text_context(
    struct codec_lm_state * st,
    int32_t                 text_token);

// ─────────────────────────────────────────────────────────────────────
// Audio embedding lookup.
// ─────────────────────────────────────────────────────────────────────

// Look up the audio embedding row for codebook `cb_idx` and code
// `code`.  Returns a pointer into the model's weight buffer (read-only;
// lifetime = lifetime of `lm`).  Returns NULL when the arguments are
// out of range.
//
// `code == -1` is reserved for "skip / pad" by callers and is rejected
// by this lookup with NULL — use codec_lm_compose_audio_embd to do a
// pad-aware sum.
const float * codec_lm_audio_embd(
    struct codec_lm * lm,
    int32_t           cb_idx,
    int32_t           code);

// Sum-of-codebook compose: write `sum_i audio_embd[i][codes[i]]` into
// `out_embd[hidden_dim]`.  `codes[i] == -1` means "skip codebook i"
// (treated as a zero contribution — same as multiplying that channel's
// embedding by zero).  All-skip is allowed and writes a zero buffer.
//
// Caller must size `out_embd` to at least `info->hidden_dim` floats.
enum codec_status codec_lm_compose_audio_embd(
    struct codec_lm *  lm,
    const int32_t *    codes,        // [n_codebook]
    float *            out_embd);    // [hidden_dim]

// Compose the next-step backbone input embedding.  For models that have
// a learned per-step positional embedding (Chatterbox T3's
// `speech_pos_emb` is the canonical example), `step` indexes into that
// table and is added on top of `compose_audio_embd`.  For models without
// such a table (CSM, generic parallel-heads-delay TTS), `step` is
// ignored and this is identical to `compose_audio_embd`.
//
// This is the embedding the host feeds back into the LM as
// `inputs_embeds` for the next decode call (Type B / embed-override
// inference pattern).
//
// Caller must size `out_embd` to at least `info->hidden_dim` floats.
enum codec_status codec_lm_compose_next_embd(
    struct codec_lm *  lm,
    const int32_t *    codes,        // [n_codebook]
    int32_t            step,         // AR step index for learned pos emb
    float *            out_embd);    // [hidden_dim]

// ─────────────────────────────────────────────────────────────────────
// Per-AR-step state machine: hidden -> codebook logits -> codes.
//
// Usage:
//
//   codec_lm_step_begin(state, h);
//   for (int k = 0; k < info->n_codebook; ++k) {
//       int32_t cb_idx, n;
//       const float * logits = codec_lm_step_logits(state, &cb_idx, &n);
//       int32_t code = caller_sample(cb_idx, logits, n);
//       codec_lm_step_push_code(state, code);
//   }
//   int32_t codes[N];
//   codec_lm_step_finish(state, codes);
//
// Invariants:
//   * Exactly one `step_begin`, then `step_logits + step_push_code`
//     called in that order `n_codebook` times, then exactly one
//     `step_finish`.  Calling out of order returns INVALID_STATE.
//   * For `parallel_heads_delay`, all N logits arrays are computed
//     during `step_begin`; `step_logits` just hands out pointers and
//     `step_push_code` records the sampled value into the delay
//     register.
//   * For `residual_depth_ar`, `step_logits(k=0)` returns c0 logits
//     computed from the backbone hidden alone; `step_push_code(c0)`
//     advances the depth decoder by one position; the next
//     `step_logits` returns c1 logits, and so on.
//   * Logits pointers returned by `step_logits` are owned by the state
//     and remain valid only until the next `step_logits` /
//     `step_push_code` / `step_finish` / `step_begin` call.
//
// Sampling is intentionally outside the API: caller wraps the returned
// logits in whatever sampler stack they want.  The expected pattern
// for llama.cpp users is:
//
//   llama_token_data * arr = alloca(n * sizeof *arr);
//   for (int i = 0; i < n; ++i) arr[i] = (llama_token_data){i, logits[i], 0};
//   llama_token_data_array cur = { arr, (size_t)n, -1, false };
//   llama_sampler_apply(audio_chains[cb_idx], &cur);
//   int32_t code = arr[cur.selected].id;
// ─────────────────────────────────────────────────────────────────────

enum codec_status codec_lm_step_begin(
    struct codec_lm_state * st,
    const float *           h_in);   // [hidden_dim]

bool codec_lm_step_pending(const struct codec_lm_state * st);

// Returns logits for the next codebook in sequence.  Out-params
// receive the codebook index (always 0,1,2,... in order) and the
// length of the logits array (= codebook_sizes[cb_idx]).  Returns
// NULL when called out of phase (e.g., before step_begin or after
// step_finish, or before push_code for the prior codebook on a kind
// that requires it).
const float * codec_lm_step_logits(
    struct codec_lm_state * st,
    int32_t *               out_cb_idx,
    int32_t *               out_n);

// Push the code sampled from the most recent step_logits.  Code value
// must be in `[0, codebook_sizes[cb_idx])`.  Returns INVALID_ARG for
// out-of-range codes; INVALID_STATE if no step_logits is pending.
enum codec_status codec_lm_step_push_code(
    struct codec_lm_state * st,
    int32_t                 code);

// Read all sampled codes for this step into `out_codes[n_codebook]`
// and reset the state machine to "between steps".  After this call,
// step_begin can be invoked again.  Returns INVALID_STATE if not all
// codebooks have been pushed.
enum codec_status codec_lm_step_finish(
    struct codec_lm_state * st,
    int32_t *               out_codes);  // [n_codebook]

// ─────────────────────────────────────────────────────────────────────
// Continuous-latent step machine (CONTINUOUS_LATENT_CFM kind only).
//
// Replaces the codebook step machine.  One call runs the whole generation
// step internally — tslm_adapter + FSQ + RALM step + LocDiT CFM diffusion —
// and writes one latent patch plus a stop flag.  No logits, no sampling.
//
//   h_in        : [hidden_dim] backbone (Barbet) hidden for this position.
//   cfg_value   : classifier-free-guidance strength (e.g. 2.0).
//   n_timesteps : CFM Euler steps (e.g. 10).
//   noise       : [patch_size*latent_dim] CFM init noise, or NULL to sample
//                 internally (pass a buffer for deterministic / parity runs).
//   out_patch   : [patch_size*latent_dim] generated latent frames (accumulate
//                 these across steps, then codec_decode_quantized_representation).
//   out_stop    : set to 1 when the stop head fires, else 0.
enum codec_status codec_lm_step_generate(
    struct codec_lm_state * st,
    const float *           h_in,
    float                   cfg_value,
    int32_t                 n_timesteps,
    const float *           noise,
    float *                 out_patch,
    int32_t *               out_stop);

// Feedback embedding for the NEXT backbone step: LocEnc(last patch) projected
// into the backbone hidden space (`enc_to_tslm_proj`).  Valid only after a
// codec_lm_step_generate call.  out_embd : [hidden_dim].
enum codec_status codec_lm_step_feedback_embd(
    struct codec_lm_state * st,
    float *                 out_embd);

// Prefill the RALM (residual LM) over the whole prompt prefix before the first
// codec_lm_step_generate.  The reference _inference runs the RALM causally over
// every prefix position: at each position it forms
//   enc_out  = fsq(tslm_adapter(h)) on audio positions, tslm_adapter(h) (no FSQ)
//              on text positions;
//   ralm_in  = fusion_concat_proj(concat(enc_out, audio_mask * feat_embed_lm)).
// codec_lm handles ONLY the text-position semantics here (audio prefix
// continuation is not yet wired): every prefill position is treated as a text
// position, so enc_out = tslm_adapter(h) WITHOUT FSQ and the feat_embed_lm half
// of the fusion input is zero.  This matches zero-shot ("null" speaker) prompts,
// which is what the current host path drives.
//
// The K/V for all `n_pos` positions is written into the persistent RALM cache,
// kv_pos is set to n_pos, and the LAST position's (lm_hidden, residual_hidden)
// are cached so the FIRST subsequent codec_lm_step_generate consumes them
// directly (the `<|audio_start|>` text position → non-FSQ lm_hidden) and runs no
// RALM step of its own (matching the reference's iteration-0 semantics).
//
//   hiddens    : [n_pos * hidden_dim] backbone (Barbet) hiddens, position-major
//                (position p's hidden at offset p*hidden_dim).
//   n_pos      : number of prefix positions (T).
//   hidden_dim : must equal info->hidden_dim.
//
// CONTINUOUS_LATENT_CFM only; other kinds return CODEC_STATUS_NOT_SUPPORTED.
// After this call the state is "primed"; codec_lm_state_reset clears it.
enum codec_status codec_lm_text_prefill(
    struct codec_lm_state * st,
    const float *           hiddens,
    int32_t                 n_pos,
    int32_t                 hidden_dim);

// Configure the min_len stop guard for continuous-latent generation.  The
// reference suppresses the stop flag for patches 0..min_len (0-based patch
// index, `if i > min_len and stop == 1: break`), so a stop head that fires
// early on the first few patches is ignored.  Default is read from the GGUF
// key `codec.lm.min_len` (falling back to 2).  Set n < 0 to restore the
// GGUF/default value.  CONTINUOUS_LATENT_CFM only; no-op for other kinds.
enum codec_status codec_lm_set_continuous_min_len(
    struct codec_lm_state * st,
    int32_t                 min_len);

// Teacher-force the NEXT codec_lm_step_generate's trajectory (parity testing).
// The supplied reference latent `patch` [patch_size*latent_dim] replaces the
// codec's own generated patch as the cond for the next step AND as the LocEnc
// feedback source for THIS step — so codec replays the reference trajectory
// exactly and every emitted patch stays comparable to the reference (free-
// running feedback diverges chaotically after a few steps under F16).  The
// patch the graph emits is still codec's own.  Consumed once per step; re-arm
// before each step.  Pass NULL to disarm.  CONTINUOUS_LATENT_CFM only.
enum codec_status codec_lm_set_teacher_patch(
    struct codec_lm_state * st,
    const float *           patch,
    int32_t                 n);

// =====================================================================
// Speaker-conditioning encoder.
//
// Generic across voice-clone families (Chatterbox cond_enc, Qwen3-TTS
// speaker_encoder, MOSS-TTSD x-vector projector, LFM2-Audio speaker
// prefix, …).  Each codec model declares its required inputs + output
// shape via `codec_lm_speaker_info`; the caller (typically the LM
// arch on the llama.cpp side) decides how to consume the resulting
// `(n_rows, hidden_dim)` matrix — prefix concat, additive overlay,
// cross-attention KV, etc.
//
// Returns NULL via `codec_lm_speaker_get_info` when the loaded
// codec_model has no speaker section (most codecs don't —
// `codec.speaker.has_encoder` GGUF metadata key absent or false).
// =====================================================================

struct codec_lm_speaker_info {
    // Required inputs.  Each codec declares which it consumes; the
    // caller passes the corresponding argument to
    // `codec_lm_speaker_encode` (NULL allowed for optional ones).
    bool    needs_ref_pcm;             // ref audio PCM (mono).  Sample
                                       // rate the encoder works at is
                                       // declared via `ref_sample_rate`.
    bool    needs_ref_speech_tokens;   // pre-encoded via codec_encode
                                       // (e.g. Chatterbox S3T codes).
    bool    needs_emotion_scalar;      // single float in [0, 1] (model-
                                       // specific semantics).  When the
                                       // caller passes NULL,
                                       // `emotion_default` is used.

    // Working sample rate the encoder expects on `ref_pcm` when
    // `needs_ref_pcm` is set.  Caller is responsible for resampling.
    // Zero when `needs_ref_pcm = false`.
    int32_t ref_sample_rate;

    // Default value the runtime substitutes when the caller passes NULL
    // for `emotion`.  Pulled from `codec.speaker.emotion_default` GGUF
    // metadata, otherwise 0.5.  Meaningful only when
    // `needs_emotion_scalar = true`.
    float   emotion_default;

    // Output shape.  How the LM consumes this is the LM arch's decision
    // — codec_lm does not bake "prefix" / "vector" / etc. semantics
    // into this field.
    int32_t n_rows;
    int32_t hidden_dim;

    // Width of the intermediate speaker-embedding vector this codec's
    // audio encoder produces.  Useful when the caller has a cached
    // embedding from a prior call (or from a pickled file like
    // Chatterbox's conds.pt) and wants to feed it back in via
    // `codec_lm_speaker_encode_from_embedding`.  Zero when the codec
    // doesn't expose a usable intermediate (i.e. when only the full
    // ref_pcm-driven path is supported).
    int32_t speaker_emb_dim;
};

// Returns NULL when no speaker section is present.  Lifetime = lifetime
// of `lm`.
const struct codec_lm_speaker_info * codec_lm_speaker_get_info(
    const struct codec_lm * lm);

// Run the speaker-conditioning encoder.
//
// Arguments are validated against `codec_lm_speaker_info`:
//   * required inputs that are NULL / zero-length → INVALID_ARG
//   * NULL `emotion` is substituted with `info->emotion_default`
//
// `out` must have room for at least `info->n_rows * info->hidden_dim`
// floats and is written as F32.  `out_n_elems` is the caller's buffer
// capacity in elements (used for a size check; runtime never writes
// past it).  Returns NOT_SUPPORTED when the codec has no speaker
// section (mirror of `codec_lm_speaker_get_info == NULL`).
enum codec_status codec_lm_speaker_encode(
    struct codec_lm *          lm,
    const struct codec_audio * ref_pcm,             // OPTIONAL per info
    const int32_t *            ref_speech_tokens,   // OPTIONAL per info
    int32_t                    n_ref_speech_tokens,
    const float *              emotion,             // NULL = use default
    float *                    out,                 // [n_rows × hidden_dim]
    int32_t                    out_n_elems);

// Like `codec_lm_speaker_encode` but takes a pre-computed
// `speaker_emb` directly (skipping the audio encoder front-end).
// Useful when the caller has cached the embedding from a prior call
// (Chatterbox's `conds.pt` pickles the post-VE 256-d vector this way)
// or has computed it via an out-of-band path.
//
// `speaker_emb_dim` must match `info->speaker_emb_dim` exactly;
// callers should query that field before calling.  Returns
// CODEC_STATUS_NOT_SUPPORTED on codecs whose encoder doesn't expose a
// usable intermediate (info->speaker_emb_dim == 0).
enum codec_status codec_lm_speaker_encode_from_embedding(
    struct codec_lm *          lm,
    const float *              speaker_emb,
    int32_t                    speaker_emb_dim,
    const int32_t *            ref_speech_tokens,   // OPTIONAL per info
    int32_t                    n_ref_speech_tokens,
    const float *              emotion,             // NULL = use default
    float *                    out,
    int32_t                    out_n_elems);

#ifdef __cplusplus
}
#endif

#endif // CODEC_LM_H
