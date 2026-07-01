#ifndef CODEC_COMMON_H
#define CODEC_COMMON_H

// Generic audio-LM API for codec.cpp.  See docs/codec_common_api.md
// for the full design.  Mirrors llama.cpp's `common/` layer pattern:
// the host (llama.rn's rn-tts, examples/tts-cli, …) keeps full
// ownership of the `llama_decode` loop, the sampler, and KV
// management; this layer provides build-time + per-step + end-of-
// sequence hooks that the host calls at three points in its existing
// AR control flow.

#include "codec.h"
#include "codec_lm.h"

#include <cstdint>
#include <map>
#include <string>
#include <vector>

// We don't depend on llama.cpp here — `llama_token` aliases its public
// type for ABI compatibility but the value is just an int32 to us.  The
// host's translation unit can include both <llama.h> and this header.
using codec_common_token = int32_t;

namespace codec_common {

// ─────────────────────────────────────────────────────────────────────
// Modality
// ─────────────────────────────────────────────────────────────────────
// Bitmask read from GGUF metadata under `codec.lm.modality.*`.  Hosts
// use it to decide whether to enable the audio path, expose a ref-audio
// input field, set up `decode_audio` at end-of-sequence, etc.  Each
// model declares its own modality at convert time — no host-side
// hardcoded model lists.
enum modality_flag : uint32_t {
    INPUT_TEXT   = 1u << 0,   // model consumes a text prompt
    INPUT_AUDIO  = 1u << 1,   // model consumes ref / prompt audio
    OUTPUT_TEXT  = 1u << 2,   // model emits text (audio chat reply)
    OUTPUT_AUDIO = 1u << 3,   // model emits speech (TTS / audio chat)
};

// ─────────────────────────────────────────────────────────────────────
// Params + I/O types
// ─────────────────────────────────────────────────────────────────────

struct audio_lm_params {
    std::string codec_path;             // codec / codec_lm GGUF path
    bool        use_gpu      = false;
    int32_t     n_threads    = 0;       // 0 → codec.cpp default
};

// Generic input descriptor.  Fill the fields the caller has; the model
// uses what its modality + speaker_info declare.  Unknown / unused
// fields are ignored — no errors for over-supply.
struct audio_lm_input {
    // Reference audio (voice clone, audio chat input, ICL prompt, …).
    const float * ref_pcm         = nullptr;
    int32_t       ref_n_samples   = 0;
    int32_t       ref_sample_rate = 0;

    // Pre-computed speaker embedding (skip the audio encoder front-end).
    // Useful when the embedding is cached (e.g. Chatterbox conds.pt).
    const float * speaker_emb     = nullptr;
    int32_t       speaker_emb_dim = 0;

    // Emotion / expressivity scalar in [0, 1].  NULL → model default.
    const float * emotion         = nullptr;

    // ICL voice-clone prompt: reference transcript paired with ref_pcm.
    std::string   ref_text;

    // Synthesis target text.
    std::string   text;

    // Model-specific knobs (language_id, speaker_id, x_vector_only_mode,
    // …).  Keys + semantics live in docs/codec_common_input_keys.md
    // (to be authored alongside the first 2–3 model integrations).
    std::map<std::string, std::string> extra;
};

// Built prompt: tokens (host feeds via `llama_decode`) + optional
// prefix embeddings (`inputs_embeds` path) for models that prepend a
// conditioning prefix.  Plus the sampler defaults the model was
// trained with — the host may override.
struct audio_lm_prompt {
    // Token ids the host concatenates onto its `embd` for `llama_decode`.
    std::vector<codec_common_token> tokens;

    // Optional: prefix embeddings prepended via the `inputs_embeds`
    // path.  Empty for Type A token-only models (OuteTTS / Orpheus).
    // Shape: `embeds_prefix_rows × embeds_prefix_hidden`, row-major.
    std::vector<float> embeds_prefix;
    int32_t            embeds_prefix_rows   = 0;
    int32_t            embeds_prefix_hidden = 0;

    // CFG-using models (Chatterbox T3): unconditional branch.  Same
    // shape as `embeds_prefix` when present; empty when CFG isn't used.
    std::vector<float> embeds_uncond;

    // Sampler hints — model's training-time defaults.  Host may
    // override; presented for one-call convenience.
    float                 default_temperature        = 0.0f;
    float                 default_top_p              = 0.0f;
    float                 default_min_p              = 0.0f;
    float                 default_repetition_penalty = 0.0f;
    float                 default_cfg_weight         = 0.0f;   // 0 = no CFG
    codec_common_token    start_token                = -1;    // -1 = no seed
    codec_common_token    stop_token                 = -1;    // -1 = use vocab EOS
};

struct audio_lm_audio_output {
    std::vector<float> pcm;            // interleaved when n_channels > 1
    int32_t            sample_rate = 0;
    int32_t            n_channels  = 1;
};

// Per-step verdict returned by `observe_token`.  Tells the host how to
// set up the NEXT `llama_decode` call.  See docs/codec_common_api.md
// for the full semantics table.
enum observe_action {
    OBSERVE_PASSTHROUGH,      // text token; render + standard token-batch path
    OBSERVE_CONSUMED,         // audio token (Type A); no render; token-batch path
    OBSERVE_CONSUMED_EMBED,   // audio token (Type B/C/D); no render; embd-batch
                              //                          path via get_next_embed
    OBSERVE_STOP,             // model emitted stop / codec_lm done
};

// ─────────────────────────────────────────────────────────────────────
// Opaque context
// ─────────────────────────────────────────────────────────────────────
struct audio_lm_context;

// ─────────────────────────────────────────────────────────────────────
// Lifecycle
// ─────────────────────────────────────────────────────────────────────
//
// `audio_lm_init` loads the codec + codec_lm from the same GGUF (or
// returns success for codec-only files, in which case the AR hooks
// remain available but report NOT_SUPPORTED at observe / build).  On
// failure returns nullptr and writes the reason into `*err` if non-null.

audio_lm_context * audio_lm_init (const audio_lm_params & p, std::string * err = nullptr);
void               audio_lm_free (audio_lm_context * ctx);

// Drop per-sequence state (accumulated codes, internal step machine,
// pending next_embed buffer).  Capabilities / loaded weights stay.
// Call between consecutive generations on the same context.
void               audio_lm_reset(audio_lm_context * ctx);

// ─────────────────────────────────────────────────────────────────────
// Capability queries (computed once at init, cheap)
// ─────────────────────────────────────────────────────────────────────
uint32_t     audio_lm_modality       (const audio_lm_context * ctx);
bool         audio_lm_has_speaker_enc(const audio_lm_context * ctx);
int32_t      audio_lm_n_codebook     (const audio_lm_context * ctx);
int32_t      audio_lm_hidden_dim     (const audio_lm_context * ctx);
const char * audio_lm_last_error     (const audio_lm_context * ctx);

// ─────────────────────────────────────────────────────────────────────
// Prompt build (one shot, before the AR loop)
//
// Reference impls per inference Type land in steps 2–5 of the roadmap.
// Returns false + populates last_error when the model doesn't support
// build_prompt yet.
// ─────────────────────────────────────────────────────────────────────
bool audio_lm_build_prompt(audio_lm_context * ctx,
                            const audio_lm_input  & in,
                            audio_lm_prompt       * out);

// ─────────────────────────────────────────────────────────────────────
// Type A audio-token-range config
//
// Type A models (OuteTTS, Orpheus, …) encode their audio as ordinary
// LM tokens sitting in a contiguous slice of the host vocabulary.
// observe_token detects "this tok is an audio code" via:
//
//   `tok ∈ [offset, offset + count)`  → audio code, value = tok - offset
//   `tok == eos_id` (when ≥ 0)         → end of audio (OBSERVE_STOP)
//   otherwise                          → text (OBSERVE_PASSTHROUGH)
//
// The values are read at init from GGUF metadata:
//
//   codec.audio_token.offset  (uint32)   — first audio-token ID
//   codec.audio_token.count   (uint32)   — number of audio tokens
//   codec.audio_token.eos_id  (int32)    — end-of-audio sentinel (-1 = none)
//
// Hosts that need to override (test rigs, models without the keys
// baked) can call `audio_lm_set_audio_token_range` after init.
// `offset < 0` disables Type A dispatch entirely (every token returns
// OBSERVE_PASSTHROUGH).
// ─────────────────────────────────────────────────────────────────────
void audio_lm_set_audio_token_range(
        audio_lm_context * ctx,
        int32_t            offset,
        int32_t            count,
        int32_t            eos_id);

void audio_lm_get_audio_token_range(
        const audio_lm_context * ctx,
        int32_t * out_offset,
        int32_t * out_count,
        int32_t * out_eos_id);

// ─────────────────────────────────────────────────────────────────────
// Type B embed-override flag
//
// When `true`, observe_token treats audio-range tokens as Type B
// (Chatterbox-style): in addition to accumulating the code, it composes
// the next backbone-input embedding (`speech_emb[code] + speech_pos_emb
// [step]`) into the context's internal buffer and returns
// OBSERVE_CONSUMED_EMBED so the host knows to switch to its
// `inputs_embeds` decode path for the next step.  When `false` (the
// default), the same audio-range tokens dispatch as Type A and the host
// keeps using the standard token-id decode path.
//
// The step counter is internal: it starts at `start_step` on reset and
// increments by 1 each time a Type B audio token is consumed.  Pass the
// initial position (Chatterbox starts at 1 because pos 0 is the
// start_speech_token's slot in the prefill).
// ─────────────────────────────────────────────────────────────────────
void audio_lm_set_uses_embed_override(audio_lm_context * ctx,
                                       bool    enabled,
                                       int32_t start_step);

bool audio_lm_get_uses_embed_override(const audio_lm_context * ctx);

// ─────────────────────────────────────────────────────────────────────
// Per-step observe (called by host after each backbone token sample)
//
// `last_hidden` is the backbone's hidden state at the just-sampled
// position (typically `llama_get_embeddings_ith(ctx, -1)`).  Required
// for Type C/D (depth decoder / parallel heads sample from it);
// ignored for Type A/B — pass nullptr.
// ─────────────────────────────────────────────────────────────────────
observe_action audio_lm_observe_token(
        audio_lm_context * ctx,
        codec_common_token tok,
        const float *      last_hidden,
        int32_t            hidden_dim);

// Only valid immediately after OBSERVE_CONSUMED_EMBED.  Points into
// `ctx`'s internal buffer; valid until the next observe_token / reset /
// free call.  Hosts that need to retain the vector across calls memcpy
// it out.
const float * audio_lm_get_next_embed(const audio_lm_context * ctx,
                                       int32_t * out_dim);

// ─────────────────────────────────────────────────────────────────────
// Continuous-latent per-step observe (BlueMagpie / VoxCPM)
//
// For continuous-latent models (codec_lm kind continuous_latent_cfm) the
// backbone emits a hidden state per step (no token, no codebook).  The host
// calls this with that hidden; codec_common runs the whole adaptor step
// (tslm_adapter + FSQ + RALM + LocDiT CFM diffusion) internally, accumulates
// the produced latent patch, and returns:
//   OBSERVE_CONSUMED_EMBED — feed `audio_lm_get_next_embed()` (the LocEnc
//                            feedback) as the next backbone input embedding.
//   OBSERVE_STOP           — the stop head fired; break and call decode_audio.
//
//   hidden     : [hidden_dim] backbone hidden (e.g. llama_get_embeddings_ith).
//   noise      : [patch_size*latent_dim] CFM init noise, or NULL to sample
//                (pass a buffer for deterministic / reproducible output).
observe_action audio_lm_observe_hidden(audio_lm_context * ctx,
                                       const float * hidden, int32_t hidden_dim,
                                       const float * noise);

// True when the loaded model is a continuous-latent kind (use observe_hidden
// instead of observe_token / observe_codes).
bool audio_lm_is_continuous(const audio_lm_context * ctx);

// Set CFG strength + diffusion steps for the continuous path (defaults 2.0 / 10).
void audio_lm_set_continuous_params(audio_lm_context * ctx, float cfg_value, int32_t n_timesteps);

// ─────────────────────────────────────────────────────────────────────
// Multi-codebook frame observe (Type C / Type D)
//
// For models that emit `n_codebook` codes per backbone step:
//
//   * Type C — residual depth-AR (CSM, Qwen3-TTS):  the host samples c0
//     from the backbone head, drives the codec_lm step machine
//     (step_begin → step_logits/push_code × (n_cb-1) → step_finish) to
//     produce c1..c{n-1}, then calls this with all N codes.
//   * Type D — parallel heads with delay (MOSS-TTSD):  the host samples
//     all N codes from N parallel heads, then calls this directly with
//     the full frame.
//
// codec_common just accumulates the frame into the per-sequence buffer
// and (when `uses_embed_override` is set) composes the next backbone-
// input embedding via `codec_lm_compose_next_embd` so the host can feed
// it back as `inputs_embeds`.
//
//   codes        — one frame of length n_codes
//   n_codes      — must equal `audio_lm_n_codebook(ctx)` for codec_lm-
//                  backed models, or 1 for codec-only Type A round-trips
//                  (in which case prefer observe_token instead).
//   last_hidden  — backbone hidden at the just-sampled position; required
//                  when uses_embed_override is set AND the kind's
//                  compose_next_embd needs it; ignored otherwise.  Pass
//                  nullptr when the host has no hidden to share.
//
// Returns OBSERVE_CONSUMED_EMBED when uses_embed_override is set and
// the embed compose succeeds; OBSERVE_CONSUMED otherwise; OBSERVE_STOP
// on a hard error (last_error gets the reason).
// ─────────────────────────────────────────────────────────────────────
observe_action audio_lm_observe_codes(
        audio_lm_context * ctx,
        const int32_t    * codes,
        int32_t            n_codes,
        const float      * last_hidden,
        int32_t            hidden_dim);

// ─────────────────────────────────────────────────────────────────────
// External codes push (offline / debug / parity round-trip)
//
// Append `n_frames * n_q` codes to the context's per-sequence
// accumulator.  Codes are interleaved (T, n_q): `codes[t*n_q + q]`.
// `n_q` must match `audio_lm_n_codebook(ctx)`.  Used by tooling that
// already has the codes (e.g. tts-cli's `decode` subcommand) and just
// wants to flow them through codec_common's `decode_audio`.
//
// `audio_lm_observe_token` (step 3+) appends codes internally as the
// AR loop runs, so the same accumulator backs both flows.
// ─────────────────────────────────────────────────────────────────────
bool audio_lm_push_codes(audio_lm_context * ctx,
                         const int32_t   * codes,
                         int32_t           n_frames,
                         int32_t           n_q);

// ─────────────────────────────────────────────────────────────────────
// End of sequence
//
// Decode accumulated audio codes → PCM.  Valid only when modality has
// OUTPUT_AUDIO and the AR loop has produced at least one audio frame.
// Returns false + populates last_error otherwise.
// ─────────────────────────────────────────────────────────────────────
bool audio_lm_decode_audio(audio_lm_context * ctx, audio_lm_audio_output * out);

}  // namespace codec_common

#endif  // CODEC_COMMON_H
