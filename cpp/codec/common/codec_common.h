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

// Borrowed handle to the underlying codec_lm (NULL when none loaded).
// Lets a host reach model-specific helpers not wrapped by codec_common
// (e.g. the Chatterbox T3 `codec_lm_chatterbox_*` prompt/tokenizer API).
struct codec_lm * audio_lm_get_lm(audio_lm_context * ctx);

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
// codec_lm end-of-audio metadata (codebook-AR kinds).
//
// Surfaces the codec_lm's `codec.lm.eos_code_c0` / `codec.lm.eos_min_step`
// (see codec_lm.h).  This is the model-owned end-of-audio decision the
// host used to hardcode; `audio_lm_observe_codes` already acts on it, but
// hosts may want to read the raw values (e.g. to display / log the stop
// code).  Writes -1 / 0 when no codec_lm is loaded or no sentinel exists.
// Any out-pointer may be NULL.
// ─────────────────────────────────────────────────────────────────────
void audio_lm_get_lm_eos(
        const audio_lm_context * ctx,
        int32_t * out_eos_code_c0,
        int32_t * out_eos_min_step);

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

// Set CFG strength + diffusion steps + min_len stop guard for the continuous
// path (defaults 2.0 / 10 / -1).  min_len < 0 leaves the model default (GGUF
// `codec.lm.min_len`, else 2): the stop flag is ignored for patches 0..min_len.
void audio_lm_set_continuous_params(audio_lm_context * ctx, float cfg_value,
                                    int32_t n_timesteps, int32_t min_len = -1);

// Prefill the RALM over the whole prompt prefix before the first
// audio_lm_observe_hidden.  `hiddens` is [n_pos * hidden_dim] backbone hiddens
// (position-major).  Continuous-latent kinds only; returns false otherwise.
// After this the first observe_hidden consumes the primed prefill state.
bool audio_lm_text_prefill(audio_lm_context * ctx, const float * hiddens,
                           int32_t n_pos, int32_t hidden_dim);

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

// ═════════════════════════════════════════════════════════════════════
// Host-driven AR (Phase B): prompt assembly + codebook step machine
//
// These entries let a host that owns a llama.cpp backbone drive the full
// synthesize loop.  Model-specific prompt formats live here (keyed on
// `codec.lm.host_arch` / `codec.lm.kind` GGUF metadata) so the host stays
// model-agnostic; the host only supplies the backbone tokenizer + decode
// loop.  The codebook step machine (Type C/D) is exposed as thin
// passthroughs onto the internal codec_lm_state so the host never touches
// raw codec_lm handles.
// ═════════════════════════════════════════════════════════════════════

// Per-model prompt-assembly descriptor.  The host builds its backbone
// prompt as tokenize(prefix + input.text + suffix) with add_bos /
// parse_special as flagged, then decodes it to seed the AR loop.
struct audio_lm_prompt_info {
    // Coarse inference type, derived from `codec.lm.kind` + metadata.
    enum kind {
        KIND_UNKNOWN = 0,
        KIND_RESIDUAL_DEPTH_AR,   // Type C — CSM, Qwen3-TTS, MOSS-TTS-Realtime
        KIND_PARALLEL_HEADS_DELAY,// Type D — MOSS-TTSD
        KIND_CONTINUOUS_CFM,      // BlueMagpie / VoxCPM
    } model_kind = KIND_UNKNOWN;

    std::string host_arch;        // "llama" / "qwen3" / "barbet"
    std::string prompt_prefix;    // literal text prepended to input.text
    std::string prompt_suffix;    // literal text appended to input.text
    bool        add_bos       = false;  // ask the tokenizer to prepend BOS
    bool        parse_special  = false; // let special tokens in the template resolve

    // cb0 source.  When true (MOSS-TTSD delay-pattern text-modality cb0),
    // the host samples cb0 from the BACKBONE's own logits and calls
    // audio_lm_step_set_text_context(cb0) before step_begin; the step
    // machine then fills cb1..N-1.  When false (CSM / Qwen3-TTS /
    // MOSS-TTS-Realtime), the step machine produces all N codebooks.
    bool cb0_from_backbone = false;

    // audio_codebook_offset — codebooks [0, offset) are text/control and
    // are NOT decoded to audio; decode_audio slices them off.  0 for CSM /
    // Qwen3-TTS; 1 for MOSS-TTSD / MOSS-TTS-Realtime.
    int32_t audio_codebook_offset = 0;

    bool    is_continuous = false;  // use the observe_hidden path instead
    int32_t n_codebook    = 0;
    int32_t hidden_dim    = 0;
    int32_t eos_code_c0   = -1;     // cb0 EOS sentinel (-1 = none)
    int32_t eos_min_step  = 0;

    // Sampling defaults (rn-tts convention): temp 0.9 / top_p 0.95 /
    // top_k 50.  Host may override.  0 → greedy.
    float   default_temperature = 0.0f;
    float   default_top_p       = 0.0f;
    int32_t default_top_k       = 0;

    // ── Streaming text↔audio interleave (MOSS-TTS-Realtime) ─────────────
    // When true the per-backbone-step input is composed as
    //   text_embd[text_token] + compose_audio_embd(prev_frame_codes)
    // where the text lane is fed one token per audio frame (the reference's
    // streaming loop) rather than all text at prefill.  The spoken text
    // goes in the ASSISTANT turn as a `text_prefix`: the host prefills the
    // system+user+assistant-open context, then the first `prefill_text_len`
    // text tokens (audio lanes = audio_pad_code, the LAST prefill row's cb0
    // lane = bos_code_c0), and thereafter steps 1:1, filling the text lane
    // with `text_pad_id` once the text is exhausted, until cb0 == eos_code_c0.
    // The text embedding table lives in the backbone (not the codec), so the
    // host adds it externally on top of compose_audio_embd — see
    // `text_externally_added`.
    bool    streaming_interleave  = false;
    bool    text_externally_added = false;  // host adds text_embd[tok] on top
                                            // of compose_audio_embd
    int32_t prefill_text_len      = 12;     // text tokens before audio opens
    int32_t text_pad_id           = -1;     // text lane fill once text ends
    int32_t audio_pad_code        = -1;     // audio lane fill during text-only
    int32_t bos_code_c0           = -1;     // audio BOS in last prefill row cb0
    float   default_repetition_penalty = 1.0f;
    int32_t repetition_window          = 0;

    // ── Sequential text→audio (LFM2-Audio) ──────────────────────────────
    // When true the model is a *sequential* multimodal LM (LFM2-Audio):
    // after the ChatML prompt is prefilled, the backbone first free-runs in
    // TEXT modality (sampling text tokens from its OWN tied-embedding lm_head
    // and feeding them back as ordinary tokens) until it emits
    // `audio_start_id`, at which point it switches to AUDIO_OUT: the residual
    // depth decoder emits one N-codebook frame per step (feedback through
    // compose_audio_codes_embd on the backbone) and generation stops when cb0
    // samples eos_code_c0 (EOAudio) or the backbone emits `text_end_id`.
    // See liquid_audio.model.lfm2_audio.LFM2AudioModel.generate_sequential.
    bool    sequential_text_audio = false;
    int32_t audio_start_id        = -1;  // text token → switch to AUDIO_OUT
    int32_t text_end_id           = -1;  // text token → end turn / stop
    int32_t max_text_tokens       = 64;  // safety cap on the text warmup phase

    // ── Merged-cb0 speech sub-range (MOSS-TTSD) ─────────────────────────
    // For parallel-heads-delay models whose cb0 is a merged text+speech
    // vocab, speech codes occupy [cb0_speech_range_start, cb0_speech_range_end)
    // of the backbone vocab (from codec.lm.speech_token_range) and end-of-
    // audio is signalled by cb0 == eos_code_c0.  The host's auto-grammar
    // (tts_auto_grammar) uses these to constrain decode-phase cb0 sampling
    // to speech tokens ∪ {eos_code_c0}, keeping the model from drifting into
    // arbitrary text tokens (babble) mid-utterance.  Both -1 when absent.
    int32_t cb0_speech_range_start = -1;  // codec.lm.cb0_speech_offset
    int32_t cb0_speech_range_end   = -1;  // codec.lm.cb0_speech_range_end (exclusive)
};

// Fill `*out` from the loaded model's metadata.  Returns false + sets
// last_error when the model has no codec_lm (no AR profile to describe).
bool audio_lm_get_prompt_info(const audio_lm_context * ctx,
                              audio_lm_prompt_info    * out);

// ─────────────────────────────────────────────────────────────────────
// Auto-grammar (prompt/metadata-derived GBNF for the backbone sampler)
//
// Returns a GBNF grammar string derived from the model's metadata (the
// `pi` filled by audio_lm_get_prompt_info) + the synthesis text, or "" for
// models that don't benefit from one.  The grammar constrains the
// BACKBONE-logits sampler only (cb0-from-backbone / text-warmup tokens);
// it is meaningless for codec_lm audio-codebook heads (arbitrary float
// arrays, no vocab).
//
// Precedent: llama.rn's rn-tts build_dynamic_grammar (OuteTTS/Soprano/
// NeuTTS families).  The concept is ported; the first real application
// here is MOSS-TTSD cb0 (merged text+speech vocab): the grammar allows
// only the speech-token range [cb0_speech_range_start, cb0_speech_range_end)
// plus the end-of-audio sentinel eos_code_c0, so the backbone can't drift
// into arbitrary text tokens mid-utterance.  A non-speech cb0 token is the
// reference's natural stop trigger, so eos_code_c0 stays permitted.
//
// `text` is unused today (MOSS-TTSD's grammar is text-independent) but is
// threaded through for future prompt-dependent grammars (OuteTTS-style
// word-sequence constraints).
std::string tts_auto_grammar(const audio_lm_prompt_info & pi,
                             const std::string & text);

// ─── Codebook step machine passthroughs (Type C/D) ──────────────────
//
// Drive the internal codec_lm_state per AR step.  Sequence per frame:
//   [set_text_context(cb0)]        (MOSS-TTSD only, before step_begin)
//   step_begin(last_hidden)
//   for cb in 0..n_codebook-1:  step_logits(&idx,&n) → sample → step_push_code
//   step_finish(codes)            → codes[0..n_codebook-1]
// then hand `codes` to audio_lm_observe_codes for accumulation + compose.
// All return false / nullptr + set last_error on misuse.
bool          audio_lm_step_set_text_context(audio_lm_context * ctx, int32_t text_token);
bool          audio_lm_step_begin      (audio_lm_context * ctx, const float * last_hidden, int32_t hidden_dim);
const float * audio_lm_step_logits     (audio_lm_context * ctx, int32_t * out_cb_idx, int32_t * out_n);
bool          audio_lm_step_push_code  (audio_lm_context * ctx, int32_t code);
bool          audio_lm_step_finish     (audio_lm_context * ctx, int32_t * out_codes, int32_t n_codes);

// ─── Multi-modal prompt embedding (Type D / merged-cb0 models) ──────
//
// For parallel-heads-delay models whose cb0 is a merged text+speech vocab
// (MOSS-TTSD), the backbone prompt embedding is NOT the plain text-token
// embedding — it's the per-position sum over all n_codebook embedding
// tables with cb0 = the text token and cb1..N-1 = the speech-pad code
// (mirrors the HF processor's `_prepare_multi_modal_inputs`).  The host
// must feed these composed embeddings via the inputs_embeds path during
// prefill instead of the raw token path.
//
// Returns true iff this model needs the composed-prompt path; when false
// the host should prefill with the plain token path.  `out_needs_composed`
// is set accordingly even on the false branch.
bool audio_lm_prompt_needs_composed_embd(const audio_lm_context * ctx);

// Compose one prompt position's backbone input embedding for text token
// `text_token`.  Writes `hidden_dim` floats into `out_embd`.  cb0 = the raw
// text token (merged vocab, no speech offset), cb1..N-1 = speech-pad code.
bool audio_lm_compose_prompt_embd(audio_lm_context * ctx,
                                  int32_t            text_token,
                                  float *            out_embd,
                                  int32_t            out_dim);

// ─── Audio-lane compose (streaming residual_depth_ar) ───────────────
// Compose the audio-lane contribution of one backbone-step input from a
// full frame of `n_codes` codebook codes (the fused compose_audio_embd:
// Σ_k audio_embd_k[codes[k]]).  For MOSS-TTS-Realtime the text lane is
// added by the host on top (text_externally_added).  Writes `out_dim`
// (= hidden) floats.  Used to build the streaming prefill rows (audio-pad
// codes, plus BOS in the last row's cb0) and — equivalently to
// audio_lm_get_next_embed — the per-step previous-frame audio embed.
bool audio_lm_compose_audio_codes_embd(audio_lm_context * ctx,
                                       const int32_t *    codes,
                                       int32_t            n_codes,
                                       float *            out_embd,
                                       int32_t            out_dim);

// ─────────────────────────────────────────────────────────────────────
// Qwen3-TTS talker prompt assembly.
//
// The talker prompt is a two-lane additive prefix (see
// modeling_qwen3_tts.py::Qwen3TTSTalkerModel.generate):
//
//   text lane  = text_projection(text_embd[text_tok])   (projected, H)
//   codec lane = codec_embedding[control_tag]           (H)
//
// summed position-wise.  With auto-language + an ECAPA x-vector the
// prefix (role header + control stream) is:
//
//   0..2 : text_proj(role_tok[0..2])                     (codec lane empty)
//   3    : text_proj(tts_pad) + codec_embd[nothink]
//   4    : text_proj(tts_pad) + codec_embd[think_bos]
//   5    : text_proj(tts_pad) + codec_embd[think_eos]
//   6    : text_proj(tts_pad) + X-VECTOR
//   7    : text_proj(tts_bos) + codec_embd[codec_pad]
//   8    : text_proj(text[0]) + codec_embd[codec_bos]
//
// after which generation runs; the trailing text (text_proj(text[i]) for
// i>=1, then tts_eos) is injected per-step by the host via
// `audio_lm_talker_trailing_text_embd`.
//
// `role_tokens` are the tokenized "<|im_start|>assistant\n" header (host
// tokenizes them), `text_tokens` the payload text tokens.  `xvector` (may
// be null) is the ECAPA x-vector (hidden floats).  On success writes the
// prefix rows (row-major, `out_n_rows * hidden` floats) into `out_embds`
// (host sizes it to at least `(3 + 5 + n_text_prefix) * hidden`), sets
// `*out_n_rows`, and reports how many payload text tokens were consumed by
// the prefix in `*out_text_consumed` (1: text[0] summed at row 8).
bool audio_lm_talker_has_projection(const audio_lm_context * ctx);

bool audio_lm_build_talker_prefix(audio_lm_context * ctx,
                                  const int32_t *    role_tokens,
                                  int32_t            n_role,
                                  const int32_t *    text_tokens,
                                  int32_t            n_text,
                                  const float *      xvector,   // may be null
                                  int32_t            xvec_dim,
                                  float *            out_embds,
                                  int32_t            out_cap_rows,
                                  int32_t *          out_n_rows,
                                  int32_t *          out_text_consumed);

// Per-step trailing-text embedding: text_proj(text_tokens[i]) for the
// i-th trailing token, or text_proj(tts_eos) once the text is exhausted.
// Writes `hidden` floats into `out_embd`.  The host sums this with the
// audio next-embed before feeding the next backbone step.
bool audio_lm_talker_trailing_text_embd(audio_lm_context * ctx,
                                        const int32_t *    text_tokens,
                                        int32_t            n_text,
                                        int32_t            trailing_idx,
                                        float *            out_embd,
                                        int32_t            out_dim);

}  // namespace codec_common

#endif  // CODEC_COMMON_H
