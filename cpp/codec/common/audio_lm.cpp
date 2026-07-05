#include "codec_common.h"

#include <cstring>
#include <new>
#include <string>
#include <vector>

namespace codec_common {

// =====================================================================
// Internal context.  Holds the codec_model + codec_context + codec_lm
// handles plus the per-sequence state buffers.
// =====================================================================
struct audio_lm_context {
    codec_model    * model      = nullptr;
    codec_context  * codec_ctx  = nullptr;
    codec_lm       * lm         = nullptr;
    codec_lm_state * state      = nullptr;

    // Cached capabilities (read once at init).
    uint32_t modality_mask = 0;
    int32_t  n_cb          = 0;
    int32_t  hidden        = 0;
    bool     has_spk_enc   = false;

    // Type A audio-token range.  `audio_tok_offset < 0` → disabled
    // (every token surfaces as PASSTHROUGH).  Set from GGUF metadata
    // at init; hosts override via `audio_lm_set_audio_token_range`.
    int32_t  audio_tok_offset = -1;
    int32_t  audio_tok_count  = 0;
    int32_t  audio_tok_eos    = -1;

    // Type B embed-override toggle.  When true, observe_token composes
    // the next backbone-input embed via `codec_lm_compose_next_embd`
    // and returns OBSERVE_CONSUMED_EMBED.  `ar_step` is the position
    // counter passed to compose_next_embd (incremented on each consume);
    // `ar_step_start` is the value `reset()` restores it to.
    bool    uses_embed_override = false;
    int32_t ar_step             = 1;
    int32_t ar_step_start       = 1;

    // Per-sequence buffers (cleared on reset).
    //
    // `codes` is laid out (T, n_cb) interleaved: codes[t*n_cb + q] = code
    // for codebook q at frame t.  Matches codec_token_buffer convention.
    std::vector<int32_t> codes;
    int32_t              codes_n_frames = 0;

    // The vector returned by `audio_lm_get_next_embed`.  Owned by ctx;
    // valid until the next observe_token / reset / free.
    std::vector<float> next_embed_buf;
    int32_t            next_embed_dim = 0;

    // Continuous-latent models (BlueMagpie/VoxCPM, codec_lm kind
    // continuous_latent_cfm).  Each backbone step produces one latent patch
    // (patch_size × latent_dim) via codec_lm_step_generate, accumulated here;
    // decode_audio runs codec_decode_quantized_representation over them.
    bool    is_continuous   = false;
    int32_t patch_size      = 0;
    int32_t latent_dim      = 0;
    float   cont_cfg        = 2.0f;
    int32_t cont_timesteps  = 10;
    int32_t cont_min_len    = -1;   // -1 = use model default (GGUF codec.lm.min_len / 2)
    std::vector<float> latents;        // channel-major [latent_dim, n_frames]
    int32_t            latent_n_frames = 0;

    mutable std::string last_error;
};

// ─────────────────────────────────────────────────────────────────────
// Modality detection
// ─────────────────────────────────────────────────────────────────────
//
// Prefer explicit `codec.lm.modality.*` GGUF keys when present (set by
// the converter).  When absent (legacy GGUFs), infer from what the
// codec / codec_lm side actually supports:
//
//   * codec has decoder              → OUTPUT_AUDIO
//   * codec has encoder OR has_spk_enc → INPUT_AUDIO
//   * codec_lm present               → INPUT_TEXT (TTS-style models
//                                                   consume a text
//                                                   prompt)
//   * OUTPUT_TEXT  → only when explicitly declared (no robust signal
//                    to infer it from existing GGUFs).
//
// This keeps the API working today for the GGUFs we already shipped;
// new converters should write the explicit keys so the heuristic isn't
// needed.

static uint32_t read_modality_or_infer(audio_lm_context * ctx) {
    uint32_t mask = 0;

    const codec_gguf_metadata * meta = codec_model_metadata(ctx->model);
    bool saw_explicit = false;

    if (meta != nullptr) {
        for (size_t i = 0; i < meta->n_items; ++i) {
            const char * key = meta->items[i].key;
            const char * val = meta->items[i].value;
            if (key == nullptr || val == nullptr) continue;
            // Match "true" loosely — codec_gguf_metadata serialises
            // bools as the strings "true" / "false".
            const bool on = (std::strcmp(val, "true") == 0 ||
                             std::strcmp(val, "1")    == 0);
            if      (std::strcmp(key, "codec.lm.modality.input_text"  ) == 0) { if (on) mask |= INPUT_TEXT;   saw_explicit = true; }
            else if (std::strcmp(key, "codec.lm.modality.input_audio" ) == 0) { if (on) mask |= INPUT_AUDIO;  saw_explicit = true; }
            else if (std::strcmp(key, "codec.lm.modality.output_text" ) == 0) { if (on) mask |= OUTPUT_TEXT;  saw_explicit = true; }
            else if (std::strcmp(key, "codec.lm.modality.output_audio") == 0) { if (on) mask |= OUTPUT_AUDIO; saw_explicit = true; }
        }
    }

    // Type A audio-token range — independent of modality but pulled
    // from the same metadata scan to avoid a second pass.
    if (meta != nullptr) {
        for (size_t i = 0; i < meta->n_items; ++i) {
            const char * key = meta->items[i].key;
            const char * val = meta->items[i].value;
            if (key == nullptr || val == nullptr) continue;
            if      (std::strcmp(key, "codec.audio_token.offset") == 0) ctx->audio_tok_offset = std::atoi(val);
            else if (std::strcmp(key, "codec.audio_token.count")  == 0) ctx->audio_tok_count  = std::atoi(val);
            else if (std::strcmp(key, "codec.audio_token.eos_id") == 0) ctx->audio_tok_eos    = std::atoi(val);
        }
    }

    if (!saw_explicit) {
        if (codec_model_has_decoder(ctx->model)) mask |= OUTPUT_AUDIO;
        // INPUT_AUDIO heuristic: the LM consumes ref / prompt audio.
        // `codec_model_has_encoder` is too generous — bidirectional
        // codecs (Mimi, Qwen3-TTS-Tokenizer, XY-Tokenizer) expose an
        // encoder even when the LM is zero-shot (CSM).  Use the LM's
        // own speaker_encoder section as the only positive signal;
        // models that consume ref audio without an explicit speaker
        // encoder (MOSS-TTSD's interleaved speech-tokenizer prompt)
        // should set `codec.lm.modality.input_audio=true` at convert
        // time to opt in.
        if (ctx->has_spk_enc) mask |= INPUT_AUDIO;
        if (ctx->lm != nullptr) mask |= INPUT_TEXT;
    }

    return mask;
}

// =====================================================================
// Lifecycle
// =====================================================================

audio_lm_context * audio_lm_init(const audio_lm_params & p, std::string * err) {
    if (p.codec_path.empty()) {
        if (err) *err = "audio_lm_init: codec_path is empty";
        return nullptr;
    }
    auto * ctx = new (std::nothrow) audio_lm_context();
    if (ctx == nullptr) {
        if (err) *err = "audio_lm_init: out of memory";
        return nullptr;
    }

    auto mp = codec_model_default_params();
    mp.use_gpu   = p.use_gpu;
    if (p.n_threads > 0) mp.n_threads = p.n_threads;
    ctx->model = codec_model_load_from_file(p.codec_path.c_str(), mp);
    if (ctx->model == nullptr) {
        if (err) *err = "audio_lm_init: codec_model_load_from_file failed for " + p.codec_path;
        delete ctx;
        return nullptr;
    }

    auto cp = codec_context_default_params();
    ctx->codec_ctx = codec_init_from_model(ctx->model, cp);
    if (ctx->codec_ctx == nullptr) {
        if (err) *err = "audio_lm_init: codec_init_from_model failed";
        codec_model_free(ctx->model);
        delete ctx;
        return nullptr;
    }

    // LM adaptor is OPTIONAL.  codec-only GGUFs (e.g. wavtokenizer.gguf,
    // dac.gguf) still work for decode_audio paths — the AR-side hooks
    // just return NOT_SUPPORTED at observe / build time.
    ctx->lm = codec_lm_create(ctx->model);
    if (ctx->lm != nullptr) {
        ctx->state = codec_lm_state_new(ctx->lm);
        const codec_lm_info * info = codec_lm_get_info(ctx->lm);
        if (info != nullptr) {
            ctx->n_cb   = info->n_codebook;
            ctx->hidden = info->hidden_dim;
            ctx->is_continuous = info->is_continuous;
            ctx->patch_size    = info->patch_size;
            ctx->latent_dim    = info->latent_dim;
        }
        ctx->has_spk_enc = (codec_lm_speaker_get_info(ctx->lm) != nullptr);
    }

    ctx->modality_mask = read_modality_or_infer(ctx);
    return ctx;
}

void audio_lm_free(audio_lm_context * ctx) {
    if (ctx == nullptr) return;
    if (ctx->state)     codec_lm_state_free(ctx->state);
    if (ctx->lm)        codec_lm_free(ctx->lm);
    if (ctx->codec_ctx) codec_free(ctx->codec_ctx);
    if (ctx->model)     codec_model_free(ctx->model);
    delete ctx;
}

void audio_lm_reset(audio_lm_context * ctx) {
    if (ctx == nullptr) return;
    ctx->codes.clear();
    ctx->codes_n_frames = 0;
    ctx->next_embed_buf.clear();
    ctx->next_embed_dim = 0;
    ctx->ar_step = ctx->ar_step_start;
    ctx->latents.clear();
    ctx->latent_n_frames = 0;
    if (ctx->state) codec_lm_state_reset(ctx->state);
    ctx->last_error.clear();
}

// =====================================================================
// Capability queries
// =====================================================================

uint32_t audio_lm_modality(const audio_lm_context * ctx) {
    return ctx ? ctx->modality_mask : 0u;
}

bool audio_lm_has_speaker_enc(const audio_lm_context * ctx) {
    return ctx ? ctx->has_spk_enc : false;
}

int32_t audio_lm_n_codebook(const audio_lm_context * ctx) {
    return ctx ? ctx->n_cb : 0;
}

int32_t audio_lm_hidden_dim(const audio_lm_context * ctx) {
    return ctx ? ctx->hidden : 0;
}

const char * audio_lm_last_error(const audio_lm_context * ctx) {
    return ctx ? ctx->last_error.c_str() : "";
}

// =====================================================================
// Prompt build
//
// For step 2 the implemented branch is the speaker-encode path: when
// the model has a speaker_encoder section AND the caller supplied either
// `ref_pcm` or `speaker_emb` (+ `ref_speech_tokens` if needed), run
// `codec_lm_speaker_encode` and populate `embeds_prefix` with the
// resulting (n_rows × hidden_dim) matrix.
//
// `tokens` is left empty — text tokenization runs in the host (which
// owns the llama_model + tokenizer), and codec_common doesn't depend on
// llama.cpp.  Once Type A reference models join, build_prompt may also
// emit a few seed tokens (e.g. audio_start markers) directly.
//
// Sampler hints + start/stop tokens are zero for now; they'll get
// populated from `codec.lm.sampling.*` GGUF keys in a follow-up — for
// step 2 the host fills them in itself.
//
// Returns true even when no speaker encoding happens (Type A / zero-
// shot models) — empty `tokens` + empty `embeds_prefix` is a valid
// "host does everything text-side" prompt.
// =====================================================================

bool audio_lm_build_prompt(audio_lm_context * ctx,
                            const audio_lm_input  & in,
                            audio_lm_prompt       * out) {
    if (ctx == nullptr || out == nullptr) return false;
    out->tokens.clear();
    out->embeds_prefix.clear();
    out->embeds_prefix_rows   = 0;
    out->embeds_prefix_hidden = 0;
    out->embeds_uncond.clear();
    out->default_temperature        = 0.0f;
    out->default_top_p              = 0.0f;
    out->default_min_p              = 0.0f;
    out->default_repetition_penalty = 0.0f;
    out->default_cfg_weight         = 0.0f;
    out->start_token                = -1;
    out->stop_token                 = -1;

    if (!ctx->has_spk_enc) {
        // Type A / zero-shot — no speaker prefix to compute.  Caller
        // still gets a successful (empty) prompt; the host AR loop runs
        // entirely on its own tokenized text input.
        return true;
    }

    // ────────────────────────────────────────────────────────────────
    // Speaker-encode branch.  Two flavours depending on what the caller
    // has on hand: (a) raw `ref_pcm` → run the full speaker encoder
    // (VE + cond_enc for Chatterbox, ECAPA-TDNN for Qwen3-TTS);
    // (b) cached `speaker_emb` → skip the front-end via `_from_embedding`.
    // ────────────────────────────────────────────────────────────────
    const codec_lm_speaker_info * si = codec_lm_speaker_get_info(ctx->lm);
    if (si == nullptr || si->n_rows <= 0 || si->hidden_dim <= 0) {
        ctx->last_error = "audio_lm_build_prompt: speaker_info missing or invalid";
        return false;
    }
    const int32_t need_elems = si->n_rows * si->hidden_dim;
    out->embeds_prefix.resize((size_t) need_elems);

    const bool has_pcm = (in.ref_pcm != nullptr && in.ref_n_samples > 0);
    const bool has_emb = (in.speaker_emb != nullptr && in.speaker_emb_dim > 0);

    if (!has_pcm && !has_emb) {
        // Caller declared neither — but the model has a speaker encoder.
        // Allow the no-speaker path (build_prompt returns success with
        // empty prefix) only when the model doesn't strictly require ref
        // audio; otherwise complain.
        if (si->needs_ref_pcm || si->needs_ref_speech_tokens) {
            ctx->last_error =
                "audio_lm_build_prompt: model requires ref_pcm / ref_speech_tokens "
                "but neither ref_pcm nor speaker_emb was provided";
            out->embeds_prefix.clear();
            return false;
        }
        out->embeds_prefix.clear();
        return true;
    }

    codec_audio audio = {};
    if (has_pcm) {
        audio.data        = in.ref_pcm;
        audio.n_samples   = in.ref_n_samples;
        audio.sample_rate = in.ref_sample_rate > 0
                            ? in.ref_sample_rate
                            : si->ref_sample_rate;
        audio.n_channels  = 1;
        audio.pcm_type    = CODEC_PCM_TYPE_F32;
    }

    // ref_speech_tokens are model-specific (Chatterbox needs them; the
    // Qwen3-TTS ECAPA path doesn't).  For now they come from `in.extra`
    // when the model needs them: key `"ref_speech_tokens_csv"` =
    // comma-separated int32 ids.  This stays a private convention until
    // the input schema doc lands; tts-cli doesn't drive it yet.
    std::vector<int32_t> ref_codes;
    const int32_t * ref_codes_ptr = nullptr;
    int32_t         ref_codes_n   = 0;
    if (si->needs_ref_speech_tokens) {
        auto it = in.extra.find("ref_speech_tokens_csv");
        if (it != in.extra.end()) {
            const std::string & s = it->second;
            size_t i = 0;
            while (i < s.size()) {
                size_t j = s.find(',', i);
                if (j == std::string::npos) j = s.size();
                if (j > i) {
                    try { ref_codes.push_back(std::stoi(s.substr(i, j - i))); }
                    catch (...) { /* skip malformed */ }
                }
                i = j + 1;
            }
        }
        if (!ref_codes.empty()) {
            ref_codes_ptr = ref_codes.data();
            ref_codes_n   = (int32_t) ref_codes.size();
        }
    }

    const enum codec_status rc = has_pcm
        ? codec_lm_speaker_encode(
            ctx->lm, &audio, ref_codes_ptr, ref_codes_n, in.emotion,
            out->embeds_prefix.data(), need_elems)
        : codec_lm_speaker_encode_from_embedding(
            ctx->lm, in.speaker_emb, in.speaker_emb_dim,
            ref_codes_ptr, ref_codes_n, in.emotion,
            out->embeds_prefix.data(), need_elems);

    if (rc != CODEC_STATUS_SUCCESS) {
        const char * raw = codec_lm_get_last_error(ctx->lm);
        ctx->last_error = std::string("audio_lm_build_prompt: speaker_encode failed (")
                          + (raw && *raw ? raw : "?") + ")";
        out->embeds_prefix.clear();
        return false;
    }
    out->embeds_prefix_rows   = si->n_rows;
    out->embeds_prefix_hidden = si->hidden_dim;
    return true;
}

// ─── Type A audio-token range config ────────────────────────────────

void audio_lm_set_audio_token_range(audio_lm_context * ctx,
                                     int32_t offset, int32_t count, int32_t eos_id) {
    if (ctx == nullptr) return;
    ctx->audio_tok_offset = offset;
    ctx->audio_tok_count  = count;
    ctx->audio_tok_eos    = eos_id;
}

void audio_lm_get_audio_token_range(const audio_lm_context * ctx,
                                     int32_t * out_offset, int32_t * out_count,
                                     int32_t * out_eos_id) {
    if (out_offset) *out_offset = ctx ? ctx->audio_tok_offset : -1;
    if (out_count ) *out_count  = ctx ? ctx->audio_tok_count  : 0;
    if (out_eos_id) *out_eos_id = ctx ? ctx->audio_tok_eos    : -1;
}

// ─── Type B embed-override config ───────────────────────────────────

void audio_lm_set_uses_embed_override(audio_lm_context * ctx,
                                       bool enabled, int32_t start_step) {
    if (ctx == nullptr) return;
    ctx->uses_embed_override = enabled;
    ctx->ar_step_start       = start_step;
    ctx->ar_step             = start_step;
}

bool audio_lm_get_uses_embed_override(const audio_lm_context * ctx) {
    return ctx ? ctx->uses_embed_override : false;
}

// ─── Per-step observe ───────────────────────────────────────────────
//
// Type A path implemented (single-codebook contiguous audio range +
// optional EOS sentinel).  Type B/C/D paths land in steps 4–5 and
// will inspect the `last_hidden` argument; for now Type A ignores it.

observe_action audio_lm_observe_token(
        audio_lm_context * ctx,
        codec_common_token tok,
        const float *      /*last_hidden*/,
        int32_t            /*hidden_dim*/) {
    if (ctx == nullptr) return OBSERVE_STOP;

    // EOS sentinel — explicit end-of-audio token.  Checked BEFORE the
    // audio range in case the converter put eos_id inside [offset,
    // offset+count) (unusual but legal).
    if (ctx->audio_tok_eos >= 0 && tok == ctx->audio_tok_eos) {
        return OBSERVE_STOP;
    }

    // Type A/B audio-range detection.  Disabled when offset < 0.
    if (ctx->audio_tok_offset >= 0 && ctx->audio_tok_count > 0 &&
        tok >= ctx->audio_tok_offset &&
        tok <  ctx->audio_tok_offset + ctx->audio_tok_count) {
        const int32_t code = tok - ctx->audio_tok_offset;
        // Append (single-cb frame).  Future multi-cb Type A variants
        // (e.g. Orpheus-style delay over N codebooks) will need a more
        // structured stride here; n_q==1 is the only shape we surface
        // today.
        const int32_t this_n_q = ctx->n_cb > 0 ? ctx->n_cb : 1;
        if (this_n_q != 1) {
            // Multi-cb Type A/B isn't part of step 3/4.  Fall through to
            // PASSTHROUGH rather than silently corrupt the accumulator.
            return OBSERVE_PASSTHROUGH;
        }
        if (ctx->n_cb == 0) ctx->n_cb = 1;
        ctx->codes.push_back(code);
        ctx->codes_n_frames += 1;

        // Type B: compose the next backbone-input embed via codec_lm.
        // Requires a codec_lm (single-cb here, so codes is just [code])
        // and surfaces OBSERVE_CONSUMED_EMBED so the host knows to feed
        // `get_next_embed`'s buffer into its inputs_embeds decode path.
        if (ctx->uses_embed_override && ctx->lm != nullptr && ctx->hidden > 0) {
            ctx->next_embed_buf.assign((size_t) ctx->hidden, 0.0f);
            const int32_t codes_single[1] = { code };
            const enum codec_status rc = codec_lm_compose_next_embd(
                ctx->lm, codes_single, ctx->ar_step,
                ctx->next_embed_buf.data());
            if (rc != CODEC_STATUS_SUCCESS) {
                const char * raw = codec_lm_get_last_error(ctx->lm);
                ctx->last_error = std::string("observe_token: compose_next_embd failed (")
                                  + (raw && *raw ? raw : "?") + ")";
                ctx->next_embed_buf.clear();
                ctx->next_embed_dim = 0;
                return OBSERVE_STOP;
            }
            ctx->next_embed_dim = ctx->hidden;
            ctx->ar_step       += 1;
            return OBSERVE_CONSUMED_EMBED;
        }

        return OBSERVE_CONSUMED;
    }

    // Anything else — BOS, text token, special marker.  Caller renders
    // it / handles it on the text path.
    return OBSERVE_PASSTHROUGH;
}

const float * audio_lm_get_next_embed(const audio_lm_context * ctx,
                                       int32_t * out_dim) {
    if (out_dim) *out_dim = ctx ? ctx->next_embed_dim : 0;
    return (ctx && !ctx->next_embed_buf.empty()) ? ctx->next_embed_buf.data() : nullptr;
}

// ─── Multi-cb observe (Type C / Type D) ─────────────────────────────
//
// Host has already done the per-frame sampling — for Type C via the
// codec_lm step machine, for Type D via parallel head outputs — and
// passes the assembled (n_cb) code vector in.  We just accumulate it
// and (optionally) compose the next backbone-input embedding.

observe_action audio_lm_observe_codes(
        audio_lm_context * ctx,
        const int32_t    * codes,
        int32_t            n_codes,
        const float      * /*last_hidden*/,
        int32_t            /*hidden_dim*/) {
    if (ctx == nullptr || codes == nullptr || n_codes <= 0) {
        if (ctx) ctx->last_error = "observe_codes: null ctx / codes or non-positive n_codes";
        return OBSERVE_STOP;
    }

    // Lock in n_cb on first frame.  Codec_lm-backed models declare it
    // ahead of time; codec-only flows may discover it from the first
    // frame's width.
    if (ctx->n_cb == 0) {
        ctx->n_cb = n_codes;
    } else if (n_codes != ctx->n_cb) {
        ctx->last_error = std::string("observe_codes: n_codes=") + std::to_string(n_codes) +
                          " mismatches model n_codebook=" + std::to_string(ctx->n_cb);
        return OBSERVE_STOP;
    }

    const size_t prev = ctx->codes.size();
    ctx->codes.resize(prev + (size_t) n_codes);
    std::memcpy(ctx->codes.data() + prev, codes, (size_t) n_codes * sizeof(int32_t));
    ctx->codes_n_frames += 1;

    // Type B/C/D: compose next backbone-input embed when override is on
    // AND we have a codec_lm to compose with.  For residual_depth_ar
    // (no learned pos table) compose_next_embd falls back to
    // compose_audio_embd — exactly the right shape for CSM.  For
    // parallel_heads_delay with a pos table (Chatterbox) the host
    // shouldn't be hitting this path with n_codes > 1, but we honour
    // it consistently.
    if (ctx->uses_embed_override && ctx->lm != nullptr && ctx->hidden > 0) {
        ctx->next_embed_buf.assign((size_t) ctx->hidden, 0.0f);
        const enum codec_status rc = codec_lm_compose_next_embd(
            ctx->lm, codes, ctx->ar_step, ctx->next_embed_buf.data());
        if (rc != CODEC_STATUS_SUCCESS) {
            const char * raw = codec_lm_get_last_error(ctx->lm);
            ctx->last_error = std::string("observe_codes: compose_next_embd failed (")
                              + (raw && *raw ? raw : "?") + ")";
            ctx->next_embed_buf.clear();
            ctx->next_embed_dim = 0;
            return OBSERVE_STOP;
        }
        ctx->next_embed_dim = ctx->hidden;
        ctx->ar_step       += 1;
        return OBSERVE_CONSUMED_EMBED;
    }

    return OBSERVE_CONSUMED;
}

// =====================================================================
// External codes push (offline / debug)
//
// Appends to the same `ctx->codes` accumulator `observe_token` will
// populate during AR.  Format mirrors codec_token_buffer: (T, n_q)
// interleaved row-major.
// =====================================================================

bool audio_lm_push_codes(audio_lm_context * ctx,
                          const int32_t   * codes,
                          int32_t           n_frames,
                          int32_t           n_q) {
    if (ctx == nullptr || codes == nullptr) return false;
    if (n_frames <= 0 || n_q <= 0) {
        ctx->last_error = "audio_lm_push_codes: non-positive n_frames / n_q";
        return false;
    }
    // codec_lm absent → we can still accumulate; decode_audio works on
    // codec_model only, n_q comes from the caller.  Just don't enforce
    // a model-side n_codebook check in that case.
    if (ctx->n_cb > 0 && n_q != ctx->n_cb) {
        ctx->last_error = "audio_lm_push_codes: n_q doesn't match model n_codebook";
        return false;
    }
    if (ctx->codes_n_frames == 0) {
        ctx->n_cb = n_q;   // remember it so decode_audio can lay out the buffer
    }
    const size_t prev = ctx->codes.size();
    ctx->codes.resize(prev + (size_t) n_frames * (size_t) n_q);
    std::memcpy(ctx->codes.data() + prev, codes,
                (size_t) n_frames * (size_t) n_q * sizeof(int32_t));
    ctx->codes_n_frames += n_frames;
    return true;
}

// =====================================================================
// End of sequence — codes → PCM
// =====================================================================

// =====================================================================
// Continuous-latent path (BlueMagpie / VoxCPM, kind continuous_latent_cfm)
// =====================================================================

bool audio_lm_is_continuous(const audio_lm_context * ctx) {
    return ctx != nullptr && ctx->is_continuous;
}

void audio_lm_set_continuous_params(audio_lm_context * ctx, float cfg_value,
                                    int32_t n_timesteps, int32_t min_len) {
    if (ctx == nullptr) return;
    if (cfg_value   > 0.0f) ctx->cont_cfg       = cfg_value;
    if (n_timesteps > 0)    ctx->cont_timesteps = n_timesteps;
    ctx->cont_min_len = min_len;   // <0 keeps the model default
    if (ctx->state != nullptr && ctx->is_continuous) {
        codec_lm_set_continuous_min_len(ctx->state, min_len);
    }
}

bool audio_lm_text_prefill(audio_lm_context * ctx, const float * hiddens,
                           int32_t n_pos, int32_t hidden_dim) {
    if (ctx == nullptr || ctx->state == nullptr || hiddens == nullptr) {
        if (ctx) ctx->last_error = "audio_lm_text_prefill: no codec_lm state";
        return false;
    }
    if (!ctx->is_continuous) {
        ctx->last_error = "audio_lm_text_prefill: model is not a continuous-latent kind";
        return false;
    }
    enum codec_status rc = codec_lm_text_prefill(ctx->state, hiddens, n_pos, hidden_dim);
    if (rc != CODEC_STATUS_SUCCESS) {
        const char * raw = codec_lm_state_get_last_error(ctx->state);
        ctx->last_error = std::string("audio_lm_text_prefill: failed (") + (raw ? raw : "?") + ")";
        return false;
    }
    return true;
}

observe_action audio_lm_observe_hidden(audio_lm_context * ctx,
                                       const float * hidden, int32_t hidden_dim,
                                       const float * noise) {
    if (ctx == nullptr || ctx->lm == nullptr || ctx->state == nullptr) {
        if (ctx) ctx->last_error = "audio_lm_observe_hidden: no codec_lm adaptor";
        return OBSERVE_STOP;
    }
    if (!ctx->is_continuous) {
        ctx->last_error = "audio_lm_observe_hidden: model is not a continuous-latent kind";
        return OBSERVE_STOP;
    }
    if (hidden == nullptr || hidden_dim != ctx->hidden) {
        ctx->last_error = "audio_lm_observe_hidden: hidden is null or wrong dim";
        return OBSERVE_STOP;
    }

    const int32_t pd = ctx->patch_size * ctx->latent_dim;
    std::vector<float> patch((size_t) pd, 0.0f);
    int32_t stop = 0;
    enum codec_status rc = codec_lm_step_generate(
        ctx->state, hidden, ctx->cont_cfg, ctx->cont_timesteps, noise, patch.data(), &stop);
    if (rc != CODEC_STATUS_SUCCESS) {
        const char * raw = codec_lm_state_get_last_error(ctx->state);
        ctx->last_error = std::string("audio_lm_observe_hidden: step_generate failed (")
                           + (raw ? raw : "?") + ")";
        return OBSERVE_STOP;
    }
    // Accumulate the patch frame-major ([patch_size, latent_dim]); transposed
    // to channel-major at decode time.
    ctx->latents.insert(ctx->latents.end(), patch.begin(), patch.end());
    ctx->latent_n_frames += ctx->patch_size;

    if (stop) {
        return OBSERVE_STOP;
    }

    // Feedback embedding for the next backbone step.
    ctx->next_embed_buf.assign((size_t) ctx->hidden, 0.0f);
    if (codec_lm_step_feedback_embd(ctx->state, ctx->next_embed_buf.data()) != CODEC_STATUS_SUCCESS) {
        ctx->last_error = "audio_lm_observe_hidden: step_feedback_embd failed";
        return OBSERVE_STOP;
    }
    ctx->next_embed_dim = ctx->hidden;
    return OBSERVE_CONSUMED_EMBED;
}

bool audio_lm_decode_audio(audio_lm_context * ctx, audio_lm_audio_output * out) {
    if (ctx == nullptr || out == nullptr) return false;

    // Continuous-latent models: decode the accumulated latent patches.
    if (ctx->is_continuous) {
        if (ctx->latent_n_frames <= 0) {
            ctx->last_error = "audio_lm_decode_audio: no latents accumulated";
            return false;
        }
        const int32_t D = ctx->latent_dim, T = ctx->latent_n_frames;
        // Transpose frame-major [T, D] → channel-major [D, T] for decode.
        std::vector<float> chan((size_t) D * T);
        for (int32_t t = 0; t < T; ++t)
            for (int32_t d = 0; d < D; ++d)
                chan[(size_t) d * T + t] = ctx->latents[(size_t) t * D + d];
        codec_pcm_buffer pcm = {};
        auto dp = codec_decode_default_params();
        const enum codec_status rc = codec_decode_quantized_representation(
            ctx->codec_ctx, chan.data(), D, T, &pcm, dp);
        if (rc != CODEC_STATUS_SUCCESS) {
            const char * raw = codec_get_last_error(ctx->codec_ctx);
            ctx->last_error = std::string("audio_lm_decode_audio: decode_latents failed (")
                               + (raw ? raw : "?") + ")";
            return false;
        }
        out->pcm.assign(pcm.data, pcm.data + (size_t) pcm.n_samples * (size_t) pcm.n_channels);
        out->sample_rate = pcm.sample_rate;
        out->n_channels  = pcm.n_channels;
        codec_pcm_buffer_free(&pcm);
        return true;
    }

    if (ctx->codes_n_frames <= 0 || ctx->n_cb <= 0) {
        ctx->last_error = "audio_lm_decode_audio: no codes accumulated";
        return false;
    }
    if (!(ctx->modality_mask & OUTPUT_AUDIO)) {
        ctx->last_error = "audio_lm_decode_audio: model has no OUTPUT_AUDIO modality";
        return false;
    }
    if (!codec_model_has_decoder(ctx->model)) {
        ctx->last_error = "audio_lm_decode_audio: codec has no decoder";
        return false;
    }

    codec_token_buffer tokens = {};
    tokens.data         = ctx->codes.data();
    tokens.n_tokens     = (int32_t) ctx->codes.size();
    tokens.n_frames     = ctx->codes_n_frames;
    tokens.n_q          = ctx->n_cb;
    tokens.codebook_size = codec_model_codebook_size(ctx->model);
    tokens.sample_rate   = codec_model_sample_rate(ctx->model);
    tokens.hop_size      = codec_model_hop_size(ctx->model);

    codec_pcm_buffer pcm = {};
    auto dp = codec_decode_default_params();
    const enum codec_status rc =
        codec_decode(ctx->codec_ctx, &tokens, &pcm, dp);
    if (rc != CODEC_STATUS_SUCCESS) {
        const char * raw = codec_get_last_error(ctx->codec_ctx);
        ctx->last_error = std::string("audio_lm_decode_audio: codec_decode failed (")
                           + (raw ? raw : "?") + ")";
        return false;
    }

    out->pcm.assign(pcm.data, pcm.data + (size_t) pcm.n_samples * (size_t) pcm.n_channels);
    out->sample_rate = pcm.sample_rate;
    out->n_channels  = pcm.n_channels;
    codec_pcm_buffer_free(&pcm);
    return true;
}

}  // namespace codec_common
