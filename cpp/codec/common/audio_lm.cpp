#include "codec_common.h"

#include <algorithm>
#include <cstring>
#include <new>
#include <string>
#include <vector>

namespace codec_common {

// Linear-interpolation resample of mono F32 PCM from `in_sr` to `out_sr`.
// Shared by every speaker-encoder path: the encoders (ECAPA-TDNN @ 24 kHz,
// Chatterbox VE @ 16 kHz, …) declare their working rate via
// `codec_lm_speaker_info::ref_sample_rate`; the runtime is responsible for
// feeding PCM at that rate (see include/codec_lm.h).  This mirrors the
// reference pipelines' librosa.resample-to-speaker_encoder_sample_rate step.
static std::vector<float> resample_linear_mono(const std::vector<float> & in,
                                               int32_t in_sr, int32_t out_sr) {
    if (in.empty() || in_sr <= 0 || out_sr <= 0 || in_sr == out_sr) return in;
    const int64_t n_in  = (int64_t) in.size();
    const int64_t n_out = n_in * out_sr / in_sr;
    std::vector<float> out((size_t) std::max<int64_t>(n_out, 1));
    for (int64_t i = 0; i < (int64_t) out.size(); ++i) {
        const double src = (double) i * in_sr / out_sr;
        int64_t i0 = (int64_t) src;
        const double f = src - (double) i0;
        const float a0 = in[(size_t) std::min<int64_t>(i0,     n_in - 1)];
        const float a1 = in[(size_t) std::min<int64_t>(i0 + 1, n_in - 1)];
        out[(size_t) i] = (float) ((double) a0 * (1.0 - f) + (double) a1 * f);
    }
    return out;
}

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

    // codes→PCM decode transform (parallel_heads_delay / control-cb0 models).
    // Populated at init from GGUF metadata + codec_lm_info; the decode path
    // reads them so no model-name switch is needed.
    //
    //   audio_cb_offset: number of leading codebooks that are PURE text/
    //     control and get dropped entirely before decode (Moshi-style
    //     c0_input_modality="text").  0 for CSM / Qwen3-TTS / Realtime and
    //     also for MOSS-TTSD (whose cb0 is a *merged* text+speech channel,
    //     handled via cb0_speech_offset below, not dropped).
    //   cb0_speech_offset: value subtracted from codebook-0 codes to map the
    //     merged text+speech vocab back into raw quantizer index space
    //     (MOSS-TTSD: speech_token_range[0]).  0 = no remap.
    //   delay_pattern[q] (indexed over the FULL n_cb): per-codebook emission
    //     delay to reverse before forming the codec buffer (MOSS-TTSD
    //     [0,1,…,7]).  Empty when all-zero.
    int32_t              audio_cb_offset   = 0;
    int32_t              cb0_speech_offset = 0;
    std::vector<int32_t> delay_pattern;   // empty → no delay to unshift

    // Merged-cb0 models (MOSS-TTSD): the prompt embedding is the sum over all
    // n_cb embedding tables (cb0=text token, cb1..N-1=speech_pad).  When true,
    // the host must feed composed prompt embeddings via inputs_embeds.
    bool    prompt_needs_composed = false;
    int32_t speech_pad_code       = 0;   // codec.lm.speech_pad_token

    // Qwen3-TTS talker prompt control tags (codec-vocab, looked up in
    // audio_embd_0 = codec_embedding).  -1 when absent (non-Qwen3-TTS).
    int32_t q3_nothink_id   = -1;
    int32_t q3_think_bos_id = -1;
    int32_t q3_think_eos_id = -1;
    int32_t q3_codec_pad_id = -1;
    int32_t q3_codec_bos_id = -1;
    int32_t q3_tts_pad_id   = -1;   // text-vocab (projected)
    int32_t q3_tts_bos_id   = -1;   // text-vocab (projected)
    int32_t q3_tts_eos_id   = -1;   // text-vocab (projected)
    bool    has_talker_proj = false;

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

    const codec_lm_gguf_metadata * meta = codec_model_metadata(ctx->model);
    bool saw_explicit = false;

    if (meta != nullptr) {
        for (size_t i = 0; i < meta->n_items; ++i) {
            const char * key = meta->items[i].key;
            const char * val = meta->items[i].value;
            if (key == nullptr || val == nullptr) continue;
            // Match "true" loosely — codec_lm_gguf_metadata serialises
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

static const char * meta_str(const audio_lm_context * ctx, const char * key);

// Determine the codes→PCM decode transform from GGUF metadata.  This is
// the single source of truth for both `audio_lm_get_prompt_info` (which
// reports it) and `audio_lm_decode_audio` (which applies it).
//
//   audio_cb_offset = 1  when codebook 0 is a text/control channel that is
//     NOT an audio quantizer level.  Two cases:
//       * parallel_heads_delay: cb0 is the backbone text vocab (MOSS-TTSD).
//       * residual_depth_ar with `c0_input_modality="text"` (Moshi).
//     For residual_depth_ar with c0 modality "audio" (CSM, Qwen3-TTS) or
//     "none" (MOSS-TTS-Realtime, LFM2) cb0 IS an audio codebook → offset 0.
//   delay_pattern = codec_lm_info.delay_pattern (over the full n_cb), used
//     to reverse the per-codebook emission shift before decode.
static void init_decode_transform(audio_lm_context * ctx) {
    ctx->audio_cb_offset   = 0;
    ctx->cb0_speech_offset = 0;
    ctx->delay_pattern.clear();
    if (ctx->lm == nullptr) return;

    const codec_lm_info * info = codec_lm_get_info(ctx->lm);
    if (info == nullptr) return;

    const char * kind = meta_str(ctx, "codec.lm.kind");
    const bool is_depth = kind && std::strcmp(kind, "residual_depth_ar") == 0;

    // Pure text/control leading codebooks are DROPPED before decode.  This
    // is the Moshi-style residual_depth_ar case (c0_input_modality="text"):
    // cb0 is a backbone text token with no audio content.  MOSS-TTSD is NOT
    // this case — its cb0 is a merged text+speech channel and stays in the
    // decode via cb0_speech_offset below.
    if (is_depth) {
        const char * c0mod = meta_str(ctx, "codec.lm.residual.c0_input_modality");
        ctx->audio_cb_offset = (c0mod && std::strcmp(c0mod, "text") == 0) ? 1 : 0;
    }

    // MOSS-TTSD merged-cb0 remap: subtract speech_token_range[0] from cb0
    // codes to map the merged text+speech vocab back into raw quantizer
    // index space (mirrors the HF processor shifting_outputs()).  Written by
    // the converter as a scalar since array element values aren't surfaced.
    if (const char * v = meta_str(ctx, "codec.lm.cb0_speech_offset")) {
        ctx->cb0_speech_offset = std::atoi(v);
    }

    if (info->delay_pattern != nullptr && info->n_codebook > 0) {
        bool any_delay = false;
        ctx->delay_pattern.assign(info->delay_pattern,
                                  info->delay_pattern + info->n_codebook);
        for (int32_t d : ctx->delay_pattern) if (d != 0) { any_delay = true; break; }
        if (!any_delay) ctx->delay_pattern.clear();   // nothing to unshift
    }

    // Merged-cb0 models need the composed multi-modal prompt embedding.  The
    // cb0_speech_offset is the distinguishing signal: it is only set for the
    // merged text+speech cb0 (MOSS-TTSD).  Read the speech-pad code used to
    // fill cb1..N-1 during prompt/prefill.
    if (ctx->cb0_speech_offset != 0) {
        ctx->prompt_needs_composed = true;
        if (const char * v = meta_str(ctx, "codec.lm.speech_pad_token")) {
            ctx->speech_pad_code = std::atoi(v);
        }
    }

    // Qwen3-TTS talker control tags + text projection (residual_depth_ar
    // with the qwen3tts metadata baked in).  Present only for the Qwen3-TTS
    // talker; other residual models leave these -1 / has_talker_proj=false.
    auto read_i = [&](const char * key, int32_t & dst) {
        if (const char * v = meta_str(ctx, key)) dst = std::atoi(v);
    };
    read_i("codec.lm.qwen3tts.nothink_id",   ctx->q3_nothink_id);
    read_i("codec.lm.qwen3tts.think_bos_id", ctx->q3_think_bos_id);
    read_i("codec.lm.qwen3tts.think_eos_id", ctx->q3_think_eos_id);
    read_i("codec.lm.pad_code_c0",           ctx->q3_codec_pad_id);
    read_i("codec.lm.bos_code_c0",           ctx->q3_codec_bos_id);
    read_i("codec.lm.qwen3tts.tts_pad_id",   ctx->q3_tts_pad_id);
    read_i("codec.lm.qwen3tts.tts_bos_id",   ctx->q3_tts_bos_id);
    read_i("codec.lm.qwen3tts.tts_eos_id",   ctx->q3_tts_eos_id);
    if (ctx->lm != nullptr) {
        ctx->has_talker_proj = codec_lm_text_proj_dim(ctx->lm) > 0;
    }
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
    init_decode_transform(ctx);
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

struct codec_lm * audio_lm_get_lm(audio_lm_context * ctx) {
    return ctx ? ctx->lm : nullptr;
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
    // Resample the caller's ref PCM to the encoder's declared working rate
    // (si->ref_sample_rate) when they differ.  The speaker encoders reject a
    // rate mismatch (see qwen3_tts_speaker_encode's guard), so this MUST run
    // for arbitrary --ref-audio inputs (24k/44.1k/48k, …), mirroring the
    // reference pipelines' resample-to-speaker_encoder_sample_rate step.
    // `resampled` must outlive the codec_lm_speaker_encode call below.
    std::vector<float> resampled;
    if (has_pcm) {
        const int32_t in_sr = in.ref_sample_rate > 0
                              ? in.ref_sample_rate
                              : si->ref_sample_rate;
        const int32_t want_sr = si->ref_sample_rate > 0
                                ? si->ref_sample_rate
                                : in_sr;
        if (in_sr != want_sr && want_sr > 0) {
            resampled = resample_linear_mono(
                std::vector<float>(in.ref_pcm, in.ref_pcm + in.ref_n_samples),
                in_sr, want_sr);
            audio.data        = resampled.data();
            audio.n_samples   = (int32_t) resampled.size();
            audio.sample_rate = want_sr;
        } else {
            audio.data        = in.ref_pcm;
            audio.n_samples   = in.ref_n_samples;
            audio.sample_rate = in_sr;
        }
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

void audio_lm_get_lm_eos(const audio_lm_context * ctx,
                         int32_t * out_eos_code_c0, int32_t * out_eos_min_step) {
    int32_t eos_c0 = -1, eos_min = 0;
    if (ctx != nullptr && ctx->lm != nullptr) {
        const codec_lm_info * info = codec_lm_get_info(ctx->lm);
        if (info != nullptr) {
            eos_c0  = info->eos_code_c0;
            eos_min = info->eos_min_step;
        }
    }
    if (out_eos_code_c0)  *out_eos_code_c0  = eos_c0;
    if (out_eos_min_step) *out_eos_min_step = eos_min;
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

    // Model-owned end-of-audio: if the codec_lm declares a cb0 EOS
    // sentinel (codec.lm.eos_code_c0 metadata), let it decide.  This is a
    // no-op when metadata is absent (eos_code_c0 defaults to -1 →
    // out_is_eos always 0) or for kinds without the concept (NOT_SUPPORTED,
    // treated as "not EOS").  The host used to hardcode this per-model.
    if (ctx->lm != nullptr && ctx->state != nullptr) {
        int32_t is_eos = 0;
        const enum codec_status ers =
            codec_lm_step_is_eos(ctx->state, codes, n_codes, &is_eos);
        if (ers == CODEC_STATUS_SUCCESS && is_eos) {
            return OBSERVE_STOP;
        }
    }

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

// =====================================================================
// Phase B: prompt assembly + codebook step machine passthroughs
// =====================================================================

// Look up a raw GGUF metadata string value by key.  Returns nullptr when
// absent.  Used to key model-specific prompt assembly off host_arch / kind.
static const char * meta_str(const audio_lm_context * ctx, const char * key) {
    if (ctx == nullptr || ctx->model == nullptr) return nullptr;
    const codec_lm_gguf_metadata * meta = codec_model_metadata(ctx->model);
    if (meta == nullptr) return nullptr;
    for (size_t i = 0; i < meta->n_items; ++i) {
        if (meta->items[i].key && std::strcmp(meta->items[i].key, key) == 0) {
            return meta->items[i].value;
        }
    }
    return nullptr;
}

// Read an integer-valued KV (stored stringified) with a default fallback.
static int32_t meta_i32_or(const audio_lm_context * ctx, const char * key,
                           int32_t dflt) {
    const char * v = meta_str(ctx, key);
    if (v == nullptr || *v == '\0') return dflt;
    return (int32_t) std::atoi(v);
}

// Read a bool-valued KV ("true"/"1" → true) with a default fallback.
static bool parse_bool_meta(const audio_lm_context * ctx, const char * key,
                            bool dflt) {
    const char * v = meta_str(ctx, key);
    if (v == nullptr || *v == '\0') return dflt;
    return v[0] == 't' || v[0] == 'T' || v[0] == '1';
}

bool audio_lm_get_prompt_info(const audio_lm_context * ctx,
                              audio_lm_prompt_info    * out) {
    if (ctx == nullptr || out == nullptr) return false;
    *out = audio_lm_prompt_info();
    if (ctx->lm == nullptr) {
        if (ctx) const_cast<audio_lm_context*>(ctx)->last_error =
            "audio_lm_get_prompt_info: no codec_lm adaptor";
        return false;
    }

    const codec_lm_info * info = codec_lm_get_info(ctx->lm);
    const char * arch = meta_str(ctx, "codec.lm.host_arch");
    const char * kind = meta_str(ctx, "codec.lm.kind");
    out->host_arch    = arch ? arch : (info && info->host_arch ? info->host_arch : "");
    out->n_codebook   = ctx->n_cb;
    out->hidden_dim   = ctx->hidden;
    out->is_continuous = ctx->is_continuous;
    if (info != nullptr) {
        out->eos_code_c0  = info->eos_code_c0;
        out->eos_min_step = info->eos_min_step;
    }
    // Merged-cb0 speech sub-range (MOSS-TTSD) — surfaced for the host's
    // auto-grammar.  cb0_speech_offset is speech_token_range[0] (start);
    // cb0_speech_range_end mirrors speech_token_range[1] (exclusive end).
    out->cb0_speech_range_start = meta_i32_or(ctx, "codec.lm.cb0_speech_offset", -1);
    out->cb0_speech_range_end   = meta_i32_or(ctx, "codec.lm.cb0_speech_range_end", -1);
    // rn-tts sampling defaults.
    out->default_temperature = 0.9f;
    out->default_top_p       = 0.95f;
    out->default_top_k       = 50;

    const bool is_delay = kind && std::strcmp(kind, "parallel_heads_delay") == 0;
    const bool is_depth = kind && std::strcmp(kind, "residual_depth_ar") == 0;
    if (ctx->is_continuous)        out->model_kind = audio_lm_prompt_info::KIND_CONTINUOUS_CFM;
    else if (is_delay)             out->model_kind = audio_lm_prompt_info::KIND_PARALLEL_HEADS_DELAY;
    else if (is_depth)             out->model_kind = audio_lm_prompt_info::KIND_RESIDUAL_DEPTH_AR;

    // ── Per-arch / per-family prompt template ──────────────────────────
    // barbet == BlueMagpie continuous latent CFM.
    if (out->host_arch == "barbet" || ctx->is_continuous) {
        // <|bm_spk|> text <|bm_audio_start|>  (control tokens, no BOS)
        out->prompt_prefix = "<|bm_spk|>";
        out->prompt_suffix = "<|bm_audio_start|>";
        out->add_bos       = false;
        out->parse_special = true;
        out->cb0_from_backbone = false;
        out->is_continuous = true;
        return true;
    }

    if (out->host_arch == "llama") {
        // CSM: [<speaker_id>]text<|end_of_text|>.  add_bos=true handles BOS;
        // speaker id defaults to 0 (host may override the prefix via extra).
        out->prompt_prefix = "[0]";
        out->prompt_suffix = "<|end_of_text|>";
        out->add_bos       = true;
        out->parse_special = true;
        out->cb0_from_backbone = false;
        out->audio_codebook_offset = ctx->audio_cb_offset;   // 0 for CSM
        return true;
    }

    // qwen3 family — MOSS-TTSD (delay) is a plain [S1] pass-through;
    // Qwen3-TTS / MOSS-TTS-Realtime (depth) use ChatML.
    if (out->host_arch == "qwen3") {
        out->audio_codebook_offset = ctx->audio_cb_offset;
        // cb0_from_backbone = "the backbone lm_head samples cb0 as a text
        // token".  MOSS-TTSD (parallel_heads_delay): cb0 is the merged
        // text+speech vocab sampled from the backbone, so true.  Qwen3-TTS /
        // MOSS-TTS-Realtime (residual_depth_ar): cb0 is an audio codebook the
        // depth decoder samples, so false.
        out->cb0_from_backbone     = is_delay;
        if (is_delay) {
            // MOSS-TTSD chat template (fnlp/MOSS-TTSD-v0.5 tokenizer_config):
            //   <|begin_of_style|>{system}<|end_of_style|>
            //   <|begin_of_text|>{text}<|end_of_text|>
            //   <|begin_of_speech|>
            // where the caller's [S1]/[S2] speaker tags map to
            // <speaker1>/<speaker2> (host does that substitution before
            // tokenizing).  The trailing <|begin_of_speech|> opens the audio
            // channel; generation then emits speech frames until cb0 samples
            // the text EOS (eos_code_c0=151643).
            out->prompt_prefix =
                "<|begin_of_style|>You are a speech synthesizer that generates "
                "natural, realistic, and human-like conversational audio from "
                "dialogue text.<|end_of_style|>\n<|begin_of_text|>";
            out->prompt_suffix = "<|end_of_text|>\n<|begin_of_speech|>";
            out->add_bos       = false;
            out->parse_special = true;
            return true;
        }
        // MOSS-TTS-Realtime (residual_depth_ar, ChatML): the reference
        // processor (processing_mossttsrealtime.py) prefixes a fixed TTS
        // system prompt describing the engine, then puts the text in a user
        // turn and opens an assistant turn where the 16 audio channels are
        // generated.  Without the system prompt the backbone hidden states
        // drift off-distribution and the depth decoder emits noise.  We
        // detect realtime by the n_codebook==16 depth layout carried in
        // metadata (Qwen3-TTS is 16-group residual too, but host_arch is
        // still "qwen3"; the realtime kind ships codec.lm.n_codebook=16 and
        // c0_input_modality="none").
        {
            const char * c0mod = meta_str(ctx, "codec.lm.residual.c0_input_modality");
            const bool is_realtime = is_depth && c0mod &&
                                     std::strcmp(c0mod, "none") == 0;
            if (is_realtime) {
                // Streaming interleave: the spoken text is fed one token per
                // audio frame into the ASSISTANT turn (the system prompt says
                // "based on the text given in the assistant").  The context
                // prefix is the system prompt + an empty user turn + the
                // assistant opener; the spoken text is NOT baked into it —
                // the host streams it through the composed per-step input.
                // (processing_mossttsrealtime.py tts_system_prompt +
                //  streaming_mossttsrealtime.py prefill/step.)
                out->prompt_prefix =
                    "<|im_start|>system\nYou are a highly expressive "
                    "text-to-speech (TTS) engine developed by Mosi "
                    "Intelligence. \nYou possess natural language "
                    "understanding, emotional modeling, and multi-style "
                    "speech generation capabilities, allowing you to generate "
                    "the corresponding speech based on the text given in the "
                    "assistant.<|im_end|>\n<|im_start|>user\n";
                out->prompt_suffix =
                    "<|im_end|>\n<|im_start|>assistant\n";
                out->add_bos       = false;
                out->parse_special = true;

                // Streaming interleave params (metadata-driven, with the
                // reference constants as fallbacks).
                out->streaming_interleave  = true;
                out->text_externally_added =
                    parse_bool_meta(ctx, "codec.lm.compose.text_externally_added", true);
                out->prefill_text_len =
                    meta_i32_or(ctx, "codec.lm.compose.prefill_text_len", 12);
                out->text_pad_id    = meta_i32_or(ctx, "codec.lm.text_pad", 151655);
                out->audio_pad_code = meta_i32_or(ctx, "codec.lm.audio_pad_token", 1024);
                out->bos_code_c0    = meta_i32_or(ctx, "codec.lm.bos_code_c0", 1025);
                // Reference streaming sampling defaults.
                out->default_temperature = 0.8f;
                out->default_top_p       = 0.6f;
                out->default_top_k       = 30;
                out->default_repetition_penalty = 1.1f;
                out->repetition_window          = 50;
                return true;
            }
        }
        // ChatML for Qwen3-TTS.
        out->prompt_prefix = "<|im_start|>user\n";
        out->prompt_suffix = "<|im_end|>\n<|im_start|>assistant\n";
        out->add_bos       = false;
        out->parse_special = true;
        return true;
    }

    // lfm2 — LFM2-Audio-1.5B sequential text→audio TTS.  The reference
    // (liquid_audio ChatState + generate_sequential) prefills a ChatML
    // conversation whose system turn primes pure read-aloud TTS, puts the
    // input text in a user turn, and opens an assistant turn.  Generation
    // then free-runs in TEXT modality (backbone tied-embedding lm_head) until
    // the model emits <|audio_start|> (id 128), switches to AUDIO_OUT and
    // depth-decodes 8-codebook Mimi frames until cb0 == EOAudio (2048) or the
    // backbone emits <|im_end|> (id 7).  The <|startoftext|> BOS is prepended
    // by the tokenizer (add_bos=true).
    if (out->host_arch == "lfm2") {
        // TTS system prompt: "Perform TTS. Use the [voice] voice." — this is
        // what flips the model into immediate AUDIO_OUT (it emits
        // <|audio_start|> as the first generated token).  Voice ∈ {US male,
        // US female, UK male, UK female}; US male is the default.  (Liquid4All/
        // liquid-audio TTS example.)
        out->prompt_prefix =
            "<|im_start|>system\nPerform TTS. Use the US male voice."
            "<|im_end|>\n<|im_start|>user\n";
        out->prompt_suffix =
            "<|im_end|>\n<|im_start|>assistant\n";
        out->add_bos       = true;   // <|startoftext|>
        out->parse_special = true;
        out->cb0_from_backbone = false;      // depth decoder emits all 8 cb
        out->audio_codebook_offset = ctx->audio_cb_offset;  // 0 for LFM2
        // Sequential text→audio switch.  Special-token ids are fixed for the
        // LFM2-Audio tokenizer (added_tokens_decoder in tokenizer_config).
        out->sequential_text_audio = true;
        out->audio_start_id = meta_i32_or(ctx, "codec.lm.audio_start_id", 128);
        out->text_end_id    = meta_i32_or(ctx, "codec.lm.text_end_id",    7);
        out->max_text_tokens = meta_i32_or(ctx, "codec.lm.max_text_tokens", 64);
        // Reference TTS defaults: greedy text, greedy audio (temperature=None
        // → argmax in generate_sequential).  Host --temp overrides.
        out->default_temperature = 0.0f;
        out->default_top_p       = 1.0f;
        out->default_top_k       = 0;
        return true;
    }

    // Unknown arch — return the raw kind but empty template.
    return true;
}

// Emit a GBNF alternation matching the decimal literals [lo, hi] (inclusive),
// as `"<" ( ... ) ">"`-style terminals are assembled by the caller.  Rather
// than a full digit-range decomposition we lean on the fact that GBNF can
// enumerate a bounded set compactly via nested digit ranges; for a 0..N-1
// speech vocab this stays small.  We build the classic "0..max" numeric rule
// used by the rn-tts NeuTTS/Soprano grammars (decade/hundred/thousand digit
// bands) generalised to `max`.
static std::string gbnf_uint_range_rule(int32_t max_inclusive) {
    // Produce alternatives for [0, max_inclusive].  We special-case up to
    // 9999 (covers MOSS-TTSD's 0..1023); larger vocabs fall back to a loose
    // "[0-9]+" (still correct membership-wise for the model's own emissions,
    // just not a tight upper bound).
    if (max_inclusive < 0) return "[0-9]+";
    if (max_inclusive > 9999) return "[0-9]+";
    // NOTE: keep the whole alternation on ONE line — llama.cpp's GBNF parser
    // treats a bare newline as the end of a rule, so `\n | ...` continuations
    // fail with "expecting name at |".
    std::string out;
    auto add = [&](const std::string & alt) {
        if (!out.empty()) out += " | ";
        out += alt;
    };
    // single digit 0..9 (bounded by max)
    {
        int hi = max_inclusive < 9 ? max_inclusive : 9;
        add("[0-" + std::to_string(hi) + "]");
    }
    // two digits 10..99
    if (max_inclusive >= 10) add("[1-9] [0-9]");
    // three digits 100..999
    if (max_inclusive >= 100) add("[1-9] [0-9] [0-9]");
    // four digits 1000..max (tight upper bound on the leading digit band)
    if (max_inclusive >= 1000) {
        // Enumerate 1000..max_inclusive by leading-digit bands so we don't
        // over-admit (e.g. max=1023 must not accept 1099).  Keep it simple:
        // a loose "[1-9] [0-9] [0-9] [0-9]" would over-admit up to 9999; for
        // the common 0..1023 case emit an exact band for the thousands.
        const int thousands = max_inclusive / 1000;      // e.g. 1
        const int rem       = max_inclusive % 1000;      // e.g. 23
        // full lower thousands: [1 .. thousands-1] followed by any 3 digits
        if (thousands >= 2) {
            add("[1-" + std::to_string(thousands - 1) + "] [0-9] [0-9] [0-9]");
        }
        // the top thousand band, capped at rem
        // thousands digit is fixed; the trailing 3 digits are 000..rem
        std::string band = "\"" + std::to_string(thousands) + "\" ";
        const int h = rem / 100, t = (rem / 10) % 10, o = rem % 10;
        // 000..(h-1)99  |  h(0..t-1)9? ... approximate tight bound
        std::string sub;
        auto addsub = [&](const std::string & a) {
            if (!sub.empty()) sub += " | ";
            sub += a;
        };
        if (h >= 1) addsub("[0-" + std::to_string(h - 1) + "] [0-9] [0-9]");
        if (t >= 1) addsub("\"" + std::to_string(h) + "\" [0-" + std::to_string(t - 1) + "] [0-9]");
        addsub("\"" + std::to_string(h) + "\" \"" + std::to_string(t) + "\" [0-" + std::to_string(o) + "]");
        add(band + "( " + sub + " )");
    }
    return out;
}

std::string tts_auto_grammar(const audio_lm_prompt_info & pi,
                             const std::string & text) {
    (void) text;  // reserved for future prompt-dependent grammars

    // MOSS-TTSD (and any merged-cb0 parallel-heads-delay model): constrain
    // decode-phase cb0 to the speech-token range ∪ {eos_code_c0}.  The
    // backbone tokens for the speech sub-range detokenize as the decimal
    // pieces "<0>".."<N-1>" (N = range_end - range_start); the end sentinel
    // detokenizes to a fixed control string.  We match those piece strings.
    if (pi.model_kind == audio_lm_prompt_info::KIND_PARALLEL_HEADS_DELAY &&
        pi.cb0_from_backbone &&
        pi.cb0_speech_range_start >= 0 &&
        pi.cb0_speech_range_end   > pi.cb0_speech_range_start) {
        const int32_t n_speech = pi.cb0_speech_range_end - pi.cb0_speech_range_start;
        const std::string num_rule = gbnf_uint_range_rule(n_speech - 1);
        // A speech frame's cb0 is one "<CODE>" piece; audio ends with the
        // end-of-speech sentinel.  The delay-pattern tail frames (cb0 forced
        // to eos while cb1..N flush) are handled by the runtime, not the
        // grammar, so `end` may appear once and then trailing eos repeats are
        // allowed too (end+ ) to cover the flush window.
        std::string g;
        g += "root ::= speech* end+\n";
        g += "speech ::= \"<\" SPEECHID \">\"\n";
        g += "end ::= \"<|end_of_speech|>\"\n";
        g += "SPEECHID ::= " + num_rule + "\n";
        return g;
    }

    return "";
}

// ─── Codebook step machine passthroughs ─────────────────────────────

bool audio_lm_step_set_text_context(audio_lm_context * ctx, int32_t text_token) {
    if (ctx == nullptr || ctx->state == nullptr) {
        if (ctx) ctx->last_error = "audio_lm_step_set_text_context: no codec_lm state";
        return false;
    }
    if (codec_lm_state_set_text_context(ctx->state, text_token) != CODEC_STATUS_SUCCESS) {
        const char * raw = codec_lm_state_get_last_error(ctx->state);
        ctx->last_error = std::string("set_text_context failed (") + (raw ? raw : "?") + ")";
        return false;
    }
    return true;
}

bool audio_lm_step_begin(audio_lm_context * ctx, const float * last_hidden, int32_t hidden_dim) {
    if (ctx == nullptr || ctx->state == nullptr) {
        if (ctx) ctx->last_error = "audio_lm_step_begin: no codec_lm state";
        return false;
    }
    if (last_hidden == nullptr || hidden_dim != ctx->hidden) {
        ctx->last_error = "audio_lm_step_begin: null hidden or wrong dim";
        return false;
    }
    if (codec_lm_step_begin(ctx->state, last_hidden) != CODEC_STATUS_SUCCESS) {
        const char * raw = codec_lm_state_get_last_error(ctx->state);
        ctx->last_error = std::string("step_begin failed (") + (raw ? raw : "?") + ")";
        return false;
    }
    return true;
}

const float * audio_lm_step_logits(audio_lm_context * ctx, int32_t * out_cb_idx, int32_t * out_n) {
    if (ctx == nullptr || ctx->state == nullptr) {
        if (ctx) ctx->last_error = "audio_lm_step_logits: no codec_lm state";
        return nullptr;
    }
    const float * lg = codec_lm_step_logits(ctx->state, out_cb_idx, out_n);
    if (lg == nullptr) {
        const char * raw = codec_lm_state_get_last_error(ctx->state);
        ctx->last_error = std::string("step_logits returned null (") + (raw ? raw : "?") + ")";
    }
    return lg;
}

bool audio_lm_step_push_code(audio_lm_context * ctx, int32_t code) {
    if (ctx == nullptr || ctx->state == nullptr) {
        if (ctx) ctx->last_error = "audio_lm_step_push_code: no codec_lm state";
        return false;
    }
    if (codec_lm_step_push_code(ctx->state, code) != CODEC_STATUS_SUCCESS) {
        const char * raw = codec_lm_state_get_last_error(ctx->state);
        ctx->last_error = std::string("step_push_code failed (") + (raw ? raw : "?") + ")";
        return false;
    }
    return true;
}

bool audio_lm_step_finish(audio_lm_context * ctx, int32_t * out_codes, int32_t n_codes) {
    if (ctx == nullptr || ctx->state == nullptr || out_codes == nullptr) {
        if (ctx) ctx->last_error = "audio_lm_step_finish: null state / out_codes";
        return false;
    }
    if (n_codes < ctx->n_cb) {
        ctx->last_error = "audio_lm_step_finish: out_codes buffer smaller than n_codebook";
        return false;
    }
    if (codec_lm_step_finish(ctx->state, out_codes) != CODEC_STATUS_SUCCESS) {
        const char * raw = codec_lm_state_get_last_error(ctx->state);
        ctx->last_error = std::string("step_finish failed (") + (raw ? raw : "?") + ")";
        return false;
    }
    return true;
}

bool audio_lm_prompt_needs_composed_embd(const audio_lm_context * ctx) {
    return ctx != nullptr && ctx->prompt_needs_composed;
}

bool audio_lm_compose_prompt_embd(audio_lm_context * ctx,
                                  int32_t            text_token,
                                  float *            out_embd,
                                  int32_t            out_dim) {
    if (ctx == nullptr || out_embd == nullptr) return false;
    if (ctx->lm == nullptr) {
        ctx->last_error = "audio_lm_compose_prompt_embd: no codec_lm adaptor";
        return false;
    }
    if (out_dim < ctx->hidden) {
        ctx->last_error = "audio_lm_compose_prompt_embd: out buffer smaller than hidden_dim";
        return false;
    }
    if (ctx->n_cb <= 0) {
        ctx->last_error = "audio_lm_compose_prompt_embd: n_codebook unknown";
        return false;
    }
    // cb0 = raw text token (merged vocab, NOT speech-offset); cb1..N-1 =
    // speech_pad code — exactly the HF processor's prompt grid before the
    // delay shift.  compose_audio_embd sums the per-channel embeddings.
    std::vector<int32_t> codes((size_t) ctx->n_cb, ctx->speech_pad_code);
    codes[0] = text_token;
    const enum codec_status rc =
        codec_lm_compose_audio_embd(ctx->lm, codes.data(), out_embd);
    if (rc != CODEC_STATUS_SUCCESS) {
        const char * raw = codec_lm_get_last_error(ctx->lm);
        ctx->last_error = std::string("audio_lm_compose_prompt_embd: compose failed (")
                           + (raw && *raw ? raw : "?") + ")";
        return false;
    }
    return true;
}

bool audio_lm_compose_audio_codes_embd(audio_lm_context * ctx,
                                       const int32_t *    codes,
                                       int32_t            n_codes,
                                       float *            out_embd,
                                       int32_t            out_dim) {
    if (ctx == nullptr || codes == nullptr || out_embd == nullptr) return false;
    if (ctx->lm == nullptr) {
        ctx->last_error = "audio_lm_compose_audio_codes_embd: no codec_lm adaptor";
        return false;
    }
    if (out_dim < ctx->hidden) {
        ctx->last_error = "audio_lm_compose_audio_codes_embd: out buffer smaller than hidden_dim";
        return false;
    }
    if (n_codes != ctx->n_cb) {
        ctx->last_error = "audio_lm_compose_audio_codes_embd: n_codes != n_codebook";
        return false;
    }
    const enum codec_status rc =
        codec_lm_compose_audio_embd(ctx->lm, codes, out_embd);
    if (rc != CODEC_STATUS_SUCCESS) {
        const char * raw = codec_lm_get_last_error(ctx->lm);
        ctx->last_error = std::string("audio_lm_compose_audio_codes_embd: compose failed (")
                           + (raw && *raw ? raw : "?") + ")";
        return false;
    }
    return true;
}

bool audio_lm_talker_has_projection(const audio_lm_context * ctx) {
    return ctx != nullptr && ctx->has_talker_proj &&
           ctx->q3_nothink_id >= 0 && ctx->q3_codec_bos_id >= 0;
}

// Helper: codec lane lookup — codec_embedding[code] (= audio_embd cb 0),
// dequanting from the F16 table.
static bool talker_codec_embd(audio_lm_context * ctx, int32_t code,
                              float * out, int32_t dim) {
    if (!codec_lm_codec_embd_row(ctx->lm, code, out, dim)) {
        const char * raw = codec_lm_get_last_error(ctx->lm);
        ctx->last_error = std::string("talker: codec_embedding lookup failed (")
                          + (raw && *raw ? raw : "?") + ")";
        return false;
    }
    return true;
}

bool audio_lm_build_talker_prefix(audio_lm_context * ctx,
                                  const int32_t *    role_tokens,
                                  int32_t            n_role,
                                  const int32_t *    text_tokens,
                                  int32_t            n_text,
                                  const float *      xvector,
                                  int32_t            xvec_dim,
                                  float *            out_embds,
                                  int32_t            out_cap_rows,
                                  int32_t *          out_n_rows,
                                  int32_t *          out_text_consumed) {
    if (!ctx || !out_embds || !out_n_rows || !out_text_consumed) return false;
    if (!audio_lm_talker_has_projection(ctx)) {
        ctx->last_error = "audio_lm_build_talker_prefix: no talker projection";
        return false;
    }
    const int32_t H = ctx->hidden;
    if (xvector != nullptr && xvec_dim != H) {
        ctx->last_error = "audio_lm_build_talker_prefix: xvector dim != hidden";
        return false;
    }
    if (n_text < 1) {
        ctx->last_error = "audio_lm_build_talker_prefix: need >=1 text token";
        return false;
    }

    // The codec control stream (auto-language, with x-vector row inserted):
    //   [nothink, think_bos, think_eos, <XVEC>, codec_pad, codec_bos]
    // The text lane aligned to it (all but last col):
    //   [tts_pad, tts_pad, tts_pad, tts_pad, tts_bos]
    // and text[0] is summed with the final codec col (codec_bos).
    const int32_t n_ctrl = 5 + (xvector ? 1 : 0);   // control rows (excl. text[0] row)
    const int32_t n_rows = n_role + n_ctrl + 1;      // + text[0] row
    if (n_rows > out_cap_rows) {
        ctx->last_error = "audio_lm_build_talker_prefix: output row cap too small";
        return false;
    }

    std::vector<float> tmp((size_t) H);
    auto proj = [&](int32_t tok, float * dst) -> bool {
        if (!codec_lm_project_text(ctx->lm, tok, dst, H)) {
            const char * raw = codec_lm_get_last_error(ctx->lm);
            ctx->last_error = std::string("talker: project_text failed (")
                              + (raw && *raw ? raw : "?") + ")";
            return false;
        }
        return true;
    };

    int32_t r = 0;
    // Role header: projected text only (codec lane empty → zero).
    for (int32_t i = 0; i < n_role; ++i, ++r) {
        if (!proj(role_tokens[i], out_embds + (size_t) r * H)) return false;
    }
    // Control stream text lane = tts_pad for all but the last (tts_bos).
    // codec lane cycles [nothink, think_bos, think_eos, XVEC, codec_pad].
    struct CtrlRow { int32_t text_tok; int32_t codec_tag; bool is_xvec; };
    std::vector<CtrlRow> ctrl;
    ctrl.push_back({ctx->q3_tts_pad_id, ctx->q3_nothink_id,   false});
    ctrl.push_back({ctx->q3_tts_pad_id, ctx->q3_think_bos_id, false});
    ctrl.push_back({ctx->q3_tts_pad_id, ctx->q3_think_eos_id, false});
    if (xvector) ctrl.push_back({ctx->q3_tts_pad_id, -1, true});
    ctrl.push_back({ctx->q3_tts_bos_id, ctx->q3_codec_pad_id, false});
    for (const CtrlRow & c : ctrl) {
        float * dst = out_embds + (size_t) r * H;
        if (!proj(c.text_tok, dst)) return false;
        if (c.is_xvec) {
            for (int32_t i = 0; i < H; ++i) dst[i] += xvector[i];
        } else {
            if (!talker_codec_embd(ctx, c.codec_tag, tmp.data(), H)) return false;
            for (int32_t i = 0; i < H; ++i) dst[i] += tmp[i];
        }
        ++r;
    }
    // Final row: text_proj(text[0]) + codec_embd[codec_bos].
    {
        float * dst = out_embds + (size_t) r * H;
        if (!proj(text_tokens[0], dst)) return false;
        if (!talker_codec_embd(ctx, ctx->q3_codec_bos_id, tmp.data(), H)) return false;
        for (int32_t i = 0; i < H; ++i) dst[i] += tmp[i];
        ++r;
    }

    *out_n_rows        = r;
    *out_text_consumed = 1;   // text[0] folded into the prefix
    return true;
}

bool audio_lm_talker_trailing_text_embd(audio_lm_context * ctx,
                                        const int32_t *    text_tokens,
                                        int32_t            n_text,
                                        int32_t            trailing_idx,
                                        float *            out_embd,
                                        int32_t            out_dim) {
    if (!ctx || !out_embd) return false;
    if (out_dim < ctx->hidden) {
        ctx->last_error = "talker_trailing_text_embd: out buffer too small";
        return false;
    }
    // trailing_idx counts from the first *trailing* token (text[1]).  Once
    // the text is exhausted, feed tts_eos (matches the reference's
    // trailing_text_hidden = proj(text[1:]) ++ tts_eos).
    const int32_t tok_pos = trailing_idx + 1;   // maps to text_tokens[tok_pos]
    const int32_t tok = (tok_pos < n_text && text_tokens)
                      ? text_tokens[tok_pos]
                      : ctx->q3_tts_eos_id;
    if (!codec_lm_project_text(ctx->lm, tok, out_embd, out_dim)) {
        const char * raw = codec_lm_get_last_error(ctx->lm);
        ctx->last_error = std::string("talker_trailing_text_embd: project failed (")
                          + (raw && *raw ? raw : "?") + ")";
        return false;
    }
    return true;
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

    // ── LM codes → codec quantizer codes transform ──────────────────────
    // The AR loop accumulated the FULL (T, n_cb) frame, including a possible
    // text/control cb0 and a per-codebook emission delay.  Mirror the
    // upstream MOSS processor (see rn-tts decodeAudioTokens):
    //   * slice codebooks [0, audio_cb_offset) — they aren't audio levels;
    //   * reverse the delay_pattern shift so codebook q at output frame t
    //     comes from input frame t + delay[audio_cb_offset + q];
    //   * decode the remaining n_q = n_cb - offset codebooks (RVQ decodes
    //     with fewer levels than the codec's native n_q — Realtime's codec
    //     has 32 levels but the LM only predicts the first 16/15).
    const int32_t n_cb_in = ctx->n_cb;
    const int32_t offset  = ctx->audio_cb_offset;
    const int32_t n_q     = n_cb_in - offset;
    if (n_q <= 0) {
        ctx->last_error = "audio_lm_decode_audio: audio_cb_offset >= n_codebook";
        return false;
    }

    // Per-audio-codebook delays (indexed within the audio slice).
    std::vector<int32_t> audio_delays((size_t) n_q, 0);
    int32_t max_delay = 0;
    if (!ctx->delay_pattern.empty() &&
        (int32_t) ctx->delay_pattern.size() >= n_cb_in) {
        for (int32_t q = 0; q < n_q; ++q) {
            const int32_t d = ctx->delay_pattern[(size_t) (offset + q)];
            audio_delays[(size_t) q] = d;
            if (d > max_delay) max_delay = d;
        }
    }

    const int32_t n_frames_in = ctx->codes_n_frames;
    if (max_delay > 0 && n_frames_in <= max_delay) {
        ctx->last_error = "audio_lm_decode_audio: too few frames to cover delay_pattern";
        return false;
    }
    const int32_t n_frames_out = (max_delay > 0) ? (n_frames_in - max_delay) : n_frames_in;

    const int32_t codebook_sz = codec_model_codebook_size(ctx->model);
    std::vector<int32_t> decode_codes;
    const int32_t * codes_ptr = ctx->codes.data();
    if (offset > 0 || max_delay > 0 || ctx->cb0_speech_offset != 0) {
        decode_codes.resize((size_t) n_frames_out * (size_t) n_q);
        for (int32_t t = 0; t < n_frames_out; ++t) {
            for (int32_t q = 0; q < n_q; ++q) {
                const int32_t src_t = t + audio_delays[(size_t) q];
                int32_t code = ctx->codes[(size_t) src_t * n_cb_in + (offset + q)];
                // Merged text+speech cb0 (MOSS-TTSD): map back to raw
                // quantizer index space.  Only the first *audio* codebook
                // (q==0 after any pure-control slice) carries the offset.
                if (q == 0 && ctx->cb0_speech_offset != 0) {
                    code -= ctx->cb0_speech_offset;
                }
                // Guard the codec's embedding get_rows against pad / control
                // codes (speech_pad=1024, bos/eos sentinels) that the LM can
                // emit before stop — the HF processor drops such frames; we
                // clamp into the valid quantizer range instead of aborting.
                if (codebook_sz > 0) {
                    if (code < 0)            code = 0;
                    if (code >= codebook_sz) code = codebook_sz - 1;
                }
                decode_codes[(size_t) t * n_q + q] = code;
            }
        }
        codes_ptr = decode_codes.data();
    }

    codec_token_buffer tokens = {};
    tokens.data         = const_cast<int32_t *>(codes_ptr);
    tokens.n_tokens     = n_frames_out * n_q;
    tokens.n_frames     = n_frames_out;
    tokens.n_q          = n_q;
    tokens.codebook_size = codec_model_codebook_size(ctx->model);
    tokens.sample_rate   = codec_model_sample_rate(ctx->model);
    tokens.hop_size      = codec_model_hop_size(ctx->model);

    codec_pcm_buffer pcm = {};
    auto dp = codec_decode_default_params();
    dp.n_q = n_q;
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
