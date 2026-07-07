#include "lm_internal.h"
#include "speaker_chatterbox.h"
#include "speaker_qwen3_tts.h"

#include "../runtime/graph.h"
#include "../runtime/graph_exec.h"
#include "../runtime/tensor_utils.h"

#include <ggml.h>
#include <gguf.h>

#include <cstring>
#include <new>
#include <string>

// =====================================================================
// codec_lm — public API entry, kind dispatch, state lifecycle.
//
// Every state owns its own codec_context (graph cache + scheduler) but
// shares the codec_model's backend + weights.  This lets multiple
// generations run in parallel and keeps the graph caches per-state, so
// a kind's per-step graphs (e.g. parallel_heads_delay's matmul cluster)
// only build once per state.
// =====================================================================

// ---------------------------------------------------------------------
// kind <-> string mapping
// ---------------------------------------------------------------------

static const char * codec_lm_kind_name_internal(enum codec_lm_kind kind) {
    switch (kind) {
        case CODEC_LM_KIND_PARALLEL_HEADS_DELAY:  return "parallel_heads_delay";
        case CODEC_LM_KIND_RESIDUAL_DEPTH_AR:     return "residual_depth_ar";
        case CODEC_LM_KIND_CONTINUOUS_LATENT_CFM: return "continuous_latent_cfm";
        case CODEC_LM_KIND_FLOW_LM:               return "flow_lm";
        case CODEC_LM_KIND_UNKNOWN:               break;
    }
    return "unknown";
}

const char * codec_lm_kind_name(enum codec_lm_kind kind) {
    return codec_lm_kind_name_internal(kind);
}

enum codec_lm_kind codec_lm_kind_from_string(const char * s) {
    if (s == nullptr || s[0] == '\0') {
        return CODEC_LM_KIND_UNKNOWN;
    }
    if (std::strcmp(s, "parallel_heads_delay") == 0) {
        return CODEC_LM_KIND_PARALLEL_HEADS_DELAY;
    }
    if (std::strcmp(s, "residual_depth_ar") == 0) {
        return CODEC_LM_KIND_RESIDUAL_DEPTH_AR;
    }
    if (std::strcmp(s, "continuous_latent_cfm") == 0) {
        return CODEC_LM_KIND_CONTINUOUS_LATENT_CFM;
    }
    if (std::strcmp(s, "flow_lm") == 0) {
        return CODEC_LM_KIND_FLOW_LM;
    }
    return CODEC_LM_KIND_UNKNOWN;
}

const codec_lm_kind_vtable * codec_lm_vtable_for_kind(enum codec_lm_kind kind) {
    switch (kind) {
        case CODEC_LM_KIND_PARALLEL_HEADS_DELAY:  return &codec_lm_vtable_parallel_heads_delay;
        case CODEC_LM_KIND_RESIDUAL_DEPTH_AR:     return &codec_lm_vtable_residual_depth_ar;
        case CODEC_LM_KIND_CONTINUOUS_LATENT_CFM: return &codec_lm_vtable_continuous_latent_cfm;
        case CODEC_LM_KIND_FLOW_LM:               return &codec_lm_vtable_flow_lm;
        case CODEC_LM_KIND_UNKNOWN:               break;
    }
    return nullptr;
}

// ---------------------------------------------------------------------
// GGUF metadata helpers (codec.lm.* namespace)
// ---------------------------------------------------------------------

std::string codec_lm_read_string_kv(const codec_model * codec, const char * key) {
    if (codec == nullptr || codec->gguf == nullptr || key == nullptr) {
        return std::string();
    }
    const int kid = lm_gguf_find_key(codec->gguf, key);
    if (kid < 0) {
        return std::string();
    }
    return codec_lm_gguf_value_to_string(codec->gguf, kid);
}

// ---------------------------------------------------------------------
// Tensor shape validation (shared between kinds that use unfused per-cb
// audio embedding / output head tables).
// ---------------------------------------------------------------------

bool codec_lm_check_unfused_audio_tables(
    codec_lm * lm,
    int32_t hidden_dim,
    const std::vector<int32_t> & codebook_sizes,
    bool tied_heads) {

    if (lm == nullptr || lm->codec == nullptr || lm->codec->weights == nullptr) {
        return false;
    }
    if (hidden_dim <= 0 || codebook_sizes.empty()) {
        lm->last_error = "invalid hidden_dim or n_codebook";
        return false;
    }

    char buf[64];
    for (size_t i = 0; i < codebook_sizes.size(); ++i) {
        const int32_t v = codebook_sizes[i];
        if (v <= 0) {
            lm->last_error = "codec.lm.codebook_sizes contains a non-positive entry";
            return false;
        }

        std::snprintf(buf, sizeof(buf), "lm.audio_embd_%zu.weight", i);
        lm_ggml_tensor * t_e = lm_ggml_get_tensor(lm->codec->weights, buf);
        if (t_e == nullptr) {
            lm->last_error = std::string("missing tensor: ") + buf;
            return false;
        }
        if (t_e->ne[0] != hidden_dim || t_e->ne[1] != v) {
            lm->last_error = std::string("shape mismatch on ") + buf +
                             " (expected [hidden, vocab])";
            return false;
        }

        if (!tied_heads) {
            std::snprintf(buf, sizeof(buf), "lm.heads_%zu.weight", i);
            lm_ggml_tensor * t_h = lm_ggml_get_tensor(lm->codec->weights, buf);
            if (t_h == nullptr) {
                lm->last_error = std::string("missing tensor: ") + buf;
                return false;
            }
            if (t_h->ne[0] != hidden_dim || t_h->ne[1] != v) {
                lm->last_error = std::string("shape mismatch on ") + buf +
                                 " (expected [hidden, vocab])";
                return false;
            }
        }
    }
    return true;
}

// ---------------------------------------------------------------------
// codec_lm lifecycle
// ---------------------------------------------------------------------

static bool codec_lm_populate_info(codec_lm * lm) {
    lm_gguf_context * gf = lm->codec->gguf;
    if (gf == nullptr) {
        lm->last_error = "codec_model has no GGUF context";
        return false;
    }

    if (!codec_read_bool_kv(gf, "codec.lm.has_adaptor", false)) {
        lm->last_error = "codec.lm.has_adaptor is absent or false";
        return false;
    }

    const std::string kind_s = codec_lm_read_string_kv(lm->codec, "codec.lm.kind");
    lm->kind = codec_lm_kind_from_string(kind_s.c_str());
    if (lm->kind == CODEC_LM_KIND_UNKNOWN) {
        lm->last_error = "unrecognised codec.lm.kind: '" + kind_s + "'";
        return false;
    }
    lm->vtable = codec_lm_vtable_for_kind(lm->kind);
    if (lm->vtable == nullptr) {
        lm->last_error = "no implementation for codec.lm.kind: " + kind_s;
        return false;
    }

    int32_t hidden             = codec_read_i32_kv(gf, "codec.lm.hidden_dim", 0);

    // FlowLM (Pocket-TTS): self-contained continuous-latent AR.  No codebooks,
    // no external backbone; uses codec.lm.d_model / ldim.  Populate minimal info
    // and skip the codebook + continuous_latent_cfm metadata paths below.
    if (lm->kind == CODEC_LM_KIND_FLOW_LM) {
        const int32_t d_model    = codec_read_i32_kv(gf, "codec.lm.d_model", 0);
        const int32_t ldim       = codec_read_i32_kv(gf, "codec.lm.ldim", 0);
        if (d_model <= 0 || ldim <= 0) {
            lm->last_error = "codec.lm flow_lm: d_model / ldim must be > 0";
            return false;
        }
        lm->host_arch_buf   = codec_lm_read_string_kv(lm->codec, "codec.lm.host_arch");
        lm->info.kind          = lm->kind;
        lm->info.hidden_dim    = d_model;
        lm->info.is_continuous = true;
        lm->info.latent_dim    = ldim;
        lm->info.patch_size    = 1;
        lm->info.n_codebook    = 0;
        lm->info.codebook_sizes = nullptr;
        lm->info.delay_pattern  = nullptr;
        lm->info.host_arch      = lm->host_arch_buf.empty() ? "" : lm->host_arch_buf.c_str();
        lm->info.eos_code_c0    = -1;
        lm->info.eos_min_step   = 0;
        return true;
    }

    // Continuous-latent kinds (CONTINUOUS_LATENT_CFM) don't have codebooks;
    // they emit a continuous latent patch per step.  Populate the continuous
    // info fields and skip the codebook-centric metadata below.
    if (lm->kind == CODEC_LM_KIND_CONTINUOUS_LATENT_CFM) {
        const int32_t patch_size = codec_read_i32_kv(gf, "codec.lm.patch_size", 0);
        const int32_t latent_dim = codec_read_i32_kv(gf, "codec.lm.latent_dim", 0);
        if (hidden <= 0 || patch_size <= 0 || latent_dim <= 0) {
            lm->last_error = "codec.lm continuous: hidden_dim / patch_size / latent_dim must be > 0";
            return false;
        }
        lm->host_arch_buf = codec_lm_read_string_kv(lm->codec, "codec.lm.host_arch");
        lm->info.kind          = lm->kind;
        lm->info.hidden_dim    = hidden;
        lm->info.is_continuous = true;
        lm->info.patch_size    = patch_size;
        lm->info.latent_dim    = latent_dim;
        lm->info.n_codebook    = 0;
        lm->info.codebook_sizes = nullptr;
        lm->info.delay_pattern  = nullptr;
        lm->info.host_arch      = lm->host_arch_buf.empty() ? "" : lm->host_arch_buf.c_str();
        // speaker section + vtable init are handled by the shared tail below.
    } else {
    const int32_t audio_embd_d = codec_read_i32_kv(gf, "codec.lm.audio_embed_dim", hidden);
    // Optional separate compose embed dim (LFM2-Audio).  When absent /
    // zero, compose output matches the depth-side audio_embed_dim
    // (CSM/Qwen3-TTS) or the model doesn't support compose at all
    // (Moshi — caller composes via backbone).  The default below picks
    // audio_embed_dim, which is the right answer for CSM/Qwen3-TTS;
    // Moshi GGUFs do nothing to expose compose either way, so this
    // field is informational only.
    const int32_t compose_embd_d = codec_read_i32_kv(
        gf, "codec.lm.compose.audio_embed_dim", audio_embd_d);
    const int32_t n_cb         = codec_read_i32_kv(gf, "codec.lm.n_codebook", 0);
    if (hidden <= 0 || audio_embd_d <= 0 || n_cb <= 0) {
        lm->last_error = "codec.lm metadata: hidden_dim / audio_embed_dim / n_codebook must be > 0";
        return false;
    }

    codec_read_i32_array_kv_vec(gf, "codec.lm.codebook_sizes", &lm->codebook_sizes_buf);
    if ((int32_t) lm->codebook_sizes_buf.size() != n_cb) {
        lm->last_error = "codec.lm.codebook_sizes length must equal n_codebook";
        return false;
    }

    lm->delay_pattern_buf.assign((size_t) n_cb, 0);
    codec_read_i32_array_kv(gf, "codec.lm.delay_pattern", lm->delay_pattern_buf.data(), n_cb);
    // (absent => left as zeros, which is the correct default)

    lm->host_arch_buf = codec_lm_read_string_kv(lm->codec, "codec.lm.host_arch");

    lm->info.kind                    = lm->kind;
    lm->info.hidden_dim              = hidden;
    lm->info.audio_embed_dim         = audio_embd_d;
    lm->info.compose_audio_embed_dim = compose_embd_d;
    lm->info.n_codebook              = n_cb;
    lm->info.codebook_sizes   = lm->codebook_sizes_buf.data();
    lm->info.delay_pattern    = lm->delay_pattern_buf.data();
    lm->info.host_arch        = lm->host_arch_buf.empty() ? "" : lm->host_arch_buf.c_str();
    }

    // End-of-audio metadata (applies to every kind; continuous kinds get
    // the -1 default since they signal stop via step_generate).  Default
    // eos_code_c0 = -1 (no sentinel), eos_min_step = 0.
    lm->info.eos_code_c0  = codec_read_i32_kv(gf, "codec.lm.eos_code_c0", -1);
    lm->info.eos_min_step = codec_read_i32_kv(gf, "codec.lm.eos_min_step", 0);

    // Speaker-conditioning encoder section is optional.  Absent means
    // codec_lm_speaker_get_info returns NULL and codec_lm_speaker_encode
    // returns CODEC_STATUS_NOT_SUPPORTED.
    lm->has_speaker_encoder = codec_read_bool_kv(gf, "codec.speaker.has_encoder", false);
    if (lm->has_speaker_encoder) {
        codec_lm_speaker_info & si = lm->speaker_info;
        si.needs_ref_pcm           = codec_read_bool_kv(gf, "codec.speaker.needs_ref_pcm", false);
        si.needs_ref_speech_tokens = codec_read_bool_kv(gf, "codec.speaker.needs_ref_speech_tokens", false);
        si.needs_emotion_scalar    = codec_read_bool_kv(gf, "codec.speaker.needs_emotion_scalar", false);
        si.ref_sample_rate         = codec_read_i32_kv (gf, "codec.speaker.ref_sample_rate", 0);
        si.emotion_default         = codec_read_f32_kv (gf, "codec.speaker.emotion_default", 0.5f);
        si.n_rows                  = codec_read_i32_kv (gf, "codec.speaker.n_rows", 0);
        si.hidden_dim              = codec_read_i32_kv (gf, "codec.speaker.hidden_dim", hidden);
        // VE / wav2vec / x-vector / … all expose a single intermediate
        // dim; Chatterbox: 256.  Reused by `_from_embedding`.
        si.speaker_emb_dim         = codec_read_i32_kv (gf, "codec.speaker.speaker_emb_dim", 0);
        if (si.n_rows <= 0 || si.hidden_dim <= 0) {
            lm->last_error =
                "codec.speaker.has_encoder=true but n_rows / hidden_dim missing or non-positive";
            return false;
        }
        if (si.needs_ref_pcm && si.ref_sample_rate <= 0) {
            lm->last_error =
                "codec.speaker.needs_ref_pcm=true but ref_sample_rate missing or non-positive";
            return false;
        }

        // Dispatch on `codec.speaker.encoder_arch` to choose the impl.
        const std::string arch_s =
            codec_lm_read_string_kv(lm->codec, "codec.speaker.encoder_arch");
        if (arch_s == "chatterbox_voice_encoder") {
            lm->speaker_arch = codec_lm::CODEC_SPEAKER_ARCH_CHATTERBOX_VOICE_ENC;
        } else if (arch_s == "qwen3_tts_ecapa_tdnn") {
            // Recognised at the dispatch layer; runtime impl is a stub
            // (codec_lm_speaker_encode returns NOT_SUPPORTED).  Lets the
            // codec be loaded + introspected (info struct, n_rows,
            // hidden_dim) while the ECAPA-TDNN port lands.
            lm->speaker_arch = codec_lm::CODEC_SPEAKER_ARCH_QWEN3_TTS_ECAPA_TDNN;
        } else {
            lm->last_error =
                "codec.speaker.encoder_arch='" + arch_s + "' is not recognised "
                "by this codec.cpp build";
            return false;
        }
    }
    return true;
}

// Per-arch init / free dispatchers.
static bool speaker_arch_init(codec_lm * lm) {
    switch (lm->speaker_arch) {
        case codec_lm::CODEC_SPEAKER_ARCH_CHATTERBOX_VOICE_ENC:
            return chatterbox_speaker_init(lm);
        case codec_lm::CODEC_SPEAKER_ARCH_QWEN3_TTS_ECAPA_TDNN:
            return qwen3_tts_speaker_init(lm);
        case codec_lm::CODEC_SPEAKER_ARCH_NONE:
            return true;
    }
    return false;
}

static void speaker_arch_free(codec_lm * lm) {
    switch (lm->speaker_arch) {
        case codec_lm::CODEC_SPEAKER_ARCH_CHATTERBOX_VOICE_ENC:
            chatterbox_speaker_free(lm);
            break;
        case codec_lm::CODEC_SPEAKER_ARCH_QWEN3_TTS_ECAPA_TDNN:
            qwen3_tts_speaker_free(lm);
            break;
        case codec_lm::CODEC_SPEAKER_ARCH_NONE:
            break;
    }
}

// Thread-local fallback for the most recent codec_lm_create failure.
// Callers that get NULL back from `codec_lm_create` can read the reason
// via `codec_lm_get_create_error` without a valid lm handle.
static thread_local std::string s_codec_lm_create_error;

struct codec_lm * codec_lm_create(struct codec_model * codec) {
    s_codec_lm_create_error.clear();
    if (codec == nullptr) {
        s_codec_lm_create_error = "codec_lm_create: codec is NULL";
        return nullptr;
    }
    codec_lm * lm = new (std::nothrow) codec_lm();
    if (lm == nullptr) {
        s_codec_lm_create_error = "codec_lm_create: out of memory";
        return nullptr;
    }
    lm->codec = codec;

    if (!codec_lm_populate_info(lm)) {
        s_codec_lm_create_error = lm->last_error;
        delete lm;
        return nullptr;
    }

    if (lm->vtable->init == nullptr || !lm->vtable->init(lm)) {
        s_codec_lm_create_error = lm->last_error.empty()
            ? "codec_lm_create: init returned false (no detail)"
            : lm->last_error;
        if (lm->vtable->free != nullptr) {
            lm->vtable->free(lm);
        }
        delete lm;
        return nullptr;
    }

    if (lm->has_speaker_encoder && !speaker_arch_init(lm)) {
        s_codec_lm_create_error = lm->last_error.empty()
            ? "codec_lm_create: speaker init returned false (no detail)"
            : lm->last_error;
        if (lm->vtable->free != nullptr) lm->vtable->free(lm);
        delete lm;
        return nullptr;
    }

    return lm;
}

const char * codec_lm_get_create_error(void) {
    return s_codec_lm_create_error.c_str();
}

void codec_lm_free(struct codec_lm * lm) {
    if (lm == nullptr) {
        return;
    }
    speaker_arch_free(lm);
    codec_lm_chatterbox_free_state(lm);
    if (lm->vtable != nullptr && lm->vtable->free != nullptr) {
        lm->vtable->free(lm);
    }
    delete lm;
}

const struct codec_lm_info * codec_lm_get_info(const struct codec_lm * lm) {
    return lm == nullptr ? nullptr : &lm->info;
}

const char * codec_lm_get_last_error(const struct codec_lm * lm) {
    return lm == nullptr ? "" : lm->last_error.c_str();
}

const char * codec_lm_state_get_last_error(const struct codec_lm_state * st) {
    return st == nullptr ? "" : st->last_error.c_str();
}

// ---------------------------------------------------------------------
// codec_lm_state lifecycle
// ---------------------------------------------------------------------

struct codec_lm_state * codec_lm_state_new(struct codec_lm * lm) {
    if (lm == nullptr || lm->vtable == nullptr) {
        return nullptr;
    }
    codec_lm_state * st = new (std::nothrow) codec_lm_state();
    if (st == nullptr) {
        return nullptr;
    }
    st->lm = lm;

    // Each state owns a codec_context that borrows the codec_model's
    // backend and weights but has its own graph cache and scheduler.
    codec_context * ctx = new (std::nothrow) codec_context();
    if (ctx == nullptr) {
        delete st;
        return nullptr;
    }
    ctx->model   = lm->codec;
    ctx->backend = lm->codec->backend;
    ctx->params  = codec_context_default_params();
    std::string err;
    if (!codec_runtime_init(ctx, &err)) {
        delete ctx;
        delete st;
        return nullptr;
    }
    st->ctx = ctx;

    st->codes_buf.assign((size_t) lm->info.n_codebook, 0);
    st->next_cb            = 0;
    st->step_in_progress   = false;
    st->logits_pending     = false;
    st->ar_frame           = 0;

    if (lm->vtable->state_init != nullptr && !lm->vtable->state_init(st)) {
        if (lm->vtable->state_free != nullptr) {
            lm->vtable->state_free(st);
        }
        codec_runtime_free(ctx);
        delete ctx;
        delete st;
        return nullptr;
    }
    return st;
}

void codec_lm_state_free(struct codec_lm_state * st) {
    if (st == nullptr) {
        return;
    }
    if (st->lm != nullptr && st->lm->vtable != nullptr && st->lm->vtable->state_free != nullptr) {
        st->lm->vtable->state_free(st);
    }
    if (st->ctx != nullptr) {
        codec_runtime_free(st->ctx);
        delete st->ctx;
    }
    delete st;
}

void codec_lm_state_reset(struct codec_lm_state * st) {
    if (st == nullptr) {
        return;
    }
    st->next_cb            = 0;
    st->step_in_progress   = false;
    st->logits_pending     = false;
    st->text_token_context = -1;
    st->ar_frame           = 0;
    std::fill(st->codes_buf.begin(), st->codes_buf.end(), 0);
    if (st->lm != nullptr && st->lm->vtable != nullptr && st->lm->vtable->state_reset != nullptr) {
        st->lm->vtable->state_reset(st);
    }
}

enum codec_status codec_lm_state_set_text_context(
        struct codec_lm_state * st, int32_t text_token) {
    if (st == nullptr) {
        return CODEC_STATUS_INVALID_ARG;
    }
    st->text_token_context = text_token;
    st->last_error.clear();
    return CODEC_STATUS_SUCCESS;
}

// ---------------------------------------------------------------------
// audio embed lookup + compose — delegate to vtable
// ---------------------------------------------------------------------

const float * codec_lm_audio_embd(struct codec_lm * lm, int32_t cb_idx, int32_t code) {
    if (lm == nullptr || lm->vtable == nullptr || lm->vtable->audio_embd == nullptr) {
        return nullptr;
    }
    if (cb_idx < 0 || cb_idx >= lm->info.n_codebook) {
        return nullptr;
    }
    if (code < 0 || code >= lm->info.codebook_sizes[cb_idx]) {
        return nullptr;
    }
    return lm->vtable->audio_embd(lm, cb_idx, code);
}

enum codec_status codec_lm_compose_audio_embd(
    struct codec_lm * lm,
    const int32_t * codes,
    float * out_embd) {
    if (lm == nullptr || codes == nullptr || out_embd == nullptr) {
        return CODEC_STATUS_INVALID_ARG;
    }
    if (lm->vtable == nullptr || lm->vtable->compose_audio_embd == nullptr) {
        return CODEC_STATUS_NOT_SUPPORTED;
    }
    return lm->vtable->compose_audio_embd(lm, codes, out_embd);
}

enum codec_status codec_lm_compose_next_embd(
    struct codec_lm * lm,
    const int32_t * codes,
    int32_t step,
    float * out_embd) {
    if (lm == nullptr || codes == nullptr || out_embd == nullptr) {
        return CODEC_STATUS_INVALID_ARG;
    }
    if (lm->vtable == nullptr) {
        return CODEC_STATUS_NOT_SUPPORTED;
    }
    // Prefer the kind-specific impl (pos-aware).  Fall back to plain
    // compose_audio_embd when the kind doesn't have a per-step pos emb
    // — step is then ignored, matching the documented contract.
    if (lm->vtable->compose_next_embd != nullptr) {
        return lm->vtable->compose_next_embd(lm, codes, step, out_embd);
    }
    if (lm->vtable->compose_audio_embd != nullptr) {
        return lm->vtable->compose_audio_embd(lm, codes, out_embd);
    }
    return CODEC_STATUS_NOT_SUPPORTED;
}

// ---------------------------------------------------------------------
// step machine — delegate to vtable, with a thin layer of state-machine
// invariant checking so kind impls don't have to repeat it.
// ---------------------------------------------------------------------

enum codec_status codec_lm_step_begin(
    struct codec_lm_state * st,
    const float * h_in) {
    if (st == nullptr || st->lm == nullptr || st->lm->vtable == nullptr) {
        return CODEC_STATUS_INVALID_ARG;
    }
    if (h_in == nullptr) {
        return CODEC_STATUS_INVALID_ARG;
    }
    if (st->step_in_progress) {
        st->last_error = "codec_lm_step_begin called twice without step_finish";
        return CODEC_STATUS_INVALID_STATE;
    }
    if (st->lm->vtable->step_begin == nullptr) {
        return CODEC_STATUS_NOT_SUPPORTED;
    }

    st->next_cb            = 0;
    st->logits_pending     = false;
    std::fill(st->codes_buf.begin(), st->codes_buf.end(), 0);

    enum codec_status rc = st->lm->vtable->step_begin(st, h_in);
    if (rc != CODEC_STATUS_SUCCESS) {
        return rc;
    }
    st->step_in_progress = true;
    return CODEC_STATUS_SUCCESS;
}

bool codec_lm_step_pending(const struct codec_lm_state * st) {
    if (st == nullptr || !st->step_in_progress) {
        return false;
    }
    return st->next_cb < (st->lm != nullptr ? st->lm->info.n_codebook : 0);
}

const float * codec_lm_step_logits(
    struct codec_lm_state * st,
    int32_t * out_cb_idx,
    int32_t * out_n) {
    if (st == nullptr || st->lm == nullptr || st->lm->vtable == nullptr) {
        return nullptr;
    }
    if (!st->step_in_progress) {
        st->last_error = "codec_lm_step_logits called outside a step (no step_begin)";
        return nullptr;
    }
    if (st->logits_pending) {
        st->last_error = "codec_lm_step_logits called twice without push_code";
        return nullptr;
    }
    if (st->next_cb >= st->lm->info.n_codebook) {
        st->last_error = "codec_lm_step_logits called past n_codebook";
        return nullptr;
    }
    if (st->lm->vtable->step_logits == nullptr) {
        return nullptr;
    }

    int32_t cb_idx = -1;
    int32_t n      = 0;
    const float * lg = st->lm->vtable->step_logits(st, &cb_idx, &n);
    if (lg == nullptr) {
        return nullptr;
    }
    if (cb_idx != st->next_cb) {
        st->last_error = "kind step_logits returned wrong cb_idx (state machine corrupted)";
        return nullptr;
    }
    if (out_cb_idx != nullptr) *out_cb_idx = cb_idx;
    if (out_n      != nullptr) *out_n      = n;
    st->logits_pending = true;
    return lg;
}

enum codec_status codec_lm_step_push_code(
    struct codec_lm_state * st,
    int32_t code) {
    if (st == nullptr || st->lm == nullptr || st->lm->vtable == nullptr) {
        return CODEC_STATUS_INVALID_ARG;
    }
    if (!st->step_in_progress) {
        st->last_error = "codec_lm_step_push_code called outside a step";
        return CODEC_STATUS_INVALID_STATE;
    }
    if (!st->logits_pending) {
        st->last_error = "codec_lm_step_push_code called without a preceding step_logits";
        return CODEC_STATUS_INVALID_STATE;
    }
    const int32_t cb = st->next_cb;
    if (code < 0 || code >= st->lm->info.codebook_sizes[cb]) {
        st->last_error = "code out of range for codebook";
        return CODEC_STATUS_INVALID_ARG;
    }

    if (st->lm->vtable->step_push_code != nullptr) {
        enum codec_status rc = st->lm->vtable->step_push_code(st, code);
        if (rc != CODEC_STATUS_SUCCESS) {
            return rc;
        }
    }
    st->codes_buf[(size_t) cb] = code;
    st->next_cb           += 1;
    st->logits_pending     = false;
    return CODEC_STATUS_SUCCESS;
}

enum codec_status codec_lm_step_finish(
    struct codec_lm_state * st,
    int32_t * out_codes) {
    if (st == nullptr || st->lm == nullptr || out_codes == nullptr) {
        return CODEC_STATUS_INVALID_ARG;
    }
    if (!st->step_in_progress) {
        st->last_error = "codec_lm_step_finish called without step_begin";
        return CODEC_STATUS_INVALID_STATE;
    }
    if (st->next_cb != st->lm->info.n_codebook) {
        st->last_error = "codec_lm_step_finish called before all codebooks were pushed";
        return CODEC_STATUS_INVALID_STATE;
    }
    if (st->logits_pending) {
        st->last_error = "codec_lm_step_finish called with a pending logits read";
        return CODEC_STATUS_INVALID_STATE;
    }
    if (st->lm->vtable != nullptr && st->lm->vtable->step_finish != nullptr) {
        enum codec_status rc = st->lm->vtable->step_finish(st, out_codes);
        if (rc != CODEC_STATUS_SUCCESS) {
            return rc;
        }
    } else {
        std::memcpy(out_codes, st->codes_buf.data(),
                    (size_t) st->lm->info.n_codebook * sizeof(int32_t));
    }
    st->step_in_progress = false;
    st->ar_frame        += 1;   // one more AR frame completed
    return CODEC_STATUS_SUCCESS;
}

// ---------------------------------------------------------------------
// End-of-audio decision
// ---------------------------------------------------------------------

enum codec_status codec_lm_step_is_eos(
    struct codec_lm_state * st,
    const int32_t * codes,
    int32_t n_codes,
    int32_t * out_is_eos) {
    if (st == nullptr || st->lm == nullptr || codes == nullptr ||
        out_is_eos == nullptr || n_codes <= 0) {
        if (st != nullptr) {
            st->last_error = "codec_lm_step_is_eos: null args or n_codes <= 0";
        }
        return CODEC_STATUS_INVALID_ARG;
    }
    *out_is_eos = 0;

    // Only codebook kinds carry a cb0 EOS sentinel.  Continuous-latent
    // kinds signal stop via codec_lm_step_generate's out_stop flag.
    if (st->lm->kind != CODEC_LM_KIND_RESIDUAL_DEPTH_AR &&
        st->lm->kind != CODEC_LM_KIND_PARALLEL_HEADS_DELAY) {
        st->last_error = "codec_lm_step_is_eos: kind has no cb0 EOS concept";
        return CODEC_STATUS_NOT_SUPPORTED;
    }

    const int32_t eos_c0 = st->lm->info.eos_code_c0;
    if (eos_c0 < 0) {
        // Model has no sentinel (e.g. Moshi) — never EOS.
        return CODEC_STATUS_SUCCESS;
    }

    // `ar_frame` was incremented by the preceding step_finish, so the
    // frame index of the just-emitted `codes` is `ar_frame - 1`.  Gate the
    // sentinel behind eos_min_step on that index.
    const int32_t this_frame = st->ar_frame - 1;
    if (this_frame >= st->lm->info.eos_min_step && codes[0] == eos_c0) {
        *out_is_eos = 1;
    }
    return CODEC_STATUS_SUCCESS;
}

// ---------------------------------------------------------------------
// continuous-latent step machine — delegate to vtable
// ---------------------------------------------------------------------

enum codec_status codec_lm_step_generate(
    struct codec_lm_state * st, const float * h_in, float cfg_value,
    int32_t n_timesteps, const float * noise, float * out_patch, int32_t * out_stop) {
    if (st == nullptr || st->lm == nullptr || h_in == nullptr || out_patch == nullptr) {
        return CODEC_STATUS_INVALID_ARG;
    }
    if (st->lm->vtable == nullptr || st->lm->vtable->step_generate == nullptr) {
        st->last_error = "codec_lm_step_generate not supported for this kind";
        return CODEC_STATUS_NOT_SUPPORTED;
    }
    return st->lm->vtable->step_generate(st, h_in, cfg_value, n_timesteps, noise, out_patch, out_stop);
}

enum codec_status codec_lm_step_feedback_embd(struct codec_lm_state * st, float * out_embd) {
    if (st == nullptr || st->lm == nullptr || out_embd == nullptr) {
        return CODEC_STATUS_INVALID_ARG;
    }
    if (st->lm->vtable == nullptr || st->lm->vtable->step_feedback_embd == nullptr) {
        st->last_error = "codec_lm_step_feedback_embd not supported for this kind";
        return CODEC_STATUS_NOT_SUPPORTED;
    }
    return st->lm->vtable->step_feedback_embd(st, out_embd);
}

enum codec_status codec_lm_text_prefill(
    struct codec_lm_state * st, const float * hiddens, int32_t n_pos, int32_t hidden_dim) {
    if (st == nullptr || st->lm == nullptr || hiddens == nullptr) {
        return CODEC_STATUS_INVALID_ARG;
    }
    if (n_pos <= 0 || hidden_dim <= 0) {
        st->last_error = "codec_lm_text_prefill: n_pos and hidden_dim must be > 0";
        return CODEC_STATUS_INVALID_ARG;
    }
    if (hidden_dim != st->lm->info.hidden_dim) {
        st->last_error = "codec_lm_text_prefill: hidden_dim mismatch";
        return CODEC_STATUS_INVALID_ARG;
    }
    if (st->lm->vtable == nullptr || st->lm->vtable->text_prefill == nullptr) {
        st->last_error = "codec_lm_text_prefill not supported for this kind";
        return CODEC_STATUS_NOT_SUPPORTED;
    }
    return st->lm->vtable->text_prefill(st, hiddens, n_pos, hidden_dim);
}

enum codec_status codec_lm_set_continuous_min_len(struct codec_lm_state * st, int32_t min_len) {
    if (st == nullptr || st->lm == nullptr) {
        return CODEC_STATUS_INVALID_ARG;
    }
    if (st->lm->vtable == nullptr || st->lm->vtable->set_min_len == nullptr) {
        st->last_error = "codec_lm_set_continuous_min_len not supported for this kind";
        return CODEC_STATUS_NOT_SUPPORTED;
    }
    return st->lm->vtable->set_min_len(st, min_len);
}

enum codec_status codec_lm_set_teacher_patch(struct codec_lm_state * st, const float * patch, int32_t n) {
    if (st == nullptr || st->lm == nullptr) {
        return CODEC_STATUS_INVALID_ARG;
    }
    if (st->lm->vtable == nullptr || st->lm->vtable->set_teacher_patch == nullptr) {
        st->last_error = "codec_lm_set_teacher_patch not supported for this kind";
        return CODEC_STATUS_NOT_SUPPORTED;
    }
    return st->lm->vtable->set_teacher_patch(st, patch, n);
}

// ---------------------------------------------------------------------
// speaker encoder — delegate to vtable
// ---------------------------------------------------------------------

const struct codec_lm_speaker_info * codec_lm_speaker_get_info(
        const struct codec_lm * lm) {
    if (lm == nullptr || !lm->has_speaker_encoder) {
        return nullptr;
    }
    return &lm->speaker_info;
}

enum codec_status codec_lm_speaker_encode(
        struct codec_lm *          lm,
        const struct codec_audio * ref_pcm,
        const int32_t *            ref_speech_tokens,
        int32_t                    n_ref_speech_tokens,
        const float *              emotion,
        float *                    out,
        int32_t                    out_n_elems) {
    if (lm == nullptr) {
        return CODEC_STATUS_INVALID_ARG;
    }
    if (!lm->has_speaker_encoder ||
        lm->speaker_arch == codec_lm::CODEC_SPEAKER_ARCH_NONE) {
        lm->last_error = "codec_lm_speaker_encode: model has no speaker section";
        return CODEC_STATUS_NOT_SUPPORTED;
    }
    if (out == nullptr) {
        lm->last_error = "codec_lm_speaker_encode: out is NULL";
        return CODEC_STATUS_INVALID_ARG;
    }
    const codec_lm_speaker_info & si = lm->speaker_info;
    const int32_t need_elems = si.n_rows * si.hidden_dim;
    if (out_n_elems < need_elems) {
        lm->last_error = "codec_lm_speaker_encode: out buffer too small";
        return CODEC_STATUS_INVALID_ARG;
    }

    if (si.needs_ref_pcm) {
        if (ref_pcm == nullptr || ref_pcm->data == nullptr || ref_pcm->n_samples <= 0) {
            lm->last_error = "codec_lm_speaker_encode: ref_pcm required but missing";
            return CODEC_STATUS_INVALID_ARG;
        }
    }
    if (si.needs_ref_speech_tokens) {
        if (ref_speech_tokens == nullptr || n_ref_speech_tokens <= 0) {
            lm->last_error = "codec_lm_speaker_encode: ref_speech_tokens required but missing";
            return CODEC_STATUS_INVALID_ARG;
        }
    }

    // NULL emotion → use the model's training default (see header for
    // semantics).  Models that don't consume emotion ignore the value
    // regardless.
    const float emotion_val = (emotion != nullptr) ? *emotion : si.emotion_default;

    switch (lm->speaker_arch) {
        case codec_lm::CODEC_SPEAKER_ARCH_CHATTERBOX_VOICE_ENC:
            return chatterbox_speaker_encode(
                lm, ref_pcm,
                ref_speech_tokens, n_ref_speech_tokens,
                emotion_val,
                out, out_n_elems);
        case codec_lm::CODEC_SPEAKER_ARCH_QWEN3_TTS_ECAPA_TDNN:
            return qwen3_tts_speaker_encode(
                lm, ref_pcm,
                ref_speech_tokens, n_ref_speech_tokens,
                emotion_val,
                out, out_n_elems);
        case codec_lm::CODEC_SPEAKER_ARCH_NONE:
            break;
    }
    lm->last_error = "codec_lm_speaker_encode: no impl for speaker_arch";
    return CODEC_STATUS_NOT_SUPPORTED;
}

enum codec_status codec_lm_speaker_encode_from_embedding(
        struct codec_lm *          lm,
        const float *              speaker_emb,
        int32_t                    speaker_emb_dim,
        const int32_t *            ref_speech_tokens,
        int32_t                    n_ref_speech_tokens,
        const float *              emotion,
        float *                    out,
        int32_t                    out_n_elems) {
    if (lm == nullptr) {
        return CODEC_STATUS_INVALID_ARG;
    }
    if (!lm->has_speaker_encoder ||
        lm->speaker_arch == codec_lm::CODEC_SPEAKER_ARCH_NONE) {
        lm->last_error = "codec_lm_speaker_encode_from_embedding: no speaker section";
        return CODEC_STATUS_NOT_SUPPORTED;
    }
    if (lm->speaker_info.speaker_emb_dim <= 0) {
        lm->last_error =
            "codec_lm_speaker_encode_from_embedding: codec doesn't expose an "
            "intermediate speaker embedding (info->speaker_emb_dim == 0)";
        return CODEC_STATUS_NOT_SUPPORTED;
    }
    if (out == nullptr || speaker_emb == nullptr) {
        lm->last_error = "codec_lm_speaker_encode_from_embedding: NULL argument";
        return CODEC_STATUS_INVALID_ARG;
    }
    if (speaker_emb_dim != lm->speaker_info.speaker_emb_dim) {
        lm->last_error =
            "codec_lm_speaker_encode_from_embedding: speaker_emb_dim mismatch";
        return CODEC_STATUS_INVALID_ARG;
    }
    const codec_lm_speaker_info & si = lm->speaker_info;
    if (out_n_elems < si.n_rows * si.hidden_dim) {
        lm->last_error =
            "codec_lm_speaker_encode_from_embedding: out buffer too small";
        return CODEC_STATUS_INVALID_ARG;
    }
    if (si.needs_ref_speech_tokens &&
        (ref_speech_tokens == nullptr || n_ref_speech_tokens <= 0)) {
        lm->last_error =
            "codec_lm_speaker_encode_from_embedding: ref_speech_tokens required";
        return CODEC_STATUS_INVALID_ARG;
    }
    const float emotion_val = (emotion != nullptr) ? *emotion : si.emotion_default;

    switch (lm->speaker_arch) {
        case codec_lm::CODEC_SPEAKER_ARCH_CHATTERBOX_VOICE_ENC:
            return chatterbox_speaker_encode_from_emb(
                lm, speaker_emb,
                ref_speech_tokens, n_ref_speech_tokens,
                emotion_val,
                out, out_n_elems);
        case codec_lm::CODEC_SPEAKER_ARCH_QWEN3_TTS_ECAPA_TDNN:
            // Qwen3-TTS's x-vector IS the (1, hidden_dim) cond_emb (no
            // perceiver / cond_enc projection downstream).  Skipping
            // ECAPA-TDNN is just `out[:hidden] = speaker_emb[:hidden]`.
            if (speaker_emb_dim != lm->speaker_info.hidden_dim) {
                lm->last_error =
                    "codec_lm_speaker_encode_from_embedding(qwen3_tts): "
                    "speaker_emb_dim must equal info->hidden_dim (1024)";
                return CODEC_STATUS_INVALID_ARG;
            }
            std::memcpy(out, speaker_emb,
                        (size_t) lm->speaker_info.hidden_dim * sizeof(float));
            return CODEC_STATUS_SUCCESS;
        case codec_lm::CODEC_SPEAKER_ARCH_NONE:
            break;
    }
    lm->last_error =
        "codec_lm_speaker_encode_from_embedding: no impl for speaker_arch";
    return CODEC_STATUS_NOT_SUPPORTED;
}
