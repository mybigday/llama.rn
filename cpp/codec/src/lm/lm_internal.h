#ifndef CODEC_LM_INTERNAL_H
#define CODEC_LM_INTERNAL_H

#include "codec_lm.h"

#include "../codec_internal.h"
#include "../runtime/graph.h"

#include <cstdint>
#include <string>
#include <vector>

struct codec_lm_kind_vtable;

// Free the lazily-allocated Chatterbox T3 host-orchestration state
// (tokenizer + dequanted embed tables) registered by the
// `codec_lm_chatterbox_*` helpers.  No-op when none was created.
// Implemented in chatterbox_t3.cpp; called from codec_lm_free.
void codec_lm_chatterbox_free_state(struct codec_lm * lm);

struct codec_lm {
    // Borrowed; not freed by codec_lm_free.  All weights and the
    // lm_ggml_backend live on this codec_model.
    codec_model * codec = nullptr;

    enum codec_lm_kind kind = CODEC_LM_KIND_UNKNOWN;
    const codec_lm_kind_vtable * vtable = nullptr;
    void * impl = nullptr;     // per-kind opaque data

    // Backing storage for codec_lm_info pointer fields.
    std::vector<int32_t> codebook_sizes_buf;
    std::vector<int32_t> delay_pattern_buf;
    std::string          host_arch_buf;

    codec_lm_info info = {};

    // Speaker-conditioning encoder info, populated from `codec.speaker.*`
    // GGUF metadata at create time.  `has_speaker_encoder = false` when
    // the section is absent (most codecs); `codec_lm_speaker_get_info`
    // returns NULL in that case.  When true, the loader reads
    // `codec.speaker.encoder_arch` to set `speaker_arch` (enum below)
    // and the public `codec_lm_speaker_encode` dispatches on it.
    // `speaker_impl` is per-arch opaque state (e.g. dequanted LSTM
    // weights, cached graph context).
    bool                       has_speaker_encoder = false;
    codec_lm_speaker_info      speaker_info        = {};
    enum codec_speaker_arch {
        CODEC_SPEAKER_ARCH_NONE                  = 0,
        CODEC_SPEAKER_ARCH_CHATTERBOX_VOICE_ENC  = 1,
        CODEC_SPEAKER_ARCH_QWEN3_TTS_ECAPA_TDNN  = 2, // runtime stub — see docs/audio_speaker_encoders.md
        // Add more here: CODEC_SPEAKER_ARCH_MOSS_TTSD_XVECTOR, …
    };
    codec_speaker_arch speaker_arch = CODEC_SPEAKER_ARCH_NONE;
    void *             speaker_impl = nullptr;

    std::string last_error;
};

struct codec_lm_state {
    codec_lm * lm = nullptr;
    void * impl = nullptr;     // per-kind opaque data

    // Each state owns its own codec_context for graph cache + scheduler.
    // The context borrows codec->backend / codec->weights, so weights are
    // shared across all states.  Independent eval_ctx + scheduler means
    // multiple states can run concurrently without interfering.
    codec_context * ctx = nullptr;

    // Per-step bookkeeping (kind-agnostic).  Codes pushed via
    // codec_lm_step_push_code accumulate here; codec_lm_step_finish
    // copies them out.  `next_cb` is the index of the codebook whose
    // logits will be returned by the next codec_lm_step_logits call,
    // and it advances on each push_code.  step_in_progress goes true
    // on step_begin and false on step_finish.
    std::vector<int32_t> codes_buf;
    int32_t next_cb        = 0;
    bool    step_in_progress = false;

    // Kind-agnostic AR frame counter for the end-of-audio decision.
    // Incremented once per successful codec_lm_step_finish (0-based frame
    // index of the NEXT frame to be produced; equivalently, the count of
    // frames already finished).  Reset to 0 by codec_lm_state_reset.  Used
    // by codec_lm_step_is_eos to gate the cb0 sentinel behind eos_min_step.
    int32_t ar_frame       = 0;
    // logits_pending == true after step_logits; cleared by push_code.
    // Used to enforce the alternating logits/push_code order.
    bool    logits_pending = false;

    // Optional pre-step context for `c0_input_modality="text"` models
    // (Moshi).  Caller stashes the text token sampled from the backbone
    // here before calling step_begin; the residual_depth_ar runtime
    // reads it at depth position 0.  Sentinel `-1` = unset.
    int32_t text_token_context = -1;

    std::string last_error;
};

// One vtable per kind.  Functions may be NULL when not applicable to a
// kind (e.g. kinds that do not need state_reset).
struct codec_lm_kind_vtable {
    enum codec_lm_kind kind;
    const char * name;

    // Lifecycle.  init returns false on failure (missing tensors, bad
    // metadata); the create path then frees the codec_lm and returns
    // NULL to the user.
    bool (*init)(codec_lm * lm);
    void (*free)(codec_lm * lm);

    bool (*state_init)(codec_lm_state * st);
    void (*state_free)(codec_lm_state * st);
    void (*state_reset)(codec_lm_state * st);

    // Step machine.  Per-kind responsibilities documented in codec_lm.h.
    enum codec_status (*step_begin)(codec_lm_state * st, const float * h_in);
    bool              (*step_pending)(const codec_lm_state * st);
    const float *     (*step_logits)(codec_lm_state * st, int32_t * out_cb_idx, int32_t * out_n);
    enum codec_status (*step_push_code)(codec_lm_state * st, int32_t code);
    enum codec_status (*step_finish)(codec_lm_state * st, int32_t * out_codes);

    // Audio embd (kind-specific because the table layout — fused vs
    // unfused — varies, even though the v1 schema standardises on
    // unfused per-cb tables).
    const float *     (*audio_embd)(codec_lm * lm, int32_t cb_idx, int32_t code);
    enum codec_status (*compose_audio_embd)(codec_lm * lm, const int32_t * codes, float * out_embd);

    // Like compose_audio_embd but adds a learned per-step positional
    // embedding when the kind has one (Chatterbox T3's `speech_pos_emb`).
    // Leave NULL for kinds without per-step pos emb — the public function
    // then falls back to compose_audio_embd, ignoring `step`.
    enum codec_status (*compose_next_embd)(codec_lm * lm, const int32_t * codes, int32_t step, float * out_embd);

    // Speaker-conditioning encoder.  Optional per arch — leave NULL when
    // the model has no speaker section (most kinds: parallel_heads_delay
    // for plain TTS, residual_depth_ar for CSM, etc.).  When non-NULL,
    // the runtime is responsible for honouring `lm->speaker_info` — it
    // must produce `n_rows * hidden_dim` F32 values in `out`.
    enum codec_status (*speaker_encode)(
        codec_lm * lm,
        const struct codec_audio * ref_pcm,
        const int32_t * ref_speech_tokens, int32_t n_ref_speech_tokens,
        float emotion,                  // pre-defaulted by lm.cpp
        float * out, int32_t out_n_elems);

    // Continuous-latent step machine (CONTINUOUS_LATENT_CFM).  NULL for
    // codebook kinds.  Placed last so existing kinds' positional vtable
    // initializers leave these zero (nullptr).
    enum codec_status (*step_generate)(codec_lm_state * st, const float * h_in,
        float cfg_value, int32_t n_timesteps, const float * noise,
        float * out_patch, int32_t * out_stop);
    enum codec_status (*step_feedback_embd)(codec_lm_state * st, float * out_embd);

    // Prefill the RALM over the whole prompt prefix (continuous_latent_cfm).
    // NULL for other kinds → codec_lm_text_prefill returns NOT_SUPPORTED.
    enum codec_status (*text_prefill)(codec_lm_state * st, const float * hiddens,
        int32_t n_pos, int32_t hidden_dim);
    // Configure the min_len stop guard (continuous_latent_cfm).  NULL for other
    // kinds → codec_lm_set_continuous_min_len returns NOT_SUPPORTED.
    enum codec_status (*set_min_len)(codec_lm_state * st, int32_t min_len);
    // Teacher-force the next step's trajectory (continuous_latent_cfm, parity
    // tests): the reference patch replaces the codec's own patch as the cond +
    // LocEnc feedback source.  NULL for other kinds.
    enum codec_status (*set_teacher_patch)(codec_lm_state * st, const float * patch, int32_t n);
};

// Map between the GGUF string and the C enum.  Returns
// CODEC_LM_KIND_UNKNOWN for unrecognised strings.
enum codec_lm_kind          codec_lm_kind_from_string(const char * s);
const codec_lm_kind_vtable * codec_lm_vtable_for_kind(enum codec_lm_kind kind);

// Per-kind vtables, exposed to lm.cpp's dispatch table.
extern const codec_lm_kind_vtable codec_lm_vtable_parallel_heads_delay;
extern const codec_lm_kind_vtable codec_lm_vtable_residual_depth_ar;
extern const codec_lm_kind_vtable codec_lm_vtable_continuous_latent_cfm;
extern const codec_lm_kind_vtable codec_lm_vtable_flow_lm;

// Read a string KV from the codec_model's GGUF.  Returns empty string
// if the key is absent.
std::string codec_lm_read_string_kv(const codec_model * codec, const char * key);

// Validate that all `lm.audio_embd_{i}.weight` (and optionally
// `lm.heads_{i}.weight`) tensors exist with the expected shapes.  Used
// by parallel_heads_delay at init time.  Any missing or wrong-shape
// tensor sets `lm->last_error` and returns false.
//
// When `tied_heads` is true, `lm.heads_{i}.weight` is not required (the
// runtime will reuse `lm.audio_embd_{i}.weight` as the head weight —
// matches the `tie_word_embeddings`-style convention upstream).
bool codec_lm_check_unfused_audio_tables(
    codec_lm * lm,
    int32_t hidden_dim,
    const std::vector<int32_t> & codebook_sizes,
    bool tied_heads);

#endif // CODEC_LM_INTERNAL_H
