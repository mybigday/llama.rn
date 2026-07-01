#ifndef RNTTS_H
#define RNTTS_H

#include <functional>
#include <vector>
#include <string>
#include "llama.h"
#include "nlohmann/json.hpp"

using json = nlohmann::ordered_json;

struct codec_model;
struct codec_context;
struct codec_lm;
struct codec_lm_state;
struct codec_lm_info;

namespace rnllama {

// Forward declarations
struct llama_rn_context;

// TTS type enumeration
enum tts_type {
    UNKNOWN = -1,
    OUTETTS_V0_1 = 0,
    OUTETTS_V0_2 = 1,
    OUTETTS_V0_3 = 2,
    OUTETTS_V1_0 = 3,
    SOPRANO_1_1_80M = 4,
    NEUTTS_NANO = 5,
    NEUTTS_AIR = 6,
    CSM_1B = 7,
    QWEN3_TTS_0_6B = 8,
    MOSS_TTS_REALTIME = 9,
    MOSS_TTSD_V07 = 10,
    CHATTERBOX_T3 = 11,             // English Chatterbox (t3_cfg)
    CHATTERBOX_T3_MULTILINGUAL = 12, // 23-language Chatterbox (t3_mtl23ls_v3)
};

// Audio completion result structure.
// `flow` tells JS which downstream path to take:
//   "tokens"      — standard completion → tryAddAudioToken → decodeAudioTokens
//   "codec_lm_ar" — generateAudioCodes (codec_lm AR loop) → decodeAudioTokens
struct llama_rn_audio_completion_result {
    std::string prompt;
    std::string grammar;
    bool embedding;
    std::string flow;
};

// Options + per-frame progress hook for the codec_lm AR driver.
struct llama_rn_audio_codes_options {
    std::string prompt;
    int  max_frames = 500;
    float temperature = 0.9f;
    float top_p = 0.95f;
    int   top_k = 50;
    uint32_t seed = 0;

    // Optional speaker-conditioning prefix.  When non-empty, the AR loop
    // feeds these `n_rows × hidden_dim` rows via `b.embd` before the
    // tokenized prompt — gives codec_lm-AR models (Chatterbox, Qwen3-TTS,
    // MOSS-TTSD voice-clone) somewhere to absorb the output of
    // `codec_lm_speaker_encode`.  Caller supplies the matrix via
    // `encodeSpeaker(...).speakerEmb` after running the speaker encoder
    // on the reference audio.  Length must be `speaker_emb_rows *
    // speaker_emb_hidden_dim`.
    std::vector<float> speaker_emb_prefix;
    int speaker_emb_rows = 0;
    int speaker_emb_hidden_dim = 0;
};

// Progress callback fires after each AR step with the just-sampled codes
// for that frame (length = n_codebook).  Return false to abort.
using llama_rn_audio_codes_progress_cb =
    std::function<bool(int step, const std::vector<int32_t> &codes)>;

struct llama_rn_audio_codes_result {
    std::vector<int32_t> codes;   // (n_frames * n_codebook) interleaved
    int n_codebook = 0;
    int n_frames = 0;
    bool stopped_on_eos = false;
    bool aborted = false;

    // Continuous-latent path (BlueMagpie-TTS / VoxCPM): the AR loop drives
    // codec_common::audio_lm_observe_hidden per backbone step, accumulates
    // latent patches internally, then audio_lm_decode_audio produces PCM
    // directly — bypassing the codec_token path entirely.  When the loaded
    // codec_lm is continuous-latent kind, `codes` stays empty and
    // `pcm` + `sample_rate` carry the decoded audio.  JS callers can just
    // play `pcm` at `sample_rate`; there's no `decodeAudioTokens` step.
    std::vector<float> pcm;
    int sample_rate = 0;
    bool is_continuous = false;
};

// On-device speaker encoding result.  Caller drops this into a speaker JSON
// (alongside any model-specific spk_emb / language tags) for voice-clone
// models (Qwen3-TTS / MOSS-TTSD).  The (T × n_q) interleaved layout matches
// what `decodeAudioTokens` and the codec_lm AR loop expect for ref_codes.
struct llama_rn_speaker_artifact {
    std::vector<int32_t> ref_codes;   // (n_frames * n_q) interleaved
    int n_q = 0;
    int n_frames = 0;
    int sample_rate = 0;
    int codebook_size = 0;
    std::string ref_text;             // pass-through from caller

    // Speaker-conditioning embedding produced by codec.cpp's
    // `codec_lm_speaker_encode`.  Present when the loaded codec.gguf has a
    // speaker section (Chatterbox / Qwen3-TTS / MOSS-TTSD voice-clone).
    // How the LM consumes this matrix is the LM arch's decision — codec.cpp
    // does not bake "prefix" / "vector" / etc. semantics into the output.
    std::vector<float> speaker_emb;   // (n_rows * hidden_dim) f32, row-major
    int speaker_n_rows = 0;           // Chatterbox: 34   Qwen3-TTS: 1
    int speaker_hidden_dim = 0;       // LM hidden size
};

// Options for encodeSpeaker.  Required inputs are picked up from the
// loaded codec's `codec_lm_speaker_info`; whatever the caller doesn't
// provide is left to defaults (e.g. emotion → emotion_default).
struct llama_rn_encode_speaker_options {
    std::vector<float> pcm;               // F32 mono samples
    int input_sample_rate = 0;
    std::string ref_text;                 // pass-through, not consumed by codec
    bool has_emotion = false;
    float emotion = 0.5f;                 // [0, 1]; only used when has_emotion
};

// Single source of truth for everything the JS layer needs to drive a TTS
// session — populated from the native profile so JS doesn't keep its own
// parallel mapping. Voice resolution lives entirely on the JS side: the
// wrapper looks up the name `default` in its per-family/per-language voice
// table, so different small-model variants (e.g. neutts-nano-german vs
// neutts-nano-spanish) can ship different reference speakers without the
// native side having to track them.
struct llama_rn_tts_capabilities {
    int type;                       // matches tts_type enum
    std::string prompt_kind;        // "outetts_legacy" | "outetts_v0_3" | "outetts_v1_0" | "soprano" | "neutts" | ""
    std::string family;             // "outetts" | "soprano" | "neutts" | ""
    bool requires_phonemes;
    std::string default_language;   // language hint for phonemizer hook ("en-us" today)
};

// (T × n_q) audio code range — start token id + count + per-codebook size.
// Mirrors the fields in `tts_model_profile::audio_token_config::code_range`
// but lives in the public header so callers / JSI can forward-declare it.
struct llama_rn_audio_code_range {
    int32_t start;
    int     count;
    int     codebook_size;
};

// TTS context for TTS-specific functionality
struct llama_rn_context_tts {
    // TTS state fields
    std::vector<llama_token> audio_tokens;
    int pending_codebook1 = -1;

    // Codec runtime handles
    ::codec_model *codec_model = nullptr;
    ::codec_context *codec_ctx = nullptr;
    // codec_lm adaptor (created lazily on first codec_lm-AR call, freed by
    // dtor).  Stays NULL when the loaded codec.gguf has no `lm.*` section,
    // in which case the model is treated as a plain codec.
    ::codec_lm *codec_lm = nullptr;
    ::codec_lm_state *codec_lm_state = nullptr;
    bool codec_lm_probed = false;
    tts_type type = UNKNOWN;

    // Vocab-probed audio token ranges.  Resolved lazily on first
    // tryAddAudioToken / isAudioToken / etc. — replaces the hardcoded
    // `tts_model_profile::audio.code_ranges` values which only matched
    // the original Llama-based releases.  OuteTTS V0.x is now Qwen2-based
    // (audio range starts at 151672, not 50307); OuteTTS V1.0 0.6B is
    // Qwen3-based (`<|c1_0|>=151669`, `<|c2_0|>=152694`, not 128256/129281).
    // Probing the actual vocab keeps the profile robust to backbone swaps.
    std::vector<llama_rn_audio_code_range> resolved_code_ranges;
    bool resolved_ranges_ready = false;

    // Constructor and destructor
    // `use_gpu` mirrors codec.cpp's `codec_model_params.use_gpu` — set true
    // to offload codec + codec_lm graphs (Mimi / S3G / depth decoder etc.)
    // to whatever backend codec.cpp's `ggml_backend_init_best` picks.
    // Defaults match the loaded backbone's GPU offload state where
    // possible; the caller (JS) can override.
    llama_rn_context_tts(const std::string &vocoder_model_path, int batch_size = -1, bool use_gpu = false);
    ~llama_rn_context_tts();

    // TTS utility methods
    void reset();
    tts_type getTTSType(llama_rn_context* main_ctx, json speaker = nullptr);
    // Detect-only variant for the JS layer to read the model's TTS family
    // without needing a speaker JSON or any text input.
    tts_type detectTTSType(llama_rn_context* main_ctx);
    // Full capability snapshot — single source of truth for JS-side wrappers.
    llama_rn_tts_capabilities getTTSCapabilities(llama_rn_context* main_ctx);
    llama_rn_audio_completion_result getFormattedAudioCompletion(llama_rn_context* main_ctx, const std::string &speaker_json_str, const std::string &text_to_speak);
    // codec_lm AR driver — drives the backbone + codec_lm step-state-machine
    // end-to-end, writes (T*n_q) interleaved codes into audio_tokens AND
    // returns them in the result for streaming convenience.  Returns
    // result.codes.empty() on failure (check logs).
    llama_rn_audio_codes_result generateAudioCodes(llama_rn_context* main_ctx, const llama_rn_audio_codes_options &opts, const llama_rn_audio_codes_progress_cb &on_frame = nullptr);
    // Continuous-latent variant (BlueMagpie-TTS / VoxCPM path).
    // Dispatched to internally by generateAudioCodes when
    // `codec_lm_get_info(codec_lm)->is_continuous` is true.  Public here
    // only so the JSI wrapper can invoke it directly if needed; JS
    // callers should stick to `generateAudioCodes` — it handles both.
    llama_rn_audio_codes_result generateAudioCodesContinuous(llama_rn_context* main_ctx, const llama_rn_audio_codes_options &opts, const llama_rn_audio_codes_progress_cb &on_frame, const ::codec_lm_info * info);
    // Encode a reference audio clip into codec tokens via the loaded
    // codec.gguf's encoder.  Used by voice-clone TTS models to register
    // a custom speaker without a Python pre-bake step.  `pcm` is F32 mono
    // at `input_sample_rate`; the codec will resample / re-channel as
    // needed (or fail with an empty result if the codec can't).
    // `ref_text` is passed through verbatim into the returned struct so
    // callers can persist a single bundle.
    llama_rn_speaker_artifact encodeSpeaker(llama_rn_context* main_ctx, const llama_rn_encode_speaker_options &opts);
    // Backwards-compat overload that just forwards into the options-struct
    // form.  Existing callers don't need to know about speaker-encoder
    // optional inputs.
    llama_rn_speaker_artifact encodeSpeaker(llama_rn_context* main_ctx, const std::vector<float> &pcm, int input_sample_rate, const std::string &ref_text);
    std::vector<float> decodeAudioTokens(llama_rn_context* main_ctx, const std::vector<llama_token> &tokens);
    std::vector<float> decodeAudioEmbeddings(llama_rn_context* main_ctx, const std::vector<float> &embeddings, int embedding_dim);
    int getAudioSampleRate() const;
    bool isAudioToken(llama_rn_context* main_ctx, llama_token token, const std::string &token_text = "");
    bool tryAddAudioToken(llama_rn_context* main_ctx, llama_token token, const std::string &token_text = "");
    bool shouldCaptureAudioEmbeddings(llama_rn_context* main_ctx);
};

}

#endif /* RNTTS_H */
