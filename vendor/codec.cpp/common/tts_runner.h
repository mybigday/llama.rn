#ifndef CODEC_TTS_RUNNER_H
#define CODEC_TTS_RUNNER_H

// codec_common::tts_runner — the reference TTS host loop.
//
// It drives the public codec_audio_lm API (include/codec_audio_lm.h) with
// llama.cpp as the backbone, the same loop a host such as llama.rn writes
// against the API: load the codec and the backbone, split the text, set the
// voice, then per segment build the prompt, sample / decode / observe each
// step and decode (or stream) the audio.  `tools/tts-cli` (one-shot
// tts_runner_synthesize) and `tools/tts-server` (a tts_runner_session) are
// thin drivers over it.
//
// The session API is built only when CODEC_TTS_BACKBONE=ON (codec_tts_runner
// links libttsbackbone).  Without it, tts-cli still runs the self-contained
// models (Pocket-TTS FlowLM) through tts_runner_synthesize_selfcontained.

#include "codec.h"
#include "codec_audio_lm.h"

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace codec_common {

// ─────────────────────────────────────────────────────────────────────
// Params — everything the reference loop needs.  Sampling is llama.cpp's
// (its defaults; the model card recommends values); decoder graph settings
// left < 0 come from the GGUF metadata.
// ─────────────────────────────────────────────────────────────────────
// Sampling settings the user set (tts_runner_params::sampling_user_set): the
// backbone GGUF's general.sampling.* fills the others, as llama.cpp does.
enum tts_sampling_field : uint32_t {
    TTS_SAMPLING_TEMP               = 1u << 0,
    TTS_SAMPLING_TOP_K              = 1u << 1,
    TTS_SAMPLING_TOP_P              = 1u << 2,
    TTS_SAMPLING_MIN_P              = 1u << 3,
    TTS_SAMPLING_PENALTY_LAST_N     = 1u << 4,
    TTS_SAMPLING_PENALTY_REPEAT     = 1u << 5,
    TTS_SAMPLING_PENALTY_FREQ       = 1u << 6,
    TTS_SAMPLING_PENALTY_PRESENT    = 1u << 7,
    TTS_SAMPLING_DRY_MULTIPLIER     = 1u << 8,
    TTS_SAMPLING_DRY_BASE           = 1u << 9,
    TTS_SAMPLING_DRY_ALLOWED_LENGTH = 1u << 10,
    TTS_SAMPLING_DRY_PENALTY_LAST_N = 1u << 11,
    TTS_SAMPLING_SAMPLERS           = 1u << 12,
};

// One llama.cpp sampling setting: its CLI flag (common/arg.cpp), its
// llama-server request field, the bit, and the field (float or int).
struct tts_sampling_option {
    const char * flag;
    const char * json;
    uint32_t     bit;
    float   codec_audio_lm_sampler::* f32;
    int32_t codec_audio_lm_sampler::* i32;
};
const std::vector<tts_sampling_option> & tts_sampling_options();

struct tts_runner_params {
    std::string codec_path;          // codec / codec_lm GGUF (required)
    std::vector<codec_model_kv_override> kv_overrides;  // codec GGUF metadata overrides
    std::string backbone_path;       // llama.cpp backbone GGUF; empty for
                                     // self-contained models (Pocket FlowLM)
    std::string text;                // synthesis target (required)
    std::string ref_audio_path;      // optional WAV for voice conditioning
    std::string ref_text;            // optional transcript of ref_audio (NeuTTS phonemization)
    std::string voice;               // builtin codec speaker name (codec_get_speaker)
    std::string speaker_file;        // speaker GGUF from codec_speaker_save
    std::string save_speaker;        // write the voice in use to this path
    const codec_speaker * voice_spk = nullptr;   // the resolved voice (tts_runner_make_voice), borrowed
    std::string instruct;            // CosyVoice instruct text (instruct2 mode)

    int32_t  n_threads = 0;
    bool     use_gpu   = false;
    int32_t  max_frames = 0;         // 0 → per-model default
    int32_t  min_frames = -1;        // -1 → the model's own minimum

    // llama.cpp sampling (common_params_sampling defaults); its pointers are
    // unused, `samplers` / `penalty_exempt` below hold them.  Segment i of a
    // long text uses seed + i (a random seed stays random).
    codec_audio_lm_sampler sampling = codec_audio_lm_default_tts_request().sampler;
    std::string            samplers;
    std::vector<int32_t>   penalty_exempt;
    uint32_t               sampling_user_set = 0;   // tts_sampling_field bits

    // Decoder graph settings; < 0 → the GGUF's.
    float   lm_guidance   = -1.0f;   // guidance of the backbone pair (Chatterbox T3)
    float   flow_guidance = -1.0f;   // guidance of a flow / CFM decoder
    int32_t flow_steps    = -1;      // steps of a flow / CFM decoder

    // Grammar (GBNF) constraint for the BACKBONE-logits sampler.  When
    // non-empty this GBNF is attached to the common_sampler that samples
    // backbone tokens (cb0-from-backbone / text warmup).  It never applies
    // to codec_lm audio-codebook heads (those are arbitrary float arrays
    // not tied to the backbone vocab).  A parse failure surfaces as a clean
    // error, not a crash.  Empty → the model's auto-grammar (if any) is used
    // instead; see codec_common::tts_auto_grammar.
    std::string grammar;

    // Long input runs a segment at a time: sentences packed up to this many
    // UTF-8 bytes per segment, blank lines always split (tts_split_text).
    // 0 = the whole text in one run.
    int32_t segment_bytes = 0;

    // Streaming.  When set, every piece of audio goes to on_audio (model
    // sample rate / channels, interleaved) as soon as it is ready and
    // result.pcm stays empty; return false to stop the run.  Pieces are whole
    // segments, or with stream_chunk_ms > 0 chunks of about that much audio
    // decoded while the AR loop runs — for models whose codes accumulate in
    // audio_lm (the others still stream per segment).
    std::function<bool(const float * pcm, int32_t n_samples, int32_t sample_rate, int32_t n_channels)> on_audio;
    int32_t stream_chunk_ms = 0;
};

// ─────────────────────────────────────────────────────────────────────
// Result — PCM + stats.  The caller writes the WAV / marshals output.
// ─────────────────────────────────────────────────────────────────────
struct tts_runner_result {
    std::vector<float> pcm;          // interleaved if n_channels > 1
    int32_t sample_rate = 0;
    int32_t n_channels  = 1;

    int32_t     n_frames = 0;        // audio frames generated
    std::string stop_reason;         // "eos_code_c0" / "stop_head" / …

    double load_ms    = 0.0;         // codec + backbone load, up to the first AR step (incl. speaker_ms)
    double speaker_ms = 0.0;         // decoder speaker from --ref-audio / --speaker-file / --voice
    double decode_ms = 0.0;          // codes -> PCM (audio_lm_decode_audio)
    double total_ms  = 0.0;

    std::string error;               // set + returns false on failure
};

// Self-contained synthesize (models that need no backbone, Pocket-TTS
// FlowLM).  Lives in the always-built codec_common library.  Returns 0 when
// the codec GGUF needs a backbone; 1 when it handled the request (out->error
// set on failure, including a codec that cannot be opened).
int tts_runner_synthesize_selfcontained(const tts_runner_params & params,
                                        tts_runner_result * out);

// Pieces of the runner that need no backbone (tts_runner_base.cpp).
// Interleaved multi-channel PCM -> mono (channel average); n_frames samples per channel.
std::vector<float> downmix_mono(const float * in, size_t n_frames, int32_t n_channels);
// Mono F32 from a WAV file (channels averaged); true with an empty `ref_pcm`
// when `path` is empty.
bool load_ref_audio(const std::string & path, std::vector<float> & ref_pcm, int32_t * out_n, int32_t * out_sr,
                    std::string * err);
codec_audio_lm * tts_runner_open_codec(const tts_runner_params & params, std::string * err);
// The request's codec_audio_lm_tts_request (pointers into `params`).
codec_audio_lm_tts_request tts_runner_request(const tts_runner_params & params);
// The request's voice: speaker_file > ref_audio (+ ref_text, when the model
// can clone) > the builtin `voice` > `voice_spk` > the model default;
// save_speaker writes it.
struct tts_runner_voice {
    codec_speaker *       owned = nullptr;
    const codec_speaker * use   = nullptr;   // NULL: the model default
    tts_runner_voice() = default;
    tts_runner_voice(const tts_runner_voice &) = delete;
    tts_runner_voice & operator=(const tts_runner_voice &) = delete;
    ~tts_runner_voice() { if (owned) codec_speaker_free(owned); }
};
bool tts_runner_make_voice(codec_audio_lm * alm, const tts_runner_params & params, tts_runner_voice * out,
                           double * ms, std::string * err);
// Moves `buf` into out->pcm (frees it).
bool tts_runner_take_pcm(codec_pcm_buffer * buf, tts_runner_result * out);
bool tts_runner_generate_self_contained(codec_audio_lm * alm, const tts_runner_params & params,
                                        tts_runner_result * out);

// ─────────────────────────────────────────────────────────────────────
// Session — load once, synthesize many (a server).  tts_runner_load reads
// the load-time fields of `params` (codec_path, kv_overrides,
// backbone_path, n_threads, use_gpu, max_frames); tts_runner_run reads the per-request ones (text,
// voice / speaker_file / ref_audio, sampler knobs, ...); max_frames stays
// the load-time one.  Runs on one session are not thread-safe: callers
// serialize them.  Only compiled into codec_tts_runner.
// ─────────────────────────────────────────────────────────────────────
struct tts_runner_session;

struct tts_runner_info {
    std::vector<std::string> voices;      // the speakers `voice` can name
    std::string              default_voice;
    bool                     takes_instruct = false;   // CosyVoice instruct mode
    std::string              error;       // speaker file could not be listed
};

tts_runner_session * tts_runner_load(const tts_runner_params & params, std::string * err);
bool                 tts_runner_run (tts_runner_session * s, const tts_runner_params & params,
                                     tts_runner_result * out);
void                 tts_runner_free(tts_runner_session * s);
// The session's codec_audio_lm, e.g. to create / load voices (serialize with the runs).
codec_audio_lm *     tts_runner_audio_lm(tts_runner_session * s);
// Voices are the codec GGUF's builtin speakers, or those in `speaker_file`
// when the runs pass one.
tts_runner_info      tts_runner_get_info(tts_runner_session * s, const std::string & speaker_file = "");

// Load, run and free in one go: `params.text` -> PCM.  Returns true on
// success (result.pcm populated); false with result.error set on failure.
// Progress goes through codec_log (codec_log_set to redirect it).  Only
// compiled into the codec_tts_runner library.
bool tts_runner_synthesize(const tts_runner_params & params,
                           tts_runner_result * out);

}  // namespace codec_common

#endif  // CODEC_TTS_RUNNER_H
