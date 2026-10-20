#ifndef CODEC_TTS_H
#define CODEC_TTS_H

#include "codec.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// =====================================================================
// codec_tts — text to speech in one call: the reference host loop of
// tts-cli / tts-server (common/tts_runner.h) behind a C ABI.  A session
// loads a codec GGUF and (unless the model is self-contained) a llama.cpp
// backbone GGUF once, then synthesizes many requests.  For a loop around
// your own llama.cpp, use codec_audio_lm.h instead.  The backbone runs in
// the llama.cpp bundled with codec.cpp (with its architecture / tokenizer
// patches), so this lives in its own library, libcodec_tts, next to
// libcodec.  Runs on one session are not thread-safe; serialize them.
// =====================================================================

struct codec_tts;

struct codec_tts_params {
    const char * codec_path;                               // codec GGUF (required)
    const char * backbone_path;                            // llama.cpp GGUF; NULL for self-contained models
    const struct codec_model_kv_override * kv_overrides;   // codec metadata overrides (may be NULL)
    int32_t      n_kv_overrides;
    int32_t      n_threads;                                // 0 = default
    bool         use_gpu;
    int32_t      max_frames;                               // longest generation; sizes the backbone
                                                           // context (0 = model default)
};

// Streaming: each piece of audio as soon as it is final (interleaved when
// n_channels > 1).  Return false to stop the run.
typedef bool (*codec_tts_audio_callback)(const float * pcm, int32_t n_samples, int32_t sample_rate,
                                         int32_t n_channels, void * user_data);

// Per-request options.  NULL strings and negative numbers keep the model's
// defaults (from its GGUF metadata); the names follow
// codec_audio_lm_tts_request.
struct codec_tts_request {
    const char * text;               // required
    const char * voice;              // builtin voice name (codec_tts_voices)
    const char * speaker_file;       // speaker GGUF (codec_speaker_save / save_speaker)
    const char * ref_audio_path;     // WAV to clone the voice from
    const char * ref_text;           // transcript of ref_audio
    const char * save_speaker;       // write the voice in use here
    const char * instruct;           // instruction text, for models that take it
    const char * preset;             // a sampling preset the model defines
    const char * grammar;            // GBNF for the backbone sampler (NULL = the model's own)
    uint32_t     seed;               // 0 = default
    float        temperature;
    float        top_p;
    int32_t      top_k;
    float        min_p;
    float        repetition_penalty;
    int32_t      min_frames;         // shortest generation before the stop token may end it
    float        lm_guidance;        // classifier-free guidance weight of the backbone pair
    float        flow_guidance;      // guidance of a flow / diffusion decoder
    int32_t      flow_steps;         // steps of a flow / diffusion decoder
    int32_t      segment_bytes;      // generate long text in segments of at most this many UTF-8
                                     // bytes (0 = the model's own segmentation)
    codec_tts_audio_callback on_audio;   // stream instead of filling out_pcm (may be NULL)
    void *       user_data;
    int32_t      stream_chunk_ms;    // with on_audio: decode while generating, about this much
                                     // audio per piece (0 = one piece per segment)
};

struct codec_tts_result {
    int32_t n_frames;             // audio frames generated
    char    stop_reason[32];      // "eos", "eos_code_c0", "max_frames", ...
    double  total_ms;
    double  decode_ms;
};

struct codec_tts_params  codec_tts_default_params(void);
struct codec_tts_request codec_tts_default_request(void);

// NULL on failure; the reason goes to `err` (may be NULL).
struct codec_tts * codec_tts_load(const struct codec_tts_params * params, char * err, size_t err_size);
void               codec_tts_free(struct codec_tts * tts);

// Synthesize one request into `out_pcm` (free with codec_pcm_buffer_free; left
// empty when the request streams through on_audio); `out_result` may be NULL.
// On failure returns false; codec_tts_last_error says why.
bool codec_tts_run(struct codec_tts * tts, const struct codec_tts_request * req,
                   struct codec_pcm_buffer * out_pcm, struct codec_tts_result * out_result);
const char * codec_tts_last_error(const struct codec_tts * tts);

// Voices `voice` can name: the codec's builtin voices, or the speakers in
// `speaker_file` (may be NULL).  Fills a per-session list read with
// codec_tts_voice; returns its size, or -1.
int32_t      codec_tts_voices       (struct codec_tts * tts, const char * speaker_file);
const char * codec_tts_voice        (const struct codec_tts * tts, int32_t i);
const char * codec_tts_default_voice(const struct codec_tts * tts);
// The model takes `instruct`.
bool         codec_tts_takes_instruct(const struct codec_tts * tts);

#ifdef __cplusplus
}
#endif

#endif  // CODEC_TTS_H
