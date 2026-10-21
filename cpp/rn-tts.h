#ifndef RNTTS_H
#define RNTTS_H

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include "llama.h"
#include "nlohmann/json.hpp"

using json = nlohmann::ordered_json;

struct codec_audio_lm;
struct codec_audio_lm_generation;
struct codec_speaker;
struct common_sampler;
struct common_params_sampling;

namespace rnllama {

struct llama_rn_context;
struct tts_guard_sampler;

// Model families llama.rn reports in getTTSCapabilities().type.  The audio
// side (prompt formats, sampling, stop rules, decoding) is codec.cpp's
// codec_audio_lm; this only names the model for the JS layer.
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
    CHATTERBOX_T3 = 11,
    CHATTERBOX_T3_MULTILINGUAL = 12,
    BLUEMAGPIE_TTS = 13,
    CHATTERBOX_TURBO = 14,
    COSYVOICE = 15,
};

// getFormattedAudioCompletion() result.  The prompt itself is codec.cpp's and
// stays native: completion() runs the armed request whatever prompt it gets.
struct llama_rn_audio_completion_result {
    std::string prompt;
    std::string grammar;
    bool embedding = false;
    std::string flow;
};

// generateAudioCodes() options (deprecated entry point).
struct llama_rn_audio_codes_options {
    std::string prompt;
    int  max_frames = 500;
    float temperature = 0.9f;
    float top_p = 0.95f;
    int   top_k = 50;
    uint32_t seed = 0;
};

using llama_rn_audio_codes_progress_cb =
    std::function<bool(int step, const std::vector<int32_t> &codes)>;

struct llama_rn_audio_codes_result {
    std::vector<int32_t> codes;   // one entry per audio frame
    int n_codebook = 0;
    int n_frames = 0;
    bool stopped_on_eos = false;
    bool aborted = false;
};

// A codec voice (codec_speaker.h).  Created and loaded voices are freed with
// codec_speaker_free; builtin ones belong to the model and are not.  Shared, so
// a generation keeps its voice for a later decode after the speaker is released.
using rn_voice = std::shared_ptr<const ::codec_speaker>;

// A registry speaker (LlamaSpeaker): a voice cloned from reference audio
// (all its conditioning is computed once, at creation) or loaded from a
// speaker file.
struct rn_speaker {
    rn_voice voice;
    // Kept for the JS shape of LlamaSpeaker; the voice is ready when created.
    int rows = 0;
    bool baked = true;
};

struct llama_rn_speaker_info {
    std::string name;
    std::string ref_text;
    bool is_default = false;
};

struct llama_rn_tts_capabilities {
    int type = UNKNOWN;             // tts_type
    std::string prompt_kind;
    std::string family;
    bool requires_phonemes = false;
    std::string default_language = "en-us";
    std::vector<llama_rn_speaker_info> builtin_voices;
    bool can_speaker_from_audio = false;
    bool can_take_instruct = false;   // getFormattedAudioCompletion's `instruct` is used
    bool needs_voice = false;         // a LlamaSpeaker is required (no builtin voice to fall back on)
    int sample_rate = 0;
    // Always empty: codec_audio_lm samples with the model's own settings.
    json sampling_defaults = json::object();
};

// Text-to-speech over codec.cpp's codec_audio_lm (include/codec_audio_lm.h):
// codec.cpp says what to feed the backbone, how to sample and when to stop;
// this drives the backbone (the main llama_context) through that loop.
struct llama_rn_context_tts {
    ::codec_audio_lm * alm = nullptr;

    // The last generation's decoder input (codec_audio_lm_get_generation,
    // segments concatenated): codec codes, which the completion result returns
    // as `audio_tokens` (frame-major, code_width per frame), or latents, which
    // it returns as `embeddings` / `embedding_dim`.  decodeAudioTokens /
    // decodeAudioEmbeddings decode such data at any later time.
    std::vector<llama_token> audio_tokens;
    std::vector<float> audio_latents;
    int audio_latent_dim = 0;
    int n_frames = 0;               // audio frames of the last generation
    std::string stop_reason;

    // Speaker id from the completion params (JSIParams); -1 = none.
    int pending_speaker_id = -1;

    std::unordered_map<int, rn_speaker> speakers;
    int next_speaker_id = 1;

    // `backbone_path` is the main model's GGUF; models whose text lane reads
    // the backbone's embedding table (MOSS-TTS-Realtime) need it.  Throws on
    // failure.
    llama_rn_context_tts(const std::string &vocoder_model_path, const std::string &backbone_path,
                         bool use_gpu = false, const std::vector<std::string> &kv_overrides = {},
                         int n_threads = 0);
    ~llama_rn_context_tts();

    // Per-completion state (rewind): the speaker id and the last generation's
    // data, which its completion result holds; the armed request survives.
    void reset();

    llama_rn_tts_capabilities getTTSCapabilities(llama_rn_context* main_ctx);
    // Checks the backbone context against what the model asks for; empty
    // when it fits, else the reason.
    std::string checkBackbone(llama_rn_context* main_ctx);

    // Arms a request: `text`, plus the voice from `speakerId` (registry), else
    // `speaker.codec_voice` (builtin voice name), else the model's default.
    // The voice is resolved here so an unknown voice fails early.  `instruct`
    // describes how to speak, for models that take one (can_take_instruct).
    llama_rn_audio_completion_result getFormattedAudioCompletion(llama_rn_context* main_ctx, const json &speaker,
                                                                 const std::string &text_to_speak, int speakerId = -1,
                                                                 const std::string &instruct = "");
    bool isArmed() const { return armed; }

    // ── Generation, inside the completion flow ───────────────────────────
    // loadPrompt calls beginGeneration (an armed request), nextToken calls
    // step, endCompletion calls endGeneration.  Long text is generated one
    // segment at a time: after a STOP, nextSegment decodes the segment and
    // prefills the next one.  beginGeneration / nextSegment set the
    // context's sampling params to the model's (restored by endGeneration);
    // the completion re-inits its sampler after them.  Throw on failure.
    bool isGenerating() const { return generating; }
    // Returns the KV position after the first segment's prompt.
    int beginGeneration(llama_rn_context* main_ctx);
    struct step_result {
        bool stop = false;          // the segment ended (see nextSegment)
        bool fed = false;           // one position was decoded at n_past
        bool audio = false;         // this step produced an audio frame
        llama_token tok = -1;       // audio: the sampled token, else the frame index
        llama_token sampled = -1;   // the backbone token sampled this step (-1: none)
    };
    // One step: sample with `smpl` when the model asks, observe, decode what
    // it feeds at `n_past`.
    step_result step(llama_rn_context* main_ctx, common_sampler * smpl, int n_past);
    // After a STOP: keeps the segment's decoder input; returns the KV position
    // after the next segment's prompt, or -1 when it was the last.
    int nextSegment(llama_rn_context* main_ctx);
    // Keeps the decoder input generated so far and restores the context.
    // Safe to call when not generating.
    void endGeneration(llama_rn_context* main_ctx);

    // DEPRECATED: arms `opts.prompt` and runs a completion.
    llama_rn_audio_codes_result generateAudioCodes(llama_rn_context* main_ctx, const llama_rn_audio_codes_options &opts,
                                                   const llama_rn_audio_codes_progress_cb &on_frame = nullptr);

    // ── Speaker registry ─────────────────────────────────────────────────
    int createSpeaker(llama_rn_context* main_ctx, const std::vector<float> &pcm, int sample_rate,
                      const std::string &ref_text, float emotion, bool has_emotion, bool bake);
    bool bakeSpeaker(llama_rn_context* main_ctx, int id);
    const rn_speaker* getSpeaker(int id) const;
    void releaseSpeaker(int id);
    // Throws on failure.
    int loadSpeaker(const std::string &path, const std::string &name);
    // Writes a registry speaker's voice to a speaker file as `name` ("" = the
    // codec's default name).  Throws on failure.
    void saveSpeaker(int id, const std::string &path, const std::string &name);
    std::vector<llama_rn_speaker_info> listSpeakers(const std::string &path);

    // PCM of a generation's codes (`audio_tokens`) or latents (`embeddings`),
    // with the voice of the last generation.  Throws on failure.
    std::vector<float> decodeAudioTokens(llama_rn_context* main_ctx, const std::vector<llama_token> &tokens);
    std::vector<float> decodeAudioEmbeddings(llama_rn_context* main_ctx, const std::vector<float> &embeddings,
                                             int embedding_dim);
    int getAudioSampleRate() const;

private:
    struct armed_voice {
        std::string voice;      // builtin voice name ("" = default)
        int speaker_id = -1;    // registry speaker
        bool operator==(const armed_voice &o) const { return voice == o.voice && speaker_id == o.speaker_id; }
    };

    // The codec voice for `v`; null for the model's default when it has none.
    // Throws on an unknown voice.
    rn_voice resolveVoice(const armed_voice &v);
    // Prompt of segment `seg` (codec_audio_lm), decoded from position 0.
    int beginSegment(llama_rn_context* main_ctx);
    // Appends the current segment's decoder input to audio_tokens / audio_latents.
    void finishSegment();
    std::vector<float> decodeGeneration(const codec_audio_lm_generation &gen, const char * what);
    int code_width = 0;             // codebooks per frame of this model's codes (0: none seen yet)

    bool armed = false;
    std::string armed_text;
    std::string armed_instruct;
    armed_voice armed_v;
    // The voice of the generation in progress, or of the last one (decode).
    rn_voice gen_voice;

    // The generation in progress.
    bool generating = false;
    bool no_guidance = false;       // the context has too few sequences for the model's guidance pair
    std::vector<std::string> segments;
    size_t seg = 0;
    bool seg_open = false;          // the current segment has not been decoded yet
    int32_t seg_frames = 0;
    int32_t seg_max_frames = 0;
    int32_t n_seq = 1;
    int32_t n_embd = 0;
    bool needs_hidden = false;
    bool sample_next = false;
    int32_t min_frames = 0;
    llama_token stop_token = -1;
    std::vector<llama_token> penalty_prompt;   // fed to the sampler before its first draw
    std::unique_ptr<tts_guard_sampler> guard;
    std::unique_ptr<::common_params_sampling> saved_sampling;
    bool saved_embedding = false;
};

}

#endif /* RNTTS_H */
