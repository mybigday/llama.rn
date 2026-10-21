#ifndef RNTTS_H
#define RNTTS_H

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include "llama.h"
#include "nlohmann/json.hpp"

using json = nlohmann::ordered_json;

struct codec_audio_lm;
struct codec_audio_lm_generation;
struct codec_audio_lm_completion;
struct codec_speaker;

namespace rnllama {

struct llama_rn_context;

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

// getFormattedAudioCompletion() result: what the request's format needs, as
// completion() parameters (snake_case, as JS passes them) in
// `completion_params`: the prompt, the grammar and whether hidden states are
// needed.  `logit_bias` (one bias per vocabulary token, the Float32Array form
// of completion()'s logit_bias) rules out the tokens the model never samples.
// Sampling and length are the caller's: completion() runs with exactly the
// parameters it is given.
struct llama_rn_audio_completion_result {
    std::string prompt;
    std::string grammar;
    bool embedding = false;
    std::string flow;
    json completion_params = json::object();
    std::vector<float> logit_bias;   // n_vocab entries, or empty
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

// Text-to-speech over codec.cpp's codec_audio_lm (include/codec_audio_lm.h).
// getFormattedAudioCompletion arms a request; the normal completion then runs
// it: loadPrompt decodes the request's prompt (loadPrompt), and every
// nextToken step samples with the completion's own sampler and reports the
// step to codec.cpp (observe), which says what the backbone reads next and
// when to stop.  endCompletion collects the decoder input (endGeneration).
struct llama_rn_context_tts {
    ::codec_audio_lm * alm = nullptr;

    // The last generation's decoder input (codec_audio_lm_get_generation):
    // codec codes, which the completion result returns as `audio_tokens`
    // (frame-major, code_width per frame), or latents, which it returns as
    // `embeddings` / `embedding_dim`.  decodeAudioTokens /
    // decodeAudioEmbeddings decode such data at any later time.
    std::vector<llama_token> audio_tokens;
    std::vector<float> audio_latents;
    int audio_latent_dim = 0;
    int n_frames = 0;               // audio frames of the last generation
    std::string stop_reason;

    // The completion's voice (JSIParams): a registry speaker id (-1 = none) or
    // a builtin voice name ("" = the model's default).
    int pending_speaker_id = -1;
    std::string pending_voice;

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
    // data, which its completion result holds.
    void reset();

    llama_rn_tts_capabilities getTTSCapabilities(llama_rn_context* main_ctx);
    // Checks the backbone context against what the model asks for; empty
    // when it fits, else the reason.
    std::string checkBackbone(llama_rn_context* main_ctx);

    // Arms a request for the next completion: `text`, plus the voice from
    // `speakerId` (registry), else `speaker.codec_voice` (builtin voice name),
    // else the model's default.  `instruct` describes how to speak, for models
    // that take one (can_take_instruct).  `options` holds codec.cpp's request
    // overrides (min_frames).  The generation samples with the completion's
    // sampling, codec.cpp's own samplers included.  Throws on failure.
    llama_rn_audio_completion_result getFormattedAudioCompletion(llama_rn_context* main_ctx, const json &speaker,
                                                                 const std::string &text_to_speak, int speakerId = -1,
                                                                 const std::string &instruct = "",
                                                                 const json &options = json::object());
    // Loads the completion's prompt (getFormattedAudioCompletion's, or the
    // caller's edit of it) MTMD-style: codec.cpp builds its chunks, tokens and
    // embedding rows, with the completion's voice (pending_speaker_id /
    // pending_voice, else the model's default) filling the <__voice__>
    // markers, and its step hook takes part in every step.  Throws on failure.
    void loadPrompt(llama_rn_context* main_ctx);
    // The last generation's decoder input into audio_tokens / audio_latents.
    // Throws on failure.
    void collectGeneration();

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
    // Codebooks per frame of `audio_tokens` (0 until this model generated).
    int getCodeWidth() const { return code_width; }

private:
    // A registry speaker, else a builtin voice by name ("" = the default).
    // Null for the model's default when it has none.  Throws on an unknown voice.
    rn_voice resolveVoice(int speaker_id, const std::string &voice_name);
    std::vector<float> decodeGeneration(const codec_audio_lm_generation &gen, const char * what);
    int code_width = 0;             // codebooks per frame of this model's codes (0: none seen yet)

    // The voice of the last completion, kept for decoding its generation later.
    rn_voice gen_voice;

    // codec.cpp's part in the completion's steps (rn-tts.cpp).
    struct step_hook;
    std::unique_ptr<step_hook> hook;
};

}

#endif /* RNTTS_H */
