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

// getFormattedAudioCompletion() result: everything the completion that runs
// the request needs.  `completion_params` holds completion() parameters
// (snake_case, as JS passes them): the model's backbone sampling, grammar,
// allowed tokens, minimum length, frame budget and whether hidden states are
// needed.  Nothing is applied behind the caller's back; completion() runs with
// exactly the parameters it is given.
struct llama_rn_audio_completion_result {
    std::string prompt;
    std::string grammar;
    bool embedding = false;
    std::string flow;
    json completion_params = json::object();
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

    // Arms a request for the next completion: `text`, plus the voice from
    // `speakerId` (registry), else `speaker.codec_voice` (builtin voice name),
    // else the model's default.  `instruct` describes how to speak, for models
    // that take one (can_take_instruct).  `options` holds codec.cpp's request
    // overrides (seed, the codec-side sampling, frame limits, guidance; see
    // getFormattedAudioCompletion in src/index.ts).  Throws on failure.
    llama_rn_audio_completion_result getFormattedAudioCompletion(llama_rn_context* main_ctx, const json &speaker,
                                                                 const std::string &text_to_speak, int speakerId = -1,
                                                                 const std::string &instruct = "",
                                                                 const json &options = json::object());
    bool isArmed() const { return armed; }
    const std::string & armedText() const { return armed_text; }

    // ── Hooks of the completion (rn-completion.cpp) ──────────────────────
    bool isGenerating() const { return generating; }
    // Starts the armed request: decodes its prompt from position 0 and returns
    // the KV position after it.  `prompt_tokens` receives the prompt's tokens
    // (for penalize_prompt).  Throws on failure.
    int beginGeneration(llama_rn_context* main_ctx, std::vector<llama_token> &prompt_tokens);
    // Whether this step samples a backbone token (else codec.cpp reads only
    // the hidden state).
    bool samplesBackbone() const { return sample_next; }
    struct verdict {
        bool stop = false;
        bool audio = false;         // this step produced an audio frame
        llama_token feed = -1;      // the token the backbone reads next; -1: an embedding row (decodeFeed)
    };
    // Reports one step: `sampled` is the token the completion sampled (-1
    // when samplesBackbone() was false).  Throws on failure.
    verdict observe(llama_rn_context* main_ctx, llama_token sampled);
    // Decodes the embedding row(s) the last observe asked for at `n_past`.
    // Returns false when none is pending.  Throws on failure.
    bool decodeFeed(llama_rn_context* main_ctx, int n_past);
    // Keeps the generation's decoder input and clears the KV cache (it holds
    // embedding rows the completion's token history does not describe).
    // Safe to call when not generating; never throws.
    void endGeneration(llama_rn_context* main_ctx);

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
    struct armed_voice {
        std::string voice;      // builtin voice name ("" = default)
        int speaker_id = -1;    // registry speaker
    };

    // The codec voice for `v`; null for the model's default when it has none.
    // Throws on an unknown voice.
    rn_voice resolveVoice(const armed_voice &v);
    // codec_audio_lm_tts_prompt + get_completion for the armed request.
    void prompt(llama_rn_context* main_ctx, const rn_voice &voice, ::codec_audio_lm_completion &c);
    std::vector<float> decodeGeneration(const codec_audio_lm_generation &gen, const char * what);
    int code_width = 0;             // codebooks per frame of this model's codes (0: none seen yet)

    bool armed = false;
    std::string armed_text;
    std::string armed_instruct;
    armed_voice armed_v;
    json armed_options = json::object();
    // The voice of the generation in progress, or of the last one (decode).
    rn_voice gen_voice;

    // The generation in progress.
    bool generating = false;
    int32_t n_seq = 1;
    int32_t n_embd = 0;
    bool needs_hidden = false;
    bool sample_next = false;
    std::vector<float> feed_rows;   // n_seq embedding rows for the next decode
};

}

#endif /* RNTTS_H */
