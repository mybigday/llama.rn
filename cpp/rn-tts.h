#ifndef RNTTS_H
#define RNTTS_H

#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
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

// The completion's prompt as codec.cpp builds it (buildPrompt): token runs
// and embedding rows, each holding its rows sequence by sequence.
struct llama_rn_tts_prompt {
    struct chunk {
        std::vector<llama_token> tokens;   // n_seq * n, or empty
        std::vector<float> embd;           // n_seq * n * n_embd, or empty
        int n = 0;
        int n_embd = 0;
    };
    std::vector<chunk> chunks;
    int n_seq = 1;
    bool needs_prompt_hidden = false;      // prefill takes every position's hidden state
    // codec.cpp's codebook head groups: one sampler each (sample_head).
    struct head_group {
        bool penalties = true;
        std::vector<llama_token> history;  // accepted into the penalties first
    };
    std::vector<head_group> head_groups;
};

// What a step says (step): the end, whether its token is audio, and the
// token the backbone reads next (-1: the embedding rows in pending_next_embd).
struct llama_rn_tts_step {
    bool stop = false;
    bool audio = false;
    llama_token feed = -1;
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
    bool uses_ref_text = false;       // a cloned voice's transcript (createSpeaker refText) is used
    int sample_rate = 0;
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
    // codec codes or latents, frame-major, which the completion result
    // returns as `audio_tokens` / `audio_latents`.  decodeAudioTokens /
    // decodeAudioLatents decode such data at any later time.
    std::vector<llama_token> audio_tokens;
    std::vector<float> audio_latents;
    int n_frames = 0;               // audio frames of the last generation
    std::string stop_reason;

    std::unordered_map<int, rn_speaker> speakers;
    int next_speaker_id = 1;

    // `backbone_path` is the main model's GGUF; models whose text lane reads
    // the backbone's embedding table (MOSS-TTS-Realtime) need it.  Throws on
    // failure.
    llama_rn_context_tts(const std::string &vocoder_model_path, const std::string &backbone_path,
                         bool use_gpu = false, const std::vector<std::string> &kv_overrides = {},
                         int n_threads = 0);
    ~llama_rn_context_tts();

    // Per-completion state (rewind): the last generation's data, which its
    // completion result holds.
    void reset();

    llama_rn_tts_capabilities getTTSCapabilities(llama_rn_context* main_ctx);
    // Checks the backbone context against what the model asks for; empty
    // when it fits, else the reason.
    std::string checkBackbone(llama_rn_context* main_ctx);

    // Arms a request for the next completion: `text`, plus the voice from
    // `speakerId` (registry), else `speaker.codec_voice` (builtin voice name),
    // else the model's default.  `instruct` describes how to speak, for models
    // that take one (can_take_instruct).  The generation samples with the
    // completion's sampling, codec.cpp's own samplers included, and its
    // n_min_predict is codec.cpp's min_frames.  Throws on failure.
    llama_rn_audio_completion_result getFormattedAudioCompletion(llama_rn_context* main_ctx, const json &speaker,
                                                                 const std::string &text_to_speak, int speakerId = -1,
                                                                 const std::string &instruct = "");
    // ── TTS flow of the completion (rn-completion.cpp) ───────────────────
    // loadPrompt arms prompt_pending; nextToken's TTS prefill decodes
    // buildPrompt (codec.cpp builds the completion's prompt MTMD-style: its
    // voice markers become embedding rows, filled by the completion's voice,
    // tts_speaker_id / tts_voice, else the model's default; its n_min_predict
    // becomes min_frames) and passes prefill the prompt's hidden states where
    // the model reads them.
    // Every step after the backbone's calls step, which has codec.cpp's
    // codebook heads drawn through `sample_head` and leaves the embedding
    // rows the backbone reads next in pending_next_embd.  Throw on failure.
    bool prompt_pending = false;
    bool sample_backbone = false;   // the next step samples a backbone token
    int n_seq = 1;                  // sequences of the generation (2: guidance pair)
    std::vector<float> pending_next_embd;   // n_seq rows, or empty
    llama_rn_tts_prompt buildPrompt(llama_rn_context* main_ctx);
    void prefill(llama_rn_context* main_ctx, const float * hiddens, int n_pos);
    llama_rn_tts_step step(llama_rn_context* main_ctx, llama_token sampled,
                           const std::function<llama_token(int group, const float * logits, int n_vocab)> &sample_head);
    // The last generation's decoder input into audio_tokens / audio_latents.
    // Throws on failure.
    void collectGeneration();

    // ── Speaker registry ─────────────────────────────────────────────────
    int createSpeaker(llama_rn_context* main_ctx, const std::vector<float> &pcm, int sample_rate,
                      const std::string &ref_text, float emotion, bool has_emotion, bool bake);
    bool bakeSpeaker(llama_rn_context* main_ctx, int id);
    const rn_speaker* getSpeaker(int id) const;
    void releaseSpeaker(int id);
    // A speaker GGUF of this model.  Throws on failure.
    int loadSpeaker(const std::string &path, const std::string &name);
    // Writes a registry speaker's voice to a speaker file as `name` ("" = the
    // codec's default name).  Throws on failure.
    void saveSpeaker(int id, const std::string &path, const std::string &name);
    std::vector<llama_rn_speaker_info> listSpeakers(const std::string &path);

    // PCM of a generation's codes (`audio_tokens`) or latents (`audio_latents`),
    // with the voice of the last generation.  Throws on failure.
    std::vector<float> decodeAudioTokens(llama_rn_context* main_ctx, const std::vector<llama_token> &tokens);
    std::vector<float> decodeAudioLatents(llama_rn_context* main_ctx, const std::vector<float> &latents);
    int getAudioSampleRate() const;

private:
    // Serializes the codec context, the speaker registry and the last
    // generation's state: a completion's steps run on its thread while
    // getFormattedAudioCompletion, speaker and decode calls (no context
    // claim, so the next request can be prepared meanwhile) run on others.
    // codec_audio_lm takes one thread at a time.  Recursive: public calls
    // nest (getFormattedAudioCompletion -> checkBackbone, -> getSpeaker).
    mutable std::recursive_mutex mutex;

    // A registry speaker, else a builtin voice by name ("" = the default).
    // Null for the model's default when it has none.  Throws on an unknown voice.
    rn_voice resolveVoice(int speaker_id, const std::string &voice_name);
    std::vector<float> decodeGeneration(const codec_audio_lm_generation &gen, const char * what);
    // Per frame of this model's generations (0: none seen yet): codebooks of
    // its codes, or the dimension of its latents.
    int code_width = 0;
    int latent_width = 0;

    // The voice of the last completion, kept for decoding its generation later.
    rn_voice gen_voice;
    // This completion built a prompt (buildPrompt); without one the codec
    // still holds an earlier completion's generation.
    bool has_generation = false;

    int n_embd = 0;
    bool needs_hidden = false;
};

}

#endif /* RNTTS_H */
