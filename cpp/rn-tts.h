#ifndef RNTTS_H
#define RNTTS_H

#include <vector>
#include <string>
#include "llama.h"
#include "nlohmann/json.hpp"
#include "common.h"

using json = nlohmann::ordered_json;

namespace rnllama {

// Forward declarations
struct llama_rn_context;

// TTS type enumeration
enum tts_type {
    UNKNOWN = -1,
    OUTETTS_V0_1 = 0,
    OUTETTS_V0_2 = 1,
    OUTETTS_V0_3 = 2,
};

// Audio completion result structure
struct llama_rn_audio_completion_result {
    std::string prompt;
    const char *grammar;
};

// TTS context for TTS-specific functionality
struct llama_rn_tts_context {
    // TTS state fields
    std::vector<llama_token> audio_tokens;
    std::vector<llama_token> guide_tokens;
    bool next_token_uses_guide_token = true;

    // Constructor and destructor
    llama_rn_tts_context() = default;
    ~llama_rn_tts_context() = default;

    // TTS utility methods
    tts_type getTTSType(llama_rn_context* main_ctx, json speaker = nullptr);
    llama_rn_audio_completion_result getFormattedAudioCompletion(llama_rn_context* main_ctx, const std::string &speaker_json_str, const std::string &text_to_speak);
    std::vector<llama_token> getAudioCompletionGuideTokens(llama_rn_context* main_ctx, const std::string &text_to_speak);
    std::vector<float> decodeAudioTokens(llama_rn_context* main_ctx, const std::vector<llama_token> &tokens);
    void setGuideTokens(const std::vector<llama_token> &tokens);

};

// TTS processing functions
std::string process_text(const std::string & text, const tts_type tts_type);
std::string audio_text_from_speaker(json speaker, const tts_type type);
std::string audio_data_from_speaker(json speaker, const tts_type type);

// Number conversion utilities
std::string number_to_words(const std::string & number_str);
std::string replace_numbers_with_words(const std::string & input_text);

// Audio processing utilities
std::vector<float> embd_to_audio(const float * embd, const int n_codes, const int n_embd, const int n_thread);

// the default speaker profile is from: https://github.com/edwko/OuteTTS/blob/main/outetts/version/v1/default_speakers/en_male_1.json
extern const std::string default_audio_text;
extern const std::string default_audio_data;
extern const char *OUTETTS_V1_GRAMMAR;
extern const char *OUTETTS_V2_GRAMMAR;

}

#endif /* RNTTS_H */
