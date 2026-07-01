#include "rn-tts.h"
#include "rn-llama.h"
#include "rn-completion.h"
#include "anyascii.h"
#include "common.h"
#include "codec.h"
#include "codec_lm.h"
#include "codec_common.h"
#include "llama.h"
#include "sampling.h"
#include <regex>
#include <map>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <codecvt>
#include <cstring>
#include <limits>
#include <locale>
#include <memory>

namespace rnllama {

// (OuteTTS legacy default voice was previously hardcoded here as
//  default_audio_text / default_audio_data — now lives in TS
//  src/tts-voices.ts as a structured `{ words: [...] }` payload that the
//  audio_text_from_speaker / audio_data_from_speaker helpers re-format.)

// Number conversion maps and functions
static const std::map<int, std::string> ones = {
    {0, "zero"}, {1, "one"}, {2, "two"}, {3, "three"}, {4, "four"},
    {5, "five"}, {6, "six"}, {7, "seven"}, {8, "eight"}, {9, "nine"},
    {10, "ten"}, {11, "eleven"}, {12, "twelve"}, {13, "thirteen"}, {14, "fourteen"},
    {15, "fifteen"}, {16, "sixteen"}, {17, "seventeen"}, {18, "eighteen"}, {19, "nineteen"}
};

static const std::map<int, std::string> tens = {
    {2, "twenty"}, {3, "thirty"}, {4, "forty"}, {5, "fifty"},
    {6, "sixty"}, {7, "seventy"}, {8, "eighty"}, {9, "ninety"}
};

// Convert a number less than 1000 to words
static std::string convert_less_than_thousand(int num) {
    std::string result;

    if (num >= 100) {
        result += ones.at(num / 100) + " hundred ";
        num %= 100;
    }

    if (num >= 20) {
        result += tens.at(num / 10);
        if (num % 10 > 0) {
            result += "-" + ones.at(num % 10);
        }
    } else if (num > 0) {
        result += ones.at(num);
    }

    return result;
}

std::string number_to_words(const std::string & number_str) {
    try {
        size_t decimal_pos = number_str.find('.');
        std::string integer_part = number_str.substr(0, decimal_pos);

        int int_number = std::stoi(integer_part);
        std::string result;

        if (int_number == 0) {
            result = "zero";
        } else {
            if (int_number >= 1000000000) {
                int billions = int_number / 1000000000;
                result += convert_less_than_thousand(billions) + " billion ";
                int_number %= 1000000000;
            }

            if (int_number >= 1000000) {
                int millions = int_number / 1000000;
                result += convert_less_than_thousand(millions) + " million ";
                int_number %= 1000000;
            }

            if (int_number >= 1000) {
                int thousands = int_number / 1000;
                result += convert_less_than_thousand(thousands) + " thousand ";
                int_number %= 1000;
            }

            if (int_number > 0) {
                result += convert_less_than_thousand(int_number);
            }
        }

        // Handle decimal part
        if (decimal_pos != std::string::npos) {
            result += " point";
            std::string decimal_part = number_str.substr(decimal_pos + 1);
            for (char digit : decimal_part) {
                result += " " + ones.at(digit - '0');
            }
        }

        return result;
    } catch (const std::exception& e) {
        // Skip if fails
        return " ";
    }
}

std::string replace_numbers_with_words(const std::string & input_text) {
    std::regex number_pattern(R"(\d+(\.\d+)?)");
    std::string result;
    auto it = std::sregex_iterator(input_text.begin(), input_text.end(), number_pattern);
    auto end = std::sregex_iterator();

    size_t last_pos = 0;
    for (std::sregex_iterator i = it; i != end; ++i) {
        const std::smatch& match = *i;
        result.append(input_text, last_pos, match.position() - last_pos);
        result.append(number_to_words(match.str()));
        last_pos = match.position() + match.length();
    }
    result.append(input_text, last_pos);

    return result;
}

static std::string anyascii_string(const std::string &input) {
    std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> converter;
    auto wstr = converter.from_bytes(input);
    std::string output;
    for (char32_t c : wstr) {
        const char *r;
        size_t rlen = anyascii(c, &r);
        output.append(r, rlen);
    }
    return output;
}

static std::string normalize_plain_tts_text(std::string text);

std::string process_text(const std::string & text, const tts_type tts_type) {
    if (tts_type == OUTETTS_V1_0) {
        return normalize_plain_tts_text(text);
    }
    if (tts_type == SOPRANO_1_1_80M) {
        return normalize_plain_tts_text(text);
    }

    std::string processed_text = replace_numbers_with_words(text);

    if (tts_type == OUTETTS_V0_2 || tts_type == OUTETTS_V0_3) {
        processed_text = anyascii_string(processed_text);

        std::regex dashes(R"([—–-])");
        processed_text = std::regex_replace(processed_text, dashes, " ");
    }

    std::transform(processed_text.begin(), processed_text.end(),
                  processed_text.begin(), ::tolower);

    std::regex special_chars(R"([-_/,\.\\])");
    processed_text = std::regex_replace(processed_text, special_chars, " ");

    std::regex non_alpha(R"([^a-z\s])");
    processed_text = std::regex_replace(processed_text, non_alpha, "");

    std::regex multiple_spaces(R"(\s+)");
    processed_text = std::regex_replace(processed_text, multiple_spaces, " ");

    processed_text = std::regex_replace(processed_text, std::regex(R"(^\s+|\s+$)"), "");

    std::string separator = (tts_type == OUTETTS_V0_3) ? "<|space|>" : "<|text_sep|>";
    processed_text = std::regex_replace(processed_text, std::regex(R"(\s)"), separator);

    return processed_text;
}

std::string audio_text_from_speaker(json speaker, const tts_type type) {
    std::string audio_text = "<|text_start|>";

    if (type == OUTETTS_V0_2 || type == OUTETTS_V0_3) {
        std::string separator = (type == OUTETTS_V0_3) ? "<|space|>" : "<|text_sep|>";
        for (const auto &word : speaker["words"]) {
            audio_text += word["word"].get<std::string>() + separator;
        }
    }

    return audio_text;
}

std::string audio_data_from_speaker(json speaker, const tts_type type) {
    std::string audio_data = "<|audio_start|>\n";

    if (type == OUTETTS_V0_2 || type == OUTETTS_V0_3) {
        std::string code_start = (type == OUTETTS_V0_3) ? "" : "<|code_start|>";
        std::string code_end = (type == OUTETTS_V0_3) ? "<|space|>" : "<|code_end|>";
        for (const auto &word : speaker["words"]) {
            std::string word_text = word["word"].get<std::string>();
            double duration = word["duration"].get<double>();
            std::vector<int> codes = word["codes"].get<std::vector<int>>();

            // Create the audio output entry
            std::ostringstream word_entry;
            word_entry << word_text << "<|t_" << std::fixed << std::setprecision(2)
                       << duration << "|>" + code_start;
            for (const auto &Code : codes) {
                word_entry << "<|" << Code << "|>";
            }
            word_entry << code_end << "\n";
            audio_data += word_entry.str();
        }
    }

    return audio_data;
}

static std::vector<std::string> feature_tokens_from_json(const json &features) {
    std::vector<std::string> result;
    const std::vector<std::string> keys = {"energy", "spectral_centroid", "pitch"};
    for (const auto &key : keys) {
        double value = features.is_object() && features.contains(key) ? features[key].get<double>() : 0.0;
        std::ostringstream out;
        out << "<|" << key << "_" << value << "|>";
        result.push_back(out.str());
    }
    return result;
}

static std::string outetts_v1_codes_from_speaker(json speaker) {
    if (!speaker.is_object() || !speaker.contains("words")) {
        return "";
    }

    std::ostringstream audio;
    if (speaker.contains("global_features") && speaker["global_features"].is_object()) {
        audio << "<|global_features_start|>";
        for (const auto &feature : feature_tokens_from_json(speaker["global_features"])) {
            audio << feature;
        }
        audio << "<|global_features_end|>\n";
    }

    for (const auto &word : speaker["words"]) {
        audio << "<|word_start|>" << word.value("word", std::string()) << "<|features|>";
        audio << "<|t_" << std::fixed << std::setprecision(2) << word.value("duration", 0.0) << "|>";
        for (const auto &feature : feature_tokens_from_json(word.value("features", json::object()))) {
            audio << feature;
        }
        audio << "<|code|>";

        if (word.contains("c1") && word.contains("c2")) {
            const auto c1 = word["c1"].get<std::vector<int>>();
            const auto c2 = word["c2"].get<std::vector<int>>();
            const size_t n = std::min(c1.size(), c2.size());
            for (size_t i = 0; i < n; ++i) {
                audio << "<|c1_" << c1[i] << "|><|c2_" << c2[i] << "|>";
            }
        }
        audio << "<|word_end|>\n";
    }

    return audio.str();
}

// Constructor and destructor implementations
llama_rn_context_tts::llama_rn_context_tts(const std::string &vocoder_model_path, int /* batch_size */, bool use_gpu) {
  struct codec_model_params model_params = codec_model_default_params();
  model_params.use_gpu = use_gpu;
  codec_model = codec_model_load_from_file(vocoder_model_path.c_str(), model_params);
  if (codec_model == nullptr) {
      throw std::runtime_error("Failed to load codec model");
  }

  struct codec_context_params context_params = codec_context_default_params();
  codec_ctx = codec_init_from_model(codec_model, context_params);
  if (codec_ctx == nullptr) {
      codec_model_free(codec_model);
      codec_model = nullptr;
      throw std::runtime_error("Failed to initialize codec context");
  }

  type = UNKNOWN; // Will be determined when used
}

llama_rn_context_tts::~llama_rn_context_tts() {
  if (codec_lm_state != nullptr) {
      codec_lm_state_free(codec_lm_state);
      codec_lm_state = nullptr;
  }
  if (codec_lm != nullptr) {
      codec_lm_free(codec_lm);
      codec_lm = nullptr;
  }
  if (codec_ctx != nullptr) {
      codec_free(codec_ctx);
      codec_ctx = nullptr;
  }
  if (codec_model != nullptr) {
      codec_model_free(codec_model);
      codec_model = nullptr;
  }
  type = UNKNOWN;
}

void llama_rn_context_tts::reset() {
    audio_tokens.clear();
    pending_codebook1 = -1;
    if (codec_lm_state != nullptr) {
        codec_lm_state_reset(codec_lm_state);
    }
}

// Forward declarations from rn-llama.h
extern bool rnllama_verbose;
void log(const char *level, const char *function, int line, const char *format, ...);

#define LOG_ERROR(MSG, ...) log("ERROR", __func__, __LINE__, MSG, ##__VA_ARGS__)
#define LOG_WARNING(MSG, ...) log("WARNING", __func__, __LINE__, MSG, ##__VA_ARGS__)
#define LOG_INFO(MSG, ...) log("INFO", __func__, __LINE__, MSG, ##__VA_ARGS__)

static std::string lowercase_copy(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return (char) std::tolower(c);
    });
    return value;
}

static bool contains_case_insensitive(const std::string &haystack, const std::string &needle) {
    return lowercase_copy(haystack).find(lowercase_copy(needle)) != std::string::npos;
}

static std::string normalize_plain_tts_text(std::string text) {
    text = std::regex_replace(text, std::regex(R"(\s+)"), " ");
    text = std::regex_replace(text, std::regex(R"(^\s+|\s+$)"), "");
    return text;
}

enum class tts_prompt_kind {
    OUTETTS_LEGACY,
    OUTETTS_V0_3,
    OUTETTS_V1_0,
    SOPRANO,
    NEUTTS,
    CSM,
    QWEN3_TTS,
    MOSS_TTS_REALTIME,
    MOSS_TTSD,
    CHATTERBOX,
};

enum class tts_decode_kind {
    CODEC_CODES,
    HIDDEN_STATES,
    CODEC_LM_AR,  // codes are produced by generateAudioCodes() running the
                  // backbone + codec_lm AR loop; the JS layer is expected
                  // to call generateAudioCodes() instead of completion().
    UNSUPPORTED,
};

struct parsed_audio_token {
    int codebook;
    int codec_value;
};

struct tts_model_profile {
    tts_type type;
    tts_prompt_kind prompt_kind;
    struct audio_token_config {
        struct code_range {
            llama_token start;
            int count;
            int codebook_size;
        };

        int n_codebook;
        std::vector<code_range> code_ranges;
    } audio;
    tts_decode_kind decode_kind;
    // For CODEC_LM_AR profiles where the codec_lm emits one text-modality
    // codebook ahead of the audio codebooks (MOSS-TTS-Realtime / MOSS-TTSD:
    // cb-0 is a text vocab token, cb-1..N-1 are RVQ audio codes), this is
    // the index of the first audio codebook (default 0 = all audio).
    // `decodeAudioTokens` slices that prefix off before forming a
    // `codec_token_buffer`; `generateAudioCodes` calls
    // `codec_lm_state_set_text_context` with the cb-0 value when offset > 0.
    int audio_codebook_offset = 0;
};

static const tts_model_profile TTS_PROFILES[] = {
    {OUTETTS_V0_1, tts_prompt_kind::OUTETTS_LEGACY, {1, {{50307, 4100, 4100}}}, tts_decode_kind::CODEC_CODES},
    {OUTETTS_V0_2, tts_prompt_kind::OUTETTS_LEGACY, {1, {{50307, 4100, 4100}}}, tts_decode_kind::CODEC_CODES},
    {OUTETTS_V0_3, tts_prompt_kind::OUTETTS_V0_3, {1, {{50307, 4100, 4100}}}, tts_decode_kind::CODEC_CODES},
    {OUTETTS_V1_0, tts_prompt_kind::OUTETTS_V1_0, {2, {{128256, 1025, 1025}, {129281, 1025, 1025}}}, tts_decode_kind::CODEC_CODES},
    {SOPRANO_1_1_80M, tts_prompt_kind::SOPRANO, {0, {}}, tts_decode_kind::HIDDEN_STATES},
    {NEUTTS_NANO,    tts_prompt_kind::NEUTTS,  {1, {{128262, 65536, 65536}}}, tts_decode_kind::CODEC_CODES},
    {NEUTTS_AIR,     tts_prompt_kind::NEUTTS,  {1, {{151671, 65536, 65536}}}, tts_decode_kind::CODEC_CODES},
    // CSM (Sesame): Llama-3.2-1B backbone + Mimi 32-codebook codec + residual-depth-AR codec_lm.
    // n_codebook / code_ranges are unused on the CODEC_LM_AR path — generateAudioCodes
    // writes raw codec values (no llama vocab offsets) directly into audio_tokens.
    {CSM_1B,         tts_prompt_kind::CSM,     {32, {}}, tts_decode_kind::CODEC_LM_AR, 0},
    // Qwen3-TTS-12Hz-0.6B: Qwen3 talker (1024 hidden, 28 layers) + qwen3_tts_tokenizer
    // 16-codebook codec with residual-depth-AR codec_lm.  cb-0 is speech-only
    // (vocab=3072), so audio_codebook_offset=0 (all 16 codebooks are audio).
    {QWEN3_TTS_0_6B, tts_prompt_kind::QWEN3_TTS, {16, {}}, tts_decode_kind::CODEC_LM_AR, 0},
    // MOSS-TTS-Realtime: Qwen3-2B backbone + MOSS-Audio-Tokenizer 16 RVQ + residual-depth-AR.
    // cb-0 is a text token sampled from the backbone's lm_head (151936 vocab); cb-1..16
    // are audio.  audio_codebook_offset=1 — decodeAudioTokens slices cb-0 off before
    // sending to codec_decode, and generateAudioCodes calls codec_lm_state_set_text_context
    // with codes[0] for the depth decoder's pos-0 input.
    {MOSS_TTS_REALTIME, tts_prompt_kind::MOSS_TTS_REALTIME, {17, {}}, tts_decode_kind::CODEC_LM_AR, 1},
    // MOSS-TTSD-v0.7: Qwen3-1.7B + XY-Tokenizer 8-codebook with parallel_heads_delay.
    // cb-0 is a merged text+speech vocab; cb-1..7 are RVQ audio with per-codebook
    // delay_pattern shift.  decodeAudioTokens reads codec_lm_info.delay_pattern to
    // un-shift each audio codebook before forming the codec_token_buffer.
    {MOSS_TTSD_V07, tts_prompt_kind::MOSS_TTSD, {8, {}}, tts_decode_kind::CODEC_LM_AR, 1},
    // Chatterbox T3 (English + multilingual variants): driven by a completely different
    // runtime — T3 backbone + S3G/S3T codec with custom prompt assembly via
    // `cond_enc.*` tensors from t3-extras.gguf.  Detection only for now; the runtime
    // path is tracked separately (see rn-tts plans / chatterbox.md in codec.cpp).
    {CHATTERBOX_T3,              tts_prompt_kind::CHATTERBOX, {1, {}}, tts_decode_kind::UNSUPPORTED, 0},
    {CHATTERBOX_T3_MULTILINGUAL, tts_prompt_kind::CHATTERBOX, {1, {}}, tts_decode_kind::UNSUPPORTED, 0},
};

// Takes the resolved (vocab-probed) code ranges + n_codebook from the
// profile, so callers can share the lookup with both the default profile
// values and the runtime-probed overrides.
static bool token_code_from_ranges(
        const std::vector<llama_rn_audio_code_range> & ranges,
        int n_codebook,
        llama_token token,
        parsed_audio_token & parsed) {
    for (size_t i = 0; i < ranges.size(); ++i) {
        const auto &range = ranges[i];
        if (token >= range.start && token < range.start + range.count) {
            const int offset = (int) (token - range.start);
            parsed.codebook = ranges.size() == 1 && n_codebook > 1 && range.codebook_size > 0
                ? (offset / range.codebook_size) % n_codebook
                : (int) i;
            parsed.codec_value = range.codebook_size > 0 ? offset % range.codebook_size : offset;
            return true;
        }
    }
    return false;
}

static const tts_model_profile & profile_for_type(tts_type type) {
    for (const auto &profile : TTS_PROFILES) {
        if (profile.type == type) {
            return profile;
        }
    }
    return TTS_PROFILES[1];
}

static tts_type type_from_string(const std::string &value) {
    if (value == "0.1" || contains_case_insensitive(value, "outetts-0.1") || contains_case_insensitive(value, "outetts 0.1")) {
        return OUTETTS_V0_1;
    }
    if (value == "0.2" || contains_case_insensitive(value, "outetts-0.2") || contains_case_insensitive(value, "outetts 0.2")) {
        return OUTETTS_V0_2;
    }
    if (value == "0.3" || contains_case_insensitive(value, "outetts-0.3") || contains_case_insensitive(value, "outetts 0.3")) {
        return OUTETTS_V0_3;
    }
    if (value == "1.0" || contains_case_insensitive(value, "outetts-1.0") ||
        contains_case_insensitive(value, "llama-outetts-1.0") || contains_case_insensitive(value, "outetts 1.0")) {
        return OUTETTS_V1_0;
    }
    if (contains_case_insensitive(value, "soprano")) {
        return SOPRANO_1_1_80M;
    }
    if (contains_case_insensitive(value, "neutts-nano") || contains_case_insensitive(value, "neutts_nano")) {
        return NEUTTS_NANO;
    }
    if (contains_case_insensitive(value, "neutts-air") || contains_case_insensitive(value, "neutts_air")) {
        return NEUTTS_AIR;
    }
    if (contains_case_insensitive(value, "csm-1b") || contains_case_insensitive(value, "csm_1b") ||
        contains_case_insensitive(value, "sesame/csm") || contains_case_insensitive(value, "sesame-csm")) {
        return CSM_1B;
    }
    if (contains_case_insensitive(value, "qwen3-tts") || contains_case_insensitive(value, "qwen3_tts")) {
        return QWEN3_TTS_0_6B;
    }
    if (contains_case_insensitive(value, "moss-tts-realtime") || contains_case_insensitive(value, "moss_tts_realtime")) {
        return MOSS_TTS_REALTIME;
    }
    if (contains_case_insensitive(value, "moss-ttsd") || contains_case_insensitive(value, "moss_ttsd")) {
        return MOSS_TTSD_V07;
    }
    if (contains_case_insensitive(value, "chatterbox-multilingual") || contains_case_insensitive(value, "chatterbox_multilingual") ||
        contains_case_insensitive(value, "mtl23ls")) {
        return CHATTERBOX_T3_MULTILINGUAL;
    }
    if (contains_case_insensitive(value, "chatterbox")) {
        return CHATTERBOX_T3;
    }
    return UNKNOWN;
}

static bool is_token_in_ranges(
    const std::vector<llama_rn_audio_code_range> & ranges,
    int n_codebook,
    llama_token token,
    int *codec_value
) {
    parsed_audio_token parsed = {-1, 0};
    if (token_code_from_ranges(ranges, n_codebook, token, parsed)) {
        if (codec_value) *codec_value = parsed.codec_value;
        return true;
    }
    return false;
}

static std::vector<std::string> split_string(const std::string &text, const std::string &delimiter) {
    std::vector<std::string> result;
    size_t start = 0;
    while (start <= text.size()) {
        const size_t end = text.find(delimiter, start);
        std::string word = text.substr(start, end == std::string::npos ? std::string::npos : end - start);
        if (!word.empty()) {
            result.push_back(word);
        }
        if (end == std::string::npos) {
            break;
        }
        start = end + delimiter.size();
    }
    return result;
}

static std::string grammar_word(const std::string &word) {
    if (word.empty()) {
        return "\"\"";
    }

    const bool ascii = std::all_of(word.begin(), word.end(), [](unsigned char c) {
        return c < 128 && c != '"' && c != '\\';
    });
    if (ascii) {
        return "\"" + word + "\"";
    }

    std::string chars;
    for (char c : word) {
        if (chars.find(c) == std::string::npos) {
            chars.push_back(c);
        }
    }
    return "[" + chars + "]+";
}

static std::string grammar_words_union(const std::vector<std::string> &words) {
    std::ostringstream out;
    for (size_t i = 0; i < words.size(); ++i) {
        if (i > 0) {
            out << " | ";
        }
        out << grammar_word(words[i]);
    }
    return out.str().empty() ? "\"\"" : out.str();
}

static std::string build_outetts_legacy_dynamic_grammar(const std::vector<std::string> &words, bool v0_3) {
    std::ostringstream sequence;
    if (words.empty()) {
        sequence << "WORD";
    } else {
        for (size_t i = 0; i < words.size(); ++i) {
            if (i > 0) {
                sequence << " audioBlock ";
            }
            sequence << grammar_word(words[i]);
        }
    }

    std::ostringstream grammar;
    grammar << "root ::= NL? " << sequence.str() << " audioBlock audioEnd NL? EOS?\n"
            << "audioBlock ::= TIME " << (v0_3 ? "" : "codeStart ") << "CODE* " << (v0_3 ? "space" : "codeEnd") << " NL?\n"
            << "EOS ::= \"<|im_end|>\"\n"
            << "audioEnd ::= \"<|audio_end|>\"\n"
            << "space ::= \"<|space|>\"\n"
            << "codeStart ::= \"<|code_start|>\"\n"
            << "codeEnd ::= \"<|code_end|>\"\n"
            << "WORD ::= " << grammar_words_union(words) << "\n"
            << "NL ::= [\\n]\n"
            << "TIME ::= \"<|t_\" DECIMAL \"|>\"\n"
            << "CODE ::= \"<|\" DIGITS \"|>\"\n"
            << "DIGITS ::= [0-9]+\n"
            << "DECIMAL ::= [0-9]+ \".\" [0-9]+\n";
    return grammar.str();
}

static std::string build_outetts_v1_dynamic_grammar(const std::vector<std::string> &words) {
    std::ostringstream grammar;
    grammar << "root ::= wordStart? leadWord wordBlock* audioEnd NL? EOS?\n"
            << "leadWord ::= WORD audioBlock\n"
            << "wordBlock ::= wordStart WORD audioBlock\n"
            << "audioBlock ::= codeBlock wordEnd NL?\n"
            << "codeBlock ::= features TIME energy spectralCentroid pitch CODE CODES*\n"
            << "EOS ::= \"<|im_end|>\"\n"
            << "audioEnd ::= \"<|audio_end|>\"\n"
            << "wordStart ::= \"<|word_start|>\"\n"
            << "wordEnd ::= \"<|word_end|>\"\n"
            << "features ::= \"<|features|>\"\n"
            << "energy ::= \"<|energy_\" DIGITS \"|>\"\n"
            << "spectralCentroid ::= \"<|spectral_centroid_\" DIGITS \"|>\"\n"
            << "pitch ::= \"<|pitch_\" DIGITS \"|>\"\n"
            << "WORD ::= " << grammar_words_union(words) << "\n"
            << "NL ::= [\\n]\n"
            << "TIME ::= \"<|t_\" DECIMAL \"|>\"\n"
            << "CODE ::= \"<|code|>\"\n"
            << "CODES ::= CODE1 CODE2\n"
            << "CODE1 ::= \"<|c1_\" DIGITS \"|>\"\n"
            << "CODE2 ::= \"<|c2_\" DIGITS \"|>\"\n"
            << "DIGITS ::= [0-9]+\n"
            << "DECIMAL ::= [0-9]+ \".\" [0-9]+\n";
    return grammar.str();
}

static std::string build_soprano_dynamic_grammar() {
    return R"(root ::= CODE* STOP?
CODE ::= "[" SOPRANO_ID "]"
STOP ::= "[STOP]"
SOPRANO_ID ::= [0-9] | [1-9] [0-9] | [1-9] [0-9] [0-9] | [1-7] [0-9] [0-9] [0-9]
)";
}

// NeuTTS Nano + Air both decode through NeuCodec, single codebook of 65536 entries.
// Constrain output to <|speech_0|>..<|speech_65535|> followed by the end marker
// so the model can't emit garbage tokens at the speech-generation stage.
static std::string build_neutts_dynamic_grammar() {
    return R"(root ::= speechToken+ end
end ::= "<|SPEECH_GENERATION_END|>"
speechToken ::= "<|speech_" SPEECH_ID "|>"
SPEECH_ID ::= [0-9]
        | [1-9] [0-9]
        | [1-9] [0-9] [0-9]
        | [1-9] [0-9] [0-9] [0-9]
        | [1-5] [0-9] [0-9] [0-9] [0-9]
        | "6" [0-4] [0-9] [0-9] [0-9]
        | "65" [0-4] [0-9] [0-9]
        | "655" [0-2] [0-9]
        | "6553" [0-5]
)";
}

static std::string build_neutts_codes_string(const std::vector<int32_t> &codes) {
    std::string out;
    out.reserve(codes.size() * 16);
    for (int32_t c : codes) {
        out += "<|speech_";
        out += std::to_string(c);
        out += "|>";
    }
    return out;
}

// NeuTTS prompt format (mirrors neuphonic/neutts neutts.py):
//   user: Convert the text to speech:<|TEXT_PROMPT_START|>{ref_phones} {input_phones}<|TEXT_PROMPT_END|>
//   assistant:<|SPEECH_GENERATION_START|>{ref_codes_as_speech_tokens}
//
// Reference voice (`ref_phones` + `ref_codes`) MUST be supplied via the
// speaker JSON — the JS wrapper resolves the built-in voice table or a
// caller-provided object before this is reached. Phonemization of both
// the reference text and the input text is also a JS-side concern via
// the `phonemizer` hook on `getFormattedAudioCompletion`.
static std::string build_neutts_prompt(json speaker, const std::string &text_to_speak) {
    if (!speaker.is_object() ||
        !speaker.contains("ref_codes") || !speaker["ref_codes"].is_array() ||
        speaker["ref_codes"].empty()) {
        LOG_ERROR("NeuTTS requires speaker.ref_codes (use the JS wrapper or pass them manually)");
        return "";
    }
    const std::string ref_phones = speaker.value("ref_phones", std::string());
    std::vector<int32_t> codes;
    codes.reserve(speaker["ref_codes"].size());
    for (const auto &c : speaker["ref_codes"]) {
        codes.push_back(c.get<int32_t>());
    }
    const std::string ref_codes_str = build_neutts_codes_string(codes);
    const std::string input_text = normalize_plain_tts_text(text_to_speak);

    std::string prompt;
    prompt.reserve(256 + ref_phones.size() + input_text.size() + ref_codes_str.size());
    prompt += "user: Convert the text to speech:<|TEXT_PROMPT_START|>";
    prompt += ref_phones;
    if (!ref_phones.empty() && !input_text.empty()) prompt += " ";
    prompt += input_text;
    prompt += "<|TEXT_PROMPT_END|>\nassistant:<|SPEECH_GENERATION_START|>";
    prompt += ref_codes_str;
    return prompt;
}

static std::string build_dynamic_grammar(const tts_model_profile &profile, const std::string &text_to_speak) {
    if (profile.prompt_kind == tts_prompt_kind::OUTETTS_LEGACY) {
        const std::string processed = process_text(text_to_speak, profile.type);
        return build_outetts_legacy_dynamic_grammar(split_string(processed, "<|text_sep|>"), false);
    }
    if (profile.prompt_kind == tts_prompt_kind::OUTETTS_V0_3) {
        const std::string processed = process_text(text_to_speak, profile.type);
        return build_outetts_legacy_dynamic_grammar(split_string(processed, "<|space|>"), true);
    }
    if (profile.prompt_kind == tts_prompt_kind::OUTETTS_V1_0) {
        const std::string processed = process_text(text_to_speak, profile.type);
        return build_outetts_v1_dynamic_grammar(split_string(processed, " "));
    }
    if (profile.prompt_kind == tts_prompt_kind::SOPRANO) {
        return build_soprano_dynamic_grammar();
    }
    if (profile.prompt_kind == tts_prompt_kind::NEUTTS) {
        return build_neutts_dynamic_grammar();
    }
    return "";
}

// Helper for the resolver: probe a vocab marker and fall back to the
// profile's hardcoded value when the marker isn't present (older variants).
static int32_t probe_or_fallback(llama_rn_context* main_ctx, const char * marker, int32_t fallback);

// Build the runtime code_range vector for a given prompt_kind by probing
// the main context's vocab.  Falls back to the static profile defaults
// when a marker can't be tokenised as a single special token (older /
// non-standard variants).  Profiles for codec_lm-AR / hidden-states / etc.
// pass through the static code_ranges unchanged.
static std::vector<llama_rn_audio_code_range> resolve_code_ranges_for_kind(
    llama_rn_context * main_ctx,
    tts_prompt_kind kind,
    const tts_model_profile::audio_token_config & defaults);

// TTS member functions
// Probe the vocab for a single-token marker. Returns the matched token id,
// or LLAMA_TOKEN_NULL when the marker doesn't tokenize to exactly one token
// (i.e. it isn't a real special / added token in this model's vocab).
static llama_token tts_probe_special_token(llama_rn_context* main_ctx, const std::string &text) {
    if (!main_ctx || !main_ctx->ctx || !main_ctx->model) return LLAMA_TOKEN_NULL;
    // Skip probing on no-vocab backbones (Chatterbox T3 placeholder, etc.)
    // — llama_vocab::tokenize hits GGML_ABORT when invoked on
    // LLAMA_VOCAB_TYPE_NONE, which would crash the process.
    const llama_vocab * vocab = llama_model_get_vocab(main_ctx->model);
    if (vocab == nullptr || llama_vocab_type(vocab) == LLAMA_VOCAB_TYPE_NONE) {
        return LLAMA_TOKEN_NULL;
    }
    auto toks = ::common_tokenize(main_ctx->ctx, text, /*add_special=*/false, /*parse_special=*/true);
    if (toks.size() != 1) return LLAMA_TOKEN_NULL;
    return toks[0];
}

static int32_t probe_or_fallback(llama_rn_context* main_ctx, const char * marker, int32_t fallback) {
    const llama_token t = tts_probe_special_token(main_ctx, marker);
    return (t != LLAMA_TOKEN_NULL) ? (int32_t) t : fallback;
}

// Resolves runtime audio-token start IDs by probing the loaded backbone's
// vocab.  OuteAI / Neuphonic ship the same model family on multiple
// backbones (e.g. OuteTTS-1.0 came on Llama-3.2-1B with `<|c1_0|>=128256`
// and on Qwen3-0.6B with `<|c1_0|>=151669`); hardcoding either offset in
// the profile table breaks the other.  Probing keeps the source of truth
// in the GGUF.  For codec_lm-AR / hidden-states / etc. there are no
// vocab-level audio ranges so this just passes through the profile's
// (often empty) defaults.
static std::vector<llama_rn_audio_code_range> resolve_code_ranges_for_kind(
    llama_rn_context * main_ctx,
    tts_prompt_kind kind,
    const tts_model_profile::audio_token_config & defaults) {

    std::vector<llama_rn_audio_code_range> out;
    if (defaults.code_ranges.empty()) {
        return out;
    }
    if (main_ctx == nullptr || main_ctx->ctx == nullptr) {
        for (const auto & r : defaults.code_ranges) {
            out.push_back({(int32_t) r.start, r.count, r.codebook_size});
        }
        return out;
    }

    auto push_one = [&](int32_t start, int count, int codebook_size) {
        out.push_back({start, count, codebook_size});
    };

    if (kind == tts_prompt_kind::OUTETTS_LEGACY || kind == tts_prompt_kind::OUTETTS_V0_3) {
        // OuteTTS V0.x audio codes are `<|0|>` .. `<|count-1|>`.
        const auto & d = defaults.code_ranges[0];
        push_one(probe_or_fallback(main_ctx, "<|0|>", (int32_t) d.start),
                 d.count, d.codebook_size);
        return out;
    }

    if (kind == tts_prompt_kind::OUTETTS_V1_0) {
        // V1.0 has two codebooks: `<|c1_0|>` and `<|c2_0|>`.
        const auto & d0 = defaults.code_ranges[0];
        const auto & d1 = defaults.code_ranges.size() > 1 ? defaults.code_ranges[1] : d0;
        push_one(probe_or_fallback(main_ctx, "<|c1_0|>", (int32_t) d0.start),
                 d0.count, d0.codebook_size);
        push_one(probe_or_fallback(main_ctx, "<|c2_0|>", (int32_t) d1.start),
                 d1.count, d1.codebook_size);
        return out;
    }

    if (kind == tts_prompt_kind::NEUTTS) {
        // NeuTTS speech vocab is `<|speech_0|>` .. `<|speech_65535|>`.
        const auto & d = defaults.code_ranges[0];
        push_one(probe_or_fallback(main_ctx, "<|speech_0|>", (int32_t) d.start),
                 d.count, d.codebook_size);
        return out;
    }

    // Other kinds (CSM / Qwen3-TTS / MOSS-TTS-Realtime / MOSS-TTSD /
    // Chatterbox / SOPRANO) don't have vocab-level audio ranges — keep
    // whatever the profile declared.
    for (const auto & r : defaults.code_ranges) {
        out.push_back({(int32_t) r.start, r.count, r.codebook_size});
    }
    return out;
}

tts_type llama_rn_context_tts::getTTSType(llama_rn_context* main_ctx, json speaker) {
    if (speaker.is_object() && speaker.contains("version")) {
        const std::string version = speaker["version"].get<std::string>();
        const tts_type speaker_type = type_from_string(version);
        if (speaker_type != UNKNOWN) {
            return speaker_type;
        }
        {
            LOG_ERROR("Unsupported speaker version '%s'\n", version.c_str());
        }
    }
    if (type != UNKNOWN) {
        return type;
    }

    // codec.gguf carries `codec.lm.*` metadata when an LM adaptor is bundled
    // (CSM, Qwen3-TTS, MOSS-TTSD, …).  If `codec_lm_create` succeeds, the
    // model is unambiguously codec_lm-driven; pick the family from the
    // adaptor's host_arch + n_codebook signature.  We keep the handle live
    // since generateAudioCodes will reuse it (`codec_lm_create` is mmap-only
    // pointer math — the cost is in the state machine, not the create call).
    if (codec_lm == nullptr && !codec_lm_probed && codec_model != nullptr) {
        codec_lm_probed = true;
        codec_lm = ::codec_lm_create(codec_model);
    }
    if (codec_lm != nullptr) {
        const ::codec_lm_info * info = ::codec_lm_get_info(codec_lm);
        const std::string host_arch = info && info->host_arch ? info->host_arch : "";
        const int n_cb = info ? info->n_codebook : 0;
        // Disambiguate codec_lm-driven models by host_arch + n_codebook.
        if (host_arch == "llama" && n_cb == 32) return CSM_1B;
        if (host_arch == "qwen3" && n_cb == 16) return QWEN3_TTS_0_6B;
        if (host_arch == "qwen3" && n_cb == 17) return MOSS_TTS_REALTIME;
        if (host_arch == "qwen3" && n_cb == 8)  return MOSS_TTSD_V07;
        // Chatterbox T3: Llama-520M backbone + parallel_heads_delay codec_lm
        // (single audio codebook).  English (`t3_cfg`) and multilingual
        // (`t3_mtl23ls_v3`) variants share the codec_lm shape — they only
        // differ in `lm.chatterbox.text_emb` dim (704 vs 2454).  We can't
        // tell them apart at this layer; route both to CHATTERBOX_T3 and
        // let the runtime (once the codec.cpp prefix-builder lands) read
        // the text-vocab size from codec_lm metadata if it needs to.
        if (host_arch == "llama" && n_cb == 1)  return CHATTERBOX_T3;
        // Other codec_lm families (LFM2-Audio, etc.) will land here once
        // their profiles are wired; fall through to vocab / name detection.
    }

    // Vocab-based detection runs next because gguf `general.name` is often
    // a meaningless HF revision hash (e.g. neuphonic/neutts-nano stamps it
    // with the snapshot id), so name / chat_template heuristics miss.
    if (main_ctx && main_ctx->ctx) {
        // NeuTTS speech vocab starts at <|speech_0|>. The id pinpoints the
        // family because each backbone has a different base vocab size.
        const llama_token tok = tts_probe_special_token(main_ctx, "<|speech_0|>");
        if (tok != LLAMA_TOKEN_NULL) {
            if (tok == 128262) return NEUTTS_NANO;
            if (tok == 151671) return NEUTTS_AIR;
        }
    }

    const char *chat_template = llama_model_chat_template(main_ctx->model, nullptr);
    const std::vector<std::string> detection_inputs = {
        chat_template ? std::string(chat_template) : "",
        main_ctx->model->name,
    };
    for (const auto &input : detection_inputs) {
        const tts_type detected = type_from_string(input);
        if (detected != UNKNOWN) {
            return detected;
        }
    }

    return OUTETTS_V0_2;
}

tts_type llama_rn_context_tts::detectTTSType(llama_rn_context* main_ctx) {
    return getTTSType(main_ctx, json::object());
}

llama_rn_tts_capabilities llama_rn_context_tts::getTTSCapabilities(llama_rn_context* main_ctx) {
    llama_rn_tts_capabilities cap = {};
    cap.type = (int) detectTTSType(main_ctx);
    if (cap.type < 0) {
        cap.requires_phonemes = false;
        return cap;
    }
    const tts_model_profile &profile = profile_for_type((tts_type) cap.type);
    switch (profile.prompt_kind) {
        case tts_prompt_kind::OUTETTS_LEGACY:
            cap.prompt_kind = "outetts_legacy";
            cap.family = "outetts";
            break;
        case tts_prompt_kind::OUTETTS_V0_3:
            cap.prompt_kind = "outetts_v0_3";
            cap.family = "outetts";
            break;
        case tts_prompt_kind::OUTETTS_V1_0:
            cap.prompt_kind = "outetts_v1_0";
            cap.family = "outetts";
            break;
        case tts_prompt_kind::SOPRANO:
            cap.prompt_kind = "soprano";
            cap.family = "soprano";
            break;
        case tts_prompt_kind::NEUTTS:
            cap.prompt_kind = "neutts";
            cap.family = "neutts";
            break;
        case tts_prompt_kind::CSM:
            cap.prompt_kind = "csm";
            cap.family = "csm";
            break;
        case tts_prompt_kind::QWEN3_TTS:
            cap.prompt_kind = "qwen3_tts";
            cap.family = "qwen3_tts";
            break;
        case tts_prompt_kind::MOSS_TTS_REALTIME:
            cap.prompt_kind = "moss_tts_realtime";
            cap.family = "moss_tts";
            break;
        case tts_prompt_kind::MOSS_TTSD:
            cap.prompt_kind = "moss_ttsd";
            cap.family = "moss_ttsd";
            break;
        case tts_prompt_kind::CHATTERBOX:
            cap.prompt_kind = (cap.type == CHATTERBOX_T3_MULTILINGUAL)
                ? "chatterbox_multilingual" : "chatterbox";
            cap.family = "chatterbox";
            break;
    }
    cap.requires_phonemes = (profile.prompt_kind == tts_prompt_kind::NEUTTS);
    cap.default_language = "en-us";
    return cap;
}

static std::string build_outetts_legacy_prompt(const tts_model_profile &profile, json speaker, const std::string &speaker_json_str, const std::string &text_to_speak) {
    // OuteTTS legacy / V0_3: speaker JSON must carry a structured `words`
    // array — JS wrappers resolve the built-in default voice into this
    // shape before reaching us. With no speaker we still emit a prompt
    // (audio_text / audio_data empty) so generation can proceed without
    // voice conditioning, but quality drops.
    std::string audio_text = "<|text_start|>";
    std::string audio_data = "<|audio_start|>\n";
    if (!speaker_json_str.empty()) {
        audio_text = audio_text_from_speaker(speaker, profile.type);
        audio_data = audio_data_from_speaker(speaker, profile.type);
    }

    return "<|im_start|>\n" + audio_text + process_text(text_to_speak, profile.type) + "<|text_end|>\n" + audio_data + "\n";
}

static std::string build_outetts_v1_prompt(json speaker, const std::string &speaker_json_str, const std::string &text_to_speak) {
    std::string text = process_text(text_to_speak, OUTETTS_V1_0);
    std::string audio_prefix;
    if (!speaker_json_str.empty()) {
        const std::string speaker_text = speaker.value("text", std::string());
        if (!speaker_text.empty()) {
            std::string separator = ". ";
            const char last = speaker_text.back();
            if (last == '.' || last == '?' || last == '!') {
                separator = " ";
            }
            text = process_text(speaker_text + separator + text_to_speak, OUTETTS_V1_0);
        }
        audio_prefix = outetts_v1_codes_from_speaker(speaker);
        if (!audio_prefix.empty()) {
            audio_prefix += "<|word_start|>";
        }
    }
    return "<|im_start|>\n<|text_start|>" + text + "<|text_end|>\n<|audio_start|>\n" + audio_prefix;
}

static std::string build_soprano_prompt(const std::string &text_to_speak) {
    return "[STOP][TEXT]" + process_text(text_to_speak, SOPRANO_1_1_80M) + "[START]";
}

// CSM prompt format (mirrors sesame/csm-1b's CsmProcessor chat template):
//
//   <|begin_of_text|>[<speaker>]<text><|end_of_text|>
//
// `speaker` is an integer id (CSM was trained with two speakers, "0" and
// "1"); the JS layer passes it via `speaker.id` in the speaker JSON.
// Defaults to 0 when absent.  The prep_csm converter mapped CSM's
// `embed_text_tokens` onto the backbone's standard `model.embed_tokens`,
// so feeding these text tokens via `b.token` (which generateAudioCodes
// does) drives the right text embedding without any out-of-band lookup.
static std::string build_csm_prompt(json speaker, const std::string &text_to_speak) {
    int speaker_id = 0;
    if (speaker.is_object() && speaker.contains("id")) {
        try {
            speaker_id = speaker["id"].get<int>();
        } catch (...) { /* fall back to 0 */ }
    }
    std::string out;
    out.reserve(text_to_speak.size() + 32);
    out += "<|begin_of_text|>[";
    out += std::to_string(speaker_id);
    out += "]";
    out += text_to_speak;
    out += "<|end_of_text|>";
    return out;
}

// Qwen3-TTS prompt format mirrors the talker's chat-style assembly:
// ChatML wrapper with the speech context starting at <|tts_start|>.  Voice-
// clone reference is opaque to the prompt builder — the JS layer drops
// ref_codes / spk_emb into the speaker JSON; generateAudioCodes consumes
// those when assembling the input embedding sequence.
static std::string build_qwen3_tts_prompt(json speaker, const std::string &text_to_speak) {
    (void) speaker;
    std::string out;
    out.reserve(text_to_speak.size() + 64);
    out += "<|im_start|>user\n";
    out += text_to_speak;
    out += "<|im_end|>\n<|im_start|>assistant\n";
    return out;
}

// MOSS-TTS-Realtime: Qwen3 talker chat-style prompt.  Realtime is a
// streaming model — the prompt is a plain user-message frame.
static std::string build_moss_tts_realtime_prompt(json speaker, const std::string &text_to_speak) {
    (void) speaker;
    std::string out;
    out.reserve(text_to_speak.size() + 64);
    out += "<|im_start|>user\n";
    out += text_to_speak;
    out += "<|im_end|>\n<|im_start|>assistant\n";
    return out;
}

// MOSS-TTSD dialogue prompt — caller passes text already in `[S1]...[S2]...`
// form (model-specific).  We pass it through.
static std::string build_moss_ttsd_prompt(json speaker, const std::string &text_to_speak) {
    (void) speaker;
    return text_to_speak;
}

// Chatterbox: runtime not yet wired (requires custom cond_enc prefix
// assembly from t3-extras.gguf — separate design discussion).  Return
// raw text so detection / capability surface works.
static std::string build_chatterbox_prompt(json speaker, const std::string &text_to_speak) {
    (void) speaker;
    return text_to_speak;
}

llama_rn_audio_completion_result llama_rn_context_tts::getFormattedAudioCompletion(llama_rn_context* main_ctx, const std::string &speaker_json_str, const std::string &text_to_speak) {
    json speaker = speaker_json_str.empty() ? json::object() : json::parse(speaker_json_str);
    const tts_type tts_type = getTTSType(main_ctx, speaker);
    if (tts_type == UNKNOWN) {
        LOG_ERROR("Unknown TTS version");
        return {"", "", false, ""};
    }

    const tts_model_profile &profile = profile_for_type(tts_type);
    const std::string grammar = build_dynamic_grammar(profile, text_to_speak);
    const bool embedding = profile.decode_kind == tts_decode_kind::HIDDEN_STATES;
    const std::string flow = (profile.decode_kind == tts_decode_kind::CODEC_LM_AR)
        ? "codec_lm_ar" : "tokens";
    switch (profile.prompt_kind) {
        case tts_prompt_kind::OUTETTS_LEGACY:
        case tts_prompt_kind::OUTETTS_V0_3:
            return {build_outetts_legacy_prompt(profile, speaker, speaker_json_str, text_to_speak), grammar, embedding, flow};
        case tts_prompt_kind::OUTETTS_V1_0:
            return {build_outetts_v1_prompt(speaker, speaker_json_str, text_to_speak), grammar, embedding, flow};
        case tts_prompt_kind::SOPRANO:
            return {build_soprano_prompt(text_to_speak), grammar, embedding, flow};
        case tts_prompt_kind::NEUTTS:
            return {build_neutts_prompt(speaker, text_to_speak), grammar, embedding, flow};
        case tts_prompt_kind::CSM:
            return {build_csm_prompt(speaker, text_to_speak), "", embedding, flow};
        case tts_prompt_kind::QWEN3_TTS:
            return {build_qwen3_tts_prompt(speaker, text_to_speak), "", embedding, flow};
        case tts_prompt_kind::MOSS_TTS_REALTIME:
            return {build_moss_tts_realtime_prompt(speaker, text_to_speak), "", embedding, flow};
        case tts_prompt_kind::MOSS_TTSD:
            return {build_moss_ttsd_prompt(speaker, text_to_speak), "", embedding, flow};
        case tts_prompt_kind::CHATTERBOX:
            return {build_chatterbox_prompt(speaker, text_to_speak), "", embedding, flow};
    }
    return {"", "", false, ""};
}

// ─────────────────────────────────────────────────────────────────────
// codec_lm AR driver — runs the backbone + codec_lm step-state-machine
// end-to-end and produces a (T × n_codebook) interleaved code matrix.
//
// The backbone is put into embeddings=true mode for the duration of this
// call so we can read its per-position hidden state via
// llama_get_embeddings_ith.  KV cache is wiped on entry and the
// embeddings flag is restored on exit so callers can reuse the context
// for normal completion afterwards.
// ─────────────────────────────────────────────────────────────────────

namespace {

// Greedy / temperature + top-k + top-p sampler over a raw logits buffer.
// Mirrors the semantics of tts.py's sample_logits — caller passes RNG
// state via `rng_state` so the same seed yields identical sequences.
int32_t sample_codec_logits(const float * logits, int32_t n,
                            float temperature, int32_t top_k, float top_p,
                            uint64_t * rng_state) {
    if (n <= 0) return 0;

    auto rand01 = [&]() {
        // xorshift64* — small, deterministic, no <random> dependency drift.
        uint64_t x = *rng_state ? *rng_state : 0x9E3779B97F4A7C15ULL;
        x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
        *rng_state = x;
        const uint64_t r = x * 2685821657736338717ULL;
        return (double)(r >> 11) / (double)(1ULL << 53);
    };

    // Greedy path: argmax over finite logits (treat NaN/+inf as -inf).
    if (temperature <= 0.0f) {
        int32_t best = 0;
        float best_v = -std::numeric_limits<float>::infinity();
        for (int32_t i = 0; i < n; ++i) {
            const float v = logits[i];
            if (std::isfinite(v) && v > best_v) { best = i; best_v = v; }
        }
        return best;
    }

    // Softmax with temperature, masking out non-finite logits.
    std::vector<double> p((size_t) n, 0.0);
    double max_l = -std::numeric_limits<double>::infinity();
    for (int32_t i = 0; i < n; ++i) {
        if (std::isfinite(logits[i]) && logits[i] > max_l) max_l = (double) logits[i];
    }
    if (!std::isfinite(max_l)) {
        return 0;
    }
    const double inv_t = 1.0 / std::max(temperature, 1e-6f);
    double sum = 0.0;
    for (int32_t i = 0; i < n; ++i) {
        if (!std::isfinite(logits[i])) { p[i] = 0.0; continue; }
        const double v = std::exp(((double) logits[i] - max_l) * inv_t);
        p[i] = v; sum += v;
    }
    if (sum <= 0.0) return 0;
    for (auto &v : p) v /= sum;

    // top-k mask
    if (top_k > 0 && top_k < n) {
        std::vector<int32_t> idx((size_t) n);
        for (int32_t i = 0; i < n; ++i) idx[i] = i;
        std::partial_sort(idx.begin(), idx.begin() + top_k, idx.end(),
                          [&](int32_t a, int32_t b){ return p[a] > p[b]; });
        std::vector<double> kept((size_t) n, 0.0);
        double s = 0.0;
        for (int32_t i = 0; i < top_k; ++i) { kept[idx[i]] = p[idx[i]]; s += p[idx[i]]; }
        if (s <= 0.0) return 0;
        for (auto &v : kept) v /= s;
        p.swap(kept);
    }

    // top-p (nucleus) mask
    if (top_p > 0.0f && top_p < 1.0f) {
        std::vector<int32_t> idx((size_t) n);
        for (int32_t i = 0; i < n; ++i) idx[i] = i;
        std::sort(idx.begin(), idx.end(),
                  [&](int32_t a, int32_t b){ return p[a] > p[b]; });
        std::vector<double> kept((size_t) n, 0.0);
        double cdf = 0.0;
        for (int32_t i = 0; i < n; ++i) {
            const int32_t k = idx[i];
            kept[k] = p[k];
            cdf += p[k];
            if (cdf >= (double) top_p) break;
        }
        double s = 0.0;
        for (auto v : kept) s += v;
        if (s <= 0.0) return 0;
        for (auto &v : kept) v /= s;
        p.swap(kept);
    }

    const double u = rand01();
    double acc = 0.0;
    for (int32_t i = 0; i < n; ++i) {
        acc += p[i];
        if (u <= acc) return i;
    }
    return n - 1;
}

// Tiny RAII wrapper for llama_batch.  llama_batch_init / llama_batch_free
// are the canonical lifetime API.
struct scoped_llama_batch {
    llama_batch b;
    bool owned = false;
    scoped_llama_batch(int32_t n_tokens, int32_t embd_dim, int32_t n_seq_max) {
        b = llama_batch_init(n_tokens, embd_dim, n_seq_max);
        owned = true;
    }
    ~scoped_llama_batch() { if (owned) llama_batch_free(b); }
    scoped_llama_batch(const scoped_llama_batch &) = delete;
    scoped_llama_batch & operator=(const scoped_llama_batch &) = delete;
};

// Saves the embeddings flag on construction and restores it on destruction
// so a thrown exception or early return doesn't strand the context in
// embeddings mode.
struct scoped_embeddings_flag {
    llama_context * ctx;
    bool prev;
    scoped_embeddings_flag(llama_context * c, bool desired) : ctx(c) {
        // llama.cpp doesn't expose a getter — the next-best thing is to
        // assume "off" pre-TTS and unconditionally restore to that.  The
        // main_ctx is created with embeddings=false for chat completion.
        prev = false;
        llama_set_embeddings(ctx, desired);
    }
    ~scoped_embeddings_flag() { llama_set_embeddings(ctx, prev); }
};

} // namespace

llama_rn_audio_codes_result llama_rn_context_tts::generateAudioCodes(
    llama_rn_context * main_ctx,
    const llama_rn_audio_codes_options & opts,
    const llama_rn_audio_codes_progress_cb & on_frame) {

    llama_rn_audio_codes_result result;

    if (main_ctx == nullptr || main_ctx->ctx == nullptr || main_ctx->model == nullptr) {
        LOG_ERROR("generateAudioCodes: main context not initialized");
        return result;
    }
    if (codec_model == nullptr) {
        LOG_ERROR("generateAudioCodes: codec model not loaded");
        return result;
    }

    // Lazy codec_lm + state init.  codec_lm is already created during
    // getTTSType() if the codec.gguf carries `lm.*` metadata; create here
    // too in case the caller skipped capability detection.
    if (codec_lm == nullptr && !codec_lm_probed) {
        codec_lm_probed = true;
        codec_lm = ::codec_lm_create(codec_model);
    }
    if (codec_lm == nullptr) {
        const char * err = ::codec_lm_get_create_error();
        LOG_ERROR("generateAudioCodes: codec.gguf has no codec_lm adaptor (%s)",
                  err && *err ? err : "no `lm.*` section in codec.gguf");
        return result;
    }
    if (codec_lm_state == nullptr) {
        codec_lm_state = ::codec_lm_state_new(codec_lm);
        if (codec_lm_state == nullptr) {
            LOG_ERROR("generateAudioCodes: codec_lm_state_new failed");
            return result;
        }
    } else {
        ::codec_lm_state_reset(codec_lm_state);
    }

    const ::codec_lm_info * info = ::codec_lm_get_info(codec_lm);
    const int n_cb     = info->n_codebook;
    const int hidden   = info->hidden_dim;
    const int audio_ed = info->audio_embed_dim;
    const int compose_ed = info->compose_audio_embed_dim > 0
        ? info->compose_audio_embed_dim : audio_ed;

    const int model_n_embd = llama_model_n_embd(main_ctx->model);
    if (model_n_embd != hidden) {
        LOG_ERROR("generateAudioCodes: backbone n_embd=%d != codec_lm hidden=%d",
                  model_n_embd, hidden);
        return result;
    }

    // Continuous-latent branch (BlueMagpie-TTS / VoxCPM, codec_lm kind
    // continuous_latent_cfm).  No per-step codebook sampling — the backbone
    // emits a hidden state each step; the codec_lm's tslm_adapter + FSQ +
    // RALM + LocDiT CFM produce one latent patch which we accumulate and
    // then feed to codec_decode_quantized_representation at the end.  We
    // delegate the whole step (step_generate + step_feedback_embd) to
    // codec_common::audio_lm_observe_hidden so the host owns only the AR
    // control flow (prompt feed, per-step llama_decode, embedding readout,
    // stop check).
    if (info->is_continuous) {
        result.is_continuous = true;
        return generateAudioCodesContinuous(main_ctx, opts, on_frame, info);
    }

    if (compose_ed != hidden) {
        LOG_ERROR("generateAudioCodes: compose_audio_embed_dim=%d != backbone hidden=%d",
                  compose_ed, hidden);
        return result;
    }

    // Tokenize prompt.  CSM's prompt embeds the BOS / EOS markers verbatim
    // (`<|begin_of_text|>...<|end_of_text|>`), so add_special=false here.
    const std::vector<llama_token> prompt_tokens = ::common_tokenize(
        main_ctx->ctx, opts.prompt, /*add_special=*/false, /*parse_special=*/true);
    if (prompt_tokens.empty()) {
        LOG_ERROR("generateAudioCodes: prompt tokenizes to zero tokens");
        return result;
    }

    // Reset state, switch into embeddings mode, wipe KV.
    audio_tokens.clear();
    pending_codebook1 = -1;
    scoped_embeddings_flag embd_guard(main_ctx->ctx, true);
    ::llama_memory_clear(::llama_get_memory(main_ctx->ctx), true);

    // Feed prompt as tokens.  prep_csm mapped CSM's `embed_text_tokens`
    // onto the standard backbone `model.embed_tokens`, so b.token feeding
    // drives the right text embedding without out-of-band lookup.
    const int32_t n_ctx_max = (int32_t) llama_n_ctx(main_ctx->ctx);
    if ((int32_t) prompt_tokens.size() >= n_ctx_max) {
        LOG_ERROR("generateAudioCodes: prompt (%zu) >= n_ctx (%d)",
                  prompt_tokens.size(), n_ctx_max);
        return result;
    }

    int32_t pos = 0;

    // Optional speaker-conditioning prefix (output of
    // `codec_lm_speaker_encode` for the loaded voice-clone codec).  Fed
    // first via b.embd so the AR loop sees it ahead of the text prompt.
    // Caller is responsible for matching speaker_emb_hidden_dim to the
    // backbone n_embd (we validate before feeding to avoid corrupting
    // the KV cache with mismatched embedding rows).
    if (!opts.speaker_emb_prefix.empty() && opts.speaker_emb_rows > 0 &&
        opts.speaker_emb_hidden_dim > 0) {
        if (opts.speaker_emb_hidden_dim != hidden) {
            LOG_ERROR("generateAudioCodes: speaker_emb_hidden_dim=%d != backbone n_embd=%d — skipping prefix",
                      opts.speaker_emb_hidden_dim, hidden);
        } else if ((int32_t) opts.speaker_emb_prefix.size() !=
                   opts.speaker_emb_rows * opts.speaker_emb_hidden_dim) {
            LOG_ERROR("generateAudioCodes: speaker_emb_prefix length %zu != rows(%d) × hidden(%d) — skipping prefix",
                      opts.speaker_emb_prefix.size(),
                      opts.speaker_emb_rows, opts.speaker_emb_hidden_dim);
        } else if ((int32_t) (pos + opts.speaker_emb_rows + prompt_tokens.size()) >= n_ctx_max) {
            LOG_ERROR("generateAudioCodes: speaker prefix (%d) + prompt (%zu) >= n_ctx (%d) — skipping prefix",
                      opts.speaker_emb_rows, prompt_tokens.size(), n_ctx_max);
        } else {
            scoped_llama_batch sb(opts.speaker_emb_rows, hidden, 1);
            llama_batch & b = sb.b;
            b.n_tokens = opts.speaker_emb_rows;
            std::memcpy(b.embd, opts.speaker_emb_prefix.data(),
                        (size_t) opts.speaker_emb_rows * (size_t) hidden * sizeof(float));
            for (int32_t i = 0; i < b.n_tokens; ++i) {
                b.pos[i]       = pos + i;
                b.n_seq_id[i]  = 1;
                b.seq_id[i][0] = 0;
                b.logits[i]    = 0; // hidden state read after the prompt
            }
            b.token = nullptr;
            if (::llama_decode(main_ctx->ctx, b) != 0) {
                LOG_ERROR("generateAudioCodes: llama_decode (speaker prefix) failed");
                return result;
            }
            pos += b.n_tokens;
        }
    }

    {
        scoped_llama_batch pb((int32_t) prompt_tokens.size(), 0, 1);
        llama_batch & b = pb.b;
        b.n_tokens = (int32_t) prompt_tokens.size();
        for (int32_t i = 0; i < b.n_tokens; ++i) {
            b.token[i]       = prompt_tokens[(size_t) i];
            b.pos[i]         = pos + i;
            b.n_seq_id[i]    = 1;
            b.seq_id[i][0]   = 0;
            b.logits[i]      = (i == b.n_tokens - 1) ? 1 : 0;
        }
        if (::llama_decode(main_ctx->ctx, b) != 0) {
            LOG_ERROR("generateAudioCodes: llama_decode (prompt) failed");
            return result;
        }
        pos += b.n_tokens;
    }

    auto read_hidden = [&]() -> const float * {
        // The last (and only) row with logits[i]==1 is at index 0 in the
        // embeddings output buffer.
        return ::llama_get_embeddings_ith(main_ctx->ctx, 0);
    };

    const float * h_in = read_hidden();
    if (h_in == nullptr) {
        LOG_ERROR("generateAudioCodes: llama_get_embeddings_ith returned NULL");
        return result;
    }

    // Single reusable per-step batch (1 embedding row).
    scoped_llama_batch step_batch(1, hidden, 1);

    std::vector<float> next_embd((size_t) hidden, 0.0f);
    std::vector<int32_t> codes((size_t) n_cb, 0);

    uint64_t rng = opts.seed ? (uint64_t) opts.seed : 0xC0DEC1ABULL;

    const int max_frames = std::max(opts.max_frames, 1);
    result.codes.reserve((size_t) max_frames * (size_t) n_cb);

    // For text-modality codec_lm models (MOSS-TTS-Realtime: cb-0 is a token
    // sampled from the backbone's lm_head over the Qwen3 text vocab), pull
    // the backbone's logits at the latest output position and stash the
    // sampled text token via codec_lm_state_set_text_context BEFORE
    // step_begin so the depth decoder's pos-0 input is wired correctly.
    // codec_lm's step_logits(0) for these models then returns the same
    // logits (tied to the backbone's text head), so we sample again into
    // codes[0] — harmless and keeps the per-step loop uniform.
    const tts_model_profile &gen_profile = profile_for_type(type);
    const bool text_modality_cb0 = (gen_profile.audio_codebook_offset > 0);

    // Per-stage perf accumulators (µs).  Logged every 25 frames so we can
    // tell whether backbone decode or codec_lm depth is the bottleneck on
    // codec_lm-AR models like CSM (32-cb residual depth) without spamming
    // logcat per-step.
    auto now_us = []() -> int64_t {
        const auto t = std::chrono::steady_clock::now().time_since_epoch();
        return std::chrono::duration_cast<std::chrono::microseconds>(t).count();
    };
    int64_t us_step_begin = 0, us_depth_sample = 0, us_step_finish = 0;
    int64_t us_compose = 0, us_backbone = 0, us_read_hidden = 0;
    const int64_t loop_t0 = now_us();

    int step = 0;
    for (; step < max_frames; ++step) {
        if (pos + 1 >= n_ctx_max) {
            LOG_WARNING("generateAudioCodes: context full at frame %d", step);
            break;
        }

        if (text_modality_cb0) {
            const float * backbone_logits = ::llama_get_logits_ith(main_ctx->ctx, 0);
            if (backbone_logits != nullptr) {
                const int32_t n_vocab = (int32_t) llama_vocab_n_tokens(
                    llama_model_get_vocab(main_ctx->model));
                const int32_t text_tok = sample_codec_logits(
                    backbone_logits, n_vocab, opts.temperature,
                    opts.top_k, opts.top_p, &rng);
                ::codec_lm_state_set_text_context(codec_lm_state, text_tok);
            }
        }

        int64_t t = now_us();
        if (::codec_lm_step_begin(codec_lm_state, h_in) != CODEC_STATUS_SUCCESS) {
            const char * err = ::codec_lm_state_get_last_error(codec_lm_state);
            LOG_ERROR("generateAudioCodes: codec_lm_step_begin failed: %s",
                      err && *err ? err : "(no error message)");
            break;
        }
        us_step_begin += now_us() - t;

        t = now_us();
        bool step_ok = true;
        for (int cb = 0; cb < n_cb; ++cb) {
            int32_t cb_idx = -1, vocab = 0;
            const float * logits = ::codec_lm_step_logits(codec_lm_state, &cb_idx, &vocab);
            if (logits == nullptr || vocab <= 0) {
                LOG_ERROR("generateAudioCodes: step_logits failed at cb=%d", cb);
                step_ok = false; break;
            }
            const int32_t code = sample_codec_logits(logits, vocab,
                opts.temperature, opts.top_k, opts.top_p, &rng);
            if (::codec_lm_step_push_code(codec_lm_state, code) != CODEC_STATUS_SUCCESS) {
                LOG_ERROR("generateAudioCodes: step_push_code failed at cb=%d code=%d",
                          cb, code);
                step_ok = false; break;
            }
        }
        us_depth_sample += now_us() - t;
        if (!step_ok) break;

        t = now_us();
        if (::codec_lm_step_finish(codec_lm_state, codes.data()) != CODEC_STATUS_SUCCESS) {
            LOG_ERROR("generateAudioCodes: codec_lm_step_finish failed");
            break;
        }
        us_step_finish += now_us() - t;

        // CSM EOS heuristic: training-time audio-EOS frame has codes all
        // zero; trip on cb-0==0 after step 0 to avoid stopping on the
        // initial frame which is also frequently zero for cb-0.
        const bool csm_eos = (type == CSM_1B) && step > 0 && codes[0] == 0;
        if (csm_eos) {
            result.stopped_on_eos = true;
        } else {
            // Append (T, n_cb) interleaved.
            const size_t prev_size = result.codes.size();
            result.codes.resize(prev_size + (size_t) n_cb);
            for (int cb = 0; cb < n_cb; ++cb) {
                result.codes[prev_size + (size_t) cb] = codes[(size_t) cb];
            }
            audio_tokens.insert(audio_tokens.end(), codes.begin(), codes.end());
        }

        if (on_frame) {
            if (!on_frame(step, codes)) {
                result.aborted = true;
                break;
            }
        }
        if (csm_eos) break;

        // Compose next-step embedding from sampled codes.
        int64_t t2 = now_us();
        if (::codec_lm_compose_audio_embd(codec_lm, codes.data(), next_embd.data())
                != CODEC_STATUS_SUCCESS) {
            const char * err = ::codec_lm_get_last_error(codec_lm);
            LOG_ERROR("generateAudioCodes: compose_audio_embd failed: %s",
                      err && *err ? err : "(no error message)");
            break;
        }
        us_compose += now_us() - t2;

        // Feed the composed audio embedding as the next position.
        t2 = now_us();
        {
            llama_batch & b = step_batch.b;
            b.n_tokens = 1;
            std::memcpy(b.embd, next_embd.data(), (size_t) hidden * sizeof(float));
            b.pos[0]       = pos;
            b.n_seq_id[0]  = 1;
            b.seq_id[0][0] = 0;
            b.logits[0]    = 1;
            // batch.token must be NULL when embd is set; llama_batch_init
            // leaves token allocated, so blank it out explicitly.
            b.token = nullptr;
            if (::llama_decode(main_ctx->ctx, b) != 0) {
                LOG_ERROR("generateAudioCodes: llama_decode (step %d) failed", step);
                break;
            }
            pos += 1;
        }
        us_backbone += now_us() - t2;

        t2 = now_us();
        h_in = read_hidden();
        if (h_in == nullptr) {
            LOG_ERROR("generateAudioCodes: hidden state read returned NULL at step %d", step);
            break;
        }
        us_read_hidden += now_us() - t2;

        if ((step + 1) % 25 == 0) {
            const int64_t loop_us = now_us() - loop_t0;
            const int n = step + 1;
            LOG_INFO("generateAudioCodes perf @ step %d: "
                     "wall=%.2fs (%.1f frames/s)  per-step avg ms — "
                     "backbone=%.2f  depth(begin+sample+finish)=%.2f  compose=%.3f  read_h=%.3f",
                     step,
                     (double) loop_us / 1e6,
                     (double) n / ((double) loop_us / 1e6),
                     (double) us_backbone / 1000.0 / n,
                     (double) (us_step_begin + us_depth_sample + us_step_finish) / 1000.0 / n,
                     (double) us_compose / 1000.0 / n,
                     (double) us_read_hidden / 1000.0 / n);
        }
    }

    if (step > 0) {
        const int64_t loop_us = now_us() - loop_t0;
        LOG_INFO("generateAudioCodes done: %d frames in %.2fs (%.1f frames/s)  "
                 "totals ms — backbone=%.0f  step_begin=%.0f  depth_sample=%.0f  "
                 "step_finish=%.0f  compose=%.0f  read_h=%.0f",
                 step, (double) loop_us / 1e6,
                 (double) step / ((double) loop_us / 1e6),
                 (double) us_backbone / 1000.0,
                 (double) us_step_begin / 1000.0,
                 (double) us_depth_sample / 1000.0,
                 (double) us_step_finish / 1000.0,
                 (double) us_compose / 1000.0,
                 (double) us_read_hidden / 1000.0);
    }

    result.n_codebook = n_cb;
    result.n_frames = (int) (result.codes.size() / (size_t) std::max(n_cb, 1));
    return result;
}

// ─────────────────────────────────────────────────────────────────────
// Continuous-latent codec_lm AR driver (BlueMagpie-TTS / VoxCPM).
//
// Each backbone step emits a hidden state (no codebook token); we hand
// it to `codec_lm_step_generate` which internally runs
// tslm_adapter → FSQ → RALM → LocDiT CFM diffusion and produces one
// latent patch (patch_size × latent_dim).  We accumulate patches, use
// `codec_lm_step_feedback_embd` (LocEnc(patch) projected into backbone
// hidden space) as the next `b.embd`, and stop when the codec_lm's stop
// head fires.  At end of loop, `codec_decode_quantized_representation`
// turns the (D × T) latent grid into PCM.
//
// Mirrors codec_common::audio_lm_observe_hidden semantics but calls the
// codec.cpp APIs directly so it can share the already-loaded codec_lm
// / codec_lm_state / codec_ctx owned by this llama_rn_context_tts (the
// codec_common wrapper opens its own codec context from a GGUF path,
// which we'd rather not duplicate).
// ─────────────────────────────────────────────────────────────────────
llama_rn_audio_codes_result llama_rn_context_tts::generateAudioCodesContinuous(
    llama_rn_context * main_ctx,
    const llama_rn_audio_codes_options & opts,
    const llama_rn_audio_codes_progress_cb & on_frame,
    const ::codec_lm_info * info) {

    llama_rn_audio_codes_result result;
    result.is_continuous = true;

    if (info == nullptr) {
        LOG_ERROR("generateAudioCodesContinuous: null codec_lm_info");
        return result;
    }
    const int hidden     = info->hidden_dim;
    const int patch_size = info->patch_size;
    const int latent_dim = info->latent_dim;
    if (patch_size <= 0 || latent_dim <= 0) {
        LOG_ERROR("generateAudioCodesContinuous: bad shape (patch=%d, latent=%d)",
                  patch_size, latent_dim);
        return result;
    }

    // Tokenize prompt (same flags as the codebook path — CSM-style prompts
    // may embed BOS/EOS markers verbatim).
    const std::vector<llama_token> prompt_tokens = ::common_tokenize(
        main_ctx->ctx, opts.prompt, /*add_special=*/false, /*parse_special=*/true);
    if (prompt_tokens.empty()) {
        LOG_ERROR("generateAudioCodesContinuous: prompt tokenizes to zero tokens");
        return result;
    }
    const int32_t n_ctx_max = (int32_t) llama_n_ctx(main_ctx->ctx);
    if ((int32_t) prompt_tokens.size() >= n_ctx_max) {
        LOG_ERROR("generateAudioCodesContinuous: prompt (%zu) >= n_ctx (%d)",
                  prompt_tokens.size(), n_ctx_max);
        return result;
    }

    // Reset per-sequence codec_lm state so step_generate starts fresh.
    if (codec_lm_state != nullptr) {
        ::codec_lm_state_reset(codec_lm_state);
    }

    audio_tokens.clear();
    pending_codebook1 = -1;
    scoped_embeddings_flag embd_guard(main_ctx->ctx, true);
    ::llama_memory_clear(::llama_get_memory(main_ctx->ctx), true);

    int32_t pos = 0;

    // Optional speaker prefix (same shape as the codebook path).
    if (!opts.speaker_emb_prefix.empty() && opts.speaker_emb_rows > 0 &&
        opts.speaker_emb_hidden_dim > 0) {
        if (opts.speaker_emb_hidden_dim != hidden) {
            LOG_ERROR("generateAudioCodesContinuous: speaker_emb_hidden_dim=%d != backbone n_embd=%d — skipping prefix",
                      opts.speaker_emb_hidden_dim, hidden);
        } else if ((int32_t) opts.speaker_emb_prefix.size() !=
                   opts.speaker_emb_rows * opts.speaker_emb_hidden_dim) {
            LOG_ERROR("generateAudioCodesContinuous: speaker_emb_prefix length mismatch — skipping prefix");
        } else if ((int32_t) (pos + opts.speaker_emb_rows + prompt_tokens.size()) >= n_ctx_max) {
            LOG_ERROR("generateAudioCodesContinuous: speaker prefix + prompt >= n_ctx — skipping prefix");
        } else {
            scoped_llama_batch sb(opts.speaker_emb_rows, hidden, 1);
            llama_batch & b = sb.b;
            b.n_tokens = opts.speaker_emb_rows;
            std::memcpy(b.embd, opts.speaker_emb_prefix.data(),
                        (size_t) opts.speaker_emb_rows * (size_t) hidden * sizeof(float));
            for (int32_t i = 0; i < b.n_tokens; ++i) {
                b.pos[i]       = pos + i;
                b.n_seq_id[i]  = 1;
                b.seq_id[i][0] = 0;
                b.logits[i]    = 0;
            }
            b.token = nullptr;
            if (::llama_decode(main_ctx->ctx, b) != 0) {
                LOG_ERROR("generateAudioCodesContinuous: llama_decode (speaker prefix) failed");
                return result;
            }
            pos += b.n_tokens;
        }
    }

    // Feed prompt as tokens; last row emits embeddings.
    {
        scoped_llama_batch pb((int32_t) prompt_tokens.size(), 0, 1);
        llama_batch & b = pb.b;
        b.n_tokens = (int32_t) prompt_tokens.size();
        for (int32_t i = 0; i < b.n_tokens; ++i) {
            b.token[i]       = prompt_tokens[(size_t) i];
            b.pos[i]         = pos + i;
            b.n_seq_id[i]    = 1;
            b.seq_id[i][0]   = 0;
            b.logits[i]      = (i == b.n_tokens - 1) ? 1 : 0;
        }
        if (::llama_decode(main_ctx->ctx, b) != 0) {
            LOG_ERROR("generateAudioCodesContinuous: llama_decode (prompt) failed");
            return result;
        }
        pos += b.n_tokens;
    }

    auto read_hidden = [&]() -> const float * {
        return ::llama_get_embeddings_ith(main_ctx->ctx, 0);
    };

    const float * h_in = read_hidden();
    if (h_in == nullptr) {
        LOG_ERROR("generateAudioCodesContinuous: llama_get_embeddings_ith returned NULL");
        return result;
    }

    // Per-step buffers.
    scoped_llama_batch step_batch(1, hidden, 1);
    std::vector<float> patch((size_t) patch_size * (size_t) latent_dim, 0.0f);
    std::vector<float> next_embd((size_t) hidden, 0.0f);

    // Accumulator laid out frame-major [T, latent_dim] while we go —
    // transposed to channel-major [latent_dim, T] before decode.
    std::vector<float> latents_tf;
    int total_frames = 0;

    // Diffusion / CFG hyper-params.  Codec_common defaults are cfg=2.0,
    // n_timesteps=10 (BlueMagpie training-time).  Expose via opts later if
    // callers need to tweak; for now use those.
    const float   cfg_value   = 2.0f;
    const int32_t n_timesteps = 10;

    const int max_frames = std::max(opts.max_frames, 1);

    int step = 0;
    for (; step < max_frames; ++step) {
        if (pos + 1 >= n_ctx_max) {
            LOG_WARNING("generateAudioCodesContinuous: context full at step %d", step);
            break;
        }

        int32_t stop = 0;
        const enum codec_status rc = ::codec_lm_step_generate(
            codec_lm_state, h_in, cfg_value, n_timesteps,
            /*noise=*/nullptr, patch.data(), &stop);
        if (rc != CODEC_STATUS_SUCCESS) {
            const char * err = ::codec_lm_state_get_last_error(codec_lm_state);
            LOG_ERROR("generateAudioCodesContinuous: step_generate failed: %s",
                      err && *err ? err : "(no error message)");
            break;
        }

        // Accumulate patch (frame-major).
        latents_tf.insert(latents_tf.end(), patch.begin(), patch.end());
        total_frames += patch_size;

        if (on_frame) {
            // Signal progress to JS via an empty codes payload — the
            // continuous path has no per-step tokens, so callers just
            // treat `step` as a frame counter.  Return false to abort.
            std::vector<int32_t> empty_codes;
            if (!on_frame(step, empty_codes)) {
                result.aborted = true;
                break;
            }
        }

        if (stop) {
            result.stopped_on_eos = true;
            break;
        }

        // Feedback embedding: LocEnc(last patch) → backbone hidden.
        if (::codec_lm_step_feedback_embd(codec_lm_state, next_embd.data())
                != CODEC_STATUS_SUCCESS) {
            LOG_ERROR("generateAudioCodesContinuous: step_feedback_embd failed");
            break;
        }

        // Feed as next backbone position.
        {
            llama_batch & b = step_batch.b;
            b.n_tokens = 1;
            std::memcpy(b.embd, next_embd.data(), (size_t) hidden * sizeof(float));
            b.pos[0]       = pos;
            b.n_seq_id[0]  = 1;
            b.seq_id[0][0] = 0;
            b.logits[0]    = 1;
            b.token        = nullptr;
            if (::llama_decode(main_ctx->ctx, b) != 0) {
                LOG_ERROR("generateAudioCodesContinuous: llama_decode (step %d) failed", step);
                break;
            }
            pos += 1;
        }

        h_in = read_hidden();
        if (h_in == nullptr) {
            LOG_ERROR("generateAudioCodesContinuous: hidden read NULL at step %d", step);
            break;
        }
    }

    if (total_frames <= 0) {
        LOG_ERROR("generateAudioCodesContinuous: no latents accumulated");
        return result;
    }

    // Transpose [T, D] → [D, T] for codec_decode_quantized_representation.
    std::vector<float> latents_dt((size_t) total_frames * (size_t) latent_dim);
    for (int32_t t = 0; t < total_frames; ++t) {
        for (int32_t d = 0; d < latent_dim; ++d) {
            latents_dt[(size_t) d * (size_t) total_frames + (size_t) t] =
                latents_tf[(size_t) t * (size_t) latent_dim + (size_t) d];
        }
    }

    struct codec_decode_params dp = codec_decode_default_params();
    if (main_ctx->params.cpuparams.n_threads > 0) {
        dp.n_threads = main_ctx->params.cpuparams.n_threads;
    }
    struct codec_pcm_buffer pcm = {};
    const enum codec_status drc = ::codec_decode_quantized_representation(
        codec_ctx, latents_dt.data(), latent_dim, total_frames, &pcm, dp);
    if (drc != CODEC_STATUS_SUCCESS) {
        const char * err = ::codec_get_last_error(codec_ctx);
        LOG_ERROR("generateAudioCodesContinuous: decode_latents failed: %s",
                  err && *err ? err : "(no error message)");
        return result;
    }

    result.pcm.assign(pcm.data, pcm.data + (size_t) pcm.n_samples * (size_t) pcm.n_channels);
    result.sample_rate = pcm.sample_rate;
    result.n_frames = total_frames;
    result.n_codebook = 0;
    ::codec_pcm_buffer_free(&pcm);

    LOG_INFO("generateAudioCodesContinuous done: %d steps, %d latent frames, %d PCM samples @ %d Hz",
             step, total_frames, (int) result.pcm.size(), result.sample_rate);
    return result;
}

// Returns the cached resolved code ranges for this context, populating
// them via vocab probing on first call.  Idempotent.
static const std::vector<llama_rn_audio_code_range> & ensure_resolved_ranges(
    llama_rn_context_tts * tts,
    llama_rn_context * main_ctx,
    const tts_model_profile & profile) {

    if (!tts->resolved_ranges_ready) {
        tts->resolved_code_ranges = resolve_code_ranges_for_kind(
            main_ctx, profile.prompt_kind, profile.audio);
        tts->resolved_ranges_ready = true;
    }
    return tts->resolved_code_ranges;
}

bool llama_rn_context_tts::isAudioToken(llama_rn_context* main_ctx, llama_token token, const std::string &token_text) {
    (void) token_text;
    if (token < 0) {
        return false;
    }

    const tts_type tts_type = getTTSType(main_ctx);
    const tts_model_profile &profile = profile_for_type(tts_type);
    const auto & ranges = ensure_resolved_ranges(this, main_ctx, profile);

    return is_token_in_ranges(ranges, profile.audio.n_codebook, token, nullptr);
}

bool llama_rn_context_tts::tryAddAudioToken(llama_rn_context* main_ctx, llama_token token, const std::string &token_text) {
    (void) token_text;
    if (token < 0) {
        return false;
    }

    const tts_type tts_type = getTTSType(main_ctx);
    const tts_model_profile &profile = profile_for_type(tts_type);
    const auto & ranges = ensure_resolved_ranges(this, main_ctx, profile);

    parsed_audio_token parsed = {-1, 0};
    if (profile.audio.n_codebook == 2) {
        if (!token_code_from_ranges(ranges, profile.audio.n_codebook, token, parsed)) {
            return false;
        }
        if (parsed.codebook == 0) {
            pending_codebook1 = parsed.codec_value;
            return true;
        }
        if (pending_codebook1 < 0) {
            return true;
        }
        audio_tokens.push_back(pending_codebook1);
        audio_tokens.push_back(parsed.codec_value);
        pending_codebook1 = -1;
        return true;
    }

    int codec_value = 0;
    if (!is_token_in_ranges(ranges, profile.audio.n_codebook, token, &codec_value)) {
        return false;
    }
    audio_tokens.push_back(codec_value);
    return true;
}

bool llama_rn_context_tts::shouldCaptureAudioEmbeddings(llama_rn_context* main_ctx) {
    const tts_type tts_type = getTTSType(main_ctx);
    return profile_for_type(tts_type).decode_kind == tts_decode_kind::HIDDEN_STATES;
}

static int codec_decode_n_q_for_profile(const tts_model_profile &profile, ::codec_model *codec_model) {
    if (profile.audio.n_codebook > 0) {
        return profile.audio.n_codebook;
    }
    return std::max(codec_model_n_q(codec_model), 1);
}

std::vector<float> llama_rn_context_tts::decodeAudioTokens(llama_rn_context* main_ctx, const std::vector<llama_token> &tokens) {
    if (codec_ctx == nullptr || codec_model == nullptr) {
        LOG_ERROR("Codec context is not initialized");
        return std::vector<float>();
    }

    tts_type tts_type = getTTSType(main_ctx);
    const tts_model_profile &profile = profile_for_type(tts_type);
    if (profile.decode_kind == tts_decode_kind::HIDDEN_STATES) {
        if (main_ctx->completion == nullptr) {
            LOG_ERROR("Completion context is not initialized");
            return std::vector<float>();
        }
        return decodeAudioEmbeddings(main_ctx, main_ctx->completion->embeddings, main_ctx->completion->embedding_dim);
    }
    if (profile.decode_kind == tts_decode_kind::UNSUPPORTED) {
        LOG_ERROR("This TTS model's codec is not supported by codec.cpp yet");
        return std::vector<float>();
    }
    std::vector<llama_token> tokens_audio = tokens;

    if (tokens_audio.empty()) {
        LOG_ERROR("No audio codec tokens found in %zu completion tokens", tokens.size());
        return std::vector<float>();
    }

    const int n_cb_in = profile.audio.n_codebook > 0 ? profile.audio.n_codebook : 1;
    const int audio_cb_off = std::max(0, std::min(profile.audio_codebook_offset, n_cb_in - 1));
    // n_q for the downstream codec_decode call counts only the audio codebooks.
    const int n_q = (audio_cb_off > 0)
        ? (n_cb_in - audio_cb_off)
        : codec_decode_n_q_for_profile(profile, codec_model);
    // Need at least one complete (T × n_cb_in) frame to produce any audio.
    if ((int)tokens_audio.size() < n_cb_in) {
        LOG_ERROR("Audio token count %zu is below the minimum frame size n_cb=%d",
                  tokens_audio.size(), n_cb_in);
        return std::vector<float>();
    }
    // If the trailing tokens don't form a complete frame, drop them and warn.
    // This typically happens when the model is cut off mid-frame (n_predict
    // hit, context full, etc.).
    const size_t remainder = tokens_audio.size() % (size_t) n_cb_in;
    if (remainder != 0) {
        const size_t aligned = tokens_audio.size() - remainder;
        LOG_WARNING("Audio token count %zu is not divisible by n_cb=%d; dropping last %zu tokens",
                    tokens_audio.size(), n_cb_in, remainder);
        tokens_audio.resize(aligned);
    }

    const size_t n_frames = tokens_audio.size() / (size_t) n_cb_in;

    // For parallel_heads_delay codec_lm (MOSS-TTSD), each codebook was
    // emitted with a per-channel delay offset; reverse the shift before
    // forming the codec_token_buffer.  Aligned frame count is
    // n_frames - max_delay.  delay_pattern[] indices line up with the
    // codec_lm's n_codebook (i.e. include cb-0); we read the audio side
    // [audio_cb_off .. n_cb_in).
    std::vector<int32_t> audio_delays((size_t) n_q, 0);
    int max_delay = 0;
    if (codec_lm != nullptr) {
        const ::codec_lm_info * lm_info = ::codec_lm_get_info(codec_lm);
        if (lm_info != nullptr && lm_info->delay_pattern != nullptr &&
            lm_info->n_codebook >= n_cb_in) {
            for (int q = 0; q < n_q; ++q) {
                const int d = lm_info->delay_pattern[audio_cb_off + q];
                audio_delays[(size_t) q] = d;
                if (d > max_delay) max_delay = d;
            }
        }
    }

    if (max_delay > 0 && (int) n_frames <= max_delay) {
        LOG_ERROR("Audio frames %zu insufficient to cover delay_pattern (max_delay=%d)",
                  n_frames, max_delay);
        return std::vector<float>();
    }
    const size_t n_frames_aligned = (max_delay > 0) ? (n_frames - (size_t) max_delay) : n_frames;

    std::vector<int32_t> codec_tokens;
    if (audio_cb_off > 0 || max_delay > 0) {
        codec_tokens.resize(n_frames_aligned * (size_t) n_q);
        for (size_t t = 0; t < n_frames_aligned; ++t) {
            for (int q = 0; q < n_q; ++q) {
                const size_t src_t = t + (size_t) audio_delays[(size_t) q];
                codec_tokens[t * (size_t) n_q + (size_t) q] =
                    (int32_t) tokens_audio[src_t * (size_t) n_cb_in + (size_t) (audio_cb_off + q)];
            }
        }
    } else {
        codec_tokens.assign(tokens_audio.begin(), tokens_audio.end());
    }
    struct codec_token_buffer token_buffer = {};
    token_buffer.data = codec_tokens.data();
    token_buffer.n_tokens = (int32_t)codec_tokens.size();
    token_buffer.n_frames = (int32_t) n_frames;
    token_buffer.n_q = n_q;
    token_buffer.codebook_size = codec_model_codebook_size(codec_model);
    token_buffer.sample_rate = codec_model_sample_rate(codec_model);
    token_buffer.hop_size = codec_model_hop_size(codec_model);

    struct codec_decode_params decode_params = codec_decode_default_params();
    if (main_ctx->params.cpuparams.n_threads > 0) {
        decode_params.n_threads = main_ctx->params.cpuparams.n_threads;
    }
    decode_params.n_q = n_q;

    struct codec_pcm_buffer pcm = {};
    const enum codec_status status = codec_decode(codec_ctx, &token_buffer, &pcm, decode_params);
    if (status != CODEC_STATUS_SUCCESS) {
        const char *err = codec_get_last_error(codec_ctx);
        LOG_ERROR("codec_decode() failed: %s", err != nullptr ? err : "unknown error");
        return std::vector<float>();
    }

    std::vector<float> audio(pcm.data, pcm.data + pcm.n_samples);
    codec_pcm_buffer_free(&pcm);
    return audio;
}

std::vector<float> llama_rn_context_tts::decodeAudioEmbeddings(llama_rn_context* main_ctx, const std::vector<float> &embeddings, int embedding_dim) {
    if (codec_ctx == nullptr || codec_model == nullptr) {
        LOG_ERROR("Codec context is not initialized");
        return std::vector<float>();
    }
    if (embeddings.empty() || embedding_dim <= 0 || embeddings.size() % (size_t) embedding_dim != 0) {
        LOG_ERROR("Invalid audio embedding shape: %zu values, dim=%d", embeddings.size(), embedding_dim);
        return std::vector<float>();
    }

    struct codec_decode_params decode_params = codec_decode_default_params();
    if (main_ctx->params.cpuparams.n_threads > 0) {
        decode_params.n_threads = main_ctx->params.cpuparams.n_threads;
    }

    struct codec_pcm_buffer pcm = {};
    const int n_frames = (int) (embeddings.size() / (size_t) embedding_dim);
    const enum codec_status status = codec_decode_quantized_representation(
        codec_ctx,
        embeddings.data(),
        embedding_dim,
        n_frames,
        &pcm,
        decode_params);
    if (status != CODEC_STATUS_SUCCESS) {
        const char *err = codec_get_last_error(codec_ctx);
        LOG_ERROR("codec_decode_quantized_representation() failed: %s", err != nullptr ? err : "unknown error");
        return std::vector<float>();
    }

    std::vector<float> audio(pcm.data, pcm.data + pcm.n_samples);
    codec_pcm_buffer_free(&pcm);
    return audio;
}

llama_rn_speaker_artifact llama_rn_context_tts::encodeSpeaker(
    llama_rn_context * main_ctx,
    const llama_rn_encode_speaker_options & opts) {

    llama_rn_speaker_artifact out;
    out.ref_text = opts.ref_text;

    if (codec_ctx == nullptr || codec_model == nullptr) {
        LOG_ERROR("encodeSpeaker: codec context not initialized");
        return out;
    }
    if (opts.pcm.empty() || opts.input_sample_rate <= 0) {
        LOG_ERROR("encodeSpeaker: empty PCM or invalid sample rate %d",
                  opts.input_sample_rate);
        return out;
    }

    struct codec_audio audio = {};
    audio.data         = opts.pcm.data();
    audio.n_samples    = (int32_t) opts.pcm.size();
    audio.sample_rate  = opts.input_sample_rate;
    audio.n_channels   = 1;
    audio.pcm_type     = CODEC_PCM_TYPE_F32;

    struct codec_encode_params params = codec_encode_default_params();
    if (main_ctx != nullptr && main_ctx->params.cpuparams.n_threads > 0) {
        params.n_threads = main_ctx->params.cpuparams.n_threads;
    }

    struct codec_token_buffer tokens = {};
    const enum codec_status status = codec_encode(codec_ctx, &audio, &tokens, params);
    if (status != CODEC_STATUS_SUCCESS) {
        const char * err = codec_get_last_error(codec_ctx);
        LOG_ERROR("encodeSpeaker: codec_encode failed: %s",
                  err && *err ? err : "(no error message)");
        return out;
    }

    if (tokens.data != nullptr && tokens.n_tokens > 0) {
        out.ref_codes.assign(tokens.data, tokens.data + tokens.n_tokens);
    }
    out.n_q           = tokens.n_q;
    out.n_frames      = tokens.n_frames;
    out.sample_rate   = tokens.sample_rate;
    out.codebook_size = tokens.codebook_size;

    // If the loaded codec.gguf exposes a speaker section (Chatterbox / Qwen3-TTS
    // / MOSS-TTSD), drive it via codec.cpp's generic codec_lm_speaker_encode.
    // The info struct tells us which inputs the codec wants (some need pcm,
    // some need the just-computed ref_codes, some need an emotion scalar) and
    // declares the output shape.  We let codec.cpp validate.
    if (codec_lm == nullptr && !codec_lm_probed && codec_model != nullptr) {
        codec_lm_probed = true;
        codec_lm = ::codec_lm_create(codec_model);
    }
    if (codec_lm != nullptr) {
        const ::codec_lm_speaker_info * sp_info = ::codec_lm_speaker_get_info(codec_lm);
        if (sp_info != nullptr && sp_info->n_rows > 0 && sp_info->hidden_dim > 0) {
            // Optional resample stub — codec.cpp expects ref_pcm at
            // sp_info->ref_sample_rate; caller is responsible for resampling.
            // We just pass the raw PCM through and trust the caller (the JS
            // wrapper or a downstream resampler) to align rates before calling.
            struct codec_audio sp_audio = {};
            sp_audio.data        = opts.pcm.data();
            sp_audio.n_samples   = (int32_t) opts.pcm.size();
            sp_audio.sample_rate = opts.input_sample_rate;
            sp_audio.n_channels  = 1;
            sp_audio.pcm_type    = CODEC_PCM_TYPE_F32;

            const float emotion_val = opts.emotion;
            const float * emotion_ptr =
                (sp_info->needs_emotion_scalar && opts.has_emotion) ? &emotion_val : nullptr;

            out.speaker_emb.assign((size_t) sp_info->n_rows * (size_t) sp_info->hidden_dim, 0.0f);
            const enum codec_status sp_status = ::codec_lm_speaker_encode(
                codec_lm,
                sp_info->needs_ref_pcm ? &sp_audio : nullptr,
                (sp_info->needs_ref_speech_tokens && !out.ref_codes.empty()) ? out.ref_codes.data() : nullptr,
                (sp_info->needs_ref_speech_tokens && !out.ref_codes.empty()) ? (int32_t) out.ref_codes.size() : 0,
                emotion_ptr,
                out.speaker_emb.data(),
                (int32_t) out.speaker_emb.size());
            if (sp_status == CODEC_STATUS_SUCCESS) {
                out.speaker_n_rows = sp_info->n_rows;
                out.speaker_hidden_dim = sp_info->hidden_dim;
            } else {
                const char * err = ::codec_lm_get_last_error(codec_lm);
                LOG_WARNING("encodeSpeaker: codec_lm_speaker_encode failed (%d): %s — speaker_emb will be empty",
                            (int) sp_status, err && *err ? err : "(no error message)");
                out.speaker_emb.clear();
            }
        }
    }

    codec_token_buffer_free(&tokens);
    return out;
}

llama_rn_speaker_artifact llama_rn_context_tts::encodeSpeaker(
    llama_rn_context * main_ctx,
    const std::vector<float> & pcm,
    int input_sample_rate,
    const std::string & ref_text) {
    llama_rn_encode_speaker_options opts;
    opts.pcm = pcm;
    opts.input_sample_rate = input_sample_rate;
    opts.ref_text = ref_text;
    return encodeSpeaker(main_ctx, opts);
}

int llama_rn_context_tts::getAudioSampleRate() const {
    return codec_model != nullptr ? codec_model_sample_rate(codec_model) : 0;
}

}
