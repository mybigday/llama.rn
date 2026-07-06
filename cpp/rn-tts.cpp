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
#include <cstdlib>
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

  // Initialize the codec_common audio_lm context alongside the codec.
  // audio_lm_init re-uses the same GGUF file (just loads it again through
  // the codec_common abstraction); it will return nullptr without error for
  // GGUFs that have no codec.lm section — that's fine, those stay on the
  // direct codec_decode path.
  {
      codec_common::audio_lm_params alm_params;
      alm_params.codec_path = vocoder_model_path;
      alm_params.use_gpu    = use_gpu;
      std::string alm_err;
      audio_lm_ctx = codec_common::audio_lm_init(alm_params, &alm_err);
      if (audio_lm_ctx == nullptr) {
          // Not a hard failure; the model may just lack an LM section.
          LOG_WARNING("audio_lm_init failed (non-fatal): %s",
                      alm_err.empty() ? "(no error)" : alm_err.c_str());
      }
  }

  type = UNKNOWN; // Will be determined when used
}

llama_rn_context_tts::~llama_rn_context_tts() {
  if (bb_sampler != nullptr) {
      common_sampler_free(bb_sampler);
      bb_sampler = nullptr;
  }
  if (audio_lm_ctx != nullptr) {
      codec_common::audio_lm_free(audio_lm_ctx);
      audio_lm_ctx = nullptr;
  }
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
    audio_embeddings.clear();
    audio_embedding_dim = 0;
    pending_feedback_embd.clear();
    audio_embeddings_pending = false;
    audio_embeddings_done = false;
    prompt_hiddens.clear();
    continuous_prefill_done = false;

    pending_speaker_emb_prefix.clear();
    pending_speaker_emb_rows = 0;
    pending_speaker_emb_hidden_dim = 0;
    pending_next_embd.clear();
    codec_lm_ar_pending_embd = false;
    codec_lm_ar_done = false;
    codec_lm_ar_step = 0;
    codec_lm_ar_rng = 0;
    codec_lm_ar_stopped_on_eos = false;

    talker_text_tokens.clear();
    talker_trailing = 0;
    // talker_prefix_{embd,rows,hidden} intentionally NOT cleared here.
    // They are set by getFormattedAudioCompletion (called before rewind/reset)
    // and consumed once by nextToken.  getFormattedAudioCompletion clears them
    // at its own entry point for non-talker models, so no stale prefix leaks
    // across generations.

    // Reset audio_lm per-sequence state (codes accumulator + step machine),
    // but keep the loaded weights + capabilities.
    if (audio_lm_ctx != nullptr) {
        codec_common::audio_lm_reset(audio_lm_ctx);
    }

    // bb_sampler is rebuilt on first use each generation (needs the model
    // pointer); free it here so it picks up fresh grammar next run.
    if (bb_sampler != nullptr) {
        common_sampler_free(bb_sampler);
        bb_sampler = nullptr;
        bb_sampler_built = false;
    }

    chatterbox_n_seq  = 0;
    chatterbox_n_past = 0;
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
    BLUEMAGPIE,
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
    // BlueMagpie-TTS (OpenFormosa): Barbet (Mamba2+attn hybrid) backbone + AudioVAE
    // continuous-latent codec_lm.  Continuous flow doesn't consume codebooks; the
    // completion loop's `tryContinuousAudioStep` hook drives codec_lm_step_generate
    // and accumulates patches into audio_embeddings, which JS pipes to
    // `decodeAudioEmbeddings`.  n_codebook / code_ranges are unused on this path
    // — same convention as the SOPRANO HIDDEN_STATES profile.  Emitted flow is
    // "continuous_embd" (see getFormattedAudioCompletion), so decode_kind stays
    // HIDDEN_STATES to keep `embedding = true` and skip token-flow handling.
    {BLUEMAGPIE_TTS, tts_prompt_kind::BLUEMAGPIE, {0, {}}, tts_decode_kind::HIDDEN_STATES, 0},
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
    if (contains_case_insensitive(value, "bluemagpie") ||
        contains_case_insensitive(value, "blue-magpie") ||
        contains_case_insensitive(value, "barbet")) {
        return BLUEMAGPIE_TTS;
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
        if (codec_lm == nullptr) {
            LOG_VERBOSE("codec_lm_create: %s", ::codec_lm_get_create_error());
        }
    }
    if (codec_lm != nullptr) {
        const ::codec_lm_info * info = ::codec_lm_get_info(codec_lm);
        const std::string host_arch = info && info->host_arch ? info->host_arch : "";
        const int n_cb = info ? info->n_codebook : 0;
        // Continuous-latent codec_lm (BlueMagpie-TTS / VoxCPM) — no codebooks,
        // AR loop emits latent patches instead.  host_arch is "barbet" for
        // BlueMagpie; VoxCPM will land here with "minicpm4" once its converter
        // exists.  Take the continuous flag as the discriminator since
        // codec_lm_info.is_continuous is set by codec.cpp on this family.
        if (info && info->is_continuous) {
            if (host_arch == "barbet") return BLUEMAGPIE_TTS;
        }
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

    // Chatterbox S3G codec: the vocoder GGUF has no embedded LM (the T3
    // backbone is a separate file), so codec_lm_create returns nullptr.
    // Detect by the codec arch enum which is always set correctly by codec.cpp.
    // Distinguish English (S3G standard) from multilingual by the backbone's
    // `general.name` — the multilingual GGUF has a meaningless hash name so
    // we fall back to CHATTERBOX_T3 (both share the same runtime path anyway).
    if (codec_model != nullptr) {
        const codec_arch ca = ::codec_model_arch(codec_model);
        if (ca == CODEC_ARCH_CHATTERBOX_S3G) {
            // Check backbone name for multilingual marker.
            if (main_ctx && main_ctx->model) {
                const std::string bname = main_ctx->model->name;
                if (contains_case_insensitive(bname, "chatterbox-multilingual") ||
                    contains_case_insensitive(bname, "chatterbox_multilingual") ||
                    contains_case_insensitive(bname, "mtl23ls")) {
                    return CHATTERBOX_T3_MULTILINGUAL;
                }
            }
            return CHATTERBOX_T3;
        }
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
        case tts_prompt_kind::BLUEMAGPIE:
            cap.prompt_kind = "bluemagpie";
            cap.family = "bluemagpie";
            break;
    }
    cap.requires_phonemes = (profile.prompt_kind == tts_prompt_kind::NEUTTS);
    // BlueMagpie was trained on Taiwanese-Mandarin data — flag zh-tw so the
    // JS-side language picker defaults right; other families keep en-us.
    cap.default_language =
        (profile.prompt_kind == tts_prompt_kind::BLUEMAGPIE) ? "zh-tw" : "en-us";
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
// so feeding these text tokens as ordinary backbone tokens drives the
// right text embedding without any out-of-band lookup.
//
// We do NOT emit a leading `<|begin_of_text|>` here: the completion
// path tokenizes with `add_bos=true` for Llama-family vocabs (CSM's
// backbone is Llama-3.2-1B), so the tokenizer already prepends the BOS.
// Embedding a literal `<|begin_of_text|>` in the prompt would
// double-BOS the sequence and drift CSM's output vs the reference
// CsmProcessor.  (Old standalone `generateAudioCodes` tokenized with
// `add_special=false`, which is why the literal worked there.)
static std::string build_csm_prompt(json speaker, const std::string &text_to_speak) {
    int speaker_id = 0;
    if (speaker.is_object() && speaker.contains("id")) {
        try {
            speaker_id = speaker["id"].get<int>();
        } catch (...) { /* fall back to 0 */ }
    }
    std::string out;
    out.reserve(text_to_speak.size() + 32);
    out += "[";
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

// BlueMagpie-TTS prompt: `[spk] + text + [audio_start]`.
// The upstream driver (`bluemagpie/model.py::_generate` → `_build_inputs`,
// speaker_slot="null" default when no centroid is provided) puts the learned
// spk token at position 0.  This slot was trained via speaker dropout — its
// embedding tells the LM "null speaker mode", which the stop_head was
// conditioned on.
//
// CRITICAL: the spk / audio_start ids are NOT the `<|speaker|>` (114674) /
// `<|audio_start|>` (114666) entries in tokenizer.json.  BlueMagpie's
// `bluemagpie/config.py::resolve_barbet_config` auto-allocates them in the
// Megatron *padding region* — spk=114826, audio_start=114822 — and the model
// was conditioned on THOSE embedding rows.  Feeding the tokenizer.json ids
// gives position 0 the wrong embedding, so the whole causal hidden trajectory
// diverges and the stop head never fires.  The converter now bakes those
// padding-region ids as CONTROL tokens `<|bm_spk|>` / `<|bm_audio_start|>`, so
// emitting those strings (parse_special=true) tokenizes onto the correct rows.
static std::string build_bluemagpie_prompt(json speaker, const std::string &text_to_speak) {
    (void) speaker;
    return "<|bm_spk|>" + text_to_speak + "<|bm_audio_start|>";
}

llama_rn_audio_completion_result llama_rn_context_tts::getFormattedAudioCompletion(llama_rn_context* main_ctx, const std::string &speaker_json_str, const std::string &text_to_speak) {
    // Always clear per-generation state that survives reset() here so that
    // stale data from a previous generation doesn't pollute a new one.
    // (reset() intentionally skips these fields so they survive the rewind()
    // that fires between getFormattedAudioCompletion and nextToken.)
    talker_prefix_embd.clear();
    talker_prefix_rows           = 0;
    talker_prefix_hidden         = 0;
    chatterbox_prefill_pending   = false;
    chatterbox_text.clear();

    json speaker = speaker_json_str.empty() ? json::object() : json::parse(speaker_json_str);
    const tts_type tts_type = getTTSType(main_ctx, speaker);
    if (tts_type == UNKNOWN) {
        LOG_ERROR("Unknown TTS version");
        return {"", "", false, ""};
    }

    const tts_model_profile &profile = profile_for_type(tts_type);
    const std::string grammar = build_dynamic_grammar(profile, text_to_speak);

    // Continuous-latent codec_lm (BlueMagpie-TTS / VoxCPM) hijacks the
    // standard `tokens` flow: the completion loop drives the codec_lm
    // step machine via `tryContinuousAudioStep` on each `llama_decode`,
    // accumulating latent patches into `audio_embeddings` which JS then
    // hands to `decodeAudioEmbeddings`.  Signalled via
    // `flow = "continuous_embd"` + `embedding = true`.  Detection is
    // dynamic: no dedicated `tts_type` yet, so we probe the codec_lm.
    const bool is_continuous_lm = this->isTTSContinuous(main_ctx);
    // Codebook codec_lm-AR (CSM / Qwen3-TTS / MOSS-TTSD /
    // MOSS-TTS-Realtime / Chatterbox) now shares the same completion
    // loop: the per-step hook `tryCodecLmAudioStep` fires from
    // `rn-completion.cpp`, appending codes to `audio_tokens` and
    // composing the next backbone embed.  Emit `flow = "tokens"` +
    // `embedding = true` so JS treats these exactly like Type A
    // (OuteTTS / Soprano / NeuTTS) — `completion` + `decodeAudioTokens`.
    // The `generateAudioCodes` wrapper still works for source compat but
    // new JS callers should skip it.
    const bool is_codec_lm_ar =
        profile.decode_kind == tts_decode_kind::CODEC_LM_AR;
    const bool embedding = is_continuous_lm || is_codec_lm_ar
        || profile.decode_kind == tts_decode_kind::HIDDEN_STATES;

    // Detect Qwen3-TTS talker path via the audio_lm_ctx (which was init'd
    // in the constructor).  When audio_lm_talker_has_projection returns true
    // the prompt is NOT a text string — it's a composed embedding prefix that
    // the completion loop injects via a manual b.embd batch before the AR
    // loop.  We build it here and stash it in the result so the completion
    // driver in rn-completion.cpp can inject it.
    //
    // Chatterbox also deviates from the text-prompt path: its backbone has
    // no text tokenizer (tokenizer.ggml.model=none), so we tokenize via the
    // codec_lm's own BPE and build the full CFG prefix entirely inside
    // tryChatterboxPrefill (called from the completion loop's prefill phase).
    // Here we just signal "chatterbox" via flow="chatterbox_embd".
    const bool is_talker = audio_lm_ctx != nullptr &&
                           codec_common::audio_lm_talker_has_projection(audio_lm_ctx);

    if (is_talker && main_ctx != nullptr && main_ctx->model != nullptr) {
        // Qwen3-TTS talker path: build embedding prefix and return it.
        // The completion loop sees flow="talker_embd" and skips normal
        // token-batch prefill, feeding these rows as b.embd instead.
        const int hidden = llama_model_n_embd(main_ctx->model);
        if (hidden <= 0) {
            LOG_ERROR("getFormattedAudioCompletion: could not get n_embd");
            return {};
        }
        const llama_vocab * vocab = llama_model_get_vocab(main_ctx->model);
        if (vocab == nullptr) {
            LOG_ERROR("getFormattedAudioCompletion: could not get vocab");
            return {};
        }

        // Tokenize role header and payload text (same as run_codebook_ar).
        auto tok_str = [&](const std::string & s, bool special) -> std::vector<llama_token> {
            const int n = llama_vocab_n_tokens(vocab);
            std::vector<llama_token> out(n + 16);
            const int got = llama_tokenize(vocab, s.c_str(), (int32_t)s.size(),
                                           out.data(), (int32_t)out.size(),
                                           /*add_special=*/false, special);
            if (got < 0) return {};
            out.resize((size_t)got);
            return out;
        };
        std::vector<llama_token> role_toks = tok_str("<|im_start|>assistant\n", true);
        std::vector<llama_token> text_toks = tok_str(text_to_speak, false);
        if (role_toks.empty() || text_toks.empty()) {
            LOG_ERROR("getFormattedAudioCompletion: talker tokenize failed");
            return {};
        }

        // Extract x-vector from speaker JSON if provided.
        std::vector<float> xvec;
        if (speaker.contains("x_vector")) {
            for (auto &v : speaker["x_vector"]) xvec.push_back(v.get<float>());
        }
        const float * xvec_ptr = (xvec.size() == (size_t)hidden) ? xvec.data() : nullptr;
        const int32_t xvec_dim = xvec_ptr ? hidden : 0;

        const int32_t cap_rows = (int32_t)role_toks.size() + 6 + 4;
        std::vector<float> prefix((size_t)cap_rows * hidden);
        int32_t n_rows = 0, consumed = 0;
        if (!codec_common::audio_lm_build_talker_prefix(
                audio_lm_ctx,
                role_toks.data(), (int32_t)role_toks.size(),
                text_toks.data(), (int32_t)text_toks.size(),
                xvec_ptr, xvec_dim,
                prefix.data(), cap_rows, &n_rows, &consumed)) {
            LOG_ERROR("getFormattedAudioCompletion: audio_lm_build_talker_prefix failed: %s",
                      codec_common::audio_lm_last_error(audio_lm_ctx));
            return {};
        }
        prefix.resize((size_t)n_rows * hidden);

        // Store the text tokens for per-step trailing text injection.
        talker_text_tokens.assign(text_toks.begin(), text_toks.end());
        talker_trailing = 0;

        // Also stash the prefix on `this` so rn-completion.cpp's nextToken
        // can inject it as an embd batch before starting the AR loop.
        // (RNLlamaJSI only serialises prompt/grammar/embedding/flow to JS,
        // so the result struct fields alone would not survive the round-trip.)
        talker_prefix_embd   = prefix;
        talker_prefix_rows   = n_rows;
        talker_prefix_hidden = hidden;

        llama_rn_audio_completion_result res;
        res.prompt               = "";           // unused — prefill via embd
        res.grammar              = "";
        res.embedding            = true;
        res.flow                 = "talker_embd";
        res.talker_prefix_embd   = std::move(prefix);
        res.talker_prefix_rows   = n_rows;
        res.talker_prefix_hidden = hidden;
        return res;
    }

    // Chatterbox: signal the completion loop to call tryChatterboxPrefill.
    // Return empty text prompt; the prefill happens in rn-completion.cpp
    // when it sees flow=="chatterbox_embd".
    const bool is_chatterbox = (profile.prompt_kind == tts_prompt_kind::CHATTERBOX) &&
                                audio_lm_ctx != nullptr;
    if (is_chatterbox) {
        // Signal rn-completion.cpp to call tryChatterboxPrefill before the
        // AR loop starts.  Survives rewind() — see talker_prefix design note.
        chatterbox_prefill_pending = true;
        chatterbox_text            = text_to_speak;  // stored here; params.prompt stays empty
        // Default CFG weight; the JS layer can override via params if needed.
        chatterbox_cfg_weight      = 0.7f;

        llama_rn_audio_completion_result res;
        // Empty prompt: the backbone has no text tokenizer (tokenizer.ggml.model=none).
        // loadPrompt("") sets n_past=-1 (empty batch guard); the Chatterbox prefill
        // block in rn-completion.cpp normalizes that to 0 and calls tryChatterboxPrefill.
        res.prompt     = "";
        res.grammar    = "";
        res.embedding  = true;
        res.flow       = "chatterbox_embd";
        return res;
    }

    const std::string flow = is_continuous_lm ? "continuous_embd" : "tokens";
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
            // Fallback when audio_lm_ctx has no talker projection (stale GGUF).
            return {build_qwen3_tts_prompt(speaker, text_to_speak), "", embedding, flow};
        case tts_prompt_kind::MOSS_TTS_REALTIME:
            return {build_moss_tts_realtime_prompt(speaker, text_to_speak), "", embedding, flow};
        case tts_prompt_kind::MOSS_TTSD:
            return {build_moss_ttsd_prompt(speaker, text_to_speak), "", embedding, flow};
        case tts_prompt_kind::CHATTERBOX:
            // Fallback when audio_lm_ctx is unavailable (UNSUPPORTED decode_kind).
            return {build_chatterbox_prompt(speaker, text_to_speak), "", embedding, flow};
        case tts_prompt_kind::BLUEMAGPIE:
            return {build_bluemagpie_prompt(speaker, text_to_speak), "", embedding, flow};
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

// generateAudioCodes — source-compat wrapper.
//
// After the codec_lm-AR refactor, all Type B/C/D flows (CSM / Qwen3-TTS /
// MOSS-TTSD / MOSS-TTS-Realtime / Chatterbox) share the standard
// `completion` loop with everything else.  The per-step codec_lm state
// machine that used to live inside this function is now
// `tryCodecLmAudioStep` (rn-tts.cpp), driven from `rn-completion.cpp`.
//
// This wrapper keeps the historical `generateAudioCodes(opts, on_frame)`
// entry point working: it seeds the codec_lm sampler + optional speaker
// prefix, primes the completion params, and runs the completion loop
// until the codec_lm signals stop (or `max_frames` frames are emitted).
// The (T × n_codebook) codes are drained from `tts_wrapper->audio_tokens`
// into the result, matching the old signature.
//
// New JS callers can skip this wrapper: `getFormattedAudioCompletion`
// now returns `flow = "tokens"` for codec_lm-AR models, so `completion` +
// `decodeAudioTokens` works the same way it does for OuteTTS / Soprano /
// NeuTTS.
llama_rn_audio_codes_result llama_rn_context_tts::generateAudioCodes(
    llama_rn_context * main_ctx,
    const llama_rn_audio_codes_options & opts,
    const llama_rn_audio_codes_progress_cb & on_frame) {

    llama_rn_audio_codes_result result;

    if (main_ctx == nullptr || main_ctx->ctx == nullptr || main_ctx->model == nullptr) {
        LOG_ERROR("generateAudioCodes: main context not initialized");
        return result;
    }
    if (main_ctx->completion == nullptr) {
        LOG_ERROR("generateAudioCodes: completion context not initialized");
        return result;
    }
    if (codec_model == nullptr) {
        LOG_ERROR("generateAudioCodes: codec model not loaded");
        return result;
    }

    // Lazy codec_lm probe — reused by the completion loop's
    // `isTTSCodecLmAR` check below.
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
    const ::codec_lm_info * info = ::codec_lm_get_info(codec_lm);
    if (info == nullptr) {
        LOG_ERROR("generateAudioCodes: codec_lm_get_info returned NULL");
        return result;
    }
    if (info->is_continuous) {
        LOG_ERROR("generateAudioCodes: this codec_lm is continuous-latent; use completion() + decodeAudioEmbeddings() instead");
        return result;
    }
    const int n_cb   = info->n_codebook;
    const int hidden = info->hidden_dim;
    const int model_n_embd = llama_model_n_embd(main_ctx->model);
    if (model_n_embd != hidden) {
        LOG_ERROR("generateAudioCodes: backbone n_embd=%d != codec_lm hidden=%d",
                  model_n_embd, hidden);
        return result;
    }

    // Stash optional speaker-conditioning prefix (output of
    // `codec_lm_speaker_encode`) so the completion loop's first
    // `llama_decode` injects it as an embd-batch before the token prompt.
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
        } else {
            pending_speaker_emb_prefix       = opts.speaker_emb_prefix;
            pending_speaker_emb_rows         = opts.speaker_emb_rows;
            pending_speaker_emb_hidden_dim   = opts.speaker_emb_hidden_dim;
        }
    }

    // Configure completion params for a codec_lm-AR run.  We reuse the
    // caller's temperature / top_p / top_k / seed (they get applied to
    // the codec_lm's codebook sampler in `tryCodecLmAudioStep` and to
    // the backbone's sampler in the standard completion path).  Each
    // frame corresponds to one `llama_decode` = one predicted token
    // slot; cap at `max_frames`.
    const int max_frames = std::max(opts.max_frames, 1);
    main_ctx->params.prompt        = opts.prompt;
    main_ctx->params.n_predict     = max_frames;
    main_ctx->params.embedding     = true;    // for llama_get_embeddings_ith
    main_ctx->params.sampling.temp = opts.temperature;
    main_ctx->params.sampling.top_p = opts.top_p;
    main_ctx->params.sampling.top_k = opts.top_k;
    if (opts.seed != 0) {
        main_ctx->params.sampling.seed = opts.seed;
    }
    // The codec_lm sampler seed lives on the tts wrapper (persists across
    // steps within a completion, re-seeded in `reset()`).
    codec_lm_ar_rng = opts.seed ? (uint64_t) opts.seed : 0xC0DEC1ABULL;

    llama_set_embeddings(main_ctx->ctx, true);

    // Drive the standard completion loop.  `rewind()` clears
    // `audio_tokens` + tts flags + resets sampler.  `loadPrompt({})`
    // tokenizes the prompt through the standard pipeline.  Then
    // `doCompletion()` runs the codec_lm step per `llama_decode`.
    main_ctx->completion->rewind();
    if (!main_ctx->completion->initSampling()) {
        LOG_ERROR("generateAudioCodes: initSampling failed");
        return result;
    }
    main_ctx->completion->loadPrompt({});
    if (main_ctx->completion->context_full) {
        LOG_ERROR("generateAudioCodes: prompt exceeds n_ctx (%d)",
                  main_ctx->n_ctx);
        return result;
    }
    main_ctx->completion->beginCompletion();

    const auto now_us = []() -> int64_t {
        const auto t = std::chrono::steady_clock::now().time_since_epoch();
        return std::chrono::duration_cast<std::chrono::microseconds>(t).count();
    };
    const int64_t loop_t0 = now_us();

    int step = 0;
    while (main_ctx->completion->has_next_token) {
        const size_t codes_before = audio_tokens.size();
        (void) main_ctx->completion->doCompletion();

        // A completed frame appended `n_cb` codes to audio_tokens.  Feed
        // the progress callback + capture stop.
        if (audio_tokens.size() >= codes_before + (size_t) n_cb) {
            std::vector<int32_t> frame_codes(
                audio_tokens.begin() + (int64_t) codes_before,
                audio_tokens.begin() + (int64_t) codes_before + n_cb);
            if (on_frame) {
                if (!on_frame(step, frame_codes)) {
                    result.aborted = true;
                    break;
                }
            }
            step++;
            if (step >= max_frames) {
                break;
            }
        }
        if (codec_lm_ar_done) {
            break;
        }
    }
    main_ctx->completion->endCompletion();

    if (step > 0) {
        const int64_t loop_us = now_us() - loop_t0;
        LOG_INFO("generateAudioCodes done: %d frames in %.2fs (%.1f frames/s)",
                 step, (double) loop_us / 1e6,
                 (double) step / std::max((double) loop_us / 1e6, 1e-6));
    }

    // Drain audio_tokens (llama_token = int32_t) into the result.
    result.codes.reserve(audio_tokens.size());
    for (const llama_token t : audio_tokens) {
        result.codes.push_back((int32_t) t);
    }
    result.n_codebook = n_cb;
    result.n_frames = (int) (result.codes.size() / (size_t) std::max(n_cb, 1));
    result.stopped_on_eos = codec_lm_ar_stopped_on_eos;
    return result;
}

// ─────────────────────────────────────────────────────────────────────
// Continuous-latent codec_lm per-step hook (BlueMagpie-TTS / VoxCPM).
//
// Called from `rn-completion.cpp`'s `nextToken` after each `llama_decode`
// when the loaded codec_lm reports `is_continuous = true`.
//
//   1. `codec_lm_step_generate` runs tslm_adapter → FSQ → RALM → LocDiT
//      CFM diffusion on the just-read backbone hidden and produces one
//      latent patch (patch_size × latent_dim).  We append it to
//      `audio_embeddings` frame-major so
//      `decodeAudioEmbeddings(audio_embeddings, latent_dim)` on the JS
//      side reproduces `codec_decode_quantized_representation`.
//   2. If the stop head fires, `audio_embeddings_done = true` and the
//      completion loop terminates — no feedback embd is produced.
//   3. Otherwise `codec_lm_step_feedback_embd` writes the LocEnc feedback
//      into `pending_feedback_embd`; the completion loop injects that
//      into the next `llama_decode` via `b.embd`.
//
// State (codec_lm_state, audio_embeddings, pending_feedback_embd) is
// reset by `llama_rn_context_tts::reset()` at rewind time.
// ─────────────────────────────────────────────────────────────────────
bool llama_rn_context_tts::isTTSContinuous(llama_rn_context * main_ctx) {
    if (codec_model == nullptr) {
        return false;
    }
    if (codec_lm == nullptr && !codec_lm_probed) {
        codec_lm_probed = true;
        codec_lm = ::codec_lm_create(codec_model);
    }
    if (codec_lm == nullptr) {
        return false;
    }
    const ::codec_lm_info * info = ::codec_lm_get_info(codec_lm);
    if (info == nullptr || !info->is_continuous) {
        return false;
    }
    if (main_ctx != nullptr && main_ctx->model != nullptr) {
        const int model_n_embd = llama_model_n_embd(main_ctx->model);
        if (model_n_embd != info->hidden_dim) {
            LOG_WARNING("isTTSContinuous: backbone n_embd=%d != codec_lm hidden=%d",
                        model_n_embd, info->hidden_dim);
        }
    }
    return true;
}

// ─────────────────────────────────────────────────────────────────────
// Codebook codec_lm-AR per-step hook (CSM / Qwen3-TTS / MOSS-TTSD /
// MOSS-TTS-Realtime / Chatterbox).
//
// Extracted verbatim (semantics-preserving) from `generateAudioCodes`
// so the completion loop in `rn-completion.cpp` can drive one codec_lm
// step per `llama_decode` iteration.  Structurally parallel to
// `tryContinuousAudioStep`: state on `llama_rn_context_tts`, no direct
// `llama_decode` calls (that's the completion loop's job).
// ─────────────────────────────────────────────────────────────────────
bool llama_rn_context_tts::isTTSCodecLmAR(llama_rn_context * main_ctx) {
    if (codec_model == nullptr) {
        return false;
    }
    if (codec_lm == nullptr && !codec_lm_probed) {
        codec_lm_probed = true;
        codec_lm = ::codec_lm_create(codec_model);
    }
    if (codec_lm == nullptr) {
        return false;
    }
    const ::codec_lm_info * info = ::codec_lm_get_info(codec_lm);
    if (info == nullptr || info->is_continuous) {
        return false;
    }
    // Only codebook AR kinds route through this hook.  Continuous-latent
    // is handled by `tryContinuousAudioStep`; UNKNOWN kinds mean the
    // codec.gguf carries a partial `lm.*` section we can't drive.
    if (info->kind != CODEC_LM_KIND_RESIDUAL_DEPTH_AR &&
        info->kind != CODEC_LM_KIND_PARALLEL_HEADS_DELAY) {
        return false;
    }
    if (main_ctx != nullptr && main_ctx->model != nullptr) {
        const int model_n_embd = llama_model_n_embd(main_ctx->model);
        if (model_n_embd != info->hidden_dim) {
            LOG_WARNING("isTTSCodecLmAR: backbone n_embd=%d != codec_lm hidden=%d",
                        model_n_embd, info->hidden_dim);
        }
    }
    return true;
}

// ─────────────────────────────────────────────────────────────────────
// audio_lm-driven codebook-AR step (Qwen3-TTS talker / MOSS-TTSD /
// MOSS-TTS-Realtime / Chatterbox).
//
// For models with audio_lm_ctx set up the full per-step sequence is:
//   [set_text_context(cb0)]   — MOSS-TTSD cb0-from-backbone only
//   audio_lm_step_begin
//   for cb in 0..n_cb-1:
//     audio_lm_step_logits → sample → audio_lm_step_push_code
//   audio_lm_step_finish
//   audio_lm_observe_codes → accumulate + compose next_embed
//   [+ trailing text embed for Qwen3-TTS talker]
//   → pending_next_embd for the completion loop to inject
//
// Returns true on success (or EOS); sets codec_lm_ar_done on stop.
// ─────────────────────────────────────────────────────────────────────
static bool try_audio_lm_step(
    codec_common::audio_lm_context * alm_ctx,
    llama_context *  lctx,
    llama_model *    lmodel,
    const float *    hidden,
    int              hidden_dim,
    // MOSS-TTSD cb0-from-backbone fields:
    bool             cb0_from_backbone,
    common_sampler ** bb_sampler_ptr,
    bool *           bb_sampler_built_ptr,
    const std::string & bb_grammar,
    // sampling params for audio codebooks:
    float samp_temp, int32_t samp_topk, float samp_topp,
    uint64_t * rng_state,
    // Qwen3-TTS talker trailing text:
    bool             is_talker,
    const std::vector<int32_t> & talker_text_toks,
    int *            talker_trailing_ptr,
    // outputs:
    std::vector<float> * next_embd_out,
    std::vector<int32_t> * codes_out,
    bool * is_eos_out,
    const char ** stop_reason_out)
{
    const int n_cb = codec_common::audio_lm_n_codebook(alm_ctx);
    if (n_cb <= 0) return false;

    // MOSS-TTSD cb0-from-backbone: sample cb0 from backbone lm_head using
    // a common_sampler with GBNF grammar (constrains to speech range ∪ eos).
    // Build the sampler lazily on first step.
    if (cb0_from_backbone) {
        if (!(*bb_sampler_built_ptr) && lmodel != nullptr) {
            common_params_sampling sp;
            sp.seed           = *rng_state ? (uint32_t)(*rng_state & 0xFFFFFFFF)
                                           : 0xC0DEC1ABu;
            sp.no_perf        = true;
            sp.temp           = samp_temp;
            sp.top_k          = samp_topk > 0 ? samp_topk : 0;
            sp.top_p          = (samp_topp > 0.0f && samp_topp < 1.0f) ? samp_topp : 1.0f;
            sp.min_p          = 0.0f;
            sp.penalty_repeat = 1.0f;
            sp.penalty_last_n = 0;
            sp.penalty_freq   = 0.0f;
            sp.penalty_present = 0.0f;
            sp.samplers = {
                COMMON_SAMPLER_TYPE_TOP_K,
                COMMON_SAMPLER_TYPE_TOP_P,
                COMMON_SAMPLER_TYPE_TEMPERATURE,
            };
            if (!bb_grammar.empty()) {
                sp.grammar = bb_grammar;
            }
            *bb_sampler_ptr = common_sampler_init(lmodel, sp);
            *bb_sampler_built_ptr = true;
        }
        if (*bb_sampler_ptr != nullptr && lctx != nullptr) {
            const int32_t c0 = common_sampler_sample(*bb_sampler_ptr, lctx, -1,
                                                      /*grammar_first=*/false);
            common_sampler_accept(*bb_sampler_ptr, c0, /*is_generated=*/true);
            if (!codec_common::audio_lm_step_set_text_context(alm_ctx, c0)) {
                return false;
            }
        }
    }

    if (!codec_common::audio_lm_step_begin(alm_ctx, hidden, hidden_dim)) {
        return false;
    }

    for (int cb = 0; cb < n_cb; ++cb) {
        int32_t cb_idx = 0, nlog = 0;
        const float * lg = codec_common::audio_lm_step_logits(alm_ctx, &cb_idx, &nlog);
        if (lg == nullptr || nlog <= 0) return false;
        // cb0-from-backbone: the code was already set via set_text_context;
        // we still need to push it through the step machine here.
        int32_t code;
        if (cb0_from_backbone && cb == 0) {
            // Re-read from the state — it was stashed by set_text_context.
            // sample_codec_logits with greedy on the first logit row works
            // only if nlog > 0; we just pick the cb0 we already sampled.
            // The step machine expects us to call push_code regardless.
            code = sample_codec_logits(lg, nlog, 0.0f, 0, 0.0f, rng_state); // greedy fallback
            // Overwrite: the real cb0 came from bb_sampler; re-derive by
            // argmax from lg (this is the codec_lm's view of cb0 logits, NOT
            // the backbone's).  For parallel_heads_delay cb0_from_backbone
            // the step machine needs us to push the backbone-sampled code.
            // We stored it in set_text_context; step_logits for cb0 returns
            // logits from the BACKBONE head (routed internally), so this
            // sample is consistent.  Use our raw sampler on those logits.
            code = sample_codec_logits(lg, nlog, samp_temp, samp_topk, samp_topp, rng_state);
        } else {
            code = sample_codec_logits(lg, nlog, samp_temp, samp_topk, samp_topp, rng_state);
        }
        if (!codec_common::audio_lm_step_push_code(alm_ctx, code)) return false;
    }

    codes_out->assign((size_t)n_cb, 0);
    if (!codec_common::audio_lm_step_finish(alm_ctx, codes_out->data(), n_cb)) return false;

    // Accumulate codes + compose next embed via audio_lm_observe_codes.
    // This handles: EOS detection, delay-unshift accumulation, and
    // next-embed composition (when uses_embed_override is set).
    auto act = codec_common::audio_lm_observe_codes(
        alm_ctx, codes_out->data(), n_cb, hidden, hidden_dim);
    if (act == codec_common::OBSERVE_STOP) {
        const char * e = codec_common::audio_lm_last_error(alm_ctx);
        if (e && *e) {
            // Real error, not just EOS.
            return false;
        }
        *is_eos_out = true;
        if (stop_reason_out) *stop_reason_out = "eos_code_c0";
        return true;
    }

    // Retrieve the next backbone input embedding.
    int32_t ndim = 0;
    const float * nb = codec_common::audio_lm_get_next_embed(alm_ctx, &ndim);
    if (nb == nullptr || ndim != hidden_dim) {
        // audio_lm_observe_codes returned OBSERVE_CONSUMED (no embed override
        // set) — fall back to direct compose_next_embd.
        next_embd_out->assign((size_t)hidden_dim, 0.0f);
        codec_lm * lm = codec_common::audio_lm_get_lm(alm_ctx);
        if (lm != nullptr) {
            ::codec_lm_compose_next_embd(lm, codes_out->data(),
                                         0, next_embd_out->data());
        }
    } else {
        next_embd_out->assign(nb, nb + ndim);
    }

    // Qwen3-TTS talker: add trailing text projection to next embed.
    if (is_talker && talker_trailing_ptr != nullptr) {
        std::vector<float> tt((size_t)hidden_dim, 0.0f);
        if (codec_common::audio_lm_talker_trailing_text_embd(
                alm_ctx,
                talker_text_toks.data(), (int32_t)talker_text_toks.size(),
                *talker_trailing_ptr,
                tt.data(), hidden_dim)) {
            for (int i = 0; i < hidden_dim; ++i) (*next_embd_out)[i] += tt[i];
        }
        ++(*talker_trailing_ptr);
    }

    *is_eos_out = false;
    return true;
}

bool llama_rn_context_tts::tryCodecLmAudioStep(
    llama_rn_context * main_ctx,
    llama_token        backbone_sampled_tok,
    const float *      hidden,
    int                hidden_dim) {

    if (codec_lm == nullptr) {
        LOG_ERROR("tryCodecLmAudioStep: codec_lm not created");
        return false;
    }
    if (hidden == nullptr || hidden_dim <= 0) {
        LOG_ERROR("tryCodecLmAudioStep: null / empty hidden state");
        return false;
    }
    const ::codec_lm_info * info = ::codec_lm_get_info(codec_lm);
    if (info == nullptr || info->is_continuous) {
        LOG_ERROR("tryCodecLmAudioStep: codec_lm is not codebook-AR");
        return false;
    }
    if (hidden_dim != info->hidden_dim) {
        LOG_ERROR("tryCodecLmAudioStep: hidden dim mismatch (got %d, expected %d)",
                  hidden_dim, info->hidden_dim);
        return false;
    }

    // Sampling params — shared between both paths.
    if (codec_lm_ar_rng == 0) {
        const auto & sp = main_ctx->params.sampling;
        codec_lm_ar_rng = sp.seed != 0 && sp.seed != (uint32_t) -1
            ? (uint64_t) sp.seed : 0xC0DEC1ABULL;
    }
    const auto & sp = main_ctx->params.sampling;
    const float   samp_temp = sp.temp > 0 ? sp.temp : 0.9f;
    const float   samp_topp = sp.top_p > 0 ? sp.top_p : 0.95f;
    const int32_t samp_topk = sp.top_k > 0 ? sp.top_k : 50;

    // ── Phase B path: audio_lm_* API ────────────────────────────────────
    // For Qwen3-TTS (talker), MOSS-TTSD (cb0_from_backbone), and
    // MOSS-TTS-Realtime (streaming_interleave) we route through the
    // codec_common audio_lm layer which owns: per-step step_begin/finish,
    // observe_codes (accumulation + EOS), and get_next_embed composition.
    // The audio_lm_ctx must have been initialised in the constructor.
    if (audio_lm_ctx != nullptr) {
        // Detect which sub-path we're on from the prompt_info.
        codec_common::audio_lm_prompt_info pi{};
        const bool have_pi = codec_common::audio_lm_get_prompt_info(audio_lm_ctx, &pi);
        const bool is_talker = codec_common::audio_lm_talker_has_projection(audio_lm_ctx);
        const bool is_cb0_bb = have_pi && pi.cb0_from_backbone;

        // MOSS-TTS-Realtime streaming_interleave path: the per-step input row
        // is text_embd[text_token] + compose_audio_embd(prev_codes), assembled
        // by try_audio_lm_step.  We signal this by checking pi.streaming_interleave.
        // For now the streaming_interleave path falls through to the audio_lm
        // step machine (the step machine is the same; the interleave part is
        // handled in the prefill setup — rn-completion.cpp isn't yet wired for
        // the full streaming-prefill).  TODO: add streaming prefill to rn-completion.
        if (is_talker || is_cb0_bb || (have_pi && pi.streaming_interleave)) {
            // Build GBNF grammar for cb0-from-backbone sampler if needed.
            std::string bb_grammar;
            if (is_cb0_bb) {
                bb_grammar = codec_common::tts_auto_grammar(pi, /*text=*/"");
            }

            std::vector<int32_t> codes;
            std::vector<float>   next_embd;
            bool is_eos = false;
            const char * stop_reason = nullptr;

            const bool ok = try_audio_lm_step(
                audio_lm_ctx,
                main_ctx->ctx, main_ctx->model,
                hidden, hidden_dim,
                is_cb0_bb,
                &bb_sampler, &bb_sampler_built,
                bb_grammar,
                samp_temp, samp_topk, samp_topp, &codec_lm_ar_rng,
                is_talker,
                talker_text_tokens, &talker_trailing,
                &next_embd, &codes, &is_eos, &stop_reason);

            if (!ok) {
                LOG_ERROR("tryCodecLmAudioStep: audio_lm step failed: %s",
                          codec_common::audio_lm_last_error(audio_lm_ctx));
                return false;
            }

            if (is_eos) {
                codec_lm_ar_done           = true;
                codec_lm_ar_stopped_on_eos = true;
                codec_lm_ar_pending_embd   = false;
                return true;
            }

            // Append codes to audio_tokens (T, n_cb interleaved).
            audio_tokens.insert(audio_tokens.end(), codes.begin(), codes.end());

            pending_next_embd      = std::move(next_embd);
            codec_lm_ar_pending_embd = true;
            codec_lm_ar_step += 1;
            return true;
        }
        // Fall through to legacy codec_lm_state_* path for models not
        // yet using the audio_lm layer (CSM, plain parallel-heads-delay
        // models without streaming_interleave).
    }

    // ── Legacy path: direct codec_lm_state_* API ────────────────────────
    // Used by CSM-1B / OuteTTS-adjacent parallel-heads models that don't
    // have prompt_info flags set.  Kept intact to avoid regressions.
    const int compose_ed = info->compose_audio_embed_dim > 0
        ? info->compose_audio_embed_dim : info->audio_embed_dim;
    if (compose_ed != info->hidden_dim) {
        LOG_ERROR("tryCodecLmAudioStep: compose_audio_embed_dim=%d != hidden=%d",
                  compose_ed, info->hidden_dim);
        return false;
    }

    if (codec_lm_state == nullptr) {
        codec_lm_state = ::codec_lm_state_new(codec_lm);
        if (codec_lm_state == nullptr) {
            LOG_ERROR("tryCodecLmAudioStep: codec_lm_state_new failed");
            return false;
        }
    }

    const int n_cb = info->n_codebook;
    const tts_model_profile & profile = profile_for_type(type);
    const bool text_modality_cb0 = (profile.audio_codebook_offset > 0);

    // Text-modality cb0 (old path — MOSS-TTSD / MOSS-TTS-Realtime when
    // audio_lm_ctx is unavailable or doesn't have the flags set).
    if (text_modality_cb0) {
        int32_t text_tok = backbone_sampled_tok;
        if (text_tok < 0) {
            const float * backbone_logits = ::llama_get_logits_ith(main_ctx->ctx, -1);
            if (backbone_logits != nullptr) {
                const int32_t n_vocab = (int32_t) llama_vocab_n_tokens(
                    llama_model_get_vocab(main_ctx->model));
                text_tok = sample_codec_logits(
                    backbone_logits, n_vocab, samp_temp,
                    samp_topk, samp_topp, &codec_lm_ar_rng);
            }
        }
        if (text_tok >= 0) {
            ::codec_lm_state_set_text_context(codec_lm_state, text_tok);
        }
    }

    if (::codec_lm_step_begin(codec_lm_state, hidden) != CODEC_STATUS_SUCCESS) {
        const char * err = ::codec_lm_state_get_last_error(codec_lm_state);
        LOG_ERROR("tryCodecLmAudioStep: codec_lm_step_begin failed: %s",
                  err && *err ? err : "(no error message)");
        return false;
    }

    for (int cb = 0; cb < n_cb; ++cb) {
        int32_t cb_idx = -1, vocab = 0;
        const float * logits = ::codec_lm_step_logits(codec_lm_state, &cb_idx, &vocab);
        if (logits == nullptr || vocab <= 0) {
            LOG_ERROR("tryCodecLmAudioStep: step_logits failed at cb=%d", cb);
            return false;
        }
        const int32_t code = sample_codec_logits(logits, vocab,
            samp_temp, samp_topk, samp_topp, &codec_lm_ar_rng);
        if (::codec_lm_step_push_code(codec_lm_state, code) != CODEC_STATUS_SUCCESS) {
            LOG_ERROR("tryCodecLmAudioStep: step_push_code failed at cb=%d code=%d",
                      cb, code);
            return false;
        }
    }

    std::vector<int32_t> codes((size_t) n_cb, 0);
    if (::codec_lm_step_finish(codec_lm_state, codes.data()) != CODEC_STATUS_SUCCESS) {
        LOG_ERROR("tryCodecLmAudioStep: codec_lm_step_finish failed");
        return false;
    }

    // Stop detection — metadata-driven EOS via codec_lm_step_is_eos,
    // falling back to CSM legacy heuristic for stale GGUFs.
    bool is_eos = false;
    if (info->eos_code_c0 >= 0) {
        int32_t eos_flag = 0;
        if (::codec_lm_step_is_eos(codec_lm_state, codes.data(), n_cb, &eos_flag)
                == CODEC_STATUS_SUCCESS) {
            is_eos = (eos_flag != 0);
        }
    } else if (type == CSM_1B) {
        is_eos = codec_lm_ar_step > 0 && codes[0] == 0;
    }
    if (is_eos) {
        codec_lm_ar_done           = true;
        codec_lm_ar_stopped_on_eos = true;
        codec_lm_ar_pending_embd   = false;
        return true;
    }

    audio_tokens.insert(audio_tokens.end(), codes.begin(), codes.end());

    pending_next_embd.assign((size_t) hidden_dim, 0.0f);
    if (::codec_lm_compose_next_embd(codec_lm, codes.data(),
                                     codec_lm_ar_step,
                                     pending_next_embd.data())
            != CODEC_STATUS_SUCCESS) {
        const char * err = ::codec_lm_get_last_error(codec_lm);
        LOG_ERROR("tryCodecLmAudioStep: compose_next_embd failed: %s",
                  err && *err ? err : "(no error message)");
        return false;
    }
    codec_lm_ar_pending_embd = true;
    codec_lm_ar_step += 1;
    return true;
}

// ─────────────────────────────────────────────────────────────────────
// Qwen3-TTS talker prefill hook.
//
// Called ONCE from rn-completion.cpp when flow == "talker_embd" and the
// prefill embd batch has already been decoded into the backbone KV cache.
// At this point `last_hidden` is the last row's backbone hidden state.
// We arm audio_lm for the embed-override path so audio_lm_observe_codes
// will compose the next backbone input embedding per step.
// ─────────────────────────────────────────────────────────────────────
bool llama_rn_context_tts::tryTalkerPrefill(
    llama_rn_context * main_ctx,
    const float *      last_hidden,
    int                hidden_dim) {
    (void) main_ctx;
    if (audio_lm_ctx == nullptr) {
        LOG_ERROR("tryTalkerPrefill: audio_lm_ctx not initialised");
        return false;
    }
    if (last_hidden == nullptr || hidden_dim <= 0) {
        LOG_ERROR("tryTalkerPrefill: null hidden state");
        return false;
    }
    // Enable embed-override: observe_codes will call compose_next_embd
    // and get_next_embed will return the composed embedding.  start_step=1
    // matches run_codebook_ar's convention.
    codec_common::audio_lm_set_uses_embed_override(audio_lm_ctx, true, 1);
    return true;
}

// ─────────────────────────────────────────────────────────────────────
// Chatterbox T3 prefill hook.
//
// Tokenizes `text` via codec_lm_chatterbox_tokenize, builds the full
// CFG prompt embedding prefix via codec_lm_chatterbox_build_prompt,
// and decodes it into two parallel KV-cache sequences (seq_id 0 = cond,
// seq_id 1 = uncond when cfg_weight > 0).  Sets chatterbox_n_seq and
// chatterbox_n_past on success.
// ─────────────────────────────────────────────────────────────────────
bool llama_rn_context_tts::tryChatterboxPrefill(
    llama_rn_context * main_ctx,
    const std::string & text,
    const float *       ref_pcm,
    int                 ref_n_samples,
    int                 ref_sample_rate,
    float               cfg_weight) {

    if (audio_lm_ctx == nullptr) {
        LOG_ERROR("tryChatterboxPrefill: audio_lm_ctx not initialised");
        return false;
    }
    if (main_ctx == nullptr || main_ctx->ctx == nullptr || main_ctx->model == nullptr) {
        LOG_ERROR("tryChatterboxPrefill: invalid backbone context");
        return false;
    }

    // Get the codec_lm handle from audio_lm_ctx.
    ::codec_lm * lm = codec_common::audio_lm_get_lm(audio_lm_ctx);
    if (lm == nullptr) {
        LOG_ERROR("tryChatterboxPrefill: no codec_lm inside audio_lm_ctx");
        return false;
    }
    const ::codec_lm_chatterbox_info * ci = ::codec_lm_chatterbox_get_info(lm);
    if (ci == nullptr) {
        LOG_ERROR("tryChatterboxPrefill: model is not a Chatterbox T3 adaptor");
        return false;
    }

    const int hidden = llama_model_n_embd(main_ctx->model);
    if (hidden <= 0) return false;

    // CFG needs seq_id 0 (cond) + seq_id 1 (uncond).  If the backbone context
    // was initialised with n_seq_max=1, we cannot use CFG — silently degrade.
    const uint32_t n_seq_max = llama_n_seq_max(main_ctx->ctx);
    if (n_seq_max < 2 && cfg_weight > 0.0f) {
        LOG_WARNING("tryChatterboxPrefill: cfg_weight=%.2f but n_seq_max=%u — disabling CFG",
                 cfg_weight, n_seq_max);
        cfg_weight = 0.0f;
    }

    // Tokenize text with the baked BPE.
    std::vector<int32_t> text_ids(text.size() + 64);
    int32_t n_text = 0;
    if (::codec_lm_chatterbox_tokenize(lm, text.c_str(),
                                        text_ids.data(), (int32_t)text_ids.size(),
                                        &n_text) != CODEC_STATUS_SUCCESS) {
        LOG_ERROR("tryChatterboxPrefill: tokenize failed: %s",
                  ::codec_lm_get_last_error(lm));
        return false;
    }
    text_ids.resize((size_t)n_text);

    const int32_t n_seq_cap = (cfg_weight > 0.0f) ? 2 : 1;
    const int32_t seq_len_cap = ci->cond_rows + (n_text + 2) + 2;
    std::vector<float> prompt((size_t)seq_len_cap * n_seq_cap * hidden);
    int32_t seq_len = 0, n_seq = 0;

    if (::codec_lm_chatterbox_build_prompt(
            lm,
            text_ids.data(), n_text,
            cfg_weight,
            /*speaker_emb=*/nullptr, 0,
            /*ref_speech_tokens=*/nullptr, 0,
            /*emotion=*/nullptr,
            ref_pcm, ref_n_samples, ref_sample_rate,
            prompt.data(), seq_len_cap * n_seq_cap,
            &seq_len, &n_seq) != CODEC_STATUS_SUCCESS) {
        LOG_ERROR("tryChatterboxPrefill: build_prompt failed: %s",
                  ::codec_lm_get_last_error(lm));
        return false;
    }

    // Decode: seq_id 0 = cond, seq_id 1 = uncond.  Both sequences share
    // the same positions [0, seq_len).  logits requested only at the last
    // position of each sequence (the start-of-speech position).
    const int32_t total = seq_len * n_seq;
    llama_batch b = llama_batch_init(total, hidden, 1);
    b.token    = nullptr;
    b.n_tokens = total;
    int32_t bi = 0;
    for (int32_t s = 0; s < n_seq; ++s) {
        for (int32_t r = 0; r < seq_len; ++r) {
            std::memcpy(b.embd + (size_t)bi * hidden,
                        prompt.data() + ((size_t)s * seq_len + r) * hidden,
                        (size_t)hidden * sizeof(float));
            b.pos[bi]       = r;
            b.n_seq_id[bi]  = 1;
            b.seq_id[bi][0] = s;
            b.logits[bi]    = (r == seq_len - 1) ? 1 : 0;
            ++bi;
        }
    }
    const int rc = llama_decode(main_ctx->ctx, b);
    llama_batch_free(b);
    if (rc != 0) {
        LOG_ERROR("tryChatterboxPrefill: llama_decode prefill failed");
        return false;
    }

    // Arm embed override and remember loop state.
    codec_common::audio_lm_set_uses_embed_override(audio_lm_ctx, true, 1);
    chatterbox_n_seq  = n_seq;
    chatterbox_n_past = seq_len;

    LOG_INFO("tryChatterboxPrefill: %d text tokens, seq_len=%d n_seq=%d",
             n_text, seq_len, n_seq);
    return true;
}

bool llama_rn_context_tts::tryContinuousAudioStep(
    llama_rn_context * main_ctx,
    const float *      hidden,
    int                hidden_dim) {
    (void) main_ctx;

    if (codec_lm == nullptr) {
        LOG_ERROR("tryContinuousAudioStep: codec_lm not created");
        return false;
    }
    if (hidden == nullptr || hidden_dim <= 0) {
        LOG_ERROR("tryContinuousAudioStep: null / empty hidden state");
        return false;
    }
    const ::codec_lm_info * info = ::codec_lm_get_info(codec_lm);
    if (info == nullptr || !info->is_continuous) {
        LOG_ERROR("tryContinuousAudioStep: codec_lm is not continuous-latent");
        return false;
    }
    if (hidden_dim != info->hidden_dim) {
        LOG_ERROR("tryContinuousAudioStep: hidden dim mismatch (got %d, expected %d)",
                  hidden_dim, info->hidden_dim);
        return false;
    }
    const int patch_size = info->patch_size;
    const int latent_dim = info->latent_dim;
    if (patch_size <= 0 || latent_dim <= 0) {
        LOG_ERROR("tryContinuousAudioStep: bad shape (patch=%d, latent=%d)",
                  patch_size, latent_dim);
        return false;
    }

    if (codec_lm_state == nullptr) {
        codec_lm_state = ::codec_lm_state_new(codec_lm);
        if (codec_lm_state == nullptr) {
            LOG_ERROR("tryContinuousAudioStep: codec_lm_state_new failed");
            return false;
        }
    }

    // BlueMagpie official release defaults (models/bluemagpie
    // release_metadata.json on the upstream HF repo): guidance_scale = 2.8,
    // num_timesteps = 9.  These replace the earlier VoxCPM-style 2.0/10.
    const float   cfg_value   = 2.8f;
    const int32_t n_timesteps = 9;

    // Per-step wall-clock timing so we can see where CPU goes on device.
    // Logged at INFO so `adb logcat` surfaces it; the constants module below
    // hosts the counters so the total-time summary fires when generation
    // finishes (either via stop head or n_predict cap).
    const auto now_us_local = []() -> int64_t {
        const auto t = std::chrono::steady_clock::now().time_since_epoch();
        return std::chrono::duration_cast<std::chrono::microseconds>(t).count();
    };
    const int64_t t_step_start = now_us_local();

    std::vector<float> patch((size_t) patch_size * (size_t) latent_dim, 0.0f);
    int32_t stop = 0;
    const int64_t t_gen_start = now_us_local();
    const enum codec_status rc = ::codec_lm_step_generate(
        codec_lm_state, hidden, cfg_value, n_timesteps,
        /*noise=*/nullptr, patch.data(), &stop);
    const int64_t t_gen_us = now_us_local() - t_gen_start;
    if (rc != CODEC_STATUS_SUCCESS) {
        const char * err = ::codec_lm_state_get_last_error(codec_lm_state);
        LOG_ERROR("tryContinuousAudioStep: step_generate failed: %s",
                  err && *err ? err : "(no error message)");
        return false;
    }

    // Accumulate patch (frame-major [T, latent_dim]).
    audio_embeddings.insert(audio_embeddings.end(), patch.begin(), patch.end());
    audio_embedding_dim = latent_dim;

    if (stop) {
        audio_embeddings_done = true;
        audio_embeddings_pending = false;
        const int step_idx = (int) (audio_embeddings.size() / (size_t) (patch_size * latent_dim)) - 1;
        LOG_INFO("continuous step %d: step_generate=%.1fms (STOP)",
                 step_idx, (double) t_gen_us / 1000.0);
        return true;
    }

    pending_feedback_embd.assign((size_t) hidden_dim, 0.0f);
    const int64_t t_fb_start = now_us_local();
    if (::codec_lm_step_feedback_embd(codec_lm_state, pending_feedback_embd.data())
            != CODEC_STATUS_SUCCESS) {
        LOG_ERROR("tryContinuousAudioStep: step_feedback_embd failed");
        return false;
    }
    const int64_t t_fb_us = now_us_local() - t_fb_start;
    const int64_t t_step_us = now_us_local() - t_step_start;
    // Log every N steps (compact) — every step is too spammy but we still
    // want a signal that things are progressing.  Sampling schedule: first 5,
    // then every 10 → catches early ramp-up + steady state.
    const int step_idx = (int) (audio_embeddings.size() / (size_t) (patch_size * latent_dim)) - 1;
    if (step_idx < 5 || (step_idx % 10) == 0) {
        LOG_INFO("continuous step %d: step_generate=%.1fms feedback=%.1fms total=%.1fms",
                 step_idx,
                 (double) t_gen_us / 1000.0,
                 (double) t_fb_us / 1000.0,
                 (double) t_step_us / 1000.0);
    }
    audio_embeddings_pending = true;
    return true;
}

bool llama_rn_context_tts::tryContinuousPrefill(
    llama_rn_context * main_ctx,
    const float *      hiddens,
    int                n_pos,
    int                dim) {
    (void) main_ctx;

    if (codec_lm == nullptr) {
        LOG_ERROR("tryContinuousPrefill: codec_lm not created");
        return false;
    }
    if (hiddens == nullptr || n_pos <= 0 || dim <= 0) {
        LOG_ERROR("tryContinuousPrefill: null / empty hiddens (n_pos=%d, dim=%d)",
                  n_pos, dim);
        return false;
    }
    const ::codec_lm_info * info = ::codec_lm_get_info(codec_lm);
    if (info == nullptr || !info->is_continuous) {
        LOG_ERROR("tryContinuousPrefill: codec_lm is not continuous-latent");
        return false;
    }
    if (dim != info->hidden_dim) {
        LOG_ERROR("tryContinuousPrefill: hidden dim mismatch (got %d, expected %d)",
                  dim, info->hidden_dim);
        return false;
    }

    if (codec_lm_state == nullptr) {
        codec_lm_state = ::codec_lm_state_new(codec_lm);
        if (codec_lm_state == nullptr) {
            LOG_ERROR("tryContinuousPrefill: codec_lm_state_new failed");
            return false;
        }
    }

    const enum codec_status rc =
        ::codec_lm_text_prefill(codec_lm_state, hiddens, n_pos, dim);
    if (rc != CODEC_STATUS_SUCCESS) {
        const char * err = ::codec_lm_state_get_last_error(codec_lm_state);
        LOG_ERROR("tryContinuousPrefill: codec_lm_text_prefill failed: %s",
                  err && *err ? err : "(no error message)");
        return false;
    }
    LOG_INFO("continuous prefill: seeded RALM K/V for %d prompt positions", n_pos);
    return true;
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

// Read a scalar int32 GGUF metadata value from the codec model, returning
// `fallback` when the key is absent.  Mirrors codec_common's `meta_str` +
// std::atoi used by audio_lm's decode transform.
static int32_t codec_meta_i32(::codec_model *codec_model, const char *key, int32_t fallback) {
    if (codec_model == nullptr || key == nullptr) return fallback;
    const struct codec_lm_gguf_metadata *meta = codec_model_metadata(codec_model);
    if (meta == nullptr) return fallback;
    for (size_t i = 0; i < meta->n_items; ++i) {
        if (meta->items[i].key != nullptr && std::strcmp(meta->items[i].key, key) == 0 &&
            meta->items[i].value != nullptr) {
            return (int32_t) std::atoi(meta->items[i].value);
        }
    }
    return fallback;
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
        // Chatterbox T3 with audio_lm path active decodes via audio_lm_decode_audio.
        if (audio_lm_ctx != nullptr) {
            codec_common::audio_lm_audio_output pcm_out;
            if (!codec_common::audio_lm_decode_audio(audio_lm_ctx, &pcm_out)) {
                LOG_ERROR("decodeAudioTokens: audio_lm_decode_audio failed: %s",
                          codec_common::audio_lm_last_error(audio_lm_ctx));
                return {};
            }
            return pcm_out.pcm;
        }
        LOG_ERROR("This TTS model's codec is not supported by codec.cpp yet");
        return std::vector<float>();
    }

    // For codec_lm-AR models that used the audio_lm_ctx path (Qwen3-TTS /
    // MOSS-TTSD / MOSS-TTS-Realtime), audio_lm_decode_audio reads the
    // internal accumulator filled by audio_lm_observe_codes, applies the
    // correct delay-pattern unshift and cb0_speech_offset remapping, and
    // calls codec_decode.  This is the codec_common-canonical decode path
    // and avoids the duplicate logic below.
    if (profile.decode_kind == tts_decode_kind::CODEC_LM_AR && audio_lm_ctx != nullptr) {
        // Check that we actually used the audio_lm step machine (not the
        // legacy codec_lm_state path): the audio_lm accumulator has codes
        // iff observe_codes was called at least once.
        codec_common::audio_lm_prompt_info pi{};
        const bool have_pi = codec_common::audio_lm_get_prompt_info(audio_lm_ctx, &pi);
        const bool used_alm_path = have_pi && (
            codec_common::audio_lm_talker_has_projection(audio_lm_ctx) ||
            pi.cb0_from_backbone ||
            pi.streaming_interleave);
        if (used_alm_path) {
            codec_common::audio_lm_audio_output pcm_out;
            if (!codec_common::audio_lm_decode_audio(audio_lm_ctx, &pcm_out)) {
                LOG_ERROR("decodeAudioTokens: audio_lm_decode_audio failed: %s",
                          codec_common::audio_lm_last_error(audio_lm_ctx));
                return {};
            }
            if (!pcm_out.pcm.empty()) return pcm_out.pcm;
            // Empty PCM from audio_lm_decode_audio — fall through to direct path
            // (may happen if the accumulator is empty due to EOS on frame 0).
            LOG_WARNING("decodeAudioTokens: audio_lm_decode_audio returned empty PCM; falling back to direct path");
        }
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

    // NOTE: this codes->PCM transform is kept in lock-step with the
    // metadata-driven `audio_lm_decode_audio` in
    // cpp/codec/common/audio_lm.cpp (upstream codec.cpp).  rn drives its own
    // AR loop and calls codec_decode directly rather than routing through an
    // audio_lm_context, so the offset/delay-unshift/cb0_speech_offset/clamp
    // logic is mirrored here.  If upstream's transform changes, update both.
    //
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

    // Merged text+speech cb0 remap (MOSS-TTSD): subtract speech_token_range[0]
    // from the first audio codebook so the merged vocab maps back to raw
    // quantizer index space (mirrors codec_common audio_lm_decode_audio's
    // cb0_speech_offset handling; the key is written by the MOSS converter).
    // Absent (== 0) for CSM / Qwen3-TTS / Realtime, so those are unaffected.
    const int32_t cb0_speech_offset = codec_meta_i32(codec_model, "codec.lm.cb0_speech_offset", 0);
    const int32_t codebook_sz = codec_model_codebook_size(codec_model);

    std::vector<int32_t> codec_tokens;
    if (audio_cb_off > 0 || max_delay > 0 || cb0_speech_offset != 0) {
        codec_tokens.resize(n_frames_aligned * (size_t) n_q);
        for (size_t t = 0; t < n_frames_aligned; ++t) {
            for (int q = 0; q < n_q; ++q) {
                const size_t src_t = t + (size_t) audio_delays[(size_t) q];
                int32_t code = (int32_t) tokens_audio[src_t * (size_t) n_cb_in + (size_t) (audio_cb_off + q)];
                if (q == 0 && cb0_speech_offset != 0) {
                    code -= cb0_speech_offset;
                }
                // Guard the codec's get_rows against pad / control codes
                // (speech_pad, bos/eos sentinels) the LM can emit before stop;
                // the HF processor drops such frames — we clamp into range.
                if (codebook_sz > 0) {
                    if (code < 0)             code = 0;
                    if (code >= codebook_sz)  code = codebook_sz - 1;
                }
                codec_tokens[t * (size_t) n_q + (size_t) q] = code;
            }
        }
    } else {
        codec_tokens.assign(tokens_audio.begin(), tokens_audio.end());
    }
    struct codec_token_buffer token_buffer = {};
    token_buffer.data = codec_tokens.data();
    token_buffer.n_tokens = (int32_t)codec_tokens.size();
    // Delay-unshift trims max_delay tail frames; the buffer carries
    // n_frames_aligned frames (== n_frames when no delay).  Upstream
    // audio_lm_decode_audio uses the trimmed count (n_frames_out) here.
    token_buffer.n_frames = (int32_t) n_frames_aligned;
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
    // The accumulator is frame-major [T, dim] (one row per latent frame), but
    // codec_decode_quantized_representation expects channel-major [dim, T]
    // (data[d * n_frames + t]) — same convention as codec_common's
    // audio_lm_decode_audio, which transposes before decoding.  Without this
    // transpose the AudioVAE decodes a scrambled latent matrix: the output
    // still *sounds* like fluent speech but says entirely wrong words.
    std::vector<float> chan_major((size_t) embedding_dim * (size_t) n_frames);
    for (int t = 0; t < n_frames; ++t) {
        for (int d = 0; d < embedding_dim; ++d) {
            chan_major[(size_t) d * n_frames + t] = embeddings[(size_t) t * embedding_dim + d];
        }
    }
    const enum codec_status status = codec_decode_quantized_representation(
        codec_ctx,
        chan_major.data(),
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
