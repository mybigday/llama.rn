#include "rn-decision.h"

#include "llama-ext.h" // staging API: llama_decision_order
#include "rn-mtmd.hpp" // base64_decode

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace rnllama {

using json = common_json;

static const char * decision_question_type_name(llama_rn_decision_question_type type) {
    switch (type) {
        case LLAMA_RN_DECISION_QUESTION_CHOICE: return "choice";
        case LLAMA_RN_DECISION_QUESTION_SCORE:  return "score";
        case LLAMA_RN_DECISION_QUESTION_NOUL:   return "noul";
    }
    return "";
}

static std::string decision_meta_str(const llama_model * model, const std::string & key) {
    char buf[256];
    const int32_t n = llama_model_meta_val_str(model, key.c_str(), buf, sizeof(buf));
    return n < 0 ? "" : std::string(buf);
}

//
// profiles
//

const char * llama_rn_decision_type_name(common_decision_type type) {
    switch (type) {
        case COMMON_DECISION_TYPE_OPENJEV: return "openjev";
        case COMMON_DECISION_TYPE_LEV:     return "lev";
        case COMMON_DECISION_TYPE_KEV:     return "kev";
        case COMMON_DECISION_TYPE_NIMBLE:  return "nimble";
        case COMMON_DECISION_TYPE_LAYA:    return "laya";
        case COMMON_DECISION_TYPE_CLEF:    return "clef";
        case COMMON_DECISION_TYPE_PPLX_DECIDER: return "pplx-decider";
        case COMMON_DECISION_TYPE_VLLM_DECISION2: return "vllm-decision2";
        case COMMON_DECISION_TYPE_VLLM_D3:      return "vllm-d3";
        case COMMON_DECISION_TYPE_LFM2_D1:      return "lfm2-d1";
        case COMMON_DECISION_TYPE_LFM2_D1_OMNI: return "lfm2-d1-omni";
        case COMMON_DECISION_TYPE_GRANITE_SWITCH: return "granite-switch";
        case COMMON_DECISION_TYPE_H2O_LIGHTNING:  return "h2o-lightning";
        case COMMON_DECISION_TYPE_MINICPM_OCR_JEV: return "openbmb-minicpm-ocr-jev";
        case COMMON_DECISION_TYPE_LFM3_D1:      return "lfm3-d1";
        case COMMON_DECISION_TYPE_LFM3_D1_OMNI: return "lfm3-d1-omni";
        case COMMON_DECISION_TYPE_UNKNOWN: return "unknown";
        default:                           return "";
    }
}

bool llama_rn_decision_profile_for(common_decision_type type, llama_rn_decision_profile & p) {
    p = llama_rn_decision_profile();
    switch (type) {
        case COMMON_DECISION_TYPE_OPENJEV:
            p.readout         = LLAMA_RN_DECISION_READOUT_LABEL_LOGITS;
            p.labels          = LLAMA_RN_DECISION_LABELS_LETTERS;
            p.noul_true_first = true;
            p.image_input     = true;
            return true;
        case COMMON_DECISION_TYPE_LEV:
            p.readout             = LLAMA_RN_DECISION_READOUT_LABEL_LOGITS;
            p.labels              = LLAMA_RN_DECISION_LABELS_CODES;
            p.label_texts         = true;
            p.sort_keys           = true; // lev was trained with sorted keys
            p.reverse_variant     = true; // to cancel the preference for the first label
            p.n_noul_ratings      = 9;    // 0 = certainly no, 8 = certainly yes
            p.temperature_buckets = LLAMA_RN_DECISION_TEMPERATURE_BUCKETS_SIZE;
            return true;
        case COMMON_DECISION_TYPE_NIMBLE:
            p.readout        = LLAMA_RN_DECISION_READOUT_LABEL_LOGITS;
            p.labels         = LLAMA_RN_DECISION_LABELS_CODES;
            p.label_texts    = true;
            p.list_questions = true;
            return true;
        case COMMON_DECISION_TYPE_KEV:
            p.readout     = LLAMA_RN_DECISION_READOUT_POINTER;
            p.marker      = LLAMA_RN_DECISION_MARKER_TEXT;
            p.marker_text = "<|box_end|>"; // the hidden state of an option is read at the token that ends it
            p.text_input  = true;
            return true;
        case COMMON_DECISION_TYPE_LAYA:
            p.readout       = LLAMA_RN_DECISION_READOUT_MARKER_EMBD;
            p.marker        = LLAMA_RN_DECISION_MARKER_VOCAB_MASK;
            p.truncate_head = true;
            return true;
        case COMMON_DECISION_TYPE_CLEF:
            p.readout         = LLAMA_RN_DECISION_READOUT_JOINT;
            p.sort_keys       = true;
            p.noul_true_first = true;
            p.choice_sorted   = true;
            p.image_input     = true;
            return true;
        case COMMON_DECISION_TYPE_PPLX_DECIDER:
            p.readout     = LLAMA_RN_DECISION_READOUT_LABEL_LOGITS;
            p.labels      = LLAMA_RN_DECISION_LABELS_CODES;
            p.label_texts = true;
            p.image_input = true;
            return true;
        case COMMON_DECISION_TYPE_LFM2_D1:
            p.readout         = LLAMA_RN_DECISION_READOUT_LABEL_LOGITS;
            p.labels          = LLAMA_RN_DECISION_LABELS_PER_QUESTION;
            p.noul_true_first = true;
            p.image_input     = true;
            p.null_state      = true;
            return true;
        case COMMON_DECISION_TYPE_LFM2_D1_OMNI:
            p.readout          = LLAMA_RN_DECISION_READOUT_MARKER_EMBD;
            p.marker           = LLAMA_RN_DECISION_MARKER_VOCAB_MASK;
            p.image_input      = true;
            p.audio_input      = true;
            p.null_state       = true;
            p.noul_yes_no      = true;
            p.escape_special   = true;
            p.media_raw_probs  = true; // image and audio answers are not calibrated
            // max_length, image_text_length and audio_text_length of the model config
            p.max_tokens       = 16384;
            p.max_tokens_image = 896;
            p.max_tokens_audio = 15360;
            return true;
        case COMMON_DECISION_TYPE_VLLM_DECISION2:
            p.readout            = LLAMA_RN_DECISION_READOUT_JOINT;
            p.joint_per_question = true; // each question has its own prompt, with the spans that the head reads
            p.sort_keys          = true;
            p.noul_criteria_order = true;
            p.strict_payload     = true;
            p.canonical_legend   = true;
            return true;
        case COMMON_DECISION_TYPE_VLLM_D3:
            p.readout          = LLAMA_RN_DECISION_READOUT_LABEL_LOGITS;
            p.labels           = LLAMA_RN_DECISION_LABELS_CODES;
            p.label_texts      = true;
            p.image_input      = true;
            p.null_state       = true;
            p.canonical_legend = true;
            p.confidence_entropy = true;
            p.optional_instructions = true; // the prompt has a default question
            p.no_prefix_cache  = true; // non-causal attention: a prefix says nothing about what follows it
            return true;
        case COMMON_DECISION_TYPE_GRANITE_SWITCH:
            // same as openjev, run by the decision aLoRA of granite switch, with code labels
            p.readout         = LLAMA_RN_DECISION_READOUT_LABEL_LOGITS;
            p.labels          = LLAMA_RN_DECISION_LABELS_CODES;
            p.label_texts     = true;
            p.noul_true_first = true;
            return true;
        case COMMON_DECISION_TYPE_H2O_LIGHTNING:
            // same as openjev, the labels are the model's own, read after an answer prefill
            p.readout         = LLAMA_RN_DECISION_READOUT_LABEL_LOGITS;
            p.labels          = LLAMA_RN_DECISION_LABELS_MODEL;
            p.label_texts     = true;
            p.noul_true_first = true;
            p.noul_yes_no     = true;
            p.image_input     = true;
            p.h2o_render      = true;
            p.optional_instructions = true; // the template has a default, the type defaults to choice
            p.score_levels_min = 1;         // the maximum is the number of labels, set at init
            p.quoted_attrs    = LLAMA_TOKEN_ATTR_CONTROL | LLAMA_TOKEN_ATTR_USER_DEFINED;
            p.quoted_min_size = 1;
            return true;
        case COMMON_DECISION_TYPE_MINICPM_OCR_JEV:
            // one score per class, the pooled embeddings output of the last token
            p.readout          = LLAMA_RN_DECISION_READOUT_POOLED;
            p.choice_sorted    = true; // the classes are fixed: "0" inconsistent, "1" consistent
            p.image_input      = true;
            p.single_question  = true;
            p.forbids_instructions = true; // the task is part of the prompt
            p.state_must_be_text   = true; // the OCR text
            p.n_images_required    = 2;    // the original image, then the one rendered from the OCR text
            return true;
        case COMMON_DECISION_TYPE_LFM3_D1:
        case COMMON_DECISION_TYPE_LFM3_D1_OMNI:
            // same as lfm2-d1, non-causal, the texts of the special tokens are broken in the input
            p.readout         = LLAMA_RN_DECISION_READOUT_LABEL_LOGITS;
            p.labels          = LLAMA_RN_DECISION_LABELS_PER_QUESTION;
            p.labels_alpha2   = true; // codes are A..Z, then AA, AB..
            p.noul_true_first = true;
            p.image_input     = true;
            p.null_state      = true;
            p.video_input     = true; // a state part may be a video, seen frame by frame
            p.break_special   = true;
            p.score_median     = true;
            p.confidence_max_prob = true;
            p.no_prefix_cache  = true; // the short convs have no memory to separate the sequences
            p.quoted_attrs    = LLAMA_TOKEN_ATTR_CONTROL;
            p.quoted_min_size = 2;
            if (type == COMMON_DECISION_TYPE_LFM3_D1_OMNI) {
                p.audio_input      = true;
                p.spoken_questions = true; // a spoken question, its instructions are an input_audio part
            } else {
                p.forbids_spoken   = true;
            }
            return true;
        default:
            return false;
    }
}

//
// model-specific setup
//

void llama_rn_decision_context::init(const llama_model * model) {
    *this = llama_rn_decision_context(); // the model can be reloaded

    const common_decision_type model_type = common_get_decision_type(model);
    declared_type = model_type;
    if (model_type == COMMON_DECISION_TYPE_NONE) {
        // a declared type wins, a system_one template is the fallback
        const char * legacy_src = llama_model_chat_template(model, "system_one");
        if (legacy_src != nullptr) {
            init_legacy(model, legacy_src);
        }
        return;
    }
    if (!llama_rn_decision_profile_for(model_type, profile)) {
        type  = COMMON_DECISION_TYPE_UNKNOWN;
        error = "this build does not support the decision type \"" +
                decision_meta_str(model, decision_meta_str(model, "general.architecture") + ".decision.type") + "\"";
        return;
    }

    const std::string prefix = decision_meta_str(model, "general.architecture") + ".decision.";

    vocab = llama_model_get_vocab(model);

    const char * tmpl_src = llama_model_chat_template(model, "systemone");
    if (tmpl_src == nullptr) {
        throw std::runtime_error("decision model has no \"systemone\" template");
    }
    tmpl = std::make_shared<const common_chat_template>(tmpl_src, "", "");

    // a label model whose template has a place for images takes them, whatever its type says
    // (a type's own prompt may have none, e.g. nimble's); fill_prompt() checks that it uses them
    if (profile.readout == LLAMA_RN_DECISION_READOUT_LABEL_LOGITS && std::string(tmpl_src).find("images") != std::string::npos) {
        profile.image_input = true;
    }

    const std::string prefix_temp = prefix + "temperature.";
    for (int32_t i = 0; i < llama_model_meta_count(model); i++) {
        char key[256];
        char val[64];
        if (llama_model_meta_key_by_index(model, i, key, sizeof(key)) < 0 || !string_starts_with(key, prefix_temp)) {
            continue;
        }
        if (llama_model_meta_val_str_by_index(model, i, val, sizeof(val)) < 0) {
            continue;
        }
        const float temp = std::strtof(val, nullptr);
        if (temp <= 0.0f) {
            throw std::runtime_error(string_format("invalid decision temperature: %s = %s", key, val));
        }
        temperatures[key + prefix_temp.size()] = temp;
    }

    // "<n_levels>.<level>" -> offset
    const std::string prefix_bias = prefix + "score_bias.";
    for (int32_t i = 0; i < llama_model_meta_count(model); i++) {
        char key[256];
        char val[64];
        if (llama_model_meta_key_by_index(model, i, key, sizeof(key)) < 0 || !string_starts_with(key, prefix_bias)) {
            continue;
        }
        if (llama_model_meta_val_str_by_index(model, i, val, sizeof(val)) < 0) {
            continue;
        }
        const auto parts = string_split<std::string>(key + prefix_bias.size(), '.');
        const size_t n     = parts.size() == 2 ? std::strtoul(parts[0].c_str(), nullptr, 10) : 0;
        const size_t level = parts.size() == 2 ? std::strtoul(parts[1].c_str(), nullptr, 10) : 0;
        if (n < 2 || n > 255 || level >= n) {
            throw std::runtime_error(string_format("invalid decision score bias: %s", key));
        }
        auto & bias = score_bias[n];
        bias.resize(n, 0.0f);
        bias[level] = std::strtof(val, nullptr);
    }

    switch (profile.labels) {
        case LLAMA_RN_DECISION_LABELS_LETTERS: {
            const std::string letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
            for (const char c : letters) {
                const auto toks = common_tokenize(vocab, std::string(1, c), false, false);
                if (toks.size() != 1) {
                    throw std::runtime_error(string_format("decision label '%c' is not a single token", c));
                }
                labels.push_back(toks[0]);
            }
        } break;
        case LLAMA_RN_DECISION_LABELS_CODES: {
            std::vector<std::string> codes;
            for (char a = 'A'; a <= 'Z'; a++) {
                codes.push_back(std::string(1, a));
            }
            for (char a = 'A'; a <= 'Z'; a++) {
                for (char b = 'A'; b <= 'Z'; b++) {
                    codes.push_back(std::string{a, b});
                }
            }
            for (const auto & code : codes) {
                const auto toks = common_tokenize(vocab, code, false, false);
                if (toks.size() == 1 && labels.size() < 255) {
                    labels.push_back(toks[0]);
                    if (profile.label_texts) {
                        label_texts.push_back(code);
                    }
                }
            }
        } break;
        case LLAMA_RN_DECISION_LABELS_MODEL: {
            // the answer is read at the token of a label, its text is given to the template
            for (uint32_t i = 0; llama_model_decision_label(model, i) != nullptr; i++) {
                const std::string text = llama_model_decision_label(model, i);
                const auto toks = common_tokenize(vocab, text, false, false);
                if (toks.size() != 1 || std::find(labels.begin(), labels.end(), toks[0]) != labels.end()) {
                    throw std::runtime_error(string_format("decision label \"%s\" is not a single distinct token", text.c_str()));
                }
                labels.push_back(toks[0]);
                label_texts.push_back(text);
            }
            if (labels.empty()) {
                throw std::runtime_error("decision model has no labels");
            }
        } break;
        case LLAMA_RN_DECISION_LABELS_PER_QUESTION:
        case LLAMA_RN_DECISION_LABELS_NONE:
            break;
    }

    if (profile.quoted_attrs != 0) {
        init_quoted_spellings(profile.quoted_attrs, profile.quoted_min_size);
    }

    switch (profile.marker) {
        case LLAMA_RN_DECISION_MARKER_TEXT: {
            const auto toks = common_tokenize(vocab, profile.marker_text, false, true);
            if (toks.size() != 1) {
                throw std::runtime_error("decision model has no " + profile.marker_text + " token");
            }
            token_marker = toks[0];
        } break;
        case LLAMA_RN_DECISION_MARKER_VOCAB_MASK: {
            token_marker = llama_vocab_mask(vocab);
            if (token_marker == LLAMA_TOKEN_NULL) {
                throw std::runtime_error("decision model has no mask token");
            }
            if (profile.truncate_head) {
                token_sep = llama_vocab_sep(vocab);
                if (token_sep == LLAMA_TOKEN_NULL) {
                    throw std::runtime_error("decision model has no sep token");
                }
                // the input must not contain the marker of the options (lfm2-d1-omni escapes it instead)
                text_marker = common_token_to_piece(vocab, token_marker, true);
            }
        } break;
        case LLAMA_RN_DECISION_MARKER_NONE:
            break;
    }

    if (profile.truncate_head) {
        const std::string val = decision_meta_str(model, prefix + "max_head_tokens");
        max_head_tokens = std::strtoul(val.c_str(), nullptr, 10);
        if (max_head_tokens == 0) {
            throw std::runtime_error("decision model has no valid max_head_tokens");
        }
    }

    if (model_type == COMMON_DECISION_TYPE_MINICPM_OCR_JEV) {
        // the classes come from the GGUF classifier labels, the request API is binary: [inconsistent, consistent]
        const uint32_t n_classes = llama_model_n_cls_out(model);
        if (n_classes != 2 || llama_model_n_embd_out(model) != (int32_t) n_classes) {
            throw std::runtime_error(string_format("decision model must have 2 classifier labels and output one score per class, got %u labels and %d scores",
                n_classes, llama_model_n_embd_out(model)));
        }
        n_options_max = n_classes;
    } else {
        n_options_max = profile.readout == LLAMA_RN_DECISION_READOUT_LABEL_LOGITS && profile.labels != LLAMA_RN_DECISION_LABELS_PER_QUESTION
            ? labels.size() : 255;
    }
    if (model_type == COMMON_DECISION_TYPE_H2O_LIGHTNING) {
        // a score question has at most one level per label
        profile.score_levels_max = labels.size();
    }
    type = model_type;
}

void llama_rn_decision_context::init_quoted_spellings(uint32_t attrs, size_t min_size) {
    for (llama_token id = 0; id < llama_vocab_n_tokens(vocab); id++) {
        if (llama_vocab_get_attr(vocab, id) & attrs) {
            const std::string text = llama_vocab_get_text(vocab, id);
            if (text.size() >= min_size) {
                quoted_spellings.push_back(text);
            }
        }
    }
    std::stable_sort(quoted_spellings.begin(), quoted_spellings.end(), [](const std::string & a, const std::string & b) {
        return a.size() > b.size();
    });
    for (const auto & spelling : quoted_spellings) {
        if (quoted_first.find(spelling[0]) == std::string::npos) {
            quoted_first += spelling[0];
        }
    }
}

json llama_rn_decision_context::info() const {
    if (!is_decision_model()) {
        return nullptr;
    }
    json out = json{
        {"type",           legacy ? "system_one" : llama_rn_decision_type_name(type)},
        {"nOptionsMax",    n_options_max},
        {"imageInput",     profile.image_input},
        {"audioInput",     profile.audio_input},
        {"textGeneration", can_generate_text()},
    };
    if (legacy) {
        out["readout"] = profile.readout == LLAMA_RN_DECISION_READOUT_RANK_HEAD ? "rank_head" : "letter_slot";
    }
    if (!error.empty()) {
        out["error"] = error;
    }
    return out;
}

size_t llama_rn_decision_prompt::n_tokens() const {
    size_t n = tokens.size();
    for (const auto & piece : chunks) {
        n += mtmd_helper_get_n_tokens(piece.get());
    }
    if (media) {
        n += mtmd_helper_get_n_tokens(media.get());
    }
    for (const auto & piece : state_pieces) {
        n += piece.chunks ? mtmd_helper_get_n_tokens(piece.chunks.get()) : piece.tokens.size();
    }
    return n;
}

//
// system_one (legacy)
//

static void decision_load_bitmaps(
        mtmd_context * mctx,
        const std::vector<std::string> & images,
        mtmd::bitmaps & bitmaps,
        bool allow_video = false,
        std::vector<mtmd_helper_video *> * videos = nullptr);
static std::shared_ptr<mtmd_input_chunks> decision_tokenize_media(
        mtmd_context * mctx,
        const std::string & prompt,
        const std::vector<const mtmd_bitmap *> & bitmaps,
        bool add_special);

// The readout follows from what the model is, nothing declares it (system-one.cpp:
// system_one_params_from_model): a classification head scores one sequence per option, a
// causal model answers at the next token after each question. The other readouts the library
// knows need graphs that only its own fork has.
void llama_rn_decision_context::init_legacy(const llama_model * model, const char * tmpl_src) {
    legacy = true;
    vocab  = llama_model_get_vocab(model);
    tmpl   = std::make_shared<const common_chat_template>(tmpl_src, "", "");

    const std::string arch = decision_meta_str(model, "general.architecture");
    const std::string causal_str = decision_meta_str(model, arch + ".attention.causal");
    const bool causal = causal_str.empty() || causal_str == "true" || causal_str == "1";
    const bool has_cls_head = llama_model_cls_label(model, 0) != nullptr;
    const bool has_pointer  = !decision_meta_str(model, arch + ".decision_head.pointer_dim").empty();

    if (has_cls_head) {
        profile.readout = LLAMA_RN_DECISION_READOUT_RANK_HEAD;
    } else if (has_pointer || !causal) {
        legacy = false;
        type   = COMMON_DECISION_TYPE_UNKNOWN;
        error  = has_pointer
            ? "this system_one model reads a pointer head (kev_pointer), which needs a graph llama.cpp does not have"
            : "this system_one model is bidirectional (masked_slot / scored_slot), which needs a graph llama.cpp does not have";
        return;
    } else {
        profile.readout     = LLAMA_RN_DECISION_READOUT_LETTER_SLOTS;
        profile.image_input = true; // a state may carry images, with the model's projector
    }

    const std::string sep = decision_meta_str(model, "system_one.segment_separator");
    if (!sep.empty()) {
        legacy_separator = sep;
    }

    // a comma-joined string, a label may begin with a space (system-one.cpp: split_array)
    std::string labels_str = decision_meta_str(model, "system_one.labels");
    const bool bracketed = labels_str.size() >= 2 && labels_str.front() == '[' && labels_str.back() == ']';
    for (const auto & piece : string_split(bracketed ? labels_str.substr(1, labels_str.size() - 2) : labels_str, ",")) {
        std::string label;
        for (const char c : piece) {
            if (c == '"' || (bracketed && c == ' ')) {
                continue;
            }
            label += c;
        }
        if (!label.empty()) {
            label_texts.push_back(label);
        }
    }
    if (label_texts.empty()) {
        labels_are_default = true;
        for (char c = 'A'; c <= 'Z'; c++) label_texts.push_back(std::string(1, c));
        for (char c = 'a'; c <= 'z'; c++) label_texts.push_back(std::string(1, c));
    }

    if (profile.readout == LLAMA_RN_DECISION_READOUT_LETTER_SLOTS) {
        for (const auto & label : label_texts) {
            const auto toks = common_tokenize(vocab, label, false, false);
            if (toks.size() != 1) {
                throw std::runtime_error("the answer label \"" + label + "\" is not a single token for this tokenizer" +
                    (labels_are_default ? " (it is the default A-Za-z, set system_one.labels)" : ""));
            }
            labels.push_back(toks[0]);
        }
        n_options_max = labels.size();
    } else {
        n_options_max = 255;
    }

    // only what the model declares, the template writes them
    auto piece = [&](llama_token id) {
        const char * text = id >= 0 ? llama_vocab_get_text(vocab, id) : nullptr;
        return text ? std::string(text) : std::string();
    };
    if (!decision_meta_str(model, "tokenizer.ggml.bos_token_id").empty()) {
        bos_text = piece(llama_vocab_bos(vocab));
    }
    if (!decision_meta_str(model, "tokenizer.ggml.eos_token_id").empty()) {
        eos_text = piece(llama_vocab_eos(vocab));
    }
}

bool llama_rn_decision_context::can_generate_text() const {
    // common_init puts these in embedding mode by their declared type, even when this build
    // cannot use them for decisions
    switch (declared_type) {
        case COMMON_DECISION_TYPE_LAYA:
        case COMMON_DECISION_TYPE_KEV:
        case COMMON_DECISION_TYPE_CLEF:
        case COMMON_DECISION_TYPE_LFM2_D1_OMNI:
        case COMMON_DECISION_TYPE_VLLM_DECISION2:
        case COMMON_DECISION_TYPE_MINICPM_OCR_JEV:
            return false;
        default:
            break;
    }
    // a classification head is read through RANK pooling, which has no logits
    return !(legacy && profile.readout == LLAMA_RN_DECISION_READOUT_RANK_HEAD);
}

void llama_rn_decision_context::check_context(llama_context * ctx) const {
    if (profile.readout == LLAMA_RN_DECISION_READOUT_RANK_HEAD && llama_pooling_type(ctx) != LLAMA_POOLING_TYPE_RANK) {
        throw std::runtime_error("this model answers with its classification head: initialize the context "
                                 "with pooling_type: 'rank' and embedding: true");
    }
}

// render, then split on the separator, dropping empty segments (system-one.cpp: render_and_split)
std::vector<std::string> llama_rn_decision_context::render_legacy(const common_json & inp) const {
    std::string rendered;
    try {
        jinja::context ctx(tmpl->source());
        jinja::global_from_json(ctx, inp, false);
        jinja::runtime runtime(ctx);
        rendered = jinja::runtime::gather_string_parts(runtime.execute(tmpl->prog))->as_string().str();
    } catch (const std::exception & e) {
        throw std::invalid_argument(std::string("system_one template failed to render: ") + e.what());
    }
    std::vector<std::string> segments;
    for (auto & segment : string_split(rendered, legacy_separator)) {
        if (!segment.empty()) {
            segments.push_back(std::move(segment));
        }
    }
    return segments;
}

static std::string decision_legacy_text(const common_json & val) {
    return val.is_string() ? val.get<std::string>() : val.dump();
}

static const char * decision_legacy_kind(llama_rn_decision_question_type type) {
    return type == LLAMA_RN_DECISION_QUESTION_NOUL ? "noul" : type == LLAMA_RN_DECISION_QUESTION_CHOICE ? "choice" : "score";
}

// The template namespace of the system_one library: an option is {label, option, desc}, where a
// noul's sides are "no" and "yes", a choice's are its keys and a score's are its level texts
static common_json decision_legacy_option(const llama_rn_decision_question & q, size_t i, const std::string & label) {
    const auto & opt = q.options[i];
    std::string text = opt.key;
    std::string desc = opt.description.is_null() ? "" : decision_legacy_text(opt.description);
    if (q.type == LLAMA_RN_DECISION_QUESTION_NOUL) {
        text = opt.key == "true" ? "yes" : "no";
    } else if (q.type == LLAMA_RN_DECISION_QUESTION_SCORE) {
        text = desc;
        desc.clear();
    }
    return common_json{
        {"label",  label},
        {"option", text},
        {"desc",   desc.empty() ? common_json() : common_json(desc)},
    };
}

// The state as the system_one template gets it: a string with a media marker where each image
// goes. The images of the request go first; a state of content parts (the library's own form)
// keeps its text and puts its images where they are.
static void decision_add_image(const common_json & url, std::vector<std::string> & images);

static std::string decision_legacy_state(const common_json & state, std::vector<std::string> & images) {
    const std::string marker = mtmd_default_marker();
    auto scrub = [&](std::string text) {
        string_replace_all(text, marker, " "); // the text must not forge an image
        return text;
    };

    std::string out;
    for (size_t i = 0; i < images.size(); i++) {
        out += marker;
    }

    const bool is_parts = state.is_array() && !state.empty() &&
        std::all_of(state.begin(), state.end(), [](const common_json & part) {
            return part.is_object() && part.contains("type") && !part.contains("role");
        });
    if (!is_parts) {
        return out + scrub(decision_legacy_text(state));
    }
    for (const auto & part : state) {
        const std::string type = part.at("type").is_string() ? part.at("type").get<std::string>() : "";
        if (type == "text") {
            out += scrub(part.contains("text") ? decision_legacy_text(part.at("text")) : "");
        } else if (type == "image_url" && part.contains("image_url")) {
            const common_json & url = part.at("image_url");
            decision_add_image(url.is_object() && url.contains("url") ? url.at("url") : url, images);
            out += marker;
        } else {
            throw std::invalid_argument("unknown content part type \"" + type + "\" in \"state\"");
        }
    }
    return out;
}

std::vector<llama_rn_decision_prompt> llama_rn_decision_context::fill_prompts_legacy(const llama_rn_decision_request & request, mtmd_context * mctx) const {
    std::vector<std::string> images = request.images;
    const std::string state = decision_legacy_state(request.state, images);
    if (!images.empty()) {
        if (profile.readout == LLAMA_RN_DECISION_READOUT_RANK_HEAD) {
            // each option is its own sequence, the images would be encoded once per option
            throw std::runtime_error("images are not supported for a classification-head (rank_head) model");
        }
        if (mctx == nullptr || !mtmd_support_vision(mctx)) {
            throw std::runtime_error("image input needs the multimodal projector of the model, see initMultimodal()");
        }
    }
    auto tokenize = [&](const std::string & segment) {
        auto ids = common_tokenize(vocab, segment, false, true);
        if (ids.empty()) {
            throw std::invalid_argument("a segment of the system_one prompt tokenized to nothing");
        }
        return ids;
    };
    auto label_of = [&](size_t i) { return i < label_texts.size() ? label_texts[i] : std::string("?"); };

    std::vector<llama_rn_decision_prompt> prompts;

    if (profile.readout == LLAMA_RN_DECISION_READOUT_RANK_HEAD) {
        // one sequence per option, the head scores it
        for (size_t qi = 0; qi < request.questions.size(); qi++) {
            const auto & q = request.questions[qi];
            for (size_t oi = 0; oi < q.options.size(); oi++) {
                common_json option = decision_legacy_option(q, oi, label_of(oi));
                option["index"] = (int) oi;
                const common_json inp = common_json{
                    {"state",     state},
                    {"question",  common_json{{"kind", decision_legacy_kind(q.type)}, {"text", decision_legacy_text(q.instructions)}}},
                    {"option",    option},
                    {"sep",       legacy_separator},
                    {"mask",      ""},
                    {"bos_token", bos_text},
                    {"eos_token", eos_text},
                };
                llama_rn_decision_prompt prompt;
                prompt.question = qi;
                prompt.pooled   = true;
                for (const auto & segment : render_legacy(inp)) {
                    const auto ids = tokenize(segment);
                    prompt.tokens.insert(prompt.tokens.end(), ids.begin(), ids.end());
                }
                if (prompt.tokens.empty()) {
                    throw std::invalid_argument("the system_one template rendered nothing for a question/option pair");
                }
                prompts.push_back(std::move(prompt));
            }
        }
        return prompts;
    }

    // letter_slot: every question in one sequence, the trailing segments are the question blocks
    common_json questions = common_json::array();
    for (size_t qi = 0; qi < request.questions.size(); qi++) {
        const auto & q = request.questions[qi];
        common_json options = common_json::array();
        for (size_t oi = 0; oi < q.options.size(); oi++) {
            options.push_back(decision_legacy_option(q, oi, label_of(oi)));
        }
        questions.push_back(common_json{
            {"k",       (int) qi + 1},
            {"key",     q.id},
            {"kind",    decision_legacy_kind(q.type)},
            {"text",    decision_legacy_text(q.instructions)},
            {"options", options},
        });
    }
    const std::vector<std::string> segments = render_legacy(common_json{
        {"state",     state},
        {"questions", questions},
        {"sep",       legacy_separator},
        {"mask",      ""},
        {"bos_token", bos_text},
        {"eos_token", eos_text},
    });
    if (segments.size() < request.questions.size()) {
        throw std::invalid_argument("the system_one template produced " + std::to_string(segments.size()) +
            " segments for " + std::to_string(request.questions.size()) +
            " questions: the separator must precede every question block");
    }

    llama_rn_decision_prompt prompt;
    prompt.labels = labels;
    const size_t first_question = segments.size() - request.questions.size();
    const std::string marker = mtmd_default_marker();
    mtmd::bitmaps bitmaps;
    decision_load_bitmaps(mctx, images, bitmaps);
    const std::vector<const mtmd_bitmap *> bitmaps_c_ptr = bitmaps.c_ptr();
    size_t next_image = 0;
    for (size_t i = 0; i < segments.size(); i++) {
        if (i < first_question && !images.empty()) {
            // with images, the state is tokenized one segment at a time, each with its own images,
            // and the question blocks follow it with their slots counted from where it ends
            size_t n_markers = 0;
            for (size_t at = segments[i].find(marker); at != std::string::npos; at = segments[i].find(marker, at + marker.size())) {
                n_markers++;
            }
            if (next_image + n_markers > images.size()) {
                throw std::invalid_argument("the system_one template wrote the state's images more than once");
            }
            llama_rn_decision_piece piece;
            if (n_markers > 0) {
                const std::vector<const mtmd_bitmap *> piece_bitmaps(
                    bitmaps_c_ptr.begin() + next_image, bitmaps_c_ptr.begin() + next_image + n_markers);
                piece.chunks = decision_tokenize_media(mctx, segments[i], piece_bitmaps, false);
                next_image += n_markers;
            } else {
                piece.tokens = tokenize(segments[i]);
            }
            prompt.state_pieces.push_back(std::move(piece));
            continue;
        }
        if (i >= first_question && i == first_question && !images.empty() && next_image != images.size()) {
            throw std::invalid_argument("the system_one template did not write the state's images");
        }
        const auto ids = tokenize(segments[i]);
        prompt.tokens.insert(prompt.tokens.end(), ids.begin(), ids.end());
        if (i >= first_question) {
            // the answer is the next token after the question's block
            prompt.slots.push_back((int32_t) prompt.tokens.size() - 1);
            prompt.slot_n_labels.push_back((int32_t) request.questions[i - first_question].options.size());
        }
    }
    prompts.push_back(std::move(prompt));
    return prompts;
}

//
// request parsing
//

llama_rn_decision_request llama_rn_decision_context::parse_request(const json & body) const {
    if (!is_supported()) {
        throw std::runtime_error(error.empty() ? "this model is not a decision model of a supported type" : error);
    }
    llama_rn_decision_request request;
    request.questions = parse_questions(body);
    request.state     = prepare_state(parse_state(body, request.images));
    return request;
}

// a spoken question (lfm3-d1-omni): its instructions are an input_audio part
static bool decision_is_spoken(const json & instructions) {
    return instructions.is_object() && instructions.contains("type") && instructions.at("type").is_string() &&
           instructions.at("type").get<std::string>() == "input_audio";
}

static bool decision_is_payload(const json & val) {
    return val.is_string() || val.is_object() || val.is_array();
}

std::vector<llama_rn_decision_question> llama_rn_decision_context::parse_questions(const json & body) const {
    // a model may take a null state, for example to ask about the media only
    if (!body.contains("state") || (body.at("state").is_null() && !profile.null_state)) {
        throw std::invalid_argument("\"state\" must be provided");
    }
    if (profile.strict_payload && !decision_is_payload(body.at("state"))) {
        throw std::invalid_argument("\"state\" must be text, an object or an array");
    }
    if (!body.contains("questions") || !body.at("questions").is_object() || body.at("questions").empty()) {
        throw std::invalid_argument("\"questions\" must be a non-empty object");
    }
    if (profile.single_question && body.at("questions").size() != 1) {
        throw std::invalid_argument("this model answers exactly one question");
    }

    std::vector<llama_rn_decision_question> questions;
    for (const auto & [id, q] : body.at("questions").items()) {
        auto err = [&id = id](const std::string & msg) {
            return std::invalid_argument("questions." + id + ": " + msg);
        };
        if (!q.is_object()) {
            throw err("must be an object");
        }
        // h2o-lightning: the instructions can be left out, the template has a default, and the type defaults to choice
        const json instructions = q.contains("instructions") ? q.at("instructions")
                                : profile.h2o_render ? json(std::string()) : json();
        if (instructions.is_null() && !profile.optional_instructions && !profile.forbids_instructions) {
            throw err("\"instructions\" must be provided");
        }
        if (profile.strict_payload && (!decision_is_payload(instructions) || instructions == "")) {
            throw err("\"instructions\" must be non-empty text, an object or an array");
        }
        if (profile.forbids_spoken && decision_is_spoken(instructions)) {
            throw err("spoken instructions need a model with audio input");
        }

        llama_rn_decision_question question;
        question.id           = id;
        question.instructions = instructions;

        const std::string type_name = q.contains("type") && q.at("type").is_string() ? q.at("type").get<std::string>()
                                    : profile.h2o_render ? "choice" : "";
        const json        criteria  = q.contains("criteria") ? q.at("criteria") : json();

        if (type_name == "choice") {
            question.type = LLAMA_RN_DECISION_QUESTION_CHOICE;
            if (!criteria.is_object() || criteria.empty()) {
                throw err("\"criteria\" must be a non-empty object");
            }
            for (const auto & [key, description] : criteria.items()) {
                if (profile.strict_payload && (key.empty() || !(description.is_null() || decision_is_payload(description)))) {
                    throw err("option keys must be non-empty, descriptions must be null, text, an object or an array");
                }
                question.options.push_back({key, description});
            }
            if (profile.strict_payload && question.options.size() < 2) {
                throw err("\"criteria\" must have at least 2 options");
            }
            if (profile.choice_sorted) {
                std::sort(question.options.begin(), question.options.end(), [](const auto & a, const auto & b) {
                    return a.key < b.key;
                });
            }
        } else if (type_name == "score") {
            question.type = LLAMA_RN_DECISION_QUESTION_SCORE;
            if (!criteria.is_array() || criteria.size() < profile.score_levels_min || criteria.size() > profile.score_levels_max) {
                throw err(string_format("\"criteria\" must be an array of %zu to %zu levels",
                                        profile.score_levels_min, profile.score_levels_max));
            }
            for (size_t i = 0; i < criteria.size(); i++) {
                if (profile.strict_payload && !decision_is_payload(criteria.at(i))) {
                    throw err("level descriptions must be text, an object or an array");
                }
                question.options.push_back({std::to_string(i), criteria.at(i)});
            }
        } else if (type_name == "noul") {
            question.type = LLAMA_RN_DECISION_QUESTION_NOUL;
            if (!criteria.is_null() && !criteria.is_object()) {
                throw err("\"criteria\" must be an object");
            }
            if (profile.strict_payload && criteria.is_object()) {
                for (const auto & [key, description] : criteria.items()) {
                    if ((key != "false" && key != "true") || !decision_is_payload(description)) {
                        throw err("\"criteria\" may only describe \"false\" and \"true\" with text, an object or an array");
                    }
                }
            }
            for (const char * key : {"false", "true"}) {
                json description;
                if (criteria.is_object() && criteria.contains(key)) {
                    description = criteria.at(key);
                } else if (criteria.is_object() && profile.noul_yes_no) {
                    const char * alias = std::string(key) == "true" ? "yes" : "no";
                    description = criteria.contains(alias) ? criteria.at(alias) : json();
                }
                question.options.push_back({key, description});
            }
            if (profile.noul_true_first ||
                (profile.noul_criteria_order && criteria.is_object() && criteria.size() == 2 && criteria.begin().key() == "true")) {
                std::swap(question.options[0], question.options[1]);
            }
        } else {
            throw err("\"type\" must be one of: choice, score, noul");
        }

        // the classes are fixed: "0" is inconsistent, "1" is consistent
        if (profile.forbids_instructions) {
            // the task text is in the template, an instructions from the request would be dropped
            if (!instructions.is_null()) {
                throw err("\"instructions\" is not used by this model, the task is part of its prompt");
            }
            if (question.type == LLAMA_RN_DECISION_QUESTION_CHOICE) {
                const auto & opts = question.options;
                if (opts.size() != 2 || opts[0].key != "0" || opts[1].key != "1" || !opts[0].description.is_null() || !opts[1].description.is_null()) {
                    throw err("\"criteria\" must be {\"0\": null, \"1\": null}");
                }
            } else if (question.type == LLAMA_RN_DECISION_QUESTION_NOUL) {
                if (!criteria.is_null()) {
                    throw err("\"criteria\" is not supported for a noul question");
                }
            } else {
                throw err("only choice and noul questions are supported");
            }
        }

        if (question.options.size() > n_options_max) {
            throw err(string_format("too many options (%zu), this model supports at most %zu", question.options.size(), n_options_max));
        }

        questions.push_back(std::move(question));
    }
    return questions;
}

static const size_t DECISION_MAX_IMAGES = 8;

// any media, mtmd tells an audio clip from an image by its content
static void decision_add_image(const json & url, std::vector<std::string> & images) {
    if (!url.is_string() || url.get<std::string>().empty()) {
        throw std::invalid_argument("an image or an audio clip must be a file path or a data URL");
    }
    if (images.size() >= DECISION_MAX_IMAGES) {
        throw std::invalid_argument(string_format("too many media files, the maximum is %zu", DECISION_MAX_IMAGES));
    }
    images.push_back(url.get<std::string>());
}

// an input_audio part (OpenAI): base64 data in "data", with its "format", or a file path or a data URL in "url"
static void decision_add_audio(const json & input_audio, std::vector<std::string> & images) {
    const json data = input_audio.is_object() && input_audio.contains("data") ? input_audio.at("data") : json();
    if (data.is_string() && !string_starts_with(data.get<std::string>(), "data:")) {
        const std::string format = input_audio.contains("format") && input_audio.at("format").is_string()
            ? input_audio.at("format").get<std::string>() : "wav";
        decision_add_image("data:audio/" + format + ";base64," + data.get<std::string>(), images);
        return;
    }
    decision_add_image(data.is_string() ? data : input_audio.is_object() && input_audio.contains("url") ? input_audio.at("url") : json(), images);
}

// the media of a content part, false if it is not one
static bool decision_take_media(const json & part, std::vector<std::string> & images, bool allow_video = false) {
    if (!part.is_object() || !part.contains("type")) {
        return false;
    }
    if (part.at("type") == "image_url" && part.contains("image_url")) {
        const json & image_url = part.at("image_url");
        decision_add_image(image_url.is_object() && image_url.contains("url") ? image_url.at("url") : image_url, images);
        return true;
    }
    if (part.at("type") == "input_audio" && part.contains("input_audio")) {
        decision_add_audio(part.at("input_audio"), images);
        return true;
    }
    // lfm3-d1: a video part, the model sees its frames
    if (allow_video && part.at("type") == "video_url" && part.contains("video_url")) {
        const json & video_url = part.at("video_url");
        decision_add_image(video_url.is_object() && video_url.contains("url") ? video_url.at("url") : video_url, images);
        return true;
    }
    return false;
}

json llama_rn_decision_context::parse_state(const json & body, std::vector<std::string> & images) const {
    if (profile.state_must_be_text && !body.at("state").is_string()) {
        throw std::invalid_argument("\"state\" must be the OCR text");
    }
    if (body.contains("videos") && !body.at("videos").is_null() && !body.at("videos").empty()) {
        throw std::invalid_argument("\"videos\" is not supported");
    }
    // "images" is an alias of "files"
    for (const char * key : {"files", "images"}) {
        if (!body.contains(key) || body.at(key).is_null()) {
            continue;
        }
        if (!body.at(key).is_array()) {
            throw std::invalid_argument(string_format("\"%s\" must be an array", key));
        }
        for (const auto & url : body.at(key)) {
            decision_add_image(url, images);
        }
    }

    // the prompt has the original image, then the image rendered from the OCR text
    if (profile.n_images_required > 0 && images.size() != profile.n_images_required) {
        throw std::invalid_argument(string_format("this model needs exactly %zu images (the original, then the rendered one), got %zu",
                                                  profile.n_images_required, images.size()));
    }

    const json & state = body.at("state");

    // a state of content parts: the images come out, the text stays (a system_one model keeps
    // the parts as they are, it places each image where it is, see decision_legacy_state())
    const bool is_parts = state.is_array() && !state.empty() &&
        std::all_of(state.begin(), state.end(), [](const json & part) {
            return part.is_object() && part.contains("type") && !part.contains("role");
        });
    if (is_parts && !legacy) {
        json parts = json::array();
        for (const auto & part : state) {
            if (!decision_take_media(part, images, profile.video_input)) {
                parts.push_back(part);
            }
        }
        return parts;
    }

    const bool is_wrapped = state.is_object() && state.contains("messages");
    const json & messages = is_wrapped ? state.at("messages") : state;
    if (!messages.is_array()) {
        return state;
    }

    // chat messages: take the image and audio parts out of the content
    json messages_out = json::array();
    for (const auto & msg : messages) {
        if (!msg.is_object() || !msg.contains("content") || !msg.at("content").is_array()) {
            messages_out.push_back(msg);
            continue;
        }
        json content = json::array();
        for (const auto & part : msg.at("content")) {
            if (!decision_take_media(part, images, profile.video_input)) {
                content.push_back(part);
            }
        }
        json msg_out = msg;
        msg_out["content"] = content;
        messages_out.push_back(msg_out);
    }

    if (!is_wrapped) {
        return messages_out;
    }
    json state_out = state;
    state_out["messages"] = messages_out;
    return state_out;
}

//
// prompt
//

// the recursive functions of this file walk request JSON, a deeper value would overflow the stack
static const int RECURSIVE_LIMIT = 128;

// replace text in all strings of a JSON value
static json decision_replace_text(const json & val, const std::string & search, const std::string & replace, int depth = 0) {
    if (depth >= RECURSIVE_LIMIT) {
        throw std::invalid_argument("JSON is nested more than " + std::to_string(RECURSIVE_LIMIT) + " levels deep");
    }
    if (val.is_string()) {
        std::string str = val.get<std::string>();
        string_replace_all(str, search, replace);
        return str;
    }
    if (val.is_array()) {
        json out = json::array();
        for (const auto & item : val) {
            out.push_back(decision_replace_text(item, search, replace, depth + 1));
        }
        return out;
    }
    if (val.is_object()) {
        json out = json::object();
        for (const auto & [key, item] : val.items()) {
            out[key] = decision_replace_text(item, search, replace, depth + 1);
        }
        return out;
    }
    return val;
}

// sort the keys of all objects of a JSON value
static json decision_sort_keys(const json & val, int depth = 0) {
    if (depth >= RECURSIVE_LIMIT) {
        throw std::invalid_argument("JSON is nested more than " + std::to_string(RECURSIVE_LIMIT) + " levels deep");
    }
    if (val.is_array()) {
        json out = json::array();
        for (const auto & item : val) {
            out.push_back(decision_sort_keys(item, depth + 1));
        }
        return out;
    }
    if (val.is_object()) {
        std::map<std::string, json> sorted;
        for (const auto & [key, item] : val.items()) {
            sorted[key] = decision_sort_keys(item, depth + 1);
        }
        json out = json::object();
        for (const auto & [key, item] : sorted) {
            out[key] = item;
        }
        return out;
    }
    return val;
}

// kev flattens a JSON value into text, the keys of an object are kept as labels (kev/api.py: render)
static std::string decision_text_render(const json & val, int indent = 0, int depth = 0) {
    if (depth >= RECURSIVE_LIMIT) {
        throw std::invalid_argument("JSON is nested more than " + std::to_string(RECURSIVE_LIMIT) + " levels deep");
    }
    const std::string pad(2 * indent, ' ');
    if (val.is_null()) {
        return "";
    }
    if (val.is_string()) {
        return val.get<std::string>();
    }
    if (val.is_boolean()) {
        return val.get<bool>() ? "True" : "False";
    }
    if (val.is_array()) {
        std::string out;
        for (const auto & item : val) {
            const std::string text = decision_text_render(item, indent + 1, depth + 1);
            out += (out.empty() ? "" : "\n") + pad + "- " + text.substr(std::min(text.size(), text.find_first_not_of(" \t\n\r")));
        }
        return out;
    }
    if (val.is_object()) {
        std::string out;
        for (const auto & [key, item] : val.items()) {
            const bool is_nested = item.is_object() || item.is_array();
            out += (out.empty() ? "" : "\n") + pad + key + (is_nested ? ":\n" : ": ") + decision_text_render(item, is_nested ? indent + 1 : 0, depth + 1);
        }
        return out;
    }
    return val.dump();
}

// write "<|name|>" with U+00A6 in place of "|", so text from the request is not parsed as special tokens
// same result as std::regex_replace with <\|([A-Za-z0-9_]+)\|>; std::regex is not used:
// libstdc++ recurses for each char, a long input overflows the stack
static std::string decision_escape_text(const std::string & text) {
    const auto is_name = [](char c) {
        return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_';
    };
    std::string out;
    size_t n_done = 0; // the chars of text that are in out
    for (size_t i = text.find("<|"); i != std::string::npos; i = text.find("<|", std::max(i + 1, n_done))) {
        size_t j = i + 2;
        while (j < text.size() && is_name(text[j])) {
            j++;
        }
        if (j > i + 2 && text.compare(j, 2, "|>") == 0) {
            out.append(text, n_done, i - n_done);
            out += "<\xC2\xA6";
            out.append(text, i + 2, j - i - 2);
            out += "\xC2\xA6>";
            n_done = j + 2;
        }
    }
    if (n_done == 0) {
        return text;
    }
    out.append(text, n_done, std::string::npos);
    return out;
}

static std::string decision_text(const json & val) {
    return decision_escape_text(decision_text_render(val));
}

// apply a text escape to all strings of a JSON value, keys included (d1-omni prompt.py: escape)
template <typename F>
static json decision_escape(const json & val, const F & escape, int depth = 0) {
    if (depth >= RECURSIVE_LIMIT) {
        throw std::invalid_argument("JSON is nested more than " + std::to_string(RECURSIVE_LIMIT) + " levels deep");
    }
    if (val.is_string()) {
        return escape(val.get<std::string>());
    }
    if (val.is_array()) {
        json out = json::array();
        for (const auto & item : val) {
            out.push_back(decision_escape(item, escape, depth + 1));
        }
        return out;
    }
    if (val.is_object()) {
        json out = json::object();
        for (const auto & [key, item] : val.items()) {
            out[escape(key)] = decision_escape(item, escape, depth + 1);
        }
        return out;
    }
    return val;
}

static json decision_escape_special(const json & val) {
    return decision_escape(val, decision_escape_text);
}

// lfm3-d1: a zero-width space after the first character of each special token text (d1 prompt.py: plain)
std::string llama_rn_decision_context::d1_break(const std::string & text) const {
    static const std::string zwsp = "\xE2\x80\x8B";
    std::string cur = text;
    // again for a match that overlapped the one broken before it
    for (bool changed = true; changed;) {
        changed = false;
        std::string out;
        for (size_t i = 0; i < cur.size();) {
            const std::string * hit = nullptr;
            if (quoted_first.find(cur[i]) != std::string::npos) {
                for (const auto & spelling : quoted_spellings) {
                    if (cur.compare(i, spelling.size(), spelling) == 0) {
                        hit = &spelling;
                        break;
                    }
                }
            }
            if (hit == nullptr) {
                out += cur[i++];
                continue;
            }
            // after the first character, its UTF-8 bytes
            const unsigned char c = (*hit)[0];
            const size_t n = c < 0x80 ? 1 : c < 0xE0 ? 2 : c < 0xF0 ? 3 : 4;
            out += hit->substr(0, n) + zwsp + hit->substr(n);
            i += hit->size();
            changed = true;
        }
        cur = std::move(out);
    }
    return cur;
}

//
// h2o-lightning: the texts are rendered the way the reference shim renders them with Python (json.dumps, str.lstrip)
//

// one code point of UTF-8 text at s[i], i moves past it; an invalid byte is read as is
static uint32_t decision_utf8_next(const std::string & s, size_t & i) {
    const unsigned char c = s[i];
    size_t n = c < 0x80 ? 1 : (c >> 5) == 0x6 ? 2 : (c >> 4) == 0xe ? 3 : (c >> 3) == 0x1e ? 4 : 1;
    if (i + n > s.size()) {
        n = 1;
    }
    uint32_t cp = n == 1 ? c : n == 2 ? (c & 0x1f) : n == 3 ? (c & 0x0f) : (c & 0x07);
    for (size_t k = 1; k < n; k++) {
        cp = (cp << 6) | (s[i + k] & 0x3f);
    }
    i += n;
    return cp;
}

// str.isspace() of Python
static bool decision_py_isspace(uint32_t cp) {
    return (cp >= 0x09 && cp <= 0x0d) || (cp >= 0x1c && cp <= 0x20) || cp == 0x85 || cp == 0xa0 || cp == 0x1680 ||
           (cp >= 0x2000 && cp <= 0x200a) || cp == 0x2028 || cp == 0x2029 || cp == 0x202f || cp == 0x205f || cp == 0x3000;
}

static std::string decision_py_rstrip(const std::string & s) {
    size_t end = 0;
    for (size_t i = 0; i < s.size();) {
        if (!decision_py_isspace(decision_utf8_next(s, i))) {
            end = i;
        }
    }
    return s.substr(0, end);
}

static std::string decision_py_lstrip(const std::string & s) {
    size_t i = 0;
    while (i < s.size()) {
        size_t next = i;
        if (!decision_py_isspace(decision_utf8_next(s, next))) {
            break;
        }
        i = next;
    }
    return s.substr(i);
}

// a code point as JSON escapes, a surrogate pair above the BMP
static std::string decision_py_escape_cp(uint32_t cp) {
    if (cp < 0x10000) {
        return string_format("\\u%04x", cp);
    }
    cp -= 0x10000;
    return string_format("\\u%04x\\u%04x", 0xd800 + (cp >> 10), 0xdc00 + (cp & 0x3ff));
}

// repr() of a Python float: the shortest digits that read back the same, in exponent form below 1e-4 and from 1e16
static std::string decision_py_float(double v) {
    char buf[64];
    // the fewest digits that read back the same
    for (int prec = 0; prec < 17; prec++) {
        snprintf(buf, sizeof(buf), "%.*e", prec, v);
        if (std::strtod(buf, nullptr) == v) {
            break;
        }
    }
    const std::string sci(buf); // for example "-1.25e+03"
    const size_t pos_e = sci.find('e');
    const int    exp   = std::atoi(sci.c_str() + pos_e + 1);
    std::string sign;
    std::string digits;
    for (size_t i = 0; i < pos_e; i++) {
        if (sci[i] == '-') {
            sign = "-";
        } else if (sci[i] != '.') {
            digits += sci[i];
        }
    }
    const int n = (int) digits.size();
    if (exp < -4 || exp >= 16) {
        return sign + digits.substr(0, 1) + (n > 1 ? "." + digits.substr(1) : "") + string_format("e%c%02d", exp < 0 ? '-' : '+', std::abs(exp));
    }
    if (exp >= n - 1) {
        return sign + digits + std::string(exp - (n - 1), '0') + ".0";
    }
    if (exp >= 0) {
        return sign + digits.substr(0, exp + 1) + "." + digits.substr(exp + 1);
    }
    return sign + "0." + std::string(-exp - 1, '0') + digits;
}

static void decision_py_str(std::string & out, const std::string & s, bool ensure_ascii) {
    out += '"';
    for (size_t i = 0; i < s.size();) {
        const unsigned char c = s[i];
        switch (c) {
            case '"':  out += "\\\""; i++; continue;
            case '\\': out += "\\\\"; i++; continue;
            case '\b': out += "\\b";  i++; continue;
            case '\f': out += "\\f";  i++; continue;
            case '\n': out += "\\n";  i++; continue;
            case '\r': out += "\\r";  i++; continue;
            case '\t': out += "\\t";  i++; continue;
            default:   break;
        }
        if (c < 0x20 || (ensure_ascii && c == 0x7f)) {
            out += string_format("\\u%04x", c);
            i++;
        } else if (c < 0x80 || !ensure_ascii) {
            out += (char) c;
            i++;
        } else {
            out += decision_py_escape_cp(decision_utf8_next(s, i));
        }
    }
    out += '"';
}

// json.dumps() of Python: indent < 0 is no indent, keys stay in their order
static void decision_py_dump(std::string & out, const json & val, int indent, int level, const std::string & item_sep, const std::string & key_sep, bool ensure_ascii) {
    if (level >= RECURSIVE_LIMIT) {
        throw std::invalid_argument("JSON is nested more than " + std::to_string(RECURSIVE_LIMIT) + " levels deep");
    }
    const auto newline = [&](int lvl) {
        return indent < 0 ? std::string() : "\n" + std::string(indent * lvl, ' ');
    };
    if (val.is_null()) {
        out += "null";
    } else if (val.is_boolean()) {
        out += val.get<bool>() ? "true" : "false";
    } else if (val.is_number_float()) {
        out += decision_py_float(val.get<double>());
    } else if (val.is_number()) {
        out += val.dump();
    } else if (val.is_string()) {
        decision_py_str(out, val.get<std::string>(), ensure_ascii);
    } else if (val.is_array()) {
        if (val.empty()) {
            out += "[]";
            return;
        }
        out += "[" + newline(level + 1);
        for (size_t i = 0; i < val.size(); i++) {
            if (i > 0) {
                out += item_sep + newline(level + 1);
            }
            decision_py_dump(out, val.at(i), indent, level + 1, item_sep, key_sep, ensure_ascii);
        }
        out += newline(level) + "]";
    } else if (val.is_object()) {
        if (val.empty()) {
            out += "{}";
            return;
        }
        out += "{" + newline(level + 1);
        bool first = true;
        for (const auto & [key, item] : val.items()) {
            if (!first) {
                out += item_sep + newline(level + 1);
            }
            first = false;
            decision_py_str(out, key, ensure_ascii);
            out += key_sep;
            decision_py_dump(out, item, indent, level + 1, item_sep, key_sep, ensure_ascii);
        }
        out += newline(level) + "}";
    }
}

static std::string decision_py_dumps(const json & val, int indent, const std::string & item_sep, const std::string & key_sep, bool ensure_ascii) {
    std::string out;
    decision_py_dump(out, val, indent, 0, item_sep, key_sep, ensure_ascii);
    return out;
}

// a state longer than the maximum keeps its head and its tail around a "..." line
// (h2o-lightning follows serve_config.json and h2o_lightning_shim.py of the model repo)
std::string llama_rn_decision_context::h2o_cut_state(const std::string & text) const {
    static const size_t DECISION_H2O_MAX_STATE_TOKENS = 32000;
    // a byte-level BPE never has more tokens than bytes
    const size_t max_state_tokens = DECISION_H2O_MAX_STATE_TOKENS;
    if (text.size() <= max_state_tokens) {
        return text;
    }
    const llama_tokens tokens = common_tokenize(vocab, text, false, true);
    if (tokens.size() <= max_state_tokens) {
        return text;
    }
    // each half is the text of its token pieces, a cut inside a character gives U+FFFD
    const size_t half = max_state_tokens / 2;
    auto text_of = [&](llama_tokens::const_iterator begin, llama_tokens::const_iterator end) {
        std::string out;
        for (auto it = begin; it != end; ++it) {
            out += common_token_to_piece(vocab, *it, true);
        }
        return json::parse(json::make(out).dump_safe()).get<std::string>();
    };
    return text_of(tokens.begin(), tokens.begin() + half) + "\n...\n" + text_of(tokens.end() - half, tokens.end());
}

// the special token texts written in the input are quoted with JSON escapes, so they stay text
std::string llama_rn_decision_context::h2o_quote(const std::string & text) const {
    std::string out;
    for (size_t i = 0; i < text.size();) {
        bool is_quoted = false;
        if (quoted_first.find(text[i]) == std::string::npos) {
            out += text[i++];
            continue;
        }
        for (const auto & spelling : quoted_spellings) {
            if (text.compare(i, spelling.size(), spelling) == 0) {
                std::string quoted = spelling;
                string_replace_all(quoted, "<", "\\u003c");
                string_replace_all(quoted, ">", "\\u003e");
                out += quoted;
                i += spelling.size();
                is_quoted = true;
                break;
            }
        }
        if (!is_quoted) {
            out += text[i++];
        }
    }
    return out;
}

json llama_rn_decision_context::prepare_state(const json & state) const {
    if (!profile.h2o_render) {
        return state;
    }
    // done once per request: compact JSON, the cut, the quoting
    return h2o_quote(h2o_cut_state(state.is_string() ? state.get<std::string>() : decision_py_dumps(state, -1, ",", ":", false)));
}

json llama_rn_decision_context::render_h2o(const json & state, const llama_rn_decision_question & question) const {
    // the instructions are JSON with an indent of 1, the descriptions are default JSON, the state is from prepare_state()
    const std::string instructions = decision_py_lstrip(question.instructions.is_string() ?
            question.instructions.get<std::string>() : decision_py_dumps(question.instructions, 1, ",", ": ", true));

    json options = json::array();
    for (size_t i = 0; i < question.options.size(); i++) {
        const auto & opt = question.options[i];
        json description; // null if not provided, the template has a default, except a score level that is shown as JSON null
        if (question.type == LLAMA_RN_DECISION_QUESTION_SCORE && opt.description.is_null()) {
            description = "null";
        } else if (!opt.description.is_null()) {
            description = h2o_quote(opt.description.is_string() ? opt.description.get<std::string>() : decision_py_dumps(opt.description, -1, ", ", ": ", false));
        }
        const std::string key = h2o_quote(opt.key);

        // the user turn is stripped like Python's str.strip(), the template only trims ASCII whitespace
        // so the text that ends the turn is stripped here, a space keeps a description that is only whitespace
        if (i + 1 == question.options.size()) {
            const bool is_key = question.type == LLAMA_RN_DECISION_QUESTION_CHOICE && description.is_null();
            if (is_key || description.is_string()) {
                const std::string text = is_key ? key : description.get<std::string>();
                const std::string stripped = decision_py_rstrip(text);
                if (stripped != text) {
                    description = stripped.empty() ? json(" ") : json(stripped);
                }
            }
        }

        json option = json::object();
        option["label"]       = label_texts[i];
        option["key"]         = key;
        option["description"] = description;
        options.push_back(option);
    }

    json inp = json::object();
    inp["type"]         = decision_question_type_name(question.type);
    inp["state"]        = state;
    inp["instructions"] = h2o_quote(instructions);
    inp["options"]      = options;
    return inp;
}

size_t llama_rn_decision_context::n_variants(const llama_rn_decision_question & question) const {
    if (profile.reverse_variant && question.type == LLAMA_RN_DECISION_QUESTION_CHOICE && question.options.size() > 1) {
        return 2;
    }
    return 1;
}

size_t llama_rn_decision_context::n_outputs(const llama_rn_decision_question & question) const {
    if (profile.n_noul_ratings > 0 && question.type == LLAMA_RN_DECISION_QUESTION_NOUL) {
        return profile.n_noul_ratings;
    }
    return question.options.size();
}

// label codes follow prompt.py of the lfm2-d1 model repo
void llama_rn_decision_context::d1_labels(const llama_rn_decision_question & question, std::vector<std::string> & texts, std::vector<std::vector<llama_token>> & groups) const {
    const size_t n_options = question.options.size();

    auto get_single_tokens = [&](const std::vector<std::string> & forms) {
        std::vector<llama_token> out;
        for (const auto & form : forms) {
            const auto toks = common_tokenize(vocab, form, false, false);
            if (toks.size() == 1 && std::find(out.begin(), out.end(), toks[0]) == out.end()) {
                out.push_back(toks[0]);
            }
        }
        return out;
    };

    if (question.type != LLAMA_RN_DECISION_QUESTION_CHOICE) {
        for (const auto & opt : question.options) {
            std::vector<llama_token> group;
            if (question.type == LLAMA_RN_DECISION_QUESTION_SCORE) {
                group = get_single_tokens({opt.key});
            } else if (opt.key == "true") {
                group = get_single_tokens({"yes", "Yes", "YES"});
            } else {
                group = get_single_tokens({"no", "No", "NO"});
            }
            if (group.empty()) {
                throw std::runtime_error("decision label is not a single token: " + opt.key);
            }
            texts.push_back(opt.key);
            groups.push_back(group);
        }
        return;
    }

    bool is_letters = true;
    for (const auto & opt : question.options) {
        is_letters = is_letters && opt.key.size() == 1 && std::isalpha((unsigned char) opt.key[0]);
    }

    std::vector<std::string> codes;
    for (size_t i = 0; i < n_options; i++) {
        if (is_letters) {
            codes.push_back(question.options[i].key);
        } else if (n_options <= 26) {
            codes.push_back(std::string(1, 'A' + i));
        } else if (profile.labels_alpha2) {
            // lfm3-d1: A..Z, then AA, AB..
            codes.push_back(i < 26 ? std::string(1, 'A' + i) : std::string{(char) ('A' + (i - 26) / 26), (char) ('A' + (i - 26) % 26)});
        } else {
            codes.push_back(string_format("%02zu", i));
        }
    }

    // a code that is taken, or not a single token, is replaced by the first free one of these
    std::vector<std::string> pool;
    for (char c = 'A'; c <= 'Z'; c++) {
        pool.push_back(std::string(1, c));
    }
    for (int i = 0; i < 100; i++) {
        pool.push_back(string_format("%02d", i));
    }
    for (char c = 'a'; c <= 'z'; c++) {
        pool.push_back(std::string(1, c));
    }
    for (int i = 0; i < 200; i++) {
        pool.push_back(string_format("#%d", i));
    }
    for (char a = 'A'; a <= 'Z'; a++) {
        for (char b = 'A'; b <= 'Z'; b++) {
            pool.push_back(std::string{a, b});
        }
    }

    std::vector<llama_token> used;
    auto take = [&](const std::string & code) {
        const auto toks = common_tokenize(vocab, code, false, false);
        if (toks.size() != 1 || std::find(used.begin(), used.end(), toks[0]) != used.end()) {
            return false;
        }
        used.push_back(toks[0]);
        // the code may also be answered with a leading space
        std::vector<llama_token> group = {toks[0]};
        for (const llama_token tok : get_single_tokens({" " + code})) {
            if (tok != toks[0]) {
                group.push_back(tok);
            }
        }
        texts.push_back(code);
        groups.push_back(group);
        return true;
    };
    for (const auto & code : codes) {
        bool is_taken = take(code);
        for (size_t i = 0; !is_taken && i < pool.size(); i++) {
            is_taken = take(pool[i]);
        }
        if (!is_taken) {
            throw std::invalid_argument(string_format("no single-token label left for %zu options", n_options));
        }
    }
}

void llama_rn_decision_context::fill_labels(const llama_rn_decision_question & question, llama_rn_decision_prompt & prompt) const {
    if (profile.labels != LLAMA_RN_DECISION_LABELS_PER_QUESTION) {
        // a rating scale reads its ratings at the first labels
        prompt.labels.assign(labels.begin(), labels.begin() + n_outputs(question));
        return;
    }
    std::vector<std::string>              texts;
    std::vector<std::vector<llama_token>> groups;
    d1_labels(question, texts, groups);
    for (const auto & group : groups) {
        prompt.labels.insert(prompt.labels.end(), group.begin(), group.end());
        prompt.label_groups.push_back(group.size());
    }
}

json llama_rn_decision_context::render_options(const llama_rn_decision_question & question, size_t variant) const {
    const size_t n_options = question.options.size();

    std::vector<std::string>              d1_texts;
    std::vector<std::vector<llama_token>> d1_groups;
    if (profile.labels == LLAMA_RN_DECISION_LABELS_PER_QUESTION) {
        d1_labels(question, d1_texts, d1_groups);
    }

    // the second variant shows the options in the reverse order
    json options = json::array();
    for (size_t i = 0; i < n_options; i++) {
        const auto & opt = question.options[variant == 0 ? i : n_options - 1 - i];
        json option = json{
            {"key",         opt.key},
            {"description", opt.description},
        };
        if (profile.text_input) {
            option["key"] = decision_text(opt.key);
            if (!opt.description.is_null()) {
                option["description"] = decision_text(opt.description);
            }
        }
        if (!label_texts.empty()) {
            option["label"] = label_texts[i];
        }
        if (!d1_texts.empty()) {
            option["label"] = d1_texts[i];
        }
        options.push_back(option);
    }
    return options;
}

// The separator marks the boundaries at which a prompt is tokenized piece by piece. A template
// never spells these out -- it writes {{ sep }}, {{ mark_question }} and {{ mark_option }} -- so
// they are internal to this file and share one prefix, which is what a single scrub can neuter.
static const std::string DECISION_MARKER   = "<<decision:";
static const std::string DECISION_NEUTERED = "<<decision ";
static const std::string DECISION_SEP      = "<<decision:sep>>";
static const std::string DECISION_STATE    = "<<decision:state>>";
static const std::string DECISION_QUESTION = "<<decision:question>>";
static const std::string DECISION_OPTION   = "<<decision:option>>";

static std::string decision_apply_template(const common_chat_template & tmpl, const json & inp) {
    jinja::context ctx(tmpl.source());
    jinja::global_from_json(ctx, inp, false);
    jinja::runtime runtime(ctx);
    const jinja::value results = runtime.execute(tmpl.prog);
    return jinja::runtime::gather_string_parts(results)->as_string().str();
}

std::string llama_rn_decision_context::render(
        const json & state,
        const std::vector<llama_rn_decision_question> & questions,
        const llama_rn_decision_question & question,
        size_t variant,
        size_t n_images,
        bool is_audio) const {
    // the template is given raw JSON values, it serializes the ones that are not strings
    // h2o-lightning: the template is given texts
    json inp = profile.h2o_render ? render_h2o(state, question) : json{
        {"id",           question.id},
        {"type",         decision_question_type_name(question.type)},
        {"instructions", question.instructions},
        {"state",        state},
        {"options",      render_options(question, variant)},
    };

    if (profile.list_questions) {
        inp["questions"] = json::array();
        for (const auto & q : questions) {
            inp["questions"].push_back(json{
                {"id",           q.id},
                {"type",         decision_question_type_name(q.type)},
                {"instructions", q.instructions},
                {"options",      render_options(q, 0)},
            });
        }
    }

    if (profile.sort_keys) {
        inp = decision_sort_keys(inp);
    }

    if (profile.text_input) {
        inp["state"]        = decision_text(state);
        inp["instructions"] = decision_text(question.instructions);
    }

    // the OCR text must not be parsed as special tokens
    if (profile.state_must_be_text) {
        inp["state"] = decision_escape_text(state.get<std::string>());
    }

    // the input must not contain the marker of the options
    if (!text_marker.empty()) {
        inp = decision_replace_text(inp, text_marker, " ");
    }

    // a spoken question (lfm3-d1-omni) is heard where its text would be, after the media of the state
    const bool is_spoken = decision_is_spoken(question.instructions);
    if (is_spoken) {
        inp["instructions"] = "";
    }
    if (profile.break_special) {
        inp = decision_escape(inp, [this](const std::string & text) { return d1_break(text); });
    }
    if (profile.escape_special) {
        inp = decision_escape_special(inp);
    }

    // the template puts one media marker per image
    json images = json::array();
    if (n_images > 0 || is_spoken) {
        inp = decision_replace_text(inp, mtmd_default_marker(), " ");
        for (size_t i = 0; i < n_images; i++) {
            images.push_back(mtmd_default_marker());
        }
    }
    inp["images"] = images;
    if (is_spoken) {
        inp["instructions"] = mtmd_default_marker();
    }

    // a template may ask for the prompt to be tokenized piece by piece by writing {{ sep }} at every boundary,
    // the input must not forge one
    if (profile.max_tokens > 0) {
        // a piece that is cut to a budget starts with the mark of what it is, see fill_prompt_budget()
        inp = decision_replace_text(inp, DECISION_MARKER, DECISION_NEUTERED);
        inp["mark_state"]    = DECISION_STATE;
        inp["mark_question"] = DECISION_QUESTION;
        inp["mark_option"]   = DECISION_OPTION;
    } else {
        inp = decision_replace_text(inp, DECISION_SEP, " ");
    }
    inp["sep"] = DECISION_SEP;
    if (profile.audio_input && profile.max_tokens > 0) {
        // only the budget prompt (lfm2-d1-omni) tells the template what kind of media it gets
        inp["audio"] = is_audio;
    }

    return decision_apply_template(*tmpl, inp);
}

std::vector<llama_rn_decision_prompt> llama_rn_decision_context::fill_prompts(const llama_rn_decision_request & request, mtmd_context * mctx) const {
    if (!is_supported()) {
        throw std::runtime_error(error.empty() ? "this model is not a decision model of a supported type" : error);
    }
    if (legacy) {
        return fill_prompts_legacy(request, mctx);
    }
    if (!request.images.empty() && !profile.image_input && !profile.audio_input) {
        throw std::runtime_error("this decision model does not take images");
    }
    if (!request.images.empty() && (mctx == nullptr || (!mtmd_support_vision(mctx) && !mtmd_support_audio(mctx)))) {
        throw std::runtime_error("image and audio input need the multimodal projector of the model, see initMultimodal()");
    }

    mtmd::bitmaps bitmaps;
    std::vector<mtmd_helper_video *> videos; // the frames of a video need its context while they are tokenized
    decision_load_bitmaps(mctx, request.images, bitmaps, profile.video_input, &videos);
    const std::vector<const mtmd_bitmap *> bitmaps_c_ptr = bitmaps.c_ptr();

    std::vector<llama_rn_decision_prompt> prompts;
    try {
        if (is_joint() && profile.joint_per_question) {
            // each question has its own prompt, with the spans that the head reads (vllm-decision2)
            for (size_t i = 0; i < request.questions.size(); i++) {
                prompts.push_back(fill_prompt_joint(request, mctx, bitmaps_c_ptr, i));
            }
        } else if (is_joint()) {
            prompts.push_back(fill_prompt_joint(request, mctx, bitmaps_c_ptr));
        } else {
            for (size_t i = 0; i < request.questions.size(); i++) {
                for (size_t variant = 0; variant < n_variants(request.questions[i]); variant++) {
                    prompts.push_back(fill_prompt(request, i, variant, mctx, bitmaps_c_ptr));
                }
            }
        }
    } catch (...) {
        for (mtmd_helper_video * video : videos) {
            mtmd_helper_video_free(video);
        }
        throw;
    }
    for (mtmd_helper_video * video : videos) {
        mtmd_helper_video_free(video);
    }
    if (profile.no_prefix_cache) {
        for (auto & p : prompts) {
            p.no_prefix_cache = true;
        }
    }
    return prompts;
}

// images: file paths or data URLs, the same as the media_paths of a completion. Loaded once per
// request: a prompt per question tokenizes the same images again, from these
// (mtmd::bitmaps cannot be returned: its destructor leaves it without a move constructor)
// allow_video: a video is kept as its frames, without its soundtrack (lfm3-d1); its contexts are
// appended to videos, which the caller keeps alive until the bitmaps are tokenized, then frees
static void decision_load_bitmaps(
        mtmd_context * mctx,
        const std::vector<std::string> & images,
        mtmd::bitmaps & bitmaps,
        bool allow_video,
        std::vector<mtmd_helper_video *> * videos) {
    auto free_videos = [&]() {
        if (videos != nullptr) {
            for (mtmd_helper_video * video : *videos) {
                mtmd_helper_video_free(video);
            }
            videos->clear();
        }
    };
    try {
        for (const auto & image : images) {
            mtmd_helper_init_opt opt = mtmd_helper_init_opt_default();
            opt.video_params.no_soundtrack = allow_video; // the d1 reference sees the frames of a video, not its soundtrack
            mtmd_helper_bitmap_wrapper out{};
            if (string_starts_with(image, "data:")) {
                const size_t comma = image.find(',');
                if (comma == std::string::npos || image.substr(0, comma).find(";base64") == std::string::npos) {
                    throw std::invalid_argument("a media data URL must be base64 encoded");
                }
                const raw_buffer data = base64_decode(image.substr(comma + 1));
                out = mtmd_helper_bitmap_init_from_buf(mctx, data.data(), data.size(), false, opt);
            } else {
                out = mtmd_helper_bitmap_init_from_file(mctx, image.c_str(), false, opt);
            }
            if (out.video_ctx != nullptr && !allow_video) {
                mtmd_helper_video_free(out.video_ctx);
                if (out.bitmap != nullptr) {
                    mtmd_bitmap_free(out.bitmap);
                }
                throw std::invalid_argument("a decision media file must be an image or an audio clip, not a video");
            }
            if (out.bitmap == nullptr) {
                if (out.video_ctx != nullptr) {
                    mtmd_helper_video_free(out.video_ctx);
                }
                throw std::invalid_argument("failed to load image or audio file: " + image.substr(0, 64));
            }
            if (out.video_ctx != nullptr) {
                videos->push_back(out.video_ctx);
            }
            bitmaps.entries.emplace_back(out.bitmap);
        }
    } catch (...) {
        free_videos();
        throw;
    }
}

static std::shared_ptr<mtmd_input_chunks> decision_tokenize_media(
        mtmd_context * mctx,
        const std::string & prompt,
        const std::vector<const mtmd_bitmap *> & bitmaps,
        bool add_special) {
    // the same flags as llama-server; a system_one state segment adds nothing, its template writes BOS
    mtmd_input_text text = {
        prompt.data(),
        prompt.size(),
        add_special,
        /* parse_special */ true,
    };
    std::shared_ptr<mtmd_input_chunks> chunks(mtmd_input_chunks_init(), mtmd_input_chunks_free);
    std::vector<const mtmd_bitmap *> bitmaps_c_ptr = bitmaps; // mtmd_tokenize takes a non-const array
    if (mtmd_tokenize(mctx, chunks.get(), &text, bitmaps_c_ptr.data(), bitmaps_c_ptr.size()) != 0) {
        throw std::runtime_error("failed to tokenize the decision prompt with its images");
    }
    return chunks;
}

llama_rn_decision_prompt llama_rn_decision_context::fill_prompt(
        const llama_rn_decision_request & request,
        size_t i_question,
        size_t variant,
        mtmd_context * mctx,
        const std::vector<const mtmd_bitmap *> & bitmaps) const {
    const auto & question = request.questions[i_question];

    llama_rn_decision_prompt prompt;
    prompt.question = i_question;
    prompt.variant  = variant;

    if (profile.max_tokens > 0) {
        fill_prompt_budget(request, i_question, variant, mctx, bitmaps, prompt);
        return prompt;
    }

    // a spoken question (lfm3-d1-omni) is heard where its text would be, after the media of the state
    const bool is_spoken = decision_is_spoken(question.instructions);
    mtmd::bitmaps spoken_bitmaps;
    std::vector<const mtmd_bitmap *> media = bitmaps;
    if (is_spoken) {
        if (mctx == nullptr || !mtmd_support_audio(mctx)) {
            throw std::runtime_error("a spoken question needs the multimodal projector of the model, see initMultimodal()");
        }
        std::vector<std::string> files;
        decision_add_audio(question.instructions.contains("input_audio") ? question.instructions.at("input_audio") : json::object(), files);
        decision_load_bitmaps(mctx, files, spoken_bitmaps);
        for (const auto & entry : spoken_bitmaps.entries) {
            media.push_back(entry.ptr.get());
        }
    }

    const std::string prompt_text = render(request.state, request.questions, question, variant, request.images.size());

    if (!media.empty()) {
        GGML_ASSERT(profile.readout == LLAMA_RN_DECISION_READOUT_LABEL_LOGITS ||
                    profile.readout == LLAMA_RN_DECISION_READOUT_POOLED); // see profile.image_input
        size_t n_markers = 0;
        for (size_t pos = prompt_text.find(mtmd_default_marker()); pos != std::string::npos; pos = prompt_text.find(mtmd_default_marker(), pos + 1)) {
            n_markers++;
        }
        if (n_markers != media.size()) {
            throw std::runtime_error("the template of this decision model has no place for images");
        }
        const std::vector<std::string> pieces = string_split(prompt_text, DECISION_SEP);
        if (pieces.size() == 1) {
            // the special tokens are added as llama-server does (process_mtmd_prompt)
            prompt.chunks.push_back(decision_tokenize_media(mctx, prompt_text, media, true));
        } else {
            // piece by piece, as a prompt without images; a piece takes the images of its markers
            size_t next_image = 0;
            for (const std::string & piece : pieces) {
                size_t n = 0;
                for (size_t pos = piece.find(mtmd_default_marker()); pos != std::string::npos; pos = piece.find(mtmd_default_marker(), pos + 1)) {
                    n++;
                }
                const std::vector<const mtmd_bitmap *> piece_bitmaps(media.begin() + next_image, media.begin() + next_image + n);
                next_image += n;
                auto chunks = decision_tokenize_media(mctx, piece, piece_bitmaps, false);
                if (mtmd_input_chunks_size(chunks.get()) > 0) {
                    prompt.chunks.push_back(std::move(chunks));
                }
            }
        }
        if (profile.readout == LLAMA_RN_DECISION_READOUT_POOLED) {
            // the pooled output holds one score per class
            prompt.pooled   = true;
            prompt.n_scores = question.options.size();
        } else {
            fill_labels(question, prompt);
        }
        return prompt;
    }

    // a template that writes no separator yields a single piece, i.e. the whole prompt at once
    for (const std::string & piece : string_split(prompt_text, DECISION_SEP)) {
        const std::vector<llama_token> piece_tokens = common_tokenize(vocab, piece, false, true);
        prompt.tokens.insert(prompt.tokens.end(), piece_tokens.begin(), piece_tokens.end());
    }
    if (prompt.tokens.empty()) {
        throw std::runtime_error("the decision prompt is empty");
    }

    switch (profile.readout) {
        case LLAMA_RN_DECISION_READOUT_LABEL_LOGITS:
            fill_labels(question, prompt);
            break;
        case LLAMA_RN_DECISION_READOUT_MARKER_EMBD:
            fill_prompt_laya(prompt.tokens, question, prompt);
            break;
        case LLAMA_RN_DECISION_READOUT_POINTER:
            // an option is read at its end token, the question at the last token
            for (size_t i = 0; i < prompt.tokens.size(); i++) {
                if (prompt.tokens[i] == token_marker) {
                    prompt.markers.push_back(i);
                }
            }
            if (prompt.markers.size() != question.options.size()) {
                throw std::runtime_error("unexpected layout of the decision prompt");
            }
            prompt.pointer = prompt.tokens.size() - 1;
            break;
        case LLAMA_RN_DECISION_READOUT_POOLED:
            // minicpm-ocr-jev: the pooled output holds one score per class
            prompt.pooled   = true;
            prompt.n_scores = question.options.size();
            break;
        case LLAMA_RN_DECISION_READOUT_JOINT:
            GGML_ABORT("a joint prompt is filled by fill_prompt_joint()");
    }
    return prompt;
}

// the prompt is: [cls] question [sep] ([marker] option)* [sep] state [sep]
// options and question are cut to fit max_head_tokens, the same way the model was trained
void llama_rn_decision_context::fill_prompt_laya(std::vector<llama_token> & tokens, const llama_rn_decision_question & question, llama_rn_decision_prompt & prompt) const {
    const size_t n_options = question.options.size();

    std::vector<size_t> markers;
    for (size_t i = 0; i < tokens.size(); i++) {
        if (tokens[i] == token_marker) {
            markers.push_back(i);
        }
    }
    const auto invalid = std::runtime_error("unexpected layout of the decision prompt");
    if (markers.size() != n_options || markers[0] < 2 || tokens[markers[0] - 1] != token_sep || tokens.back() != token_sep) {
        throw invalid;
    }
    const size_t head_end = markers[0] - 1;
    const size_t opts_end = std::find(tokens.begin() + markers.back(), tokens.end(), token_sep) - tokens.begin();
    if (opts_end + 1 >= tokens.size()) {
        throw invalid;
    }

    // marker + text of each option
    std::vector<std::vector<llama_token>> options;
    size_t n_options_tokens = 0;
    auto set_max = [&](size_t n_max) {
        n_options_tokens = 0;
        for (auto & opt : options) {
            opt.resize(std::min(opt.size(), n_max));
            n_options_tokens += opt.size();
        }
    };
    for (size_t i = 0; i < n_options; i++) {
        const size_t end = i + 1 < n_options ? markers[i + 1] : opts_end;
        options.emplace_back(tokens.begin() + markers[i], tokens.begin() + end);
    }
    set_max(max_option_tokens + 1);
    if (n_options_tokens + 16 > max_head_tokens) {
        // too many or too long options, shrink them evenly
        set_max(std::max((size_t) 4, (max_head_tokens - std::min(max_head_tokens, (size_t) 16)) / n_options));
    }
    const size_t n_question_max = std::max((size_t) 8, max_head_tokens - std::min(max_head_tokens, n_options_tokens));

    std::vector<llama_token> out;
    out.push_back(tokens[0]);
    out.insert(out.end(), tokens.begin() + 1, tokens.begin() + std::min(head_end, 1 + n_question_max));
    out.push_back(token_sep);
    for (const auto & opt : options) {
        prompt.markers.push_back(out.size());
        out.insert(out.end(), opt.begin(), opt.end());
    }
    out.insert(out.end(), tokens.begin() + opts_end, tokens.end());
    tokens = std::move(out);

    // the output has one score per question type
    prompt.column = question.type;
}

// Each piece is cut to its budget as the model was trained (d1-omni prompt.py: encode), the state
// gets the room that is left. The media come first, the text after them has a budget of its own.
void llama_rn_decision_context::fill_prompt_budget(
        const llama_rn_decision_request & request,
        size_t i_question,
        size_t variant,
        mtmd_context * mctx,
        const std::vector<const mtmd_bitmap *> & bitmaps,
        llama_rn_decision_prompt & prompt) const {
    const auto & question = request.questions[i_question];
    const auto invalid = std::runtime_error("unexpected layout of the decision prompt");

    // the prompt depends on the kind of media, mtmd tells an audio clip from an image by its content
    std::string markers;
    size_t n_media  = 0;
    bool   is_audio = false;
    if (!request.images.empty()) {
        for (size_t i = 0; i < request.images.size(); i++) {
            markers += mtmd_default_marker();
        }
        prompt.media = decision_tokenize_media(mctx, markers, bitmaps, true);
        n_media = mtmd_helper_get_n_tokens(prompt.media.get());
        for (size_t i = 0; i < mtmd_input_chunks_size(prompt.media.get()); i++) {
            is_audio = is_audio || mtmd_input_chunk_get_type(mtmd_input_chunks_get(prompt.media.get(), i)) == MTMD_INPUT_CHUNK_TYPE_AUDIO;
        }
    }
    if (is_audio && request.images.size() > 1) {
        throw std::invalid_argument("a request has images or one audio clip, not both");
    }

    const std::string prompt_text = render(request.state, request.questions, question, variant, request.images.size(), is_audio);
    const std::vector<std::string> pieces = string_split(prompt_text, DECISION_SEP);
    if (pieces.empty() || pieces[0] != markers) {
        throw invalid;
    }

    size_t n_max = profile.max_tokens;
    if (!request.images.empty()) {
        n_max = std::min(is_audio ? profile.max_tokens_audio : profile.max_tokens_image, profile.max_tokens - std::min(profile.max_tokens, n_media));
        if (n_max < 64) {
            throw std::invalid_argument(string_format("the media take %zu of the %zu positions, send fewer images", n_media, profile.max_tokens));
        }
    }

    // the options get max(96, min(24 n + 32, max / 2)) tokens, shared evenly
    const int64_t n_options      = question.options.size();
    const int64_t n_budget       = std::max<int64_t>(96, std::min<int64_t>(24 * n_options + 32, n_max / 2));
    const int64_t n_option_max   = std::max<int64_t>(2, (n_budget - 3 * n_options) / n_options);
    const int64_t n_question_max = std::max<int64_t>(16, n_budget);

    std::vector<llama_token> head; // before the state
    std::vector<llama_token> body; // the state
    std::vector<llama_token> tail; // after the state
    bool has_state = false;
    for (size_t i_piece = 1; i_piece < pieces.size(); i_piece++) {
        std::string piece = pieces[i_piece];
        int64_t n_piece_max = -1;
        bool is_state = false;
        if (string_starts_with(piece, DECISION_STATE)) {
            piece    = piece.substr(DECISION_STATE.size());
            is_state = true;
        } else if (string_starts_with(piece, DECISION_QUESTION)) {
            piece       = piece.substr(DECISION_QUESTION.size());
            n_piece_max = n_question_max;
        } else if (string_starts_with(piece, DECISION_OPTION)) {
            piece       = piece.substr(DECISION_OPTION.size());
            n_piece_max = n_option_max;
        }

        std::vector<llama_token> tokens = common_tokenize(vocab, piece, false, true);
        if (n_piece_max >= 0 && (int64_t) tokens.size() > n_piece_max) {
            tokens.resize(n_piece_max);
        }

        if (is_state) {
            if (has_state) {
                throw invalid;
            }
            body      = std::move(tokens);
            has_state = true;
        } else {
            auto & dst = has_state ? tail : head;
            dst.insert(dst.end(), tokens.begin(), tokens.end());
        }
    }
    if (!has_state) {
        throw invalid;
    }

    const size_t n_room = n_max - std::min(n_max, head.size() + tail.size());
    body.resize(std::min(body.size(), n_room));

    prompt.tokens = std::move(head);
    prompt.tokens.insert(prompt.tokens.end(), body.begin(), body.end());
    prompt.tokens.insert(prompt.tokens.end(), tail.begin(), tail.end());
    prompt.tokens.resize(std::min(prompt.tokens.size(), n_max));

    for (size_t i = 0; i < prompt.tokens.size(); i++) {
        if (prompt.tokens[i] == token_marker) {
            prompt.markers.push_back(n_media + i);
        }
    }
    if ((int64_t) prompt.markers.size() != n_options) {
        throw std::invalid_argument("the options do not fit in the context");
    }

    // the output has one score per question type
    prompt.column = question.type;
}

// given to the template: text between the pieces of the prompt, and at the start of the span of a question or of an option
llama_rn_decision_prompt llama_rn_decision_context::fill_prompt_joint(
        const llama_rn_decision_request & request,
        mtmd_context * mctx,
        const std::vector<const mtmd_bitmap *> & bitmaps,
        int32_t only_question) const {
    const auto & questions = request.questions;
    const size_t first_q = only_question >= 0 ? (size_t) only_question : 0;
    const size_t n_q     = only_question >= 0 ? 1 : questions.size();

    json inp_questions = json::array();
    for (size_t qi = first_q; qi < first_q + n_q; qi++) {
        const auto & question = questions[qi];
        json options = json::array();
        for (const auto & opt : question.options) {
            options.push_back(json{
                {"key",         opt.key},
                {"description", opt.description},
            });
        }
        inp_questions.push_back(json{
            {"id",           question.id},
            {"type",         decision_question_type_name(question.type)},
            {"instructions", question.instructions},
            {"options",      options},
        });
    }

    // the template is given raw JSON values, and no marker in the input
    json inp = json{
        {"state",     request.state},
        {"questions", inp_questions},
    };
    if (profile.sort_keys) {
        inp = decision_sort_keys(inp);
    }
    inp = decision_replace_text(inp, DECISION_MARKER, DECISION_NEUTERED);

    // the template puts one media marker per image
    json images = json::array();
    if (!request.images.empty()) {
        inp = decision_replace_text(inp, mtmd_default_marker(), " ");
        for (size_t i = 0; i < request.images.size(); i++) {
            images.push_back(mtmd_default_marker());
        }
    }
    inp["images"]        = images;
    inp["sep"]           = DECISION_SEP;
    inp["mark_question"] = DECISION_QUESTION;
    inp["mark_option"]   = DECISION_OPTION;

    const std::string prompt_text = decision_apply_template(*tmpl, inp);
    const auto invalid = std::runtime_error("unexpected layout of the decision prompt");

    // the model was trained with the pieces tokenized one by one
    const std::vector<std::string> pieces = string_split(prompt_text, DECISION_SEP);
    size_t i_piece = 0;

    llama_rn_decision_prompt prompt;
    prompt.question = first_q;
    if (!request.images.empty()) {
        // the text before the images is tokenized with them, a vision token separates the pieces anyway
        std::string head;
        while (i_piece < pieces.size() && head.find(mtmd_default_marker()) == std::string::npos) {
            head += pieces[i_piece++];
        }
        if (head.find(mtmd_default_marker()) == std::string::npos || head.find(DECISION_MARKER) != std::string::npos) {
            throw invalid;
        }
        prompt.media = decision_tokenize_media(mctx, head, bitmaps, true);
    }

    size_t i_question = 0;
    for (; i_piece < pieces.size(); i_piece++) {
        std::string piece = pieces[i_piece];
        int32_t order = LLAMA_DECISION_ORDER_NONE;
        if (string_starts_with(piece, DECISION_QUESTION)) {
            piece = piece.substr(DECISION_QUESTION.size());
            if (i_question >= n_q) {
                throw invalid;
            }
            switch (questions[first_q + i_question++].type) {
                case LLAMA_RN_DECISION_QUESTION_NOUL:   order = LLAMA_DECISION_ORDER_QUESTION_NOUL;   break;
                case LLAMA_RN_DECISION_QUESTION_CHOICE: order = LLAMA_DECISION_ORDER_QUESTION_CHOICE; break;
                case LLAMA_RN_DECISION_QUESTION_SCORE:  order = LLAMA_DECISION_ORDER_QUESTION_SCORE;  break;
            }
        } else if (string_starts_with(piece, DECISION_OPTION)) {
            piece = piece.substr(DECISION_OPTION.size());
            order = LLAMA_DECISION_ORDER_OPTION;
            prompt.n_scores++;
        }

        const std::vector<llama_token> piece_tokens = common_tokenize(vocab, piece, false, true);
        if (order != LLAMA_DECISION_ORDER_NONE && piece_tokens.empty()) {
            throw std::invalid_argument("the instructions and the options of a question must not be empty");
        }
        prompt.tokens.insert(prompt.tokens.end(), piece_tokens.begin(), piece_tokens.end());
        prompt.order.resize(prompt.tokens.size(), order);
    }

    size_t n_options = 0;
    for (size_t qi = first_q; qi < first_q + n_q; qi++) {
        n_options += questions[qi].options.size();
    }
    if (i_question != n_q || (size_t) prompt.n_scores != n_options) {
        throw invalid;
    }
    return prompt;
}

//
// answer
//

float llama_rn_decision_context::get_temperature(const llama_rn_decision_question & question) const {
    const size_t n = question.options.size();
    const std::string type_name = decision_question_type_name(question.type);

    // the temperature can depend on the number of options, the buckets are the ones used to fit it
    std::string bucket;
    if (profile.temperature_buckets == LLAMA_RN_DECISION_TEMPERATURE_BUCKETS_SIZE) {
        bucket = n <= 8 ? "small" : n <= 26 ? "mid" : "large";
    } else {
        bucket = n <= 2 ? "2" : n <= 5 ? "3_5" : n <= 10 ? "6_10" : "11";
    }

    for (const auto & name : {type_name + "." + bucket, type_name}) {
        const auto it = temperatures.find(name);
        if (it != temperatures.end()) {
            return it->second;
        }
    }
    return 1.0f;
}

// confidence formulas are the ones published by TypeSafe

static double decision_confidence_choice(const std::vector<double> & probs) {
    if (probs.size() < 2) {
        return 1.0;
    }
    const double uniform = 1.0 / probs.size();
    const double p_max   = *std::max_element(probs.begin(), probs.end());
    return std::max(0.0, (p_max - uniform) / (1.0 - uniform));
}

static double decision_confidence_score(const std::vector<double> & probs) {
    if (probs.size() < 2) {
        return 1.0;
    }
    const size_t n    = probs.size();
    const size_t mode = std::max_element(probs.begin(), probs.end()) - probs.begin();

    // mean distance to the mode, relative to the one of a uniform distribution around its center
    double dist         = 0.0;
    double dist_uniform = 0.0;
    for (size_t i = 0; i < n; i++) {
        dist         += probs[i] * std::fabs((double) i - (double) mode);
        dist_uniform += std::fabs((double) i - (n - 1) / 2.0) / n;
    }
    return std::max(0.0, 1.0 - dist / dist_uniform);
}

// VLLM_DECISION2, VLLM_D3: normalized entropy, as in the reference implementation
static double decision_confidence_entropy(const std::vector<double> & probs) {
    if (probs.size() < 2) {
        return 1.0;
    }
    double entropy = 0.0;
    for (const double p : probs) {
        if (p > 0.0) {
            entropy -= p * std::log(p);
        }
    }
    return std::max(0.0, std::min(1.0, 1.0 - entropy / std::log((double) probs.size())));
}

double llama_rn_decision_context::get_decision_confidence(llama_rn_decision_question_type question_type, const std::vector<double> & probs) const {
    if (profile.confidence_entropy) {
        return decision_confidence_entropy(probs);
    }
    // lfm3-d1: the probability of the answer (d1 api.py)
    if (profile.confidence_max_prob) {
        return *std::max_element(probs.begin(), probs.end());
    }
    switch (question_type) {
        case LLAMA_RN_DECISION_QUESTION_CHOICE: return decision_confidence_choice(probs);
        case LLAMA_RN_DECISION_QUESTION_SCORE:  return decision_confidence_score(probs);
        case LLAMA_RN_DECISION_QUESTION_NOUL:   break;
    }
    throw std::runtime_error("no confidence for this question type");
}

json llama_rn_decision_context::format_answer(const llama_rn_decision_question & question, const std::vector<std::vector<float>> & scores, bool has_media) const {
    const size_t n = n_outputs(question);
    if (scores.size() != n_variants(question)) {
        throw std::runtime_error("decision result does not match the number of variants");
    }

    // softmax over the outputs of each variant, then the average of the variants
    float temperature = has_media && profile.media_raw_probs ? 1.0f : get_temperature(question);
    // a model can have its own temperature for requests with media
    const auto it_image = temperatures.find("image");
    if (has_media && it_image != temperatures.end()) {
        temperature = it_image->second;
    }
    std::vector<double> probs(n, 0.0);
    for (size_t v = 0; v < scores.size(); v++) {
        const auto & s = scores[v];
        if (s.size() != n) {
            throw std::runtime_error("decision result does not match the number of options");
        }
        // a joint head returns NaN if it could not use the decision order, +inf would make the softmax NaN
        if (std::any_of(s.begin(), s.end(), [](float x) { return std::isnan(x) || x == INFINITY; })) {
            throw std::runtime_error("the model could not evaluate the decision");
        }
        std::vector<double> logits(s.begin(), s.end());
        if (question.type == LLAMA_RN_DECISION_QUESTION_SCORE && score_bias.count(n) > 0) {
            for (size_t i = 0; i < n; i++) {
                logits[i] += score_bias.at(n)[i];
            }
        }
        const double logit_max = *std::max_element(logits.begin(), logits.end());
        std::vector<double> p(n);
        double sum = 0.0;
        for (size_t i = 0; i < n; i++) {
            p[i] = std::exp((logits[i] - logit_max) / temperature);
            sum += p[i];
        }
        for (size_t i = 0; i < n; i++) {
            // the second variant is in the reverse order
            probs[v == 0 ? i : n - 1 - i] += p[i] / sum / scores.size();
        }
    }

    json answer = json{{"type", decision_question_type_name(question.type)}};

    if (question.type == LLAMA_RN_DECISION_QUESTION_NOUL) {
        if (profile.n_noul_ratings > 0) {
            double expected = 0.0;
            for (size_t i = 0; i < n; i++) {
                expected += probs[i] * i / (n - 1);
            }
            answer["noul"] = expected;
            return answer;
        }
        for (size_t i = 0; i < n; i++) {
            if (question.options[i].key == "true") {
                answer["noul"] = probs[i];
            }
        }
        return answer;
    }

    json probabilities = json::object();
    for (size_t i = 0; i < n; i++) {
        probabilities[question.options[i].key] = probs[i];
    }

    if (question.type == LLAMA_RN_DECISION_QUESTION_CHOICE) {
        const size_t best = std::max_element(probs.begin(), probs.end()) - probs.begin();
        answer["choice"]        = question.options[best].key;
        answer["probabilities"] = probabilities;
        answer["confidence"]    = get_decision_confidence(question.type, probs);
    } else {
        double expected = 0.0;
        json legend = json::object();
        for (size_t i = 0; i < n; i++) {
            expected += i * probs[i];
            const json & description = question.options[i].description;
            legend[question.options[i].key] = profile.canonical_legend && !description.is_string()
                ? json(decision_sort_keys(description).dump()) : description;
        }
        // lfm3-d1: the median level, the first whose cumulative probability reaches one half (d1 api.py)
        if (profile.score_median) {
            double mass = 0.0;
            size_t level = n - 1;
            for (size_t i = 0; i < n && mass < 0.5; i++) {
                mass += probs[i];
                level = i;
            }
            expected = (double) level;
        }
        answer["score"]         = expected;
        answer["legend"]        = legend;
        answer["probabilities"] = probabilities;
        answer["confidence"]    = get_decision_confidence(question.type, probs);
    }
    return answer;
}

json llama_rn_decision_context::format_result(
        const llama_rn_decision_request & request,
        const std::vector<llama_rn_decision_prompt> & prompts,
        const std::vector<std::vector<float>> & scores,
        const std::string & model_name) const {
    if (scores.size() != prompts.size()) {
        throw std::runtime_error("decision result does not match the number of prompts");
    }

    size_t n_input_tokens = 0;
    for (const auto & prompt : prompts) {
        n_input_tokens += prompt.n_tokens();
    }

    const bool has_media = !request.images.empty();

    json answers = json::object();
    if (is_joint() && profile.joint_per_question) {
        // one prompt per question (vllm-decision2), its scores are the options of that question
        for (size_t i = 0; i < prompts.size(); i++) {
            const auto & question = request.questions[prompts[i].question];
            answers[question.id] = format_answer(question, {scores[i]}, has_media);
        }
    } else if (is_joint()) {
        // the scores of all the options of all the questions, in order
        size_t i_score = 0;
        for (const auto & question : request.questions) {
            const size_t n = question.options.size();
            if (i_score + n > scores[0].size()) {
                throw std::runtime_error("decision result does not match the number of options");
            }
            answers[question.id] = format_answer(question, {std::vector<float>(scores[0].begin() + i_score, scores[0].begin() + i_score + n)}, has_media);
            i_score += n;
        }
    } else {
        std::vector<std::vector<std::vector<float>>> per_question(request.questions.size());
        for (size_t i = 0; i < prompts.size(); i++) {
            auto & variants = per_question[prompts[i].question];
            if (prompts[i].pooled) {
                // one prompt per option, each scored to one number
                if (variants.empty()) {
                    variants.emplace_back();
                }
                variants[0].insert(variants[0].end(), scores[i].begin(), scores[i].end());
                continue;
            }
            variants.push_back(scores[i]);
        }
        for (size_t i = 0; i < request.questions.size(); i++) {
            answers[request.questions[i].id] = format_answer(request.questions[i], per_question[i], has_media);
        }
    }

    return json{
        {"model",   model_name},
        {"answers", answers},
        {"usage",   json{
            {"input_tokens",  n_input_tokens},
            {"output_tokens", 0},
        }},
    };
}

//
// evaluation
//

// the logits of the labels, the max of each group if the labels are grouped
static std::vector<float> decision_read_labels(const float * logits, const llama_rn_decision_prompt & prompt, size_t n_labels) {
    std::vector<float> scores;
    for (size_t i = 0; i < n_labels; i++) {
        scores.push_back(logits[prompt.labels[i]]);
    }
    if (prompt.label_groups.empty()) {
        return scores;
    }
    std::vector<float> out;
    size_t i = 0;
    for (const int32_t n : prompt.label_groups) {
        GGML_ASSERT(n > 0 && i + n <= scores.size());
        out.push_back(*std::max_element(scores.begin() + i, scores.begin() + i + n));
        i += n;
    }
    return out;
}

// Evaluate a prompt with media in one batch from the start of the sequence, the media as embeddings
// next to the text (llama-server: add_prompt_mixed). A model without memory, or with non-causal
// attention, must see the whole prompt at once. Every entry is an output, batch index = prompt position.
static void decision_eval_mixed(
        mtmd_context * mctx,
        llama_context * ctx,
        llama_seq_id seq_id,
        const llama_rn_decision_prompt & prompt) {
    GGML_ASSERT(mctx != nullptr && prompt.media);
    const size_t n_total = prompt.n_tokens();
    if (n_total > (size_t) llama_n_batch(ctx)) {
        throw std::runtime_error(string_format(
            "the decision prompt does not fit in one batch (%zu tokens, n_batch = %u)", n_total, llama_n_batch(ctx)));
    }

    llama_memory_t mem = llama_get_memory(ctx);
    if (mem != nullptr) {
        llama_memory_seq_rm(mem, seq_id, -1, -1);
    }

    const int32_t n_embd   = llama_model_n_embd_inp(llama_get_model(ctx));
    const bool    is_mrope = mtmd_decode_use_mrope(mctx);
    const mtmd_input_chunks * chunks = prompt.media.get();

    // the batch only points at the embeddings, they are kept until it is decoded
    std::vector<std::vector<float>> embds;
    mtmd::batch_ptr mbatch;
    common_batch batch(ctx);
    llama_pos n_past = 0;
    for (size_t i = 0; i < mtmd_input_chunks_size(chunks); i++) {
        const mtmd_input_chunk * chunk = mtmd_input_chunks_get(chunks, i);
        const size_t n_tokens = mtmd_input_chunk_get_n_tokens(chunk);
        if (mtmd_input_chunk_get_type(chunk) == MTMD_INPUT_CHUNK_TYPE_TEXT) {
            size_t n_text = 0;
            const llama_token * tokens = mtmd_input_chunk_get_tokens_text(chunk, &n_text);
            for (size_t j = 0; j < n_text; j++) {
                batch.add(tokens[j], n_past++, seq_id, true);
            }
            continue;
        }

        // encoded with the media chunks that follow it, as llama-server does (encode_mtmd_chunk)
        const float * out = mbatch ? mtmd_batch_get_output_embd(mbatch.get(), chunk) : nullptr;
        if (out == nullptr) {
            mbatch.reset(mtmd_batch_init(mctx));
            GGML_ASSERT(mtmd_batch_add_chunk(mbatch.get(), chunk) == 0);
            for (size_t k = i + 1; k < mtmd_input_chunks_size(chunks); k++) {
                const mtmd_input_chunk * next = mtmd_input_chunks_get(chunks, k);
                if (mtmd_input_chunk_get_type(next) != MTMD_INPUT_CHUNK_TYPE_TEXT && mtmd_batch_add_chunk(mbatch.get(), next) != 0) {
                    break; // the batch is full, or the chunk cannot go with the others
                }
            }
            if (mtmd_batch_encode(mbatch.get()) != 0) {
                throw std::runtime_error("failed to encode the media of the decision prompt");
            }
            out = mtmd_batch_get_output_embd(mbatch.get(), chunk);
            GGML_ASSERT(out != nullptr);
        }
        embds.emplace_back(out, out + n_tokens * n_embd);
        const float * embd = embds.back().data();

        // positions are the ones of mtmd_helper_decode_image_chunk()
        const mtmd_image_tokens * image = mtmd_input_chunk_get_tokens_image(chunk);
        for (size_t j = 0; j < n_tokens; j++) {
            const llama_pos p = n_past + (llama_pos) j;
            llama_pos pos[GGML_MROPE_SECTIONS] = { p, p, p, p };
            if (is_mrope && image) {
                const mtmd_decoder_pos rel = mtmd_image_tokens_get_decoder_pos(image, n_past, j);
                pos[0] = rel.t;
                pos[1] = rel.y;
                pos[2] = rel.x;
                pos[3] = rel.z;
            }
            batch.add_embd({ embd + j * n_embd, 1, (size_t) n_embd }, pos, seq_id, true);
        }
        n_past += mtmd_input_chunk_get_n_pos(chunk);
    }

    // the decision order covers the text after the media, the media have none
    for (size_t j = 0; j < prompt.tokens.size(); j++) {
        const int32_t idx = batch.add(prompt.tokens[j], n_past++, seq_id, true);
        if (!prompt.order.empty()) {
            batch.tokens[idx].decision_order = prompt.order[j];
        }
    }
    GGML_ASSERT((size_t) batch.size() == n_total);

    const int32_t ret = llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get());
    if (ret != 0) {
        if (mem != nullptr) {
            llama_memory_seq_rm(mem, seq_id, -1, -1);
        }
        throw std::runtime_error(string_format("failed to evaluate the decision prompt (error %d)", ret));
    }
}

// the chunks of a prompt's pieces, in order
static std::vector<const mtmd_input_chunk *> decision_flatten(const std::vector<std::shared_ptr<mtmd_input_chunks>> & pieces) {
    std::vector<const mtmd_input_chunk *> out;
    for (const auto & piece : pieces) {
        for (size_t i = 0; i < mtmd_input_chunks_size(piece.get()); i++) {
            out.push_back(mtmd_input_chunks_get(piece.get(), i));
        }
    }
    return out;
}

// Evaluate a prompt with images from the start of the sequence, the way llama-server does
// (process_mtmd_chunk): a media chunk is encoded in one mtmd batch with the media chunks
// that follow it, then its embeddings are decoded; text is decoded in batches of n_batch.
// Only the last token of the prompt has an output. Evaluation starts at token first_token of
// chunk first_chunk, at position n_past: what comes before is already in the sequence.
static void decision_eval_chunks(
        mtmd_context * mctx,
        llama_context * ctx,
        llama_seq_id seq_id,
        const std::vector<const mtmd_input_chunk *> & chunks,
        llama_pos & n_past,
        bool logits_last,
        size_t first_chunk = 0,
        size_t first_token = 0) {
    const size_t  n_chunks = chunks.size();
    const int32_t n_batch  = llama_n_batch(ctx);

    mtmd::batch_ptr mbatch;
    for (size_t i = first_chunk; i < n_chunks; i++) {
        const mtmd_input_chunk * chunk = chunks[i];
        const bool is_last = i + 1 == n_chunks;

        if (mtmd_input_chunk_get_type(chunk) == MTMD_INPUT_CHUNK_TYPE_TEXT) {
            size_t n_tokens = 0;
            const llama_token * tokens = mtmd_input_chunk_get_tokens_text(chunk, &n_tokens);
            const size_t first = i == first_chunk ? first_token : 0;
            common_batch batch(ctx);
            for (size_t j = first; j < n_tokens; j += n_batch) {
                const size_t end = std::min(n_tokens, j + n_batch);
                batch.clear();
                for (size_t k = j; k < end; k++) {
                    batch.add(tokens[k], n_past + (k - first), seq_id, logits_last && is_last && k + 1 == n_tokens);
                }
                const int32_t ret = llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get());
                if (ret != 0) {
                    throw std::runtime_error(string_format("failed to evaluate the decision prompt (error %d)", ret));
                }
            }
            n_past += n_tokens - first;
            continue;
        }

        if (is_last && logits_last) {
            throw std::runtime_error("a decision prompt must end with text");
        }

        float * embd = mbatch ? mtmd_batch_get_output_embd(mbatch.get(), chunk) : nullptr;
        if (embd == nullptr) {
            mbatch.reset(mtmd_batch_init(mctx));
            GGML_ASSERT(mtmd_batch_add_chunk(mbatch.get(), chunk) == 0);
            for (size_t j = i + 1; j < n_chunks; j++) {
                const mtmd_input_chunk * next = chunks[j];
                if (mtmd_input_chunk_get_type(next) == MTMD_INPUT_CHUNK_TYPE_TEXT) {
                    continue;
                }
                if (mtmd_batch_add_chunk(mbatch.get(), next) != 0) {
                    break; // the batch is full, or the chunk cannot go with the others
                }
            }
            if (mtmd_batch_encode(mbatch.get()) != 0) {
                throw std::runtime_error("failed to encode the images of the decision prompt");
            }
            embd = mtmd_batch_get_output_embd(mbatch.get(), chunk);
            GGML_ASSERT(embd != nullptr);
        }

        llama_pos new_n_past = n_past;
        if (mtmd_helper_decode_image_chunk(mctx, ctx, chunk, embd, n_past, seq_id, n_batch, &new_n_past, nullptr, nullptr) != 0) {
            throw std::runtime_error("failed to decode the images of the decision prompt");
        }
        n_past = new_n_past;
    }
}

// Where two prompts with images first differ: a media chunk matches when it is the same image
// (its id is the image's hash) with the same tokens, a text chunk token by token. The last token of
// b is always evaluated again, its logits are read.
static void decision_common_prefix(
        const std::vector<const mtmd_input_chunk *> & a,
        const std::vector<const mtmd_input_chunk *> & b,
        size_t & i_chunk,
        size_t & i_token,
        llama_pos & n_pos) {
    i_chunk = 0;
    i_token = 0;
    n_pos   = 0;
    const size_t n_b = b.size();
    const size_t n   = std::min(a.size(), n_b);
    for (; i_chunk < n; i_chunk++) {
        const mtmd_input_chunk * ca = a[i_chunk];
        const mtmd_input_chunk * cb = b[i_chunk];
        const auto type = mtmd_input_chunk_get_type(cb);
        if (mtmd_input_chunk_get_type(ca) != type) {
            return;
        }
        if (type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
            size_t n_a = 0;
            size_t n_b_tokens = 0;
            const llama_token * ta = mtmd_input_chunk_get_tokens_text(ca, &n_a);
            const llama_token * tb = mtmd_input_chunk_get_tokens_text(cb, &n_b_tokens);
            size_t k = 0;
            while (k < n_a && k < n_b_tokens && ta[k] == tb[k]) {
                k++;
            }
            if (k < n_a || k < n_b_tokens) {
                i_token = k;
                n_pos  += k;
                break;
            }
            n_pos += k;
            continue;
        }
        const char * id_a = mtmd_input_chunk_get_id(ca);
        const char * id_b = mtmd_input_chunk_get_id(cb);
        if (id_a == nullptr || id_b == nullptr || std::strcmp(id_a, id_b) != 0 ||
            mtmd_input_chunk_get_n_tokens(ca) != mtmd_input_chunk_get_n_tokens(cb)) {
            return;
        }
        n_pos += mtmd_input_chunk_get_n_pos(cb);
    }

    // the last token of b must be evaluated: if b matched up to its very end, step back onto it
    if (i_chunk >= n_b) {
        i_chunk = n_b - 1;
        i_token = mtmd_input_chunk_get_n_tokens(b[i_chunk]);
    }
    const mtmd_input_chunk * last = b[n_b - 1];
    const size_t n_last = mtmd_input_chunk_get_n_tokens(last);
    if (i_chunk == n_b - 1 && i_token >= n_last) {
        if (mtmd_input_chunk_get_type(last) != MTMD_INPUT_CHUNK_TYPE_TEXT || n_last == 0) {
            i_chunk = 0; // a prompt ends with text, see decision_eval_chunks()
            i_token = 0;
            n_pos   = 0;
            return;
        }
        i_token = n_last - 1;
        n_pos  -= 1;
    }
}

// system_one with images: the state pieces from the start of the sequence (text, or text and
// images), then the question blocks in one batch, whose slots are read there
static std::vector<float> decision_eval_state_pieces(
        llama_context * ctx,
        llama_seq_id seq_id,
        const llama_rn_decision_prompt & prompt,
        mtmd_context * mctx) {
    llama_memory_t mem   = llama_get_memory(ctx);
    const size_t n_batch = llama_n_batch(ctx);
    if (mem != nullptr) {
        llama_memory_seq_rm(mem, seq_id, -1, -1);
    }
    if (prompt.tokens.size() > n_batch) {
        throw std::runtime_error(string_format(
            "the questions of the decision prompt do not fit in one batch (%zu tokens, n_batch = %zu)",
            prompt.tokens.size(), n_batch));
    }

    common_batch batch(ctx);
    auto decode = [&](const std::vector<llama_token> & tokens, llama_pos n_past, const std::vector<int32_t> & outputs) {
        for (size_t i = 0; i < tokens.size(); i += n_batch) {
            const size_t end = std::min(tokens.size(), i + n_batch);
            batch.clear();
            for (size_t j = i; j < end; j++) {
                const bool output = std::find(outputs.begin(), outputs.end(), (int32_t) j) != outputs.end();
                batch.add(tokens[j], n_past + j, seq_id, output);
            }
            const int32_t ret = llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get());
            if (ret != 0) {
                throw std::runtime_error(string_format("failed to evaluate the decision prompt (error %d)", ret));
            }
        }
    };

    std::vector<float> scores;
    try {
        llama_pos n_past = 0;
        for (const auto & piece : prompt.state_pieces) {
            if (piece.chunks) {
                GGML_ASSERT(mctx != nullptr);
                decision_eval_chunks(mctx, ctx, seq_id, decision_flatten({piece.chunks}), n_past, false);
            } else {
                decode(piece.tokens, n_past, {});
                n_past += piece.tokens.size();
            }
        }
        decode(prompt.tokens, n_past, prompt.slots);

        for (size_t s = 0; s < prompt.slots.size(); s++) {
            const float * logits = llama_get_logits_ith(ctx, prompt.slots[s]);
            if (logits == nullptr) {
                throw std::runtime_error("failed to get logits");
            }
            for (int32_t i = 0; i < prompt.slot_n_labels[s]; i++) {
                scores.push_back(logits[prompt.labels[i]]);
            }
        }
    } catch (...) {
        if (mem != nullptr) {
            llama_memory_seq_rm(mem, seq_id, -1, -1);
        }
        throw;
    }
    return scores;
}

static std::vector<float> decision_read_outputs(
        llama_context * ctx,
        llama_seq_id seq_id,
        const llama_rn_decision_prompt & prompt,
        size_t n_tokens,
        size_t i_last);

std::vector<float> llama_rn_decision_eval(
        llama_context * ctx,
        llama_seq_id seq_id,
        const llama_rn_decision_prompt & prompt,
        llama_rn_decision_cache & cache,
        mtmd_context * mctx) {
    if (!prompt.chunks.empty()) {
        // continue from the previous prompt where the two agree: its images are encoded and
        // decoded already, as llama-server keeps them in the slot
        llama_memory_t mem = llama_get_memory(ctx);
        GGML_ASSERT(mctx != nullptr);
        size_t    first_chunk = 0;
        size_t    first_token = 0;
        llama_pos n_past      = 0;
        const std::vector<const mtmd_input_chunk *> chunks = decision_flatten(prompt.chunks);
        if (mem != nullptr && !cache.chunks.empty() && !prompt.no_prefix_cache) {
            decision_common_prefix(decision_flatten(cache.chunks), chunks, first_chunk, first_token, n_past);
        }
        // a recurrent state cannot be rolled back, start over
        if (mem == nullptr || !llama_memory_seq_rm(mem, seq_id, n_past, -1)) {
            if (mem != nullptr) {
                llama_memory_seq_rm(mem, seq_id, -1, -1);
            }
            first_chunk = 0;
            first_token = 0;
            n_past      = 0;
        }
        cache.clear();
        try {
            decision_eval_chunks(mctx, ctx, seq_id, chunks, n_past, true, first_chunk, first_token);
        } catch (...) {
            if (mem != nullptr) {
                llama_memory_seq_rm(mem, seq_id, -1, -1);
            }
            throw;
        }
        cache.chunks = prompt.chunks;
        const float * logits = llama_get_logits_ith(ctx, -1);
        if (logits == nullptr) {
            throw std::runtime_error("failed to get logits");
        }
        return decision_read_labels(logits, prompt, prompt.labels.size());
    }

    if (!prompt.state_pieces.empty()) {
        cache.clear();
        return decision_eval_state_pieces(ctx, seq_id, prompt, mctx);
    }

    if (prompt.media) {
        cache.clear();
        decision_eval_mixed(mctx, ctx, seq_id, prompt);
        return decision_read_outputs(ctx, seq_id, prompt, prompt.n_tokens(), 0);
    }

    const auto & tokens   = prompt.tokens;
    const size_t n_tokens = tokens.size();
    const size_t n_batch  = llama_n_batch(ctx);
    llama_memory_t mem    = llama_get_memory(ctx);

    // the outputs are read from the last batch, which holds every position that is read
    const size_t first_read = prompt.need_embd() || !prompt.slots.empty() ? (size_t) prompt.pos_first() : n_tokens - 1;
    if (n_tokens - first_read > n_batch || (mem == nullptr && n_tokens > n_batch)) {
        throw std::runtime_error(string_format(
            "the decision prompt does not fit in one batch (%zu tokens, n_batch = %zu)",
            mem == nullptr ? n_tokens : n_tokens - first_read, n_batch));
    }

    // continue from the previous prompt of the sequence, every position that is read is evaluated again
    size_t n_keep = 0;
    if (mem != nullptr) {
        if (prompt.no_prefix_cache) {
            // what the sequence holds says nothing about this prompt (a non-causal model, or
            // short convs with no memory to separate the sequences)
            llama_memory_seq_rm(mem, seq_id, -1, -1);
        } else {
            while (n_keep < cache.tokens.size() && n_keep < first_read && cache.tokens[n_keep] == tokens[n_keep]) {
                n_keep++;
            }
            // a recurrent state cannot be rolled back, start over
            if (!llama_memory_seq_rm(mem, seq_id, n_keep, -1)) {
                llama_memory_seq_rm(mem, seq_id, -1, -1);
                n_keep = 0;
            }
        }
    }
    cache.chunks.clear();
    cache.tokens.assign(tokens.begin(), tokens.begin() + n_keep);

    common_batch batch(ctx);
    size_t i_last = n_keep; // first position of the last batch
    for (size_t i = n_keep; i < n_tokens; ) {
        size_t end = std::min(n_tokens, i + n_batch);
        if (end < n_tokens && end > first_read) {
            end = first_read; // keep what is read together in the next batch
        }
        batch.clear();
        for (size_t j = i; j < end; j++) {
            // the embeddings output needs every token as an output, as llama-server marks them;
            // the logits are read at the last token only
            const bool output = prompt.need_embd() ||
                (prompt.slots.empty() ? j + 1 == n_tokens
                                      : std::find(prompt.slots.begin(), prompt.slots.end(), (int32_t) j) != prompt.slots.end());
            const int32_t idx = batch.add(tokens[j], j, seq_id, output);
            if (!prompt.order.empty()) {
                batch.tokens[idx].decision_order = prompt.order[j];
            }
        }
        const int32_t ret = llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get());
        if (ret != 0) {
            cache.tokens.clear();
            if (mem != nullptr) {
                llama_memory_seq_rm(mem, seq_id, -1, -1);
            }
            throw std::runtime_error(string_format("failed to evaluate the decision prompt (error %d)", ret));
        }
        cache.tokens.insert(cache.tokens.end(), tokens.begin() + i, tokens.begin() + end);
        i_last = i;
        i = end;
    }
    if (mem == nullptr) {
        cache.tokens.clear();
    }

    return decision_read_outputs(ctx, seq_id, prompt, n_tokens, i_last);
}

// the scores of a decoded prompt: n_tokens is its length, i_last its position at index 0 of the last batch
static std::vector<float> decision_read_outputs(
        llama_context * ctx,
        llama_seq_id seq_id,
        const llama_rn_decision_prompt & prompt,
        size_t n_tokens,
        size_t i_last) {
    std::vector<float> scores;

    if (!prompt.labels.empty()) {
        // a slot per question reads its own options' labels, else the last token reads them all
        const std::vector<int32_t> slots = prompt.slots.empty() ? std::vector<int32_t>{(int32_t) n_tokens - 1} : prompt.slots;
        for (size_t s = 0; s < slots.size(); s++) {
            const float * logits = llama_get_logits_ith(ctx, slots[s] - (int32_t) i_last);
            if (logits == nullptr) {
                throw std::runtime_error("failed to get logits");
            }
            const size_t n = prompt.slot_n_labels.empty() ? prompt.labels.size() : (size_t) prompt.slot_n_labels[s];
            const std::vector<float> slot_scores = decision_read_labels(logits, prompt, n);
            scores.insert(scores.end(), slot_scores.begin(), slot_scores.end());
        }
        return scores;
    }

    if (prompt.pooled) {
        const float * embd = llama_get_embeddings_seq(ctx, seq_id);
        if (embd == nullptr) {
            throw std::runtime_error("failed to get the pooled output, is the context in RANK pooling?");
        }
        // RANK_HEAD (legacy) reads one score; POOLED reads one value per class
        const size_t n = prompt.n_scores > 0 ? (size_t) prompt.n_scores : 1;
        for (size_t i = 0; i < n; i++) {
            scores.push_back(embd[i]);
        }
        return scores;
    }

    auto get_embd = [&](size_t pos) {
        const float * embd = llama_get_embeddings_ith(ctx, pos - i_last);
        if (embd == nullptr) {
            throw std::runtime_error("failed to get embeddings");
        }
        return embd;
    };

    // joint head: the scores are the first rows
    for (int32_t i = 0; i < prompt.n_scores; i++) {
        scores.push_back(get_embd(i_last + i)[0]);
    }

    const int32_t n_embd_out = llama_model_n_embd_out(llama_get_model(ctx));
    const int32_t n_pointer  = n_embd_out / 2;
    const float * embd_q = prompt.pointer >= 0 ? get_embd(prompt.pointer) : nullptr;
    GGML_ASSERT(prompt.column >= 0 && prompt.column < n_embd_out);

    for (const int32_t marker : prompt.markers) {
        const float * embd = get_embd(marker);
        if (embd_q == nullptr) {
            scores.push_back(embd[prompt.column]);
            continue;
        }
        float dot = 0.0f;
        for (int32_t i = 0; i < n_pointer; i++) {
            dot += embd_q[i] * embd[n_pointer + i];
        }
        scores.push_back(dot / sqrtf((float) n_pointer));
    }
    return scores;
}

} // namespace rnllama
