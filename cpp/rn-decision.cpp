#include "rn-decision.h"

#include "llama-ext.h" // staging API: llama_decision_order
#include "rn-mtmd.hpp" // base64_decode

#include <algorithm>
#include <cmath>
#include <regex>
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
    if (model_type == COMMON_DECISION_TYPE_NONE) {
        // a declared type wins, a system_one template is the fallback
        const char * legacy_src = llama_model_chat_template(model, "system_one");
        if (legacy_src != nullptr) {
            init_legacy(model, legacy_src);
        }
        return;
    }
    if (!llama_rn_decision_profile_for(model_type, profile)) {
        type = COMMON_DECISION_TYPE_UNKNOWN;
        return;
    }

    const std::string prefix = decision_meta_str(model, "general.architecture") + ".decision.";

    vocab = llama_model_get_vocab(model);

    const char * tmpl_src = llama_model_chat_template(model, "systemone");
    if (tmpl_src == nullptr) {
        throw std::runtime_error("decision model has no \"systemone\" template");
    }
    tmpl = std::make_shared<const common_chat_template>(tmpl_src, "", "");

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
        case LLAMA_RN_DECISION_LABELS_NONE:
            break;
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
            token_sep    = llama_vocab_sep(vocab);
            if (token_marker == LLAMA_TOKEN_NULL || token_sep == LLAMA_TOKEN_NULL) {
                throw std::runtime_error("decision model has no mask or sep token");
            }
            text_marker = common_token_to_piece(vocab, token_marker, true);
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

    n_options_max = profile.readout == LLAMA_RN_DECISION_READOUT_LABEL_LOGITS ? labels.size() : 255;
    type = model_type;
}

json llama_rn_decision_context::info() const {
    if (!is_decision_model()) {
        return nullptr;
    }
    json out = json{
        {"type",           legacy ? "system_one" : llama_rn_decision_type_name(type)},
        {"nOptionsMax",    n_options_max},
        {"imageInput",     profile.image_input},
        // the models that read the embeddings output run in embedding mode, they have no logits
        {"textGeneration", is_supported() && (profile.readout == LLAMA_RN_DECISION_READOUT_LABEL_LOGITS ||
                                              profile.readout == LLAMA_RN_DECISION_READOUT_LETTER_SLOTS)},
    };
    if (legacy) {
        out["readout"] = profile.readout == LLAMA_RN_DECISION_READOUT_RANK_HEAD ? "rank_head" : "letter_slot";
    }
    if (!error.empty()) {
        out["error"] = error;
    }
    return out;
}

//
// system_one (legacy)
//

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
        profile.readout = LLAMA_RN_DECISION_READOUT_LETTER_SLOTS;
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

std::vector<llama_rn_decision_prompt> llama_rn_decision_context::fill_prompts_legacy(const llama_rn_decision_request & request) const {
    const std::string state = decision_legacy_text(request.state);
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
    for (size_t i = 0; i < segments.size(); i++) {
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
    llama_rn_decision_request request;
    request.questions = parse_questions(body);
    request.state     = parse_state(body, request.images);
    return request;
}

std::vector<llama_rn_decision_question> llama_rn_decision_context::parse_questions(const json & body) const {
    if (!body.contains("state") || body.at("state").is_null()) {
        throw std::invalid_argument("\"state\" must be provided");
    }
    if (!body.contains("questions") || !body.at("questions").is_object() || body.at("questions").empty()) {
        throw std::invalid_argument("\"questions\" must be a non-empty object");
    }

    std::vector<llama_rn_decision_question> questions;
    for (const auto & [id, q] : body.at("questions").items()) {
        auto err = [&id = id](const std::string & msg) {
            return std::invalid_argument("questions." + id + ": " + msg);
        };
        if (!q.is_object()) {
            throw err("must be an object");
        }
        if (!q.contains("instructions") || q.at("instructions").is_null()) {
            throw err("\"instructions\" must be provided");
        }

        llama_rn_decision_question question;
        question.id           = id;
        question.instructions = q.at("instructions");

        const std::string type_name = q.contains("type") && q.at("type").is_string() ? q.at("type").get<std::string>() : "";
        const json        criteria  = q.contains("criteria") ? q.at("criteria") : json();

        if (type_name == "choice") {
            question.type = LLAMA_RN_DECISION_QUESTION_CHOICE;
            if (!criteria.is_object() || criteria.empty()) {
                throw err("\"criteria\" must be a non-empty object");
            }
            for (const auto & [key, description] : criteria.items()) {
                question.options.push_back({key, description});
            }
            if (profile.choice_sorted) {
                std::sort(question.options.begin(), question.options.end(), [](const auto & a, const auto & b) {
                    return a.key < b.key;
                });
            }
        } else if (type_name == "score") {
            question.type = LLAMA_RN_DECISION_QUESTION_SCORE;
            if (!criteria.is_array() || criteria.size() < 2 || criteria.size() > 10) {
                throw err("\"criteria\" must be an array of 2 to 10 levels");
            }
            for (size_t i = 0; i < criteria.size(); i++) {
                question.options.push_back({std::to_string(i), criteria.at(i)});
            }
        } else if (type_name == "noul") {
            question.type = LLAMA_RN_DECISION_QUESTION_NOUL;
            if (!criteria.is_null() && !criteria.is_object()) {
                throw err("\"criteria\" must be an object");
            }
            for (const char * key : {"false", "true"}) {
                question.options.push_back({key, criteria.is_object() && criteria.contains(key) ? criteria.at(key) : json()});
            }
            if (profile.noul_true_first) {
                std::swap(question.options[0], question.options[1]);
            }
        } else {
            throw err("\"type\" must be one of: choice, score, noul");
        }

        if (question.options.size() > n_options_max) {
            throw err(string_format("too many options (%zu), this model supports at most %zu", question.options.size(), n_options_max));
        }

        questions.push_back(std::move(question));
    }
    return questions;
}

static const size_t DECISION_MAX_IMAGES = 8;

static void decision_add_image(const json & url, std::vector<std::string> & images) {
    if (!url.is_string() || url.get<std::string>().empty()) {
        throw std::invalid_argument("an image must be a file path or a data URL");
    }
    if (images.size() >= DECISION_MAX_IMAGES) {
        throw std::invalid_argument(string_format("too many images, the maximum is %zu", DECISION_MAX_IMAGES));
    }
    images.push_back(url.get<std::string>());
}

json llama_rn_decision_context::parse_state(const json & body, std::vector<std::string> & images) const {
    if (body.contains("images") && !body.at("images").is_null()) {
        if (!body.at("images").is_array()) {
            throw std::invalid_argument("\"images\" must be an array");
        }
        for (const auto & url : body.at("images")) {
            decision_add_image(url, images);
        }
    }

    const json & state = body.at("state");
    const bool is_wrapped = state.is_object() && state.contains("messages");
    const json & messages = is_wrapped ? state.at("messages") : state;
    if (!messages.is_array()) {
        return state;
    }

    // chat messages: take the image parts out of the content
    json messages_out = json::array();
    for (const auto & msg : messages) {
        if (!msg.is_object() || !msg.contains("content") || !msg.at("content").is_array()) {
            messages_out.push_back(msg);
            continue;
        }
        json content = json::array();
        for (const auto & part : msg.at("content")) {
            const bool is_image = part.is_object() && part.contains("type") && part.at("type") == "image_url" && part.contains("image_url");
            if (is_image) {
                const json & image_url = part.at("image_url");
                decision_add_image(image_url.is_object() && image_url.contains("url") ? image_url.at("url") : image_url, images);
            } else {
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

// replace text in all strings of a JSON value
static json decision_replace_text(const json & val, const std::string & search, const std::string & replace) {
    if (val.is_string()) {
        std::string str = val.get<std::string>();
        string_replace_all(str, search, replace);
        return str;
    }
    if (val.is_array()) {
        json out = json::array();
        for (const auto & item : val) {
            out.push_back(decision_replace_text(item, search, replace));
        }
        return out;
    }
    if (val.is_object()) {
        json out = json::object();
        for (const auto & [key, item] : val.items()) {
            out[key] = decision_replace_text(item, search, replace);
        }
        return out;
    }
    return val;
}

// sort the keys of all objects of a JSON value
static json decision_sort_keys(const json & val) {
    if (val.is_array()) {
        json out = json::array();
        for (const auto & item : val) {
            out.push_back(decision_sort_keys(item));
        }
        return out;
    }
    if (val.is_object()) {
        std::map<std::string, json> sorted;
        for (const auto & [key, item] : val.items()) {
            sorted[key] = decision_sort_keys(item);
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
static std::string decision_text_render(const json & val, int indent = 0) {
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
            const std::string text = decision_text_render(item, indent + 1);
            out += (out.empty() ? "" : "\n") + pad + "- " + text.substr(std::min(text.size(), text.find_first_not_of(" \t\n\r")));
        }
        return out;
    }
    if (val.is_object()) {
        std::string out;
        for (const auto & [key, item] : val.items()) {
            const bool is_nested = item.is_object() || item.is_array();
            out += (out.empty() ? "" : "\n") + pad + key + (is_nested ? ":\n" : ": ") + decision_text_render(item, is_nested ? indent + 1 : 0);
        }
        return out;
    }
    return val.dump();
}

// text input: special tokens written in the text must not be parsed as such
static std::string decision_text(const json & val) {
    static const std::regex re_special("<\\|([A-Za-z0-9_]+)\\|>");
    return std::regex_replace(decision_text_render(val), re_special, "<\xC2\xA6$1\xC2\xA6>");
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

json llama_rn_decision_context::render_options(const llama_rn_decision_question & question, size_t variant) const {
    const size_t n_options = question.options.size();

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
        size_t n_images) const {
    // the template is given raw JSON values, it serializes the ones that are not strings
    json inp = json{
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

    // the input must not contain the marker of the options
    if (!text_marker.empty()) {
        inp = decision_replace_text(inp, text_marker, " ");
    }

    // the template puts one media marker per image
    json images = json::array();
    if (n_images > 0) {
        inp = decision_replace_text(inp, mtmd_default_marker(), " ");
        for (size_t i = 0; i < n_images; i++) {
            images.push_back(mtmd_default_marker());
        }
    }
    inp["images"] = images;

    // a template may ask for the prompt to be tokenized piece by piece by writing {{ sep }} at every boundary,
    // the input must not forge one
    inp        = decision_replace_text(inp, DECISION_SEP, " ");
    inp["sep"] = DECISION_SEP;

    return decision_apply_template(*tmpl, inp);
}

std::vector<llama_rn_decision_prompt> llama_rn_decision_context::fill_prompts(const llama_rn_decision_request & request, mtmd_context * mctx) const {
    if (!is_supported()) {
        throw std::runtime_error(error.empty() ? "this model is not a decision model of a supported type" : error);
    }
    if (legacy) {
        if (!request.images.empty()) {
            throw std::runtime_error("image input is not supported for a system_one model");
        }
        return fill_prompts_legacy(request);
    }
    if (!request.images.empty() && !profile.image_input) {
        throw std::runtime_error("this decision model does not take images");
    }
    if (!request.images.empty() && (mctx == nullptr || !mtmd_support_vision(mctx))) {
        throw std::runtime_error("image input needs the multimodal projector of the model, see initMultimodal()");
    }

    std::vector<llama_rn_decision_prompt> prompts;
    if (is_joint()) {
        prompts.push_back(fill_prompt_joint(request));
        return prompts;
    }
    for (size_t i = 0; i < request.questions.size(); i++) {
        for (size_t variant = 0; variant < n_variants(request.questions[i]); variant++) {
            prompts.push_back(fill_prompt(request, i, variant, mctx));
        }
    }
    return prompts;
}

// images: file paths or data URLs, the same as the media_paths of a completion
static std::shared_ptr<mtmd_input_chunks> decision_tokenize_media(
        mtmd_context * mctx,
        const std::string & prompt,
        const std::vector<std::string> & images) {
    mtmd::bitmaps bitmaps;
    for (const auto & image : images) {
        mtmd_helper_bitmap_wrapper out{};
        if (string_starts_with(image, "data:")) {
            const size_t comma = image.find(',');
            if (comma == std::string::npos || image.substr(0, comma).find(";base64") == std::string::npos) {
                throw std::invalid_argument("an image data URL must be base64 encoded");
            }
            const raw_buffer data = base64_decode(image.substr(comma + 1));
            out = mtmd_helper_bitmap_init_from_buf(mctx, data.data(), data.size(), false, mtmd_helper_init_opt_default());
        } else {
            out = mtmd_helper_bitmap_init_from_file(mctx, image.c_str(), false, mtmd_helper_init_opt_default());
        }
        if (out.video_ctx != nullptr) {
            mtmd_helper_video_free(out.video_ctx);
            if (out.bitmap != nullptr) {
                mtmd_bitmap_free(out.bitmap);
            }
            throw std::invalid_argument("a decision image must be an image, not a video");
        }
        if (out.bitmap == nullptr) {
            throw std::invalid_argument("failed to load image: " + image.substr(0, 64));
        }
        bitmaps.entries.emplace_back(out.bitmap);
    }

    // the same flags as llama-server
    mtmd_input_text text = {
        prompt.data(),
        prompt.size(),
        /* add_special   */ true,
        /* parse_special */ true,
    };
    std::shared_ptr<mtmd_input_chunks> chunks(mtmd_input_chunks_init(), mtmd_input_chunks_free);
    auto bitmaps_c_ptr = bitmaps.c_ptr();
    if (mtmd_tokenize(mctx, chunks.get(), &text, bitmaps_c_ptr.data(), bitmaps_c_ptr.size()) != 0) {
        throw std::runtime_error("failed to tokenize the decision prompt with its images");
    }
    return chunks;
}

llama_rn_decision_prompt llama_rn_decision_context::fill_prompt(
        const llama_rn_decision_request & request,
        size_t i_question,
        size_t variant,
        mtmd_context * mctx) const {
    const auto & question = request.questions[i_question];
    const std::string prompt_text = render(request.state, request.questions, question, variant, request.images.size());

    llama_rn_decision_prompt prompt;
    prompt.question = i_question;
    prompt.variant  = variant;

    if (!request.images.empty()) {
        GGML_ASSERT(profile.readout == LLAMA_RN_DECISION_READOUT_LABEL_LOGITS); // see profile.image_input
        if (prompt_text.find(DECISION_SEP) != std::string::npos) {
            // the pieces would have to be tokenized around the media chunks
            throw std::runtime_error("this decision model tokenizes its prompt piece by piece, "
                                     "which is not supported together with an image");
        }
        prompt.chunks = decision_tokenize_media(mctx, prompt_text, request.images);
        prompt.labels.assign(labels.begin(), labels.begin() + n_outputs(question));
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
            // a rating scale reads its ratings at the first labels
            prompt.labels.assign(labels.begin(), labels.begin() + n_outputs(question));
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

// given to the template: text between the pieces of the prompt, and at the start of the span of a question or of an option
llama_rn_decision_prompt llama_rn_decision_context::fill_prompt_joint(const llama_rn_decision_request & request) const {
    const auto & questions = request.questions;

    json inp_questions = json::array();
    for (const auto & question : questions) {
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
    inp["sep"]           = DECISION_SEP;
    inp["mark_question"] = DECISION_QUESTION;
    inp["mark_option"]   = DECISION_OPTION;

    const std::string prompt_text = decision_apply_template(*tmpl, inp);

    // the model was trained with the pieces tokenized one by one
    llama_rn_decision_prompt prompt;
    size_t i_question = 0;
    for (std::string piece : string_split(prompt_text, DECISION_SEP)) {
        int32_t order = LLAMA_DECISION_ORDER_NONE;
        if (string_starts_with(piece, DECISION_QUESTION)) {
            piece = piece.substr(DECISION_QUESTION.size());
            if (i_question >= questions.size()) {
                throw std::runtime_error("unexpected layout of the decision prompt");
            }
            switch (questions[i_question++].type) {
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
    for (const auto & question : questions) {
        n_options += question.options.size();
    }
    if (i_question != questions.size() || (size_t) prompt.n_scores != n_options) {
        throw std::runtime_error("unexpected layout of the decision prompt");
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

json llama_rn_decision_context::format_answer(const llama_rn_decision_question & question, const std::vector<std::vector<float>> & scores) const {
    const size_t n = n_outputs(question);
    if (scores.size() != n_variants(question)) {
        throw std::runtime_error("decision result does not match the number of variants");
    }

    // softmax over the outputs of each variant, then the average of the variants
    const float temperature = get_temperature(question);
    std::vector<double> probs(n, 0.0);
    for (size_t v = 0; v < scores.size(); v++) {
        const auto & s = scores[v];
        if (s.size() != n) {
            throw std::runtime_error("decision result does not match the number of options");
        }
        // a joint head returns NaN if it could not use the decision order
        if (std::any_of(s.begin(), s.end(), [](float x) { return std::isnan(x); })) {
            throw std::runtime_error("the model could not evaluate the decision");
        }
        const float score_max = *std::max_element(s.begin(), s.end());
        std::vector<double> p(n);
        double sum = 0.0;
        for (size_t i = 0; i < n; i++) {
            p[i] = std::exp((double) (s[i] - score_max) / temperature);
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
        answer["confidence"]    = decision_confidence_choice(probs);
    } else {
        double expected = 0.0;
        json legend = json::object();
        for (size_t i = 0; i < n; i++) {
            expected += i * probs[i];
            legend[question.options[i].key] = question.options[i].description;
        }
        answer["score"]         = expected;
        answer["legend"]        = legend;
        answer["probabilities"] = probabilities;
        answer["confidence"]    = decision_confidence_score(probs);
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

    json answers = json::object();
    if (is_joint()) {
        // the scores of all the options of all the questions, in order
        size_t i_score = 0;
        for (const auto & question : request.questions) {
            const size_t n = question.options.size();
            if (i_score + n > scores[0].size()) {
                throw std::runtime_error("decision result does not match the number of options");
            }
            answers[question.id] = format_answer(question, {std::vector<float>(scores[0].begin() + i_score, scores[0].begin() + i_score + n)});
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
            answers[request.questions[i].id] = format_answer(request.questions[i], per_question[i]);
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

// Evaluate a prompt with images from the start of the sequence, the way llama-server does
// (process_mtmd_chunk): a media chunk is encoded in one mtmd batch with the media chunks
// that follow it, then its embeddings are decoded; text is decoded in batches of n_batch.
// Only the last token of the prompt has an output.
static void decision_eval_chunks(mtmd_context * mctx, llama_context * ctx, llama_seq_id seq_id, const mtmd_input_chunks * chunks) {
    const size_t  n_chunks = mtmd_input_chunks_size(chunks);
    const int32_t n_batch  = llama_n_batch(ctx);

    llama_pos n_past = 0;
    mtmd::batch_ptr mbatch;
    for (size_t i = 0; i < n_chunks; i++) {
        const mtmd_input_chunk * chunk = mtmd_input_chunks_get(chunks, i);
        const bool is_last = i + 1 == n_chunks;

        if (mtmd_input_chunk_get_type(chunk) == MTMD_INPUT_CHUNK_TYPE_TEXT) {
            size_t n_tokens = 0;
            const llama_token * tokens = mtmd_input_chunk_get_tokens_text(chunk, &n_tokens);
            common_batch batch(ctx);
            for (size_t j = 0; j < n_tokens; j += n_batch) {
                const size_t end = std::min(n_tokens, j + n_batch);
                batch.clear();
                for (size_t k = j; k < end; k++) {
                    batch.add(tokens[k], n_past + k, seq_id, is_last && k + 1 == n_tokens);
                }
                const int32_t ret = llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get());
                if (ret != 0) {
                    throw std::runtime_error(string_format("failed to evaluate the decision prompt (error %d)", ret));
                }
            }
            n_past += n_tokens;
            continue;
        }

        if (is_last) {
            throw std::runtime_error("a decision prompt must end with text");
        }

        float * embd = mbatch ? mtmd_batch_get_output_embd(mbatch.get(), chunk) : nullptr;
        if (embd == nullptr) {
            mbatch.reset(mtmd_batch_init(mctx));
            GGML_ASSERT(mtmd_batch_add_chunk(mbatch.get(), chunk) == 0);
            for (size_t j = i + 1; j < n_chunks; j++) {
                const mtmd_input_chunk * next = mtmd_input_chunks_get(chunks, j);
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

std::vector<float> llama_rn_decision_eval(
        llama_context * ctx,
        llama_seq_id seq_id,
        const llama_rn_decision_prompt & prompt,
        std::vector<llama_token> & cached,
        mtmd_context * mctx) {
    if (prompt.chunks) {
        // the media chunks are evaluated from the start, every time
        llama_memory_t mem = llama_get_memory(ctx);
        cached.clear();
        if (mem != nullptr) {
            llama_memory_seq_rm(mem, seq_id, -1, -1);
        }
        GGML_ASSERT(mctx != nullptr);
        try {
            decision_eval_chunks(mctx, ctx, seq_id, prompt.chunks.get());
        } catch (...) {
            if (mem != nullptr) {
                llama_memory_seq_rm(mem, seq_id, -1, -1);
            }
            throw;
        }
        const float * logits = llama_get_logits_ith(ctx, -1);
        if (logits == nullptr) {
            throw std::runtime_error("failed to get logits");
        }
        std::vector<float> scores;
        for (const llama_token label : prompt.labels) {
            scores.push_back(logits[label]);
        }
        return scores;
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
        while (n_keep < cached.size() && n_keep < first_read && cached[n_keep] == tokens[n_keep]) {
            n_keep++;
        }
        // a recurrent state cannot be rolled back, start over
        if (!llama_memory_seq_rm(mem, seq_id, n_keep, -1)) {
            llama_memory_seq_rm(mem, seq_id, -1, -1);
            n_keep = 0;
        }
    }
    cached.assign(tokens.begin(), tokens.begin() + n_keep);

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
            cached.clear();
            if (mem != nullptr) {
                llama_memory_seq_rm(mem, seq_id, -1, -1);
            }
            throw std::runtime_error(string_format("failed to evaluate the decision prompt (error %d)", ret));
        }
        cached.insert(cached.end(), tokens.begin() + i, tokens.begin() + end);
        i_last = i;
        i = end;
    }
    if (mem == nullptr) {
        cached.clear();
    }

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
            for (size_t i = 0; i < n; i++) {
                scores.push_back(logits[prompt.labels[i]]);
            }
        }
        return scores;
    }

    if (prompt.pooled) {
        const float * embd = llama_get_embeddings_seq(ctx, seq_id);
        if (embd == nullptr) {
            throw std::runtime_error("failed to get the pooled output, is the context in RANK pooling?");
        }
        scores.push_back(embd[0]);
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
