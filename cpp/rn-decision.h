#ifndef RN_DECISION_H
#define RN_DECISION_H

// Typed decision models (TypeSafe /v1/systemone API): a model answers typed
// questions about a state in one forward pass, no token is generated.
//
// Ported from llama.cpp tools/server/server-decision.{h,cpp} at b11385 (bf9a0cc),
// plus eb2b96dd5 "server : tokenize a decision prompt piece by piece when its
// template asks", which is not upstream yet: render() passing {{ sep }} to every
// template and fill_prompt() splitting on it come from that commit.
//
// The blobs ported from, checked by scripts/sync-vendor.sh: when upstream changes
// them, carry the change over and update these lines.
// upstream-blob: tools/server/server-decision.cpp 522aa2bed4ff96c7e16d7c508b1f605363d1af5a
// upstream-blob: tools/server/server-decision.h 93480ba26974519cf7920105af640421eb58827d
//
// The functions keep upstream's names and split so that an upstream change can
// be carried over function by function. What differs:
//   - the per-model branches of upstream are a profile table, see
//     llama_rn_decision_profile_for(): a new model that combines conventions
//     that already exist is one more row, not new code
//   - a template may ask for its prompt to be tokenized piece by piece ({{ sep }})
//     for every type, not only for the joint one
//   - the decode loop is llama.rn's own (llama_rn_decision_eval), the server's
//     task queue and its shared-prefix grouping are not ported
//   - a model with no <arch>.decision.type but a `system_one` template is read the
//     way mybigday's system-one library does (tools/system-one at feat/system-one):
//     the readout is derived from what the model is rather than declared, see
//     init_legacy(). Only the readouts upstream's graphs can serve are taken:
//     letter_slot (causal) and rank_head (a classification head)

#include "common.h"
#include "chat.h"
#include "json.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include <algorithm>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

// JSON is upstream's common_json here, so that its code carries over as it is;
// the callers convert at the boundary
namespace rnllama {

enum llama_rn_decision_question_type {
    LLAMA_RN_DECISION_QUESTION_CHOICE,
    LLAMA_RN_DECISION_QUESTION_SCORE,
    LLAMA_RN_DECISION_QUESTION_NOUL,
};

// where the score of an option is read
enum llama_rn_decision_readout {
    LLAMA_RN_DECISION_READOUT_LABEL_LOGITS, // logits of one label token per option, at the last prompt token
    LLAMA_RN_DECISION_READOUT_MARKER_EMBD,  // embeddings[column] at one marker token per option, column = question type
    LLAMA_RN_DECISION_READOUT_POINTER,      // scaled dot of the last token's query and the key at each option's marker
    LLAMA_RN_DECISION_READOUT_JOINT,        // all questions in one prompt, the head writes option i's score in row i
    // system_one (legacy) readouts
    LLAMA_RN_DECISION_READOUT_LETTER_SLOTS, // all questions in one prompt, label logits at the end of each question's segment
    LLAMA_RN_DECISION_READOUT_RANK_HEAD,    // one prompt per option, scored by the classification head (RANK pooling)
};

// the label tokens of LABEL_LOGITS
enum llama_rn_decision_labels {
    LLAMA_RN_DECISION_LABELS_NONE,
    LLAMA_RN_DECISION_LABELS_LETTERS, // A..Z then a..z, every one must be a single token
    LLAMA_RN_DECISION_LABELS_CODES,   // A..Z then AA..ZZ, the ones that are a single token, at most 255
};

// the marker token of MARKER_EMBD and POINTER
enum llama_rn_decision_marker {
    LLAMA_RN_DECISION_MARKER_NONE,
    LLAMA_RN_DECISION_MARKER_VOCAB_MASK, // the vocab's mask token, the vocab's sep token delimits the options
    LLAMA_RN_DECISION_MARKER_TEXT,       // marker_text, a single special token
};

enum llama_rn_decision_temperature_buckets {
    LLAMA_RN_DECISION_TEMPERATURE_BUCKETS_DEFAULT, // 2, 3_5, 6_10, 11
    LLAMA_RN_DECISION_TEMPERATURE_BUCKETS_SIZE,    // small, mid, large
};

// The conventions a decision type was trained with. Upstream spells each one
// as an `if (type == ...)` branch.
struct llama_rn_decision_profile {
    llama_rn_decision_readout readout = LLAMA_RN_DECISION_READOUT_LABEL_LOGITS;
    llama_rn_decision_labels  labels  = LLAMA_RN_DECISION_LABELS_NONE;
    llama_rn_decision_marker  marker  = LLAMA_RN_DECISION_MARKER_NONE;
    std::string marker_text;

    bool label_texts      = false; // the template is given the label of each option
    bool list_questions   = false; // the template is given all the questions of the request
    bool sort_keys        = false; // the template input has its object keys sorted
    bool text_input       = false; // state, instructions and options are flattened to text (kev)
    bool noul_true_first  = false; // noul options are [true, false] instead of [false, true]
    bool choice_sorted    = false; // choice options are in the order of their keys
    bool reverse_variant  = false; // a choice is also shown in reverse order, the two are averaged
    bool truncate_head    = false; // question and options are cut to <arch>.decision.max_head_tokens
    bool image_input      = false; // the prompt has a place for images
    size_t n_noul_ratings = 0;     // if set, noul is the expected value of a rating scale of that many labels

    llama_rn_decision_temperature_buckets temperature_buckets = LLAMA_RN_DECISION_TEMPERATURE_BUCKETS_DEFAULT;
};

// the profile of a type, false if the type is not supported
bool llama_rn_decision_profile_for(common_decision_type type, llama_rn_decision_profile & profile);

const char * llama_rn_decision_type_name(common_decision_type type);

struct llama_rn_decision_option {
    std::string key;
    common_json description; // null if not provided
};

struct llama_rn_decision_question {
    std::string id;
    llama_rn_decision_question_type type;
    common_json instructions;
    std::vector<llama_rn_decision_option> options; // in the order of the model outputs
};

// A piece of a system_one state: text, or text and images tokenized by mtmd
struct llama_rn_decision_piece {
    std::vector<llama_token>           tokens;
    std::shared_ptr<mtmd_input_chunks> chunks;
};

// One prompt to evaluate, and where its result is read
struct llama_rn_decision_prompt {
    size_t question = 0; // index in the request, unused by a joint prompt
    size_t variant  = 0;

    std::vector<llama_token> tokens;
    // a prompt with images is evaluated from its chunks, tokens is then empty
    std::shared_ptr<mtmd_input_chunks> chunks;
    // system_one with images: the state, segment by segment, evaluated before tokens (the
    // question blocks, whose slots are relative to them)
    std::vector<llama_rn_decision_piece> state_pieces;

    size_t n_tokens() const {
        size_t n = chunks ? mtmd_helper_get_n_tokens(chunks.get()) : tokens.size();
        for (const auto & piece : state_pieces) {
            n += piece.chunks ? mtmd_helper_get_n_tokens(piece.chunks.get()) : piece.tokens.size();
        }
        return n;
    }

    std::vector<llama_token> labels;  // LABEL_LOGITS: logits of these tokens, at the last token
    std::vector<int32_t>     slots;   // LETTER_SLOTS: where each question's labels are read
    std::vector<int32_t>     slot_n_labels; // LETTER_SLOTS: how many labels each slot reads
    bool                     pooled = false; // RANK_HEAD: the score is the sequence's pooled output
    std::vector<int32_t>     markers; // MARKER_EMBD, POINTER: prompt positions of the options
    int32_t                  column  = 0;
    int32_t                  pointer = -1; // POINTER: prompt position of the query

    std::vector<int32_t> order;    // JOINT: one llama_decision_order per token
    int32_t              n_scores = 0;

    bool need_embd() const {
        return !markers.empty() || !order.empty() || pooled;
    }

    // first prompt position that is read, -1 if only the last token is
    int32_t pos_first() const {
        if (!order.empty() || pooled) {
            return 0;
        }
        if (!slots.empty()) {
            return *std::min_element(slots.begin(), slots.end());
        }
        int32_t pos = pointer;
        for (const int32_t marker : markers) {
            pos = pos < 0 ? marker : std::min(pos, marker);
        }
        return pos;
    }
};

struct llama_rn_decision_request {
    common_json state;
    std::vector<llama_rn_decision_question> questions;
    std::vector<std::string> images; // file paths or data URLs
};

struct llama_rn_decision_context {
    common_decision_type      type = COMMON_DECISION_TYPE_NONE;
    llama_rn_decision_profile profile;
    size_t                    n_options_max = 0;
    bool                      legacy = false; // a system_one model, see init_legacy()
    std::string               error;          // why a decision model cannot be used, if type is UNKNOWN

    // read the "<arch>.decision.*" metadata, type stays NONE if the model has none
    // throws if the model is a decision model that cannot be used
    void init(const llama_model * model);

    bool is_supported() const {
        return legacy || (type != COMMON_DECISION_TYPE_NONE && type != COMMON_DECISION_TYPE_UNKNOWN);
    }

    bool is_decision_model() const {
        return legacy || type != COMMON_DECISION_TYPE_NONE;
    }

    // all the questions are answered from one prompt
    bool is_joint() const {
        return profile.readout == LLAMA_RN_DECISION_READOUT_JOINT ||
               profile.readout == LLAMA_RN_DECISION_READOUT_LETTER_SLOTS;
    }

    // throws if the context cannot serve the readout, e.g. rank_head without RANK pooling
    void check_context(llama_context * ctx) const;

    // model.decision of the JS API, null if the model is not a decision model
    common_json info() const;

    // throws std::invalid_argument on bad input
    llama_rn_decision_request parse_request(const common_json & body) const;

    // number of prompts that are evaluated to answer this question, each one shows the options in a different order
    size_t n_variants(const llama_rn_decision_question & question) const;

    // all the prompts of a request, the ones that start alike next to each other
    // mctx: the multimodal context, only used if the request has images (nullptr if not initialized)
    std::vector<llama_rn_decision_prompt> fill_prompts(const llama_rn_decision_request & request, mtmd_context * mctx) const;

    // scores: the raw model outputs of each prompt, in the order of fill_prompts()
    common_json format_result(
            const llama_rn_decision_request & request,
            const std::vector<llama_rn_decision_prompt> & prompts,
            const std::vector<std::vector<float>> & scores,
            const std::string & model_name) const;

private:
    const llama_vocab * vocab = nullptr;
    std::shared_ptr<const common_chat_template> tmpl; // the "systemone" template

    std::map<std::string, float> temperatures; // "<type>" or "<type>.<n_options bucket>"

    std::vector<llama_token> labels;
    std::vector<std::string> label_texts;

    llama_token token_marker      = LLAMA_TOKEN_NULL;
    llama_token token_sep         = LLAMA_TOKEN_NULL;
    std::string text_marker;
    size_t      max_head_tokens   = 0;
    size_t      max_option_tokens = 48;

    // system_one
    std::string legacy_separator = "\x1e"; // system_one.segment_separator
    std::string bos_text;
    std::string eos_text;
    bool        labels_are_default = false;

    void init_legacy(const llama_model * model, const char * tmpl_src);
    std::vector<std::string> render_legacy(const common_json & inp) const;
    std::vector<llama_rn_decision_prompt> fill_prompts_legacy(const llama_rn_decision_request & request, mtmd_context * mctx) const;

    std::vector<llama_rn_decision_question> parse_questions(const common_json & body) const;
    common_json parse_state(const common_json & body, std::vector<std::string> & images) const;

    std::string render(
            const common_json & state,
            const std::vector<llama_rn_decision_question> & questions,
            const llama_rn_decision_question & question,
            size_t variant,
            size_t n_images) const;
    common_json render_options(const llama_rn_decision_question & question, size_t variant) const;
    size_t n_outputs(const llama_rn_decision_question & question) const;

    llama_rn_decision_prompt fill_prompt(
            const llama_rn_decision_request & request,
            size_t i_question,
            size_t variant,
            mtmd_context * mctx) const;
    llama_rn_decision_prompt fill_prompt_joint(const llama_rn_decision_request & request) const;
    void fill_prompt_laya(std::vector<llama_token> & tokens, const llama_rn_decision_question & question, llama_rn_decision_prompt & prompt) const;

    float get_temperature(const llama_rn_decision_question & question) const;
    common_json format_answer(const llama_rn_decision_question & question, const std::vector<std::vector<float>> & scores) const;
};

// A decision request queued on the slot manager
struct llama_rn_decision_job {
    llama_rn_decision_request request;
    std::vector<llama_rn_decision_prompt> prompts;
    // the response, or {"error": message}
    std::function<void(int32_t request_id, const common_json & result)> on_result;
};

// What a sequence holds from the previous prompt, so that the next one evaluates only what
// differs from it. Empty for a cleared sequence.
struct llama_rn_decision_cache {
    std::vector<llama_token>           tokens; // a text prompt
    std::shared_ptr<mtmd_input_chunks> chunks; // a prompt with images

    void clear() {
        tokens.clear();
        chunks.reset();
    }
};

// Evaluate one prompt on sequence seq_id and return its raw scores, throws on failure.
// mctx: the multimodal context, needed by a prompt with images.
// cache: what the sequence holds, the common prefix is not evaluated again; updated on return.
std::vector<float> llama_rn_decision_eval(
        llama_context * ctx,
        llama_seq_id seq_id,
        const llama_rn_decision_prompt & prompt,
        llama_rn_decision_cache & cache,
        mtmd_context * mctx = nullptr);

} // namespace rnllama

#endif /* RN_DECISION_H */
