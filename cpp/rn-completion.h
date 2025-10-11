#ifndef RN_COMPLETION_H
#define RN_COMPLETION_H

#include "common.h"
#include "llama.h"
#include "sampling.h"
#include "nlohmann/json.hpp"
#include "chat.h"

using json = nlohmann::ordered_json;

namespace rnllama {

// Utility functions
static inline void llama_batch_clear(llama_batch *batch) {
    batch->n_tokens = 0;
}

// Forward declarations
struct llama_rn_context;

// Types defined in rn-llama.h (needed here for compilation)
enum stop_type
{
    STOP_FULL,
    STOP_PARTIAL,
};

struct completion_token_output
{
    struct token_prob
    {
        llama_token tok;
        float prob;
    };

    std::vector<token_prob> probs;
    llama_token tok;
    std::string text;  // Token text (decoded)
    int32_t request_id = -1;  // Request ID for parallel processing
};

struct completion_chat_output
{
  std::string content;
  std::string reasoning_content;
  std::vector<common_chat_tool_call> tool_calls;
  std::string accumulated_text;
};

// Completion context class
struct llama_rn_context_completion {
    // Reference to parent context
    llama_rn_context* parent_ctx;

    // Completion state fields
    bool is_predicting = false;
    bool is_interrupted = false;
    bool has_next_token = false;
    std::string prefill_text;
    std::string generated_text;
    std::vector<completion_token_output> generated_token_probs;
    size_t num_prompt_tokens = 0;
    size_t num_tokens_predicted = 0;
    llama_pos n_past = 0;
    size_t n_remain = 0;
    std::vector<llama_token> embd;
    bool incomplete = false;
    bool context_full = false;
    bool truncated = false;
    bool stopped_eos = false;
    bool stopped_word = false;
    bool stopped_limit = false;
    std::string stopping_word;
    // Current completion parameters for chat parsing
    int current_chat_format = COMMON_CHAT_FORMAT_CONTENT_ONLY;
    common_reasoning_format current_reasoning_format = COMMON_REASONING_FORMAT_NONE;
    bool current_thinking_forced_open = false;

    // Sampling context
    common_sampler *ctx_sampling = nullptr;

    // Constructor
    llama_rn_context_completion(llama_rn_context* parent);

    // Destructor
    ~llama_rn_context_completion();

    // Completion processing methods
    void rewind();
    bool initSampling();
    void truncatePrompt(std::vector<llama_token> &prompt_tokens);
    void loadPrompt(const std::vector<std::string> &media_paths);
    void beginCompletion();
    void beginCompletion(int chat_format, common_reasoning_format reasoning_format, bool thinking_forced_open);
    void endCompletion();
    completion_token_output nextToken();
    size_t findStoppingStrings(const std::string &text, const size_t last_token_size, const stop_type type);
    completion_token_output doCompletion();
    completion_chat_output parseChatOutput(bool is_partial);

    // Embedding methods
    std::vector<float> embedding(common_params &embd_params);
    std::vector<float> rerank(const std::string &query, const std::vector<std::string> &documents);

    // Benchmarking methods
    std::string bench(int pp, int tg, int pl, int nr);

    // Multimodal processing methods
    void processMedia(
      const std::string &prompt,
      const std::vector<std::string> &media_paths
    );
};

} // namespace rnllama

#endif /* RN_COMPLETION_H */
