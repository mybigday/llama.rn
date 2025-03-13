#ifndef RNLLAMA_H
#define RNLLAMA_H

#include <sstream>
#include <iostream>
#include "chat.h"
#include "common.h"
#include "ggml.h"
#include "gguf.h"
#include "llama.h"
#include "llama-impl.h"
#include "sampling.h"
#if defined(__ANDROID__)
#include <android/log.h>
#endif

namespace rnllama {

std::string tokens_to_output_formatted_string(const llama_context *ctx, const llama_token token);

std::string tokens_to_str(llama_context *ctx, const std::vector<llama_token>::const_iterator begin, const std::vector<llama_token>::const_iterator end);

lm_ggml_type kv_cache_type_from_str(const std::string & s);

enum stop_type
{
    STOP_FULL,
    STOP_PARTIAL,
};

// completion token output with probabilities
struct completion_token_output
{
    struct token_prob
    {
        llama_token tok;
        float prob;
    };

    std::vector<token_prob> probs;
    llama_token tok;
};

// Main context class
struct llama_rn_context {
    bool is_predicting = false;
    bool is_interrupted = false;
    bool has_next_token = false;
    std::string generated_text;
    std::vector<completion_token_output> generated_token_probs;

    size_t num_prompt_tokens = 0;
    size_t num_tokens_predicted = 0;
    size_t n_past = 0;
    size_t n_remain = 0;

    std::vector<llama_token> embd;
    common_params params;
    common_init_result llama_init;

    llama_model *model = nullptr;
    float loading_progress = 0;
    bool is_load_interrupted = false;

    llama_context *ctx = nullptr;
    common_sampler *ctx_sampling = nullptr;
    common_chat_templates_ptr templates;

    int n_ctx;

    bool truncated = false;
    bool stopped_eos = false;
    bool stopped_word = false;
    bool stopped_limit = false;
    std::string stopping_word;
    bool incomplete = false;

    std::vector<common_adapter_lora_info> lora;

    ~llama_rn_context();

    void rewind();
    bool initSampling();
    bool loadModel(common_params &params_);
    bool validateModelChatTemplate(bool use_jinja, const char *name) const;
    common_chat_params getFormattedChatWithJinja(
      const std::string &messages,
      const std::string &chat_template,
      const std::string &json_schema,
      const std::string &tools,
      const bool &parallel_tool_calls,
      const std::string &tool_choice
    ) const;
    std::string getFormattedChat(
      const std::string &messages,
      const std::string &chat_template
    ) const;
    void truncatePrompt(std::vector<llama_token> &prompt_tokens);
    void loadPrompt();
    void beginCompletion();
    completion_token_output nextToken();
    size_t findStoppingStrings(const std::string &text, const size_t last_token_size, const stop_type type);
    completion_token_output doCompletion();
    std::vector<float> getEmbedding(common_params &embd_params);
    std::string bench(int pp, int tg, int pl, int nr);
    int applyLoraAdapters(std::vector<common_adapter_lora_info> lora);
    void removeLoraAdapters();
    std::vector<common_adapter_lora_info> getLoadedLoraAdapters();
};\

// Logging macros
extern bool rnllama_verbose;

#if RNLLAMA_VERBOSE != 1
#define LOG_VERBOSE(MSG, ...)
#else
#define LOG_VERBOSE(MSG, ...)                                       \
    do                                                              \
    {                                                               \
        if (rnllama_verbose)                                        \
        {                                                           \
            log("VERBOSE", __func__, __LINE__, MSG, ##__VA_ARGS__); \
        }                                                           \
    } while (0)
#endif

#define LOG_ERROR(MSG, ...) log("ERROR", __func__, __LINE__, MSG, ##__VA_ARGS__)
#define LOG_WARNING(MSG, ...) log("WARNING", __func__, __LINE__, MSG, ##__VA_ARGS__)
#define LOG_INFO(MSG, ...) log("INFO", __func__, __LINE__, MSG, ##__VA_ARGS__)

} // namespace rnllama

#endif /* RNLLAMA_H */
