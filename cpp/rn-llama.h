#ifndef RNLLAMA_H
#define RNLLAMA_H

#include <sstream>
#include <iostream>
#include <thread>
#include <codecvt>
#include "chat.h"
#include "common.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include "gguf.h"
#include "llama.h"
#include "llama-model.h"
#include "llama-impl.h"
#include "sampling.h"
#include "nlohmann/json.hpp"
#include "rn-tts.h"
#if defined(__ANDROID__)
#include <android/log.h>
#endif

using json = nlohmann::ordered_json;

namespace rnllama {

std::string tokens_to_output_formatted_string(const llama_context *ctx, const llama_token token);

std::string tokens_to_str(llama_context *ctx, const std::vector<llama_token>::const_iterator begin, const std::vector<llama_token>::const_iterator end);

lm_ggml_type kv_cache_type_from_str(const std::string & s);

enum llama_flash_attn_type flash_attn_type_from_str(const std::string & s);

// Forward declarations - actual definitions are in rn-completion.h
// Note: enum forward declarations not allowed in C++, using include in implementation file
struct completion_token_output;
struct completion_chat_output;
struct llama_rn_context_mtmd;

struct llama_rn_context_tts;

struct llama_rn_context_completion;

struct llama_rn_slot_manager;

struct llama_rn_tokenize_result {
  std::vector<llama_token> tokens;
  bool has_media = false;
  std::vector<std::string> bitmap_hashes;
  std::vector<size_t> chunk_pos; // both text and media
  std::vector<size_t> chunk_pos_media; // media only
};

// Main context class
struct llama_rn_context {
    // Model state fields
    llama_model *model = nullptr;
    float loading_progress = 0;
    bool is_load_interrupted = false;
    common_params params;
    common_init_result llama_init;
    llama_context *ctx = nullptr;
    common_chat_templates_ptr templates;
    int n_ctx;

    // Completion context (DEPRECATED: Use slot_manager for parallel decoding)
    llama_rn_context_completion *completion = nullptr;

    // NEW: Slot manager for parallel decoding
    llama_rn_slot_manager *slot_manager = nullptr;
    bool parallel_mode_enabled = false;

    lm_ggml_threadpool *threadpool = nullptr;
    lm_ggml_threadpool *threadpool_batch = nullptr;

    ~llama_rn_context();

    bool loadModel(common_params &params_);
    void cleanupThreadpools();
    bool attachThreadpoolsIfAvailable();

    // Parallel decoding methods
    void enableParallelMode(int32_t n_parallel, int32_t n_batch = 512);
    void disableParallelMode();

    // Model methods
    bool validateModelChatTemplate(bool use_jinja, const char *name) const;
    common_chat_params getFormattedChatWithJinja(
      const std::string& messages,
      const std::string& chat_template,
      const std::string& json_schema,
      const std::string& tools,
      const bool& parallel_tool_calls,
      const std::string& tool_choice,
      const bool& enable_thinking,
      const bool& add_generation_prompt = true,
      const std::string& now_str = "",
      const std::map<std::string, std::string>& chat_template_kwargs = {}
    ) const;
    std::string getFormattedChat(
      const std::string &messages,
      const std::string &chat_template
    ) const;
    llama_rn_tokenize_result tokenize(const std::string &text, const std::vector<std::string> &media_paths);

    // Lora methods
    std::vector<common_adapter_lora_info> lora;
    int applyLoraAdapters(std::vector<common_adapter_lora_info> lora);
    void removeLoraAdapters();
    std::vector<common_adapter_lora_info> getLoadedLoraAdapters();

    // Multimodal fields and methods
    llama_rn_context_mtmd *mtmd_wrapper = nullptr;
    bool has_multimodal = false;
    bool initMultimodal(const std::string &mmproj_path, bool use_gpu);
    bool isMultimodalEnabled() const;
    bool isMultimodalSupportVision() const;
    bool isMultimodalSupportAudio() const;
    void releaseMultimodal();

    // TTS fields and methods (delegated to TTS context)
    llama_rn_context_tts *tts_wrapper = nullptr;
    bool has_vocoder = false;
    bool initVocoder(const std::string &vocoder_model_path, int batch_size = -1);
    bool isVocoderEnabled() const;
    void releaseVocoder();

    // Cache management
    void clearCache(bool clear_data = false);
};

// Utility functions
inline void llama_batch_add(llama_batch *batch, llama_token id, llama_pos pos, std::vector<llama_seq_id> seq_ids, bool logits) {
    batch->token   [batch->n_tokens] = id;
    batch->pos     [batch->n_tokens] = pos;
    batch->n_seq_id[batch->n_tokens] = seq_ids.size();
    for (size_t i = 0; i < seq_ids.size(); i++) {
        batch->seq_id[batch->n_tokens][i] = seq_ids[i];
    }
    batch->logits  [batch->n_tokens] = logits ? 1 : 0;
    batch->n_tokens += 1;
}

// Device info functions
std::string get_backend_devices_info();

// Logging functions
void log(const char *level, const char *function, int line, const char *format, ...);

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
