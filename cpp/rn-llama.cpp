#include "rn-llama.h"
#include "rn-tts.h"
#include "rn-mtmd.hpp"
#include "rn-completion.h"

// Include multimodal support
#include "tools/mtmd/mtmd.h"
#include "tools/mtmd/mtmd-helper.h"
#include "tools/mtmd/clip.h"

namespace rnllama {

static const std::vector<lm_ggml_type> kv_cache_types = {
    LM_GGML_TYPE_F32,
    LM_GGML_TYPE_F16,
    LM_GGML_TYPE_BF16,
    LM_GGML_TYPE_Q8_0,
    LM_GGML_TYPE_Q4_0,
    LM_GGML_TYPE_Q4_1,
    LM_GGML_TYPE_IQ4_NL,
    LM_GGML_TYPE_Q5_0,
    LM_GGML_TYPE_Q5_1,
};

lm_ggml_type kv_cache_type_from_str(const std::string & s) {
    if (s.empty()) {
        return LM_GGML_TYPE_F16; // Default to F16 if empty string
    }

    for (const auto & type : kv_cache_types) {
        if (lm_ggml_type_name(type) == s) {
            return type;
        }
    }

    // Return default type instead of throwing to avoid crashes
    return LM_GGML_TYPE_F16;
}


void log(const char *level, const char *function, int line,
                       const char *format, ...)
{
    va_list args;
    #if defined(__ANDROID__)
        char prefix[256];
        snprintf(prefix, sizeof(prefix), "%s:%d %s", function, line, format);

        va_start(args, format);
        android_LogPriority priority;
        if (strcmp(level, "ERROR") == 0) {
            priority = ANDROID_LOG_ERROR;
        } else if (strcmp(level, "WARNING") == 0) {
            priority = ANDROID_LOG_WARN;
        } else if (strcmp(level, "INFO") == 0) {
            priority = ANDROID_LOG_INFO;
        } else {
            priority = ANDROID_LOG_DEBUG;
        }
        __android_log_vprint(priority, "RNLlama", prefix, args);
        va_end(args);
    #else
        printf("[%s] %s:%d ", level, function, line);
        va_start(args, format);
        vprintf(format, args);
        va_end(args);
        printf("\n");
    #endif
}

static bool ends_with(const std::string &str, const std::string &suffix)
{
    return str.size() >= suffix.size() &&
           0 == str.compare(str.size() - suffix.size(), suffix.size(), suffix);
}

static size_t find_partial_stop_string(const std::string &stop,
                                       const std::string &text)
{
    if (!text.empty() && !stop.empty())
    {
        const char text_last_char = text.back();
        for (int64_t char_index = stop.size() - 1; char_index >= 0; char_index--)
        {
            if (stop[char_index] == text_last_char)
            {
                const std::string current_partial = stop.substr(0, char_index + 1);
                if (ends_with(text, current_partial))
                {
                    return text.size() - char_index - 1;
                }
            }
        }
    }
    return std::string::npos;
}

// format incomplete utf-8 multibyte character for output
std::string tokens_to_output_formatted_string(const llama_context *ctx, const llama_token token)
{
    std::string out = token == -1 ? "" : common_token_to_piece(ctx, token);
    // if the size is 1 and first bit is 1, meaning it's a partial character
    //   (size > 1 meaning it's already a known token)
    if (out.size() == 1 && (out[0] & 0x80) == 0x80)
    {
        std::stringstream ss;
        ss << std::hex << (out[0] & 0xff);
        std::string res(ss.str());
        out = "byte: \\x" + res;
    }
    return out;
}

std::string tokens_to_str(llama_context *ctx, const std::vector<llama_token>::const_iterator begin, const std::vector<llama_token>::const_iterator end)
{
    std::string ret;
    for (auto it = begin; it != end; ++it)
    {
        ret += common_token_to_piece(ctx, *it);
    }
    return ret;
}


llama_rn_context::~llama_rn_context() {
    if (completion != nullptr) {
        delete completion;
        completion = nullptr;
    }

    releaseMultimodal();
    releaseVocoder();
}

bool llama_rn_context::loadModel(common_params &params_)
{
    params = params_;
    llama_init = common_init_from_params(params);
    model = llama_init.model.get();
    ctx = llama_init.context.get();
    if (model == nullptr)
    {
        LOG_ERROR("unable to load model: %s", params_.model.path.c_str());
        return false;
    }
    templates = common_chat_templates_init(model, params.chat_template);
    n_ctx = llama_n_ctx(ctx);

    // Initialize completion context
    if (completion != nullptr) {
        delete completion;
    }
    completion = new llama_rn_context_completion(this);

    // Initialize context shift flag
    LOG_INFO("ctx_shift: %s", params.ctx_shift ? "enabled" : "disabled");

    // We can uncomment for debugging or after this fix: https://github.com/ggerganov/llama.cpp/pull/11101
    // LOG_INFO("%s\n", common_params_get_system_info(params).c_str());

    return true;
}


bool llama_rn_context::validateModelChatTemplate(bool use_jinja, const char *name) const {
    const char * tmpl = llama_model_chat_template(model, name);
    if (tmpl == nullptr) {
      return false;
    }
    return common_chat_verify_template(tmpl, use_jinja);
}

common_chat_params llama_rn_context::getFormattedChatWithJinja(
        const std::string& messages,
        const std::string& chat_template,
        const std::string& json_schema,
        const std::string& tools,
        const bool& parallel_tool_calls,
        const std::string& tool_choice,
        const bool& enable_thinking,
        const bool& add_generation_prompt,
        const std::string& now_str,
        const std::map<std::string, std::string>& chat_template_kwargs
) const {
    common_chat_templates_inputs inputs;
    inputs.use_jinja = true;
    inputs.messages = common_chat_msgs_parse_oaicompat(json::parse(messages));
    auto useTools = !tools.empty();
    if (useTools) {
        inputs.tools = common_chat_tools_parse_oaicompat(json::parse(tools));
    }
    inputs.parallel_tool_calls = parallel_tool_calls;
    if (!tool_choice.empty()) {
        inputs.tool_choice = common_chat_tool_choice_parse_oaicompat(tool_choice);
    }
    if (!json_schema.empty()) {
        inputs.json_schema = json::parse(json_schema);
    }
    inputs.enable_thinking = enable_thinking;
    inputs.add_generation_prompt = add_generation_prompt;

    // Handle now parameter - parse timestamp or use current time
    if (!now_str.empty()) {
        try {
            // Try to parse as timestamp (seconds since epoch)
            auto timestamp = std::stoll(now_str);
            inputs.now = std::chrono::system_clock::from_time_t(timestamp);
        } catch (...) {
            // If parsing fails, use current time
            inputs.now = std::chrono::system_clock::now();
        }
    }

    inputs.chat_template_kwargs = chat_template_kwargs;

    // If chat_template is provided, create new one and use it (probably slow)
    if (!chat_template.empty()) {
        auto tmps = common_chat_templates_init(model, chat_template);
        return common_chat_templates_apply(tmps.get(), inputs);
    } else {
        return common_chat_templates_apply(templates.get(), inputs);
    }
}

std::string llama_rn_context::getFormattedChat(
  const std::string &messages,
  const std::string &chat_template
) const {
    common_chat_templates_inputs inputs;
    inputs.messages = common_chat_msgs_parse_oaicompat(json::parse(messages));
    inputs.use_jinja = false;

    // If chat_template is provided, create new one and use it (probably slow)
    if (!chat_template.empty()) {
        auto tmps = common_chat_templates_init(model, chat_template);
        return common_chat_templates_apply(tmps.get(), inputs).prompt;
    } else {
        return common_chat_templates_apply(templates.get(), inputs).prompt;
    }
}

llama_rn_tokenize_result llama_rn_context::tokenize(const std::string &text, const std::vector<std::string> &media_paths) {
  if (media_paths.size() > 0) {
      if (!isMultimodalEnabled()) {
          throw std::runtime_error("Multimodal is not enabled but media paths are provided");
      }
      auto result = tokenizeWithMedia(mtmd_wrapper, text, media_paths);
      mtmd_input_chunks_free(result.chunks);
      llama_rn_tokenize_result tokenize_result = {
          .tokens = result.tokens,
          .has_media = true,
          .bitmap_hashes = result.bitmap_hashes,
          .chunk_pos = result.chunk_pos,
          .chunk_pos_media = result.chunk_pos_media,
      };
      return tokenize_result;
  }
  std::vector<llama_token> text_tokens;
  text_tokens = common_tokenize(ctx, text, false);
  llama_rn_tokenize_result tokenize_result = {
      .tokens = text_tokens,
      .has_media = false,
      .bitmap_hashes = {},
      .chunk_pos = {},
      .chunk_pos_media = {},
  };
  return tokenize_result;
}

int llama_rn_context::applyLoraAdapters(std::vector<common_adapter_lora_info> lora) {
    for (auto &la : lora) {
        la.ptr = llama_adapter_lora_init(model, la.path.c_str());
        if (la.ptr == nullptr) {
            LOG_ERROR("failed to apply lora adapter '%s'\n", la.path.c_str());
            return -1;
        }
    }
    this->lora = lora;
    common_set_adapter_lora(ctx, lora);
    return 0;
}

void llama_rn_context::removeLoraAdapters() {
    this->lora.clear();
    common_set_adapter_lora(ctx, this->lora); // apply empty list
}

std::vector<common_adapter_lora_info> llama_rn_context::getLoadedLoraAdapters() {
    return this->lora;
}

bool llama_rn_context::initMultimodal(const std::string &mmproj_path, bool use_gpu) {
    try {
        mtmd_wrapper = new llama_rn_context_mtmd(mmproj_path, use_gpu, model, ctx, params, has_multimodal, params);
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR("[DEBUG] Failed to initialize multimodal: %s", e.what());
        return false;
    }
}

bool llama_rn_context::isMultimodalEnabled() const {
    return mtmd_wrapper != nullptr && mtmd_wrapper->isEnabled(has_multimodal);
}

bool llama_rn_context::isMultimodalSupportVision() const {
    return isMultimodalEnabled() && mtmd_wrapper->supportVision();
}

bool llama_rn_context::isMultimodalSupportAudio() const {
    return isMultimodalEnabled() && mtmd_wrapper->supportAudio();
}

void llama_rn_context::releaseMultimodal() {
    if (mtmd_wrapper != nullptr) {
        delete mtmd_wrapper;
        mtmd_wrapper = nullptr;
        has_multimodal = false;
    }
}

bool llama_rn_context::initVocoder(const std::string &vocoder_model_path, int batch_size) {
    try {
        tts_wrapper = new llama_rn_context_tts(vocoder_model_path, batch_size);
        has_vocoder = true;
        return true;
    } catch (const std::exception& e) {
        has_vocoder = false;
        return false;
    }
}

bool llama_rn_context::isVocoderEnabled() const {
    return has_vocoder && tts_wrapper != nullptr;
}

void llama_rn_context::releaseVocoder() {
    if (tts_wrapper != nullptr) {
        delete tts_wrapper;
        tts_wrapper = nullptr;
    }
    has_vocoder = false;
}

}
