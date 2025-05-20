#include "rn-llama.h"

// Include multimodal support
#include "tools/mtmd/mtmd.h"
#include "tools/mtmd/clip.h"

namespace rnllama {

// Computes FNV-1a hash of the data
static std::string fnv_hash(const uint8_t * data, size_t len) {
    const uint64_t fnv_prime = 0x100000001b3ULL;
    uint64_t hash = 0xcbf29ce484222325ULL;

    for (size_t i = 0; i < len; ++i) {
        hash ^= data[i];
        hash *= fnv_prime;
    }
    return std::to_string(hash);
}

static const std::string base64_chars =
             "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
             "abcdefghijklmnopqrstuvwxyz"
             "0123456789+/";

// Base64 decoding function
static std::vector<uint8_t> base64_decode(const std::string &encoded_string) {
    std::vector<uint8_t> decoded;
    int in_len = encoded_string.size();
    int i = 0;
    int j = 0;
    int in_ = 0;
    unsigned char char_array_4[4], char_array_3[3];

    while (in_len-- && (encoded_string[in_] != '=')) {
        if (isspace(encoded_string[in_])) {
            in_++;
            continue;
        }

        if (encoded_string[in_] == '=' || base64_chars.find(encoded_string[in_]) == std::string::npos) {
            break;
        }

        char_array_4[i++] = encoded_string[in_]; in_++;
        if (i == 4) {
            for (i = 0; i < 4; i++) {
                char_array_4[i] = base64_chars.find(char_array_4[i]);
            }

            char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
            char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
            char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];

            for (i = 0; i < 3; i++) {
                decoded.push_back(char_array_3[i]);
            }
            i = 0;
        }
    }

    if (i) {
        for (j = i; j < 4; j++) {
            char_array_4[j] = 0;
        }

        for (j = 0; j < 4; j++) {
            char_array_4[j] = base64_chars.find(char_array_4[j]);
        }

        char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
        char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
        char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];

        for (j = 0; j < i - 1; j++) {
            decoded.push_back(char_array_3[j]);
        }
    }

    return decoded;
}

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
    for (const auto & type : kv_cache_types) {
        if (lm_ggml_type_name(type) == s) {
            return type;
        }
    }
    throw std::runtime_error("Unsupported cache type: " + s);
}

static void llama_batch_clear(llama_batch *batch) {
    batch->n_tokens = 0;
}

static void llama_batch_add(llama_batch *batch, llama_token id, llama_pos pos, std::vector<llama_seq_id> seq_ids, bool logits) {
    batch->token   [batch->n_tokens] = id;
    batch->pos     [batch->n_tokens] = pos;
    batch->n_seq_id[batch->n_tokens] = seq_ids.size();
    for (size_t i = 0; i < seq_ids.size(); i++) {
        batch->seq_id[batch->n_tokens][i] = seq_ids[i];
    }
    batch->logits  [batch->n_tokens] = logits ? 1 : 0;
    batch->n_tokens += 1;
}

// NOTE: Edit from https://github.com/ggerganov/llama.cpp/blob/master/examples/server/server.cpp

static void log(const char *level, const char *function, int line,
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

static size_t common_part(const std::vector<llama_token> &a, const std::vector<llama_token> &b)
{
    size_t i;
    for (i = 0; i < a.size() && i < b.size() && a[i] == b[i]; i++)
    {
    }
    return i;
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

struct llama_rn_context_mtmd {
  mtmd_context *mtmd_ctx = nullptr;
};

llama_rn_context::~llama_rn_context() {
    if (ctx_sampling != nullptr) {
        common_sampler_free(ctx_sampling);
    }

    releaseMultimodal();
}

void llama_rn_context::rewind() {
    is_interrupted = false;
    params.antiprompt.clear();
    params.sampling.grammar.clear();
    num_prompt_tokens = 0;
    num_tokens_predicted = 0;
    generated_text = "";
    generated_text.reserve(params.n_ctx);
    generated_token_probs.clear();
    truncated = false;
    context_full = false;
    stopped_eos = false;
    stopped_word = false;
    stopped_limit = false;
    stopping_word = "";
    incomplete = false;
    n_remain = 0;
    n_past = 0;
    params.sampling.n_prev = n_ctx;
}

bool llama_rn_context::initSampling() {
    if (ctx_sampling != nullptr) {
        common_sampler_free(ctx_sampling);
    }
    ctx_sampling = common_sampler_init(model, params.sampling);
    return ctx_sampling != nullptr;
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
  const std::string &messages,
  const std::string &chat_template,
  const std::string &json_schema,
  const std::string &tools,
  const bool &parallel_tool_calls,
  const std::string &tool_choice
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
    inputs.extract_reasoning = params.reasoning_format != COMMON_REASONING_FORMAT_NONE;

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

void llama_rn_context::truncatePrompt(std::vector<llama_token> &prompt_tokens) {
    const int n_left = n_ctx - params.n_keep;
    const int n_block_size = n_left / 2;
    const int erased_blocks = (prompt_tokens.size() - params.n_keep - n_block_size) / n_block_size;

    // Keep n_keep tokens at start of prompt (at most n_ctx - 4)
    std::vector<llama_token> new_tokens(prompt_tokens.begin(), prompt_tokens.begin() + params.n_keep);

    new_tokens.insert(new_tokens.end(), prompt_tokens.begin() + params.n_keep + erased_blocks * n_block_size, prompt_tokens.end());

    LOG_INFO("input truncated, n_ctx: %d, n_keep: %d, n_left: %d, old_size: %d, new_size: %d",
        n_ctx,
        params.n_keep,
        n_left,
        prompt_tokens.size(),
        new_tokens.size()
    );

    truncated = true;
    prompt_tokens = new_tokens;
}

void llama_rn_context::loadPrompt(const std::vector<std::string> &image_paths) {
    bool has_images = !image_paths.empty() && isMultimodalEnabled();

    LOG_INFO("[DEBUG] loadPrompt: has_images=%d, prompt='%s', image_paths_count=%zu",
             has_images ? 1 : 0, params.prompt.c_str(), image_paths.size());

    // Step 1: Process input (different for text-only vs. multimodal)
    std::vector<llama_token> text_tokens;

    if (!has_images) {
        // Text-only path
        text_tokens = ::common_tokenize(ctx, params.prompt, true, true);
        num_prompt_tokens = text_tokens.size();

        // LOG tokens
        std::stringstream ss;
        ss << "\n" << __func__ << ": prompt_tokens = ";
        for (auto& token : text_tokens) {
            ss << token << " ";
        }
        LOG_INFO("%s\n", ss.str().c_str());

        if (params.n_keep < 0) {
            params.n_keep = (int)num_prompt_tokens;
        }
        params.n_keep = std::min(n_ctx - 4, params.n_keep);

        // Handle truncation if needed
        if (num_prompt_tokens >= (size_t)n_ctx) {
            if (!params.ctx_shift) {
                context_full = true;
                return;
            }
            truncatePrompt(text_tokens);
            num_prompt_tokens = text_tokens.size();
            LM_GGML_ASSERT(num_prompt_tokens < (size_t)n_ctx);
        }

        // Update sampling context
        for (auto & token : text_tokens) {
            common_sampler_accept(ctx_sampling, token, false);
        }

        // compare the evaluated prompt with the new prompt
        n_past = common_part(embd, text_tokens);

        embd = text_tokens;
        if (n_past == num_prompt_tokens) {
            // we have to evaluate at least 1 token to generate logits.
            n_past--;
        }

        // Manage KV cache
        llama_kv_self_seq_rm(ctx, 0, n_past, -1);

        LOG_VERBOSE("prompt ingested, n_past: %d, cached: %s, to_eval: %s",
            n_past,
            tokens_to_str(ctx, embd.cbegin(), embd.cbegin() + n_past).c_str(),
            tokens_to_str(ctx, embd.cbegin() + n_past, embd.cend()).c_str()
        );
    } else {
        // Multimodal path - process all images
        if (!processImage(image_paths, params.prompt, text_tokens)) {
            LOG_ERROR("[DEBUG] Failed to process images", "");
            return;
        }
        num_prompt_tokens = text_tokens.size();
    }

    has_next_token = true;

    LOG_INFO("[DEBUG] Input processed: n_past=%d, embd.size=%zu, num_prompt_tokens=%zu, has_images=%d",
             n_past, embd.size(), num_prompt_tokens, has_images ? 1 : 0);
}

void llama_rn_context::beginCompletion() {
    // number of tokens to keep when resetting context
    n_remain = params.n_predict;
    llama_perf_context_reset(ctx);
    is_predicting = true;
}

completion_token_output llama_rn_context::nextToken()
{
    completion_token_output result;
    result.tok = -1;

    if (embd.size() >= (size_t)params.n_ctx)
    {
        if (!params.ctx_shift) {
            // If context shifting is disabled, stop generation
            LOG_WARNING("context full, n_ctx: %d, tokens: %d", params.n_ctx, embd.size());
            has_next_token = false;
            context_full = true;
            return result;
        }

        // Shift context

        const int n_left    = n_past - params.n_keep - 1;
        const int n_discard = n_left/2;

        llama_kv_self_seq_rm (ctx, 0, params.n_keep + 1            , params.n_keep + n_discard + 1);
        llama_kv_self_seq_add(ctx, 0, params.n_keep + 1 + n_discard, n_past, -n_discard);

        for (size_t i = params.n_keep + 1 + n_discard; i < embd.size(); i++)
        {
            embd[i - n_discard] = embd[i];
        }
        embd.resize(embd.size() - n_discard);

        n_past -= n_discard;
        truncated = true;

        LOG_VERBOSE("context shifted, new n_past: %d, new size: %d", n_past, embd.size());
    }

    bool tg = true;
    while (n_past < embd.size())
    {
        int n_eval = (int)embd.size() - n_past;
        tg = n_eval == 1;
        if (n_eval > params.n_batch)
        {
            n_eval = params.n_batch;
        }
        if (llama_decode(ctx, llama_batch_get_one(&embd[n_past], n_eval)))
        {
            LOG_ERROR("failed to eval, n_eval: %d, n_past: %d, n_threads: %d, embd: %s",
                n_eval,
                n_past,
                params.cpuparams.n_threads,
                tokens_to_str(ctx, embd.cbegin() + n_past, embd.cend()).c_str()
            );
            has_next_token = false;
            return result;
        }
        n_past += n_eval;

        if(is_interrupted) {
            LOG_INFO("Decoding Interrupted");
            embd.resize(n_past);
            has_next_token = false;
            return result;
        }
    }

    const llama_vocab* vocab = llama_model_get_vocab(model);

    if (params.n_predict == 0)
    {
        has_next_token = false;
        result.tok = llama_vocab_eos(vocab);
        return result;
    }

    {
        // out of user input, sample next token
        std::vector<llama_token_data> candidates;
        candidates.reserve(llama_vocab_n_tokens(vocab));

        result.tok = common_sampler_sample(ctx_sampling, ctx, -1);

        llama_token_data_array cur_p = *common_sampler_get_candidates(ctx_sampling);

        const int32_t n_probs = params.sampling.n_probs;

        // deprecated
        /*if (params.sampling.temp <= 0 && n_probs > 0)
        {
            // For llama_sample_token_greedy we need to sort candidates
            llama_sampler_init_softmax();

        }*/


        for (size_t i = 0; i < std::min(cur_p.size, (size_t)n_probs); ++i)
        {
            result.probs.push_back({cur_p.data[i].id, cur_p.data[i].p});
        }

        common_sampler_accept(ctx_sampling, result.tok, true);
        if (tg) {
            num_tokens_predicted++;
        }
    }

    // add it to the context
    embd.push_back(result.tok);
    // decrement remaining sampling budget
    --n_remain;

    if (!embd.empty() && embd.back() == llama_vocab_eos(vocab))
    {
        // stopping_word = llama_token_to_piece(ctx, embd.back());
        has_next_token = false;
        stopped_eos = true;
        LOG_VERBOSE("eos token found", "");
        return result;
    }

    has_next_token = params.n_predict == -1 || n_remain != 0;
    return result;
}

size_t llama_rn_context::findStoppingStrings(const std::string &text, const size_t last_token_size,
                            const stop_type type)
{
    size_t stop_pos = std::string::npos;
    for (const std::string &word : params.antiprompt)
    {
        size_t pos;
        if (type == STOP_FULL)
        {
            const size_t tmp = word.size() + last_token_size;
            const size_t from_pos = text.size() > tmp ? text.size() - tmp : 0;
            pos = text.find(word, from_pos);
        }
        else
        {
            pos = find_partial_stop_string(word, text);
        }
        if (pos != std::string::npos &&
            (stop_pos == std::string::npos || pos < stop_pos))
        {
            if (type == STOP_FULL)
            {
                stopping_word = word;
                stopped_word = true;
                has_next_token = false;
            }
            stop_pos = pos;
        }
    }
    return stop_pos;
}

completion_token_output llama_rn_context::doCompletion()
{
    const completion_token_output token_with_probs = nextToken();

    const std::string token_text = token_with_probs.tok == -1 ? "" : common_token_to_piece(ctx, token_with_probs.tok);
    generated_text += token_text;

    if (params.sampling.n_probs > 0)
    {
        generated_token_probs.push_back(token_with_probs);
    }

    // check if there is incomplete UTF-8 character at the end
    for (unsigned i = 1; i < 5 && i <= generated_text.size(); ++i) {
        unsigned char c = generated_text[generated_text.size() - i];
        if ((c & 0xC0) == 0x80) {
            // continuation byte: 10xxxxxx
            continue;
        }
        if ((c & 0xE0) == 0xC0) {
            // 2-byte character: 110xxxxx ...
            incomplete = i < 2;
        } else if ((c & 0xF0) == 0xE0) {
            // 3-byte character: 1110xxxx ...
            incomplete = i < 3;
        } else if ((c & 0xF8) == 0xF0) {
            // 4-byte character: 11110xxx ...
            incomplete = i < 4;
        }
        // else 1-byte character or invalid byte
        break;
    }

    if (incomplete && !has_next_token)
    {
        has_next_token = true;
        n_remain++;
    }

    if (!has_next_token && n_remain == 0)
    {
        stopped_limit = true;
    }

    LOG_VERBOSE("next token, token: %s, token_text: %s, has_next_token: %d, n_remain: %d, num_tokens_predicted: %d, stopped_eos: %d, stopped_word: %d, stopped_limit: %d, stopping_word: %s",
        common_token_to_piece(ctx, token_with_probs.tok),
        tokens_to_output_formatted_string(ctx, token_with_probs.tok).c_str(),
        has_next_token,
        n_remain,
        num_tokens_predicted,
        stopped_eos,
        stopped_word,
        stopped_limit,
        stopping_word.c_str()
    );
    return token_with_probs;
}

std::vector<float> llama_rn_context::getEmbedding(common_params &embd_params)
{
    static const int n_embd = llama_model_n_embd(llama_get_model(ctx));
    if (!embd_params.embedding)
    {
        LOG_WARNING("embedding disabled, embedding: %s", embd_params.embedding);
        return std::vector<float>(n_embd, 0.0f);
    }
    float *data;

    const enum llama_pooling_type pooling_type = llama_pooling_type(ctx);
    printf("pooling_type: %d\n", pooling_type);
    if (pooling_type == LLAMA_POOLING_TYPE_NONE) {
        data = llama_get_embeddings(ctx);
    } else {
        data = llama_get_embeddings_seq(ctx, 0);
    }

    if (!data) {
        return std::vector<float>(n_embd, 0.0f);
    }
    std::vector<float> embedding(data, data + n_embd), out(data, data + n_embd);
    common_embd_normalize(embedding.data(), out.data(), n_embd, embd_params.embd_normalize);
    return out;
}

std::string llama_rn_context::bench(int pp, int tg, int pl, int nr)
{
    if (is_predicting) {
        LOG_ERROR("cannot benchmark while predicting", "");
        return std::string("[]");
    }

    is_predicting = true;

    double pp_avg = 0;
    double tg_avg = 0;

    double pp_std = 0;
    double tg_std = 0;

    // TODO: move batch into llama_rn_context (related https://github.com/mybigday/llama.rn/issues/30)
    llama_batch batch = llama_batch_init(
        std::min(pp, params.n_ubatch), // max n_tokens is limited by n_ubatch
        0,                         // No embeddings
        1                          // Single sequence
    );

    for (int i = 0; i < nr; i++)
    {
        llama_batch_clear(&batch);

        const int n_tokens = pp;

        for (int i = 0; i < n_tokens; i++)
        {
            llama_batch_add(&batch, 0, i, {0}, false);
        }
        batch.logits[batch.n_tokens - 1] = 1; // true

        llama_kv_self_clear(ctx);

        const int64_t t_pp_start = llama_time_us();
        if (llama_decode(ctx, batch) != 0)
        {
            LOG_ERROR("llama_decode() failed during prompt", "");
        }
        const int64_t t_pp_end = llama_time_us();
        llama_kv_self_clear(ctx);

        if (is_interrupted) break;

        const int64_t t_tg_start = llama_time_us();

        for (int i = 0; i < tg; i++)
        {
            llama_batch_clear(&batch);

            for (int j = 0; j < pl; j++)
            {
                llama_batch_add(&batch, 0, i, {j}, true);
            }

            if (llama_decode(ctx, batch) != 0)
            {
                LOG_ERROR("llama_decode() failed during text generation", "");
            }
            if (is_interrupted) break;
        }

        const int64_t t_tg_end = llama_time_us();

        llama_kv_self_clear(ctx);

        const double t_pp = (t_pp_end - t_pp_start) / 1000000.0;
        const double t_tg = (t_tg_end - t_tg_start) / 1000000.0;

        const double speed_pp = pp / t_pp;
        const double speed_tg = (pl * tg) / t_tg;

        pp_avg += speed_pp;
        tg_avg += speed_tg;

        pp_std += speed_pp * speed_pp;
        tg_std += speed_tg * speed_tg;
    }

    pp_avg /= nr;
    tg_avg /= nr;

    if (nr > 1) {
        pp_std = sqrt(pp_std / (nr - 1) - pp_avg * pp_avg * nr / (nr - 1));
        tg_std = sqrt(tg_std / (nr - 1) - tg_avg * tg_avg * nr / (nr - 1));
    } else {
        pp_std = 0;
        tg_std = 0;
    }

    if (is_interrupted) llama_kv_self_clear(ctx);
    is_predicting = false;

    char model_desc[128];
    llama_model_desc(model, model_desc, sizeof(model_desc));
    return std::string("[\"") + model_desc + std::string("\",") +
        std::to_string(llama_model_size(model)) + std::string(",") +
        std::to_string(llama_model_n_params(model)) + std::string(",") +
        std::to_string(pp_avg) + std::string(",") +
        std::to_string(pp_std) + std::string(",") +
        std::to_string(tg_avg) + std::string(",") +
        std::to_string(tg_std) +
        std::string("]");
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
    LOG_INFO("[DEBUG] Initializing multimodal with mmproj path: %s", mmproj_path.c_str());

    if (model == nullptr) {
        LOG_ERROR("[DEBUG] Model not loaded, cannot initialize multimodal", "");
        return false;
    }

    LOG_INFO("[DEBUG] Model info: n_ctx=%d, n_embd=%d",
             llama_n_ctx(ctx),
             llama_model_n_embd(model));

    // Initialize mtmd context
    mtmd_context_params mtmd_params = mtmd_context_params_default();
    mtmd_params.use_gpu = use_gpu;
    mtmd_params.print_timings = false;
    mtmd_params.n_threads = params.cpuparams.n_threads;
    mtmd_params.verbosity = (lm_ggml_log_level)LM_GGML_LOG_LEVEL_INFO;

    LOG_INFO("[DEBUG] Initializing mtmd context with threads=%d", mtmd_params.n_threads);

    auto mtmd_ctx = mtmd_init_from_file(mmproj_path.c_str(), model, mtmd_params);
    if (mtmd_ctx == nullptr) {
        LOG_ERROR("[DEBUG] Failed to initialize multimodal context with mmproj: %s", mmproj_path.c_str());
        return false;
    }
    mtmd_wrapper = new llama_rn_context_mtmd();
    mtmd_wrapper->mtmd_ctx = mtmd_ctx;

    has_multimodal = true;

    // Check if the model uses M-RoPE or non-causal attention
    bool uses_mrope = mtmd_decode_use_mrope(mtmd_ctx);
    bool uses_non_causal = mtmd_decode_use_non_causal(mtmd_ctx);
    LOG_INFO("[DEBUG] Model multimodal properties: uses_mrope=%d, uses_non_causal=%d",
             uses_mrope ? 1 : 0,
             uses_non_causal ? 1 : 0);

    // Disable context shifting when multimodal is enabled
    // This is because an image chunk may contain multiple tokens
    // and context shifting could break the image representation
    params.ctx_shift = false;

    // params.n_cache_reuse = 0;

    LOG_INFO("Multimodal context initialized successfully with mmproj: %s", mmproj_path.c_str());
    LOG_INFO("Context shifting disabled for multimodal support");
    return true;
}

bool llama_rn_context::processImage(
    const std::vector<std::string> &image_paths,
    const std::string &prompt,
    std::vector<llama_token> &text_tokens
) {
    if (!isMultimodalEnabled()) {
        LOG_ERROR("[DEBUG] Multimodal context not initialized", "");
        return false;
    }

    // Multimodal path
    std::string full_prompt = prompt;
    // Add image marker if it doesn't already exist
    if (full_prompt.find("<__image__>") == std::string::npos) {
        full_prompt += " <__image__>";
    }

    LOG_INFO("[DEBUG] Processing message with role=user, content=%s", full_prompt.c_str());
    LOG_INFO("[DEBUG] Processing %zu images with prompt: %s", image_paths.size(), prompt.c_str());
    LOG_INFO("[DEBUG] Current context state: n_past=%d, n_ctx=%d", n_past, n_ctx);

    // Prepare bitmaps array for all images
    mtmd::bitmaps bitmaps;

    std::vector<std::string> bitmap_hashes;

    // Load all images
    for (const auto& image_path : image_paths) {
        LOG_INFO("[DEBUG] Loading image: %s",
                 image_path.substr(0, 50).c_str()); // Only log part of path for base64

        // Check if it's a base64 image
        if (image_path.compare(0, 11, "data:image/") == 0) {
            LOG_INFO("[DEBUG] Detected base64 encoded image");

            // Parse base64 data
            std::vector<std::string> parts;
            size_t comma_pos = image_path.find(',');
            if (comma_pos == std::string::npos) {
                LOG_ERROR("[DEBUG] Invalid base64 image format, missing comma separator");
                bitmaps.entries.clear();
                return false;
            }

            std::string header = image_path.substr(0, comma_pos);
            std::string base64_data = image_path.substr(comma_pos + 1);

            if (header.find("base64") == std::string::npos) {
                LOG_ERROR("[DEBUG] Image must be base64 encoded");
                bitmaps.entries.clear();
                return false;
            }

            // Decode base64
            try {
                // Decode base64 to binary
                std::vector<uint8_t> image_data = base64_decode(base64_data);
                LOG_INFO("[DEBUG] Base64 decoded, size: %zu bytes", image_data.size());

                // Load bitmap from memory buffer using direct initialization
                mtmd::bitmap bmp(mtmd_helper_bitmap_init_from_buf(image_data.data(), image_data.size()));
                if (!bmp.ptr) {
                    LOG_ERROR("[DEBUG] Failed to load base64 image");
                    bitmaps.entries.clear();
                    return false;
                }

                // Calculate bitmap hash (for KV caching)
                std::string hash = fnv_hash(bmp.data(), bmp.nx()*bmp.ny()*3);
                bmp.set_id(hash.c_str());
                LOG_INFO("[DEBUG] Bitmap hash: %s", hash.c_str());
                bitmaps.entries.push_back(std::move(bmp));
                bitmap_hashes.push_back(hash.c_str());
            } catch (const std::exception& e) {
                LOG_ERROR("[DEBUG] Failed to decode base64 image: %s", e.what());
                bitmaps.entries.clear();
                return false;
            }
        } else if (image_path.compare(0, 7, "http://") == 0 || image_path.compare(0, 8, "https://") == 0) {
            // HTTP URLs are not supported yet
            LOG_ERROR("[DEBUG] HTTP/HTTPS URLs are not supported yet: %s", image_path.c_str());
            bitmaps.entries.clear();
            return false;
        } else {
            // Regular file path
            LOG_INFO("[DEBUG] Loading image from file");

            // Check if file exists
            FILE* file = fopen(image_path.c_str(), "rb");
            if (file == nullptr) {
                LOG_ERROR("[DEBUG] File does not exist or cannot be opened: %s (errno: %d, %s)",
                         image_path.c_str(), errno, strerror(errno));

                bitmaps.entries.clear();
                return false;
            }

            // Get file size
            fseek(file, 0, SEEK_END);
            long file_size = ftell(file);
            fseek(file, 0, SEEK_SET);
            LOG_INFO("[DEBUG] File exists and size is %ld bytes", file_size);
            fclose(file);

            // Create bitmap directly
            mtmd::bitmap bmp(mtmd_helper_bitmap_init_from_file(image_path.c_str()));
            if (!bmp.ptr) {
                LOG_ERROR("[DEBUG] Failed to load image");
                bitmaps.entries.clear();
                return false;
            }

            // Calculate bitmap hash (for KV caching)
            std::string hash = fnv_hash(bmp.data(), bmp.nx()*bmp.ny()*3);
            bmp.set_id(hash.c_str());
            LOG_INFO("[DEBUG] Bitmap hash: %s", hash.c_str());
            bitmaps.entries.push_back(std::move(bmp));
            bitmap_hashes.push_back(hash.c_str());
        }
    }

    // Create input chunks
    LOG_INFO("[DEBUG] Initializing input chunks");
    mtmd_input_chunks* chunks = mtmd_input_chunks_init();
    if (chunks == nullptr) {
        LOG_ERROR("[DEBUG] Failed to initialize input chunks", "");
        bitmaps.entries.clear();
        return false;
    }

    // Create input text
    LOG_INFO("[DEBUG] Setting up input text with add_special=%d, parse_special=%d",
             n_past == 0 ? 1 : 0, 1);
    mtmd_input_text input_text;
    input_text.text = full_prompt.c_str(); // Use the full prompt with image marker
    input_text.add_special = n_past == 0;  // Add BOS token if this is the first message
    input_text.parse_special = true;       // Parse special tokens like <__image__>

    /**
     * Tokenize the text and images together.
     *
     * Example of tokenization for "foo bar <__image__> baz <__image__>":
     *
     * 1. Input text with image markers:
     *
     *    "foo bar <__image__> baz <__image__>"
     *
     * 2. Model-specific markers are added.
     *
     * 3. Text is split and tokenized into chunks:
     *
     *    ┌─────────────┐  ┌─────────────────────────┐  ┌─────────┐  ┌─────────────────────────┐
     *    │ TEXT CHUNK  │  │ IMAGE CHUNK             │  │ TEXT    │  │ IMAGE CHUNK             │
     *    │ "foo bar "  │  │                         │  │ " baz " │  │                         │
     *    └─────────────┘  └─────────────────────────┘  └─────────┘  └─────────────────────────┘
     *          │                     │                      │                    │
     *          ▼                     ▼                      ▼                    ▼
     *    ┌─────────────┐  ┌─────────────────────────┐  ┌─────────┐  ┌─────────────────────────┐
     *    │ [1234,5678] │  │ Image Data Structure    │  │ [9012]  │  │ Image Data Structure    │
     *    └─────────────┘  └─────────────────────────┘  └─────────┘  └─────────────────────────┘
     *
     * 4. Image token structure differences:
     *
     *    For Qwen2VL (uses M-RoPE with 2D positions):
     *    ┌─────────────────────────────────────────┐
     *    │ IMAGE_CHUNK                             │
     *    │ ┌───────────────────────────────────┐   │
     *    │ │ mtmd_image_tokens:                │   │
     *    │ │  nx = 16, ny = 16                 │   │ ← 2D grid (16×16 = 256 tokens)
     *    │ │  use_mrope_pos = true             │   │ ← Uses M-RoPE positioning
     *    │ │  batch_f32 = [image_embeddings]   │   │
     *    │ └───────────────────────────────────┘   │
     *    └─────────────────────────────────────────┘
     *
     *    For other models (uses 1D positions):
     *    ┌─────────────────────────────────────────┐
     *    │ IMAGE_CHUNK                             │
     *    │ ┌───────────────────────────────────┐   │
     *    │ │ mtmd_image_tokens:                │   │
     *    │ │  nx = 256, ny = 1                 │   │ ← 1D sequence (256 tokens)
     *    │ │  use_mrope_pos = false            │   │ ← Uses standard positioning
     *    │ │  batch_f32 = [image_embeddings]   │   │
     *    │ └───────────────────────────────────┘   │
     *    └─────────────────────────────────────────┘
     *
     * 5. Final chunks array:
     *    chunks[0] = TEXT_CHUNK([1234, 5678])
     *    chunks[1] = IMAGE_CHUNK(first_image)
     *    chunks[2] = TEXT_CHUNK([9012])
     *    chunks[3] = IMAGE_CHUNK(second_image)
     */
    LOG_INFO("[DEBUG] Tokenizing text and %zu images", bitmaps.entries.size());
    auto bitmaps_c_ptr = bitmaps.c_ptr();
    int32_t res = mtmd_tokenize(mtmd_wrapper->mtmd_ctx, chunks, &input_text, bitmaps_c_ptr.data(), bitmaps_c_ptr.size());
    if (res != 0) {
        LOG_ERROR("[DEBUG] Failed to tokenize images and text: %d", res);
        mtmd_input_chunks_free(chunks);
        bitmaps.entries.clear();
        return false;
    }

    // Log chunk information
    size_t num_chunks = mtmd_input_chunks_size(chunks);
    LOG_INFO("[DEBUG] Tokenization successful: num_chunks=%zu", num_chunks);

    // Clear text_tokens before adding new tokens
    text_tokens.clear();

    // Create a vector to store all tokens (both text and image)
    std::vector<llama_token> all_tokens;

    // Track the total number of tokens (both text and image)
    size_t total_token_count = 0;

    /**
     * Evaluate the chunks.
     *
     * For our example "foo bar <__image__> baz <__image__>":
     *
     * Token organization in memory:
     *
     *    all_tokens: [t0][t1][NULL][NULL]...[NULL][t2][NULL][NULL]...[NULL]
     *    positions:   0   1    2    3   ...  257   258  259  260 ...  514
     *    chunk_pos:   0        2                   258  259
     *
     *    Where:
     *    - [t0][t1] are text tokens for "foo bar " (positions 0-1)
     *    - [NULL]x256 are placeholder tokens for the first image (positions 2-257)
     *    - [t2] is the text token for " baz " (position 258)
     *    - [NULL]x256 are placeholder tokens for the second image (positions 259-514)
     */
    std::vector<size_t> chunk_pos;
    std::vector<size_t> chunk_pos_images;
    for (size_t i = 0; i < num_chunks; i++) {
        chunk_pos.push_back(total_token_count);

        const mtmd_input_chunk* chunk = mtmd_input_chunks_get(chunks, i);
        mtmd_input_chunk_type chunk_type = mtmd_input_chunk_get_type(chunk);

        if (chunk_type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
            size_t n_tokens;
            const llama_token* tokens = mtmd_input_chunk_get_tokens_text(chunk, &n_tokens);
            LOG_INFO("[DEBUG] Chunk %zu: type=TEXT, n_tokens=%zu", i, n_tokens);

            // Add text tokens
            text_tokens.insert(text_tokens.end(), tokens, tokens + n_tokens);
            all_tokens.insert(all_tokens.end(), tokens, tokens + n_tokens);
            total_token_count += n_tokens;
        } else if (chunk_type == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
            chunk_pos_images.push_back(total_token_count);

            const mtmd_image_tokens* img_tokens = mtmd_input_chunk_get_tokens_image(chunk);
            size_t n_tokens = mtmd_image_tokens_get_n_tokens(img_tokens);
            size_t n_pos = mtmd_image_tokens_get_n_pos(img_tokens);
            LOG_INFO("[DEBUG] Chunk %zu: type=IMAGE, n_tokens=%zu, n_pos=%zu",
                     i, n_tokens, n_pos);

            for (size_t j = 0; j < n_pos; j++) {
                all_tokens.push_back(LLAMA_TOKEN_NULL); // Placeholder token
            }
            total_token_count += n_pos;
        }
    }

    // Check if we have enough context space for all tokens
    if (n_past + all_tokens.size() >= (size_t)n_ctx) {
        LOG_ERROR("[DEBUG] Not enough context space: n_past=%d, tokens=%zu, n_ctx=%d",
                 n_past, all_tokens.size(), n_ctx);
        mtmd_input_chunks_free(chunks);
        bitmaps.entries.clear();
        context_full = true;
        return false;
    }

    n_past = common_part(embd, all_tokens);

    llama_pos new_n_past = n_past;

    // Compare bitmap hashes, if they are not the same, backtrack n_past to the position of the first mismatch
    if (mtmd_bitmap_past_hashes.size() > 0) {
        for (size_t i = 0; i < bitmap_hashes.size(); i++) {
            auto pos = chunk_pos_images[i];
            if (n_past < pos) {
                break;
            }
            if (i >= mtmd_bitmap_past_hashes.size()) {
                break;
            }
            if (bitmap_hashes[i] != mtmd_bitmap_past_hashes[i]) {
                LOG_INFO(
                    "[DEBUG] Bitmap hash mismatch at position %zu, %s != %s",
                    i, bitmap_hashes[i].c_str(), mtmd_bitmap_past_hashes[i].c_str()
                );
                n_past = chunk_pos_images[i];
                new_n_past = n_past;
                break;
            }
        }
    }

    // Clear all KV cache entries after position n_past
    llama_kv_self_seq_rm(ctx, 0, n_past, -1);

    LOG_INFO("[DEBUG] Evaluating chunks: n_past=%d, n_batch=%d", n_past, params.n_batch);

    for (size_t i = 0; i < chunk_pos.size(); i++) {

        LOG_INFO("[DEBUG] Evaluating chunk %zu: n_past=%d, chunk_pos=%zu", i, n_past, chunk_pos[i]);

        // Process chunk only if it's after the current n_past
        if (chunk_pos[i] >= n_past) {
            bool chunk_logits_last = (i == num_chunks - 1);
            auto chunk = mtmd_input_chunks_get(chunks, i);

            int32_t res = mtmd_helper_eval_chunk_single(
                mtmd_wrapper->mtmd_ctx,
                ctx,
                chunk,
                n_past,
                0,
                params.n_batch,
                true,
                &new_n_past
            );
            if (res != 0) {
                LOG_ERROR("[DEBUG] Failed to evaluate chunks", "");
                mtmd_input_chunks_free(chunks);
                bitmaps.entries.clear();
                return res;
            }
            n_past = new_n_past;
        }
    }

    if (n_past == total_token_count && n_past > 0 && all_tokens[n_past - 1] != LLAMA_TOKEN_NULL) {
        // we have to evaluate at least 1 token to generate logits.
        n_past--;
    }

    // Update embd with all tokens (both text and image)
    embd = all_tokens;

    mtmd_bitmap_past_hashes = bitmap_hashes;

    // Update sampling context with text tokens only
    for (auto & token : all_tokens) {
        if (token == LLAMA_TOKEN_NULL) {
            continue;
        }
        common_sampler_accept(ctx_sampling, token, false);
    }

    // Clean up image resources
    LOG_INFO("[DEBUG] Cleaning up resources");
    mtmd_input_chunks_free(chunks);
    bitmaps.entries.clear();
    return true;
}

bool llama_rn_context::isMultimodalEnabled() const {
    return has_multimodal && mtmd_wrapper != nullptr;
}

void llama_rn_context::releaseMultimodal() {
    if (mtmd_wrapper && mtmd_wrapper->mtmd_ctx != nullptr) {
        mtmd_free(mtmd_wrapper->mtmd_ctx);
        mtmd_wrapper->mtmd_ctx = nullptr;
        delete mtmd_wrapper;
        mtmd_wrapper = nullptr;
        has_multimodal = false;
    }
}

}
