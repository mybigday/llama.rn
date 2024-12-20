#ifndef RNLLAMA_H
#define RNLLAMA_H

#include <sstream>
#include <iostream>
#include "common.h"
#include "ggml.h"
#include "llama.h"
#include "llama-impl.h"
#include "sampling.h"

namespace rnllama {

const std::vector<lm_ggml_type> kv_cache_types = {
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

static lm_ggml_type kv_cache_type_from_str(const std::string & s) {
    for (const auto & type : kv_cache_types) {
        if (lm_ggml_type_name(type) == s) {
            return type;
        }
    }
    throw std::runtime_error("Unsupported cache type: " + s);
}

static std::string lm_gguf_data_to_str(enum lm_gguf_type type, const void * data, int i) {
    switch (type) {
        case LM_GGUF_TYPE_UINT8:   return std::to_string(((const uint8_t  *)data)[i]);
        case LM_GGUF_TYPE_INT8:    return std::to_string(((const int8_t   *)data)[i]);
        case LM_GGUF_TYPE_UINT16:  return std::to_string(((const uint16_t *)data)[i]);
        case LM_GGUF_TYPE_INT16:   return std::to_string(((const int16_t  *)data)[i]);
        case LM_GGUF_TYPE_UINT32:  return std::to_string(((const uint32_t *)data)[i]);
        case LM_GGUF_TYPE_INT32:   return std::to_string(((const int32_t  *)data)[i]);
        case LM_GGUF_TYPE_UINT64:  return std::to_string(((const uint64_t *)data)[i]);
        case LM_GGUF_TYPE_INT64:   return std::to_string(((const int64_t  *)data)[i]);
        case LM_GGUF_TYPE_FLOAT32: return std::to_string(((const float    *)data)[i]);
        case LM_GGUF_TYPE_FLOAT64: return std::to_string(((const double   *)data)[i]);
        case LM_GGUF_TYPE_BOOL:    return ((const bool *)data)[i] ? "true" : "false";
        default:                   return "unknown type: " + std::to_string(type);
    }
}

static std::string lm_gguf_kv_to_str(const struct lm_gguf_context * ctx_gguf, int i) {
    const enum lm_gguf_type type = lm_gguf_get_kv_type(ctx_gguf, i);

    switch (type) {
        case LM_GGUF_TYPE_STRING:
            return lm_gguf_get_val_str(ctx_gguf, i);
        case LM_GGUF_TYPE_ARRAY:
            {
                const enum lm_gguf_type arr_type = lm_gguf_get_arr_type(ctx_gguf, i);
                int arr_n = lm_gguf_get_arr_n(ctx_gguf, i);
                const void * data = lm_gguf_get_arr_data(ctx_gguf, i);
                std::stringstream ss;
                ss << "[";
                for (int j = 0; j < arr_n; j++) {
                    if (arr_type == LM_GGUF_TYPE_STRING) {
                        std::string val = lm_gguf_get_arr_str(ctx_gguf, i, j);
                        // escape quotes
                        replace_all(val, "\\", "\\\\");
                        replace_all(val, "\"", "\\\"");
                        ss << '"' << val << '"';
                    } else if (arr_type == LM_GGUF_TYPE_ARRAY) {
                        ss << "???";
                    } else {
                        ss << lm_gguf_data_to_str(arr_type, data, j);
                    }
                    if (j < arr_n - 1) {
                        ss << ", ";
                    }
                }
                ss << "]";
                return ss.str();
            }
        default:
            return lm_gguf_data_to_str(type, lm_gguf_get_val_data(ctx_gguf, i), 0);
    }
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
    printf("[%s] %s:%d ", level, function, line);

    va_list args;
    va_start(args, format);
    vprintf(format, args);
    va_end(args);

    printf("\n");
}

static bool rnllama_verbose = false;

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
static std::string tokens_to_output_formatted_string(const llama_context *ctx, const llama_token token)
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

template <class Iter>
static std::string tokens_to_str(llama_context *ctx, Iter begin, Iter end)
{
    std::string ret;
    for (; begin != end; ++begin)
    {
        ret += common_token_to_piece(ctx, *begin);
    }
    return ret;
}

struct llama_rn_context
{
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

    llama_model *model = nullptr;
    float loading_progress = 0;
    bool is_load_interrupted = false;

    llama_context *ctx = nullptr;
    common_sampler *ctx_sampling = nullptr;

    int n_ctx;

    bool truncated = false;
    bool stopped_eos = false;
    bool stopped_word = false;
    bool stopped_limit = false;
    std::string stopping_word;
    bool incomplete = false;

    std::vector<common_lora_adapter_container> lora_adapters;

    ~llama_rn_context()
    {
        if (ctx)
        {
            llama_free(ctx);
            ctx = nullptr;
        }
        if (model)
        {
            llama_free_model(model);
            model = nullptr;
        }
        if (ctx_sampling != nullptr)
        {
            common_sampler_free(ctx_sampling);
        }
    }

    void rewind()
    {
        is_interrupted = false;
        params.antiprompt.clear();
        params.sampling.grammar.clear();
        num_prompt_tokens = 0;
        num_tokens_predicted = 0;
        generated_text = "";
        generated_text.reserve(params.n_ctx);
        generated_token_probs.clear();
        truncated = false;
        stopped_eos = false;
        stopped_word = false;
        stopped_limit = false;
        stopping_word = "";
        incomplete = false;
        n_remain = 0;
        n_past = 0;
        params.sampling.n_prev = n_ctx;
    }

    bool initSampling() {
        if (ctx_sampling != nullptr) {
            common_sampler_free(ctx_sampling);
        }
        ctx_sampling = common_sampler_init(model, params.sampling);
        return ctx_sampling != nullptr;
    }

    bool loadModel(common_params &params_)
    {
        params = params_;
        common_init_result result = common_init_from_params(params);
        model = result.model;
        ctx = result.context;
        if (model == nullptr)
        {
           LOG_ERROR("unable to load model: %s", params_.model.c_str());
           return false;
        }
        n_ctx = llama_n_ctx(ctx);
        return true;
    }

    bool validateModelChatTemplate() const {
        std::vector<char> model_template(2048, 0); // longest known template is about 1200 bytes
        std::string template_key = "tokenizer.chat_template";
        int32_t res = llama_model_meta_val_str(model, template_key.c_str(), model_template.data(), model_template.size());
        if (res >= 0) {
            llama_chat_message chat[] = {{"user", "test"}};
            std::string tmpl = std::string(model_template.data(), model_template.size());
            int32_t chat_res = llama_chat_apply_template(model, tmpl.c_str(), chat, 1, true, nullptr, 0);
            return chat_res > 0;
        }
        return res > 0;
    }

    void truncatePrompt(std::vector<llama_token> &prompt_tokens) {
        const int n_left = n_ctx - params.n_keep;
        const int n_block_size = n_left / 2;
        const int erased_blocks = (prompt_tokens.size() - params.n_keep - n_block_size) / n_block_size;

        // Keep n_keep tokens at start of prompt (at most n_ctx - 4)
        std::vector<llama_token> new_tokens(prompt_tokens.begin(), prompt_tokens.begin() + params.n_keep);

        new_tokens.insert(new_tokens.end(), prompt_tokens.begin() + params.n_keep + erased_blocks * n_block_size, prompt_tokens.end());

        LOG_VERBOSE("input truncated, n_ctx: %d, n_keep: %d, n_left: %d, new_tokens: %s, num_prompt_tokens: %d",
            n_ctx,
            params.n_keep,
            n_left,
            tokens_to_str(ctx, new_tokens.cbegin(), new_tokens.cend()).c_str(),
            new_tokens.size()
        );

        truncated = true;
        prompt_tokens = new_tokens;
    }

    void loadPrompt()
    {
        std::vector<llama_token> prompt_tokens = ::common_tokenize(ctx, params.prompt, true, true);
        num_prompt_tokens = prompt_tokens.size();

        // LOG tokens
        std::stringstream ss;
        ss << "\n" << __func__ << ": prompt_tokens = ";
        for (auto& token : prompt_tokens) {
            ss << token << " ";
        }
        LOG_INFO("%s\n", ss.str().c_str());

        if (params.n_keep < 0)
        {
            params.n_keep = (int)num_prompt_tokens;
        }
        params.n_keep = std::min(n_ctx - 4, params.n_keep);

        // if input prompt is too big, truncate like normal
        if (num_prompt_tokens >= (size_t) n_ctx)
        {
            truncatePrompt(prompt_tokens);
            num_prompt_tokens = prompt_tokens.size();

            LM_GGML_ASSERT(num_prompt_tokens < (size_t) n_ctx);
        }
        // push the prompt into the sampling context (do not apply grammar)
        for (auto & token : prompt_tokens)
        {
           common_sampler_accept(ctx_sampling, token, false);
        }

        // compare the evaluated prompt with the new prompt
        n_past = common_part(embd, prompt_tokens);

        embd = prompt_tokens;
        if (n_past == num_prompt_tokens)
        {
            // we have to evaluate at least 1 token to generate logits.
            n_past--;
        }

        // since #3228 we now have to manually manage the KV cache
        llama_kv_cache_seq_rm(ctx, 0, n_past, -1);

        LOG_VERBOSE("prompt ingested, n_past: %d, cached: %s, to_eval: %s",
            n_past,
            tokens_to_str(ctx, embd.cbegin(), embd.cbegin() + n_past).c_str(),
            tokens_to_str(ctx, embd.cbegin() + n_past, embd.cend()).c_str()
        );

        has_next_token = true;
    }

    void beginCompletion()
    {
        // number of tokens to keep when resetting context
        n_remain = params.n_predict;
        llama_perf_context_reset(ctx);
        is_predicting = true;
    }

    completion_token_output nextToken()
    {
        completion_token_output result;
        result.tok = -1;

        if (embd.size() >= (size_t)params.n_ctx)
        {
            // Shift context

            const int n_left    = n_past - params.n_keep - 1;
            const int n_discard = n_left/2;

            llama_kv_cache_seq_rm (ctx, 0, params.n_keep + 1            , params.n_keep + n_discard + 1);
            llama_kv_cache_seq_add(ctx, 0, params.n_keep + 1 + n_discard, n_past, -n_discard);

            for (size_t i = params.n_keep + 1 + n_discard; i < embd.size(); i++)
            {
                embd[i - n_discard] = embd[i];
            }
            embd.resize(embd.size() - n_discard);

            n_past -= n_discard;

            LOG_VERBOSE("input truncated, n_ctx: %d, n_keep: %d, n_left: %d, new_tokens: %s",
                params.n_ctx,
                params.n_keep,
                n_left
            );
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

        if (params.n_predict == 0)
        {
            has_next_token = false;
            result.tok = llama_token_eos(model);
            return result;
        }

        {
            // out of user input, sample next token
            std::vector<llama_token_data> candidates;
            candidates.reserve(llama_n_vocab(model));

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

        if (!embd.empty() && embd.back() == llama_token_eos(model))
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

    size_t findStoppingStrings(const std::string &text, const size_t last_token_size,
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

    completion_token_output doCompletion()
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

    std::vector<float> getEmbedding(common_params &embd_params)
    {
        static const int n_embd = llama_n_embd(llama_get_model(ctx));
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

    std::string bench(int pp, int tg, int pl, int nr)
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
        llama_batch batch = llama_batch_init(512, 0, 1);

        for (int i = 0; i < nr; i++)
        {
            llama_batch_clear(&batch);

            const int n_tokens = pp;

            for (int i = 0; i < n_tokens; i++)
            {
                llama_batch_add(&batch, 0, i, {0}, false);
            }
            batch.logits[batch.n_tokens - 1] = 1; // true

            llama_kv_cache_clear(ctx);

            const int64_t t_pp_start = llama_time_us();
            if (llama_decode(ctx, batch) != 0)
            {
                LOG_ERROR("llama_decode() failed during prompt", "");
            }
            const int64_t t_pp_end = llama_time_us();
            llama_kv_cache_clear(ctx);

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

            llama_kv_cache_clear(ctx);

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

        if (is_interrupted) llama_kv_cache_clear(ctx);
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

    int applyLoraAdapters(std::vector<common_lora_adapter_info> lora_adapters) {
        this->lora_adapters.clear();
        auto containers = std::vector<common_lora_adapter_container>();
        for (auto & la : lora_adapters) {
            common_lora_adapter_container loaded_la;
            loaded_la.path = la.path;
            loaded_la.scale = la.scale;
            loaded_la.adapter = llama_lora_adapter_init(model, la.path.c_str());
            if (loaded_la.adapter == nullptr) {
                LOG_ERROR("%s: failed to apply lora adapter '%s'\n", __func__, la.path.c_str());
                return -1;
            }

            this->lora_adapters.push_back(loaded_la);
            containers.push_back(loaded_la);
        }
        common_lora_adapters_apply(ctx, containers);
        return 0;
    }

    void removeLoraAdapters() {
        this->lora_adapters.clear();
        common_lora_adapters_apply(ctx, this->lora_adapters); // apply empty list
    }

    std::vector<common_lora_adapter_container> getLoadedLoraAdapters() {
        return this->lora_adapters;
    }
};

}

#endif /* LLAMA_H */
