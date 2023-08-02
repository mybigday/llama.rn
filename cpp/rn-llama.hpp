#ifndef RNLLAMA_H
#define RNLLAMA_H

#include <sstream>
#include <iostream>
#include "common.h"
#include "llama.h"

namespace rnllama {

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
    std::string out = token == -1 ? "" : llama_token_to_str(ctx, token);
    // if first bit is 1, meaning it's a partial character
    if (out.size() > 0 && (out[0] & 0x80) == 0x80)
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
        ret += llama_token_to_str(ctx, *begin);
    }
    return ret;
}

struct llama_rn_context
{
    bool has_next_token = false;
    std::string generated_text;
    std::vector<completion_token_output> generated_token_probs;

    size_t num_prompt_tokens = 0;
    size_t num_tokens_predicted = 0;
    size_t n_past = 0;
    size_t n_remain = 0;

    std::vector<llama_token> embd;
    std::vector<llama_token> last_n_tokens;

    llama_model *model = nullptr;
    llama_context *ctx = nullptr;
    gpt_params params;

    bool truncated = false;
    bool stopped_eos = false;
    bool stopped_word = false;
    bool stopped_limit = false;
    std::string stopping_word;
    int32_t multibyte_pending = 0;

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
    }

    void rewind()
    {
        params.antiprompt.clear();
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
        multibyte_pending = 0;

        n_remain = 0;
        n_past = 0;
    }

    bool loadModel(gpt_params &params_)
    {
        params = params_;
        std::tie(model, ctx) = llama_init_from_gpt_params(params);
        if (model == nullptr)
        {
           LOG_ERROR("unable to load model: %s", params_.model.c_str());
           return false;
        }

        last_n_tokens.resize(params.n_ctx);
        std::fill(last_n_tokens.begin(), last_n_tokens.end(), 0);
        return true;
    }

    void loadPrompt()
    {
        params.prompt.insert(0, 1, ' '); // always add a first space
        std::vector<llama_token> prompt_tokens = ::llama_tokenize(ctx, params.prompt, true);
        num_prompt_tokens = prompt_tokens.size();

        if (params.n_keep < 0)
        {
            params.n_keep = (int)num_prompt_tokens;
        }
        params.n_keep = std::min(params.n_ctx - 4, params.n_keep);

        // if input prompt is too big, truncate like normal
        if (num_prompt_tokens >= (size_t)params.n_ctx)
        {
            const int n_left = (params.n_ctx - params.n_keep) / 2;
            std::vector<llama_token> new_tokens(prompt_tokens.begin(), prompt_tokens.begin() + params.n_keep);
            const int erased_blocks = (num_prompt_tokens - params.n_keep - n_left - 1) / n_left;
            new_tokens.insert(new_tokens.end(), prompt_tokens.begin() + params.n_keep + erased_blocks * n_left, prompt_tokens.end());
            std::copy(prompt_tokens.end() - params.n_ctx, prompt_tokens.end(), last_n_tokens.begin());

            LOG_VERBOSE("input truncated, n_ctx: %d, n_keep: %d, n_left: %d, new_tokens: %s",
                params.n_ctx, params.n_keep, n_left, tokens_to_str(ctx, new_tokens.cbegin(), new_tokens.cend()).c_str()
            );
            truncated = true;
            prompt_tokens = new_tokens;
        }
        else
        {
            const size_t ps = num_prompt_tokens;
            std::fill(last_n_tokens.begin(), last_n_tokens.end() - ps, 0);
            std::copy(prompt_tokens.begin(), prompt_tokens.end(), last_n_tokens.end() - ps);
        }

        // compare the evaluated prompt with the new prompt
        n_past = common_part(embd, prompt_tokens);
        embd = prompt_tokens;
        if (n_past == num_prompt_tokens)
        {
            // we have to evaluate at least 1 token to generate logits.
            n_past--;
        }

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
        llama_set_rng_seed(ctx, params.seed);
    }

    completion_token_output nextToken()
    {
        completion_token_output result;
        result.tok = -1;

        if (embd.size() >= (size_t)params.n_ctx)
        {
            // Reset context
            const int n_left = (params.n_ctx - params.n_keep) / 2;

            std::vector<llama_token> new_tokens(embd.begin(), embd.begin() + params.n_keep);
            new_tokens.insert(new_tokens.end(), embd.end() - n_left, embd.end());
            embd = new_tokens;
            n_past = params.n_keep;
            truncated = true;
            LOG_VERBOSE("input truncated, n_ctx: %d, n_keep: %d, n_left: %d, new_tokens: %s", 
                params.n_ctx,
                params.n_keep,
                n_left,
                tokens_to_str(ctx, new_tokens.cbegin(), new_tokens.cend()).c_str()
            );
        }

        while (n_past < embd.size())
        {
            int n_eval = (int)embd.size() - n_past;
            if (n_eval > params.n_batch)
            {
                n_eval = params.n_batch;
            }
            if (llama_eval(ctx, &embd[n_past], n_eval, n_past, params.n_threads))
            {
                LOG_ERROR("failed to eval, n_eval: %d, n_past: %d, n_threads: %d, embd: %s",
                    n_eval,
                    n_past,
                    params.n_threads,
                    tokens_to_str(ctx, embd.cbegin() + n_past, embd.cend()).c_str()
                );
                has_next_token = false;
                return result;
            }
            n_past += n_eval;
        }

        if (params.n_predict == 0)
        {
            has_next_token = false;
            result.tok = llama_token_eos();
            return result;
        }

        // out of user input, sample next token
        const float temp = params.temp;
        const int32_t top_k = params.top_k <= 0 ? llama_n_vocab(ctx) : params.top_k;
        const float top_p = params.top_p;
        const float tfs_z = params.tfs_z;
        const float typical_p = params.typical_p;
        const int32_t repeat_last_n = params.repeat_last_n < 0 ? params.n_ctx : params.repeat_last_n;
        const float repeat_penalty = params.repeat_penalty;
        const float alpha_presence = params.presence_penalty;
        const float alpha_frequency = params.frequency_penalty;
        const int mirostat = params.mirostat;
        const float mirostat_tau = params.mirostat_tau;
        const float mirostat_eta = params.mirostat_eta;
        const bool penalize_nl = params.penalize_nl;
        const int32_t n_probs = params.n_probs;

        {
            auto *logits = llama_get_logits(ctx);
            auto n_vocab = llama_n_vocab(ctx);

            // Apply params.logit_bias map
            for (const auto &it : params.logit_bias)
            {
                logits[it.first] += it.second;
            }

            std::vector<llama_token_data> candidates;
            candidates.reserve(n_vocab);
            for (llama_token token_id = 0; token_id < n_vocab; token_id++)
            {
                candidates.emplace_back(llama_token_data{token_id, logits[token_id], 0.0f});
            }

            llama_token_data_array candidates_p = {candidates.data(), candidates.size(), false};

            // Apply penalties
            float nl_logit = logits[llama_token_nl()];
            auto last_n_repeat = std::min(std::min((int)last_n_tokens.size(), repeat_last_n), params.n_ctx);
            llama_sample_repetition_penalty(ctx, &candidates_p,
                                            last_n_tokens.data() + last_n_tokens.size() - last_n_repeat,
                                            last_n_repeat, repeat_penalty);
            llama_sample_frequency_and_presence_penalties(ctx, &candidates_p,
                                                          last_n_tokens.data() + last_n_tokens.size() - last_n_repeat,
                                                          last_n_repeat, alpha_frequency, alpha_presence);
            if (!penalize_nl)
            {
                logits[llama_token_nl()] = nl_logit;
            }

            if (temp <= 0)
            {
                // Greedy sampling
                result.tok = llama_sample_token_greedy(ctx, &candidates_p);
                if (n_probs > 0)
                {
                    llama_sample_softmax(ctx, &candidates_p);
                }
            }
            else
            {
                if (mirostat == 1)
                {
                    static float mirostat_mu = 2.0f * mirostat_tau;
                    const int mirostat_m = 100;
                    llama_sample_temperature(ctx, &candidates_p, temp);
                    result.tok = llama_sample_token_mirostat(ctx, &candidates_p, mirostat_tau, mirostat_eta, mirostat_m, &mirostat_mu);
                }
                else if (mirostat == 2)
                {
                    static float mirostat_mu = 2.0f * mirostat_tau;
                    llama_sample_temperature(ctx, &candidates_p, temp);
                    result.tok = llama_sample_token_mirostat_v2(ctx, &candidates_p, mirostat_tau, mirostat_eta, &mirostat_mu);
                }
                else
                {
                    // Temperature sampling
                    size_t min_keep = std::max(1, n_probs);
                    llama_sample_top_k(ctx, &candidates_p, top_k, min_keep);
                    llama_sample_tail_free(ctx, &candidates_p, tfs_z, min_keep);
                    llama_sample_typical(ctx, &candidates_p, typical_p, min_keep);
                    llama_sample_top_p(ctx, &candidates_p, top_p, min_keep);
                    llama_sample_temperature(ctx, &candidates_p, temp);
                    result.tok = llama_sample_token(ctx, &candidates_p);
                }
            }

            for (size_t i = 0; i < std::min(candidates_p.size, (size_t)n_probs); ++i)
            {
                result.probs.push_back({candidates_p.data[i].id, candidates_p.data[i].p});
            }
            last_n_tokens.erase(last_n_tokens.begin());
            last_n_tokens.push_back(result.tok);
            num_tokens_predicted++;
        }

        // add it to the context
        embd.push_back(result.tok);
        // decrement remaining sampling budget
        --n_remain;

        if (!embd.empty() && embd.back() == llama_token_eos())
        {
            // stopping_word = llama_token_to_str(ctx, embd.back());
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

        const std::string token_text = token_with_probs.tok == -1 ? "" : llama_token_to_str(ctx, token_with_probs.tok);
        generated_text += token_text;

        if (params.n_probs > 0)
        {
            generated_token_probs.push_back(token_with_probs);
        }

        if (multibyte_pending > 0)
        {
            multibyte_pending -= token_text.size();
        }
        else if (token_text.size() == 1)
        {
            const char c = token_text[0];
            // 2-byte characters: 110xxxxx 10xxxxxx
            if ((c & 0xE0) == 0xC0)
            {
                multibyte_pending = 1;
                // 3-byte characters: 1110xxxx 10xxxxxx 10xxxxxx
            }
            else if ((c & 0xF0) == 0xE0)
            {
                multibyte_pending = 2;
                // 4-byte characters: 11110xxx 10xxxxxx 10xxxxxx 10xxxxxx
            }
            else if ((c & 0xF8) == 0xF0)
            {
                multibyte_pending = 3;
            }
            else
            {
                multibyte_pending = 0;
            }
        }

        if (multibyte_pending > 0 && !has_next_token)
        {
            has_next_token = true;
            n_remain++;
        }

        if (!has_next_token && n_remain == 0)
        {
            stopped_limit = true;
        }

        LOG_VERBOSE("next token, token: %s, token_text: %s, has_next_token: %d, n_remain: %d, num_tokens_predicted: %d, stopped_eos: %d, stopped_word: %d, stopped_limit: %d, stopping_word: %s",
            llama_token_to_str(ctx, token_with_probs.tok),
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

    std::vector<float> getEmbedding()
    {
        static const int n_embd = llama_n_embd(ctx);
        if (!params.embedding)
        {
            LOG_WARNING("embedding disabled, embedding: %s", params.embedding);
            return std::vector<float>(n_embd, 0.0f);
        }
        const float *data = llama_get_embeddings(ctx);
        std::vector<float> embedding(data, data + n_embd);
        return embedding;
    }
};

}

#endif /* LLAMA_H */
