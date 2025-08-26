#include "rn-completion.h"
#include "rn-llama.h"
#include "rn-tts.h"
#include "rn-mtmd.hpp"

// Include multimodal support
#include "tools/mtmd/mtmd.h"
#include "tools/mtmd/mtmd-helper.h"
#include "tools/mtmd/clip.h"

namespace rnllama {

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

// Helper function to format rerank task: [BOS]query[EOS][SEP]doc[EOS]
static std::vector<llama_token> format_rerank(const llama_vocab * vocab, const std::vector<llama_token> & query, const std::vector<llama_token> & doc) {
    std::vector<llama_token> result;

    // Get EOS token - use SEP token as fallback if EOS is not available
    llama_token eos_token = llama_vocab_eos(vocab);
    if (eos_token == LLAMA_TOKEN_NULL) {
        eos_token = llama_vocab_sep(vocab);
    }

    result.reserve(doc.size() + query.size() + 4);
    if (llama_vocab_get_add_bos(vocab)) {
        result.push_back(llama_vocab_bos(vocab));
    }
    result.insert(result.end(), query.begin(), query.end());
    if (llama_vocab_get_add_eos(vocab)) {
        result.push_back(eos_token);
    }
    if (llama_vocab_get_add_sep(vocab)) {
        result.push_back(llama_vocab_sep(vocab));
    }
    result.insert(result.end(), doc.begin(), doc.end());
    if (llama_vocab_get_add_eos(vocab)) {
        result.push_back(eos_token);
    }

    return result;
}

// Constructor
llama_rn_context_completion::llama_rn_context_completion(llama_rn_context* parent)
    : parent_ctx(parent) {
}

// Destructor
llama_rn_context_completion::~llama_rn_context_completion() {
    if (ctx_sampling != nullptr) {
        common_sampler_free(ctx_sampling);
        ctx_sampling = nullptr;
    }
}

void llama_rn_context_completion::rewind() {
    is_interrupted = false;
    parent_ctx->params.antiprompt.clear();
    parent_ctx->params.sampling.grammar.clear();
    num_prompt_tokens = 0;
    num_tokens_predicted = 0;
    prefill_text = "";
    generated_text = "";
    generated_text.reserve(parent_ctx->params.n_ctx);
    truncated = false;
    context_full = false;
    stopped_eos = false;
    stopped_word = false;
    stopped_limit = false;
    stopping_word = "";
    incomplete = false;
    n_remain = 0;
    n_past = 0;
    parent_ctx->params.sampling.n_prev = parent_ctx->n_ctx;
    if (parent_ctx->isVocoderEnabled()) {
        parent_ctx->tts_wrapper->audio_tokens.clear();
        parent_ctx->tts_wrapper->next_token_uses_guide_token = true;
        parent_ctx->tts_wrapper->guide_tokens.clear();
    }
}

bool llama_rn_context_completion::initSampling() {
    if (ctx_sampling != nullptr) {
        common_sampler_free(ctx_sampling);
    }
    ctx_sampling = common_sampler_init(parent_ctx->model, parent_ctx->params.sampling);
    return ctx_sampling != nullptr;
}

void llama_rn_context_completion::truncatePrompt(std::vector<llama_token> &prompt_tokens) {
    const int n_left = parent_ctx->n_ctx - parent_ctx->params.n_keep;
    const int n_block_size = n_left / 2;
    const int erased_blocks = (prompt_tokens.size() - parent_ctx->params.n_keep - n_block_size) / n_block_size;

    // Keep n_keep tokens at start of prompt (at most n_ctx - 4)
    std::vector<llama_token> new_tokens(prompt_tokens.begin(), prompt_tokens.begin() + parent_ctx->params.n_keep);

    new_tokens.insert(new_tokens.end(), prompt_tokens.begin() + parent_ctx->params.n_keep + erased_blocks * n_block_size, prompt_tokens.end());

    LOG_INFO("input truncated, n_ctx: %d, n_keep: %d, n_left: %d, old_size: %d, new_size: %d",
        parent_ctx->n_ctx,
        parent_ctx->params.n_keep,
        n_left,
        prompt_tokens.size(),
        new_tokens.size()
    );

    truncated = true;
    prompt_tokens = new_tokens;
}

void llama_rn_context_completion::loadPrompt(const std::vector<std::string> &media_paths) {
    bool has_media = !media_paths.empty();

    if (!has_media) {
        std::vector<llama_token> text_tokens;
        // Text-only path
        text_tokens = ::common_tokenize(parent_ctx->ctx, parent_ctx->params.prompt, true, true);
        num_prompt_tokens = text_tokens.size();

        // LOG tokens
        std::stringstream ss;
        ss << "\n" << __func__ << ": prompt_tokens = ";
        for (auto& token : text_tokens) {
            ss << token << " ";
        }
        LOG_INFO("%s\n", ss.str().c_str());

        if (parent_ctx->params.n_keep < 0) {
            parent_ctx->params.n_keep = (int)num_prompt_tokens;
        }
        parent_ctx->params.n_keep = std::min(parent_ctx->n_ctx - 4, parent_ctx->params.n_keep);

        // Handle truncation if needed
        if (num_prompt_tokens >= (size_t)parent_ctx->n_ctx) {
            if (!parent_ctx->params.ctx_shift) {
                context_full = true;
                return;
            }
            truncatePrompt(text_tokens);
            num_prompt_tokens = text_tokens.size();
            LM_GGML_ASSERT(num_prompt_tokens < (size_t)parent_ctx->n_ctx);
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
        auto * kv = llama_get_memory(parent_ctx->ctx);
        llama_memory_seq_rm(kv, 0, n_past, -1);

        LOG_VERBOSE("prompt ingested, n_past: %d, cached: %s, to_eval: %s",
            n_past,
            tokens_to_str(parent_ctx->ctx, embd.cbegin(), embd.cbegin() + n_past).c_str(),
            tokens_to_str(parent_ctx->ctx, embd.cbegin() + n_past, embd.cend()).c_str()
        );
    } else {
        // Multimodal path - process all media paths
        processMedia(parent_ctx->params.prompt, media_paths);
        num_prompt_tokens = embd.size();
    }

    has_next_token = true;

    LOG_INFO("[DEBUG] Input processed: n_past=%d, embd.size=%zu, num_prompt_tokens=%zu, has_media=%d",
             n_past, embd.size(), num_prompt_tokens, has_media ? 1 : 0);
}

void llama_rn_context_completion::beginCompletion() {
    beginCompletion(COMMON_CHAT_FORMAT_CONTENT_ONLY, COMMON_REASONING_FORMAT_NONE, false);
}

void llama_rn_context_completion::beginCompletion(int chat_format, common_reasoning_format reasoning_format, bool thinking_forced_open) {
    // number of tokens to keep when resetting context
    n_remain = parent_ctx->params.n_predict;
    llama_perf_context_reset(parent_ctx->ctx);
    is_predicting = true;

    current_chat_format = chat_format;
    current_reasoning_format = reasoning_format;
    current_thinking_forced_open = thinking_forced_open;
}

void llama_rn_context_completion::endCompletion() {
    is_predicting = false;
}

completion_token_output llama_rn_context_completion::nextToken()
{
    completion_token_output result;
    result.tok = -1;

    if (embd.size() >= (size_t)parent_ctx->params.n_ctx)
    {
        if (!parent_ctx->params.ctx_shift) {
            // If context shifting is disabled, stop generation
            LOG_WARNING("context full, n_ctx: %d, tokens: %d", parent_ctx->params.n_ctx, embd.size());
            has_next_token = false;
            context_full = true;
            return result;
        }

        // Shift context

        const int n_left    = n_past - parent_ctx->params.n_keep - 1;
        const int n_discard = n_left/2;

        auto * kv = llama_get_memory(parent_ctx->ctx);
        llama_memory_seq_rm (kv, 0, parent_ctx->params.n_keep + 1            , parent_ctx->params.n_keep + n_discard + 1);
        llama_memory_seq_add(kv, 0, parent_ctx->params.n_keep + 1 + n_discard, n_past, -n_discard);

        for (size_t i = parent_ctx->params.n_keep + 1 + n_discard; i < embd.size(); i++)
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
        if (n_eval > parent_ctx->params.n_batch)
        {
            n_eval = parent_ctx->params.n_batch;
        }
        if (llama_decode(parent_ctx->ctx, llama_batch_get_one(&embd[n_past], n_eval)))
        {
            LOG_ERROR("failed to eval, n_eval: %d, n_past: %d, n_threads: %d, embd: %s",
                n_eval,
                n_past,
                parent_ctx->params.cpuparams.n_threads,
                tokens_to_str(parent_ctx->ctx, embd.cbegin() + n_past, embd.cend()).c_str()
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

    const llama_vocab* vocab = llama_model_get_vocab(parent_ctx->model);

    if (parent_ctx->params.n_predict == 0)
    {
        has_next_token = false;
        result.tok = llama_vocab_eos(vocab);
        return result;
    }

    {
        // out of user input, sample next token
        std::vector<llama_token_data> candidates;
        candidates.reserve(llama_vocab_n_tokens(vocab));

        llama_token new_token_id = common_sampler_sample(ctx_sampling, parent_ctx->ctx, -1);

        if (llama_vocab_is_eog(vocab, new_token_id)) {
            has_next_token = false;
            stopped_eos = true;
            LOG_VERBOSE("EOS: %s", common_token_to_piece(parent_ctx->ctx, new_token_id).c_str());
            return result;
        }

        if (parent_ctx->tts_wrapper != nullptr && parent_ctx->tts_wrapper->next_token_uses_guide_token && !parent_ctx->tts_wrapper->guide_tokens.empty() && !llama_vocab_is_control(vocab, new_token_id)) {
            new_token_id = parent_ctx->tts_wrapper->guide_tokens[0];
            parent_ctx->tts_wrapper->guide_tokens.erase(parent_ctx->tts_wrapper->guide_tokens.begin());
        }
        if (parent_ctx->tts_wrapper != nullptr) {
            parent_ctx->tts_wrapper->next_token_uses_guide_token = (new_token_id == 198);
        }
        result.tok = new_token_id;

        llama_token_data_array cur_p = *common_sampler_get_candidates(ctx_sampling);

        const int32_t n_probs = parent_ctx->params.sampling.n_probs;

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

    has_next_token = parent_ctx->params.n_predict == -1 || n_remain != 0;
    return result;
}

size_t llama_rn_context_completion::findStoppingStrings(const std::string &text, const size_t last_token_size,
                            const stop_type type)
{
    size_t stop_pos = std::string::npos;
    for (const std::string &word : parent_ctx->params.antiprompt)
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

completion_token_output llama_rn_context_completion::doCompletion()
{
    completion_token_output token_with_probs = nextToken();

    const std::string token_text = token_with_probs.tok == -1 ? "" : common_token_to_piece(parent_ctx->ctx, token_with_probs.tok);
    generated_text += token_text;

    if (parent_ctx->isVocoderEnabled()) {
        tts_type type = parent_ctx->tts_wrapper->getTTSType(parent_ctx);
        if (parent_ctx->tts_wrapper->type == UNKNOWN) {
            parent_ctx->tts_wrapper->type = type;
        }
        if ((type == OUTETTS_V0_2 || type == OUTETTS_V0_3) && (token_with_probs.tok >= 151672 && token_with_probs.tok <= 155772)) {
            parent_ctx->tts_wrapper->audio_tokens.push_back(token_with_probs.tok);
        }
    }

    if (parent_ctx->params.sampling.n_probs > 0)
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
        common_token_to_piece(parent_ctx->ctx, token_with_probs.tok),
        tokens_to_output_formatted_string(parent_ctx->ctx, token_with_probs.tok).c_str(),
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

completion_partial_output llama_rn_context_completion::getPartialOutput(const std::string &token_text) {
    common_chat_syntax syntax;
    syntax.format = static_cast<common_chat_format>(current_chat_format);
    syntax.reasoning_format = current_reasoning_format;
    syntax.thinking_forced_open = current_thinking_forced_open;
    syntax.parse_tool_calls = true;

    common_chat_msg parsed_msg = common_chat_parse(prefill_text + generated_text, true, syntax);

    completion_partial_output result;

    result.content = parsed_msg.content;
    result.reasoning_content = parsed_msg.reasoning_content;
    result.accumulated_text = prefill_text + generated_text;
    result.tool_calls = parsed_msg.tool_calls;

    return result;
}

std::vector<float> llama_rn_context_completion::getEmbedding(common_params &embd_params)
{
    static const int n_embd = llama_model_n_embd(llama_get_model(parent_ctx->ctx));
    if (!embd_params.embedding)
    {
        LOG_WARNING("embedding disabled, embedding: %s", embd_params.embedding);
        return std::vector<float>(n_embd, 0.0f);
    }
    float *data;

    const enum llama_pooling_type pooling_type = llama_pooling_type(parent_ctx->ctx);
    printf("pooling_type: %d\n", pooling_type);
    if (pooling_type == LLAMA_POOLING_TYPE_NONE) {
        data = llama_get_embeddings(parent_ctx->ctx);
    } else {
        data = llama_get_embeddings_seq(parent_ctx->ctx, 0);
    }

    if (!data) {
        return std::vector<float>(n_embd, 0.0f);
    }
    std::vector<float> embedding(data, data + n_embd), out(data, data + n_embd);
    common_embd_normalize(embedding.data(), out.data(), n_embd, embd_params.embd_normalize);
    return out;
}

std::vector<float> llama_rn_context_completion::rerank(const std::string &query, const std::vector<std::string> &documents)
{
    std::vector<float> scores;

    // Check if this model supports reranking (requires rank pooling type)
    const enum llama_pooling_type pooling_type = llama_pooling_type(parent_ctx->ctx);
    if (pooling_type != LLAMA_POOLING_TYPE_RANK) {
        throw std::runtime_error("reranking not supported, pooling_type: " + std::to_string(pooling_type));
    }

    if (!parent_ctx->params.embedding) {
        throw std::runtime_error("embedding disabled but required for reranking");
    }

    const llama_vocab * vocab = llama_model_get_vocab(parent_ctx->model);
    std::vector<llama_token> query_tokens = common_tokenize(vocab, query, false, true);

    scores.reserve(documents.size());

    for (size_t i = 0; i < documents.size(); ++i) {
        rewind();
        embd = {};

        const std::string & document = documents[i];

        std::vector<llama_token> doc_tokens = common_tokenize(vocab, document, false, true);

        std::vector<llama_token> rerank_tokens = format_rerank(vocab, query_tokens, doc_tokens);

        llama_memory_clear(llama_get_memory(parent_ctx->ctx), false);

        // Process the rerank input
        try {
            parent_ctx->params.prompt = tokens_to_str(parent_ctx->ctx, rerank_tokens.begin(), rerank_tokens.end());
            initSampling();
            loadPrompt({}); // No media paths for rerank
            beginCompletion();
            doCompletion();

            // Get the rerank score (single embedding value for rank pooling)
            float *data = llama_get_embeddings_seq(parent_ctx->ctx, 0);
            if (data) {
                scores.push_back(data[0]); // For rank pooling, the score is the first (and only) dimension
            } else {
                scores.push_back(-1e6f); // Default low score if computation failed
            }
        } catch (const std::exception &e) {
            LOG_WARNING("rerank computation failed for document %zu: %s", i, e.what());
            scores.push_back(-1e6f);
        }
        endCompletion();

        // Clear KV cache again to prepare for next document or restore original state
        llama_memory_clear(llama_get_memory(parent_ctx->ctx), false);
    }

    return scores;
}

std::string llama_rn_context_completion::bench(int pp, int tg, int pl, int nr)
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
        std::min(pp, parent_ctx->params.n_ubatch), // max n_tokens is limited by n_ubatch
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

        llama_memory_clear(llama_get_memory(parent_ctx->ctx), true);

        const int64_t t_pp_start = llama_time_us();
        if (llama_decode(parent_ctx->ctx, batch) != 0)
        {
            LOG_ERROR("llama_decode() failed during prompt", "");
        }
        const int64_t t_pp_end = llama_time_us();

        llama_memory_clear(llama_get_memory(parent_ctx->ctx), true);

        if (is_interrupted) break;

        const int64_t t_tg_start = llama_time_us();

        for (int i = 0; i < tg; i++)
        {
            llama_batch_clear(&batch);

            for (int j = 0; j < pl; j++)
            {
                llama_batch_add(&batch, 0, i, {j}, true);
            }

            if (llama_decode(parent_ctx->ctx, batch) != 0)
            {
                LOG_ERROR("llama_decode() failed during text generation", "");
            }
            if (is_interrupted) break;
        }

        const int64_t t_tg_end = llama_time_us();

        llama_memory_clear(llama_get_memory(parent_ctx->ctx), true);

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

    if (is_interrupted) llama_memory_clear(llama_get_memory(parent_ctx->ctx), true);
    endCompletion();

    char model_desc[128];
    llama_model_desc(parent_ctx->model, model_desc, sizeof(model_desc));
    return std::string("[\"") + model_desc + std::string("\",") +
        std::to_string(llama_model_size(parent_ctx->model)) + std::string(",") +
        std::to_string(llama_model_n_params(parent_ctx->model)) + std::string(",") +
        std::to_string(pp_avg) + std::string(",") +
        std::to_string(pp_std) + std::string(",") +
        std::to_string(tg_avg) + std::string(",") +
        std::to_string(tg_std) +
        std::string("]");
}

void llama_rn_context_completion::processMedia(
    const std::string &prompt,
    const std::vector<std::string> &media_paths
) {
    if (!parent_ctx->isMultimodalEnabled()) {
        throw std::runtime_error("Multimodal is not enabled but image paths are provided");
    }

    // Delegate to the mtmd_wrapper method
    parent_ctx->mtmd_wrapper->processMedia(
        parent_ctx->ctx,
        prompt,
        media_paths,
        parent_ctx->n_ctx,
        parent_ctx->params.n_batch,
        n_past,
        embd,
        context_full,
        ctx_sampling
    );
}

} // namespace rnllama
