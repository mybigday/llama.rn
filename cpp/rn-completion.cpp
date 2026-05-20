#include "rn-completion.h"
#include "rn-llama.h"
#include "rn-tts.h"
#include "rn-mtmd.hpp"
#include "rn-common.hpp"

#include <algorithm>
#include <cstring>
#include <cstdlib>
#include <limits>

// Include multimodal support
#include "tools/mtmd/mtmd.h"
#include "tools/mtmd/mtmd-helper.h"
#include "tools/mtmd/clip.h"

namespace rnllama {

// Constructor
llama_rn_context_completion::llama_rn_context_completion(llama_rn_context* parent)
    : parent_ctx(parent) {
}

// Destructor
llama_rn_context_completion::~llama_rn_context_completion() {
    resetSpeculative();
    if (ctx_sampling != nullptr) {
        common_sampler_free(ctx_sampling);
        ctx_sampling = nullptr;
    }
}

void llama_rn_context_completion::rewind() {
    resetSpeculative();
    is_interrupted = false;
    parent_ctx->params.antiprompt.clear();
    parent_ctx->params.sampling.grammar = {};
    parent_ctx->params.sampling.grammar_lazy = false;
    parent_ctx->params.sampling.grammar_triggers.clear();
    parent_ctx->params.sampling.preserved_tokens.clear();
    parent_ctx->params.sampling.generation_prompt.clear();
    num_prompt_tokens = 0;
    num_tokens_predicted = 0;
    num_draft_tokens = 0;
    num_draft_tokens_accepted = 0;
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

    // Check if this is an encoder-decoder model (like T5)
    const bool is_enc_dec = llama_model_has_encoder(parent_ctx->model);
    const auto vocab = llama_model_get_vocab(parent_ctx->model);
    const bool add_bos = llama_vocab_get_add_bos(vocab);

    if (!has_media) {
        std::vector<llama_token> text_tokens;
        // Text-only path - use modified tokenization for encoder-decoder models
        text_tokens = ::common_tokenize(parent_ctx->ctx, parent_ctx->params.prompt, add_bos || is_enc_dec, true);
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

        // NOTE: Do NOT feed prompt tokens into the sampler.
        // The penalty sampler should only track generated tokens.
        // Feeding prompt tokens causes <|im_end|> (which appears
        // in every ChatML prompt) to be penalised by repeat_penalty
        // / frequency_penalty, preventing EOS and producing
        // extremely verbose output on Qwen-family models.

        // compare the evaluated prompt with the new prompt
        n_past = is_enc_dec ? 0 : find_common_prefix_length(embd, text_tokens);

        embd = text_tokens;
        if (n_past == num_prompt_tokens) {
            // we have to evaluate at least 1 token to generate logits.
            n_past--;
        }

        // Manage KV cache
        auto * kv = llama_get_memory(parent_ctx->ctx);
        bool cache_remove_success = llama_memory_seq_rm(kv, 0, n_past, -1);

        // For hybrid models (LFM-2, Granite, Mamba, etc.), partial cache removal may fail
        // In that case, do a full cache clear to prevent contamination
        if (!cache_remove_success) {
            LOG_WARNING("Partial cache removal failed (likely hybrid/recurrent model), doing full cache clear");
            llama_memory_clear(kv, false);
            embd.clear();
            n_past = 0;
            // Re-assign all tokens to embd since we cleared everything
            embd = text_tokens;
        }

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

    // Handle encoder-decoder models (like T5) with special encoding phase
    if (is_enc_dec && !has_media) {
        // For encoder-decoder models, we need to encode the input tokens first
        if (embd.size() > n_past) {
            // Encode tokens in batches using n_batch as chunk size
            int n_past_batch = n_past;
            int n_remaining = embd.size() - n_past;

            while (n_remaining > 0) {
                int n_eval = n_remaining;
                if (n_eval > parent_ctx->params.n_batch) {
                    n_eval = parent_ctx->params.n_batch;
                }

                int ret = llama_encode(parent_ctx->ctx, llama_batch_get_one(embd.data() + n_past_batch, n_eval));
                if (ret < 0) {
                    LOG_ERROR("Failed to encode token batch, code: %d, n_eval: %d, n_past_batch: %d", ret, n_eval, n_past_batch);
                    has_next_token = false;
                    return;
                }

                n_past_batch += n_eval;
                n_remaining -= n_eval;
                n_past += n_eval;
            }
        }
        // Update token count for encoding
        num_prompt_tokens = embd.size();

        // Add decoder start token for encoder-decoder models
        llama_token decode_bos = llama_model_decoder_start_token(parent_ctx->model);
        if (decode_bos == LLAMA_TOKEN_NULL) {
            decode_bos = llama_vocab_bos(vocab);
        }

        // Add the decoder start token to begin generation
        embd.emplace_back(decode_bos);
        common_sampler_accept(ctx_sampling, decode_bos, false);

        LOG_INFO("[DEBUG] T5 encoding complete, added decoder BOS token: %d", decode_bos);
    }

    has_next_token = true;

    LOG_INFO("[DEBUG] Input processed: n_past=%d, embd.size=%zu, num_prompt_tokens=%zu, has_media=%d",
            n_past, embd.size(), num_prompt_tokens, has_media ? 1 : 0);
}

void llama_rn_context_completion::beginCompletion() {
    beginCompletion(COMMON_CHAT_FORMAT_CONTENT_ONLY, COMMON_REASONING_FORMAT_NONE);
}

void llama_rn_context_completion::beginCompletion(int chat_format, common_reasoning_format reasoning_format, const std::string &generation_prompt, const std::string &chat_parser) {
    // number of tokens to keep when resetting context
    n_remain = parent_ctx->params.n_predict;
    llama_perf_context_reset(parent_ctx->ctx);
    is_predicting = true;

    current_chat_format = chat_format;
    current_reasoning_format = reasoning_format;
    current_generation_prompt = generation_prompt;
    current_chat_parser = chat_parser;
}

void llama_rn_context_completion::endCompletion() {
    is_predicting = false;
}

bool llama_rn_context_completion::shouldUseMTP() const {
    const auto & types = parent_ctx->params.speculative.types;
    return std::find(types.begin(), types.end(), COMMON_SPECULATIVE_TYPE_DRAFT_MTP) != types.end() &&
        parent_ctx->params.speculative.draft.n_max > 0;
}

void llama_rn_context_completion::resetSpeculative() {
    if (spec != nullptr) {
        common_speculative_free(spec);
        spec = nullptr;
    }
    spec_ctx.reset();
    if (spec_batch_initialized) {
        llama_batch_free(spec_batch);
        spec_batch = {};
        spec_batch_initialized = false;
    }
    spec_prompt.clear();
    spec_id_last = LLAMA_TOKEN_NULL;
    spec_n_past = 0;
    spec_draft.clear();
    spec_pending_tokens.clear();
}

void llama_rn_context_completion::initMTP() {
    if (!shouldUseMTP()) {
        return;
    }
    if (llama_model_has_encoder(parent_ctx->model)) {
        throw std::runtime_error("MTP speculative decoding is only supported for decoder-only models");
    }
    if (embd.empty()) {
        throw std::runtime_error("MTP speculative decoding requires a non-empty prompt");
    }

    const auto n_mtp = parent_ctx->params.speculative.draft.n_max;
    if ((llama_model_is_recurrent(parent_ctx->model) || llama_model_is_hybrid(parent_ctx->model)) &&
        llama_n_rs_seq(parent_ctx->ctx) < (uint32_t) n_mtp) {
        throw std::runtime_error(
            "MTP for recurrent or hybrid models must be enabled when loading the model "
            "with speculative.type='draft-mtp' and speculative.n_max/spec_draft_n_max set");
    }

    resetSpeculative();

    auto cparams = common_context_params_to_llama(parent_ctx->params);
    cparams.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
    cparams.n_rs_seq = 0;

    spec_ctx.reset(llama_init_from_model(parent_ctx->model, cparams));
    if (spec_ctx == nullptr) {
        throw std::runtime_error("failed to create MTP draft context");
    }

    parent_ctx->params.speculative.draft.ctx_tgt = parent_ctx->ctx;
    parent_ctx->params.speculative.draft.ctx_dft = spec_ctx.get();

    spec = common_speculative_init(parent_ctx->params.speculative, 1);
    if (spec == nullptr) {
        throw std::runtime_error("failed to initialize MTP speculative decoding");
    }

    spec_batch = llama_batch_init(llama_n_batch(parent_ctx->ctx), 0, 1);
    spec_batch_initialized = true;

    llama_memory_clear(llama_get_memory(parent_ctx->ctx), false);
    llama_memory_clear(llama_get_memory(spec_ctx.get()), false);
    n_past = 0;

    evalMTPPrompt();
}

void llama_rn_context_completion::evalMTPPrompt() {
    const llama_seq_id seq_id = 0;
    const size_t n_prompt = embd.size();

    spec_prompt.clear();
    spec_pending_tokens.clear();
    spec_draft.clear();
    spec_id_last = embd.back();

    if (n_prompt > 1) {
        spec_prompt.assign(embd.begin(), embd.end() - 1);
    }

    const int32_t n_batch = std::max<int32_t>(1, llama_n_batch(parent_ctx->ctx));
    size_t offset = 0;

    while (offset < spec_prompt.size()) {
        common_batch_clear(spec_batch);

        const size_t n_eval = std::min<size_t>(n_batch, spec_prompt.size() - offset);
        for (size_t i = 0; i < n_eval; ++i) {
            // MTP consumes pre-norm embeddings from every target row, but prompt logits are unused.
            // Keep one output row per decode batch to preserve the usual llama.cpp graph shape.
            const bool needs_logits = i + 1 == n_eval;
            common_batch_add(spec_batch, spec_prompt[offset + i],
                             (llama_pos) (offset + i), { seq_id }, needs_logits);
        }

        const int ret = llama_decode(parent_ctx->ctx, spec_batch);
        if (ret != 0) {
            throw std::runtime_error("failed to evaluate MTP prompt batch, ret=" + std::to_string(ret));
        }
        if (!common_speculative_process(spec, spec_batch)) {
            throw std::runtime_error("failed to process MTP prompt batch");
        }

        offset += n_eval;
    }

    spec_n_past = (llama_pos) spec_prompt.size();
    n_past = spec_n_past;

    common_speculative_begin(spec, seq_id, spec_prompt);
}

bool llama_rn_context_completion::refillMTPTokens() {
    const llama_seq_id seq_id = 0;

    if (spec_id_last == LLAMA_TOKEN_NULL || stopped_eos || stopped_limit || context_full) {
        return false;
    }
    if (parent_ctx->params.n_predict >= 0 && n_remain == 0) {
        stopped_limit = true;
        has_next_token = false;
        return false;
    }

    const int32_t n_ctx = parent_ctx->params.n_ctx;
    if (spec_n_past + 1 >= n_ctx) {
        context_full = true;
        has_next_token = false;
        return false;
    }

    spec_draft.clear();

    const int32_t remaining =
        parent_ctx->params.n_predict < 0 ? std::numeric_limits<int32_t>::max() : (int32_t) n_remain;
    const int32_t n_draft_remaining = remaining == std::numeric_limits<int32_t>::max()
        ? parent_ctx->params.speculative.draft.n_max
        : std::max<int32_t>(0, remaining - 1);
    const int32_t n_draft_ctx = std::max<int32_t>(0, n_ctx - (int32_t) spec_n_past - 1);
    const int32_t n_draft_batch = std::max<int32_t>(0, llama_n_batch(parent_ctx->ctx) - 1);
    const int32_t n_draft_limit = std::min<int32_t>(
        parent_ctx->params.speculative.draft.n_max,
        std::min<int32_t>(n_draft_remaining, std::min<int32_t>(n_draft_ctx, n_draft_batch)));

    if (n_draft_limit > 0) {
        common_speculative_get_draft_params(spec, seq_id) = {
            /* .drafting = */ true,
            /* .n_max    = */ n_draft_limit,
            /* .n_past   = */ spec_n_past,
            /* .id_last  = */ spec_id_last,
            /* .prompt   = */ &spec_prompt,
            /* .result   = */ &spec_draft,
        };
        common_speculative_draft(spec);

        if ((int32_t) spec_draft.size() > n_draft_limit) {
            spec_draft.resize(n_draft_limit);
        }

        common_context_seq_rm(spec_ctx.get(), seq_id, spec_n_past, -1);
    }

    const size_t n_draft = spec_draft.size();
    num_draft_tokens += n_draft;

    common_batch_clear(spec_batch);
    common_batch_add(spec_batch, spec_id_last, spec_n_past, { seq_id }, true);
    for (size_t i = 0; i < n_draft; ++i) {
        common_batch_add(spec_batch, spec_draft[i],
                         spec_n_past + (llama_pos) i + 1, { seq_id }, true);
    }

    const int ret = llama_decode(parent_ctx->ctx, spec_batch);
    if (ret != 0) {
        throw std::runtime_error("failed to evaluate MTP target batch, ret=" + std::to_string(ret));
    }
    if (!common_speculative_process(spec, spec_batch)) {
        throw std::runtime_error("failed to process MTP target batch");
    }

    auto accepted = common_sampler_sample_and_accept_n(ctx_sampling, parent_ctx->ctx, spec_draft);
    if (accepted.empty()) {
        return false;
    }

    size_t accepted_count = accepted.size();
    bool saw_eos = false;
    const llama_vocab* vocab = llama_model_get_vocab(parent_ctx->model);
    for (size_t i = 0; i < accepted.size(); ++i) {
        if (llama_vocab_is_eog(vocab, accepted[i])) {
            accepted_count = i + 1;
            saw_eos = true;
            break;
        }

        completion_token_output output;
        output.tok = accepted[i];
        output.text = common_token_to_piece(parent_ctx->ctx, accepted[i]);
        spec_pending_tokens.push_back(std::move(output));
    }

    const size_t n_accepted_draft = saw_eos
        ? accepted_count - 1
        : accepted.size() - 1;
    if (n_draft > 0) {
        const size_t n_accepted = std::min(n_accepted_draft, n_draft);
        num_draft_tokens_accepted += n_accepted;
        common_speculative_accept(spec, seq_id, (uint16_t) n_accepted);
    }

    for (size_t i = 0; i < accepted_count; ++i) {
        spec_prompt.push_back(spec_id_last);
        spec_id_last = accepted[i];
    }

    spec_n_past += (llama_pos) accepted_count;
    n_past = spec_n_past;

    common_context_seq_rm(parent_ctx->ctx, seq_id, spec_n_past, -1);
    common_context_seq_rm(spec_ctx.get(), seq_id, spec_n_past, -1);

    if (saw_eos) {
        stopped_eos = true;
        has_next_token = false;
    }

    if (parent_ctx->params.n_predict >= 0) {
        const size_t emitted = spec_pending_tokens.size();
        n_remain = emitted >= n_remain ? 0 : n_remain - emitted;
        if (n_remain == 0 && !saw_eos) {
            stopped_limit = true;
            has_next_token = false;
        }
    }

    return !spec_pending_tokens.empty();
}

completion_token_output llama_rn_context_completion::nextTokenMTP() {
    completion_token_output result;
    result.tok = -1;

    if (spec == nullptr) {
        initMTP();
    }

    if (spec_pending_tokens.empty() && !refillMTPTokens()) {
        return result;
    }

    result = std::move(spec_pending_tokens.front());
    spec_pending_tokens.pop_front();
    num_tokens_predicted++;
    has_next_token = !spec_pending_tokens.empty() || (!stopped_eos && !stopped_limit && !context_full);
    return result;
}

completion_token_output llama_rn_context_completion::nextToken()
{
    if (shouldUseMTP()) {
        return nextTokenMTP();
    }

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

        const int32_t n_probs = parent_ctx->params.sampling.n_probs;
        if (n_probs > 0) {
          llama_token_data_array cur_p = *common_sampler_get_candidates(ctx_sampling, true);
          for (size_t i = 0; i < std::min(cur_p.size, (size_t)n_probs); ++i)
          {
              result.probs.push_back({cur_p.data[i].id, cur_p.data[i].p});
          }
        }

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
        result.text = common_token_to_piece(parent_ctx->ctx, new_token_id);

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

completion_chat_output llama_rn_context_completion::parseChatOutput(bool is_partial) {
    common_chat_parser_params syntax;
    syntax.format = static_cast<common_chat_format>(current_chat_format);
    syntax.reasoning_format = current_reasoning_format;
    syntax.generation_prompt = current_generation_prompt;
    syntax.parse_tool_calls = true;

    // Load the PEG parser if available (required for COMMON_CHAT_FORMAT_PEG_* formats)
    if (!current_chat_parser.empty()) {
        syntax.parser.load(current_chat_parser);
    }

    common_chat_msg parsed_msg = common_chat_parse(prefill_text + generated_text, is_partial, syntax);

    completion_chat_output result;

    result.content = parsed_msg.content;
    result.reasoning_content = parsed_msg.reasoning_content;
    result.accumulated_text = prefill_text + generated_text;
    result.tool_calls = parsed_msg.tool_calls;

    return result;
}

std::vector<float> llama_rn_context_completion::embedding(common_params &embd_params)
{
    llama_memory_clear(llama_get_memory(parent_ctx->ctx), true);

    rewind();
    llama_perf_context_reset(parent_ctx->ctx);
    if (!initSampling()) {
        throw std::runtime_error("Failed to initialize sampling");
    }
    beginCompletion();
    loadPrompt({});
    doCompletion();
    endCompletion();

    static const int n_embd = llama_model_n_embd(llama_get_model(parent_ctx->ctx));
    if (!embd_params.embedding)
    {
        LOG_WARNING("embedding disabled, embedding: %s", embd_params.embedding);
        return std::vector<float>(n_embd, 0.0f);
    }
    float *data;

    const enum llama_pooling_type pooling_type = llama_pooling_type(parent_ctx->ctx);
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

        std::vector<llama_token> rerank_tokens = format_rerank_tokens(vocab, query_tokens, doc_tokens);

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

std::string llama_rn_context_completion::bench(int pp, int tg, int pl, int nr) {
    if (is_predicting) {
        LOG_ERROR("cannot benchmark while predicting", "");
        return std::string("{}");
    }

    if (pp <= 0 || tg <= 0 || pl <= 0 || nr <= 0) {
        LOG_ERROR("invalid benchmark parameters pp=%d tg=%d pl=%d nr=%d", pp, tg, pl, nr);
        return std::string("{}");
    }

    is_predicting = true;

    auto * ctx = parent_ctx->ctx;
    auto * model = parent_ctx->model;
    auto * mem = llama_get_memory(ctx);

    const bool is_pp_shared = parent_ctx->params.is_pp_shared;
    const bool kv_unified   = parent_ctx->params.kv_unified;
    const int32_t n_batch   = parent_ctx->params.n_batch;
    const int32_t n_ubatch  = parent_ctx->params.n_ubatch;
    const int32_t flash_attn = static_cast<int32_t>(parent_ctx->params.flash_attn_type);
    const int32_t n_gpu_layers = parent_ctx->params.n_gpu_layers;
    const int32_t n_threads = llama_n_threads(ctx);
    const int32_t n_threads_batch = llama_n_threads_batch(ctx);
    const int32_t n_kv_max = llama_n_ctx(ctx);

    const int32_t n_ctx_req = is_pp_shared
        ? (kv_unified ? pp : pl * pp) + pl * tg
        : pl * (pp + tg);

    if (n_ctx_req > n_kv_max) {
        LOG_ERROR("benchmark requires n_ctx=%d but only %d available", n_ctx_req, n_kv_max);
        endCompletion();
        return std::string("{}");
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int32_t n_vocab = vocab ? llama_vocab_n_tokens(vocab) : 0;

    auto get_token_rand = [n_vocab]() -> llama_token {
        if (n_vocab <= 0) {
            return 0;
        }
        return std::rand() % n_vocab;
    };

    llama_batch batch = llama_batch_init(n_kv_max, 0, 1);

    auto decode_helper = [ctx](llama_batch & batch_ref, int32_t n_batch_ref, bool synchronize) -> bool {
        const int32_t total = batch_ref.n_tokens;
        for (int32_t i = 0; i < total; i += n_batch_ref) {
            const int32_t n_tokens_step = std::min(n_batch_ref, total - i);

            llama_batch batch_view = {
                n_tokens_step,
                batch_ref.token    + i,
                nullptr,
                batch_ref.pos      + i,
                batch_ref.n_seq_id + i,
                batch_ref.seq_id   + i,
                batch_ref.logits   + i,
            };

            const int ret = llama_decode(ctx, batch_view);
            if (ret != 0) {
                LOG_ERROR("llama_decode() failed during benchmark, n_batch=%d ret=%d", n_batch_ref, ret);
                return false;
            }

            if (synchronize) {
                llama_synchronize(ctx);
            }
        }

        return true;
    };

    // warm up like the CLI benchmark
    llama_batch_clear(&batch);
    const int warmup_tokens = std::min(16, n_kv_max);
    for (int i = 0; i < warmup_tokens; ++i) {
        llama_batch_add(&batch, get_token_rand(), i, {0}, i == warmup_tokens - 1);
    }
    if (!decode_helper(batch, n_batch, true)) {
        llama_batch_free(batch);
        endCompletion();
        return std::string("{}");
    }

    double acc_t_pp = 0.0;
    double acc_t_tg = 0.0;
    double acc_speed_pp = 0.0;
    double acc_speed_tg = 0.0;
    double acc_t_total = 0.0;
    double acc_speed_total = 0.0;

    int runs_completed = 0;

    for (int run = 0; run < nr && !is_interrupted; ++run) {
        bool run_failed = false;

        llama_batch_clear(&batch);

        const int prompt_sequences = is_pp_shared ? 1 : pl;
        for (int seq = 0; seq < prompt_sequences; ++seq) {
            for (int i = 0; i < pp; ++i) {
                llama_batch_add(&batch, get_token_rand(), i, {static_cast<llama_seq_id>(seq)}, i == pp - 1);
            }
        }

        llama_memory_clear(mem, false);

        const auto t_pp_start = lm_ggml_time_us();
        if (!decode_helper(batch, n_batch, false)) {
            run_failed = true;
            break;
        }

        llama_synchronize(ctx);
        const auto t_pp_end = lm_ggml_time_us();

        if (is_pp_shared && pl > 1) {
            for (int32_t seq = 1; seq < pl; ++seq) {
                llama_memory_seq_cp(mem, 0, seq, -1, -1);
            }

            if (!kv_unified) {
                llama_batch_clear(&batch);
                llama_batch_add(&batch, get_token_rand(), pp, {0}, true);
                if (!decode_helper(batch, n_batch, true)) {
                    run_failed = true;
                    break;
                }
                llama_memory_seq_rm(mem, 0, pp, -1);
            }
        }

        if (run_failed) {
            break;
        }

        const auto t_tg_start = lm_ggml_time_us();

        for (int i = 0; i < tg; ++i) {
            llama_batch_clear(&batch);

            for (int seq = 0; seq < pl; ++seq) {
                llama_batch_add(&batch, get_token_rand(), pp + i, {static_cast<llama_seq_id>(seq)}, true);
            }

            if (!decode_helper(batch, n_batch, true)) {
                run_failed = true;
                break;
            }
        }

        if (run_failed) {
            break;
        }

        const auto t_tg_end = lm_ggml_time_us();

        const double t_pp = (t_pp_end - t_pp_start) / 1e6;
        const double t_tg = (t_tg_end - t_tg_start) / 1e6;
        const double t_total = t_pp + t_tg;

        const double prompt_tokens = is_pp_shared ? static_cast<double>(pp) : static_cast<double>(pl * pp);
        const double generated_tokens = static_cast<double>(pl * tg);

        const double speed_pp = t_pp > 0.0 ? prompt_tokens / t_pp : 0.0;
        const double speed_tg = t_tg > 0.0 ? generated_tokens / t_tg : 0.0;
        const double speed_total = t_total > 0.0 ? (prompt_tokens + generated_tokens) / t_total : 0.0;

        acc_t_pp += t_pp;
        acc_t_tg += t_tg;
        acc_speed_pp += speed_pp;
        acc_speed_tg += speed_tg;
        acc_t_total += t_total;
        acc_speed_total += speed_total;

        ++runs_completed;
    }

    llama_memory_clear(mem, false);

    const double divisor = runs_completed > 0 ? static_cast<double>(runs_completed) : 1.0;

    json result_json = {
        {"n_kv_max", n_kv_max},
        {"n_batch", n_batch},
        {"n_ubatch", n_ubatch},
        {"flash_attn", flash_attn},
        {"is_pp_shared", is_pp_shared ? 1 : 0},
        {"n_gpu_layers", n_gpu_layers},
        {"n_threads", n_threads},
        {"n_threads_batch", n_threads_batch},
        {"pp", pp},
        {"tg", tg},
        {"pl", pl},
        {"n_kv", n_ctx_req},
        {"t_pp", acc_t_pp / divisor},
        {"speed_pp", acc_speed_pp / divisor},
        {"t_tg", acc_t_tg / divisor},
        {"speed_tg", acc_speed_tg / divisor},
        {"t", acc_t_total / divisor},
        {"speed", acc_speed_total / divisor}
    };

    llama_batch_free(batch);
    endCompletion();

    return result_json.dump();
}

void llama_rn_context_completion::processMedia(
    const std::string &prompt,
    const std::vector<std::string> &media_paths
) {
    if (!parent_ctx->isMultimodalEnabled()) {
        throw std::runtime_error("Multimodal is not enabled but image paths are provided");
    }

    // Delegate to the mtmd_wrapper method
    // For non-parallel mode, use the global bitmap_past_hashes from mtmd_wrapper
    parent_ctx->mtmd_wrapper->processMedia(
        parent_ctx->ctx,
        prompt,
        media_paths,
        parent_ctx->n_ctx,
        parent_ctx->params.n_batch,
        n_past,
        embd,
        context_full,
        ctx_sampling,
        parent_ctx->mtmd_wrapper->bitmap_past_hashes,
        0  // Use sequence ID 0 for non-parallel mode
    );
}

} // namespace rnllama
