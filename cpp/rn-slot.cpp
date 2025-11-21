#include "rn-slot.h"
#include "rn-completion.h"
#include "rn-llama.h"
#include "rn-common.hpp"
#include "chat.h"
#include <cstring>
#include <mutex>

namespace rnllama {

// Constructor
llama_rn_slot::llama_rn_slot() :
    id(-1),
    request_id(-1),
    state(SLOT_STATE_IDLE),
    task_type(SLOT_TASK_TYPE_COMPLETION),
    parent_ctx(nullptr),
    n_ctx(0),
    n_past(0),
    n_decoded(0),
    n_remaining(-1),
    i_batch(-1),
    embd_normalize(-1),
    num_prompt_tokens(0),
    num_tokens_predicted(0),
    incomplete(false),
    context_full(false),
    truncated(false),
    stopped_eos(false),
    stopped_word(false),
    stopped_limit(false),
    current_chat_format(0),
    current_reasoning_format(COMMON_REASONING_FORMAT_NONE),
    current_thinking_forced_open(false),
    ctx_sampling(nullptr),
    t_start_process(0),
    t_start_generation(0),
    t_last_used(0),
    n_prompt_tokens_cache(0),
    n_prompt_tokens_processed(0),
    t_prompt_processing(0.0),
    t_token_generation(0.0),
    is_interrupted(false),
    prompt_processing_finished(false),
    media_processed(false),
    rerank_current_index(0),
    save_state_size(-1)
{
}

// Destructor
llama_rn_slot::~llama_rn_slot() {
    if (ctx_sampling != nullptr) {
        common_sampler_free(ctx_sampling);
        ctx_sampling = nullptr;
    }
}

// Reset to IDLE state
void llama_rn_slot::reset() {
    state = SLOT_STATE_IDLE;
    request_id = -1;
    n_past = 0;
    n_decoded = 0;
    n_remaining = -1;
    i_batch = -1;

    // Clear token vectors
    prompt_tokens.clear();
    generated_tokens.clear();
    generated_text.clear();
    embd.clear();

    // Reset state fields
    prefill_text.clear();
    generated_token_probs.clear();
    num_prompt_tokens = 0;
    num_tokens_predicted = 0;
    incomplete = false;
    context_full = false;
    truncated = false;
    stopped_eos = false;
    stopped_word = false;
    stopped_limit = false;
    stopping_word.clear();
    stop_words.clear();
    error_message.clear();

    // Clear multimodal state
    bitmap_past_hashes.clear();
    media_paths.clear();
    prompt_text.clear();
    media_processed = false;

    // Reset chat parsing state
    current_chat_format = 0;
    current_reasoning_format = COMMON_REASONING_FORMAT_NONE;
    current_thinking_forced_open = false;

    // Reset flags
    is_interrupted = false;
    prompt_processing_finished = false;

    // Free sampling context
    if (ctx_sampling != nullptr) {
        common_sampler_free(ctx_sampling);
        ctx_sampling = nullptr;
    }

    // Clear callbacks
    on_token_callback = nullptr;
    on_complete_callback = nullptr;
    on_embedding_callback = nullptr;
    on_rerank_callback = nullptr;

    // Reset task-specific data
    task_type = SLOT_TASK_TYPE_COMPLETION;
    embd_normalize = -1;
    rerank_prompt_tokens.clear();
    rerank_scores.clear();
    rerank_current_index = 0;

    // Reset state management
    if (!load_state_path.empty() || !save_state_path.empty()) {
        LOG_VERBOSE("Slot %d: Clearing state paths (load=%s, save=%s)",
                   id,
                   load_state_path.empty() ? "none" : load_state_path.c_str(),
                   save_state_path.empty() ? "none" : save_state_path.c_str());
    }
    load_state_path.clear();
    save_state_path.clear();
    save_state_size = -1;

    // Reset timing fields
    t_start_process = 0;
    t_start_generation = 0;
    n_prompt_tokens_cache = 0;
    n_prompt_tokens_processed = 0;
    t_prompt_processing = 0.0;
    t_token_generation = 0.0;

    // Note: Keep cache_tokens for potential reuse
    // Note: Keep t_last_used for LRU tracking
}

// Load prompt tokens
void llama_rn_slot::load_prompt(const std::vector<llama_token>& tokens) {
    prompt_tokens = tokens;
    num_prompt_tokens = tokens.size();
    state = SLOT_STATE_PROCESSING_PROMPT;

    // Check if we have loaded state
    bool has_loaded_state = (!load_state_path.empty() && !cache_tokens.empty());

    if (has_loaded_state) {
        // Find how many tokens match between cached state and new prompt
        size_t n_matching = find_common_prefix_length(cache_tokens, tokens);

        if (n_matching > 0) {
            // We can reuse the KV cache for the matching prefix
            // Adjust n_past to the matching length
            LOG_INFO("Slot %d (req=%d): Reusing loaded state (%zu matching tokens from %zu cached, %zu prompt tokens)",
                     id, request_id, n_matching, cache_tokens.size(), tokens.size());

            // Set n_past to the matching prefix length
            // Any remaining tokens will be processed through build_batch
            n_past = n_matching;
            n_decoded = 0;

            // Track cached tokens for timing metrics
            n_prompt_tokens_cache = n_matching;

            // Clear KV cache beyond the matching prefix
            if (parent_ctx && parent_ctx->ctx) {
                auto * kv = llama_get_memory(parent_ctx->ctx);
                llama_memory_seq_rm(kv, id, n_matching, -1);
                LOG_VERBOSE("Slot %d: Cleared KV cache beyond position %zu", id, n_matching);
            }

            // Update cache_tokens to include the full prompt
            cache_tokens = tokens;
        } else {
            // No matching tokens, start fresh
            LOG_WARNING("Slot %d (req=%d): Loaded state doesn't match prompt (0 matching tokens), clearing cache",
                       id, request_id);

            n_past = 0;
            n_decoded = 0;
            n_prompt_tokens_cache = 0;

            // Clear KV cache for this slot's sequence
            if (parent_ctx && parent_ctx->ctx) {
                auto * kv = llama_get_memory(parent_ctx->ctx);
                llama_memory_seq_rm(kv, id, 0, parent_ctx->params.n_ctx);
                LOG_VERBOSE("Slot %d: Cleared KV cache for sequence (0 to %d)", id, parent_ctx->params.n_ctx);
            }

            cache_tokens = tokens;
        }
    } else {
        // No loaded state, start fresh
        n_past = 0;
        n_decoded = 0;
        n_prompt_tokens_cache = 0;

        // Clear KV cache for this slot's sequence to ensure clean state
        if (parent_ctx && parent_ctx->ctx) {
            auto * kv = llama_get_memory(parent_ctx->ctx);
            llama_memory_seq_rm(kv, id, 0, parent_ctx->params.n_ctx);
            LOG_VERBOSE("Slot %d: Cleared KV cache for sequence (0 to %d)", id, parent_ctx->params.n_ctx);
        }

        // Initialize cache_tokens with prompt tokens
        cache_tokens = tokens;
    }
}

// Check if there are generated tokens to retrieve
bool llama_rn_slot::has_next_token() const {
    return !generated_tokens.empty() && state != SLOT_STATE_IDLE;
}

// Get next generated token
completion_token_output llama_rn_slot::get_next_token() {
    if (generated_tokens.empty()) {
        completion_token_output empty_token;
        empty_token.tok = -1;
        return empty_token;
    }

    llama_token token = generated_tokens.front();
    generated_tokens.erase(generated_tokens.begin());

    completion_token_output output;
    output.tok = token;
    output.request_id = request_id;

    // Find matching probabilities if available
    for (const auto& token_prob : generated_token_probs) {
        if (token_prob.tok == token) {
            output.probs = token_prob.probs;
            break;
        }
    }

    return output;
}

// Parse chat output (tool calls, reasoning content, etc.)
completion_chat_output llama_rn_slot::parseChatOutput(bool is_partial) {
    common_chat_syntax syntax;
    syntax.format = static_cast<common_chat_format>(current_chat_format);
    syntax.reasoning_format = current_reasoning_format;  // Already the correct enum type
    syntax.thinking_forced_open = current_thinking_forced_open;
    syntax.parse_tool_calls = true;

    std::string full_text = prefill_text + generated_text;

    common_chat_msg parsed_msg = common_chat_parse(full_text, is_partial, syntax);

    completion_chat_output result;
    result.content = parsed_msg.content;
    result.reasoning_content = parsed_msg.reasoning_content;
    result.accumulated_text = full_text;
    result.tool_calls = parsed_msg.tool_calls;

    return result;
}

// Get timing information for this slot
slot_timings llama_rn_slot::get_timings() const {
    slot_timings timings;
    timings.cache_n = n_prompt_tokens_cache;

    timings.prompt_n = n_prompt_tokens_processed;
    timings.prompt_ms = t_prompt_processing * 1e3;  // Convert seconds to milliseconds for output
    if (n_prompt_tokens_processed > 0 && t_prompt_processing > 0.0) {
        timings.prompt_per_token_ms = (t_prompt_processing * 1e3) / n_prompt_tokens_processed;
        timings.prompt_per_second = n_prompt_tokens_processed / t_prompt_processing;
    }

    timings.predicted_n = n_decoded;
    timings.predicted_ms = t_token_generation * 1e3;  // Convert seconds to milliseconds for output
    if (n_decoded > 0 && t_token_generation > 0.0) {
        timings.predicted_per_token_ms = (t_token_generation * 1e3) / n_decoded;
        timings.predicted_per_second = n_decoded / t_token_generation;
    }

    return timings;
}

// Load state into this slot's sequence
bool llama_rn_slot::load_state() {
    if (!parent_ctx || !parent_ctx->ctx) {
        LOG_ERROR("Slot %d: Cannot load state - context not initialized", id);
        return false;
    }

#ifdef LM_GGML_USE_OPENCL
    const auto &model_devices = parent_ctx->llama_init.model->devices;
    auto has_opencl = false;
    for (const auto &dev : model_devices) {
        const char *dev_name = lm_ggml_backend_dev_name(dev);
        if (strncmp(dev_name, "GPUOpenCL", 9) == 0) {
            has_opencl = true;
        }
    }
    // TODO: Figure out how to handle this in a more elegant way
    if (has_opencl && !parent_ctx->params.kv_unified) {
        LOG_ERROR("Slot %d: Cannot load state - kv_unified is not enabled with OpenCL backend", id);
        return false;
    }
    if (has_opencl && parent_ctx->params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_DISABLED) {
        LOG_ERROR("Slot %d: Cannot load state - flash_attn_type is not disabled with OpenCL backend", id);
        return false;
    }
#endif

    if (load_state_path.empty()) {
        LOG_VERBOSE("Slot %d: No state path to load from", id);
        return true;  // Nothing to load is not an error
    }

    LOG_INFO("Slot %d: Loading state from: %s", id, load_state_path.c_str());

    // Start timing
    const int64_t t_load_start = lm_ggml_time_us();

    auto * kv = llama_get_memory(parent_ctx->ctx);
    llama_memory_seq_rm(kv, id, -1, -1);
    LOG_VERBOSE("Slot %d: Cleared ALL KV cache cells for sequence %d", id, id);

    // Get size needed for token output buffer
    size_t max_tokens = parent_ctx->params.n_ctx;
    std::vector<llama_token> state_tokens(max_tokens);
    size_t n_token_count_out = 0;

    // Load state from file into this slot's sequence
    if (!llama_state_seq_load_file(
        parent_ctx->ctx,
        load_state_path.c_str(),
        id,  // Use slot ID as sequence ID
        state_tokens.data(),
        state_tokens.size(),
        &n_token_count_out
    )) {
        LOG_ERROR("Slot %d: Failed to load state from file: %s", id, load_state_path.c_str());
        return false;
    }

    // Resize token vector to actual size
    state_tokens.resize(n_token_count_out);

    // Remove LLAMA_TOKEN_NULL tokens
    auto null_token_iter = std::find(state_tokens.begin(), state_tokens.end(), LLAMA_TOKEN_NULL);
    if (null_token_iter != state_tokens.end()) {
        state_tokens.resize(std::distance(state_tokens.begin(), null_token_iter));
    }

    // Update slot state
    embd = state_tokens;
    n_past = state_tokens.size();
    cache_tokens = state_tokens;

    // Remove any KV cache cells beyond n_past to ensure clean state
    // This is needed because the state file might contain more cells than tokens
    // (e.g., from a previous save that had more data)
    llama_memory_seq_rm(kv, id, n_past, -1);
    LOG_VERBOSE("Slot %d: Removed KV cache cells beyond position %d", id, n_past);

    // Calculate elapsed time
    const int64_t t_load_end = lm_ggml_time_us();
    const double t_load_ms = (t_load_end - t_load_start) / 1000.0;

    return true;
}

// Save state from this slot's sequence
bool llama_rn_slot::save_state() {
    if (!parent_ctx || !parent_ctx->ctx) {
        LOG_ERROR("Slot %d: Cannot save state - context not initialized", id);
        return false;
    }


#ifdef LM_GGML_USE_OPENCL
    const auto &model_devices = parent_ctx->llama_init.model->devices;
    auto has_opencl = false;
    for (const auto &dev : model_devices) {
        const char *dev_name = lm_ggml_backend_dev_name(dev);
        if (strncmp(dev_name, "GPUOpenCL", 9) == 0) {
            has_opencl = true;
        }
    }
    // TODO: Figure out how to handle this in a more elegant way
    if (has_opencl && !parent_ctx->params.kv_unified) {
        LOG_ERROR("Slot %d: Cannot save state - kv_unified is not enabled with OpenCL backend", id);
        return false;
    }
    if (has_opencl && parent_ctx->params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_DISABLED) {
        LOG_ERROR("Slot %d: Cannot save state - flash_attn_type is not disabled with OpenCL backend", id);
        return false;
    }
#endif

    if (save_state_path.empty()) {
        LOG_VERBOSE("Slot %d: No state path to save to", id);
        return true;  // Not specified is not an error
    }

    LOG_INFO("Slot %d: Saving state to: %s", id, save_state_path.c_str());

    // Start timing
    const int64_t t_save_start = lm_ggml_time_us();

    // Get tokens for this state (cache_tokens represents all processed tokens in KV cache)
    std::vector<llama_token> state_tokens = cache_tokens;

    // Remove LLAMA_TOKEN_NULL tokens
    auto null_token_iter = std::find(state_tokens.begin(), state_tokens.end(), LLAMA_TOKEN_NULL);
    if (null_token_iter != state_tokens.end()) {
        state_tokens.resize(std::distance(state_tokens.begin(), null_token_iter));
    }

    if (state_tokens.empty()) {
        LOG_WARNING("Slot %d: No tokens to save for state", id);
        return false;
    }

    // Determine how many tokens to save
    size_t default_size = state_tokens.size();
    size_t actual_save_size = default_size;

    if (save_state_size > 0 && (size_t)save_state_size <= default_size) {
        actual_save_size = save_state_size;
    }

    // Save with size - 1 to force re-processing of last token when loading
    // This ensures fresh logits are generated for sampling
    // Only do this if we have more than 1 token to save
    if (actual_save_size > 1) {
        actual_save_size--;
        LOG_VERBOSE("Slot %d: Saving %zu tokens (reduced by 1 for logits regeneration)",
                   id, actual_save_size);
    }

    // Save state to file
    if (!llama_state_seq_save_file(
        parent_ctx->ctx,
        save_state_path.c_str(),
        id,  // Use slot ID as sequence ID
        state_tokens.data(),
        actual_save_size
    )) {
        LOG_ERROR("Slot %d: Failed to save state to file: %s", id, save_state_path.c_str());
        return false;
    }

    // Calculate elapsed time
    const int64_t t_save_end = lm_ggml_time_us();
    const double t_save_ms = (t_save_end - t_save_start) / 1000.0;

    return true;
}

} // namespace rnllama
