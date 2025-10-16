#include "rn-slot.h"
#include "rn-completion.h"
#include "rn-llama.h"
#include "chat.h"
#include <cstring>

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
    is_interrupted(false),
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
    load_state_path.clear();
    save_state_path.clear();
    save_state_size = -1;

    // Note: Keep cache_tokens for potential reuse
    // Note: Keep t_last_used for LRU tracking
}

// Load prompt tokens
void llama_rn_slot::load_prompt(const std::vector<llama_token>& tokens) {
    prompt_tokens = tokens;
    num_prompt_tokens = tokens.size();
    state = SLOT_STATE_PROCESSING_PROMPT;

    // Check if we have loaded state and if the prompt matches
    bool has_loaded_state = (!load_state_path.empty() && !cache_tokens.empty());
    bool prompt_matches = false;

    if (has_loaded_state && tokens.size() <= cache_tokens.size()) {
        // Check if the new prompt tokens match the beginning of the loaded state tokens
        prompt_matches = std::equal(tokens.begin(), tokens.end(), cache_tokens.begin());
    }

    if (has_loaded_state && prompt_matches) {
        // Reusing loaded state: keep KV cache and n_past
        // Note: State is saved with size-1, so the last token needs to be processed
        // This ensures we get fresh logits for sampling
        LOG_INFO("Slot %d: Reusing loaded state (%d cached tokens, %d prompt tokens)",
                 id, (int)cache_tokens.size(), (int)tokens.size());
        // n_past is already set by load_state, keep it
        // The last prompt token will be processed through build_batch to generate logits
        n_decoded = 0;
    } else {
        // Starting fresh: clear KV cache and reset state
        if (has_loaded_state) {
            LOG_WARNING("Slot %d: Loaded state doesn't match prompt (%zu cached vs %zu prompt tokens), clearing cache",
                       id, cache_tokens.size(), tokens.size());
        }

        n_past = 0;
        n_decoded = 0;

        // Clear KV cache for this slot's sequence to ensure clean state
        // This is crucial when reusing slots for different requests
        if (parent_ctx && parent_ctx->ctx) {
            auto * kv = llama_get_memory(parent_ctx->ctx);
            // Use n_ctx as the end position to ensure all positions are cleared
            llama_memory_seq_rm(kv, id, 0, parent_ctx->params.n_ctx);
            LOG_VERBOSE("Slot %d: Cleared KV cache for sequence (0 to %d)", id, parent_ctx->params.n_ctx);
        }

        // Initialize cache_tokens with prompt tokens
        // This will be extended with generated tokens during completion
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

// Load state into this slot's sequence
bool llama_rn_slot::load_state() {
    if (!parent_ctx || !parent_ctx->ctx) {
        LOG_ERROR("Slot %d: Cannot load state - context not initialized", id);
        return false;
    }

    if (load_state_path.empty()) {
        LOG_VERBOSE("Slot %d: No state path to load from", id);
        return true;  // Nothing to load is not an error
    }

    LOG_INFO("Slot %d: Loading state from: %s", id, load_state_path.c_str());

    // Start timing
    const int64_t t_load_start = lm_ggml_time_us();

    // Clear existing KV cache for this sequence before loading
    // This ensures we start fresh with only the loaded state
    auto * kv = llama_get_memory(parent_ctx->ctx);
    // Use -1 for both positions to clear all cells for this sequence
    llama_memory_seq_rm(kv, id, -1, -1);
    LOG_VERBOSE("Slot %d: Cleared existing KV cache before loading state", id);

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

    LOG_INFO("Slot %d: State loaded successfully from %s, n_past=%d, tokens=%zu, time=%.2f ms",
             id, load_state_path.c_str(), n_past, state_tokens.size(), t_load_ms);

    return true;
}

// Save state from this slot's sequence
bool llama_rn_slot::save_state() {
    if (!parent_ctx || !parent_ctx->ctx) {
        LOG_ERROR("Slot %d: Cannot save state - context not initialized", id);
        return false;
    }

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

    LOG_INFO("Slot %d: State saved successfully to %s (%zu tokens of %zu total), time=%.2f ms",
             id, save_state_path.c_str(), actual_save_size, default_size, t_save_ms);

    return true;
}

} // namespace rnllama
