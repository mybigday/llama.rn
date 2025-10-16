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

    // Reset session state management
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
    n_past = 0;
    n_decoded = 0;

    // Clear KV cache for this slot's sequence to ensure clean state
    // This is crucial when reusing slots for different requests
    if (parent_ctx && parent_ctx->ctx) {
        auto * kv = llama_get_memory(parent_ctx->ctx);
        llama_memory_seq_rm(kv, id, 0, -1);
        LOG_VERBOSE("Slot %d: Cleared KV cache for sequence", id);
    }

    // Note: For now, we don't reuse cache across different requests in the same slot
    // This ensures correctness at the cost of some performance
    // Future optimization: implement smart cache reuse with proper validation
    cache_tokens.clear();
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

// Load session state into this slot's sequence
bool llama_rn_slot::load_session_state() {
    if (!parent_ctx || !parent_ctx->ctx) {
        LOG_ERROR("Slot %d: Cannot load session state - context not initialized", id);
        return false;
    }

    if (load_state_path.empty()) {
        LOG_VERBOSE("Slot %d: No session state path to load from", id);
        return true;  // Nothing to load is not an error
    }

    LOG_INFO("Slot %d: Loading session state from: %s", id, load_state_path.c_str());

    // Get size needed for token output buffer
    size_t max_tokens = parent_ctx->params.n_ctx;
    std::vector<llama_token> session_tokens(max_tokens);
    size_t n_token_count_out = 0;

    // Load state from file into this slot's sequence
    if (!llama_state_seq_load_file(
        parent_ctx->ctx,
        load_state_path.c_str(),
        id,  // Use slot ID as sequence ID
        session_tokens.data(),
        session_tokens.size(),
        &n_token_count_out
    )) {
        LOG_ERROR("Slot %d: Failed to load session state from file: %s", id, load_state_path.c_str());
        return false;
    }

    // Resize token vector to actual size
    session_tokens.resize(n_token_count_out);

    // Remove LLAMA_TOKEN_NULL tokens
    auto null_token_iter = std::find(session_tokens.begin(), session_tokens.end(), LLAMA_TOKEN_NULL);
    if (null_token_iter != session_tokens.end()) {
        session_tokens.resize(std::distance(session_tokens.begin(), null_token_iter));
    }

    // Update slot state
    embd = session_tokens;
    n_past = session_tokens.size();
    cache_tokens = session_tokens;

    LOG_INFO("Slot %d: Session state loaded successfully from %s, n_past=%d, tokens=%zu",
             id, load_state_path.c_str(), n_past, session_tokens.size());

    return true;
}

// Save session state from this slot's sequence
bool llama_rn_slot::save_session_state() {
    if (!parent_ctx || !parent_ctx->ctx) {
        LOG_ERROR("Slot %d: Cannot save session state - context not initialized", id);
        return false;
    }

    if (save_state_path.empty()) {
        LOG_VERBOSE("Slot %d: No session state path to save to", id);
        return true;  // Not specified is not an error
    }

    LOG_INFO("Slot %d: Saving session state to: %s", id, save_state_path.c_str());

    // Get tokens for this session (from embd or cache_tokens)
    std::vector<llama_token> session_tokens = embd;
    if (session_tokens.empty()) {
        session_tokens = cache_tokens;
    }

    // Remove LLAMA_TOKEN_NULL tokens
    auto null_token_iter = std::find(session_tokens.begin(), session_tokens.end(), LLAMA_TOKEN_NULL);
    if (null_token_iter != session_tokens.end()) {
        session_tokens.resize(std::distance(session_tokens.begin(), null_token_iter));
    }

    if (session_tokens.empty()) {
        LOG_WARNING("Slot %d: No tokens to save for session", id);
        return false;
    }

    // Determine how many tokens to save
    size_t default_size = session_tokens.size();
    size_t actual_save_size = default_size;

    if (save_state_size > 0 && (size_t)save_state_size <= default_size) {
        actual_save_size = save_state_size;
    }

    // Save state to file
    if (!llama_state_seq_save_file(
        parent_ctx->ctx,
        save_state_path.c_str(),
        id,  // Use slot ID as sequence ID
        session_tokens.data(),
        actual_save_size
    )) {
        LOG_ERROR("Slot %d: Failed to save session state to file: %s", id, save_state_path.c_str());
        return false;
    }

    LOG_INFO("Slot %d: Session state saved successfully to %s (%zu tokens of %zu total)",
             id, save_state_path.c_str(), actual_save_size, default_size);

    return true;
}

} // namespace rnllama
