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
    rerank_current_index(0)
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

} // namespace rnllama
