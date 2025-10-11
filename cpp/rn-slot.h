#ifndef RN_SLOT_H
#define RN_SLOT_H

#include "common.h"
#include "llama.h"
#include "sampling.h"
#include <vector>
#include <string>
#include <functional>

namespace rnllama {

// Forward declarations
struct llama_rn_context;
struct completion_token_output;
struct completion_chat_output;

// Slot states
enum llama_rn_slot_state {
    SLOT_STATE_IDLE = 0,           // Available for new requests
    SLOT_STATE_PROCESSING_PROMPT,  // Processing input tokens
    SLOT_STATE_GENERATING,         // Generating output tokens
    SLOT_STATE_DONE                // Completed (ready for cleanup)
};

// Per-slot completion state
struct llama_rn_slot {
    // Slot identification
    int32_t id;                    // Slot index (0 to n_parallel-1)
    int32_t request_id;            // Unique request identifier
    llama_rn_slot_state state;

    // Context management
    llama_rn_context* parent_ctx;  // Parent context reference
    int32_t n_ctx;                 // Context size for this slot
    llama_pos n_past;              // Number of tokens processed
    int32_t n_decoded;             // Tokens generated so far
    int32_t n_remaining;           // Tokens left to generate (-1 = unlimited)
    int32_t i_batch;               // Position in current batch

    // Token management
    std::vector<llama_token> prompt_tokens;
    std::vector<llama_token> cache_tokens;  // For KV cache reuse
    std::vector<llama_token> generated_tokens;
    std::string generated_text;

    // Completion state (migrated from llama_rn_context_completion)
    std::string prefill_text;
    std::vector<completion_token_output> generated_token_probs;
    size_t num_prompt_tokens;
    size_t num_tokens_predicted;
    std::vector<llama_token> embd;
    bool incomplete;
    bool context_full;
    bool truncated;
    bool stopped_eos;
    bool stopped_word;
    bool stopped_limit;
    std::string stopping_word;
    std::vector<std::string> stop_words;  // Stop words for this slot

    // Chat parsing state
    int current_chat_format;
    common_reasoning_format current_reasoning_format;
    bool current_thinking_forced_open;

    // Sampling context (per-slot)
    common_sampler* ctx_sampling;

    // Timing
    int64_t t_start_process;
    int64_t t_start_generation;
    int64_t t_last_used;

    // Cancellation flag (per-slot)
    bool is_interrupted;

    // Token callback (per-slot)
    std::function<void(const completion_token_output&)> on_token_callback;

    // Completion callback (per-slot)
    std::function<void(llama_rn_slot*)> on_complete_callback;

    // Constructor
    llama_rn_slot();

    // Destructor
    ~llama_rn_slot();

    // Methods
    void reset();                          // Reset to IDLE state
    void load_prompt(const std::vector<llama_token>& tokens);
    bool has_next_token() const;
    completion_token_output get_next_token();
    completion_chat_output parseChatOutput(bool is_partial);
};

} // namespace rnllama

#endif /* RN_SLOT_H */