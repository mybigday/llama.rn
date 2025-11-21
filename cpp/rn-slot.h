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

// Slot timings result
struct slot_timings {
    int32_t cache_n = -1;                  // Number of prompt tokens from cache

    int32_t prompt_n = -1;                 // Number of prompt tokens processed
    double prompt_ms = 0.0;                // Total time for prompt processing (ms)
    double prompt_per_token_ms = 0.0;      // Time per prompt token (ms)
    double prompt_per_second = 0.0;        // Tokens per second for prompt processing

    int32_t predicted_n = -1;              // Number of tokens generated
    double predicted_ms = 0.0;             // Total time for token generation (ms)
    double predicted_per_token_ms = 0.0;   // Time per generated token (ms)
    double predicted_per_second = 0.0;     // Tokens per second for generation
};

// Slot task types
enum llama_rn_slot_task_type {
    SLOT_TASK_TYPE_COMPLETION = 0,
    SLOT_TASK_TYPE_EMBEDDING,
    SLOT_TASK_TYPE_RERANK,
};

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
    llama_rn_slot_task_type task_type; // Current task type assigned to slot

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

    // Multimodal state (per-slot)
    std::vector<std::string> bitmap_past_hashes;  // For multimodal KV cache reuse
    std::vector<std::string> media_paths;         // Media paths for deferred processing
    std::string prompt_text;                      // Original prompt text for media processing
    bool media_processed;                         // Flag indicating if media has been processed

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
    std::string error_message;             // Error message if completion failed

    // Chat parsing state
    int current_chat_format;
    common_reasoning_format current_reasoning_format;
    bool current_thinking_forced_open;

    // Sampling context (per-slot)
    common_params* params;
    common_sampler* ctx_sampling;

    // Timing
    int64_t t_start_process;       // Start time for processing (us)
    int64_t t_start_generation;    // Start time for generation (us)
    int64_t t_last_used;           // Last time slot was used (us)

    // Timing metrics
    int32_t n_prompt_tokens_cache;     // Number of prompt tokens from cache
    int32_t n_prompt_tokens_processed; // Number of prompt tokens processed
    double t_prompt_processing;        // Time for prompt processing (seconds)
    double t_token_generation;         // Time for token generation (seconds)

    // Cancellation flag (per-slot)
    bool is_interrupted;

    // Flag to indicate prompt processing just finished (timing needs to be calculated after decode)
    bool prompt_processing_finished;

    // Token callback (per-slot)
    std::function<void(const completion_token_output&)> on_token_callback;

    // Completion callback (per-slot)
    std::function<void(llama_rn_slot*)> on_complete_callback;

    // Embedding task state
    int embd_normalize;
    std::function<void(int32_t, const std::vector<float>&)> on_embedding_callback;

    // Rerank task state
    std::function<void(int32_t, const std::vector<float>&)> on_rerank_callback;
    std::vector<std::vector<llama_token>> rerank_prompt_tokens;
    std::vector<float> rerank_scores;
    size_t rerank_current_index;

    // State management (per-slot)
    std::string load_state_path;      // Path to load state from before processing
    std::string save_state_path;      // Path to save state to after completion
    int32_t save_state_size;          // Number of tokens to save (0 or -1 = all tokens)

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

    // Timing methods
    slot_timings get_timings() const;      // Get timing information for this slot

    // State methods
    bool load_state();             // Load state into this slot's sequence
    bool save_state();             // Save state from this slot's sequence
};

} // namespace rnllama

#endif /* RN_SLOT_H */
