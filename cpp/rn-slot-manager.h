#ifndef RN_SLOT_MANAGER_H
#define RN_SLOT_MANAGER_H

#include "rn-slot.h"
#include "common.h"
#include "llama.h"
#include <vector>
#include <deque>
#include <map>
#include <functional>

namespace rnllama {

// Forward declarations
struct llama_rn_context;
struct completion_token_output;

// Queued request structure
struct llama_rn_queued_request {
    int32_t request_id;
    common_params params;
    std::vector<llama_token> prompt_tokens;
    std::function<void(const completion_token_output&)> on_token;
    std::function<void(llama_rn_slot*)> on_complete;

    // Media paths for multimodal
    std::vector<std::string> media_paths;

    // Chat format parameters
    int chat_format;
    common_reasoning_format reasoning_format;
    bool thinking_forced_open;

    // Prefill text
    std::string prefill_text;

    llama_rn_queued_request() :
        request_id(-1),
        chat_format(0),
        reasoning_format(COMMON_REASONING_FORMAT_NONE),
        thinking_forced_open(false)
    {}
};

// Slot manager for parallel decoding
struct llama_rn_slot_manager {
    // Parent context reference
    llama_rn_context* parent_ctx;

    // Slot pool
    std::vector<llama_rn_slot> slots;
    int32_t n_parallel;                    // Number of parallel slots

    // Request queue
    std::deque<llama_rn_queued_request> queue_requests;

    // Request tracking
    std::map<int32_t, llama_rn_slot*> active_requests;  // request_id -> slot
    int32_t next_request_id;

    // Batch processing
    llama_batch batch;
    int32_t n_batch;                       // Max batch size

    // Configuration
    float slot_prompt_similarity;          // Threshold for cache reuse (0.0-1.0)
    bool continuous_batching;              // Allow mixing prompt/generation

    // Constructor
    llama_rn_slot_manager(llama_rn_context* ctx);

    // Destructor
    ~llama_rn_slot_manager();

    // Initialization
    bool init(int32_t n_parallel, int32_t n_batch, int32_t n_ctx);

    // Request management
    int32_t queue_request(
        const common_params& params,
        const std::vector<llama_token>& prompt,
        const std::vector<std::string>& media_paths,
        int chat_format,
        common_reasoning_format reasoning_format,
        bool thinking_forced_open,
        const std::string& prefill_text,
        std::function<void(const completion_token_output&)> on_token,
        std::function<void(llama_rn_slot*)> on_complete
    );

    // Slot management
    llama_rn_slot* get_available_slot(const std::vector<llama_token>& prompt);
    llama_rn_slot* get_slot_by_request_id(int32_t request_id);
    void release_slot(llama_rn_slot* slot);
    void cancel_request(int32_t request_id);

    // Main processing loop
    void update_slots();

    // Helper methods
    float compute_similarity(const std::vector<llama_token>& a,
                            const std::vector<llama_token>& b);
    void build_batch();
    bool process_batch();
    void sample_and_callback();

    // Process pending queue
    void process_pending_queue();

    // Release completed slots
    void release_completed_slots();
};

} // namespace rnllama

#endif /* RN_SLOT_MANAGER_H */