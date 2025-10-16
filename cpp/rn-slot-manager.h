#ifndef RN_SLOT_MANAGER_H
#define RN_SLOT_MANAGER_H

#include "rn-slot.h"
#include "common.h"
#include "llama.h"
#include <vector>
#include <deque>
#include <map>
#include <functional>
#include <mutex>
#include <thread>
#include <atomic>
#include <condition_variable>

namespace rnllama {

// Forward declarations
struct llama_rn_context;
struct completion_token_output;

// Queued request structure
struct llama_rn_queued_request {
    int32_t request_id;
    llama_rn_slot_task_type task_type;
    common_params params;
    std::vector<llama_token> prompt_tokens;
    std::function<void(const completion_token_output&)> on_token;
    std::function<void(llama_rn_slot*)> on_complete;

    // Media paths for multimodal
    std::vector<std::string> media_paths;
    std::string prompt_text;  // Original prompt text (needed for media processing)

    // Chat format parameters
    int chat_format;
    common_reasoning_format reasoning_format;
    bool thinking_forced_open;

    // Prefill text
    std::string prefill_text;

    // Embedding parameters
    int embd_normalize;
    std::function<void(int32_t, const std::vector<float>&)> on_embedding;

    // Rerank parameters
    std::vector<std::vector<llama_token>> rerank_prompt_tokens;
    std::function<void(int32_t, const std::vector<float>&)> on_rerank;

    // State management
    std::string load_state_path;       // File path to load state from before processing
    std::string save_state_path;       // File path to save state to after completion
    int32_t save_state_size;           // Number of tokens to save (0 or -1 = all tokens)

    llama_rn_queued_request() :
        request_id(-1),
        task_type(SLOT_TASK_TYPE_COMPLETION),
        chat_format(0),
        reasoning_format(COMMON_REASONING_FORMAT_NONE),
        thinking_forced_open(false),
        embd_normalize(-1),
        save_state_size(-1)
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

    // Processing loop control
    std::mutex slots_mutex;                // Mutex for thread-safe access to slots
    std::condition_variable slots_cv;      // Condition variable for efficient waiting
    std::thread processing_thread;         // Background processing thread
    std::atomic<bool> processing_active;   // Flag to control processing loop

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
        const std::string& prompt_text,
        int chat_format,
        common_reasoning_format reasoning_format,
        bool thinking_forced_open,
        const std::string& prefill_text,
        const std::string& load_state_path,
        const std::string& save_state_path,
        int32_t save_state_size,
        std::function<void(const completion_token_output&)> on_token,
        std::function<void(llama_rn_slot*)> on_complete
    );

    int32_t queue_embedding_request(
        const std::vector<llama_token>& tokens,
        int embd_normalize,
        std::function<void(int32_t, const std::vector<float>&)> on_result
    );

    int32_t queue_rerank_request(
        const std::string& query,
        const std::vector<std::string>& documents,
        int normalize,
        std::function<void(int32_t, const std::vector<float>&)> on_results
    );

    // Slot management
    llama_rn_slot* get_available_slot(const std::vector<llama_token>& prompt);
    llama_rn_slot* get_slot_by_request_id(int32_t request_id);
    void release_slot(llama_rn_slot* slot);
    void cancel_request(int32_t request_id);

    // Processing loop management
    void start_processing_loop();
    void stop_processing_loop();

    // Main processing loop (protected by mutex)
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
