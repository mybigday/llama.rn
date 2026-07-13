#ifndef RN_COMPLETION_H
#define RN_COMPLETION_H

#include "common.h"
#include "llama.h"
#include "rn-llama.h"
#include "sampling.h"
#include "nlohmann/json.hpp"
#include "chat.h"
#include "speculative.h"
#include <deque>

using json = nlohmann::ordered_json;

namespace rnllama {

// Utility functions
static inline void llama_batch_clear(llama_batch *batch) {
    batch->n_tokens = 0;
}

// Forward declarations
struct llama_rn_context;

// A snapshot of the non-rollbackable memory state (recurrent/SWA cells,
// PARTIAL_ONLY) at a token boundary.
struct rn_state_checkpoint {
    // Exact token sequence the snapshot represents (memory positions [0, n)).
    std::vector<llama_token> tokens;
    // PARTIAL_ONLY per-sequence state blob for seq_id 0.
    std::vector<uint8_t> data;

    size_t n_tokens() const { return tokens.size(); }
    size_t size_bytes() const {
        return data.size() + tokens.size() * sizeof(llama_token);
    }
};

// Types defined in rn-llama.h (needed here for compilation)
enum stop_type
{
    STOP_FULL,
    STOP_PARTIAL,
};

struct completion_token_output
{
    struct token_prob
    {
        llama_token tok;
        float prob;
    };

    std::vector<token_prob> probs;
    llama_token tok;
    std::string text;  // Token text (decoded)
    int32_t request_id = -1;  // Request ID for parallel processing
};

struct completion_chat_output
{
  std::string content;
  std::string reasoning_content;
  std::vector<common_chat_tool_call> tool_calls;
  std::string accumulated_text;
};

// Completion context class
struct llama_rn_context_completion {
    // Reference to parent context
    llama_rn_context* parent_ctx;

    // Completion state fields
    bool is_predicting = false;
    bool is_interrupted = false;
    bool has_next_token = false;
    std::string prefill_text;
    std::string generated_text;
    utf8_stream_gate utf8_gate;
    std::vector<completion_token_output> generated_token_probs;
    size_t num_draft_tokens = 0;
    size_t num_draft_tokens_accepted = 0;
    size_t num_prompt_tokens = 0;
    size_t num_tokens_predicted = 0;
    int64_t t_start_generation = 0;
    double t_token_generation = 0.0;
    llama_pos n_past = 0;
    size_t n_remain = 0;
    std::vector<llama_token> embd;

    // --- Prompt state cache (recurrent / hybrid / SWA prefix reuse) ----------
    // Saved snapshots, oldest first; empty for pure-attention models (seq_rm
    // already reuses the prefix for free).
    std::vector<rn_state_checkpoint> state_checkpoints;
    bool state_cache_enabled = false;   // set once, from the model architecture
    bool state_cache_probed = false;    // whether we've inspected the model yet
    bool prompt_checkpoint_pending = false; // prompt-region snapshots armed for this ingest
    // embedding()/rerank() run throwaway prompts through this path; they must
    // not capture into the chat's cache. Set per loadPrompt.
    bool state_cache_capture_allowed = true;
    // Bounds, copied from the host-set config at probeStateCache() time. The
    // byte budget is the real limiter (snapshots are tens of KiB to ~22 MiB
    // per model, fixed in the token count).
    size_t state_cache_max_checkpoints = 8;
    size_t state_cache_budget_bytes = (size_t) 160 * 1024 * 1024;
    // Minimum spacing between message-boundary restore points on a COLD ingest:
    // a boundary less than this far past the previous one saves less reprocess
    // than a full-state snapshot costs (see computeMessageBoundaries).
    llama_pos state_ckpt_min_gap = 64;
    // Fallback snapshot interval during a COLD prompt ingest (0 disables); used
    // only when the prompt exposes no message boundaries (base models, multimodal).
    size_t state_ckpt_prefill_interval = 256;
    // Message-boundary snapshot positions for the current prompt, ascending
    // (see computeMessageBoundaries). Computed per loadPrompt.
    std::vector<llama_pos> boundary_ckpts;
    bool incomplete = false;
    bool context_full = false;
    bool truncated = false;
    bool stopped_eos = false;
    bool stopped_word = false;
    bool stopped_limit = false;
    std::string stopping_word;
    // Current completion parameters for chat parsing
    int current_chat_format = COMMON_CHAT_FORMAT_CONTENT_ONLY;
    common_reasoning_format current_reasoning_format = COMMON_REASONING_FORMAT_NONE;
    std::string current_generation_prompt;
    std::string current_chat_parser;  // Serialized PEG parser for chat output parsing

    // Sampling context
    common_sampler *ctx_sampling = nullptr;

    // Speculative decoding context for MTP.
    common_speculative *spec = nullptr;
    llama_context_ptr spec_ctx;
    llama_batch spec_batch = {};
    bool spec_batch_initialized = false;
    llama_tokens spec_prompt;
    llama_token spec_id_last = LLAMA_TOKEN_NULL;
    llama_pos spec_n_past = 0;
    llama_tokens spec_draft;
    std::deque<completion_token_output> spec_pending_tokens;
    // Number of prompt tokens the last MTP prompt eval actually decoded (vs.
    // reused from the cache). Instrumentation for the reuse tests.
    size_t mtp_prompt_reprocessed = 0;
    // Set once a mem-shared MTP draft (ctx_other == target) is detected: the
    // checkpoint cache is disabled for it, since upstream state save/restore is
    // a no-op on shared-cell caches (TAG_KV_CACHE_SHARE_CELLS). Test-visible.
    bool mtp_draft_mem_shared = false;

    // Constructor
    llama_rn_context_completion(llama_rn_context* parent);

    // Destructor
    ~llama_rn_context_completion();

    // Completion processing methods
    void rewind();
    bool initSampling();

    // Prompt state cache helpers (see rn_state_checkpoint).
    void probeStateCache();                       // detect whether the model needs it
    void captureStateCheckpoint();                // snapshot memory at current n_past (embd)
    // Snapshot at position n, tagged with seq[0, n) (MTP path uses spec_prompt).
    void captureStateCheckpoint(const std::vector<llama_token> &seq, size_t n);
    int  findStateCheckpoint(const std::vector<llama_token> &target, size_t max_len) const;
    bool restoreStateCheckpoint(size_t index);    // restore snapshot into seq 0
    // Restore the longest snapshot prefixing `target` (length <= max_reuse,
    // < total_tokens) and truncate the live prefix to it. Sets n_past_out.
    bool recoverStateCheckpoint(const std::vector<llama_token> &target, size_t max_reuse,
                                size_t total_tokens, llama_pos &n_past_out);
    void evictStateCheckpoints();                 // enforce count / byte bounds
    void clearStateCheckpoints();                 // drop all snapshots
    void eraseStateCheckpointAt(size_t n_tokens); // drop the snapshot at a boundary
    void truncatePrompt(std::vector<llama_token> &prompt_tokens);
    void loadPrompt(const std::vector<std::string> &media_paths, bool allow_state_cache = true);
    // Ascending message-boundary positions: the first content token after each
    // run of chat-template delimiter tokens (CONTROL/USER_DEFINED). Boundaries
    // closer than min_gap to the previous one are dropped (min_gap == 1 keeps
    // all — used to recover the last boundary as the cold-ingest frontier).
    std::vector<llama_pos> computeMessageBoundaries(const std::vector<llama_token> &tokens,
                                                    llama_pos min_gap) const;
    void beginCompletion();
    void beginCompletion(int chat_format, common_reasoning_format reasoning_format, const std::string &generation_prompt = "", const std::string &chat_parser = "");
    void endCompletion();
    void resetGenerationTimings();
    void startGenerationTiming();
    void updateGenerationTiming();
    completion_token_output nextToken();
    bool shouldUseMTP() const;
    void resetSpeculative();
    void initMTP();
    void evalMTPPrompt();
    bool refillMTPTokens();
    completion_token_output nextTokenMTP();
    size_t findStoppingStrings(const std::string &text, const size_t last_token_size, const stop_type type);
    completion_token_output doCompletion();
    completion_chat_output parseChatOutput(bool is_partial);

    // Embedding methods
    std::vector<float> embedding(common_params &embd_params);
    std::vector<float> rerank(const std::string &query, const std::vector<std::string> &documents);

    // Benchmarking methods
    std::string bench(int pp, int tg, int pl, int nr);

    // Multimodal processing methods
    void processMedia(
      const std::string &prompt,
      const std::vector<std::string> &media_paths
    );
};

} // namespace rnllama

#endif /* RN_COMPLETION_H */
