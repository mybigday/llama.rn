#include "rn-slot-manager.h"
#include "rn-completion.h"
#include "rn-llama.h"
#include "rn-mtmd.hpp"
#include "rn-common.hpp"
#include "ggml.h"
#include <algorithm>
#include <cstring>

namespace rnllama {

// Constructor
llama_rn_slot_manager::llama_rn_slot_manager(llama_rn_context* ctx) :
    parent_ctx(ctx),
    n_parallel(1),
    next_request_id(1),
    n_batch(512),
    slot_prompt_similarity(0.5f),
    continuous_batching(false),
    processing_active(false)
{
    // Initialize batch to zero/null - will be properly allocated later
    std::memset(&batch, 0, sizeof(batch));
}

// Destructor
llama_rn_slot_manager::~llama_rn_slot_manager() {
    // Stop processing loop if active
    stop_processing_loop();

    // Free batch
    if (batch.token != nullptr) {
        llama_batch_free(batch);
    }

    // Slots will be freed automatically by vector destructor
}

// Initialize slot manager
bool llama_rn_slot_manager::init(int32_t n_parallel_, int32_t n_batch_, int32_t n_ctx) {
    n_parallel = n_parallel_;
    n_batch = n_batch_;

    LOG_INFO("Initializing slot manager with %d parallel slots, batch size %d", n_parallel, n_batch);

    // Allocate slots
    slots.resize(n_parallel);

    // Initialize each slot
    int32_t n_ctx_per_slot = n_ctx / n_parallel;
    for (int32_t i = 0; i < n_parallel; i++) {
        slots[i].id = i;
        slots[i].parent_ctx = parent_ctx;
        slots[i].n_ctx = n_ctx_per_slot;
        slots[i].state = SLOT_STATE_IDLE;
        slots[i].request_id = -1;
        LOG_VERBOSE("Slot %d initialized with context size %d", i, n_ctx_per_slot);
    }

    // Allocate batch
    batch = llama_batch_init(n_batch, 0, n_parallel);
    if (batch.token == nullptr) {
        LOG_ERROR("Failed to allocate batch");
        return false;
    }

    LOG_INFO("Slot manager initialized successfully");
    return true;
}

// Queue a new request
int32_t llama_rn_slot_manager::queue_request(
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
) {
    // Generate unique request ID
    int32_t request_id = next_request_id++;

    LOG_INFO("Queuing request %d with %zu prompt tokens (load_state=%s, save_state=%s, save_size=%d)",
             request_id, prompt.size(),
             load_state_path.empty() ? "no" : load_state_path.c_str(),
             save_state_path.empty() ? "no" : save_state_path.c_str(),
             save_state_size);

    // Create queued request
    llama_rn_queued_request request;
    request.request_id = request_id;
    request.task_type = SLOT_TASK_TYPE_COMPLETION;
    request.params = params;
    request.prompt_tokens = prompt;
    request.media_paths = media_paths;
    request.prompt_text = prompt_text;
    request.chat_format = chat_format;
    request.reasoning_format = reasoning_format;
    request.thinking_forced_open = thinking_forced_open;
    request.prefill_text = prefill_text;
    request.load_state_path = load_state_path;
    request.save_state_path = save_state_path;
    request.save_state_size = save_state_size;
    request.on_token = on_token;
    request.on_complete = on_complete;

    // Add to queue
    {
        std::lock_guard<std::mutex> lock(slots_mutex);
        queue_requests.emplace_back(std::move(request));
    }

    // Notify processing thread that new work is available
    slots_cv.notify_one();

    return request_id;
}

// Queue an embedding task for parallel processing
int32_t llama_rn_slot_manager::queue_embedding_request(
    const std::vector<llama_token>& tokens,
    int embd_normalize,
    std::function<void(int32_t, const std::vector<float>&)> on_result
) {
    if (parent_ctx == nullptr || parent_ctx->model == nullptr || parent_ctx->ctx == nullptr) {
        LOG_ERROR("Cannot queue embedding: context not initialized");
        return -1;
    }

    int32_t request_id = next_request_id++;

    if (!parent_ctx->params.embedding) {
        LOG_WARNING("Embedding disabled in model parameters; returning zero vector");
        if (on_result) {
            const int n_embd = llama_model_n_embd(parent_ctx->model);
            std::vector<float> empty_embedding(n_embd, 0.0f);
            on_result(request_id, empty_embedding);
        }
        return request_id;
    }

    llama_rn_queued_request request;
    request.request_id = request_id;
    request.task_type = SLOT_TASK_TYPE_EMBEDDING;
    request.prompt_tokens = tokens;
    request.embd_normalize = embd_normalize;
    request.on_embedding = on_result;

    {
        std::lock_guard<std::mutex> lock(slots_mutex);
        queue_requests.emplace_back(std::move(request));
    }

    slots_cv.notify_one();

    return request_id;
}

// Queue a rerank task for parallel processing
int32_t llama_rn_slot_manager::queue_rerank_request(
    const std::string& query,
    const std::vector<std::string>& documents,
    int normalize,
    std::function<void(int32_t, const std::vector<float>&)> on_results
) {
    if (parent_ctx == nullptr || parent_ctx->model == nullptr || parent_ctx->ctx == nullptr) {
        LOG_ERROR("Cannot queue rerank: context not initialized");
        return -1;
    }

    int32_t request_id = next_request_id++;

    const enum llama_pooling_type pooling_type = llama_pooling_type(parent_ctx->ctx);
    if (pooling_type != LLAMA_POOLING_TYPE_RANK) {
        LOG_ERROR("Reranking not supported by current model (pooling_type=%d)", pooling_type);
        if (on_results) {
            std::vector<float> scores(documents.size(), -1e6f);
            on_results(request_id, scores);
        }
        return request_id;
    }

    if (!parent_ctx->params.embedding) {
        LOG_ERROR("Embedding disabled but required for reranking");
        if (on_results) {
            std::vector<float> scores(documents.size(), -1e6f);
            on_results(request_id, scores);
        }
        return request_id;
    }

    const llama_vocab* vocab = llama_model_get_vocab(parent_ctx->model);
    if (vocab == nullptr) {
        LOG_ERROR("Failed to get vocabulary for rerank task");
        if (on_results) {
            std::vector<float> scores(documents.size(), -1e6f);
            on_results(request_id, scores);
        }
        return request_id;
    }

    llama_rn_queued_request request;
    request.request_id = request_id;
    request.task_type = SLOT_TASK_TYPE_RERANK;
    request.embd_normalize = normalize;
    request.on_rerank = on_results;

    try {
        std::vector<llama_token> query_tokens = common_tokenize(vocab, query, false, true);
        request.rerank_prompt_tokens.reserve(documents.size());

        const bool add_bos = llama_vocab_get_add_bos(vocab);
        const bool is_enc_dec = llama_model_has_encoder(parent_ctx->model);

        for (const std::string& doc : documents) {
            std::vector<llama_token> doc_tokens = common_tokenize(vocab, doc, false, true);
            std::vector<llama_token> rerank_tokens = format_rerank_tokens(vocab, query_tokens, doc_tokens);

            // Convert tokens back to text and re-tokenize using context-aware settings
            std::string rerank_text = tokens_to_str(parent_ctx->ctx, rerank_tokens.begin(), rerank_tokens.end());
            std::vector<llama_token> prompt_tokens = common_tokenize(
                parent_ctx->ctx,
                rerank_text,
                add_bos || is_enc_dec,
                true
            );

            request.rerank_prompt_tokens.push_back(std::move(prompt_tokens));
        }

        if (!request.rerank_prompt_tokens.empty()) {
            request.prompt_tokens = request.rerank_prompt_tokens.front();
        }
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to tokenize rerank inputs: %s", e.what());
        if (on_results) {
            std::vector<float> scores(documents.size(), -1e6f);
            on_results(request_id, scores);
        }
        return request_id;
    }

    if (request.rerank_prompt_tokens.empty()) {
        LOG_INFO("Rerank request %d has no documents; returning empty result", request_id);
        if (on_results) {
            on_results(request_id, {});
        }
        return request_id;
    }

    {
        std::lock_guard<std::mutex> lock(slots_mutex);
        queue_requests.emplace_back(std::move(request));
    }

    slots_cv.notify_one();

    return request_id;
}

// Get available slot (LRU strategy for now, similarity matching in Phase 3)
llama_rn_slot* llama_rn_slot_manager::get_available_slot(const std::vector<llama_token>& prompt) {
    llama_rn_slot* best_slot = nullptr;
    int64_t oldest_time = INT64_MAX;

    // Find idle or done slot with oldest t_last_used (LRU)
    for (auto& slot : slots) {
        if (slot.state == SLOT_STATE_IDLE || slot.state == SLOT_STATE_DONE) {
            if (slot.t_last_used < oldest_time) {
                oldest_time = slot.t_last_used;
                best_slot = &slot;
            }
        }
    }

    if (best_slot != nullptr) {
        LOG_VERBOSE("Selected slot %d (LRU)", best_slot->id);
    } else {
        LOG_VERBOSE("No available slots");
    }

    return best_slot;
}

// Get slot by request ID
llama_rn_slot* llama_rn_slot_manager::get_slot_by_request_id(int32_t request_id) {
    auto it = active_requests.find(request_id);
    if (it != active_requests.end()) {
        return it->second;
    }
    return nullptr;
}

// Release slot
void llama_rn_slot_manager::release_slot(llama_rn_slot* slot) {
    LOG_VERBOSE("Releasing slot %d", slot->id);

    // Update last used timestamp for LRU tracking
    slot->t_last_used = lm_ggml_time_us();

    // Reset slot (cache_tokens is preserved by reset() for potential reuse)
    slot->reset();
}

// Cancel request
void llama_rn_slot_manager::cancel_request(int32_t request_id) {
    LOG_INFO("Cancelling request %d", request_id);

    // Check if request is active
    auto it = active_requests.find(request_id);
    if (it != active_requests.end()) {
        llama_rn_slot* slot = it->second;
        // Mark as interrupted and set to DONE state
        // Don't call release_slot yet - let update_slots handle cleanup
        slot->is_interrupted = true;
        slot->state = SLOT_STATE_DONE;
        active_requests.erase(it);
        LOG_INFO("Request %d cancelled (was active in slot %d)", request_id, slot->id);
        return;
    }

    // Remove from pending queue
    auto pending_it = std::remove_if(queue_requests.begin(), queue_requests.end(),
        [request_id](const llama_rn_queued_request& req) {
            return req.request_id == request_id;
        });
    if (pending_it != queue_requests.end()) {
        queue_requests.erase(pending_it, queue_requests.end());
        LOG_INFO("Request %d cancelled (was in pending queue)", request_id);
        return;
    }

    LOG_WARNING("Request %d not found for cancellation", request_id);
}

// Compute similarity between two token sequences (stub for Phase 3)
float llama_rn_slot_manager::compute_similarity(
    const std::vector<llama_token>& a,
    const std::vector<llama_token>& b
) {
    // Longest Common Prefix (LCP) approach
    size_t common_prefix = 0;
    size_t max_len = std::min(a.size(), b.size());
    for (size_t i = 0; i < max_len; i++) {
        if (a[i] == b[i]) {
            common_prefix++;
        } else {
            break;
        }
    }

    if (a.empty() && b.empty()) return 1.0f;
    if (a.empty() || b.empty()) return 0.0f;

    return static_cast<float>(common_prefix) / static_cast<float>(std::max(a.size(), b.size()));
}

// Process pending queue
void llama_rn_slot_manager::process_pending_queue() {
    while (!queue_requests.empty()) {
        llama_rn_queued_request& request = queue_requests.front();

        const std::vector<llama_token>* prompt_view = nullptr;
        std::vector<llama_token> empty_prompt;

        if (request.task_type == SLOT_TASK_TYPE_RERANK) {
            if (!request.rerank_prompt_tokens.empty()) {
                prompt_view = &request.rerank_prompt_tokens.front();
            }
        } else {
            prompt_view = &request.prompt_tokens;
        }

        if (prompt_view == nullptr) {
            prompt_view = &empty_prompt;
        }

        llama_rn_slot* slot = get_available_slot(*prompt_view);
        if (slot == nullptr) {
            LOG_VERBOSE(
                "No available slots, stopping queue processing (request %d at front)",
                request.request_id
            );
            break;
        }

        // Assign request to slot
        slot->request_id = request.request_id;
        slot->task_type = request.task_type;
        slot->is_interrupted = false;

        // Reset callbacks from previous usage
        slot->on_token_callback = nullptr;
        slot->on_complete_callback = nullptr;
        slot->on_embedding_callback = nullptr;
        slot->on_rerank_callback = nullptr;

        // Ensure we start without a sampling context unless set below
        if (slot->ctx_sampling != nullptr) {
            common_sampler_free(slot->ctx_sampling);
            slot->params = nullptr;
            slot->ctx_sampling = nullptr;
        }

        switch (request.task_type) {
            case SLOT_TASK_TYPE_COMPLETION: {
                slot->params = &request.params;
                slot->ctx_sampling = common_sampler_init(parent_ctx->model, request.params.sampling);

                // Assign state parameters
                slot->load_state_path = request.load_state_path;
                slot->save_state_path = request.save_state_path;
                slot->save_state_size = request.save_state_size;

                // Load state if provided
                if (!slot->load_state_path.empty()) {
                    if (!slot->load_state()) {
                        LOG_ERROR("Failed to load state for slot %d, request %d",
                                  slot->id, request.request_id);
                        // Mark slot as done with error
                        slot->state = SLOT_STATE_DONE;
                        slot->incomplete = true;
                        slot->error_message = "Failed to load state from: " + slot->load_state_path;
                        if (request.on_complete) {
                            request.on_complete(slot);
                        }
                        queue_requests.pop_front();
                        continue;
                    }
                }

                // Start timing AFTER state loading completes
                slot->t_start_process = lm_ggml_time_us();

                // Always load prompt - it will detect and preserve state if appropriate
                bool has_media = !request.media_paths.empty();
                if (has_media && parent_ctx->isMultimodalEnabled()) {
                    LOG_INFO("Storing %zu media paths for deferred processing in slot %d",
                             request.media_paths.size(), slot->id);
                    slot->media_paths = request.media_paths;
                    slot->prompt_text = request.prompt_text;
                    slot->media_processed = false;
                    slot->load_prompt(request.prompt_tokens);
                } else {
                    slot->media_paths.clear();
                    slot->prompt_text.clear();
                    slot->media_processed = true;
                    slot->load_prompt(request.prompt_tokens);
                }
                slot->i_batch = -1;

                slot->on_token_callback = request.on_token;
                slot->on_complete_callback = request.on_complete;
                slot->current_chat_format = request.chat_format;
                slot->current_reasoning_format = request.reasoning_format;
                slot->current_thinking_forced_open = request.thinking_forced_open;
                slot->prefill_text = request.prefill_text;
                slot->n_remaining = request.params.n_predict;
                slot->stop_words = request.params.antiprompt;
                break;
            }

            case SLOT_TASK_TYPE_EMBEDDING: {
                slot->params = &request.params;
                // Start timing (no state loading for embeddings)
                slot->t_start_process = lm_ggml_time_us();

                slot->media_paths.clear();
                slot->prompt_text.clear();
                slot->media_processed = true;
                slot->embd_normalize = request.embd_normalize;
                slot->on_embedding_callback = request.on_embedding;
                slot->n_remaining = -1;
                slot->stop_words.clear();
                slot->load_prompt(request.prompt_tokens);
                slot->i_batch = -1;
                break;
            }

            case SLOT_TASK_TYPE_RERANK: {
                slot->params = nullptr;
                // Start timing (memory clear is part of the task, not overhead)
                slot->t_start_process = lm_ggml_time_us();

                if (parent_ctx && parent_ctx->ctx) {
                    llama_memory_clear(llama_get_memory(parent_ctx->ctx), false);
                }
                if (request.rerank_prompt_tokens.empty()) {
                    LOG_WARNING("Rerank request %d has no documents to process", request.request_id);
                    if (request.on_rerank) {
                        request.on_rerank(request.request_id, {});
                    }
                    queue_requests.pop_front();
                    continue;
                }

                slot->media_paths.clear();
                slot->prompt_text.clear();
                slot->media_processed = true;
                slot->embd_normalize = request.embd_normalize;
                slot->on_rerank_callback = request.on_rerank;
                slot->rerank_prompt_tokens = std::move(request.rerank_prompt_tokens);
                slot->rerank_scores.assign(slot->rerank_prompt_tokens.size(), 0.0f);
                slot->rerank_current_index = 0;
                slot->n_remaining = -1;
                slot->stop_words.clear();
                slot->load_prompt(slot->rerank_prompt_tokens[0]);
                slot->i_batch = -1;
                break;
            }

            default:
                LOG_ERROR("Unknown task type %d for request %d", request.task_type, request.request_id);
                queue_requests.pop_front();
                continue;
        }

        // Track active request
        active_requests[request.request_id] = slot;

        // Remove from queue
        queue_requests.pop_front();
    }
}

// Build batch from all active slots
void llama_rn_slot_manager::build_batch() {
    // Clear the batch
    batch.n_tokens = 0;

    // First pass: Add tokens from GENERATING slots (previously sampled tokens)
    for (auto& slot : slots) {
        if (slot.state == SLOT_STATE_GENERATING) {
            // Only add if we have generated tokens (skip first iteration after prompt)
            if (!slot.generated_tokens.empty()) {
                // Get the last generated token
                llama_token token = slot.generated_tokens.back();

                // Add to batch with this slot's sequence ID
                llama_batch_add(&batch, token, slot.n_past, {slot.id}, true);

                // Mark position in batch for this slot
                slot.i_batch = batch.n_tokens - 1;

                slot.n_past++; // Increment for next token

                LOG_VERBOSE("Slot %d: Added generated token %d at pos %d", slot.id, token, slot.n_past - 1);
            }
        }
    }

    // Second pass: Add prompt tokens from PROCESSING_PROMPT slots
    for (auto& slot : slots) {
        if (slot.state == SLOT_STATE_PROCESSING_PROMPT) {
            // Check if we need to process media first (deferred processing)
            // Process media at the very start (n_past == 0) before any prompt tokens
            if (!slot.media_processed && !slot.media_paths.empty() && slot.n_past == 0) {
                LOG_INFO("Slot %d: Processing media before prompt tokens", slot.id);

                try {
                    // Clear KV cache for this slot's sequence to ensure clean state
                    if (parent_ctx && parent_ctx->ctx) {
                        auto * kv = llama_get_memory(parent_ctx->ctx);
                        llama_memory_seq_rm(kv, slot.id, 0, -1);
                        LOG_VERBOSE("Slot %d: Cleared KV cache for sequence", slot.id);
                    }

                    // Process media using the stored prompt_text and media_paths
                    slot.embd.clear();
                    llama_pos n_past_before = slot.n_past;
                    slot.n_past = 0;
                    bool context_full = false;

                    parent_ctx->mtmd_wrapper->processMedia(
                        parent_ctx->ctx,
                        slot.prompt_text,
                        slot.media_paths,
                        parent_ctx->n_ctx,
                        n_batch,
                        slot.n_past,
                        slot.embd,
                        context_full,
                        slot.ctx_sampling,
                        slot.bitmap_past_hashes,
                        slot.id  // Use slot ID as sequence ID for parallel processing
                    );

                    if (context_full) {
                        LOG_ERROR("Context full after processing media for slot %d", slot.id);
                        slot.context_full = true;
                        slot.state = SLOT_STATE_DONE;
                        if (slot.on_complete_callback) {
                            slot.on_complete_callback(&slot);
                        }
                        continue;
                    }

                    // Update prompt tokens with the processed result from processMedia
                    slot.prompt_tokens = slot.embd;
                    slot.num_prompt_tokens = slot.embd.size();
                    slot.media_processed = true;

                    // processMedia() fills ALL tokens into KV cache, so update n_past to match
                    // This prevents the while loop from re-adding tokens that are already in KV cache
                    slot.n_past = slot.num_prompt_tokens;

                    // Transition to GENERATING state immediately since all prompt tokens are processed
                    slot.state = SLOT_STATE_GENERATING;

                    // Mark that prompt processing just finished - timing will be calculated after decode
                    // Note: for media processing, processMedia() already decoded everything, so timing
                    // calculation will happen immediately after this in the main loop
                    slot.prompt_processing_finished = true;
                    slot.n_prompt_tokens_processed = slot.num_prompt_tokens - slot.n_prompt_tokens_cache;

                    // Set i_batch to -1 to indicate logits from media processing are ready to sample
                    // In sample_and_callback(), batch index -1 will be handled specially
                    slot.i_batch = -1;

                    LOG_INFO("Slot %d: Media processed, transitioned to GENERATING state, n_past=%d, num_prompt_tokens=%zu",
                            slot.id, slot.n_past, slot.num_prompt_tokens);

                    // Continue to next slot - this slot is ready to sample in sample_and_callback()
                    continue;

                } catch (const std::exception& e) {
                    LOG_ERROR("Failed to process media for slot %d: %s", slot.id, e.what());
                    slot.state = SLOT_STATE_DONE;
                    slot.incomplete = true;
                    if (slot.on_complete_callback) {
                        slot.on_complete_callback(&slot);
                    }
                    continue;
                }
            }

            // Process tokens up to n_batch limit (only for non-media slots)
            while (slot.n_past < (llama_pos)slot.num_prompt_tokens && batch.n_tokens < n_batch) {
                llama_token token = slot.prompt_tokens[slot.n_past];

                // Skip LLAMA_TOKEN_NULL - these are media placeholders already in KV cache
                if (token == LLAMA_TOKEN_NULL) {
                    LOG_VERBOSE("Slot %d: Skipping NULL token at pos %d (media chunk)", slot.id, slot.n_past);
                    slot.n_past++;
                    continue;
                }

                // Request logits for all tokens when embeddings/rerank are needed
                bool need_logits = true;
                if (slot.task_type == SLOT_TASK_TYPE_COMPLETION) {
                    need_logits = (slot.n_past == (llama_pos)(slot.num_prompt_tokens - 1));
                }

                // Add to batch with this slot's sequence ID
                llama_batch_add(&batch, token, slot.n_past, {slot.id}, need_logits);

                // Mark position in batch for this slot (will be overwritten each iteration)
                slot.i_batch = batch.n_tokens - 1;

                slot.n_past++;
            }

            // If we've processed all prompt tokens, transition based on task type
            if (slot.n_past >= (llama_pos)slot.num_prompt_tokens) {
                slot.state = SLOT_STATE_GENERATING;

                // Mark that prompt processing just finished - timing will be calculated after decode
                slot.prompt_processing_finished = true;
                slot.n_prompt_tokens_processed = slot.num_prompt_tokens - slot.n_prompt_tokens_cache;

                if (slot.task_type == SLOT_TASK_TYPE_COMPLETION) {
                    LOG_INFO("Slot %d: Transitioned to GENERATING state", slot.id);
                } else if (slot.task_type == SLOT_TASK_TYPE_EMBEDDING) {
                    LOG_INFO("Slot %d: Prompt processed for embedding task", slot.id);
                } else if (slot.task_type == SLOT_TASK_TYPE_RERANK) {
                    LOG_INFO("Slot %d: Prompt processed for rerank task (doc %zu/%zu)",
                             slot.id,
                             slot.rerank_current_index + 1,
                             slot.rerank_prompt_tokens.size());
                }
            }

            LOG_VERBOSE("Slot %d: Processed prompt tokens, n_past=%d/%zu",
                       slot.id, slot.n_past, slot.num_prompt_tokens);
        }
    }

    LOG_VERBOSE("Batch built with %d tokens", batch.n_tokens);
}

bool llama_rn_slot_manager::process_batch() {
    if (batch.n_tokens == 0) {
        // No tokens to process
        return true;
    }

    if (parent_ctx == nullptr || parent_ctx->ctx == nullptr) {
        LOG_ERROR("Cannot process batch: context is null");
        return false;
    }

    // Call llama_decode with the unified batch
    int ret = llama_decode(parent_ctx->ctx, batch);

    if (ret != 0) {
        // Decode failed
        if (ret == 1) {
            LOG_ERROR("llama_decode failed: could not find a KV slot for the batch");
        } else {
            LOG_ERROR("llama_decode failed with code: %d", ret);
        }

        // Try with smaller batch size next time
        if (n_batch > 32) {
            n_batch = n_batch / 2;
            LOG_WARNING("Reducing batch size to %d", n_batch);
        }

        return false;
    }

    // Synchronize to ensure GPU work completes before timing measurements
    // This is critical for accurate performance metrics when using Metal/GPU
    llama_synchronize(parent_ctx->ctx);

    LOG_VERBOSE("Batch processed successfully");
    return true;
}

void llama_rn_slot_manager::sample_and_callback() {
    if (parent_ctx == nullptr || parent_ctx->ctx == nullptr) {
        return;
    }

    const llama_vocab* vocab = llama_model_get_vocab(parent_ctx->model);
    const int n_embd = llama_model_n_embd(parent_ctx->model);

    auto get_embedding_ptr = [&](llama_rn_slot& slot) -> const float* {
        const float* data = llama_get_embeddings_seq(parent_ctx->ctx, slot.id);
        if (data == nullptr) {
            int idx = slot.i_batch;
            if (idx < 0 || idx >= batch.n_tokens) {
                idx = batch.n_tokens - 1;
            }
            if (idx >= 0) {
                data = llama_get_embeddings_ith(parent_ctx->ctx, idx);
            }
        }
        if (data == nullptr) {
            data = llama_get_embeddings(parent_ctx->ctx);
        }
        return data;
    };

    // Process each slot in GENERATING state
    for (auto& slot : slots) {
        if (slot.state != SLOT_STATE_GENERATING) {
            continue;
        }

        // Check if interrupted
        if (slot.is_interrupted) {
            LOG_INFO("Slot %d: Generation interrupted", slot.id);
            slot.state = SLOT_STATE_DONE;
            continue;
        }
        switch (slot.task_type) {
            case SLOT_TASK_TYPE_COMPLETION: {
                if (slot.ctx_sampling == nullptr) {
                    LOG_WARNING("Slot %d: Sampling context is null, marking as done", slot.id);
                    slot.state = SLOT_STATE_DONE;
                    continue;
                }

                if (slot.i_batch == -1) {
                    LOG_VERBOSE("Slot %d: Sampling from media processing logits (batch index -1)", slot.id);
                } else if (slot.i_batch < 0 || slot.i_batch >= batch.n_tokens) {
                    LOG_WARNING("Slot %d: Invalid batch position %d", slot.id, slot.i_batch);
                    continue;
                }

                llama_token new_token_id = common_sampler_sample(slot.ctx_sampling, parent_ctx->ctx, slot.i_batch);
                common_sampler_accept(slot.ctx_sampling, new_token_id, true);

                if (llama_vocab_is_eog(vocab, new_token_id)) {
                    slot.stopped_eos = true;
                    slot.state = SLOT_STATE_DONE;
                    LOG_INFO("Slot %d: Stopped on EOS token", slot.id);

                    // Save state if path is provided
                    if (!slot.save_state_path.empty()) {
                        slot.save_state();
                    }

                    if (slot.on_complete_callback) {
                        slot.on_complete_callback(&slot);
                    }
                    continue;
                }

                std::string token_text = common_token_to_piece(parent_ctx->ctx, new_token_id);
                slot.generated_text += token_text;

                // Update token generation timing
                const int64_t t_current = lm_ggml_time_us();
                slot.t_token_generation = (t_current - slot.t_start_generation) / 1e6;

                completion_token_output token_output;
                token_output.tok = new_token_id;
                token_output.text = token_text;
                token_output.request_id = slot.request_id;

                const int32_t n_probs = slot.params->sampling.n_probs;
                if (n_probs > 0) {
                  llama_token_data_array cur_p = *common_sampler_get_candidates(slot.ctx_sampling, true);
                  for (size_t i = 0; i < std::min(cur_p.size, (size_t)n_probs); ++i)
                  {
                      token_output.probs.push_back({cur_p.data[i].id, cur_p.data[i].p});
                  }
                }

                slot.generated_tokens.push_back(new_token_id);
                slot.n_decoded++;

                // Update cache_tokens to keep track of all processed tokens
                // This is needed for state saving
                slot.cache_tokens.push_back(new_token_id);

                if (slot.on_token_callback) {
                    slot.on_token_callback(token_output);
                }

                bool should_stop = false;

                if (slot.n_remaining > 0) {
                    slot.n_remaining--;
                    if (slot.n_remaining == 0) {
                        slot.stopped_limit = true;
                        should_stop = true;
                        LOG_INFO("Slot %d: Stopped on token limit", slot.id);
                    }
                }

                if (slot.n_past >= slot.n_ctx) {
                    slot.context_full = true;
                    should_stop = true;
                    LOG_WARNING("Slot %d: Context full", slot.id);
                }

                if (!slot.stop_words.empty() && !slot.generated_text.empty()) {
                    const std::string& text = slot.generated_text;
                    const size_t last_token_size = token_text.size();

                    for (const std::string& word : slot.stop_words) {
                        const size_t search_start = text.size() > word.size() + last_token_size
                            ? text.size() - word.size() - last_token_size
                            : 0;
                        size_t pos = text.find(word, search_start);

                        if (pos != std::string::npos) {
                            slot.stopped_word = true;
                            slot.stopping_word = word;
                            should_stop = true;
                            LOG_INFO("Slot %d: Stopped on word '%s'", slot.id, word.c_str());
                            break;
                        }
                    }
                }

                if (should_stop) {
                    slot.state = SLOT_STATE_DONE;

                    // Save state if path is provided
                    if (!slot.save_state_path.empty()) {
                        slot.save_state();
                    }

                    if (slot.on_complete_callback) {
                        slot.on_complete_callback(&slot);
                    }
                }

                LOG_VERBOSE("Slot %d: Generated token %d ('%s'), n_past=%d, n_decoded=%d",
                           slot.id, new_token_id, token_text.c_str(), slot.n_past, slot.n_decoded);
                break;
            }

            case SLOT_TASK_TYPE_EMBEDDING: {
                const float* data = get_embedding_ptr(slot);

                std::vector<float> embedding(n_embd, 0.0f);
                if (data != nullptr) {
                    embedding.assign(data, data + n_embd);
                }

                std::vector<float> normalized(n_embd, 0.0f);
                if (n_embd >= 4) {
                    LOG_INFO("Embedding data: 0: %f 1: %f 2: %f 3: %f",
                             embedding[0], embedding[1], embedding[2], embedding[3]);
                }
                LOG_INFO("Normalizing embedding with normalize=%d", slot.embd_normalize);
                common_embd_normalize(embedding.data(), normalized.data(), n_embd, slot.embd_normalize);
                if (n_embd >= 4) {
                    LOG_INFO("Normalized embedding data: 0: %f 1: %f 2: %f 3: %f",
                             normalized[0], normalized[1], normalized[2], normalized[3]);
                }

                if (slot.on_embedding_callback) {
                    slot.on_embedding_callback(slot.request_id, normalized);
                }

                slot.state = SLOT_STATE_DONE;
                continue;
            }

            case SLOT_TASK_TYPE_RERANK: {
                const float* data = get_embedding_ptr(slot);
                float score = data ? data[0] : -1e6f;

                LOG_INFO("Rerank data: 0: %f", score);

                if (slot.rerank_current_index < slot.rerank_scores.size()) {
                    slot.rerank_scores[slot.rerank_current_index] = score;
                }

                slot.rerank_current_index++;

                if (slot.rerank_current_index < slot.rerank_prompt_tokens.size()) {
                    if (parent_ctx && parent_ctx->ctx) {
                        llama_memory_clear(llama_get_memory(parent_ctx->ctx), false);
                    }
                    slot.load_prompt(slot.rerank_prompt_tokens[slot.rerank_current_index]);
                    slot.state = SLOT_STATE_PROCESSING_PROMPT;
                    slot.i_batch = -1;
                    continue;
                }

                if (slot.on_rerank_callback) {
                    slot.on_rerank_callback(slot.request_id, slot.rerank_scores);
                }

                if (parent_ctx && parent_ctx->ctx) {
                    llama_memory_clear(llama_get_memory(parent_ctx->ctx), false);
                }

                slot.state = SLOT_STATE_DONE;
                continue;
            }

            default:
                LOG_ERROR("Slot %d: Unknown task type %d in sampling", slot.id, slot.task_type);
                slot.state = SLOT_STATE_DONE;
                continue;
        }
    }
}

// Release completed slots
void llama_rn_slot_manager::release_completed_slots() {
    for (auto& slot : slots) {
        if (slot.state == SLOT_STATE_DONE) {
            // Remove from active requests
            auto it = active_requests.find(slot.request_id);
            if (it != active_requests.end()) {
                active_requests.erase(it);
            }

            // Release slot
            release_slot(&slot);
        }
    }
}

// Main processing loop
void llama_rn_slot_manager::update_slots() {
    // Step 1: Process pending queue (with mutex)
    {
        std::lock_guard<std::mutex> lock(slots_mutex);
        process_pending_queue();
    }

    // Step 2: Check if any slots are active (with mutex)
    bool has_active = false;
    {
        std::lock_guard<std::mutex> lock(slots_mutex);
        for (const auto& slot : slots) {
            if (slot.state == SLOT_STATE_PROCESSING_PROMPT || slot.state == SLOT_STATE_GENERATING) {
                has_active = true;
                break;
            }
        }
    }

    if (!has_active) {
        // No active slots, return early
        return;
    }

    // Step 3: Build batch from all active slots (with mutex)
    {
        std::lock_guard<std::mutex> lock(slots_mutex);
        build_batch();
    }

    // Step 4: Process batch if we have tokens (NO mutex - llama_decode is thread-safe)
    if (batch.n_tokens > 0) {
        bool success = process_batch();
        if (!success) {
            LOG_ERROR("Batch processing failed");
            // Mark all active slots as done with error (with mutex)
            std::lock_guard<std::mutex> lock(slots_mutex);
            for (auto& slot : slots) {
                if (slot.state == SLOT_STATE_PROCESSING_PROMPT || slot.state == SLOT_STATE_GENERATING) {
                    slot.state = SLOT_STATE_DONE;
                    slot.incomplete = true;
                    if (slot.on_complete_callback) {
                        slot.on_complete_callback(&slot);
                    }
                }
            }
            return;
        }

        // Step 4.5: Calculate timing for slots that just finished prompt processing
        // This must happen AFTER batch has been decoded
        {
            std::lock_guard<std::mutex> lock(slots_mutex);
            const int64_t t_now = lm_ggml_time_us();
            for (auto& slot : slots) {
                if (slot.prompt_processing_finished) {
                    slot.t_start_generation = t_now;
                    slot.t_prompt_processing = (slot.t_start_generation - slot.t_start_process) / 1e6;
                    slot.prompt_processing_finished = false;  // Clear the flag

                    LOG_VERBOSE("Slot %d: Prompt processing complete, time=%.3fs, tokens=%d (cached=%d)",
                               slot.id, slot.t_prompt_processing, slot.n_prompt_tokens_processed, slot.n_prompt_tokens_cache);
                }
            }
        }
    }

    // Step 5: Sample tokens and invoke callbacks for GENERATING slots (with mutex)
    {
        std::lock_guard<std::mutex> lock(slots_mutex);
        sample_and_callback();
    }

    // Step 6: Release completed slots (with mutex)
    {
        std::lock_guard<std::mutex> lock(slots_mutex);
        release_completed_slots();
    }

    // Step 7: Process pending queue again - assign requests to newly freed slots (with mutex)
    {
        std::lock_guard<std::mutex> lock(slots_mutex);
        process_pending_queue();
    }
}

// Start background processing loop
void llama_rn_slot_manager::start_processing_loop() {
    // Check if already running
    if (processing_active.load()) {
        LOG_WARNING("Processing loop already active");
        return;
    }

    processing_active.store(true);

    // Start processing thread
    processing_thread = std::thread([this]() {
        LOG_INFO("Processing loop started");

        while (processing_active.load()) {
            // Call update_slots (protected by mutex)
            update_slots();

            // Wait for new work instead of sleeping
            // This efficiently blocks until notified or until there's work to do
            std::unique_lock<std::mutex> lock(slots_mutex);

            // Check if we have any active work or pending requests
            bool has_work = !queue_requests.empty();
            if (!has_work) {
                for (const auto& slot : slots) {
                    if (slot.state == SLOT_STATE_PROCESSING_PROMPT || slot.state == SLOT_STATE_GENERATING) {
                        has_work = true;
                        break;
                    }
                }
            }

            // If no work, wait for notification
            if (!has_work && processing_active.load()) {
                slots_cv.wait(lock, [this]() {
                    // Wake up if: there are pending requests, or processing should stop
                    return !queue_requests.empty() || !processing_active.load();
                });
            }
        }

        LOG_INFO("Processing loop stopped");
    });
}

// Stop background processing loop
void llama_rn_slot_manager::stop_processing_loop() {
    if (!processing_active.load()) {
        return;
    }

    LOG_INFO("Stopping processing loop...");
    processing_active.store(false);

    // Notify condition variable to wake up the thread
    slots_cv.notify_all();

    // Wait for processing thread to finish
    if (processing_thread.joinable()) {
        processing_thread.join();
    }

    LOG_INFO("Processing loop stopped");
}

} // namespace rnllama
