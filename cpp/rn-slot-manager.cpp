#include "rn-slot-manager.h"
#include "rn-completion.h"
#include "rn-llama.h"
#include "ggml.h"
#include <algorithm>
#include <cstring>

namespace rnllama {

// Helper function to check if string ends with suffix
static bool ends_with(const std::string& str, const std::string& suffix) {
    if (suffix.size() > str.size()) return false;
    return std::equal(suffix.rbegin(), suffix.rend(), str.rbegin());
}

// Helper function to find partial stop string at end of text
static size_t find_partial_stop_string(const std::string& stop, const std::string& text) {
    if (!text.empty() && !stop.empty()) {
        const char text_last_char = text.back();
        for (int64_t char_index = stop.size() - 1; char_index >= 0; char_index--) {
            if (stop[char_index] == text_last_char) {
                const std::string current_partial = stop.substr(0, char_index + 1);
                if (ends_with(text, current_partial)) {
                    return text.size() - char_index - 1;
                }
            }
        }
    }
    return std::string::npos;
}

// Constructor
llama_rn_slot_manager::llama_rn_slot_manager(llama_rn_context* ctx) :
    parent_ctx(ctx),
    n_parallel(1),
    next_request_id(1),
    n_batch(512),
    slot_prompt_similarity(0.5f),
    continuous_batching(false)
{
    // Initialize batch to zero/null - will be properly allocated later
    std::memset(&batch, 0, sizeof(batch));
}

// Destructor
llama_rn_slot_manager::~llama_rn_slot_manager() {
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
    int chat_format,
    common_reasoning_format reasoning_format,
    bool thinking_forced_open,
    const std::string& prefill_text,
    std::function<void(const completion_token_output&)> on_token,
    std::function<void(llama_rn_slot*)> on_complete
) {
    // Generate unique request ID
    int32_t request_id = next_request_id++;

    LOG_INFO("Queuing request %d with %zu prompt tokens", request_id, prompt.size());

    // Create queued request
    llama_rn_queued_request request;
    request.request_id = request_id;
    request.params = params;
    request.prompt_tokens = prompt;
    request.media_paths = media_paths;
    request.chat_format = chat_format;
    request.reasoning_format = reasoning_format;
    request.thinking_forced_open = thinking_forced_open;
    request.prefill_text = prefill_text;
    request.on_token = on_token;
    request.on_complete = on_complete;

    // Add to queue
    queue_requests.push_back(request);

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

    // Save cache tokens for potential reuse
    slot->cache_tokens = slot->prompt_tokens;
    slot->t_last_used = lm_ggml_time_us();

    // Reset slot
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

    // Remove from deferred queue
    auto deferred_it = std::remove_if(queue_deferred.begin(), queue_deferred.end(),
        [request_id](const llama_rn_queued_request& req) {
            return req.request_id == request_id;
        });
    if (deferred_it != queue_deferred.end()) {
        queue_deferred.erase(deferred_it, queue_deferred.end());
        LOG_INFO("Request %d cancelled (was in deferred queue)", request_id);
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

        // Try to get available slot
        llama_rn_slot* slot = get_available_slot(request.prompt_tokens);

        if (slot == nullptr) {
            // No slots available, defer all remaining requests
            LOG_VERBOSE("No available slots, deferring request %d", request.request_id);
            queue_deferred.push_back(request);
            queue_requests.pop_front();
            continue;
        }

        // Assign request to slot
        LOG_INFO("Assigning request %d to slot %d", request.request_id, slot->id);
        slot->request_id = request.request_id;
        slot->load_prompt(request.prompt_tokens);
        slot->on_token_callback = request.on_token;
        slot->on_complete_callback = request.on_complete;
        slot->current_chat_format = request.chat_format;
        slot->current_reasoning_format = request.reasoning_format;
        slot->current_thinking_forced_open = request.thinking_forced_open;
        slot->prefill_text = request.prefill_text;
        slot->t_start_process = lm_ggml_time_us();

        // Set token generation limit from params
        slot->n_remaining = request.params.n_predict;

        // Copy stop words from params
        slot->stop_words = request.params.antiprompt;

        // Initialize sampling context from params
        if (slot->ctx_sampling != nullptr) {
            common_sampler_free(slot->ctx_sampling);
        }
        slot->ctx_sampling = common_sampler_init(parent_ctx->model, request.params.sampling);

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
            // Process tokens up to n_batch limit
            while (slot.n_past < (llama_pos)slot.num_prompt_tokens && batch.n_tokens < n_batch) {
                llama_token token = slot.prompt_tokens[slot.n_past];

                // Only request logits for the last token of the prompt
                bool need_logits = (slot.n_past == (llama_pos)(slot.num_prompt_tokens - 1));

                // Add to batch with this slot's sequence ID
                llama_batch_add(&batch, token, slot.n_past, {slot.id}, need_logits);

                // Mark position in batch for this slot (will be overwritten each iteration)
                slot.i_batch = batch.n_tokens - 1;

                slot.n_past++;
            }

            // If we've processed all prompt tokens, transition to GENERATING state
            if (slot.n_past >= (llama_pos)slot.num_prompt_tokens) {
                slot.state = SLOT_STATE_GENERATING;
                slot.t_start_generation = lm_ggml_time_us();
                LOG_INFO("Slot %d: Transitioned to GENERATING state", slot.id);
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

    LOG_VERBOSE("Processing batch with %d tokens", batch.n_tokens);

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

    LOG_VERBOSE("Batch processed successfully");
    return true;
}

void llama_rn_slot_manager::sample_and_callback() {
    if (parent_ctx == nullptr || parent_ctx->ctx == nullptr) {
        return;
    }

    const llama_vocab* vocab = llama_model_get_vocab(parent_ctx->model);

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

        // Check if we have a valid batch position
        if (slot.i_batch < 0 || slot.i_batch >= batch.n_tokens) {
            LOG_WARNING("Slot %d: Invalid batch position %d", slot.id, slot.i_batch);
            continue;
        }

        // Safety check: ensure sampling context is valid
        if (slot.ctx_sampling == nullptr) {
            LOG_WARNING("Slot %d: Sampling context is null, marking as done", slot.id);
            slot.state = SLOT_STATE_DONE;
            continue;
        }

        // Sample token using the slot's sampling context
        llama_token new_token_id = common_sampler_sample(slot.ctx_sampling, parent_ctx->ctx, slot.i_batch);

        // Accept the token
        common_sampler_accept(slot.ctx_sampling, new_token_id, true);

        // Check for EOS token
        if (llama_vocab_is_eog(vocab, new_token_id)) {
            slot.stopped_eos = true;
            slot.state = SLOT_STATE_DONE;
            LOG_INFO("Slot %d: Stopped on EOS token", slot.id);

            // Call completion callback
            if (slot.on_complete_callback) {
                slot.on_complete_callback(&slot);
            }
            continue;
        }

        // Convert token to text
        std::string token_text = common_token_to_piece(parent_ctx->ctx, new_token_id);
        slot.generated_text += token_text;

        // Create token output
        completion_token_output token_output;
        token_output.tok = new_token_id;
        token_output.text = token_text;
        token_output.request_id = slot.request_id;

        // Get token probabilities if requested
        llama_token_data_array* cur_p = common_sampler_get_candidates(slot.ctx_sampling, true);
        if (cur_p != nullptr) {
            // Add top probabilities (limit to reasonable number)
            const int32_t n_probs = std::min(10, (int32_t)cur_p->size);
            for (int32_t i = 0; i < n_probs; ++i) {
                token_output.probs.push_back({cur_p->data[i].id, cur_p->data[i].p});
            }
        }

        // Store token for next iteration
        slot.generated_tokens.push_back(new_token_id);
        slot.n_decoded++;
        // Note: n_past is incremented in build_batch() when adding to batch

        // Call token callback
        if (slot.on_token_callback) {
            slot.on_token_callback(token_output);
        }

        // Check stopping conditions
        bool should_stop = false;

        // Check token limit (n_remaining is set from params.n_predict)
        if (slot.n_remaining > 0) {
            slot.n_remaining--;
            if (slot.n_remaining == 0) {
                slot.stopped_limit = true;
                should_stop = true;
                LOG_INFO("Slot %d: Stopped on token limit", slot.id);
            }
        }

        // Check context full
        if (slot.n_past >= slot.n_ctx) {
            slot.context_full = true;
            should_stop = true;
            LOG_WARNING("Slot %d: Context full", slot.id);
        }

        // Check stopping words
        if (!slot.stop_words.empty() && !slot.generated_text.empty()) {
            const std::string& text = slot.generated_text;
            const size_t last_token_size = token_text.size();

            for (const std::string& word : slot.stop_words) {
                // Look for full match in the recent text
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

            // Call completion callback
            if (slot.on_complete_callback) {
                slot.on_complete_callback(&slot);
            }
        }

        LOG_VERBOSE("Slot %d: Generated token %d ('%s'), n_past=%d, n_decoded=%d",
                   slot.id, new_token_id, token_text.c_str(), slot.n_past, slot.n_decoded);
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

// Promote deferred requests back to main queue
void llama_rn_slot_manager::promote_deferred_requests() {
    if (!queue_deferred.empty()) {
        LOG_VERBOSE("Promoting %zu deferred requests", queue_deferred.size());
        // Move all deferred requests back to main queue
        queue_requests.insert(queue_requests.end(), queue_deferred.begin(), queue_deferred.end());
        queue_deferred.clear();
    }
}

// Main processing loop
void llama_rn_slot_manager::update_slots() {
    // Step 1: Process pending queue
    process_pending_queue();

    // Step 2: Check if any slots are active
    bool has_active = false;
    for (const auto& slot : slots) {
        if (slot.state == SLOT_STATE_PROCESSING_PROMPT || slot.state == SLOT_STATE_GENERATING) {
            has_active = true;
            break;
        }
    }

    if (!has_active) {
        // No active slots, return early
        return;
    }

    // Step 3: Build batch from all active slots
    build_batch();

    // Step 4: Process batch if we have tokens
    if (batch.n_tokens > 0) {
        bool success = process_batch();
        if (!success) {
            LOG_ERROR("Batch processing failed");
            // Mark all active slots as done with error
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
    }

    // Step 5: Sample tokens and invoke callbacks for GENERATING slots
    sample_and_callback();

    // Step 6: Release completed slots
    release_completed_slots();

    // Step 7: Promote deferred requests back to main queue
    promote_deferred_requests();
}

} // namespace rnllama