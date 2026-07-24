#pragma once

#include "JSIHelpers.h"
#include "JSINativeHeaders.h"
#include <algorithm>
#include <iterator>
#include <string>
#include <vector>

namespace rnllama_jsi {

    inline jsi::Object createTokenizeResult(jsi::Runtime& runtime, const rnllama::llama_rn_tokenize_result& result) {
        jsi::Object res(runtime);
        
        jsi::Array tokens = jsi::Array(runtime, result.tokens.size());
        for (size_t i = 0; i < result.tokens.size(); ++i) {
            tokens.setValueAtIndex(runtime, i, (double)result.tokens[i]);
        }
        res.setProperty(runtime, "tokens", tokens);
        
        res.setProperty(runtime, "has_media", result.has_media);
        
        jsi::Array hashes = jsi::Array(runtime, result.bitmap_hashes.size());
        for (size_t i = 0; i < result.bitmap_hashes.size(); ++i) {
            hashes.setValueAtIndex(runtime, i, jsi::String::createFromUtf8(runtime, result.bitmap_hashes[i]));
        }
        res.setProperty(runtime, "bitmap_hashes", hashes);
        
        jsi::Array chunk_pos = jsi::Array(runtime, result.chunk_pos.size());
        for (size_t i = 0; i < result.chunk_pos.size(); ++i) {
            chunk_pos.setValueAtIndex(runtime, i, (double)result.chunk_pos[i]);
        }
        res.setProperty(runtime, "chunk_pos", chunk_pos);
        
        jsi::Array chunk_pos_media = jsi::Array(runtime, result.chunk_pos_media.size());
        for (size_t i = 0; i < result.chunk_pos_media.size(); ++i) {
            chunk_pos_media.setValueAtIndex(runtime, i, (double)result.chunk_pos_media[i]);
        }
        res.setProperty(runtime, "chunk_pos_media", chunk_pos_media);

        return res;
    }

    inline jsi::Object loadSession(jsi::Runtime& runtime, rnllama::llama_rn_context* ctx, const std::string& path) {
        if (!ctx || !ctx->completion) {
            throw std::runtime_error("Context or completion not initialized");
        }
        if (ctx->slot_manager != nullptr) {
            // llama_state_load_file restores every sequence at once, which
            // would corrupt in-flight parallel slots
            throw std::runtime_error("Session load is not supported while parallel mode is enabled");
        }

        auto& embd = ctx->completion->embd;

        size_t n_token_count_out = 0;
        embd.resize(llama_n_ctx(ctx->ctx));
        if (!llama_state_load_file(ctx->ctx, path.c_str(), embd.data(), embd.size(), &n_token_count_out)) {
             throw std::runtime_error("Failed to load session");
        }
        // Keep LLAMA_TOKEN_NULL media placeholders: they hold the positions of
        // media evaluated into the restored memory
        embd.resize(n_token_count_out);

        // The restored memory may hold more positions than the token list
        // (legacy files saved from multimodal sequences or trimmed saves).
        // Reconcile so the next completion can resume; degrade to an empty
        // cache when the memory cannot be rolled back. M-RoPE media histories
        // legitimately hold fewer time positions than placeholder tokens.
        auto * kv = llama_get_memory(ctx->ctx);
        const llama_pos n_tokens = (llama_pos) embd.size();
        const llama_pos pos_max = llama_memory_seq_pos_max(kv, 0);
        bool resumable = pos_max + 1 == n_tokens ||
                         (rnllama::model_uses_mrope(ctx->model) && pos_max >= 0 &&
                          pos_max + 1 < n_tokens);
        if (!resumable && pos_max + 1 > n_tokens) {
            resumable = llama_memory_seq_rm(kv, 0, n_tokens, -1) &&
                        llama_memory_seq_pos_max(kv, 0) + 1 == n_tokens;
            if (resumable) {
                // SWA caches prune cells behind the attention window; after
                // rolling back, the window ending at n_tokens must be intact.
                // Recurrent/hybrid models are exempt (their pos_min reflects
                // the recurrent tail; seq_rm itself enforces their safety).
                const bool is_recurrent_or_hybrid =
                    llama_model_is_recurrent(ctx->model) || llama_model_is_hybrid(ctx->model);
                const int32_t n_swa = ctx->params.swa_full ? 0 : llama_model_n_swa(ctx->model);
                if (n_swa > 0 && !is_recurrent_or_hybrid) {
                    const llama_pos pos_min = llama_memory_seq_pos_min(kv, 0);
                    resumable = pos_min == 0 ||
                                (pos_min > 0 && pos_min < std::max<llama_pos>(0, n_tokens - n_swa));
                }
            }
        }
        if (!resumable) {
            llama_memory_seq_rm(kv, 0, 0, -1);
            embd.clear();
        }

        // Media identity for placeholder positions; absent for text-only or
        // legacy files (media is then conservatively reprocessed)
        ctx->setMediaHashes(embd.empty() ? std::vector<std::string>{}
                                         : rnllama::read_state_meta(path));

        // Placeholders are not vocab ids - drop them from the prompt string
        std::vector<llama_token> text_tokens;
        text_tokens.reserve(embd.size());
        std::copy_if(embd.begin(), embd.end(), std::back_inserter(text_tokens),
                     [](llama_token t) { return t != LLAMA_TOKEN_NULL; });
        const std::string text = rnllama::tokens_to_str(ctx->ctx, text_tokens.cbegin(), text_tokens.cend());

        jsi::Object result(runtime);
        result.setProperty(runtime, "tokens_loaded", (double)embd.size());
        result.setProperty(runtime, "prompt", jsi::String::createFromUtf8(runtime, text));
        return result;
    }

    inline int saveSession(rnllama::llama_rn_context* ctx, const std::string& path, int size) {
        if (!ctx || !ctx->completion) {
            throw std::runtime_error("Context or completion not initialized");
        }
        if (ctx->slot_manager != nullptr) {
            // The single-completion embd does not describe the parallel slots'
            // sequences; a whole-context save would be inconsistent
            throw std::runtime_error("Session save is not supported while parallel mode is enabled");
        }

        // Keep LLAMA_TOKEN_NULL media placeholders: llama_state_save_file
        // serializes the whole memory, so the token list must cover the same
        // positions or the file cannot be resumed
        std::vector<llama_token> session_tokens = ctx->completion->embd;

        int default_size = session_tokens.size();
        int save_size = size > 0 && size <= default_size ? size : default_size;

        if (!llama_state_save_file(ctx->ctx, path.c_str(), session_tokens.data(), save_size)) {
             throw std::runtime_error("Failed to save session");
        }

        // Persist media identity only when the saved prefix actually holds
        // media and no media position was cut off; anything else cannot be
        // verified on reload
        const bool media_retained =
            std::find(session_tokens.begin(), session_tokens.begin() + save_size,
                      LLAMA_TOKEN_NULL) != session_tokens.begin() + save_size &&
            std::find(session_tokens.begin() + save_size, session_tokens.end(),
                      LLAMA_TOKEN_NULL) == session_tokens.end();
        rnllama::write_state_meta(path, media_retained ? ctx->getMediaHashes()
                                                       : std::vector<std::string>{});

        return save_size;
    }
}
