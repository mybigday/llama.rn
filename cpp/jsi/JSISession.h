#pragma once

#include "JSIHelpers.h"
#include "JSINativeHeaders.h"
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

        size_t n_token_count_out = 0;
        ctx->completion->embd.resize(ctx->params.n_ctx);
        if (!llama_state_load_file(ctx->ctx, path.c_str(), ctx->completion->embd.data(), ctx->completion->embd.capacity(), &n_token_count_out)) {
             throw std::runtime_error("Failed to load session");
        }
        ctx->completion->embd.resize(n_token_count_out);
        
        auto null_token_iter = std::find(ctx->completion->embd.begin(), ctx->completion->embd.end(), LLAMA_TOKEN_NULL);
        if (null_token_iter != ctx->completion->embd.end()) {
            ctx->completion->embd.resize(std::distance(ctx->completion->embd.begin(), null_token_iter));
        }
        
        const std::string text = rnllama::tokens_to_str(ctx->ctx, ctx->completion->embd.cbegin(), ctx->completion->embd.cend());
        
        jsi::Object result(runtime);
        result.setProperty(runtime, "tokens_loaded", (double)n_token_count_out);
        result.setProperty(runtime, "prompt", jsi::String::createFromUtf8(runtime, text));
        return result;
    }

    inline int saveSession(rnllama::llama_rn_context* ctx, const std::string& path, int size) {
        if (!ctx || !ctx->completion) {
            throw std::runtime_error("Context or completion not initialized");
        }
        
        std::vector<llama_token> session_tokens = ctx->completion->embd;
        auto null_token_iter = std::find(session_tokens.begin(), session_tokens.end(), LLAMA_TOKEN_NULL);
        if (null_token_iter != session_tokens.end()) {
            session_tokens.resize(std::distance(session_tokens.begin(), null_token_iter));
        }
        
        int default_size = session_tokens.size();
        int save_size = size > 0 && size <= default_size ? size : default_size;
        
        if (!llama_state_save_file(ctx->ctx, path.c_str(), session_tokens.data(), save_size)) {
             throw std::runtime_error("Failed to save session");
        }
        return session_tokens.size();
    }
}
