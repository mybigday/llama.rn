#include "graph.h"

#include <algorithm>
#include <array>
#include <cstdlib>

static bool codec_graph_key_equal(const codec_graph_cache_key & a, const codec_graph_cache_key & b) {
    return a.kind == b.kind &&
           a.n_frames == b.n_frames &&
           a.n_q == b.n_q &&
           a.hop == b.hop &&
           a.n_in == b.n_in &&
           a.latent_dim == b.latent_dim;
}

void codec_graph_release(codec_context * ctx) {
    if (ctx == nullptr) {
        return;
    }
    if (ctx->eval_ctx != nullptr) {
        lm_ggml_free(ctx->eval_ctx);
        ctx->eval_ctx = nullptr;
    }
    ctx->eval_graph = nullptr;
    ctx->eval_output = nullptr;
    ctx->eval_entry = nullptr;
    ctx->eval_alloc_entry = nullptr;
}

static bool codec_graph_ensure_eval_arena(codec_context * ctx, size_t required_size, std::string * error) {
    if (ctx == nullptr) {
        if (error != nullptr) {
            *error = "invalid eval arena context";
        }
        return false;
    }

    const size_t min_size = required_size > 0 ? required_size : 1024;
    if (ctx->eval_arena_size >= min_size && ctx->eval_arena_buf != nullptr) {
        return true;
    }

    void * new_buf = std::realloc(ctx->eval_arena_buf, min_size);
    if (new_buf == nullptr) {
        if (error != nullptr) {
            *error = "failed to grow eval arena, required size: " + std::to_string(required_size);
        }
        return false;
    }

    ctx->eval_arena_buf = new_buf;
    ctx->eval_arena_size = min_size;
    return true;
}

bool codec_graph_cache_get_or_build(
    codec_context * ctx,
    codec_graph_cache_key key,
    size_t mem_size,
    codec_graph_build_fn build_fn,
    const void * user_data,
    size_t user_data_size,
    codec_graph_cache_entry ** out_entry,
    std::string * error) {

    if (ctx == nullptr || build_fn == nullptr || out_entry == nullptr) {
        if (error != nullptr) {
            *error = "invalid graph cache arguments";
        }
        return false;
    }

    if (user_data_size > 0 && user_data == nullptr) {
        if (error != nullptr) {
            *error = "missing graph builder user_data";
        }
        return false;
    }

    codec_graph_release(ctx);

    codec_graph_cache_entry * cached = nullptr;
    for (codec_graph_cache_entry & entry : ctx->graph_cache) {
        if (codec_graph_key_equal(entry.key, key)) {
            cached = &entry;
            break;
        }
    }

    if (cached == nullptr) {
        codec_graph_cache_entry entry;
        entry.key = key;
        entry.required_mem_size = mem_size;
        entry.build_fn = build_fn;
        entry.last_graph_size = 0;
        entry.allocated = false;
        if (user_data_size > 0) {
            const uint8_t * src = static_cast<const uint8_t *>(user_data);
            entry.build_user_data.assign(src, src + user_data_size);
        }
        ctx->graph_cache.push_back(entry);
        cached = &ctx->graph_cache.back();
    } else {
        cached->required_mem_size = std::max(cached->required_mem_size, mem_size);
    }

    if (!codec_graph_ensure_eval_arena(ctx, cached->required_mem_size, error)) {
        return false;
    }

    lm_ggml_init_params p = {
        /*.mem_size   =*/ ctx->eval_arena_size,
        /*.mem_buffer =*/ ctx->eval_arena_buf,
        /*.no_alloc   =*/ true,
    };

    ctx->eval_ctx = lm_ggml_init(p);
    if (ctx->eval_ctx == nullptr) {
        if (error != nullptr) {
            *error = "failed to create eval context";
        }
        codec_graph_release(ctx);
        return false;
    }

    lm_ggml_tensor * out = nullptr;
    void * build_data = cached->build_user_data.empty() ? nullptr : cached->build_user_data.data();
    if (!cached->build_fn(ctx->eval_ctx, build_data, &out) || out == nullptr) {
        if (error != nullptr) {
            *error = "failed to build graph";
        }
        codec_graph_release(ctx);
        return false;
    }

    size_t graph_size = LM_GGML_DEFAULT_GRAPH_SIZE;
    if (key.kind == CODEC_GRAPH_WT_ENCODE || key.kind == CODEC_GRAPH_WT_DECODE) {
        graph_size = LM_GGML_DEFAULT_GRAPH_SIZE * 32;
    } else if (key.kind == CODEC_GRAPH_Q3T_DECODE) {
        graph_size = LM_GGML_DEFAULT_GRAPH_SIZE * 64;
    }

    ctx->eval_graph = lm_ggml_new_graph_custom(ctx->eval_ctx, graph_size, false);
    lm_ggml_build_forward_expand(ctx->eval_graph, out);
    ctx->eval_output = out;
    ctx->eval_entry = cached;

    cached->required_mem_size = std::max(cached->required_mem_size, lm_ggml_used_mem(ctx->eval_ctx));
    cached->last_graph_size = lm_ggml_graph_n_nodes(ctx->eval_graph);
    *out_entry = cached;
    return true;
}

lm_ggml_tensor * codec_graph_get_tensor(codec_context * ctx, codec_graph_cache_entry * entry, const char * name) {
    if (ctx == nullptr || entry == nullptr || name == nullptr || ctx->eval_entry != entry || ctx->eval_ctx == nullptr) {
        return nullptr;
    }
    return lm_ggml_get_tensor(ctx->eval_ctx, name);
}
