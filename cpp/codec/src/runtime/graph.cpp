#include "graph.h"
#include "perf_log.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>

static bool codec_graph_key_equal(const codec_graph_cache_key & a, const codec_graph_cache_key & b) {
    return a.kind == b.kind &&
           a.n_frames == b.n_frames &&
           a.n_q == b.n_q &&
           a.hop == b.hop &&
           a.n_in == b.n_in &&
           a.latent_dim == b.latent_dim;
}

struct codec_graph_count_state {
    std::vector<lm_ggml_tensor *> visited;
    size_t n_nodes = 0;
    size_t n_leafs = 0;
};

static void codec_graph_count_visit(lm_ggml_tensor * node, codec_graph_count_state * state) {
    if (node == nullptr || state == nullptr) {
        return;
    }

    if (std::find(state->visited.begin(), state->visited.end(), node) != state->visited.end()) {
        return;
    }
    state->visited.push_back(node);

    for (int i = 0; i < LM_GGML_MAX_SRC; ++i) {
        codec_graph_count_visit(node->src[i], state);
    }

    if (node->op == LM_GGML_OP_NONE && (node->flags & LM_GGML_TENSOR_FLAG_PARAM) == 0) {
        ++state->n_leafs;
    } else {
        ++state->n_nodes;
    }
}

static codec_graph_count_state codec_graph_count_exact(lm_ggml_tensor * out) {
    codec_graph_count_state state;
    codec_graph_count_visit(out, &state);
    return state;
}

size_t codec_graph_size_exact(
    const struct codec_model * /*model*/,
    const struct codec_graph_cache_key * /*key*/,
    const void * /*user_data*/,
    lm_ggml_tensor * out) {

    const codec_graph_count_state state = codec_graph_count_exact(out);
    return std::max<size_t>(1, std::max(state.n_nodes, state.n_leafs));
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
    ctx->eval_graph_allocated = false;
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

// Default arena size for ggml metadata.  lm_ggml_init in no_alloc mode
// pre-allocates this many bytes for the eval context's tensor / op /
// graph metadata; the actual data lives in the scheduler's galloc-managed
// buffer.  Largest observed usage across all models is chatterbox_s3g's
// 48 MB unrolled-CFM graph (2026-05-07), so 128 MB has ~2.5× headroom.
// The arena is mostly virtual: pages are committed on touch, so an
// oversized default doesn't materially affect RSS — but it bloats VIRT
// and is wasteful if a future backend pre-faults arena pages.
static constexpr size_t kCodecGraphDefaultArenaBytes = (size_t) 128 * 1024 * 1024;

bool codec_graph_cache_get_or_build(
    codec_context * ctx,
    codec_graph_cache_key key,
    codec_graph_build_fn build_fn,
    const void * user_data,
    size_t user_data_size,
    codec_graph_cache_entry ** out_entry,
    std::string * error) {

    char detail_buf[96];
    std::snprintf(detail_buf, sizeof(detail_buf),
                  "kind=%d n_frames=%d n_q=%d n_in=%d",
                  key.kind, key.n_frames, key.n_q, key.n_in);
    CODEC_PERF_SCOPE_D("graph_build", detail_buf);

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

    codec_graph_cache_entry * cached = nullptr;
    for (codec_graph_cache_entry & entry : ctx->graph_cache) {
        if (codec_graph_key_equal(entry.key, key)) {
            cached = &entry;
            break;
        }
    }

    // Consecutive-call fast path: if the resolved entry is the exact same entry
    // built by the immediately previous call, its eval graph is still alive and
    // galloc-allocated, and the incoming user_data bytes are byte-identical to
    // the stored build_user_data, then the previously built graph, its tensors,
    // and the galloc allocation are all still valid.  Skip the rebuild + the
    // galloc re-plan entirely and reuse them; codec_graph_get_tensor still
    // resolves names via the live eval_ctx.  This only triggers for back-to-back
    // identical (entry, user_data) calls — any intervening call to a different
    // entry sets ctx->eval_entry != cached and forces the slow path (the safety
    // gate that prevents dangling galloc allocations).
    if (cached != nullptr && ctx->eval_entry == cached && ctx->eval_ctx != nullptr &&
        ctx->eval_graph != nullptr && ctx->eval_graph_allocated) {
        const size_t nbytes = cached->build_user_data.size();
        const bool same_bytes =
            (user_data_size == nbytes) &&
            (nbytes == 0 || std::memcmp(cached->build_user_data.data(), user_data, nbytes) == 0);
        if (same_bytes) {
            *out_entry = cached;
            return true;
        }
    }

    // Slow path: a rebuild is actually needed.  Release the previously built
    // eval graph now (this also clears eval_graph_allocated so codec_graph_prepare_io
    // re-plans galloc for the new graph).  Doing the release here — rather than
    // unconditionally at function entry — is what lets the fast path above keep
    // the prior allocation alive.
    codec_graph_release(ctx);

    if (cached == nullptr) {
        codec_graph_cache_entry entry;
        entry.key = key;
        entry.required_mem_size = kCodecGraphDefaultArenaBytes;
        entry.build_fn = build_fn;
        entry.last_graph_size = 0;
        entry.last_sched_graph_size = 0;
        if (user_data_size > 0) {
            const uint8_t * src = static_cast<const uint8_t *>(user_data);
            entry.build_user_data.assign(src, src + user_data_size);
        }
        ctx->graph_cache.push_back(entry);
        cached = &ctx->graph_cache.back();
    } else if (user_data_size > 0) {
        // Existing entry, but the user_data bytes differ (or were never stored):
        // refresh the stored copy so the rebuilt graph uses the current inputs
        // and the next fast-path comparison is against the right bytes.  (Fixes
        // a latent stale-user_data hazard: build_fn reads build_user_data.data().)
        const uint8_t * src = static_cast<const uint8_t *>(user_data);
        cached->build_user_data.assign(src, src + user_data_size);
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

    // Mark every unallocated leaf as a graph input so galloc keeps it persistent
    // and we can write to it via codec_runtime_write_tensor between alloc and
    // compute.  Model weights live in a different context with buffers already
    // attached, so they're skipped naturally.  Also mark every named non-leaf as
    // output: the codebase convention is to name only tensors that the runtime
    // reads back via codec_graph_get_tensor + codec_runtime_read_tensor, and
    // without OUTPUT flagging galloc reuses their buffers after their last
    // graph use (e.g. snac's three RVQ code tensors).
    // Auto-flag inputs only: leafs without a buffer are tensors the runtime
    // will write to via codec_runtime_write_tensor.  Auto-flagging outputs by
    // "non-leaf with a non-empty name" is too coarse — ggml internally names
    // many op outputs (views, conts, reshapes, etc.) after their source
    // tensor, so the heuristic flags hundreds of intermediate tensors and
    // bloats the persistent buffer (xcodec2 encode: 1340 names → 5.8 GB
    // pinned).  Models that read non-terminal outputs must call
    // lm_ggml_set_output explicitly inside their build_fn (see snac.cpp).
    for (lm_ggml_tensor * t = lm_ggml_get_first_tensor(ctx->eval_ctx); t != nullptr;
         t = lm_ggml_get_next_tensor(ctx->eval_ctx, t)) {
        if (t->op == LM_GGML_OP_NONE && t->buffer == nullptr && t->view_src == nullptr) {
            lm_ggml_set_input(t);
        }
    }
    lm_ggml_set_output(out);

    size_t graph_size = 0;
    if (ctx->model != nullptr && ctx->model->vtable != nullptr && ctx->model->vtable->graph_size != nullptr) {
        graph_size = ctx->model->vtable->graph_size(ctx->model, &cached->key, build_data, out);
    }

    // Side-output tensors (extra lm_ggml_set_output calls inside build_fn,
    // not reachable from `out`) need to be expanded into the graph as
    // separate roots — without that, sched_alloc_graph never sees them
    // and their `t->buffer` stays NULL, making the runtime fail on
    // codec_runtime_read_tensor.  Walk eval_ctx, collect them.
    std::vector<lm_ggml_tensor *> side_outputs;
    for (lm_ggml_tensor * t = lm_ggml_get_first_tensor(ctx->eval_ctx); t != nullptr;
         t = lm_ggml_get_next_tensor(ctx->eval_ctx, t)) {
        if ((t->flags & LM_GGML_TENSOR_FLAG_OUTPUT) && t != out) {
            side_outputs.push_back(t);
        }
    }

    // Count the union of nodes/leafs reachable from `out` AND every
    // side-output, sharing one visited set so shared sub-DAG (e.g.
    // h_in feeding all N parallel heads) is counted once.  This walks
    // through external `src` pointers too (e.g. weights in model->weights
    // referenced from eval_ctx ops), which ggml's graph builder also
    // visits — so the count matches the actual graph hash-set demand.
    codec_graph_count_state counts;
    codec_graph_count_visit(out, &counts);
    for (lm_ggml_tensor * t : side_outputs) {
        codec_graph_count_visit(t, &counts);
    }
    // ggml's internal hash set sizes from this; sum (rather than max)
    // of nodes + leafs is the correct bound for what lm_ggml_visit_parents
    // will insert.  Take the larger of (vtable hint, side-output-aware
    // count) — the vtable hint typically only walks from `out` and so
    // undercounts when build_fn flagged extra lm_ggml_set_output side
    // branches (e.g. parallel_heads_delay's N parallel logit roots).
    {
        const size_t side_aware = std::max<size_t>(1, counts.n_nodes + counts.n_leafs);
        if (graph_size < side_aware) {
            graph_size = side_aware;
        }
    }
    ctx->eval_graph = lm_ggml_new_graph_custom(ctx->eval_ctx, graph_size, false);
    lm_ggml_build_forward_expand(ctx->eval_graph, out);
    for (lm_ggml_tensor * t : side_outputs) {
        lm_ggml_build_forward_expand(ctx->eval_graph, t);
    }
    ctx->eval_output = out;
    ctx->eval_entry = cached;

    // Ratchet up the arena ceiling if this graph needed more than the default
    // — the next call for this entry will allocate enough on its own.  We
    // never shrink: the eval arena is reused across cache entries and a
    // realloc-down would just churn since the next entry might want the
    // larger size again.
    const size_t used_mem = lm_ggml_used_mem(ctx->eval_ctx);
    if (used_mem > cached->required_mem_size) {
        cached->required_mem_size = used_mem;
    }
    cached->last_graph_size = (int32_t) graph_size;
    cached->last_sched_graph_size = (int32_t) std::max<size_t>(1, counts.n_nodes + counts.n_leafs);
    *out_entry = cached;
    return true;
}

lm_ggml_tensor * codec_graph_get_tensor(codec_context * ctx, codec_graph_cache_entry * entry, const char * name) {
    if (ctx == nullptr || entry == nullptr || name == nullptr || ctx->eval_entry != entry || ctx->eval_ctx == nullptr) {
        return nullptr;
    }
    return lm_ggml_get_tensor(ctx->eval_ctx, name);
}
