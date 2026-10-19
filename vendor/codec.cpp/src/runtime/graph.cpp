#include "graph.h"
#include "perf_log.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <unordered_set>

static bool codec_graph_key_equal(const codec_graph_cache_key & a, const codec_graph_cache_key & b) {
    return a.kind == b.kind &&
           a.n_frames == b.n_frames &&
           a.n_q == b.n_q &&
           a.hop == b.hop &&
           a.n_in == b.n_in &&
           a.latent_dim == b.latent_dim;
}

struct codec_graph_count_state {
    std::unordered_set<const ggml_tensor *> visited;
    size_t n_nodes = 0;
    size_t n_leafs = 0;
};

static void codec_graph_count_visit(ggml_tensor * node, codec_graph_count_state * state) {
    if (node == nullptr || state == nullptr) {
        return;
    }

    if (!state->visited.insert(node).second) {
        return;
    }

    for (int i = 0; i < GGML_MAX_SRC; ++i) {
        codec_graph_count_visit(node->src[i], state);
    }

    if (node->op == GGML_OP_NONE && (node->flags & GGML_TENSOR_FLAG_PARAM) == 0) {
        ++state->n_leafs;
    } else {
        ++state->n_nodes;
    }
}

static codec_graph_count_state codec_graph_count_exact(ggml_tensor * out) {
    codec_graph_count_state state;
    codec_graph_count_visit(out, &state);
    return state;
}

size_t codec_graph_size_exact(
    const struct codec_model * /*model*/,
    const struct codec_graph_cache_key * /*key*/,
    const void * /*user_data*/,
    ggml_tensor * out) {

    const codec_graph_count_state state = codec_graph_count_exact(out);
    return std::max<size_t>(1, std::max(state.n_nodes, state.n_leafs));
}

// Context whose graph is being built (set around build_fn), so blocks can
// register constants without threading codec_context through every builder.
static thread_local codec_context * t_building_ctx = nullptr;

ggml_tensor * codec_graph_const(
        ggml_context * ctx_eval, const char * name, ggml_type type, int n_dims,
        const int64_t * ne, const void * data) {
    codec_context * ctx = t_building_ctx;
    if (ctx == nullptr || ctx_eval == nullptr || ctx_eval != ctx->eval_ctx || name == nullptr ||
            data == nullptr || n_dims < 1 || n_dims > GGML_MAX_DIMS) {
        return nullptr;
    }
    // Search the registered constants, not ctx_eval: ggml_get_tensor scans every
    // object of a graph that can hold 100k+ nodes.
    for (const codec_graph_const_input & c : ctx->eval_consts) {
        if (std::strcmp(ggml_get_name(c.tensor), name) == 0) {
            return c.tensor;
        }
    }
    ggml_tensor * t = ggml_new_tensor(ctx_eval, type, n_dims, ne);
    if (t == nullptr) {
        return nullptr;
    }
    ggml_set_name(t, name);
    ggml_set_input(t);
    codec_graph_const_input c;
    c.tensor = t;
    const uint8_t * src = static_cast<const uint8_t *>(data);
    c.data.assign(src, src + ggml_nbytes(t));
    ctx->eval_consts.push_back(std::move(c));
    return t;
}

ggml_tensor * codec_graph_const_f32(
        ggml_context * ctx_eval, const char * name, const std::vector<float> & data,
        int64_t ne0, int64_t ne1, int64_t ne2) {
    if ((int64_t) data.size() != ne0 * ne1 * ne2) {
        return nullptr;
    }
    const int64_t ne[3] = {ne0, ne1, ne2};
    const int n_dims = ne2 > 1 ? 3 : (ne1 > 1 ? 2 : 1);
    return codec_graph_const(ctx_eval, name, GGML_TYPE_F32, n_dims, ne, data.data());
}

ggml_tensor * codec_graph_pin_cpu(ggml_tensor * node) {
    if (t_building_ctx != nullptr && node != nullptr) {
        t_building_ctx->eval_cpu_nodes.push_back(node);
    }
    return node;
}

bool codec_graph_backend_is_gpu() {
    return t_building_ctx != nullptr && t_building_ctx->backend != nullptr &&
           ggml_backend_dev_type(ggml_backend_get_device(t_building_ctx->backend)) != GGML_BACKEND_DEVICE_TYPE_CPU;
}

const char * codec_graph_backend_reg_name() {
    if (t_building_ctx == nullptr || t_building_ctx->backend == nullptr) return "";
    ggml_backend_dev_t dev = ggml_backend_get_device(t_building_ctx->backend);
    return dev != nullptr ? ggml_backend_reg_name(ggml_backend_dev_backend_reg(dev)) : "";
}

ggml_tensor ** codec_graph_weight_cast_slot(ggml_context * ctx_eval, const ggml_tensor * w) {
    if (t_building_ctx == nullptr || ctx_eval != t_building_ctx->eval_ctx || w == nullptr) return nullptr;
    return &t_building_ctx->eval_weight_casts[w];
}

bool codec_graph_backend_supports(const ggml_tensor * node) {
#ifdef CODEC_DEBUG_TOOLS
    static const bool force_composite = [] {
        const char * v = std::getenv("CODEC_FORCE_COMPOSITE_OPS");
        return v != nullptr && v[0] != '\0' && v[0] != '0';
    }();
    if (force_composite) return false;
#endif
    if (t_building_ctx == nullptr || t_building_ctx->backend == nullptr || node == nullptr) return true;
    return ggml_backend_supports_op(t_building_ctx->backend, node);
}

static ggml_tensor * codec_graph_last_tensor(ggml_context * ctx) {
    ggml_tensor * last = nullptr;
    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
        last = t;
    }
    return last;
}

codec_graph_cpu_scope::codec_graph_cpu_scope(ggml_context * ctx_eval)
    : ctx(ctx_eval), last_before(codec_graph_last_tensor(ctx_eval)) {}

codec_graph_cpu_scope::~codec_graph_cpu_scope() {
    // A view must live on its source's backend: pin views only of pinned tensors.
    std::unordered_set<const ggml_tensor *> pinned;
    ggml_tensor * t = last_before != nullptr ? ggml_get_next_tensor(ctx, last_before) : ggml_get_first_tensor(ctx);
    for (; t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
        if (t->op == GGML_OP_NONE) {
            continue;
        }
        if (t->view_src != nullptr && pinned.count(t->view_src) == 0) {
            continue;
        }
        pinned.insert(t);
        codec_graph_pin_cpu(t);
    }
}

void codec_graph_release(codec_context * ctx) {
    if (ctx == nullptr) {
        return;
    }

    // The scheduler retains pointers into an allocated graph, so release its
    // graph state while the eval context is still alive. A scheduler reset is
    // sufficient for most graphs, but Metal can retain stale allocations for
    // Chatterbox S3G's large decode graph and return zeroes on the next decode.
    // Recreate that scheduler when codec_sched_ensure_capacity is called again.
    if (ctx->sched != nullptr && ctx->eval_graph_allocated) {
        if (ctx->eval_entry != nullptr &&
            ctx->eval_entry->key.kind == CODEC_GRAPH_CHATTERBOX_S3G_DECODE) {
            ggml_backend_sched_free(ctx->sched);
            ctx->sched = nullptr;
            ctx->sched_reserved_graph_size = 0;
        } else {
            ggml_backend_sched_reset(ctx->sched);
        }
    }

    if (ctx->eval_ctx != nullptr) {
        ggml_free(ctx->eval_ctx);
        ctx->eval_ctx = nullptr;
    }
    ctx->eval_graph = nullptr;
    ctx->eval_output = nullptr;
    ctx->eval_entry = nullptr;
    ctx->eval_consts.clear();
    ctx->eval_cpu_nodes.clear();
    ctx->eval_weight_casts.clear();
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

// Default arena size for ggml metadata.  ggml_init in no_alloc mode
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

    ggml_init_params p = {
        /*.mem_size   =*/ ctx->eval_arena_size,
        /*.mem_buffer =*/ ctx->eval_arena_buf,
        /*.no_alloc   =*/ true,
    };

    ctx->eval_ctx = ggml_init(p);
    if (ctx->eval_ctx == nullptr) {
        if (error != nullptr) {
            *error = "failed to create eval context";
        }
        codec_graph_release(ctx);
        return false;
    }

    ggml_tensor * out = nullptr;
    void * build_data = cached->build_user_data.empty() ? nullptr : cached->build_user_data.data();
    t_building_ctx = ctx;
    const bool built = cached->build_fn(ctx->eval_ctx, build_data, &out);
    t_building_ctx = nullptr;
    if (!built || out == nullptr) {
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
    // ggml_set_output explicitly inside their build_fn (see snac.cpp).
    for (ggml_tensor * t = ggml_get_first_tensor(ctx->eval_ctx); t != nullptr;
         t = ggml_get_next_tensor(ctx->eval_ctx, t)) {
        if (t->op == GGML_OP_NONE && t->buffer == nullptr && t->view_src == nullptr) {
            ggml_set_input(t);
        }
    }
    ggml_set_output(out);

    size_t graph_size = 0;
    if (ctx->model != nullptr && ctx->model->vtable != nullptr && ctx->model->vtable->graph_size != nullptr) {
        graph_size = ctx->model->vtable->graph_size(ctx->model, &cached->key, build_data, out);
    }

    // Side-output tensors (extra ggml_set_output calls inside build_fn,
    // not reachable from `out`) need to be expanded into the graph as
    // separate roots — without that, sched_alloc_graph never sees them
    // and their `t->buffer` stays NULL, making the runtime fail on
    // codec_runtime_read_tensor.  Walk eval_ctx, collect them.
    std::vector<ggml_tensor *> side_outputs;
    for (ggml_tensor * t = ggml_get_first_tensor(ctx->eval_ctx); t != nullptr;
         t = ggml_get_next_tensor(ctx->eval_ctx, t)) {
        if ((t->flags & GGML_TENSOR_FLAG_OUTPUT) && t != out) {
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
    for (ggml_tensor * t : side_outputs) {
        codec_graph_count_visit(t, &counts);
    }
    // ggml's internal hash set sizes from this; sum (rather than max)
    // of nodes + leafs is the correct bound for what ggml_visit_parents
    // will insert.  Take the larger of (vtable hint, side-output-aware
    // count) — the vtable hint typically only walks from `out` and so
    // undercounts when build_fn flagged extra ggml_set_output side
    // branches (e.g. parallel_heads_delay's N parallel logit roots).
    {
        const size_t side_aware = std::max<size_t>(1, counts.n_nodes + counts.n_leafs);
        if (graph_size < side_aware) {
            graph_size = side_aware;
        }
    }
    ctx->eval_graph = ggml_new_graph_custom(ctx->eval_ctx, graph_size, false);
    ggml_build_forward_expand(ctx->eval_graph, out);
    for (ggml_tensor * t : side_outputs) {
        ggml_build_forward_expand(ctx->eval_graph, t);
    }
    ctx->eval_output = out;
    ctx->eval_entry = cached;

    // Ratchet up the arena ceiling if this graph needed more than the default
    // — the next call for this entry will allocate enough on its own.  We
    // never shrink: the eval arena is reused across cache entries and a
    // realloc-down would just churn since the next entry might want the
    // larger size again.
    const size_t used_mem = ggml_used_mem(ctx->eval_ctx);
    {
        char det[96];
        std::snprintf(det, sizeof(det), "kind=%d meta_bytes=%zu nodes=%d", key.kind, used_mem, (int) counts.n_nodes);
        codec_perf_event("graph_meta", det);
    }
    if (used_mem > cached->required_mem_size) {
        cached->required_mem_size = used_mem;
    }
    cached->last_graph_size = (int32_t) graph_size;
    cached->last_sched_graph_size = (int32_t) std::max<size_t>(1, counts.n_nodes + counts.n_leafs);
    *out_entry = cached;
    return true;
}

ggml_tensor * codec_graph_get_tensor(codec_context * ctx, codec_graph_cache_entry * entry, const char * name) {
    if (ctx == nullptr || entry == nullptr || name == nullptr || ctx->eval_entry != entry || ctx->eval_ctx == nullptr) {
        return nullptr;
    }
    return ggml_get_tensor(ctx->eval_ctx, name);
}
