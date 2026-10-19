#include "graph_exec.h"

#include "perf_log.h"

#include <ggml-alloc.h>
#include <ggml.h>

#if defined(__linux__)
#include <sched.h>
#include <unistd.h>
#endif

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#ifdef CODEC_DEBUG_TOOLS
// ---------------------------------------------------------------------------
// Per-node profiler (CODEC_DEBUG_TOOLS builds; active when CODEC_OP_PROFILE=<path>).
//
// Registers a ggml_backend_sched eval callback that timestamps successive node
// observation callbacks and accumulates wall time + call count per
// (op, shape-class).  The scheduler invokes the callback per resulting node
// twice: ask=true (batching query — we always answer yes so every node is
// observed) then ask=false (the observation, fired *after* the node computed).
//
// We attribute the interval between two consecutive ask=false callbacks to the
// node observed at the *end* of that interval (its result just became ready),
// which is the standard "delta since previous observe" accounting.  The first
// node's interval also carries the compute-start overhead (split setup, input
// copies), a negligible edge effect over thousands of nodes.
//
// A summary is appended to the file at the end of each compute.
// ---------------------------------------------------------------------------
namespace {

struct op_profile_bucket {
    int64_t total_us = 0;
    int64_t count    = 0;
};

struct op_profiler {
    const char *                          path      = nullptr;
    bool                                  resolved  = false;
    bool                                  active    = false;   // inside one compute
    int64_t                               t_prev_us = 0;
    std::map<std::string, op_profile_bucket> buckets;

    const char * resolve() {
        if (!resolved) {
            path = std::getenv("CODEC_OP_PROFILE");
            if (path != nullptr && path[0] == '\0') path = nullptr;
            resolved = true;
        }
        return path;
    }
};

op_profiler & profiler() {
    static op_profiler p;
    return p;
}

// Coarse shape-class key: op name + inner-two dims of the result, so distinct
// matmul sizes (e.g. h_dit×T vs qkv) bucket separately without exploding.
std::string profile_key(const ggml_tensor * t) {
    char buf[96];
    std::snprintf(buf, sizeof(buf), "%-16s [%lld,%lld,%lld,%lld]",
                  ggml_op_name(t->op),
                  (long long) t->ne[0], (long long) t->ne[1],
                  (long long) t->ne[2], (long long) t->ne[3]);
    return std::string(buf);
}

// The interval since the previous observation is the compute of `node` (its
// result just became ready).
void profile_observe(ggml_tensor * node) {
    op_profiler & p = profiler();
    const int64_t now = ggml_time_us();
    op_profile_bucket & b = p.buckets[profile_key(node)];
    b.total_us += now - p.t_prev_us;
    b.count    += 1;
    p.t_prev_us = now;
}

bool profile_eval_callback(ggml_tensor * t, bool ask, void * /*user_data*/) {
    if (ask) {
        return true;  // observe every node
    }
    profile_observe(t);
    return true;
}

void profile_begin() {
    op_profiler & p = profiler();
    if (p.resolve() == nullptr) { p.active = false; return; }
    p.active    = true;
    p.t_prev_us = ggml_time_us();
}

void profile_end(int graph_kind, int n_nodes) {
    op_profiler & p = profiler();
    if (!p.active) return;

    // Aggregate by op-name (strip the shape suffix) for a compact top-level view,
    // then dump the detailed per-shape buckets.
    std::map<std::string, op_profile_bucket> by_op;
    int64_t grand_us = 0;
    for (auto & kv : p.buckets) {
        grand_us += kv.second.total_us;
        std::string op = kv.first.substr(0, kv.first.find(' '));
        op_profile_bucket & b = by_op[op];
        b.total_us += kv.second.total_us;
        b.count    += kv.second.count;
    }

    FILE * f = std::fopen(p.path, "ab");
    if (f == nullptr) { p.buckets.clear(); return; }
    std::fprintf(f, "==== op_profile kind=%d nodes=%d total_us=%lld ====\n",
                 graph_kind, n_nodes, (long long) grand_us);
    // sort by-op view by total_us desc
    std::vector<std::pair<std::string, op_profile_bucket>> ops(by_op.begin(), by_op.end());
    std::sort(ops.begin(), ops.end(),
              [](const auto & a, const auto & b) { return a.second.total_us > b.second.total_us; });
    std::fprintf(f, "-- by op --\n");
    for (auto & kv : ops) {
        std::fprintf(f, "  %-16s us=%-10lld cnt=%-6lld pct=%5.1f\n",
                     kv.first.c_str(), (long long) kv.second.total_us,
                     (long long) kv.second.count,
                     grand_us ? 100.0 * (double) kv.second.total_us / (double) grand_us : 0.0);
    }
    // detailed per-shape buckets, sorted by total_us desc
    std::vector<std::pair<std::string, op_profile_bucket>> det(p.buckets.begin(), p.buckets.end());
    std::sort(det.begin(), det.end(),
              [](const auto & a, const auto & b) { return a.second.total_us > b.second.total_us; });
    std::fprintf(f, "-- by op+shape (top 40) --\n");
    int shown = 0;
    for (auto & kv : det) {
        if (shown++ >= 40) break;
        std::fprintf(f, "  %-40s us=%-10lld cnt=%-6lld pct=%5.1f\n",
                     kv.first.c_str(), (long long) kv.second.total_us,
                     (long long) kv.second.count,
                     grand_us ? 100.0 * (double) kv.second.total_us / (double) grand_us : 0.0);
    }
    std::fclose(f);
    p.buckets.clear();
}

}  // namespace
#endif // CODEC_DEBUG_TOOLS

static bool codec_backend_is_cpu(ggml_backend_t backend) {
    if (backend == nullptr) {
        return false;
    }
    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    if (dev == nullptr) {
        return false;
    }
    return ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU;
}

static ggml_backend_t codec_get_default_backend(codec_context * ctx) {
    // If main backend is CPU, use it; otherwise use CPU backend if available
    if (ctx->backend != nullptr && codec_backend_is_cpu(ctx->backend)) {
        return ctx->backend;
    }
    return ctx->cpu_backend;
}

static void codec_backend_set_n_threads(ggml_backend_t backend, int32_t n_threads) {
    if (backend == nullptr || n_threads <= 0) {
        return;
    }
    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    if (dev == nullptr) {
        return;
    }
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    if (reg == nullptr) {
        return;
    }
    ggml_backend_set_n_threads_t fn = reinterpret_cast<ggml_backend_set_n_threads_t>(
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads"));
    if (fn != nullptr) {
        fn(backend, n_threads);
    }
}

// The CPU registry's threadpool entry points (resolved once; null without a CPU backend).
struct codec_cpu_threadpool_fns {
    ggml_threadpool_t (*make)(struct ggml_threadpool_params *) = nullptr;
    void (*destroy)(ggml_threadpool_t) = nullptr;
    void (*attach)(ggml_backend_t, ggml_threadpool_t) = nullptr;
};

static const codec_cpu_threadpool_fns & codec_cpu_threadpool_api() {
    static const codec_cpu_threadpool_fns fns = [] {
        codec_cpu_threadpool_fns f;
        ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        ggml_backend_reg_t reg = dev != nullptr ? ggml_backend_dev_backend_reg(dev) : nullptr;
        if (reg != nullptr) {
            f.make = reinterpret_cast<decltype(f.make)>(ggml_backend_reg_get_proc_address(reg, "ggml_threadpool_new"));
            f.destroy = reinterpret_cast<decltype(f.destroy)>(ggml_backend_reg_get_proc_address(reg, "ggml_threadpool_free"));
            f.attach = reinterpret_cast<decltype(f.attach)>(
                ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_set_threadpool"));
        }
        if (f.make == nullptr || f.destroy == nullptr || f.attach == nullptr) f = codec_cpu_threadpool_fns{};
        return f;
    }();
    return fns;
}

// The pool's workers inherit the CPU affinity of the thread that creates them,
// and they live as long as the model. A caller whose mask fits the pool (an
// embedder keeping compute on its big cores) passes it on unchanged; one pinned
// narrower than the pool (e.g. by a strict llama.cpp threadpool) would leave
// every worker sharing those cores, so on Linux / Android the pool is then
// created from a fresh thread widened to every CPU. The caller's own mask is
// never touched.
static ggml_threadpool_t codec_make_unpinned_threadpool(const codec_cpu_threadpool_fns & api,
                                                        struct ggml_threadpool_params tpp) {
#if defined(__linux__)
    cpu_set_t caller;
    CPU_ZERO(&caller);
    if (sched_getaffinity(0, sizeof(caller), &caller) != 0 || CPU_COUNT(&caller) >= tpp.n_threads) {
        return api.make(&tpp);
    }
    ggml_threadpool_t tp = nullptr;
    std::thread maker([&] {
        cpu_set_t all;
        CPU_ZERO(&all);
        const long n_cpu = sysconf(_SC_NPROCESSORS_CONF);
        for (long i = 0; i < n_cpu && i < CPU_SETSIZE; ++i) CPU_SET((int) i, &all);
        sched_setaffinity(0, sizeof(all), &all);   // best effort: the cpuset may cap it
        tp = api.make(&tpp);
    });
    maker.join();
    return tp;
#else
    return api.make(&tpp);
#endif
}

// Attach the model's persistent CPU threadpool (grown to n_threads) to the
// context's CPU backends.
static void codec_attach_cpu_threadpool(codec_context * ctx, int32_t n_threads) {
    const codec_cpu_threadpool_fns & api = codec_cpu_threadpool_api();
    codec_model * model = ctx->model;
    if (api.make == nullptr || model == nullptr || n_threads <= 1) {
        return;
    }
    if (model->cpu_threadpool == nullptr || model->cpu_threadpool_n < n_threads) {
        ggml_threadpool_t tp = codec_make_unpinned_threadpool(api, ggml_threadpool_params_default(n_threads));
        if (tp == nullptr) {
            return;
        }
        if (model->cpu_threadpool != nullptr) {
            api.destroy(model->cpu_threadpool);   // contexts re-attach before their next compute
        }
        model->cpu_threadpool = tp;
        model->cpu_threadpool_n = n_threads;
    }
    for (ggml_backend_t b : {ctx->backend, ctx->cpu_backend}) {
        if (codec_backend_is_cpu(b)) {
            api.attach(b, model->cpu_threadpool);
        }
    }
}

void codec_model_free_cpu_threadpool(codec_model * model) {
    if (model != nullptr && model->cpu_threadpool != nullptr) {
        codec_cpu_threadpool_api().destroy(model->cpu_threadpool);
        model->cpu_threadpool = nullptr;
        model->cpu_threadpool_n = 0;
    }
}

static bool codec_sched_ensure_capacity(codec_context * ctx, int32_t required, std::string * error) {
    if (ctx == nullptr || ctx->backend == nullptr) {
        if (error != nullptr) {
            *error = "invalid scheduler context";
        }
        return false;
    }
    if (required <= 0) {
        required = 1;
    }

    const size_t target = (size_t) required;
    if ((int32_t) target <= ctx->sched_reserved_graph_size && ctx->sched != nullptr) {
        return true;
    }

    if (ctx->sched != nullptr) {
        ggml_backend_sched_free(ctx->sched);
        ctx->sched = nullptr;
    }

    std::array<ggml_backend_t, 2> backends = { ctx->backend, nullptr };
    int n_backends = 1;
    if (!codec_backend_is_cpu(ctx->backend) && ctx->cpu_backend != nullptr) {
        backends[1] = ctx->cpu_backend;
        n_backends = 2;
    }

#ifdef CODEC_DEBUG_TOOLS
    const bool op_offload = std::getenv("CODEC_NO_OP_OFFLOAD") == nullptr;
#else
    const bool op_offload = true;
#endif
    ctx->sched = ggml_backend_sched_new(backends.data(), nullptr, n_backends, target, false, op_offload);
    if (ctx->sched == nullptr) {
        if (error != nullptr) {
            *error = "failed to recreate backend scheduler";
        }
        return false;
    }

    ctx->sched_reserved_graph_size = (int32_t) target;
    return true;
}

bool codec_runtime_init(codec_context * ctx, std::string * error) {
    if (ctx == nullptr || ctx->model == nullptr) {
        if (error != nullptr) {
            *error = "invalid runtime init arguments";
        }
        return false;
    }

    ctx->backend = ctx->model->backend;
    ctx->eval_arena_buf = nullptr;
    ctx->eval_arena_size = 0;
    ctx->eval_ctx = nullptr;
    ctx->eval_graph = nullptr;
    ctx->eval_output = nullptr;
    ctx->eval_entry = nullptr;
    ctx->sched_reserved_graph_size = 0;
    if (ctx->backend == nullptr) {
        if (error != nullptr) {
            *error = "model backend is null";
        }
        return false;
    }

    std::array<ggml_backend_t, 2> backends = { ctx->backend, nullptr };
    int n_backends = 1;

    if (!codec_backend_is_cpu(ctx->backend)) {
        ctx->cpu_backend = ggml_backend_init_by_name("CPU", nullptr);
        if (ctx->cpu_backend != nullptr) {
            backends[1] = ctx->cpu_backend;
            n_backends = 2;
        }
    }

    return true;
}

bool codec_graph_prepare_io(
    codec_context * ctx,
    codec_graph_cache_entry * entry,
    std::string * error) {
    CODEC_PERF_SCOPE("graph_prepare_io");
    if (ctx == nullptr || entry == nullptr || ctx->eval_entry != entry || ctx->eval_graph == nullptr || ctx->backend == nullptr) {
        if (error != nullptr) {
            *error = "invalid graph prepare arguments";
        }
        return false;
    }

    // The scheduler must be sized large enough for this graph before we can
    // alloc into it.  galloc-managed memory reuse depends on the topological
    // analysis happening on the same graph that we're about to compute, so we
    // alloc here (after build, before write_tensor) and compute later.
    int32_t required = entry->last_sched_graph_size;
    if (required <= 0) {
        required = std::max(1, ggml_graph_n_nodes(ctx->eval_graph));
    }
    if (!codec_sched_ensure_capacity(ctx, required, error)) {
        return false;
    }

    if (ctx->eval_graph_allocated) {
        // Consecutive-call fast path (same graph, same allocation): the graph
        // constants are leaf inputs whose buffers galloc may hand to later
        // nodes after their last use, so write them again before each compute.
        for (const codec_graph_const_input & c : ctx->eval_consts) {
            ggml_backend_tensor_set(c.tensor, c.data.data(), 0, c.data.size());
        }
        return true;
    }

    ggml_backend_sched_reset(ctx->sched);
    if (ctx->cpu_backend != nullptr && !codec_backend_is_cpu(ctx->backend)) {
        for (ggml_tensor * node : ctx->eval_cpu_nodes) {
            ggml_backend_sched_set_tensor_backend(ctx->sched, node, ctx->cpu_backend);
        }
    }
    if (!ggml_backend_sched_alloc_graph(ctx->sched, ctx->eval_graph)) {
        if (error != nullptr) {
            *error = "failed to allocate graph in scheduler";
        }
        return false;
    }
    ctx->eval_graph_allocated = true;

    for (const codec_graph_const_input & c : ctx->eval_consts) {
        ggml_backend_tensor_set(c.tensor, c.data.data(), 0, c.data.size());
    }

    {
        char det[80];
        const size_t bytes = ggml_backend_sched_get_buffer_size(ctx->sched, codec_get_default_backend(ctx));
        std::snprintf(det, sizeof(det), "kind=%d bytes=%zu", entry->key.kind, bytes);
        codec_perf_event("graph_alloc_buf", det);
    }

    return true;
}

#ifdef CODEC_DEBUG_TOOLS
// CODEC_AUDIT_UPCAST=1: report every matmul / convtr operand that is a stored
// weight cast to another type (weights must run at their stored type).
static const ggml_tensor * audit_upcast_leaf(const ggml_tensor * t, int depth = 0) {
    if (t == nullptr || depth > 16) return nullptr;
    switch (t->op) {
        case GGML_OP_RESHAPE: case GGML_OP_VIEW: case GGML_OP_PERMUTE: case GGML_OP_TRANSPOSE:
        case GGML_OP_CONT: case GGML_OP_PAD: case GGML_OP_CONCAT: case GGML_OP_SCALE:
            return audit_upcast_leaf(t->src[0], depth + 1);
        case GGML_OP_CPY: case GGML_OP_DUP: {
            const ggml_tensor * s = t->src[0];
            if (s != nullptr && s->op == GGML_OP_NONE && s->buffer != nullptr && s->type != t->type &&
                (s->flags & GGML_TENSOR_FLAG_INPUT) == 0 && std::strncmp(s->name, "lut.", 4) != 0) return s;
            return audit_upcast_leaf(s, depth + 1);
        }
        default: return nullptr;
    }
}
static void audit_upcast(ggml_cgraph * g) {
    static const bool on = std::getenv("CODEC_AUDIT_UPCAST") != nullptr;
    if (!on) return;
    static std::set<std::string> seen;
    for (int i = 0; i < ggml_graph_n_nodes(g); ++i) {
        const ggml_tensor * n = ggml_graph_node(g, i);
        if (n->op != GGML_OP_MUL_MAT && n->op != GGML_OP_MUL_MAT_ID && n->op != GGML_OP_CONV_TRANSPOSE_1D &&
            n->op != GGML_OP_OUT_PROD) continue;
        for (int k = 0; k < 2; ++k) {
            const ggml_tensor * leaf = audit_upcast_leaf(n->src[k]);
            if (leaf != nullptr && seen.insert(leaf->name).second) {
                std::fprintf(stderr, "AUDIT-UPCAST %s src%d <- %s (%s) via %s\n", ggml_op_name(n->op), k, leaf->name,
                             ggml_type_name(leaf->type), n->name);
            }
        }
    }
}
#endif // CODEC_DEBUG_TOOLS

bool codec_graph_compute(
    codec_context * ctx,
    codec_graph_cache_entry * entry,
    int32_t n_threads,
    std::string * error) {

    char detail_buf[64];
    std::snprintf(detail_buf, sizeof(detail_buf),
                  "kind=%d nodes=%d",
                  entry ? entry->key.kind : -1,
                  ctx && ctx->eval_graph ? ggml_graph_n_nodes(ctx->eval_graph) : 0);
    CODEC_PERF_SCOPE_D("graph_compute", detail_buf);

    if (ctx == nullptr || entry == nullptr || ctx->eval_entry != entry || ctx->eval_graph == nullptr || ctx->backend == nullptr) {
        if (error != nullptr) {
            *error = "invalid graph compute arguments";
        }
        return false;
    }

    codec_backend_set_n_threads(ctx->backend, n_threads);
    if (ctx->cpu_backend != nullptr) {
        codec_backend_set_n_threads(ctx->cpu_backend, n_threads);
    }
    codec_attach_cpu_threadpool(ctx, n_threads);

    if (!codec_graph_prepare_io(ctx, entry, error)) {
        return false;
    }

#ifdef CODEC_DEBUG_TOOLS
    const bool profile_on = profiler().resolve() != nullptr;
    if (profile_on) {
        ggml_backend_sched_set_eval_callback(ctx->sched, profile_eval_callback, nullptr);
        profile_begin();
    }
    audit_upcast(ctx->eval_graph);
#endif

    const ggml_status st = ggml_backend_sched_graph_compute(ctx->sched, ctx->eval_graph);

#ifdef CODEC_DEBUG_TOOLS
    if (profile_on) {
        profile_end(entry ? entry->key.kind : -1, ggml_graph_n_nodes(ctx->eval_graph));
        ggml_backend_sched_set_eval_callback(ctx->sched, nullptr, nullptr);
    }
#endif

    if (st != GGML_STATUS_SUCCESS) {
        if (error != nullptr) {
            *error = "graph scheduler compute failed";
        }
        return false;
    }

    return true;
}

void codec_runtime_free(codec_context * ctx) {
    if (ctx == nullptr) {
        return;
    }

    codec_graph_release(ctx);
    ctx->graph_cache.clear();

    if (ctx->eval_arena_buf != nullptr) {
        std::free(ctx->eval_arena_buf);
        ctx->eval_arena_buf = nullptr;
        ctx->eval_arena_size = 0;
    }

    if (ctx->sched != nullptr) {
        ggml_backend_sched_free(ctx->sched);
        ctx->sched = nullptr;
    }

    if (ctx->cpu_backend != nullptr) {
        ggml_backend_free(ctx->cpu_backend);
        ctx->cpu_backend = nullptr;
    }
}
