#include "graph_exec.h"

#include "perf_log.h"

#include <ggml-alloc.h>
#include <ggml.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <utility>
#include <vector>

// ---------------------------------------------------------------------------
// Per-node profiler (zero-cost unless CODEC_OP_PROFILE=<path> is set).
//
// Registers a lm_ggml_backend_sched eval callback that timestamps successive node
// observation callbacks and accumulates wall time + call count per
// (op, shape-class).  The scheduler invokes the callback per resulting node
// twice: ask=true (batching query — we always answer yes so every node is
// observed) then ask=false (the observation, fired *after* the node computed).
//
// We attribute the interval between two consecutive ask=false callbacks to the
// node observed at the *end* of that interval (its result just became ready),
// which is the standard "delta since previous observe" accounting.  The very
// first node's own compute isn't isolated (it shares the interval from compute
// start), a negligible edge effect over thousands of nodes.
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
    lm_ggml_tensor *                         prev_node = nullptr;
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
std::string profile_key(const lm_ggml_tensor * t) {
    char buf[96];
    std::snprintf(buf, sizeof(buf), "%-16s [%lld,%lld,%lld,%lld]",
                  lm_ggml_op_name(t->op),
                  (long long) t->ne[0], (long long) t->ne[1],
                  (long long) t->ne[2], (long long) t->ne[3]);
    return std::string(buf);
}

// Attribute the elapsed interval to the *previous* observed node.
void profile_observe(lm_ggml_tensor * node) {
    op_profiler & p = profiler();
    const int64_t now = lm_ggml_time_us();
    if (p.prev_node != nullptr) {
        op_profile_bucket & b = p.buckets[profile_key(p.prev_node)];
        b.total_us += now - p.t_prev_us;
        b.count    += 1;
    }
    p.prev_node = node;
    p.t_prev_us = now;
}

bool profile_eval_callback(lm_ggml_tensor * t, bool ask, void * /*user_data*/) {
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
    p.prev_node = nullptr;
    p.t_prev_us = lm_ggml_time_us();
}

void profile_end(int graph_kind, int n_nodes) {
    op_profiler & p = profiler();
    if (!p.active) return;
    // flush the last observed node using compute-end as its interval boundary.
    if (p.prev_node != nullptr) {
        op_profile_bucket & b = p.buckets[profile_key(p.prev_node)];
        b.total_us += lm_ggml_time_us() - p.t_prev_us;
        b.count    += 1;
    }
    p.prev_node = nullptr;

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

static bool codec_backend_is_cpu(lm_ggml_backend_t backend) {
    if (backend == nullptr) {
        return false;
    }
    lm_ggml_backend_dev_t dev = lm_ggml_backend_get_device(backend);
    if (dev == nullptr) {
        return false;
    }
    return lm_ggml_backend_dev_type(dev) == LM_GGML_BACKEND_DEVICE_TYPE_CPU;
}

static lm_ggml_backend_t codec_get_default_backend(codec_context * ctx) {
    // If main backend is CPU, use it; otherwise use CPU backend if available
    if (ctx->backend != nullptr && codec_backend_is_cpu(ctx->backend)) {
        return ctx->backend;
    }
    return ctx->cpu_backend;
}

static void codec_backend_set_n_threads(lm_ggml_backend_t backend, int32_t n_threads) {
    if (backend == nullptr || n_threads <= 0) {
        return;
    }
    lm_ggml_backend_dev_t dev = lm_ggml_backend_get_device(backend);
    if (dev == nullptr) {
        return;
    }
    lm_ggml_backend_reg_t reg = lm_ggml_backend_dev_backend_reg(dev);
    if (reg == nullptr) {
        return;
    }
    lm_ggml_backend_set_n_threads_t fn = reinterpret_cast<lm_ggml_backend_set_n_threads_t>(
        lm_ggml_backend_reg_get_proc_address(reg, "lm_ggml_backend_set_n_threads"));
    if (fn != nullptr) {
        fn(backend, n_threads);
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
        lm_ggml_backend_sched_free(ctx->sched);
        ctx->sched = nullptr;
    }

    std::array<lm_ggml_backend_t, 2> backends = { ctx->backend, nullptr };
    int n_backends = 1;
    if (!codec_backend_is_cpu(ctx->backend) && ctx->cpu_backend != nullptr) {
        backends[1] = ctx->cpu_backend;
        n_backends = 2;
    }

    const bool op_offload = std::getenv("CODEC_NO_OP_OFFLOAD") == nullptr;
    ctx->sched = lm_ggml_backend_sched_new(backends.data(), nullptr, n_backends, target, false, op_offload);
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

    std::array<lm_ggml_backend_t, 2> backends = { ctx->backend, nullptr };
    int n_backends = 1;

    if (!codec_backend_is_cpu(ctx->backend)) {
        ctx->cpu_backend = lm_ggml_backend_init_by_name("CPU", nullptr);
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
        required = std::max(1, lm_ggml_graph_n_nodes(ctx->eval_graph));
    }
    if (!codec_sched_ensure_capacity(ctx, required, error)) {
        return false;
    }

    if (ctx->eval_graph_allocated) {
        return true;
    }

    lm_ggml_backend_sched_reset(ctx->sched);
    if (!lm_ggml_backend_sched_alloc_graph(ctx->sched, ctx->eval_graph)) {
        if (error != nullptr) {
            *error = "failed to allocate graph in scheduler";
        }
        return false;
    }
    ctx->eval_graph_allocated = true;

    {
        char det[80];
        const size_t bytes = lm_ggml_backend_sched_get_buffer_size(ctx->sched, codec_get_default_backend(ctx));
        std::snprintf(det, sizeof(det), "kind=%d bytes=%zu", entry->key.kind, bytes);
        codec_perf_event("graph_alloc_buf", det);
    }

    return true;
}

bool codec_graph_compute(
    codec_context * ctx,
    codec_graph_cache_entry * entry,
    int32_t n_threads,
    std::string * error) {

    char detail_buf[64];
    std::snprintf(detail_buf, sizeof(detail_buf),
                  "kind=%d nodes=%d",
                  entry ? entry->key.kind : -1,
                  ctx && ctx->eval_graph ? lm_ggml_graph_n_nodes(ctx->eval_graph) : 0);
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

    if (!codec_graph_prepare_io(ctx, entry, error)) {
        return false;
    }

    const bool profile_on = profiler().resolve() != nullptr;
    if (profile_on) {
        lm_ggml_backend_sched_set_eval_callback(ctx->sched, profile_eval_callback, nullptr);
        profile_begin();
    }

    const lm_ggml_status st = lm_ggml_backend_sched_graph_compute(ctx->sched, ctx->eval_graph);

    if (profile_on) {
        profile_end(entry ? entry->key.kind : -1, lm_ggml_graph_n_nodes(ctx->eval_graph));
        lm_ggml_backend_sched_set_eval_callback(ctx->sched, nullptr, nullptr);
    }

    if (st != LM_GGML_STATUS_SUCCESS) {
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
        lm_ggml_backend_sched_free(ctx->sched);
        ctx->sched = nullptr;
    }

    if (ctx->cpu_backend != nullptr) {
        lm_ggml_backend_free(ctx->cpu_backend);
        ctx->cpu_backend = nullptr;
    }
}
