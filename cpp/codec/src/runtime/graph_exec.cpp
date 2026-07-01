#include "graph_exec.h"

#include "perf_log.h"

#include <ggml-alloc.h>
#include <ggml.h>

#include <array>
#include <cstdlib>

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

    ctx->sched = lm_ggml_backend_sched_new(backends.data(), nullptr, n_backends, target, false, true);
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

    const lm_ggml_status st = lm_ggml_backend_sched_graph_compute(ctx->sched, ctx->eval_graph);
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
