#include "graph_exec.h"

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
        return true;
    }

    const size_t min_size = (size_t) required;
    const size_t base_size = LM_GGML_DEFAULT_GRAPH_SIZE * 8;
    size_t target = std::max(base_size, min_size * 2);
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
    ctx->eval_alloc_entry = nullptr;
    ctx->sched_reserved_graph_size = 0;
    ctx->sched_needs_reset = false;
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

    const size_t sched_graph_size = LM_GGML_DEFAULT_GRAPH_SIZE * 8;
    ctx->sched = lm_ggml_backend_sched_new(backends.data(), nullptr, n_backends, sched_graph_size, false, true);
    if (ctx->sched == nullptr) {
        if (ctx->cpu_backend != nullptr) {
            lm_ggml_backend_free(ctx->cpu_backend);
            ctx->cpu_backend = nullptr;
        }
        if (error != nullptr) {
            *error = "failed to create backend scheduler";
        }
        return false;
    }
    ctx->sched_reserved_graph_size = (int32_t) sched_graph_size;

    return true;
}

bool codec_graph_prepare_io(
    codec_context * ctx,
    codec_graph_cache_entry * entry,
    std::string * error) {
    if (ctx == nullptr || entry == nullptr || ctx->eval_entry != entry || ctx->eval_graph == nullptr || ctx->sched == nullptr || ctx->backend == nullptr) {
        if (error != nullptr) {
            *error = "invalid graph prepare arguments";
        }
        return false;
    }

    if (ctx->eval_alloc_entry == entry) {
        return true;
    }

    // If we previously allocated a graph in the scheduler and we are about to allocate a different one,
    // we must reset the scheduler to avoid dangling allocations (see ggml-backend.h docs).
    if (ctx->sched_needs_reset && !entry->allocated) {
        lm_ggml_backend_sched_reset(ctx->sched);
        ctx->sched_needs_reset = false;
        // All previous allocations are invalid after reset.
        for (codec_graph_cache_entry & e : ctx->graph_cache) {
            e.allocated = false;
        }
    }

    // Allocate all tensors in the context to the backend buffer using lm_ggml_backend_alloc_ctx_tensors
    lm_ggml_backend_t cpu_backend = codec_get_default_backend(ctx);
    if (cpu_backend == nullptr || ctx->eval_ctx == nullptr) {
        if (error != nullptr) {
            *error = "no CPU backend or eval context";
        }
        return false;
    }

    lm_ggml_backend_buffer_t buf = lm_ggml_backend_alloc_ctx_tensors(ctx->eval_ctx, cpu_backend);
    if (buf == nullptr) {
        if (error != nullptr) {
            *error = "failed to allocate tensors to backend";
        }
        return false;
    }

    entry->allocated = true;
    ctx->eval_alloc_entry = entry;
    ctx->sched_needs_reset = true;
    return true;
}

bool codec_graph_compute(
    codec_context * ctx,
    codec_graph_cache_entry * entry,
    int32_t n_threads,
    std::string * error) {

    if (ctx == nullptr || entry == nullptr || ctx->eval_entry != entry || ctx->eval_graph == nullptr || ctx->sched == nullptr || ctx->backend == nullptr) {
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

    const int32_t n_nodes = lm_ggml_graph_n_nodes(ctx->eval_graph);
    const int32_t required = n_nodes > 0 ? n_nodes * 2 : 0;
    if (!codec_sched_ensure_capacity(ctx, required, error)) {
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
