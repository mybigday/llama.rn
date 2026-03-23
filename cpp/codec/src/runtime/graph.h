#ifndef CODEC_RUNTIME_GRAPH_H
#define CODEC_RUNTIME_GRAPH_H

#include "../codec_internal.h"

enum codec_graph_kind {
    CODEC_GRAPH_WT_DECODE = 1,
    CODEC_GRAPH_WT_ENCODE = 2,
    CODEC_GRAPH_DAC_DECODE = 3,
    CODEC_GRAPH_DAC_ENCODE = 4,
    CODEC_GRAPH_DAC_DECODE_LATENT = 5,
    CODEC_GRAPH_MIMI_DECODE = 6,
    CODEC_GRAPH_MIMI_ENCODE = 7,
    CODEC_GRAPH_Q3T_DECODE = 8,
};

bool codec_runtime_init(codec_context * ctx, std::string * error);
void codec_runtime_free(codec_context * ctx);

bool codec_graph_cache_get_or_build(
    codec_context * ctx,
    codec_graph_cache_key key,
    size_t mem_size,
    codec_graph_build_fn build_fn,
    const void * user_data,
    size_t user_data_size,
    codec_graph_cache_entry ** out_entry,
    std::string * error);

bool codec_graph_compute(
    codec_context * ctx,
    codec_graph_cache_entry * entry,
    int32_t n_threads,
    std::string * error);

bool codec_graph_prepare_io(
    codec_context * ctx,
    codec_graph_cache_entry * entry,
    std::string * error);

void codec_graph_release(codec_context * ctx);
lm_ggml_tensor * codec_graph_get_tensor(codec_context * ctx, codec_graph_cache_entry * entry, const char * name);

struct codec_graph_eval_guard {
    explicit codec_graph_eval_guard(codec_context * ctx_) : ctx(ctx_) {}
    ~codec_graph_eval_guard() {
        codec_graph_release(ctx);
    }
    codec_context * ctx;
};

#endif
