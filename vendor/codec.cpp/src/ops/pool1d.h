#ifndef CODEC_OPS_POOL1D_H
#define CODEC_OPS_POOL1D_H

#include <ggml.h>

// ggml_pool_1d(x, op, k, s, p) on time-major [t, c, ...] input, built as a
// [t, 1, c] POOL_2D: same arithmetic, but ggml-vulkan has no POOL_1D (the
// op would split the graph to the CPU).
ggml_tensor * codec_op_pool_1d(
    ggml_context * ctx,
    ggml_tensor * x,
    enum ggml_op_pool op,
    int32_t k,
    int32_t s,
    int32_t p);

ggml_tensor * codec_op_max_pool1d(
    ggml_context * ctx,
    ggml_tensor * x,
    int32_t kernel,
    int32_t pad);

ggml_tensor * codec_op_avg_pool1d(
    ggml_context * ctx,
    ggml_tensor * x,
    int32_t kernel,
    int32_t pad);

#endif
