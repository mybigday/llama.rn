#ifndef CODEC_OPS_POOL1D_H
#define CODEC_OPS_POOL1D_H

#include <ggml.h>

lm_ggml_tensor * codec_op_max_pool1d(
    lm_ggml_context * ctx,
    lm_ggml_tensor * x,
    int32_t kernel,
    int32_t pad);

lm_ggml_tensor * codec_op_avg_pool1d(
    lm_ggml_context * ctx,
    lm_ggml_tensor * x,
    int32_t kernel,
    int32_t pad);

#endif
