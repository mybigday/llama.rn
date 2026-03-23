#ifndef CODEC_OPS_CONV1D_H
#define CODEC_OPS_CONV1D_H

#include "../codec_internal.h"

lm_ggml_tensor * codec_conv1d(
    lm_ggml_context * ctx,
    lm_ggml_tensor * x,
    lm_ggml_tensor * w,
    lm_ggml_tensor * b,
    int32_t stride,
    int32_t dilation,
    int32_t padding);

lm_ggml_tensor * codec_conv1d_depthwise(
    lm_ggml_context * ctx,
    lm_ggml_tensor * x,
    lm_ggml_tensor * w,
    lm_ggml_tensor * b,
    int32_t stride,
    int32_t dilation,
    int32_t padding);

lm_ggml_tensor * codec_conv1d_causal(
    lm_ggml_context * ctx,
    lm_ggml_tensor * x,
    lm_ggml_tensor * w,
    lm_ggml_tensor * b,
    int32_t stride,
    int32_t dilation);

lm_ggml_tensor * codec_conv1d_depthwise_causal(
    lm_ggml_context * ctx,
    lm_ggml_tensor * x,
    lm_ggml_tensor * w,
    lm_ggml_tensor * b,
    int32_t stride,
    int32_t dilation);

#endif
