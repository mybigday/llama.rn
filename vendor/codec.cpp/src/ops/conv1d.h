#ifndef CODEC_OPS_CONV1D_H
#define CODEC_OPS_CONV1D_H

#include "../codec_internal.h"

// Conv kernels are (K, C_in, C_out) or flattened (K*C_in, 1, C_out) (same bytes;
// the layout block-quantized kernels need), at any stored type.  Kernel
// size of either form, given the input channel count (0 if they don't fit).
int64_t codec_conv1d_kernel_size(const ggml_tensor * w, int64_t c_in);

ggml_tensor * codec_conv1d(
    ggml_context * ctx,
    ggml_tensor * x,
    ggml_tensor * w,
    ggml_tensor * b,
    int32_t stride,
    int32_t dilation,
    int32_t padding);

ggml_tensor * codec_conv1d_depthwise(
    ggml_context * ctx,
    ggml_tensor * x,
    ggml_tensor * w,
    ggml_tensor * b,
    int32_t stride,
    int32_t dilation,
    int32_t padding);

ggml_tensor * codec_conv1d_causal(
    ggml_context * ctx,
    ggml_tensor * x,
    ggml_tensor * w,
    ggml_tensor * b,
    int32_t stride,
    int32_t dilation);

ggml_tensor * codec_conv1d_causal_replicate(
    ggml_context * ctx,
    ggml_tensor * x,
    ggml_tensor * w,
    ggml_tensor * b,
    int32_t stride,
    int32_t dilation);

ggml_tensor * codec_conv1d_depthwise_causal(
    ggml_context * ctx,
    ggml_tensor * x,
    ggml_tensor * w,
    ggml_tensor * b,
    int32_t stride,
    int32_t dilation);

#endif
