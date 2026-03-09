#include "convtr1d.h"

#include "lm_ggml_ops.h"

#include <algorithm>

lm_ggml_tensor * codec_convtr1d(
    lm_ggml_context * ctx,
    lm_ggml_tensor * x,
    lm_ggml_tensor * w,
    lm_ggml_tensor * b,
    int32_t stride,
    int32_t padding,
    int32_t dilation) {

    if (ctx == nullptr || x == nullptr || w == nullptr || stride <= 0 || dilation <= 0 || padding < 0) {
        return nullptr;
    }

    lm_ggml_tensor * y = lm_ggml_conv_transpose_1d(ctx, w, x, stride, 0, dilation);
    if (b != nullptr) {
        lm_ggml_tensor * b2 = lm_ggml_reshape_2d(ctx, b, 1, y->ne[1]);
        y = lm_ggml_add(ctx, y, lm_ggml_repeat(ctx, b2, y));
    }
    if (padding > 0) {
        y = codec_op_crop_1d(ctx, y, padding, padding);
    }
    return lm_ggml_cont(ctx, y);
}

lm_ggml_tensor * codec_convtr1d_causal(
    lm_ggml_context * ctx,
    lm_ggml_tensor * x,
    lm_ggml_tensor * w,
    lm_ggml_tensor * b,
    int32_t stride,
    int32_t dilation) {

    if (ctx == nullptr || x == nullptr || w == nullptr || stride <= 0 || dilation <= 0) {
        return nullptr;
    }

    lm_ggml_tensor * y = codec_convtr1d(ctx, x, w, b, stride, 0, dilation);
    if (y == nullptr) {
        return nullptr;
    }

    const int32_t kernel = (int32_t) w->ne[0];
    const int32_t crop_right = std::max(0, kernel - stride);
    return codec_op_crop_1d(ctx, y, 0, crop_right);
}
