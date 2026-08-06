#include "conv1d.h"

#include "lm_ggml_ops.h"
#include "../runtime/tensor_utils.h"

// Conv weights are loaded directly from the GGUF context. Quantized weight
// types can't be reshaped (their row size is fixed by the block format), so
// cast to F32 before any reshape/im2col path. F16 weights are kept as F16 to
// preserve the fast im2col-F16 path that lm_ggml_conv_1d uses.
static lm_ggml_tensor * codec_conv1d_prepare_w(lm_ggml_context * ctx, lm_ggml_tensor * w) {
    if (w == nullptr) {
        return nullptr;
    }
    if (w->type == LM_GGML_TYPE_F32 || w->type == LM_GGML_TYPE_F16) {
        return w;
    }
    return lm_ggml_cast(ctx, w, LM_GGML_TYPE_F32);
}

static lm_ggml_tensor * codec_conv1d_pointwise_impl(
    lm_ggml_context * ctx,
    lm_ggml_tensor * x,
    lm_ggml_tensor * w) {

    if (ctx == nullptr || x == nullptr || w == nullptr) {
        return nullptr;
    }
    if (x->ne[1] != w->ne[1] || w->ne[0] != 1) {
        return nullptr;
    }

    lm_ggml_tensor * x_ct = lm_ggml_cont(ctx, lm_ggml_transpose(ctx, x));       // [c_in, t]
    lm_ggml_tensor * w_ic = lm_ggml_reshape_2d(ctx, w, w->ne[1], w->ne[2]);  // [c_in, c_out]
    lm_ggml_tensor * y_ct = lm_ggml_mul_mat(ctx, w_ic, x_ct);                // [c_out, t]
    if (y_ct == nullptr) {
        return nullptr;
    }

    return lm_ggml_cont(ctx, lm_ggml_transpose(ctx, y_ct));                  // [t, c_out]
}

static lm_ggml_tensor * codec_conv1d_impl(
    lm_ggml_context * ctx,
    lm_ggml_tensor * x,
    lm_ggml_tensor * w,
    int32_t stride,
    int32_t padding,
    int32_t dilation) {

    if (ctx == nullptr || x == nullptr || w == nullptr || stride <= 0 || dilation <= 0 || padding < 0) {
        return nullptr;
    }

    if (w->ne[0] == 1 && stride == 1 && dilation == 1 && padding == 0) {
        return codec_conv1d_pointwise_impl(ctx, x, w);
    }

    const lm_ggml_type im2col_type = w->type == LM_GGML_TYPE_F16 ? LM_GGML_TYPE_F16 : LM_GGML_TYPE_F32;
    lm_ggml_tensor * im2col = lm_ggml_im2col(ctx, w, x, stride, 0, padding, 0, dilation, 0, false, im2col_type);
    if (im2col == nullptr) {
        return nullptr;
    }

    lm_ggml_tensor * lhs = lm_ggml_reshape_2d(ctx, im2col, im2col->ne[0], im2col->ne[2] * im2col->ne[1]);
    lm_ggml_tensor * rhs = lm_ggml_reshape_2d(ctx, w, w->ne[0] * w->ne[1], w->ne[2]);
    lm_ggml_tensor * y = lm_ggml_mul_mat(ctx, lhs, rhs);
    if (y == nullptr) {
        return nullptr;
    }

    return lm_ggml_reshape_3d(ctx, y, im2col->ne[1], w->ne[2], im2col->ne[2]);
}

static lm_ggml_tensor * codec_conv1d_depthwise_impl(
    lm_ggml_context * ctx,
    lm_ggml_tensor * x,
    lm_ggml_tensor * w,
    int32_t stride,
    int32_t padding,
    int32_t dilation) {

    if (ctx == nullptr || x == nullptr || w == nullptr || stride <= 0 || dilation <= 0 || padding < 0) {
        return nullptr;
    }

    lm_ggml_tensor * x4 = lm_ggml_reshape_4d(ctx, x, x->ne[0], 1, x->ne[1], x->ne[2]);
    const lm_ggml_type im2col_type = w->type == LM_GGML_TYPE_F16 ? LM_GGML_TYPE_F16 : LM_GGML_TYPE_F32;
    lm_ggml_tensor * im2col = lm_ggml_im2col(ctx, w, x4, stride, 0, padding, 0, dilation, 0, false, im2col_type);
    if (im2col == nullptr) {
        return nullptr;
    }

    lm_ggml_tensor * y = lm_ggml_mul_mat(ctx, im2col, w);
    if (y == nullptr) {
        return nullptr;
    }

    return lm_ggml_reshape_3d(ctx, y, y->ne[0], y->ne[2], 1);
}

lm_ggml_tensor * codec_conv1d(
    lm_ggml_context * ctx,
    lm_ggml_tensor * x,
    lm_ggml_tensor * w,
    lm_ggml_tensor * b,
    int32_t stride,
    int32_t dilation,
    int32_t padding) {

    if (ctx == nullptr || x == nullptr || w == nullptr || stride <= 0 || dilation <= 0 || padding < 0) {
        return nullptr;
    }

    w = codec_conv1d_prepare_w(ctx, w);
    b = codec_graph_cast_f32(ctx, b);

    lm_ggml_tensor * y = codec_conv1d_impl(ctx, x, w, stride, padding, dilation);
    if (b != nullptr) {
        lm_ggml_tensor * b2 = lm_ggml_reshape_2d(ctx, b, 1, y->ne[1]);
        y = lm_ggml_add(ctx, y, lm_ggml_repeat(ctx, b2, y));
    }
    y = lm_ggml_cont(ctx, y);
    // Squeeze the trailing batch dim when input was 2D so callers don't
    // need to manually reshape after every non-pointwise conv (the im2col
    // path always returns ne=(t_out, c_out, 1)).
    if (x->ne[2] <= 1 && y->ne[2] == 1 && y->ne[3] == 1) {
        y = lm_ggml_reshape_2d(ctx, y, y->ne[0], y->ne[1]);
    }
    return y;
}

lm_ggml_tensor * codec_conv1d_depthwise(
    lm_ggml_context * ctx,
    lm_ggml_tensor * x,
    lm_ggml_tensor * w,
    lm_ggml_tensor * b,
    int32_t stride,
    int32_t dilation,
    int32_t padding) {

    if (ctx == nullptr || x == nullptr || w == nullptr || stride <= 0 || dilation <= 0 || padding < 0) {
        return nullptr;
    }

    w = codec_conv1d_prepare_w(ctx, w);
    b = codec_graph_cast_f32(ctx, b);

    lm_ggml_tensor * y = codec_conv1d_depthwise_impl(ctx, x, w, stride, padding, dilation);
    if (b != nullptr) {
        lm_ggml_tensor * b2 = lm_ggml_reshape_2d(ctx, b, 1, y->ne[1]);
        y = lm_ggml_add(ctx, y, lm_ggml_repeat(ctx, b2, y));
    }
    return lm_ggml_cont(ctx, y);
}

lm_ggml_tensor * codec_conv1d_causal(
    lm_ggml_context * ctx,
    lm_ggml_tensor * x,
    lm_ggml_tensor * w,
    lm_ggml_tensor * b,
    int32_t stride,
    int32_t dilation) {

    if (ctx == nullptr || x == nullptr || w == nullptr || stride <= 0 || dilation <= 0) {
        return nullptr;
    }

    if (w->ne[0] < stride) {
        return nullptr;
    }

    w = codec_conv1d_prepare_w(ctx, w);
    b = codec_graph_cast_f32(ctx, b);

    const int32_t kernel = (int32_t) w->ne[0];
    const int32_t kernel_eff = (kernel - 1) * dilation + 1;
    const int32_t pad_left = kernel_eff - stride;
    const int32_t t_in = (int32_t) x->ne[0];
    const int32_t extra_pad = t_in > 0 ? (((t_in + stride - 1) / stride) * stride - t_in) : 0;
    lm_ggml_tensor * x_pad = codec_op_pad_1d(ctx, x, pad_left, extra_pad);
    if (x_pad == nullptr) {
        return nullptr;
    }

    lm_ggml_tensor * y = codec_conv1d_impl(ctx, x_pad, w, stride, 0, dilation);
    if (b != nullptr) {
        lm_ggml_tensor * b2 = lm_ggml_reshape_2d(ctx, b, 1, y->ne[1]);
        y = lm_ggml_add(ctx, y, lm_ggml_repeat(ctx, b2, y));
    }
    return lm_ggml_cont(ctx, y);
}

lm_ggml_tensor * codec_conv1d_causal_replicate(
    lm_ggml_context * ctx,
    lm_ggml_tensor * x,
    lm_ggml_tensor * w,
    lm_ggml_tensor * b,
    int32_t stride,
    int32_t dilation) {

    if (ctx == nullptr || x == nullptr || w == nullptr || stride <= 0 || dilation <= 0) {
        return nullptr;
    }

    if (w->ne[0] < stride) {
        return nullptr;
    }

    w = codec_conv1d_prepare_w(ctx, w);
    b = codec_graph_cast_f32(ctx, b);

    const int32_t kernel = (int32_t) w->ne[0];
    const int32_t kernel_eff = (kernel - 1) * dilation + 1;
    const int32_t pad_left = kernel_eff - stride;
    const int32_t t_in = (int32_t) x->ne[0];
    const int32_t extra_pad = t_in > 0 ? (((t_in + stride - 1) / stride) * stride - t_in) : 0;
    lm_ggml_tensor * x_pad = codec_op_pad_1d_replicate(ctx, x, pad_left, extra_pad);
    if (x_pad == nullptr) {
        return nullptr;
    }

    lm_ggml_tensor * y = codec_conv1d_impl(ctx, x_pad, w, stride, 0, dilation);
    if (b != nullptr) {
        lm_ggml_tensor * b2 = lm_ggml_reshape_2d(ctx, b, 1, y->ne[1]);
        y = lm_ggml_add(ctx, y, lm_ggml_repeat(ctx, b2, y));
    }
    return lm_ggml_cont(ctx, y);
}

lm_ggml_tensor * codec_conv1d_depthwise_causal(
    lm_ggml_context * ctx,
    lm_ggml_tensor * x,
    lm_ggml_tensor * w,
    lm_ggml_tensor * b,
    int32_t stride,
    int32_t dilation) {

    if (ctx == nullptr || x == nullptr || w == nullptr || stride <= 0 || dilation <= 0) {
        return nullptr;
    }

    if (w->ne[0] < stride) {
        return nullptr;
    }

    w = codec_conv1d_prepare_w(ctx, w);
    b = codec_graph_cast_f32(ctx, b);

    const int32_t kernel = (int32_t) w->ne[0];
    const int32_t kernel_eff = (kernel - 1) * dilation + 1;
    const int32_t pad_left = kernel_eff - stride;
    const int32_t t_in = (int32_t) x->ne[0];
    const int32_t extra_pad = t_in > 0 ? (((t_in + stride - 1) / stride) * stride - t_in) : 0;
    lm_ggml_tensor * x_pad = codec_op_pad_1d(ctx, x, pad_left, extra_pad);
    if (x_pad == nullptr) {
        return nullptr;
    }

    lm_ggml_tensor * y = codec_conv1d_depthwise_impl(ctx, x_pad, w, stride, 0, dilation);
    if (b != nullptr) {
        lm_ggml_tensor * b2 = lm_ggml_reshape_2d(ctx, b, 1, y->ne[1]);
        y = lm_ggml_add(ctx, y, lm_ggml_repeat(ctx, b2, y));
    }
    return lm_ggml_cont(ctx, y);
}
