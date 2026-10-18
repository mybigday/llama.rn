#include "conv1d.h"

#include "ggml_ops.h"
#include "../runtime/graph.h"
#include "../runtime/tensor_utils.h"

#include <algorithm>
#include <cstdio>
#include <cstring>

// Conv weights are used at their stored type (F32, F16, BF16 or block-quantized):
// the GGUF's precision is the compute precision.  F32/F16 kernels take the
// im2col x weight orientation (F16 kernels with an F16 im2col); every other type
// is mul_mat's src0 against an F32 im2col, the only orientation ggml runs for it.
static bool codec_conv1d_w_is_src1(const ggml_tensor * w) {
    return w->type == GGML_TYPE_F32 || w->type == GGML_TYPE_F16;
}

// A conv kernel is stored (K, C_in, C_out) or flattened (K*C_in, 1, C_out): the
// same bytes, with rows long enough to block-quantize.  With the input channel
// count the two never conflict (for C_in = 1 they read the same).
static bool codec_conv1d_w_is_flat(const ggml_tensor * w, int64_t c_in) {
    return w->ne[1] != c_in && w->ne[1] == 1 && w->ne[3] == 1;
}

int64_t codec_conv1d_kernel_size(const ggml_tensor * w, int64_t c_in) {
    if (w == nullptr) return 0;
    if (!codec_conv1d_w_is_flat(w, c_in)) return w->ne[0];
    return c_in > 0 && w->ne[0] % c_in == 0 ? w->ne[0] / c_in : 0;
}

// y = conv(x, w) as im2col . w with the kernel as mul_mat's src[0]: every stored
// type, flattened kernels included.  x: [T, C_in, N]  ->  [T_out, C_out, N].
static ggml_tensor * codec_conv1d_im2col_w_src0(ggml_context * ctx, ggml_tensor * x, ggml_tensor * w, int64_t K,
                                               int32_t stride, int32_t padding, int32_t dilation) {
    const int64_t C_in = x->ne[1];
    const int64_t C_out = w->ne[2];
    // im2col reads only the kernel's shape: a data-less [K, C_in] stand-in.
    ggml_tensor * kshape = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, C_in);
    ggml_tensor * cols = ggml_im2col(ctx, kshape, x, stride, 0, padding, 0, dilation, 0, false, GGML_TYPE_F32);
    if (cols == nullptr) return nullptr;
    const int64_t t_out = cols->ne[1], n = cols->ne[2];
    ggml_tensor * w2d = ggml_reshape_2d(ctx, w, K * C_in, C_out);
    ggml_tensor * y = ggml_mul_mat(ctx, w2d, ggml_reshape_2d(ctx, cols, K * C_in, t_out * n));          // [C_out, T_out*N]
    y = ggml_reshape_3d(ctx, y, C_out, t_out, n);
    return ggml_cont(ctx, ggml_permute(ctx, y, 1, 0, 2, 3));                                        // [T_out, C_out, N]
}

static ggml_tensor * codec_conv1d_pointwise_impl(
    ggml_context * ctx,
    ggml_tensor * x,
    ggml_tensor * w) {

    if (ctx == nullptr || x == nullptr || w == nullptr) {
        return nullptr;
    }
    if (x->ne[1] != w->ne[1] || w->ne[0] != 1) {
        return nullptr;
    }

    ggml_tensor * x_ct = ggml_cont(ctx, ggml_transpose(ctx, x));       // [c_in, t]
    ggml_tensor * w_ic = ggml_reshape_2d(ctx, w, w->ne[1], w->ne[2]);  // [c_in, c_out]
    ggml_tensor * y_ct = ggml_mul_mat(ctx, w_ic, x_ct);                // [c_out, t]
    if (y_ct == nullptr) {
        return nullptr;
    }

    return ggml_cont(ctx, ggml_transpose(ctx, y_ct));                  // [t, c_out]
}

static ggml_tensor * codec_conv1d_impl(
    ggml_context * ctx,
    ggml_tensor * x,
    ggml_tensor * w,
    int32_t stride,
    int32_t padding,
    int32_t dilation) {

    if (ctx == nullptr || x == nullptr || w == nullptr || stride <= 0 || dilation <= 0 || padding < 0) {
        return nullptr;
    }
    w = codec_graph_stored_weight(ctx, w);
    if (codec_conv1d_w_is_flat(w, x->ne[1])) {
        const int64_t K = codec_conv1d_kernel_size(w, x->ne[1]);
        if (K <= 0) {
            std::fprintf(stderr, "codec_conv1d: flattened kernel '%s' [%lld, 1, %lld] does not fit %lld input channels\n",
                         w->name, (long long) w->ne[0], (long long) w->ne[2], (long long) x->ne[1]);
            return nullptr;
        }
        return codec_conv1d_im2col_w_src0(ctx, x, w, K, stride, padding, dilation);
    }

    if (w->ne[0] == 1 && stride == 1 && dilation == 1 && padding == 0) {
        return codec_conv1d_pointwise_impl(ctx, x, w);
    }

    // GPUs: one GEMM per tap over a shifted view of channel-major x instead of
    // im2col (Metal's 1D im2col runs K threads per threadgroup; ggml-vulkan's is
    // slow too): dac decode 352 -> 80 ms on Vulkan, 1214 -> 1051 ms on Metal.
    // On Metal the [T_out, C_out] per-tap GEMMs lose to im2col for long
    // sequences with few channels (nemo decode, 27-54 channels: 1.5 s -> 3.1 s),
    // so it takes the GEMM path only from 64 input and output channels; Vulkan
    // gains even there (nemo decode 339 -> 67 ms).  Other GPU backends are
    // unmeasured and follow Vulkan.  On CPUs im2col stays faster (x86, M1,
    // Pixel 7 Pro).
    //   y[t, co] = sum_k sum_ci x[t + k*d, ci] * w[k, ci, co]
    const bool metal = std::strcmp(codec_graph_backend_reg_name(), "MTL") == 0;
    // The taps are views of the permuted kernel, so it must be a type ggml copies
    // everywhere (F32/F16); an F16 kernel also needs taps that start 64 B aligned
    // (ggml_pad, used to align F32 taps, is F32-only on the GPU backends).
    const int64_t K = w->ne[0], C_in = w->ne[1], C_out = w->ne[2];
    const bool tap_type_ok = w->type == GGML_TYPE_F32 || (w->type == GGML_TYPE_F16 && (C_in * C_out) % 32 == 0);
    const bool gemm_taps = tap_type_ok && (!metal || std::min(w->ne[1], w->ne[2]) >= 64);
    if (codec_graph_backend_is_gpu() && gemm_taps && stride == 1 && x->ne[2] <= 1 && x->ne[3] <= 1 &&
        x->ne[1] == w->ne[1]) {
        ggml_tensor * xp = padding > 0 ? codec_op_pad_1d(ctx, x, padding, padding) : x;
        const int64_t t_out = xp->ne[0] - (int64_t) dilation * (K - 1);
        if (t_out <= 0) return nullptr;
        ggml_tensor * x_ct = ggml_cont(ctx, ggml_transpose(ctx, xp));                              // [C_in, T]
        // Vulkan binds each operand at its view offset, which must meet the device's
        // minStorageBufferOffsetAlignment (64 B on Mali) unless the shader handles a
        // misaligned start; ggml-vulkan allows that only for mul_mat's src1. The
        // shifted x views (offset k*d*C_in floats) are therefore src1, and every tap
        // of the weights starts on a 64 B boundary. Metal binds any offset and runs
        // the [T_out, C_out] orientation ~6% faster (DAC), so it keeps x as src0 for
        // F32 kernels (an F16 src1 under an F32 src0 is not a Metal mul_mat).
        const bool x_src0 = metal && w->type == GGML_TYPE_F32;
        const int64_t tap = metal || w->type != GGML_TYPE_F32 ? C_in * C_out : (C_in * C_out + 15) / 16 * 16;
        ggml_tensor ** slot = codec_graph_weight_cast_slot(ctx, w);
        ggml_tensor * wk_all = slot != nullptr ? *slot : nullptr;
        if (wk_all == nullptr) {
            wk_all = ggml_cont(ctx, ggml_permute(ctx, w, 2, 0, 1, 3));                              // [C_in, C_out, K]
            wk_all = ggml_reshape_2d(ctx, wk_all, C_in * C_out, K);
            if (tap != C_in * C_out) wk_all = ggml_pad(ctx, wk_all, (int) (tap - C_in * C_out), 0, 0, 0);
            if (slot != nullptr) *slot = wk_all;                                                    // [tap, K]
        }
        ggml_tensor * y = nullptr;
        for (int64_t k = 0; k < K; ++k) {
            ggml_tensor * xk = ggml_view_2d(ctx, x_ct, C_in, t_out, x_ct->nb[1], (size_t) (k * dilation) * x_ct->nb[1]);
            ggml_tensor * wk = ggml_view_2d(ctx, wk_all, C_in, C_out, ggml_row_size(w->type, C_in), (size_t) k * wk_all->nb[1]);
            ggml_tensor * yk = x_src0 ? ggml_mul_mat(ctx, xk, wk)                                  // [T_out, C_out]
                                      : ggml_mul_mat(ctx, wk, xk);                                 // [C_out, T_out]
            y = y == nullptr ? yk : ggml_add(ctx, y, yk);
        }
        if (!x_src0) y = ggml_cont(ctx, ggml_transpose(ctx, y));
        return ggml_reshape_3d(ctx, y, t_out, C_out, 1);
    }

    if (!codec_conv1d_w_is_src1(w)) {
        return codec_conv1d_im2col_w_src0(ctx, x, w, w->ne[0], stride, padding, dilation);
    }
    const ggml_type im2col_type = w->type == GGML_TYPE_F16 ? GGML_TYPE_F16 : GGML_TYPE_F32;
    ggml_tensor * im2col = ggml_im2col(ctx, w, x, stride, 0, padding, 0, dilation, 0, false, im2col_type);
    if (im2col == nullptr) {
        return nullptr;
    }

    ggml_tensor * cols = ggml_reshape_2d(ctx, im2col, im2col->ne[0], im2col->ne[2] * im2col->ne[1]);
    ggml_tensor * w2d = ggml_reshape_2d(ctx, w, w->ne[0] * w->ne[1], w->ne[2]);
    ggml_tensor * y = ggml_mul_mat(ctx, cols, w2d);
    if (y == nullptr) {
        return nullptr;
    }

    return ggml_reshape_3d(ctx, y, im2col->ne[1], w->ne[2], im2col->ne[2]);
}

static ggml_tensor * codec_conv1d_depthwise_impl(
    ggml_context * ctx,
    ggml_tensor * x,
    ggml_tensor * w,
    int32_t stride,
    int32_t padding,
    int32_t dilation) {

    if (ctx == nullptr || x == nullptr || w == nullptr || stride <= 0 || dilation <= 0 || padding < 0) {
        return nullptr;
    }
    w = codec_graph_stored_weight(ctx, w);

    // Stride 1, unbatched: y = sum_k x[t + k*d] * w[k] as K broadcast multiply-adds
    // over shifted views — F32 throughout, no K-fold im2col copy.
    if (stride == 1 && x->ne[2] <= 1 && x->ne[3] <= 1 && w->ne[1] == 1 && w->ne[2] == x->ne[1]) {
        const int64_t K = w->ne[0], C = x->ne[1];
        ggml_tensor * xp = padding > 0 ? codec_op_pad_1d(ctx, x, padding, padding) : x;
        if (!ggml_is_contiguous(xp)) xp = ggml_cont(ctx, xp);
        const int64_t t_out = xp->ne[0] - (int64_t) dilation * (K - 1);
        if (t_out <= 0) return nullptr;
        ggml_tensor * wt = ggml_cont(ctx, ggml_transpose(ctx, ggml_reshape_2d(ctx, codec_graph_cast_f32(ctx, w), K, C)));  // [C, K]
        ggml_tensor * y = nullptr;
        for (int64_t k = 0; k < K; ++k) {
            ggml_tensor * xk = ggml_view_2d(ctx, xp, t_out, C, xp->nb[1], (size_t) (k * dilation) * xp->nb[0]);
            ggml_tensor * wk = ggml_view_2d(ctx, wt, 1, C, wt->nb[0], (size_t) k * wt->nb[1]);              // [1, C]
            ggml_tensor * term = ggml_mul(ctx, xk, wk);
            y = y == nullptr ? term : ggml_add(ctx, y, term);
        }
        return ggml_reshape_3d(ctx, y, t_out, C, 1);
    }

    ggml_tensor * x4 = ggml_reshape_4d(ctx, x, x->ne[0], 1, x->ne[1], x->ne[2]);
    const ggml_type im2col_type = w->type == GGML_TYPE_F16 ? GGML_TYPE_F16 : GGML_TYPE_F32;
    ggml_tensor * im2col = ggml_im2col(ctx, w, x4, stride, 0, padding, 0, dilation, 0, false, im2col_type);
    if (im2col == nullptr) {
        return nullptr;
    }

    if (!codec_conv1d_w_is_src1(w)) {
        ggml_tensor * y = ggml_mul_mat(ctx, w, im2col);                                         // [1, T_out, C]
        return ggml_reshape_3d(ctx, y, y->ne[1], y->ne[2], 1);
    }
    ggml_tensor * y = ggml_mul_mat(ctx, im2col, w);
    if (y == nullptr) {
        return nullptr;
    }

    return ggml_reshape_3d(ctx, y, y->ne[0], y->ne[2], 1);
}

ggml_tensor * codec_conv1d(
    ggml_context * ctx,
    ggml_tensor * x,
    ggml_tensor * w,
    ggml_tensor * b,
    int32_t stride,
    int32_t dilation,
    int32_t padding) {

    if (ctx == nullptr || x == nullptr || w == nullptr || stride <= 0 || dilation <= 0 || padding < 0) {
        return nullptr;
    }

    b = codec_graph_cast_f32(ctx, b);

    ggml_tensor * y = codec_conv1d_impl(ctx, x, w, stride, padding, dilation);
    if (b != nullptr) {
        ggml_tensor * b2 = ggml_reshape_2d(ctx, b, 1, y->ne[1]);
        y = ggml_add(ctx, y, b2);
    }
    y = ggml_cont(ctx, y);
    // Squeeze the trailing batch dim when input was 2D so callers don't
    // need to manually reshape after every non-pointwise conv (the im2col
    // path always returns ne=(t_out, c_out, 1)).
    if (x->ne[2] <= 1 && y->ne[2] == 1 && y->ne[3] == 1) {
        y = ggml_reshape_2d(ctx, y, y->ne[0], y->ne[1]);
    }
    return y;
}

ggml_tensor * codec_conv1d_depthwise(
    ggml_context * ctx,
    ggml_tensor * x,
    ggml_tensor * w,
    ggml_tensor * b,
    int32_t stride,
    int32_t dilation,
    int32_t padding) {

    if (ctx == nullptr || x == nullptr || w == nullptr || stride <= 0 || dilation <= 0 || padding < 0) {
        return nullptr;
    }

    b = codec_graph_cast_f32(ctx, b);

    ggml_tensor * y = codec_conv1d_depthwise_impl(ctx, x, w, stride, padding, dilation);
    if (b != nullptr) {
        ggml_tensor * b2 = ggml_reshape_2d(ctx, b, 1, y->ne[1]);
        y = ggml_add(ctx, y, b2);
    }
    return ggml_cont(ctx, y);
}

ggml_tensor * codec_conv1d_causal(
    ggml_context * ctx,
    ggml_tensor * x,
    ggml_tensor * w,
    ggml_tensor * b,
    int32_t stride,
    int32_t dilation) {

    if (ctx == nullptr || x == nullptr || w == nullptr || stride <= 0 || dilation <= 0) {
        return nullptr;
    }

    const int32_t kernel = (int32_t) codec_conv1d_kernel_size(w, x->ne[1]);
    if (kernel < stride) {
        return nullptr;
    }

    b = codec_graph_cast_f32(ctx, b);

    const int32_t kernel_eff = (kernel - 1) * dilation + 1;
    const int32_t pad_left = kernel_eff - stride;
    const int32_t t_in = (int32_t) x->ne[0];
    const int32_t extra_pad = t_in > 0 ? (((t_in + stride - 1) / stride) * stride - t_in) : 0;
    ggml_tensor * x_pad = codec_op_pad_1d(ctx, x, pad_left, extra_pad);
    if (x_pad == nullptr) {
        return nullptr;
    }

    ggml_tensor * y = codec_conv1d_impl(ctx, x_pad, w, stride, 0, dilation);
    if (b != nullptr) {
        ggml_tensor * b2 = ggml_reshape_2d(ctx, b, 1, y->ne[1]);
        y = ggml_add(ctx, y, b2);
    }
    return ggml_cont(ctx, y);
}

ggml_tensor * codec_conv1d_causal_replicate(
    ggml_context * ctx,
    ggml_tensor * x,
    ggml_tensor * w,
    ggml_tensor * b,
    int32_t stride,
    int32_t dilation) {

    if (ctx == nullptr || x == nullptr || w == nullptr || stride <= 0 || dilation <= 0) {
        return nullptr;
    }

    const int32_t kernel = (int32_t) codec_conv1d_kernel_size(w, x->ne[1]);
    if (kernel < stride) {
        return nullptr;
    }

    b = codec_graph_cast_f32(ctx, b);

    const int32_t kernel_eff = (kernel - 1) * dilation + 1;
    const int32_t pad_left = kernel_eff - stride;
    const int32_t t_in = (int32_t) x->ne[0];
    const int32_t extra_pad = t_in > 0 ? (((t_in + stride - 1) / stride) * stride - t_in) : 0;
    ggml_tensor * x_pad = codec_op_pad_1d_replicate(ctx, x, pad_left, extra_pad);
    if (x_pad == nullptr) {
        return nullptr;
    }

    ggml_tensor * y = codec_conv1d_impl(ctx, x_pad, w, stride, 0, dilation);
    if (b != nullptr) {
        ggml_tensor * b2 = ggml_reshape_2d(ctx, b, 1, y->ne[1]);
        y = ggml_add(ctx, y, b2);
    }
    return ggml_cont(ctx, y);
}

ggml_tensor * codec_conv1d_depthwise_causal(
    ggml_context * ctx,
    ggml_tensor * x,
    ggml_tensor * w,
    ggml_tensor * b,
    int32_t stride,
    int32_t dilation) {

    if (ctx == nullptr || x == nullptr || w == nullptr || stride <= 0 || dilation <= 0) {
        return nullptr;
    }

    const int32_t kernel = (int32_t) w->ne[0];   // depthwise (K, 1, C)
    if (kernel < stride) {
        return nullptr;
    }

    b = codec_graph_cast_f32(ctx, b);

    const int32_t kernel_eff = (kernel - 1) * dilation + 1;
    const int32_t pad_left = kernel_eff - stride;
    const int32_t t_in = (int32_t) x->ne[0];
    const int32_t extra_pad = t_in > 0 ? (((t_in + stride - 1) / stride) * stride - t_in) : 0;
    ggml_tensor * x_pad = codec_op_pad_1d(ctx, x, pad_left, extra_pad);
    if (x_pad == nullptr) {
        return nullptr;
    }

    ggml_tensor * y = codec_conv1d_depthwise_impl(ctx, x_pad, w, stride, 0, dilation);
    if (b != nullptr) {
        ggml_tensor * b2 = ggml_reshape_2d(ctx, b, 1, y->ne[1]);
        y = ggml_add(ctx, y, b2);
    }
    return ggml_cont(ctx, y);
}
