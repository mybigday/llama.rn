#include "convtr1d.h"

#include "ggml_ops.h"
#include "../runtime/graph.h"
#include "../runtime/tensor_utils.h"

#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

// Zero-pad ne[1] by `left` / `right` rows.  Metal's PAD only pads at the end, so
// the left side is a concatenated zero constant (cf. codec_op_pad_1d).
static ggml_tensor * codec_convtr1d_pad_rows(ggml_context * ctx, ggml_tensor * x, int64_t left, int64_t right) {
    if (left > 0) {
        const int64_t ne[4] = {x->ne[0], left, x->ne[2], x->ne[3]};
        const std::string name = "lut.zeros.f32." + std::to_string(ne[0]) + "x" + std::to_string(ne[1]) + "x" +
                                 std::to_string(ne[2]) + "x" + std::to_string(ne[3]);
        std::vector<float> zeros((size_t) (ne[0] * ne[1] * ne[2] * ne[3]), 0.0f);
        ggml_tensor * z = codec_graph_const(ctx, name.c_str(), GGML_TYPE_F32, 4, ne, zeros.data());
        x = z != nullptr ? ggml_concat(ctx, z, x, 1) : ggml_pad_ext(ctx, x, 0, 0, (int) left, 0, 0, 0, 0, 0);
    }
    return right > 0 ? ggml_pad(ctx, x, 0, (int) right, 0, 0) : x;
}

// ConvTranspose1d as one GEMM plus ceil(K/stride) shifted adds (dilation 1).
// ggml's CPU CONV_TRANSPOSE_1D runs one dot product per output sample with no
// blocking; this is the same arithmetic (bit-level equal up to summation order,
// at the kernel's stored precision) at GEMM speed.
//   cols[k + K*co, t] = sum_ci w[k, co, ci] * x[t, ci]
//   y[t*s + k, co]   += cols[k + K*co, t];  with k = j*s + p, phase block j
//   lands j frames later.
// The kernel is (K, C_out, C_in) (ggml's ConvTranspose1d layout, transposed
// here) or C_in-major (C_in, C_out, K): rows over C_in, so it is mul_mat's src[0]
// as stored, the layout block-quantized kernels need (cols rows are then co + C_out*k).
static bool codec_convtr1d_w_ci_major(const ggml_tensor * w, int64_t c_in) {
    return w->ne[2] != c_in && w->ne[0] == c_in;
}

int64_t codec_convtr1d_kernel_size(const ggml_tensor * w, int64_t c_in) {
    if (w == nullptr) return 0;
    return codec_convtr1d_w_ci_major(w, c_in) ? w->ne[2] : w->ne[0];
}

static ggml_tensor * codec_convtr1d_gemm(ggml_context * ctx, ggml_tensor * x, ggml_tensor * w, int32_t stride) {
    const int64_t C_in = x->ne[1], T = x->ne[0];
    const bool ci_major = codec_convtr1d_w_ci_major(w, C_in);
    const int64_t K0 = ci_major ? w->ne[2] : w->ne[0];
    const int64_t C_out = w->ne[1];
    // A kernel that is not a multiple of the stride is treated as one with zero
    // taps appended: its last phase block is padded (on the F32 GEMM output, so
    // the kernel itself stays untouched); they only add samples past the true
    // output end, cropped below.
    const int64_t K = (K0 + stride - 1) / stride * stride;
    const int64_t J = K / stride;
    ggml_tensor * wt = ci_major ? ggml_reshape_2d(ctx, w, C_in, C_out * K0)                              // [C_in, C_out*K0]
                                : ggml_cont(ctx, ggml_transpose(ctx, ggml_reshape_2d(ctx, w, K0 * C_out, C_in)));  // [C_in, K0*C_out]
    ggml_tensor * xt = ggml_cont(ctx, ggml_transpose(ctx, ggml_reshape_2d(ctx, x, T, C_in)));           // [C_in, T]
    ggml_tensor * cols = ggml_mul_mat(ctx, wt, xt);                                                     // [K0*C_out, T]
    const size_t es = cols->nb[0];
    ggml_tensor * y = nullptr;
    for (int64_t j = 0; j < J; ++j) {
        const int64_t len = std::min<int64_t>(stride, K0 - j * stride);
        ggml_tensor * blk;
        if (ci_major) {
            blk = ggml_view_3d(ctx, cols, C_out, len, T, es * C_out, cols->nb[1], es * C_out * j * stride);   // [C_out, len, T]
            blk = ggml_cont(ctx, ggml_permute(ctx, blk, 2, 0, 1, 3));                                        // [len, T, C_out]
        } else {
            blk = ggml_view_3d(ctx, cols, len, C_out, T, es * K0, cols->nb[1], es * j * stride);              // [len, C_out, T]
            blk = ggml_cont(ctx, ggml_permute(ctx, blk, 0, 2, 1, 3));                                        // [len, T, C_out]
        }
        if (len < stride) blk = ggml_pad(ctx, blk, (int) (stride - len), 0, 0, 0);                     // [s, T, C_out]
        blk = codec_convtr1d_pad_rows(ctx, blk, j, J - 1 - j);                                         // [s, T+J-1, C_out]
        y = y == nullptr ? blk : ggml_add(ctx, y, blk);
    }
    y = ggml_reshape_2d(ctx, y, stride * (T + J - 1), C_out);
    return K == K0 ? y : codec_op_crop_1d(ctx, y, 0, (int32_t) (K - K0));
}

ggml_tensor * codec_convtr1d(
    ggml_context * ctx,
    ggml_tensor * x,
    ggml_tensor * w,
    ggml_tensor * b,
    int32_t stride,
    int32_t padding,
    int32_t dilation) {

    if (ctx == nullptr || x == nullptr || w == nullptr || stride <= 0 || dilation <= 0 || padding < 0) {
        return nullptr;
    }
    if (x->ne[2] > 1 && x->ne[3] <= 1) {
        // The GEMM path runs one sequence: convolve each [t, c] item, stack them.
        ggml_tensor * y = nullptr;
        for (int64_t i = 0; i < x->ne[2]; ++i) {
            ggml_tensor * xi = ggml_cont(ctx, ggml_view_3d(ctx, x, x->ne[0], x->ne[1], 1, x->nb[1], x->nb[2], (size_t) i * x->nb[2]));
            ggml_tensor * yi = codec_convtr1d(ctx, xi, w, b, stride, padding, dilation);
            if (yi == nullptr) return nullptr;
            y = y == nullptr ? yi : ggml_concat(ctx, y, yi, 2);
        }
        return y;
    }

    w = codec_graph_stored_weight(ctx, w);
    b = codec_graph_cast_f32(ctx, b);

    // The kernel runs at its stored type.  The GEMM path (dilation 1, unbatched)
    // takes any type in the C_in-major layout and F32/F16/BF16 in ggml's (it
    // transposes those); ggml's CONV_TRANSPOSE_1D takes F32/F16 (K, C_out, C_in).
    const bool ci_major = codec_convtr1d_w_ci_major(w, x->ne[1]);
    const bool gemm = dilation == 1 && x->ne[2] <= 1 && x->ne[3] <= 1 && (ci_major || x->ne[1] == w->ne[2]);
    const bool float_w = w->type == GGML_TYPE_F32 || w->type == GGML_TYPE_F16 || w->type == GGML_TYPE_BF16;
    const bool ok = gemm ? (ci_major || float_w)
                         : (!ci_major && (w->type == GGML_TYPE_F32 || w->type == GGML_TYPE_F16));
    if (!ok) {
        codec_log(CODEC_LOG_LEVEL_ERROR, "codec_convtr1d: kernel '%s' (%s, %s layout) is not supported for %s ConvTranspose1d\n",
                     w->name, ggml_type_name(w->type), ci_major ? "C_in-major" : "(K, C_out, C_in)",
                     gemm ? "this" : "a dilated/batched");
        return nullptr;
    }
    ggml_tensor * y = gemm ? codec_convtr1d_gemm(ctx, x, w, stride) : ggml_conv_transpose_1d(ctx, w, x, stride, 0, dilation);
    if (b != nullptr) {
        ggml_tensor * b2 = ggml_reshape_2d(ctx, b, 1, y->ne[1]);
        y = ggml_add(ctx, y, b2);
    }
    if (padding > 0) {
        y = codec_op_crop_1d(ctx, y, padding, padding);
    }
    return ggml_cont(ctx, y);
}

ggml_tensor * codec_convtr1d_causal(
    ggml_context * ctx,
    ggml_tensor * x,
    ggml_tensor * w,
    ggml_tensor * b,
    int32_t stride,
    int32_t dilation) {

    if (ctx == nullptr || x == nullptr || w == nullptr || stride <= 0 || dilation <= 0) {
        return nullptr;
    }

    ggml_tensor * y = codec_convtr1d(ctx, x, w, b, stride, 0, dilation);
    if (y == nullptr) {
        return nullptr;
    }

    const int32_t kernel = (int32_t) codec_convtr1d_kernel_size(w, x->ne[1]);
    const int32_t crop_right = std::max(0, kernel - stride);
    return codec_op_crop_1d(ctx, y, 0, crop_right);
}
