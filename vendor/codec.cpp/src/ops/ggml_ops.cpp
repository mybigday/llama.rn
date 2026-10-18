#include "ggml_ops.h"

#include "conv1d.h"
#include "lm_attn.h"
#include "rope.h"
#include "../runtime/graph.h"
#include "../runtime/tensor_utils.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

ggml_tensor * codec_op_unary(ggml_context * ctx, ggml_tensor * x, codec_unary_op op) {
    if (ctx == nullptr || x == nullptr) {
        return nullptr;
    }

    switch (op) {
        case CODEC_UNARY_SIGMOID: {
            ggml_tensor * x_half = ggml_scale(ctx, x, 0.5f);
            ggml_tensor * th = ggml_tanh(ctx, x_half);
            return ggml_scale_bias(ctx, th, 0.5f, 0.5f);
        }
        case CODEC_UNARY_ELU: {
            ggml_tensor * native = ggml_elu(ctx, x);
            if (codec_graph_backend_supports(native)) return native;
            // elu(x) = relu(x) + exp(min(x, 0)) - 1 for backends without ELU
            // (Vulkan: ggml_elu split the graph to the CPU).
            if (!ggml_is_contiguous(x)) x = ggml_cont(ctx, x);
            ggml_tensor * neg_part = ggml_neg(ctx, ggml_relu(ctx, ggml_neg(ctx, x)));     // min(x, 0)
            ggml_tensor * em1 = ggml_scale_bias(ctx, ggml_exp(ctx, neg_part), 1.0f, -1.0f);
            return ggml_add(ctx, ggml_relu(ctx, x), em1);
        }
        case CODEC_UNARY_SILU:
            return ggml_silu(ctx, x);
        case CODEC_UNARY_GELU_ERF:
            return ggml_gelu_erf(ctx, x);
        case CODEC_UNARY_MISH: {
            // mish(x) = x * tanh(softplus(x))
            ggml_tensor * sp = ggml_softplus(ctx, x);
            ggml_tensor * t = ggml_tanh(ctx, sp);
            return ggml_mul(ctx, x, t);
        }
        default:
            return nullptr;
    }
}

ggml_tensor * codec_op_layer_norm(ggml_context * ctx, ggml_tensor * x, float eps, ggml_tensor * gamma, ggml_tensor * beta) {
    if (ctx == nullptr || x == nullptr) {
        return nullptr;
    }

    gamma = codec_graph_cast_f32(ctx, gamma);
    beta = codec_graph_cast_f32(ctx, beta);

    ggml_tensor * y = ggml_norm(ctx, x, eps);
    if (gamma != nullptr) {
        ggml_tensor * g2 = ggml_reshape_2d(ctx, gamma, 1, x->ne[1]);
        y = ggml_mul(ctx, y, g2);
    }
    if (beta != nullptr) {
        ggml_tensor * b2 = ggml_reshape_2d(ctx, beta, 1, x->ne[1]);
        y = ggml_add(ctx, y, b2);
    }
    return y;
}

ggml_tensor * codec_op_layer_norm_ct(ggml_context * ctx, ggml_tensor * x_ct, float eps, ggml_tensor * gamma, ggml_tensor * beta) {
    if (ctx == nullptr || x_ct == nullptr || gamma == nullptr || beta == nullptr) {
        return nullptr;
    }

    gamma = codec_graph_cast_f32(ctx, gamma);
    beta = codec_graph_cast_f32(ctx, beta);

    ggml_tensor * y = ggml_norm(ctx, x_ct, eps);
    ggml_tensor * g2 = ggml_reshape_2d(ctx, gamma, x_ct->ne[0], 1);
    ggml_tensor * b2 = ggml_reshape_2d(ctx, beta, x_ct->ne[0], 1);
    y = ggml_mul(ctx, y, g2);
    y = ggml_add(ctx, y, b2);
    return y;
}

ggml_tensor * codec_op_layer_norm_tc(ggml_context * ctx, ggml_tensor * x_tc, float eps, ggml_tensor * gamma, ggml_tensor * beta) {
    if (ctx == nullptr || x_tc == nullptr || gamma == nullptr || beta == nullptr) {
        return nullptr;
    }

    ggml_tensor * x_ct = ggml_cont(ctx, ggml_transpose(ctx, x_tc));
    ggml_tensor * y_ct = codec_op_layer_norm_ct(ctx, x_ct, eps, gamma, beta);
    if (y_ct == nullptr) {
        return nullptr;
    }
    return ggml_cont(ctx, ggml_transpose(ctx, y_ct));
}

ggml_tensor * codec_op_rms_norm_ct(ggml_context * ctx, ggml_tensor * x_ct, float eps, ggml_tensor * gamma) {
    if (ctx == nullptr || x_ct == nullptr || gamma == nullptr) {
        return nullptr;
    }

    ggml_tensor * y = ggml_rms_norm(ctx, x_ct, eps);
    return codec_op_channel_scale(ctx, y, gamma);
}

ggml_tensor * codec_op_group_norm(ggml_context * ctx, ggml_tensor * x, int32_t n_groups, float eps, ggml_tensor * gamma, ggml_tensor * beta) {
    if (ctx == nullptr || x == nullptr || n_groups <= 0 || x->ne[1] % n_groups != 0) {
        return nullptr;
    }

    gamma = codec_graph_cast_f32(ctx, gamma);
    beta = codec_graph_cast_f32(ctx, beta);

    const int64_t cpg = x->ne[1] / n_groups;
    ggml_tensor * x3 = ggml_reshape_3d(ctx, x, x->ne[0], cpg, n_groups);
    ggml_tensor * y = ggml_reshape_2d(ctx, ggml_group_norm(ctx, x3, n_groups, eps), x->ne[0], x->ne[1]);

    if (gamma != nullptr) {
        ggml_tensor * g2 = ggml_reshape_2d(ctx, gamma, 1, x->ne[1]);
        y = ggml_mul(ctx, y, g2);
    }
    if (beta != nullptr) {
        ggml_tensor * b2 = ggml_reshape_2d(ctx, beta, 1, x->ne[1]);
        y = ggml_add(ctx, y, b2);
    }
    return y;
}

ggml_tensor * codec_op_linear(ggml_context * ctx, ggml_tensor * x, ggml_tensor * w, ggml_tensor * b) {
    if (ctx == nullptr || x == nullptr || w == nullptr) {
        return nullptr;
    }

    // The weight runs at its stored type (mul_mat converts x to its dot type).
    w = codec_graph_stored_weight(ctx, w);
    ggml_tensor * b_f32 = codec_graph_cast_f32(ctx, b);

    ggml_tensor * y = ggml_mul_mat(ctx, w, x);
    if (b_f32 != nullptr) {
        ggml_tensor * b2 = ggml_reshape_2d(ctx, b_f32, w->ne[1], 1);
        y = ggml_add(ctx, y, b2);
    }
    return y;
}

ggml_tensor * codec_op_linear_tc(ggml_context * ctx, ggml_tensor * x_tc, ggml_tensor * w, ggml_tensor * b) {
    if (ctx == nullptr || x_tc == nullptr || w == nullptr) {
        return nullptr;
    }

    ggml_tensor * x_ct = ggml_cont(ctx, ggml_transpose(ctx, x_tc));
    ggml_tensor * y_ct = codec_op_linear(ctx, x_ct, w, b);
    if (y_ct == nullptr) {
        return nullptr;
    }
    return ggml_cont(ctx, ggml_transpose(ctx, y_ct));
}

static ggml_tensor * codec_op_sin_poly(ggml_context * ctx, ggml_tensor * x) {
    if (!ggml_is_contiguous(x)) x = ggml_cont(ctx, x);
    // r = x - 2*pi*round(x / (2*pi)) in two steps (Cody-Waite: 6.28125 is exact).
    const float two_pi_hi = 6.28125f, two_pi_lo = 1.9353071795864769e-3f;
    ggml_tensor * k = ggml_round(ctx, ggml_scale(ctx, x, 0.15915494309189535f));
    ggml_tensor * r = ggml_sub(ctx, x, ggml_scale(ctx, k, two_pi_hi));
    r = ggml_sub(ctx, r, ggml_scale(ctx, k, two_pi_lo));
    // sin(r) = r * P(r^2), minimax on [-pi, pi].
    static const float c[6] = {9.999995828e-01f, -1.666655391e-01f, 8.332410827e-03f,
                               -1.980882371e-04f, 2.699913694e-06f, -2.036979119e-08f};
    ggml_tensor * r2 = ggml_sqr(ctx, r);
    ggml_tensor * p = ggml_scale_bias(ctx, r2, c[5], c[4]);
    for (int i = 3; i >= 0; --i) {
        p = ggml_scale_bias(ctx, ggml_mul(ctx, p, r2), 1.0f, c[i]);
    }
    return ggml_mul(ctx, p, r);
}

ggml_tensor * codec_op_sin(ggml_context * ctx, ggml_tensor * x) {
    if (ctx == nullptr || x == nullptr) return nullptr;
    ggml_tensor * native = ggml_sin(ctx, x);
    return codec_graph_backend_supports(native) ? native : codec_op_sin_poly(ctx, x);
}

ggml_tensor * codec_op_cos(ggml_context * ctx, ggml_tensor * x) {
    if (ctx == nullptr || x == nullptr) return nullptr;
    ggml_tensor * native = ggml_cos(ctx, x);
    if (codec_graph_backend_supports(native)) return native;
    if (!ggml_is_contiguous(x)) x = ggml_cont(ctx, x);
    return codec_op_sin_poly(ctx, ggml_scale_bias(ctx, x, 1.0f, 1.5707963267948966f));
}

ggml_tensor * codec_op_leaky_relu(ggml_context * ctx, ggml_tensor * x, float slope) {
    if (ctx == nullptr || x == nullptr) return nullptr;
    ggml_tensor * native = ggml_leaky_relu(ctx, x, slope, false);
    if (codec_graph_backend_supports(native)) return native;
    if (!ggml_is_contiguous(x)) x = ggml_cont(ctx, x);
    return ggml_add(ctx, ggml_relu(ctx, x), ggml_scale(ctx, ggml_relu(ctx, ggml_neg(ctx, x)), -slope));
}

ggml_tensor * codec_op_arange(ggml_context * ctx, float start, float stop, float step) {
    if (ctx == nullptr || step == 0.0f) return nullptr;
    ggml_tensor * native = ggml_arange(ctx, start, stop, step);
    if (codec_graph_backend_supports(native)) return native;
    const int64_t n = (int64_t) std::ceil((stop - start) / step);
    if (n <= 0) return nullptr;
    std::vector<float> v((size_t) n);
    for (int64_t i = 0; i < n; ++i) v[(size_t) i] = start + step * (float) i;   // ggml_arange's formula
    char name[96];
    std::snprintf(name, sizeof(name), "lut.arange.%a.%a.%a", start, stop, step);
    const int64_t ne[1] = {n};
    ggml_tensor * t = codec_graph_const(ctx, name, GGML_TYPE_F32, 1, ne, v.data());
    return t != nullptr ? t : native;
}

ggml_tensor * codec_op_snake(ggml_context * ctx, ggml_tensor * x, ggml_tensor * alpha, float eps) {
    if (ctx == nullptr || x == nullptr || alpha == nullptr) {
        return nullptr;
    }

    alpha = codec_graph_cast_f32(ctx, alpha);

    ggml_tensor * alpha_2d = ggml_reshape_2d(ctx, alpha, 1, x->ne[1]);
    // ggml_clamp is in-place: clamp a per-channel copy, then broadcast it.
    ggml_tensor * alpha_clamped = ggml_clamp(ctx, ggml_cont(ctx, alpha_2d), eps, FLT_MAX);
    ggml_tensor * ax = ggml_mul(ctx, x, alpha_clamped);
    ggml_tensor * s = codec_op_sin(ctx, ax);
    ggml_tensor * s2 = ggml_mul(ctx, s, s);
    ggml_tensor * frac = ggml_div(ctx, s2, alpha_clamped);
    return ggml_add(ctx, x, frac);
}

ggml_tensor * codec_op_snake_beta(ggml_context * ctx, ggml_tensor * x, ggml_tensor * alpha, ggml_tensor * inv_beta, float eps) {
    if (ctx == nullptr || x == nullptr || alpha == nullptr || inv_beta == nullptr) {
        return nullptr;
    }

    alpha = codec_graph_cast_f32(ctx, alpha);
    inv_beta = codec_graph_cast_f32(ctx, inv_beta);

    ggml_tensor * alpha_2d = ggml_reshape_2d(ctx, alpha, 1, x->ne[1]);
    // ggml_clamp is in-place: clamp a per-channel copy, then broadcast it.
    ggml_tensor * alpha_clamped = ggml_clamp(ctx, ggml_cont(ctx, alpha_2d), eps, FLT_MAX);
    ggml_tensor * ax = ggml_mul(ctx, x, alpha_clamped);
    ggml_tensor * s = codec_op_sin(ctx, ax);
    ggml_tensor * s2 = ggml_mul(ctx, s, s);
    ggml_tensor * invb_2d = ggml_reshape_2d(ctx, inv_beta, 1, x->ne[1]);
    ggml_tensor * frac = ggml_mul(ctx, s2, invb_2d);
    return ggml_add(ctx, x, frac);
}

ggml_tensor * codec_op_pad_1d(ggml_context * ctx, ggml_tensor * x, int32_t pad_left, int32_t pad_right) {
    if (ctx == nullptr || x == nullptr || pad_left < 0 || pad_right < 0) {
        return nullptr;
    }
    if (pad_left == 0) {
        return pad_right > 0 ? ggml_pad(ctx, x, pad_right, 0, 0, 0) : x;
    }
    // Metal's PAD only pads at the end: prepend a zero constant instead.
    const int64_t ne[4] = {pad_left, x->ne[1], x->ne[2], x->ne[3]};
    const std::string name = std::string("lut.zeros.") + ggml_type_name(x->type) + "." + std::to_string(ne[0]) +
                             "x" + std::to_string(ne[1]) + "x" + std::to_string(ne[2]) + "x" + std::to_string(ne[3]);
    std::vector<uint8_t> zeros((size_t) (ggml_row_size(x->type, ne[0]) * ne[1] * ne[2] * ne[3]), 0);
    ggml_tensor * z = codec_graph_const(ctx, name.c_str(), x->type, 4, ne, zeros.data());
    if (z == nullptr) {
        // Not inside a graph build: fall back to the native op.
        return ggml_pad_ext(ctx, x, pad_left, pad_right, 0, 0, 0, 0, 0, 0);
    }
    ggml_tensor * y = ggml_concat(ctx, z, x, 0);
    return pad_right > 0 ? ggml_pad(ctx, y, pad_right, 0, 0, 0) : y;
}

ggml_tensor * codec_op_pad_reflect_tc(ggml_context * ctx, ggml_tensor * x_tc, int32_t pad_left, int32_t pad_right) {
    if (ctx == nullptr || x_tc == nullptr || pad_left < 0 || pad_right < 0) {
        return nullptr;
    }
    const int64_t t = x_tc->ne[0];
    if (pad_left == 0 && pad_right == 0) {
        return x_tc;
    }
    if (pad_left >= t || pad_right >= t) {
        return nullptr;
    }
    std::vector<int32_t> idx((size_t) (t + pad_left + pad_right));
    for (int64_t i = 0; i < (int64_t) idx.size(); ++i) {
        int64_t j = i - pad_left;
        if (j < 0) j = -j;
        if (j >= t) j = 2 * (t - 1) - j;
        idx[(size_t) i] = (int32_t) j;
    }
    const std::string name = "lut.reflect." + std::to_string(t) + "." + std::to_string(pad_left) + "." +
                             std::to_string(pad_right);
    const int64_t ne[1] = {(int64_t) idx.size()};
    ggml_tensor * rows = codec_graph_const(ctx, name.c_str(), GGML_TYPE_I32, 1, ne, idx.data());
    if (rows == nullptr) {
        return nullptr;
    }
    ggml_tensor * x_ct = ggml_cont(ctx, ggml_transpose(ctx, x_tc));               // [c, t]: rows are time steps
    return ggml_cont(ctx, ggml_transpose(ctx, ggml_get_rows(ctx, x_ct, rows)));    // [t + pads, c]
}

ggml_tensor * codec_op_clamp_min(ggml_context * ctx, ggml_tensor * x, float lo) {
    if (ctx == nullptr || x == nullptr) {
        return nullptr;
    }
    return ggml_scale_bias(ctx, ggml_relu(ctx, ggml_scale_bias(ctx, x, 1.0f, -lo)), 1.0f, lo);
}

ggml_tensor * codec_op_round_bounded(ggml_context * ctx, ggml_tensor * x, int32_t max_abs) {
    if (ctx == nullptr || x == nullptr || max_abs <= 0) {
        return nullptr;
    }
    ggml_tensor * xc = ggml_is_contiguous(x) ? x : ggml_cont(ctx, x);
    const int64_t n = ggml_nelements(xc);
    // [max_abs, n]: every element against thresholds t = 0.5, 1.5, ...
    // round(x) = #{t : x >= t} - #{t : -x >= t} = sum_t step(x + t) - step(t - x).
    ggml_tensor * xr = ggml_repeat_4d(ctx, ggml_reshape_2d(ctx, xc, 1, n), max_abs, n, 1, 1);
    ggml_tensor * thr = codec_op_arange(ctx, 0.5f, (float) max_abs, 1.0f);
    ggml_tensor * up = ggml_sum_rows(ctx, ggml_step(ctx, ggml_add(ctx, xr, thr)));
    ggml_tensor * dn = ggml_sum_rows(ctx, ggml_step(ctx, ggml_add(ctx, ggml_scale(ctx, xr, -1.0f), thr)));
    return ggml_reshape(ctx, ggml_sub(ctx, up, dn), xc);
}

ggml_tensor * codec_op_pad_1d_replicate(ggml_context * ctx, ggml_tensor * x, int32_t pad_left, int32_t pad_right) {
    if (ctx == nullptr || x == nullptr || pad_left < 0 || pad_right < 0) {
        return nullptr;
    }
    if (pad_left == 0 && pad_right == 0) {
        return x;
    }

    ggml_tensor * out = x;
    if (pad_left > 0) {
        ggml_tensor * left = ggml_view_2d(ctx, x, 1, x->ne[1], x->nb[1], 0);
        ggml_tensor * left_target = ggml_new_tensor_2d(ctx, x->type, pad_left, x->ne[1]);
        ggml_tensor * left_rep = ggml_repeat(ctx, left, left_target);
        out = ggml_concat(ctx, left_rep, out, 0);
    }

    if (pad_right > 0) {
        const size_t offset = (size_t) (x->ne[0] - 1) * (size_t) x->nb[0];
        ggml_tensor * right = ggml_view_2d(ctx, x, 1, x->ne[1], x->nb[1], offset);
        ggml_tensor * right_target = ggml_new_tensor_2d(ctx, x->type, pad_right, x->ne[1]);
        ggml_tensor * right_rep = ggml_repeat(ctx, right, right_target);
        out = ggml_concat(ctx, out, right_rep, 0);
    }

    return ggml_cont(ctx, out);
}

ggml_tensor * codec_op_crop_1d(ggml_context * ctx, ggml_tensor * x, int32_t crop_left, int32_t crop_right) {
    if (ctx == nullptr || x == nullptr || crop_left < 0 || crop_right < 0) {
        return nullptr;
    }
    const int64_t out_t = x->ne[0] - crop_left - crop_right;
    if (out_t <= 0) {
        return nullptr;
    }
    ggml_tensor * view = ggml_view_2d(ctx, x, out_t, x->ne[1], x->nb[1], (size_t) crop_left * sizeof(float));
    return ggml_cont(ctx, view);
}

ggml_tensor * codec_op_causal_crop_1d(ggml_context * ctx, ggml_tensor * x, int32_t target_t) {
    if (ctx == nullptr || x == nullptr || target_t <= 0 || x->ne[0] < target_t) {
        return nullptr;
    }
    return codec_op_crop_1d(ctx, x, 0, (int32_t)x->ne[0] - target_t);
}

ggml_tensor * codec_op_channel_scale(ggml_context * ctx, ggml_tensor * x, ggml_tensor * scale) {
    if (ctx == nullptr || x == nullptr || scale == nullptr) {
        return nullptr;
    }
    scale = codec_graph_cast_f32(ctx, scale);
    // ggml_mul broadcasts src1 over ne>=1 dims natively (ne[1]=1 repeats over
    // x's ne[1]); no explicit ggml_repeat needed.  scale is (ne[0],) → view as
    // (ne[0], 1) so the leading dims match x for the broadcast.
    ggml_tensor * s2 = ggml_reshape_2d(ctx, scale, x->ne[0], 1);
    return ggml_mul(ctx, x, s2);
}

ggml_tensor * codec_op_tokens_to_features(ggml_context * ctx, ggml_tensor * tokens, int32_t out_channels) {
    if (ctx == nullptr || tokens == nullptr || out_channels <= 0) {
        return nullptr;
    }

    ggml_tensor * x = ggml_scale(ctx, tokens, 1.0f / 1024.0f);
    ggml_tensor * base = nullptr;
    if (x->ne[1] <= 1) {
        base = ggml_cont(ctx, ggml_view_2d(ctx, x, x->ne[0], 1, x->nb[1], 0));
    } else {
        // Aggregate all quantizers instead of only using q=0.
        ggml_tensor * x_qt = ggml_cont(ctx, ggml_transpose(ctx, x)); // [q, t]
        ggml_tensor * mean_1t = ggml_scale(ctx, ggml_sum_rows(ctx, x_qt), 1.0f / (float) x_qt->ne[0]); // [1, t]
        base = ggml_cont(ctx, ggml_transpose(ctx, mean_1t));         // [t, 1]
    }
    if (out_channels == 1) {
        return base;
    }
    return ggml_repeat(ctx, base, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, x->ne[0], out_channels));
}

ggml_tensor * codec_op_convnext_block_ct(
    ggml_context * ctx,
    ggml_tensor * x_ct,
    ggml_tensor * dw_w,
    ggml_tensor * dw_b,
    ggml_tensor * ln_w,
    ggml_tensor * ln_b,
    ggml_tensor * pw1_w,
    ggml_tensor * pw1_b,
    ggml_tensor * pw2_w,
    ggml_tensor * pw2_b,
    ggml_tensor * gamma,
    int32_t dw_padding) {

    if (ctx == nullptr || x_ct == nullptr || dw_w == nullptr || ln_w == nullptr || pw1_w == nullptr || pw2_w == nullptr) {
        return nullptr;
    }

    ggml_tensor * res_tc = ggml_cont(ctx, ggml_transpose(ctx, x_ct)); // [t, c]
    ggml_tensor * x_dw = codec_conv1d_depthwise(ctx, res_tc, dw_w, dw_b, 1, 1, dw_padding);
    if (x_dw == nullptr) {
        return nullptr;
    }

    ggml_tensor * h_ct = codec_op_layer_norm_ct(
        ctx, ggml_cont(ctx, ggml_transpose(ctx, x_dw)), 1e-6f, ln_w, ln_b);
    if (h_ct == nullptr) {
        return nullptr;
    }

    ggml_tensor * pw1 = codec_op_linear(ctx, h_ct, pw1_w, pw1_b);
    if (pw1 == nullptr) {
        return nullptr;
    }
    pw1 = ggml_gelu_erf(ctx, pw1);

    ggml_tensor * pw2 = codec_op_linear(ctx, pw1, pw2_w, pw2_b);
    if (pw2 == nullptr) {
        return nullptr;
    }
    if (gamma != nullptr) {
        pw2 = codec_op_channel_scale(ctx, pw2, gamma);
        if (pw2 == nullptr) {
            return nullptr;
        }
    }
    return ggml_add(ctx, x_ct, pw2);
}

ggml_tensor * codec_op_causal_block1d_tc(
    ggml_context * ctx,
    ggml_tensor * x_tc,
    ggml_tensor * conv_w,
    ggml_tensor * conv_b,
    ggml_tensor * ln_w,
    ggml_tensor * ln_b) {
    if (ctx == nullptr || x_tc == nullptr || conv_w == nullptr) return nullptr;
    ggml_tensor * y = codec_conv1d_causal(ctx, x_tc, conv_w, conv_b, /*stride=*/1, /*dilation=*/1);
    if (y == nullptr) return nullptr;
    y = codec_op_layer_norm_tc(ctx, y, /*eps=*/1e-5f, ln_w, ln_b);
    if (y == nullptr) return nullptr;
    return codec_op_unary(ctx, y, CODEC_UNARY_MISH);
}

ggml_tensor * codec_op_hifigan_resblock_branch_ct(
    ggml_context * ctx,
    ggml_tensor * x_tc,
    ggml_tensor * a1,
    ggml_tensor * a2,
    ggml_tensor * c1_w,
    ggml_tensor * c1_b,
    ggml_tensor * c2_w,
    ggml_tensor * c2_b,
    int32_t kernel_size,
    int32_t dilation) {
    if (ctx == nullptr || x_tc == nullptr || a1 == nullptr || a2 == nullptr ||
        c1_w == nullptr || c2_w == nullptr || kernel_size <= 0 || dilation <= 0) {
        return nullptr;
    }
    const int32_t pad1 = (kernel_size * dilation - dilation) / 2;
    const int32_t pad2 = (kernel_size - 1) / 2;
    ggml_tensor * h = codec_op_snake(ctx, x_tc, a1, /*eps=*/1e-9f);
    if (h == nullptr) return nullptr;
    h = codec_conv1d(ctx, h, c1_w, c1_b, /*stride=*/1, /*dilation=*/dilation, /*padding=*/pad1);
    if (h == nullptr) return nullptr;
    h = codec_op_snake(ctx, h, a2, /*eps=*/1e-9f);
    if (h == nullptr) return nullptr;
    h = codec_conv1d(ctx, h, c2_w, c2_b, /*stride=*/1, /*dilation=*/1, /*padding=*/pad2);
    if (h == nullptr) return nullptr;
    return ggml_add(ctx, h, x_tc);
}

ggml_tensor * codec_op_cfm_causal_resnet_block_tc(
    ggml_context * ctx,
    ggml_tensor * x_tc,
    ggml_tensor * t_emb,
    ggml_tensor * b1_conv_w, ggml_tensor * b1_conv_b,
    ggml_tensor * b1_ln_w,   ggml_tensor * b1_ln_b,
    ggml_tensor * b2_conv_w, ggml_tensor * b2_conv_b,
    ggml_tensor * b2_ln_w,   ggml_tensor * b2_ln_b,
    ggml_tensor * mlp_w,     ggml_tensor * mlp_b,
    ggml_tensor * res_w,     ggml_tensor * res_b) {
    if (ctx == nullptr || x_tc == nullptr || t_emb == nullptr ||
        b1_conv_w == nullptr || b2_conv_w == nullptr ||
        mlp_w == nullptr || res_w == nullptr) return nullptr;

    ggml_tensor * h = codec_op_causal_block1d_tc(ctx, x_tc, b1_conv_w, b1_conv_b, b1_ln_w, b1_ln_b);
    if (h == nullptr) return nullptr;

    // Time embedding mlp: mish(t_emb) → Linear(time_embed_dim → out_dim) → broadcast over t.
    ggml_tensor * tm = codec_op_unary(ctx, t_emb, CODEC_UNARY_MISH);
    if (tm == nullptr) return nullptr;
    ggml_tensor * tm_2d = ggml_reshape_2d(ctx, tm, tm->ne[0], 1);
    ggml_tensor * tm_proj = codec_mul_mat(ctx, mlp_w, tm_2d);  // [out, 1]
    if (tm_proj == nullptr) return nullptr;
    if (mlp_b != nullptr) {
        ggml_tensor * mlp_b_2d = ggml_reshape_2d(ctx, mlp_b, mlp_b->ne[0], 1);
        tm_proj = ggml_add(ctx, tm_proj, mlp_b_2d);
    }
    ggml_tensor * tm_to = ggml_cont(ctx, ggml_transpose(ctx, tm_proj));  // [1, out]
    h = ggml_add(ctx, h, tm_to);

    h = codec_op_causal_block1d_tc(ctx, h, b2_conv_w, b2_conv_b, b2_ln_w, b2_ln_b);
    if (h == nullptr) return nullptr;

    ggml_tensor * res = codec_conv1d(ctx, x_tc, res_w, res_b, /*stride=*/1, /*dilation=*/1, /*padding=*/0);
    if (res == nullptr) return nullptr;
    return ggml_add(ctx, h, res);
}

ggml_tensor * codec_op_basic_transformer_block_ct(
    ggml_context * ctx,
    ggml_tensor * x_ct,
    ggml_tensor * norm1_w, ggml_tensor * norm1_b,
    ggml_tensor * qw, ggml_tensor * kw, ggml_tensor * vw,
    ggml_tensor * ow, ggml_tensor * ob,
    ggml_tensor * norm3_w, ggml_tensor * norm3_b,
    ggml_tensor * ff1_w, ggml_tensor * ff1_b,
    ggml_tensor * ff2_w, ggml_tensor * ff2_b,
    int32_t head_dim,
    int32_t num_heads) {
    if (ctx == nullptr || x_ct == nullptr || qw == nullptr || kw == nullptr || vw == nullptr ||
        ow == nullptr || ff1_w == nullptr || ff2_w == nullptr || head_dim <= 0 || num_heads <= 0) {
        return nullptr;
    }
    const int32_t inner = head_dim * num_heads;
    const int32_t T = (int32_t) x_ct->ne[1];

    ggml_tensor * h_ct = codec_op_layer_norm_ct(ctx, x_ct, 1e-5f, norm1_w, norm1_b);
    if (h_ct == nullptr) return nullptr;
    ggml_tensor * q_ct = codec_op_linear(ctx, h_ct, qw, /*b=*/nullptr);
    ggml_tensor * k_ct = codec_op_linear(ctx, h_ct, kw, /*b=*/nullptr);
    ggml_tensor * v_ct = codec_op_linear(ctx, h_ct, vw, /*b=*/nullptr);
    if (q_ct == nullptr || k_ct == nullptr || v_ct == nullptr) return nullptr;

    auto to_dth = [&](ggml_tensor * x_ct_in) {
        ggml_tensor * r = ggml_reshape_3d(ctx, x_ct_in, head_dim, num_heads, T);
        return ggml_cont(ctx, ggml_permute(ctx, r, 0, 2, 1, 3));
    };
    codec_lm_attn_params attn_p = {};
    attn_p.scale = 1.0f / std::sqrt((float) head_dim);
    attn_p.causal = false;
    ggml_tensor * attn_dth = codec_op_lm_attn_ctx_dth(ctx, to_dth(q_ct), to_dth(k_ct), to_dth(v_ct), &attn_p);
    if (attn_dth == nullptr) return nullptr;

    ggml_tensor * attn_ct = ggml_reshape_2d(
        ctx,
        ggml_cont(ctx, ggml_permute(ctx, attn_dth, 0, 2, 1, 3)),
        inner, T);
    ggml_tensor * proj_ct = codec_op_linear(ctx, attn_ct, ow, ob);
    if (proj_ct == nullptr) return nullptr;
    x_ct = ggml_add(ctx, x_ct, proj_ct);

    ggml_tensor * f_ct = codec_op_layer_norm_ct(ctx, x_ct, 1e-5f, norm3_w, norm3_b);
    if (f_ct == nullptr) return nullptr;
    ggml_tensor * ff = codec_op_linear(ctx, f_ct, ff1_w, ff1_b);
    if (ff == nullptr) return nullptr;
    ff = codec_op_unary(ctx, ff, CODEC_UNARY_GELU_ERF);
    ff = codec_op_linear(ctx, ff, ff2_w, ff2_b);
    if (ff == nullptr) return nullptr;
    return ggml_add(ctx, x_ct, ff);
}

ggml_tensor * codec_op_whisper_encoder_layer_tc(
    ggml_context * ctx,
    ggml_tensor * x_tc,
    ggml_tensor * n1w, ggml_tensor * n1b,
    ggml_tensor * qw,  ggml_tensor * qb,
    ggml_tensor * kw,
    ggml_tensor * vw,  ggml_tensor * vb,
    ggml_tensor * ow,  ggml_tensor * ob,
    ggml_tensor * n2w, ggml_tensor * n2b,
    ggml_tensor * fc1w, ggml_tensor * fc1b,
    ggml_tensor * fc2w, ggml_tensor * fc2b,
    int32_t head_dim,
    int32_t n_heads,
    int32_t n_valid) {

    if (ctx == nullptr || x_tc == nullptr || qw == nullptr || kw == nullptr || vw == nullptr ||
        ow == nullptr || fc1w == nullptr || fc2w == nullptr || head_dim <= 0 || n_heads <= 0) {
        return nullptr;
    }
    const int64_t t      = x_tc->ne[0];
    const int32_t hidden = head_dim * n_heads;

    ggml_tensor * res = x_tc;
    ggml_tensor * h = codec_op_layer_norm_tc(ctx, x_tc, 1e-5f, n1w, n1b);
    if (h == nullptr) return nullptr;
    ggml_tensor * h_ct = ggml_cont(ctx, ggml_transpose(ctx, h));     // [c, t]

    auto linear_with_bias = [&](ggml_tensor * w, ggml_tensor * b) -> ggml_tensor * {
        ggml_tensor * y = codec_mul_mat(ctx, w, h_ct);                // [out, t]
        if (y == nullptr) return nullptr;
        if (b != nullptr) {
            ggml_tensor * b2 = ggml_reshape_2d(ctx, b, b->ne[0], 1);
            y = ggml_add(ctx, y, b2);
        }
        return y;
    };

    ggml_tensor * q = linear_with_bias(qw, qb);                       // [hidden, t]
    ggml_tensor * k = linear_with_bias(kw, nullptr);                  // k: bias-free
    ggml_tensor * v = linear_with_bias(vw, vb);
    if (q == nullptr || k == nullptr || v == nullptr) return nullptr;

    auto to_dth = [&](ggml_tensor * x_ct_in) -> ggml_tensor * {
        ggml_tensor * r = ggml_reshape_3d(ctx, x_ct_in, head_dim, n_heads, t);
        return ggml_cont(ctx, ggml_permute(ctx, r, 0, 2, 1, 3));      // [d, t, h]
    };

    codec_lm_attn_params attn_p = {};
    attn_p.scale   = 1.0f / std::sqrt((float) head_dim);
    attn_p.causal  = false;
    attn_p.window  = 0;
    attn_p.n_valid = n_valid;
    ggml_tensor * attn_dth = codec_op_lm_attn_ctx_dth(ctx, to_dth(q), to_dth(k), to_dth(v), &attn_p);
    if (attn_dth == nullptr) return nullptr;
    ggml_tensor * attn_dht = ggml_cont(ctx, ggml_permute(ctx, attn_dth, 0, 2, 1, 3));
    ggml_tensor * attn_ct  = ggml_reshape_2d(ctx, attn_dht, hidden, t);

    ggml_tensor * out_ct = codec_mul_mat(ctx, ow, attn_ct);             // [hidden, t]
    if (ob != nullptr) {
        ggml_tensor * ob_2d = ggml_reshape_2d(ctx, ob, ob->ne[0], 1);
        out_ct = ggml_add(ctx, out_ct, ob_2d);
    }
    ggml_tensor * out_tc = ggml_cont(ctx, ggml_transpose(ctx, out_ct));
    x_tc = ggml_add(ctx, res, out_tc);

    res = x_tc;
    h = codec_op_layer_norm_tc(ctx, x_tc, 1e-5f, n2w, n2b);
    h_ct = ggml_cont(ctx, ggml_transpose(ctx, h));
    ggml_tensor * f1 = linear_with_bias(fc1w, fc1b);                   // [ffn, t]
    ggml_tensor * f1_tc = ggml_cont(ctx, ggml_transpose(ctx, f1));
    f1_tc = ggml_gelu_erf(ctx, f1_tc);
    f1 = ggml_cont(ctx, ggml_transpose(ctx, f1_tc));                   // [ffn, t]
    ggml_tensor * f2 = codec_mul_mat(ctx, fc2w, f1);                    // [hidden, t]
    if (fc2b != nullptr) {
        ggml_tensor * b2 = ggml_reshape_2d(ctx, fc2b, fc2b->ne[0], 1);
        f2 = ggml_add(ctx, f2, b2);
    }
    ggml_tensor * f2_tc = ggml_cont(ctx, ggml_transpose(ctx, f2));
    return ggml_add(ctx, res, f2_tc);
}

ggml_tensor * codec_op_add_sliced_pos_emb_tc(
    ggml_context * ctx,
    ggml_tensor * x_tc,
    ggml_tensor * pos) {
    if (ctx == nullptr || x_tc == nullptr || pos == nullptr) return x_tc;
    const int64_t t = x_tc->ne[0];
    if (t <= 0) return x_tc;
    ggml_tensor * pos_view = ggml_view_2d(ctx, pos, pos->ne[0], t, pos->nb[1], 0);
    ggml_tensor * pos_tc   = ggml_cont(ctx, ggml_transpose(ctx, pos_view));
    return ggml_add(ctx, x_tc, pos_tc);
}

ggml_tensor * codec_op_l2_normalize_tc(
    ggml_context * ctx,
    ggml_tensor * x_tc,
    float eps) {
    if (ctx == nullptr || x_tc == nullptr) return nullptr;
    ggml_tensor * x_ct = ggml_cont(ctx, ggml_transpose(ctx, x_tc));            // [c, t]
    ggml_tensor * sq = ggml_mul(ctx, x_ct, x_ct);
    ggml_tensor * sum_sq = ggml_sum_rows(ctx, sq);                             // [1, t]
    sum_sq = ggml_scale_bias(ctx, sum_sq, 1.0f, eps);
    ggml_tensor * norm = ggml_sqrt(ctx, sum_sq);
    ggml_tensor * ones = ggml_scale_bias(ctx, norm, 0.0f, 1.0f);
    ggml_tensor * inv = ggml_div(ctx, ones, norm);
    ggml_tensor * x_n = ggml_mul(ctx, x_ct, inv);
    return ggml_cont(ctx, ggml_transpose(ctx, x_n));                           // [t, c]
}

ggml_tensor * codec_op_sinusoidal_time_emb(
    ggml_context * ctx,
    float t_v,
    int32_t dim,
    float scale) {
    if (ctx == nullptr || dim <= 0 || (dim & 1) != 0) return nullptr;
    const int32_t half = dim / 2;
    ggml_tensor * idx = codec_op_arange(ctx, 0.0f, (float) half, 1.0f);
    ggml_tensor * log_idx = ggml_scale(ctx, idx, -std::log(10000.0f) / (float) (half - 1));
    ggml_tensor * freqs = ggml_exp(ctx, log_idx);
    ggml_tensor * e = ggml_scale(ctx, freqs, t_v * scale);
    return ggml_concat(ctx, codec_op_sin(ctx, e), codec_op_cos(ctx, e), /*dim=*/0);
}

// Shared-filter FIR along time as block-Toeplitz GEMMs:
//   y[n, c] = scale * sum_m x[m, c] * h[tap(n, m)]     (tap outside [0, K): no term)
// with tap shift-invariant under (n + bo, m + bi).  Time is cut into blocks of
// bi input / bo output samples; output block j is M0 . x_blk[j] + M1 . x_blk[j+1]
// with [bi, bo] matrices gathered from h in-graph.  Replaces a depthwise im2col
// conv (a K-fold copy of x) by one small GEMM: the same F32 products in a
// different summation order.  nullptr when a tap reaches past those two blocks
// or outside a graph build.
static ggml_tensor * codec_op_fir_blocked_tc(
    ggml_context * ctx, ggml_tensor * x_tc, ggml_tensor * h, float scale,
    int64_t bi, int64_t bo, int64_t t_out, const std::function<int64_t(int64_t, int64_t)> & tap) {

    const int64_t K = h->ne[0], c = x_tc->ne[1];
    auto valid = [&](int64_t k) { return k >= 0 && k < K; };
    // Block j = 1 keeps every index non-negative; blocks j-1 and j+2 must be empty.
    std::vector<int32_t> idx((size_t) (2 * bi * bo));
    for (int64_t o = 0; o < bo; ++o) {
        for (int64_t i = 0; i < bi; ++i) {
            if (valid(tap(bo + o, i)) || valid(tap(bo + o, 3 * bi + i))) return nullptr;
            for (int64_t r = 0; r < 2; ++r) {
                const int64_t k = tap(bo + o, (1 + r) * bi + i);
                idx[(size_t) ((r * bo + o) * bi + i)] = (int32_t) (valid(k) ? k : K);   // K -> appended zero
            }
        }
    }
    const std::string sig(reinterpret_cast<const char *>(idx.data()), idx.size() * sizeof(int32_t));
    const std::string key = std::to_string(K) + "." + std::to_string(bi) + "." + std::to_string(bo) + "." +
                            std::to_string(std::hash<std::string>{}(sig));
    const int64_t ne_idx[1] = {2 * bi * bo};
    ggml_tensor * idx_t = codec_graph_const(ctx, ("lut.fir_idx." + key).c_str(), GGML_TYPE_I32, 1, ne_idx, idx.data());
    const float zero = 0.0f;
    const int64_t ne_z[1] = {1};
    ggml_tensor * z = codec_graph_const(ctx, "lut.zeros.f32.1", GGML_TYPE_F32, 1, ne_z, &zero);
    if (idx_t == nullptr || z == nullptr) return nullptr;

    // [1, K+1] rows of h (scaled) plus a zero row -> gathered [bi, 2*bo]: M0 | M1.
    ggml_tensor * hz = ggml_concat(ctx, ggml_scale(ctx, codec_graph_cast_f32(ctx, h), scale), z, 0);
    ggml_tensor * m = ggml_get_rows(ctx, ggml_reshape_2d(ctx, hz, 1, K + 1), idx_t);       // [1, 2*bo*bi]
    m = ggml_reshape_2d(ctx, m, bi, 2 * bo);

    const int64_t nblk = (t_out + bo - 1) / bo;
    const int64_t len = (nblk + 1) * bi;
    ggml_tensor * x = x_tc;
    if (x->ne[0] > len) x = ggml_cont(ctx, ggml_view_2d(ctx, x, len, c, x->nb[1], 0));
    if (x->ne[0] < len) x = ggml_pad(ctx, x, (int) (len - x->ne[0]), 0, 0, 0);
    if (!ggml_is_contiguous(x)) x = ggml_cont(ctx, x);
    // One GEMM over every block (contiguous operands only: ggml-vulkan mishandled
    // a strided [bi, nblk, c] view as src1), then block j of the output is
    // M0 . x_blk[j] + M1 . x_blk[j+1] = rows [0, bo) of block j + rows [bo, 2bo)
    // of block j+1.
    // A single 2D GEMM over the flattened [bi, (nblk+1)*c] blocks rather than a
    // batch of c small ones (Metal ran the batched form ~5x slower).
    ggml_tensor * yy = ggml_mul_mat(ctx, m, ggml_reshape_2d(ctx, x, bi, (nblk + 1) * c));
    yy = ggml_reshape_3d(ctx, yy, 2 * bo, nblk + 1, c);                                   // [2*bo, nblk+1, c]
    ggml_tensor * y0 = ggml_view_3d(ctx, yy, bo, nblk, c, yy->nb[1], yy->nb[2], 0);
    ggml_tensor * y1 = ggml_view_3d(ctx, yy, bo, nblk, c, yy->nb[1], yy->nb[2], yy->nb[1] + bo * yy->nb[0]);
    ggml_tensor * y = ggml_add(ctx, y0, y1);   // row-contiguous views: fine for binary ops everywhere
    y = ggml_reshape_2d(ctx, y, bo * nblk, c);
    return y->ne[0] == t_out ? y : ggml_cont(ctx, ggml_view_2d(ctx, y, t_out, c, y->nb[1], 0));
}

ggml_tensor * codec_op_alias_free_snake_beta_tc(
    ggml_context * ctx,
    ggml_tensor * x_tc,
    ggml_tensor * alpha,
    ggml_tensor * inv_beta,
    ggml_tensor * kernel_12) {

    if (ctx == nullptr || x_tc == nullptr || alpha == nullptr || inv_beta == nullptr || kernel_12 == nullptr) {
        return nullptr;
    }
    if (kernel_12->ne[0] != 12) return nullptr;
    const int64_t c = x_tc->ne[1];

    // Replicate-pad the input by 5 each side.  This matches alias_free_torch's
    // `self.pad = kernel_size // ratio - 1 = 5` before the conv-transpose.
    ggml_tensor * x_rp = codec_op_pad_1d_replicate(ctx, x_tc, 5, 5);
    if (x_rp == nullptr) return nullptr;
    const int64_t t_rp = x_rp->ne[0];

    // Zero-insert by 2: rebuild the padded sequence as
    // [x[0], 0, x[1], 0, …, x[t_rp-1], 0] using ggml_pad along a new
    // innermost axis.  Reshape (t_rp, c) → (1, t_rp, c) → ggml_pad(0, 1, 0…)
    // → (2, t_rp, c) with the second slot zeroed → reshape to (2*t_rp, c).
    // Upsample: zero-insert by 2, pad 11, 12-tap correlation, x2, trim 1 + 15/15
    // collapses to y[n] = 2 * sum_m x_rp[m] * h[2m - n - 4], n < 2t.
    ggml_tensor * y_up = codec_op_fir_blocked_tc(ctx, x_rp, kernel_12, 2.0f, 16, 32, 2 * x_tc->ne[0],
        [](int64_t n, int64_t m) { return 2 * m - n - 4; });
    if (y_up != nullptr) {
        ggml_tensor * y_act = codec_op_snake_beta(ctx, y_up, alpha, inv_beta, /*eps=*/1e-9f);
        if (y_act == nullptr) return nullptr;
        // Downsample: replicate-pad (5, 6), 12-tap correlation at stride 2.
        ggml_tensor * d_pad = codec_op_pad_1d_replicate(ctx, y_act, 5, 6);
        ggml_tensor * y_dn = codec_op_fir_blocked_tc(ctx, d_pad, kernel_12, 1.0f, 32, 16, x_tc->ne[0],
            [](int64_t n, int64_t m) { return m - 2 * n; });
        if (y_dn != nullptr) return y_dn;
        ggml_tensor * w_dw = ggml_repeat(ctx, ggml_reshape_3d(ctx, kernel_12, 12, 1, 1),
                                         ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 12, 1, c));
        return codec_conv1d_depthwise(ctx, d_pad, w_dw, nullptr, /*stride=*/2, /*dilation=*/1, /*padding=*/0);
    }

    ggml_tensor * x_3d = ggml_reshape_3d(ctx, x_rp, 1, t_rp, c);
    ggml_tensor * x_zero_3d = ggml_pad(ctx, x_3d, 1, 0, 0, 0);  // ne=(2, t_rp, c)
    if (x_zero_3d == nullptr) return nullptr;
    x_zero_3d = ggml_cont(ctx, x_zero_3d);
    ggml_tensor * x_zi = ggml_reshape_2d(ctx, x_zero_3d, 2 * t_rp, c);

    // Pad with 11 zeros on each side, then depthwise-correlate with the
    // (symmetric) 12-tap Kaiser kernel.  Conv-transpose stride 2 of input
    // length L produces 2L+10 outputs; our zero-inserted/padded direct-conv
    // produces 2L+11, one too long, so we trim the trailing sample after.
    ggml_tensor * x_zp = codec_op_pad_1d(ctx, x_zi, 11, 11);

    ggml_tensor * w_3d = ggml_reshape_3d(ctx, kernel_12, 12, 1, 1);
    ggml_tensor * w_dst = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 12, 1, c);
    ggml_tensor * w_dw = ggml_repeat(ctx, w_3d, w_dst);

    ggml_tensor * y = codec_conv1d_depthwise(ctx, x_zp, w_dw, nullptr, /*stride=*/1, /*dilation=*/1, /*padding=*/0);
    if (y == nullptr) return nullptr;
    // y length = (2*t_rp + 22) - 12 + 1 = 2*t_rp + 11.  Trim 1 from end → 2*t_rp + 10.
    y = codec_op_crop_1d(ctx, y, 0, 1);

    // Multiply by ratio=2 (matches `x = ratio * F.conv_transpose1d(...)`).
    y = ggml_scale(ctx, y, 2.0f);

    // Trim Activation1d's pad_left=15 / pad_right=15 → length 2t.
    y = codec_op_crop_1d(ctx, y, 15, 15);

    // Snake-beta non-linearity at the upsampled rate.
    ggml_tensor * y_act = codec_op_snake_beta(ctx, y, alpha, inv_beta, /*eps=*/1e-9f);
    if (y_act == nullptr) return nullptr;

    // Downsample 2×: replicate-pad (5, 6) then depthwise conv stride=2.
    ggml_tensor * d_pad = codec_op_pad_1d_replicate(ctx, y_act, 5, 6);
    return codec_conv1d_depthwise(ctx, d_pad, w_dw, nullptr, /*stride=*/2, /*dilation=*/1, /*padding=*/0);
}

ggml_tensor * codec_op_vocos_resnet_block_tc(
    ggml_context * ctx,
    ggml_tensor * x_tc,
    ggml_tensor * n1_w, ggml_tensor * n1_b,
    ggml_tensor * c1_w, ggml_tensor * c1_b,
    ggml_tensor * n2_w, ggml_tensor * n2_b,
    ggml_tensor * c2_w, ggml_tensor * c2_b) {

    if (ctx == nullptr || x_tc == nullptr ||
        n1_w == nullptr || n1_b == nullptr || c1_w == nullptr || c1_b == nullptr ||
        n2_w == nullptr || n2_b == nullptr || c2_w == nullptr || c2_b == nullptr) {
        return nullptr;
    }

    ggml_tensor * h = codec_op_group_norm(ctx, x_tc, 32, 1e-6f, n1_w, n1_b);
    if (h == nullptr) return nullptr;
    h = ggml_silu(ctx, h);
    h = codec_conv1d(ctx, h, c1_w, c1_b, 1, 1, 1);
    if (h == nullptr) return nullptr;
    h = codec_op_group_norm(ctx, h, 32, 1e-6f, n2_w, n2_b);
    if (h == nullptr) return nullptr;
    h = ggml_silu(ctx, h);
    h = codec_conv1d(ctx, h, c2_w, c2_b, 1, 1, 1);
    if (h == nullptr) return nullptr;
    return ggml_add(ctx, x_tc, h);
}

ggml_tensor * codec_op_roformer_block_ct(
    ggml_context * ctx,
    ggml_tensor * x_ct,
    ggml_tensor * att_norm_w,
    ggml_tensor * ffn_norm_w,
    ggml_tensor * c_attn_w,
    ggml_tensor * c_proj_w,
    ggml_tensor * fc1_w,
    ggml_tensor * fc2_w,
    int32_t head_dim,
    int32_t n_heads,
    float rope_theta) {

    if (ctx == nullptr || x_ct == nullptr || att_norm_w == nullptr || ffn_norm_w == nullptr ||
        c_attn_w == nullptr || c_proj_w == nullptr || fc1_w == nullptr || fc2_w == nullptr) {
        return nullptr;
    }

    const int32_t hidden_dim = (int32_t) x_ct->ne[0];
    if (hidden_dim != head_dim * n_heads) {
        return nullptr;
    }

    ggml_tensor * h = codec_op_rms_norm_ct(ctx, x_ct, 1e-6f, att_norm_w);
    if (h == nullptr) return nullptr;

    ggml_tensor * qkv = codec_mul_mat(ctx, c_attn_w, h);                                  // [3*hidden, t]
    if (qkv == nullptr) return nullptr;

    const int64_t t = qkv->ne[1];
    ggml_tensor * q = ggml_cont(ctx, ggml_view_2d(ctx, qkv, hidden_dim, t, qkv->nb[1], 0));
    ggml_tensor * k = ggml_cont(ctx, ggml_view_2d(ctx, qkv, hidden_dim, t, qkv->nb[1], (size_t) hidden_dim * qkv->nb[0]));
    ggml_tensor * v = ggml_cont(ctx, ggml_view_2d(ctx, qkv, hidden_dim, t, qkv->nb[1], (size_t) hidden_dim * qkv->nb[0] * 2));

    ggml_tensor * q_dht = ggml_reshape_3d(ctx, q, head_dim, n_heads, t);                  // [d, h, t]
    ggml_tensor * k_dht = ggml_reshape_3d(ctx, k, head_dim, n_heads, t);
    ggml_tensor * v_dth = ggml_permute(ctx, ggml_reshape_3d(ctx, v, head_dim, n_heads, t),
                                       0, 2, 1, 3);                                       // [d, t, h]

    ggml_tensor * q_rope_dht = codec_op_rope(ctx, q_dht, head_dim, rope_theta, 1.0f, CODEC_ROPE_MODE_NORMAL);
    ggml_tensor * k_rope_dht = codec_op_rope(ctx, k_dht, head_dim, rope_theta, 1.0f, CODEC_ROPE_MODE_NORMAL);
    if (q_rope_dht == nullptr || k_rope_dht == nullptr) return nullptr;
    ggml_tensor * q_rope = ggml_cont(ctx, ggml_permute(ctx, q_rope_dht, 0, 2, 1, 3));     // [d, t, h]
    ggml_tensor * k_rope = ggml_cont(ctx, ggml_permute(ctx, k_rope_dht, 0, 2, 1, 3));     // [d, t, h]

    codec_lm_attn_params attn_p = {};
    attn_p.scale = 1.0f / std::sqrt((float) head_dim);
    attn_p.causal = false;
    ggml_tensor * attn_ctx = codec_op_lm_attn_ctx_dth(ctx, q_rope, k_rope, v_dth, &attn_p); // [d, t, h]
    if (attn_ctx == nullptr) return nullptr;

    ggml_tensor * attn_ct = ggml_reshape_2d(
        ctx,
        ggml_cont(ctx, ggml_permute(ctx, attn_ctx, 0, 2, 1, 3)),
        hidden_dim,
        t);
    ggml_tensor * attn_proj = codec_mul_mat(ctx, c_proj_w, attn_ct);                       // [hidden, t]
    if (attn_proj == nullptr) return nullptr;
    x_ct = ggml_add(ctx, x_ct, attn_proj);

    ggml_tensor * m = codec_op_rms_norm_ct(ctx, x_ct, 1e-6f, ffn_norm_w);
    if (m == nullptr) return nullptr;
    ggml_tensor * ff = codec_mul_mat(ctx, fc1_w, m);
    ff = ggml_silu(ctx, ff);
    ff = codec_mul_mat(ctx, fc2_w, ff);
    if (ff == nullptr) return nullptr;
    return ggml_add(ctx, x_ct, ff);
}

ggml_tensor * codec_op_espnet_rel_pos_emb(
    ggml_context * ctx,
    int32_t t,
    int32_t d_model) {
    if (ctx == nullptr || t <= 0 || d_model <= 0 || (d_model & 1) != 0) return nullptr;
    const int32_t half = d_model / 2;
    const int32_t pe_len = 2 * t - 1;

    // Position values per row r ∈ [0, 2T-2] are p_r = (T-1) - r, giving the
    // sequence [T-1, T-2, ..., 0, -1, ..., -(T-1)].
    ggml_tensor * row_idx = codec_op_arange(ctx, 0.0f, (float) pe_len, 1.0f);    // [pe_len]
    ggml_tensor * pos = ggml_scale_bias(ctx, row_idx, -1.0f, (float) (t - 1));

    // freq[k] = exp(-2k * log(10000) / d_model) for k ∈ [0, half).
    ggml_tensor * k_idx = codec_op_arange(ctx, 0.0f, (float) half, 1.0f);
    ggml_tensor * log_k = ggml_scale(ctx, k_idx, -2.0f * std::log(10000.0f) / (float) d_model);
    ggml_tensor * freqs = ggml_exp(ctx, log_k);                                // [half]

    // Outer product: angle[k, r] = freqs[k] * pos[r]. Broadcast both to [half, pe_len].
    ggml_tensor * pos_2d = ggml_reshape_2d(ctx, pos, 1, pe_len);
    ggml_tensor * freqs_2d = ggml_reshape_2d(ctx, freqs, half, 1);
    ggml_tensor * tmpl = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, half, pe_len);
    ggml_tensor * angle = ggml_mul(ctx,
        ggml_repeat(ctx, freqs_2d, tmpl),
        ggml_repeat(ctx, pos_2d, tmpl));

    // Interleave sin/cos along ne[0]: stack as [half, 2, pe_len], permute axes
    // (ne[0],ne[1]) → (ne[1],ne[0]) so the interleaved layout falls out as the
    // contiguous flatten over (k, j∈{sin,cos}, r).
    ggml_tensor * sin_3d = ggml_reshape_3d(ctx, codec_op_sin(ctx, angle), half, 1, pe_len);
    ggml_tensor * cos_3d = ggml_reshape_3d(ctx, codec_op_cos(ctx, angle), half, 1, pe_len);
    ggml_tensor * stacked = ggml_concat(ctx, sin_3d, cos_3d, /*dim=*/1);       // [half, 2, pe_len]
    ggml_tensor * permuted = ggml_cont(ctx, ggml_permute(ctx, stacked, 1, 0, 2, 3));
    return ggml_reshape_2d(ctx, permuted, d_model, pe_len);
}
