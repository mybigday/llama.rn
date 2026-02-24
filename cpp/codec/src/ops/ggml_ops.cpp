#include "lm_ggml_ops.h"

#include <algorithm>
#include <cfloat>

lm_ggml_tensor * codec_op_unary(lm_ggml_context * ctx, lm_ggml_tensor * x, codec_unary_op op) {
    if (ctx == nullptr || x == nullptr) {
        return nullptr;
    }

    switch (op) {
        case CODEC_UNARY_SIGMOID: {
            lm_ggml_tensor * x_half = lm_ggml_scale(ctx, x, 0.5f);
            lm_ggml_tensor * th = lm_ggml_tanh(ctx, x_half);
            return lm_ggml_scale_bias(ctx, th, 0.5f, 0.5f);
        }
        case CODEC_UNARY_ELU:
            return lm_ggml_elu(ctx, x);
        case CODEC_UNARY_SILU:
            return lm_ggml_silu(ctx, x);
        case CODEC_UNARY_GELU_ERF:
            return lm_ggml_gelu_erf(ctx, x);
        default:
            return nullptr;
    }
}

lm_ggml_tensor * codec_op_layer_norm(lm_ggml_context * ctx, lm_ggml_tensor * x, float eps, lm_ggml_tensor * gamma, lm_ggml_tensor * beta) {
    if (ctx == nullptr || x == nullptr) {
        return nullptr;
    }

    lm_ggml_tensor * y = lm_ggml_norm(ctx, x, eps);
    if (gamma != nullptr) {
        lm_ggml_tensor * g2 = lm_ggml_reshape_2d(ctx, gamma, 1, x->ne[1]);
        y = lm_ggml_mul(ctx, y, lm_ggml_repeat(ctx, g2, y));
    }
    if (beta != nullptr) {
        lm_ggml_tensor * b2 = lm_ggml_reshape_2d(ctx, beta, 1, x->ne[1]);
        y = lm_ggml_add(ctx, y, lm_ggml_repeat(ctx, b2, y));
    }
    return y;
}

lm_ggml_tensor * codec_op_group_norm(lm_ggml_context * ctx, lm_ggml_tensor * x, int32_t n_groups, float eps, lm_ggml_tensor * gamma, lm_ggml_tensor * beta) {
    if (ctx == nullptr || x == nullptr || n_groups <= 0 || x->ne[1] % n_groups != 0) {
        return nullptr;
    }

    const int64_t cpg = x->ne[1] / n_groups;
    lm_ggml_tensor * x3 = lm_ggml_reshape_3d(ctx, x, x->ne[0], cpg, n_groups);
    lm_ggml_tensor * y = lm_ggml_reshape_2d(ctx, lm_ggml_group_norm(ctx, x3, n_groups, eps), x->ne[0], x->ne[1]);

    if (gamma != nullptr) {
        lm_ggml_tensor * g2 = lm_ggml_reshape_2d(ctx, gamma, 1, x->ne[1]);
        y = lm_ggml_mul(ctx, y, lm_ggml_repeat(ctx, g2, y));
    }
    if (beta != nullptr) {
        lm_ggml_tensor * b2 = lm_ggml_reshape_2d(ctx, beta, 1, x->ne[1]);
        y = lm_ggml_add(ctx, y, lm_ggml_repeat(ctx, b2, y));
    }
    return y;
}

lm_ggml_tensor * codec_op_linear(lm_ggml_context * ctx, lm_ggml_tensor * x, lm_ggml_tensor * w, lm_ggml_tensor * b) {
    if (ctx == nullptr || x == nullptr || w == nullptr) {
        return nullptr;
    }

    lm_ggml_tensor * y = lm_ggml_mul_mat(ctx, w, x);
    if (b != nullptr) {
        lm_ggml_tensor * b2 = lm_ggml_reshape_2d(ctx, b, w->ne[1], 1);
        y = lm_ggml_add(ctx, y, lm_ggml_repeat(ctx, b2, y));
    }
    return y;
}

lm_ggml_tensor * codec_op_snake(lm_ggml_context * ctx, lm_ggml_tensor * x, lm_ggml_tensor * alpha, float eps) {
    if (ctx == nullptr || x == nullptr || alpha == nullptr) {
        return nullptr;
    }

    lm_ggml_tensor * alpha_2d = lm_ggml_reshape_2d(ctx, alpha, 1, x->ne[1]);
    lm_ggml_tensor * alpha_rep = lm_ggml_repeat(ctx, alpha_2d, x);
    lm_ggml_tensor * alpha_clamped = lm_ggml_clamp(ctx, alpha_rep, eps, FLT_MAX);
    lm_ggml_tensor * ax = lm_ggml_mul(ctx, alpha_clamped, x);
    lm_ggml_tensor * s = lm_ggml_sin(ctx, ax);
    lm_ggml_tensor * s2 = lm_ggml_mul(ctx, s, s);
    lm_ggml_tensor * frac = lm_ggml_div(ctx, s2, alpha_clamped);
    return lm_ggml_add(ctx, x, frac);
}

lm_ggml_tensor * codec_op_snake_beta(lm_ggml_context * ctx, lm_ggml_tensor * x, lm_ggml_tensor * alpha, lm_ggml_tensor * inv_beta, float eps) {
    if (ctx == nullptr || x == nullptr || alpha == nullptr || inv_beta == nullptr) {
        return nullptr;
    }

    lm_ggml_tensor * alpha_2d = lm_ggml_reshape_2d(ctx, alpha, 1, x->ne[1]);
    lm_ggml_tensor * alpha_rep = lm_ggml_repeat(ctx, alpha_2d, x);
    lm_ggml_tensor * alpha_clamped = lm_ggml_clamp(ctx, alpha_rep, eps, FLT_MAX);
    lm_ggml_tensor * ax = lm_ggml_mul(ctx, alpha_clamped, x);
    lm_ggml_tensor * s = lm_ggml_sin(ctx, ax);
    lm_ggml_tensor * s2 = lm_ggml_mul(ctx, s, s);
    lm_ggml_tensor * invb_2d = lm_ggml_reshape_2d(ctx, inv_beta, 1, x->ne[1]);
    lm_ggml_tensor * invb_rep = lm_ggml_repeat(ctx, invb_2d, x);
    lm_ggml_tensor * frac = lm_ggml_mul(ctx, s2, invb_rep);
    return lm_ggml_add(ctx, x, frac);
}

lm_ggml_tensor * codec_op_pad_1d(lm_ggml_context * ctx, lm_ggml_tensor * x, int32_t pad_left, int32_t pad_right) {
    if (ctx == nullptr || x == nullptr || pad_left < 0 || pad_right < 0) {
        return nullptr;
    }
    return lm_ggml_pad_ext(ctx, x, pad_left, pad_right, 0, 0, 0, 0, 0, 0);
}

lm_ggml_tensor * codec_op_pad_1d_replicate(lm_ggml_context * ctx, lm_ggml_tensor * x, int32_t pad_left, int32_t pad_right) {
    if (ctx == nullptr || x == nullptr || pad_left < 0 || pad_right < 0) {
        return nullptr;
    }
    if (pad_left == 0 && pad_right == 0) {
        return x;
    }

    lm_ggml_tensor * out = x;
    if (pad_left > 0) {
        lm_ggml_tensor * left = lm_ggml_view_2d(ctx, x, 1, x->ne[1], x->nb[1], 0);
        lm_ggml_tensor * left_target = lm_ggml_new_tensor_2d(ctx, x->type, pad_left, x->ne[1]);
        lm_ggml_tensor * left_rep = lm_ggml_repeat(ctx, left, left_target);
        out = lm_ggml_concat(ctx, left_rep, out, 0);
    }

    if (pad_right > 0) {
        const size_t offset = (size_t) (x->ne[0] - 1) * (size_t) x->nb[0];
        lm_ggml_tensor * right = lm_ggml_view_2d(ctx, x, 1, x->ne[1], x->nb[1], offset);
        lm_ggml_tensor * right_target = lm_ggml_new_tensor_2d(ctx, x->type, pad_right, x->ne[1]);
        lm_ggml_tensor * right_rep = lm_ggml_repeat(ctx, right, right_target);
        out = lm_ggml_concat(ctx, out, right_rep, 0);
    }

    return lm_ggml_cont(ctx, out);
}

lm_ggml_tensor * codec_op_crop_1d(lm_ggml_context * ctx, lm_ggml_tensor * x, int32_t crop_left, int32_t crop_right) {
    if (ctx == nullptr || x == nullptr || crop_left < 0 || crop_right < 0) {
        return nullptr;
    }
    const int64_t out_t = x->ne[0] - crop_left - crop_right;
    if (out_t <= 0) {
        return nullptr;
    }
    lm_ggml_tensor * view = lm_ggml_view_2d(ctx, x, out_t, x->ne[1], x->nb[1], (size_t) crop_left * sizeof(float));
    return lm_ggml_cont(ctx, view);
}

lm_ggml_tensor * codec_op_causal_crop_1d(lm_ggml_context * ctx, lm_ggml_tensor * x, int32_t target_t) {
    if (ctx == nullptr || x == nullptr || target_t <= 0 || x->ne[0] < target_t) {
        return nullptr;
    }
    return codec_op_crop_1d(ctx, x, 0, (int32_t)x->ne[0] - target_t);
}

lm_ggml_tensor * codec_op_channel_scale(lm_ggml_context * ctx, lm_ggml_tensor * x, lm_ggml_tensor * scale) {
    if (ctx == nullptr || x == nullptr || scale == nullptr) {
        return nullptr;
    }
    lm_ggml_tensor * s2 = lm_ggml_reshape_2d(ctx, scale, x->ne[0], 1);
    return lm_ggml_mul(ctx, x, lm_ggml_repeat(ctx, s2, x));
}

lm_ggml_tensor * codec_op_tokens_to_features(lm_ggml_context * ctx, lm_ggml_tensor * tokens, int32_t out_channels) {
    if (ctx == nullptr || tokens == nullptr || out_channels <= 0) {
        return nullptr;
    }

    lm_ggml_tensor * x = lm_ggml_scale(ctx, tokens, 1.0f / 1024.0f);
    lm_ggml_tensor * base = nullptr;
    if (x->ne[1] <= 1) {
        base = lm_ggml_cont(ctx, lm_ggml_view_2d(ctx, x, x->ne[0], 1, x->nb[1], 0));
    } else {
        // Aggregate all quantizers instead of only using q=0.
        lm_ggml_tensor * x_qt = lm_ggml_cont(ctx, lm_ggml_transpose(ctx, x)); // [q, t]
        lm_ggml_tensor * mean_1t = lm_ggml_scale(ctx, lm_ggml_sum_rows(ctx, x_qt), 1.0f / (float) x_qt->ne[0]); // [1, t]
        base = lm_ggml_cont(ctx, lm_ggml_transpose(ctx, mean_1t));         // [t, 1]
    }
    if (out_channels == 1) {
        return base;
    }
    return lm_ggml_repeat(ctx, base, lm_ggml_new_tensor_2d(ctx, LM_GGML_TYPE_F32, x->ne[0], out_channels));
}
