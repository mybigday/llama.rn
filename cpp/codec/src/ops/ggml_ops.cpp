#include "lm_ggml_ops.h"

#include "conv1d.h"
#include "lm_attn.h"
#include "rope.h"
#include "../runtime/tensor_utils.h"

#include <algorithm>
#include <cfloat>
#include <cmath>

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
        case CODEC_UNARY_MISH: {
            // mish(x) = x * tanh(softplus(x))
            lm_ggml_tensor * sp = lm_ggml_softplus(ctx, x);
            lm_ggml_tensor * t = lm_ggml_tanh(ctx, sp);
            return lm_ggml_mul(ctx, x, t);
        }
        default:
            return nullptr;
    }
}

lm_ggml_tensor * codec_op_layer_norm(lm_ggml_context * ctx, lm_ggml_tensor * x, float eps, lm_ggml_tensor * gamma, lm_ggml_tensor * beta) {
    if (ctx == nullptr || x == nullptr) {
        return nullptr;
    }

    gamma = codec_graph_cast_f32(ctx, gamma);
    beta = codec_graph_cast_f32(ctx, beta);

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

lm_ggml_tensor * codec_op_layer_norm_ct(lm_ggml_context * ctx, lm_ggml_tensor * x_ct, float eps, lm_ggml_tensor * gamma, lm_ggml_tensor * beta) {
    if (ctx == nullptr || x_ct == nullptr || gamma == nullptr || beta == nullptr) {
        return nullptr;
    }

    gamma = codec_graph_cast_f32(ctx, gamma);
    beta = codec_graph_cast_f32(ctx, beta);

    lm_ggml_tensor * y = lm_ggml_norm(ctx, x_ct, eps);
    lm_ggml_tensor * g2 = lm_ggml_reshape_2d(ctx, gamma, x_ct->ne[0], 1);
    lm_ggml_tensor * b2 = lm_ggml_reshape_2d(ctx, beta, x_ct->ne[0], 1);
    y = lm_ggml_mul(ctx, y, lm_ggml_repeat(ctx, g2, y));
    y = lm_ggml_add(ctx, y, lm_ggml_repeat(ctx, b2, y));
    return y;
}

lm_ggml_tensor * codec_op_layer_norm_tc(lm_ggml_context * ctx, lm_ggml_tensor * x_tc, float eps, lm_ggml_tensor * gamma, lm_ggml_tensor * beta) {
    if (ctx == nullptr || x_tc == nullptr || gamma == nullptr || beta == nullptr) {
        return nullptr;
    }

    lm_ggml_tensor * x_ct = lm_ggml_cont(ctx, lm_ggml_transpose(ctx, x_tc));
    lm_ggml_tensor * y_ct = codec_op_layer_norm_ct(ctx, x_ct, eps, gamma, beta);
    if (y_ct == nullptr) {
        return nullptr;
    }
    return lm_ggml_cont(ctx, lm_ggml_transpose(ctx, y_ct));
}

lm_ggml_tensor * codec_op_rms_norm_ct(lm_ggml_context * ctx, lm_ggml_tensor * x_ct, float eps, lm_ggml_tensor * gamma) {
    if (ctx == nullptr || x_ct == nullptr || gamma == nullptr) {
        return nullptr;
    }

    lm_ggml_tensor * y = lm_ggml_rms_norm(ctx, x_ct, eps);
    return codec_op_channel_scale(ctx, y, gamma);
}

lm_ggml_tensor * codec_op_group_norm(lm_ggml_context * ctx, lm_ggml_tensor * x, int32_t n_groups, float eps, lm_ggml_tensor * gamma, lm_ggml_tensor * beta) {
    if (ctx == nullptr || x == nullptr || n_groups <= 0 || x->ne[1] % n_groups != 0) {
        return nullptr;
    }

    gamma = codec_graph_cast_f32(ctx, gamma);
    beta = codec_graph_cast_f32(ctx, beta);

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

    // Cast weight to F32 to preserve numerical parity with the legacy
    // dequant-into-graph-buffer path. lm_ggml_mul_mat itself supports quantized
    // weights natively, but mixing Q8_0 weight + F32 activations introduces
    // additional per-block quantization on the activation side.
    lm_ggml_tensor * w_f32 = codec_graph_cast_f32(ctx, w);
    lm_ggml_tensor * b_f32 = codec_graph_cast_f32(ctx, b);

    lm_ggml_tensor * y = lm_ggml_mul_mat(ctx, w_f32, x);
    if (b_f32 != nullptr) {
        lm_ggml_tensor * b2 = lm_ggml_reshape_2d(ctx, b_f32, w_f32->ne[1], 1);
        y = lm_ggml_add(ctx, y, lm_ggml_repeat(ctx, b2, y));
    }
    return y;
}

lm_ggml_tensor * codec_op_linear_tc(lm_ggml_context * ctx, lm_ggml_tensor * x_tc, lm_ggml_tensor * w, lm_ggml_tensor * b) {
    if (ctx == nullptr || x_tc == nullptr || w == nullptr) {
        return nullptr;
    }

    lm_ggml_tensor * x_ct = lm_ggml_cont(ctx, lm_ggml_transpose(ctx, x_tc));
    lm_ggml_tensor * y_ct = codec_op_linear(ctx, x_ct, w, b);
    if (y_ct == nullptr) {
        return nullptr;
    }
    return lm_ggml_cont(ctx, lm_ggml_transpose(ctx, y_ct));
}

lm_ggml_tensor * codec_op_snake(lm_ggml_context * ctx, lm_ggml_tensor * x, lm_ggml_tensor * alpha, float eps) {
    if (ctx == nullptr || x == nullptr || alpha == nullptr) {
        return nullptr;
    }

    alpha = codec_graph_cast_f32(ctx, alpha);

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

    alpha = codec_graph_cast_f32(ctx, alpha);
    inv_beta = codec_graph_cast_f32(ctx, inv_beta);

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
    scale = codec_graph_cast_f32(ctx, scale);
    // lm_ggml_mul broadcasts src1 over ne>=1 dims natively (ne[1]=1 repeats over
    // x's ne[1]); no explicit lm_ggml_repeat needed.  scale is (ne[0],) → view as
    // (ne[0], 1) so the leading dims match x for the broadcast.
    lm_ggml_tensor * s2 = lm_ggml_reshape_2d(ctx, scale, x->ne[0], 1);
    return lm_ggml_mul(ctx, x, s2);
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

lm_ggml_tensor * codec_op_convnext_block_ct(
    lm_ggml_context * ctx,
    lm_ggml_tensor * x_ct,
    lm_ggml_tensor * dw_w,
    lm_ggml_tensor * dw_b,
    lm_ggml_tensor * ln_w,
    lm_ggml_tensor * ln_b,
    lm_ggml_tensor * pw1_w,
    lm_ggml_tensor * pw1_b,
    lm_ggml_tensor * pw2_w,
    lm_ggml_tensor * pw2_b,
    lm_ggml_tensor * gamma,
    int32_t dw_padding) {

    if (ctx == nullptr || x_ct == nullptr || dw_w == nullptr || ln_w == nullptr || pw1_w == nullptr || pw2_w == nullptr) {
        return nullptr;
    }

    lm_ggml_tensor * res_tc = lm_ggml_cont(ctx, lm_ggml_transpose(ctx, x_ct)); // [t, c]
    lm_ggml_tensor * x_dw = codec_conv1d_depthwise(ctx, res_tc, dw_w, dw_b, 1, 1, dw_padding);
    if (x_dw == nullptr) {
        return nullptr;
    }

    lm_ggml_tensor * h_ct = codec_op_layer_norm_ct(
        ctx, lm_ggml_cont(ctx, lm_ggml_transpose(ctx, x_dw)), 1e-6f, ln_w, ln_b);
    if (h_ct == nullptr) {
        return nullptr;
    }

    lm_ggml_tensor * pw1 = codec_op_linear(ctx, h_ct, pw1_w, pw1_b);
    if (pw1 == nullptr) {
        return nullptr;
    }
    pw1 = lm_ggml_gelu_erf(ctx, pw1);

    lm_ggml_tensor * pw2 = codec_op_linear(ctx, pw1, pw2_w, pw2_b);
    if (pw2 == nullptr) {
        return nullptr;
    }
    if (gamma != nullptr) {
        pw2 = codec_op_channel_scale(ctx, pw2, gamma);
        if (pw2 == nullptr) {
            return nullptr;
        }
    }
    return lm_ggml_add(ctx, x_ct, pw2);
}

lm_ggml_tensor * codec_op_causal_block1d_tc(
    lm_ggml_context * ctx,
    lm_ggml_tensor * x_tc,
    lm_ggml_tensor * conv_w,
    lm_ggml_tensor * conv_b,
    lm_ggml_tensor * ln_w,
    lm_ggml_tensor * ln_b) {
    if (ctx == nullptr || x_tc == nullptr || conv_w == nullptr) return nullptr;
    lm_ggml_tensor * y = codec_conv1d_causal(ctx, x_tc, conv_w, conv_b, /*stride=*/1, /*dilation=*/1);
    if (y == nullptr) return nullptr;
    y = codec_op_layer_norm_tc(ctx, y, /*eps=*/1e-5f, ln_w, ln_b);
    if (y == nullptr) return nullptr;
    return codec_op_unary(ctx, y, CODEC_UNARY_MISH);
}

lm_ggml_tensor * codec_op_hifigan_resblock_branch_ct(
    lm_ggml_context * ctx,
    lm_ggml_tensor * x_tc,
    lm_ggml_tensor * a1,
    lm_ggml_tensor * a2,
    lm_ggml_tensor * c1_w,
    lm_ggml_tensor * c1_b,
    lm_ggml_tensor * c2_w,
    lm_ggml_tensor * c2_b,
    int32_t kernel_size,
    int32_t dilation) {
    if (ctx == nullptr || x_tc == nullptr || a1 == nullptr || a2 == nullptr ||
        c1_w == nullptr || c2_w == nullptr || kernel_size <= 0 || dilation <= 0) {
        return nullptr;
    }
    const int32_t pad1 = (kernel_size * dilation - dilation) / 2;
    const int32_t pad2 = (kernel_size - 1) / 2;
    lm_ggml_tensor * h = codec_op_snake(ctx, x_tc, a1, /*eps=*/1e-9f);
    if (h == nullptr) return nullptr;
    h = codec_conv1d(ctx, h, c1_w, c1_b, /*stride=*/1, /*dilation=*/dilation, /*padding=*/pad1);
    if (h == nullptr) return nullptr;
    h = codec_op_snake(ctx, h, a2, /*eps=*/1e-9f);
    if (h == nullptr) return nullptr;
    h = codec_conv1d(ctx, h, c2_w, c2_b, /*stride=*/1, /*dilation=*/1, /*padding=*/pad2);
    if (h == nullptr) return nullptr;
    return lm_ggml_add(ctx, h, x_tc);
}

lm_ggml_tensor * codec_op_cfm_causal_resnet_block_tc(
    lm_ggml_context * ctx,
    lm_ggml_tensor * x_tc,
    lm_ggml_tensor * t_emb,
    lm_ggml_tensor * b1_conv_w, lm_ggml_tensor * b1_conv_b,
    lm_ggml_tensor * b1_ln_w,   lm_ggml_tensor * b1_ln_b,
    lm_ggml_tensor * b2_conv_w, lm_ggml_tensor * b2_conv_b,
    lm_ggml_tensor * b2_ln_w,   lm_ggml_tensor * b2_ln_b,
    lm_ggml_tensor * mlp_w,     lm_ggml_tensor * mlp_b,
    lm_ggml_tensor * res_w,     lm_ggml_tensor * res_b) {
    if (ctx == nullptr || x_tc == nullptr || t_emb == nullptr ||
        b1_conv_w == nullptr || b2_conv_w == nullptr ||
        mlp_w == nullptr || res_w == nullptr) return nullptr;

    lm_ggml_tensor * h = codec_op_causal_block1d_tc(ctx, x_tc, b1_conv_w, b1_conv_b, b1_ln_w, b1_ln_b);
    if (h == nullptr) return nullptr;

    // Time embedding mlp: mish(t_emb) → Linear(time_embed_dim → out_dim) → broadcast over t.
    lm_ggml_tensor * tm = codec_op_unary(ctx, t_emb, CODEC_UNARY_MISH);
    if (tm == nullptr) return nullptr;
    lm_ggml_tensor * tm_2d = lm_ggml_reshape_2d(ctx, tm, tm->ne[0], 1);
    lm_ggml_tensor * tm_proj = lm_ggml_mul_mat(ctx, mlp_w, tm_2d);  // [out, 1]
    if (tm_proj == nullptr) return nullptr;
    if (mlp_b != nullptr) {
        lm_ggml_tensor * mlp_b_2d = lm_ggml_reshape_2d(ctx, mlp_b, mlp_b->ne[0], 1);
        tm_proj = lm_ggml_add(ctx, tm_proj, mlp_b_2d);
    }
    lm_ggml_tensor * tm_to = lm_ggml_cont(ctx, lm_ggml_transpose(ctx, tm_proj));  // [1, out]
    h = lm_ggml_add(ctx, h, lm_ggml_repeat(ctx, tm_to, h));

    h = codec_op_causal_block1d_tc(ctx, h, b2_conv_w, b2_conv_b, b2_ln_w, b2_ln_b);
    if (h == nullptr) return nullptr;

    lm_ggml_tensor * res = codec_conv1d(ctx, x_tc, res_w, res_b, /*stride=*/1, /*dilation=*/1, /*padding=*/0);
    if (res == nullptr) return nullptr;
    return lm_ggml_add(ctx, h, res);
}

lm_ggml_tensor * codec_op_basic_transformer_block_tc(
    lm_ggml_context * ctx,
    lm_ggml_tensor * x_tc,
    lm_ggml_tensor * norm1_w, lm_ggml_tensor * norm1_b,
    lm_ggml_tensor * qw, lm_ggml_tensor * kw, lm_ggml_tensor * vw,
    lm_ggml_tensor * ow, lm_ggml_tensor * ob,
    lm_ggml_tensor * norm3_w, lm_ggml_tensor * norm3_b,
    lm_ggml_tensor * ff1_w, lm_ggml_tensor * ff1_b,
    lm_ggml_tensor * ff2_w, lm_ggml_tensor * ff2_b,
    int32_t head_dim,
    int32_t num_heads) {
    if (ctx == nullptr || x_tc == nullptr || qw == nullptr || kw == nullptr || vw == nullptr ||
        ow == nullptr || ff1_w == nullptr || ff2_w == nullptr || head_dim <= 0 || num_heads <= 0) {
        return nullptr;
    }
    const int32_t inner = head_dim * num_heads;
    const int32_t T = (int32_t) x_tc->ne[0];

    lm_ggml_tensor * h_tc = codec_op_layer_norm_tc(ctx, x_tc, 1e-5f, norm1_w, norm1_b);
    if (h_tc == nullptr) return nullptr;
    lm_ggml_tensor * h_ct = lm_ggml_cont(ctx, lm_ggml_transpose(ctx, h_tc));
    lm_ggml_tensor * q_ct = codec_op_linear(ctx, h_ct, qw, /*b=*/nullptr);
    lm_ggml_tensor * k_ct = codec_op_linear(ctx, h_ct, kw, /*b=*/nullptr);
    lm_ggml_tensor * v_ct = codec_op_linear(ctx, h_ct, vw, /*b=*/nullptr);
    if (q_ct == nullptr || k_ct == nullptr || v_ct == nullptr) return nullptr;

    auto to_dth = [&](lm_ggml_tensor * x_ct_in) {
        lm_ggml_tensor * r = lm_ggml_reshape_3d(ctx, x_ct_in, head_dim, num_heads, T);
        return lm_ggml_cont(ctx, lm_ggml_permute(ctx, r, 0, 2, 1, 3));
    };
    codec_lm_attn_params attn_p = {};
    attn_p.scale = 1.0f / std::sqrt((float) head_dim);
    attn_p.causal = false;
    lm_ggml_tensor * attn_dth = codec_op_lm_attn_ctx_dth(ctx, to_dth(q_ct), to_dth(k_ct), to_dth(v_ct), &attn_p);
    if (attn_dth == nullptr) return nullptr;

    lm_ggml_tensor * attn_ct = lm_ggml_reshape_2d(
        ctx,
        lm_ggml_cont(ctx, lm_ggml_permute(ctx, attn_dth, 0, 2, 1, 3)),
        inner, T);
    lm_ggml_tensor * proj_ct = codec_op_linear(ctx, attn_ct, ow, ob);
    if (proj_ct == nullptr) return nullptr;
    x_tc = lm_ggml_add(ctx, x_tc, lm_ggml_cont(ctx, lm_ggml_transpose(ctx, proj_ct)));

    lm_ggml_tensor * f_tc = codec_op_layer_norm_tc(ctx, x_tc, 1e-5f, norm3_w, norm3_b);
    if (f_tc == nullptr) return nullptr;
    lm_ggml_tensor * f_ct = lm_ggml_cont(ctx, lm_ggml_transpose(ctx, f_tc));
    lm_ggml_tensor * ff = codec_op_linear(ctx, f_ct, ff1_w, ff1_b);
    if (ff == nullptr) return nullptr;
    ff = codec_op_unary(ctx, ff, CODEC_UNARY_GELU_ERF);
    ff = codec_op_linear(ctx, ff, ff2_w, ff2_b);
    if (ff == nullptr) return nullptr;
    return lm_ggml_add(ctx, x_tc, lm_ggml_cont(ctx, lm_ggml_transpose(ctx, ff)));
}

lm_ggml_tensor * codec_op_whisper_encoder_layer_tc(
    lm_ggml_context * ctx,
    lm_ggml_tensor * x_tc,
    lm_ggml_tensor * n1w, lm_ggml_tensor * n1b,
    lm_ggml_tensor * qw,  lm_ggml_tensor * qb,
    lm_ggml_tensor * kw,
    lm_ggml_tensor * vw,  lm_ggml_tensor * vb,
    lm_ggml_tensor * ow,  lm_ggml_tensor * ob,
    lm_ggml_tensor * n2w, lm_ggml_tensor * n2b,
    lm_ggml_tensor * fc1w, lm_ggml_tensor * fc1b,
    lm_ggml_tensor * fc2w, lm_ggml_tensor * fc2b,
    int32_t head_dim,
    int32_t n_heads,
    int32_t n_valid) {

    if (ctx == nullptr || x_tc == nullptr || qw == nullptr || kw == nullptr || vw == nullptr ||
        ow == nullptr || fc1w == nullptr || fc2w == nullptr || head_dim <= 0 || n_heads <= 0) {
        return nullptr;
    }
    const int64_t t      = x_tc->ne[0];
    const int32_t hidden = head_dim * n_heads;

    lm_ggml_tensor * res = x_tc;
    lm_ggml_tensor * h = codec_op_layer_norm_tc(ctx, x_tc, 1e-5f, n1w, n1b);
    if (h == nullptr) return nullptr;
    lm_ggml_tensor * h_ct = lm_ggml_cont(ctx, lm_ggml_transpose(ctx, h));     // [c, t]

    auto linear_with_bias = [&](lm_ggml_tensor * w, lm_ggml_tensor * b) -> lm_ggml_tensor * {
        lm_ggml_tensor * y = lm_ggml_mul_mat(ctx, w, h_ct);                // [out, t]
        if (y == nullptr) return nullptr;
        if (b != nullptr) {
            lm_ggml_tensor * b2 = lm_ggml_reshape_2d(ctx, b, b->ne[0], 1);
            lm_ggml_tensor * br = lm_ggml_repeat(ctx, b2, y);
            y = lm_ggml_add(ctx, y, br);
        }
        return y;
    };

    lm_ggml_tensor * q = linear_with_bias(qw, qb);                       // [hidden, t]
    lm_ggml_tensor * k = linear_with_bias(kw, nullptr);                  // k: bias-free
    lm_ggml_tensor * v = linear_with_bias(vw, vb);
    if (q == nullptr || k == nullptr || v == nullptr) return nullptr;

    auto to_dth = [&](lm_ggml_tensor * x_ct_in) -> lm_ggml_tensor * {
        lm_ggml_tensor * r = lm_ggml_reshape_3d(ctx, x_ct_in, head_dim, n_heads, t);
        return lm_ggml_cont(ctx, lm_ggml_permute(ctx, r, 0, 2, 1, 3));      // [d, t, h]
    };

    codec_lm_attn_params attn_p = {};
    attn_p.scale   = 1.0f / std::sqrt((float) head_dim);
    attn_p.causal  = false;
    attn_p.window  = 0;
    attn_p.n_valid = n_valid;
    lm_ggml_tensor * attn_dth = codec_op_lm_attn_ctx_dth(ctx, to_dth(q), to_dth(k), to_dth(v), &attn_p);
    if (attn_dth == nullptr) return nullptr;
    lm_ggml_tensor * attn_dht = lm_ggml_cont(ctx, lm_ggml_permute(ctx, attn_dth, 0, 2, 1, 3));
    lm_ggml_tensor * attn_ct  = lm_ggml_reshape_2d(ctx, attn_dht, hidden, t);

    lm_ggml_tensor * out_ct = lm_ggml_mul_mat(ctx, ow, attn_ct);             // [hidden, t]
    if (ob != nullptr) {
        lm_ggml_tensor * ob_2d = lm_ggml_reshape_2d(ctx, ob, ob->ne[0], 1);
        lm_ggml_tensor * obr = lm_ggml_repeat(ctx, ob_2d, out_ct);
        out_ct = lm_ggml_add(ctx, out_ct, obr);
    }
    lm_ggml_tensor * out_tc = lm_ggml_cont(ctx, lm_ggml_transpose(ctx, out_ct));
    x_tc = lm_ggml_add(ctx, res, out_tc);

    res = x_tc;
    h = codec_op_layer_norm_tc(ctx, x_tc, 1e-5f, n2w, n2b);
    h_ct = lm_ggml_cont(ctx, lm_ggml_transpose(ctx, h));
    lm_ggml_tensor * f1 = linear_with_bias(fc1w, fc1b);                   // [ffn, t]
    lm_ggml_tensor * f1_tc = lm_ggml_cont(ctx, lm_ggml_transpose(ctx, f1));
    f1_tc = lm_ggml_gelu_erf(ctx, f1_tc);
    f1 = lm_ggml_cont(ctx, lm_ggml_transpose(ctx, f1_tc));                   // [ffn, t]
    lm_ggml_tensor * f2 = lm_ggml_mul_mat(ctx, fc2w, f1);                    // [hidden, t]
    if (fc2b != nullptr) {
        lm_ggml_tensor * b2 = lm_ggml_reshape_2d(ctx, fc2b, fc2b->ne[0], 1);
        lm_ggml_tensor * br = lm_ggml_repeat(ctx, b2, f2);
        f2 = lm_ggml_add(ctx, f2, br);
    }
    lm_ggml_tensor * f2_tc = lm_ggml_cont(ctx, lm_ggml_transpose(ctx, f2));
    return lm_ggml_add(ctx, res, f2_tc);
}

lm_ggml_tensor * codec_op_add_sliced_pos_emb_tc(
    lm_ggml_context * ctx,
    lm_ggml_tensor * x_tc,
    lm_ggml_tensor * pos) {
    if (ctx == nullptr || x_tc == nullptr || pos == nullptr) return x_tc;
    const int64_t t = x_tc->ne[0];
    if (t <= 0) return x_tc;
    lm_ggml_tensor * pos_view = lm_ggml_view_2d(ctx, pos, pos->ne[0], t, pos->nb[1], 0);
    lm_ggml_tensor * pos_tc   = lm_ggml_cont(ctx, lm_ggml_transpose(ctx, pos_view));
    return lm_ggml_add(ctx, x_tc, pos_tc);
}

lm_ggml_tensor * codec_op_l2_normalize_tc(
    lm_ggml_context * ctx,
    lm_ggml_tensor * x_tc,
    float eps) {
    if (ctx == nullptr || x_tc == nullptr) return nullptr;
    lm_ggml_tensor * x_ct = lm_ggml_cont(ctx, lm_ggml_transpose(ctx, x_tc));            // [c, t]
    lm_ggml_tensor * sq = lm_ggml_mul(ctx, x_ct, x_ct);
    lm_ggml_tensor * sum_sq = lm_ggml_sum_rows(ctx, sq);                             // [1, t]
    sum_sq = lm_ggml_scale_bias(ctx, sum_sq, 1.0f, eps);
    lm_ggml_tensor * norm = lm_ggml_sqrt(ctx, sum_sq);
    lm_ggml_tensor * ones = lm_ggml_scale_bias(ctx, norm, 0.0f, 1.0f);
    lm_ggml_tensor * inv = lm_ggml_div(ctx, ones, norm);
    lm_ggml_tensor * inv_rep = lm_ggml_repeat(ctx, inv, x_ct);
    lm_ggml_tensor * x_n = lm_ggml_mul(ctx, x_ct, inv_rep);
    return lm_ggml_cont(ctx, lm_ggml_transpose(ctx, x_n));                           // [t, c]
}

lm_ggml_tensor * codec_op_sinusoidal_time_emb(
    lm_ggml_context * ctx,
    float t_v,
    int32_t dim,
    float scale) {
    if (ctx == nullptr || dim <= 0 || (dim & 1) != 0) return nullptr;
    const int32_t half = dim / 2;
    lm_ggml_tensor * idx = lm_ggml_arange(ctx, 0.0f, (float) half, 1.0f);
    lm_ggml_tensor * log_idx = lm_ggml_scale(ctx, idx, -std::log(10000.0f) / (float) (half - 1));
    lm_ggml_tensor * freqs = lm_ggml_exp(ctx, log_idx);
    lm_ggml_tensor * e = lm_ggml_scale(ctx, freqs, t_v * scale);
    return lm_ggml_concat(ctx, lm_ggml_sin(ctx, e), lm_ggml_cos(ctx, e), /*dim=*/0);
}

lm_ggml_tensor * codec_op_alias_free_snake_beta_tc(
    lm_ggml_context * ctx,
    lm_ggml_tensor * x_tc,
    lm_ggml_tensor * alpha,
    lm_ggml_tensor * inv_beta,
    lm_ggml_tensor * kernel_12) {

    if (ctx == nullptr || x_tc == nullptr || alpha == nullptr || inv_beta == nullptr || kernel_12 == nullptr) {
        return nullptr;
    }
    if (kernel_12->ne[0] != 12) return nullptr;
    const int64_t c = x_tc->ne[1];

    // Replicate-pad the input by 5 each side.  This matches alias_free_torch's
    // `self.pad = kernel_size // ratio - 1 = 5` before the conv-transpose.
    lm_ggml_tensor * x_rp = codec_op_pad_1d_replicate(ctx, x_tc, 5, 5);
    if (x_rp == nullptr) return nullptr;
    const int64_t t_rp = x_rp->ne[0];

    // Zero-insert by 2: rebuild the padded sequence as
    // [x[0], 0, x[1], 0, …, x[t_rp-1], 0] using lm_ggml_pad along a new
    // innermost axis.  Reshape (t_rp, c) → (1, t_rp, c) → lm_ggml_pad(0, 1, 0…)
    // → (2, t_rp, c) with the second slot zeroed → reshape to (2*t_rp, c).
    lm_ggml_tensor * x_3d = lm_ggml_reshape_3d(ctx, x_rp, 1, t_rp, c);
    lm_ggml_tensor * x_zero_3d = lm_ggml_pad(ctx, x_3d, 1, 0, 0, 0);  // ne=(2, t_rp, c)
    if (x_zero_3d == nullptr) return nullptr;
    x_zero_3d = lm_ggml_cont(ctx, x_zero_3d);
    lm_ggml_tensor * x_zi = lm_ggml_reshape_2d(ctx, x_zero_3d, 2 * t_rp, c);

    // Pad with 11 zeros on each side, then depthwise-correlate with the
    // (symmetric) 12-tap Kaiser kernel.  Conv-transpose stride 2 of input
    // length L produces 2L+10 outputs; our zero-inserted/padded direct-conv
    // produces 2L+11, one too long, so we trim the trailing sample after.
    lm_ggml_tensor * x_zp = codec_op_pad_1d(ctx, x_zi, 11, 11);

    lm_ggml_tensor * w_3d = lm_ggml_reshape_3d(ctx, kernel_12, 12, 1, 1);
    lm_ggml_tensor * w_dst = lm_ggml_new_tensor_3d(ctx, LM_GGML_TYPE_F32, 12, 1, c);
    lm_ggml_tensor * w_dw = lm_ggml_repeat(ctx, w_3d, w_dst);

    lm_ggml_tensor * y = codec_conv1d_depthwise(ctx, x_zp, w_dw, nullptr, /*stride=*/1, /*dilation=*/1, /*padding=*/0);
    if (y == nullptr) return nullptr;
    // y length = (2*t_rp + 22) - 12 + 1 = 2*t_rp + 11.  Trim 1 from end → 2*t_rp + 10.
    y = codec_op_crop_1d(ctx, y, 0, 1);

    // Multiply by ratio=2 (matches `x = ratio * F.conv_transpose1d(...)`).
    y = lm_ggml_scale(ctx, y, 2.0f);

    // Trim Activation1d's pad_left=15 / pad_right=15 → length 2t.
    y = codec_op_crop_1d(ctx, y, 15, 15);

    // Snake-beta non-linearity at the upsampled rate.
    lm_ggml_tensor * y_act = codec_op_snake_beta(ctx, y, alpha, inv_beta, /*eps=*/1e-9f);
    if (y_act == nullptr) return nullptr;

    // Downsample 2×: replicate-pad (5, 6) then depthwise conv stride=2.
    lm_ggml_tensor * d_pad = codec_op_pad_1d_replicate(ctx, y_act, 5, 6);
    return codec_conv1d_depthwise(ctx, d_pad, w_dw, nullptr, /*stride=*/2, /*dilation=*/1, /*padding=*/0);
}

lm_ggml_tensor * codec_op_vocos_resnet_block_tc(
    lm_ggml_context * ctx,
    lm_ggml_tensor * x_tc,
    lm_ggml_tensor * n1_w, lm_ggml_tensor * n1_b,
    lm_ggml_tensor * c1_w, lm_ggml_tensor * c1_b,
    lm_ggml_tensor * n2_w, lm_ggml_tensor * n2_b,
    lm_ggml_tensor * c2_w, lm_ggml_tensor * c2_b) {

    if (ctx == nullptr || x_tc == nullptr ||
        n1_w == nullptr || n1_b == nullptr || c1_w == nullptr || c1_b == nullptr ||
        n2_w == nullptr || n2_b == nullptr || c2_w == nullptr || c2_b == nullptr) {
        return nullptr;
    }

    lm_ggml_tensor * h = codec_op_group_norm(ctx, x_tc, 32, 1e-6f, n1_w, n1_b);
    if (h == nullptr) return nullptr;
    h = lm_ggml_silu(ctx, h);
    h = codec_conv1d(ctx, h, c1_w, c1_b, 1, 1, 1);
    if (h == nullptr) return nullptr;
    h = codec_op_group_norm(ctx, h, 32, 1e-6f, n2_w, n2_b);
    if (h == nullptr) return nullptr;
    h = lm_ggml_silu(ctx, h);
    h = codec_conv1d(ctx, h, c2_w, c2_b, 1, 1, 1);
    if (h == nullptr) return nullptr;
    return lm_ggml_add(ctx, x_tc, h);
}

lm_ggml_tensor * codec_op_roformer_block_ct(
    lm_ggml_context * ctx,
    lm_ggml_tensor * x_ct,
    lm_ggml_tensor * att_norm_w,
    lm_ggml_tensor * ffn_norm_w,
    lm_ggml_tensor * c_attn_w,
    lm_ggml_tensor * c_proj_w,
    lm_ggml_tensor * fc1_w,
    lm_ggml_tensor * fc2_w,
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

    lm_ggml_tensor * h = codec_op_rms_norm_ct(ctx, x_ct, 1e-6f, att_norm_w);
    if (h == nullptr) return nullptr;

    lm_ggml_tensor * qkv = lm_ggml_mul_mat(ctx, c_attn_w, h);                                  // [3*hidden, t]
    if (qkv == nullptr) return nullptr;

    const int64_t t = qkv->ne[1];
    lm_ggml_tensor * q = lm_ggml_cont(ctx, lm_ggml_view_2d(ctx, qkv, hidden_dim, t, qkv->nb[1], 0));
    lm_ggml_tensor * k = lm_ggml_cont(ctx, lm_ggml_view_2d(ctx, qkv, hidden_dim, t, qkv->nb[1], (size_t) hidden_dim * qkv->nb[0]));
    lm_ggml_tensor * v = lm_ggml_cont(ctx, lm_ggml_view_2d(ctx, qkv, hidden_dim, t, qkv->nb[1], (size_t) hidden_dim * qkv->nb[0] * 2));

    lm_ggml_tensor * q_dht = lm_ggml_reshape_3d(ctx, q, head_dim, n_heads, t);                  // [d, h, t]
    lm_ggml_tensor * k_dht = lm_ggml_reshape_3d(ctx, k, head_dim, n_heads, t);
    lm_ggml_tensor * v_dth = lm_ggml_permute(ctx, lm_ggml_reshape_3d(ctx, v, head_dim, n_heads, t),
                                       0, 2, 1, 3);                                       // [d, t, h]

    lm_ggml_tensor * q_rope_dht = codec_op_rope(ctx, q_dht, head_dim, rope_theta, 1.0f, CODEC_ROPE_MODE_NORMAL);
    lm_ggml_tensor * k_rope_dht = codec_op_rope(ctx, k_dht, head_dim, rope_theta, 1.0f, CODEC_ROPE_MODE_NORMAL);
    if (q_rope_dht == nullptr || k_rope_dht == nullptr) return nullptr;
    lm_ggml_tensor * q_rope = lm_ggml_cont(ctx, lm_ggml_permute(ctx, q_rope_dht, 0, 2, 1, 3));     // [d, t, h]
    lm_ggml_tensor * k_rope = lm_ggml_cont(ctx, lm_ggml_permute(ctx, k_rope_dht, 0, 2, 1, 3));     // [d, t, h]

    codec_lm_attn_params attn_p = {};
    attn_p.scale = 1.0f / std::sqrt((float) head_dim);
    attn_p.causal = false;
    lm_ggml_tensor * attn_ctx = codec_op_lm_attn_ctx_dth(ctx, q_rope, k_rope, v_dth, &attn_p); // [d, t, h]
    if (attn_ctx == nullptr) return nullptr;

    lm_ggml_tensor * attn_ct = lm_ggml_reshape_2d(
        ctx,
        lm_ggml_cont(ctx, lm_ggml_permute(ctx, attn_ctx, 0, 2, 1, 3)),
        hidden_dim,
        t);
    lm_ggml_tensor * attn_proj = lm_ggml_mul_mat(ctx, c_proj_w, attn_ct);                       // [hidden, t]
    if (attn_proj == nullptr) return nullptr;
    x_ct = lm_ggml_add(ctx, x_ct, attn_proj);

    lm_ggml_tensor * m = codec_op_rms_norm_ct(ctx, x_ct, 1e-6f, ffn_norm_w);
    if (m == nullptr) return nullptr;
    lm_ggml_tensor * ff = lm_ggml_mul_mat(ctx, fc1_w, m);
    ff = lm_ggml_silu(ctx, ff);
    ff = lm_ggml_mul_mat(ctx, fc2_w, ff);
    if (ff == nullptr) return nullptr;
    return lm_ggml_add(ctx, x_ct, ff);
}

lm_ggml_tensor * codec_op_espnet_rel_pos_emb(
    lm_ggml_context * ctx,
    int32_t t,
    int32_t d_model) {
    if (ctx == nullptr || t <= 0 || d_model <= 0 || (d_model & 1) != 0) return nullptr;
    const int32_t half = d_model / 2;
    const int32_t pe_len = 2 * t - 1;

    // Position values per row r ∈ [0, 2T-2] are p_r = (T-1) - r, giving the
    // sequence [T-1, T-2, ..., 0, -1, ..., -(T-1)].
    lm_ggml_tensor * row_idx = lm_ggml_arange(ctx, 0.0f, (float) pe_len, 1.0f);    // [pe_len]
    lm_ggml_tensor * pos = lm_ggml_scale_bias(ctx, row_idx, -1.0f, (float) (t - 1));

    // freq[k] = exp(-2k * log(10000) / d_model) for k ∈ [0, half).
    lm_ggml_tensor * k_idx = lm_ggml_arange(ctx, 0.0f, (float) half, 1.0f);
    lm_ggml_tensor * log_k = lm_ggml_scale(ctx, k_idx, -2.0f * std::log(10000.0f) / (float) d_model);
    lm_ggml_tensor * freqs = lm_ggml_exp(ctx, log_k);                                // [half]

    // Outer product: angle[k, r] = freqs[k] * pos[r]. Broadcast both to [half, pe_len].
    lm_ggml_tensor * pos_2d = lm_ggml_reshape_2d(ctx, pos, 1, pe_len);
    lm_ggml_tensor * freqs_2d = lm_ggml_reshape_2d(ctx, freqs, half, 1);
    lm_ggml_tensor * tmpl = lm_ggml_new_tensor_2d(ctx, LM_GGML_TYPE_F32, half, pe_len);
    lm_ggml_tensor * angle = lm_ggml_mul(ctx,
        lm_ggml_repeat(ctx, freqs_2d, tmpl),
        lm_ggml_repeat(ctx, pos_2d, tmpl));

    // Interleave sin/cos along ne[0]: stack as [half, 2, pe_len], permute axes
    // (ne[0],ne[1]) → (ne[1],ne[0]) so the interleaved layout falls out as the
    // contiguous flatten over (k, j∈{sin,cos}, r).
    lm_ggml_tensor * sin_3d = lm_ggml_reshape_3d(ctx, lm_ggml_sin(ctx, angle), half, 1, pe_len);
    lm_ggml_tensor * cos_3d = lm_ggml_reshape_3d(ctx, lm_ggml_cos(ctx, angle), half, 1, pe_len);
    lm_ggml_tensor * stacked = lm_ggml_concat(ctx, sin_3d, cos_3d, /*dim=*/1);       // [half, 2, pe_len]
    lm_ggml_tensor * permuted = lm_ggml_cont(ctx, lm_ggml_permute(ctx, stacked, 1, 0, 2, 3));
    return lm_ggml_reshape_2d(ctx, permuted, d_model, pe_len);
}
