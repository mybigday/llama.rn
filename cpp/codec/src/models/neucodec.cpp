#include "neucodec.h"

#include "../ops/conv1d.h"
#include "../ops/lm_ggml_ops.h"
#include "../ops/lm_attn.h"
#include "../ops/local_attn.h"
#include "../ops/rope.h"
#include "../ops/pool1d.h"
#include "../runtime/audio_dsp.h"
#include "../runtime/graph.h"
#include "../runtime/tensor_utils.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <new>
#include <string>
#include <vector>

static std::string codec_neu_name_tok() { return "neucodec.decode.tok"; }
static std::string codec_neu_name_codebook() { return "neucodec.decode.codebook"; }
static std::string codec_neu_name_quant_w() { return "neucodec.decode.quant.project_out.w"; }
static std::string codec_neu_name_quant_b() { return "neucodec.decode.quant.project_out.b"; }
static std::string codec_neu_name_fc_post_w() { return "neucodec.decode.fc_post_a.w"; }
static std::string codec_neu_name_fc_post_b() { return "neucodec.decode.fc_post_a.b"; }
static std::string codec_neu_name_embed_w() { return "neucodec.decode.embed.w"; }
static std::string codec_neu_name_embed_b() { return "neucodec.decode.embed.b"; }
static std::string codec_neu_name_final_ln_w() { return "neucodec.decode.final_ln.w"; }
static std::string codec_neu_name_final_ln_b() { return "neucodec.decode.final_ln.b"; }
static std::string codec_neu_name_head_w() { return "neucodec.decode.head.out.w"; }
static std::string codec_neu_name_head_b() { return "neucodec.decode.head.out.b"; }
static std::string codec_neu_name_istft_window() { return "neucodec.decode.istft.window"; }

static std::string codec_neu_name_prior(int32_t li, const char * suffix) {
    return "neucodec.decode.prior." + std::to_string(li) + "." + suffix;
}

static std::string codec_neu_name_post(int32_t li, const char * suffix) {
    return "neucodec.decode.post." + std::to_string(li) + "." + suffix;
}

static std::string codec_neu_name_transformer(int32_t li, const char * suffix) {
    return "neucodec.decode.transformer." + std::to_string(li) + "." + suffix;
}

static std::string codec_neu_encode_name(const std::string & name) {
    if (name.rfind("neucodec.encode.", 0) != 0) {
        return name;
    }
    uint64_t h = 1469598103934665603ull;
    for (char c : name) {
        h ^= (uint8_t) c;
        h *= 1099511628211ull;
    }
    char buf[32];
    std::snprintf(buf, sizeof(buf), "nce.%016llx", (unsigned long long) h);
    return std::string(buf);
}

static void codec_neu_set_enc_name(lm_ggml_tensor * tensor, const std::string & name) {
    if (tensor == nullptr) {
        return;
    }
    const std::string short_name = codec_neu_encode_name(name);
    lm_ggml_set_name(tensor, short_name.c_str());
}


// Returns the raw GGUF weight tensor for `name`, applying encoder-name hashing
// for `neucodec.encode.*` entries (which are stored under short `nce.*` digests).
// Decode-side names pass through unchanged.
static lm_ggml_tensor * codec_neu_get_tensor(codec_model * model, const std::string & name) {
    return codec_model_get_tensor(model, codec_neu_encode_name(name));
}

// Resolve a logical NeuCodec tensor name to its (possibly hashed) GGUF entry,
// then return it cast to F32 in the eval graph. Encode-side tensors live under
// hashed names (codec_neu_encode_name); decode-side names pass through.
static lm_ggml_tensor * codec_neu_W(lm_ggml_context * ctx_eval, const codec_model * model, const std::string & name) {
    return codec_graph_weight(ctx_eval, model, codec_neu_encode_name(name));
}

static float codec_neu_silu(float x) {
    // Match torch.nn.functional.silu behavior with numerically stable sigmoid.
    // This avoids large-|x| overflow drift when precomputing dynamic position bias.
    if (x >= 0.0f) {
        const float e = std::exp(-x);
        return x / (1.0f + e);
    }
    const float e = std::exp(x);
    return x * e / (1.0f + e);
}

static bool codec_neu_build_dynamic_pos_bias(
    codec_model * model,
    const std::string & prefix,
    int32_t max_dist,
    std::vector<float> * out,
    std::string * err) {

    if (model == nullptr || out == nullptr || max_dist <= 0) {
        if (err) *err = "invalid dynamic_pos_bias args";
        return false;
    }

    const std::string w0 = prefix + ".mlp.0.weight";
    const std::string b0 = prefix + ".mlp.0.bias";
    const std::string w1 = prefix + ".mlp.2.weight";
    const std::string b1 = prefix + ".mlp.2.bias";
    const std::string w2 = prefix + ".mlp.4.weight";
    const std::string b2 = prefix + ".mlp.4.bias";

    lm_ggml_tensor * tw0 = codec_neu_get_tensor(model, w0);
    lm_ggml_tensor * tb0 = codec_neu_get_tensor(model, b0);
    lm_ggml_tensor * tw1 = codec_neu_get_tensor(model, w1);
    lm_ggml_tensor * tb1 = codec_neu_get_tensor(model, b1);
    lm_ggml_tensor * tw2 = codec_neu_get_tensor(model, w2);
    lm_ggml_tensor * tb2 = codec_neu_get_tensor(model, b2);
    if (tw0 == nullptr || tb0 == nullptr || tw1 == nullptr || tb1 == nullptr || tw2 == nullptr || tb2 == nullptr) {
        if (err) *err = "missing dynamic_pos_bias tensors for " + prefix;
        return false;
    }

    std::vector<float> w0v, b0v, w1v, b1v, w2v, b2v;
    if (!codec_tensor_as_vec_f32(tw0, &w0v) ||
        !codec_tensor_as_vec_f32(tb0, &b0v) ||
        !codec_tensor_as_vec_f32(tw1, &w1v) ||
        !codec_tensor_as_vec_f32(tb1, &b1v) ||
        !codec_tensor_as_vec_f32(tw2, &w2v) ||
        !codec_tensor_as_vec_f32(tb2, &b2v)) {
        if (err) *err = "failed reading dynamic_pos_bias tensors";
        return false;
    }

    const int32_t dim = (int32_t) b0v.size();
    const int32_t heads = (int32_t) b2v.size();
    if ((int32_t) w0v.size() != dim * 1 || (int32_t) w1v.size() != dim * dim || (int32_t) w2v.size() != heads * dim) {
        if (err) *err = "unexpected dynamic_pos_bias tensor shapes";
        return false;
    }

    out->assign((size_t) heads * (size_t) max_dist, 0.0f);
    std::vector<float> y0((size_t) dim, 0.0f);
    std::vector<float> y1((size_t) dim, 0.0f);
    std::vector<float> y2((size_t) heads, 0.0f);

    for (int32_t d = 0; d < max_dist; ++d) {
        const float x = (float) d;
        for (int32_t i = 0; i < dim; ++i) {
            float v = w0v[(size_t) i] * x + b0v[(size_t) i];
            y0[(size_t) i] = codec_neu_silu(v);
        }

        for (int32_t j = 0; j < dim; ++j) {
            float acc = b1v[(size_t) j];
            const float * w_row = &w1v[(size_t) j * (size_t) dim];
            for (int32_t i = 0; i < dim; ++i) {
                acc += w_row[(size_t) i] * y0[(size_t) i];
            }
            y1[(size_t) j] = codec_neu_silu(acc);
        }

        for (int32_t h = 0; h < heads; ++h) {
            float acc = b2v[(size_t) h];
            const float * w_row = &w2v[(size_t) h * (size_t) dim];
            for (int32_t i = 0; i < dim; ++i) {
                acc += w_row[(size_t) i] * y1[(size_t) i];
            }
            y2[(size_t) h] = acc;
        }

        for (int32_t h = 0; h < heads; ++h) {
            (*out)[(size_t) h * (size_t) max_dist + (size_t) d] = y2[(size_t) h];
        }
    }

    return true;
}

static lm_ggml_tensor * codec_neu_grn_tc(
    lm_ggml_context * ctx_eval,
    lm_ggml_tensor * x_tc,
    lm_ggml_tensor * gamma,
    lm_ggml_tensor * beta,
    float eps) {

    if (ctx_eval == nullptr || x_tc == nullptr || gamma == nullptr || beta == nullptr) {
        return nullptr;
    }
    (void) eps;
    // distill-neucodec GRN is applied in channels_last format with T=1 reduction axes,
    // which simplifies to x + gamma * x + beta.
    lm_ggml_tensor * g2 = lm_ggml_reshape_2d(ctx_eval, gamma, 1, x_tc->ne[1]);
    lm_ggml_tensor * b2 = lm_ggml_reshape_2d(ctx_eval, beta, 1, x_tc->ne[1]);
    lm_ggml_tensor * y = lm_ggml_mul(ctx_eval, x_tc, lm_ggml_repeat(ctx_eval, g2, x_tc));
    y = lm_ggml_add(ctx_eval, y, lm_ggml_repeat(ctx_eval, b2, y));
    y = lm_ggml_add(ctx_eval, y, x_tc);
    return y;
}

static lm_ggml_tensor * codec_neu_snake_tc(
    lm_ggml_context * ctx_eval,
    lm_ggml_tensor * x_tc,
    lm_ggml_tensor * alpha,
    float eps) {

    if (ctx_eval == nullptr || x_tc == nullptr || alpha == nullptr) {
        return nullptr;
    }
    lm_ggml_tensor * alpha_2d = lm_ggml_reshape_2d(ctx_eval, alpha, 1, x_tc->ne[1]);
    lm_ggml_tensor * alpha_rep = lm_ggml_repeat(ctx_eval, alpha_2d, x_tc);
    lm_ggml_tensor * alpha_eps = lm_ggml_scale_bias(ctx_eval, alpha_rep, 1.0f, eps);
    lm_ggml_tensor * ax = lm_ggml_mul(ctx_eval, alpha_rep, x_tc);
    lm_ggml_tensor * s = lm_ggml_sin(ctx_eval, ax);
    lm_ggml_tensor * s2 = lm_ggml_mul(ctx_eval, s, s);
    lm_ggml_tensor * frac = lm_ggml_div(ctx_eval, s2, alpha_eps);
    return lm_ggml_add(ctx_eval, x_tc, frac);
}

static lm_ggml_tensor * codec_neu_conv1d_grouped(
    lm_ggml_context * ctx_eval,
    lm_ggml_tensor * x_tc,
    lm_ggml_tensor * w,
    lm_ggml_tensor * b,
    int32_t stride,
    int32_t dilation,
    int32_t padding,
    int32_t groups) {

    if (ctx_eval == nullptr || x_tc == nullptr || w == nullptr || groups <= 0) {
        return nullptr;
    }
    const int32_t in_channels = (int32_t) x_tc->ne[1];
    const int32_t out_channels = (int32_t) w->ne[2];
    if (in_channels % groups != 0 || out_channels % groups != 0) {
        return nullptr;
    }
    const int32_t in_g = in_channels / groups;
    const int32_t out_g = out_channels / groups;

    lm_ggml_tensor * out = nullptr;
    for (int32_t g = 0; g < groups; ++g) {
        const size_t x_off = (size_t) g * (size_t) in_g * x_tc->nb[1];
        lm_ggml_tensor * x_g = lm_ggml_view_2d(ctx_eval, x_tc, (int32_t) x_tc->ne[0], in_g, x_tc->nb[1], x_off);

        const size_t w_off = (size_t) g * (size_t) out_g * w->nb[2];
        lm_ggml_tensor * w_g = lm_ggml_view_3d(ctx_eval, w, (int32_t) w->ne[0], in_g, out_g, w->nb[1], w->nb[2], w_off);
        lm_ggml_tensor * b_g = nullptr;
        if (b != nullptr) {
            const size_t b_off = (size_t) g * (size_t) out_g * b->nb[0];
            b_g = lm_ggml_view_1d(ctx_eval, b, out_g, b_off);
        }

        lm_ggml_tensor * y_g = codec_conv1d(ctx_eval, x_g, w_g, b_g, stride, dilation, padding);
        if (y_g == nullptr) {
            return nullptr;
        }
        out = out == nullptr ? y_g : lm_ggml_concat(ctx_eval, out, y_g, 1);
    }
    return lm_ggml_cont(ctx_eval, out);
}

static lm_ggml_tensor * codec_neu_resnet_block(
    lm_ggml_context * ctx_eval,
    lm_ggml_tensor * x_tc,
    lm_ggml_tensor * n1_w,
    lm_ggml_tensor * n1_b,
    lm_ggml_tensor * c1_w,
    lm_ggml_tensor * c1_b,
    lm_ggml_tensor * n2_w,
    lm_ggml_tensor * n2_b,
    lm_ggml_tensor * c2_w,
    lm_ggml_tensor * c2_b) {

    if (ctx_eval == nullptr || x_tc == nullptr) {
        return nullptr;
    }

    lm_ggml_tensor * h = codec_op_group_norm(ctx_eval, x_tc, 32, 1e-6f, n1_w, n1_b);
    if (h == nullptr) {
        return nullptr;
    }
    h = lm_ggml_silu(ctx_eval, h);
    h = codec_conv1d(ctx_eval, h, c1_w, c1_b, 1, 1, 1);
    if (h == nullptr) {
        return nullptr;
    }
    h = codec_op_group_norm(ctx_eval, h, 32, 1e-6f, n2_w, n2_b);
    if (h == nullptr) {
        return nullptr;
    }
    h = lm_ggml_silu(ctx_eval, h);
    h = codec_conv1d(ctx_eval, h, c2_w, c2_b, 1, 1, 1);
    if (h == nullptr) {
        return nullptr;
    }

    return lm_ggml_add(ctx_eval, x_tc, h);
}

static lm_ggml_tensor * codec_neu_transformer_block(
    lm_ggml_context * ctx_eval,
    lm_ggml_tensor * x_ct,
    lm_ggml_tensor * att_norm_w,
    lm_ggml_tensor * ffn_norm_w,
    lm_ggml_tensor * att_c_attn_w,
    lm_ggml_tensor * att_c_proj_w,
    lm_ggml_tensor * mlp_fc1_w,
    lm_ggml_tensor * mlp_fc2_w,
    int32_t head_dim,
    int32_t n_heads,
    float rope_theta) {

    if (ctx_eval == nullptr || x_ct == nullptr) {
        return nullptr;
    }
    const int32_t hidden_dim = (int32_t) x_ct->ne[0];
    if (hidden_dim != head_dim * n_heads) {
        return nullptr;
    }

    lm_ggml_tensor * h = codec_op_rms_norm_ct(ctx_eval, x_ct, 1e-6f, att_norm_w);
    if (h == nullptr) {
        return nullptr;
    }

    lm_ggml_tensor * qkv = lm_ggml_mul_mat(ctx_eval, att_c_attn_w, h); // [3*hidden, t]
    if (qkv == nullptr) {
        return nullptr;
    }

    const int64_t t = qkv->ne[1];
    lm_ggml_tensor * q = lm_ggml_cont(ctx_eval, lm_ggml_view_2d(ctx_eval, qkv, hidden_dim, t, qkv->nb[1], 0));
    lm_ggml_tensor * k = lm_ggml_cont(ctx_eval, lm_ggml_view_2d(ctx_eval, qkv, hidden_dim, t, qkv->nb[1], (size_t) hidden_dim * qkv->nb[0]));
    lm_ggml_tensor * v = lm_ggml_cont(ctx_eval, lm_ggml_view_2d(ctx_eval, qkv, hidden_dim, t, qkv->nb[1], (size_t) hidden_dim * qkv->nb[0] * 2));

    lm_ggml_tensor * q_dht = lm_ggml_reshape_3d(ctx_eval, q, head_dim, n_heads, t); // [d, h, t]
    lm_ggml_tensor * k_dht = lm_ggml_reshape_3d(ctx_eval, k, head_dim, n_heads, t); // [d, h, t]
    lm_ggml_tensor * v_dth = lm_ggml_permute(ctx_eval, lm_ggml_reshape_3d(ctx_eval, v, head_dim, n_heads, t), 0, 2, 1, 3); // [d, t, h]

    lm_ggml_tensor * q_rope_dht = codec_op_rope(ctx_eval, q_dht, head_dim, rope_theta, 1.0f, CODEC_ROPE_MODE_NORMAL);
    lm_ggml_tensor * k_rope_dht = codec_op_rope(ctx_eval, k_dht, head_dim, rope_theta, 1.0f, CODEC_ROPE_MODE_NORMAL);
    lm_ggml_tensor * q_rope = q_rope_dht ? lm_ggml_cont(ctx_eval, lm_ggml_permute(ctx_eval, q_rope_dht, 0, 2, 1, 3)) : nullptr; // [d, t, h]
    lm_ggml_tensor * k_rope = k_rope_dht ? lm_ggml_cont(ctx_eval, lm_ggml_permute(ctx_eval, k_rope_dht, 0, 2, 1, 3)) : nullptr; // [d, t, h]
    if (q_rope == nullptr || k_rope == nullptr) {
        return nullptr;
    }

    codec_lm_attn_params attn_p = {};
    attn_p.scale = 1.0f / std::sqrt((float) head_dim);
    attn_p.causal = false;
    lm_ggml_tensor * attn_ctx = codec_op_lm_attn_ctx_dth(ctx_eval, q_rope, k_rope, v_dth, &attn_p); // [d, t, h]
    if (attn_ctx == nullptr) {
        return nullptr;
    }
    lm_ggml_tensor * attn_ct = lm_ggml_reshape_2d(
        ctx_eval,
        lm_ggml_cont(ctx_eval, lm_ggml_permute(ctx_eval, attn_ctx, 0, 2, 1, 3)),
        hidden_dim,
        t);

    lm_ggml_tensor * attn_proj = lm_ggml_mul_mat(ctx_eval, att_c_proj_w, attn_ct); // [hidden, t]
    if (attn_proj == nullptr) {
        return nullptr;
    }
    x_ct = lm_ggml_add(ctx_eval, x_ct, attn_proj);

    lm_ggml_tensor * m = codec_op_rms_norm_ct(ctx_eval, x_ct, 1e-6f, ffn_norm_w);
    if (m == nullptr) {
        return nullptr;
    }
    lm_ggml_tensor * ff = lm_ggml_mul_mat(ctx_eval, mlp_fc1_w, m);
    ff = lm_ggml_silu(ctx_eval, ff);
    ff = lm_ggml_mul_mat(ctx_eval, mlp_fc2_w, ff);
    if (ff == nullptr) {
        return nullptr;
    }
    x_ct = lm_ggml_add(ctx_eval, x_ct, ff);
    return x_ct;
}

static lm_ggml_tensor * codec_neu_attention_full_tc(
    lm_ggml_context * ctx_eval,
    lm_ggml_tensor * x_tc,
    lm_ggml_tensor * w_q,
    lm_ggml_tensor * b_q,
    lm_ggml_tensor * w_k,
    lm_ggml_tensor * b_k,
    lm_ggml_tensor * w_v,
    lm_ggml_tensor * b_v,
    lm_ggml_tensor * w_o,
    lm_ggml_tensor * b_o,
    int32_t head_dim,
    int32_t n_heads) {

    if (ctx_eval == nullptr || x_tc == nullptr || w_q == nullptr || w_k == nullptr || w_v == nullptr || w_o == nullptr) {
        return nullptr;
    }

    lm_ggml_tensor * x_ct = lm_ggml_cont(ctx_eval, lm_ggml_transpose(ctx_eval, x_tc)); // [c, t]
    lm_ggml_tensor * q_ct = codec_op_linear(ctx_eval, x_ct, w_q, b_q);
    lm_ggml_tensor * k_ct = codec_op_linear(ctx_eval, x_ct, w_k, b_k);
    lm_ggml_tensor * v_ct = codec_op_linear(ctx_eval, x_ct, w_v, b_v);
    if (q_ct == nullptr || k_ct == nullptr || v_ct == nullptr) {
        return nullptr;
    }

    const int64_t t = q_ct->ne[1];
    lm_ggml_tensor * q_dht = lm_ggml_reshape_3d(ctx_eval, q_ct, head_dim, n_heads, t); // [d,h,t]
    lm_ggml_tensor * k_dht = lm_ggml_reshape_3d(ctx_eval, k_ct, head_dim, n_heads, t);
    lm_ggml_tensor * v_dth = lm_ggml_permute(ctx_eval, lm_ggml_reshape_3d(ctx_eval, v_ct, head_dim, n_heads, t), 0, 2, 1, 3); // [d,t,h]

    lm_ggml_tensor * q_dth = lm_ggml_cont(ctx_eval, lm_ggml_permute(ctx_eval, q_dht, 0, 2, 1, 3)); // [d,t,h]
    lm_ggml_tensor * k_dth = lm_ggml_cont(ctx_eval, lm_ggml_permute(ctx_eval, k_dht, 0, 2, 1, 3)); // [d,t,h]
    codec_lm_attn_params attn_p = {};
    attn_p.scale = 1.0f / std::sqrt((float) head_dim);
    attn_p.causal = false;
    lm_ggml_tensor * attn_ctx = codec_op_lm_attn_ctx_dth(ctx_eval, q_dth, k_dth, v_dth, &attn_p); // [d,t,h]
    if (attn_ctx == nullptr) {
        return nullptr;
    }
    lm_ggml_tensor * attn_ct = lm_ggml_reshape_2d(
        ctx_eval,
        lm_ggml_cont(ctx_eval, lm_ggml_permute(ctx_eval, attn_ctx, 0, 2, 1, 3)),
        head_dim * n_heads,
        t);

    lm_ggml_tensor * out_ct = codec_op_linear(ctx_eval, attn_ct, w_o, b_o);
    if (out_ct == nullptr) {
        return nullptr;
    }
    return lm_ggml_cont(ctx_eval, lm_ggml_transpose(ctx_eval, out_ct)); // [t,c]
}

static lm_ggml_tensor * codec_neu_geglu_tc(
    lm_ggml_context * ctx_eval,
    lm_ggml_tensor * x_tc,
    int32_t inner_dim) {

    if (ctx_eval == nullptr || x_tc == nullptr || inner_dim <= 0) {
        return nullptr;
    }
    const int32_t t = (int32_t) x_tc->ne[0];
    const int32_t c = (int32_t) x_tc->ne[1];
    if (c != inner_dim * 2) {
        return nullptr;
    }

    lm_ggml_tensor * x1 = lm_ggml_view_2d(ctx_eval, x_tc, t, inner_dim, x_tc->nb[1], 0);
    lm_ggml_tensor * x2 = lm_ggml_view_2d(ctx_eval, x_tc, t, inner_dim, x_tc->nb[1], (size_t) inner_dim * x_tc->nb[1]);
    x1 = lm_ggml_cont(ctx_eval, x1);
    x2 = lm_ggml_cont(ctx_eval, x2);
    lm_ggml_tensor * x2_gelu = lm_ggml_gelu_erf(ctx_eval, x2);
    return lm_ggml_mul(ctx_eval, x1, x2_gelu);
}

static lm_ggml_tensor * codec_neu_local_mha_tc(
    lm_ggml_context * ctx_eval,
    lm_ggml_tensor * x_tc,
    lm_ggml_tensor * ln_w,
    lm_ggml_tensor * ln_b,
    lm_ggml_tensor * w_qkv,
    lm_ggml_tensor * w_out,
    int32_t heads,
    int32_t head_dim,
    lm_ggml_tensor * mask_score_kqh,
    float ln_eps) {

    if (ctx_eval == nullptr || x_tc == nullptr || w_qkv == nullptr || w_out == nullptr ||
        mask_score_kqh == nullptr) {
        return nullptr;
    }
    lm_ggml_tensor * h_tc = x_tc;
    if (ln_w != nullptr && ln_b != nullptr) {
        h_tc = codec_op_layer_norm_tc(ctx_eval, h_tc, ln_eps, ln_w, ln_b);
        if (h_tc == nullptr) {
            return nullptr;
        }
    }

    lm_ggml_tensor * qkv_tc = codec_op_linear_tc(ctx_eval, h_tc, w_qkv, nullptr); // [t, 3*dim]
    if (qkv_tc == nullptr) {
        return nullptr;
    }

    const int32_t t = (int32_t) qkv_tc->ne[0];
    const int32_t qkv_out = (int32_t) qkv_tc->ne[1];
    if (qkv_out % 3 != 0) {
        return nullptr;
    }
    const int32_t inner = qkv_out / 3;
    if (inner != heads * head_dim) {
        return nullptr;
    }
    lm_ggml_tensor * q_tc = lm_ggml_view_2d(ctx_eval, qkv_tc, t, inner, qkv_tc->nb[1], 0);
    lm_ggml_tensor * k_tc = lm_ggml_view_2d(ctx_eval, qkv_tc, t, inner, qkv_tc->nb[1], (size_t) inner * qkv_tc->nb[1]);
    lm_ggml_tensor * v_tc = lm_ggml_view_2d(ctx_eval, qkv_tc, t, inner, qkv_tc->nb[1], (size_t) inner * qkv_tc->nb[1] * 2);
    q_tc = lm_ggml_cont(ctx_eval, q_tc);
    k_tc = lm_ggml_cont(ctx_eval, k_tc);
    v_tc = lm_ggml_cont(ctx_eval, v_tc);

    lm_ggml_tensor * q_ct = lm_ggml_cont(ctx_eval, lm_ggml_transpose(ctx_eval, q_tc));
    lm_ggml_tensor * k_ct = lm_ggml_cont(ctx_eval, lm_ggml_transpose(ctx_eval, k_tc));
    lm_ggml_tensor * v_ct = lm_ggml_cont(ctx_eval, lm_ggml_transpose(ctx_eval, v_tc));
    lm_ggml_tensor * q_dht = lm_ggml_reshape_3d(ctx_eval, q_ct, head_dim, heads, t);
    lm_ggml_tensor * k_dht = lm_ggml_reshape_3d(ctx_eval, k_ct, head_dim, heads, t);
    lm_ggml_tensor * v_dht = lm_ggml_reshape_3d(ctx_eval, v_ct, head_dim, heads, t);
    lm_ggml_tensor * q_dth = lm_ggml_cont(ctx_eval, lm_ggml_permute(ctx_eval, q_dht, 0, 2, 1, 3));
    lm_ggml_tensor * k_dth = lm_ggml_cont(ctx_eval, lm_ggml_permute(ctx_eval, k_dht, 0, 2, 1, 3));
    lm_ggml_tensor * v_dth = lm_ggml_cont(ctx_eval, lm_ggml_permute(ctx_eval, v_dht, 0, 2, 1, 3));

    lm_ggml_tensor * attn_dth = codec_op_local_attn(ctx_eval, q_dth, k_dth, v_dth, mask_score_kqh, head_dim, heads);
    if (attn_dth == nullptr) {
        return nullptr;
    }

    lm_ggml_tensor * attn_dht = lm_ggml_cont(ctx_eval, lm_ggml_permute(ctx_eval, attn_dth, 0, 2, 1, 3));
    lm_ggml_tensor * attn_ct = lm_ggml_reshape_2d(ctx_eval, attn_dht, inner, t);
    lm_ggml_tensor * out_tc = codec_op_linear_tc(ctx_eval, lm_ggml_cont(ctx_eval, lm_ggml_transpose(ctx_eval, attn_ct)), w_out, nullptr);
    return out_tc;
}


struct neucodec_decode_build {
    int32_t t = 0;
    int32_t q = 0;
    int32_t codebook_dim = 0;
    int32_t codebook_size = 0;
    int32_t vq_dim = 0;
    int32_t hidden_dim = 0;
    int32_t num_layers = 0;
    int32_t num_heads = 0;
    int32_t head_dim = 0;
    int32_t head_out_dim = 0;
    float rope_theta = 10000.0f;
    const codec_model * model = nullptr;
};

static bool codec_neu_build_decode(lm_ggml_context * ctx_eval, void * user_data, lm_ggml_tensor ** out) {
    neucodec_decode_build * p = static_cast<neucodec_decode_build *>(user_data);
    if (ctx_eval == nullptr || p == nullptr || out == nullptr || p->model == nullptr) {
        return false;
    }
    if (p->t <= 0 || p->q <= 0 || p->codebook_dim <= 0 || p->codebook_size <= 0 ||
        p->vq_dim <= 0 || p->hidden_dim <= 0 || p->num_layers <= 0 || p->num_heads <= 0 || p->head_dim <= 0 ||
        p->head_out_dim <= 0) {
        return false;
    }

    auto W = [&](const std::string & name) -> lm_ggml_tensor * {
        return codec_neu_W(ctx_eval, p->model, name);
    };

    lm_ggml_tensor * t_tok = lm_ggml_new_tensor_2d(ctx_eval, LM_GGML_TYPE_I32, p->t, p->q);
    lm_ggml_set_name(t_tok, codec_neu_name_tok().c_str());

    lm_ggml_tensor * t_codebook = W(codec_neu_name_codebook());
    if (t_codebook == nullptr) {
        return false;
    }

    lm_ggml_tensor * t_idx = lm_ggml_view_1d(ctx_eval, t_tok, p->t, 0);
    lm_ggml_tensor * t_q = lm_ggml_get_rows(ctx_eval, t_codebook, t_idx); // [codebook_dim, t]

    lm_ggml_tensor * t_qp_w = W(codec_neu_name_quant_w());
    lm_ggml_tensor * t_qp_b = W(codec_neu_name_quant_b());
    if (t_qp_w == nullptr || t_qp_b == nullptr) {
        return false;
    }

    lm_ggml_tensor * x_ct = codec_op_linear(ctx_eval, t_q, t_qp_w, t_qp_b); // [vq_dim, t]
    if (x_ct == nullptr) {
        return false;
    }

    lm_ggml_tensor * t_fc_w = W(codec_neu_name_fc_post_w());
    lm_ggml_tensor * t_fc_b = W(codec_neu_name_fc_post_b());
    if (t_fc_w == nullptr || t_fc_b == nullptr) {
        return false;
    }
    x_ct = codec_op_linear(ctx_eval, x_ct, t_fc_w, t_fc_b); // [hidden_dim, t]
    if (x_ct == nullptr) {
        return false;
    }

    lm_ggml_tensor * x_tc = lm_ggml_cont(ctx_eval, lm_ggml_transpose(ctx_eval, x_ct)); // [t, hidden]

    lm_ggml_tensor * t_embed_w = W(codec_neu_name_embed_w());
    lm_ggml_tensor * t_embed_b = W(codec_neu_name_embed_b());
    if (t_embed_w == nullptr || t_embed_b == nullptr) {
        return false;
    }
    x_tc = codec_conv1d(ctx_eval, x_tc, t_embed_w, t_embed_b, 1, 1, 3);
    if (x_tc == nullptr) {
        return false;
    }

    for (int32_t li = 0; li < 2; ++li) {
        lm_ggml_tensor * n1_w = W(codec_neu_name_prior(li, "norm1.w"));
        lm_ggml_tensor * n1_b = W(codec_neu_name_prior(li, "norm1.b"));
        lm_ggml_tensor * c1_w = W(codec_neu_name_prior(li, "conv1.w"));
        lm_ggml_tensor * c1_b = W(codec_neu_name_prior(li, "conv1.b"));
        lm_ggml_tensor * n2_w = W(codec_neu_name_prior(li, "norm2.w"));
        lm_ggml_tensor * n2_b = W(codec_neu_name_prior(li, "norm2.b"));
        lm_ggml_tensor * c2_w = W(codec_neu_name_prior(li, "conv2.w"));
        lm_ggml_tensor * c2_b = W(codec_neu_name_prior(li, "conv2.b"));
        if (n1_w == nullptr || n1_b == nullptr || c1_w == nullptr || c1_b == nullptr ||
            n2_w == nullptr || n2_b == nullptr || c2_w == nullptr || c2_b == nullptr) {
            return false;
        }

        x_tc = codec_neu_resnet_block(ctx_eval, x_tc, n1_w, n1_b, c1_w, c1_b, n2_w, n2_b, c2_w, c2_b);
        if (x_tc == nullptr) {
            return false;
        }
    }

    x_ct = lm_ggml_cont(ctx_eval, lm_ggml_transpose(ctx_eval, x_tc)); // [hidden, t]

    for (int32_t li = 0; li < p->num_layers; ++li) {
        lm_ggml_tensor * attn_w = W(codec_neu_name_transformer(li, "att_norm.w"));
        lm_ggml_tensor * ffn_w = W(codec_neu_name_transformer(li, "ffn_norm.w"));
        lm_ggml_tensor * c_attn_w = W(codec_neu_name_transformer(li, "att.c_attn.w"));
        lm_ggml_tensor * c_proj_w = W(codec_neu_name_transformer(li, "att.c_proj.w"));
        lm_ggml_tensor * fc1_w = W(codec_neu_name_transformer(li, "mlp.fc1.w"));
        lm_ggml_tensor * fc2_w = W(codec_neu_name_transformer(li, "mlp.fc2.w"));
        if (attn_w == nullptr || ffn_w == nullptr || c_attn_w == nullptr || c_proj_w == nullptr ||
            fc1_w == nullptr || fc2_w == nullptr) {
            return false;
        }

        x_ct = codec_neu_transformer_block(
            ctx_eval,
            x_ct,
            attn_w,
            ffn_w,
            c_attn_w,
            c_proj_w,
            fc1_w,
            fc2_w,
            p->head_dim,
            p->num_heads,
            p->rope_theta);
        if (x_ct == nullptr) {
            return false;
        }
    }

    x_tc = lm_ggml_cont(ctx_eval, lm_ggml_transpose(ctx_eval, x_ct)); // [t, hidden]

    for (int32_t li = 0; li < 2; ++li) {
        lm_ggml_tensor * n1_w = W(codec_neu_name_post(li, "norm1.w"));
        lm_ggml_tensor * n1_b = W(codec_neu_name_post(li, "norm1.b"));
        lm_ggml_tensor * c1_w = W(codec_neu_name_post(li, "conv1.w"));
        lm_ggml_tensor * c1_b = W(codec_neu_name_post(li, "conv1.b"));
        lm_ggml_tensor * n2_w = W(codec_neu_name_post(li, "norm2.w"));
        lm_ggml_tensor * n2_b = W(codec_neu_name_post(li, "norm2.b"));
        lm_ggml_tensor * c2_w = W(codec_neu_name_post(li, "conv2.w"));
        lm_ggml_tensor * c2_b = W(codec_neu_name_post(li, "conv2.b"));
        if (n1_w == nullptr || n1_b == nullptr || c1_w == nullptr || c1_b == nullptr ||
            n2_w == nullptr || n2_b == nullptr || c2_w == nullptr || c2_b == nullptr) {
            return false;
        }

        x_tc = codec_neu_resnet_block(ctx_eval, x_tc, n1_w, n1_b, c1_w, c1_b, n2_w, n2_b, c2_w, c2_b);
        if (x_tc == nullptr) {
            return false;
        }
    }

    x_ct = lm_ggml_cont(ctx_eval, lm_ggml_transpose(ctx_eval, x_tc)); // [hidden, t]

    lm_ggml_tensor * t_fln_w = W(codec_neu_name_final_ln_w());
    lm_ggml_tensor * t_fln_b = W(codec_neu_name_final_ln_b());
    if (t_fln_w == nullptr || t_fln_b == nullptr) {
        return false;
    }
    x_ct = codec_op_layer_norm_ct(ctx_eval, x_ct, 1e-6f, t_fln_w, t_fln_b);
    if (x_ct == nullptr) {
        return false;
    }

    lm_ggml_tensor * t_head_w = W(codec_neu_name_head_w());
    lm_ggml_tensor * t_head_b = W(codec_neu_name_head_b());
    if (t_head_w == nullptr || t_head_b == nullptr) {
        return false;
    }
    lm_ggml_tensor * t_out = codec_op_linear(ctx_eval, x_ct, t_head_w, t_head_b); // [out_dim, t]
    if (t_out == nullptr) {
        return false;
    }
    lm_ggml_set_name(t_out, "neucodec.decode.head.out");
    *out = t_out;
    return true;
}

static bool codec_neu_init_decode_build(
    codec_context * ctx,
    const codec_neucodec * neu,
    int32_t t,
    int32_t q,
    neucodec_decode_build * build,
    std::string * err) {

    if (ctx == nullptr || ctx->model == nullptr || neu == nullptr || build == nullptr || t <= 0 || q <= 0) {
        if (err != nullptr) {
            *err = "invalid NeuCodec decode build arguments";
        }
        return false;
    }

    if (neu->hidden_dim != neu->num_heads * neu->head_dim) {
        if (err != nullptr) {
            *err = "NeuCodec head_dim * num_heads mismatch";
        }
        return false;
    }

    build->t = t;
    build->q = q;
    build->model = ctx->model;
    build->codebook_dim = neu->codebook_dim;
    build->codebook_size = neu->codebook_size;
    build->vq_dim = neu->vq_dim;
    build->hidden_dim = neu->hidden_dim;
    build->num_layers = neu->num_layers;
    build->num_heads = neu->num_heads;
    build->head_dim = neu->head_dim;
    build->head_out_dim = neu->n_fft + 2;
    build->rope_theta = neu->rope_theta;

    if (build->head_out_dim <= 0) {
        if (err != nullptr) {
            *err = "invalid NeuCodec head output dimension";
        }
        return false;
    }
    return true;
}

struct neucodec_encode_build {
    int32_t n_in = 0;
    int32_t n_in_sem = 0;
    int32_t n_q = 1;
    int32_t codebook_dim = 8;
    int32_t codebook_size = 65536;
    int32_t encoder_type = 0;
    int32_t hubert_hidden = 768;
    int32_t hubert_heads = 12;
    int32_t hubert_intermediate = 3072;
    int32_t hubert_layers = 2;
    int32_t hubert_pos_k = 128;
    int32_t hubert_pos_groups = 16;
    float hubert_ln_eps = 1e-5f;
    int32_t hubert_feat_layers = 7;
    int32_t hubert_conv_dim[7] = { 512, 512, 512, 512, 512, 512, 512 };
    int32_t hubert_conv_kernel[7] = { 10, 3, 3, 3, 3, 2, 2 };
    int32_t hubert_conv_stride[7] = { 5, 2, 2, 2, 2, 2, 2 };
    int32_t local_window = 300;
    int32_t local_down_window = 1500;
    const codec_local_attn_params * down_attn = nullptr;
    const codec_local_attn_params * local_attn = nullptr;
    const codec_model * model = nullptr;
};

static lm_ggml_tensor * codec_neu_build_distill_first_block(
    lm_ggml_context * ctx_eval,
    const codec_model * model,
    lm_ggml_tensor * x_tc) {

    if (ctx_eval == nullptr || model == nullptr || x_tc == nullptr) {
        return nullptr;
    }

    const int32_t pool_kernels[5] = { 1, 5, 11, 21, 45 };
    lm_ggml_tensor * concat = nullptr;
    for (int32_t i = 0; i < 5; ++i) {
        const int32_t k = pool_kernels[i];
        lm_ggml_tensor * x_abs = lm_ggml_abs(ctx_eval, x_tc);
        lm_ggml_tensor * x_max = codec_op_max_pool1d(ctx_eval, x_abs, k, k / 2);
        lm_ggml_tensor * x_avg = codec_op_avg_pool1d(ctx_eval, x_max, k, k / 2);

        const std::string prefix = "neucodec.encode.distill.codec_encoder.encoder.blocks.0.blocks." + std::to_string(i) + ".1.";
        lm_ggml_tensor * w = codec_neu_W(ctx_eval, model, prefix + "weight");
        lm_ggml_tensor * b = codec_neu_W(ctx_eval, model, prefix + "bias");
        if (w == nullptr || b == nullptr) {
            return nullptr;
        }
        lm_ggml_tensor * y = codec_conv1d(ctx_eval, x_avg, w, b, 1, 1, 3);
        if (y == nullptr) {
            return nullptr;
        }
        concat = concat == nullptr ? y : lm_ggml_concat(ctx_eval, concat, y, 1);
    }

    lm_ggml_tensor * w1 = codec_neu_W(ctx_eval, model, "neucodec.encode.distill.codec_encoder.encoder.blocks.0.conv_1.weight");
    lm_ggml_tensor * b1 = codec_neu_W(ctx_eval, model, "neucodec.encode.distill.codec_encoder.encoder.blocks.0.conv_1.bias");
    if (w1 == nullptr || b1 == nullptr) {
        return nullptr;
    }
    lm_ggml_tensor * h = codec_conv1d(ctx_eval, concat, w1, b1, 1, 1, 0);
    if (h == nullptr) {
        return nullptr;
    }
    h = lm_ggml_gelu_erf(ctx_eval, h);

    lm_ggml_tensor * x_cat = lm_ggml_concat(ctx_eval, h, x_tc, 1);
    lm_ggml_tensor * w2 = codec_neu_W(ctx_eval, model, "neucodec.encode.distill.codec_encoder.encoder.blocks.0.conv_2.weight");
    lm_ggml_tensor * b2 = codec_neu_W(ctx_eval, model, "neucodec.encode.distill.codec_encoder.encoder.blocks.0.conv_2.bias");
    if (w2 == nullptr || b2 == nullptr) {
        return nullptr;
    }
    lm_ggml_tensor * y = codec_conv1d(ctx_eval, x_cat, w2, b2, 1, 1, 0);
    return y;
}

static lm_ggml_tensor * codec_neu_build_distill_base_unit(
    lm_ggml_context * ctx_eval,
    const codec_model * model,
    lm_ggml_tensor * x_tc,
    const std::string & prefix) {

    if (ctx_eval == nullptr || model == nullptr || x_tc == nullptr) {
        return nullptr;
    }
    lm_ggml_tensor * dw_w = codec_neu_W(ctx_eval, model, prefix + ".dw_conv.weight");
    lm_ggml_tensor * dw_b = codec_neu_W(ctx_eval, model, prefix + ".dw_conv.bias");
    if (dw_w == nullptr || dw_b == nullptr) {
        return nullptr;
    }
    lm_ggml_tensor * h = codec_conv1d_depthwise(ctx_eval, x_tc, dw_w, dw_b, 1, 1, 3);
    if (h == nullptr) {
        return nullptr;
    }

    lm_ggml_tensor * pw1_w = codec_neu_W(ctx_eval, model, prefix + ".pw_conv1.weight");
    lm_ggml_tensor * pw1_b = codec_neu_W(ctx_eval, model, prefix + ".pw_conv1.bias");
    if (pw1_w == nullptr || pw1_b == nullptr) {
        return nullptr;
    }
    h = codec_op_linear_tc(ctx_eval, h, pw1_w, pw1_b);
    if (h == nullptr) {
        return nullptr;
    }

    lm_ggml_tensor * act_a = codec_neu_W(ctx_eval, model, prefix + ".act.alpha");
    if (act_a == nullptr) {
        return nullptr;
    }
    h = codec_neu_snake_tc(ctx_eval, h, act_a, 1.1920929e-7f);
    if (h == nullptr) {
        return nullptr;
    }

    lm_ggml_tensor * grn_g = codec_neu_W(ctx_eval, model, prefix + ".grn.gamma");
    lm_ggml_tensor * grn_b = codec_neu_W(ctx_eval, model, prefix + ".grn.beta");
    if (grn_g == nullptr || grn_b == nullptr) {
        return nullptr;
    }
    h = codec_neu_grn_tc(ctx_eval, h, grn_g, grn_b, 1.1920929e-7f);
    if (h == nullptr) {
        return nullptr;
    }

    lm_ggml_tensor * pw2_w = codec_neu_W(ctx_eval, model, prefix + ".pw_conv2.weight");
    lm_ggml_tensor * pw2_b = codec_neu_W(ctx_eval, model, prefix + ".pw_conv2.bias");
    if (pw2_w == nullptr || pw2_b == nullptr) {
        return nullptr;
    }
    h = codec_op_linear_tc(ctx_eval, h, pw2_w, pw2_b);
    if (h == nullptr) {
        return nullptr;
    }

    return lm_ggml_add(ctx_eval, x_tc, h);
}

static lm_ggml_tensor * codec_neu_build_distill_local_trans(
    lm_ggml_context * ctx_eval,
    const codec_model * model,
    lm_ggml_tensor * x_tc,
    const std::string & prefix,
    int32_t depth,
    int32_t heads,
    int32_t head_dim,
    const char * mask_tensor_name) {

    if (ctx_eval == nullptr || model == nullptr || x_tc == nullptr || depth <= 0 ||
        mask_tensor_name == nullptr) {
        return nullptr;
    }

    const int32_t dim = (int32_t) x_tc->ne[1];
    (void) dim;
    const int32_t inner_dim = (int32_t) ((int32_t) x_tc->ne[1] * 4 * 2 / 3);

    // Per-trans-stage local-attention score-bias leaf, shared across all
    // `depth` LocalMHA layers (they all see the same t and the same
    // `attn_params`).  Filled CPU-side by the encoder's runtime path before
    // graph compute.
    const int32_t t = (int32_t) x_tc->ne[0];
    lm_ggml_tensor * mask_kqh = lm_ggml_new_tensor_3d(ctx_eval, LM_GGML_TYPE_F32, t, t, heads);
    lm_ggml_set_name(mask_kqh, mask_tensor_name);

    for (int32_t li = 0; li < depth; ++li) {
        const std::string lp = prefix + ".layers." + std::to_string(li);
        // LocalMHA
        lm_ggml_tensor * ln_w = codec_neu_W(ctx_eval, model, lp + ".0.norm.weight");
        lm_ggml_tensor * ln_b = codec_neu_W(ctx_eval, model, lp + ".0.norm.bias");
        lm_ggml_tensor * w_qkv = codec_neu_W(ctx_eval, model, lp + ".0.to_qkv.weight");
        lm_ggml_tensor * w_out = codec_neu_W(ctx_eval, model, lp + ".0.to_out.weight");
        if (ln_w == nullptr || ln_b == nullptr || w_qkv == nullptr || w_out == nullptr) {
            return nullptr;
        }

        lm_ggml_tensor * attn = codec_neu_local_mha_tc(ctx_eval, x_tc, ln_w, ln_b, w_qkv, w_out, heads, head_dim, mask_kqh, 1e-5f);
        if (attn == nullptr) {
            return nullptr;
        }
        x_tc = lm_ggml_add(ctx_eval, x_tc, attn);

        // FeedForward
        lm_ggml_tensor * ff_ln_w = codec_neu_W(ctx_eval, model, lp + ".1.0.weight");
        lm_ggml_tensor * ff_ln_b = codec_neu_W(ctx_eval, model, lp + ".1.0.bias");
        lm_ggml_tensor * ff_w1 = codec_neu_W(ctx_eval, model, lp + ".1.1.weight");
        lm_ggml_tensor * ff_w2 = codec_neu_W(ctx_eval, model, lp + ".1.4.weight");
        if (ff_ln_w == nullptr || ff_ln_b == nullptr || ff_w1 == nullptr || ff_w2 == nullptr) {
            return nullptr;
        }

        lm_ggml_tensor * ff = codec_op_layer_norm_tc(ctx_eval, x_tc, 1e-5f, ff_ln_w, ff_ln_b);
        if (ff == nullptr) {
            return nullptr;
        }
        ff = codec_op_linear_tc(ctx_eval, ff, ff_w1, nullptr);
        if (ff == nullptr) {
            return nullptr;
        }
        ff = codec_neu_geglu_tc(ctx_eval, ff, inner_dim);
        if (ff == nullptr) {
            return nullptr;
        }
        ff = codec_op_linear_tc(ctx_eval, ff, ff_w2, nullptr);
        if (ff == nullptr) {
            return nullptr;
        }
        x_tc = lm_ggml_add(ctx_eval, x_tc, ff);
    }

    return x_tc;
}

static bool codec_neu_build_encode(lm_ggml_context * ctx_eval, void * user_data, lm_ggml_tensor ** out) {
    neucodec_encode_build * p = static_cast<neucodec_encode_build *>(user_data);
    if (ctx_eval == nullptr || p == nullptr || out == nullptr || p->model == nullptr) {
        return false;
    }
    if (p->n_in <= 0 || p->n_in_sem <= 0) {
        return false;
    }
    if (p->encoder_type != 1) {
        return false;
    }

    auto W = [&](const std::string & name) { return codec_neu_W(ctx_eval, p->model, name); };

    lm_ggml_tensor * t_pcm = lm_ggml_new_tensor_2d(ctx_eval, LM_GGML_TYPE_F32, p->n_in, 1);
    codec_neu_set_enc_name(t_pcm, "neucodec.encode.pcm");
    lm_ggml_tensor * t_sem = lm_ggml_new_tensor_2d(ctx_eval, LM_GGML_TYPE_F32, p->n_in_sem, 1);
    codec_neu_set_enc_name(t_sem, "neucodec.encode.sem");

    // Distill acoustic encoder
    lm_ggml_tensor * x = codec_neu_build_distill_first_block(ctx_eval, p->model, t_pcm);
    if (x == nullptr) {
        return false;
    }

    // stage 0
    x = codec_neu_build_distill_base_unit(ctx_eval, p->model, x, "neucodec.encode.distill.codec_encoder.encoder.blocks.1.0.module");
    if (x == nullptr) return false;
    lm_ggml_tensor * d0_w = W("neucodec.encode.distill.codec_encoder.encoder.blocks.2.0.weight");
    lm_ggml_tensor * d0_b = W("neucodec.encode.distill.codec_encoder.encoder.blocks.2.0.bias");
    if (d0_w == nullptr || d0_b == nullptr) return false;
    x = codec_conv1d(ctx_eval, x, d0_w, d0_b, 4, 1, 0);
    if (x == nullptr) return false;

    // stage 1
    x = codec_neu_build_distill_base_unit(ctx_eval, p->model, x, "neucodec.encode.distill.codec_encoder.encoder.blocks.3.0.module");
    if (x == nullptr) return false;
    lm_ggml_tensor * d1_w = W("neucodec.encode.distill.codec_encoder.encoder.blocks.4.0.weight");
    lm_ggml_tensor * d1_b = W("neucodec.encode.distill.codec_encoder.encoder.blocks.4.0.bias");
    if (d1_w == nullptr || d1_b == nullptr) return false;
    x = codec_conv1d(ctx_eval, x, d1_w, d1_b, 4, 1, 0);
    if (x == nullptr) return false;

    // stage 2
    x = codec_neu_build_distill_base_unit(ctx_eval, p->model, x, "neucodec.encode.distill.codec_encoder.encoder.blocks.5.0.module");
    if (x == nullptr) return false;
    lm_ggml_tensor * d2_w = W("neucodec.encode.distill.codec_encoder.encoder.blocks.6.0.weight");
    lm_ggml_tensor * d2_b = W("neucodec.encode.distill.codec_encoder.encoder.blocks.6.0.bias");
    if (d2_w == nullptr || d2_b == nullptr) return false;
    x = codec_conv1d(ctx_eval, x, d2_w, d2_b, 4, 1, 0);
    if (x == nullptr) return false;

    // final stage (2 blocks)
    x = codec_neu_build_distill_base_unit(ctx_eval, p->model, x, "neucodec.encode.distill.codec_encoder.encoder.blocks.7.0.module");
    if (x == nullptr) return false;
    x = codec_neu_build_distill_base_unit(ctx_eval, p->model, x, "neucodec.encode.distill.codec_encoder.encoder.blocks.7.1.module");
    if (x == nullptr) return false;
    lm_ggml_tensor * d3_w = W("neucodec.encode.distill.codec_encoder.encoder.blocks.8.weight");
    lm_ggml_tensor * d3_b = W("neucodec.encode.distill.codec_encoder.encoder.blocks.8.bias");
    if (d3_w == nullptr || d3_b == nullptr) return false;
    x = codec_conv1d(ctx_eval, x, d3_w, d3_b, 1, 1, 1);
    if (x == nullptr) return false;
    // en_encoder down_trans
    x = codec_neu_build_distill_local_trans(
        ctx_eval,
        p->model,
        x,
        "neucodec.encode.distill.codec_encoder.en_encoder.down_trans.trans",
        2,
        6,
        512 / 4,
        "neucodec.encode.distill.down_mask");
    if (x == nullptr) {
        return false;
    }
    lm_ggml_tensor * down_w = W("neucodec.encode.distill.codec_encoder.en_encoder.down_trans.down_layer.weight");
    lm_ggml_tensor * down_b = W("neucodec.encode.distill.codec_encoder.en_encoder.down_trans.down_layer.bias");
    if (down_w == nullptr || down_b == nullptr) return false;
    x = codec_conv1d(ctx_eval, x, down_w, down_b, 5, 1, 0);
    if (x == nullptr) return false;
    // en_encoder local_trans
    x = codec_neu_build_distill_local_trans(
        ctx_eval,
        p->model,
        x,
        "neucodec.encode.distill.codec_encoder.en_encoder.local_trans",
        3,
        6,
        512 / 4,
        "neucodec.encode.distill.local_mask");
    if (x == nullptr) {
        return false;
    }
    // fc_sq_prior (512->768)
    lm_ggml_tensor * t_fc_sq_w = W("neucodec.encode.fc_sq_prior.w");
    lm_ggml_tensor * t_fc_sq_b = W("neucodec.encode.fc_sq_prior.b");
    if (t_fc_sq_w == nullptr || t_fc_sq_b == nullptr) return false;
    lm_ggml_tensor * fsq_tc = codec_op_linear_tc(ctx_eval, x, t_fc_sq_w, t_fc_sq_b);
    if (fsq_tc == nullptr) return false;
    // HuBERT semantic model
    lm_ggml_tensor * sem = t_sem;
    for (int32_t li = 0; li < p->hubert_feat_layers; ++li) {
        lm_ggml_tensor * w = W("neucodec.encode.hubert.feat.conv." + std::to_string(li) + ".w");
        if (w == nullptr) return false;
        sem = codec_conv1d(ctx_eval, sem, w, nullptr, p->hubert_conv_stride[li], 1, 0);
        if (sem == nullptr) return false;
        if (li == 0) {
            lm_ggml_tensor * gn_w = W("neucodec.encode.hubert.feat.conv.0.gn.w");
            lm_ggml_tensor * gn_b = W("neucodec.encode.hubert.feat.conv.0.gn.b");
            if (gn_w == nullptr || gn_b == nullptr) return false;
            sem = codec_op_group_norm(ctx_eval, sem, p->hubert_conv_dim[li], p->hubert_ln_eps, gn_w, gn_b);
        }
        sem = lm_ggml_gelu_erf(ctx_eval, sem);
    }

    lm_ggml_tensor * feat_w = W("neucodec.encode.hubert.feature_projection.w");
    lm_ggml_tensor * feat_b = W("neucodec.encode.hubert.feature_projection.b");
    if (feat_w == nullptr || feat_b == nullptr) return false;
    lm_ggml_tensor * hs = codec_op_linear_tc(ctx_eval, sem, feat_w, feat_b);
    if (hs == nullptr) return false;

    // positional conv
    lm_ggml_tensor * pos_w = W("neucodec.encode.hubert.encoder.pos_conv.w");
    lm_ggml_tensor * pos_b = W("neucodec.encode.hubert.encoder.pos_conv.b");
    if (pos_w == nullptr || pos_b == nullptr) return false;
    lm_ggml_tensor * pos = codec_neu_conv1d_grouped(ctx_eval, hs, pos_w, pos_b, 1, 1, p->hubert_pos_k / 2, p->hubert_pos_groups);
    if (pos == nullptr) return false;
    if ((p->hubert_pos_k % 2) == 0) {
        pos = codec_op_crop_1d(ctx_eval, pos, 0, 1);
    }
    pos = lm_ggml_gelu_erf(ctx_eval, pos);
    hs = lm_ggml_add(ctx_eval, hs, pos);

    // encoder layer norm
    lm_ggml_tensor * enc_ln_w = W("neucodec.encode.hubert.encoder.layer_norm.w");
    lm_ggml_tensor * enc_ln_b = W("neucodec.encode.hubert.encoder.layer_norm.b");
    if (enc_ln_w == nullptr || enc_ln_b == nullptr) return false;
    hs = codec_op_layer_norm_tc(ctx_eval, hs, p->hubert_ln_eps, enc_ln_w, enc_ln_b);
    if (hs == nullptr) return false;

    const int32_t h_head_dim = p->hubert_hidden / p->hubert_heads;
    for (int32_t li = 0; li < p->hubert_layers; ++li) {
        const std::string lp = "neucodec.encode.hubert.encoder.layers." + std::to_string(li);
        lm_ggml_tensor * res = hs;
        lm_ggml_tensor * q_w = W(lp + ".att.q.w");
        lm_ggml_tensor * q_b = W(lp + ".att.q.b");
        lm_ggml_tensor * k_w = W(lp + ".att.k.w");
        lm_ggml_tensor * k_b = W(lp + ".att.k.b");
        lm_ggml_tensor * v_w = W(lp + ".att.v.w");
        lm_ggml_tensor * v_b = W(lp + ".att.v.b");
        lm_ggml_tensor * o_w = W(lp + ".att.o.w");
        lm_ggml_tensor * o_b = W(lp + ".att.o.b");
        if (q_w == nullptr || q_b == nullptr || k_w == nullptr || k_b == nullptr ||
            v_w == nullptr || v_b == nullptr || o_w == nullptr || o_b == nullptr) {
            return false;
        }

        lm_ggml_tensor * attn = codec_neu_attention_full_tc(ctx_eval, hs, q_w, q_b, k_w, k_b, v_w, v_b, o_w, o_b, h_head_dim, p->hubert_heads);
        if (attn == nullptr) return false;
        hs = lm_ggml_add(ctx_eval, res, attn);

        lm_ggml_tensor * ln_w = W(lp + ".ln.w");
        lm_ggml_tensor * ln_b = W(lp + ".ln.b");
        if (ln_w == nullptr || ln_b == nullptr) return false;
        hs = codec_op_layer_norm_tc(ctx_eval, hs, p->hubert_ln_eps, ln_w, ln_b);
        if (hs == nullptr) return false;

        lm_ggml_tensor * ff1_w = W(lp + ".ffn.fc1.w");
        lm_ggml_tensor * ff1_b = W(lp + ".ffn.fc1.b");
        lm_ggml_tensor * ff2_w = W(lp + ".ffn.fc2.w");
        lm_ggml_tensor * ff2_b = W(lp + ".ffn.fc2.b");
        if (ff1_w == nullptr || ff1_b == nullptr || ff2_w == nullptr || ff2_b == nullptr) return false;
        lm_ggml_tensor * ff = codec_op_linear_tc(ctx_eval, hs, ff1_w, ff1_b);
        if (ff == nullptr) return false;
        ff = lm_ggml_gelu_erf(ctx_eval, ff);
        ff = codec_op_linear_tc(ctx_eval, ff, ff2_w, ff2_b);
        if (ff == nullptr) return false;
        hs = lm_ggml_add(ctx_eval, hs, ff);

        lm_ggml_tensor * ffn_ln_w = W(lp + ".ffn_ln.w");
        lm_ggml_tensor * ffn_ln_b = W(lp + ".ffn_ln.b");
        if (ffn_ln_w == nullptr || ffn_ln_b == nullptr) return false;
        hs = codec_op_layer_norm_tc(ctx_eval, hs, p->hubert_ln_eps, ffn_ln_w, ffn_ln_b);
        if (hs == nullptr) return false;
    }
    // Semantic encoder conv stack
    lm_ggml_tensor * sem_init_w = W("neucodec.encode.semantic_encoder.initial_conv.w");
    lm_ggml_tensor * sem1_w = W("neucodec.encode.semantic_encoder.residual.1.w");
    lm_ggml_tensor * sem1_b = W("neucodec.encode.semantic_encoder.residual.1.b");
    lm_ggml_tensor * sem2_w = W("neucodec.encode.semantic_encoder.residual.3.w");
    lm_ggml_tensor * sem2_b = W("neucodec.encode.semantic_encoder.residual.3.b");
    lm_ggml_tensor * sem_out_w = W("neucodec.encode.semantic_encoder.final_conv.w");
    if (sem_init_w == nullptr || sem1_w == nullptr || sem1_b == nullptr ||
        sem2_w == nullptr || sem2_b == nullptr || sem_out_w == nullptr) {
        return false;
    }

    lm_ggml_tensor * sem_tc = codec_conv1d(ctx_eval, hs, sem_init_w, nullptr, 1, 1, 1);
    if (sem_tc == nullptr) return false;
    sem_tc = lm_ggml_relu(ctx_eval, sem_tc);
    // SemanticEncoder uses inplace ReLU in residual path.
    lm_ggml_tensor * sem_res = sem_tc;
    sem_tc = codec_conv1d(ctx_eval, sem_tc, sem1_w, sem1_b, 1, 1, 1);
    sem_tc = lm_ggml_relu(ctx_eval, sem_tc);
    sem_tc = codec_conv1d(ctx_eval, sem_tc, sem2_w, sem2_b, 1, 1, 1);
    sem_tc = lm_ggml_add(ctx_eval, sem_tc, sem_res);
    sem_tc = codec_conv1d(ctx_eval, sem_tc, sem_out_w, nullptr, 1, 1, 1);
    if (sem_tc == nullptr) return false;

    // match lengths
    if (sem_tc->ne[0] != fsq_tc->ne[0]) {
        const int32_t min_t = (int32_t) std::min(sem_tc->ne[0], fsq_tc->ne[0]);
        sem_tc = codec_op_crop_1d(ctx_eval, sem_tc, 0, (int32_t) sem_tc->ne[0] - min_t);
        fsq_tc = codec_op_crop_1d(ctx_eval, fsq_tc, 0, (int32_t) fsq_tc->ne[0] - min_t);
    }

    lm_ggml_tensor * concat = lm_ggml_concat(ctx_eval, sem_tc, fsq_tc, 1);
    lm_ggml_tensor * fc_w = W("neucodec.encode.fc_prior.w");
    lm_ggml_tensor * fc_b = W("neucodec.encode.fc_prior.b");
    if (fc_w == nullptr || fc_b == nullptr) return false;
    lm_ggml_tensor * prior_tc = codec_op_linear_tc(ctx_eval, concat, fc_w, fc_b);
    if (prior_tc == nullptr) return false;
    // FSQ project_in
    lm_ggml_tensor * proj_w = W("neucodec.encode.quant.project_in.w");
    lm_ggml_tensor * proj_b = W("neucodec.encode.quant.project_in.b");
    if (proj_w == nullptr || proj_b == nullptr) return false;
    lm_ggml_tensor * z_tc = codec_op_linear_tc(ctx_eval, prior_tc, proj_w, proj_b);
    if (z_tc == nullptr) return false;
    // FSQ bound + quantize (vector-quantize-pytorch FSQ)
    const float eps = 1e-3f;
    const float half_l = (3.0f * (1.0f + eps)) / 2.0f;
    const float offset = 0.5f;
    const float shift = std::atanh(offset / half_l);
    const float half_width = 2.0f;

    // ResidualFSQ initializes residual with bound(x), then FSQ quantize()
    // applies bound() again before rounding.
    lm_ggml_tensor * z_bound = lm_ggml_tanh(ctx_eval, lm_ggml_scale_bias(ctx_eval, z_tc, 1.0f, shift));
    z_bound = lm_ggml_scale_bias(ctx_eval, z_bound, half_l, -offset);
    z_bound = lm_ggml_tanh(ctx_eval, lm_ggml_scale_bias(ctx_eval, z_bound, 1.0f, shift));
    z_bound = lm_ggml_scale_bias(ctx_eval, z_bound, half_l, -offset);
    lm_ggml_tensor * zq = lm_ggml_scale(ctx_eval, lm_ggml_round(ctx_eval, z_bound), 1.0f / half_width);

    // indices
    lm_ggml_tensor * basis = lm_ggml_new_tensor_1d(ctx_eval, LM_GGML_TYPE_F32, p->codebook_dim);
    codec_neu_set_enc_name(basis, "neucodec.encode.fsq.basis");
    lm_ggml_tensor * z_scaled = lm_ggml_scale_bias(ctx_eval, zq, half_width, half_width);
    lm_ggml_tensor * basis_2d = lm_ggml_reshape_2d(ctx_eval, basis, 1, p->codebook_dim);
    lm_ggml_tensor * basis_rep = lm_ggml_repeat(ctx_eval, basis_2d, z_scaled);
    lm_ggml_tensor * z_mul = lm_ggml_mul(ctx_eval, z_scaled, basis_rep);
    lm_ggml_tensor * z_ct = lm_ggml_cont(ctx_eval, lm_ggml_transpose(ctx_eval, z_mul));
    lm_ggml_tensor * idx_sum = lm_ggml_sum_rows(ctx_eval, z_ct);
    lm_ggml_tensor * idx_1d = lm_ggml_reshape_1d(ctx_eval, idx_sum, z_mul->ne[0]);
    lm_ggml_tensor * idx_2d = lm_ggml_reshape_2d(ctx_eval, idx_1d, (int32_t) z_mul->ne[0], 1);
    lm_ggml_tensor * out_t = lm_ggml_cont(ctx_eval, idx_2d);
    codec_neu_set_enc_name(out_t, "neucodec.encode.out");
    *out = out_t;
    return true;
}

static bool codec_neu_init_encode_build(
    const codec_neucodec * neu,
    int32_t n_in,
    int32_t n_in_sem,
    neucodec_encode_build * build,
    std::string * err) {

    if (neu == nullptr || build == nullptr || n_in <= 0 || n_in_sem <= 0) {
        if (err) *err = "invalid NeuCodec encode build arguments";
        return false;
    }
    build->n_in = n_in;
    build->n_in_sem = n_in_sem;
    build->n_q = neu->n_q;
    build->codebook_dim = neu->codebook_dim;
    build->codebook_size = neu->codebook_size;
    build->encoder_type = neu->encoder_type;
    build->hubert_hidden = neu->hubert_hidden;
    build->hubert_heads = neu->hubert_heads;
    build->hubert_intermediate = neu->hubert_intermediate;
    build->hubert_layers = neu->hubert_layers;
    build->hubert_pos_k = neu->hubert_pos_k;
    build->hubert_pos_groups = neu->hubert_pos_groups;
    build->hubert_ln_eps = neu->hubert_ln_eps;
    build->hubert_feat_layers = neu->hubert_feat_layers;
    for (int i = 0; i < neu->hubert_feat_layers; ++i) {
        build->hubert_conv_dim[i] = neu->hubert_conv_dim[i];
        build->hubert_conv_kernel[i] = neu->hubert_conv_kernel[i];
        build->hubert_conv_stride[i] = neu->hubert_conv_stride[i];
    }
    build->local_window = 300;
    build->local_down_window = 1500;
    build->down_attn = &neu->distill_attn_down;
    build->local_attn = &neu->distill_attn_local;
    return true;
}

static enum codec_status codec_neu_decode_graph(
    struct codec_context * ctx,
    const struct codec_token_buffer * tokens,
    int32_t use_n_q,
    struct codec_pcm_buffer * out_pcm) {

    codec_neucodec & neu = *static_cast<codec_neucodec *>(ctx->model->impl);
    if (tokens == nullptr || tokens->data == nullptr || tokens->n_frames <= 0 || tokens->n_q < use_n_q) {
        codec_context_set_error(ctx, "invalid NeuCodec token buffer");
        return CODEC_STATUS_INVALID_ARG;
    }

    const int32_t t = tokens->n_frames;
    const int32_t q = use_n_q;
    const int32_t hop = std::max(1, neu.hop_size);
    codec_graph_eval_guard eval_guard(ctx);
    std::string err;

    neucodec_decode_build build = {};
    if (!codec_neu_init_decode_build(ctx, &neu, t, q, &build, &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    codec_graph_cache_entry * entry = nullptr;
    err.clear();
    if (!codec_graph_cache_get_or_build(
            ctx,
            { CODEC_GRAPH_NEUCODEC_DECODE, /*n_frames=*/t, /*n_q=*/q, /*hop=*/hop, /*n_in=*/0, /*latent_dim=*/0 },
            codec_neu_build_decode,
            &build,
            sizeof(build),
            &entry,
            &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    lm_ggml_tensor * t_tok = codec_graph_get_tensor(ctx, entry, codec_neu_name_tok().c_str());
    lm_ggml_tensor * t_out = codec_graph_get_tensor(ctx, entry, "neucodec.decode.head.out");
    if (t_tok == nullptr || t_out == nullptr) {
        codec_context_set_error(ctx, "cached NeuCodec decode graph is invalid");
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    if (!codec_graph_prepare_io(ctx, entry, &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    std::vector<int32_t> tok_i32((size_t) t * (size_t) q, 0);
    for (int32_t ti = 0; ti < t; ++ti) {
        for (int32_t qi = 0; qi < q; ++qi) {
            int32_t tok = tokens->data[(size_t) ti * (size_t) tokens->n_q + (size_t) qi];
            tok = std::max(0, std::min(build.codebook_size - 1, tok));
            tok_i32[(size_t) qi * (size_t) t + (size_t) ti] = tok;
        }
    }

    if (!codec_runtime_write_tensor(t_tok, tok_i32.data(), tok_i32.size() * sizeof(int32_t), &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    const int32_t n_threads = ctx->model->n_threads > 0 ? ctx->model->n_threads : 1;
    if (!codec_graph_compute(ctx, entry, n_threads, &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    std::vector<float> head((size_t) build.head_out_dim * (size_t) t, 0.0f);
    if (!codec_runtime_read_tensor(t_out, head.data(), head.size() * sizeof(float), &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    std::vector<float> window;
    lm_ggml_tensor * w_tensor = codec_model_get_tensor(ctx->model, codec_neu_name_istft_window());
    if (w_tensor != nullptr) {
        codec_tensor_as_vec_f32(w_tensor, &window);
    }

    std::vector<float> pcm_v;
    if (!codec_runtime_istft_from_head(head, build.head_out_dim, t, hop, w_tensor != nullptr ? &window : nullptr, false, -1, &pcm_v, &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }
    float * pcm = static_cast<float *>(std::malloc(pcm_v.size() * sizeof(float)));
    if (pcm == nullptr) {
        codec_context_set_error(ctx, "failed to allocate pcm output");
        return CODEC_STATUS_INTERNAL_ERROR;
    }
    std::memcpy(pcm, pcm_v.data(), pcm_v.size() * sizeof(float));

    codec_pcm_buffer_reset(out_pcm);
    out_pcm->data = pcm;
    out_pcm->n_samples = (int32_t) pcm_v.size();
    out_pcm->sample_rate = neu.sample_rate;
    out_pcm->n_channels = 1;

    return CODEC_STATUS_SUCCESS;
}

enum codec_status codec_neucodec_init(struct codec_model * model) {
    if (model == nullptr || model->impl == nullptr || model->gguf == nullptr) {
        return CODEC_STATUS_INVALID_ARG;
    }

    codec_neucodec & neu = *static_cast<codec_neucodec *>(model->impl);
    neu.sample_rate = codec_read_i32_kv(model->gguf, "codec.sample_rate", neu.sample_rate);
    neu.encode_sample_rate = codec_read_i32_kv(model->gguf, "codec.encode_sample_rate", neu.encode_sample_rate);
    neu.hop_size = codec_read_i32_kv(model->gguf, "codec.hop_size", neu.hop_size);
    neu.n_fft = codec_read_i32_kv(model->gguf, "codec.n_fft", neu.n_fft);
    neu.n_q = codec_read_i32_kv(model->gguf, "codec.n_q", neu.n_q);
    neu.codebook_size = codec_read_i32_kv(model->gguf, "codec.codebook_size", neu.codebook_size);
    neu.codebook_dim = codec_read_i32_kv(model->gguf, "codec.codebook_dim", neu.codebook_dim);
    neu.latent_dim = codec_read_i32_kv(model->gguf, "codec.latent_dim", neu.latent_dim);
    neu.hidden_dim = codec_read_i32_kv(model->gguf, "neucodec.hidden_dim", neu.hidden_dim);
    neu.vq_dim = codec_read_i32_kv(model->gguf, "neucodec.vq_dim", neu.vq_dim);
    neu.num_layers = codec_read_i32_kv(model->gguf, "neucodec.num_layers", neu.num_layers);
    neu.num_heads = codec_read_i32_kv(model->gguf, "neucodec.num_heads", neu.num_heads);
    neu.head_dim = codec_read_i32_kv(model->gguf, "neucodec.head_dim", neu.head_dim);
    neu.rope_theta = codec_read_f32_kv(model->gguf, "neucodec.rope_theta", neu.rope_theta);
    neu.has_encoder = codec_read_bool_kv(model->gguf, "codec.has_encoder", neu.has_encoder);
    neu.has_decoder = codec_read_bool_kv(model->gguf, "codec.has_decoder", neu.has_decoder);
    neu.encoder_type = (model->arch == CODEC_ARCH_DISTILL_NEUCODEC) ? 1 : 0;
    const int key = lm_gguf_find_key(model->gguf, "neucodec.encoder_type");
    if (key >= 0 && lm_gguf_get_kv_type(model->gguf, key) == LM_GGUF_TYPE_STRING) {
        const char * val = lm_gguf_get_val_str(model->gguf, key);
        if (val != nullptr && std::strcmp(val, "distill") == 0) {
            neu.encoder_type = 1;
        } else {
            neu.encoder_type = 0;
        }
    }
    if (model->arch == CODEC_ARCH_DISTILL_NEUCODEC && neu.encoder_type != 1) {
        return CODEC_STATUS_INVALID_STATE;
    }

    if (neu.encoder_type == 1) {
        neu.hubert_hidden = codec_read_i32_kv(model->gguf, "neucodec.hubert.hidden_size", neu.hubert_hidden);
        neu.hubert_heads = codec_read_i32_kv(model->gguf, "neucodec.hubert.num_heads", neu.hubert_heads);
        neu.hubert_intermediate = codec_read_i32_kv(model->gguf, "neucodec.hubert.intermediate_size", neu.hubert_intermediate);
        neu.hubert_layers = codec_read_i32_kv(model->gguf, "neucodec.hubert.num_layers", neu.hubert_layers);
        neu.hubert_pos_k = codec_read_i32_kv(model->gguf, "neucodec.hubert.num_conv_pos_embeddings", neu.hubert_pos_k);
        neu.hubert_pos_groups = codec_read_i32_kv(model->gguf, "neucodec.hubert.num_conv_pos_embedding_groups", neu.hubert_pos_groups);
        neu.hubert_ln_eps = codec_read_f32_kv(model->gguf, "neucodec.hubert.layer_norm_eps", neu.hubert_ln_eps);
        codec_read_i32_array_kv(model->gguf, "neucodec.hubert.conv_dim", neu.hubert_conv_dim, neu.hubert_feat_layers);
        codec_read_i32_array_kv(model->gguf, "neucodec.hubert.conv_kernel", neu.hubert_conv_kernel, neu.hubert_feat_layers);
        codec_read_i32_array_kv(model->gguf, "neucodec.hubert.conv_stride", neu.hubert_conv_stride, neu.hubert_feat_layers);
    }

    model->sample_rate = neu.sample_rate;
    model->encode_sample_rate = neu.encode_sample_rate;
    model->has_encoder = neu.has_encoder;
    model->has_decoder = neu.has_decoder;
    model->hop_size = neu.hop_size;
    model->n_q = neu.n_q;
    model->codebook_size = neu.codebook_size;
    model->latent_dim = neu.latent_dim;
    model->n_fft = neu.n_fft;
    model->win_length = neu.n_fft;

    return CODEC_STATUS_SUCCESS;
}

enum codec_status codec_neucodec_decode(
    struct codec_context * ctx,
    const struct codec_token_buffer * tokens,
    struct codec_pcm_buffer * out_pcm,
    struct codec_decode_params params) {

    codec_neucodec & neu = *static_cast<codec_neucodec *>(ctx->model->impl);
    if (!neu.has_decoder) {
        codec_context_set_error(ctx, "model metadata indicates no decoder");
        return CODEC_STATUS_INVALID_STATE;
    }

    const int32_t model_n_q = std::max(1, neu.n_q);
    const int32_t use_n_q = params.n_q == 0 ? model_n_q : params.n_q;
    if (params.n_q < 0 || use_n_q < 1 || use_n_q > model_n_q) {
        codec_context_set_error(ctx, "NeuCodec decode n_q must be 0 or in [1, model_n_q]");
        return CODEC_STATUS_INVALID_ARG;
    }

    return codec_neu_decode_graph(ctx, tokens, use_n_q, out_pcm);
}

static enum codec_status codec_neu_encode_graph(
    codec_context * ctx,
    const std::vector<float> & pcm,
    struct codec_token_buffer * out_tokens) {

    codec_neucodec & neu = *static_cast<codec_neucodec *>(ctx->model->impl);
    if (pcm.empty()) {
        codec_context_set_error(ctx, "invalid NeuCodec PCM input");
        return CODEC_STATUS_INVALID_ARG;
    }
    if (neu.encoder_type != 1) {
        codec_context_set_error(ctx, "NeuCodec encoder_type not supported (only distill implemented)");
        return CODEC_STATUS_NOT_SUPPORTED;
    }

    const int32_t n_in = (int32_t) pcm.size();
    const int32_t pad_for_wav = 320 - (n_in % 320);
    const int32_t n_in_pad = n_in + pad_for_wav;
    const int32_t n_in_sem = n_in_pad + 320;

    std::vector<float> pcm_pad((size_t) n_in_pad, 0.0f);
    std::memcpy(pcm_pad.data(), pcm.data(), pcm.size() * sizeof(float));
    std::vector<float> sem_pad((size_t) n_in_sem, 0.0f);
    std::memcpy(sem_pad.data() + 160, pcm_pad.data(), pcm_pad.size() * sizeof(float));

    codec_graph_eval_guard eval_guard(ctx);
    std::string err;

    neucodec_encode_build build = {};
    if (!codec_neu_init_encode_build(&neu, n_in_pad, n_in_sem, &build, &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }
    build.model = ctx->model;

    codec_graph_cache_entry * entry = nullptr;
    if (!codec_graph_cache_get_or_build(
            ctx,
            { CODEC_GRAPH_NEUCODEC_ENCODE, /*n_frames=*/0, /*n_q=*/build.n_q, /*hop=*/0, /*n_in=*/n_in_pad, /*latent_dim=*/0 },
            codec_neu_build_encode,
            &build,
            sizeof(build),
            &entry,
            &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    lm_ggml_tensor * t_pcm = codec_graph_get_tensor(ctx, entry, codec_neu_encode_name("neucodec.encode.pcm").c_str());
    lm_ggml_tensor * t_sem = codec_graph_get_tensor(ctx, entry, codec_neu_encode_name("neucodec.encode.sem").c_str());
    lm_ggml_tensor * t_out = codec_graph_get_tensor(ctx, entry, codec_neu_encode_name("neucodec.encode.out").c_str());
    lm_ggml_tensor * t_basis = codec_graph_get_tensor(ctx, entry, codec_neu_encode_name("neucodec.encode.fsq.basis").c_str());
    if (t_pcm == nullptr || t_sem == nullptr || t_out == nullptr || t_basis == nullptr) {
        codec_context_set_error(ctx, "cached NeuCodec encode graph is invalid");
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    if (!codec_graph_prepare_io(ctx, entry, &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    if (!codec_runtime_write_tensor(t_pcm, pcm_pad.data(), pcm_pad.size() * sizeof(float), &err) ||
        !codec_runtime_write_tensor(t_sem, sem_pad.data(), sem_pad.size() * sizeof(float), &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    const float basis_vals[8] = { 1, 4, 16, 64, 256, 1024, 4096, 16384 };
    if (!codec_runtime_write_tensor(t_basis, basis_vals, sizeof(basis_vals), &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    // Local-attention score-bias masks (causal + block window + per-head
    // rel-pos bias).  Built CPU-side from the cached `attn_params`, then
    // written to graph leaves before compute.
    auto write_local_attn_mask = [&](const char * tname,
                                      const codec_local_attn_params & ap) -> bool {
        lm_ggml_tensor * t_mask = codec_graph_get_tensor(ctx, entry, tname);
        if (t_mask == nullptr) return true;   // distill stage not built (e.g. legacy graph)
        const int32_t t_len = (int32_t) t_mask->ne[0];
        std::vector<float> mask((size_t) t_len * (size_t) t_len * (size_t) ap.heads, 0.0f);
        codec_local_attn_fill_mask(&ap, t_len, mask.data());
        return codec_runtime_write_tensor(t_mask, mask.data(), mask.size() * sizeof(float), &err);
    };
    if (!write_local_attn_mask("neucodec.encode.distill.down_mask",  neu.distill_attn_down) ||
        !write_local_attn_mask("neucodec.encode.distill.local_mask", neu.distill_attn_local)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    if (!codec_graph_compute(ctx, entry, ctx->model->n_threads, &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    if (t_out->type != LM_GGML_TYPE_F32 || t_out->ne[1] != 1) {
        codec_context_set_error(ctx, "unexpected NeuCodec token tensor shape/type");
        return CODEC_STATUS_INTERNAL_ERROR;
    }
    const int32_t n_frames = (int32_t) t_out->ne[0];
    std::vector<float> out_f((size_t) n_frames, 0.0f);
    if (!codec_runtime_read_tensor(t_out, out_f.data(), out_f.size() * sizeof(float), &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }
    std::vector<int32_t> tok((size_t) n_frames, 0);
    for (int32_t ti = 0; ti < n_frames; ++ti) {
        float v = out_f[(size_t) ti];
        if (!std::isfinite(v)) v = 0.0f;
        int32_t idx = (int32_t) std::lrintf(v);
        tok[(size_t) ti] = std::max(0, std::min(neu.codebook_size - 1, idx));
    }

    int32_t * data = static_cast<int32_t *>(std::malloc(tok.size() * sizeof(int32_t)));
    if (data == nullptr) {
        codec_context_set_error(ctx, "failed to allocate token output");
        return CODEC_STATUS_INTERNAL_ERROR;
    }
    std::memcpy(data, tok.data(), tok.size() * sizeof(int32_t));
    codec_token_buffer_reset(out_tokens);
    out_tokens->data = data;
    out_tokens->n_tokens = n_frames;
    out_tokens->n_frames = n_frames;
    out_tokens->n_q = 1;
    out_tokens->codebook_size = neu.codebook_size;
    out_tokens->sample_rate = neu.encode_sample_rate;
    out_tokens->hop_size = neu.hop_size;
    return CODEC_STATUS_SUCCESS;
}

enum codec_status codec_neucodec_encode(
    struct codec_context * ctx,
    const std::vector<float> & pcm,
    struct codec_token_buffer * out_tokens,
    struct codec_latent_buffer * out_latent,
    struct codec_encode_params params) {

    codec_neucodec & neu = *static_cast<codec_neucodec *>(ctx->model->impl);
    if (!neu.has_encoder) {
        codec_context_set_error(ctx, "model metadata indicates no encoder");
        return CODEC_STATUS_INVALID_STATE;
    }

    (void) out_latent;
    if (params.n_q != 0 && params.n_q != 1) {
        codec_context_set_error(ctx, "NeuCodec encode n_q must be 0 or 1");
        return CODEC_STATUS_INVALID_ARG;
    }

    if (neu.encoder_type == 1 && !neu.distill_bias_ready) {
        std::string err;
        const int32_t down_max_dist = 3000;
        const int32_t local_max_dist = 600;
        if (!codec_neu_build_dynamic_pos_bias(ctx->model, "neucodec.encode.distill.codec_encoder.en_encoder.down_trans.trans.dynamic_pos_bias",
                                              down_max_dist, &neu.distill_bias_down, &err) ||
            !codec_neu_build_dynamic_pos_bias(ctx->model, "neucodec.encode.distill.codec_encoder.en_encoder.local_trans.dynamic_pos_bias",
                                              local_max_dist, &neu.distill_bias_local, &err)) {
            codec_context_set_error(ctx, err);
            return CODEC_STATUS_INTERNAL_ERROR;
        }
        neu.distill_attn_down.bias = neu.distill_bias_down.data();
        neu.distill_attn_down.heads = 6;
        neu.distill_attn_down.head_dim = 128;
        // local_attention uses look_backward=1, so effective causal context is 2 * window_size.
        neu.distill_attn_down.window = 3000;
        neu.distill_attn_down.max_dist = down_max_dist;

        neu.distill_attn_local.bias = neu.distill_bias_local.data();
        neu.distill_attn_local.heads = 6;
        neu.distill_attn_local.head_dim = 128;
        neu.distill_attn_local.window = 600;
        neu.distill_attn_local.max_dist = local_max_dist;

        neu.distill_bias_ready = true;
    }

    return codec_neu_encode_graph(ctx, pcm, out_tokens);
}

static void * codec_neu_create_impl() {
    return new (std::nothrow) codec_neucodec();
}

static void codec_neu_destroy_impl(void * ptr) {
    delete static_cast<codec_neucodec *>(ptr);
}

static enum codec_status codec_neu_decode_wrap(
    struct codec_context * ctx,
    const struct codec_token_buffer * tokens,
    struct codec_pcm_buffer * out_pcm,
    struct codec_decode_params params) {

    return codec_neucodec_decode(ctx, tokens, out_pcm, params);
}

static enum codec_status codec_neu_encode_wrap(
    struct codec_context * ctx,
    const std::vector<float> & pcm,
    struct codec_token_buffer * out_tokens,
    struct codec_latent_buffer * out_latent,
    struct codec_encode_params params) {

    return codec_neucodec_encode(ctx, pcm, out_tokens, out_latent, params);
}

const struct codec_model_vtable * codec_neucodec_vtable() {
    static const codec_model_vtable vtable = {
        CODEC_ARCH_NEUCODEC,
        "neucodec",
        codec_neu_create_impl,
        codec_neu_destroy_impl,
        codec_neucodec_init,
        codec_graph_size_exact,
        codec_neu_encode_wrap,
        codec_neu_decode_wrap,
    };
    return &vtable;
}

const struct codec_model_vtable * codec_distill_neucodec_vtable() {
    static const codec_model_vtable vtable = {
        CODEC_ARCH_DISTILL_NEUCODEC,
        "distill_neucodec",
        codec_neu_create_impl,
        codec_neu_destroy_impl,
        codec_neucodec_init,
        codec_graph_size_exact,
        codec_neu_encode_wrap,
        codec_neu_decode_wrap,
    };
    return &vtable;
}
