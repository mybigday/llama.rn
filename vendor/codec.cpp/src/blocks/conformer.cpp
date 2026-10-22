#include "conformer.h"

#include "../ops/conv1d.h"
#include "../ops/ggml_ops.h"
#include "../ops/lm_attn.h"
#include "../runtime/graph.h"
#include "../runtime/tensor_utils.h"

#include <cmath>
#include <string>

ggml_tensor * codec_block_conformer_embed(
        ggml_context * ctx, const codec_model * model, const std::string & prefix,
        const codec_conformer_params & p, ggml_tensor * x_ct) {
    auto W = [&](const std::string & n) { return codec_graph_weight(ctx, model, prefix + "." + n); };
    ggml_tensor * lw = W("lin.w");
    ggml_tensor * nw = W("ln.w");
    if (x_ct == nullptr || lw == nullptr || nw == nullptr) return nullptr;
    ggml_tensor * h = codec_op_linear(ctx, x_ct, lw, W("lin.b"));
    if (h == nullptr) return nullptr;
    h = ggml_cont(ctx, ggml_transpose(ctx, h));
    h = codec_op_layer_norm_tc(ctx, h, p.embed_eps, nw, W("ln.b"));
    if (h == nullptr) return nullptr;
    return ggml_scale(ctx, h, std::sqrt((float) p.hidden));
}

ggml_tensor * codec_block_conformer_layer(
        ggml_context * ctx, const codec_model * model, const std::string & prefix,
        const codec_conformer_params & p, ggml_tensor * x_tc, ggml_tensor * pos_emb) {
    auto W = [&](const std::string & n) { return codec_graph_weight(ctx, model, prefix + "." + n); };
    ggml_tensor * nmw = W("norm_mha.w");
    ggml_tensor * nfw = W("norm_ff.w");
    ggml_tensor * qw = W("attn.q.w");
    ggml_tensor * pw = W("attn.pos.w");
    ggml_tensor * pbu = W("attn.pbu");
    ggml_tensor * pbv = W("attn.pbv");
    if (x_tc == nullptr || pos_emb == nullptr || nmw == nullptr || nfw == nullptr || qw == nullptr ||
        pw == nullptr || pbu == nullptr || pbv == nullptr || p.n_heads <= 0 || p.hidden % p.n_heads != 0) {
        return nullptr;
    }
    const int32_t T = (int32_t) x_tc->ne[0];
    const int32_t hd = p.hidden / p.n_heads;

    ggml_tensor * h_tc = codec_op_layer_norm_tc(ctx, x_tc, p.block_eps, nmw, W("norm_mha.b"));
    if (h_tc == nullptr) return nullptr;
    ggml_tensor * h_ct = ggml_cont(ctx, ggml_transpose(ctx, h_tc));
    ggml_tensor * q_ct = codec_op_linear(ctx, h_ct, qw, W("attn.q.b"));
    ggml_tensor * k_ct = codec_op_linear(ctx, h_ct, W("attn.k.w"), W("attn.k.b"));
    ggml_tensor * v_ct = codec_op_linear(ctx, h_ct, W("attn.v.w"), W("attn.v.b"));
    ggml_tensor * p_ct = codec_op_linear(ctx, pos_emb, pw, nullptr);
    if (q_ct == nullptr || k_ct == nullptr || v_ct == nullptr || p_ct == nullptr) return nullptr;

    auto to_dth = [&](ggml_tensor * t, int32_t n) {
        return ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_3d(ctx, t, hd, p.n_heads, n), 0, 2, 1, 3));
    };
    codec_lm_attn_params ap = {};
    ap.scale = 1.0f / std::sqrt((float) hd);
    ap.causal = p.causal;
    ggml_tensor * a = codec_op_lm_attn_rel_pos_dth(ctx, to_dth(q_ct, T), to_dth(k_ct, T), to_dth(v_ct, T),
                                                   to_dth(p_ct, 2 * T - 1), pbu, pbv, &ap);
    if (a == nullptr) return nullptr;
    a = ggml_reshape_2d(ctx, ggml_cont(ctx, ggml_permute(ctx, a, 0, 2, 1, 3)), p.hidden, T);
    ggml_tensor * o = codec_op_linear(ctx, a, W("attn.o.w"), W("attn.o.b"));
    if (o == nullptr) return nullptr;
    x_tc = ggml_add(ctx, x_tc, ggml_cont(ctx, ggml_transpose(ctx, o)));

    ggml_tensor * f = codec_op_layer_norm_tc(ctx, x_tc, p.block_eps, nfw, W("norm_ff.b"));
    if (f == nullptr) return nullptr;
    f = codec_op_linear(ctx, ggml_cont(ctx, ggml_transpose(ctx, f)), W("ff.w1.w"), W("ff.w1.b"));
    if (f == nullptr) return nullptr;
    f = codec_op_linear(ctx, codec_op_unary(ctx, f, CODEC_UNARY_SILU), W("ff.w2.w"), W("ff.w2.b"));
    if (f == nullptr) return nullptr;
    return ggml_add(ctx, x_tc, ggml_cont(ctx, ggml_transpose(ctx, f)));
}

ggml_tensor * codec_block_conformer_stack(
        ggml_context * ctx, const codec_model * model, const std::string & prefix,
        const codec_conformer_params & p, ggml_tensor * x_tc, int32_t n_layers) {
    if (x_tc == nullptr) return nullptr;
    ggml_tensor * pe = codec_op_espnet_rel_pos_emb(ctx, (int32_t) x_tc->ne[0], p.hidden);
    if (pe == nullptr) return nullptr;
    for (int32_t i = 0; i < n_layers && x_tc != nullptr; ++i) {
        x_tc = codec_block_conformer_layer(ctx, model, prefix + "." + std::to_string(i), p, x_tc, pe);
    }
    return x_tc;
}

ggml_tensor * codec_block_pre_lookahead(
        ggml_context * ctx, const codec_model * model, const std::string & prefix, ggml_tensor * x_tc) {
    auto W = [&](const std::string & n) { return codec_graph_weight(ctx, model, prefix + "." + n); };
    ggml_tensor * c1w = W("cv1.w");
    ggml_tensor * c2w = W("cv2.w");
    if (x_tc == nullptr || c1w == nullptr || c2w == nullptr) return nullptr;
    ggml_tensor * h = codec_op_pad_1d(ctx, x_tc, 0, (int32_t) codec_conv1d_kernel_size(c1w, x_tc->ne[1]) - 1);
    h = codec_conv1d(ctx, h, c1w, W("cv1.b"), 1, 1, 0);
    if (h == nullptr) return nullptr;
    h = codec_op_leaky_relu(ctx, h, 0.01f);
    h = codec_op_pad_1d(ctx, h, (int32_t) codec_conv1d_kernel_size(c2w, h->ne[1]) - 1, 0);
    h = codec_conv1d(ctx, h, c2w, W("cv2.b"), 1, 1, 0);
    if (h == nullptr) return nullptr;
    return ggml_add(ctx, h, x_tc);
}

ggml_tensor * codec_block_upsample_conv(
        ggml_context * ctx, const codec_model * model, const std::string & prefix,
        ggml_tensor * x_tc, int32_t stride) {
    ggml_tensor * w = codec_graph_weight(ctx, model, prefix + ".w");
    ggml_tensor * b = codec_graph_weight(ctx, model, prefix + ".b");
    if (x_tc == nullptr || w == nullptr) return nullptr;
    ggml_tensor * up = ggml_interpolate(ctx, x_tc, x_tc->ne[0] * stride, x_tc->ne[1], x_tc->ne[2], x_tc->ne[3],
                                        GGML_SCALE_MODE_NEAREST);
    up = codec_op_pad_1d(ctx, up, 2 * stride, 0);
    return codec_conv1d(ctx, up, w, b, 1, 1, 0);
}
