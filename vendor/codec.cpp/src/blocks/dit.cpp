#include "dit.h"

#include "../ops/conv1d.h"

#include "cfm.h"
#include "../ops/ggml_ops.h"
#include "../ops/lm_attn.h"
#include "../ops/rope.h"
#include "../runtime/graph.h"
#include "../runtime/tensor_utils.h"

#include <cmath>
#include <string>

namespace {

// Grouped Conv1d (k, groups) on x [T, C] with left zero padding k - 1:
// im2col over a [T, C/groups, groups] view gives each group's columns, then
// one batched matmul against the weight viewed as [k * C/groups, C/groups, groups].
ggml_tensor * grouped_causal_conv(ggml_context * ctx, ggml_tensor * x_tc, ggml_tensor * w, ggml_tensor * b,
                                  int32_t groups) {
    const int64_t T = x_tc->ne[0];
    const int64_t C = x_tc->ne[1];
    const int64_t cg = C / groups;
    const int64_t k = codec_conv1d_kernel_size(w, cg);
    ggml_tensor * xp = codec_op_pad_1d(ctx, x_tc, (int32_t) (k - 1), 0);
    ggml_tensor * xg = ggml_reshape_3d(ctx, ggml_cont(ctx, xp), T + k - 1, cg, groups);
    ggml_tensor * kshape = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, cg);
    ggml_tensor * cols = ggml_im2col(ctx, kshape, xg, 1, 0, 0, 0, 1, 0, false, GGML_TYPE_F32);   // [k*cg, T, groups]
    ggml_tensor * wg = ggml_reshape_3d(ctx, w, k * cg, cg, groups);
    ggml_tensor * y = codec_mul_mat(ctx, wg, cols);                                          // [cg, T, groups]
    y = ggml_reshape_2d(ctx, ggml_cont(ctx, ggml_permute(ctx, y, 1, 0, 2, 3)), T, C);       // [T, C]
    return ggml_add(ctx, y, ggml_reshape_2d(ctx, b, 1, C));
}

// LayerNorm (no affine, eps 1e-6) * (1 + scale) + shift on x [C, T]; scale / shift are [C] views.
ggml_tensor * modulate(ggml_context * ctx, ggml_tensor * x_ct, ggml_tensor * scale, ggml_tensor * shift) {
    ggml_tensor * n = ggml_norm(ctx, x_ct, 1e-6f);
    n = ggml_add(ctx, n, ggml_mul(ctx, n, scale));
    return ggml_add(ctx, n, shift);
}

ggml_tensor * chunk(ggml_context * ctx, ggml_tensor * v, int64_t dim, int64_t i) {
    return ggml_view_1d(ctx, v, dim, (size_t) i * dim * ggml_element_size(v));
}

}  // namespace

ggml_tensor * codec_block_dit(
        ggml_context * ctx, const codec_model * model, const char * prefix,
        const codec_dit_params & p,
        ggml_tensor * x, ggml_tensor * mu, ggml_tensor * spks, ggml_tensor * cond, float t) {
    const std::string pre(prefix);
    auto W = [&](const std::string & n) { return codec_graph_weight(ctx, model, pre + "." + n); };
    if (ctx == nullptr || x == nullptr || mu == nullptr || spks == nullptr || cond == nullptr ||
        p.dim % p.n_heads != 0) {
        return nullptr;
    }
    const int64_t T = x->ne[0];
    const int64_t D = p.dim;
    const int32_t hd = p.dim / p.n_heads;

    ggml_tensor * te = codec_block_cfm_time_mlp(ctx, model, prefix, t, p.time_sin_dim, p.time_scale);   // [D]
    if (te == nullptr) return nullptr;
    ggml_tensor * te_act = ggml_reshape_2d(ctx, codec_op_unary(ctx, te, CODEC_UNARY_SILU), D, 1);

    // InputEmbedding: proj([x, cond, mu, spks]) + CausalConvPositionEmbedding.
    ggml_tensor * spks_rep = ggml_repeat(ctx, ggml_reshape_2d(ctx, spks, 1, spks->ne[0]), x);
    ggml_tensor * in = ggml_concat(ctx, ggml_concat(ctx, ggml_concat(ctx, x, cond, 1), mu, 1), spks_rep, 1);
    ggml_tensor * h = codec_op_linear(ctx, ggml_cont(ctx, ggml_transpose(ctx, in)), W("in.proj.w"), W("in.proj.b"));
    if (h == nullptr) return nullptr;                                                        // [D, T]
    ggml_tensor * h_tc = ggml_cont(ctx, ggml_transpose(ctx, h));
    ggml_tensor * cw1 = W("in.cp1.w");
    ggml_tensor * cw2 = W("in.cp2.w");
    if (cw1 == nullptr || cw2 == nullptr) return nullptr;
    ggml_tensor * pos = codec_op_unary(ctx, grouped_causal_conv(ctx, h_tc, cw1, W("in.cp1.b"), p.conv_groups), CODEC_UNARY_MISH);
    pos = codec_op_unary(ctx, grouped_causal_conv(ctx, pos, cw2, W("in.cp2.b"), p.conv_groups), CODEC_UNARY_MISH);
    h = ggml_add(ctx, h, ggml_cont(ctx, ggml_transpose(ctx, pos)));                         // [D, T]

    // x-transformers apply_rotary_pos_emb on (b, n, D) before the head split:
    // GPT-J pairs over the first rope_dim channels, the rest pass through.
    auto rope = [&](ggml_tensor * t_ct) {
        ggml_tensor * r = codec_op_rope(ctx, ggml_reshape_3d(ctx, t_ct, D, T, 1), p.rope_dim, 10000.0f, 1.0f,
                                        CODEC_ROPE_MODE_NORMAL);
        return r == nullptr ? nullptr : ggml_reshape_2d(ctx, ggml_cont(ctx, r), D, T);
    };

    for (int32_t i = 0; i < p.depth; ++i) {
        const std::string b = "blk." + std::to_string(i) + ".";
        ggml_tensor * ada = codec_op_linear(ctx, te_act, W(b + "ada.w"), W(b + "ada.b"));    // [6D, 1]
        if (ada == nullptr) return nullptr;
        ada = ggml_reshape_1d(ctx, ada, 6 * D);
        ggml_tensor * shift_msa = chunk(ctx, ada, D, 0);
        ggml_tensor * scale_msa = chunk(ctx, ada, D, 1);
        ggml_tensor * gate_msa = chunk(ctx, ada, D, 2);
        ggml_tensor * shift_mlp = chunk(ctx, ada, D, 3);
        ggml_tensor * scale_mlp = chunk(ctx, ada, D, 4);
        ggml_tensor * gate_mlp = chunk(ctx, ada, D, 5);

        ggml_tensor * nx = modulate(ctx, h, scale_msa, shift_msa);
        ggml_tensor * q = rope(codec_op_linear(ctx, nx, W(b + "attn.q.w"), W(b + "attn.q.b")));
        ggml_tensor * k = rope(codec_op_linear(ctx, nx, W(b + "attn.k.w"), W(b + "attn.k.b")));
        ggml_tensor * v = codec_op_linear(ctx, nx, W(b + "attn.v.w"), W(b + "attn.v.b"));
        if (q == nullptr || k == nullptr || v == nullptr) return nullptr;
        ggml_tensor * a = codec_op_attn_ct(ctx, q, k, v, hd, p.n_heads, 1.0f / std::sqrt((float) hd));   // [D, T]
        if (a == nullptr) return nullptr;
        a = codec_op_linear(ctx, a, W(b + "attn.o.w"), W(b + "attn.o.b"));
        if (a == nullptr) return nullptr;
        h = ggml_add(ctx, h, ggml_mul(ctx, a, gate_msa));

        ggml_tensor * f = modulate(ctx, h, scale_mlp, shift_mlp);
        f = codec_op_linear(ctx, f, W(b + "ff1.w"), W(b + "ff1.b"));
        if (f == nullptr) return nullptr;
        f = codec_op_linear(ctx, ggml_gelu(ctx, f), W(b + "ff2.w"), W(b + "ff2.b"));          // GELU (tanh)
        if (f == nullptr) return nullptr;
        h = ggml_add(ctx, h, ggml_mul(ctx, f, gate_mlp));
    }

    // AdaLayerNormZero_Final: (scale, shift) — note the order.
    ggml_tensor * fin = codec_op_linear(ctx, te_act, W("out_ada.w"), W("out_ada.b"));
    if (fin == nullptr) return nullptr;
    fin = ggml_reshape_1d(ctx, fin, 2 * D);
    h = modulate(ctx, h, chunk(ctx, fin, D, 0), chunk(ctx, fin, D, 1));
    ggml_tensor * out = codec_op_linear(ctx, h, W("proj_out.w"), W("proj_out.b"));             // [80, T]
    return out == nullptr ? nullptr : ggml_cont(ctx, ggml_transpose(ctx, out));
}
