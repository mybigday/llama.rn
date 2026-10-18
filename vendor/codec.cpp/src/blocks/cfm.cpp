#include "cfm.h"

#include "../ops/conv1d.h"
#include "../ops/convtr1d.h"
#include "../ops/ggml_ops.h"
#include "../runtime/graph.h"
#include "../runtime/tensor_utils.h"

#include <cmath>
#include <string>

namespace {

constexpr float kHalfPi = 0.5f * 3.14159265358979323846f;

// Block1D: Conv1d(k3, p1) → GroupNorm(groups) → Mish (non-causal Matcha).
ggml_tensor * block1d_gn(ggml_context * ctx, ggml_tensor * x, ggml_tensor * cw, ggml_tensor * cb,
                         ggml_tensor * nw, ggml_tensor * nb, int32_t groups) {
    ggml_tensor * y = codec_conv1d(ctx, x, cw, cb, 1, 1, 1);
    if (y == nullptr) return nullptr;
    y = codec_op_group_norm(ctx, y, groups, 1e-5f, nw, nb);
    if (y == nullptr) return nullptr;
    return codec_op_unary(ctx, y, CODEC_UNARY_MISH);
}

ggml_tensor * resnet(ggml_context * ctx, const codec_model * model, const std::string & pfx,
                     const codec_cfm_unet_params & p, ggml_tensor * x, ggml_tensor * t_emb) {
    auto W = [&](const std::string & n) { return codec_graph_weight(ctx, model, pfx + "." + n); };
    if (p.causal) {
        return codec_op_cfm_causal_resnet_block_tc(ctx, x, t_emb,
            W("b1.cv.w"), W("b1.cv.b"), W("b1.ln.w"), W("b1.ln.b"),
            W("b2.cv.w"), W("b2.cv.b"), W("b2.ln.w"), W("b2.ln.b"),
            W("mlp.w"), W("mlp.b"), W("res.w"), W("res.b"));
    }
    ggml_tensor * mlp_w = W("mlp.w");
    ggml_tensor * mlp_b = W("mlp.b");
    if (mlp_w == nullptr) return nullptr;
    ggml_tensor * h = block1d_gn(ctx, x, W("b1.cv.w"), W("b1.cv.b"), W("b1.ln.w"), W("b1.ln.b"), p.gn_groups);
    if (h == nullptr) return nullptr;
    ggml_tensor * tm = codec_op_unary(ctx, t_emb, CODEC_UNARY_MISH);
    ggml_tensor * tp = codec_mul_mat(ctx, mlp_w, ggml_reshape_2d(ctx, tm, tm->ne[0], 1));
    if (mlp_b != nullptr) tp = ggml_add(ctx, tp, ggml_reshape_2d(ctx, mlp_b, mlp_b->ne[0], 1));
    h = ggml_add(ctx, h, ggml_cont(ctx, ggml_transpose(ctx, tp)));
    h = block1d_gn(ctx, h, W("b2.cv.w"), W("b2.cv.b"), W("b2.ln.w"), W("b2.ln.b"), p.gn_groups);
    if (h == nullptr) return nullptr;
    ggml_tensor * res = codec_conv1d(ctx, x, W("res.w"), W("res.b"), 1, 1, 0);
    if (res == nullptr) return nullptr;
    return ggml_add(ctx, h, res);
}

ggml_tensor * transformers(ggml_context * ctx, const codec_model * model, const std::string & pfx,
                           const codec_cfm_unet_params & p, ggml_tensor * x) {
    if (p.n_transformers <= 0) return x;
    x = ggml_cont(ctx, ggml_transpose(ctx, x));  // [c, t] through the whole stack
    for (int32_t j = 0; j < p.n_transformers && x != nullptr; ++j) {
        const std::string tp = pfx + ".t." + std::to_string(j) + ".";
        auto W = [&](const std::string & n) { return codec_graph_weight(ctx, model, tp + n); };
        x = codec_op_basic_transformer_block_ct(ctx, x,
            W("norm1.w"), W("norm1.b"),
            W("attn.q.w"), W("attn.k.w"), W("attn.v.w"), W("attn.o.w"), W("attn.o.b"),
            W("norm3.w"), W("norm3.b"),
            W("ff.w1.w"), W("ff.w1.b"), W("ff.w2.w"), W("ff.w2.b"),
            p.head_dim, p.n_heads);
    }
    return x == nullptr ? nullptr : ggml_cont(ctx, ggml_transpose(ctx, x));
}

// Down/up resampling conv `x` of a level: the last level keeps the length
// (Conv1d k3 p1, or CausalConv1d k3); inner levels halve (Conv1d k3 s2 p1)
// or double (ConvTranspose1d k4 s2 p1).
ggml_tensor * level_conv(ggml_context * ctx, ggml_tensor * x, ggml_tensor * w, ggml_tensor * b,
                         bool causal, bool is_last, bool down) {
    if (w == nullptr) return nullptr;
    if (is_last) {
        return causal ? codec_conv1d_causal(ctx, x, w, b, 1, 1) : codec_conv1d(ctx, x, w, b, 1, 1, 1);
    }
    return down ? codec_conv1d(ctx, x, w, b, 2, 1, 1) : codec_convtr1d(ctx, x, w, b, 2, 1, 1);
}

ggml_tensor * crop_t(ggml_context * ctx, ggml_tensor * x, int64_t t) {
    if (x->ne[0] == t) return x;
    return ggml_cont(ctx, ggml_view_2d(ctx, x, t, x->ne[1], x->nb[1], 0));
}

}  // namespace

ggml_tensor * codec_block_cfm_time_mlp(
        ggml_context * ctx, const codec_model * model, const char * prefix,
        float t, int32_t sin_dim, float scale) {
    const std::string pfx = std::string(prefix) + ".t.";
    ggml_tensor * l1w = codec_graph_weight(ctx, model, pfx + "l1.w");
    ggml_tensor * l1b = codec_graph_weight(ctx, model, pfx + "l1.b");
    ggml_tensor * l2w = codec_graph_weight(ctx, model, pfx + "l2.w");
    ggml_tensor * l2b = codec_graph_weight(ctx, model, pfx + "l2.b");
    if (l1w == nullptr || l2w == nullptr) return nullptr;
    ggml_tensor * te = codec_op_sinusoidal_time_emb(ctx, t, sin_dim, scale);
    if (te == nullptr) return nullptr;
    te = codec_op_linear(ctx, ggml_reshape_2d(ctx, te, sin_dim, 1), l1w, l1b);
    if (te == nullptr) return nullptr;
    te = codec_op_linear(ctx, codec_op_unary(ctx, te, CODEC_UNARY_SILU), l2w, l2b);
    if (te == nullptr) return nullptr;
    return ggml_reshape_1d(ctx, te, te->ne[0]);
}

ggml_tensor * codec_block_cfm_unet(
        ggml_context * ctx, const codec_model * model, const char * prefix,
        const codec_cfm_unet_params & p,
        ggml_tensor * x, ggml_tensor * mu, ggml_tensor * spks, ggml_tensor * cond,
        ggml_tensor * t_emb) {
    if (ctx == nullptr || x == nullptr || mu == nullptr || spks == nullptr || cond == nullptr ||
        t_emb == nullptr || p.n_levels <= 0) {
        return nullptr;
    }
    const std::string pre(prefix);
    auto W = [&](const std::string & n) { return codec_graph_weight(ctx, model, pre + "." + n); };

    ggml_tensor * spks_rep = ggml_repeat(ctx, ggml_reshape_2d(ctx, spks, 1, p.out_channels), x);
    ggml_tensor * h = ggml_concat(ctx, ggml_concat(ctx, ggml_concat(ctx, x, mu, 1), spks_rep, 1), cond, 1);

    std::vector<ggml_tensor *> skips;
    for (int32_t i = 0; i < p.n_levels; ++i) {
        const std::string lp = pre + ".dn." + std::to_string(i);
        h = resnet(ctx, model, lp + ".r", p, h, t_emb);
        if (h == nullptr) return nullptr;
        h = transformers(ctx, model, lp, p, h);
        if (h == nullptr) return nullptr;
        skips.push_back(h);
        h = level_conv(ctx, h, W("dn." + std::to_string(i) + ".x.w"), W("dn." + std::to_string(i) + ".x.b"),
                       p.causal, i == p.n_levels - 1, /*down=*/true);
        if (h == nullptr) return nullptr;
    }
    for (int32_t i = 0; i < p.n_mid; ++i) {
        const std::string lp = pre + ".md." + std::to_string(i);
        h = resnet(ctx, model, lp + ".r", p, h, t_emb);
        if (h == nullptr) return nullptr;
        h = transformers(ctx, model, lp, p, h);
        if (h == nullptr) return nullptr;
    }
    for (int32_t i = 0; i < p.n_levels; ++i) {
        const std::string lp = pre + ".up." + std::to_string(i);
        ggml_tensor * skip = skips[(size_t) (p.n_levels - 1 - i)];
        h = ggml_concat(ctx, crop_t(ctx, h, skip->ne[0]), skip, 1);
        h = resnet(ctx, model, lp + ".r", p, h, t_emb);
        if (h == nullptr) return nullptr;
        h = transformers(ctx, model, lp, p, h);
        if (h == nullptr) return nullptr;
        h = level_conv(ctx, h, W("up." + std::to_string(i) + ".x.w"), W("up." + std::to_string(i) + ".x.b"),
                       p.causal, i == p.n_levels - 1, /*down=*/false);
        if (h == nullptr) return nullptr;
    }
    if (p.causal) {
        h = codec_op_causal_block1d_tc(ctx, h, W("final.cv.w"), W("final.cv.b"), W("final.ln.w"), W("final.ln.b"));
    } else {
        h = block1d_gn(ctx, h, W("final.cv.w"), W("final.cv.b"), W("final.ln.w"), W("final.ln.b"), p.gn_groups);
    }
    if (h == nullptr) return nullptr;
    return codec_conv1d(ctx, h, W("proj.w"), W("proj.b"), 1, 1, 0);
}

ggml_tensor * codec_block_cfm_solve(
        ggml_context * ctx, const codec_cfm_solver_params & p,
        ggml_tensor * z, ggml_tensor * mu, ggml_tensor * spks, ggml_tensor * cond,
        const codec_cfm_estimator_fn & estimator) {
    if (ctx == nullptr || z == nullptr || p.n_steps <= 0) return nullptr;
    ggml_tensor * mu0 = ggml_scale(ctx, mu, 0.0f);
    ggml_tensor * spks0 = ggml_scale(ctx, spks, 0.0f);
    ggml_tensor * cond0 = ggml_scale(ctx, cond, 0.0f);
    ggml_tensor * x = z;
    for (int32_t i = 0; i < p.n_steps; ++i) {
        const float lt = (float) i / (float) p.n_steps;
        const float lr = (float) (i + 1) / (float) p.n_steps;
        const float t = p.meanflow ? lt : 1.0f - std::cos(lt * kHalfPi);
        const float r = p.meanflow ? lr : 1.0f - std::cos(lr * kHalfPi);
        const float dt = r - t;
        ggml_tensor * v = estimator(x, mu, spks, cond, t, r);
        if (v == nullptr) return nullptr;
        if (p.meanflow || p.cfg_rate == 0.0f) {
            x = ggml_add(ctx, x, ggml_scale(ctx, v, dt));
            continue;
        }
        ggml_tensor * vu = estimator(x, mu0, spks0, cond0, t, r);
        if (vu == nullptr) return nullptr;
        ggml_tensor * g = ggml_sub(ctx, ggml_scale(ctx, v, 1.0f + p.cfg_rate), ggml_scale(ctx, vu, p.cfg_rate));
        x = ggml_add(ctx, x, ggml_scale(ctx, g, dt));
    }
    return x;
}
