#include "models.h"

static constexpr int SPK_RES2NET_SCALE = 8; // enc_res2net_scale
static constexpr int SPK_DILATIONS[3]  = { 2, 3, 4 }; // enc_dilations[1..3]

// conv1d, kernel K, padding "same" (reflect), dilation d
// x: [C, T] (ne[0]=C, ne[1]=T) -> [out_c, T]
lm_ggml_tensor * clip_graph_qwen3tts_spkenc::conv1d_same(lm_ggml_tensor * x, lm_ggml_tensor * w, lm_ggml_tensor * b, int dilation) const {
    const int K   = (int) w->ne[0];
    const int IC  = (int) w->ne[1];
    const int OC  = (int) w->ne[2];
    const int pad = ((K - 1) * dilation) / 2;

    // lm_ggml_pad_reflect_1d pads ne[0], so bring T onto ne[0] first, same layout as im2col wants
    lm_ggml_tensor * x_t = lm_ggml_cont(ctx0, lm_ggml_transpose(ctx0, x)); // [T, IC]
    if (pad > 0) {
        x_t = lm_ggml_pad_reflect_1d(ctx0, x_t, pad, pad); // [T + 2*pad, IC]
    }
    lm_ggml_tensor * x4d = lm_ggml_reshape_4d(ctx0, x_t, x_t->ne[0], IC, 1, 1);

    // dummy F32 kernel, im2col only reads its shape, so a quantized w does not assert
    lm_ggml_tensor * dummy = lm_ggml_new_tensor_4d(ctx0, LM_GGML_TYPE_F32, K, IC, 1, 1);

    lm_ggml_tensor * col = lm_ggml_im2col(ctx0, dummy, x4d, 1, 1, 0, 0, dilation, 1, false, LM_GGML_TYPE_F32);
    const int64_t T_out = col->ne[1];
    col = lm_ggml_reshape_2d(ctx0, col, (int64_t) K * IC, T_out);

    lm_ggml_tensor * w2d = lm_ggml_reshape_2d(ctx0, w, (int64_t) K * IC, OC);
    lm_ggml_tensor * y   = lm_ggml_mul_mat(ctx0, w2d, col); // [OC, T_out]
    lm_ggml_mul_mat_set_prec(y, LM_GGML_PREC_F32);

    lm_ggml_tensor * b2d = lm_ggml_reshape_2d(ctx0, b, OC, 1);
    y = lm_ggml_add(ctx0, y, b2d);
    return y;
}

// Res2Net: split channel axis into `scale` chunks, chain dilated conv1d branches
// x: [C, T] -> [C, T]
lm_ggml_tensor * clip_graph_qwen3tts_spkenc::res2net(lm_ggml_tensor * x, const clip_layer & layer, int dilation, int scale) const {
    const int64_t C  = x->ne[0];
    const int64_t T  = x->ne[1];
    const int64_t Cs = C / scale;

    std::vector<lm_ggml_tensor *> outs;
    outs.reserve(scale);

    auto chunk = [&](int i) -> lm_ggml_tensor * {
        return lm_ggml_view_2d(ctx0, x, Cs, T, x->nb[1], (size_t) i * Cs * x->nb[0]);
    };

    lm_ggml_tensor * prev = nullptr;
    for (int i = 0; i < scale; i++) {
        lm_ggml_tensor * c = lm_ggml_cont(ctx0, chunk(i));
        if (i == 0) {
            outs.push_back(c);
            continue;
        }
        lm_ggml_tensor * inp = (i >= 2) ? lm_ggml_add(ctx0, c, prev) : c;
        lm_ggml_tensor * y   = conv1d_same(inp, layer.res2_conv_w[i - 1], layer.res2_conv_b[i - 1], dilation);
        y                 = lm_ggml_relu(ctx0, y);
        outs.push_back(y);
        prev = y;
    }

    lm_ggml_tensor * acc = outs[0];
    for (int i = 1; i < scale; i++) {
        acc = lm_ggml_concat(ctx0, acc, outs[i], 0);
    }
    return acc;
}

// squeeze-and-excitation gate. x: [C, T] -> [C, T]
lm_ggml_tensor * clip_graph_qwen3tts_spkenc::se_block(lm_ggml_tensor * x, const clip_layer & layer) const {
    // temporal mean, keepdim: transpose so T is on ne[0], reduce, transpose back
    lm_ggml_tensor * x_t  = lm_ggml_cont(ctx0, lm_ggml_transpose(ctx0, x));    // [T, C]
    lm_ggml_tensor * mean = lm_ggml_mean(ctx0, x_t);                        // [1, C]
    mean               = lm_ggml_cont(ctx0, lm_ggml_transpose(ctx0, mean)); // [C, 1]

    lm_ggml_tensor * h = conv1d_same(mean, layer.se_conv1_w, layer.se_conv1_b, 1);
    h = lm_ggml_relu(ctx0, h);
    h = conv1d_same(h, layer.se_conv2_w, layer.se_conv2_b, 1);
    h = lm_ggml_sigmoid(ctx0, h); // [C, 1]

    return lm_ggml_mul(ctx0, x, h); // broadcast gate over T
}

// tdnn1 -> res2net -> tdnn2 -> se, plus residual. x: [C, T] -> [C, T]
lm_ggml_tensor * clip_graph_qwen3tts_spkenc::se_res2net_block(lm_ggml_tensor * x, const clip_layer & layer, int dilation, int scale) const {
    lm_ggml_tensor * residual = x;
    lm_ggml_tensor * h  = conv1d_same(x, layer.conv_pw1_w, layer.conv_pw1_b, 1); // tdnn1
    h = lm_ggml_relu(ctx0, h);
    h = res2net(h, layer, dilation, scale);
    h = conv1d_same(h, layer.conv_pw2_w, layer.conv_pw2_b, 1); // tdnn2
    h = lm_ggml_relu(ctx0, h);
    h = se_block(h, layer);
    return lm_ggml_add(ctx0, h, residual);
}

// attentive statistics pooling. x: [C, T] -> [2*C, 1]
lm_ggml_tensor * clip_graph_qwen3tts_spkenc::attentive_stats_pool(lm_ggml_tensor * x) const {
    const int64_t T = x->ne[1];

    // mean over T: [C, 1]
    lm_ggml_tensor * x_t  = lm_ggml_cont(ctx0, lm_ggml_transpose(ctx0, x));
    lm_ggml_tensor * mean = lm_ggml_mean(ctx0, x_t);
    mean               = lm_ggml_cont(ctx0, lm_ggml_transpose(ctx0, mean));

    // std over T: sqrt(clamp(mean((x - mean)^2), eps))
    lm_ggml_tensor * mean_rep = lm_ggml_repeat(ctx0, mean, x);
    lm_ggml_tensor * centered = lm_ggml_sub(ctx0, x, mean_rep);
    lm_ggml_tensor * var_t    = lm_ggml_cont(ctx0, lm_ggml_transpose(ctx0, lm_ggml_sqr(ctx0, centered)));
    lm_ggml_tensor * var      = lm_ggml_mean(ctx0, var_t);
    var                    = lm_ggml_cont(ctx0, lm_ggml_transpose(ctx0, var));
    var                    = lm_ggml_scale_bias(ctx0, var, 1.0f, 1e-12f);
    lm_ggml_tensor * std      = lm_ggml_sqrt(ctx0, var);

    // attention input: cat([x, mean, std]) along channel axis -> [3C, T]
    lm_ggml_tensor * std_rep = lm_ggml_repeat(ctx0, std, x);
    lm_ggml_tensor * cat     = lm_ggml_concat(ctx0, x, mean_rep, 0);
    cat                   = lm_ggml_concat(ctx0, cat, std_rep, 0);

    // attention TDNN (3C -> attn_c) + ReLU, tanh, then 1x1 conv (attn_c -> C)
    lm_ggml_tensor * a = conv1d_same(cat, model.spk_asp_tdnn_w, model.spk_asp_tdnn_b, 1);
    a = lm_ggml_relu(ctx0, a);
    a = lm_ggml_tanh(ctx0, a);
    a = conv1d_same(a, model.spk_asp_attn_w, model.spk_asp_attn_b, 1);

    // softmax over T
    lm_ggml_tensor * a_t = lm_ggml_cont(ctx0, lm_ggml_transpose(ctx0, a));   // [T, C]
    lm_ggml_tensor * w_t = lm_ggml_soft_max(ctx0, a_t);
    lm_ggml_tensor * w   = lm_ggml_cont(ctx0, lm_ggml_transpose(ctx0, w_t)); // [C, T]

    // weighted mean: sum(w * x) over T, multiply by T to undo lm_ggml_mean's 1/T scaling
    lm_ggml_tensor * wx     = lm_ggml_mul(ctx0, w, x);
    lm_ggml_tensor * wx_t   = lm_ggml_cont(ctx0, lm_ggml_transpose(ctx0, wx));
    lm_ggml_tensor * w_mean = lm_ggml_mean(ctx0, wx_t);
    w_mean = lm_ggml_scale(ctx0, w_mean, (float) T);
    w_mean = lm_ggml_cont(ctx0, lm_ggml_transpose(ctx0, w_mean)); // [C, 1]

    // weighted std: sum(w * (x - w_mean)^2) over T
    lm_ggml_tensor * w_mean_rep = lm_ggml_repeat(ctx0, w_mean, x);
    lm_ggml_tensor * dev        = lm_ggml_sub(ctx0, x, w_mean_rep);
    lm_ggml_tensor * w_var_in   = lm_ggml_mul(ctx0, w, lm_ggml_sqr(ctx0, dev));
    lm_ggml_tensor * w_var_t    = lm_ggml_cont(ctx0, lm_ggml_transpose(ctx0, w_var_in));
    lm_ggml_tensor * w_var      = lm_ggml_mean(ctx0, w_var_t);
    w_var                    = lm_ggml_scale(ctx0, w_var, (float) T);
    w_var                    = lm_ggml_cont(ctx0, lm_ggml_transpose(ctx0, w_var));
    w_var                    = lm_ggml_scale_bias(ctx0, w_var, 1.0f, 1e-12f);
    lm_ggml_tensor * w_std      = lm_ggml_sqrt(ctx0, w_var);

    return lm_ggml_concat(ctx0, w_mean, w_std, 0); // [2C, 1]
}

lm_ggml_cgraph * clip_graph_qwen3tts_spkenc::build() {
    // inp_raw: [T, n_mel, 1, 1], from mtmd_audio_preprocessor_qwen3tts_spk
    lm_ggml_tensor * inp = build_inp_raw(1);
    inp = lm_ggml_reshape_2d(ctx0, inp, inp->ne[0], inp->ne[1]);

    // this file's convention is [C, T]; the preprocessor delivers [T, C]
    lm_ggml_tensor * mel = lm_ggml_cont(ctx0, lm_ggml_transpose(ctx0, inp)); // [n_mel, T]
    cb(mel, "mel", -1);

    // frontend conv0 TDNN k=5, dilation=1: 128 -> 512
    lm_ggml_tensor * cur = conv1d_same(mel, model.conv1d_1_w, model.conv1d_1_b, 1);
    cur = lm_ggml_relu(ctx0, cur);
    cb(cur, "frontend", -1);

    // 3 SE-Res2Net blocks at dilations 2, 3, 4
    LM_GGML_ASSERT((int) model.layers.size() == 3);
    std::vector<lm_ggml_tensor *> blk_out(3);
    for (int il = 0; il < 3; il++) {
        cur = se_res2net_block(cur, model.layers[il], SPK_DILATIONS[il], SPK_RES2NET_SCALE);
        blk_out[il] = cur;
        cb(cur, "block_out", il);
    }

    // multi-layer feature aggregation: cat blk[0..2] then TDNN k=1 + ReLU
    lm_ggml_tensor * cat = lm_ggml_concat(ctx0, blk_out[0], blk_out[1], 0);
    cat = lm_ggml_concat(ctx0, cat, blk_out[2], 0); // [1536, T]
    lm_ggml_tensor * mfa = conv1d_same(cat, model.conv_out_w, model.conv_out_b, 1);
    mfa = lm_ggml_relu(ctx0, mfa);
    cb(mfa, "mfa", -1);

    // attentive statistics pooling: [1536, T] -> [3072, 1]
    lm_ggml_tensor * stats = attentive_stats_pool(mfa);
    cb(stats, "asp", -1);

    // final FC k=1: [3072, 1] -> [enc_dim, 1]
    lm_ggml_tensor * emb = conv1d_same(stats, model.mm_fc_w, model.mm_fc_b, 1);

    emb = lm_ggml_reshape_1d(ctx0, emb, emb->ne[0]);
    emb = lm_ggml_cont(ctx0, emb);
    cb(emb, "spk_embedding", -1);

    lm_ggml_build_forward_expand(gf, emb);
    return gf;
}
