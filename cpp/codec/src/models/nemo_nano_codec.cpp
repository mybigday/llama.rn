#include "nemo_nano_codec.h"

#include "../ops/conv1d.h"
#include "../ops/convtr1d.h"
#include "../ops/lm_ggml_ops.h"
#include "../runtime/graph.h"
#include "../runtime/tensor_utils.h"

#include <ggml-cpu.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static lm_ggml_tensor * nemo_conv1d_replicate(
    lm_ggml_context * ctx,
    lm_ggml_tensor * x,
    lm_ggml_tensor * w,
    lm_ggml_tensor * b,
    int32_t stride,
    int32_t dilation,
    int32_t padding) {

    if (ctx == nullptr || x == nullptr || w == nullptr || stride <= 0 || dilation <= 0 || padding < 0) {
        return nullptr;
    }

    lm_ggml_tensor * x_pad = codec_op_pad_1d_replicate(ctx, x, padding, padding);
    if (x_pad == nullptr) {
        return nullptr;
    }

    lm_ggml_tensor * w_f32 = codec_graph_cast_f32(ctx, w);
    lm_ggml_tensor * b_f32 = codec_graph_cast_f32(ctx, b);
    lm_ggml_tensor * im2col = lm_ggml_im2col(ctx, w_f32, x_pad, stride, 0, 0, 0, dilation, 0, false, LM_GGML_TYPE_F32);
    lm_ggml_tensor * im2col_2d = lm_ggml_reshape_2d(ctx, im2col, im2col->ne[0], (im2col->ne[2] * im2col->ne[1]));
    lm_ggml_tensor * w_2d = lm_ggml_reshape_2d(ctx, w_f32, (w_f32->ne[0] * w_f32->ne[1]), w_f32->ne[2]);
    lm_ggml_tensor * y = lm_ggml_mul_mat(ctx, im2col_2d, w_2d);
    y = lm_ggml_reshape_3d(ctx, y, im2col->ne[1], w_f32->ne[2], im2col->ne[2]);
    if (b_f32 != nullptr) {
        lm_ggml_tensor * b2 = lm_ggml_reshape_2d(ctx, b_f32, 1, y->ne[1]);
        y = lm_ggml_add(ctx, y, lm_ggml_repeat(ctx, b2, y));
    }
    return lm_ggml_cont(ctx, y);
}

static std::string nemo_enc_down_w_name(int32_t i) { return "nemo.enc.down." + std::to_string(i) + ".w"; }
static std::string nemo_enc_down_b_name(int32_t i) { return "nemo.enc.down." + std::to_string(i) + ".b"; }

static std::string nemo_enc_res_in_w(int32_t l, int32_t b, int32_t r) {
    return "nemo.enc.res.l" + std::to_string(l) + ".b" + std::to_string(b) + ".r" + std::to_string(r) + ".in.w";
}
static std::string nemo_enc_res_in_b(int32_t l, int32_t b, int32_t r) {
    return "nemo.enc.res.l" + std::to_string(l) + ".b" + std::to_string(b) + ".r" + std::to_string(r) + ".in.b";
}
static std::string nemo_enc_res_sk_w(int32_t l, int32_t b, int32_t r) {
    return "nemo.enc.res.l" + std::to_string(l) + ".b" + std::to_string(b) + ".r" + std::to_string(r) + ".sk.w";
}
static std::string nemo_enc_res_sk_b(int32_t l, int32_t b, int32_t r) {
    return "nemo.enc.res.l" + std::to_string(l) + ".b" + std::to_string(b) + ".r" + std::to_string(r) + ".sk.b";
}

static std::string nemo_dec_up_w_name(int32_t i) { return "nemo.dec.up." + std::to_string(i) + ".w"; }
static std::string nemo_dec_up_b_name(int32_t i) { return "nemo.dec.up." + std::to_string(i) + ".b"; }
static std::string nemo_dec_act_name(int32_t i) { return "nemo.dec.act." + std::to_string(i) + ".a"; }
static std::string nemo_dec_res_in_w(int32_t l, int32_t b, int32_t r) {
    return "nemo.dec.res.l" + std::to_string(l) + ".b" + std::to_string(b) + ".r" + std::to_string(r) + ".in.w";
}
static std::string nemo_dec_res_in_b(int32_t l, int32_t b, int32_t r) {
    return "nemo.dec.res.l" + std::to_string(l) + ".b" + std::to_string(b) + ".r" + std::to_string(r) + ".in.b";
}
static std::string nemo_dec_res_sk_w(int32_t l, int32_t b, int32_t r) {
    return "nemo.dec.res.l" + std::to_string(l) + ".b" + std::to_string(b) + ".r" + std::to_string(r) + ".sk.w";
}
static std::string nemo_dec_res_sk_b(int32_t l, int32_t b, int32_t r) {
    return "nemo.dec.res.l" + std::to_string(l) + ".b" + std::to_string(b) + ".r" + std::to_string(r) + ".sk.b";
}
static std::string nemo_dec_res_in_a(int32_t l, int32_t b, int32_t r) {
    return "nemo.dec.res.l" + std::to_string(l) + ".b" + std::to_string(b) + ".r" + std::to_string(r) + ".in.a";
}
static std::string nemo_dec_res_sk_a(int32_t l, int32_t b, int32_t r) {
    return "nemo.dec.res.l" + std::to_string(l) + ".b" + std::to_string(b) + ".r" + std::to_string(r) + ".sk.a";
}

static std::string nemo_fsq_name(const std::string & suffix) { return "nemo.fsq." + suffix; }

struct nemo_encode_build {
    int32_t n_in = 0;
    int32_t hop = 0;
    int32_t n_q = 0;
    int32_t codebook_dim = 0;
    int32_t codebook_size = 0;
    const codec_model * model = nullptr;
};

struct nemo_decode_build {
    int32_t t = 0;
    int32_t q = 0;
    int32_t codebook_dim = 0;
    int32_t codebook_size = 0;
    const codec_model * model = nullptr;
};

static bool nemo_build_encode(lm_ggml_context * ctx_eval, void * user_data, lm_ggml_tensor ** out) {
    nemo_encode_build * p = static_cast<nemo_encode_build *>(user_data);
    if (ctx_eval == nullptr || p == nullptr || out == nullptr || p->n_in <= 0 || p->n_q <= 0 || p->model == nullptr) {
        return false;
    }

    auto W = [&](const std::string & name) -> lm_ggml_tensor * {
        return codec_graph_weight(ctx_eval, p->model, name);
    };

    lm_ggml_tensor * t_pcm = lm_ggml_new_tensor_2d(ctx_eval, LM_GGML_TYPE_F32, p->n_in, 1);
    lm_ggml_set_name(t_pcm, "nemo.encode.pcm");

    lm_ggml_tensor * x = t_pcm;

    lm_ggml_tensor * t_pre_w = W("nemo.enc.pre.w");
    lm_ggml_tensor * t_pre_b = W("nemo.enc.pre.b");
    if (t_pre_w == nullptr || t_pre_b == nullptr) {
        return false;
    }
    x = nemo_conv1d_replicate(ctx_eval, x, t_pre_w, t_pre_b, 1, 1, 3);
    if (x == nullptr) {
        return false;
    }
    lm_ggml_set_name(x, "nemo.enc.pre.out");

    const int32_t down_rates[5] = { 2, 3, 6, 7, 7 };
    int32_t in_channels = 24;

    for (int32_t li = 0; li < 5; ++li) {
        lm_ggml_tensor * res_sum = nullptr;
        for (int32_t bi = 0; bi < 3; ++bi) {
            lm_ggml_tensor * x_block = x;
            const int32_t k = (bi == 1) ? 7 : (bi == 2 ? 11 : 3);
            const int32_t dilations[3] = { 1, 3, 5 };
            for (int32_t ri = 0; ri < 3; ++ri) {
                lm_ggml_tensor * t_in_w = W(nemo_enc_res_in_w(li, bi, ri));
                lm_ggml_tensor * t_in_b = W(nemo_enc_res_in_b(li, bi, ri));
                lm_ggml_tensor * t_sk_w = W(nemo_enc_res_sk_w(li, bi, ri));
                lm_ggml_tensor * t_sk_b = W(nemo_enc_res_sk_b(li, bi, ri));
                if (t_in_w == nullptr || t_in_b == nullptr || t_sk_w == nullptr || t_sk_b == nullptr) {
                    return false;
                }

                lm_ggml_tensor * h = lm_ggml_leaky_relu(ctx_eval, x_block, 0.01f, false);
                const int32_t pad_in = (k * dilations[ri] - dilations[ri]) / 2;
                const int32_t pad_sk = k / 2;
                h = nemo_conv1d_replicate(ctx_eval, h, t_in_w, t_in_b, 1, dilations[ri], pad_in);
                h = lm_ggml_leaky_relu(ctx_eval, h, 0.01f, false);
                h = nemo_conv1d_replicate(ctx_eval, h, t_sk_w, t_sk_b, 1, 1, pad_sk);
                x_block = lm_ggml_add(ctx_eval, x_block, h);
                lm_ggml_set_name(x_block, ("nemo.enc.l" + std::to_string(li) + ".b" + std::to_string(bi) + ".r" + std::to_string(ri) + ".out").c_str());
            }
            res_sum = res_sum == nullptr ? x_block : lm_ggml_add(ctx_eval, res_sum, x_block);
        }

        x = lm_ggml_scale(ctx_eval, res_sum, 1.0f / 3.0f);
        x = lm_ggml_leaky_relu(ctx_eval, x, 0.01f, false);

        const int32_t out_channels = in_channels * 2;
        const int32_t stride = down_rates[li];
        const int32_t kernel = 2 * stride;
        const int32_t padding = (kernel - stride + 1) / 2;

        lm_ggml_tensor * t_dw_w = W(nemo_enc_down_w_name(li));
        lm_ggml_tensor * t_dw_b = W(nemo_enc_down_b_name(li));
        if (t_dw_w == nullptr || t_dw_b == nullptr) {
            return false;
        }
        x = nemo_conv1d_replicate(ctx_eval, x, t_dw_w, t_dw_b, stride, 1, padding);
        if (x == nullptr) {
            return false;
        }
        lm_ggml_set_name(x, ("nemo.enc.down." + std::to_string(li) + ".out").c_str());
        in_channels = out_channels;
    }

    x = lm_ggml_leaky_relu(ctx_eval, x, 0.01f, false);
    lm_ggml_tensor * t_post_w = W("nemo.enc.post.w");
    lm_ggml_tensor * t_post_b = W("nemo.enc.post.b");
    if (t_post_w == nullptr || t_post_b == nullptr) {
        return false;
    }
    x = nemo_conv1d_replicate(ctx_eval, x, t_post_w, t_post_b, 1, 1, 3);
    if (x == nullptr) {
        return false;
    }
    lm_ggml_set_name(x, "nemo.enc.post.out");

    // FSQ encode per group
    lm_ggml_tensor * t_scale = W(nemo_fsq_name("scale"));
    lm_ggml_tensor * t_out_scale = W(nemo_fsq_name("out_scale"));
    lm_ggml_tensor * t_out_offset = W(nemo_fsq_name("out_offset"));
    lm_ggml_tensor * t_in_shift = W(nemo_fsq_name("in_shift"));
    lm_ggml_tensor * t_dim_base = W(nemo_fsq_name("dim_base"));
    if (t_scale == nullptr || t_out_scale == nullptr || t_out_offset == nullptr || t_in_shift == nullptr || t_dim_base == nullptr) {
        return false;
    }

    lm_ggml_tensor * tokens = nullptr;
    const int32_t t = (int32_t) x->ne[0];
    for (int32_t g = 0; g < p->n_q; ++g) {
        const size_t offset = (size_t) g * (size_t) p->codebook_dim * x->nb[1];
        lm_ggml_tensor * x_g = lm_ggml_view_2d(ctx_eval, x, t, p->codebook_dim, x->nb[1], offset);

        lm_ggml_tensor * x_add = lm_ggml_add(ctx_eval, x_g, lm_ggml_repeat(ctx_eval, lm_ggml_reshape_2d(ctx_eval, t_in_shift, 1, p->codebook_dim), x_g));
        lm_ggml_tensor * x_tanh = lm_ggml_tanh(ctx_eval, x_add);
        lm_ggml_tensor * x_mul = lm_ggml_mul(ctx_eval, x_tanh, lm_ggml_repeat(ctx_eval, lm_ggml_reshape_2d(ctx_eval, t_out_scale, 1, p->codebook_dim), x_g));
        lm_ggml_tensor * x_comp = lm_ggml_sub(ctx_eval, x_mul, lm_ggml_repeat(ctx_eval, lm_ggml_reshape_2d(ctx_eval, t_out_offset, 1, p->codebook_dim), x_g));
        lm_ggml_tensor * x_round = lm_ggml_round(ctx_eval, x_comp);
        lm_ggml_tensor * x_norm = lm_ggml_div(ctx_eval, x_round, lm_ggml_repeat(ctx_eval, lm_ggml_reshape_2d(ctx_eval, t_scale, 1, p->codebook_dim), x_g));

        lm_ggml_tensor * x_nonneg = lm_ggml_add(ctx_eval, lm_ggml_mul(ctx_eval, x_norm, lm_ggml_repeat(ctx_eval, lm_ggml_reshape_2d(ctx_eval, t_scale, 1, p->codebook_dim), x_g)),
                                          lm_ggml_repeat(ctx_eval, lm_ggml_reshape_2d(ctx_eval, t_scale, 1, p->codebook_dim), x_g));
        lm_ggml_tensor * x_idx = lm_ggml_mul(ctx_eval, x_nonneg, lm_ggml_repeat(ctx_eval, lm_ggml_reshape_2d(ctx_eval, t_dim_base, 1, p->codebook_dim), x_g));
        lm_ggml_tensor * x_idx_ct = lm_ggml_cont(ctx_eval, lm_ggml_transpose(ctx_eval, x_idx)); // [dim, t]
        lm_ggml_tensor * idx_sum = lm_ggml_sum_rows(ctx_eval, x_idx_ct); // [1, t]
        lm_ggml_tensor * idx_1d = lm_ggml_reshape_1d(ctx_eval, idx_sum, t);
        lm_ggml_tensor * idx_i32 = lm_ggml_cast(ctx_eval, idx_1d, LM_GGML_TYPE_I32);
        lm_ggml_tensor * idx_2d = lm_ggml_reshape_2d(ctx_eval, idx_i32, t, 1);

        tokens = tokens == nullptr ? idx_2d : lm_ggml_concat(ctx_eval, tokens, idx_2d, 1);
    }

    lm_ggml_tensor * t_out = lm_ggml_cont(ctx_eval, tokens);
    lm_ggml_set_name(t_out, "nemo.encode.out");
    *out = t_out;
    return true;
}

static bool nemo_build_decode(lm_ggml_context * ctx_eval, void * user_data, lm_ggml_tensor ** out) {
    nemo_decode_build * p = static_cast<nemo_decode_build *>(user_data);
    if (ctx_eval == nullptr || p == nullptr || out == nullptr || p->t <= 0 || p->q <= 0 || p->model == nullptr) {
        return false;
    }

    auto W = [&](const std::string & name) -> lm_ggml_tensor * {
        return codec_graph_weight(ctx_eval, p->model, name);
    };

    lm_ggml_tensor * t_tok = lm_ggml_new_tensor_2d(ctx_eval, LM_GGML_TYPE_I32, p->t, p->q);
    lm_ggml_set_name(t_tok, "nemo.decode.tok");

    lm_ggml_tensor * x_ct = nullptr;
    for (int32_t g = 0; g < p->q; ++g) {
        lm_ggml_tensor * t_codebook = W("nemo.fsq.codebook." + std::to_string(g));
        if (t_codebook == nullptr) {
            return false;
        }

        lm_ggml_tensor * t_idx = lm_ggml_view_1d(ctx_eval, t_tok, p->t, (size_t) g * t_tok->nb[1]);
        lm_ggml_tensor * t_emb = lm_ggml_get_rows(ctx_eval, t_codebook, t_idx); // [codebook_dim, t]
        x_ct = (x_ct == nullptr) ? t_emb : lm_ggml_concat(ctx_eval, x_ct, t_emb, 0);
    }

    lm_ggml_tensor * x = lm_ggml_cont(ctx_eval, lm_ggml_transpose(ctx_eval, x_ct)); // [t, c]
    lm_ggml_set_name(x, "nemo.dec.embed.out");

    lm_ggml_tensor * t_pre_w = W("nemo.dec.pre.w");
    lm_ggml_tensor * t_pre_b = W("nemo.dec.pre.b");
    if (t_pre_w == nullptr || t_pre_b == nullptr) {
        return false;
    }
    x = codec_conv1d_causal(ctx_eval, x, t_pre_w, t_pre_b, 1, 1);
    if (x == nullptr) {
        return false;
    }
    lm_ggml_set_name(x, "nemo.dec.pre.out");

    const int32_t up_rates[5] = { 7, 7, 6, 3, 2 };
    int32_t in_channels = 864;

    for (int32_t li = 0; li < 5; ++li) {
        lm_ggml_tensor * t_act = W(nemo_dec_act_name(li));
        if (t_act == nullptr) {
            return false;
        }
        lm_ggml_tensor * x_left = lm_ggml_view_2d(ctx_eval, x, (int32_t) x->ne[0], in_channels / 2, x->nb[1], 0);
        lm_ggml_tensor * x_right = lm_ggml_view_2d(ctx_eval, x, (int32_t) x->ne[0], in_channels - in_channels / 2, x->nb[1], (size_t) (in_channels / 2) * x->nb[1]);
        lm_ggml_tensor * x_snake = codec_op_snake(ctx_eval, x_left, t_act, 1e-9f);
        lm_ggml_tensor * x_lr = lm_ggml_leaky_relu(ctx_eval, x_right, 0.01f, false);
        lm_ggml_tensor * x_cat = lm_ggml_concat(ctx_eval, x_snake, x_lr, 1);
        x = x_cat;

        const int32_t out_channels = in_channels / 2;
        const int32_t stride = up_rates[li];
        lm_ggml_tensor * t_up_w = W(nemo_dec_up_w_name(li));
        lm_ggml_tensor * t_up_b = W(nemo_dec_up_b_name(li));
        if (t_up_w == nullptr || t_up_b == nullptr) {
            return false;
        }
        x = codec_convtr1d_causal(ctx_eval, x, t_up_w, t_up_b, stride, 1);
        if (x == nullptr) {
            return false;
        }
        lm_ggml_set_name(x, ("nemo.dec.up." + std::to_string(li) + ".out").c_str());
        in_channels = out_channels;

        lm_ggml_tensor * res_sum = nullptr;
        for (int32_t bi = 0; bi < 3; ++bi) {
            lm_ggml_tensor * x_block = x;
            const int32_t k = (bi == 1) ? 7 : (bi == 2 ? 11 : 3);
            const int32_t dilations[3] = { 1, 3, 5 };
            for (int32_t ri = 0; ri < 3; ++ri) {
                lm_ggml_tensor * t_in_a = W(nemo_dec_res_in_a(li, bi, ri));
                lm_ggml_tensor * t_sk_a = W(nemo_dec_res_sk_a(li, bi, ri));
                lm_ggml_tensor * t_in_w = W(nemo_dec_res_in_w(li, bi, ri));
                lm_ggml_tensor * t_in_b = W(nemo_dec_res_in_b(li, bi, ri));
                lm_ggml_tensor * t_sk_w = W(nemo_dec_res_sk_w(li, bi, ri));
                lm_ggml_tensor * t_sk_b = W(nemo_dec_res_sk_b(li, bi, ri));
                if (t_in_a == nullptr || t_sk_a == nullptr || t_in_w == nullptr || t_in_b == nullptr || t_sk_w == nullptr || t_sk_b == nullptr) {
                    return false;
                }

                lm_ggml_tensor * x_left_r = lm_ggml_view_2d(ctx_eval, x_block, (int32_t) x_block->ne[0], in_channels / 2, x_block->nb[1], 0);
                lm_ggml_tensor * x_right_r = lm_ggml_view_2d(ctx_eval, x_block, (int32_t) x_block->ne[0], in_channels - in_channels / 2, x_block->nb[1], (size_t) (in_channels / 2) * x_block->nb[1]);
                lm_ggml_tensor * x_snake_r = codec_op_snake(ctx_eval, x_left_r, t_in_a, 1e-9f);
                lm_ggml_tensor * x_lr_r = lm_ggml_leaky_relu(ctx_eval, x_right_r, 0.01f, false);
                lm_ggml_tensor * x_act = lm_ggml_concat(ctx_eval, x_snake_r, x_lr_r, 1);

                lm_ggml_tensor * h = codec_conv1d_causal(ctx_eval, x_act, t_in_w, t_in_b, 1, dilations[ri]);
                if (h == nullptr) {
                    return false;
                }

                lm_ggml_tensor * h_left = lm_ggml_view_2d(ctx_eval, h, (int32_t) h->ne[0], in_channels / 2, h->nb[1], 0);
                lm_ggml_tensor * h_right = lm_ggml_view_2d(ctx_eval, h, (int32_t) h->ne[0], in_channels - in_channels / 2, h->nb[1], (size_t) (in_channels / 2) * h->nb[1]);
                lm_ggml_tensor * h_snake = codec_op_snake(ctx_eval, h_left, t_sk_a, 1e-9f);
                lm_ggml_tensor * h_lr = lm_ggml_leaky_relu(ctx_eval, h_right, 0.01f, false);
                lm_ggml_tensor * h_act = lm_ggml_concat(ctx_eval, h_snake, h_lr, 1);

                h = codec_conv1d_causal(ctx_eval, h_act, t_sk_w, t_sk_b, 1, 1);
                if (h == nullptr) {
                    return false;
                }
                x_block = lm_ggml_add(ctx_eval, x_block, h);
                lm_ggml_set_name(x_block, ("nemo.dec.l" + std::to_string(li) + ".b" + std::to_string(bi) + ".r" + std::to_string(ri) + ".out").c_str());
            }
            res_sum = res_sum == nullptr ? x_block : lm_ggml_add(ctx_eval, res_sum, x_block);
        }
        x = lm_ggml_scale(ctx_eval, res_sum, 1.0f / 3.0f);
    }

    lm_ggml_tensor * t_post_a = W("nemo.dec.post.a");
    if (t_post_a == nullptr) {
        return false;
    }
    lm_ggml_tensor * x_left_f = lm_ggml_view_2d(ctx_eval, x, (int32_t) x->ne[0], in_channels / 2, x->nb[1], 0);
    lm_ggml_tensor * x_right_f = lm_ggml_view_2d(ctx_eval, x, (int32_t) x->ne[0], in_channels - in_channels / 2, x->nb[1], (size_t) (in_channels / 2) * x->nb[1]);
    lm_ggml_tensor * x_snake_f = codec_op_snake(ctx_eval, x_left_f, t_post_a, 1e-9f);
    lm_ggml_tensor * x_lr_f = lm_ggml_leaky_relu(ctx_eval, x_right_f, 0.01f, false);
    lm_ggml_tensor * x_act_f = lm_ggml_concat(ctx_eval, x_snake_f, x_lr_f, 1);
    x = x_act_f;
    lm_ggml_set_name(x, "nemo.dec.post.act");

    lm_ggml_tensor * t_post_w = W("nemo.dec.post.w");
    lm_ggml_tensor * t_post_b = W("nemo.dec.post.b");
    if (t_post_w == nullptr || t_post_b == nullptr) {
        return false;
    }
    lm_ggml_tensor * t_pcm = codec_conv1d_causal(ctx_eval, x, t_post_w, t_post_b, 1, 1);
    if (t_pcm == nullptr) {
        return false;
    }

    lm_ggml_tensor * t_out = lm_ggml_clamp(ctx_eval, t_pcm, -1.0f, 1.0f);
    lm_ggml_set_name(t_pcm, "nemo.dec.post.out");
    lm_ggml_set_name(t_out, "nemo.decode.out");
    *out = t_out;
    return true;
}

static enum codec_status nemo_encode_graph(
    codec_context * ctx,
    const std::vector<float> & pcm,
    struct codec_token_buffer * out_tokens,
    int32_t hop_size,
    int32_t sample_rate) {

    codec_nemo_nano_codec & nemo = *static_cast<codec_nemo_nano_codec *>(ctx->model->impl);
    const int32_t n_in = (int32_t) pcm.size();
    const int32_t n_q = std::max(1, nemo.n_q);
    const int32_t codebook_dim = std::max(1, nemo.codebook_dim);
    const int32_t codebook_size = std::max(2, nemo.codebook_size);

    codec_graph_eval_guard eval_guard(ctx);
    nemo_encode_build build = { n_in, hop_size, n_q, codebook_dim, codebook_size, ctx->model };
    codec_graph_cache_entry * entry = nullptr;
    std::string err;
    if (!codec_graph_cache_get_or_build(
            ctx,
            { CODEC_GRAPH_NEMO_NANO_ENCODE, /*n_frames=*/0, /*n_q=*/n_q, /*hop=*/hop_size, /*n_in=*/n_in, /*latent_dim=*/codebook_dim },
            nemo_build_encode,
            &build,
            sizeof(build),
            &entry,
            &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    lm_ggml_tensor * t_pcm = codec_graph_get_tensor(ctx, entry, "nemo.encode.pcm");
    lm_ggml_tensor * t_out = codec_graph_get_tensor(ctx, entry, "nemo.encode.out");
    if (t_pcm == nullptr || t_out == nullptr) {
        codec_context_set_error(ctx, "cached NeMo encode graph is invalid");
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    if (!codec_graph_prepare_io(ctx, entry, &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }
    if (!codec_runtime_write_tensor(t_pcm, pcm.data(), pcm.size() * sizeof(float), &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    const int32_t n_threads = ctx->model->n_threads > 0 ? ctx->model->n_threads : 1;
    if (!codec_graph_compute(ctx, entry, n_threads, &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    const int32_t n_frames = (int32_t) t_out->ne[0];
    const int32_t nq = (int32_t) t_out->ne[1];

    std::vector<int32_t> tok_tq;
    if (!codec_runtime_read_tensor_i32_2d_tq(t_out, &tok_tq, &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    int32_t * data = static_cast<int32_t *>(std::malloc(tok_tq.size() * sizeof(int32_t)));
    if (data == nullptr) {
        codec_context_set_error(ctx, "failed to allocate token output");
        return CODEC_STATUS_INTERNAL_ERROR;
    }
    std::memcpy(data, tok_tq.data(), tok_tq.size() * sizeof(int32_t));
    codec_token_buffer_reset(out_tokens);
    out_tokens->data = data;
    out_tokens->n_tokens = n_frames * nq;
    out_tokens->n_frames = n_frames;
    out_tokens->n_q = nq;
    out_tokens->codebook_size = codebook_size;
    out_tokens->sample_rate = sample_rate;
    out_tokens->hop_size = hop_size;

    return CODEC_STATUS_SUCCESS;
}

static enum codec_status nemo_decode_graph(
    codec_context * ctx,
    const struct codec_token_buffer * tokens,
    int32_t use_n_q,
    struct codec_pcm_buffer * out_pcm,
    int32_t hop_size,
    int32_t sample_rate) {

    codec_nemo_nano_codec & nemo = *static_cast<codec_nemo_nano_codec *>(ctx->model->impl);
    if (tokens == nullptr || tokens->data == nullptr || tokens->n_frames <= 0) {
        codec_context_set_error(ctx, "invalid NeMo token buffer");
        return CODEC_STATUS_INVALID_ARG;
    }

    const int32_t t = tokens->n_frames;
    const int32_t q = use_n_q;
    const int32_t codebook_dim = std::max(1, nemo.codebook_dim);
    const int32_t codebook_size = std::max(2, nemo.codebook_size);
    codec_graph_eval_guard eval_guard(ctx);

    nemo_decode_build build = { t, q, codebook_dim, codebook_size, ctx->model };
    codec_graph_cache_entry * entry = nullptr;
    std::string err;
    if (!codec_graph_cache_get_or_build(
            ctx,
            { CODEC_GRAPH_NEMO_NANO_DECODE, /*n_frames=*/t, /*n_q=*/q, /*hop=*/hop_size, /*n_in=*/0, /*latent_dim=*/0 },
            nemo_build_decode,
            &build,
            sizeof(build),
            &entry,
            &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    lm_ggml_tensor * t_tok = codec_graph_get_tensor(ctx, entry, "nemo.decode.tok");
    lm_ggml_tensor * t_out = codec_graph_get_tensor(ctx, entry, "nemo.decode.out");
    if (t_tok == nullptr || t_out == nullptr) {
        codec_context_set_error(ctx, "cached NeMo decode graph is invalid");
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    if (!codec_graph_prepare_io(ctx, entry, &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    // tokens->data is [t, q] (t-major). ggml expects [q, t] with t as ne0.
    std::vector<int32_t> tok_i32((size_t) t * (size_t) q, 0);
    const int32_t * src = static_cast<const int32_t *>(tokens->data);
    for (int32_t ti = 0; ti < t; ++ti) {
        for (int32_t qi = 0; qi < q; ++qi) {
            int32_t v = src[(size_t) ti * (size_t) q + (size_t) qi];
            v = std::max(0, std::min(codebook_size - 1, v));
            tok_i32[(size_t) qi * (size_t) t + (size_t) ti] = v;
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

    const int32_t n_pcm = (int32_t) t_out->ne[0];
    std::vector<float> pcm((size_t) n_pcm, 0.0f);
    if (!codec_runtime_read_tensor(t_out, pcm.data(), pcm.size() * sizeof(float), &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    codec_pcm_buffer_reset(out_pcm);
    out_pcm->n_channels = 1;
    out_pcm->sample_rate = sample_rate;
    out_pcm->n_samples = n_pcm;
    out_pcm->data = static_cast<float *>(std::malloc(pcm.size() * sizeof(float)));
    if (out_pcm->data == nullptr) {
        codec_context_set_error(ctx, "failed to allocate PCM output");
        return CODEC_STATUS_INTERNAL_ERROR;
    }
    std::memcpy(out_pcm->data, pcm.data(), pcm.size() * sizeof(float));
    return CODEC_STATUS_SUCCESS;
}

enum codec_status codec_nemo_nano_codec_init(struct codec_model * model) {
    if (model == nullptr) {
        return CODEC_STATUS_INVALID_ARG;
    }

    codec_nemo_nano_codec nemo;
    nemo.sample_rate = codec_read_i32_kv(model->gguf, "codec.sample_rate", 22050);
    nemo.hop_size = codec_read_i32_kv(model->gguf, "codec.hop_size", 1764);
    nemo.n_q = codec_read_i32_kv(model->gguf, "codec.n_q", 4);
    nemo.codebook_size = codec_read_i32_kv(model->gguf, "codec.codebook_size", 4032);
    nemo.codebook_dim = codec_read_i32_kv(model->gguf, "codec.codebook_dim", 4);
    nemo.latent_dim = codec_read_i32_kv(model->gguf, "codec.latent_dim", 16);
    nemo.has_encoder = codec_read_bool_kv(model->gguf, "codec.has_encoder", true);
    nemo.has_decoder = codec_read_bool_kv(model->gguf, "codec.has_decoder", true);

    model->sample_rate = nemo.sample_rate;
    model->hop_size = nemo.hop_size;
    model->n_q = nemo.n_q;
    model->codebook_size = nemo.codebook_size;
    model->latent_dim = nemo.latent_dim;
    model->has_encoder = nemo.has_encoder;
    model->has_decoder = nemo.has_decoder;
    model->impl = new (std::nothrow) codec_nemo_nano_codec(nemo);
    if (model->impl == nullptr) {
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    return CODEC_STATUS_SUCCESS;
}

enum codec_status codec_nemo_nano_codec_encode(
    struct codec_context * ctx,
    const std::vector<float> & pcm,
    struct codec_token_buffer * out_tokens,
    struct codec_latent_buffer * /*out_latent*/,
    struct codec_encode_params params) {

    codec_nemo_nano_codec & nemo = *static_cast<codec_nemo_nano_codec *>(ctx->model->impl);
    if (!nemo.has_encoder) {
        codec_context_set_error(ctx, "model metadata indicates no encoder");
        return CODEC_STATUS_INVALID_STATE;
    }
    if (pcm.empty()) {
        codec_context_set_error(ctx, "empty pcm");
        return CODEC_STATUS_INVALID_ARG;
    }

    const int32_t model_n_q = std::max(1, nemo.n_q);
    const int32_t use_n_q = params.n_q == 0 ? model_n_q : params.n_q;
    if (params.n_q < 0 || use_n_q < 1 || use_n_q > model_n_q) {
        codec_context_set_error(ctx, "NeMo encode n_q must be 0 or in [1, model_n_q]");
        return CODEC_STATUS_INVALID_ARG;
    }

    const int32_t hop = std::max(1, params.hop_size > 0 ? params.hop_size : nemo.hop_size);
    return nemo_encode_graph(ctx, pcm, out_tokens, hop, nemo.sample_rate);
}

enum codec_status codec_nemo_nano_codec_decode(
    struct codec_context * ctx,
    const struct codec_token_buffer * tokens,
    struct codec_pcm_buffer * out_pcm,
    struct codec_decode_params params) {

    codec_nemo_nano_codec & nemo = *static_cast<codec_nemo_nano_codec *>(ctx->model->impl);
    if (!nemo.has_decoder) {
        codec_context_set_error(ctx, "model metadata indicates no decoder");
        return CODEC_STATUS_INVALID_STATE;
    }
    const int32_t model_n_q = std::max(1, nemo.n_q);
    const int32_t use_n_q = params.n_q == 0 ? model_n_q : params.n_q;
    if (params.n_q < 0 || use_n_q < 1 || use_n_q > model_n_q) {
        codec_context_set_error(ctx, "NeMo decode n_q must be 0 or in [1, model_n_q]");
        return CODEC_STATUS_INVALID_ARG;
    }
    const int32_t hop = std::max(1, nemo.hop_size);
    return nemo_decode_graph(ctx, tokens, use_n_q, out_pcm, hop, nemo.sample_rate);
}

static void * nemo_create_impl() {
    return new (std::nothrow) codec_nemo_nano_codec();
}

static void nemo_destroy_impl(void * ptr) {
    delete static_cast<codec_nemo_nano_codec *>(ptr);
}

static enum codec_status nemo_encode_wrap(
    struct codec_context * ctx,
    const std::vector<float> & pcm,
    struct codec_token_buffer * out_tokens,
    struct codec_latent_buffer * out_latent,
    struct codec_encode_params params) {
    return codec_nemo_nano_codec_encode(ctx, pcm, out_tokens, out_latent, params);
}

static enum codec_status nemo_decode_wrap(
    struct codec_context * ctx,
    const struct codec_token_buffer * tokens,
    struct codec_pcm_buffer * out_pcm,
    struct codec_decode_params params) {
    return codec_nemo_nano_codec_decode(ctx, tokens, out_pcm, params);
}

const struct codec_model_vtable * codec_nemo_nano_codec_vtable() {
    static codec_model_vtable vtable = {
        CODEC_ARCH_NEMO_NANO_CODEC,
        "nemo_nano_codec",
        nemo_create_impl,
        nemo_destroy_impl,
        codec_nemo_nano_codec_init,
        codec_graph_size_exact,
        nemo_encode_wrap,
        nemo_decode_wrap,
        nullptr,
    };
    return &vtable;
}
