#include "wavtokenizer.h"

#include "../ops/conv1d.h"
#include "../ops/convtr1d.h"
#include "../ops/lm_ggml_ops.h"
#include "../ops/rvq.h"
#include "../runtime/graph.h"
#include "../runtime/tensor_utils.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <new>
#include <string>
#include <vector>

enum codec_status codec_wavtokenizer_init(struct codec_model * model) {
    codec_wavtokenizer_large & wt = *static_cast<codec_wavtokenizer_large *>(model->impl);

    wt.sample_rate = codec_read_i32_kv(model->gguf, "codec.sample_rate", 24000);
    wt.hop_size = codec_read_i32_kv(model->gguf, "codec.hop_size", 320);
    wt.has_encoder = codec_read_bool_kv(model->gguf, "codec.has_encoder", true);
    wt.has_decoder = codec_read_bool_kv(model->gguf, "codec.has_decoder", true);
    wt.n_q = codec_read_i32_kv(model->gguf, "codec.n_q", codec_infer_n_q_from_tensor_names(model));

    wt.vq_embed = lm_ggml_get_tensor(model->weights, "vq.vq.layers.0._codebook.embed");
    if (wt.vq_embed != nullptr) {
        wt.codebook_dim = (int32_t) wt.vq_embed->ne[0];
        wt.codebook_size = (int32_t) wt.vq_embed->ne[1];
    }
    if (wt.codebook_size <= 0) {
        wt.codebook_size = codec_read_i32_kv(model->gguf, "codec.codebook_size", 1024);
    }

    model->sample_rate = wt.sample_rate;
    model->has_encoder = wt.has_encoder;
    model->has_decoder = wt.has_decoder;
    model->hop_size = wt.hop_size;
    model->n_q = wt.n_q;
    model->codebook_size = wt.codebook_size;
    model->latent_dim = wt.codebook_dim > 0 ? wt.codebook_dim : 1;

    static const char * const keys_n_fft[] = { "codec.n_fft", "codec.stft.n_fft" };
    static const char * const keys_win_length[] = { "codec.win_length", "codec.stft.win_length" };
    static const char * const keys_n_mels[] = { "codec.n_mels", "codec.mel.n_mels" };

    model->n_fft = codec_read_i32_kv_any(model->gguf, keys_n_fft, 2, -1);
    model->win_length = codec_read_i32_kv_any(model->gguf, keys_win_length, 2, -1);
    model->n_mels = codec_read_i32_kv_any(model->gguf, keys_n_mels, 2, -1);

    return CODEC_STATUS_SUCCESS;
}

struct wt_decode_build {
    int32_t t;
    int32_t q;
    int32_t hop;
    int32_t codebook_dim;
    int32_t codebook_size;
    int32_t backbone_dim;
    int32_t backbone_intermediate;
    int32_t n_convnext;
    int32_t head_out_dim;
    int32_t use_adanorm;
    int32_t use_pos_net;
};

static std::string codec_wt_decode_codebook_tensor_name(int32_t qi) {
    return "wt.decode.vq.q" + std::to_string(qi) + ".codebook";
}

static lm_ggml_tensor * codec_wt_sum_codebook_features(
    lm_ggml_context * ctx_eval,
    lm_ggml_tensor * t_tok,
    int32_t t,
    int32_t q,
    int32_t codebook_dim,
    int32_t codebook_size) {

    lm_ggml_tensor * sum = nullptr;
    for (int32_t qi = 0; qi < q; ++qi) {
        lm_ggml_tensor * t_codebook = lm_ggml_new_tensor_2d(ctx_eval, LM_GGML_TYPE_F32, codebook_dim, codebook_size);
        lm_ggml_set_name(t_codebook, codec_wt_decode_codebook_tensor_name(qi).c_str());
        lm_ggml_tensor * t_idx = lm_ggml_view_1d(ctx_eval, t_tok, t, (size_t) qi * t_tok->nb[1]);
        lm_ggml_tensor * t_q = lm_ggml_get_rows(ctx_eval, t_codebook, t_idx); // [codebook_dim, t]
        sum = (sum == nullptr) ? t_q : lm_ggml_add(ctx_eval, sum, t_q);
    }
    return sum;
}

static lm_ggml_tensor * codec_wt_layer_norm_ct(
    lm_ggml_context * ctx_eval,
    lm_ggml_tensor * x_ct,
    lm_ggml_tensor * gamma,
    lm_ggml_tensor * beta) {

    if (ctx_eval == nullptr || x_ct == nullptr || gamma == nullptr || beta == nullptr) {
        return nullptr;
    }
    lm_ggml_tensor * y = lm_ggml_norm(ctx_eval, x_ct, 1e-6f);
    lm_ggml_tensor * g2 = lm_ggml_reshape_2d(ctx_eval, gamma, x_ct->ne[0], 1);
    lm_ggml_tensor * b2 = lm_ggml_reshape_2d(ctx_eval, beta, x_ct->ne[0], 1);
    y = lm_ggml_mul(ctx_eval, y, lm_ggml_repeat(ctx_eval, g2, y));
    y = lm_ggml_add(ctx_eval, y, lm_ggml_repeat(ctx_eval, b2, y));
    return y;
}

static std::string codec_wt_decode_embed_w_name() { return "wt.decode.bb.embed.w"; }
static std::string codec_wt_decode_embed_b_name() { return "wt.decode.bb.embed.b"; }
static std::string codec_wt_decode_norm_w_name() { return "wt.decode.bb.norm.w"; }
static std::string codec_wt_decode_norm_b_name() { return "wt.decode.bb.norm.b"; }
static std::string codec_wt_decode_final_ln_w_name() { return "wt.decode.bb.final_ln.w"; }
static std::string codec_wt_decode_final_ln_b_name() { return "wt.decode.bb.final_ln.b"; }
static std::string codec_wt_decode_head_w_name() { return "wt.decode.head.out.w"; }
static std::string codec_wt_decode_head_b_name() { return "wt.decode.head.out.b"; }

static std::string codec_wt_decode_pos_name(int32_t li, const char * suffix) {
    return "wt.decode.bb.pos_net." + std::to_string(li) + "." + suffix;
}

static std::string codec_wt_decode_blk_dw_w_name(int32_t li) { return "wt.decode.bb.l" + std::to_string(li) + ".dw.w"; }
static std::string codec_wt_decode_blk_dw_b_name(int32_t li) { return "wt.decode.bb.l" + std::to_string(li) + ".dw.b"; }
static std::string codec_wt_decode_blk_ln_w_name(int32_t li) { return "wt.decode.bb.l" + std::to_string(li) + ".ln.w"; }
static std::string codec_wt_decode_blk_ln_b_name(int32_t li) { return "wt.decode.bb.l" + std::to_string(li) + ".ln.b"; }
static std::string codec_wt_decode_blk_pw1_w_name(int32_t li) { return "wt.decode.bb.l" + std::to_string(li) + ".pw1.w"; }
static std::string codec_wt_decode_blk_pw1_b_name(int32_t li) { return "wt.decode.bb.l" + std::to_string(li) + ".pw1.b"; }
static std::string codec_wt_decode_blk_pw2_w_name(int32_t li) { return "wt.decode.bb.l" + std::to_string(li) + ".pw2.w"; }
static std::string codec_wt_decode_blk_pw2_b_name(int32_t li) { return "wt.decode.bb.l" + std::to_string(li) + ".pw2.b"; }
static std::string codec_wt_decode_blk_gamma_name(int32_t li) { return "wt.decode.bb.l" + std::to_string(li) + ".gamma"; }

static lm_ggml_tensor * codec_wt_pos_group_norm(
    lm_ggml_context * ctx_eval,
    lm_ggml_tensor * x,
    lm_ggml_tensor * gamma,
    lm_ggml_tensor * beta) {

    return codec_op_group_norm(ctx_eval, x, 32, 1e-6f, gamma, beta);
}

static lm_ggml_tensor * codec_wt_pos_resblock(
    lm_ggml_context * ctx_eval,
    lm_ggml_tensor * x,
    lm_ggml_tensor * n1_w,
    lm_ggml_tensor * n1_b,
    lm_ggml_tensor * c1_w,
    lm_ggml_tensor * c1_b,
    lm_ggml_tensor * n2_w,
    lm_ggml_tensor * n2_b,
    lm_ggml_tensor * c2_w,
    lm_ggml_tensor * c2_b) {

    lm_ggml_tensor * h = codec_wt_pos_group_norm(ctx_eval, x, n1_w, n1_b);
    if (h == nullptr) {
        return nullptr;
    }
    h = codec_op_unary(ctx_eval, h, CODEC_UNARY_SILU);
    h = codec_conv1d(ctx_eval, h, c1_w, c1_b, 1, 1, 1);
    if (h == nullptr) {
        return nullptr;
    }
    h = codec_wt_pos_group_norm(ctx_eval, h, n2_w, n2_b);
    if (h == nullptr) {
        return nullptr;
    }
    h = codec_op_unary(ctx_eval, h, CODEC_UNARY_SILU);
    h = codec_conv1d(ctx_eval, h, c2_w, c2_b, 1, 1, 1);
    if (h == nullptr) {
        return nullptr;
    }
    return lm_ggml_add(ctx_eval, x, h);
}

static lm_ggml_tensor * codec_wt_pos_attn(
    lm_ggml_context * ctx_eval,
    lm_ggml_tensor * x,
    lm_ggml_tensor * n_w,
    lm_ggml_tensor * n_b,
    lm_ggml_tensor * q_w,
    lm_ggml_tensor * q_b,
    lm_ggml_tensor * k_w,
    lm_ggml_tensor * k_b,
    lm_ggml_tensor * v_w,
    lm_ggml_tensor * v_b,
    lm_ggml_tensor * o_w,
    lm_ggml_tensor * o_b,
    int32_t dim) {

    lm_ggml_tensor * h = codec_wt_pos_group_norm(ctx_eval, x, n_w, n_b);
    if (h == nullptr) {
        return nullptr;
    }

    lm_ggml_tensor * q = codec_conv1d(ctx_eval, h, q_w, q_b, 1, 1, 0);
    lm_ggml_tensor * k = codec_conv1d(ctx_eval, h, k_w, k_b, 1, 1, 0);
    lm_ggml_tensor * v = codec_conv1d(ctx_eval, h, v_w, v_b, 1, 1, 0);
    if (q == nullptr || k == nullptr || v == nullptr) {
        return nullptr;
    }

    lm_ggml_tensor * q_ct = lm_ggml_cont(ctx_eval, lm_ggml_transpose(ctx_eval, q));
    lm_ggml_tensor * k_ct = lm_ggml_cont(ctx_eval, lm_ggml_transpose(ctx_eval, k));
    if (q_ct == nullptr || k_ct == nullptr || v == nullptr) {
        return nullptr;
    }

    lm_ggml_tensor * scores = lm_ggml_mul_mat(ctx_eval, k_ct, q_ct); // [t, t]
    if (scores == nullptr) {
        return nullptr;
    }
    const float scale = dim > 0 ? (1.0f / std::sqrt((float) dim)) : 1.0f;
    scores = lm_ggml_scale(ctx_eval, scores, scale);
    lm_ggml_tensor * probs = lm_ggml_soft_max(ctx_eval, scores);
    if (probs == nullptr) {
        return nullptr;
    }

    lm_ggml_tensor * v_tc = lm_ggml_cont(ctx_eval, v); // [t, c]
    lm_ggml_tensor * ctx_ct = lm_ggml_mul_mat(ctx_eval, v_tc, probs); // [c, t]
    if (ctx_ct == nullptr) {
        return nullptr;
    }
    lm_ggml_tensor * ctx_tc = lm_ggml_cont(ctx_eval, lm_ggml_transpose(ctx_eval, ctx_ct));
    if (ctx_tc == nullptr) {
        return nullptr;
    }
    lm_ggml_tensor * proj = codec_conv1d(ctx_eval, ctx_tc, o_w, o_b, 1, 1, 0);
    if (proj == nullptr) {
        return nullptr;
    }
    return lm_ggml_add(ctx_eval, x, proj);
}

static bool codec_wt_build_decode(lm_ggml_context * ctx_eval, void * user_data, lm_ggml_tensor ** out) {
    wt_decode_build * p = static_cast<wt_decode_build *>(user_data);
    if (ctx_eval == nullptr || p == nullptr || out == nullptr || p->t <= 0 || p->q <= 0 || p->codebook_dim <= 0 || p->codebook_size <= 1 ||
        p->backbone_dim <= 0 || p->backbone_intermediate <= 0 || p->n_convnext <= 0 || p->head_out_dim <= 0) {
        return false;
    }

    lm_ggml_tensor * t_tok = lm_ggml_new_tensor_2d(ctx_eval, LM_GGML_TYPE_I32, p->t, p->q);
    lm_ggml_set_name(t_tok, "wt.decode.tok");

    lm_ggml_tensor * t_feat_ct = codec_wt_sum_codebook_features(ctx_eval, t_tok, p->t, p->q, p->codebook_dim, p->codebook_size);
    if (t_feat_ct == nullptr) {
        return false;
    }
    lm_ggml_tensor * x = lm_ggml_cont(ctx_eval, lm_ggml_transpose(ctx_eval, t_feat_ct)); // [t, c]

    lm_ggml_tensor * t_embed_w = lm_ggml_new_tensor_3d(ctx_eval, LM_GGML_TYPE_F32, 7, p->codebook_dim, p->backbone_dim);
    lm_ggml_set_name(t_embed_w, codec_wt_decode_embed_w_name().c_str());
    lm_ggml_tensor * t_embed_b = lm_ggml_new_tensor_1d(ctx_eval, LM_GGML_TYPE_F32, p->backbone_dim);
    lm_ggml_set_name(t_embed_b, codec_wt_decode_embed_b_name().c_str());
    x = codec_conv1d(ctx_eval, x, t_embed_w, t_embed_b, 1, 1, 3);
    if (x == nullptr) {
        return false;
    }

    if (p->use_pos_net) {
        for (int32_t li = 0; li < 2; ++li) {
            lm_ggml_tensor * n1_w = lm_ggml_new_tensor_1d(ctx_eval, LM_GGML_TYPE_F32, p->backbone_dim);
            lm_ggml_set_name(n1_w, codec_wt_decode_pos_name(li, "norm1.w").c_str());
            lm_ggml_tensor * n1_b = lm_ggml_new_tensor_1d(ctx_eval, LM_GGML_TYPE_F32, p->backbone_dim);
            lm_ggml_set_name(n1_b, codec_wt_decode_pos_name(li, "norm1.b").c_str());
            lm_ggml_tensor * c1_w = lm_ggml_new_tensor_3d(ctx_eval, LM_GGML_TYPE_F32, 3, p->backbone_dim, p->backbone_dim);
            lm_ggml_set_name(c1_w, codec_wt_decode_pos_name(li, "conv1.w").c_str());
            lm_ggml_tensor * c1_b = lm_ggml_new_tensor_1d(ctx_eval, LM_GGML_TYPE_F32, p->backbone_dim);
            lm_ggml_set_name(c1_b, codec_wt_decode_pos_name(li, "conv1.b").c_str());
            lm_ggml_tensor * n2_w = lm_ggml_new_tensor_1d(ctx_eval, LM_GGML_TYPE_F32, p->backbone_dim);
            lm_ggml_set_name(n2_w, codec_wt_decode_pos_name(li, "norm2.w").c_str());
            lm_ggml_tensor * n2_b = lm_ggml_new_tensor_1d(ctx_eval, LM_GGML_TYPE_F32, p->backbone_dim);
            lm_ggml_set_name(n2_b, codec_wt_decode_pos_name(li, "norm2.b").c_str());
            lm_ggml_tensor * c2_w = lm_ggml_new_tensor_3d(ctx_eval, LM_GGML_TYPE_F32, 3, p->backbone_dim, p->backbone_dim);
            lm_ggml_set_name(c2_w, codec_wt_decode_pos_name(li, "conv2.w").c_str());
            lm_ggml_tensor * c2_b = lm_ggml_new_tensor_1d(ctx_eval, LM_GGML_TYPE_F32, p->backbone_dim);
            lm_ggml_set_name(c2_b, codec_wt_decode_pos_name(li, "conv2.b").c_str());
            x = codec_wt_pos_resblock(ctx_eval, x, n1_w, n1_b, c1_w, c1_b, n2_w, n2_b, c2_w, c2_b);
            if (x == nullptr) {
                return false;
            }
        }

        lm_ggml_tensor * attn_n_w = lm_ggml_new_tensor_1d(ctx_eval, LM_GGML_TYPE_F32, p->backbone_dim);
        lm_ggml_set_name(attn_n_w, codec_wt_decode_pos_name(2, "norm.w").c_str());
        lm_ggml_tensor * attn_n_b = lm_ggml_new_tensor_1d(ctx_eval, LM_GGML_TYPE_F32, p->backbone_dim);
        lm_ggml_set_name(attn_n_b, codec_wt_decode_pos_name(2, "norm.b").c_str());
        lm_ggml_tensor * q_w = lm_ggml_new_tensor_3d(ctx_eval, LM_GGML_TYPE_F32, 1, p->backbone_dim, p->backbone_dim);
        lm_ggml_set_name(q_w, codec_wt_decode_pos_name(2, "q.w").c_str());
        lm_ggml_tensor * q_b = lm_ggml_new_tensor_1d(ctx_eval, LM_GGML_TYPE_F32, p->backbone_dim);
        lm_ggml_set_name(q_b, codec_wt_decode_pos_name(2, "q.b").c_str());
        lm_ggml_tensor * k_w = lm_ggml_new_tensor_3d(ctx_eval, LM_GGML_TYPE_F32, 1, p->backbone_dim, p->backbone_dim);
        lm_ggml_set_name(k_w, codec_wt_decode_pos_name(2, "k.w").c_str());
        lm_ggml_tensor * k_b = lm_ggml_new_tensor_1d(ctx_eval, LM_GGML_TYPE_F32, p->backbone_dim);
        lm_ggml_set_name(k_b, codec_wt_decode_pos_name(2, "k.b").c_str());
        lm_ggml_tensor * v_w = lm_ggml_new_tensor_3d(ctx_eval, LM_GGML_TYPE_F32, 1, p->backbone_dim, p->backbone_dim);
        lm_ggml_set_name(v_w, codec_wt_decode_pos_name(2, "v.w").c_str());
        lm_ggml_tensor * v_b = lm_ggml_new_tensor_1d(ctx_eval, LM_GGML_TYPE_F32, p->backbone_dim);
        lm_ggml_set_name(v_b, codec_wt_decode_pos_name(2, "v.b").c_str());
        lm_ggml_tensor * o_w = lm_ggml_new_tensor_3d(ctx_eval, LM_GGML_TYPE_F32, 1, p->backbone_dim, p->backbone_dim);
        lm_ggml_set_name(o_w, codec_wt_decode_pos_name(2, "proj_out.w").c_str());
        lm_ggml_tensor * o_b = lm_ggml_new_tensor_1d(ctx_eval, LM_GGML_TYPE_F32, p->backbone_dim);
        lm_ggml_set_name(o_b, codec_wt_decode_pos_name(2, "proj_out.b").c_str());
        x = codec_wt_pos_attn(ctx_eval, x, attn_n_w, attn_n_b, q_w, q_b, k_w, k_b, v_w, v_b, o_w, o_b, p->backbone_dim);
        if (x == nullptr) {
            return false;
        }

        for (int32_t li = 3; li < 5; ++li) {
            lm_ggml_tensor * n1_w = lm_ggml_new_tensor_1d(ctx_eval, LM_GGML_TYPE_F32, p->backbone_dim);
            lm_ggml_set_name(n1_w, codec_wt_decode_pos_name(li, "norm1.w").c_str());
            lm_ggml_tensor * n1_b = lm_ggml_new_tensor_1d(ctx_eval, LM_GGML_TYPE_F32, p->backbone_dim);
            lm_ggml_set_name(n1_b, codec_wt_decode_pos_name(li, "norm1.b").c_str());
            lm_ggml_tensor * c1_w = lm_ggml_new_tensor_3d(ctx_eval, LM_GGML_TYPE_F32, 3, p->backbone_dim, p->backbone_dim);
            lm_ggml_set_name(c1_w, codec_wt_decode_pos_name(li, "conv1.w").c_str());
            lm_ggml_tensor * c1_b = lm_ggml_new_tensor_1d(ctx_eval, LM_GGML_TYPE_F32, p->backbone_dim);
            lm_ggml_set_name(c1_b, codec_wt_decode_pos_name(li, "conv1.b").c_str());
            lm_ggml_tensor * n2_w = lm_ggml_new_tensor_1d(ctx_eval, LM_GGML_TYPE_F32, p->backbone_dim);
            lm_ggml_set_name(n2_w, codec_wt_decode_pos_name(li, "norm2.w").c_str());
            lm_ggml_tensor * n2_b = lm_ggml_new_tensor_1d(ctx_eval, LM_GGML_TYPE_F32, p->backbone_dim);
            lm_ggml_set_name(n2_b, codec_wt_decode_pos_name(li, "norm2.b").c_str());
            lm_ggml_tensor * c2_w = lm_ggml_new_tensor_3d(ctx_eval, LM_GGML_TYPE_F32, 3, p->backbone_dim, p->backbone_dim);
            lm_ggml_set_name(c2_w, codec_wt_decode_pos_name(li, "conv2.w").c_str());
            lm_ggml_tensor * c2_b = lm_ggml_new_tensor_1d(ctx_eval, LM_GGML_TYPE_F32, p->backbone_dim);
            lm_ggml_set_name(c2_b, codec_wt_decode_pos_name(li, "conv2.b").c_str());
            x = codec_wt_pos_resblock(ctx_eval, x, n1_w, n1_b, c1_w, c1_b, n2_w, n2_b, c2_w, c2_b);
            if (x == nullptr) {
                return false;
            }
        }

        lm_ggml_tensor * gn_w = lm_ggml_new_tensor_1d(ctx_eval, LM_GGML_TYPE_F32, p->backbone_dim);
        lm_ggml_set_name(gn_w, codec_wt_decode_pos_name(5, "w").c_str());
        lm_ggml_tensor * gn_b = lm_ggml_new_tensor_1d(ctx_eval, LM_GGML_TYPE_F32, p->backbone_dim);
        lm_ggml_set_name(gn_b, codec_wt_decode_pos_name(5, "b").c_str());
        x = codec_wt_pos_group_norm(ctx_eval, x, gn_w, gn_b);
        if (x == nullptr) {
            return false;
        }
    }

    lm_ggml_tensor * t_inln_w = lm_ggml_new_tensor_1d(ctx_eval, LM_GGML_TYPE_F32, p->backbone_dim);
    lm_ggml_set_name(t_inln_w, codec_wt_decode_norm_w_name().c_str());
    lm_ggml_tensor * t_inln_b = lm_ggml_new_tensor_1d(ctx_eval, LM_GGML_TYPE_F32, p->backbone_dim);
    lm_ggml_set_name(t_inln_b, codec_wt_decode_norm_b_name().c_str());

    lm_ggml_tensor * x_ct = lm_ggml_cont(ctx_eval, lm_ggml_transpose(ctx_eval, x)); // [c, t]
    x_ct = codec_wt_layer_norm_ct(ctx_eval, x_ct, t_inln_w, t_inln_b);
    if (x_ct == nullptr) {
        return false;
    }

    for (int32_t li = 0; li < p->n_convnext; ++li) {
        lm_ggml_tensor * res_tc = lm_ggml_cont(ctx_eval, lm_ggml_transpose(ctx_eval, x_ct));

        lm_ggml_tensor * t_dw_w = lm_ggml_new_tensor_3d(ctx_eval, LM_GGML_TYPE_F32, 7, 1, p->backbone_dim);
        lm_ggml_set_name(t_dw_w, codec_wt_decode_blk_dw_w_name(li).c_str());
        lm_ggml_tensor * t_dw_b = lm_ggml_new_tensor_1d(ctx_eval, LM_GGML_TYPE_F32, p->backbone_dim);
        lm_ggml_set_name(t_dw_b, codec_wt_decode_blk_dw_b_name(li).c_str());
        lm_ggml_tensor * x_dw = codec_conv1d_depthwise(ctx_eval, res_tc, t_dw_w, t_dw_b, 1, 1, 3);
        if (x_dw == nullptr) {
            return false;
        }

        lm_ggml_tensor * t_lnw = lm_ggml_new_tensor_1d(ctx_eval, LM_GGML_TYPE_F32, p->backbone_dim);
        lm_ggml_set_name(t_lnw, codec_wt_decode_blk_ln_w_name(li).c_str());
        lm_ggml_tensor * t_lnb = lm_ggml_new_tensor_1d(ctx_eval, LM_GGML_TYPE_F32, p->backbone_dim);
        lm_ggml_set_name(t_lnb, codec_wt_decode_blk_ln_b_name(li).c_str());
        lm_ggml_tensor * x_blk_ct = codec_wt_layer_norm_ct(ctx_eval, lm_ggml_cont(ctx_eval, lm_ggml_transpose(ctx_eval, x_dw)), t_lnw, t_lnb);
        if (x_blk_ct == nullptr) {
            return false;
        }

        lm_ggml_tensor * t_pw1_w = lm_ggml_new_tensor_2d(ctx_eval, LM_GGML_TYPE_F32, p->backbone_dim, p->backbone_intermediate);
        lm_ggml_set_name(t_pw1_w, codec_wt_decode_blk_pw1_w_name(li).c_str());
        lm_ggml_tensor * t_pw1_b = lm_ggml_new_tensor_1d(ctx_eval, LM_GGML_TYPE_F32, p->backbone_intermediate);
        lm_ggml_set_name(t_pw1_b, codec_wt_decode_blk_pw1_b_name(li).c_str());
        lm_ggml_tensor * x_pw = codec_op_linear(ctx_eval, x_blk_ct, t_pw1_w, t_pw1_b);
        if (x_pw == nullptr) {
            return false;
        }
        x_pw = lm_ggml_gelu_erf(ctx_eval, x_pw);

        lm_ggml_tensor * t_pw2_w = lm_ggml_new_tensor_2d(ctx_eval, LM_GGML_TYPE_F32, p->backbone_intermediate, p->backbone_dim);
        lm_ggml_set_name(t_pw2_w, codec_wt_decode_blk_pw2_w_name(li).c_str());
        lm_ggml_tensor * t_pw2_b = lm_ggml_new_tensor_1d(ctx_eval, LM_GGML_TYPE_F32, p->backbone_dim);
        lm_ggml_set_name(t_pw2_b, codec_wt_decode_blk_pw2_b_name(li).c_str());
        lm_ggml_tensor * x_pw2 = codec_op_linear(ctx_eval, x_pw, t_pw2_w, t_pw2_b);
        if (x_pw2 == nullptr) {
            return false;
        }

        lm_ggml_tensor * t_gamma = lm_ggml_new_tensor_1d(ctx_eval, LM_GGML_TYPE_F32, p->backbone_dim);
        lm_ggml_set_name(t_gamma, codec_wt_decode_blk_gamma_name(li).c_str());
        x_pw2 = codec_op_channel_scale(ctx_eval, x_pw2, t_gamma);
        if (x_pw2 == nullptr) {
            return false;
        }
        x_ct = lm_ggml_add(ctx_eval, lm_ggml_cont(ctx_eval, lm_ggml_transpose(ctx_eval, res_tc)), x_pw2);
    }

    lm_ggml_tensor * t_fln_w = lm_ggml_new_tensor_1d(ctx_eval, LM_GGML_TYPE_F32, p->backbone_dim);
    lm_ggml_set_name(t_fln_w, codec_wt_decode_final_ln_w_name().c_str());
    lm_ggml_tensor * t_fln_b = lm_ggml_new_tensor_1d(ctx_eval, LM_GGML_TYPE_F32, p->backbone_dim);
    lm_ggml_set_name(t_fln_b, codec_wt_decode_final_ln_b_name().c_str());
    x_ct = codec_wt_layer_norm_ct(ctx_eval, x_ct, t_fln_w, t_fln_b);
    if (x_ct == nullptr) {
        return false;
    }

    lm_ggml_tensor * t_head_w = lm_ggml_new_tensor_2d(ctx_eval, LM_GGML_TYPE_F32, p->backbone_dim, p->head_out_dim);
    lm_ggml_set_name(t_head_w, codec_wt_decode_head_w_name().c_str());
    lm_ggml_tensor * t_head_b = lm_ggml_new_tensor_1d(ctx_eval, LM_GGML_TYPE_F32, p->head_out_dim);
    lm_ggml_set_name(t_head_b, codec_wt_decode_head_b_name().c_str());
    lm_ggml_tensor * t_head = codec_op_linear(ctx_eval, x_ct, t_head_w, t_head_b); // [out_dim, t]
    if (t_head == nullptr) {
        return false;
    }

    lm_ggml_tensor * t_out = lm_ggml_cont(ctx_eval, t_head);
    lm_ggml_set_name(t_out, "wt.decode.head.out");
    *out = t_out;
    return true;
}

struct wt_encode_build {
    int32_t n_in;
    int32_t hop;
    int32_t n_q;
    int32_t codebook_dim;
    int32_t codebook_size;
};

static std::string codec_wt_encode_pad_left_name(int32_t id) {
    return "wt.encode.pad." + std::to_string(id) + ".left";
}

static std::string codec_wt_encode_pad_right_name(int32_t id) {
    return "wt.encode.pad." + std::to_string(id) + ".right";
}

static int32_t codec_wt_extra_padding_for_conv1d(int32_t length, int32_t kernel_eff, int32_t stride, int32_t padding_total) {
    const double n_frames = ((double) (length - kernel_eff + padding_total)) / (double) stride + 1.0;
    const int32_t n_frames_ceil = (int32_t) std::ceil(n_frames);
    const int32_t ideal_length = (n_frames_ceil - 1) * stride + (kernel_eff - padding_total);
    return ideal_length - length;
}

static lm_ggml_tensor * codec_wt_pad1d_reflect(
    lm_ggml_context * ctx_eval,
    lm_ggml_tensor * x,
    int32_t pad_left,
    int32_t pad_right,
    int32_t pad_id) {

    if (ctx_eval == nullptr || x == nullptr || pad_left < 0 || pad_right < 0) {
        return nullptr;
    }
    if (pad_left == 0 && pad_right == 0) {
        return x;
    }

    int32_t t = (int32_t) x->ne[0];
    int32_t extra_reflect = 0;
    const int32_t max_pad = std::max(pad_left, pad_right);
    if (t <= max_pad) {
        extra_reflect = max_pad - t + 1;
        x = codec_op_pad_1d(ctx_eval, x, 0, extra_reflect);
        if (x == nullptr) {
            return nullptr;
        }
        t += extra_reflect;
    }

    lm_ggml_tensor * x_ct = lm_ggml_cont(ctx_eval, lm_ggml_transpose(ctx_eval, x)); // [c, t]
    if (x_ct == nullptr) {
        return nullptr;
    }

    lm_ggml_tensor * out = x;

    if (pad_left > 0) {
        lm_ggml_tensor * idx_left = lm_ggml_new_tensor_1d(ctx_eval, LM_GGML_TYPE_I32, pad_left);
        lm_ggml_set_name(idx_left, codec_wt_encode_pad_left_name(pad_id).c_str());
        lm_ggml_tensor * left_ct = lm_ggml_get_rows(ctx_eval, x_ct, idx_left); // [c, pad_left]
        if (left_ct == nullptr) {
            return nullptr;
        }
        lm_ggml_tensor * left_tc = lm_ggml_cont(ctx_eval, lm_ggml_transpose(ctx_eval, left_ct));
        out = lm_ggml_concat(ctx_eval, left_tc, out, 0);
    }

    if (pad_right > 0) {
        lm_ggml_tensor * idx_right = lm_ggml_new_tensor_1d(ctx_eval, LM_GGML_TYPE_I32, pad_right);
        lm_ggml_set_name(idx_right, codec_wt_encode_pad_right_name(pad_id).c_str());
        lm_ggml_tensor * right_ct = lm_ggml_get_rows(ctx_eval, x_ct, idx_right); // [c, pad_right]
        if (right_ct == nullptr) {
            return nullptr;
        }
        lm_ggml_tensor * right_tc = lm_ggml_cont(ctx_eval, lm_ggml_transpose(ctx_eval, right_ct));
        out = lm_ggml_concat(ctx_eval, out, right_tc, 0);
    }

    if (extra_reflect > 0) {
        out = codec_op_crop_1d(ctx_eval, out, 0, extra_reflect);
    }

    return out == nullptr ? nullptr : lm_ggml_cont(ctx_eval, out);
}

static lm_ggml_tensor * codec_wt_sconv1d(
    lm_ggml_context * ctx_eval,
    lm_ggml_tensor * x,
    int32_t in_ch,
    int32_t out_ch,
    int32_t kernel,
    int32_t stride,
    int32_t dilation,
    int32_t * pad_id,
    const std::string & w_name,
    const std::string & b_name) {

    if (ctx_eval == nullptr || x == nullptr || pad_id == nullptr) {
        return nullptr;
    }
    lm_ggml_tensor * t_w = lm_ggml_new_tensor_3d(ctx_eval, LM_GGML_TYPE_F32, kernel, in_ch, out_ch);
    lm_ggml_set_name(t_w, w_name.c_str());
    lm_ggml_tensor * t_b = lm_ggml_new_tensor_1d(ctx_eval, LM_GGML_TYPE_F32, out_ch);
    lm_ggml_set_name(t_b, b_name.c_str());

    const int32_t kernel_eff = (kernel - 1) * dilation + 1;
    const int32_t padding_total = kernel_eff - stride;
    const int32_t extra_padding = codec_wt_extra_padding_for_conv1d((int32_t) x->ne[0], kernel_eff, stride, padding_total);
    int32_t pad_right = padding_total / 2;
    int32_t pad_left = padding_total - pad_right;
    pad_right += extra_padding;

    const int32_t id = (*pad_id)++;
    lm_ggml_tensor * x_pad = codec_wt_pad1d_reflect(ctx_eval, x, pad_left, pad_right, id);
    if (x_pad == nullptr) {
        return nullptr;
    }
    return codec_conv1d(ctx_eval, x_pad, t_w, t_b, stride, dilation, 0);
}

static std::string codec_wt_encode_conv_w_name(int32_t li, const char * suffix) {
    return "wt.encode.l" + std::to_string(li) + "." + suffix;
}

static std::string codec_wt_encode_resblock_name(int32_t ri, const char * suffix) {
    return "wt.encode.rb" + std::to_string(ri) + "." + suffix;
}

static std::string codec_wt_encode_down_w_name(int32_t di, const char * suffix) {
    return "wt.encode.ds" + std::to_string(di) + "." + suffix;
}

static std::string codec_wt_encode_lstm_name(int32_t li, const char * suffix) {
    return "wt.encode.lstm" + std::to_string(li) + "." + suffix;
}

static lm_ggml_tensor * codec_wt_encode_resblock(
    lm_ggml_context * ctx_eval,
    lm_ggml_tensor * x,
    int32_t dim,
    int32_t res_id,
    int32_t * pad_id) {

    if (ctx_eval == nullptr || x == nullptr || pad_id == nullptr) {
        return nullptr;
    }
    const int32_t hidden = dim / 2;
    lm_ggml_tensor * h = lm_ggml_elu(ctx_eval, x);
    h = codec_wt_sconv1d(
        ctx_eval,
        h,
        dim,
        hidden,
        /*kernel=*/3,
        /*stride=*/1,
        /*dilation=*/1,
        pad_id,
        codec_wt_encode_resblock_name(res_id, "c1.w"),
        codec_wt_encode_resblock_name(res_id, "c1.b"));
    if (h == nullptr) {
        return nullptr;
    }
    h = lm_ggml_elu(ctx_eval, h);
    h = codec_wt_sconv1d(
        ctx_eval,
        h,
        hidden,
        dim,
        /*kernel=*/1,
        /*stride=*/1,
        /*dilation=*/1,
        pad_id,
        codec_wt_encode_resblock_name(res_id, "c2.w"),
        codec_wt_encode_resblock_name(res_id, "c2.b"));
    if (h == nullptr) {
        return nullptr;
    }

    lm_ggml_tensor * sc = codec_wt_sconv1d(
        ctx_eval,
        x,
        dim,
        dim,
        /*kernel=*/1,
        /*stride=*/1,
        /*dilation=*/1,
        pad_id,
        codec_wt_encode_resblock_name(res_id, "sc.w"),
        codec_wt_encode_resblock_name(res_id, "sc.b"));
    if (sc == nullptr) {
        return nullptr;
    }

    return lm_ggml_add(ctx_eval, sc, h);
}

static lm_ggml_tensor * codec_wt_encode_lstm_layer(
    lm_ggml_context * ctx_eval,
    lm_ggml_tensor * x_tc,
    int32_t dim,
    int32_t layer,
    bool skip) {

    if (ctx_eval == nullptr || x_tc == nullptr || dim <= 0) {
        return nullptr;
    }
    const int32_t t = (int32_t) x_tc->ne[0];
    if (t <= 0) {
        return nullptr;
    }

    lm_ggml_tensor * w_ih = lm_ggml_new_tensor_2d(ctx_eval, LM_GGML_TYPE_F32, dim, 4 * dim);
    lm_ggml_set_name(w_ih, codec_wt_encode_lstm_name(layer, "w_ih").c_str());
    lm_ggml_tensor * w_hh = lm_ggml_new_tensor_2d(ctx_eval, LM_GGML_TYPE_F32, dim, 4 * dim);
    lm_ggml_set_name(w_hh, codec_wt_encode_lstm_name(layer, "w_hh").c_str());
    lm_ggml_tensor * b_ih = lm_ggml_new_tensor_1d(ctx_eval, LM_GGML_TYPE_F32, 4 * dim);
    lm_ggml_set_name(b_ih, codec_wt_encode_lstm_name(layer, "b_ih").c_str());
    lm_ggml_tensor * b_hh = lm_ggml_new_tensor_1d(ctx_eval, LM_GGML_TYPE_F32, 4 * dim);
    lm_ggml_set_name(b_hh, codec_wt_encode_lstm_name(layer, "b_hh").c_str());

    lm_ggml_tensor * x_ct = lm_ggml_cont(ctx_eval, lm_ggml_transpose(ctx_eval, x_tc)); // [dim, t]
    lm_ggml_tensor * y_ct = nullptr;
    lm_ggml_tensor * h_prev = nullptr;
    lm_ggml_tensor * c_prev = nullptr;

    for (int32_t ti = 0; ti < t; ++ti) {
        const size_t offset = (size_t) ti * (size_t) x_ct->nb[1];
        lm_ggml_tensor * x_t = lm_ggml_view_2d(ctx_eval, x_ct, dim, 1, x_ct->nb[1], offset); // [dim, 1]
        x_t = lm_ggml_cont(ctx_eval, x_t);
        if (h_prev == nullptr) {
            h_prev = lm_ggml_scale(ctx_eval, x_t, 0.0f);
            c_prev = lm_ggml_scale(ctx_eval, x_t, 0.0f);
        }

        lm_ggml_tensor * gates = lm_ggml_add(
            ctx_eval,
            lm_ggml_mul_mat(ctx_eval, w_ih, x_t),
            lm_ggml_mul_mat(ctx_eval, w_hh, h_prev));
        lm_ggml_tensor * b_ih2 = lm_ggml_reshape_2d(ctx_eval, b_ih, 4 * dim, 1);
        lm_ggml_tensor * b_hh2 = lm_ggml_reshape_2d(ctx_eval, b_hh, 4 * dim, 1);
        gates = lm_ggml_add(ctx_eval, lm_ggml_add(ctx_eval, gates, b_ih2), b_hh2);

        const size_t gate_stride = (size_t) dim * gates->nb[0];
        lm_ggml_tensor * gate_i = lm_ggml_view_2d(ctx_eval, gates, dim, 1, gates->nb[1], 0);
        lm_ggml_tensor * gate_f = lm_ggml_view_2d(ctx_eval, gates, dim, 1, gates->nb[1], gate_stride);
        lm_ggml_tensor * gate_g = lm_ggml_view_2d(ctx_eval, gates, dim, 1, gates->nb[1], 2 * gate_stride);
        lm_ggml_tensor * gate_o = lm_ggml_view_2d(ctx_eval, gates, dim, 1, gates->nb[1], 3 * gate_stride);

        lm_ggml_tensor * i = lm_ggml_sigmoid(ctx_eval, gate_i);
        lm_ggml_tensor * f = lm_ggml_sigmoid(ctx_eval, gate_f);
        lm_ggml_tensor * g = lm_ggml_tanh(ctx_eval, gate_g);
        lm_ggml_tensor * o = lm_ggml_sigmoid(ctx_eval, gate_o);

        lm_ggml_tensor * c_t = lm_ggml_add(ctx_eval, lm_ggml_mul(ctx_eval, f, c_prev), lm_ggml_mul(ctx_eval, i, g));
        lm_ggml_tensor * h_t = lm_ggml_mul(ctx_eval, o, lm_ggml_tanh(ctx_eval, c_t));

        y_ct = (y_ct == nullptr) ? h_t : lm_ggml_concat(ctx_eval, y_ct, h_t, 1);
        h_prev = h_t;
        c_prev = c_t;
    }

    lm_ggml_tensor * y_tc = lm_ggml_cont(ctx_eval, lm_ggml_transpose(ctx_eval, y_ct));
    if (skip) {
        y_tc = lm_ggml_add(ctx_eval, y_tc, x_tc);
    }
    return y_tc;
}

static bool codec_wt_build_encode(lm_ggml_context * ctx_eval, void * user_data, lm_ggml_tensor ** out) {
    wt_encode_build * p = static_cast<wt_encode_build *>(user_data);
    lm_ggml_tensor * t_pcm = lm_ggml_new_tensor_2d(ctx_eval, LM_GGML_TYPE_F32, p->n_in, 1);
    lm_ggml_set_name(t_pcm, "wt.encode.pcm");

    int32_t pad_id = 0;

    lm_ggml_tensor * x = codec_wt_sconv1d(
        ctx_eval,
        t_pcm,
        /*in_ch=*/1,
        /*out_ch=*/32,
        /*kernel=*/7,
        /*stride=*/1,
        /*dilation=*/1,
        &pad_id,
        codec_wt_encode_conv_w_name(0, "w"),
        codec_wt_encode_conv_w_name(0, "b"));
    if (x == nullptr) {
        return false;
    }

    x = codec_wt_encode_resblock(ctx_eval, x, /*dim=*/32, /*res_id=*/0, &pad_id);
    x = lm_ggml_elu(ctx_eval, x);
    x = codec_wt_sconv1d(
        ctx_eval,
        x,
        /*in_ch=*/32,
        /*out_ch=*/64,
        /*kernel=*/4,
        /*stride=*/2,
        /*dilation=*/1,
        &pad_id,
        codec_wt_encode_down_w_name(0, "w"),
        codec_wt_encode_down_w_name(0, "b"));
    if (x == nullptr) {
        return false;
    }

    x = codec_wt_encode_resblock(ctx_eval, x, /*dim=*/64, /*res_id=*/1, &pad_id);
    x = lm_ggml_elu(ctx_eval, x);
    x = codec_wt_sconv1d(
        ctx_eval,
        x,
        /*in_ch=*/64,
        /*out_ch=*/128,
        /*kernel=*/8,
        /*stride=*/4,
        /*dilation=*/1,
        &pad_id,
        codec_wt_encode_down_w_name(1, "w"),
        codec_wt_encode_down_w_name(1, "b"));
    if (x == nullptr) {
        return false;
    }

    x = codec_wt_encode_resblock(ctx_eval, x, /*dim=*/128, /*res_id=*/2, &pad_id);
    x = lm_ggml_elu(ctx_eval, x);
    x = codec_wt_sconv1d(
        ctx_eval,
        x,
        /*in_ch=*/128,
        /*out_ch=*/256,
        /*kernel=*/10,
        /*stride=*/5,
        /*dilation=*/1,
        &pad_id,
        codec_wt_encode_down_w_name(2, "w"),
        codec_wt_encode_down_w_name(2, "b"));
    if (x == nullptr) {
        return false;
    }

    x = codec_wt_encode_resblock(ctx_eval, x, /*dim=*/256, /*res_id=*/3, &pad_id);
    x = lm_ggml_elu(ctx_eval, x);
    x = codec_wt_sconv1d(
        ctx_eval,
        x,
        /*in_ch=*/256,
        /*out_ch=*/512,
        /*kernel=*/16,
        /*stride=*/8,
        /*dilation=*/1,
        &pad_id,
        codec_wt_encode_down_w_name(3, "w"),
        codec_wt_encode_down_w_name(3, "b"));
    if (x == nullptr) {
        return false;
    }

    x = codec_wt_encode_lstm_layer(ctx_eval, x, /*dim=*/512, /*layer=*/0, /*skip=*/true);
    x = codec_wt_encode_lstm_layer(ctx_eval, x, /*dim=*/512, /*layer=*/1, /*skip=*/true);
    if (x == nullptr) {
        return false;
    }

    x = lm_ggml_elu(ctx_eval, x);
    x = codec_wt_sconv1d(
        ctx_eval,
        x,
        /*in_ch=*/512,
        /*out_ch=*/512,
        /*kernel=*/7,
        /*stride=*/1,
        /*dilation=*/1,
        &pad_id,
        codec_wt_encode_conv_w_name(15, "w"),
        codec_wt_encode_conv_w_name(15, "b"));
    if (x == nullptr) {
        return false;
    }

    lm_ggml_tensor * residual = lm_ggml_cont(ctx_eval, lm_ggml_transpose(ctx_eval, x)); // [dim, t]
    lm_ggml_tensor * tokens = nullptr;
    for (int32_t qi = 0; qi < p->n_q; ++qi) {
        lm_ggml_tensor * t_codebook = lm_ggml_new_tensor_2d(ctx_eval, LM_GGML_TYPE_F32, p->codebook_dim, p->codebook_size);
        lm_ggml_set_name(t_codebook, ("wt.encode.vq.q" + std::to_string(qi) + ".codebook").c_str());
        codec_rvq_layer_result_ggml layer = {};
        if (!codec_rvq_build_layer_ggml(ctx_eval, residual, t_codebook, &layer)) {
            return false;
        }
        residual = layer.residual;

        lm_ggml_tensor * idx2d = lm_ggml_reshape_2d(ctx_eval, layer.indices, layer.indices->ne[0], 1);
        tokens = (tokens == nullptr) ? idx2d : lm_ggml_concat(ctx_eval, tokens, idx2d, 1);
    }

    lm_ggml_tensor * t_out = lm_ggml_cont(ctx_eval, tokens);
    lm_ggml_set_name(t_out, "wt.encode.out");
    *out = t_out;
    return true;
}

struct wt_encode_pad_info {
    int32_t id;
    int32_t length;
    int32_t pad_left;
    int32_t pad_right;
};

static bool codec_wt_copy_conv1d_weight_to_3d(
    codec_context * ctx,
    const std::string & src_name,
    lm_ggml_tensor * dst,
    std::string * err);

static bool codec_wt_copy_linear_weight_to_2d(
    codec_context * ctx,
    const std::string & src_name,
    lm_ggml_tensor * dst,
    std::string * err);

static bool codec_wt_copy_bias_1d(codec_context * ctx, const std::string & src_name, lm_ggml_tensor * dst, std::string * err);

static int32_t codec_wt_encode_pad_for_sconv(
    int32_t length,
    int32_t kernel,
    int32_t stride,
    int32_t dilation,
    int32_t * pad_id,
    std::vector<wt_encode_pad_info> * pads) {

    const int32_t kernel_eff = (kernel - 1) * dilation + 1;
    const int32_t padding_total = kernel_eff - stride;
    const int32_t extra_padding = codec_wt_extra_padding_for_conv1d(length, kernel_eff, stride, padding_total);
    int32_t pad_right = padding_total / 2;
    int32_t pad_left = padding_total - pad_right;
    pad_right += extra_padding;

    if (pads != nullptr) {
        pads->push_back({ *pad_id, length, pad_left, pad_right });
    }
    (*pad_id)++;

    const int32_t padded = length + pad_left + pad_right;
    return (padded - kernel_eff) / stride + 1;
}

static void codec_wt_encode_collect_pad_info(int32_t n_in, std::vector<wt_encode_pad_info> * pads) {
    if (pads == nullptr) {
        return;
    }
    pads->clear();
    int32_t pad_id = 0;
    int32_t t = n_in;

    t = codec_wt_encode_pad_for_sconv(t, 7, 1, 1, &pad_id, pads);
    // Resblock 0 (dim=32): conv1 k=3, conv2 k=1, shortcut k=1
    t = codec_wt_encode_pad_for_sconv(t, 3, 1, 1, &pad_id, pads);
    t = codec_wt_encode_pad_for_sconv(t, 1, 1, 1, &pad_id, pads);
    t = codec_wt_encode_pad_for_sconv(t, 1, 1, 1, &pad_id, pads);
    t = codec_wt_encode_pad_for_sconv(t, 4, 2, 1, &pad_id, pads);

    // Resblock 1 (dim=64)
    t = codec_wt_encode_pad_for_sconv(t, 3, 1, 1, &pad_id, pads);
    t = codec_wt_encode_pad_for_sconv(t, 1, 1, 1, &pad_id, pads);
    t = codec_wt_encode_pad_for_sconv(t, 1, 1, 1, &pad_id, pads);
    t = codec_wt_encode_pad_for_sconv(t, 8, 4, 1, &pad_id, pads);

    // Resblock 2 (dim=128)
    t = codec_wt_encode_pad_for_sconv(t, 3, 1, 1, &pad_id, pads);
    t = codec_wt_encode_pad_for_sconv(t, 1, 1, 1, &pad_id, pads);
    t = codec_wt_encode_pad_for_sconv(t, 1, 1, 1, &pad_id, pads);
    t = codec_wt_encode_pad_for_sconv(t, 10, 5, 1, &pad_id, pads);

    // Resblock 3 (dim=256)
    t = codec_wt_encode_pad_for_sconv(t, 3, 1, 1, &pad_id, pads);
    t = codec_wt_encode_pad_for_sconv(t, 1, 1, 1, &pad_id, pads);
    t = codec_wt_encode_pad_for_sconv(t, 1, 1, 1, &pad_id, pads);
    t = codec_wt_encode_pad_for_sconv(t, 16, 8, 1, &pad_id, pads);

    // Final conv
    t = codec_wt_encode_pad_for_sconv(t, 7, 1, 1, &pad_id, pads);

    (void) t;
}

static bool codec_wt_write_encode_pad_indices(
    codec_context * ctx,
    codec_graph_cache_entry * entry,
    int32_t n_in,
    std::string * err) {

    if (ctx == nullptr || entry == nullptr || n_in <= 0) {
        if (err != nullptr) {
            *err = "invalid WavTokenizer encode pad arguments";
        }
        return false;
    }

    std::vector<wt_encode_pad_info> pads;
    codec_wt_encode_collect_pad_info(n_in, &pads);

    for (const auto & pad : pads) {
        if (pad.pad_left == 0 && pad.pad_right == 0) {
            continue;
        }

        int32_t t = pad.length;
        int32_t extra_reflect = 0;
        const int32_t max_pad = std::max(pad.pad_left, pad.pad_right);
        if (t <= max_pad) {
            extra_reflect = max_pad - t + 1;
            t += extra_reflect;
        }

        if (pad.pad_left > 0) {
            lm_ggml_tensor * t_left = codec_graph_get_tensor(ctx, entry, codec_wt_encode_pad_left_name(pad.id).c_str());
            if (t_left == nullptr) {
                if (err != nullptr) {
                    *err = "missing WavTokenizer encode pad left tensor";
                }
                return false;
            }
            std::vector<int32_t> idx_left((size_t) pad.pad_left, 0);
            for (int32_t i = 0; i < pad.pad_left; ++i) {
                idx_left[(size_t) i] = pad.pad_left - i;
            }
            if (!codec_runtime_write_tensor(t_left, idx_left.data(), idx_left.size() * sizeof(int32_t), err)) {
                return false;
            }
        }

        if (pad.pad_right > 0) {
            lm_ggml_tensor * t_right = codec_graph_get_tensor(ctx, entry, codec_wt_encode_pad_right_name(pad.id).c_str());
            if (t_right == nullptr) {
                if (err != nullptr) {
                    *err = "missing WavTokenizer encode pad right tensor";
                }
                return false;
            }
            std::vector<int32_t> idx_right((size_t) pad.pad_right, 0);
            for (int32_t i = 0; i < pad.pad_right; ++i) {
                idx_right[(size_t) i] = (t - 2) - i;
            }
            if (!codec_runtime_write_tensor(t_right, idx_right.data(), idx_right.size() * sizeof(int32_t), err)) {
                return false;
            }
        }
    }

    return true;
}

static bool codec_wt_write_encode_weights(
    codec_context * ctx,
    codec_graph_cache_entry * entry,
    const wt_encode_build & build,
    std::string * err) {

    if (ctx == nullptr || entry == nullptr) {
        if (err != nullptr) {
            *err = "invalid WavTokenizer encode weight arguments";
        }
        return false;
    }
    auto graph = [ctx, entry](const std::string & name) {
        return codec_graph_get_tensor(ctx, entry, name.c_str());
    };

    if (!codec_wt_copy_conv1d_weight_to_3d(ctx, "enc.model.0.conv.conv.weight", graph(codec_wt_encode_conv_w_name(0, "w")), err) ||
        !codec_wt_copy_bias_1d(ctx, "enc.model.0.conv.conv.bias", graph(codec_wt_encode_conv_w_name(0, "b")), err)) {
        return false;
    }

    const char * const res_blocks[] = { "enc.model.1", "enc.model.4", "enc.model.7", "enc.model.10" };
    for (int32_t ri = 0; ri < 4; ++ri) {
        const std::string p = std::string(res_blocks[ri]) + ".";
        if (!codec_wt_copy_conv1d_weight_to_3d(ctx, p + "block.1.conv.conv.weight", graph(codec_wt_encode_resblock_name(ri, "c1.w")), err) ||
            !codec_wt_copy_bias_1d(ctx, p + "block.1.conv.conv.bias", graph(codec_wt_encode_resblock_name(ri, "c1.b")), err) ||
            !codec_wt_copy_conv1d_weight_to_3d(ctx, p + "block.3.conv.conv.weight", graph(codec_wt_encode_resblock_name(ri, "c2.w")), err) ||
            !codec_wt_copy_bias_1d(ctx, p + "block.3.conv.conv.bias", graph(codec_wt_encode_resblock_name(ri, "c2.b")), err) ||
            !codec_wt_copy_conv1d_weight_to_3d(ctx, p + "shortcut.conv.conv.weight", graph(codec_wt_encode_resblock_name(ri, "sc.w")), err) ||
            !codec_wt_copy_bias_1d(ctx, p + "shortcut.conv.conv.bias", graph(codec_wt_encode_resblock_name(ri, "sc.b")), err)) {
            return false;
        }
    }

    const char * const downs[] = { "enc.model.3", "enc.model.6", "enc.model.9", "enc.model.12" };
    for (int32_t di = 0; di < 4; ++di) {
        const std::string p = std::string(downs[di]) + ".";
        if (!codec_wt_copy_conv1d_weight_to_3d(ctx, p + "conv.conv.weight", graph(codec_wt_encode_down_w_name(di, "w")), err) ||
            !codec_wt_copy_bias_1d(ctx, p + "conv.conv.bias", graph(codec_wt_encode_down_w_name(di, "b")), err)) {
            return false;
        }
    }

    for (int32_t li = 0; li < 2; ++li) {
        const std::string p = "enc.model.13.lstm.";
        const std::string suffix = "_l" + std::to_string(li);
        if (!codec_wt_copy_linear_weight_to_2d(ctx, p + "weight_ih" + suffix, graph(codec_wt_encode_lstm_name(li, "w_ih")), err) ||
            !codec_wt_copy_linear_weight_to_2d(ctx, p + "weight_hh" + suffix, graph(codec_wt_encode_lstm_name(li, "w_hh")), err) ||
            !codec_wt_copy_bias_1d(ctx, p + "bias_ih" + suffix, graph(codec_wt_encode_lstm_name(li, "b_ih")), err) ||
            !codec_wt_copy_bias_1d(ctx, p + "bias_hh" + suffix, graph(codec_wt_encode_lstm_name(li, "b_hh")), err)) {
            return false;
        }
    }

    if (!codec_wt_copy_conv1d_weight_to_3d(ctx, "enc.model.15.conv.conv.weight", graph(codec_wt_encode_conv_w_name(15, "w")), err) ||
        !codec_wt_copy_bias_1d(ctx, "enc.model.15.conv.conv.bias", graph(codec_wt_encode_conv_w_name(15, "b")), err)) {
        return false;
    }

    for (int32_t qi = 0; qi < build.n_q; ++qi) {
        lm_ggml_tensor * t_codebook = graph(std::string("wt.encode.vq.q") + std::to_string(qi) + ".codebook");
        if (t_codebook == nullptr) {
            if (err != nullptr) {
                *err = "missing WavTokenizer encode codebook tensor";
            }
            return false;
        }

        const std::string n0 = "vq.vq.layers." + std::to_string(qi) + "._codebook.embed";
        const std::string n1 = "vq.vq.layers." + std::to_string(qi) + ".codebook.embed";
        lm_ggml_tensor * src = lm_ggml_get_tensor(ctx->model->weights, n0.c_str());
        if (src == nullptr) {
            src = lm_ggml_get_tensor(ctx->model->weights, n1.c_str());
        }
        if (src == nullptr) {
            if (err != nullptr) {
                *err = "missing WavTokenizer codebook tensor";
            }
            return false;
        }
        std::vector<float> cb;
        if (!codec_tensor_as_vec_f32(src, &cb)) {
            if (err != nullptr) {
                *err = "failed reading WavTokenizer codebook tensor";
            }
            return false;
        }
        const int32_t ncb0 = (int32_t) codec_ne(src, 0);
        const int32_t ncb1 = (int32_t) codec_ne(src, 1);
        std::vector<float> cb_dst((size_t) build.codebook_dim * (size_t) build.codebook_size, 0.0f);
        if (ncb0 == build.codebook_dim && ncb1 == build.codebook_size) {
            cb_dst = cb;
        } else if (ncb0 == build.codebook_size && ncb1 == build.codebook_dim) {
            for (int32_t i = 0; i < build.codebook_dim; ++i) {
                for (int32_t j = 0; j < build.codebook_size; ++j) {
                    cb_dst[(size_t) i + (size_t) build.codebook_dim * (size_t) j] =
                        cb[(size_t) j + (size_t) build.codebook_size * (size_t) i];
                }
            }
        } else {
            if (err != nullptr) {
                *err = "unexpected WavTokenizer codebook shape";
            }
            return false;
        }

        if (!codec_runtime_write_tensor(t_codebook, cb_dst.data(), cb_dst.size() * sizeof(float), err)) {
            return false;
        }
    }

    return true;
}

static lm_ggml_tensor * codec_wt_get_tensor(codec_model * model, const std::string & name) {
    if (model == nullptr || model->weights == nullptr) {
        return nullptr;
    }
    return lm_ggml_get_tensor(model->weights, name.c_str());
}

static bool codec_wt_copy_conv1d_weight_to_3d(
    codec_context * ctx,
    const std::string & src_name,
    lm_ggml_tensor * dst,
    std::string * err) {

    lm_ggml_tensor * src = codec_wt_get_tensor(ctx->model, src_name);
    if (src == nullptr) {
        if (err != nullptr) {
            *err = "missing WavTokenizer tensor: " + src_name;
        }
        return false;
    }
    std::vector<float> src_v;
    if (!codec_tensor_as_vec_f32(src, &src_v)) {
        if (err != nullptr) {
            *err = "failed reading WavTokenizer tensor: " + src_name;
        }
        return false;
    }
    const int32_t dk = (int32_t) codec_ne(dst, 0);
    const int32_t din = (int32_t) codec_ne(dst, 1);
    const int32_t dout = (int32_t) codec_ne(dst, 2);
    const int32_t n0 = (int32_t) codec_ne(src, 0);
    const int32_t n1 = (int32_t) codec_ne(src, 1);
    const int32_t n2 = (int32_t) codec_ne(src, 2);
    std::vector<float> dst_v((size_t) dk * (size_t) din * (size_t) dout, 0.0f);

    if (n0 == dk && n1 == din && n2 == dout) {
        dst_v = src_v;
    } else if (n0 == dout && n1 == din && n2 == dk) {
        for (int32_t k = 0; k < dk; ++k) {
            for (int32_t i = 0; i < din; ++i) {
                for (int32_t o = 0; o < dout; ++o) {
                    const size_t src_idx = (size_t) o + (size_t) dout * ((size_t) i + (size_t) din * (size_t) k);
                    const size_t dst_idx = (size_t) k + (size_t) dk * ((size_t) i + (size_t) din * (size_t) o);
                    dst_v[dst_idx] = src_v[src_idx];
                }
            }
        }
    } else {
        if (err != nullptr) {
            *err = "unexpected WavTokenizer conv1d shape: " + src_name;
        }
        return false;
    }
    return codec_runtime_write_tensor(dst, dst_v.data(), dst_v.size() * sizeof(float), err);
}

static bool codec_wt_copy_linear_weight_to_2d(
    codec_context * ctx,
    const std::string & src_name,
    lm_ggml_tensor * dst,
    std::string * err) {

    lm_ggml_tensor * src = codec_wt_get_tensor(ctx->model, src_name);
    if (src == nullptr) {
        if (err != nullptr) {
            *err = "missing WavTokenizer tensor: " + src_name;
        }
        return false;
    }
    std::vector<float> src_v;
    if (!codec_tensor_as_vec_f32(src, &src_v)) {
        if (err != nullptr) {
            *err = "failed reading WavTokenizer tensor: " + src_name;
        }
        return false;
    }
    const int32_t din = (int32_t) codec_ne(dst, 0);
    const int32_t dout = (int32_t) codec_ne(dst, 1);
    const int32_t n0 = (int32_t) codec_ne(src, 0);
    const int32_t n1 = (int32_t) codec_ne(src, 1);
    std::vector<float> dst_v((size_t) din * (size_t) dout, 0.0f);
    if (n0 == din && n1 == dout) {
        dst_v = src_v;
    } else if (n0 == dout && n1 == din) {
        for (int32_t i = 0; i < din; ++i) {
            for (int32_t o = 0; o < dout; ++o) {
                dst_v[(size_t) i + (size_t) din * (size_t) o] = src_v[(size_t) o + (size_t) dout * (size_t) i];
            }
        }
    } else {
        if (err != nullptr) {
            *err = "unexpected WavTokenizer linear shape: " + src_name;
        }
        return false;
    }
    return codec_runtime_write_tensor(dst, dst_v.data(), dst_v.size() * sizeof(float), err);
}

static bool codec_wt_copy_bias_1d(codec_context * ctx, const std::string & src_name, lm_ggml_tensor * dst, std::string * err) {
    lm_ggml_tensor * src = codec_wt_get_tensor(ctx->model, src_name);
    if (src == nullptr) {
        if (err != nullptr) {
            *err = "missing WavTokenizer tensor: " + src_name;
        }
        return false;
    }
    std::vector<float> v;
    if (!codec_tensor_as_vec_f32(src, &v) || (int32_t) v.size() != (int32_t) codec_ne(dst, 0)) {
        if (err != nullptr) {
            *err = "invalid WavTokenizer bias tensor: " + src_name;
        }
        return false;
    }
    return codec_runtime_write_tensor(dst, v.data(), v.size() * sizeof(float), err);
}

static bool codec_wt_copy_embedding_row0(codec_context * ctx, const std::string & src_name, lm_ggml_tensor * dst, std::string * err) {
    lm_ggml_tensor * src = codec_wt_get_tensor(ctx->model, src_name);
    if (src == nullptr) {
        if (err != nullptr) {
            *err = "missing WavTokenizer tensor: " + src_name;
        }
        return false;
    }
    std::vector<float> src_v;
    if (!codec_tensor_as_vec_f32(src, &src_v)) {
        if (err != nullptr) {
            *err = "failed reading WavTokenizer tensor: " + src_name;
        }
        return false;
    }
    const int32_t d = (int32_t) codec_ne(dst, 0);
    const int32_t n0 = (int32_t) codec_ne(src, 0);
    const int32_t n1 = (int32_t) codec_ne(src, 1);
    std::vector<float> out((size_t) d, 0.0f);
    if (n0 == d && n1 >= 1) {
        for (int32_t i = 0; i < d; ++i) {
            out[(size_t) i] = src_v[(size_t) i];
        }
    } else if (n1 == d && n0 >= 1) {
        for (int32_t i = 0; i < d; ++i) {
            out[(size_t) i] = src_v[(size_t) 0 + (size_t) n0 * (size_t) i];
        }
    } else {
        if (err != nullptr) {
            *err = "unexpected WavTokenizer embedding shape: " + src_name;
        }
        return false;
    }
    return codec_runtime_write_tensor(dst, out.data(), out.size() * sizeof(float), err);
}

static bool codec_wt_init_decode_build(codec_context * ctx, int32_t t, int32_t q, wt_decode_build * build, std::string * err) {
    if (ctx == nullptr || ctx->model == nullptr || build == nullptr || t <= 0 || q <= 0) {
        if (err != nullptr) {
            *err = "invalid WavTokenizer decode build arguments";
        }
        return false;
    }
    const codec_wavtokenizer_large & wt = *static_cast<const codec_wavtokenizer_large *>(ctx->model->impl);
    build->t = t;
    build->q = q;
    build->hop = std::max(1, wt.hop_size);
    build->codebook_dim = std::max(1, wt.codebook_dim);
    build->codebook_size = std::max(2, wt.codebook_size);

    lm_ggml_tensor * embed_w = codec_wt_get_tensor(ctx->model, "dec.bb.embed.weight");
    lm_ggml_tensor * embed_b = codec_wt_get_tensor(ctx->model, "dec.bb.embed.bias");
    if (embed_w == nullptr || embed_b == nullptr) {
        if (err != nullptr) {
            *err = "missing WavTokenizer backbone embed tensors";
        }
        return false;
    }
    const int32_t eb0 = (int32_t) codec_ne(embed_w, 0);
    const int32_t eb1 = (int32_t) codec_ne(embed_w, 1);
    const int32_t eb2 = (int32_t) codec_ne(embed_w, 2);
    const int32_t bo = (int32_t) codec_ne(embed_b, 0);
    if (eb2 == bo) {
        build->backbone_dim = eb2;
        build->codebook_dim = eb1;
    } else if (eb0 == bo) {
        build->backbone_dim = eb0;
        build->codebook_dim = eb1;
    } else {
        if (err != nullptr) {
            *err = "unexpected WavTokenizer embed shape";
        }
        return false;
    }

    build->n_convnext = 0;
    for (int32_t li = 0; li < 64; ++li) {
        if (codec_wt_get_tensor(ctx->model, "dec.bb.cnx." + std::to_string(li) + ".dwconv.weight") == nullptr) {
            break;
        }
        build->n_convnext = li + 1;
    }
    if (build->n_convnext <= 0) {
        if (err != nullptr) {
            *err = "no WavTokenizer convnext layers found";
        }
        return false;
    }

    lm_ggml_tensor * pw1 = codec_wt_get_tensor(ctx->model, "dec.bb.cnx.0.pwconv1.weight");
    if (pw1 == nullptr) {
        if (err != nullptr) {
            *err = "missing WavTokenizer pwconv1 tensor";
        }
        return false;
    }
    build->backbone_intermediate = std::max((int32_t) codec_ne(pw1, 0), (int32_t) codec_ne(pw1, 1));

    lm_ggml_tensor * head_b = codec_wt_get_tensor(ctx->model, "dec.head.out.bias");
    if (head_b == nullptr) {
        if (err != nullptr) {
            *err = "missing WavTokenizer head bias tensor";
        }
        return false;
    }
    build->head_out_dim = (int32_t) codec_ne(head_b, 0);
    build->use_adanorm = codec_wt_get_tensor(ctx->model, "dec.bb.norm.scale.weight") != nullptr ? 1 : 0;
    build->use_pos_net = codec_wt_get_tensor(ctx->model, "dec.bb.pos_net.0.conv1.weight") != nullptr ? 1 : 0;
    return true;
}

static bool codec_wt_write_decode_weights(codec_context * ctx, codec_graph_cache_entry * entry, const wt_decode_build & build, std::string * err) {
    auto graph = [&](const std::string & n) { return codec_graph_get_tensor(ctx, entry, n.c_str()); };

    if (!codec_wt_copy_conv1d_weight_to_3d(ctx, "dec.bb.embed.weight", graph(codec_wt_decode_embed_w_name()), err) ||
        !codec_wt_copy_bias_1d(ctx, "dec.bb.embed.bias", graph(codec_wt_decode_embed_b_name()), err)) {
        return false;
    }

    if (build.use_pos_net) {
        for (int32_t li = 0; li < 2; ++li) {
            const std::string p = "dec.bb.pos_net." + std::to_string(li) + ".";
            if (!codec_wt_copy_bias_1d(ctx, p + "norm1.weight", graph(codec_wt_decode_pos_name(li, "norm1.w")), err) ||
                !codec_wt_copy_bias_1d(ctx, p + "norm1.bias", graph(codec_wt_decode_pos_name(li, "norm1.b")), err) ||
                !codec_wt_copy_conv1d_weight_to_3d(ctx, p + "conv1.weight", graph(codec_wt_decode_pos_name(li, "conv1.w")), err) ||
                !codec_wt_copy_bias_1d(ctx, p + "conv1.bias", graph(codec_wt_decode_pos_name(li, "conv1.b")), err) ||
                !codec_wt_copy_bias_1d(ctx, p + "norm2.weight", graph(codec_wt_decode_pos_name(li, "norm2.w")), err) ||
                !codec_wt_copy_bias_1d(ctx, p + "norm2.bias", graph(codec_wt_decode_pos_name(li, "norm2.b")), err) ||
                !codec_wt_copy_conv1d_weight_to_3d(ctx, p + "conv2.weight", graph(codec_wt_decode_pos_name(li, "conv2.w")), err) ||
                !codec_wt_copy_bias_1d(ctx, p + "conv2.bias", graph(codec_wt_decode_pos_name(li, "conv2.b")), err)) {
                return false;
            }
        }

        {
            const std::string p = "dec.bb.pos_net.2.";
            if (!codec_wt_copy_bias_1d(ctx, p + "norm.weight", graph(codec_wt_decode_pos_name(2, "norm.w")), err) ||
                !codec_wt_copy_bias_1d(ctx, p + "norm.bias", graph(codec_wt_decode_pos_name(2, "norm.b")), err) ||
                !codec_wt_copy_conv1d_weight_to_3d(ctx, p + "q.weight", graph(codec_wt_decode_pos_name(2, "q.w")), err) ||
                !codec_wt_copy_bias_1d(ctx, p + "q.bias", graph(codec_wt_decode_pos_name(2, "q.b")), err) ||
                !codec_wt_copy_conv1d_weight_to_3d(ctx, p + "k.weight", graph(codec_wt_decode_pos_name(2, "k.w")), err) ||
                !codec_wt_copy_bias_1d(ctx, p + "k.bias", graph(codec_wt_decode_pos_name(2, "k.b")), err) ||
                !codec_wt_copy_conv1d_weight_to_3d(ctx, p + "v.weight", graph(codec_wt_decode_pos_name(2, "v.w")), err) ||
                !codec_wt_copy_bias_1d(ctx, p + "v.bias", graph(codec_wt_decode_pos_name(2, "v.b")), err) ||
                !codec_wt_copy_conv1d_weight_to_3d(ctx, p + "proj_out.weight", graph(codec_wt_decode_pos_name(2, "proj_out.w")), err) ||
                !codec_wt_copy_bias_1d(ctx, p + "proj_out.bias", graph(codec_wt_decode_pos_name(2, "proj_out.b")), err)) {
                return false;
            }
        }

        for (int32_t li = 3; li < 5; ++li) {
            const std::string p = "dec.bb.pos_net." + std::to_string(li) + ".";
            if (!codec_wt_copy_bias_1d(ctx, p + "norm1.weight", graph(codec_wt_decode_pos_name(li, "norm1.w")), err) ||
                !codec_wt_copy_bias_1d(ctx, p + "norm1.bias", graph(codec_wt_decode_pos_name(li, "norm1.b")), err) ||
                !codec_wt_copy_conv1d_weight_to_3d(ctx, p + "conv1.weight", graph(codec_wt_decode_pos_name(li, "conv1.w")), err) ||
                !codec_wt_copy_bias_1d(ctx, p + "conv1.bias", graph(codec_wt_decode_pos_name(li, "conv1.b")), err) ||
                !codec_wt_copy_bias_1d(ctx, p + "norm2.weight", graph(codec_wt_decode_pos_name(li, "norm2.w")), err) ||
                !codec_wt_copy_bias_1d(ctx, p + "norm2.bias", graph(codec_wt_decode_pos_name(li, "norm2.b")), err) ||
                !codec_wt_copy_conv1d_weight_to_3d(ctx, p + "conv2.weight", graph(codec_wt_decode_pos_name(li, "conv2.w")), err) ||
                !codec_wt_copy_bias_1d(ctx, p + "conv2.bias", graph(codec_wt_decode_pos_name(li, "conv2.b")), err)) {
                return false;
            }
        }

        if (!codec_wt_copy_bias_1d(ctx, "dec.bb.pos_net.5.weight", graph(codec_wt_decode_pos_name(5, "w")), err) ||
            !codec_wt_copy_bias_1d(ctx, "dec.bb.pos_net.5.bias", graph(codec_wt_decode_pos_name(5, "b")), err)) {
            return false;
        }
    }

    if (build.use_adanorm) {
        if (!codec_wt_copy_embedding_row0(ctx, "dec.bb.norm.scale.weight", graph(codec_wt_decode_norm_w_name()), err) ||
            !codec_wt_copy_embedding_row0(ctx, "dec.bb.norm.shift.weight", graph(codec_wt_decode_norm_b_name()), err)) {
            return false;
        }
    } else {
        if (!codec_wt_copy_bias_1d(ctx, "dec.bb.norm.weight", graph(codec_wt_decode_norm_w_name()), err) ||
            !codec_wt_copy_bias_1d(ctx, "dec.bb.norm.bias", graph(codec_wt_decode_norm_b_name()), err)) {
            return false;
        }
    }

    for (int32_t li = 0; li < build.n_convnext; ++li) {
        const std::string p = "dec.bb.cnx." + std::to_string(li) + ".";
        if (!codec_wt_copy_conv1d_weight_to_3d(ctx, p + "dwconv.weight", graph(codec_wt_decode_blk_dw_w_name(li)), err) ||
            !codec_wt_copy_bias_1d(ctx, p + "dwconv.bias", graph(codec_wt_decode_blk_dw_b_name(li)), err) ||
            !codec_wt_copy_linear_weight_to_2d(ctx, p + "pwconv1.weight", graph(codec_wt_decode_blk_pw1_w_name(li)), err) ||
            !codec_wt_copy_bias_1d(ctx, p + "pwconv1.bias", graph(codec_wt_decode_blk_pw1_b_name(li)), err) ||
            !codec_wt_copy_linear_weight_to_2d(ctx, p + "pwconv2.weight", graph(codec_wt_decode_blk_pw2_w_name(li)), err) ||
            !codec_wt_copy_bias_1d(ctx, p + "pwconv2.bias", graph(codec_wt_decode_blk_pw2_b_name(li)), err) ||
            !codec_wt_copy_bias_1d(ctx, p + "gamma", graph(codec_wt_decode_blk_gamma_name(li)), err)) {
            return false;
        }
        if (build.use_adanorm) {
            if (!codec_wt_copy_embedding_row0(ctx, p + "norm.scale.weight", graph(codec_wt_decode_blk_ln_w_name(li)), err) ||
                !codec_wt_copy_embedding_row0(ctx, p + "norm.shift.weight", graph(codec_wt_decode_blk_ln_b_name(li)), err)) {
                return false;
            }
        } else {
            if (!codec_wt_copy_bias_1d(ctx, p + "norm.weight", graph(codec_wt_decode_blk_ln_w_name(li)), err) ||
                !codec_wt_copy_bias_1d(ctx, p + "norm.bias", graph(codec_wt_decode_blk_ln_b_name(li)), err)) {
                return false;
            }
        }
    }

    if (!codec_wt_copy_bias_1d(ctx, "dec.bb.fln.weight", graph(codec_wt_decode_final_ln_w_name()), err) ||
        !codec_wt_copy_bias_1d(ctx, "dec.bb.fln.bias", graph(codec_wt_decode_final_ln_b_name()), err) ||
        !codec_wt_copy_linear_weight_to_2d(ctx, "dec.head.out.weight", graph(codec_wt_decode_head_w_name()), err) ||
        !codec_wt_copy_bias_1d(ctx, "dec.head.out.bias", graph(codec_wt_decode_head_b_name()), err)) {
        return false;
    }
    return true;
}

static bool codec_wt_istft_from_head(
    const std::vector<float> & head,
    int32_t out_dim,
    int32_t n_frames,
    int32_t hop,
    std::vector<float> * out_pcm,
    std::string * err) {

    if (out_pcm == nullptr || out_dim <= 0 || n_frames <= 0 || hop <= 0 || (out_dim % 2) != 0) {
        if (err != nullptr) {
            *err = "invalid WavTokenizer ISTFT arguments";
        }
        return false;
    }
    const int32_t n_bins = out_dim / 2;
    const int32_t n_fft = 2 * (n_bins - 1);
    const float pi = 3.14159265358979323846f;
    if (n_fft <= 0) {
        if (err != nullptr) {
            *err = "invalid WavTokenizer head output dimension";
        }
        return false;
    }
    const int32_t pad = (n_fft - hop) / 2;
    const int32_t out_size = (n_frames - 1) * hop + n_fft;
    std::vector<float> window((size_t) n_fft, 0.0f);
    for (int32_t n = 0; n < n_fft; ++n) {
        window[(size_t) n] = 0.5f - 0.5f * std::cos(2.0f * pi * (float) n / (float) (n_fft - 1));
    }
    std::vector<float> y((size_t) out_size, 0.0f);
    std::vector<float> env((size_t) out_size, 0.0f);
    std::vector<float> frame((size_t) n_fft, 0.0f);

    for (int32_t ti = 0; ti < n_frames; ++ti) {
        for (int32_t n = 0; n < n_fft; ++n) {
            float sum = 0.0f;
            float mag0 = std::exp(head[(size_t) 0 + (size_t) out_dim * (size_t) ti]);
            if (mag0 > 1e2f) {
                mag0 = 1e2f;
            }
            const float re0 = mag0 * std::cos(head[(size_t) n_bins + (size_t) out_dim * (size_t) ti]);
            sum += re0;
            float magn = std::exp(head[(size_t) (n_bins - 1) + (size_t) out_dim * (size_t) ti]);
            if (magn > 1e2f) {
                magn = 1e2f;
            }
            const float ren = magn * std::cos(head[(size_t) (2 * n_bins - 1) + (size_t) out_dim * (size_t) ti]);
            sum += ren * ((n & 1) ? -1.0f : 1.0f);
            for (int32_t k = 1; k < n_bins - 1; ++k) {
                float mag = std::exp(head[(size_t) k + (size_t) out_dim * (size_t) ti]);
                if (mag > 1e2f) {
                    mag = 1e2f;
                }
                const float ph = head[(size_t) (n_bins + k) + (size_t) out_dim * (size_t) ti];
                const float re = mag * std::cos(ph);
                const float im = mag * std::sin(ph);
                const float ang = 2.0f * pi * (float) k * (float) n / (float) n_fft;
                sum += 2.0f * (re * std::cos(ang) - im * std::sin(ang));
            }
            frame[(size_t) n] = (sum / (float) n_fft) * window[(size_t) n];
        }
        const int32_t off = ti * hop;
        for (int32_t n = 0; n < n_fft; ++n) {
            y[(size_t) (off + n)] += frame[(size_t) n];
            env[(size_t) (off + n)] += window[(size_t) n] * window[(size_t) n];
        }
    }

    const int32_t out_begin = std::max(0, pad);
    const int32_t out_end = std::max(out_begin, out_size - pad);
    out_pcm->assign((size_t) (out_end - out_begin), 0.0f);
    for (int32_t i = out_begin; i < out_end; ++i) {
        const float den = env[(size_t) i] > 1e-11f ? env[(size_t) i] : 1.0f;
        (*out_pcm)[(size_t) (i - out_begin)] = y[(size_t) i] / den;
    }
    return true;
}

static enum codec_status codec_wt_decode_graph(
    struct codec_context * ctx,
    const struct codec_token_buffer * tokens,
    int32_t use_n_q,
    struct codec_pcm_buffer * out_pcm) {

    codec_wavtokenizer_large & wt = *static_cast<codec_wavtokenizer_large *>(ctx->model->impl);
    if (tokens == nullptr || tokens->data == nullptr || tokens->n_frames <= 0 || tokens->n_q < use_n_q) {
        codec_context_set_error(ctx, "invalid WavTokenizer token buffer");
        return CODEC_STATUS_INVALID_ARG;
    }

    const int32_t t = tokens->n_frames;
    const int32_t q = use_n_q;
    const int32_t hop = std::max(1, wt.hop_size);
    const size_t mem = 32 * 1024 * 1024 + (size_t) t * (size_t) q * sizeof(float) * 16;
    codec_graph_eval_guard eval_guard(ctx);
    std::string err;

    wt_decode_build build = {};
    if (!codec_wt_init_decode_build(ctx, t, q, &build, &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }
    codec_graph_cache_entry * entry = nullptr;
    err.clear();
    if (!codec_graph_cache_get_or_build(
            ctx,
            { CODEC_GRAPH_WT_DECODE, /*n_frames=*/t, /*n_q=*/q, /*hop=*/hop, /*n_in=*/0, /*latent_dim=*/0 },
            mem,
            codec_wt_build_decode,
            &build,
            sizeof(build),
            &entry,
            &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    lm_ggml_tensor * t_tok = codec_graph_get_tensor(ctx, entry, "wt.decode.tok");
    lm_ggml_tensor * t_out = codec_graph_get_tensor(ctx, entry, "wt.decode.head.out");
    if (t_tok == nullptr || t_out == nullptr) {
        codec_context_set_error(ctx, "cached WavTokenizer decode graph is invalid");
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
    for (int32_t qi = 0; qi < q; ++qi) {
        lm_ggml_tensor * t_codebook = codec_graph_get_tensor(ctx, entry, codec_wt_decode_codebook_tensor_name(qi).c_str());
        if (t_codebook == nullptr) {
            codec_context_set_error(ctx, "cached WavTokenizer decode graph is missing codebook tensors");
            return CODEC_STATUS_INTERNAL_ERROR;
        }

        const std::string n0 = "vq.vq.layers." + std::to_string(qi) + "._codebook.embed";
        const std::string n1 = "vq.vq.layers." + std::to_string(qi) + ".codebook.embed";
        lm_ggml_tensor * src = lm_ggml_get_tensor(ctx->model->weights, n0.c_str());
        if (src == nullptr) {
            src = lm_ggml_get_tensor(ctx->model->weights, n1.c_str());
        }
        if (src == nullptr) {
            codec_context_set_error(ctx, "missing WavTokenizer codebook tensor");
            return CODEC_STATUS_INTERNAL_ERROR;
        }
        std::vector<float> cb;
        if (!codec_tensor_as_vec_f32(src, &cb)) {
            codec_context_set_error(ctx, "failed reading WavTokenizer codebook tensor");
            return CODEC_STATUS_INTERNAL_ERROR;
        }
        const int32_t ncb0 = (int32_t) codec_ne(src, 0);
        const int32_t ncb1 = (int32_t) codec_ne(src, 1);
        std::vector<float> cb_dst((size_t) build.codebook_dim * (size_t) build.codebook_size, 0.0f);
        if (ncb0 == build.codebook_dim && ncb1 == build.codebook_size) {
            cb_dst = cb;
        } else if (ncb0 == build.codebook_size && ncb1 == build.codebook_dim) {
            for (int32_t i = 0; i < build.codebook_dim; ++i) {
                for (int32_t j = 0; j < build.codebook_size; ++j) {
                    cb_dst[(size_t) i + (size_t) build.codebook_dim * (size_t) j] =
                        cb[(size_t) j + (size_t) build.codebook_size * (size_t) i];
                }
            }
        } else {
            codec_context_set_error(ctx, "unexpected WavTokenizer codebook shape");
            return CODEC_STATUS_INTERNAL_ERROR;
        }

        if (!codec_runtime_write_tensor(t_codebook, cb_dst.data(), cb_dst.size() * sizeof(float), &err)) {
            codec_context_set_error(ctx, err);
            return CODEC_STATUS_INTERNAL_ERROR;
        }
    }

    if (!codec_wt_write_decode_weights(ctx, entry, build, &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
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
    std::vector<float> pcm_v;
    if (!codec_wt_istft_from_head(head, build.head_out_dim, t, hop, &pcm_v, &err)) {
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
    out_pcm->sample_rate = wt.sample_rate;
    out_pcm->n_channels = 1;

    return CODEC_STATUS_SUCCESS;
}

enum codec_status codec_wavtokenizer_decode(
    struct codec_context * ctx,
    const struct codec_token_buffer * tokens,
    struct codec_pcm_buffer * out_pcm,
    struct codec_decode_params params) {

    codec_wavtokenizer_large & wt = *static_cast<codec_wavtokenizer_large *>(ctx->model->impl);
    if (!wt.has_decoder) {
        codec_context_set_error(ctx, "model metadata indicates no decoder");
        return CODEC_STATUS_INVALID_STATE;
    }

    const int32_t model_n_q = std::max(1, wt.n_q);
    const int32_t use_n_q = params.n_q == 0 ? model_n_q : params.n_q;
    if (params.n_q < 0 || use_n_q < 1 || use_n_q > model_n_q) {
        codec_context_set_error(ctx, "WavTokenizer decode n_q must be 0 or in [1, model_n_q]");
        return CODEC_STATUS_INVALID_ARG;
    }

    return codec_wt_decode_graph(ctx, tokens, use_n_q, out_pcm);
}

enum codec_status codec_wavtokenizer_encode(
    struct codec_context * ctx,
    const std::vector<float> & pcm,
    struct codec_token_buffer * out_tokens,
    struct codec_encode_params params) {

    codec_wavtokenizer_large & wt = *static_cast<codec_wavtokenizer_large *>(ctx->model->impl);
    if (!wt.has_encoder) {
        codec_context_set_error(ctx, "model metadata indicates no encoder");
        return CODEC_STATUS_INVALID_STATE;
    }
    if (pcm.empty()) {
        codec_context_set_error(ctx, "empty pcm");
        return CODEC_STATUS_INVALID_ARG;
    }

    const int32_t model_n_q = std::max(1, wt.n_q);
    const int32_t use_n_q = params.n_q == 0 ? model_n_q : params.n_q;
    if (params.n_q < 0 || use_n_q < 1 || use_n_q > model_n_q) {
        codec_context_set_error(ctx, "WavTokenizer encode n_q must be 0 or in [1, model_n_q]");
        return CODEC_STATUS_INVALID_ARG;
    }

    const int32_t hop = std::max(1, params.hop_size > 0 ? params.hop_size : wt.hop_size);
    const int32_t n_in = (int32_t) pcm.size();

    const size_t mem = 32 * 1024 * 1024 + (size_t) n_in * sizeof(float) * 16;
    codec_graph_eval_guard eval_guard(ctx);
    const int32_t codebook_dim = std::max(1, wt.codebook_dim);
    const int32_t codebook_size = std::max(2, wt.codebook_size);
    wt_encode_build build = { n_in, hop, use_n_q, codebook_dim, codebook_size };
    codec_graph_cache_entry * entry = nullptr;
    std::string err;
    if (!codec_graph_cache_get_or_build(
            ctx,
            { CODEC_GRAPH_WT_ENCODE, /*n_frames=*/0, /*n_q=*/use_n_q, /*hop=*/hop, /*n_in=*/n_in, /*latent_dim=*/codebook_dim },
            mem,
            codec_wt_build_encode,
            &build,
            sizeof(build),
            &entry,
            &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    lm_ggml_tensor * t_pcm = codec_graph_get_tensor(ctx, entry, "wt.encode.pcm");
    lm_ggml_tensor * t_out = codec_graph_get_tensor(ctx, entry, "wt.encode.out");
    if (t_pcm == nullptr || t_out == nullptr) {
        codec_context_set_error(ctx, "cached WavTokenizer encode graph is invalid");
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

    if (!codec_wt_write_encode_pad_indices(ctx, entry, n_in, &err) ||
        !codec_wt_write_encode_weights(ctx, entry, build, &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    const int32_t n_threads = ctx->model->n_threads > 0 ? ctx->model->n_threads : 1;
    if (!codec_graph_compute(ctx, entry, n_threads, &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    const int32_t n_frames = (int32_t) t_out->ne[0];
    const int32_t n_q = (int32_t) t_out->ne[1];
    std::vector<int32_t> tok((size_t) n_frames * (size_t) n_q, 0);
    if (!codec_runtime_read_tensor(t_out, tok.data(), tok.size() * sizeof(int32_t), &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    int32_t * data = static_cast<int32_t *>(std::malloc((size_t) n_frames * (size_t) n_q * sizeof(int32_t)));
    if (data == nullptr) {
        codec_context_set_error(ctx, "failed to allocate token output");
        return CODEC_STATUS_INTERNAL_ERROR;
    }
    std::memcpy(data, tok.data(), tok.size() * sizeof(int32_t));

    codec_token_buffer_reset(out_tokens);
    out_tokens->data = data;
    out_tokens->n_tokens = n_frames * n_q;
    out_tokens->n_frames = n_frames;
    out_tokens->n_q = n_q;
    out_tokens->codebook_size = codebook_size;
    out_tokens->sample_rate = wt.sample_rate;
    out_tokens->hop_size = hop;

    return CODEC_STATUS_SUCCESS;
}

static void * codec_wavtokenizer_create_impl() {
    return new (std::nothrow) codec_wavtokenizer_large();
}

static void codec_wavtokenizer_destroy_impl(void * ptr) {
    delete static_cast<codec_wavtokenizer_large *>(ptr);
}

static enum codec_status codec_wavtokenizer_encode_wrap(
    struct codec_context * ctx,
    const std::vector<float> & pcm,
    struct codec_token_buffer * out_tokens,
    struct codec_latent_buffer * /*out_latent*/,
    struct codec_encode_params params) {
    return codec_wavtokenizer_encode(ctx, pcm, out_tokens, params);
}

static enum codec_status codec_wavtokenizer_decode_wrap(
    struct codec_context * ctx,
    const struct codec_token_buffer * tokens,
    struct codec_pcm_buffer * out_pcm,
    struct codec_decode_params params) {
    return codec_wavtokenizer_decode(ctx, tokens, out_pcm, params);
}

const struct codec_model_vtable * codec_wavtokenizer_vtable() {
    static const codec_model_vtable vtable = {
        CODEC_ARCH_WAVTOKENIZER_LARGE,
        "WavTokenizer-Large",
        codec_wavtokenizer_create_impl,
        codec_wavtokenizer_destroy_impl,
        codec_wavtokenizer_init,
        codec_wavtokenizer_encode_wrap,
        codec_wavtokenizer_decode_wrap,
        nullptr,
    };
    return &vtable;
}
