#include "wavtokenizer.h"

#include "../ops/conv1d.h"
#include "../ops/convtr1d.h"
#include "../ops/lm_ggml_ops.h"
#include "../ops/rvq.h"
#include "../runtime/audio_dsp.h"
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
    const codec_model * model;
};

// Resolve a 2D linear weight stored in PyTorch `(out, in)` layout (after GGUF
// dim-reversal: ne=(in, out)) — canonical for lm_ggml_mul_mat. If the GGUF instead
// stores `(in, out)` (legacy axis ordering producing ne=(out, in)), permute via
// lm_ggml_cont(lm_ggml_transpose(t)) so the graph sees ne=(in, out).
static lm_ggml_tensor * codec_wt_W_linear(lm_ggml_context * ctx_eval, const codec_model * model, const std::string & name,
                                        int32_t in_dim, int32_t out_dim) {
    if (ctx_eval == nullptr || model == nullptr || model->weights == nullptr) {
        return nullptr;
    }
    lm_ggml_tensor * src = lm_ggml_get_tensor(model->weights, name.c_str());
    if (src == nullptr) {
        return nullptr;
    }
    if (src->ne[0] == in_dim && src->ne[1] == out_dim) {
        return codec_graph_cast_f32(ctx_eval, src);
    }
    if (src->ne[0] == out_dim && src->ne[1] == in_dim) {
        return lm_ggml_cont(ctx_eval, lm_ggml_transpose(ctx_eval, codec_graph_cast_f32(ctx_eval, src)));
    }
    return nullptr;
}

static lm_ggml_tensor * codec_wt_W_adanorm_row0(lm_ggml_context * ctx_eval, const codec_model * model, const std::string & name) {
    if (ctx_eval == nullptr || model == nullptr || model->weights == nullptr) {
        return nullptr;
    }
    lm_ggml_tensor * src = lm_ggml_get_tensor(model->weights, name.c_str());
    if (src == nullptr) {
        return nullptr;
    }
    // src is (hidden, 4) — take a 1D view over the first row (ne[0]=hidden contiguous).
    lm_ggml_tensor * row0 = lm_ggml_view_1d(ctx_eval, src, src->ne[0], 0);
    return codec_graph_cast_f32(ctx_eval, row0);
}

static lm_ggml_tensor * codec_wt_sum_codebook_features(
    lm_ggml_context * ctx_eval,
    const codec_model * model,
    lm_ggml_tensor * t_tok,
    int32_t t,
    int32_t q) {

    if (model == nullptr || model->weights == nullptr) {
        return nullptr;
    }
    lm_ggml_tensor * sum = nullptr;
    for (int32_t qi = 0; qi < q; ++qi) {
        // Try several legacy GGUF naming variants for the codebook embedding.
        const std::string n0 = "vq.vq.layers." + std::to_string(qi) + "._codebook.embed";
        const std::string n1 = "vq.vq.layers." + std::to_string(qi) + ".codebook.embed";
        lm_ggml_tensor * t_codebook = lm_ggml_get_tensor(model->weights, n0.c_str());
        if (t_codebook == nullptr) {
            t_codebook = lm_ggml_get_tensor(model->weights, n1.c_str());
        }
        if (t_codebook == nullptr) {
            return nullptr;
        }
        // Codebook is stored (codebook_dim, codebook_size). lm_ggml_get_rows treats
        // ne[1] as the row index dim, so this picks rows of length codebook_dim
        // indexed by token id.
        lm_ggml_tensor * t_codebook_f32 = codec_graph_cast_f32(ctx_eval, t_codebook);
        lm_ggml_tensor * t_idx = lm_ggml_view_1d(ctx_eval, t_tok, t, (size_t) qi * t_tok->nb[1]);
        lm_ggml_tensor * t_q = lm_ggml_get_rows(ctx_eval, t_codebook_f32, t_idx); // [codebook_dim, t]
        sum = (sum == nullptr) ? t_q : lm_ggml_add(ctx_eval, sum, t_q);
    }
    return sum;
}

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
        p->backbone_dim <= 0 || p->backbone_intermediate <= 0 || p->n_convnext <= 0 || p->head_out_dim <= 0 || p->model == nullptr) {
        return false;
    }

    auto W = [&](const std::string & name) { return codec_graph_weight(ctx_eval, p->model, name); };
    // Adanorm scale/shift: GGUF stores (hidden, 4); decode uses row 0 only.
    auto Wn = [&](const std::string & name) -> lm_ggml_tensor * {
        if (p->use_adanorm) {
            return codec_wt_W_adanorm_row0(ctx_eval, p->model, name);
        }
        return W(name);
    };

    lm_ggml_tensor * t_tok = lm_ggml_new_tensor_2d(ctx_eval, LM_GGML_TYPE_I32, p->t, p->q);
    lm_ggml_set_name(t_tok, "wt.decode.tok");

    lm_ggml_tensor * t_feat_ct = codec_wt_sum_codebook_features(ctx_eval, p->model, t_tok, p->t, p->q);
    if (t_feat_ct == nullptr) {
        return false;
    }
    lm_ggml_tensor * x = lm_ggml_cont(ctx_eval, lm_ggml_transpose(ctx_eval, t_feat_ct)); // [t, c]

    lm_ggml_tensor * t_embed_w = W("dec.bb.embed.weight");
    lm_ggml_tensor * t_embed_b = W("dec.bb.embed.bias");
    if (t_embed_w == nullptr || t_embed_b == nullptr) {
        return false;
    }
    x = codec_conv1d(ctx_eval, x, t_embed_w, t_embed_b, 1, 1, 3);
    if (x == nullptr) {
        return false;
    }

    if (p->use_pos_net) {
        for (int32_t li = 0; li < 2; ++li) {
            const std::string p_ = "dec.bb.pos_net." + std::to_string(li) + ".";
            lm_ggml_tensor * n1_w = W(p_ + "norm1.weight");
            lm_ggml_tensor * n1_b = W(p_ + "norm1.bias");
            lm_ggml_tensor * c1_w = W(p_ + "conv1.weight");
            lm_ggml_tensor * c1_b = W(p_ + "conv1.bias");
            lm_ggml_tensor * n2_w = W(p_ + "norm2.weight");
            lm_ggml_tensor * n2_b = W(p_ + "norm2.bias");
            lm_ggml_tensor * c2_w = W(p_ + "conv2.weight");
            lm_ggml_tensor * c2_b = W(p_ + "conv2.bias");
            if (n1_w == nullptr || n1_b == nullptr || c1_w == nullptr || c1_b == nullptr ||
                n2_w == nullptr || n2_b == nullptr || c2_w == nullptr || c2_b == nullptr) {
                return false;
            }
            x = codec_wt_pos_resblock(ctx_eval, x, n1_w, n1_b, c1_w, c1_b, n2_w, n2_b, c2_w, c2_b);
            if (x == nullptr) {
                return false;
            }
        }

        const std::string pa = "dec.bb.pos_net.2.";
        lm_ggml_tensor * attn_n_w = W(pa + "norm.weight");
        lm_ggml_tensor * attn_n_b = W(pa + "norm.bias");
        lm_ggml_tensor * q_w = W(pa + "q.weight");
        lm_ggml_tensor * q_b = W(pa + "q.bias");
        lm_ggml_tensor * k_w = W(pa + "k.weight");
        lm_ggml_tensor * k_b = W(pa + "k.bias");
        lm_ggml_tensor * v_w = W(pa + "v.weight");
        lm_ggml_tensor * v_b = W(pa + "v.bias");
        lm_ggml_tensor * o_w = W(pa + "proj_out.weight");
        lm_ggml_tensor * o_b = W(pa + "proj_out.bias");
        if (attn_n_w == nullptr || attn_n_b == nullptr || q_w == nullptr || q_b == nullptr ||
            k_w == nullptr || k_b == nullptr || v_w == nullptr || v_b == nullptr ||
            o_w == nullptr || o_b == nullptr) {
            return false;
        }
        x = codec_wt_pos_attn(ctx_eval, x, attn_n_w, attn_n_b, q_w, q_b, k_w, k_b, v_w, v_b, o_w, o_b, p->backbone_dim);
        if (x == nullptr) {
            return false;
        }

        for (int32_t li = 3; li < 5; ++li) {
            const std::string p_ = "dec.bb.pos_net." + std::to_string(li) + ".";
            lm_ggml_tensor * n1_w = W(p_ + "norm1.weight");
            lm_ggml_tensor * n1_b = W(p_ + "norm1.bias");
            lm_ggml_tensor * c1_w = W(p_ + "conv1.weight");
            lm_ggml_tensor * c1_b = W(p_ + "conv1.bias");
            lm_ggml_tensor * n2_w = W(p_ + "norm2.weight");
            lm_ggml_tensor * n2_b = W(p_ + "norm2.bias");
            lm_ggml_tensor * c2_w = W(p_ + "conv2.weight");
            lm_ggml_tensor * c2_b = W(p_ + "conv2.bias");
            if (n1_w == nullptr || n1_b == nullptr || c1_w == nullptr || c1_b == nullptr ||
                n2_w == nullptr || n2_b == nullptr || c2_w == nullptr || c2_b == nullptr) {
                return false;
            }
            x = codec_wt_pos_resblock(ctx_eval, x, n1_w, n1_b, c1_w, c1_b, n2_w, n2_b, c2_w, c2_b);
            if (x == nullptr) {
                return false;
            }
        }

        lm_ggml_tensor * gn_w = W("dec.bb.pos_net.5.weight");
        lm_ggml_tensor * gn_b = W("dec.bb.pos_net.5.bias");
        if (gn_w == nullptr || gn_b == nullptr) {
            return false;
        }
        x = codec_wt_pos_group_norm(ctx_eval, x, gn_w, gn_b);
        if (x == nullptr) {
            return false;
        }
    }

    // Backbone input layer norm. With adanorm, scale/shift are 2D (hidden, 4) —
    // take row 0; without adanorm they're plain 1D weights/biases.
    lm_ggml_tensor * t_inln_w = p->use_adanorm ? Wn("dec.bb.norm.scale.weight") : W("dec.bb.norm.weight");
    lm_ggml_tensor * t_inln_b = p->use_adanorm ? Wn("dec.bb.norm.shift.weight") : W("dec.bb.norm.bias");
    if (t_inln_w == nullptr || t_inln_b == nullptr) {
        return false;
    }

    lm_ggml_tensor * x_ct = lm_ggml_cont(ctx_eval, lm_ggml_transpose(ctx_eval, x)); // [c, t]
    x_ct = codec_op_layer_norm_ct(ctx_eval, x_ct, 1e-6f, t_inln_w, t_inln_b);
    if (x_ct == nullptr) {
        return false;
    }

    for (int32_t li = 0; li < p->n_convnext; ++li) {
        const std::string p_ = "dec.bb.cnx." + std::to_string(li) + ".";
        lm_ggml_tensor * t_dw_w = W(p_ + "dwconv.weight");
        lm_ggml_tensor * t_dw_b = W(p_ + "dwconv.bias");
        lm_ggml_tensor * t_lnw = p->use_adanorm ? Wn(p_ + "norm.scale.weight") : W(p_ + "norm.weight");
        lm_ggml_tensor * t_lnb = p->use_adanorm ? Wn(p_ + "norm.shift.weight") : W(p_ + "norm.bias");
        lm_ggml_tensor * t_pw1_w = W(p_ + "pwconv1.weight");
        lm_ggml_tensor * t_pw1_b = W(p_ + "pwconv1.bias");
        lm_ggml_tensor * t_pw2_w = W(p_ + "pwconv2.weight");
        lm_ggml_tensor * t_pw2_b = W(p_ + "pwconv2.bias");
        lm_ggml_tensor * t_gamma = W(p_ + "gamma");
        if (t_dw_w == nullptr || t_dw_b == nullptr || t_lnw == nullptr || t_lnb == nullptr ||
            t_pw1_w == nullptr || t_pw1_b == nullptr || t_pw2_w == nullptr || t_pw2_b == nullptr ||
            t_gamma == nullptr) {
            return false;
        }
        x_ct = codec_op_convnext_block_ct(
            ctx_eval, x_ct, t_dw_w, t_dw_b, t_lnw, t_lnb,
            t_pw1_w, t_pw1_b, t_pw2_w, t_pw2_b, t_gamma, 3);
        if (x_ct == nullptr) {
            return false;
        }
    }

    lm_ggml_tensor * t_fln_w = W("dec.bb.fln.weight");
    lm_ggml_tensor * t_fln_b = W("dec.bb.fln.bias");
    if (t_fln_w == nullptr || t_fln_b == nullptr) {
        return false;
    }
    x_ct = codec_op_layer_norm_ct(ctx_eval, x_ct, 1e-6f, t_fln_w, t_fln_b);
    if (x_ct == nullptr) {
        return false;
    }

    lm_ggml_tensor * t_head_w = W("dec.head.out.weight");
    lm_ggml_tensor * t_head_b = W("dec.head.out.bias");
    if (t_head_w == nullptr || t_head_b == nullptr) {
        return false;
    }
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
    const codec_model * model;
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

// Streaming conv1d that fetches weights from the GGUF directly and applies
// reflect-padding via codec_wt_pad1d_reflect. *pad_id is incremented per call so
// each pad slot owns a unique input/output index pair.
static lm_ggml_tensor * codec_wt_sconv1d(
    lm_ggml_context * ctx_eval,
    const codec_model * model,
    lm_ggml_tensor * x,
    int32_t kernel,
    int32_t stride,
    int32_t dilation,
    int32_t * pad_id,
    const std::string & w_name,
    const std::string & b_name) {

    if (ctx_eval == nullptr || model == nullptr || x == nullptr || pad_id == nullptr) {
        return nullptr;
    }
    lm_ggml_tensor * t_w = codec_graph_weight(ctx_eval, model, w_name);
    lm_ggml_tensor * t_b = codec_graph_weight(ctx_eval, model, b_name);
    if (t_w == nullptr || t_b == nullptr) {
        return nullptr;
    }

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

// Wavtokenizer encoder residual block. `model_block_prefix` is the GGUF tensor
// prefix (e.g. "enc.model.1") — block conv1 is at `<prefix>.block.1.conv.conv.*`,
// block conv2 at `.block.3.conv.conv.*`, shortcut at `.shortcut.conv.conv.*`.
static lm_ggml_tensor * codec_wt_encode_resblock(
    lm_ggml_context * ctx_eval,
    const codec_model * model,
    lm_ggml_tensor * x,
    const std::string & model_block_prefix,
    int32_t * pad_id) {

    if (ctx_eval == nullptr || model == nullptr || x == nullptr || pad_id == nullptr) {
        return nullptr;
    }
    lm_ggml_tensor * h = lm_ggml_elu(ctx_eval, x);
    h = codec_wt_sconv1d(
        ctx_eval,
        model,
        h,
        /*kernel=*/3,
        /*stride=*/1,
        /*dilation=*/1,
        pad_id,
        model_block_prefix + ".block.1.conv.conv.weight",
        model_block_prefix + ".block.1.conv.conv.bias");
    if (h == nullptr) {
        return nullptr;
    }
    h = lm_ggml_elu(ctx_eval, h);
    h = codec_wt_sconv1d(
        ctx_eval,
        model,
        h,
        /*kernel=*/1,
        /*stride=*/1,
        /*dilation=*/1,
        pad_id,
        model_block_prefix + ".block.3.conv.conv.weight",
        model_block_prefix + ".block.3.conv.conv.bias");
    if (h == nullptr) {
        return nullptr;
    }

    lm_ggml_tensor * sc = codec_wt_sconv1d(
        ctx_eval,
        model,
        x,
        /*kernel=*/1,
        /*stride=*/1,
        /*dilation=*/1,
        pad_id,
        model_block_prefix + ".shortcut.conv.conv.weight",
        model_block_prefix + ".shortcut.conv.conv.bias");
    if (sc == nullptr) {
        return nullptr;
    }

    return lm_ggml_add(ctx_eval, sc, h);
}

// `model_lstm_prefix` is the GGUF prefix (e.g. "enc.model.13.lstm");
// per-layer weights/biases are at `<prefix>.{weight,bias}_{ih,hh}_l{li}`.
static lm_ggml_tensor * codec_wt_encode_lstm_layers(
    lm_ggml_context * ctx_eval,
    const codec_model * model,
    lm_ggml_tensor * x_tc,
    int32_t dim,
    int32_t n_layers,
    bool skip,
    const std::string & model_lstm_prefix) {

    if (ctx_eval == nullptr || model == nullptr || x_tc == nullptr || dim <= 0 || n_layers <= 0) {
        return nullptr;
    }
    const int32_t t = (int32_t) x_tc->ne[0];
    if (t <= 0) {
        return nullptr;
    }

    std::vector<lm_ggml_tensor *> w_ih((size_t) n_layers, nullptr);
    std::vector<lm_ggml_tensor *> w_hh((size_t) n_layers, nullptr);
    std::vector<lm_ggml_tensor *> b_ih((size_t) n_layers, nullptr);
    std::vector<lm_ggml_tensor *> b_hh((size_t) n_layers, nullptr);
    for (int32_t li = 0; li < n_layers; ++li) {
        const std::string suf = "_l" + std::to_string(li);
        w_ih[(size_t) li] = codec_wt_W_linear(ctx_eval, model, model_lstm_prefix + ".weight_ih" + suf, dim, 4 * dim);
        w_hh[(size_t) li] = codec_wt_W_linear(ctx_eval, model, model_lstm_prefix + ".weight_hh" + suf, dim, 4 * dim);
        b_ih[(size_t) li] = codec_graph_weight(ctx_eval, model, model_lstm_prefix + ".bias_ih" + suf);
        b_hh[(size_t) li] = codec_graph_weight(ctx_eval, model, model_lstm_prefix + ".bias_hh" + suf);
        if (w_ih[(size_t) li] == nullptr || w_hh[(size_t) li] == nullptr ||
            b_ih[(size_t) li] == nullptr || b_hh[(size_t) li] == nullptr) {
            return nullptr;
        }
    }

    lm_ggml_tensor * x_ct = lm_ggml_cont(ctx_eval, lm_ggml_transpose(ctx_eval, x_tc)); // [dim, t]
    lm_ggml_tensor * y_ct = nullptr;
    std::vector<lm_ggml_tensor *> h_prev((size_t) n_layers, nullptr);
    std::vector<lm_ggml_tensor *> c_prev((size_t) n_layers, nullptr);

    for (int32_t ti = 0; ti < t; ++ti) {
        const size_t offset = (size_t) ti * (size_t) x_ct->nb[1];
        lm_ggml_tensor * layer_in = lm_ggml_view_2d(ctx_eval, x_ct, dim, 1, x_ct->nb[1], offset); // [dim, 1]
        layer_in = lm_ggml_cont(ctx_eval, layer_in);

        for (int32_t li = 0; li < n_layers; ++li) {
            if (h_prev[(size_t) li] == nullptr) {
                h_prev[(size_t) li] = lm_ggml_scale(ctx_eval, layer_in, 0.0f);
                c_prev[(size_t) li] = lm_ggml_scale(ctx_eval, layer_in, 0.0f);
            }

            lm_ggml_tensor * gates = lm_ggml_add(
                ctx_eval,
                lm_ggml_mul_mat(ctx_eval, w_ih[(size_t) li], layer_in),
                lm_ggml_mul_mat(ctx_eval, w_hh[(size_t) li], h_prev[(size_t) li]));
            lm_ggml_tensor * b_ih2 = lm_ggml_reshape_2d(ctx_eval, b_ih[(size_t) li], 4 * dim, 1);
            lm_ggml_tensor * b_hh2 = lm_ggml_reshape_2d(ctx_eval, b_hh[(size_t) li], 4 * dim, 1);
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

            lm_ggml_tensor * c_t = lm_ggml_add(ctx_eval, lm_ggml_mul(ctx_eval, f, c_prev[(size_t) li]), lm_ggml_mul(ctx_eval, i, g));
            lm_ggml_tensor * h_t = lm_ggml_mul(ctx_eval, o, lm_ggml_tanh(ctx_eval, c_t));

            h_prev[(size_t) li] = h_t;
            c_prev[(size_t) li] = c_t;
            layer_in = h_t;
        }

        y_ct = (y_ct == nullptr) ? layer_in : lm_ggml_concat(ctx_eval, y_ct, layer_in, 1);
    }

    lm_ggml_tensor * y_tc = lm_ggml_cont(ctx_eval, lm_ggml_transpose(ctx_eval, y_ct));
    if (skip) {
        y_tc = lm_ggml_add(ctx_eval, y_tc, x_tc);
    }
    return y_tc;
}

static bool codec_wt_build_encode(lm_ggml_context * ctx_eval, void * user_data, lm_ggml_tensor ** out) {
    wt_encode_build * p = static_cast<wt_encode_build *>(user_data);
    if (p->model == nullptr) {
        return false;
    }
    lm_ggml_tensor * t_pcm = lm_ggml_new_tensor_2d(ctx_eval, LM_GGML_TYPE_F32, p->n_in, 1);
    lm_ggml_set_name(t_pcm, "wt.encode.pcm");

    int32_t pad_id = 0;

    auto sconv = [&](lm_ggml_tensor * x_in, int32_t kernel, int32_t stride, int32_t dilation, const std::string & prefix) {
        return codec_wt_sconv1d(
            ctx_eval, p->model, x_in, kernel, stride, dilation, &pad_id,
            prefix + ".weight", prefix + ".bias");
    };

    lm_ggml_tensor * x = sconv(t_pcm, /*k=*/7, /*s=*/1, /*d=*/1, "enc.model.0.conv.conv");
    if (x == nullptr) {
        return false;
    }

    x = codec_wt_encode_resblock(ctx_eval, p->model, x, "enc.model.1", &pad_id);
    if (x == nullptr) return false;
    x = lm_ggml_elu(ctx_eval, x);
    x = sconv(x, 4, 2, 1, "enc.model.3.conv.conv");
    if (x == nullptr) return false;

    x = codec_wt_encode_resblock(ctx_eval, p->model, x, "enc.model.4", &pad_id);
    if (x == nullptr) return false;
    x = lm_ggml_elu(ctx_eval, x);
    x = sconv(x, 8, 4, 1, "enc.model.6.conv.conv");
    if (x == nullptr) return false;

    x = codec_wt_encode_resblock(ctx_eval, p->model, x, "enc.model.7", &pad_id);
    if (x == nullptr) return false;
    x = lm_ggml_elu(ctx_eval, x);
    x = sconv(x, 10, 5, 1, "enc.model.9.conv.conv");
    if (x == nullptr) return false;

    x = codec_wt_encode_resblock(ctx_eval, p->model, x, "enc.model.10", &pad_id);
    if (x == nullptr) return false;
    x = lm_ggml_elu(ctx_eval, x);
    x = sconv(x, 16, 8, 1, "enc.model.12.conv.conv");
    if (x == nullptr) return false;

    x = codec_wt_encode_lstm_layers(ctx_eval, p->model, x, /*dim=*/512, /*n_layers=*/2, /*skip=*/true, "enc.model.13.lstm");
    if (x == nullptr) {
        return false;
    }

    x = lm_ggml_elu(ctx_eval, x);
    x = sconv(x, 7, 1, 1, "enc.model.15.conv.conv");
    if (x == nullptr) return false;

    lm_ggml_tensor * residual = lm_ggml_cont(ctx_eval, lm_ggml_transpose(ctx_eval, x)); // [dim, t]
    lm_ggml_tensor * tokens = nullptr;
    for (int32_t qi = 0; qi < p->n_q; ++qi) {
        // Codebook lives under `vq.vq.layers.X._codebook.embed` (or
        // legacy `.codebook.embed`) in the GGUF.
        const std::string n0 = "vq.vq.layers." + std::to_string(qi) + "._codebook.embed";
        const std::string n1 = "vq.vq.layers." + std::to_string(qi) + ".codebook.embed";
        lm_ggml_tensor * t_codebook = codec_graph_weight_or_null(ctx_eval, p->model, n0);
        if (t_codebook == nullptr) {
            t_codebook = codec_graph_weight(ctx_eval, p->model, n1);
        }
        if (t_codebook == nullptr) {
            return false;
        }
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

    lm_ggml_tensor * embed_w = codec_model_get_tensor(ctx->model, "dec.bb.embed.weight");
    lm_ggml_tensor * embed_b = codec_model_get_tensor(ctx->model, "dec.bb.embed.bias");
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
        if (codec_model_get_tensor(ctx->model, "dec.bb.cnx." + std::to_string(li) + ".dwconv.weight") == nullptr) {
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

    lm_ggml_tensor * pw1 = codec_model_get_tensor(ctx->model, "dec.bb.cnx.0.pwconv1.weight");
    if (pw1 == nullptr) {
        if (err != nullptr) {
            *err = "missing WavTokenizer pwconv1 tensor";
        }
        return false;
    }
    build->backbone_intermediate = std::max((int32_t) codec_ne(pw1, 0), (int32_t) codec_ne(pw1, 1));

    lm_ggml_tensor * head_b = codec_model_get_tensor(ctx->model, "dec.head.out.bias");
    if (head_b == nullptr) {
        if (err != nullptr) {
            *err = "missing WavTokenizer head bias tensor";
        }
        return false;
    }
    build->head_out_dim = (int32_t) codec_ne(head_b, 0);
    build->use_adanorm = codec_model_get_tensor(ctx->model, "dec.bb.norm.scale.weight") != nullptr ? 1 : 0;
    build->use_pos_net = codec_model_get_tensor(ctx->model, "dec.bb.pos_net.0.conv1.weight") != nullptr ? 1 : 0;
    build->model = ctx->model;
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
    if (!codec_runtime_istft_from_head(head, build.head_out_dim, t, hop, nullptr, false, -1, &pcm_v, &err)) {
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

    codec_graph_eval_guard eval_guard(ctx);
    const int32_t codebook_dim = std::max(1, wt.codebook_dim);
    const int32_t codebook_size = std::max(2, wt.codebook_size);
    wt_encode_build build = { n_in, hop, use_n_q, codebook_dim, codebook_size, ctx->model };
    codec_graph_cache_entry * entry = nullptr;
    std::string err;
    if (!codec_graph_cache_get_or_build(
            ctx,
            { CODEC_GRAPH_WT_ENCODE, /*n_frames=*/0, /*n_q=*/use_n_q, /*hop=*/hop, /*n_in=*/n_in, /*latent_dim=*/codebook_dim },
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

    if (!codec_wt_write_encode_pad_indices(ctx, entry, n_in, &err)) {
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
    std::vector<int32_t> tok;
    if (!codec_runtime_read_tensor_i32_2d_tq(t_out, &tok, &err)) {
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
        codec_graph_size_exact,
        codec_wavtokenizer_encode_wrap,
        codec_wavtokenizer_decode_wrap,
        nullptr,
    };
    return &vtable;
}
