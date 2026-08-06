#include "qwen3_tts_tokenizer.h"

#include "../ops/conv1d.h"
#include "../ops/convtr1d.h"
#include "../ops/lm_ggml_ops.h"
#include "../ops/rope.h"
#include "../runtime/graph.h"
#include "../runtime/tensor_utils.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <string>
#include <vector>

static constexpr int32_t CODEC_Q3T_RES_UNITS = 3;
static constexpr int32_t CODEC_Q3T_RES_DILATIONS[CODEC_Q3T_RES_UNITS] = { 1, 3, 9 };

enum codec_status codec_qwen3_tts_tokenizer_init(struct codec_model * model) {
    codec_qwen3_tts_tokenizer_impl & impl = *static_cast<codec_qwen3_tts_tokenizer_impl *>(model->impl);
    codec_qwen3_tts_tokenizer & q3 = impl.q3;
    codec_mimi & mimi = impl.mimi;

    q3.sample_rate = codec_read_i32_kv(model->gguf, "codec.sample_rate", 24000);
    q3.hop_size = codec_read_i32_kv(model->gguf, "codec.hop_size", 1920);
    q3.n_q = codec_read_i32_kv(model->gguf, "codec.n_q", 16);
    q3.codebook_size = codec_read_i32_kv(model->gguf, "codec.codebook_size", 2048);
    q3.codebook_dim = codec_read_i32_kv(model->gguf, "codec.codebook_dim", 1024);
    q3.latent_dim = codec_read_i32_kv(model->gguf, "codec.latent_dim", 1024);
    q3.has_encoder = codec_read_bool_kv(model->gguf, "codec.has_encoder", true);
    q3.has_decoder = codec_read_bool_kv(model->gguf, "codec.has_decoder", true);

    q3.hidden_size = codec_read_i32_kv(model->gguf, "qwen3.decoder.hidden_size", 1024);
    q3.num_hidden_layers = codec_read_i32_kv(model->gguf, "qwen3.decoder.num_hidden_layers", 8);
    q3.num_attention_heads = codec_read_i32_kv(model->gguf, "qwen3.decoder.num_attention_heads", 16);
    q3.num_key_value_heads = codec_read_i32_kv(model->gguf, "qwen3.decoder.num_key_value_heads", q3.num_attention_heads);
    q3.head_dim = codec_read_i32_kv(model->gguf, "qwen3.decoder.head_dim", 64);
    q3.intermediate_size = codec_read_i32_kv(model->gguf, "qwen3.decoder.intermediate_size", 3072);
    q3.rope_theta = codec_read_f32_kv(model->gguf, "qwen3.decoder.rope_theta", 10000.0f);
    q3.sliding_window = codec_read_i32_kv(model->gguf, "qwen3.decoder.sliding_window", 0);
    q3.decoder_dim = codec_read_i32_kv(model->gguf, "qwen3.decoder.decoder_dim", 1536);

    // upsample arrays
    q3.n_upsample_rates = 0;
    q3.n_upsampling_ratios = 0;
    {
        const int key_id = lm_gguf_find_key(model->gguf, "qwen3.decoder.upsample_rates");
        if (key_id >= 0 && lm_gguf_get_kv_type(model->gguf, key_id) == LM_GGUF_TYPE_ARRAY) {
            const enum lm_gguf_type t = lm_gguf_get_arr_type(model->gguf, key_id);
            const size_t n = lm_gguf_get_arr_n(model->gguf, key_id);
            const size_t copy_n = std::min(n, (size_t) CODEC_Q3T_MAX_UPSAMPLE);
            q3.n_upsample_rates = (int32_t) copy_n;
            const void * data = lm_gguf_get_arr_data(model->gguf, key_id);
            for (size_t i = 0; i < copy_n; ++i) {
                if (t == LM_GGUF_TYPE_INT32) {
                    q3.upsample_rates[i] = ((const int32_t *) data)[i];
                } else if (t == LM_GGUF_TYPE_UINT32) {
                    q3.upsample_rates[i] = (int32_t) ((const uint32_t *) data)[i];
                } else {
                    q3.upsample_rates[i] = 0;
                }
            }
        }
    }
    {
        const int key_id = lm_gguf_find_key(model->gguf, "qwen3.decoder.upsampling_ratios");
        if (key_id >= 0 && lm_gguf_get_kv_type(model->gguf, key_id) == LM_GGUF_TYPE_ARRAY) {
            const enum lm_gguf_type t = lm_gguf_get_arr_type(model->gguf, key_id);
            const size_t n = lm_gguf_get_arr_n(model->gguf, key_id);
            const size_t copy_n = std::min(n, (size_t) CODEC_Q3T_MAX_UPSAMPLE);
            q3.n_upsampling_ratios = (int32_t) copy_n;
            const void * data = lm_gguf_get_arr_data(model->gguf, key_id);
            for (size_t i = 0; i < copy_n; ++i) {
                if (t == LM_GGUF_TYPE_INT32) {
                    q3.upsampling_ratios[i] = ((const int32_t *) data)[i];
                } else if (t == LM_GGUF_TYPE_UINT32) {
                    q3.upsampling_ratios[i] = (int32_t) ((const uint32_t *) data)[i];
                } else {
                    q3.upsampling_ratios[i] = 0;
                }
            }
        }
    }

    // Initialize Mimi encoder metadata from qwen3.encoder.* keys
    mimi.sample_rate = codec_read_i32_kv(model->gguf, "codec.sample_rate", 24000);
    mimi.hop_size = codec_read_i32_kv(model->gguf, "codec.hop_size", 1920);
    mimi.n_q = codec_read_i32_kv(model->gguf, "qwen3.encoder.n_q", q3.n_q);
    mimi.num_semantic_quantizers = codec_read_i32_kv(model->gguf, "codec.num_semantic_quantizers", 1);
    mimi.codebook_size = codec_read_i32_kv(model->gguf, "qwen3.encoder.codebook_size", q3.codebook_size);
    mimi.codebook_dim = codec_read_i32_kv(model->gguf, "qwen3.encoder.codebook_dim", q3.codebook_dim);
    mimi.hidden_size = codec_read_i32_kv(model->gguf, "qwen3.encoder.hidden_size", 512);
    mimi.num_hidden_layers = codec_read_i32_kv(model->gguf, "qwen3.encoder.num_hidden_layers", 8);
    mimi.num_attention_heads = codec_read_i32_kv(model->gguf, "qwen3.encoder.num_attention_heads", 8);
    mimi.head_dim = codec_read_i32_kv(model->gguf, "qwen3.encoder.head_dim", 64);
    mimi.intermediate_size = codec_read_i32_kv(model->gguf, "qwen3.encoder.intermediate_size", 2048);
    mimi.rope_theta = codec_read_f32_kv(model->gguf, "qwen3.encoder.rope_theta", 10000.0f);
    mimi.rope_scaling_factor = codec_read_f32_kv(model->gguf, "qwen3.encoder.rope_scaling_factor", 1.0f);
    mimi.has_encoder = q3.has_encoder;
    mimi.has_decoder = false;

    model->sample_rate = q3.sample_rate;
    model->has_encoder = q3.has_encoder;
    model->has_decoder = q3.has_decoder;
    model->hop_size = q3.hop_size;
    model->n_q = q3.n_q;
    model->codebook_size = q3.codebook_size;
    model->n_fft = -1;
    model->win_length = -1;
    model->n_mels = -1;
    model->latent_dim = q3.latent_dim;

    return CODEC_STATUS_SUCCESS;
}

struct q3t_decode_build {
    int32_t t;
    int32_t q;
    int32_t n_sem;
    int32_t codebook_size;
    int32_t codebook_dim;
    int32_t codebook_dim_half;
    int32_t latent_dim;
    int32_t hidden_size;
    int32_t transformer_layers;
    int32_t transformer_heads;
    int32_t transformer_kv_heads;
    int32_t transformer_head_dim;
    int32_t transformer_intermediate;
    float rope_theta;
    int32_t sliding_window;
    int32_t n_upsample_rates;
    int32_t n_upsampling_ratios;
    int32_t upsample_rates[CODEC_Q3T_MAX_UPSAMPLE];
    int32_t upsampling_ratios[CODEC_Q3T_MAX_UPSAMPLE];
    int32_t decoder_dim;
    const codec_model * model;
};

// Add bias broadcast over channels; no-op when bias is null. Used for
// attention q/k/v/o biases which are optional in some Qwen3 checkpoints.
static lm_ggml_tensor * codec_q3t_add_bias_ct(lm_ggml_context * ctx_eval, lm_ggml_tensor * x, lm_ggml_tensor * b) {
    if (b == nullptr) {
        return x;
    }
    return lm_ggml_add(ctx_eval, x, lm_ggml_repeat(ctx_eval, lm_ggml_reshape_2d(ctx_eval, b, x->ne[0], 1), x));
}

static lm_ggml_tensor * codec_q3t_convnext_block(
    lm_ggml_context * ctx_eval,
    lm_ggml_tensor * x_tc,
    lm_ggml_tensor * dw_w,
    lm_ggml_tensor * dw_b,
    lm_ggml_tensor * ln_w,
    lm_ggml_tensor * ln_b,
    lm_ggml_tensor * pw1_w,
    lm_ggml_tensor * pw1_b,
    lm_ggml_tensor * pw2_w,
    lm_ggml_tensor * pw2_b,
    lm_ggml_tensor * gamma) {

    if (ctx_eval == nullptr || x_tc == nullptr) {
        return nullptr;
    }

    lm_ggml_tensor * res = x_tc;
    lm_ggml_tensor * h = codec_conv1d_depthwise_causal(ctx_eval, x_tc, dw_w, dw_b, 1, 1);
    if (h == nullptr) {
        return nullptr;
    }

    lm_ggml_tensor * h_ct = lm_ggml_cont(ctx_eval, lm_ggml_transpose(ctx_eval, h)); // [c, t]
    h_ct = codec_op_layer_norm_ct(ctx_eval, h_ct, 1e-6f, ln_w, ln_b);
    if (h_ct == nullptr) {
        return nullptr;
    }
    lm_ggml_tensor * pw1 = lm_ggml_mul_mat(ctx_eval, pw1_w, h_ct);
    if (pw1_b != nullptr) {
        lm_ggml_tensor * b2 = lm_ggml_reshape_2d(ctx_eval, pw1_b, pw1_w->ne[1], 1);
        pw1 = lm_ggml_add(ctx_eval, pw1, lm_ggml_repeat(ctx_eval, b2, pw1));
    }
    pw1 = lm_ggml_gelu_erf(ctx_eval, pw1);
    lm_ggml_tensor * pw2 = lm_ggml_mul_mat(ctx_eval, pw2_w, pw1);
    if (pw2_b != nullptr) {
        lm_ggml_tensor * b2 = lm_ggml_reshape_2d(ctx_eval, pw2_b, pw2_w->ne[1], 1);
        pw2 = lm_ggml_add(ctx_eval, pw2, lm_ggml_repeat(ctx_eval, b2, pw2));
    }
    pw2 = codec_op_channel_scale(ctx_eval, pw2, gamma);
    lm_ggml_tensor * out_tc = lm_ggml_cont(ctx_eval, lm_ggml_transpose(ctx_eval, pw2)); // [t, c]
    return lm_ggml_add(ctx_eval, res, out_tc);
}

static lm_ggml_tensor * codec_q3t_apply_sliding_mask(
    lm_ggml_context * ctx_eval,
    lm_ggml_tensor * attn_scores,
    int32_t window) {

    if (ctx_eval == nullptr || attn_scores == nullptr) {
        return nullptr;
    }
    attn_scores = lm_ggml_diag_mask_inf_inplace(ctx_eval, attn_scores, 0);
    if (window <= 0) {
        return attn_scores;
    }
    const int32_t n_past = std::max(0, window - 1);
    lm_ggml_tensor * t = lm_ggml_cont(ctx_eval, lm_ggml_transpose(ctx_eval, attn_scores));
    t = lm_ggml_diag_mask_inf_inplace(ctx_eval, t, n_past);
    return lm_ggml_cont(ctx_eval, lm_ggml_transpose(ctx_eval, t));
}

static lm_ggml_tensor * codec_q3t_repeat_kv(
    lm_ggml_context * ctx_eval,
    lm_ggml_tensor * x,
    int32_t n_heads,
    int32_t n_kv) {

    if (n_kv <= 0 || n_heads <= 0 || n_kv == n_heads) {
        return x;
    }
    if (n_heads % n_kv != 0) {
        return nullptr;
    }
    lm_ggml_tensor * target = lm_ggml_new_tensor_3d(ctx_eval, x->type, x->ne[0], x->ne[1], n_heads);
    return lm_ggml_repeat(ctx_eval, x, target);
}

static std::string codec_q3t_decode_codebook_tensor_name(int32_t qi) {
    return "q3t.dec.q.l" + std::to_string(qi) + ".codebook";
}

static std::string codec_q3t_decode_idx_tensor_name(int32_t qi) {
    return "q3t.dec.q" + std::to_string(qi) + ".idx";
}

static std::string codec_q3t_decode_pt_layer_name(int32_t li, const char * suffix) {
    return "q3t.dec.pt.l" + std::to_string(li) + "." + suffix;
}

static bool codec_q3t_build_decode(lm_ggml_context * ctx_eval, void * user_data, lm_ggml_tensor ** out) {
    q3t_decode_build * p = static_cast<q3t_decode_build *>(user_data);
    if (ctx_eval == nullptr || p == nullptr || out == nullptr || p->t <= 0 || p->q <= 0 ||
        p->codebook_size <= 1 || p->codebook_dim <= 0 || p->codebook_dim_half <= 0 ||
        p->latent_dim <= 0 || p->hidden_size <= 0 || p->transformer_layers <= 0 ||
        p->transformer_heads <= 0 || p->transformer_head_dim <= 0 || p->transformer_intermediate <= 0 ||
        p->model == nullptr) {
        return false;
    }
    const int32_t attn_dim = p->transformer_heads * p->transformer_head_dim;
    const int32_t kv_dim = p->transformer_kv_heads * p->transformer_head_dim;

    auto W = [&](const std::string & name) -> lm_ggml_tensor * {
        return codec_graph_weight(ctx_eval, p->model, name);
    };
    auto Wopt = [&](const std::string & name) -> lm_ggml_tensor * {
        return codec_graph_weight_or_null(ctx_eval, p->model, name);
    };

    // Codebook lookup per-quantizer
    lm_ggml_tensor * sem_sum = nullptr;
    lm_ggml_tensor * acu_sum = nullptr;
    for (int32_t qi = 0; qi < p->q; ++qi) {
        lm_ggml_tensor * t_idx = lm_ggml_new_tensor_1d(ctx_eval, LM_GGML_TYPE_I32, p->t);
        lm_ggml_set_name(t_idx, codec_q3t_decode_idx_tensor_name(qi).c_str());

        lm_ggml_tensor * t_codebook = W(codec_q3t_decode_codebook_tensor_name(qi));
        if (t_codebook == nullptr) {
            return false;
        }

        lm_ggml_tensor * t_qi = lm_ggml_get_rows(ctx_eval, t_codebook, t_idx); // [cb_dim_half, t]
        if (t_qi == nullptr) {
            return false;
        }
        if (qi == 0) {
            lm_ggml_set_name(t_qi, "q3t.dec.q0");
        } else if (qi == 1) {
            lm_ggml_set_name(t_qi, "q3t.dec.q1");
        }
        if (qi < p->n_sem) {
            sem_sum = sem_sum == nullptr ? t_qi : lm_ggml_add(ctx_eval, sem_sum, t_qi);
        } else {
            acu_sum = acu_sum == nullptr ? t_qi : lm_ggml_add(ctx_eval, acu_sum, t_qi);
        }
    }
    if (sem_sum == nullptr) {
        return false;
    }

    lm_ggml_tensor * t_sem_op_w = W("q3t.dec.q.s.op.w");
    if (t_sem_op_w == nullptr) {
        return false;
    }
    lm_ggml_tensor * sem_ct = lm_ggml_cont(ctx_eval, sem_sum); // [cb_dim_half, t]
    lm_ggml_tensor * sem_out = lm_ggml_mul_mat(ctx_eval, t_sem_op_w, sem_ct); // [cb_dim, t]
    sem_out = lm_ggml_cont(ctx_eval, lm_ggml_transpose(ctx_eval, sem_out)); // [t, cb_dim]
    if (sem_out == nullptr) {
        return false;
    }
    lm_ggml_set_name(sem_out, "q3t.dec.sem");

    lm_ggml_tensor * x_tc = sem_out;
    if (acu_sum != nullptr) {
        lm_ggml_tensor * t_acu_op_w = W("q3t.dec.q.a.op.w");
        if (t_acu_op_w == nullptr) {
            return false;
        }
        lm_ggml_tensor * acu_ct = lm_ggml_cont(ctx_eval, acu_sum); // [cb_dim_half, t]
        lm_ggml_tensor * acu_out = lm_ggml_mul_mat(ctx_eval, t_acu_op_w, acu_ct); // [cb_dim, t]
        acu_out = lm_ggml_cont(ctx_eval, lm_ggml_transpose(ctx_eval, acu_out)); // [t, cb_dim]
        if (acu_out == nullptr) {
            return false;
        }
        x_tc = lm_ggml_add(ctx_eval, sem_out, acu_out);
    }
    lm_ggml_set_name(x_tc, "q3t.dec.qsum");

    lm_ggml_tensor * t_pre_w = W("q3t.dec.pre.conv.w");
    lm_ggml_tensor * t_pre_b = W("q3t.dec.pre.conv.b");
    if (t_pre_w == nullptr || t_pre_b == nullptr) {
        return false;
    }
    x_tc = codec_conv1d_causal(ctx_eval, x_tc, t_pre_w, t_pre_b, 1, 1);
    if (x_tc == nullptr) {
        return false;
    }
    lm_ggml_set_name(x_tc, "q3t.dec.pre");

    lm_ggml_tensor * x_ct = lm_ggml_cont(ctx_eval, lm_ggml_transpose(ctx_eval, x_tc)); // [latent_dim, t]

    // pre-transformer input/output projections
    lm_ggml_tensor * t_in_w = W("q3t.dec.pt.in.w");
    lm_ggml_tensor * t_in_b = W("q3t.dec.pt.in.b");
    lm_ggml_tensor * t_out_w = W("q3t.dec.pt.out.w");
    lm_ggml_tensor * t_out_b = W("q3t.dec.pt.out.b");
    if (t_in_w == nullptr || t_in_b == nullptr || t_out_w == nullptr || t_out_b == nullptr) {
        return false;
    }

    x_ct = lm_ggml_mul_mat(ctx_eval, t_in_w, x_ct);
    lm_ggml_tensor * in_b2 = lm_ggml_reshape_2d(ctx_eval, t_in_b, t_in_w->ne[1], 1);
    x_ct = lm_ggml_add(ctx_eval, x_ct, lm_ggml_repeat(ctx_eval, in_b2, x_ct));

    for (int32_t li = 0; li < p->transformer_layers; ++li) {
        lm_ggml_tensor * inln_w = W(codec_q3t_decode_pt_layer_name(li, "inln.w"));
        lm_ggml_tensor * paln_w = W(codec_q3t_decode_pt_layer_name(li, "paln.w"));
        lm_ggml_tensor * q_w = W(codec_q3t_decode_pt_layer_name(li, "attn.q.w"));
        lm_ggml_tensor * k_w = W(codec_q3t_decode_pt_layer_name(li, "attn.k.w"));
        lm_ggml_tensor * v_w = W(codec_q3t_decode_pt_layer_name(li, "attn.v.w"));
        lm_ggml_tensor * o_w = W(codec_q3t_decode_pt_layer_name(li, "attn.o.w"));
        lm_ggml_tensor * q_b = Wopt(codec_q3t_decode_pt_layer_name(li, "attn.q.b"));
        lm_ggml_tensor * k_b = Wopt(codec_q3t_decode_pt_layer_name(li, "attn.k.b"));
        lm_ggml_tensor * v_b = Wopt(codec_q3t_decode_pt_layer_name(li, "attn.v.b"));
        lm_ggml_tensor * o_b = Wopt(codec_q3t_decode_pt_layer_name(li, "attn.o.b"));
        lm_ggml_tensor * fc_gate = W(codec_q3t_decode_pt_layer_name(li, "mlp.gate.w"));
        lm_ggml_tensor * fc_up = W(codec_q3t_decode_pt_layer_name(li, "mlp.up.w"));
        lm_ggml_tensor * fc_down = W(codec_q3t_decode_pt_layer_name(li, "mlp.down.w"));
        lm_ggml_tensor * sa_scale = W(codec_q3t_decode_pt_layer_name(li, "sa.scale"));
        lm_ggml_tensor * mlp_scale = W(codec_q3t_decode_pt_layer_name(li, "mlp.scale"));
        if (inln_w == nullptr || paln_w == nullptr || q_w == nullptr || k_w == nullptr ||
            v_w == nullptr || o_w == nullptr || fc_gate == nullptr || fc_up == nullptr ||
            fc_down == nullptr || sa_scale == nullptr || mlp_scale == nullptr) {
            return false;
        }

        lm_ggml_tensor * h = codec_op_rms_norm_ct(ctx_eval, x_ct, 1e-5f, inln_w);
        if (h == nullptr) {
            return false;
        }

        lm_ggml_tensor * q = lm_ggml_mul_mat(ctx_eval, q_w, h);
        lm_ggml_tensor * k = lm_ggml_mul_mat(ctx_eval, k_w, h);
        lm_ggml_tensor * v = lm_ggml_mul_mat(ctx_eval, v_w, h);
        if (q == nullptr || k == nullptr || v == nullptr) {
            return false;
        }
        q = codec_q3t_add_bias_ct(ctx_eval, q, q_b);
        k = codec_q3t_add_bias_ct(ctx_eval, k, k_b);
        v = codec_q3t_add_bias_ct(ctx_eval, v, v_b);

        const int64_t t_cur = q->ne[1];
        lm_ggml_tensor * q_dth = lm_ggml_permute(ctx_eval, lm_ggml_reshape_3d(ctx_eval, q, p->transformer_head_dim, p->transformer_heads, t_cur), 0, 2, 1, 3);
        lm_ggml_tensor * k_dth = lm_ggml_permute(ctx_eval, lm_ggml_reshape_3d(ctx_eval, k, p->transformer_head_dim, p->transformer_kv_heads, t_cur), 0, 2, 1, 3);
        lm_ggml_tensor * v_dth = lm_ggml_permute(ctx_eval, lm_ggml_reshape_3d(ctx_eval, v, p->transformer_head_dim, p->transformer_kv_heads, t_cur), 0, 2, 1, 3);

        lm_ggml_tensor * q_rope = codec_op_rope(ctx_eval, q_dth, p->transformer_head_dim, p->rope_theta, 1.0f, CODEC_ROPE_MODE_NEOX);
        lm_ggml_tensor * k_rope = codec_op_rope(ctx_eval, k_dth, p->transformer_head_dim, p->rope_theta, 1.0f, CODEC_ROPE_MODE_NEOX);
        if (q_rope == nullptr || k_rope == nullptr) {
            return false;
        }
        q_rope = codec_q3t_repeat_kv(ctx_eval, q_rope, p->transformer_heads, p->transformer_kv_heads);
        k_rope = codec_q3t_repeat_kv(ctx_eval, k_rope, p->transformer_heads, p->transformer_kv_heads);
        lm_ggml_tensor * v_rep = codec_q3t_repeat_kv(ctx_eval, v_dth, p->transformer_heads, p->transformer_kv_heads);
        if (q_rope == nullptr || k_rope == nullptr || v_rep == nullptr) {
            return false;
        }

        lm_ggml_tensor * attn_scores = lm_ggml_mul_mat(ctx_eval, lm_ggml_cont(ctx_eval, k_rope), q_rope); // [t, t, h]
        attn_scores = lm_ggml_scale_inplace(ctx_eval, attn_scores, 1.0f / std::sqrt((float) p->transformer_head_dim));
        attn_scores = codec_q3t_apply_sliding_mask(ctx_eval, attn_scores, p->sliding_window);
        lm_ggml_tensor * attn_probs = lm_ggml_soft_max(ctx_eval, attn_scores);

        lm_ggml_tensor * v_tdh = lm_ggml_permute(ctx_eval, v_rep, 1, 0, 2, 3);
        lm_ggml_tensor * attn_ctx = lm_ggml_mul_mat(ctx_eval, lm_ggml_cont(ctx_eval, v_tdh), attn_probs); // [d, t, h]
        lm_ggml_tensor * attn_ct = lm_ggml_reshape_2d(
            ctx_eval,
            lm_ggml_cont(ctx_eval, lm_ggml_permute(ctx_eval, attn_ctx, 0, 2, 1, 3)),
            attn_dim,
            t_cur);
        lm_ggml_tensor * attn_proj = lm_ggml_mul_mat(ctx_eval, o_w, attn_ct);
        attn_proj = codec_q3t_add_bias_ct(ctx_eval, attn_proj, o_b);
        x_ct = lm_ggml_add(ctx_eval, x_ct, codec_op_channel_scale(ctx_eval, attn_proj, sa_scale));

        lm_ggml_tensor * m = codec_op_rms_norm_ct(ctx_eval, x_ct, 1e-5f, paln_w);
        lm_ggml_tensor * gate = lm_ggml_mul_mat(ctx_eval, fc_gate, m);
        lm_ggml_tensor * up = lm_ggml_mul_mat(ctx_eval, fc_up, m);
        gate = lm_ggml_silu(ctx_eval, gate);
        lm_ggml_tensor * prod = lm_ggml_mul(ctx_eval, gate, up);
        lm_ggml_tensor * down = lm_ggml_mul_mat(ctx_eval, fc_down, prod);
        x_ct = lm_ggml_add(ctx_eval, x_ct, codec_op_channel_scale(ctx_eval, down, mlp_scale));
    }

    lm_ggml_tensor * t_norm_w = W("q3t.dec.pt.norm.w");
    if (t_norm_w == nullptr) {
        return false;
    }
    x_ct = codec_op_rms_norm_ct(ctx_eval, x_ct, 1e-5f, t_norm_w);

    x_ct = lm_ggml_mul_mat(ctx_eval, t_out_w, x_ct);
    lm_ggml_tensor * out_b2 = lm_ggml_reshape_2d(ctx_eval, t_out_b, t_out_w->ne[1], 1);
    x_ct = lm_ggml_add(ctx_eval, x_ct, lm_ggml_repeat(ctx_eval, out_b2, x_ct));

    x_tc = lm_ggml_cont(ctx_eval, lm_ggml_transpose(ctx_eval, x_ct)); // [t, latent_dim]
    lm_ggml_set_name(x_tc, "q3t.dec.pt");

    // upsampling ratios (convtr + convnext)
    for (int32_t ui = 0; ui < p->n_upsampling_ratios; ++ui) {
        const std::string base = "q3t.dec.up" + std::to_string(ui);
        lm_ggml_tensor * t_up_w = W(base + ".tr.w");
        lm_ggml_tensor * t_up_b = W(base + ".tr.b");
        lm_ggml_tensor * t_dw_w = W(base + ".cnx.dw.w");
        lm_ggml_tensor * t_dw_b = W(base + ".cnx.dw.b");
        lm_ggml_tensor * t_ln_w = W(base + ".cnx.norm.w");
        lm_ggml_tensor * t_ln_b = W(base + ".cnx.norm.b");
        lm_ggml_tensor * t_pw1_w = W(base + ".cnx.pw1.w");
        lm_ggml_tensor * t_pw1_b = W(base + ".cnx.pw1.b");
        lm_ggml_tensor * t_pw2_w = W(base + ".cnx.pw2.w");
        lm_ggml_tensor * t_pw2_b = W(base + ".cnx.pw2.b");
        lm_ggml_tensor * t_gamma = W(base + ".cnx.gamma");
        if (t_up_w == nullptr || t_up_b == nullptr || t_dw_w == nullptr || t_dw_b == nullptr ||
            t_ln_w == nullptr || t_ln_b == nullptr || t_pw1_w == nullptr || t_pw1_b == nullptr ||
            t_pw2_w == nullptr || t_pw2_b == nullptr || t_gamma == nullptr) {
            return false;
        }
        x_tc = codec_convtr1d_causal(ctx_eval, x_tc, t_up_w, t_up_b, p->upsampling_ratios[ui], 1);
        x_tc = codec_q3t_convnext_block(ctx_eval, x_tc, t_dw_w, t_dw_b, t_ln_w, t_ln_b, t_pw1_w, t_pw1_b, t_pw2_w, t_pw2_b, t_gamma);
        lm_ggml_set_name(x_tc, (base + ".out").c_str());
    }

    // decoder conv stack
    lm_ggml_tensor * t_dec0_w = W("q3t.dec.d0.w");
    lm_ggml_tensor * t_dec0_b = W("q3t.dec.d0.b");
    if (t_dec0_w == nullptr || t_dec0_b == nullptr) {
        return false;
    }
    x_tc = codec_conv1d_causal(ctx_eval, x_tc, t_dec0_w, t_dec0_b, 1, 1);

    int32_t cur_dim = p->decoder_dim;
    for (int32_t bi = 0; bi < p->n_upsample_rates; ++bi) {
        const std::string base = "q3t.dec.b" + std::to_string(bi);
        const int32_t out_dim = p->decoder_dim / (1 << (bi + 1));
        const int32_t up_rate = p->upsample_rates[bi];

        lm_ggml_tensor * t_s0_a = W(base + ".s0.a");
        lm_ggml_tensor * t_s0_b = W(base + ".s0.binv");
        lm_ggml_tensor * t_tr_w = W(base + ".tr.w");
        lm_ggml_tensor * t_tr_b = W(base + ".tr.b");
        if (t_s0_a == nullptr || t_s0_b == nullptr || t_tr_w == nullptr || t_tr_b == nullptr) {
            return false;
        }
        x_tc = codec_op_snake_beta(ctx_eval, x_tc, t_s0_a, t_s0_b, 1e-9f);
        x_tc = codec_convtr1d_causal(ctx_eval, x_tc, t_tr_w, t_tr_b, up_rate, 1);

        for (int32_t ri = 0; ri < CODEC_Q3T_RES_UNITS; ++ri) {
            const std::string rbase = base + ".r" + std::to_string(ri);
            lm_ggml_tensor * t_s1_a = W(rbase + ".s1.a");
            lm_ggml_tensor * t_s1_b = W(rbase + ".s1.binv");
            lm_ggml_tensor * t_c1_w = W(rbase + ".c1.w");
            lm_ggml_tensor * t_c1_b = W(rbase + ".c1.b");
            lm_ggml_tensor * t_s2_a = W(rbase + ".s2.a");
            lm_ggml_tensor * t_s2_b = W(rbase + ".s2.binv");
            lm_ggml_tensor * t_c2_w = W(rbase + ".c2.w");
            lm_ggml_tensor * t_c2_b = W(rbase + ".c2.b");
            if (t_s1_a == nullptr || t_s1_b == nullptr || t_c1_w == nullptr || t_c1_b == nullptr ||
                t_s2_a == nullptr || t_s2_b == nullptr || t_c2_w == nullptr || t_c2_b == nullptr) {
                return false;
            }

            lm_ggml_tensor * res = x_tc;
            x_tc = codec_op_snake_beta(ctx_eval, x_tc, t_s1_a, t_s1_b, 1e-9f);
            x_tc = codec_conv1d_causal(ctx_eval, x_tc, t_c1_w, t_c1_b, 1, CODEC_Q3T_RES_DILATIONS[ri]);
            x_tc = codec_op_snake_beta(ctx_eval, x_tc, t_s2_a, t_s2_b, 1e-9f);
            x_tc = codec_conv1d_causal(ctx_eval, x_tc, t_c2_w, t_c2_b, 1, 1);
            x_tc = lm_ggml_add(ctx_eval, res, x_tc);
        }
        cur_dim = out_dim;
    }
    lm_ggml_set_name(x_tc, "q3t.dec.dec_out");

    lm_ggml_tensor * t_fs_a = W("q3t.dec.final.s.a");
    lm_ggml_tensor * t_fs_b = W("q3t.dec.final.s.binv");
    lm_ggml_tensor * t_out_final_w = W("q3t.dec.final.w");
    lm_ggml_tensor * t_out_final_b = W("q3t.dec.final.b");
    if (t_fs_a == nullptr || t_fs_b == nullptr || t_out_final_w == nullptr || t_out_final_b == nullptr) {
        return false;
    }
    x_tc = codec_op_snake_beta(ctx_eval, x_tc, t_fs_a, t_fs_b, 1e-9f);

    lm_ggml_tensor * t_out = codec_conv1d_causal(ctx_eval, x_tc, t_out_final_w, t_out_final_b, 1, 1);
    if (t_out == nullptr) {
        return false;
    }
    t_out = lm_ggml_clamp(ctx_eval, t_out, -1.0f, 1.0f);
    lm_ggml_set_name(t_out, "q3t.dec.out");
    *out = t_out;
    return true;
}

static bool codec_q3t_init_decode_build(codec_context * ctx, int32_t t, int32_t q, q3t_decode_build * build, std::string * err) {
    if (ctx == nullptr || ctx->model == nullptr || build == nullptr || t <= 0 || q <= 0) {
        if (err != nullptr) {
            *err = "invalid Qwen3 decode build arguments";
        }
        return false;
    }
    codec_qwen3_tts_tokenizer_impl & impl = *static_cast<codec_qwen3_tts_tokenizer_impl *>(ctx->model->impl);
    codec_qwen3_tts_tokenizer & q3 = impl.q3;
    build->t = t;
    build->q = q;
    build->n_sem = 1;
    build->codebook_size = std::max(2, q3.codebook_size);
    build->codebook_dim = std::max(1, q3.codebook_dim);
    if (build->codebook_dim % 2 != 0) {
        if (err != nullptr) {
            *err = "Qwen3 codebook_dim must be even";
        }
        return false;
    }
    build->codebook_dim_half = build->codebook_dim / 2;
    build->latent_dim = std::max(1, q3.latent_dim);
    build->hidden_size = std::max(1, q3.hidden_size);
    build->transformer_layers = std::max(1, q3.num_hidden_layers);
    build->transformer_heads = std::max(1, q3.num_attention_heads);
    build->transformer_kv_heads = std::max(1, q3.num_key_value_heads);
    build->transformer_head_dim = std::max(1, q3.head_dim);
    build->transformer_intermediate = std::max(1, q3.intermediate_size);
    build->rope_theta = q3.rope_theta;
    build->sliding_window = q3.sliding_window;
    build->decoder_dim = std::max(1, q3.decoder_dim);

    build->n_upsample_rates = std::max(0, std::min(q3.n_upsample_rates, CODEC_Q3T_MAX_UPSAMPLE));
    build->n_upsampling_ratios = std::max(0, std::min(q3.n_upsampling_ratios, CODEC_Q3T_MAX_UPSAMPLE));
    for (int32_t i = 0; i < build->n_upsample_rates; ++i) {
        build->upsample_rates[i] = std::max(1, q3.upsample_rates[i]);
    }
    for (int32_t i = 0; i < build->n_upsampling_ratios; ++i) {
        build->upsampling_ratios[i] = std::max(1, q3.upsampling_ratios[i]);
    }
    build->model = ctx->model;
    return true;
}


enum codec_status codec_qwen3_tts_tokenizer_decode(
    struct codec_context * ctx,
    const struct codec_token_buffer * tokens,
    struct codec_pcm_buffer * out_pcm,
    struct codec_decode_params params) {

    if (ctx == nullptr || ctx->model == nullptr || tokens == nullptr || out_pcm == nullptr) {
        return CODEC_STATUS_INVALID_ARG;
    }

    codec_qwen3_tts_tokenizer_impl & impl = *static_cast<codec_qwen3_tts_tokenizer_impl *>(ctx->model->impl);
    codec_qwen3_tts_tokenizer & q3 = impl.q3;
    if (!q3.has_decoder) {
        codec_context_set_error(ctx, "model metadata indicates no decoder");
        return CODEC_STATUS_INVALID_STATE;
    }

    const int32_t model_n_q = std::max(1, q3.n_q);
    const int32_t use_n_q = params.n_q == 0 ? model_n_q : params.n_q;
    if (params.n_q < 0 || use_n_q < 1 || use_n_q > model_n_q) {
        codec_context_set_error(ctx, "Qwen3 decode n_q must be 0 or in [1, model_n_q]");
        return CODEC_STATUS_INVALID_ARG;
    }

    if (tokens->data == nullptr || tokens->n_frames <= 0 || tokens->n_q < use_n_q) {
        codec_context_set_error(ctx, "invalid Qwen3 token buffer");
        return CODEC_STATUS_INVALID_ARG;
    }

    const int32_t t = tokens->n_frames;
    const int32_t q = use_n_q;
    codec_graph_eval_guard eval_guard(ctx);
    q3t_decode_build build = {};
    std::string err;
    if (!codec_q3t_init_decode_build(ctx, t, q, &build, &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }
    codec_graph_cache_entry * entry = nullptr;
    if (!codec_graph_cache_get_or_build(
            ctx,
            { CODEC_GRAPH_Q3T_DECODE, /*n_frames=*/t, /*n_q=*/q, /*hop=*/build.codebook_dim, /*n_in=*/0, /*latent_dim=*/build.latent_dim },
            codec_q3t_build_decode,
            &build,
            sizeof(build),
            &entry,
            &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    lm_ggml_tensor * t_out = codec_graph_get_tensor(ctx, entry, "q3t.dec.out");
    if (t_out == nullptr) {
        codec_context_set_error(ctx, "cached Qwen3 decode graph is invalid");
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    if (!codec_graph_prepare_io(ctx, entry, &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    // tokens -> per-quantizer index vectors (column-major layout)
    std::vector<int32_t> tok_i32((size_t) t * (size_t) q, 0);
    for (int32_t ti = 0; ti < t; ++ti) {
        for (int32_t qi = 0; qi < q; ++qi) {
            int32_t tok = tokens->data[(size_t) ti * (size_t) tokens->n_q + (size_t) qi];
            tok = std::max(0, std::min(build.codebook_size - 1, tok));
            tok_i32[(size_t) qi * (size_t) t + (size_t) ti] = tok;
        }
    }
    for (int32_t qi = 0; qi < q; ++qi) {
        const std::string name = codec_q3t_decode_idx_tensor_name(qi);
        lm_ggml_tensor * t_idx = codec_graph_get_tensor(ctx, entry, name.c_str());
        if (t_idx == nullptr) {
            codec_context_set_error(ctx, "cached Qwen3 decode graph is invalid");
            return CODEC_STATUS_INTERNAL_ERROR;
        }
        const size_t offset = (size_t) qi * (size_t) t;
        if (!codec_runtime_write_tensor(t_idx, tok_i32.data() + offset, (size_t) t * sizeof(int32_t), &err)) {
            codec_context_set_error(ctx, err);
            return CODEC_STATUS_INTERNAL_ERROR;
        }
    }

    const int32_t n_threads = ctx->model->n_threads > 0 ? ctx->model->n_threads : 1;
    if (!codec_graph_compute(ctx, entry, n_threads, &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    const int32_t n_samples = (int32_t) t_out->ne[0];
    std::vector<float> out(n_samples, 0.0f);
    if (!codec_runtime_read_tensor(t_out, out.data(), out.size() * sizeof(float), &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    float * data = static_cast<float *>(std::malloc(out.size() * sizeof(float)));
    if (data == nullptr) {
        codec_context_set_error(ctx, "failed to allocate output PCM");
        return CODEC_STATUS_INTERNAL_ERROR;
    }
    std::copy(out.begin(), out.end(), data);

    codec_pcm_buffer_reset(out_pcm);
    out_pcm->data = data;
    out_pcm->n_samples = n_samples;
    out_pcm->sample_rate = q3.sample_rate;
    out_pcm->n_channels = 1;

    return CODEC_STATUS_SUCCESS;
}

enum codec_status codec_qwen3_tts_tokenizer_encode(
    struct codec_context * ctx,
    const std::vector<float> & pcm,
    struct codec_token_buffer * out_tokens,
    struct codec_encode_params params) {

    if (ctx == nullptr || ctx->model == nullptr || ctx->model->impl == nullptr) {
        return CODEC_STATUS_INVALID_ARG;
    }
    codec_qwen3_tts_tokenizer_impl & impl = *static_cast<codec_qwen3_tts_tokenizer_impl *>(ctx->model->impl);
    return codec_mimi_encode_with(ctx, &impl.mimi, pcm, out_tokens, params);
}

static void * codec_qwen3_tts_tokenizer_create_impl() {
    return new (std::nothrow) codec_qwen3_tts_tokenizer_impl();
}

static void codec_qwen3_tts_tokenizer_destroy_impl(void * ptr) {
    delete static_cast<codec_qwen3_tts_tokenizer_impl *>(ptr);
}

static enum codec_status codec_qwen3_tts_tokenizer_encode_wrap(
    struct codec_context * ctx,
    const std::vector<float> & pcm,
    struct codec_token_buffer * out_tokens,
    struct codec_latent_buffer * /*out_latent*/,
    struct codec_encode_params params) {
    return codec_qwen3_tts_tokenizer_encode(ctx, pcm, out_tokens, params);
}

static enum codec_status codec_qwen3_tts_tokenizer_decode_wrap(
    struct codec_context * ctx,
    const struct codec_token_buffer * tokens,
    struct codec_pcm_buffer * out_pcm,
    struct codec_decode_params params) {
    return codec_qwen3_tts_tokenizer_decode(ctx, tokens, out_pcm, params);
}

const struct codec_model_vtable * codec_qwen3_tts_tokenizer_vtable() {
    static const codec_model_vtable vtable = {
        CODEC_ARCH_QWEN3_TTS_TOKENIZER,
        "Qwen3-TTS-Tokenizer",
        codec_qwen3_tts_tokenizer_create_impl,
        codec_qwen3_tts_tokenizer_destroy_impl,
        codec_qwen3_tts_tokenizer_init,
        codec_graph_size_exact,
        codec_qwen3_tts_tokenizer_encode_wrap,
        codec_qwen3_tts_tokenizer_decode_wrap,
        nullptr,
    };
    return &vtable;
}
