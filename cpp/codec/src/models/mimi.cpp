#include "mimi.h"

#include "../ops/conv1d.h"
#include "../ops/convtr1d.h"
#include "../ops/lm_ggml_ops.h"
#include "../ops/lm_attn.h"
#include "../ops/rope.h"
#include "../ops/rvq.h"
#include "../runtime/graph.h"
#include "../runtime/tensor_utils.h"

#include <algorithm>
#include <array>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <new>
#include <string>
#include <vector>

enum codec_status codec_mimi_init(struct codec_model * model) {
    codec_mimi & mimi = *static_cast<codec_mimi *>(model->impl);

    mimi.sample_rate = codec_read_i32_kv(model->gguf, "codec.sample_rate", 24000);
    mimi.hop_size = codec_read_i32_kv(model->gguf, "codec.hop_size", 1920);
    mimi.n_q = codec_read_i32_kv(model->gguf, "codec.n_q", 32);
    mimi.num_semantic_quantizers = codec_read_i32_kv(model->gguf, "codec.num_semantic_quantizers", 1);
    mimi.codebook_size = codec_read_i32_kv(model->gguf, "codec.codebook_size", 2048);
    mimi.codebook_dim = codec_read_i32_kv(model->gguf, "codec.codebook_dim", 256);
    mimi.hidden_size = codec_read_i32_kv(model->gguf, "codec.latent_dim", 512);
    mimi.num_hidden_layers = codec_read_i32_kv(model->gguf, "codec.num_hidden_layers", 8);
    mimi.num_attention_heads = codec_read_i32_kv(model->gguf, "codec.num_attention_heads", 8);
    mimi.head_dim = codec_read_i32_kv(model->gguf, "codec.head_dim", 64);
    mimi.intermediate_size = codec_read_i32_kv(model->gguf, "codec.intermediate_size", 2048);
    mimi.rope_theta = codec_read_f32_kv(model->gguf, "codec.rope_theta", 10000.0f);
    mimi.rope_scaling_factor = codec_read_f32_kv(model->gguf, "codec.rope_scaling_factor", 1.0f);
    mimi.has_encoder = codec_read_bool_kv(model->gguf, "codec.has_encoder", false);
    mimi.has_decoder = codec_read_bool_kv(model->gguf, "codec.has_decoder", true);

    model->sample_rate = mimi.sample_rate;
    model->has_encoder = mimi.has_encoder;
    model->has_decoder = mimi.has_decoder;
    model->hop_size = mimi.hop_size;
    model->n_q = mimi.n_q;
    model->codebook_size = mimi.codebook_size;
    model->n_fft = -1;
    model->win_length = -1;
    model->n_mels = -1;
    model->latent_dim = mimi.hidden_size;

    return CODEC_STATUS_SUCCESS;
}

struct mimi_decode_build {
    int32_t t;
    int32_t q;
    int32_t hop;
    int32_t n_sem;
    int32_t codebook_dim;
    int32_t hidden_size;
    int32_t codebook_size;
    int32_t upsample_kernel;
    int32_t upsample_stride;
    int32_t transformer_layers;
    int32_t transformer_heads;
    int32_t transformer_head_dim;
    int32_t transformer_intermediate;
    float rope_theta;
    float rope_scaling_factor;
    int32_t dec_l0_kernel;
    int32_t dec_l0_out;
    int32_t dec_l2_kernel;
    int32_t dec_l2_out;
    int32_t dec_l5_kernel;
    int32_t dec_l5_out;
    int32_t dec_l8_kernel;
    int32_t dec_l8_out;
    int32_t dec_l11_kernel;
    int32_t dec_l11_out;
    int32_t dec_l14_kernel;
    int32_t dec_l14_out;
    const codec_model * model;
};

struct mimi_encode_frontend_conv_desc {
    int32_t out_c;
    int32_t in_c;
    int32_t kernel;
    int32_t stride;
};

struct mimi_encode_frontend_build {
    int32_t n_in;
    std::array<mimi_encode_frontend_conv_desc, 14> conv;
};

static constexpr std::array<const char *, 14> CODEC_MIMI_ENC_FRONTEND_WEIGHT_NAMES = {
    "enc.l0.conv.w",
    "enc.l1.block.1.conv.w",
    "enc.l1.block.3.conv.w",
    "enc.l3.conv.w",
    "enc.l4.block.1.conv.w",
    "enc.l4.block.3.conv.w",
    "enc.l6.conv.w",
    "enc.l7.block.1.conv.w",
    "enc.l7.block.3.conv.w",
    "enc.l9.conv.w",
    "enc.l10.block.1.conv.w",
    "enc.l10.block.3.conv.w",
    "enc.l12.conv.w",
    "enc.l14.conv.w",
};

static constexpr std::array<const char *, 14> CODEC_MIMI_ENC_FRONTEND_BIAS_NAMES = {
    "enc.l0.conv.b",
    "enc.l1.block.1.conv.b",
    "enc.l1.block.3.conv.b",
    "enc.l3.conv.b",
    "enc.l4.block.1.conv.b",
    "enc.l4.block.3.conv.b",
    "enc.l6.conv.b",
    "enc.l7.block.1.conv.b",
    "enc.l7.block.3.conv.b",
    "enc.l9.conv.b",
    "enc.l10.block.1.conv.b",
    "enc.l10.block.3.conv.b",
    "enc.l12.conv.b",
    "enc.l14.conv.b",
};

static constexpr std::array<int32_t, 14> CODEC_MIMI_ENC_FRONTEND_STRIDES = {
    1, 1, 1, 4, 1, 1, 5, 1, 1, 6, 1, 1, 8, 1,
};



static bool codec_mimi_init_encode_frontend_build(codec_context * ctx, int32_t n_in, mimi_encode_frontend_build * build, std::string * err) {
    if (ctx == nullptr || ctx->model == nullptr || build == nullptr || n_in <= 0) {
        if (err != nullptr) {
            *err = "invalid Mimi encode frontend build arguments";
        }
        return false;
    }

    build->n_in = n_in;
    int32_t prev_c = 1;
    for (size_t i = 0; i < CODEC_MIMI_ENC_FRONTEND_WEIGHT_NAMES.size(); ++i) {
        lm_ggml_tensor * tw = codec_model_get_tensor(ctx->model, CODEC_MIMI_ENC_FRONTEND_WEIGHT_NAMES[i]);
        lm_ggml_tensor * tb = codec_model_get_tensor(ctx->model, CODEC_MIMI_ENC_FRONTEND_BIAS_NAMES[i]);
        if (tw == nullptr || tb == nullptr) {
            if (err != nullptr) {
                *err = "missing Mimi encode frontend tensor at layer " + std::to_string(i);
            }
            return false;
        }
        if (codec_ne(tw, 0) <= 0 || codec_ne(tw, 1) <= 0 || codec_ne(tw, 2) <= 0 || codec_ne(tb, 0) != codec_ne(tw, 2)) {
            if (err != nullptr) {
                *err = "invalid Mimi encode frontend tensor shape at layer " + std::to_string(i);
            }
            return false;
        }

        mimi_encode_frontend_conv_desc d = {
            (int32_t) codec_ne(tw, 2),
            (int32_t) codec_ne(tw, 1),
            (int32_t) codec_ne(tw, 0),
            CODEC_MIMI_ENC_FRONTEND_STRIDES[i],
        };
        if (d.in_c != prev_c) {
            if (err != nullptr) {
                *err = "Mimi encode frontend channel mismatch at layer " + std::to_string(i);
            }
            return false;
        }
        build->conv[i] = d;
        prev_c = d.out_c;
    }

    return true;
}

static int32_t codec_mimi_encode_time_stride_reduce(int32_t t, int32_t stride) {
    if (t <= 0 || stride <= 0) {
        return 0;
    }
    return (t + stride - 1) / stride;
}

static lm_ggml_tensor * codec_mimi_resblock_ggml(
    lm_ggml_context * ctx_eval,
    lm_ggml_tensor * x,
    lm_ggml_tensor * w1,
    lm_ggml_tensor * b1,
    lm_ggml_tensor * w2,
    lm_ggml_tensor * b2) {

    if (ctx_eval == nullptr || x == nullptr || w1 == nullptr || b1 == nullptr || w2 == nullptr || b2 == nullptr) {
        return nullptr;
    }

    lm_ggml_tensor * h = lm_ggml_elu(ctx_eval, x);
    lm_ggml_tensor * y1 = codec_conv1d_causal(ctx_eval, h, w1, b1, 1, 1);
    if (y1 == nullptr) {
        return nullptr;
    }
    y1 = lm_ggml_elu(ctx_eval, y1);
    lm_ggml_tensor * y2 = codec_conv1d_causal(ctx_eval, y1, w2, b2, 1, 1);
    if (y2 == nullptr) {
        return nullptr;
    }
    return lm_ggml_cont(ctx_eval, lm_ggml_add(ctx_eval, x, y2));
}

struct mimi_encode_transformer_build {
    int32_t t;
    int32_t c;
    int32_t n_layers;
    int32_t n_heads;
    int32_t head_dim;
    int32_t intermediate_size;
    float rope_theta;
    float rope_scaling_factor;
};

struct mimi_encode_downsample_build {
    int32_t t;
    int32_t in_c;
    int32_t out_c;
    int32_t kernel;
    int32_t stride;
};

static constexpr int32_t CODEC_MIMI_MAX_RVQ_LAYERS = 32;
static constexpr int32_t CODEC_MIMI_RVQ_GROUP_SEMANTIC = 0;
static constexpr int32_t CODEC_MIMI_RVQ_GROUP_ACOUSTIC = 1;

struct mimi_encode_rvq_layer_desc {
    int32_t group = CODEC_MIMI_RVQ_GROUP_SEMANTIC;
    int32_t group_layer = 0;
};

struct mimi_encode_build {
    mimi_encode_frontend_build frontend = {};
    mimi_encode_transformer_build transformer = {};
    mimi_encode_downsample_build downsample = {};
    int32_t n_q = 0;
    int32_t codebook_dim = 0;
    int32_t codebook_size = 0;
    std::array<mimi_encode_rvq_layer_desc, CODEC_MIMI_MAX_RVQ_LAYERS> rvq_layers = {};
    const codec_model * model = nullptr;
};

static std::string codec_mimi_encode_rvq_indices_tensor_name(int32_t layer) {
    return "mimi.encode_unified.rvq.l" + std::to_string(layer) + ".indices";
}

static const char * CODEC_MIMI_ENCODE_PCM_TENSOR = "mimi.encode_unified.pcm";
static const char * CODEC_MIMI_ENCODE_INDICES_TENSOR = "mimi.encode_unified.indices";
static const char * CODEC_MIMI_ENCODE_RVQ_SEM_IP_TENSOR = "mimi.encode_unified.rvq.q.s.ip.w";
static const char * CODEC_MIMI_ENCODE_RVQ_ACU_IP_TENSOR = "mimi.encode_unified.rvq.q.a.ip.w";

static bool codec_mimi_init_encode_build(
    codec_context * ctx,
    const codec_mimi * mimi,
    int32_t n_in,
    mimi_encode_build * build,
    std::string * err) {

    if (ctx == nullptr || ctx->model == nullptr || mimi == nullptr || build == nullptr || n_in <= 0) {
        if (err != nullptr) {
            *err = "invalid Mimi encode build arguments";
        }
        return false;
    }

    const codec_mimi & mm = *mimi;
    if (!codec_mimi_init_encode_frontend_build(ctx, n_in, &build->frontend, err)) {
        return false;
    }

    int32_t t_frontend = n_in;
    for (const mimi_encode_frontend_conv_desc & d : build->frontend.conv) {
        t_frontend = codec_mimi_encode_time_stride_reduce(t_frontend, d.stride);
        if (t_frontend <= 0) {
            if (err != nullptr) {
                *err = "Mimi encode frontend produced invalid length";
            }
            return false;
        }
    }

    const int32_t hidden = build->frontend.conv.back().out_c;
    const int32_t n_layers = std::max(1, mm.num_hidden_layers);
    const int32_t n_heads = std::max(1, mm.num_attention_heads);
    const int32_t head_dim = std::max(1, mm.head_dim);
    if (n_heads * head_dim != hidden) {
        if (err != nullptr) {
            *err = "Mimi encoder transformer config mismatch";
        }
        return false;
    }

    build->transformer = {
        /*t=*/t_frontend,
        /*c=*/hidden,
        /*n_layers=*/n_layers,
        /*n_heads=*/n_heads,
        /*head_dim=*/head_dim,
        /*intermediate_size=*/mm.intermediate_size > 0 ? mm.intermediate_size : 2048,
        /*rope_theta=*/mm.rope_theta,
        /*rope_scaling_factor=*/mm.rope_scaling_factor,
    };

    lm_ggml_tensor * downsample_w = codec_model_get_tensor(ctx->model, "dn.cv.w");
    if (downsample_w == nullptr || codec_ne(downsample_w, 0) <= 0 || codec_ne(downsample_w, 1) <= 0 || codec_ne(downsample_w, 2) <= 0) {
        if (err != nullptr) {
            *err = "missing Mimi downsample weight";
        }
        return false;
    }

    build->downsample = {
        /*t=*/t_frontend,
        /*in_c=*/(int32_t) codec_ne(downsample_w, 1),
        /*out_c=*/(int32_t) codec_ne(downsample_w, 2),
        /*kernel=*/(int32_t) codec_ne(downsample_w, 0),
        /*stride=*/2,
    };
    if (build->downsample.in_c != hidden || build->downsample.out_c != hidden) {
        if (err != nullptr) {
            *err = "Mimi downsample channel mismatch";
        }
        return false;
    }

    build->n_q = std::max(1, std::min(mm.n_q, CODEC_MIMI_MAX_RVQ_LAYERS));
    build->codebook_dim = std::max(1, mm.codebook_dim);
    build->codebook_size = std::max(2, mm.codebook_size);

    const int32_t n_sem_cfg = std::max(0, mm.num_semantic_quantizers);
    const int32_t n_sem = std::max(1, std::min(n_sem_cfg, build->n_q));
    for (int32_t qi = 0; qi < build->n_q; ++qi) {
        if (qi < n_sem) {
            build->rvq_layers[(size_t) qi] = { CODEC_MIMI_RVQ_GROUP_SEMANTIC, qi };
        } else {
            build->rvq_layers[(size_t) qi] = { CODEC_MIMI_RVQ_GROUP_ACOUSTIC, qi - n_sem };
        }
    }

    return true;
}




static bool codec_mimi_build_encode(lm_ggml_context * ctx_eval, void * user_data, lm_ggml_tensor ** out) {
    mimi_encode_build * p = static_cast<mimi_encode_build *>(user_data);
    if (ctx_eval == nullptr || p == nullptr || out == nullptr || p->model == nullptr) {
        return false;
    }
    if (p->frontend.n_in <= 0 || p->transformer.t <= 0 || p->transformer.c <= 0 || p->n_q <= 0 || p->n_q > CODEC_MIMI_MAX_RVQ_LAYERS) {
        return false;
    }

    auto W = [&](const std::string & name) { return codec_graph_weight(ctx_eval, p->model, name); };

    lm_ggml_tensor * t_pcm = lm_ggml_new_tensor_2d(ctx_eval, LM_GGML_TYPE_F32, p->frontend.n_in, 1);
    lm_ggml_set_name(t_pcm, CODEC_MIMI_ENCODE_PCM_TENSOR);

    // Frontend conv weights — names live under the GGUF prefixes captured in
    // CODEC_MIMI_ENC_FRONTEND_WEIGHT_NAMES / _BIAS_NAMES (e.g. enc.l0.conv.w).
    std::array<lm_ggml_tensor *, 14> t_w = {};
    std::array<lm_ggml_tensor *, 14> t_b = {};
    for (size_t i = 0; i < p->frontend.conv.size(); ++i) {
        const mimi_encode_frontend_conv_desc & d = p->frontend.conv[i];
        if (d.out_c <= 0 || d.in_c <= 0 || d.kernel <= 0 || d.stride <= 0) {
            return false;
        }
        t_w[i] = W(CODEC_MIMI_ENC_FRONTEND_WEIGHT_NAMES[i]);
        t_b[i] = W(CODEC_MIMI_ENC_FRONTEND_BIAS_NAMES[i]);
        if (t_w[i] == nullptr || t_b[i] == nullptr) {
            return false;
        }
    }

    lm_ggml_tensor * x = codec_conv1d_causal(ctx_eval, t_pcm, t_w[0], t_b[0], p->frontend.conv[0].stride, 1);
    if (x == nullptr) {
        return false;
    }
    lm_ggml_tensor * t_l0_dbg = lm_ggml_new_tensor_2d(ctx_eval, LM_GGML_TYPE_F32, x->ne[0], x->ne[1]);
    lm_ggml_set_name(t_l0_dbg, "mimi.encode_frontend.l0");
    x = lm_ggml_cpy(ctx_eval, x, t_l0_dbg);
    x = codec_mimi_resblock_ggml(ctx_eval, x, t_w[1], t_b[1], t_w[2], t_b[2]);
    if (x == nullptr) {
        return false;
    }
    x = lm_ggml_elu(ctx_eval, x);

    x = codec_conv1d_causal(ctx_eval, x, t_w[3], t_b[3], p->frontend.conv[3].stride, 1);
    if (x == nullptr) {
        return false;
    }
    x = codec_mimi_resblock_ggml(ctx_eval, x, t_w[4], t_b[4], t_w[5], t_b[5]);
    if (x == nullptr) {
        return false;
    }
    x = lm_ggml_elu(ctx_eval, x);

    x = codec_conv1d_causal(ctx_eval, x, t_w[6], t_b[6], p->frontend.conv[6].stride, 1);
    if (x == nullptr) {
        return false;
    }
    x = codec_mimi_resblock_ggml(ctx_eval, x, t_w[7], t_b[7], t_w[8], t_b[8]);
    if (x == nullptr) {
        return false;
    }
    x = lm_ggml_elu(ctx_eval, x);

    x = codec_conv1d_causal(ctx_eval, x, t_w[9], t_b[9], p->frontend.conv[9].stride, 1);
    if (x == nullptr) {
        return false;
    }
    x = codec_mimi_resblock_ggml(ctx_eval, x, t_w[10], t_b[10], t_w[11], t_b[11]);
    if (x == nullptr) {
        return false;
    }
    x = lm_ggml_elu(ctx_eval, x);

    x = codec_conv1d_causal(ctx_eval, x, t_w[12], t_b[12], p->frontend.conv[12].stride, 1);
    if (x == nullptr) {
        return false;
    }
    x = lm_ggml_elu(ctx_eval, x);
    x = codec_conv1d_causal(ctx_eval, x, t_w[13], t_b[13], p->frontend.conv[13].stride, 1);
    if (x == nullptr) {
        return false;
    }

    lm_ggml_tensor * t_front_dbg = lm_ggml_new_tensor_2d(ctx_eval, LM_GGML_TYPE_F32, x->ne[0], x->ne[1]);
    lm_ggml_set_name(t_front_dbg, "mimi.encode_frontend.out");
    x = lm_ggml_cpy(ctx_eval, x, t_front_dbg);

    // Transformer layers operate on channel-major [c, t].
    x = lm_ggml_cont(ctx_eval, lm_ggml_transpose(ctx_eval, x));

    const float freq_scale = p->transformer.rope_scaling_factor > 0.0f ? 1.0f / p->transformer.rope_scaling_factor : 1.0f;

    for (int32_t li = 0; li < p->transformer.n_layers; ++li) {
        // Encode transformer GGUF prefix is `etr.lX.*` per write_encode_transformer_weights mapping.
        const std::string lp = "etr.l" + std::to_string(li);
        lm_ggml_tensor * inln_w = W(lp + ".inln.w");
        lm_ggml_tensor * inln_b = W(lp + ".inln.b");
        lm_ggml_tensor * paln_w = W(lp + ".paln.w");
        lm_ggml_tensor * paln_b = W(lp + ".paln.b");
        lm_ggml_tensor * q_w = W(lp + ".attn.q_proj.w");
        lm_ggml_tensor * k_w = W(lp + ".attn.k_proj.w");
        lm_ggml_tensor * v_w = W(lp + ".attn.v_proj.w");
        lm_ggml_tensor * o_w = W(lp + ".attn.o_proj.w");
        lm_ggml_tensor * fc1_w = W(lp + ".mlp.fc1.w");
        lm_ggml_tensor * fc2_w = W(lp + ".mlp.fc2.w");
        lm_ggml_tensor * sa_scale = W(lp + ".sa_ls.scale");
        lm_ggml_tensor * mlp_scale = W(lp + ".mlp_ls.scale");
        if (inln_w == nullptr || inln_b == nullptr || paln_w == nullptr || paln_b == nullptr ||
            q_w == nullptr || k_w == nullptr || v_w == nullptr || o_w == nullptr ||
            fc1_w == nullptr || fc2_w == nullptr || sa_scale == nullptr || mlp_scale == nullptr) {
            return false;
        }

        lm_ggml_tensor * h = codec_op_layer_norm_ct(ctx_eval, x, 1e-5f, inln_w, inln_b);
        if (h == nullptr) {
            return false;
        }

        lm_ggml_tensor * q = lm_ggml_mul_mat(ctx_eval, q_w, h);
        lm_ggml_tensor * k = lm_ggml_mul_mat(ctx_eval, k_w, h);
        lm_ggml_tensor * v = lm_ggml_mul_mat(ctx_eval, v_w, h);
        if (q == nullptr || k == nullptr || v == nullptr) {
            return false;
        }

        const int64_t t_cur = q->ne[1];
        lm_ggml_tensor * q_dth = lm_ggml_permute(ctx_eval, lm_ggml_reshape_3d(ctx_eval, q, p->transformer.head_dim, p->transformer.n_heads, t_cur), 0, 2, 1, 3);
        lm_ggml_tensor * k_dth = lm_ggml_permute(ctx_eval, lm_ggml_reshape_3d(ctx_eval, k, p->transformer.head_dim, p->transformer.n_heads, t_cur), 0, 2, 1, 3);
        lm_ggml_tensor * v_dth = lm_ggml_permute(ctx_eval, lm_ggml_reshape_3d(ctx_eval, v, p->transformer.head_dim, p->transformer.n_heads, t_cur), 0, 2, 1, 3);

        lm_ggml_tensor * q_rope = codec_op_rope(ctx_eval, q_dth, p->transformer.head_dim, p->transformer.rope_theta, freq_scale, CODEC_ROPE_MODE_NEOX);
        lm_ggml_tensor * k_rope = codec_op_rope(ctx_eval, k_dth, p->transformer.head_dim, p->transformer.rope_theta, freq_scale, CODEC_ROPE_MODE_NEOX);
        if (q_rope == nullptr || k_rope == nullptr) {
            return false;
        }

        codec_lm_attn_params attn_p = {};
        attn_p.scale = 1.0f / std::sqrt((float) p->transformer.head_dim);
        attn_p.causal = true;
        lm_ggml_tensor * attn_ctx = codec_op_lm_attn_ctx_dth(ctx_eval, q_rope, k_rope, v_dth, &attn_p);
        if (attn_ctx == nullptr) {
            return false;
        }
        lm_ggml_tensor * attn_ct = lm_ggml_reshape_2d(
            ctx_eval,
            lm_ggml_cont(ctx_eval, lm_ggml_permute(ctx_eval, attn_ctx, 0, 2, 1, 3)),
            p->transformer.c,
            t_cur);
        if (attn_ct == nullptr) {
            return false;
        }
        lm_ggml_tensor * attn_proj = lm_ggml_mul_mat(ctx_eval, o_w, attn_ct);
        if (attn_proj == nullptr) {
            return false;
        }
        x = lm_ggml_add(ctx_eval, x, codec_op_channel_scale(ctx_eval, attn_proj, sa_scale));

        lm_ggml_tensor * m = codec_op_layer_norm_ct(ctx_eval, x, 1e-5f, paln_w, paln_b);
        if (m == nullptr) {
            return false;
        }
        m = lm_ggml_mul_mat(ctx_eval, fc1_w, m);
        if (m == nullptr) {
            return false;
        }
        m = lm_ggml_gelu_erf(ctx_eval, m);
        m = lm_ggml_mul_mat(ctx_eval, fc2_w, m);
        if (m == nullptr) {
            return false;
        }
        x = lm_ggml_add(ctx_eval, x, codec_op_channel_scale(ctx_eval, m, mlp_scale));
    }
    // Convolution/downsample path expects time-major [t, c].
    x = lm_ggml_cont(ctx_eval, lm_ggml_transpose(ctx_eval, x));
    lm_ggml_tensor * t_x_dbg = lm_ggml_new_tensor_2d(ctx_eval, LM_GGML_TYPE_F32, x->ne[0], x->ne[1]);
    lm_ggml_set_name(t_x_dbg, "mimi.encode_transformer.out");
    x = lm_ggml_cpy(ctx_eval, x, t_x_dbg);

    lm_ggml_tensor * t_downsample_w = W("dn.cv.w");
    if (t_downsample_w == nullptr) {
        return false;
    }
    x = codec_conv1d_causal_replicate(ctx_eval, x, t_downsample_w, nullptr, p->downsample.stride, 1);
    if (x == nullptr) {
        return false;
    }
    lm_ggml_set_name(x, "mimi.encode_downsample.out");

    lm_ggml_tensor * t_qs_ip_w = W("q.s.ip.w");
    lm_ggml_tensor * t_qa_ip_w = W("q.a.ip.w");
    if (t_qs_ip_w == nullptr || t_qa_ip_w == nullptr) {
        return false;
    }

    lm_ggml_tensor * x_ct = lm_ggml_cont(ctx_eval, lm_ggml_transpose(ctx_eval, x)); // [downsample.out_c, t]
    lm_ggml_tensor * sem_residual = lm_ggml_mul_mat(ctx_eval, t_qs_ip_w, x_ct);
    lm_ggml_tensor * acu_residual = lm_ggml_mul_mat(ctx_eval, t_qa_ip_w, x_ct);
    if (sem_residual == nullptr || acu_residual == nullptr) {
        return false;
    }
    lm_ggml_set_name(sem_residual, "mimi.encode_rvq.sem_residual.l0");
    lm_ggml_set_name(acu_residual, "mimi.encode_rvq.acu_residual.l0");

    std::array<lm_ggml_tensor *, CODEC_MIMI_MAX_RVQ_LAYERS> layer_indices = {};
    for (int32_t li = 0; li < p->n_q; ++li) {
        const mimi_encode_rvq_layer_desc & d = p->rvq_layers[(size_t) li];
        // Codebook is materialized in the converter; mimi stores under
        // `.codebook.embed` (long) and Qwen3-TTS-Tokenizer (which reuses the
        // mimi encoder) uses `.cb.embed` (short). Try both.
        const std::string cb_base = (d.group == CODEC_MIMI_RVQ_GROUP_SEMANTIC ? "q.s.layers." : "q.a.layers.")
            + std::to_string(d.group_layer);
        lm_ggml_tensor * t_codebook = codec_graph_weight_or_null(ctx_eval, p->model, cb_base + ".codebook.embed");
        if (t_codebook == nullptr) {
            t_codebook = codec_graph_weight(ctx_eval, p->model, cb_base + ".cb.embed");
        }
        if (t_codebook == nullptr) {
            return false;
        }

        if (d.group == CODEC_MIMI_RVQ_GROUP_ACOUSTIC && d.group_layer == 4) {
            lm_ggml_tensor * t_dbg = lm_ggml_new_tensor_2d(ctx_eval, LM_GGML_TYPE_F32, acu_residual->ne[0], acu_residual->ne[1]);
            lm_ggml_set_name(t_dbg, "mimi.encode_rvq.acu_input_q5");
            acu_residual = lm_ggml_cpy(ctx_eval, acu_residual, t_dbg);
        }
        codec_rvq_layer_result_ggml res = {};
        if (d.group == CODEC_MIMI_RVQ_GROUP_SEMANTIC) {
            if (!codec_rvq_build_layer_ggml(ctx_eval, sem_residual, t_codebook, &res)) {
                return false;
            }
            lm_ggml_set_name(res.residual, ("mimi.encode_rvq.sem_residual.l" + std::to_string(li + 1)).c_str());
            sem_residual = res.residual;
        } else {
            if (!codec_rvq_build_layer_ggml(ctx_eval, acu_residual, t_codebook, &res)) {
                return false;
            }
            lm_ggml_set_name(res.residual, ("mimi.encode_rvq.acu_residual.l" + std::to_string(d.group_layer + 1)).c_str());
            acu_residual = res.residual;
        }

        lm_ggml_set_name(res.indices, codec_mimi_encode_rvq_indices_tensor_name(li).c_str());
        layer_indices[(size_t) li] = lm_ggml_reshape_2d(ctx_eval, res.indices, 1, res.indices->ne[0]);
        if (layer_indices[(size_t) li] == nullptr) {
            return false;
        }
    }

    lm_ggml_tensor * indices_qt = layer_indices[0];
    for (int32_t li = 1; li < p->n_q; ++li) {
        indices_qt = lm_ggml_concat(ctx_eval, indices_qt, layer_indices[(size_t) li], 0);
    }

    lm_ggml_tensor * indices_tq = lm_ggml_cont(ctx_eval, lm_ggml_transpose(ctx_eval, indices_qt));
    lm_ggml_set_name(indices_tq, CODEC_MIMI_ENCODE_INDICES_TENSOR);
    *out = indices_tq;
    return true;
}


// Materialized RVQ codebooks live under `q.{s,a}.layers.X.codebook.embed` with
// shape (codebook_dim, codebook_size). The mimi converter pre-computes these
// from `embed_sum / cluster_usage` so the runtime can lm_ggml_get_rows directly.
// `q_begin..q_end` selects the semantic (0..n_sem) or acoustic (n_sem..q) slice.
static lm_ggml_tensor * codec_mimi_sum_codebook_lookup(
    lm_ggml_context * ctx_eval,
    const codec_model * model,
    lm_ggml_tensor * t_tok,
    int32_t t,
    int32_t q_begin,
    int32_t q_end,
    int32_t n_sem) {

    if (model == nullptr) {
        return nullptr;
    }
    lm_ggml_tensor * sum = nullptr;
    for (int32_t qi = q_begin; qi < q_end; ++qi) {
        // Layers 0..n_sem-1 live under q.s; layers n_sem..end under q.a (with
        // index reset to 0). Match the converter's name remapping.
        const std::string base = qi < n_sem
            ? ("q.s.layers." + std::to_string(qi))
            : ("q.a.layers." + std::to_string(qi - n_sem));
        lm_ggml_tensor * t_codebook = codec_graph_weight(ctx_eval, model, base + ".codebook.embed");
        if (t_codebook == nullptr) {
            return nullptr;
        }

        lm_ggml_tensor * t_idx = lm_ggml_view_1d(ctx_eval, t_tok, t, (size_t) qi * t_tok->nb[1]);
        lm_ggml_tensor * t_qi = lm_ggml_get_rows(ctx_eval, t_codebook, t_idx); // [codebook_dim, t]
        if (t_qi == nullptr) {
            return nullptr;
        }
        sum = (sum == nullptr) ? t_qi : lm_ggml_add(ctx_eval, sum, t_qi);
    }
    return sum;
}

static bool codec_mimi_build_decode(lm_ggml_context * ctx_eval, void * user_data, lm_ggml_tensor ** out) {
    mimi_decode_build * p = static_cast<mimi_decode_build *>(user_data);
    if (ctx_eval == nullptr || p == nullptr || out == nullptr || p->t <= 0 || p->q <= 0 || p->n_sem <= 0 ||
        p->codebook_dim <= 0 || p->hidden_size <= 0 || p->codebook_size <= 1 || p->n_sem > p->q ||
        p->upsample_kernel <= 0 || p->upsample_stride <= 0 || p->transformer_layers <= 0 ||
        p->transformer_heads <= 0 || p->transformer_head_dim <= 0 || p->transformer_intermediate <= 0 ||
        p->transformer_heads * p->transformer_head_dim != p->hidden_size) {
        return false;
    }

    lm_ggml_tensor * t_tok = lm_ggml_new_tensor_2d(ctx_eval, LM_GGML_TYPE_I32, p->t, p->q);
    lm_ggml_set_name(t_tok, "mimi.decode.tok");

    auto mul_mat_checked = [&](const char * tag, lm_ggml_tensor * a, lm_ggml_tensor * b) -> lm_ggml_tensor * {
        if (a == nullptr || b == nullptr) {
            return nullptr;
        }
        if (a->ne[0] != b->ne[0] || (b->ne[2] % a->ne[2]) != 0 || (b->ne[3] % a->ne[3]) != 0) {
            (void) tag;
            return nullptr;
        }
        return lm_ggml_mul_mat(ctx_eval, a, b);
    };

    if (p->model == nullptr) {
        return false;
    }
    auto W = [&](const std::string & name) { return codec_graph_weight(ctx_eval, p->model, name); };

    // Per-layer RVQ decode: sum semantic/acoustic codebook vectors, then project each branch to hidden_size.
    const int32_t n_acu = std::max(0, p->q - p->n_sem);
    lm_ggml_tensor * t_sem_sum = codec_mimi_sum_codebook_lookup(ctx_eval, p->model, t_tok, p->t, 0, p->n_sem, p->n_sem);
    if (t_sem_sum == nullptr) {
        return false;
    }
    lm_ggml_tensor * t_sem_op_w = W("q.s.op.w");
    if (t_sem_op_w == nullptr) {
        return false;
    }
    lm_ggml_tensor * t_latent_ct = mul_mat_checked("rvq.sem_proj", t_sem_op_w, t_sem_sum); // [hidden, t]
    if (t_latent_ct == nullptr) {
        return false;
    }

    if (n_acu > 0) {
        lm_ggml_tensor * t_acu_sum = codec_mimi_sum_codebook_lookup(ctx_eval, p->model, t_tok, p->t, p->n_sem, p->q, p->n_sem);
        if (t_acu_sum == nullptr) {
            return false;
        }
        lm_ggml_tensor * t_acu_op_w = W("q.a.op.w");
        if (t_acu_op_w == nullptr) {
            return false;
        }
        lm_ggml_tensor * t_acu_latent_ct = mul_mat_checked("rvq.acu_proj", t_acu_op_w, t_acu_sum); // [hidden, t]
        if (t_acu_latent_ct == nullptr) {
            return false;
        }
        t_latent_ct = lm_ggml_add(ctx_eval, t_latent_ct, t_acu_latent_ct);
    }

    // Upsample ConvTranspose1d. The original HF weight is depthwise (k, 1, c);
    // the converter expands it to a dense diagonal (k, c, c) since lm_ggml_conv_transpose_1d isn't depthwise.
    lm_ggml_tensor * t_up_w = W("up.cv.w");
    if (t_up_w == nullptr) {
        return false;
    }

    lm_ggml_tensor * x = lm_ggml_cont(ctx_eval, lm_ggml_transpose(ctx_eval, t_latent_ct)); // [t, c]
    x = codec_convtr1d_causal(ctx_eval, x, t_up_w, nullptr, p->upsample_stride, 1);
    if (x == nullptr) {
        return false;
    }

    // Decoder transformer.
    lm_ggml_tensor * x_ct = lm_ggml_cont(ctx_eval, lm_ggml_transpose(ctx_eval, x)); // [c, t]
    const float freq_scale = p->rope_scaling_factor > 0.0f ? 1.0f / p->rope_scaling_factor : 1.0f;

    for (int32_t li = 0; li < p->transformer_layers; ++li) {
        // Decode transformer GGUF prefix is `dtr.lX.*` (per write_decode_transformer_weights).
        const std::string lp = "dtr.l" + std::to_string(li);
        lm_ggml_tensor * inln_w = W(lp + ".inln.w");
        lm_ggml_tensor * inln_b = W(lp + ".inln.b");
        lm_ggml_tensor * paln_w = W(lp + ".paln.w");
        lm_ggml_tensor * paln_b = W(lp + ".paln.b");
        lm_ggml_tensor * q_w = W(lp + ".attn.q_proj.w");
        lm_ggml_tensor * k_w = W(lp + ".attn.k_proj.w");
        lm_ggml_tensor * v_w = W(lp + ".attn.v_proj.w");
        lm_ggml_tensor * o_w = W(lp + ".attn.o_proj.w");
        lm_ggml_tensor * fc1_w = W(lp + ".mlp.fc1.w");
        lm_ggml_tensor * fc2_w = W(lp + ".mlp.fc2.w");
        lm_ggml_tensor * sa_scale = W(lp + ".sa_ls.scale");
        lm_ggml_tensor * mlp_scale = W(lp + ".mlp_ls.scale");
        if (inln_w == nullptr || inln_b == nullptr || paln_w == nullptr || paln_b == nullptr ||
            q_w == nullptr || k_w == nullptr || v_w == nullptr || o_w == nullptr ||
            fc1_w == nullptr || fc2_w == nullptr || sa_scale == nullptr || mlp_scale == nullptr) {
            return false;
        }

        lm_ggml_tensor * h = codec_op_layer_norm_ct(ctx_eval, x_ct, 1e-5f, inln_w, inln_b);
        const int32_t t_cur = (int32_t) h->ne[1];

        lm_ggml_tensor * q = mul_mat_checked("dtr.q_proj", q_w, h);
        lm_ggml_tensor * k = mul_mat_checked("dtr.k_proj", k_w, h);
        lm_ggml_tensor * v = mul_mat_checked("dtr.v_proj", v_w, h);
        if (q == nullptr || k == nullptr || v == nullptr) {
            return false;
        }

        lm_ggml_tensor * q_dth = lm_ggml_permute(ctx_eval, lm_ggml_reshape_3d(ctx_eval, q, p->transformer_head_dim, p->transformer_heads, t_cur), 0, 2, 1, 3);
        lm_ggml_tensor * k_dth = lm_ggml_permute(ctx_eval, lm_ggml_reshape_3d(ctx_eval, k, p->transformer_head_dim, p->transformer_heads, t_cur), 0, 2, 1, 3);
        lm_ggml_tensor * v_dth = lm_ggml_permute(ctx_eval, lm_ggml_reshape_3d(ctx_eval, v, p->transformer_head_dim, p->transformer_heads, t_cur), 0, 2, 1, 3);

        lm_ggml_tensor * q_rope = codec_op_rope(ctx_eval, q_dth, p->transformer_head_dim, p->rope_theta, freq_scale, CODEC_ROPE_MODE_NEOX);
        lm_ggml_tensor * k_rope = codec_op_rope(ctx_eval, k_dth, p->transformer_head_dim, p->rope_theta, freq_scale, CODEC_ROPE_MODE_NEOX);
        if (q_rope == nullptr || k_rope == nullptr) {
            return false;
        }

        codec_lm_attn_params attn_p = {};
        attn_p.scale = 1.0f / std::sqrt((float) p->transformer_head_dim);
        attn_p.causal = true;
        lm_ggml_tensor * ctx_3d = codec_op_lm_attn_ctx_dth(ctx_eval, q_rope, k_rope, v_dth, &attn_p);
        if (ctx_3d == nullptr) {
            return false;
        }
        lm_ggml_tensor * ctx_2d = lm_ggml_reshape_2d(
            ctx_eval,
            lm_ggml_cont(ctx_eval, lm_ggml_permute(ctx_eval, ctx_3d, 0, 2, 1, 3)),
            p->hidden_size,
            t_cur);
        lm_ggml_tensor * attn_out = mul_mat_checked("dtr.o_proj", o_w, ctx_2d);
        if (attn_out == nullptr) {
            return false;
        }
        x_ct = lm_ggml_add(ctx_eval, x_ct, codec_op_channel_scale(ctx_eval, attn_out, sa_scale));

        lm_ggml_tensor * m = codec_op_layer_norm_ct(ctx_eval, x_ct, 1e-5f, paln_w, paln_b);
        m = mul_mat_checked("dtr.fc1", fc1_w, m);
        if (m == nullptr) {
            return false;
        }
        m = lm_ggml_gelu_erf(ctx_eval, m);
        m = mul_mat_checked("dtr.fc2", fc2_w, m);
        if (m == nullptr) {
            return false;
        }
        x_ct = lm_ggml_add(ctx_eval, x_ct, codec_op_channel_scale(ctx_eval, m, mlp_scale));
    }

    x = lm_ggml_cont(ctx_eval, lm_ggml_transpose(ctx_eval, x_ct)); // [t, c]

    // Decoder conv stack — GGUF tensor names match write_decode_decoder_weights mapping.
    lm_ggml_tensor * t_l0_w = W("dec.l0.conv.w");
    lm_ggml_tensor * t_l0_b = W("dec.l0.conv.b");
    lm_ggml_tensor * t_l2_w = W("dec.l2.conv.w");
    lm_ggml_tensor * t_l2_b = W("dec.l2.conv.b");
    lm_ggml_tensor * t_r0_c1_w = W("dec.l3.block.1.conv.w");
    lm_ggml_tensor * t_r0_c1_b = W("dec.l3.block.1.conv.b");
    lm_ggml_tensor * t_r0_c2_w = W("dec.l3.block.3.conv.w");
    lm_ggml_tensor * t_r0_c2_b = W("dec.l3.block.3.conv.b");
    lm_ggml_tensor * t_l5_w = W("dec.l5.conv.w");
    lm_ggml_tensor * t_l5_b = W("dec.l5.conv.b");
    lm_ggml_tensor * t_r1_c1_w = W("dec.l6.block.1.conv.w");
    lm_ggml_tensor * t_r1_c1_b = W("dec.l6.block.1.conv.b");
    lm_ggml_tensor * t_r1_c2_w = W("dec.l6.block.3.conv.w");
    lm_ggml_tensor * t_r1_c2_b = W("dec.l6.block.3.conv.b");
    lm_ggml_tensor * t_l8_w = W("dec.l8.conv.w");
    lm_ggml_tensor * t_l8_b = W("dec.l8.conv.b");
    lm_ggml_tensor * t_r2_c1_w = W("dec.l9.block.1.conv.w");
    lm_ggml_tensor * t_r2_c1_b = W("dec.l9.block.1.conv.b");
    lm_ggml_tensor * t_r2_c2_w = W("dec.l9.block.3.conv.w");
    lm_ggml_tensor * t_r2_c2_b = W("dec.l9.block.3.conv.b");
    lm_ggml_tensor * t_l11_w = W("dec.l11.conv.w");
    lm_ggml_tensor * t_l11_b = W("dec.l11.conv.b");
    lm_ggml_tensor * t_r3_c1_w = W("dec.l12.block.1.conv.w");
    lm_ggml_tensor * t_r3_c1_b = W("dec.l12.block.1.conv.b");
    lm_ggml_tensor * t_r3_c2_w = W("dec.l12.block.3.conv.w");
    lm_ggml_tensor * t_r3_c2_b = W("dec.l12.block.3.conv.b");
    lm_ggml_tensor * t_l14_w = W("dec.l14.conv.w");
    lm_ggml_tensor * t_l14_b = W("dec.l14.conv.b");
    if (t_l0_w == nullptr || t_l0_b == nullptr || t_l2_w == nullptr || t_l2_b == nullptr ||
        t_r0_c1_w == nullptr || t_r0_c1_b == nullptr || t_r0_c2_w == nullptr || t_r0_c2_b == nullptr ||
        t_l5_w == nullptr || t_l5_b == nullptr ||
        t_r1_c1_w == nullptr || t_r1_c1_b == nullptr || t_r1_c2_w == nullptr || t_r1_c2_b == nullptr ||
        t_l8_w == nullptr || t_l8_b == nullptr ||
        t_r2_c1_w == nullptr || t_r2_c1_b == nullptr || t_r2_c2_w == nullptr || t_r2_c2_b == nullptr ||
        t_l11_w == nullptr || t_l11_b == nullptr ||
        t_r3_c1_w == nullptr || t_r3_c1_b == nullptr || t_r3_c2_w == nullptr || t_r3_c2_b == nullptr ||
        t_l14_w == nullptr || t_l14_b == nullptr) {
        return false;
    }

    x = codec_conv1d_causal(ctx_eval, x, t_l0_w, t_l0_b, 1, 1);
    if (x == nullptr) {
        return false;
    }
    x = lm_ggml_elu(ctx_eval, x);
    x = codec_convtr1d_causal(ctx_eval, x, t_l2_w, t_l2_b, 8, 1);
    x = codec_mimi_resblock_ggml(ctx_eval, x, t_r0_c1_w, t_r0_c1_b, t_r0_c2_w, t_r0_c2_b);
    if (x == nullptr) {
        return false;
    }
    x = lm_ggml_elu(ctx_eval, x);
    x = codec_convtr1d_causal(ctx_eval, x, t_l5_w, t_l5_b, 6, 1);
    x = codec_mimi_resblock_ggml(ctx_eval, x, t_r1_c1_w, t_r1_c1_b, t_r1_c2_w, t_r1_c2_b);
    if (x == nullptr) {
        return false;
    }
    x = lm_ggml_elu(ctx_eval, x);
    x = codec_convtr1d_causal(ctx_eval, x, t_l8_w, t_l8_b, 5, 1);
    x = codec_mimi_resblock_ggml(ctx_eval, x, t_r2_c1_w, t_r2_c1_b, t_r2_c2_w, t_r2_c2_b);
    if (x == nullptr) {
        return false;
    }
    x = lm_ggml_elu(ctx_eval, x);
    x = codec_convtr1d_causal(ctx_eval, x, t_l11_w, t_l11_b, 4, 1);
    x = codec_mimi_resblock_ggml(ctx_eval, x, t_r3_c1_w, t_r3_c1_b, t_r3_c2_w, t_r3_c2_b);
    if (x == nullptr) {
        return false;
    }
    x = lm_ggml_elu(ctx_eval, x);
    lm_ggml_tensor * t_pcm = codec_conv1d_causal(ctx_eval, x, t_l14_w, t_l14_b, 1, 1);
    if (t_pcm == nullptr) {
        return false;
    }
    lm_ggml_tensor * t_out = lm_ggml_cont(ctx_eval, t_pcm);
    lm_ggml_set_name(t_out, "mimi.decode.out");

    *out = t_out;
    return true;
}






static bool codec_mimi_init_decode_build(
    codec_context * ctx,
    const codec_mimi * mimi,
    int32_t t,
    int32_t q,
    int32_t n_sem,
    mimi_decode_build * build,
    std::string * err) {

    if (ctx == nullptr || ctx->model == nullptr || mimi == nullptr || build == nullptr ||
        t <= 0 || q <= 0 || n_sem <= 0 || n_sem > q) {
        if (err != nullptr) {
            *err = "invalid Mimi decode build arguments";
        }
        return false;
    }

    const codec_mimi & mm = *mimi;
    build->t = t;
    build->q = q;
    build->hop = std::max(1, mm.hop_size);
    build->n_sem = n_sem;
    build->codebook_dim = std::max(1, mm.codebook_dim);
    build->hidden_size = std::max(1, mm.hidden_size);
    build->codebook_size = std::max(2, mm.codebook_size);
    build->transformer_layers = std::max(1, mm.num_hidden_layers);
    build->transformer_heads = std::max(1, mm.num_attention_heads);
    build->transformer_head_dim = std::max(1, mm.head_dim);
    build->transformer_intermediate = std::max(1, mm.intermediate_size);
    build->rope_theta = mm.rope_theta;
    build->rope_scaling_factor = mm.rope_scaling_factor;

    auto require_w = [&](const char * name) -> lm_ggml_tensor * {
        lm_ggml_tensor * t_w = codec_model_get_tensor(ctx->model, name);
        if (t_w == nullptr && err != nullptr) {
            *err = std::string("missing Mimi decoder tensor: ") + name;
        }
        return t_w;
    };

    lm_ggml_tensor * up_w = require_w("up.cv.w");
    lm_ggml_tensor * l0_w = require_w("dec.l0.conv.w");
    lm_ggml_tensor * l0_b = require_w("dec.l0.conv.b");
    lm_ggml_tensor * l2_w = require_w("dec.l2.conv.w");
    lm_ggml_tensor * l2_b = require_w("dec.l2.conv.b");
    lm_ggml_tensor * l5_w = require_w("dec.l5.conv.w");
    lm_ggml_tensor * l5_b = require_w("dec.l5.conv.b");
    lm_ggml_tensor * l8_w = require_w("dec.l8.conv.w");
    lm_ggml_tensor * l8_b = require_w("dec.l8.conv.b");
    lm_ggml_tensor * l11_w = require_w("dec.l11.conv.w");
    lm_ggml_tensor * l11_b = require_w("dec.l11.conv.b");
    lm_ggml_tensor * l14_w = require_w("dec.l14.conv.w");
    lm_ggml_tensor * l14_b = require_w("dec.l14.conv.b");
    if (up_w == nullptr || l0_w == nullptr || l0_b == nullptr || l2_w == nullptr || l2_b == nullptr ||
        l5_w == nullptr || l5_b == nullptr || l8_w == nullptr || l8_b == nullptr || l11_w == nullptr ||
        l11_b == nullptr || l14_w == nullptr || l14_b == nullptr) {
        return false;
    }

    auto infer_conv1d = [&](lm_ggml_tensor * w, lm_ggml_tensor * b, int32_t * kernel, int32_t * in_c, int32_t * out_c) -> bool {
        const int32_t n0 = (int32_t) codec_ne(w, 0);
        const int32_t n1 = (int32_t) codec_ne(w, 1);
        const int32_t n2 = (int32_t) codec_ne(w, 2);
        const int32_t bo = (int32_t) codec_ne(b, 0);
        if (n0 == bo) {
            *out_c = n0;
            *in_c = n1;
            *kernel = n2;
            return true;
        }
        if (n2 == bo) {
            *out_c = n2;
            *in_c = n1;
            *kernel = n0;
            return true;
        }
        return false;
    };

    auto infer_convtr = [&](lm_ggml_tensor * w, lm_ggml_tensor * b, int32_t * kernel, int32_t * in_c, int32_t * out_c) -> bool {
        const int32_t n0 = (int32_t) codec_ne(w, 0);
        const int32_t n1 = (int32_t) codec_ne(w, 1);
        const int32_t n2 = (int32_t) codec_ne(w, 2);
        const int32_t bo = (int32_t) codec_ne(b, 0);
        if (n1 != bo) {
            return false;
        }
        *out_c = bo;
        if (n0 <= 64) {
            *kernel = n0;
            *in_c = n2;
            return true;
        }
        *in_c = n0;
        *kernel = n2;
        return true;
    };

    // up.cv.w is stored as a dense ConvTranspose1d weight (k, in=c, out=c) — the
    // converter expands the original depthwise (k, 1, c) layout into a diagonal.
    const int32_t up0 = (int32_t) codec_ne(up_w, 0);
    const int32_t up1 = (int32_t) codec_ne(up_w, 1);
    const int32_t up2 = (int32_t) codec_ne(up_w, 2);
    if (up1 != build->hidden_size || up2 != build->hidden_size) {
        if (err != nullptr) {
            *err = "unexpected Mimi upsample channel dimension";
        }
        return false;
    }
    build->upsample_kernel = up0;
    build->upsample_stride = 2;
    int32_t tmp_in = 0, tmp_out = 0;
    if (!infer_conv1d(l0_w, l0_b, &build->dec_l0_kernel, &tmp_in, &build->dec_l0_out) ||
        !infer_convtr(l2_w, l2_b, &build->dec_l2_kernel, &tmp_in, &build->dec_l2_out) ||
        !infer_convtr(l5_w, l5_b, &build->dec_l5_kernel, &tmp_in, &build->dec_l5_out) ||
        !infer_convtr(l8_w, l8_b, &build->dec_l8_kernel, &tmp_in, &build->dec_l8_out) ||
        !infer_convtr(l11_w, l11_b, &build->dec_l11_kernel, &tmp_in, &build->dec_l11_out) ||
        !infer_conv1d(l14_w, l14_b, &build->dec_l14_kernel, &tmp_in, &tmp_out)) {
        if (err != nullptr) {
            *err = "failed to infer Mimi decoder layer shapes";
        }
        return false;
    }

    build->dec_l14_out = (int32_t) codec_ne(l14_b, 0);
    if (build->dec_l14_out != 1) {
        if (err != nullptr) {
            *err = "Mimi decoder output channels must be 1";
        }
        return false;
    }

    build->model = ctx->model;
    return true;
}


enum codec_status codec_mimi_decode_with(
    struct codec_context * ctx,
    struct codec_mimi * mimi,
    const struct codec_token_buffer * tokens,
    struct codec_pcm_buffer * out_pcm,
    struct codec_decode_params params) {

    if (mimi == nullptr) {
        codec_context_set_error(ctx, "invalid Mimi metadata");
        return CODEC_STATUS_INVALID_ARG;
    }
    codec_mimi & mm = *mimi;
    if (!mm.has_decoder) {
        codec_context_set_error(ctx, "model metadata indicates no decoder");
        return CODEC_STATUS_INVALID_STATE;
    }

    const int32_t model_n_q = std::max(1, mm.n_q);
    const int32_t use_n_q = params.n_q == 0 ? model_n_q : params.n_q;
    if (params.n_q < 0 || use_n_q < 1 || use_n_q > model_n_q) {
        codec_context_set_error(ctx, "Mimi decode n_q must be 0 or in [1, model_n_q]");
        return CODEC_STATUS_INVALID_ARG;
    }

    if (tokens == nullptr || tokens->data == nullptr || tokens->n_frames <= 0 || tokens->n_q < use_n_q) {
        codec_context_set_error(ctx, "invalid Mimi token buffer");
        return CODEC_STATUS_INVALID_ARG;
    }

    const int32_t t = tokens->n_frames;
    const int32_t q = use_n_q;

    codec_graph_eval_guard eval_guard(ctx);
    const int32_t n_sem = std::max(1, std::min(mm.num_semantic_quantizers, q));
    mimi_decode_build build = {};
    std::string err;
    if (!codec_mimi_init_decode_build(ctx, &mm, t, q, n_sem, &build, &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }
    codec_graph_cache_entry * entry = nullptr;
    if (!codec_graph_cache_get_or_build(
            ctx,
            { CODEC_GRAPH_MIMI_DECODE, /*n_frames=*/t, /*n_q=*/q, /*hop=*/build.hop, /*n_in=*/0, /*latent_dim=*/0 },
            codec_mimi_build_decode,
            &build,
            sizeof(build),
            &entry,
            &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    lm_ggml_tensor * t_tok = codec_graph_get_tensor(ctx, entry, "mimi.decode.tok");
    lm_ggml_tensor * t_out = codec_graph_get_tensor(ctx, entry, "mimi.decode.out");
    if (t_tok == nullptr || t_out == nullptr) {
        codec_context_set_error(ctx, "cached Mimi decode graph is invalid");
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

    const int32_t n_samples = (int32_t) t_out->ne[0];
    float * pcm = static_cast<float *>(std::malloc((size_t) n_samples * sizeof(float)));
    if (pcm == nullptr) {
        codec_context_set_error(ctx, "failed to allocate pcm output");
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    if (!codec_runtime_read_tensor(t_out, pcm, (size_t) n_samples * sizeof(float), &err)) {
        std::free(pcm);
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    codec_pcm_buffer_reset(out_pcm);
    out_pcm->data = pcm;
    out_pcm->n_samples = n_samples;
    out_pcm->sample_rate = mm.sample_rate;
    out_pcm->n_channels = 1;

    return CODEC_STATUS_SUCCESS;
}

enum codec_status codec_mimi_decode(
    struct codec_context * ctx,
    const struct codec_token_buffer * tokens,
    struct codec_pcm_buffer * out_pcm,
    struct codec_decode_params params) {

    if (ctx == nullptr || ctx->model == nullptr) {
        return CODEC_STATUS_INVALID_ARG;
    }
    codec_mimi & mm = *static_cast<codec_mimi *>(ctx->model->impl);
    return codec_mimi_decode_with(ctx, &mm, tokens, out_pcm, params);
}

enum codec_status codec_mimi_encode_with(
    struct codec_context * ctx,
    struct codec_mimi * mimi,
    const std::vector<float> & pcm,
    struct codec_token_buffer * out_tokens,
    struct codec_encode_params params) {

    if (mimi == nullptr) {
        codec_context_set_error(ctx, "invalid Mimi metadata");
        return CODEC_STATUS_INVALID_ARG;
    }
    codec_mimi & mm = *mimi;
    if (!mm.has_encoder) {
        codec_context_set_error(ctx, "model metadata indicates no encoder");
        return CODEC_STATUS_INVALID_STATE;
    }

    if (pcm.empty()) {
        codec_context_set_error(ctx, "empty pcm");
        return CODEC_STATUS_INVALID_ARG;
    }

    const int32_t model_n_q = std::max(1, mm.n_q);
    const int32_t use_n_q = params.n_q == 0 ? model_n_q : params.n_q;
    if (params.n_q < 0 || use_n_q < 1 || use_n_q > model_n_q) {
        codec_context_set_error(ctx, "Mimi encode n_q must be 0 or in [1, model_n_q]");
        return CODEC_STATUS_INVALID_ARG;
    }

    std::string err;
    const int32_t n_in = (int32_t) pcm.size();
    const float * pcm_data = pcm.data();
    size_t pcm_bytes = pcm.size() * sizeof(float);
    mimi_encode_build build = {};
    if (!codec_mimi_init_encode_build(ctx, &mm, n_in, &build, &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }
    build.model = ctx->model;

    // Eval arena only needs to hold ggml tensor metadata (no_alloc=true), which scales
    // with graph size, not tensor element counts. Avoid huge allocations for long inputs.
    codec_graph_eval_guard eval_guard(ctx);
    codec_graph_cache_entry * entry = nullptr;
    if (!codec_graph_cache_get_or_build(
            ctx,
            { CODEC_GRAPH_MIMI_ENCODE, /*n_frames=*/0, /*n_q=*/build.n_q, /*hop=*/0, /*n_in=*/n_in, /*latent_dim=*/build.transformer.c },
            codec_mimi_build_encode,
            &build,
            sizeof(build),
            &entry,
            &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    lm_ggml_tensor * t_pcm = codec_graph_get_tensor(ctx, entry, CODEC_MIMI_ENCODE_PCM_TENSOR);
    lm_ggml_tensor * t_indices = codec_graph_get_tensor(ctx, entry, CODEC_MIMI_ENCODE_INDICES_TENSOR);
    if (t_pcm == nullptr || t_indices == nullptr) {
        codec_context_set_error(ctx, "cached Mimi encode graph is invalid");
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    if (!codec_graph_prepare_io(ctx, entry, &err) ||
        !codec_runtime_write_tensor(t_pcm, pcm_data, pcm_bytes, &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    const int32_t n_threads = ctx->model->n_threads > 0 ? ctx->model->n_threads : 1;
    if (!codec_graph_compute(ctx, entry, n_threads, &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    const int32_t t = (int32_t) t_indices->ne[0];
    const int32_t n_q_graph = (int32_t) t_indices->ne[1];
    if (t <= 0 || n_q_graph < use_n_q) {
        codec_context_set_error(ctx, "Mimi encode output shape mismatch");
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    std::vector<int32_t> all_codes_tq;
    if (!codec_runtime_read_tensor_i32_2d_tq(t_indices, &all_codes_tq, &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    int32_t * data = static_cast<int32_t *>(std::malloc((size_t) t * (size_t) use_n_q * sizeof(int32_t)));
    if (data == nullptr) {
        codec_context_set_error(ctx, "failed to allocate token output");
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    for (int32_t ti = 0; ti < t; ++ti) {
        for (int32_t qi = 0; qi < use_n_q; ++qi) {
            data[(size_t) ti * (size_t) use_n_q + (size_t) qi] =
                all_codes_tq[(size_t) ti * (size_t) n_q_graph + (size_t) qi];
        }
    }

    codec_token_buffer_reset(out_tokens);
    out_tokens->data = data;
    out_tokens->n_tokens = t * use_n_q;
    out_tokens->n_frames = t;
    out_tokens->n_q = use_n_q;
    out_tokens->codebook_size = std::max(2, mm.codebook_size);
    out_tokens->sample_rate = mm.sample_rate;
    out_tokens->hop_size = std::max(1, mm.hop_size);

    return CODEC_STATUS_SUCCESS;
}

enum codec_status codec_mimi_encode(
    struct codec_context * ctx,
    const std::vector<float> & pcm,
    struct codec_token_buffer * out_tokens,
    struct codec_encode_params params) {

    if (ctx == nullptr || ctx->model == nullptr) {
        return CODEC_STATUS_INVALID_ARG;
    }
    codec_mimi & mm = *static_cast<codec_mimi *>(ctx->model->impl);
    return codec_mimi_encode_with(ctx, &mm, pcm, out_tokens, params);
}

static void * codec_mimi_create_impl() {
    return new (std::nothrow) codec_mimi();
}

static void codec_mimi_destroy_impl(void * ptr) {
    delete static_cast<codec_mimi *>(ptr);
}

static enum codec_status codec_mimi_encode_wrap(
    struct codec_context * ctx,
    const std::vector<float> & pcm,
    struct codec_token_buffer * out_tokens,
    struct codec_latent_buffer * /*out_latent*/,
    struct codec_encode_params params) {
    return codec_mimi_encode(ctx, pcm, out_tokens, params);
}

static enum codec_status codec_mimi_decode_wrap(
    struct codec_context * ctx,
    const struct codec_token_buffer * tokens,
    struct codec_pcm_buffer * out_pcm,
    struct codec_decode_params params) {
    return codec_mimi_decode(ctx, tokens, out_pcm, params);
}

const struct codec_model_vtable * codec_mimi_vtable() {
    static const codec_model_vtable vtable = {
        CODEC_ARCH_MIMI,
        "Mimi",
        codec_mimi_create_impl,
        codec_mimi_destroy_impl,
        codec_mimi_init,
        codec_graph_size_exact,
        codec_mimi_encode_wrap,
        codec_mimi_decode_wrap,
        nullptr,
    };
    return &vtable;
}
