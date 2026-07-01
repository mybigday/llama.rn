#include "xcodec2.h"

#include "../ops/conv1d.h"
#include "../ops/lm_ggml_ops.h"
#include "../ops/lm_attn.h"
#include "../runtime/audio_dsp.h"
#include "../runtime/graph.h"
#include "../runtime/lm_gguf_kv.h"
#include "../runtime/tensor_utils.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <new>
#include <string>
#include <vector>

static const char * codec_x2_name_tok()         { return "xcodec2.decode.tok"; }
static const char * codec_x2_name_head_out()    { return "xcodec2.decode.head.out"; }
static const char * codec_x2_name_codebook()    { return "xcodec2.decode.codebook"; }
static const char * codec_x2_name_quant_w()     { return "xcodec2.decode.quant.project_out.w"; }
static const char * codec_x2_name_quant_b()     { return "xcodec2.decode.quant.project_out.b"; }
static const char * codec_x2_name_fc_post_w()   { return "xcodec2.decode.fc_post_a.w"; }
static const char * codec_x2_name_fc_post_b()   { return "xcodec2.decode.fc_post_a.b"; }
static const char * codec_x2_name_embed_w()     { return "xcodec2.decode.embed.w"; }
static const char * codec_x2_name_embed_b()     { return "xcodec2.decode.embed.b"; }
static const char * codec_x2_name_final_ln_w()  { return "xcodec2.decode.final_ln.w"; }
static const char * codec_x2_name_final_ln_b()  { return "xcodec2.decode.final_ln.b"; }
static const char * codec_x2_name_head_w()      { return "xcodec2.decode.head.out.w"; }
static const char * codec_x2_name_head_b()      { return "xcodec2.decode.head.out.b"; }
static const char * codec_x2_name_istft_window(){ return "xcodec2.decode.istft.window"; }

static std::string codec_x2_name_prior(int32_t li, const char * suffix) {
    return "xcodec2.decode.prior." + std::to_string(li) + "." + suffix;
}
static std::string codec_x2_name_post(int32_t li, const char * suffix) {
    return "xcodec2.decode.post." + std::to_string(li) + "." + suffix;
}
static std::string codec_x2_name_xfm(int32_t li, const char * suffix) {
    return "xcodec2.decode.transformer." + std::to_string(li) + "." + suffix;
}

struct xcodec2_decode_build {
    int32_t t            = 0;
    int32_t q            = 0;
    int32_t codebook_dim = 0;
    int32_t codebook_size = 0;
    int32_t vq_dim       = 0;
    int32_t hidden_dim   = 0;
    int32_t num_layers   = 0;
    int32_t num_heads    = 0;
    int32_t head_dim     = 0;
    int32_t head_out_dim = 0;
    float   rope_theta   = 10000.0f;
    const codec_model * model = nullptr;
};

static bool codec_x2_build_decode(lm_ggml_context * ctx_eval, void * user_data, lm_ggml_tensor ** out) {
    xcodec2_decode_build * p = static_cast<xcodec2_decode_build *>(user_data);
    if (ctx_eval == nullptr || p == nullptr || out == nullptr || p->model == nullptr) {
        return false;
    }
    if (p->t <= 0 || p->q <= 0 || p->codebook_dim <= 0 || p->codebook_size <= 0 ||
        p->vq_dim <= 0 || p->hidden_dim <= 0 || p->num_layers <= 0 ||
        p->num_heads <= 0 || p->head_dim <= 0 || p->head_out_dim <= 0) {
        return false;
    }

    auto W = [&](const std::string & name) -> lm_ggml_tensor * {
        return codec_graph_weight(ctx_eval, p->model, name);
    };

    // 1) tokens [t, q] — q is currently always 1.
    lm_ggml_tensor * t_tok = lm_ggml_new_tensor_2d(ctx_eval, LM_GGML_TYPE_I32, p->t, p->q);
    lm_ggml_set_name(t_tok, codec_x2_name_tok());

    lm_ggml_tensor * t_codebook = W(codec_x2_name_codebook());
    if (t_codebook == nullptr) return false;

    // codebook layout: ne=[codebook_dim=8, codebook_size=65536]; gather rows.
    lm_ggml_tensor * t_idx = lm_ggml_view_1d(ctx_eval, t_tok, p->t, 0);
    lm_ggml_tensor * t_q = lm_ggml_get_rows(ctx_eval, t_codebook, t_idx);              // [codebook_dim, t]

    // 2) FSQ.project_out: codebook_dim → vq_dim
    lm_ggml_tensor * x_ct = codec_op_linear(ctx_eval, t_q,
                                         W(codec_x2_name_quant_w()),
                                         W(codec_x2_name_quant_b()));            // [vq_dim, t]
    if (x_ct == nullptr) return false;

    // 3) fc_post_a: vq_dim → hidden_dim
    x_ct = codec_op_linear(ctx_eval, x_ct,
                           W(codec_x2_name_fc_post_w()),
                           W(codec_x2_name_fc_post_b()));                         // [hidden, t]
    if (x_ct == nullptr) return false;

    // ---- Vocos backbone ----
    // embed Conv1d(k=7, p=3) operates on [t, c] layout in this codebase.
    lm_ggml_tensor * x_tc = lm_ggml_cont(ctx_eval, lm_ggml_transpose(ctx_eval, x_ct));     // [t, hidden]
    x_tc = codec_conv1d(ctx_eval, x_tc,
                        W(codec_x2_name_embed_w()),
                        W(codec_x2_name_embed_b()),
                        /*stride=*/1, /*dilation=*/1, /*padding=*/3);
    if (x_tc == nullptr) return false;

    // prior_net: 2 ResnetBlocks (GN-SiLU-Conv-GN-SiLU-Conv + skip)
    for (int32_t li = 0; li < 2; ++li) {
        x_tc = codec_op_vocos_resnet_block_tc(
            ctx_eval, x_tc,
            W(codec_x2_name_prior(li, "norm1.w")), W(codec_x2_name_prior(li, "norm1.b")),
            W(codec_x2_name_prior(li, "conv1.w")), W(codec_x2_name_prior(li, "conv1.b")),
            W(codec_x2_name_prior(li, "norm2.w")), W(codec_x2_name_prior(li, "norm2.b")),
            W(codec_x2_name_prior(li, "conv2.w")), W(codec_x2_name_prior(li, "conv2.b")));
        if (x_tc == nullptr) return false;
    }

    x_ct = lm_ggml_cont(ctx_eval, lm_ggml_transpose(ctx_eval, x_tc));                   // [hidden, t]

    // 12 RoFormer transformer blocks (RMSNorm, RoPE q/k, MLP fc1→SiLU→fc2 no bias)
    for (int32_t li = 0; li < p->num_layers; ++li) {
        x_ct = codec_op_roformer_block_ct(
            ctx_eval, x_ct,
            W(codec_x2_name_xfm(li, "att_norm.w")),
            W(codec_x2_name_xfm(li, "ffn_norm.w")),
            W(codec_x2_name_xfm(li, "att.c_attn.w")),
            W(codec_x2_name_xfm(li, "att.c_proj.w")),
            W(codec_x2_name_xfm(li, "mlp.fc1.w")),
            W(codec_x2_name_xfm(li, "mlp.fc2.w")),
            p->head_dim, p->num_heads, p->rope_theta);
        if (x_ct == nullptr) return false;
    }

    x_tc = lm_ggml_cont(ctx_eval, lm_ggml_transpose(ctx_eval, x_ct));                   // [t, hidden]

    // post_net: 2 ResnetBlocks
    for (int32_t li = 0; li < 2; ++li) {
        x_tc = codec_op_vocos_resnet_block_tc(
            ctx_eval, x_tc,
            W(codec_x2_name_post(li, "norm1.w")), W(codec_x2_name_post(li, "norm1.b")),
            W(codec_x2_name_post(li, "conv1.w")), W(codec_x2_name_post(li, "conv1.b")),
            W(codec_x2_name_post(li, "norm2.w")), W(codec_x2_name_post(li, "norm2.b")),
            W(codec_x2_name_post(li, "conv2.w")), W(codec_x2_name_post(li, "conv2.b")));
        if (x_tc == nullptr) return false;
    }

    x_ct = lm_ggml_cont(ctx_eval, lm_ggml_transpose(ctx_eval, x_tc));                   // [hidden, t]

    // final_layer_norm (LayerNorm along channels)
    x_ct = codec_op_layer_norm_ct(ctx_eval, x_ct, 1e-6f,
                                  W(codec_x2_name_final_ln_w()),
                                  W(codec_x2_name_final_ln_b()));
    if (x_ct == nullptr) return false;

    // 4) ISTFT head: Linear(hidden, n_fft+2) → produces (mag,phase) per frame.
    //    Actual iSTFT (mag/phase → complex → OLA) runs on the host so we can
    //    reuse the shared HiFi-GAN-style helper with the periodic Hann window.
    lm_ggml_tensor * t_out = codec_op_linear(ctx_eval, x_ct,
                                          W(codec_x2_name_head_w()),
                                          W(codec_x2_name_head_b()));             // [out, t]
    if (t_out == nullptr) return false;
    lm_ggml_set_name(t_out, codec_x2_name_head_out());
    *out = t_out;
    return true;
}

static enum codec_status codec_x2_decode_graph(
    struct codec_context * ctx,
    const struct codec_token_buffer * tokens,
    int32_t use_n_q,
    struct codec_pcm_buffer * out_pcm) {

    codec_xcodec2 & x2 = *static_cast<codec_xcodec2 *>(ctx->model->impl);
    if (tokens == nullptr || tokens->data == nullptr || tokens->n_frames <= 0 || tokens->n_q < use_n_q) {
        codec_context_set_error(ctx, "invalid xcodec2 token buffer");
        return CODEC_STATUS_INVALID_ARG;
    }

    const int32_t t   = tokens->n_frames;
    const int32_t q   = use_n_q;
    const int32_t hop = std::max(1, x2.hop_size);
    codec_graph_eval_guard eval_guard(ctx);
    std::string err;

    xcodec2_decode_build build = {};
    build.t              = t;
    build.q              = q;
    build.model          = ctx->model;
    build.codebook_dim   = x2.codebook_dim;
    build.codebook_size  = x2.codebook_size;
    build.vq_dim         = x2.vq_dim;
    build.hidden_dim     = x2.hidden_dim;
    build.num_layers     = x2.num_layers;
    build.num_heads      = x2.num_heads;
    build.head_dim       = x2.head_dim;
    build.head_out_dim   = x2.n_fft + 2;
    build.rope_theta     = x2.rope_theta;

    if (x2.hidden_dim != x2.num_heads * x2.head_dim) {
        codec_context_set_error(ctx, "xcodec2 head_dim * num_heads != hidden_dim");
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    codec_graph_cache_entry * entry = nullptr;
    if (!codec_graph_cache_get_or_build(
            ctx,
            { CODEC_GRAPH_XCODEC2_DECODE, /*n_frames=*/t, /*n_q=*/q, /*hop=*/hop, /*n_in=*/0, /*latent_dim=*/0 },
            codec_x2_build_decode,
            &build,
            sizeof(build),
            &entry,
            &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    lm_ggml_tensor * t_tok = codec_graph_get_tensor(ctx, entry, codec_x2_name_tok());
    lm_ggml_tensor * t_out = codec_graph_get_tensor(ctx, entry, codec_x2_name_head_out());
    if (t_tok == nullptr || t_out == nullptr) {
        codec_context_set_error(ctx, "cached xcodec2 decode graph is invalid");
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

    // ISTFTHead (Vocos style with `padding="same"`): trim = (n_fft - hop)/2,
    // symmetric Hann (the registered buffer from `torch.hann_window` matches
    // codec_runtime_istft_from_head's default symmetric window).
    std::vector<float> window;
    lm_ggml_tensor * w_tensor = codec_model_get_tensor(ctx->model, codec_x2_name_istft_window());
    if (w_tensor != nullptr) {
        codec_tensor_as_vec_f32(w_tensor, &window);
    }

    std::vector<float> pcm_v;
    if (!codec_runtime_istft_from_head(
            head, build.head_out_dim, t, hop,
            w_tensor != nullptr ? &window : nullptr,
            /*skip_dc_nyquist=*/false,
            /*trim_pad_override=*/-1,
            &pcm_v, &err)) {
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
    out_pcm->data        = pcm;
    out_pcm->n_samples   = (int32_t) pcm_v.size();
    out_pcm->sample_rate = x2.sample_rate;
    out_pcm->n_channels  = 1;
    return CODEC_STATUS_SUCCESS;
}

enum codec_status codec_xcodec2_decode(
    struct codec_context * ctx,
    const struct codec_token_buffer * tokens,
    struct codec_pcm_buffer * out_pcm,
    struct codec_decode_params params) {

    codec_xcodec2 & x2 = *static_cast<codec_xcodec2 *>(ctx->model->impl);
    if (!x2.has_decoder) {
        codec_context_set_error(ctx, "model metadata indicates no decoder");
        return CODEC_STATUS_INVALID_STATE;
    }

    const int32_t model_n_q = std::max(1, x2.n_q);
    const int32_t use_n_q = params.n_q == 0 ? model_n_q : params.n_q;
    if (params.n_q < 0 || use_n_q < 1 || use_n_q > model_n_q) {
        codec_context_set_error(ctx, "xcodec2 decode n_q must be 0 or in [1, model_n_q]");
        return CODEC_STATUS_INVALID_ARG;
    }

    return codec_x2_decode_graph(ctx, tokens, use_n_q, out_pcm);
}

// =====================================================================
// Encoder
// =====================================================================
//
// xcodec2 encode (matches HKUSTAudio/xcodec2 `encode_code`):
//   PCM -> [BigCodec acoustic encoder]              -> a_ct  (1024, T_ac)
//   mel-fbank features -> [w2v-bert l0..l15 + SemanticEncoder] -> s_ct  (1024, T_sem)
//   concat([s, a]) along channel -> 2048-d           -> fc_prior(2048->2048)
//   FSQ.project_in (2048 -> 8) -> bound -> round   -> indices [T]
//
// Mel features (160-d after stride-2 stacking) are computed CPU-side via
// `codec_runtime_w2v_bert_features`, mirroring the way the decoder runs the
// final iSTFT on the host.
// =====================================================================

struct xcodec2_encode_build {
    int32_t n_pcm        = 0;
    int32_t n_sem_frames = 0;
    int32_t n_acoustic   = 0;     // expected output frames of CodecEnc
    int32_t n_codes      = 0;     // min(n_acoustic, n_sem_frames)
    int32_t hidden       = 1024;
    int32_t vq_dim       = 2048;
    int32_t cb_dim       = 8;     // FSQ levels count
    int32_t w2v_layers   = 16;
    int32_t w2v_hidden   = 1024;
    int32_t w2v_heads    = 16;
    int32_t w2v_head_dim = 64;
    int32_t w2v_input_dim = 160;
    int32_t w2v_dw_kernel = 31;
    int32_t w2v_intermediate = 4096;
    int32_t w2v_left_max = 64;
    int32_t w2v_right_max = 8;
    float   w2v_eps      = 1e-5f;
    int32_t enc_ngf      = 48;
    const codec_xcodec2 * cfg = nullptr;
    const codec_model * model = nullptr;
};

static const char * codec_x2_name_pcm()      { return "xcodec2.encode.pcm"; }
static const char * codec_x2_name_mel_in()   { return "xcodec2.encode.mel_features"; }
static const char * codec_x2_name_codes()    { return "xcodec2.encode.codes"; }
static const char * codec_x2_name_alias()    { return "xcodec2.enc.alias.filter"; }

// ----- BigCodec acoustic encoder helpers -------------------------------

static lm_ggml_tensor * codec_x2_residual_unit(
    lm_ggml_context * ctx,
    lm_ggml_tensor * x_tc,
    lm_ggml_tensor * a1_alpha, lm_ggml_tensor * a1_inv_beta,
    lm_ggml_tensor * c1_w,     lm_ggml_tensor * c1_b,
    lm_ggml_tensor * a2_alpha, lm_ggml_tensor * a2_inv_beta,
    lm_ggml_tensor * c2_w,     lm_ggml_tensor * c2_b,
    lm_ggml_tensor * alias_kernel,
    int32_t dilation) {

    if (ctx == nullptr || x_tc == nullptr) return nullptr;
    lm_ggml_tensor * h = codec_op_alias_free_snake_beta_tc(ctx, x_tc, a1_alpha, a1_inv_beta, alias_kernel);
    if (h == nullptr) return nullptr;
    // First conv: kernel 7, dilation `d`, padding ((7-1)*d)/2 = 3*d.
    h = codec_conv1d(ctx, h, c1_w, c1_b, /*stride=*/1, /*dilation=*/dilation, /*padding=*/3 * dilation);
    if (h == nullptr) return nullptr;
    h = codec_op_alias_free_snake_beta_tc(ctx, h, a2_alpha, a2_inv_beta, alias_kernel);
    if (h == nullptr) return nullptr;
    // Second conv: kernel 1.
    h = codec_conv1d(ctx, h, c2_w, c2_b, /*stride=*/1, /*dilation=*/1, /*padding=*/0);
    if (h == nullptr) return nullptr;
    return lm_ggml_add(ctx, x_tc, h);
}

static lm_ggml_tensor * codec_x2_encoder_block(
    lm_ggml_context * ctx,
    lm_ggml_tensor * x_tc,
    int32_t bi,
    int32_t stride,
    const codec_model * model,
    lm_ggml_tensor * alias_kernel) {

    auto W = [&](const std::string & name) -> lm_ggml_tensor * {
        return codec_graph_weight(ctx, model, name);
    };

    const int32_t dilations[3] = { 1, 3, 9 };
    for (int32_t ri = 0; ri < 3; ++ri) {
        const std::string base = "xcodec2.enc.codec.b" + std::to_string(bi) + ".r" + std::to_string(ri);
        x_tc = codec_x2_residual_unit(
            ctx, x_tc,
            W(base + ".act1.alpha"), W(base + ".act1.inv_beta"),
            W(base + ".conv1.w"),    W(base + ".conv1.b"),
            W(base + ".act2.alpha"), W(base + ".act2.inv_beta"),
            W(base + ".conv2.w"),    W(base + ".conv2.b"),
            alias_kernel,
            dilations[ri]);
        if (x_tc == nullptr) return nullptr;
    }
    // Final SnakeBeta + Activation1d, then downsample WNConv1d
    // (kernel = 2*stride, padding = stride/2 + stride%2).
    const std::string a = "xcodec2.enc.codec.b" + std::to_string(bi) + ".act";
    x_tc = codec_op_alias_free_snake_beta_tc(ctx, x_tc, W(a + ".alpha"), W(a + ".inv_beta"), alias_kernel);
    if (x_tc == nullptr) return nullptr;

    const std::string d = "xcodec2.enc.codec.b" + std::to_string(bi) + ".down";
    const int32_t kernel = 2 * stride;
    const int32_t padding = stride / 2 + stride % 2;
    x_tc = codec_conv1d(ctx, x_tc, W(d + ".w"), W(d + ".b"), /*stride=*/stride, /*dilation=*/1, /*padding=*/padding);
    return x_tc;
}

// ----- Wav2Vec2-Bert conformer layer ----------------------------------

static lm_ggml_tensor * codec_x2_w2v_attn(
    lm_ggml_context * ctx,
    lm_ggml_tensor * x_tc,
    lm_ggml_tensor * q_w, lm_ggml_tensor * q_b,
    lm_ggml_tensor * k_w, lm_ggml_tensor * k_b,
    lm_ggml_tensor * v_w, lm_ggml_tensor * v_b,
    lm_ggml_tensor * o_w, lm_ggml_tensor * o_b,
    lm_ggml_tensor * dist_emb_dn,
    lm_ggml_tensor * bucket_idx_1d,
    int32_t head_dim,
    int32_t n_heads) {

    if (ctx == nullptr || x_tc == nullptr) return nullptr;
    const int64_t t = x_tc->ne[0];

    lm_ggml_tensor * Q = codec_op_linear_tc(ctx, x_tc, q_w, q_b);   // [t, c]
    lm_ggml_tensor * K = codec_op_linear_tc(ctx, x_tc, k_w, k_b);
    lm_ggml_tensor * V = codec_op_linear_tc(ctx, x_tc, v_w, v_b);
    if (Q == nullptr || K == nullptr || V == nullptr) return nullptr;

    auto to_dth = [&](lm_ggml_tensor * x_tc_in) {
        // x_tc has ne=(t, c=h*d).
        lm_ggml_tensor * x_ct = lm_ggml_cont(ctx, lm_ggml_transpose(ctx, x_tc_in));        // [c, t]
        lm_ggml_tensor * x_dht = lm_ggml_reshape_3d(ctx, x_ct, head_dim, n_heads, t);   // [d, h, t]
        return lm_ggml_cont(ctx, lm_ggml_permute(ctx, x_dht, 0, 2, 1, 3));              // [d, t, h]
    };

    lm_ggml_tensor * q_dth = to_dth(Q);
    lm_ggml_tensor * k_dth = to_dth(K);
    lm_ggml_tensor * v_dth = to_dth(V);
    if (q_dth == nullptr || k_dth == nullptr || v_dth == nullptr) return nullptr;

    codec_lm_attn_params attn_p = {};
    attn_p.scale = 1.0f / std::sqrt((float) head_dim);
    attn_p.causal = false;

    lm_ggml_tensor * ctx_dth = codec_op_lm_attn_rel_key_dth(ctx, q_dth, k_dth, v_dth,
                                                         dist_emb_dn, bucket_idx_1d, &attn_p);
    if (ctx_dth == nullptr) return nullptr;

    // Permute back to (t, c) for linear_out.
    lm_ggml_tensor * dht = lm_ggml_cont(ctx, lm_ggml_permute(ctx, ctx_dth, 0, 2, 1, 3));  // [d, h, t]
    lm_ggml_tensor * c_t = lm_ggml_reshape_2d(ctx, dht, head_dim * n_heads, t);        // [c, t]
    lm_ggml_tensor * tc = lm_ggml_cont(ctx, lm_ggml_transpose(ctx, c_t));                  // [t, c]
    return codec_op_linear_tc(ctx, tc, o_w, o_b);
}

static lm_ggml_tensor * codec_x2_w2v_conv_module(
    lm_ggml_context * ctx,
    lm_ggml_tensor * x_tc,
    const codec_model * model,
    int32_t li,
    int32_t hidden,
    int32_t dw_kernel) {

    auto W = [&](const std::string & name) -> lm_ggml_tensor * {
        return codec_graph_weight(ctx, model, name);
    };
    const std::string base = "xcodec2.w2v.l" + std::to_string(li) + ".conv";

    // pre-LN
    lm_ggml_tensor * h = codec_op_layer_norm_tc(ctx, x_tc, 1e-5f, W(base + ".ln.w"), W(base + ".ln.b"));
    if (h == nullptr) return nullptr;

    // Pointwise conv1: 1024 → 2048 (no bias).  Run as conv1d k=1.
    h = codec_conv1d(ctx, h, W(base + ".pw1.w"), nullptr, /*stride=*/1, /*dilation=*/1, /*padding=*/0);
    if (h == nullptr) return nullptr;

    // GLU along channel axis: split (2C) into (a, b), output a * sigmoid(b).
    // h shape [t, 2C].  GLU: out[t, c] = a[t, c] * sigmoid(b[t, c]).
    const int64_t t = h->ne[0];
    const int64_t two_c = h->ne[1];
    if (two_c != 2 * hidden) return nullptr;
    lm_ggml_tensor * a = lm_ggml_cont(ctx, lm_ggml_view_2d(ctx, h, t, hidden, h->nb[1], 0));
    lm_ggml_tensor * b = lm_ggml_cont(ctx, lm_ggml_view_2d(ctx, h, t, hidden, h->nb[1], (size_t) hidden * h->nb[1]));
    lm_ggml_tensor * sig_b = codec_op_unary(ctx, b, CODEC_UNARY_SIGMOID);
    h = lm_ggml_mul(ctx, a, sig_b);   // [t, hidden]

    // Causal pad: prepend (k-1) zeros along time, then depthwise conv k=31 stride 1 no pad.
    h = codec_op_pad_1d(ctx, h, dw_kernel - 1, 0);
    h = codec_conv1d_depthwise(ctx, h, W(base + ".dw.w"), nullptr, /*stride=*/1, /*dilation=*/1, /*padding=*/0);
    if (h == nullptr) return nullptr;

    // depthwise_layer_norm operates on [t, c] (LayerNorm over channels) and
    // then SiLU.
    h = codec_op_layer_norm_tc(ctx, h, 1e-5f, W(base + ".dw_ln.w"), W(base + ".dw_ln.b"));
    h = lm_ggml_silu(ctx, h);

    // Pointwise conv2: 1024 → 1024 (no bias) — same pattern.
    h = codec_conv1d(ctx, h, W(base + ".pw2.w"), nullptr, 1, 1, 0);
    return h;
}

static lm_ggml_tensor * codec_x2_w2v_layer(
    lm_ggml_context * ctx,
    lm_ggml_tensor * x_tc,
    const codec_model * model,
    int32_t li,
    const xcodec2_encode_build * p,
    lm_ggml_tensor * dist_emb_dn,
    lm_ggml_tensor * bucket_idx_1d) {

    auto W = [&](const std::string & name) -> lm_ggml_tensor * {
        return codec_graph_weight(ctx, model, name);
    };
    const std::string base = "xcodec2.w2v.l" + std::to_string(li);

    auto ln = [&](lm_ggml_tensor * x, const std::string & name) {
        return codec_op_layer_norm_tc(ctx, x, p->w2v_eps, W(name + ".w"), W(name + ".b"));
    };

    // ffn1: pre-LN -> fc1 -> SiLU -> fc2 -> *0.5 + residual
    lm_ggml_tensor * res = x_tc;
    lm_ggml_tensor * h = ln(x_tc, base + ".ffn1_ln");
    h = codec_op_linear_tc(ctx, h, W(base + ".ffn1.fc1.w"), W(base + ".ffn1.fc1.b"));
    h = lm_ggml_silu(ctx, h);
    h = codec_op_linear_tc(ctx, h, W(base + ".ffn1.fc2.w"), W(base + ".ffn1.fc2.b"));
    if (h == nullptr) return nullptr;
    h = lm_ggml_scale(ctx, h, 0.5f);
    x_tc = lm_ggml_add(ctx, res, h);

    // self_attn: pre-LN, rel-key attention, residual.
    res = x_tc;
    h = ln(x_tc, base + ".attn_ln");
    h = codec_x2_w2v_attn(
        ctx, h,
        W(base + ".attn.q.w"), W(base + ".attn.q.b"),
        W(base + ".attn.k.w"), W(base + ".attn.k.b"),
        W(base + ".attn.v.w"), W(base + ".attn.v.b"),
        W(base + ".attn.o.w"), W(base + ".attn.o.b"),
        dist_emb_dn, bucket_idx_1d,
        p->w2v_head_dim, p->w2v_heads);
    if (h == nullptr) return nullptr;
    x_tc = lm_ggml_add(ctx, res, h);

    // conv module: residual + module(x).
    res = x_tc;
    h = codec_x2_w2v_conv_module(ctx, x_tc, model, li, p->w2v_hidden, p->w2v_dw_kernel);
    if (h == nullptr) return nullptr;
    x_tc = lm_ggml_add(ctx, res, h);

    // ffn2 (same as ffn1): pre-LN -> fc1 -> SiLU -> fc2 -> *0.5 + residual.
    res = x_tc;
    h = ln(x_tc, base + ".ffn2_ln");
    h = codec_op_linear_tc(ctx, h, W(base + ".ffn2.fc1.w"), W(base + ".ffn2.fc1.b"));
    h = lm_ggml_silu(ctx, h);
    h = codec_op_linear_tc(ctx, h, W(base + ".ffn2.fc2.w"), W(base + ".ffn2.fc2.b"));
    h = lm_ggml_scale(ctx, h, 0.5f);
    x_tc = lm_ggml_add(ctx, res, h);

    // final LN.
    return ln(x_tc, base + ".final_ln");
}

static bool codec_x2_build_encode(lm_ggml_context * ctx_eval, void * user_data, lm_ggml_tensor ** out) {
    xcodec2_encode_build * p = static_cast<xcodec2_encode_build *>(user_data);
    if (ctx_eval == nullptr || p == nullptr || out == nullptr || p->model == nullptr) {
        return false;
    }
    if (p->n_pcm <= 0 || p->n_sem_frames <= 0 || p->n_codes <= 0) return false;

    auto W = [&](const std::string & name) -> lm_ggml_tensor * {
        return codec_graph_weight(ctx_eval, p->model, name);
    };

    // ----- Inputs -----
    lm_ggml_tensor * t_pcm = lm_ggml_new_tensor_2d(ctx_eval, LM_GGML_TYPE_F32, p->n_pcm, 1);
    lm_ggml_set_name(t_pcm, codec_x2_name_pcm());
    lm_ggml_tensor * t_mel = lm_ggml_new_tensor_2d(ctx_eval, LM_GGML_TYPE_F32, p->w2v_input_dim, p->n_sem_frames);
    lm_ggml_set_name(t_mel, codec_x2_name_mel_in());

    lm_ggml_tensor * alias = W(codec_x2_name_alias());
    if (alias == nullptr) return false;

    // ===== Acoustic path: BigCodec encoder =====
    lm_ggml_tensor * x_tc = codec_conv1d(ctx_eval, t_pcm,
                                      W("xcodec2.enc.codec.conv0.w"),
                                      W("xcodec2.enc.codec.conv0.b"),
                                      /*stride=*/1, /*dilation=*/1, /*padding=*/3);
    if (x_tc == nullptr) return false;

    const int32_t up_ratios[5] = { 2, 2, 4, 4, 5 };
    for (int32_t bi = 0; bi < 5; ++bi) {
        x_tc = codec_x2_encoder_block(ctx_eval, x_tc, bi + 1, up_ratios[bi], p->model, alias);
        if (x_tc == nullptr) return false;
    }
    // Final SnakeBeta + 1024-projection conv (k=3, p=1).
    x_tc = codec_op_alias_free_snake_beta_tc(
        ctx_eval, x_tc,
        W("xcodec2.enc.codec.final.act.alpha"),
        W("xcodec2.enc.codec.final.act.inv_beta"),
        alias);
    if (x_tc == nullptr) return false;
    x_tc = codec_conv1d(ctx_eval, x_tc,
                        W("xcodec2.enc.codec.final.conv.w"),
                        W("xcodec2.enc.codec.final.conv.b"),
                        /*stride=*/1, /*dilation=*/1, /*padding=*/1);
    if (x_tc == nullptr) return false;
    // Crop to expected acoustic length (handles BigCodec's stride padding tail).
    const int64_t t_ac = x_tc->ne[0];
    if (t_ac > p->n_codes) {
        x_tc = codec_op_crop_1d(ctx_eval, x_tc, 0, (int32_t) (t_ac - p->n_codes));
    }
    lm_ggml_tensor * acoustic_tc = x_tc;  // [n_codes, 1024]

    // ===== Semantic path: Wav2Vec2-Bert feature_projection + 16 conformer layers =====
    // mel input has ne=(input_dim=160, n_sem_frames). Transpose to (n_sem, 160).
    lm_ggml_tensor * mel_tc = lm_ggml_cont(ctx_eval, lm_ggml_transpose(ctx_eval, t_mel));   // [n_sem, 160]
    lm_ggml_tensor * h = codec_op_layer_norm_tc(
        ctx_eval, mel_tc, p->w2v_eps,
        W("xcodec2.w2v.feat_ln.w"), W("xcodec2.w2v.feat_ln.b"));
    if (h == nullptr) return false;
    h = codec_op_linear_tc(ctx_eval, h,
                           W("xcodec2.w2v.feat_proj.w"), W("xcodec2.w2v.feat_proj.b"));  // [n_sem, 1024]
    if (h == nullptr) return false;

    // Build the bucket index tensor used by every conformer layer's rel-key
    // attention.  `bucket(t_q, t_k) = clamp(t_k - t_q, -L, R) + L`.
    const int64_t t_sem = h->ne[0];
    lm_ggml_tensor * t_arange = lm_ggml_arange(ctx_eval, 0.0f, (float) t_sem, 1.0f);              // [t_sem]
    lm_ggml_tensor * tk_2d = lm_ggml_reshape_2d(ctx_eval, t_arange, t_sem, 1);
    lm_ggml_tensor * tq_2d = lm_ggml_reshape_2d(ctx_eval, t_arange, 1, t_sem);
    lm_ggml_tensor * full = lm_ggml_new_tensor_2d(ctx_eval, LM_GGML_TYPE_F32, t_sem, t_sem);
    lm_ggml_tensor * tk_rep = lm_ggml_repeat(ctx_eval, tk_2d, full);
    lm_ggml_tensor * tq_rep = lm_ggml_repeat(ctx_eval, tq_2d, full);
    lm_ggml_tensor * dist = lm_ggml_sub(ctx_eval, tk_rep, tq_rep);
    dist = lm_ggml_clamp(ctx_eval, dist, (float) -p->w2v_left_max, (float) p->w2v_right_max);
    lm_ggml_tensor * bucket_f = lm_ggml_scale_bias(ctx_eval, dist, 1.0f, (float) p->w2v_left_max);
    lm_ggml_tensor * bucket_i32 = lm_ggml_cast(ctx_eval, bucket_f, LM_GGML_TYPE_I32);
    lm_ggml_tensor * bucket_1d = lm_ggml_reshape_1d(ctx_eval, bucket_i32, t_sem * t_sem);

    // 16 conformer layers.
    for (int32_t li = 0; li < p->w2v_layers; ++li) {
        lm_ggml_tensor * dist_emb = W("xcodec2.w2v.l" + std::to_string(li) + ".attn.dist.w");
        if (dist_emb == nullptr) return false;
        h = codec_x2_w2v_layer(ctx_eval, h, p->model, li, p, dist_emb, bucket_1d);
        if (h == nullptr) return false;
    }

    // SemanticEncoder: initial_conv → (ReLU → conv → ReLU → conv) + residual → final_conv.
    // The residual is `ReLU(initial_conv(x))` — the upstream `nn.ReLU(inplace=True)`
    // mutates the input buffer before the residual sum happens, so what looks
    // like `residual = initial_conv(x)` in the source actually feeds the
    // post-ReLU value into the `+ x` skip connection.
    h = codec_conv1d(ctx_eval, h,
                     W("xcodec2.sem.initial.w"), nullptr,
                     /*stride=*/1, /*dilation=*/1, /*padding=*/1);
    if (h == nullptr) return false;
    h = lm_ggml_relu(ctx_eval, h);
    lm_ggml_tensor * sem_res = h;
    h = codec_conv1d(ctx_eval, h, W("xcodec2.sem.r1.w"), W("xcodec2.sem.r1.b"), 1, 1, 1);
    h = lm_ggml_relu(ctx_eval, h);
    h = codec_conv1d(ctx_eval, h, W("xcodec2.sem.r3.w"), W("xcodec2.sem.r3.b"), 1, 1, 1);
    h = lm_ggml_add(ctx_eval, h, sem_res);
    h = codec_conv1d(ctx_eval, h, W("xcodec2.sem.final.w"), nullptr, 1, 1, 1);
    if (h == nullptr) return false;

    // Match lengths against the acoustic path: truncate to min(T_ac, T_sem).
    const int64_t t_sem_out = h->ne[0];
    const int64_t t_min = std::min(t_sem_out, (int64_t) p->n_codes);
    if (t_sem_out > t_min) {
        h = codec_op_crop_1d(ctx_eval, h, 0, (int32_t) (t_sem_out - t_min));
    }
    if (acoustic_tc->ne[0] > t_min) {
        acoustic_tc = codec_op_crop_1d(ctx_eval, acoustic_tc, 0, (int32_t) (acoustic_tc->ne[0] - t_min));
    }

    // ===== Concat + fc_prior + FSQ encode =====
    // Both `h` and `acoustic_tc` are [t, 1024].  Concat along channel = ne[1].
    lm_ggml_tensor * concat = lm_ggml_concat(ctx_eval, h, acoustic_tc, /*dim=*/1);  // [t, 2048]
    lm_ggml_tensor * prior = codec_op_linear_tc(
        ctx_eval, concat,
        W("xcodec2.enc.fc_prior.w"), W("xcodec2.enc.fc_prior.b"));            // [t, 2048]
    if (prior == nullptr) return false;

    // FSQ.project_in: 2048 → 8 (== codebook_dim).
    lm_ggml_tensor * z = codec_op_linear_tc(
        ctx_eval, prior,
        W("xcodec2.enc.quant.project_in.w"),
        W("xcodec2.enc.quant.project_in.b"));                                 // [t, 8]
    if (z == nullptr) return false;

    // FSQ bound + quantize (matches vector_quantize_pytorch FSQ — half_l*tanh(z+shift)-offset, applied twice).
    const float eps = 1e-3f;
    const float half_l = (3.0f * (1.0f + eps)) / 2.0f;
    const float offset = 0.5f;
    const float shift = std::atanh(offset / half_l);
    const float half_width = 2.0f;

    auto fsq_bound = [&](lm_ggml_tensor * x) {
        lm_ggml_tensor * y = lm_ggml_tanh(ctx_eval, lm_ggml_scale_bias(ctx_eval, x, 1.0f, shift));
        return lm_ggml_scale_bias(ctx_eval, y, half_l, -offset);
    };
    z = fsq_bound(z);
    z = fsq_bound(z);
    lm_ggml_tensor * zq = lm_ggml_scale(ctx_eval, lm_ggml_round(ctx_eval, z), 1.0f / half_width);

    // Codebook indices: sum over feature axis (ne[1]) of zq_scaled * basis[k].
    // basis[k] = product(levels[:k]) = (1, 4, 16, 64, 256, 1024, 4096, 16384) for levels=[4]^8.
    const int32_t cb_dim = p->cb_dim;
    lm_ggml_tensor * basis = lm_ggml_new_tensor_1d(ctx_eval, LM_GGML_TYPE_F32, cb_dim);
    lm_ggml_set_name(basis, "xcodec2.enc.fsq.basis");

    lm_ggml_tensor * z_scaled = lm_ggml_scale_bias(ctx_eval, zq, half_width, half_width);  // [t, cb_dim]
    lm_ggml_tensor * basis_2d = lm_ggml_reshape_2d(ctx_eval, basis, 1, cb_dim);
    lm_ggml_tensor * basis_rep = lm_ggml_repeat(ctx_eval, basis_2d, z_scaled);
    lm_ggml_tensor * z_mul = lm_ggml_mul(ctx_eval, z_scaled, basis_rep);                    // [t, cb_dim]
    lm_ggml_tensor * z_ct = lm_ggml_cont(ctx_eval, lm_ggml_transpose(ctx_eval, z_mul));        // [cb_dim, t]
    lm_ggml_tensor * idx_sum = lm_ggml_sum_rows(ctx_eval, z_ct);                            // [1, t]
    lm_ggml_tensor * idx_1d = lm_ggml_reshape_1d(ctx_eval, idx_sum, z_mul->ne[0]);          // [t]
    lm_ggml_tensor * idx_2d = lm_ggml_reshape_2d(ctx_eval, idx_1d, (int32_t) z_mul->ne[0], 1);
    lm_ggml_tensor * out_t = lm_ggml_cont(ctx_eval, idx_2d);
    lm_ggml_set_name(out_t, codec_x2_name_codes());
    *out = out_t;
    return true;
}

static enum codec_status codec_x2_encode_graph(
    struct codec_context * ctx,
    const std::vector<float> & pcm,
    struct codec_token_buffer * out_tokens) {

    codec_xcodec2 & x2 = *static_cast<codec_xcodec2 *>(ctx->model->impl);
    if (pcm.empty()) {
        codec_context_set_error(ctx, "invalid xcodec2 PCM input");
        return CODEC_STATUS_INVALID_ARG;
    }

    // ----- Step 1: CPU mel-fbank features (matches SeamlessM4TFeatureExtractor) -----
    std::vector<float> mel_filters;
    {
        lm_ggml_tensor * t_mf = codec_model_get_tensor(ctx->model, "xcodec2.enc.mel.filters");
        if (t_mf == nullptr || !codec_tensor_as_vec_f32(t_mf, &mel_filters)) {
            codec_context_set_error(ctx, "missing or invalid xcodec2.enc.mel.filters");
            return CODEC_STATUS_INTERNAL_ERROR;
        }
    }
    std::vector<float> mel_window;
    {
        lm_ggml_tensor * t_mw = codec_model_get_tensor(ctx->model, "xcodec2.enc.mel.window");
        if (t_mw == nullptr || !codec_tensor_as_vec_f32(t_mw, &mel_window)) {
            codec_context_set_error(ctx, "missing or invalid xcodec2.enc.mel.window");
            return CODEC_STATUS_INTERNAL_ERROR;
        }
    }

    std::vector<float> mel_features;
    int32_t n_sem_frames = 0;
    std::string err;
    const int32_t n_freq = x2.mel_n_fft / 2 + 1;
    if (!codec_runtime_w2v_bert_features(
            pcm, mel_filters, n_freq, x2.mel_n_mels, mel_window,
            x2.mel_n_fft, x2.mel_win, x2.mel_hop,
            x2.mel_preemphasis, x2.mel_floor, x2.mel_stride,
            &mel_features, &n_sem_frames, &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    // ----- Step 2: derive expected acoustic frame count and the common N -----
    const int32_t n_pcm = (int32_t) pcm.size();
    const int32_t hop = std::max(1, x2.hop_size);
    const int32_t n_acoustic = n_pcm / hop;
    const int32_t n_codes = std::min(n_acoustic, n_sem_frames);
    if (n_codes <= 0) {
        codec_context_set_error(ctx, "xcodec2 encode produced no frames");
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    // ----- Step 3: build & run the encode graph -----
    xcodec2_encode_build build = {};
    build.n_pcm        = n_pcm;
    build.n_sem_frames = n_sem_frames;
    build.n_acoustic   = n_acoustic;
    build.n_codes      = n_codes;
    build.hidden       = x2.hidden_dim;
    build.vq_dim       = x2.vq_dim;
    build.cb_dim       = x2.codebook_dim;
    build.w2v_layers   = x2.w2v_layers;
    build.w2v_hidden   = x2.w2v_hidden;
    build.w2v_heads    = x2.w2v_heads;
    build.w2v_head_dim = x2.w2v_head_dim;
    build.w2v_input_dim = x2.w2v_input_dim;
    build.w2v_dw_kernel = x2.w2v_dw_kernel;
    build.w2v_intermediate = x2.w2v_intermediate;
    build.w2v_left_max = x2.w2v_left_max_pos;
    build.w2v_right_max = x2.w2v_right_max_pos;
    build.w2v_eps      = x2.w2v_layer_norm_eps;
    build.enc_ngf      = x2.enc_ngf;
    build.cfg          = &x2;
    build.model        = ctx->model;

    // The encode graph is wide: 16 conformer layers + 5 BigCodec blocks each
    // with 3 alias-free Activation1d × {WNConv k=7 + WNConv k=1} residual
    // units.  Each conformer attention also instantiates a (T, T, head_dim)
    // gathered-distance-embedding tensor.  Budget grows roughly linearly with
    // T_pcm for the BigCodec stack and quadratically with T_sem for the
    // attention rel-key gather, so size both terms generously.
    codec_graph_eval_guard eval_guard(ctx);

    codec_graph_cache_entry * entry = nullptr;
    if (!codec_graph_cache_get_or_build(
            ctx,
            { CODEC_GRAPH_XCODEC2_ENCODE,
              /*n_frames=*/n_codes, /*n_q=*/1, /*hop=*/hop,
              /*n_in=*/n_pcm, /*latent_dim=*/n_sem_frames },
            codec_x2_build_encode,
            &build,
            sizeof(build),
            &entry,
            &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    lm_ggml_tensor * t_pcm = codec_graph_get_tensor(ctx, entry, codec_x2_name_pcm());
    lm_ggml_tensor * t_mel = codec_graph_get_tensor(ctx, entry, codec_x2_name_mel_in());
    lm_ggml_tensor * t_basis = codec_graph_get_tensor(ctx, entry, "xcodec2.enc.fsq.basis");
    lm_ggml_tensor * t_out = codec_graph_get_tensor(ctx, entry, codec_x2_name_codes());
    if (t_pcm == nullptr || t_mel == nullptr || t_basis == nullptr || t_out == nullptr) {
        codec_context_set_error(ctx, "cached xcodec2 encode graph is invalid");
        return CODEC_STATUS_INTERNAL_ERROR;
    }
    if (!codec_graph_prepare_io(ctx, entry, &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    // Inputs: PCM, mel features (laid out with input_dim innermost = 160).
    if (!codec_runtime_write_tensor(t_pcm, pcm.data(), pcm.size() * sizeof(float), &err) ||
        !codec_runtime_write_tensor(t_mel, mel_features.data(),
                                    mel_features.size() * sizeof(float), &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    // FSQ basis: [1, 4, 16, 64, 256, 1024, 4096, 16384].
    const float basis_vals[8] = { 1, 4, 16, 64, 256, 1024, 4096, 16384 };
    if (!codec_runtime_write_tensor(t_basis, basis_vals, sizeof(basis_vals), &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    const int32_t n_threads = ctx->model->n_threads > 0 ? ctx->model->n_threads : 1;
    if (!codec_graph_compute(ctx, entry, n_threads, &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    if (t_out->type != LM_GGML_TYPE_F32 || t_out->ne[1] != 1) {
        codec_context_set_error(ctx, "unexpected xcodec2 token tensor shape/type");
        return CODEC_STATUS_INTERNAL_ERROR;
    }
    const int32_t n_frames_out = (int32_t) t_out->ne[0];
    std::vector<float> out_f((size_t) n_frames_out, 0.0f);
    if (!codec_runtime_read_tensor(t_out, out_f.data(), out_f.size() * sizeof(float), &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    int32_t * data = static_cast<int32_t *>(std::malloc((size_t) n_frames_out * sizeof(int32_t)));
    if (data == nullptr) {
        codec_context_set_error(ctx, "failed to allocate token output");
        return CODEC_STATUS_INTERNAL_ERROR;
    }
    for (int32_t ti = 0; ti < n_frames_out; ++ti) {
        float v = out_f[(size_t) ti];
        if (!std::isfinite(v)) v = 0.0f;
        int32_t idx = (int32_t) std::lrintf(v);
        if (idx < 0) idx = 0;
        if (idx > x2.codebook_size - 1) idx = x2.codebook_size - 1;
        data[ti] = idx;
    }

    codec_token_buffer_reset(out_tokens);
    out_tokens->data = data;
    out_tokens->n_tokens = n_frames_out;
    out_tokens->n_frames = n_frames_out;
    out_tokens->n_q = 1;
    out_tokens->codebook_size = x2.codebook_size;
    out_tokens->sample_rate = x2.encode_sample_rate;
    out_tokens->hop_size = x2.hop_size;
    return CODEC_STATUS_SUCCESS;
}

enum codec_status codec_xcodec2_encode(
    struct codec_context * ctx,
    const std::vector<float> & pcm,
    struct codec_token_buffer * out_tokens,
    struct codec_latent_buffer * out_latent,
    struct codec_encode_params params) {

    (void) out_latent;
    codec_xcodec2 & x2 = *static_cast<codec_xcodec2 *>(ctx->model->impl);
    if (!x2.has_encoder) {
        codec_context_set_error(ctx, "model metadata indicates no encoder");
        return CODEC_STATUS_INVALID_STATE;
    }
    if (params.n_q != 0 && params.n_q != 1) {
        codec_context_set_error(ctx, "xcodec2 encode n_q must be 0 or 1");
        return CODEC_STATUS_INVALID_ARG;
    }
    return codec_x2_encode_graph(ctx, pcm, out_tokens);
}

enum codec_status codec_xcodec2_init(struct codec_model * model) {
    if (model == nullptr || model->impl == nullptr || model->gguf == nullptr) {
        return CODEC_STATUS_INVALID_ARG;
    }
    codec_xcodec2 & x2 = *static_cast<codec_xcodec2 *>(model->impl);

    x2.sample_rate        = codec_read_i32_kv(model->gguf, "codec.sample_rate", x2.sample_rate);
    x2.encode_sample_rate = codec_read_i32_kv(model->gguf, "codec.encode_sample_rate", x2.encode_sample_rate);
    x2.hop_size           = codec_read_i32_kv(model->gguf, "codec.hop_size", x2.hop_size);
    x2.n_fft              = codec_read_i32_kv(model->gguf, "codec.n_fft", x2.n_fft);
    x2.n_q                = codec_read_i32_kv(model->gguf, "codec.n_q", x2.n_q);
    x2.codebook_size      = codec_read_i32_kv(model->gguf, "codec.codebook_size", x2.codebook_size);
    x2.codebook_dim       = codec_read_i32_kv(model->gguf, "codec.codebook_dim", x2.codebook_dim);
    x2.latent_dim         = codec_read_i32_kv(model->gguf, "codec.latent_dim", x2.latent_dim);
    x2.has_encoder        = codec_read_bool_kv(model->gguf, "codec.has_encoder", x2.has_encoder);
    x2.has_decoder        = codec_read_bool_kv(model->gguf, "codec.has_decoder", x2.has_decoder);

    x2.hidden_dim  = codec_read_i32_kv(model->gguf, "xcodec2.hidden_dim", x2.hidden_dim);
    x2.vq_dim      = codec_read_i32_kv(model->gguf, "xcodec2.vq_dim", x2.vq_dim);
    x2.num_layers  = codec_read_i32_kv(model->gguf, "xcodec2.num_layers", x2.num_layers);
    x2.num_heads   = codec_read_i32_kv(model->gguf, "xcodec2.num_heads", x2.num_heads);
    x2.head_dim    = codec_read_i32_kv(model->gguf, "xcodec2.head_dim", x2.head_dim);
    x2.rope_theta  = codec_read_f32_kv(model->gguf, "xcodec2.rope_theta", x2.rope_theta);

    model->sample_rate        = x2.sample_rate;
    model->encode_sample_rate = x2.encode_sample_rate;
    model->has_encoder        = x2.has_encoder;
    model->has_decoder        = x2.has_decoder;
    model->hop_size           = x2.hop_size;
    model->n_q                = x2.n_q;
    model->codebook_size      = x2.codebook_size;
    model->latent_dim         = x2.latent_dim;
    model->n_fft              = x2.n_fft;
    model->win_length         = x2.n_fft;
    return CODEC_STATUS_SUCCESS;
}

static void * codec_x2_create_impl() {
    return new (std::nothrow) codec_xcodec2();
}
static void codec_x2_destroy_impl(void * ptr) {
    delete static_cast<codec_xcodec2 *>(ptr);
}

static enum codec_status codec_x2_decode_wrap(
    struct codec_context * ctx,
    const struct codec_token_buffer * tokens,
    struct codec_pcm_buffer * out_pcm,
    struct codec_decode_params params) {
    return codec_xcodec2_decode(ctx, tokens, out_pcm, params);
}

static enum codec_status codec_x2_encode_wrap(
    struct codec_context * ctx,
    const std::vector<float> & pcm,
    struct codec_token_buffer * out_tokens,
    struct codec_latent_buffer * out_latent,
    struct codec_encode_params params) {
    return codec_xcodec2_encode(ctx, pcm, out_tokens, out_latent, params);
}

const struct codec_model_vtable * codec_xcodec2_vtable() {
    static const codec_model_vtable vtable = {
        CODEC_ARCH_XCODEC2,
        "xcodec2",
        codec_x2_create_impl,
        codec_x2_destroy_impl,
        codec_xcodec2_init,
        codec_graph_size_exact,
        codec_x2_encode_wrap,
        codec_x2_decode_wrap,
    };
    return &vtable;
}
