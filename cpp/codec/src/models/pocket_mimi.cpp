#include "pocket_mimi.h"

#include "../ops/conv1d.h"
#include "../ops/convtr1d.h"
#include "../ops/lm_ggml_ops.h"
#include "../ops/lm_attn.h"
#include "../ops/rope.h"
#include "../runtime/graph.h"
#include "../runtime/tensor_utils.h"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <new>
#include <string>
#include <vector>

// =====================================================================
// Pocket-TTS custom Mimi variant.
//
// Continuous-latent codec: FlowLM emits a 32-dim latent per 12.5 Hz frame.
//
//   DECODE  latent [32, T]
//     -> quantizer.output_proj  (Conv1d k1, 32 -> 512, no bias)
//     -> upsample               (ConvTranspose1d depthwise, stride 16, k32,
//                                expanded to dense at convert time)
//     -> decoder_transformer    (2x LayerScale block: LN -> RoPE causal attn
//                                -> LayerScale ; LN -> GELU FFN -> LayerScale)
//     -> SEANet decoder         (conv k7; [convtr sN; ELU-resnet] x3; conv k3)
//     -> PCM [1, T*1920]
//
//   ENCODE  PCM [1, N]
//     -> SEANet encoder         (conv k7; [ELU-resnet; strided conv] x3;
//                                conv k3)
//     -> encoder_transformer    (same block)
//     -> downsample             (Conv1d 512 -> 32, stride 16, k32, replicate
//                                pad, no bias)
//     -> latent [32, T]
//
// All convs are causal (StreamingConv1d with pad_mode constant = left-pad
// kernel-stride).  downsample uses pad_mode="replicate".  Transposed convs
// right-trim kernel-stride (StreamingConvTranspose1d non-streaming).
// Transformer FFN uses F.gelu (erf).  Attention has a sliding window of
// context=250 frames (only bites at T>250 ~= 20 s).
// =====================================================================

namespace {

const char * pm_name_lat()   { return "pocket_mimi.decode.lat"; }
const char * pm_name_audio() { return "pocket_mimi.decode.audio"; }
const char * pm_name_pcm()   { return "pocket_mimi.encode.pcm"; }
const char * pm_name_mu()    { return "pocket_mimi.encode.mu"; }

// One LayerScale transformer block operating on channel-major x_ct [c, t].
// Matches mimi.cpp's decode/encode transformer block (LN -> RoPE causal attn
// -> LayerScale add ; LN -> fc1 -> GELU -> fc2 -> LayerScale add).
lm_ggml_tensor * pm_transformer_block(
    lm_ggml_context * ctx_eval,
    lm_ggml_tensor * x_ct,
    const codec_model * model,
    const std::string & prefix,     // e.g. "pocket_mimi.dtr.l0"
    int32_t head_dim,
    int32_t n_heads,
    int32_t c,
    float   max_period,
    int32_t context) {

    auto W = [&](const std::string & n) { return codec_graph_weight(ctx_eval, model, n); };

    lm_ggml_tensor * inln_w = W(prefix + ".inln.w");
    lm_ggml_tensor * inln_b = W(prefix + ".inln.b");
    lm_ggml_tensor * paln_w = W(prefix + ".paln.w");
    lm_ggml_tensor * paln_b = W(prefix + ".paln.b");
    lm_ggml_tensor * q_w = W(prefix + ".attn.q_proj.w");
    lm_ggml_tensor * k_w = W(prefix + ".attn.k_proj.w");
    lm_ggml_tensor * v_w = W(prefix + ".attn.v_proj.w");
    lm_ggml_tensor * o_w = W(prefix + ".attn.o_proj.w");
    lm_ggml_tensor * fc1_w = W(prefix + ".mlp.fc1.w");
    lm_ggml_tensor * fc2_w = W(prefix + ".mlp.fc2.w");
    lm_ggml_tensor * sa_scale = W(prefix + ".sa_ls.scale");
    lm_ggml_tensor * mlp_scale = W(prefix + ".mlp_ls.scale");
    if (inln_w == nullptr || inln_b == nullptr || paln_w == nullptr || paln_b == nullptr ||
        q_w == nullptr || k_w == nullptr || v_w == nullptr || o_w == nullptr ||
        fc1_w == nullptr || fc2_w == nullptr || sa_scale == nullptr || mlp_scale == nullptr) {
        return nullptr;
    }

    lm_ggml_tensor * h = codec_op_layer_norm_ct(ctx_eval, x_ct, 1e-5f, inln_w, inln_b);
    if (h == nullptr) return nullptr;
    const int64_t t_cur = h->ne[1];

    lm_ggml_tensor * q = lm_ggml_mul_mat(ctx_eval, q_w, h);
    lm_ggml_tensor * k = lm_ggml_mul_mat(ctx_eval, k_w, h);
    lm_ggml_tensor * v = lm_ggml_mul_mat(ctx_eval, v_w, h);
    if (q == nullptr || k == nullptr || v == nullptr) return nullptr;

    lm_ggml_tensor * q_dth = lm_ggml_permute(ctx_eval, lm_ggml_reshape_3d(ctx_eval, q, head_dim, n_heads, t_cur), 0, 2, 1, 3);
    lm_ggml_tensor * k_dth = lm_ggml_permute(ctx_eval, lm_ggml_reshape_3d(ctx_eval, k, head_dim, n_heads, t_cur), 0, 2, 1, 3);
    lm_ggml_tensor * v_dth = lm_ggml_permute(ctx_eval, lm_ggml_reshape_3d(ctx_eval, v, head_dim, n_heads, t_cur), 0, 2, 1, 3);

    lm_ggml_tensor * q_rope = codec_op_rope(ctx_eval, q_dth, head_dim, max_period, 1.0f, CODEC_ROPE_MODE_NORMAL);
    lm_ggml_tensor * k_rope = codec_op_rope(ctx_eval, k_dth, head_dim, max_period, 1.0f, CODEC_ROPE_MODE_NORMAL);
    if (q_rope == nullptr || k_rope == nullptr) return nullptr;

    codec_lm_attn_params attn_p = {};
    attn_p.scale = 1.0f / std::sqrt((float) head_dim);
    attn_p.causal = true;
    attn_p.window = context > 0 ? context : 0;
    lm_ggml_tensor * attn_ctx = codec_op_lm_attn_ctx_dth(ctx_eval, q_rope, k_rope, v_dth, &attn_p);
    if (attn_ctx == nullptr) return nullptr;

    lm_ggml_tensor * attn_ct = lm_ggml_reshape_2d(
        ctx_eval, lm_ggml_cont(ctx_eval, lm_ggml_permute(ctx_eval, attn_ctx, 0, 2, 1, 3)), c, t_cur);
    lm_ggml_tensor * attn_proj = lm_ggml_mul_mat(ctx_eval, o_w, attn_ct);
    if (attn_proj == nullptr) return nullptr;
    x_ct = lm_ggml_add(ctx_eval, x_ct, codec_op_channel_scale(ctx_eval, attn_proj, sa_scale));

    lm_ggml_tensor * m = codec_op_layer_norm_ct(ctx_eval, x_ct, 1e-5f, paln_w, paln_b);
    if (m == nullptr) return nullptr;
    m = lm_ggml_mul_mat(ctx_eval, fc1_w, m);
    if (m == nullptr) return nullptr;
    m = lm_ggml_gelu_erf(ctx_eval, m);
    m = lm_ggml_mul_mat(ctx_eval, fc2_w, m);
    if (m == nullptr) return nullptr;
    x_ct = lm_ggml_add(ctx_eval, x_ct, codec_op_channel_scale(ctx_eval, m, mlp_scale));
    return x_ct;
}

// SEANet residual block: x + Conv1x1(ELU(Conv_k3(ELU(x)))).  Both convs
// causal, length-preserving.  x is time-major [t, c].
lm_ggml_tensor * pm_seanet_resblock(
    lm_ggml_context * ctx_eval,
    lm_ggml_tensor * x,
    lm_ggml_tensor * c1_w, lm_ggml_tensor * c1_b,
    lm_ggml_tensor * c2_w, lm_ggml_tensor * c2_b) {
    if (x == nullptr || c1_w == nullptr || c1_b == nullptr || c2_w == nullptr || c2_b == nullptr) {
        return nullptr;
    }
    lm_ggml_tensor * h = lm_ggml_elu(ctx_eval, x);
    h = codec_conv1d_causal(ctx_eval, h, c1_w, c1_b, 1, 1);
    if (h == nullptr) return nullptr;
    h = lm_ggml_elu(ctx_eval, h);
    h = codec_conv1d_causal(ctx_eval, h, c2_w, c2_b, 1, 1);
    if (h == nullptr) return nullptr;
    return lm_ggml_cont(ctx_eval, lm_ggml_add(ctx_eval, x, h));
}

struct pm_decode_build {
    int32_t n_frames = 0;
    int32_t latent_dim = 0;
    codec_pocket_mimi cfg;
    const codec_model * model = nullptr;
};

bool pm_build_decode(lm_ggml_context * ctx_eval, void * user_data, lm_ggml_tensor ** out) {
    pm_decode_build * p = static_cast<pm_decode_build *>(user_data);
    if (ctx_eval == nullptr || p == nullptr || out == nullptr || p->model == nullptr) return false;
    if (p->n_frames <= 0 || p->latent_dim <= 0) return false;
    const codec_pocket_mimi & m = p->cfg;

    auto W = [&](const std::string & n) { return codec_graph_weight(ctx_eval, p->model, n); };

    // Latent input: ne=(t, latent_dim); buffer is [latent_dim, n_frames]
    // row-major (= PyTorch z[D, T]).
    lm_ggml_tensor * t_lat = lm_ggml_new_tensor_2d(ctx_eval, LM_GGML_TYPE_F32, p->n_frames, p->latent_dim);
    lm_ggml_set_name(t_lat, pm_name_lat());

    // quantizer.output_proj: Conv1d k1, 32 -> 512 (pointwise, no bias).
    lm_ggml_tensor * x = codec_conv1d(ctx_eval, t_lat, W("pocket_mimi.quant.out_proj.w"), nullptr, 1, 1, 0);
    if (x == nullptr) return false;   // [t, 512]

    // upsample: depthwise ConvTranspose1d (dense-expanded), stride 16, k32.
    x = codec_convtr1d_causal(ctx_eval, x, W("pocket_mimi.upsample.w"), nullptr, m.resample_stride, 1);
    if (x == nullptr) return false;   // [t*16, 512]

    // decoder_transformer (channel-major).
    lm_ggml_tensor * x_ct = lm_ggml_cont(ctx_eval, lm_ggml_transpose(ctx_eval, x));  // [512, t*16]
    for (int32_t li = 0; li < m.tf_layers; ++li) {
        x_ct = pm_transformer_block(ctx_eval, x_ct, p->model, "pocket_mimi.dtr.l" + std::to_string(li),
                                    m.tf_head_dim, m.tf_heads, m.outer_dim, m.tf_max_period, m.tf_context);
        if (x_ct == nullptr) return false;
    }
    x = lm_ggml_cont(ctx_eval, lm_ggml_transpose(ctx_eval, x_ct));  // [t*16, 512]

    // SEANet decoder conv stack.
    lm_ggml_tensor * l0_w = W("pocket_mimi.dec.l0.w");
    lm_ggml_tensor * l0_b = W("pocket_mimi.dec.l0.b");
    lm_ggml_tensor * l2_w = W("pocket_mimi.dec.l2.w");
    lm_ggml_tensor * l2_b = W("pocket_mimi.dec.l2.b");
    lm_ggml_tensor * r0c1_w = W("pocket_mimi.dec.r0.c1.w");
    lm_ggml_tensor * r0c1_b = W("pocket_mimi.dec.r0.c1.b");
    lm_ggml_tensor * r0c2_w = W("pocket_mimi.dec.r0.c2.w");
    lm_ggml_tensor * r0c2_b = W("pocket_mimi.dec.r0.c2.b");
    lm_ggml_tensor * l5_w = W("pocket_mimi.dec.l5.w");
    lm_ggml_tensor * l5_b = W("pocket_mimi.dec.l5.b");
    lm_ggml_tensor * r1c1_w = W("pocket_mimi.dec.r1.c1.w");
    lm_ggml_tensor * r1c1_b = W("pocket_mimi.dec.r1.c1.b");
    lm_ggml_tensor * r1c2_w = W("pocket_mimi.dec.r1.c2.w");
    lm_ggml_tensor * r1c2_b = W("pocket_mimi.dec.r1.c2.b");
    lm_ggml_tensor * l8_w = W("pocket_mimi.dec.l8.w");
    lm_ggml_tensor * l8_b = W("pocket_mimi.dec.l8.b");
    lm_ggml_tensor * r2c1_w = W("pocket_mimi.dec.r2.c1.w");
    lm_ggml_tensor * r2c1_b = W("pocket_mimi.dec.r2.c1.b");
    lm_ggml_tensor * r2c2_w = W("pocket_mimi.dec.r2.c2.w");
    lm_ggml_tensor * r2c2_b = W("pocket_mimi.dec.r2.c2.b");
    lm_ggml_tensor * l11_w = W("pocket_mimi.dec.l11.w");
    lm_ggml_tensor * l11_b = W("pocket_mimi.dec.l11.b");
    if (l0_w == nullptr || l0_b == nullptr || l2_w == nullptr || l2_b == nullptr ||
        r0c1_w == nullptr || r0c1_b == nullptr || r0c2_w == nullptr || r0c2_b == nullptr ||
        l5_w == nullptr || l5_b == nullptr ||
        r1c1_w == nullptr || r1c1_b == nullptr || r1c2_w == nullptr || r1c2_b == nullptr ||
        l8_w == nullptr || l8_b == nullptr ||
        r2c1_w == nullptr || r2c1_b == nullptr || r2c2_w == nullptr || r2c2_b == nullptr ||
        l11_w == nullptr || l11_b == nullptr) {
        return false;
    }

    x = codec_conv1d_causal(ctx_eval, x, l0_w, l0_b, 1, 1);          // conv 512->512 k7
    if (x == nullptr) return false;
    x = lm_ggml_elu(ctx_eval, x);
    x = codec_convtr1d_causal(ctx_eval, x, l2_w, l2_b, m.decoder_ratios[0], 1);   // convtr s6
    x = pm_seanet_resblock(ctx_eval, x, r0c1_w, r0c1_b, r0c2_w, r0c2_b);
    if (x == nullptr) return false;
    x = lm_ggml_elu(ctx_eval, x);
    x = codec_convtr1d_causal(ctx_eval, x, l5_w, l5_b, m.decoder_ratios[1], 1);   // convtr s5
    x = pm_seanet_resblock(ctx_eval, x, r1c1_w, r1c1_b, r1c2_w, r1c2_b);
    if (x == nullptr) return false;
    x = lm_ggml_elu(ctx_eval, x);
    x = codec_convtr1d_causal(ctx_eval, x, l8_w, l8_b, m.decoder_ratios[2], 1);   // convtr s4
    x = pm_seanet_resblock(ctx_eval, x, r2c1_w, r2c1_b, r2c2_w, r2c2_b);
    if (x == nullptr) return false;
    x = lm_ggml_elu(ctx_eval, x);
    lm_ggml_tensor * t_pcm = codec_conv1d_causal(ctx_eval, x, l11_w, l11_b, 1, 1);   // conv 64->1 k3
    if (t_pcm == nullptr) return false;

    lm_ggml_tensor * t_out = lm_ggml_cont(ctx_eval, t_pcm);
    lm_ggml_set_name(t_out, pm_name_audio());
    *out = t_out;
    return true;
}

struct pm_encode_build {
    int32_t n_pcm = 0;
    codec_pocket_mimi cfg;
    const codec_model * model = nullptr;
};

bool pm_build_encode(lm_ggml_context * ctx_eval, void * user_data, lm_ggml_tensor ** out) {
    pm_encode_build * p = static_cast<pm_encode_build *>(user_data);
    if (ctx_eval == nullptr || p == nullptr || out == nullptr || p->model == nullptr || p->n_pcm <= 0) return false;
    const codec_pocket_mimi & m = p->cfg;

    auto W = [&](const std::string & n) { return codec_graph_weight(ctx_eval, p->model, n); };

    lm_ggml_tensor * t_pcm = lm_ggml_new_tensor_2d(ctx_eval, LM_GGML_TYPE_F32, p->n_pcm, 1);
    lm_ggml_set_name(t_pcm, pm_name_pcm());

    // SEANet encoder conv stack.
    lm_ggml_tensor * l0_w = W("pocket_mimi.enc.l0.w");
    lm_ggml_tensor * l0_b = W("pocket_mimi.enc.l0.b");
    lm_ggml_tensor * r0c1_w = W("pocket_mimi.enc.r0.c1.w");
    lm_ggml_tensor * r0c1_b = W("pocket_mimi.enc.r0.c1.b");
    lm_ggml_tensor * r0c2_w = W("pocket_mimi.enc.r0.c2.w");
    lm_ggml_tensor * r0c2_b = W("pocket_mimi.enc.r0.c2.b");
    lm_ggml_tensor * l3_w = W("pocket_mimi.enc.l3.w");
    lm_ggml_tensor * l3_b = W("pocket_mimi.enc.l3.b");
    lm_ggml_tensor * r1c1_w = W("pocket_mimi.enc.r1.c1.w");
    lm_ggml_tensor * r1c1_b = W("pocket_mimi.enc.r1.c1.b");
    lm_ggml_tensor * r1c2_w = W("pocket_mimi.enc.r1.c2.w");
    lm_ggml_tensor * r1c2_b = W("pocket_mimi.enc.r1.c2.b");
    lm_ggml_tensor * l6_w = W("pocket_mimi.enc.l6.w");
    lm_ggml_tensor * l6_b = W("pocket_mimi.enc.l6.b");
    lm_ggml_tensor * r2c1_w = W("pocket_mimi.enc.r2.c1.w");
    lm_ggml_tensor * r2c1_b = W("pocket_mimi.enc.r2.c1.b");
    lm_ggml_tensor * r2c2_w = W("pocket_mimi.enc.r2.c2.w");
    lm_ggml_tensor * r2c2_b = W("pocket_mimi.enc.r2.c2.b");
    lm_ggml_tensor * l9_w = W("pocket_mimi.enc.l9.w");
    lm_ggml_tensor * l9_b = W("pocket_mimi.enc.l9.b");
    lm_ggml_tensor * l11_w = W("pocket_mimi.enc.l11.w");
    lm_ggml_tensor * l11_b = W("pocket_mimi.enc.l11.b");
    if (l0_w == nullptr || l0_b == nullptr ||
        r0c1_w == nullptr || r0c1_b == nullptr || r0c2_w == nullptr || r0c2_b == nullptr ||
        l3_w == nullptr || l3_b == nullptr ||
        r1c1_w == nullptr || r1c1_b == nullptr || r1c2_w == nullptr || r1c2_b == nullptr ||
        l6_w == nullptr || l6_b == nullptr ||
        r2c1_w == nullptr || r2c1_b == nullptr || r2c2_w == nullptr || r2c2_b == nullptr ||
        l9_w == nullptr || l9_b == nullptr || l11_w == nullptr || l11_b == nullptr) {
        return false;
    }

    lm_ggml_tensor * x = codec_conv1d_causal(ctx_eval, t_pcm, l0_w, l0_b, 1, 1);    // conv 1->64 k7
    if (x == nullptr) return false;
    x = pm_seanet_resblock(ctx_eval, x, r0c1_w, r0c1_b, r0c2_w, r0c2_b);
    if (x == nullptr) return false;
    x = lm_ggml_elu(ctx_eval, x);
    x = codec_conv1d_causal(ctx_eval, x, l3_w, l3_b, m.encoder_ratios[0], 1);    // down s4
    if (x == nullptr) return false;
    x = pm_seanet_resblock(ctx_eval, x, r1c1_w, r1c1_b, r1c2_w, r1c2_b);
    if (x == nullptr) return false;
    x = lm_ggml_elu(ctx_eval, x);
    x = codec_conv1d_causal(ctx_eval, x, l6_w, l6_b, m.encoder_ratios[1], 1);    // down s5
    if (x == nullptr) return false;
    x = pm_seanet_resblock(ctx_eval, x, r2c1_w, r2c1_b, r2c2_w, r2c2_b);
    if (x == nullptr) return false;
    x = lm_ggml_elu(ctx_eval, x);
    x = codec_conv1d_causal(ctx_eval, x, l9_w, l9_b, m.encoder_ratios[2], 1);    // down s6
    if (x == nullptr) return false;
    x = lm_ggml_elu(ctx_eval, x);
    x = codec_conv1d_causal(ctx_eval, x, l11_w, l11_b, 1, 1);                    // conv 512->512 k3
    if (x == nullptr) return false;

    // encoder_transformer (channel-major).
    lm_ggml_tensor * x_ct = lm_ggml_cont(ctx_eval, lm_ggml_transpose(ctx_eval, x));  // [512, t]
    for (int32_t li = 0; li < m.tf_layers; ++li) {
        x_ct = pm_transformer_block(ctx_eval, x_ct, p->model, "pocket_mimi.etr.l" + std::to_string(li),
                                    m.tf_head_dim, m.tf_heads, m.outer_dim, m.tf_max_period, m.tf_context);
        if (x_ct == nullptr) return false;
    }
    x = lm_ggml_cont(ctx_eval, lm_ggml_transpose(ctx_eval, x_ct));  // [t, 512]

    // downsample: Conv1d 512 -> 32, stride 16, k32, replicate pad, no bias.
    x = codec_conv1d_causal_replicate(ctx_eval, x, W("pocket_mimi.downsample.w"), nullptr, m.resample_stride, 1);
    if (x == nullptr) return false;

    // x is [t_lat, ldim] (ne0=t_lat, ne1=ldim); its raw buffer is channel-major
    // [ldim, t_lat] (buffer[d*t_lat + t]), matching the decode latent input.
    lm_ggml_set_name(x, pm_name_mu());
    *out = x;
    return true;
}

}  // namespace

// ---------------------------------------------------------------------
// decode_latent entry
// ---------------------------------------------------------------------

static enum codec_status codec_pocket_mimi_decode_latent(
    struct codec_context * ctx,
    const float * quantized_representation,
    int32_t latent_dim,
    int32_t n_frames,
    struct codec_pcm_buffer * out_pcm,
    struct codec_decode_params params) {
    (void) params;

    codec_pocket_mimi & m = *static_cast<codec_pocket_mimi *>(ctx->model->impl);
    if (!m.has_decoder) {
        codec_context_set_error(ctx, "model metadata indicates no decoder");
        return CODEC_STATUS_INVALID_STATE;
    }
    if (quantized_representation == nullptr || n_frames <= 0) {
        codec_context_set_error(ctx, "invalid latent input");
        return CODEC_STATUS_INVALID_ARG;
    }
    if (latent_dim != m.latent_dim) {
        codec_context_set_error(ctx, "latent_dim mismatch with model");
        return CODEC_STATUS_INVALID_ARG;
    }

    pm_decode_build build = {};
    build.n_frames = n_frames;
    build.latent_dim = latent_dim;
    build.cfg = m;
    build.model = ctx->model;

    codec_graph_eval_guard guard(ctx);
    std::string err;
    codec_graph_cache_entry * entry = nullptr;
    if (!codec_graph_cache_get_or_build(
            ctx,
            { CODEC_GRAPH_POCKET_MIMI_DECODE, /*n_frames=*/n_frames, /*n_q=*/0,
              /*hop=*/m.hop_size, /*n_in=*/0, /*latent_dim=*/latent_dim },
            pm_build_decode, &build, sizeof(build), &entry, &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    lm_ggml_tensor * t_lat = codec_graph_get_tensor(ctx, entry, pm_name_lat());
    lm_ggml_tensor * t_out = codec_graph_get_tensor(ctx, entry, pm_name_audio());
    if (t_lat == nullptr || t_out == nullptr) {
        codec_context_set_error(ctx, "cached Pocket-Mimi decode graph is invalid");
        return CODEC_STATUS_INTERNAL_ERROR;
    }
    if (!codec_graph_prepare_io(ctx, entry, &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }
    if (!codec_runtime_write_tensor(t_lat, quantized_representation,
                                    (size_t) n_frames * (size_t) latent_dim * sizeof(float), &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }
    const int32_t n_threads = ctx->model->n_threads > 0 ? ctx->model->n_threads : 1;
    if (!codec_graph_compute(ctx, entry, n_threads, &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }
    if (t_out->type != LM_GGML_TYPE_F32 || t_out->ne[1] != 1) {
        codec_context_set_error(ctx, "unexpected Pocket-Mimi output shape/type");
        return CODEC_STATUS_INTERNAL_ERROR;
    }
    const int32_t n_samples = (int32_t) t_out->ne[0];
    std::vector<float> pcm_v((size_t) n_samples, 0.0f);
    if (!codec_runtime_read_tensor(t_out, pcm_v.data(), pcm_v.size() * sizeof(float), &err)) {
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
    out_pcm->n_samples = n_samples;
    out_pcm->sample_rate = m.sample_rate;
    out_pcm->n_channels = 1;
    return CODEC_STATUS_SUCCESS;
}

// ---------------------------------------------------------------------
// encode entry (audio → latent mu)
// ---------------------------------------------------------------------

static enum codec_status codec_pocket_mimi_encode(
    struct codec_context * ctx,
    const std::vector<float> & pcm,
    struct codec_token_buffer * out_tokens,
    struct codec_latent_buffer * out_latent,
    struct codec_encode_params params) {
    (void) out_tokens; (void) params;

    codec_pocket_mimi & m = *static_cast<codec_pocket_mimi *>(ctx->model->impl);
    if (!m.has_encoder) {
        codec_context_set_error(ctx, "model metadata indicates no encoder");
        return CODEC_STATUS_INVALID_STATE;
    }
    if (out_latent == nullptr || pcm.empty()) {
        codec_context_set_error(ctx, "invalid encode input/output");
        return CODEC_STATUS_INVALID_ARG;
    }

    // pad_for_conv1d: pad to a multiple of frame_size (hop) so the whole final
    // window is covered (matches MimiModel.encode_to_latent).
    const int32_t hop = m.hop_size > 0 ? m.hop_size : 1;
    const int32_t n_in = (int32_t) pcm.size();
    const int32_t n_pcm = ((n_in + hop - 1) / hop) * hop;
    std::vector<float> pad((size_t) n_pcm, 0.0f);
    std::memcpy(pad.data(), pcm.data(), pcm.size() * sizeof(float));
    const int32_t n_frames = n_pcm / hop;

    pm_encode_build build = {};
    build.n_pcm = n_pcm;
    build.cfg = m;
    build.model = ctx->model;

    codec_graph_eval_guard guard(ctx);
    std::string err;
    codec_graph_cache_entry * entry = nullptr;
    if (!codec_graph_cache_get_or_build(
            ctx,
            { CODEC_GRAPH_POCKET_MIMI_ENCODE, /*n_frames=*/n_frames, /*n_q=*/0,
              /*hop=*/hop, /*n_in=*/n_pcm, /*latent_dim=*/m.latent_dim },
            pm_build_encode, &build, sizeof(build), &entry, &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }
    lm_ggml_tensor * t_pcm = codec_graph_get_tensor(ctx, entry, pm_name_pcm());
    lm_ggml_tensor * t_mu = codec_graph_get_tensor(ctx, entry, pm_name_mu());
    if (t_pcm == nullptr || t_mu == nullptr) {
        codec_context_set_error(ctx, "cached Pocket-Mimi encode graph is invalid");
        return CODEC_STATUS_INTERNAL_ERROR;
    }
    if (!codec_graph_prepare_io(ctx, entry, &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }
    if (!codec_runtime_write_tensor(t_pcm, pad.data(), pad.size() * sizeof(float), &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }
    const int32_t nth = ctx->model->n_threads > 0 ? ctx->model->n_threads : 1;
    if (!codec_graph_compute(ctx, entry, nth, &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }
    // mu ne=(t_lat, latent_dim) → raw buffer is channel-major [latent_dim, t_lat].
    const int32_t t_lat = (int32_t) t_mu->ne[0];
    const int32_t ldim = (int32_t) t_mu->ne[1];
    float * data = static_cast<float *>(std::malloc((size_t) t_lat * (size_t) ldim * sizeof(float)));
    if (data == nullptr) { codec_context_set_error(ctx, "oom"); return CODEC_STATUS_INTERNAL_ERROR; }
    if (!codec_runtime_read_tensor(t_mu, data, (size_t) t_lat * (size_t) ldim * sizeof(float), &err)) {
        std::free(data);
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }
    codec_latent_buffer_reset(out_latent);
    out_latent->data = data;
    out_latent->latent_dim = ldim;
    out_latent->n_frames = t_lat;
    out_latent->sample_rate = m.sample_rate;
    out_latent->hop_size = hop;
    return CODEC_STATUS_SUCCESS;
}

// ---------------------------------------------------------------------
// init + vtable
// ---------------------------------------------------------------------

enum codec_status codec_pocket_mimi_init(struct codec_model * model) {
    if (model == nullptr || model->impl == nullptr || model->gguf == nullptr) {
        return CODEC_STATUS_INVALID_ARG;
    }
    codec_pocket_mimi & m = *static_cast<codec_pocket_mimi *>(model->impl);

    m.sample_rate   = codec_read_i32_kv(model->gguf, "codec.sample_rate", m.sample_rate);
    m.hop_size      = codec_read_i32_kv(model->gguf, "codec.hop_size", m.hop_size);
    m.latent_dim    = codec_read_i32_kv(model->gguf, "codec.latent_dim", m.latent_dim);
    m.seanet_dim    = codec_read_i32_kv(model->gguf, "pocket_mimi.seanet_dim", m.seanet_dim);
    m.inner_dim     = codec_read_i32_kv(model->gguf, "pocket_mimi.inner_dim", m.inner_dim);
    m.outer_dim     = codec_read_i32_kv(model->gguf, "pocket_mimi.outer_dim", m.outer_dim);
    m.quantizer_dim = codec_read_i32_kv(model->gguf, "pocket_mimi.quantizer_dim", m.quantizer_dim);
    m.tf_layers     = codec_read_i32_kv(model->gguf, "pocket_mimi.tf_layers", m.tf_layers);
    m.tf_heads      = codec_read_i32_kv(model->gguf, "pocket_mimi.tf_heads", m.tf_heads);
    m.tf_head_dim   = codec_read_i32_kv(model->gguf, "pocket_mimi.tf_head_dim", m.tf_head_dim);
    m.tf_ffn        = codec_read_i32_kv(model->gguf, "pocket_mimi.tf_ffn", m.tf_ffn);
    m.tf_context    = codec_read_i32_kv(model->gguf, "pocket_mimi.tf_context", m.tf_context);
    m.tf_max_period = codec_read_f32_kv(model->gguf, "pocket_mimi.tf_max_period", m.tf_max_period);
    m.has_encoder   = codec_read_bool_kv(model->gguf, "codec.has_encoder", m.has_encoder);
    m.has_decoder   = codec_read_bool_kv(model->gguf, "codec.has_decoder", m.has_decoder);
    codec_read_i32_array_kv(model->gguf, "pocket_mimi.decoder_ratios", m.decoder_ratios, 3);
    codec_read_i32_array_kv(model->gguf, "pocket_mimi.encoder_ratios", m.encoder_ratios, 3);

    // resample stride = encoder_frame_rate / frame_rate.  encoder_frame_rate =
    // sample_rate / prod(ratios); frame_rate = sample_rate / hop.  So stride =
    // hop / prod(ratios).
    int32_t prod = 1;
    for (int i = 0; i < m.n_ratios; ++i) prod *= (m.decoder_ratios[i] > 0 ? m.decoder_ratios[i] : 1);
    m.resample_stride = prod > 0 ? m.hop_size / prod : 16;
    m.resample_kernel = 2 * m.resample_stride;

    model->sample_rate = m.sample_rate;
    model->encode_sample_rate = m.sample_rate;
    model->has_encoder = m.has_encoder;
    model->has_decoder = m.has_decoder;
    model->hop_size = m.hop_size;
    model->n_q = 0;
    model->latent_dim = m.latent_dim;
    return CODEC_STATUS_SUCCESS;
}

static void * codec_pocket_mimi_create_impl() {
    return new (std::nothrow) codec_pocket_mimi();
}
static void codec_pocket_mimi_destroy_impl(void * ptr) {
    delete static_cast<codec_pocket_mimi *>(ptr);
}

const struct codec_model_vtable * codec_pocket_mimi_vtable() {
    static const codec_model_vtable vtable = {
        CODEC_ARCH_POCKET_MIMI,
        "pocket_mimi",
        codec_pocket_mimi_create_impl,
        codec_pocket_mimi_destroy_impl,
        codec_pocket_mimi_init,
        codec_graph_size_exact,
        codec_pocket_mimi_encode,
        /*decode=*/nullptr,
        codec_pocket_mimi_decode_latent,
    };
    return &vtable;
}
