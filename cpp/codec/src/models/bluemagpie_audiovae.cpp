#include "bluemagpie_audiovae.h"

#include "../ops/conv1d.h"
#include "../ops/convtr1d.h"
#include "../ops/lm_ggml_ops.h"
#include "../runtime/graph.h"
#include "../runtime/lm_gguf_kv.h"
#include "../runtime/tensor_utils.h"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <new>
#include <string>
#include <vector>

// =====================================================================
// BlueMagpie / VoxCPM2 AudioVAE V2 — continuous-latent decoder.
//
// CausalDecoder (depthwise=True, cond_type="scale_bias"):
//   model.0   depthwise WNCausalConv1d(64, 64, k=7, groups=64)
//   model.1   pointwise WNCausalConv1d(64, 2048, k=1)
//   model.2-7 6 CausalDecoderBlocks (rates [8,6,5,2,2,2]):
//               sr_cond (x = x*scale + bias, 48 kHz row baked at convert)
//               Snake → WNCausalTransposeConv1d(in,out,k=2*stride) →
//               3 CausalResidualUnits (dilations 1/3/9)
//   model.8   Snake1d(32)
//   model.9   WNCausalConv1d(32, 1, k=7)
//   Tanh
//
// All convs are causal (left-pad by (k-1)*dilation, length preserved); the
// transpose conv runs raw then right-trims `2*ceil(s/2) - (s%2)` (= stride),
// upsampling by exactly `stride`.  Snake is sign-preserving (alphas can be
// negative) — NOT codec_op_snake.  NoiseBlock is off (deterministic decode).
// =====================================================================

namespace {

const char * codec_bm_name_lat()   { return "bluemagpie.audiovae.lat"; }
const char * codec_bm_name_audio() { return "bluemagpie.audiovae.audio"; }

// Sign-preserving Snake:  y = x + (1 / (alpha + 1e-9)) * sin(alpha * x)^2.
lm_ggml_tensor * codec_bm_snake_tc(lm_ggml_context * ctx, lm_ggml_tensor * x_tc, lm_ggml_tensor * alpha) {
    if (ctx == nullptr || x_tc == nullptr || alpha == nullptr) return nullptr;
    alpha = codec_graph_cast_f32(ctx, alpha);
    lm_ggml_tensor * a_2d = lm_ggml_reshape_2d(ctx, alpha, 1, x_tc->ne[1]);
    lm_ggml_tensor * a_rep = lm_ggml_repeat(ctx, a_2d, x_tc);              // [t, c]
    lm_ggml_tensor * a_eps = lm_ggml_scale_bias(ctx, a_rep, 1.0f, 1e-9f);  // alpha + 1e-9
    lm_ggml_tensor * ax = lm_ggml_mul(ctx, a_rep, x_tc);
    lm_ggml_tensor * s = lm_ggml_sin(ctx, ax);
    lm_ggml_tensor * s2 = lm_ggml_mul(ctx, s, s);
    lm_ggml_tensor * frac = lm_ggml_div(ctx, s2, a_eps);
    return lm_ggml_add(ctx, x_tc, frac);
}

// Per-channel affine:  x = x * scale[c] + bias[c]  (x_tc: ne=(t, c)).
lm_ggml_tensor * codec_bm_affine_tc(lm_ggml_context * ctx, lm_ggml_tensor * x_tc,
                                 lm_ggml_tensor * scale, lm_ggml_tensor * bias) {
    if (ctx == nullptr || x_tc == nullptr || scale == nullptr || bias == nullptr) return nullptr;
    scale = codec_graph_cast_f32(ctx, scale);
    bias = codec_graph_cast_f32(ctx, bias);
    lm_ggml_tensor * s_2d = lm_ggml_reshape_2d(ctx, scale, 1, x_tc->ne[1]);  // [1, c] — broadcasts over t
    lm_ggml_tensor * b_2d = lm_ggml_reshape_2d(ctx, bias, 1, x_tc->ne[1]);
    x_tc = lm_ggml_mul(ctx, x_tc, s_2d);
    return lm_ggml_add(ctx, x_tc, b_2d);
}

// CausalResidualUnit: y = x + Conv1×1(Snake(Conv7-dilated-depthwise-causal(Snake(x)))).
// Causal convs preserve length, so the residual add needs no crop.
lm_ggml_tensor * codec_bm_residual_unit_tc(
    lm_ggml_context * ctx, lm_ggml_tensor * x_tc,
    lm_ggml_tensor * a1, lm_ggml_tensor * c1_w, lm_ggml_tensor * c1_b,
    lm_ggml_tensor * a2, lm_ggml_tensor * c2_w, lm_ggml_tensor * c2_b,
    int32_t dilation) {

    if (ctx == nullptr || x_tc == nullptr) return nullptr;
    lm_ggml_tensor * h = codec_bm_snake_tc(ctx, x_tc, a1);
    if (h == nullptr) return nullptr;
    h = codec_conv1d_depthwise_causal(ctx, h, c1_w, c1_b, /*stride=*/1, dilation);  // k=7
    if (h == nullptr) return nullptr;
    h = codec_bm_snake_tc(ctx, h, a2);
    if (h == nullptr) return nullptr;
    h = codec_conv1d(ctx, h, c2_w, c2_b, /*stride=*/1, /*dilation=*/1, /*padding=*/0);  // k=1
    if (h == nullptr) return nullptr;
    return lm_ggml_add(ctx, x_tc, h);
}

lm_ggml_tensor * codec_bm_decoder_block(
    lm_ggml_context * ctx_eval, lm_ggml_tensor * x_tc, int32_t bi, int32_t stride,
    const codec_model * model) {

    auto W = [&](const std::string & name) -> lm_ggml_tensor * {
        return codec_graph_weight(ctx_eval, model, name);
    };
    const std::string base = "bluemagpie.dec.b" + std::to_string(bi);

    // sr_cond (scale_bias) applied to the block input.
    x_tc = codec_bm_affine_tc(ctx_eval, x_tc, W(base + ".cond.scale"), W(base + ".cond.bias"));
    if (x_tc == nullptr) return nullptr;

    x_tc = codec_bm_snake_tc(ctx_eval, x_tc, W(base + ".act.alpha"));
    if (x_tc == nullptr) return nullptr;

    // CausalTransposeConv1d: raw convtr (padding=0) then right-trim by
    // 2*ceil(stride/2) - (stride%2)  (= stride here), upsampling by `stride`.
    lm_ggml_tensor * y = codec_convtr1d(ctx_eval, x_tc, W(base + ".convtr.w"), W(base + ".convtr.b"),
                                     /*stride=*/stride, /*padding=*/0, /*dilation=*/1);
    if (y == nullptr) return nullptr;
    const int32_t crop_right = 2 * ((stride + 1) / 2) - (stride % 2);
    if (crop_right > 0) {
        y = codec_op_crop_1d(ctx_eval, y, /*crop_left=*/0, /*crop_right=*/crop_right);
        if (y == nullptr) return nullptr;
    }
    x_tc = y;

    const int32_t dilations[3] = { 1, 3, 9 };
    for (int32_t ri = 0; ri < 3; ++ri) {
        const std::string r = base + ".r" + std::to_string(ri);
        x_tc = codec_bm_residual_unit_tc(
            ctx_eval, x_tc,
            W(r + ".act1.alpha"), W(r + ".conv1.w"), W(r + ".conv1.b"),
            W(r + ".act2.alpha"), W(r + ".conv2.w"), W(r + ".conv2.b"),
            dilations[ri]);
        if (x_tc == nullptr) return nullptr;
    }
    return x_tc;
}

struct bm_decode_build {
    int32_t n_frames   = 0;
    int32_t latent_dim = 0;
    int32_t n_blocks   = 0;
    int32_t decoder_rates[6] = { 0, 0, 0, 0, 0, 0 };
    const codec_model * model = nullptr;
};

bool codec_bm_build_decode(lm_ggml_context * ctx_eval, void * user_data, lm_ggml_tensor ** out) {
    bm_decode_build * p = static_cast<bm_decode_build *>(user_data);
    if (ctx_eval == nullptr || p == nullptr || out == nullptr || p->model == nullptr) return false;
    if (p->n_frames <= 0 || p->latent_dim <= 0) return false;

    auto W = [&](const std::string & name) -> lm_ggml_tensor * {
        return codec_graph_weight(ctx_eval, p->model, name);
    };

    // Latent input: ne=(t, latent_dim); buffer is [latent_dim, n_frames]
    // row-major (= PyTorch z[D, T]) — matches codec_decode_quantized_representation.
    lm_ggml_tensor * t_lat = lm_ggml_new_tensor_2d(ctx_eval, LM_GGML_TYPE_F32, p->n_frames, p->latent_dim);
    lm_ggml_set_name(t_lat, codec_bm_name_lat());

    // Input convs: depthwise causal (k=7) → pointwise (k=1).
    lm_ggml_tensor * x_tc = codec_conv1d_depthwise_causal(ctx_eval, t_lat,
                                                       W("bluemagpie.dec.conv_in_dw.w"),
                                                       W("bluemagpie.dec.conv_in_dw.b"),
                                                       /*stride=*/1, /*dilation=*/1);
    if (x_tc == nullptr) return false;
    x_tc = codec_conv1d(ctx_eval, x_tc,
                        W("bluemagpie.dec.conv_in_pw.w"), W("bluemagpie.dec.conv_in_pw.b"),
                        /*stride=*/1, /*dilation=*/1, /*padding=*/0);
    if (x_tc == nullptr) return false;

    for (int32_t bi = 0; bi < p->n_blocks; ++bi) {
        x_tc = codec_bm_decoder_block(ctx_eval, x_tc, bi, p->decoder_rates[bi], p->model);
        if (x_tc == nullptr) return false;
    }

    // Final Snake → WNCausalConv(32→1, k=7) → Tanh.
    x_tc = codec_bm_snake_tc(ctx_eval, x_tc, W("bluemagpie.dec.act_final.alpha"));
    if (x_tc == nullptr) return false;
    x_tc = codec_conv1d_causal(ctx_eval, x_tc,
                               W("bluemagpie.dec.conv_out.w"), W("bluemagpie.dec.conv_out.b"),
                               /*stride=*/1, /*dilation=*/1);
    if (x_tc == nullptr) return false;
    x_tc = lm_ggml_tanh(ctx_eval, x_tc);
    lm_ggml_set_name(x_tc, codec_bm_name_audio());
    *out = x_tc;
    return true;
}

// CausalEncoderBlock: 3 residual units (dilation 1/3/9) → Snake → strided
// causal downsample conv (full, not depthwise).
lm_ggml_tensor * codec_bm_encoder_block(
    lm_ggml_context * ctx_eval, lm_ggml_tensor * x_tc, int32_t bi, int32_t stride,
    const codec_model * model) {

    auto W = [&](const std::string & name) -> lm_ggml_tensor * {
        return codec_graph_weight(ctx_eval, model, name);
    };
    const std::string base = "bluemagpie.enc.b" + std::to_string(bi);
    const int32_t dilations[3] = { 1, 3, 9 };
    for (int32_t ri = 0; ri < 3; ++ri) {
        const std::string r = base + ".r" + std::to_string(ri);
        x_tc = codec_bm_residual_unit_tc(
            ctx_eval, x_tc,
            W(r + ".act1.alpha"), W(r + ".conv1.w"), W(r + ".conv1.b"),
            W(r + ".act2.alpha"), W(r + ".conv2.w"), W(r + ".conv2.b"),
            dilations[ri]);
        if (x_tc == nullptr) return nullptr;
    }
    x_tc = codec_bm_snake_tc(ctx_eval, x_tc, W(base + ".act.alpha"));
    if (x_tc == nullptr) return nullptr;
    // Strided causal downsample (CausalConv1d, full conv).
    return codec_conv1d_causal(ctx_eval, x_tc, W(base + ".down.w"), W(base + ".down.b"),
                               /*stride=*/stride, /*dilation=*/1);
}

struct bm_encode_build {
    int32_t n_pcm;
    int32_t latent_dim;
    int32_t n_blocks;
    int32_t encoder_rates[4];
    const codec_model * model;
};

bool codec_bm_build_encode(lm_ggml_context * ctx, void * ud, lm_ggml_tensor ** out) {
    bm_encode_build * p = static_cast<bm_encode_build *>(ud);
    if (ctx == nullptr || p == nullptr || out == nullptr || p->model == nullptr || p->n_pcm <= 0) return false;
    auto W = [&](const std::string & name) -> lm_ggml_tensor * { return codec_graph_weight(ctx, p->model, name); };

    lm_ggml_tensor * t_pcm = lm_ggml_new_tensor_2d(ctx, LM_GGML_TYPE_F32, p->n_pcm, 1);   // (n_pcm, 1)
    lm_ggml_set_name(t_pcm, "bluemagpie.audiovae.pcm");

    // conv0: full causal conv 1 → encoder_dim (k=7).
    lm_ggml_tensor * x = codec_conv1d_causal(ctx, t_pcm, W("bluemagpie.enc.conv0.w"), W("bluemagpie.enc.conv0.b"),
                                          /*stride=*/1, /*dilation=*/1);
    if (x == nullptr) return false;
    for (int32_t bi = 0; bi < p->n_blocks; ++bi) {
        x = codec_bm_encoder_block(ctx, x, bi + 1, p->encoder_rates[bi], p->model);
        if (x == nullptr) return false;
    }
    // fc_mu: full causal conv → latent_dim (k=3).
    x = codec_conv1d_causal(ctx, x, W("bluemagpie.enc.fc_mu.w"), W("bluemagpie.enc.fc_mu.b"),
                            /*stride=*/1, /*dilation=*/1);
    if (x == nullptr) return false;
    lm_ggml_set_name(x, "bluemagpie.audiovae.mu");   // ne=(t_lat, latent_dim) → channel-major buffer
    *out = x;
    return true;
}

}  // namespace

// ---------------------------------------------------------------------
// decode_latent entry
// ---------------------------------------------------------------------

static enum codec_status codec_bluemagpie_audiovae_decode_latent(
    struct codec_context * ctx,
    const float * quantized_representation,
    int32_t latent_dim,
    int32_t n_frames,
    struct codec_pcm_buffer * out_pcm,
    struct codec_decode_params params) {
    (void) params;

    codec_bluemagpie_audiovae & vae = *static_cast<codec_bluemagpie_audiovae *>(ctx->model->impl);
    if (!vae.has_decoder) {
        codec_context_set_error(ctx, "model metadata indicates no decoder");
        return CODEC_STATUS_INVALID_STATE;
    }
    if (quantized_representation == nullptr || n_frames <= 0) {
        codec_context_set_error(ctx, "invalid latent input");
        return CODEC_STATUS_INVALID_ARG;
    }
    if (latent_dim != vae.latent_dim) {
        codec_context_set_error(ctx, "latent_dim mismatch with model");
        return CODEC_STATUS_INVALID_ARG;
    }

    bm_decode_build build = {};
    build.n_frames   = n_frames;
    build.latent_dim = latent_dim;
    build.n_blocks   = vae.n_blocks;
    for (int i = 0; i < 6; ++i) build.decoder_rates[i] = vae.decoder_rates[i];
    build.model = ctx->model;

    codec_graph_eval_guard guard(ctx);
    std::string err;

    codec_graph_cache_entry * entry = nullptr;
    if (!codec_graph_cache_get_or_build(
            ctx,
            { CODEC_GRAPH_BLUEMAGPIE_AUDIOVAE_DECODE,
              /*n_frames=*/n_frames, /*n_q=*/0, /*hop=*/vae.decode_hop,
              /*n_in=*/0, /*latent_dim=*/latent_dim },
            codec_bm_build_decode, &build, sizeof(build), &entry, &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    lm_ggml_tensor * t_lat = codec_graph_get_tensor(ctx, entry, codec_bm_name_lat());
    lm_ggml_tensor * t_out = codec_graph_get_tensor(ctx, entry, codec_bm_name_audio());
    if (t_lat == nullptr || t_out == nullptr) {
        codec_context_set_error(ctx, "cached BlueMagpie AudioVAE graph is invalid");
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
        codec_context_set_error(ctx, "unexpected BlueMagpie AudioVAE output shape/type");
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
    out_pcm->data        = pcm;
    out_pcm->n_samples   = n_samples;
    out_pcm->sample_rate = vae.sample_rate;
    out_pcm->n_channels  = 1;
    return CODEC_STATUS_SUCCESS;
}

// ---------------------------------------------------------------------
// encode entry (audio → latent mu)
// ---------------------------------------------------------------------

static enum codec_status codec_bluemagpie_audiovae_encode(
    struct codec_context * ctx,
    const std::vector<float> & pcm,
    struct codec_token_buffer * out_tokens,
    struct codec_latent_buffer * out_latent,
    struct codec_encode_params params) {
    (void) out_tokens; (void) params;

    codec_bluemagpie_audiovae & vae = *static_cast<codec_bluemagpie_audiovae *>(ctx->model->impl);
    if (!vae.has_encoder) {
        codec_context_set_error(ctx, "model metadata indicates no encoder");
        return CODEC_STATUS_INVALID_STATE;
    }
    if (out_latent == nullptr || pcm.empty()) {
        codec_context_set_error(ctx, "invalid encode input/output");
        return CODEC_STATUS_INVALID_ARG;
    }

    // Pad to a multiple of the encoder hop (right pad, matches AudioVAE.preprocess).
    const int32_t hop = vae.encode_hop > 0 ? vae.encode_hop : 1;
    const int32_t n_in = (int32_t) pcm.size();
    const int32_t n_pcm = ((n_in + hop - 1) / hop) * hop;
    std::vector<float> pad((size_t) n_pcm, 0.0f);
    std::memcpy(pad.data(), pcm.data(), pcm.size() * sizeof(float));
    const int32_t n_frames = n_pcm / hop;

    bm_encode_build build = {};
    build.n_pcm = n_pcm;
    build.latent_dim = vae.latent_dim;
    build.n_blocks = vae.n_enc_blocks;
    for (int i = 0; i < 4; ++i) build.encoder_rates[i] = vae.encoder_rates[i];
    build.model = ctx->model;

    codec_graph_eval_guard guard(ctx);
    std::string err;
    codec_graph_cache_entry * entry = nullptr;
    if (!codec_graph_cache_get_or_build(
            ctx,
            { CODEC_GRAPH_BLUEMAGPIE_AUDIOVAE_ENCODE,
              /*n_frames=*/n_frames, /*n_q=*/0, /*hop=*/hop, /*n_in=*/n_pcm, /*latent_dim=*/vae.latent_dim },
            codec_bm_build_encode, &build, sizeof(build), &entry, &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }
    lm_ggml_tensor * t_pcm = codec_graph_get_tensor(ctx, entry, "bluemagpie.audiovae.pcm");
    lm_ggml_tensor * t_mu = codec_graph_get_tensor(ctx, entry, "bluemagpie.audiovae.mu");
    if (t_pcm == nullptr || t_mu == nullptr) {
        codec_context_set_error(ctx, "cached BlueMagpie encode graph is invalid");
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
    // mu ne=(t_lat, latent_dim) → raw buffer is channel-major [latent_dim, t_lat]
    // (buffer[d*t_lat + t]), matching codec_decode_quantized_representation input.
    const int32_t t_lat = (int32_t) t_mu->ne[0];
    const int32_t ldim  = (int32_t) t_mu->ne[1];
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
    out_latent->sample_rate = vae.encode_sample_rate;
    out_latent->hop_size = hop;
    return CODEC_STATUS_SUCCESS;
}

// ---------------------------------------------------------------------
// init + vtable
// ---------------------------------------------------------------------

enum codec_status codec_bluemagpie_audiovae_init(struct codec_model * model) {
    if (model == nullptr || model->impl == nullptr || model->gguf == nullptr) {
        return CODEC_STATUS_INVALID_ARG;
    }
    codec_bluemagpie_audiovae & vae = *static_cast<codec_bluemagpie_audiovae *>(model->impl);

    vae.sample_rate        = codec_read_i32_kv(model->gguf, "codec.sample_rate", vae.sample_rate);
    vae.encode_sample_rate = codec_read_i32_kv(model->gguf, "codec.encode_sample_rate", vae.encode_sample_rate);
    vae.latent_dim         = codec_read_i32_kv(model->gguf, "codec.latent_dim", vae.latent_dim);
    vae.decode_hop         = codec_read_i32_kv(model->gguf, "codec.decode_hop_size", vae.decode_hop);
    vae.decoder_dim        = codec_read_i32_kv(model->gguf, "bluemagpie.decoder_dim", vae.decoder_dim);
    vae.has_encoder        = codec_read_bool_kv(model->gguf, "codec.has_encoder", vae.has_encoder);
    vae.has_decoder        = codec_read_bool_kv(model->gguf, "codec.has_decoder", vae.has_decoder);
    vae.encode_hop         = codec_read_i32_kv(model->gguf, "codec.hop_size", vae.encode_hop);
    vae.encoder_dim        = codec_read_i32_kv(model->gguf, "bluemagpie.encoder_dim", vae.encoder_dim);
    codec_read_i32_array_kv(model->gguf, "bluemagpie.decoder_rates", vae.decoder_rates, 6);
    codec_read_i32_array_kv(model->gguf, "bluemagpie.encoder_rates", vae.encoder_rates, 4);

    // Count actual decoder blocks from the rates array (trailing zeros = unused).
    int32_t nb = 0;
    for (int i = 0; i < 6; ++i) if (vae.decoder_rates[i] > 0) nb = i + 1;
    vae.n_blocks = nb > 0 ? nb : 6;
    int32_t neb = 0;
    for (int i = 0; i < 4; ++i) if (vae.encoder_rates[i] > 0) neb = i + 1;
    vae.n_enc_blocks = neb > 0 ? neb : 4;

    model->sample_rate        = vae.sample_rate;
    model->encode_sample_rate = vae.encode_sample_rate;
    model->has_encoder        = vae.has_encoder;
    model->has_decoder        = vae.has_decoder;
    model->hop_size           = vae.decode_hop;
    model->n_q                = 0;
    model->latent_dim         = vae.latent_dim;
    return CODEC_STATUS_SUCCESS;
}

static void * codec_bluemagpie_audiovae_create_impl() {
    return new (std::nothrow) codec_bluemagpie_audiovae();
}
static void codec_bluemagpie_audiovae_destroy_impl(void * ptr) {
    delete static_cast<codec_bluemagpie_audiovae *>(ptr);
}

const struct codec_model_vtable * codec_bluemagpie_audiovae_vtable() {
    static const codec_model_vtable vtable = {
        CODEC_ARCH_BLUEMAGPIE_AUDIOVAE,
        "bluemagpie_audiovae",
        codec_bluemagpie_audiovae_create_impl,
        codec_bluemagpie_audiovae_destroy_impl,
        codec_bluemagpie_audiovae_init,
        codec_graph_size_exact,
        codec_bluemagpie_audiovae_encode,
        /*decode=*/nullptr,
        codec_bluemagpie_audiovae_decode_latent,
    };
    return &vtable;
}
