#include "soprano.h"

#include "../ops/conv1d.h"
#include "../ops/lm_ggml_ops.h"
#include "../runtime/audio_dsp.h"
#include "../runtime/graph.h"
#include "../runtime/tensor_utils.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <new>
#include <string>
#include <vector>

static std::string codec_sop_name_embed_w() { return "sop.decode.embed.w"; }
static std::string codec_sop_name_embed_b() { return "sop.decode.embed.b"; }
static std::string codec_sop_name_norm_w() { return "sop.decode.norm.w"; }
static std::string codec_sop_name_norm_b() { return "sop.decode.norm.b"; }
static std::string codec_sop_name_fln_w() { return "sop.decode.fln.w"; }
static std::string codec_sop_name_fln_b() { return "sop.decode.fln.b"; }
static std::string codec_sop_name_head_w() { return "sop.decode.head.out.w"; }
static std::string codec_sop_name_head_b() { return "sop.decode.head.out.b"; }
static std::string codec_sop_name_istft_window() { return "sop.decode.istft.window"; }

static std::string codec_sop_name_cnx_dw_w(int32_t li) { return "sop.decode.cnx." + std::to_string(li) + ".dw.w"; }
static std::string codec_sop_name_cnx_dw_b(int32_t li) { return "sop.decode.cnx." + std::to_string(li) + ".dw.b"; }
static std::string codec_sop_name_cnx_ln_w(int32_t li) { return "sop.decode.cnx." + std::to_string(li) + ".ln.w"; }
static std::string codec_sop_name_cnx_ln_b(int32_t li) { return "sop.decode.cnx." + std::to_string(li) + ".ln.b"; }
static std::string codec_sop_name_cnx_pw1_w(int32_t li) { return "sop.decode.cnx." + std::to_string(li) + ".pw1.w"; }
static std::string codec_sop_name_cnx_pw1_b(int32_t li) { return "sop.decode.cnx." + std::to_string(li) + ".pw1.b"; }
static std::string codec_sop_name_cnx_pw2_w(int32_t li) { return "sop.decode.cnx." + std::to_string(li) + ".pw2.w"; }
static std::string codec_sop_name_cnx_pw2_b(int32_t li) { return "sop.decode.cnx." + std::to_string(li) + ".pw2.b"; }
static std::string codec_sop_name_cnx_gamma(int32_t li) { return "sop.decode.cnx." + std::to_string(li) + ".gamma"; }


struct sop_decode_build {
    int32_t t = 0;
    int32_t in_ch = 0;
    int32_t dim = 0;
    int32_t intermediate = 0;
    int32_t n_layers = 0;
    int32_t dw_kernel = 0;
    int32_t head_out_dim = 0;
    const codec_model * model = nullptr;
};

static bool codec_sop_build_decode(lm_ggml_context * ctx_eval, void * user_data, lm_ggml_tensor ** out) {
    sop_decode_build * p = static_cast<sop_decode_build *>(user_data);
    if (ctx_eval == nullptr || p == nullptr || out == nullptr || p->model == nullptr) {
        return false;
    }
    if (p->t <= 0 || p->in_ch <= 0 || p->dim <= 0 || p->intermediate <= 0 || p->n_layers <= 0 || p->dw_kernel <= 0 || p->head_out_dim <= 0) {
        return false;
    }

    auto W = [&](const std::string & name) -> lm_ggml_tensor * {
        return codec_graph_weight(ctx_eval, p->model, name);
    };

    lm_ggml_tensor * t_in = lm_ggml_new_tensor_2d(ctx_eval, LM_GGML_TYPE_F32, p->t, p->in_ch);
    lm_ggml_set_name(t_in, "sop.decode.in");

    lm_ggml_tensor * t_emb_w = W(codec_sop_name_embed_w());
    lm_ggml_tensor * t_emb_b = W(codec_sop_name_embed_b());
    if (t_emb_w == nullptr || t_emb_b == nullptr) {
        return false;
    }
    lm_ggml_tensor * x = codec_conv1d(ctx_eval, t_in, t_emb_w, t_emb_b, 1, 1, 0); // [t, dim]
    if (x == nullptr) {
        return false;
    }

    lm_ggml_tensor * t_norm_w = W(codec_sop_name_norm_w());
    lm_ggml_tensor * t_norm_b = W(codec_sop_name_norm_b());
    if (t_norm_w == nullptr || t_norm_b == nullptr) {
        return false;
    }

    lm_ggml_tensor * x_ct = lm_ggml_cont(ctx_eval, lm_ggml_transpose(ctx_eval, x)); // [c, t]
    lm_ggml_set_name(x_ct, "sop.stage.embed.ct");
    x_ct = codec_op_layer_norm_ct(ctx_eval, x_ct, 1e-6f, t_norm_w, t_norm_b);
    if (x_ct == nullptr) {
        return false;
    }
    lm_ggml_set_name(x_ct, "sop.stage.norm.ct");

    const int32_t pad = p->dw_kernel / 2;
    for (int32_t li = 0; li < p->n_layers; ++li) {
        lm_ggml_tensor * t_dw_w = W(codec_sop_name_cnx_dw_w(li));
        lm_ggml_tensor * t_dw_b = W(codec_sop_name_cnx_dw_b(li));
        lm_ggml_tensor * t_ln_w = W(codec_sop_name_cnx_ln_w(li));
        lm_ggml_tensor * t_ln_b = W(codec_sop_name_cnx_ln_b(li));
        lm_ggml_tensor * t_pw1_w = W(codec_sop_name_cnx_pw1_w(li));
        lm_ggml_tensor * t_pw1_b = W(codec_sop_name_cnx_pw1_b(li));
        lm_ggml_tensor * t_pw2_w = W(codec_sop_name_cnx_pw2_w(li));
        lm_ggml_tensor * t_pw2_b = W(codec_sop_name_cnx_pw2_b(li));
        lm_ggml_tensor * t_gamma = W(codec_sop_name_cnx_gamma(li));
        if (t_dw_w == nullptr || t_dw_b == nullptr || t_ln_w == nullptr || t_ln_b == nullptr ||
            t_pw1_w == nullptr || t_pw1_b == nullptr || t_pw2_w == nullptr || t_pw2_b == nullptr ||
            t_gamma == nullptr) {
            return false;
        }
        x_ct = codec_op_convnext_block_ct(
            ctx_eval, x_ct, t_dw_w, t_dw_b, t_ln_w, t_ln_b,
            t_pw1_w, t_pw1_b, t_pw2_w, t_pw2_b, t_gamma, pad);
        if (x_ct == nullptr) {
            return false;
        }
        lm_ggml_set_name(x_ct, ("sop.stage.block." + std::to_string(li) + ".ct").c_str());
    }

    lm_ggml_tensor * t_fln_w = W(codec_sop_name_fln_w());
    lm_ggml_tensor * t_fln_b = W(codec_sop_name_fln_b());
    if (t_fln_w == nullptr || t_fln_b == nullptr) {
        return false;
    }
    x_ct = codec_op_layer_norm_ct(ctx_eval, x_ct, 1e-6f, t_fln_w, t_fln_b);
    if (x_ct == nullptr) {
        return false;
    }
    lm_ggml_set_name(x_ct, "sop.stage.final.ct");

    lm_ggml_tensor * t_head_w = W(codec_sop_name_head_w());
    lm_ggml_tensor * t_head_b = W(codec_sop_name_head_b());
    if (t_head_w == nullptr || t_head_b == nullptr) {
        return false;
    }
    lm_ggml_tensor * t_head = codec_op_linear(ctx_eval, x_ct, t_head_w, t_head_b); // [out_dim, t]
    if (t_head == nullptr) {
        return false;
    }
    lm_ggml_tensor * t_out = lm_ggml_cont(ctx_eval, t_head);
    lm_ggml_set_name(t_out, "sop.decode.head.out");
    *out = t_out;
    return true;
}

enum codec_status codec_soprano_init(struct codec_model * model) {
    if (model == nullptr || model->gguf == nullptr) {
        return CODEC_STATUS_INVALID_ARG;
    }

    codec_soprano & sop = *static_cast<codec_soprano *>(model->impl);

    sop.sample_rate = codec_read_i32_kv(model->gguf, "codec.sample_rate", sop.sample_rate);
    sop.hop_size = codec_read_i32_kv(model->gguf, "codec.hop_size", sop.hop_size);
    sop.n_fft = codec_read_i32_kv(model->gguf, "codec.n_fft", sop.n_fft);
    sop.win_length = codec_read_i32_kv(model->gguf, "codec.win_length", sop.win_length);
    sop.latent_dim = codec_read_i32_kv(model->gguf, "codec.latent_dim", sop.latent_dim);
    sop.decoder_dim = codec_read_i32_kv(model->gguf, "soprano.decoder_dim", sop.decoder_dim);
    sop.intermediate_dim = codec_read_i32_kv(model->gguf, "soprano.intermediate_dim", sop.intermediate_dim);
    sop.num_layers = codec_read_i32_kv(model->gguf, "soprano.num_layers", sop.num_layers);
    sop.upscale = codec_read_i32_kv(model->gguf, "soprano.upscale", sop.upscale);
    sop.dw_kernel = codec_read_i32_kv(model->gguf, "soprano.dw_kernel", sop.dw_kernel);
    sop.has_encoder = codec_read_bool_kv(model->gguf, "codec.has_encoder", false);
    sop.has_decoder = codec_read_bool_kv(model->gguf, "codec.has_decoder", true);

    model->sample_rate = sop.sample_rate;
    model->hop_size = sop.hop_size;
    model->n_fft = sop.n_fft;
    model->win_length = sop.win_length;
    model->latent_dim = sop.latent_dim;
    model->has_encoder = sop.has_encoder;
    model->has_decoder = sop.has_decoder;
    model->n_q = 0;
    model->codebook_size = 0;

    return CODEC_STATUS_SUCCESS;
}

static bool codec_sop_init_decode_build(codec_context * ctx, int32_t t, sop_decode_build * build, std::string * err) {
    if (ctx == nullptr || ctx->model == nullptr || build == nullptr || t <= 0) {
        if (err != nullptr) {
            *err = "invalid Soprano decode build arguments";
        }
        return false;
    }
    const codec_soprano & sop = *static_cast<const codec_soprano *>(ctx->model->impl);
    build->t = t;
    build->in_ch = std::max(1, sop.latent_dim);
    build->dim = std::max(1, sop.decoder_dim);
    build->intermediate = std::max(1, sop.intermediate_dim);
    build->n_layers = std::max(1, sop.num_layers);
    build->dw_kernel = std::max(1, sop.dw_kernel);
    build->model = ctx->model;

    lm_ggml_tensor * head_b = codec_model_get_tensor(ctx->model, codec_sop_name_head_b());
    if (head_b == nullptr) {
        if (err != nullptr) {
            *err = "missing Soprano head bias tensor";
        }
        return false;
    }
    build->head_out_dim = (int32_t) codec_ne(head_b, 0);
    return true;
}

enum codec_status codec_soprano_decode(
    struct codec_context * ctx,
    const struct codec_token_buffer * tokens,
    struct codec_pcm_buffer * out_pcm,
    struct codec_decode_params params) {

    (void) tokens;
    (void) params;
    if (ctx == nullptr || out_pcm == nullptr) {
        return CODEC_STATUS_INVALID_ARG;
    }
    codec_context_set_error(ctx, "Soprano decoder does not accept token inputs; use decode_latent");
    return CODEC_STATUS_NOT_SUPPORTED;
}

enum codec_status codec_soprano_decode_latent(
    struct codec_context * ctx,
    const float * quantized_representation,
    int32_t latent_dim,
    int32_t n_frames,
    struct codec_pcm_buffer * out_pcm,
    struct codec_decode_params params) {

    if (ctx == nullptr || ctx->model == nullptr || out_pcm == nullptr || quantized_representation == nullptr) {
        return CODEC_STATUS_INVALID_ARG;
    }

    codec_soprano & sop = *static_cast<codec_soprano *>(ctx->model->impl);
    if (!sop.has_decoder) {
        codec_context_set_error(ctx, "model metadata indicates no decoder");
        return CODEC_STATUS_INVALID_STATE;
    }
    if (latent_dim <= 0 || n_frames <= 0) {
        codec_context_set_error(ctx, "invalid Soprano latent shape");
        return CODEC_STATUS_INVALID_ARG;
    }
    if (latent_dim != sop.latent_dim) {
        codec_context_set_error(ctx, "Soprano latent_dim mismatch");
        return CODEC_STATUS_INVALID_ARG;
    }

    const int32_t upscale = std::max(1, sop.upscale);
    const int32_t t_up = upscale * (n_frames - 1) + 1;

    std::vector<float> up((size_t) t_up * (size_t) latent_dim, 0.0f);
    for (int32_t c = 0; c < latent_dim; ++c) {
        for (int32_t ti = 0; ti < t_up; ++ti) {
            const int32_t base = std::min(n_frames - 1, ti / upscale);
            const int32_t next = std::min(n_frames - 1, base + 1);
            const float frac = (float)(ti - base * upscale) / (float) upscale;
            const float v0 = quantized_representation[(size_t) base * (size_t) latent_dim + (size_t) c];
            const float v1 = quantized_representation[(size_t) next * (size_t) latent_dim + (size_t) c];
            // ggml tensors are column-major: ne0 is contiguous.
            up[(size_t) c * (size_t) t_up + (size_t) ti] = v0 + (v1 - v0) * frac;
        }
    }

    const int32_t hop = std::max(1, sop.hop_size);
    codec_graph_eval_guard eval_guard(ctx);
    std::string err;
    sop_decode_build build = {};
    if (!codec_sop_init_decode_build(ctx, t_up, &build, &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    codec_graph_cache_entry * entry = nullptr;
    if (!codec_graph_cache_get_or_build(
            ctx,
            { CODEC_GRAPH_SOPRANO_DECODE, /*n_frames=*/t_up, /*n_q=*/0, /*hop=*/hop, /*n_in=*/0, /*latent_dim=*/latent_dim },
            codec_sop_build_decode,
            &build,
            sizeof(build),
            &entry,
            &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    lm_ggml_tensor * t_in = codec_graph_get_tensor(ctx, entry, "sop.decode.in");
    lm_ggml_tensor * t_out = codec_graph_get_tensor(ctx, entry, "sop.decode.head.out");
    if (t_in == nullptr || t_out == nullptr) {
        codec_context_set_error(ctx, "cached Soprano decode graph is invalid");
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    if (!codec_graph_prepare_io(ctx, entry, &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    if (!codec_runtime_write_tensor(t_in, up.data(), up.size() * sizeof(float), &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    const int32_t n_threads = params.n_threads > 0 ? params.n_threads : std::max(1, ctx->model->n_threads);
    if (!codec_graph_compute(ctx, entry, n_threads, &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    std::vector<float> head((size_t) build.head_out_dim * (size_t) t_up, 0.0f);
    if (!codec_runtime_read_tensor(t_out, head.data(), head.size() * sizeof(float), &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    std::vector<float> window;
    lm_ggml_tensor * w_tensor = codec_model_get_tensor(ctx->model, codec_sop_name_istft_window());
    if (w_tensor != nullptr && w_tensor->ne[1] == 1 && w_tensor->ne[2] == 1) {
        if (!codec_tensor_as_vec_f32(w_tensor, &window)) {
            window.clear();
        }
    }

    std::vector<float> pcm_v;
    const std::vector<float> * win_ptr = window.empty() ? nullptr : &window;
    if (!codec_runtime_istft_from_head(head, build.head_out_dim, t_up, hop, win_ptr, true, -1, &pcm_v, &err)) {
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
    out_pcm->sample_rate = sop.sample_rate;
    out_pcm->n_channels = 1;

    return CODEC_STATUS_SUCCESS;
}

static void * codec_sop_create_impl() {
    return new (std::nothrow) codec_soprano();
}

static void codec_sop_destroy_impl(void * impl) {
    codec_soprano * sop = static_cast<codec_soprano *>(impl);
    delete sop;
}

static enum codec_status codec_sop_init_wrap(struct codec_model * model) {
    return codec_soprano_init(model);
}

static enum codec_status codec_sop_decode_wrap(
    struct codec_context * ctx,
    const struct codec_token_buffer * tokens,
    struct codec_pcm_buffer * out_pcm,
    struct codec_decode_params params) {
    return codec_soprano_decode(ctx, tokens, out_pcm, params);
}

static enum codec_status codec_sop_decode_latent_wrap(
    struct codec_context * ctx,
    const float * quantized_representation,
    int32_t latent_dim,
    int32_t n_frames,
    struct codec_pcm_buffer * out_pcm,
    struct codec_decode_params params) {
    return codec_soprano_decode_latent(ctx, quantized_representation, latent_dim, n_frames, out_pcm, params);
}

const struct codec_model_vtable * codec_soprano_vtable() {
    static const codec_model_vtable vtable = {
        CODEC_ARCH_SOPRANO,
        "Soprano",
        codec_sop_create_impl,
        codec_sop_destroy_impl,
        codec_sop_init_wrap,
        codec_graph_size_exact,
        nullptr,
        codec_sop_decode_wrap,
        codec_sop_decode_latent_wrap,
    };
    return &vtable;
}
