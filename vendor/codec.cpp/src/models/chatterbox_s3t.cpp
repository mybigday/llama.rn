#include "chatterbox_s3t.h"

#include "../blocks/s3_tokenizer.h"
#include "../runtime/graph.h"
#include "../runtime/tensor_utils.h"

#include <cstring>
#include <new>
#include <string>
#include <vector>

struct codec_chatterbox_s3t_build {
    int32_t n_samples = 0;
    codec_s3_tokenizer_params params;
    const codec_model * model = nullptr;
};

static bool codec_chatterbox_s3t_build_encode(ggml_context * ctx_eval, void * user_data, ggml_tensor ** out) {
    const auto * p = static_cast<const codec_chatterbox_s3t_build *>(user_data);
    if (ctx_eval == nullptr || p == nullptr || out == nullptr || p->model == nullptr || p->n_samples <= 0) {
        return false;
    }
    ggml_tensor * pcm = ggml_new_tensor_1d(ctx_eval, GGML_TYPE_F32, p->n_samples);
    ggml_set_name(pcm, "s3t.encode.pcm");
    ggml_tensor * tok = codec_block_s3_tokenizer(ctx_eval, p->model, p->params, pcm);
    if (tok == nullptr) {
        return false;
    }
    ggml_set_name(tok, "s3t.encode.out");
    *out = tok;
    return true;
}

enum codec_status codec_chatterbox_s3t_init(struct codec_model * model) {
    if (model == nullptr || model->impl == nullptr || model->gguf == nullptr) {
        return CODEC_STATUS_INVALID_ARG;
    }

    codec_chatterbox_s3t & s3t = *static_cast<codec_chatterbox_s3t *>(model->impl);
    s3t.sample_rate = codec_read_i32_kv(model->gguf, "codec.sample_rate", s3t.sample_rate);
    s3t.encode_sample_rate = codec_read_i32_kv(model->gguf, "codec.encode_sample_rate", s3t.encode_sample_rate);
    s3t.hop_size = codec_read_i32_kv(model->gguf, "codec.hop_size", s3t.hop_size);
    s3t.n_q = codec_read_i32_kv(model->gguf, "codec.n_q", s3t.n_q);
    s3t.codebook_size = codec_read_i32_kv(model->gguf, "codec.codebook_size", s3t.codebook_size);
    s3t.n_fft = codec_read_i32_kv(model->gguf, "codec.n_fft", s3t.n_fft);
    s3t.win_length = codec_read_i32_kv(model->gguf, "codec.win_length", s3t.win_length);
    s3t.n_mels = codec_read_i32_kv(model->gguf, "codec.n_mels", s3t.n_mels);
    s3t.audio_state = codec_read_i32_kv(model->gguf, "chatterbox_s3t.audio_state", s3t.audio_state);
    s3t.audio_head = codec_read_i32_kv(model->gguf, "chatterbox_s3t.audio_head", s3t.audio_head);
    s3t.audio_layer = codec_read_i32_kv(model->gguf, "chatterbox_s3t.audio_layer", s3t.audio_layer);
    s3t.fsmn_kernel_size = codec_read_i32_kv(model->gguf, "chatterbox_s3t.fsmn_kernel_size", s3t.fsmn_kernel_size);
    s3t.rope_theta = codec_read_f32_kv(model->gguf, "chatterbox_s3t.rope_theta", s3t.rope_theta);
    s3t.has_encoder = codec_read_bool_kv(model->gguf, "codec.has_encoder", s3t.has_encoder);
    s3t.has_decoder = codec_read_bool_kv(model->gguf, "codec.has_decoder", s3t.has_decoder);

    model->sample_rate = s3t.sample_rate;
    model->encode_sample_rate = s3t.encode_sample_rate;
    model->has_encoder = s3t.has_encoder;
    model->has_decoder = s3t.has_decoder;
    model->hop_size = s3t.hop_size;
    model->n_q = s3t.n_q;
    model->codebook_size = s3t.codebook_size;
    model->n_fft = s3t.n_fft;
    model->win_length = s3t.win_length;
    model->n_mels = s3t.n_mels;
    model->latent_dim = -1;

    if (s3t.n_q != 1 || s3t.codebook_size != 6561 || s3t.audio_state <= 0 || s3t.audio_head <= 0 ||
        s3t.audio_state % s3t.audio_head != 0 || s3t.audio_layer <= 0 || s3t.fsmn_kernel_size <= 0 ||
        (s3t.fsmn_kernel_size % 2) == 0 || s3t.rope_theta <= 0.0f) {
        return CODEC_STATUS_INVALID_ARG;
    }

    return CODEC_STATUS_SUCCESS;
}

enum codec_status codec_chatterbox_s3t_encode(
    struct codec_context * ctx,
    const std::vector<float> & pcm,
    struct codec_token_buffer * out_tokens,
    struct codec_latent_buffer * out_latent,
    struct codec_encode_params params) {
    if (ctx == nullptr || ctx->model == nullptr || out_tokens == nullptr) {
        return CODEC_STATUS_INVALID_ARG;
    }

    codec_chatterbox_s3t & s3t = *static_cast<codec_chatterbox_s3t *>(ctx->model->impl);
    if (!s3t.has_encoder) {
        codec_context_set_error(ctx, "model metadata indicates no encoder");
        return CODEC_STATUS_INVALID_STATE;
    }
    if (params.n_q != 0 && params.n_q != 1) {
        codec_context_set_error(ctx, "Chatterbox-S3T encode n_q must be 0 or 1");
        return CODEC_STATUS_INVALID_ARG;
    }
    (void) out_latent;

    if (pcm.empty()) {
        codec_context_set_error(ctx, "empty Chatterbox-S3T PCM input");
        return CODEC_STATUS_INVALID_ARG;
    }
    codec_chatterbox_s3t_build build = {};
    build.n_samples = (int32_t) pcm.size();
    build.params.n_fft = s3t.n_fft;
    build.params.n_mels = s3t.n_mels;
    build.params.hidden = s3t.audio_state;
    build.params.n_heads = s3t.audio_head;
    build.params.n_layers = s3t.audio_layer;
    build.params.fsmn_kernel = s3t.fsmn_kernel_size;
    build.params.rope_theta = s3t.rope_theta;
    build.model = ctx->model;

    std::string err;
    std::vector<int32_t> tok;
    codec_graph_eval_guard eval_guard(ctx);
    codec_graph_cache_entry * entry = nullptr;
    if (!codec_graph_cache_get_or_build(
            ctx, {CODEC_GRAPH_CHATTERBOX_S3T_ENCODE, build.n_samples, 1, s3t.hop_size, 0, build.params.hidden},
            codec_chatterbox_s3t_build_encode, &build, sizeof(build), &entry, &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }
    ggml_tensor * t_pcm = codec_graph_get_tensor(ctx, entry, "s3t.encode.pcm");
    ggml_tensor * t_out = codec_graph_get_tensor(ctx, entry, "s3t.encode.out");
    if (t_pcm == nullptr || t_out == nullptr) {
        codec_context_set_error(ctx, "cached Chatterbox-S3T encode graph is invalid");
        return CODEC_STATUS_INTERNAL_ERROR;
    }
    tok.resize((size_t) ggml_nelements(t_out));
    if (!codec_graph_prepare_io(ctx, entry, &err) ||
        !codec_runtime_write_tensor(t_pcm, pcm.data(), pcm.size() * sizeof(float), &err) ||
        !codec_graph_compute(ctx, entry, ctx->model->n_threads, &err) ||
        !codec_runtime_read_tensor(t_out, tok.data(), tok.size() * sizeof(int32_t), &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }
    const int32_t t_tok = (int32_t) tok.size();

    int32_t * data = static_cast<int32_t *>(std::malloc((size_t) t_tok * sizeof(int32_t)));
    if (data == nullptr) {
        codec_context_set_error(ctx, "failed to allocate token output");
        return CODEC_STATUS_INTERNAL_ERROR;
    }
    std::memcpy(data, tok.data(), (size_t) t_tok * sizeof(int32_t));

    codec_token_buffer_reset(out_tokens);
    out_tokens->data = data;
    out_tokens->n_tokens = t_tok;
    out_tokens->n_frames = t_tok;
    out_tokens->n_q = 1;
    out_tokens->codebook_size = s3t.codebook_size;
    out_tokens->sample_rate = s3t.sample_rate;
    out_tokens->hop_size = s3t.hop_size;
    return CODEC_STATUS_SUCCESS;
}

static void * codec_chatterbox_s3t_create_impl() {
    return new (std::nothrow) codec_chatterbox_s3t();
}

static void codec_chatterbox_s3t_destroy_impl(void * ptr) {
    delete static_cast<codec_chatterbox_s3t *>(ptr);
}

const struct codec_model_vtable * codec_chatterbox_s3t_vtable() {
    static const codec_model_vtable vtable = {
        CODEC_ARCH_CHATTERBOX_S3T,
        "Chatterbox-S3T",
        codec_chatterbox_s3t_create_impl,
        codec_chatterbox_s3t_destroy_impl,
        codec_chatterbox_s3t_init,
        codec_graph_size_exact,
        codec_chatterbox_s3t_encode,
        nullptr,
        nullptr,
    };
    return &vtable;
}
