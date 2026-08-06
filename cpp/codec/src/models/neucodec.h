#ifndef CODEC_MODELS_NEUCODEC_H
#define CODEC_MODELS_NEUCODEC_H

#include "../codec_internal.h"

#include <vector>

#include "../ops/local_attn.h"

struct codec_neucodec {
    int32_t sample_rate = 24000;
    int32_t encode_sample_rate = 16000;
    int32_t hop_size = 480;
    int32_t n_fft = 1920;
    int32_t n_q = 1;
    int32_t codebook_size = 65536;
    int32_t codebook_dim = 8;
    int32_t vq_dim = 2048;
    int32_t latent_dim = 1024;
    int32_t hidden_dim = 1024;
    int32_t num_layers = 12;
    int32_t num_heads = 16;
    int32_t head_dim = 64;
    float rope_theta = 10000.0f;
    int32_t encoder_type = 0; // 0=base, 1=distill
    bool has_encoder = false;
    bool has_decoder = true;

    // distill hubert semantic model config
    int32_t hubert_hidden = 768;
    int32_t hubert_heads = 12;
    int32_t hubert_intermediate = 3072;
    int32_t hubert_layers = 2;
    int32_t hubert_pos_k = 128;
    int32_t hubert_pos_groups = 16;
    float hubert_ln_eps = 1e-5f;
    int32_t hubert_feat_layers = 7;
    int32_t hubert_conv_dim[7] = { 512, 512, 512, 512, 512, 512, 512 };
    int32_t hubert_conv_kernel[7] = { 10, 3, 3, 3, 3, 2, 2 };
    int32_t hubert_conv_stride[7] = { 5, 2, 2, 2, 2, 2, 2 };

    bool distill_bias_ready = false;
    int32_t distill_bias_down_max_dist = 0;
    int32_t distill_bias_local_max_dist = 0;
    std::vector<float> distill_bias_down;
    std::vector<float> distill_bias_local;
    codec_local_attn_params distill_attn_down;
    codec_local_attn_params distill_attn_local;
};

enum codec_status codec_neucodec_init(struct codec_model * model);

enum codec_status codec_neucodec_decode(
    struct codec_context * ctx,
    const struct codec_token_buffer * tokens,
    struct codec_pcm_buffer * out_pcm,
    struct codec_decode_params params);

enum codec_status codec_neucodec_encode(
    struct codec_context * ctx,
    const std::vector<float> & pcm,
    struct codec_token_buffer * out_tokens,
    struct codec_latent_buffer * out_latent,
    struct codec_encode_params params);

const struct codec_model_vtable * codec_neucodec_vtable();
const struct codec_model_vtable * codec_distill_neucodec_vtable();

#endif
