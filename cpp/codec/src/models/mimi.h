#ifndef CODEC_MODEL_MIMI_H
#define CODEC_MODEL_MIMI_H

#include "../codec_internal.h"

struct codec_mimi {
    int32_t sample_rate = 24000;
    int32_t hop_size = 1920;
    int32_t n_q = 32;
    int32_t num_semantic_quantizers = 1;
    int32_t codebook_size = 2048;
    int32_t codebook_dim = 256;
    int32_t hidden_size = 512;
    int32_t num_hidden_layers = 8;
    int32_t num_attention_heads = 8;
    int32_t head_dim = 64;
    int32_t intermediate_size = 2048;
    float rope_theta = 10000.0f;
    float rope_scaling_factor = 1.0f;
    bool has_encoder = false;
    bool has_decoder = false;
};

enum codec_status codec_mimi_init(struct codec_model * model);

enum codec_status codec_mimi_decode(
    struct codec_context * ctx,
    const struct codec_token_buffer * tokens,
    struct codec_pcm_buffer * out_pcm,
    struct codec_decode_params params);

enum codec_status codec_mimi_encode(
    struct codec_context * ctx,
    const std::vector<float> & pcm,
    struct codec_token_buffer * out_tokens,
    struct codec_encode_params params);

enum codec_status codec_mimi_encode_with(
    struct codec_context * ctx,
    struct codec_mimi * mimi,
    const std::vector<float> & pcm,
    struct codec_token_buffer * out_tokens,
    struct codec_encode_params params);
enum codec_status codec_mimi_decode_with(
    struct codec_context * ctx,
    struct codec_mimi * mimi,
    const struct codec_token_buffer * tokens,
    struct codec_pcm_buffer * out_pcm,
    struct codec_decode_params params);
const struct codec_model_vtable * codec_mimi_vtable();

#endif
