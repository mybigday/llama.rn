#ifndef CODEC_MODELS_DAC_H
#define CODEC_MODELS_DAC_H

#include "../codec_internal.h"

struct codec_dac {
    int32_t sample_rate = 24000;
    int32_t hop_size = 512;
    int32_t n_q = 4;
    int32_t codebook_size = 1024;
    int32_t latent_dim = 1024;
    int32_t codebook_dim = 8;
    bool has_encoder = false;
    bool has_decoder = false;
};

enum codec_status codec_dac_init(struct codec_model * model);
enum codec_status codec_dac_encode(
    struct codec_context * ctx,
    const std::vector<float> & pcm,
    struct codec_token_buffer * out_tokens,
    struct codec_latent_buffer * out_latent,
    struct codec_encode_params params);
enum codec_status codec_dac_decode(
    struct codec_context * ctx,
    const struct codec_token_buffer * tokens,
    struct codec_pcm_buffer * out_pcm,
    struct codec_decode_params params);
enum codec_status codec_dac_decode_latent(
    struct codec_context * ctx,
    const float * qr,
    int32_t latent_dim,
    int32_t n_frames,
    struct codec_pcm_buffer * out_pcm,
    struct codec_decode_params params);
const struct codec_model_vtable * codec_dac_vtable();

#endif
