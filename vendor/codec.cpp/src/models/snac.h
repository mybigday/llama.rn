#ifndef CODEC_MODELS_SNAC_H
#define CODEC_MODELS_SNAC_H

#include "../codec_internal.h"

#include <vector>

struct codec_snac {
    int32_t sample_rate    = 24000;
    int32_t encode_sample_rate = 24000;
    int32_t hop_size       = 512;        // np.prod(encoder_rates)
    int32_t pad_to         = 2048;       // hop * vq_strides[0]
    int32_t n_q            = 3;          // number of VQ levels
    int32_t codebook_size  = 4096;
    int32_t codebook_dim   = 8;
    int32_t latent_dim     = 768;
    int32_t encoder_dim    = 48;
    int32_t decoder_dim    = 1024;
    int32_t encoder_rates[4]  = { 2, 4, 8, 8 };
    int32_t decoder_rates[4]  = { 8, 8, 4, 2 };
    int32_t vq_strides[3]     = { 4, 2, 1 };
    bool    has_encoder    = true;
    bool    has_decoder    = true;
    bool    depthwise      = true;
};

enum codec_status codec_snac_init(struct codec_model * model);

enum codec_status codec_snac_encode(
    struct codec_context * ctx,
    const std::vector<float> & pcm,
    struct codec_token_buffer * out_tokens,
    struct codec_latent_buffer * out_latent,
    struct codec_encode_params params);

enum codec_status codec_snac_decode(
    struct codec_context * ctx,
    const struct codec_token_buffer * tokens,
    struct codec_pcm_buffer * out_pcm,
    struct codec_decode_params params);

const struct codec_model_vtable * codec_snac_vtable();

#endif
