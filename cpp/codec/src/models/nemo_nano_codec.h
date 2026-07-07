#ifndef CODEC_MODELS_NEMO_NANO_CODEC_H
#define CODEC_MODELS_NEMO_NANO_CODEC_H

#include "../codec_internal.h"

struct codec_nemo_nano_codec {
    int32_t sample_rate = 22050;
    int32_t hop_size = 1764;
    int32_t n_q = 4;
    int32_t codebook_size = 4032;
    int32_t codebook_dim = 4;
    int32_t latent_dim = 16;
    bool has_encoder = true;
    bool has_decoder = true;
};

enum codec_status codec_nemo_nano_codec_init(struct codec_model * model);
enum codec_status codec_nemo_nano_codec_encode(
    struct codec_context * ctx,
    const std::vector<float> & pcm,
    struct codec_token_buffer * out_tokens,
    struct codec_latent_buffer * out_latent,
    struct codec_encode_params params);
enum codec_status codec_nemo_nano_codec_decode(
    struct codec_context * ctx,
    const struct codec_token_buffer * tokens,
    struct codec_pcm_buffer * out_pcm,
    struct codec_decode_params params);
const struct codec_model_vtable * codec_nemo_nano_codec_vtable();

#endif
