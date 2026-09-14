#ifndef CODEC_MODEL_SOPRANO_H
#define CODEC_MODEL_SOPRANO_H

#include "../codec_internal.h"

struct codec_soprano {
    int32_t sample_rate = 32000;
    int32_t hop_size = 512;
    int32_t n_fft = 2048;
    int32_t win_length = 2048;
    int32_t latent_dim = 512;
    int32_t decoder_dim = 768;
    int32_t intermediate_dim = 2304;
    int32_t num_layers = 8;
    int32_t upscale = 4;
    int32_t dw_kernel = 3;
    bool has_encoder = false;
    bool has_decoder = true;
};

enum codec_status codec_soprano_init(struct codec_model * model);
enum codec_status codec_soprano_decode(
    struct codec_context * ctx,
    const struct codec_token_buffer * tokens,
    struct codec_pcm_buffer * out_pcm,
    struct codec_decode_params params);
enum codec_status codec_soprano_decode_latent(
    struct codec_context * ctx,
    const float * quantized_representation,
    int32_t latent_dim,
    int32_t n_frames,
    struct codec_pcm_buffer * out_pcm,
    struct codec_decode_params params);
const struct codec_model_vtable * codec_soprano_vtable();

#endif
