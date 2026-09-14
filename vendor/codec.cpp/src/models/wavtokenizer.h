#ifndef CODEC_MODELS_WAVTOKENIZER_H
#define CODEC_MODELS_WAVTOKENIZER_H

#include "../codec_internal.h"

struct codec_wavtokenizer_large {
    int32_t sample_rate = 24000;
    int32_t hop_size = 320;
    int32_t n_q = 1;
    int32_t codebook_size = 0;
    int32_t codebook_dim = 0;
    bool has_encoder = false;
    bool has_decoder = false;

    struct ggml_tensor * vq_embed = nullptr;
};

enum codec_status codec_wavtokenizer_init(struct codec_model * model);
enum codec_status codec_wavtokenizer_encode(
    struct codec_context * ctx,
    const std::vector<float> & pcm,
    struct codec_token_buffer * out_tokens,
    struct codec_encode_params params);
enum codec_status codec_wavtokenizer_decode(
    struct codec_context * ctx,
    const struct codec_token_buffer * tokens,
    struct codec_pcm_buffer * out_pcm,
    struct codec_decode_params params);
const struct codec_model_vtable * codec_wavtokenizer_vtable();

#endif
