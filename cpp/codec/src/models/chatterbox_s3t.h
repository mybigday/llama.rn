#ifndef CODEC_MODEL_CHATTERBOX_S3T_H
#define CODEC_MODEL_CHATTERBOX_S3T_H

#include "../codec_internal.h"

struct codec_chatterbox_s3t {
    int32_t sample_rate = 24000;
    int32_t encode_sample_rate = 16000;
    int32_t hop_size = 960;
    int32_t n_q = 1;
    int32_t codebook_size = 6561;
    int32_t n_fft = 400;
    int32_t win_length = 400;
    int32_t n_mels = 128;
    int32_t audio_state = 1280;
    int32_t audio_head = 20;
    int32_t audio_layer = 6;
    int32_t fsmn_kernel_size = 31;
    float rope_theta = 10000.0f;
    bool has_encoder = true;
    bool has_decoder = false;
};

enum codec_status codec_chatterbox_s3t_init(struct codec_model * model);
enum codec_status codec_chatterbox_s3t_encode(
    struct codec_context * ctx,
    const std::vector<float> & pcm,
    struct codec_token_buffer * out_tokens,
    struct codec_latent_buffer * out_latent,
    struct codec_encode_params params);
const struct codec_model_vtable * codec_chatterbox_s3t_vtable();

#endif
