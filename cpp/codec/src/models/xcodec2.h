#ifndef CODEC_MODELS_XCODEC2_H
#define CODEC_MODELS_XCODEC2_H

#include "../codec_internal.h"

struct codec_xcodec2 {
    int32_t sample_rate    = 16000;
    int32_t encode_sample_rate = 16000;
    int32_t hop_size       = 320;
    int32_t n_fft          = 1280;
    int32_t n_q            = 1;
    int32_t codebook_size  = 65536;
    int32_t codebook_dim   = 8;
    int32_t vq_dim         = 2048;
    int32_t latent_dim     = 1024;
    int32_t hidden_dim     = 1024;
    int32_t num_layers     = 12;
    int32_t num_heads      = 16;
    int32_t head_dim       = 64;
    float   rope_theta     = 10000.0f;
    bool    has_encoder    = false;
    bool    has_decoder    = true;

    // BigCodec acoustic encoder constants.
    int32_t enc_ngf        = 48;
    int32_t enc_up_ratios[5] = { 2, 2, 4, 4, 5 };
    int32_t enc_dilations[3] = { 1, 3, 9 };

    // Wav2Vec2-Bert semantic encoder slice (16 layers).
    int32_t w2v_layers     = 16;
    int32_t w2v_hidden     = 1024;
    int32_t w2v_heads      = 16;
    int32_t w2v_head_dim   = 64;
    int32_t w2v_intermediate = 4096;
    int32_t w2v_left_max_pos = 64;
    int32_t w2v_right_max_pos = 8;
    int32_t w2v_dw_kernel  = 31;
    int32_t w2v_input_dim  = 160;
    float   w2v_layer_norm_eps = 1e-5f;

    // Mel-fbank config (Wav2Vec2-Bert / SeamlessM4T defaults).
    int32_t mel_n_fft      = 512;
    int32_t mel_win        = 400;
    int32_t mel_hop        = 160;
    int32_t mel_n_mels     = 80;
    int32_t mel_stride     = 2;
    float   mel_preemphasis = 0.97f;
    float   mel_floor      = 1.192092955078125e-07f;
};

enum codec_status codec_xcodec2_init(struct codec_model * model);

enum codec_status codec_xcodec2_decode(
    struct codec_context * ctx,
    const struct codec_token_buffer * tokens,
    struct codec_pcm_buffer * out_pcm,
    struct codec_decode_params params);

#include <vector>

enum codec_status codec_xcodec2_encode(
    struct codec_context * ctx,
    const std::vector<float> & pcm,
    struct codec_token_buffer * out_tokens,
    struct codec_latent_buffer * out_latent,
    struct codec_encode_params params);

const struct codec_model_vtable * codec_xcodec2_vtable();

#endif
