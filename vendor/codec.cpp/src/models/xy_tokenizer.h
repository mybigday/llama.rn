#ifndef CODEC_MODELS_XY_TOKENIZER_H
#define CODEC_MODELS_XY_TOKENIZER_H

#include "../codec_internal.h"

#include <vector>

// XY-Tokenizer (OpenMOSS-Team/XY_Tokenizer_TTSD_V0_hf): 16 kHz-in / 24 kHz-out
// neural codec.  Encode = mel-fbank + parallel Whisper-style semantic +
// acoustic encoders + adapters + ResidualDownConv + RVQ8.  Decode = RVQ
// inverse + post_rvq_adapter + UpConv + Whisper-style mel decoder + ConvNeXt
// Vocos backbone + iSTFT head.  See `scripts/converters/xy_tokenizer.py` for
// the full data flow and tensor namespace.
struct codec_xy_tokenizer {
    int32_t encode_sample_rate = 16000;
    int32_t sample_rate        = 24000;   // decode sample rate
    int32_t encoder_downsample_rate = 1280;
    int32_t decoder_upsample_rate   = 1920;
    int32_t latent_dim         = 3072;
    int32_t rvq_dim            = 512;
    int32_t codebook_dim       = 512;
    int32_t codebook_size      = 1024;
    int32_t n_q                = 8;

    // Mel-fbank (Whisper-style) parameters.
    int32_t mel_n_mels         = 80;
    int32_t mel_n_fft          = 400;
    int32_t mel_hop_length     = 160;
    int32_t mel_chunk_length_s = 30;

    // Per-module dimensions, populated from GGUF metadata + tensor shapes.
    int32_t sem_enc_d_model        = 768;
    int32_t sem_enc_n_heads        = 12;
    int32_t sem_enc_n_layers       = 12;
    int32_t sem_enc_ffn_dim        = 3072;
    int32_t sem_enc_max_pos        = 1500;
    int32_t sem_enc_stride         = 2;
    // (acoustic encoder shares the same shape — values copied at init time.)

    int32_t sem_enc_adapter_n_layers   = 4;
    int32_t pre_rvq_adapter_in_dim     = 1536;
    int32_t pre_rvq_adapter_n_layers   = 4;
    int32_t post_rvq_adapter_n_layers  = 4;

    int32_t downsample_avg_pooler  = 4;
    int32_t upsample_stride        = 4;

    // Vocos vocoder.
    int32_t vocos_n_blocks         = 30;
    int32_t vocos_dim              = 512;
    int32_t vocos_intermediate     = 4096;
    int32_t vocos_n_fft            = 960;
    int32_t vocos_hop              = 240;
    int32_t vocos_head_out_dim     = 962;     // n_fft + 2

    bool has_encoder = true;
    bool has_decoder = true;
};

enum codec_status codec_xy_tokenizer_init(struct codec_model * model);

enum codec_status codec_xy_tokenizer_encode(
    struct codec_context * ctx,
    const std::vector<float> & pcm,
    struct codec_token_buffer * out_tokens,
    struct codec_latent_buffer * out_latent,
    struct codec_encode_params params);

enum codec_status codec_xy_tokenizer_decode(
    struct codec_context * ctx,
    const struct codec_token_buffer * tokens,
    struct codec_pcm_buffer * out_pcm,
    struct codec_decode_params params);

const struct codec_model_vtable * codec_xy_tokenizer_vtable();

#endif
