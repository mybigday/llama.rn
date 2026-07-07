#ifndef CODEC_MODELS_MOSS_AUDIO_H
#define CODEC_MODELS_MOSS_AUDIO_H

#include "../codec_internal.h"

#include <vector>

// MOSS-Audio-Tokenizer (OpenMOSS-Team) — pure-Transformer audio codec.
// Both the Nano (22 M) and the full 1.6 B variant share this architecture;
// per-block dims, layer counts and context-duration are read from GGUF
// metadata.

#define CODEC_MOSS_MAX_MODULES 16

struct codec_moss_audio {
    int32_t sample_rate    = 48000;
    int32_t encode_sample_rate = 48000;
    int32_t hop_size       = 3840;
    int32_t n_q            = 16;
    int32_t codebook_size  = 1024;
    int32_t codebook_dim   = 8;
    int32_t latent_dim     = 768;
    int32_t rvq_dim        = 512;
    int32_t number_channels = 1;
    bool    channel_interleave = true;
    bool    has_encoder    = true;
    bool    has_decoder    = true;
    float   context_duration = 10.0f;

    // Encoder + decoder module list.  module_type = 0 (PatchedPretransform) or
    // 1 (Transformer).  All other fields are only meaningful when type == 1.
    int32_t enc_n_modules = 0;
    int32_t dec_n_modules = 0;
    int32_t enc_module_type[CODEC_MOSS_MAX_MODULES]   = {0};
    int32_t enc_patch_size[CODEC_MOSS_MAX_MODULES]    = {0};
    int32_t enc_in_dim[CODEC_MOSS_MAX_MODULES]        = {0};
    int32_t enc_out_dim[CODEC_MOSS_MAX_MODULES]       = {0};
    int32_t enc_d_model[CODEC_MOSS_MAX_MODULES]       = {0};
    int32_t enc_n_heads[CODEC_MOSS_MAX_MODULES]       = {0};
    int32_t enc_n_layers[CODEC_MOSS_MAX_MODULES]      = {0};
    int32_t enc_ffn_dim[CODEC_MOSS_MAX_MODULES]       = {0};
    float   enc_context_duration[CODEC_MOSS_MAX_MODULES]  = {0};
    float   enc_max_period[CODEC_MOSS_MAX_MODULES]    = {0};
    float   enc_layer_scale[CODEC_MOSS_MAX_MODULES]   = {0};
    int32_t dec_module_type[CODEC_MOSS_MAX_MODULES]   = {0};
    int32_t dec_patch_size[CODEC_MOSS_MAX_MODULES]    = {0};
    int32_t dec_in_dim[CODEC_MOSS_MAX_MODULES]        = {0};
    int32_t dec_out_dim[CODEC_MOSS_MAX_MODULES]       = {0};
    int32_t dec_d_model[CODEC_MOSS_MAX_MODULES]       = {0};
    int32_t dec_n_heads[CODEC_MOSS_MAX_MODULES]       = {0};
    int32_t dec_n_layers[CODEC_MOSS_MAX_MODULES]      = {0};
    int32_t dec_ffn_dim[CODEC_MOSS_MAX_MODULES]       = {0};
    float   dec_context_duration[CODEC_MOSS_MAX_MODULES]  = {0};
    float   dec_max_period[CODEC_MOSS_MAX_MODULES]    = {0};
    float   dec_layer_scale[CODEC_MOSS_MAX_MODULES]   = {0};
};

enum codec_status codec_moss_audio_init(struct codec_model * model);

enum codec_status codec_moss_audio_encode(
    struct codec_context * ctx,
    const std::vector<float> & pcm,
    struct codec_token_buffer * out_tokens,
    struct codec_latent_buffer * out_latent,
    struct codec_encode_params params);

enum codec_status codec_moss_audio_decode(
    struct codec_context * ctx,
    const struct codec_token_buffer * tokens,
    struct codec_pcm_buffer * out_pcm,
    struct codec_decode_params params);

const struct codec_model_vtable * codec_moss_audio_vtable();

#endif
