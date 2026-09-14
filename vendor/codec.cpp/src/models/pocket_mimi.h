#ifndef CODEC_MODELS_POCKET_MIMI_H
#define CODEC_MODELS_POCKET_MIMI_H

#include "../codec_internal.h"

// Pocket-TTS custom Mimi variant (kyutai/pocket-tts).
//
// A small streaming Mimi: 24 kHz, 12.5 Hz frames, SEANet ratios [6,5,4]
// (decode = upsample), inner_dim 32 / outer 512, a 2-layer LayerScale
// transformer on each side, and a DummyQuantizer output_proj (Conv1d k1,
// 32 -> 512).  Latent-direct: the FlowLM emits a 32-dim continuous latent
// per frame which is decoded to PCM (no codebooks, no RVQ).
//
// Decode path : latent [32, T] -> output_proj (32->512) -> upsample
//   (depthwise convtr, stride 16) -> decoder_transformer -> SEANet decoder
//   -> PCM [1, T*1920].
// Encode path : PCM -> SEANet encoder -> encoder_transformer -> downsample
//   (conv 512->32, stride 16) -> latent [32, T] (voice cloning).
struct codec_pocket_mimi {
    int32_t sample_rate  = 24000;
    int32_t hop_size     = 1920;   // sample_rate / frame_rate = 24000 / 12.5
    int32_t latent_dim   = 32;     // continuous latent (quantizer.dimension)
    int32_t seanet_dim   = 512;    // SEANet dimension (= outer_dim)
    int32_t inner_dim    = 32;
    int32_t outer_dim    = 512;
    int32_t quantizer_dim = 32;

    // Mimi transformer (shared enc/dec hyperparams).
    int32_t tf_layers   = 2;
    int32_t tf_heads    = 8;
    int32_t tf_head_dim = 64;
    int32_t tf_ffn      = 2048;
    int32_t tf_context  = 250;
    float   tf_max_period = 10000.0f;

    // SEANet ratios (decode = upsample order, encode reverses).
    int32_t decoder_ratios[3] = { 6, 5, 4 };
    int32_t encoder_ratios[3] = { 4, 5, 6 };
    int32_t n_ratios = 3;

    // Transposed-conv upsample stride (product of decoder ratios / 8): the
    // downsample/upsample stride is encoder_frame_rate / frame_rate = 2.  In
    // this model outer resampling is at stride 16 (kernel 32).
    int32_t resample_stride = 16;
    int32_t resample_kernel = 32;

    bool has_encoder = true;
    bool has_decoder = true;
};

enum codec_status codec_pocket_mimi_init(struct codec_model * model);

const struct codec_model_vtable * codec_pocket_mimi_vtable();

#endif // CODEC_MODELS_POCKET_MIMI_H
