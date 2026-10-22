#ifndef CODEC_MODEL_COSYVOICE_H
#define CODEC_MODEL_COSYVOICE_H

#include "../blocks/cfm.h"
#include "../blocks/conformer.h"
#include "../blocks/hift.h"
#include "../codec_internal.h"

#include <string>

// CosyVoice token2wav (flow + HiFT) for every release, switched by GGUF
// metadata (`cosyvoice.version` and the per-component keys below). See
// scripts/converters/cosyvoice.py for the variant table.
struct codec_cosyvoice {
    int32_t version = 2;
    int32_t sample_rate = 24000;
    int32_t hop_size = 960;
    int32_t token_rate = 25;
    int32_t codebook_size = 6561;

    // Flow.
    int32_t n_timesteps = 10;
    float cfg_rate = 0.7f;
    float speed = 1.0f;                   // mel time-stretch before HiFT (non-streaming)
    float max_prompt_s = 0.0f;            // flow prompt kept (trailing seconds); 0 = all of it
    int32_t token_mel_ratio = 2;          // v2 / v3
    std::string length_regulator;         // v1: "joint_nearest" | "split_linear"
    int32_t lr_blocks = 4;
    codec_conformer_params enc;
    int32_t enc_layers = 6;
    int32_t enc_up_layers = 0;            // v2 UpsampleConformerEncoder
    codec_cfm_unet_params unet;
    int32_t dit_depth = 22;               // v3
    int32_t dit_dim = 1024;
    int32_t dit_heads = 16;
    int32_t dit_rope_dim = 64;

    codec_hift_params hift;
    bool hift_causal = false;

    // Reference-audio speaker encoders.
    bool has_speaker_encoder = false;
    int32_t mel_n_fft = 1920;
    int32_t mel_hop = 480;
    int32_t tokenizer_layers = 6;
};

const struct codec_model_vtable * codec_cosyvoice_vtable();

#endif
