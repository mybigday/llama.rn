#ifndef CODEC_MODELS_BLUEMAGPIE_AUDIOVAE_H
#define CODEC_MODELS_BLUEMAGPIE_AUDIOVAE_H

#include "../codec_internal.h"

// BlueMagpie / VoxCPM2 AudioVAE V2 — continuous-latent decoder.
//
// Decodes a continuous latent sequence [latent_dim, n_frames] (the CFM
// diffusion output) into a 48 kHz waveform via the causal depthwise conv
// stack.  This is the bottom of the BlueMagpie stack; the LM / LocEnc /
// LocDiT / RALM modules feed it through the codec_lm adaptor in later slices.
//
// Decode-only for now (has_encoder=false): the encoder (ref/prompt-audio →
// latent) lands when the voice-clone input modes are wired.
struct codec_bluemagpie_audiovae {
    int32_t sample_rate        = 48000;   // decode (output) sample rate
    int32_t encode_sample_rate = 16000;   // encoder working rate (unused yet)
    int32_t latent_dim         = 64;
    int32_t decoder_dim        = 2048;
    int32_t decode_hop         = 1920;    // np.prod(decoder_rates)
    int32_t decoder_rates[6]   = { 8, 6, 5, 2, 2, 2 };
    int32_t n_blocks           = 6;
    int32_t encode_hop         = 640;     // np.prod(encoder_rates)
    int32_t encoder_rates[4]   = { 2, 5, 8, 8 };
    int32_t n_enc_blocks       = 4;
    int32_t encoder_dim        = 128;
    bool    has_encoder        = true;
    bool    has_decoder        = true;
    bool    depthwise          = true;
};

enum codec_status codec_bluemagpie_audiovae_init(struct codec_model * model);

const struct codec_model_vtable * codec_bluemagpie_audiovae_vtable();

#endif // CODEC_MODELS_BLUEMAGPIE_AUDIOVAE_H
