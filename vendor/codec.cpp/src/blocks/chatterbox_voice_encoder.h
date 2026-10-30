#ifndef CODEC_BLOCKS_CHATTERBOX_VOICE_ENCODER_H
#define CODEC_BLOCKS_CHATTERBOX_VOICE_ENCODER_H

#include "../codec_internal.h"

// Chatterbox / Resemblyzer VoiceEncoder: 40-bin power mel (librosa
// center=True) → overlapping 160-frame partials → 3-layer LSTM (unrolled,
// batched over partials) → Linear + ReLU → L2 norm → mean over partials →
// L2 norm. Weights live under `<prefix>.` (`mel_basis`, `window`,
// `lstm_<l>.{W_ih,W_hh,b_ih,b_hh}`, `proj.{weight,bias}`).
struct codec_voice_encoder_params {
    int32_t n_mels = 40;
    int32_t hidden = 256;
    int32_t n_layers = 3;
    int32_t embed = 256;
    int32_t n_fft = 400;
    int32_t hop = 160;
    int32_t partial_frames = 160;
    int32_t sample_rate = 16000;
    float overlap = 0.5f;
    float rate = 1.3f;
    float min_coverage = 0.8f;
    bool final_relu = true;
};

// Partials for `n_samples` (VoiceEncoder.embeds_from_wavs / get_num_wins);
// 0 when the clip is too short.
int32_t codec_voice_encoder_n_partials(const codec_voice_encoder_params & p, int64_t n_samples);

// Mono PCM at p.sample_rate [n] → speaker embedding [embed]. Call from a
// graph build_fn.
ggml_tensor * codec_block_chatterbox_voice_encoder(
    ggml_context * ctx,
    const codec_model * model,
    const char * prefix,
    const codec_voice_encoder_params & p,
    ggml_tensor * pcm);

#endif
