#ifndef CODEC_BLOCKS_HIFT_H
#define CODEC_BLOCKS_HIFT_H

#include "../codec_internal.h"

#include <random>
#include <vector>

// HiFTGenerator (CosyVoice v1/v2/v3 and Chatterbox S3Gen vocoder): ConvRNN F0
// predictor → NSF harmonic source → STFT of the source → conv_pre / upsample /
// source fusion / resblocks → conv_post head (log-mag + phase) → iSTFT.
// Weights live under `<prefix>.` (`f0.cn.<i>`, `f0.cls`, `src.lin`,
// `conv_pre`, `up.<i>`, `src_dn.<i>`, `src_rb.<i>`, `rb.<i*n_rb+j>`,
// `conv_post`; resblocks carry `a1/a2.<d>` Snake alphas and `cv1/cv2.<d>`).
struct codec_hift_params {
    int32_t sample_rate = 24000;
    int32_t in_channels = 80;
    int32_t n_fft = 16;
    int32_t hop = 4;
    int32_t nb_harmonics = 8;
    float nsf_alpha = 0.1f;
    float nsf_sigma = 0.003f;
    float nsf_voiced_threshold = 10.0f;
    float lrelu_slope = 0.1f;
    float audio_limit = 0.99f;
    std::vector<int32_t> upsample_rates = {8, 5, 3};
    std::vector<int32_t> upsample_kernels = {16, 11, 7};
    std::vector<int32_t> resblock_kernels = {3, 7, 11};
    std::vector<int32_t> resblock_dilations = {1, 3, 5};
    std::vector<int32_t> source_resblock_kernels = {7, 7, 11};
    // 1: SineGen (cumsum of per-sample phase increments; CosyVoice v1,
    //    Chatterbox). 2: SineGen2 (frame-rate phase, linearly upsampled;
    //    CosyVoice2/3 at 24 kHz).
    int32_t sinegen = 1;
    // CausalHiFTGenerator (CosyVoice3): causal / right-look-ahead conv padding,
    // nearest-upsample + Conv1d instead of ConvTranspose1d, nearest phase
    // upsampling in SineGen2 and a uniform U[0, 1) noise buffer.
    bool causal = false;
};

// prod(upsample_rates) * hop: PCM samples per mel frame.
int32_t codec_hift_upsample_factor(const codec_hift_params & p);

// mel [T, in_channels] (ne[0] = T) → PCM [T * upsample_factor].
// `nsf_phase` [nb_harmonics + 1] (initial harmonic phases, SineGen only;
// ignored by SineGen2) and `nsf_noise` [T_pcm, nb_harmonics + 1] (standard
// normal; U[0, 1) for the causal generator) are graph inputs owned by the
// caller. Call from a graph build_fn.
ggml_tensor * codec_block_hift(
    ggml_context * ctx,
    const codec_model * model,
    const char * prefix,
    const codec_hift_params & p,
    ggml_tensor * mel_tc,
    ggml_tensor * nsf_phase,
    ggml_tensor * nsf_noise);

// Host-side random inputs for codec_block_hift, drawn from `rng` (phase
// U(-pi, pi) with harmonic 0 fixed to 0, then the noise: standard normal, or
// U[0, 1) for the causal generator).
void codec_hift_fill_random(
    const codec_hift_params & p,
    int32_t n_pcm,
    std::mt19937_64 & rng,
    std::vector<float> * phase,
    std::vector<float> * noise);

#endif
