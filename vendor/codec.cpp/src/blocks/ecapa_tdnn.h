#ifndef CODEC_BLOCKS_ECAPA_TDNN_H
#define CODEC_BLOCKS_ECAPA_TDNN_H

#include "../codec_internal.h"

// ECAPA-TDNN speaker encoder (Qwen3-TTS `Qwen3TTSSpeakerEncoder`): log
// magnitude mel → TDNN → SE-Res2Net blocks → multi-layer aggregation →
// attentive statistics pooling → Linear. Convs use reflect "same" padding.
// Weights live under `<prefix>.` (`mel_basis`, `window`, `blocks.<i>.*`,
// `mfa.conv`, `asp.{tdnn.conv,conv}`, `fc`).
constexpr int32_t kCodecEcapaMaxBlocks = 8;

struct codec_ecapa_params {
    int32_t n_fft = 1024;
    int32_t hop = 256;
    int32_t n_mels = 128;
    int32_t res2net_scale = 8;
    int32_t n_blocks = 0;   // entries in channels/kernels/dilations (last = MFA)
    int32_t channels[kCodecEcapaMaxBlocks] = {};
    int32_t kernels[kCodecEcapaMaxBlocks] = {};
    int32_t dilations[kCodecEcapaMaxBlocks] = {};
};

// Mel frames for `n_samples`; the network needs at least 2.
int32_t codec_ecapa_n_frames(const codec_ecapa_params & p, int64_t n_samples);

// Mono PCM [n] → embedding [enc_dim]. Call from a graph build_fn.
ggml_tensor * codec_block_ecapa_tdnn(
    ggml_context * ctx,
    const codec_model * model,
    const char * prefix,
    const codec_ecapa_params & p,
    ggml_tensor * pcm);

#endif
