#ifndef CODEC_BLOCKS_CAMPPLUS_H
#define CODEC_BLOCKS_CAMPPLUS_H

#include "../codec_internal.h"

// CAMPPlus speaker embedding (3D-Speaker; the x-vector encoder of CosyVoice /
// Chatterbox S3Gen): Kaldi fbank + CMN → FCM head → TDNN → 3 CAM dense
// blocks → statistics pooling → 192-d embedding. Weights live under
// `<prefix>.` (`fbank_mel` + network, BatchNorm folded at conversion; see
// scripts/converters/campplus.py).

constexpr int32_t kCodecCampplusEmbedDim = 192;

// True when `n_samples` of 16 kHz audio are long enough for the network.
bool codec_campplus_accepts(int64_t n_samples);

// 16 kHz mono PCM [n] → embedding [192]. Call from a graph build_fn.
// `onnx_seg_pooling`: average a partial last 100-frame CAM segment over the
// full window, as CosyVoice's `campplus.onnx` export does (its AveragePool
// divides by the kernel size); PyTorch avg_pool1d(ceil_mode) divides by the
// frames actually present.
ggml_tensor * codec_block_campplus(
    ggml_context * ctx,
    const codec_model * model,
    const char * prefix,
    ggml_tensor * pcm16k,
    bool onnx_seg_pooling = false);

#endif
