#ifndef CODEC_BLOCKS_XVECTOR_SINCNET_H
#define CODEC_BLOCKS_XVECTOR_SINCNET_H

#include "../codec_internal.h"

// pyannote XVectorSincNet speaker embedding (SincNet front-end → TDNN stack →
// stats pooling → Linear 3000→512), optionally followed by a projection head
// L2-normalise → Linear(512, D) → LayerNorm(D) (KittenTTS 2 `spk_proj`).
// Weights live under `<prefix>.` (sinc filters precomputed, see
// scripts/converters/xvector_sincnet.py); the head under `<proj_prefix>.`
// (nullptr = the raw 512-d embedding).

// True when `n_samples` of 16 kHz audio are long enough for the network.
bool codec_xvector_sincnet_accepts(int64_t n_samples);

// 16 kHz mono PCM [n] → embedding [512] or [D]. Call from a graph build_fn.
ggml_tensor * codec_block_xvector_sincnet(
    ggml_context * ctx,
    const codec_model * model,
    const char * prefix,
    const char * proj_prefix,
    ggml_tensor * pcm16k);

#endif
