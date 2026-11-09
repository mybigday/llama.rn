#ifndef CODEC_BLOCKS_DIT_H
#define CODEC_BLOCKS_DIT_H

#include "../codec_internal.h"

// F5-TTS-style DiT velocity estimator used by CosyVoice3's flow
// (cosyvoice/flow/DiT/dit.py, non-streaming = full attention).
// Weights under `<prefix>.`: `t.l1/l2` (TimestepEmbedding), `in.proj`,
// `in.cp1/cp2` (CausalConvPositionEmbedding, grouped k=31 convs),
// `blk.<i>.{ada, attn.{q,k,v,o}, ff1, ff2}`, `out_ada`, `proj_out`.
struct codec_dit_params {
    int32_t depth = 22;
    int32_t dim = 1024;
    int32_t n_heads = 16;
    int32_t rope_dim = 64;       // rotary applied to the first rope_dim channels of q/k only
    int32_t conv_groups = 16;
    int32_t time_sin_dim = 256;
    float time_scale = 1000.0f;
};

// x, mu, cond: [T, 80]; spks: [80]. Returns the velocity [T, 80].
ggml_tensor * codec_block_dit(
    ggml_context * ctx, const codec_model * model, const char * prefix,
    const codec_dit_params & p,
    ggml_tensor * x, ggml_tensor * mu, ggml_tensor * spks, ggml_tensor * cond, float t);

#endif
