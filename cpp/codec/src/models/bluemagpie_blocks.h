#ifndef CODEC_MODELS_BLUEMAGPIE_BLOCKS_H
#define CODEC_MODELS_BLUEMAGPIE_BLOCKS_H

#include "../codec_internal.h"

#include <ggml.h>

#include <string>

// Shared BlueMagpie/VoxCPM ggml building blocks (parity-verified in
// src/models/bluemagpie_locenc_debug.cpp).  Reused by the codec_lm
// continuous_latent_cfm kind to assemble the per-step graph.

// MiniCPM decoder block (RMSNorm + GQA + baked-RoPE + SwiGLU, use_mup=false).
// x_ht ne=(hidden, T); cos_dt/sin_dt ne=(head_dim, T) or NULL (no-rope).
lm_ggml_tensor * codec_bm_minicpm_block_ht(
    lm_ggml_context * ctx, lm_ggml_tensor * x_ht, const std::string & prefix, const codec_model * model,
    int32_t n_heads, int32_t n_kv, int32_t head_dim, float eps,
    lm_ggml_tensor * cos_dt, lm_ggml_tensor * sin_dt, bool causal);

// LocDiT estimator core: pre-projected x_h/cond_h (h_dit,P), mu_h (h_dit,n_mu),
// t_h (h_dit,1) → predicted velocity patch (latent_dim, P).
lm_ggml_tensor * bm_locdit_core(
    lm_ggml_context * ctx, const codec_model * model,
    lm_ggml_tensor * x_h, lm_ggml_tensor * cond_h, lm_ggml_tensor * mu_h, lm_ggml_tensor * t_h,
    lm_ggml_tensor * cos_t, lm_ggml_tensor * sin_t,
    int32_t n_layers, int32_t n_heads, int32_t n_kv, int32_t head_dim, float eps,
    int32_t P, int32_t h_dit, int32_t n_mu);

#endif // CODEC_MODELS_BLUEMAGPIE_BLOCKS_H
