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
// rope_pos: optional shared I32 position vector (length >= T) for native
// lm_ggml_rope_ext; if NULL the block builds its own [0,T) arange (one per block).
// Hoisting it out of the block kills the per-block ARANGE node.
lm_ggml_tensor * codec_bm_minicpm_block_ht(
    lm_ggml_context * ctx, lm_ggml_tensor * x_ht, const std::string & prefix, const codec_model * model,
    int32_t n_heads, int32_t n_kv, int32_t head_dim, float eps,
    lm_ggml_tensor * cos_dt, lm_ggml_tensor * sin_dt, bool causal,
    lm_ggml_tensor * rope_pos = nullptr);

// Batched variant: x_htb ne=(hidden, T, B).  B branches share the linear
// weights (folded into the token dim) but attend independently (B in ne[3]).
lm_ggml_tensor * codec_bm_minicpm_block_htb(
    lm_ggml_context * ctx, lm_ggml_tensor * x_htb, const std::string & prefix, const codec_model * model,
    int32_t n_heads, int32_t n_kv, int32_t head_dim, float eps,
    lm_ggml_tensor * cos_dt, lm_ggml_tensor * sin_dt, bool causal,
    lm_ggml_tensor * rope_pos = nullptr);

// LocDiT estimator core: pre-projected x_h/cond_h (h_dit,P), mu_h (h_dit,n_mu),
// t_h (h_dit,1) → predicted velocity patch (latent_dim, P).
lm_ggml_tensor * bm_locdit_core(
    lm_ggml_context * ctx, const codec_model * model,
    lm_ggml_tensor * x_h, lm_ggml_tensor * cond_h, lm_ggml_tensor * mu_h, lm_ggml_tensor * t_h,
    lm_ggml_tensor * cos_t, lm_ggml_tensor * sin_t,
    int32_t n_layers, int32_t n_heads, int32_t n_kv, int32_t head_dim, float eps,
    int32_t P, int32_t h_dit, int32_t n_mu);

// Batched CFG LocDiT: pos (mu_h) and neg (mu_zero_h) run jointly; returns both
// velocity patches (latent_dim, P) via pos_out / neg_out.
void bm_locdit_core_batched(
    lm_ggml_context * ctx, const codec_model * model,
    lm_ggml_tensor * x_h, lm_ggml_tensor * cond_h, lm_ggml_tensor * mu_h, lm_ggml_tensor * mu_zero_h,
    lm_ggml_tensor * t_h, lm_ggml_tensor * cos_t, lm_ggml_tensor * sin_t,
    int32_t n_layers, int32_t n_heads, int32_t n_kv, int32_t head_dim, float eps,
    int32_t P, int32_t h_dit, int32_t n_mu,
    lm_ggml_tensor ** pos_out, lm_ggml_tensor ** neg_out);

#endif // CODEC_MODELS_BLUEMAGPIE_BLOCKS_H
