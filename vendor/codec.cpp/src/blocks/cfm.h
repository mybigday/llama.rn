#ifndef CODEC_BLOCKS_CFM_H
#define CODEC_BLOCKS_CFM_H

#include "../codec_internal.h"

#include <functional>

// Conditional flow matching pieces shared by CosyVoice v1/v2/v3 and
// Chatterbox S3Gen: the Matcha-style 1D U-Net estimator
// (ConditionalDecoder / CausalConditionalDecoder) and the unrolled Euler
// solver with classifier-free guidance.
//
// U-Net weights under `<prefix>.`: `t.l1`, `t.l2` (TimestepEmbedding),
// `t_mix` (MeanFlow only), `dn.<i>.{r,t.<j>,x}`, `md.<i>.{r,t.<j>}`,
// `up.<i>.{r,t.<j>,x}`, `final.{cv,ln}`, `proj`. Resnet blocks `r` carry
// `b1/b2.{cv,ln}` (ln = LayerNorm for causal, GroupNorm for non-causal),
// `mlp`, `res`; transformers `t.<j>` the diffusers BasicTransformerBlock set.
struct codec_cfm_unet_params {
    int32_t in_channels = 320;     // packed [x, mu, spks, cond]
    int32_t out_channels = 80;
    int32_t n_levels = 1;          // len(channels)
    int32_t n_mid = 12;
    int32_t n_transformers = 4;    // per block
    int32_t head_dim = 64;
    int32_t n_heads = 8;
    bool causal = true;            // CausalBlock1D (LayerNorm) vs Block1D (GroupNorm)
    int32_t gn_groups = 8;
    float time_scale = 1000.0f;
};

// SinusoidalPosEmb(dim)(t * scale) → TimestepEmbedding (Linear, SiLU, Linear)
// under `<prefix>.t.l1/l2`. Returns [time_embed_dim].
ggml_tensor * codec_block_cfm_time_mlp(
    ggml_context * ctx, const codec_model * model, const char * prefix,
    float t, int32_t sin_dim, float scale);

// One estimator call. x, mu, cond: [T, 80]; spks: [80]; t_emb: [time_embed_dim].
ggml_tensor * codec_block_cfm_unet(
    ggml_context * ctx, const codec_model * model, const char * prefix,
    const codec_cfm_unet_params & p,
    ggml_tensor * x, ggml_tensor * mu, ggml_tensor * spks, ggml_tensor * cond,
    ggml_tensor * t_emb);

struct codec_cfm_solver_params {
    int32_t n_steps = 10;
    float cfg_rate = 0.7f;   // 0 drops the unconditional pass
    bool meanflow = false;   // linear t, no CFG, estimator sees (t, r)
};

// Velocity estimate for (x, mu, spks, cond) at time t (MeanFlow: target r).
using codec_cfm_estimator_fn = std::function<ggml_tensor *(
    ggml_tensor * x, ggml_tensor * mu, ggml_tensor * spks, ggml_tensor * cond, float t, float r)>;

// Unrolled ODE from noise z [T, 80] to the mel estimate [T, 80]: cosine
// t-schedule (linear for MeanFlow), v = (1 + cfg) * v_cond - cfg * v_uncond
// with the unconditional pass seeing zeroed mu / spks / cond.
ggml_tensor * codec_block_cfm_solve(
    ggml_context * ctx, const codec_cfm_solver_params & p,
    ggml_tensor * z, ggml_tensor * mu, ggml_tensor * spks, ggml_tensor * cond,
    const codec_cfm_estimator_fn & estimator);

#endif
