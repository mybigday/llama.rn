#ifndef CODEC_OPS_GGML_OPS_H
#define CODEC_OPS_GGML_OPS_H

#include "../codec_internal.h"

enum codec_unary_op {
    CODEC_UNARY_SIGMOID = 0,
    CODEC_UNARY_ELU = 1,
    CODEC_UNARY_SILU = 2,
    CODEC_UNARY_GELU_ERF = 3,
    CODEC_UNARY_MISH = 4,
};

ggml_tensor * codec_op_unary(ggml_context * ctx, ggml_tensor * x, codec_unary_op op);
ggml_tensor * codec_op_layer_norm(ggml_context * ctx, ggml_tensor * x, float eps, ggml_tensor * gamma, ggml_tensor * beta);
ggml_tensor * codec_op_layer_norm_ct(ggml_context * ctx, ggml_tensor * x_ct, float eps, ggml_tensor * gamma, ggml_tensor * beta);
ggml_tensor * codec_op_layer_norm_tc(ggml_context * ctx, ggml_tensor * x_tc, float eps, ggml_tensor * gamma, ggml_tensor * beta);
ggml_tensor * codec_op_rms_norm_ct(ggml_context * ctx, ggml_tensor * x_ct, float eps, ggml_tensor * gamma);
ggml_tensor * codec_op_group_norm(ggml_context * ctx, ggml_tensor * x, int32_t n_groups, float eps, ggml_tensor * gamma, ggml_tensor * beta);
ggml_tensor * codec_op_linear(ggml_context * ctx, ggml_tensor * x, ggml_tensor * w, ggml_tensor * b);
ggml_tensor * codec_op_linear_tc(ggml_context * ctx, ggml_tensor * x_tc, ggml_tensor * w, ggml_tensor * b);
ggml_tensor * codec_op_snake(ggml_context * ctx, ggml_tensor * x, ggml_tensor * alpha, float eps);
ggml_tensor * codec_op_snake_beta(ggml_context * ctx, ggml_tensor * x, ggml_tensor * alpha, ggml_tensor * inv_beta, float eps);
ggml_tensor * codec_op_pad_1d(ggml_context * ctx, ggml_tensor * x, int32_t pad_left, int32_t pad_right);
ggml_tensor * codec_op_pad_1d_replicate(ggml_context * ctx, ggml_tensor * x, int32_t pad_left, int32_t pad_right);
ggml_tensor * codec_op_causal_crop_1d(ggml_context * ctx, ggml_tensor * x, int32_t target_t);
ggml_tensor * codec_op_crop_1d(ggml_context * ctx, ggml_tensor * x, int32_t crop_left, int32_t crop_right);
ggml_tensor * codec_op_channel_scale(ggml_context * ctx, ggml_tensor * x, ggml_tensor * scale);

ggml_tensor * codec_op_tokens_to_features(ggml_context * ctx, ggml_tensor * tokens, int32_t out_channels);

// ConvNeXt block (Vocos-style): residual + (depthwise conv → LayerNorm → linear → GELU → linear → channel-scale).
// `x_ct` is `[c, t]`; biases and `gamma` are optional (pass nullptr to skip).
// `dw_padding` is the symmetric (non-causal) padding for the depthwise conv;
// for causal variants compose your own block.
ggml_tensor * codec_op_convnext_block_ct(
    ggml_context * ctx,
    ggml_tensor * x_ct,
    ggml_tensor * dw_w,
    ggml_tensor * dw_b,
    ggml_tensor * ln_w,
    ggml_tensor * ln_b,
    ggml_tensor * pw1_w,
    ggml_tensor * pw1_b,
    ggml_tensor * pw2_w,
    ggml_tensor * pw2_b,
    ggml_tensor * gamma,
    int32_t dw_padding);

// Matcha CausalBlock1D: causal conv1d (left-pad k-1) → LayerNorm along channels → Mish.
// `x_tc` is `[t, c]`; the conv weight is loaded directly, kernel is inferred from
// `conv_w->ne[0]`. Used by Matcha-TTS / Chatterbox CFM decoders and any other
// causal diffusion U-Net.
ggml_tensor * codec_op_causal_block1d_tc(
    ggml_context * ctx,
    ggml_tensor * x_tc,
    ggml_tensor * conv_w,
    ggml_tensor * conv_b,
    ggml_tensor * ln_w,
    ggml_tensor * ln_b);

// One branch of a HiFi-GAN-style ResBlock: `x + conv1d(snake(conv1d(snake(x))))`
// where the first conv has user-provided dilation and the second conv runs at
// dilation=1. `x_tc` is `[t, c]`; padding is symmetric (`d*(k-1)/2` for the
// dilated conv, `(k-1)/2` for the second). Caller stacks branches with
// distinct dilations to build the full ResBlock.
ggml_tensor * codec_op_hifigan_resblock_branch_ct(
    ggml_context * ctx,
    ggml_tensor * x_tc,
    ggml_tensor * a1,
    ggml_tensor * a2,
    ggml_tensor * c1_w,
    ggml_tensor * c1_b,
    ggml_tensor * c2_w,
    ggml_tensor * c2_b,
    int32_t kernel_size,
    int32_t dilation);

// Matcha CausalResnetBlock1D forward:
//   h  = causal_block(x)                   (block1: causal-conv → LN_tc → Mish)
//   h += linear(mish(t_emb)).unsqueeze(t)  (broadcast over time)
//   h  = causal_block(h)                   (block2 with the same shape pattern)
//   return h + res_conv1x1(x)
// `t_emb` is `[time_embed_dim]`; `mlp_w` has PyTorch shape `(out_c, time_embed_dim)`
// and `res_w` is a 1×1 conv of shape `(out_c, in_c, 1)`. `x_tc` is `[t, c=in_c]`.
ggml_tensor * codec_op_cfm_causal_resnet_block_tc(
    ggml_context * ctx,
    ggml_tensor * x_tc,
    ggml_tensor * t_emb,
    ggml_tensor * b1_conv_w, ggml_tensor * b1_conv_b,
    ggml_tensor * b1_ln_w,   ggml_tensor * b1_ln_b,
    ggml_tensor * b2_conv_w, ggml_tensor * b2_conv_b,
    ggml_tensor * b2_ln_w,   ggml_tensor * b2_ln_b,
    ggml_tensor * mlp_w,     ggml_tensor * mlp_b,
    ggml_tensor * res_w,     ggml_tensor * res_b);

// Diffusers BasicTransformerBlock without cross-attention (standard
// `LayerNorm → self-attn → +res` then `LayerNorm → GELU FFN → +res`). Used by
// any diffusion model that pulls in `diffusers.models.attention.BasicTransformerBlock`.
// `x_tc` is `[t, c]`; norms run along the channel dim. Q/K/V have no bias;
// only `out` and the FFN linears carry biases. `head_dim * num_heads` must
// equal the inner attention dim implied by `qw`'s rows.
ggml_tensor * codec_op_basic_transformer_block_tc(
    ggml_context * ctx,
    ggml_tensor * x_tc,
    ggml_tensor * norm1_w, ggml_tensor * norm1_b,
    ggml_tensor * qw, ggml_tensor * kw, ggml_tensor * vw,
    ggml_tensor * ow, ggml_tensor * ob,
    ggml_tensor * norm3_w, ggml_tensor * norm3_b,
    ggml_tensor * ff1_w, ggml_tensor * ff1_b,
    ggml_tensor * ff2_w, ggml_tensor * ff2_b,
    int32_t head_dim,
    int32_t num_heads);

// Whisper / `nn.MultiheadAttention`-style encoder layer (HF Whisper, XY-Tokenizer
// `OmniWhisperTransformerLayer`, etc.):
//   x = x + out_proj(attn(LN(x)))     (q has bias, k bias-free, v has bias)
//   x = x + fc2(GELU-erf(fc1(LN(x)))) (both fc1 and fc2 carry biases)
//
// Non-causal, sliceable via `n_valid` (when 0 < n_valid < t, attention scores
// for keys at positions ≥ n_valid are -inf-masked and rows for queries
// ≥ n_valid are zeroed — matches HF's `valid_q`/`valid_k` SDPA bias path).
// Pass `n_valid = 0` (or t) to disable masking.  GELU is the *exact* erf-based
// variant (matches PyTorch `F.gelu` with default `approximate='none'`).
//
// `x_tc` is `[t, c=hidden]`.  Returns `[t, c=hidden]`.
ggml_tensor * codec_op_whisper_encoder_layer_tc(
    ggml_context * ctx,
    ggml_tensor * x_tc,
    ggml_tensor * n1w, ggml_tensor * n1b,
    ggml_tensor * qw,  ggml_tensor * qb,
    ggml_tensor * kw,
    ggml_tensor * vw,  ggml_tensor * vb,
    ggml_tensor * ow,  ggml_tensor * ob,
    ggml_tensor * n2w, ggml_tensor * n2b,
    ggml_tensor * fc1w, ggml_tensor * fc1b,
    ggml_tensor * fc2w, ggml_tensor * fc2b,
    int32_t head_dim,
    int32_t n_heads,
    int32_t n_valid);

// Slice the first `t = x_tc->ne[0]` rows of `pos` (saved with PyTorch shape
// `(max_pos, d_model)`, hence ggml ne=(d_model, max_pos)) and add to `x_tc`.
// Returns `x_tc + pos[:t]`.  Fundamental "sinusoidal pos-emb add" pattern
// shared by Whisper-style encoders.
ggml_tensor * codec_op_add_sliced_pos_emb_tc(
    ggml_context * ctx,
    ggml_tensor * x_tc,
    ggml_tensor * pos);

// L2-normalize each frame of an `[t, c]` tensor along the channel axis
// (`torch.nn.functional.normalize(x, dim=1)` on a (B, C, T) tensor).
// Used by every cosine-NN RVQ codec (SNAC, MOSS-Audio, …).
ggml_tensor * codec_op_l2_normalize_tc(
    ggml_context * ctx,
    ggml_tensor * x_tc,
    float eps);

// Sinusoidal time embedding (Diffusion / flow-matching SinusoidalPosEmb).
// Builds the embedding entirely in-graph from `ggml_arange + sin/cos`:
//   half = dim/2;  freq[k] = exp(-k * log(10000)/(half-1))
//   emb  = concat(sin(scale*t*freq), cos(scale*t*freq))   length = dim.
// `t_v` is a compile-time constant scalar; the graph bakes it into a
// `ggml_scale`. Output is a 1-D tensor of length `dim`.
ggml_tensor * codec_op_sinusoidal_time_emb(
    ggml_context * ctx,
    float t_v,
    int32_t dim,
    float scale);

// BigVGAN-style anti-aliased SnakeBeta activation (`alias_free_torch.Activation1d`).
// Wraps `snake_beta` between a 2× Kaiser-FIR upsample and a matching downsample
// so the non-linearity is computed at twice the input rate, suppressing
// aliasing.  `x_tc` is `[t, c]`; output is also `[t, c]` (same length).
// `alpha` and `inv_beta` are per-channel `[c]` (already exp-baked at convert
// time).  `kernel_12` is the shared 12-tap symmetric Kaiser-sinc kernel
// (palindromic and identical for up/down, see BigCodec checkpoint).
ggml_tensor * codec_op_alias_free_snake_beta_tc(
    ggml_context * ctx,
    ggml_tensor * x_tc,
    ggml_tensor * alpha,
    ggml_tensor * inv_beta,
    ggml_tensor * kernel_12);

// Vocos-style ResnetBlock1D: GroupNorm(32) → SiLU → Conv1d(k=3, p=1) → GroupNorm(32) → SiLU → Conv1d(k=3, p=1) + residual.
// `x_tc` is `[t, c]`; both convs are kernel_size=3, stride=1, dilation=1, padding=1
// (i.e. preserves time). Used by the Vocos backbone (prior_net + post_net) in
// xcodec2 / NeuCodec / similar Vocos vocoders.
ggml_tensor * codec_op_vocos_resnet_block_tc(
    ggml_context * ctx,
    ggml_tensor * x_tc,
    ggml_tensor * n1_w, ggml_tensor * n1_b,
    ggml_tensor * c1_w, ggml_tensor * c1_b,
    ggml_tensor * n2_w, ggml_tensor * n2_b,
    ggml_tensor * c2_w, ggml_tensor * c2_b);

// bs_roformer-style transformer block (RMSNorm pre-attn, RoPE on Q/K, full
// self-attention, MLP fc1→SiLU→fc2 with no bias). The attention matrix uses a
// combined `c_attn` (`[3*h*d, h*d]`) as in Karpathy nanoGPT and the head dim
// equals the RoPE dim. `x_ct` is `[c, t]`; output is `[c, t]`.
ggml_tensor * codec_op_roformer_block_ct(
    ggml_context * ctx,
    ggml_tensor * x_ct,
    ggml_tensor * att_norm_w,
    ggml_tensor * ffn_norm_w,
    ggml_tensor * c_attn_w,
    ggml_tensor * c_proj_w,
    ggml_tensor * fc1_w,
    ggml_tensor * fc2_w,
    int32_t head_dim,
    int32_t n_heads,
    float rope_theta);

// Espnet-style relative positional encoding for a Conformer with `T` query
// positions, built entirely in-graph. Output ne = (d_model, 2T-1):
//   row r covers position p_r ∈ [T-1, T-2, ..., 0, -1, ..., -(T-1)]
//   pe[r, 2k]   = sin(p_r * freq[k])
//   pe[r, 2k+1] = cos(p_r * freq[k])
// where freq[k] = exp(-2k * log(10000)/d_model). The interleaved sin/cos
// layout matches the Espnet `pe[:, 0::2] / 1::2` convention so a Linear(d, d)
// (`linear_pos`) trained on that ordering applies directly.
ggml_tensor * codec_op_espnet_rel_pos_emb(
    ggml_context * ctx,
    int32_t t,
    int32_t d_model);

#endif
