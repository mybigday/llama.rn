#pragma once

#include "../codec_internal.h"

struct codec_lm_attn_params {
    float scale = 0.0f;   // if <= 0, use 1 / sqrt(head_dim)
    bool causal = false;
    // Optional sliding-window limit on causal attention.  If `>0`, query
    // position i attends only to keys j ∈ [i - window + 1, i].  Set to 0
    // (default) to disable; only meaningful when `causal` is true.  Used by
    // MOSS-Audio-Tokenizer's per-block context budgets.
    int32_t window = 0;
    // Optional padding mask: if `n_valid > 0` and `n_valid < t`, attention
    // scores for keys at positions `>= n_valid` are set to -inf, and the
    // output rows for queries at positions `>= n_valid` are forced to zero.
    // Set to 0 (default) to disable.  Used by MOSS-Audio-Tokenizer to honour
    // `input_lengths` after PCM padding.
    int32_t n_valid = 0;
};

// q_dth, k_dth, v_dth are [head_dim, t, n_heads]
// returns context tensor [head_dim, t, n_heads]
lm_ggml_tensor * codec_op_lm_attn_ctx_dth(
    lm_ggml_context * ctx,
    lm_ggml_tensor * q_dth,
    lm_ggml_tensor * k_dth,
    lm_ggml_tensor * v_dth,
    const codec_lm_attn_params * params);

// Espnet rel-shift trick. Input has ne[0]=2t-1, ne[1]=t, ne[2]=heads — the
// matrix-BD shape from a Conformer self-attention with relative positional
// encoding (matrix_ac shape (t, t, h) ≠ matrix_bd shape (2t-1, t, h)).
// Returns ne[0]=t, ne[1]=t, ne[2]=heads via the standard zero-pad / view-shift
// permutation described in https://arxiv.org/abs/1901.02860 §B.
lm_ggml_tensor * codec_op_rel_shift_espnet(lm_ggml_context * ctx, lm_ggml_tensor * x);

// Conformer self-attention with Espnet relative positional encoding.
// Inputs:
//   q_dth, k_dth, v_dth: [head_dim, t, n_heads]
//   p_dth (linear_pos(pos_emb) reshaped to per-head): [head_dim, 2t-1, n_heads]
//   pos_bias_u, pos_bias_v: [head_dim, n_heads]  (broadcast over t)
// Returns context tensor [head_dim, t, n_heads].
lm_ggml_tensor * codec_op_lm_attn_rel_pos_dth(
    lm_ggml_context * ctx,
    lm_ggml_tensor * q_dth,
    lm_ggml_tensor * k_dth,
    lm_ggml_tensor * v_dth,
    lm_ggml_tensor * p_dth,
    lm_ggml_tensor * pos_bias_u,
    lm_ggml_tensor * pos_bias_v,
    const codec_lm_attn_params * params);

// Shaw-style relative-key self-attention (used by Wav2Vec2-Bert /
// SeamlessM4T conformer). Adds Q · D[bucket(j-i)] / sqrt(d) on top of the
// standard q@k.T / sqrt(d) attention scores, where D is a learned distance
// embedding shared across heads.
//   q_dth, k_dth, v_dth: [head_dim, t, n_heads]
//   dist_emb_dn:         [head_dim, n_buckets]  (n_buckets = left_max+right_max+1)
//   bucket_idx_1d:       int32 [t*t]  with row-major layout (t_k inner, t_q outer)
//                        and value `clamp(t_k - t_q, -left_max, right_max) + left_max`.
// Returns context tensor [head_dim, t, n_heads].
lm_ggml_tensor * codec_op_lm_attn_rel_key_dth(
    lm_ggml_context * ctx,
    lm_ggml_tensor * q_dth,
    lm_ggml_tensor * k_dth,
    lm_ggml_tensor * v_dth,
    lm_ggml_tensor * dist_emb_dn,
    lm_ggml_tensor * bucket_idx_1d,
    const codec_lm_attn_params * params);


// Per-position / shared Linear projection.
//
// Applies `out[:, t] = w_t @ x[:, t]` to an `(in_dim, T)` sequence and
// returns an `(out_dim, T)` 2D tensor.  The weight `w` may be either:
//   - 2D `(in_dim, out_dim)`: SHARED across all T positions.  Math
//     reduces to a plain `lm_ggml_mul_mat(w, x)`.
//   - 3D `(in_dim, out_dim, N)` with `N >= T`: PER-POSITION (Moshi
//     flexible weights, LFM2-Audio per-position in_proj).  The op
//     slices the first T weight positions and applies per-pos via a
//     batched mul_mat (with the input as the LHS operand because
//     ggml's batch-broadcast rule only lets `a` broadcast).
//
// Casts `w` to F32 internally if it isn't already; the caller can pass
// the raw GGUF weight tensor.  Used by codec_lm depth-decoder graphs
// to keep the shared and flexible runtime paths on one helper.
lm_ggml_tensor * codec_op_lm_per_pos_linear(
    lm_ggml_context * ctx,
    lm_ggml_tensor * w,
    lm_ggml_tensor * x_2d,
    int32_t out_dim,
    int32_t T);


// Llama-style depth-decoder transformer block:
//   x = x + o_proj(GQA-attn(RMSNorm_1(x) -> q/k/v, optional QK-norm,
//                           optional RoPE))
//   x = x + ffn_down(SiLU(ffn_gate(RMSNorm_2(x))) * ffn_up(RMSNorm_2(x)))
//
// All of `qw / kw / vw / ow / ffn_gate / ffn_up / ffn_down` may be 2D
// (shared across positions, CSM/Qwen3-TTS/LFM2-Audio) or 3D
// `(in_dim, out_dim, N>=T)` (per-position, Moshi flexible) —
// `codec_op_lm_per_pos_linear` handles either case internally.  Norm
// weights are 1D: `head_dim` for q/k-norm, `hidden` for attn / ffn
// norm.  Weights may be F16 / quantized; the op casts them to F32 on
// the fly via `codec_graph_cast_f32`.
//
// Optional knobs:
//   - `q_norm_w` / `k_norm_w`: per-head RMSNorm on q/k (Qwen3 family,
//     LFM2-Audio).  Pass `nullptr` to skip.
//   - `use_rope`: when true, `lm_ggml_rope_ext` is applied to q and k with
//     mode `rope_mode` (`LM_GGML_ROPE_TYPE_NEOX` for Llama family,
//     `LM_GGML_ROPE_TYPE_NORMAL` for GPT-J pair layout / LFM2-Audio),
//     positions `t_pos`, and optional `freq_factors` (for llama3 RoPE
//     scaling).  Pass `false` (and `nullptr` t_pos / freq_factors) to
//     skip RoPE entirely (Moshi).
//
// Input `x_ht` shape is `(hidden, T)`; output same.  GQA: `n_heads`
// must be a multiple of `n_kv_heads`.
lm_ggml_tensor * codec_op_lm_llama_depth_block(
    lm_ggml_context * ctx,
    lm_ggml_tensor * x_ht,
    lm_ggml_tensor * attn_norm_w,
    lm_ggml_tensor * qw, lm_ggml_tensor * kw, lm_ggml_tensor * vw, lm_ggml_tensor * ow,
    lm_ggml_tensor * q_norm_w, lm_ggml_tensor * k_norm_w,
    lm_ggml_tensor * t_pos, lm_ggml_tensor * freq_factors,
    lm_ggml_tensor * ffn_norm_w,
    lm_ggml_tensor * ffn_gate, lm_ggml_tensor * ffn_up, lm_ggml_tensor * ffn_down,
    int32_t head_dim,
    int32_t n_heads,
    int32_t n_kv_heads,
    float   rope_theta,
    float   rms_eps,
    int32_t rope_mode,
    bool    use_rope);

