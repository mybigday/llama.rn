#ifndef CODEC_OPS_LOCAL_ATTN_H
#define CODEC_OPS_LOCAL_ATTN_H

#include <ggml.h>
#include <cstdint>

// Per-block local-causal attention (NeuCodec distill).  For each query at
// position i (block b = i / W, where W = `window/2`) the valid keys are
// `[(b-1)·W, i]` plus an optional per-head relative-position bias
// `bias[h, |i - j|]` (zero outside `max_dist`).
struct codec_local_attn_params {
    const float * bias = nullptr; // [heads, max_dist]
    int32_t heads = 0;
    int32_t head_dim = 0;
    int32_t window = 0;           // full attention span (= 2 · block size)
    int32_t max_dist = 0;
};

// SDPA with a precomputed score bias added to `(K · Q^T) / sqrt(d)` before
// softmax.  `score_bias_kqh` has shape `(t_k, t_q, n_heads)` — fill it via
// `codec_local_attn_fill_mask` (CPU-side, before graph compute).  Backed
// purely by ggml ops so the attention can offload to the active backend.
lm_ggml_tensor * codec_op_local_attn(
    lm_ggml_context * ctx,
    lm_ggml_tensor * q_dth,
    lm_ggml_tensor * k_dth,
    lm_ggml_tensor * v_dth,
    lm_ggml_tensor * score_bias_kqh,
    int32_t head_dim,
    int32_t n_heads);

// CPU-side filler for the score-bias tensor consumed by codec_op_local_attn.
// `out` is a host buffer of `t * t * params->heads` floats laid out as
// `data[k + q*t + h*t*t]` (matching ggml ne=(t_k, t_q, heads)).  Fills with
// `params->bias[h, |q-k|]` (zero past `max_dist`) for valid (q, k) pairs and
// -INFINITY for pairs outside the causal block window or with `k > q`.
void codec_local_attn_fill_mask(
    const codec_local_attn_params * params,
    int32_t t,
    float * out);

#endif
