#include "local_attn.h"

#include <ggml.h>

#include <algorithm>
#include <cmath>

lm_ggml_tensor * codec_op_local_attn(
    lm_ggml_context * ctx,
    lm_ggml_tensor * q_dth,
    lm_ggml_tensor * k_dth,
    lm_ggml_tensor * v_dth,
    lm_ggml_tensor * score_bias_kqh,
    int32_t head_dim,
    int32_t n_heads) {

    if (ctx == nullptr || q_dth == nullptr || k_dth == nullptr || v_dth == nullptr ||
        score_bias_kqh == nullptr || head_dim <= 0 || n_heads <= 0) {
        return nullptr;
    }

    const float scale = 1.0f / std::sqrt((float) head_dim);
    lm_ggml_tensor * k_cont = lm_ggml_cont(ctx, k_dth);
    lm_ggml_tensor * scores = lm_ggml_mul_mat(ctx, k_cont, q_dth);   // (t_k, t_q, h)
    if (scores == nullptr) return nullptr;
    scores = lm_ggml_scale_inplace(ctx, scores, scale);
    scores = lm_ggml_add(ctx, scores, score_bias_kqh);            // causal + block + rel-pos

    lm_ggml_tensor * probs = lm_ggml_soft_max(ctx, scores);
    if (probs == nullptr) return nullptr;
    lm_ggml_tensor * v_tdh = lm_ggml_cont(ctx, lm_ggml_permute(ctx, v_dth, 1, 0, 2, 3));
    return lm_ggml_mul_mat(ctx, v_tdh, probs);                     // (d, t, h)
}

void codec_local_attn_fill_mask(
    const codec_local_attn_params * params,
    int32_t t,
    float * out) {

    if (params == nullptr || out == nullptr || t <= 0) return;

    const int32_t heads     = std::max(1, params->heads);
    const int32_t attn_span = std::max(2, params->window);
    const int32_t W         = std::max(1, attn_span / 2);    // block size
    const int32_t max_dist  = std::max(1, params->max_dist);
    const float * bias      = params->bias;

    // Layout: data[k + q*t + h*t*t].
    for (int32_t h = 0; h < heads; ++h) {
        for (int32_t q = 0; q < t; ++q) {
            const int32_t i_in    = q % W;
            const int32_t max_d   = i_in + W;
            const int32_t k_lo    = std::max(0, q - max_d);   // earliest allowed key
            float * row           = out + (size_t) h * t * t + (size_t) q * t;
            for (int32_t k = 0; k < t; ++k) {
                float v;
                if (k > q || k < k_lo) {
                    v = -INFINITY;
                } else {
                    const int32_t d = q - k;                  // d in [0, max_d]
                    v = (bias != nullptr && d < max_dist)
                            ? bias[(size_t) h * (size_t) max_dist + (size_t) d]
                            : 0.0f;
                }
                row[k] = v;
            }
        }
    }
}
