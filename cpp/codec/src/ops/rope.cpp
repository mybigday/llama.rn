#include "rope.h"

lm_ggml_tensor * codec_op_rope(
    lm_ggml_context * ctx,
    lm_ggml_tensor * x_dth,
    int32_t n_dims,
    float freq_base,
    float freq_scale) {
    if (ctx == nullptr || x_dth == nullptr) {
        return nullptr;
    }

    const int64_t d = x_dth->ne[0];
    const int64_t t = x_dth->ne[1];
    if (d <= 0 || t <= 0 || x_dth->ne[2] <= 0 || n_dims <= 0 || n_dims % 2 != 0 || n_dims > d) {
        return nullptr;
    }
    if (freq_base <= 0.0f || freq_scale <= 0.0f) {
        return nullptr;
    }

    lm_ggml_tensor * t_pos = lm_ggml_cast(ctx, lm_ggml_arange(ctx, 0.0f, (float) t, 1.0f), LM_GGML_TYPE_I32);
    if (t_pos == nullptr) {
        return nullptr;
    }

    lm_ggml_tensor * x_dht = lm_ggml_permute(ctx, x_dth, 0, 2, 1, 3); // [d, h, t]
    lm_ggml_tensor * y_dht = lm_ggml_rope_ext(
        ctx,
        x_dht,
        t_pos,
        nullptr,
        n_dims,
        LM_GGML_ROPE_TYPE_NORMAL,
        0,
        freq_base,
        freq_scale,
        0.0f,
        1.0f,
        0.0f,
        0.0f);
    if (y_dht == nullptr) {
        return nullptr;
    }

    return lm_ggml_cont(ctx, lm_ggml_permute(ctx, y_dht, 0, 2, 1, 3)); // [d, t, h]
}
