#ifndef CODEC_OPS_LM_GGML_OPS_H
#define CODEC_OPS_LM_GGML_OPS_H

#include "../codec_internal.h"

enum codec_unary_op {
    CODEC_UNARY_SIGMOID = 0,
    CODEC_UNARY_ELU = 1,
    CODEC_UNARY_SILU = 2,
    CODEC_UNARY_GELU_ERF = 3,
};

lm_ggml_tensor * codec_op_unary(lm_ggml_context * ctx, lm_ggml_tensor * x, codec_unary_op op);
lm_ggml_tensor * codec_op_layer_norm(lm_ggml_context * ctx, lm_ggml_tensor * x, float eps, lm_ggml_tensor * gamma, lm_ggml_tensor * beta);
lm_ggml_tensor * codec_op_group_norm(lm_ggml_context * ctx, lm_ggml_tensor * x, int32_t n_groups, float eps, lm_ggml_tensor * gamma, lm_ggml_tensor * beta);
lm_ggml_tensor * codec_op_linear(lm_ggml_context * ctx, lm_ggml_tensor * x, lm_ggml_tensor * w, lm_ggml_tensor * b);
lm_ggml_tensor * codec_op_snake(lm_ggml_context * ctx, lm_ggml_tensor * x, lm_ggml_tensor * alpha, float eps);
lm_ggml_tensor * codec_op_snake_beta(lm_ggml_context * ctx, lm_ggml_tensor * x, lm_ggml_tensor * alpha, lm_ggml_tensor * inv_beta, float eps);
lm_ggml_tensor * codec_op_pad_1d(lm_ggml_context * ctx, lm_ggml_tensor * x, int32_t pad_left, int32_t pad_right);
lm_ggml_tensor * codec_op_pad_1d_replicate(lm_ggml_context * ctx, lm_ggml_tensor * x, int32_t pad_left, int32_t pad_right);
lm_ggml_tensor * codec_op_causal_crop_1d(lm_ggml_context * ctx, lm_ggml_tensor * x, int32_t target_t);
lm_ggml_tensor * codec_op_crop_1d(lm_ggml_context * ctx, lm_ggml_tensor * x, int32_t crop_left, int32_t crop_right);
lm_ggml_tensor * codec_op_channel_scale(lm_ggml_context * ctx, lm_ggml_tensor * x, lm_ggml_tensor * scale);

lm_ggml_tensor * codec_op_tokens_to_features(lm_ggml_context * ctx, lm_ggml_tensor * tokens, int32_t out_channels);

#endif
