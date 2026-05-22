#ifndef CODEC_OPS_ROPE_H
#define CODEC_OPS_ROPE_H

#include "../codec_internal.h"

lm_ggml_tensor * codec_op_rope(
    lm_ggml_context * ctx,
    lm_ggml_tensor * x_dth,
    int32_t n_dims,
    float freq_base,
    float freq_scale);

#endif
