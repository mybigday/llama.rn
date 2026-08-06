#ifndef CODEC_OPS_ROPE_H
#define CODEC_OPS_ROPE_H

#include "../codec_internal.h"

enum codec_rope_mode {
    CODEC_ROPE_MODE_NORMAL = LM_GGML_ROPE_TYPE_NORMAL,
    CODEC_ROPE_MODE_NEOX   = LM_GGML_ROPE_TYPE_NEOX,
};

lm_ggml_tensor * codec_op_rope(
    lm_ggml_context * ctx,
    lm_ggml_tensor * x_dth,
    int32_t n_dims,
    float freq_base,
    float freq_scale,
    codec_rope_mode mode);

#endif
