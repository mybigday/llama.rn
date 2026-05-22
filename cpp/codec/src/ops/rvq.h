#ifndef CODEC_OPS_RVQ_H
#define CODEC_OPS_RVQ_H

#include "../codec_internal.h"

struct codec_rvq_layer_result_ggml {
    lm_ggml_tensor * indices = nullptr;  // [t] I32
    lm_ggml_tensor * residual = nullptr; // [d, t] F32
};

lm_ggml_tensor * codec_rvq_argmin_map_custom1(
    lm_ggml_context * ctx_eval,
    lm_ggml_tensor * distances_ct);

bool codec_rvq_build_layer_ggml(
    lm_ggml_context * ctx_eval,
    lm_ggml_tensor * residual_ct,
    lm_ggml_tensor * codebook_dc,
    codec_rvq_layer_result_ggml * out);

#endif // CODEC_OPS_RVQ_H
