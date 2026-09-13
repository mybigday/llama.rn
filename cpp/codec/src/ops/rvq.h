#ifndef CODEC_OPS_RVQ_H
#define CODEC_OPS_RVQ_H

#include "../codec_internal.h"

struct codec_rvq_layer_result_ggml {
    ggml_tensor * indices = nullptr;  // [t] I32
    ggml_tensor * residual = nullptr; // [d, t] F32
};

ggml_tensor * codec_rvq_argmin_map_custom1(
    ggml_context * ctx_eval,
    ggml_tensor * distances_ct);

ggml_tensor * codec_rvq_select_indices_ggml(
    ggml_context * ctx_eval,
    ggml_tensor * residual_ct,
    ggml_tensor * codebook_dc);

bool codec_rvq_build_layer_ggml(
    ggml_context * ctx_eval,
    ggml_tensor * residual_ct,
    ggml_tensor * codebook_dc,
    codec_rvq_layer_result_ggml * out);

#endif // CODEC_OPS_RVQ_H
