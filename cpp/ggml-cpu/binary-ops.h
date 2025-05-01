#pragma once

#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

void lm_ggml_compute_forward_add_non_quantized(const struct lm_ggml_compute_params * params, struct lm_ggml_tensor * dst);
void lm_ggml_compute_forward_sub(const struct lm_ggml_compute_params * params, struct lm_ggml_tensor * dst);
void lm_ggml_compute_forward_mul(const struct lm_ggml_compute_params * params, struct lm_ggml_tensor * dst);
void lm_ggml_compute_forward_div(const struct lm_ggml_compute_params * params, struct lm_ggml_tensor * dst);

#ifdef __cplusplus
}
#endif
