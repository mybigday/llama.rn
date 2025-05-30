#pragma once

#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

void lm_ggml_compute_forward_abs(const struct lm_ggml_compute_params * params, struct lm_ggml_tensor * dst);
void lm_ggml_compute_forward_sgn(const struct lm_ggml_compute_params * params, struct lm_ggml_tensor * dst);
void lm_ggml_compute_forward_neg(const struct lm_ggml_compute_params * params, struct lm_ggml_tensor * dst);
void lm_ggml_compute_forward_step(const struct lm_ggml_compute_params * params, struct lm_ggml_tensor * dst);
void lm_ggml_compute_forward_tanh(const struct lm_ggml_compute_params * params, struct lm_ggml_tensor * dst);
void lm_ggml_compute_forward_elu(const struct lm_ggml_compute_params * params, struct lm_ggml_tensor * dst);
void lm_ggml_compute_forward_relu(const struct lm_ggml_compute_params * params, struct lm_ggml_tensor * dst);
void lm_ggml_compute_forward_sigmoid(const struct lm_ggml_compute_params * params, struct lm_ggml_tensor * dst);
void lm_ggml_compute_forward_hardsigmoid(const struct lm_ggml_compute_params * params, struct lm_ggml_tensor * dst);
void lm_ggml_compute_forward_exp(const struct lm_ggml_compute_params * params, struct lm_ggml_tensor * dst);
void lm_ggml_compute_forward_hardswish(const struct lm_ggml_compute_params * params, struct lm_ggml_tensor * dst);
void lm_ggml_compute_forward_sqr(const struct lm_ggml_compute_params * params, struct lm_ggml_tensor * dst);
void lm_ggml_compute_forward_sqrt(const struct lm_ggml_compute_params * params, struct lm_ggml_tensor * dst);
void lm_ggml_compute_forward_sin(const struct lm_ggml_compute_params * params, struct lm_ggml_tensor * dst);
void lm_ggml_compute_forward_cos(const struct lm_ggml_compute_params * params, struct lm_ggml_tensor * dst);
void lm_ggml_compute_forward_log(const struct lm_ggml_compute_params * params, struct lm_ggml_tensor * dst);

#ifdef __cplusplus
}
#endif
