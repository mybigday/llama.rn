#pragma once

#include "ggml.h"

// GGML internal header

#ifdef __cplusplus
extern "C" {
#endif

// Quantization
void quantize_mat_q8_0(const float * LM_GGML_RESTRICT x, void * LM_GGML_RESTRICT y, int64_t nrows, int64_t n_per_row, int64_t blck_size_interleave);

// GEMV
void lm_ggml_gemv_q4_0_4x4_q8_0(int n, float * LM_GGML_RESTRICT s, size_t bs, const void * LM_GGML_RESTRICT vx, const void * LM_GGML_RESTRICT vy, int nr, int nc);
void lm_ggml_gemv_q4_0_4x8_q8_0(int n, float * LM_GGML_RESTRICT s, size_t bs, const void * LM_GGML_RESTRICT vx, const void * LM_GGML_RESTRICT vy, int nr, int nc);
void lm_ggml_gemv_q4_0_8x8_q8_0(int n, float * LM_GGML_RESTRICT s, size_t bs, const void * LM_GGML_RESTRICT vx, const void * LM_GGML_RESTRICT vy, int nr, int nc);

// GEMM
void lm_ggml_gemm_q4_0_4x4_q8_0(int n, float * LM_GGML_RESTRICT s, size_t bs, const void * LM_GGML_RESTRICT vx, const void * LM_GGML_RESTRICT vy, int nr, int nc);
void lm_ggml_gemm_q4_0_4x8_q8_0(int n, float * LM_GGML_RESTRICT s, size_t bs, const void * LM_GGML_RESTRICT vx, const void * LM_GGML_RESTRICT vy, int nr, int nc);
void lm_ggml_gemm_q4_0_8x8_q8_0(int n, float * LM_GGML_RESTRICT s, size_t bs, const void * LM_GGML_RESTRICT vx, const void * LM_GGML_RESTRICT vy, int nr, int nc);

void           lm_ggml_aarch64_repack_tensor(struct lm_ggml_tensor * cur, enum lm_ggml_type repack_type, const void * data, size_t data_size);
enum lm_ggml_type lm_ggml_aarch64_get_optimal_repack_type(const struct lm_ggml_tensor * cur);

#ifdef __cplusplus
}
#endif

