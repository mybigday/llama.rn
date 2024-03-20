#pragma once

#define LM_GGML_COMMON_DECL_C
#include "ggml-common.h"

#include "ggml.h"

// GGML internal header

#ifdef __cplusplus
extern "C" {
#endif

// Quantization
void quantize_row_q4_0_reference(const float * LM_GGML_RESTRICT x, block_q4_0 * LM_GGML_RESTRICT y, int k);
void quantize_row_q4_1_reference(const float * LM_GGML_RESTRICT x, block_q4_1 * LM_GGML_RESTRICT y, int k);
void quantize_row_q5_0_reference(const float * LM_GGML_RESTRICT x, block_q5_0 * LM_GGML_RESTRICT y, int k);
void quantize_row_q5_1_reference(const float * LM_GGML_RESTRICT x, block_q5_1 * LM_GGML_RESTRICT y, int k);
void quantize_row_q8_0_reference(const float * LM_GGML_RESTRICT x, block_q8_0 * LM_GGML_RESTRICT y, int k);
void quantize_row_q8_1_reference(const float * LM_GGML_RESTRICT x, block_q8_1 * LM_GGML_RESTRICT y, int k);

void quantize_row_q2_K_reference(const float * LM_GGML_RESTRICT x, block_q2_K * LM_GGML_RESTRICT y, int k);
void quantize_row_q3_K_reference(const float * LM_GGML_RESTRICT x, block_q3_K * LM_GGML_RESTRICT y, int k);
void quantize_row_q4_K_reference(const float * LM_GGML_RESTRICT x, block_q4_K * LM_GGML_RESTRICT y, int k);
void quantize_row_q5_K_reference(const float * LM_GGML_RESTRICT x, block_q5_K * LM_GGML_RESTRICT y, int k);
void quantize_row_q6_K_reference(const float * LM_GGML_RESTRICT x, block_q6_K * LM_GGML_RESTRICT y, int k);
void quantize_row_q8_K_reference(const float * LM_GGML_RESTRICT x, block_q8_K * LM_GGML_RESTRICT y, int k);

void quantize_row_iq3_xxs_reference(const float * LM_GGML_RESTRICT x, block_iq3_xxs * LM_GGML_RESTRICT y, int k);
void quantize_row_iq4_nl_reference (const float * LM_GGML_RESTRICT x, block_iq4_nl  * LM_GGML_RESTRICT y, int k);
void quantize_row_iq4_xs_reference (const float * LM_GGML_RESTRICT x, block_iq4_xs  * LM_GGML_RESTRICT y, int k);
void quantize_row_iq3_s_reference  (const float * LM_GGML_RESTRICT x, block_iq3_s   * LM_GGML_RESTRICT y, int k);
void quantize_row_iq2_s_reference  (const float * LM_GGML_RESTRICT x, block_iq2_s   * LM_GGML_RESTRICT y, int k);

void quantize_row_q4_0(const float * LM_GGML_RESTRICT x, void * LM_GGML_RESTRICT y, int k);
void quantize_row_q4_1(const float * LM_GGML_RESTRICT x, void * LM_GGML_RESTRICT y, int k);
void quantize_row_q5_0(const float * LM_GGML_RESTRICT x, void * LM_GGML_RESTRICT y, int k);
void quantize_row_q5_1(const float * LM_GGML_RESTRICT x, void * LM_GGML_RESTRICT y, int k);
void quantize_row_q8_0(const float * LM_GGML_RESTRICT x, void * LM_GGML_RESTRICT y, int k);
void quantize_row_q8_1(const float * LM_GGML_RESTRICT x, void * LM_GGML_RESTRICT y, int k);

void quantize_row_q2_K(const float * LM_GGML_RESTRICT x, void * LM_GGML_RESTRICT y, int k);
void quantize_row_q3_K(const float * LM_GGML_RESTRICT x, void * LM_GGML_RESTRICT y, int k);
void quantize_row_q4_K(const float * LM_GGML_RESTRICT x, void * LM_GGML_RESTRICT y, int k);
void quantize_row_q5_K(const float * LM_GGML_RESTRICT x, void * LM_GGML_RESTRICT y, int k);
void quantize_row_q6_K(const float * LM_GGML_RESTRICT x, void * LM_GGML_RESTRICT y, int k);
void quantize_row_q8_K(const float * LM_GGML_RESTRICT x, void * LM_GGML_RESTRICT y, int k);

void quantize_row_iq3_xxs(const float * LM_GGML_RESTRICT x, void * LM_GGML_RESTRICT y, int k);
void quantize_row_iq4_nl (const float * LM_GGML_RESTRICT x, void * LM_GGML_RESTRICT y, int k);
void quantize_row_iq4_xs (const float * LM_GGML_RESTRICT x, void * LM_GGML_RESTRICT y, int k);
void quantize_row_iq3_s  (const float * LM_GGML_RESTRICT x, void * LM_GGML_RESTRICT y, int k);
void quantize_row_iq2_s  (const float * LM_GGML_RESTRICT x, void * LM_GGML_RESTRICT y, int k);

// Dequantization
void dequantize_row_q4_0(const block_q4_0 * LM_GGML_RESTRICT x, float * LM_GGML_RESTRICT y, int k);
void dequantize_row_q4_1(const block_q4_1 * LM_GGML_RESTRICT x, float * LM_GGML_RESTRICT y, int k);
void dequantize_row_q5_0(const block_q5_0 * LM_GGML_RESTRICT x, float * LM_GGML_RESTRICT y, int k);
void dequantize_row_q5_1(const block_q5_1 * LM_GGML_RESTRICT x, float * LM_GGML_RESTRICT y, int k);
void dequantize_row_q8_0(const block_q8_0 * LM_GGML_RESTRICT x, float * LM_GGML_RESTRICT y, int k);
//void dequantize_row_q8_1(const block_q8_1 * LM_GGML_RESTRICT x, float * LM_GGML_RESTRICT y, int k);

void dequantize_row_q2_K(const block_q2_K * LM_GGML_RESTRICT x, float * LM_GGML_RESTRICT y, int k);
void dequantize_row_q3_K(const block_q3_K * LM_GGML_RESTRICT x, float * LM_GGML_RESTRICT y, int k);
void dequantize_row_q4_K(const block_q4_K * LM_GGML_RESTRICT x, float * LM_GGML_RESTRICT y, int k);
void dequantize_row_q5_K(const block_q5_K * LM_GGML_RESTRICT x, float * LM_GGML_RESTRICT y, int k);
void dequantize_row_q6_K(const block_q6_K * LM_GGML_RESTRICT x, float * LM_GGML_RESTRICT y, int k);
void dequantize_row_q8_K(const block_q8_K * LM_GGML_RESTRICT x, float * LM_GGML_RESTRICT y, int k);

void dequantize_row_iq2_xxs(const block_iq2_xxs * LM_GGML_RESTRICT x, float * LM_GGML_RESTRICT y, int k);
void dequantize_row_iq2_xs (const block_iq2_xs  * LM_GGML_RESTRICT x, float * LM_GGML_RESTRICT y, int k);
void dequantize_row_iq2_s  (const block_iq2_s   * LM_GGML_RESTRICT x, float * LM_GGML_RESTRICT y, int k);
void dequantize_row_iq3_xxs(const block_iq3_xxs * LM_GGML_RESTRICT x, float * LM_GGML_RESTRICT y, int k);
void dequantize_row_iq1_s  (const block_iq1_s   * LM_GGML_RESTRICT x, float * LM_GGML_RESTRICT y, int k);
void dequantize_row_iq4_nl (const block_iq4_nl  * LM_GGML_RESTRICT x, float * LM_GGML_RESTRICT y, int k);
void dequantize_row_iq4_xs (const block_iq4_xs  * LM_GGML_RESTRICT x, float * LM_GGML_RESTRICT y, int k);
void dequantize_row_iq3_s  (const block_iq3_s   * LM_GGML_RESTRICT x, float * LM_GGML_RESTRICT y, int k);

// Dot product
void lm_ggml_vec_dot_q4_0_q8_0(int n, float * LM_GGML_RESTRICT s, size_t bs, const void * LM_GGML_RESTRICT vx, size_t bx, const void * LM_GGML_RESTRICT vy, size_t by, int nrc);
void lm_ggml_vec_dot_q4_1_q8_1(int n, float * LM_GGML_RESTRICT s, size_t bs, const void * LM_GGML_RESTRICT vx, size_t bx, const void * LM_GGML_RESTRICT vy, size_t by, int nrc);
void lm_ggml_vec_dot_q5_0_q8_0(int n, float * LM_GGML_RESTRICT s, size_t bs, const void * LM_GGML_RESTRICT vx, size_t bx, const void * LM_GGML_RESTRICT vy, size_t by, int nrc);
void lm_ggml_vec_dot_q5_1_q8_1(int n, float * LM_GGML_RESTRICT s, size_t bs, const void * LM_GGML_RESTRICT vx, size_t bx, const void * LM_GGML_RESTRICT vy, size_t by, int nrc);
void lm_ggml_vec_dot_q8_0_q8_0(int n, float * LM_GGML_RESTRICT s, size_t bs, const void * LM_GGML_RESTRICT vx, size_t bx, const void * LM_GGML_RESTRICT vy, size_t by, int nrc);

void lm_ggml_vec_dot_q2_K_q8_K(int n, float * LM_GGML_RESTRICT s, size_t bs, const void * LM_GGML_RESTRICT vx, size_t bx, const void * LM_GGML_RESTRICT vy, size_t by, int nrc);
void lm_ggml_vec_dot_q3_K_q8_K(int n, float * LM_GGML_RESTRICT s, size_t bs, const void * LM_GGML_RESTRICT vx, size_t bx, const void * LM_GGML_RESTRICT vy, size_t by, int nrc);
void lm_ggml_vec_dot_q4_K_q8_K(int n, float * LM_GGML_RESTRICT s, size_t bs, const void * LM_GGML_RESTRICT vx, size_t bx, const void * LM_GGML_RESTRICT vy, size_t by, int nrc);
void lm_ggml_vec_dot_q5_K_q8_K(int n, float * LM_GGML_RESTRICT s, size_t bs, const void * LM_GGML_RESTRICT vx, size_t bx, const void * LM_GGML_RESTRICT vy, size_t by, int nrc);
void lm_ggml_vec_dot_q6_K_q8_K(int n, float * LM_GGML_RESTRICT s, size_t bs, const void * LM_GGML_RESTRICT vx, size_t bx, const void * LM_GGML_RESTRICT vy, size_t by, int nrc);

void lm_ggml_vec_dot_iq2_xxs_q8_K(int n, float * LM_GGML_RESTRICT s, size_t bs, const void * LM_GGML_RESTRICT vx, size_t bx, const void * LM_GGML_RESTRICT vy, size_t by, int nrc);
void lm_ggml_vec_dot_iq2_xs_q8_K (int n, float * LM_GGML_RESTRICT s, size_t bs, const void * LM_GGML_RESTRICT vx, size_t bx, const void * LM_GGML_RESTRICT vy, size_t by, int nrc);
void lm_ggml_vec_dot_iq2_s_q8_K  (int n, float * LM_GGML_RESTRICT s, size_t bs, const void * LM_GGML_RESTRICT vx, size_t bx, const void * LM_GGML_RESTRICT vy, size_t by, int nrc);
void lm_ggml_vec_dot_iq3_xxs_q8_K(int n, float * LM_GGML_RESTRICT s, size_t bs, const void * LM_GGML_RESTRICT vx, size_t bx, const void * LM_GGML_RESTRICT vy, size_t by, int nrc);
void lm_ggml_vec_dot_iq1_s_q8_K  (int n, float * LM_GGML_RESTRICT s, size_t bs, const void * LM_GGML_RESTRICT vx, size_t bx, const void * LM_GGML_RESTRICT vy, size_t by, int nrc);
void lm_ggml_vec_dot_iq4_nl_q8_0 (int n, float * LM_GGML_RESTRICT s, size_t bs, const void * LM_GGML_RESTRICT vx, size_t bx, const void * LM_GGML_RESTRICT vy, size_t by, int nrc);
void lm_ggml_vec_dot_iq4_xs_q8_K (int n, float * LM_GGML_RESTRICT s, size_t bs, const void * LM_GGML_RESTRICT vx, size_t bx, const void * LM_GGML_RESTRICT vy, size_t by, int nrc);
void lm_ggml_vec_dot_iq3_s_q8_K  (int n, float * LM_GGML_RESTRICT s, size_t bs, const void * LM_GGML_RESTRICT vx, size_t bx, const void * LM_GGML_RESTRICT vy, size_t by, int nrc);

// Quantization utilizing an importance matrix (a.k.a. "Activation aWare Quantization")
size_t quantize_iq2_xxs(const float * LM_GGML_RESTRICT src, void * LM_GGML_RESTRICT dst, int nrows, int n_per_row, const float * imatrix);
size_t quantize_iq2_xs (const float * LM_GGML_RESTRICT src, void * LM_GGML_RESTRICT dst, int nrows, int n_per_row, const float * imatrix);
size_t quantize_iq2_s  (const float * LM_GGML_RESTRICT src, void * LM_GGML_RESTRICT dst, int nrows, int n_per_row, const float * imatrix);
size_t quantize_iq3_xxs(const float * LM_GGML_RESTRICT src, void * LM_GGML_RESTRICT dst, int nrows, int n_per_row, const float * imatrix);
size_t quantize_iq1_s  (const float * LM_GGML_RESTRICT src, void * LM_GGML_RESTRICT dst, int nrows, int n_per_row, const float * imatrix);
size_t quantize_iq4_nl (const float * LM_GGML_RESTRICT src, void * LM_GGML_RESTRICT dst, int nrows, int n_per_row, const float * imatrix);
size_t quantize_iq4_xs (const float * LM_GGML_RESTRICT src, void * LM_GGML_RESTRICT dst, int nrows, int n_per_row, const float * imatrix);
size_t quantize_iq3_s  (const float * LM_GGML_RESTRICT src, void * LM_GGML_RESTRICT dst, int nrows, int n_per_row, const float * imatrix);

size_t quantize_q2_K(const float * LM_GGML_RESTRICT src, void * LM_GGML_RESTRICT dst, int nrows, int n_per_row, const float * imatrix);
size_t quantize_q3_K(const float * LM_GGML_RESTRICT src, void * LM_GGML_RESTRICT dst, int nrows, int n_per_row, const float * imatrix);
size_t quantize_q4_K(const float * LM_GGML_RESTRICT src, void * LM_GGML_RESTRICT dst, int nrows, int n_per_row, const float * imatrix);
size_t quantize_q5_K(const float * LM_GGML_RESTRICT src, void * LM_GGML_RESTRICT dst, int nrows, int n_per_row, const float * imatrix);
size_t quantize_q6_K(const float * LM_GGML_RESTRICT src, void * LM_GGML_RESTRICT dst, int nrows, int n_per_row, const float * imatrix);
size_t quantize_q4_0(const float * LM_GGML_RESTRICT src, void * LM_GGML_RESTRICT dst, int nrows, int n_per_row, const float * imatrix);
size_t quantize_q4_1(const float * LM_GGML_RESTRICT src, void * LM_GGML_RESTRICT dst, int nrows, int n_per_row, const float * imatrix);
size_t quantize_q5_0(const float * LM_GGML_RESTRICT src, void * LM_GGML_RESTRICT dst, int nrows, int n_per_row, const float * imatrix);
size_t quantize_q5_1(const float * LM_GGML_RESTRICT src, void * LM_GGML_RESTRICT dst, int nrows, int n_per_row, const float * imatrix);
size_t quantize_q8_0(const float * LM_GGML_RESTRICT src, void * LM_GGML_RESTRICT dst, int nrows, int n_per_row, const float * imatrix);

void iq2xs_init_impl(enum lm_ggml_type type);
void iq2xs_free_impl(enum lm_ggml_type type);
void iq3xs_init_impl(int grid_size);
void iq3xs_free_impl(int grid_size);

#ifdef __cplusplus
}
#endif

