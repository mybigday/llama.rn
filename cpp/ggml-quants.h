#pragma once

#define LM_GGML_COMMON_DECL_C
#include "ggml-common.h"

#include "ggml.h"

// GGML internal header

#ifdef __cplusplus
extern "C" {
#endif

// NOTE: these functions are defined as LM_GGML_API because they used by the CPU backend

// Quantization
LM_GGML_API void quantize_row_q4_0_ref(const float * LM_GGML_RESTRICT x, block_q4_0 * LM_GGML_RESTRICT y, int64_t k);
LM_GGML_API void quantize_row_q4_1_ref(const float * LM_GGML_RESTRICT x, block_q4_1 * LM_GGML_RESTRICT y, int64_t k);
LM_GGML_API void quantize_row_q5_0_ref(const float * LM_GGML_RESTRICT x, block_q5_0 * LM_GGML_RESTRICT y, int64_t k);
LM_GGML_API void quantize_row_q5_1_ref(const float * LM_GGML_RESTRICT x, block_q5_1 * LM_GGML_RESTRICT y, int64_t k);
LM_GGML_API void quantize_row_q8_0_ref(const float * LM_GGML_RESTRICT x, block_q8_0 * LM_GGML_RESTRICT y, int64_t k);
LM_GGML_API void quantize_row_q8_1_ref(const float * LM_GGML_RESTRICT x, block_q8_1 * LM_GGML_RESTRICT y, int64_t k);

LM_GGML_API void quantize_row_q2_K_ref(const float * LM_GGML_RESTRICT x, block_q2_K * LM_GGML_RESTRICT y, int64_t k);
LM_GGML_API void quantize_row_q3_K_ref(const float * LM_GGML_RESTRICT x, block_q3_K * LM_GGML_RESTRICT y, int64_t k);
LM_GGML_API void quantize_row_q4_K_ref(const float * LM_GGML_RESTRICT x, block_q4_K * LM_GGML_RESTRICT y, int64_t k);
LM_GGML_API void quantize_row_q5_K_ref(const float * LM_GGML_RESTRICT x, block_q5_K * LM_GGML_RESTRICT y, int64_t k);
LM_GGML_API void quantize_row_q6_K_ref(const float * LM_GGML_RESTRICT x, block_q6_K * LM_GGML_RESTRICT y, int64_t k);
LM_GGML_API void quantize_row_q8_K_ref(const float * LM_GGML_RESTRICT x, block_q8_K * LM_GGML_RESTRICT y, int64_t k);

LM_GGML_API void quantize_row_tq1_0_ref(const float * LM_GGML_RESTRICT x, block_tq1_0 * LM_GGML_RESTRICT y, int64_t k);
LM_GGML_API void quantize_row_tq2_0_ref(const float * LM_GGML_RESTRICT x, block_tq2_0 * LM_GGML_RESTRICT y, int64_t k);

LM_GGML_API void quantize_row_iq3_xxs_ref(const float * LM_GGML_RESTRICT x, block_iq3_xxs * LM_GGML_RESTRICT y, int64_t k);
LM_GGML_API void quantize_row_iq4_nl_ref (const float * LM_GGML_RESTRICT x, block_iq4_nl  * LM_GGML_RESTRICT y, int64_t k);
LM_GGML_API void quantize_row_iq4_xs_ref (const float * LM_GGML_RESTRICT x, block_iq4_xs  * LM_GGML_RESTRICT y, int64_t k);
LM_GGML_API void quantize_row_iq3_s_ref  (const float * LM_GGML_RESTRICT x, block_iq3_s   * LM_GGML_RESTRICT y, int64_t k);
LM_GGML_API void quantize_row_iq2_s_ref  (const float * LM_GGML_RESTRICT x, block_iq2_s   * LM_GGML_RESTRICT y, int64_t k);

// Dequantization
LM_GGML_API void dequantize_row_q4_0(const block_q4_0 * LM_GGML_RESTRICT x, float * LM_GGML_RESTRICT y, int64_t k);
LM_GGML_API void dequantize_row_q4_1(const block_q4_1 * LM_GGML_RESTRICT x, float * LM_GGML_RESTRICT y, int64_t k);
LM_GGML_API void dequantize_row_q5_0(const block_q5_0 * LM_GGML_RESTRICT x, float * LM_GGML_RESTRICT y, int64_t k);
LM_GGML_API void dequantize_row_q5_1(const block_q5_1 * LM_GGML_RESTRICT x, float * LM_GGML_RESTRICT y, int64_t k);
LM_GGML_API void dequantize_row_q8_0(const block_q8_0 * LM_GGML_RESTRICT x, float * LM_GGML_RESTRICT y, int64_t k);
//LM_GGML_API void dequantize_row_q8_1(const block_q8_1 * LM_GGML_RESTRICT x, float * LM_GGML_RESTRICT y, int64_t k);

LM_GGML_API void dequantize_row_q2_K(const block_q2_K * LM_GGML_RESTRICT x, float * LM_GGML_RESTRICT y, int64_t k);
LM_GGML_API void dequantize_row_q3_K(const block_q3_K * LM_GGML_RESTRICT x, float * LM_GGML_RESTRICT y, int64_t k);
LM_GGML_API void dequantize_row_q4_K(const block_q4_K * LM_GGML_RESTRICT x, float * LM_GGML_RESTRICT y, int64_t k);
LM_GGML_API void dequantize_row_q5_K(const block_q5_K * LM_GGML_RESTRICT x, float * LM_GGML_RESTRICT y, int64_t k);
LM_GGML_API void dequantize_row_q6_K(const block_q6_K * LM_GGML_RESTRICT x, float * LM_GGML_RESTRICT y, int64_t k);
LM_GGML_API void dequantize_row_q8_K(const block_q8_K * LM_GGML_RESTRICT x, float * LM_GGML_RESTRICT y, int64_t k);

LM_GGML_API void dequantize_row_tq1_0(const block_tq1_0 * LM_GGML_RESTRICT x, float * LM_GGML_RESTRICT y, int64_t k);
LM_GGML_API void dequantize_row_tq2_0(const block_tq2_0 * LM_GGML_RESTRICT x, float * LM_GGML_RESTRICT y, int64_t k);

LM_GGML_API void dequantize_row_iq2_xxs(const block_iq2_xxs * LM_GGML_RESTRICT x, float * LM_GGML_RESTRICT y, int64_t k);
LM_GGML_API void dequantize_row_iq2_xs (const block_iq2_xs  * LM_GGML_RESTRICT x, float * LM_GGML_RESTRICT y, int64_t k);
LM_GGML_API void dequantize_row_iq2_s  (const block_iq2_s   * LM_GGML_RESTRICT x, float * LM_GGML_RESTRICT y, int64_t k);
LM_GGML_API void dequantize_row_iq3_xxs(const block_iq3_xxs * LM_GGML_RESTRICT x, float * LM_GGML_RESTRICT y, int64_t k);
LM_GGML_API void dequantize_row_iq1_s  (const block_iq1_s   * LM_GGML_RESTRICT x, float * LM_GGML_RESTRICT y, int64_t k);
LM_GGML_API void dequantize_row_iq1_m  (const block_iq1_m   * LM_GGML_RESTRICT x, float * LM_GGML_RESTRICT y, int64_t k);
LM_GGML_API void dequantize_row_iq4_nl (const block_iq4_nl  * LM_GGML_RESTRICT x, float * LM_GGML_RESTRICT y, int64_t k);
LM_GGML_API void dequantize_row_iq4_xs (const block_iq4_xs  * LM_GGML_RESTRICT x, float * LM_GGML_RESTRICT y, int64_t k);
LM_GGML_API void dequantize_row_iq3_s  (const block_iq3_s   * LM_GGML_RESTRICT x, float * LM_GGML_RESTRICT y, int64_t k);

// Quantization utilizing an importance matrix (a.k.a. "Activation aWare Quantization")
LM_GGML_API size_t quantize_iq2_xxs(const float * LM_GGML_RESTRICT src, void * LM_GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
LM_GGML_API size_t quantize_iq2_xs (const float * LM_GGML_RESTRICT src, void * LM_GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
LM_GGML_API size_t quantize_iq2_s  (const float * LM_GGML_RESTRICT src, void * LM_GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
LM_GGML_API size_t quantize_iq3_xxs(const float * LM_GGML_RESTRICT src, void * LM_GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
LM_GGML_API size_t quantize_iq1_s  (const float * LM_GGML_RESTRICT src, void * LM_GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
LM_GGML_API size_t quantize_iq1_m  (const float * LM_GGML_RESTRICT src, void * LM_GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
LM_GGML_API size_t quantize_iq4_nl (const float * LM_GGML_RESTRICT src, void * LM_GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
LM_GGML_API size_t quantize_iq4_xs (const float * LM_GGML_RESTRICT src, void * LM_GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
LM_GGML_API size_t quantize_iq3_s  (const float * LM_GGML_RESTRICT src, void * LM_GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);

LM_GGML_API size_t quantize_tq1_0(const float * LM_GGML_RESTRICT src, void * LM_GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
LM_GGML_API size_t quantize_tq2_0(const float * LM_GGML_RESTRICT src, void * LM_GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);

LM_GGML_API size_t quantize_q2_K(const float * LM_GGML_RESTRICT src, void * LM_GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
LM_GGML_API size_t quantize_q3_K(const float * LM_GGML_RESTRICT src, void * LM_GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
LM_GGML_API size_t quantize_q4_K(const float * LM_GGML_RESTRICT src, void * LM_GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
LM_GGML_API size_t quantize_q5_K(const float * LM_GGML_RESTRICT src, void * LM_GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
LM_GGML_API size_t quantize_q6_K(const float * LM_GGML_RESTRICT src, void * LM_GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
LM_GGML_API size_t quantize_q4_0(const float * LM_GGML_RESTRICT src, void * LM_GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
LM_GGML_API size_t quantize_q4_1(const float * LM_GGML_RESTRICT src, void * LM_GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
LM_GGML_API size_t quantize_q5_0(const float * LM_GGML_RESTRICT src, void * LM_GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
LM_GGML_API size_t quantize_q5_1(const float * LM_GGML_RESTRICT src, void * LM_GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
LM_GGML_API size_t quantize_q8_0(const float * LM_GGML_RESTRICT src, void * LM_GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);

LM_GGML_API void lm_iq2xs_init_impl(enum lm_ggml_type type);
LM_GGML_API void lm_iq2xs_free_impl(enum lm_ggml_type type);
LM_GGML_API void lm_iq3xs_init_impl(int grid_size);
LM_GGML_API void lm_iq3xs_free_impl(int grid_size);

#ifdef __cplusplus
}
#endif
