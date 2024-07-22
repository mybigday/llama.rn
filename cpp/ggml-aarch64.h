// SPDX-FileCopyrightText: Copyright 2024 Arm Ltd.
#pragma once

#define LM_GGML_COMMON_DECL_C
#include "ggml-common.h"

#include "ggml.h"

// GGML internal header

#ifdef __cplusplus
extern "C" {
#endif

// Quantization
void quantize_q8_0_4x4(const float * LM_GGML_RESTRICT x, void * LM_GGML_RESTRICT y, int64_t k);
void quantize_q8_0_4x8(const float * LM_GGML_RESTRICT x, void * LM_GGML_RESTRICT y, int64_t k);

void quantize_mat_q8_0(const float * LM_GGML_RESTRICT x, void * LM_GGML_RESTRICT y, int64_t nrows, int64_t n_per_row, int64_t blck_size_interleave);

// Quantization utilizing an importance matrix (a.k.a. "Activation aWare Quantization")
size_t quantize_q4_0_4x4(const float * LM_GGML_RESTRICT src, void * LM_GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
size_t quantize_q4_0_4x8(const float * LM_GGML_RESTRICT src, void * LM_GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
size_t quantize_q4_0_8x8(const float * LM_GGML_RESTRICT src, void * LM_GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);

// GEMV
void lm_ggml_gemv_q4_0_4x4_q8_0(int n, float * LM_GGML_RESTRICT s, size_t bs, const void * LM_GGML_RESTRICT vx, const void * LM_GGML_RESTRICT vy, int nr, int nc);
void lm_ggml_gemv_q4_0_4x8_q8_0(int n, float * LM_GGML_RESTRICT s, size_t bs, const void * LM_GGML_RESTRICT vx, const void * LM_GGML_RESTRICT vy, int nr, int nc);
void lm_ggml_gemv_q4_0_8x8_q8_0(int n, float * LM_GGML_RESTRICT s, size_t bs, const void * LM_GGML_RESTRICT vx, const void * LM_GGML_RESTRICT vy, int nr, int nc);

// GEMM
void lm_ggml_gemm_q4_0_4x4_q8_0(int n, float * LM_GGML_RESTRICT s, size_t bs, const void * LM_GGML_RESTRICT vx, const void * LM_GGML_RESTRICT vy, int nr, int nc);
void lm_ggml_gemm_q4_0_4x8_q8_0(int n, float * LM_GGML_RESTRICT s, size_t bs, const void * LM_GGML_RESTRICT vx, const void * LM_GGML_RESTRICT vy, int nr, int nc);
void lm_ggml_gemm_q4_0_8x8_q8_0(int n, float * LM_GGML_RESTRICT s, size_t bs, const void * LM_GGML_RESTRICT vx, const void * LM_GGML_RESTRICT vy, int nr, int nc);

#ifdef __cplusplus
}
#endif

