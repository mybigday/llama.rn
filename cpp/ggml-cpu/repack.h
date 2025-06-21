#pragma once

#define LM_GGML_COMMON_DECL_CPP
#include "ggml-common.h"

#include "traits.h"
#include "ggml.h"

// GGML internal header

lm_ggml_backend_buffer_type_t lm_ggml_backend_cpu_repack_buffer_type(void);

template <int K> constexpr int QK_0() {
    if constexpr (K == 4) {
        return QK4_0;
    }
    if constexpr (K == 8) {
        return QK8_0;
    }
    return -1;
}

template <int K, int N> struct block {
    lm_ggml_half d[N];                         // deltas for N qK_0 blocks
    int8_t    qs[(QK_0<K>() * N * K) / 8];  // quants for N qK_0 blocks
};

// control size
static_assert(sizeof(block<4, 4>) == 4 * sizeof(lm_ggml_half) + QK8_0 * 2, "wrong block<4,4> size/padding");
static_assert(sizeof(block<4, 8>) == 8 * sizeof(lm_ggml_half) + QK8_0 * 4, "wrong block<4,8> size/padding");
static_assert(sizeof(block<8, 4>) == 4 * sizeof(lm_ggml_half) + QK8_0 * 4, "wrong block<8,4> size/padding");
static_assert(sizeof(block<8, 8>) == 8 * sizeof(lm_ggml_half) + QK8_0 * 8, "wrong block<8,8> size/padding");

using block_q4_0x4 = block<4, 4>;
using block_q4_0x8 = block<4, 8>;
using block_q8_0x4 = block<8, 4>;
using block_q8_0x8 = block<8, 8>;

struct block_q4_Kx8 {
    lm_ggml_half d[8];      // super-block scale for quantized scales
    lm_ggml_half dmin[8];   // super-block scale for quantized mins
    uint8_t scales[96];  // scales and mins, quantized with 6 bits
    uint8_t qs[1024];    // 4--bit quants
};

static_assert(sizeof(block_q4_Kx8) == sizeof(lm_ggml_half) * 16 + K_SCALE_SIZE * 8 + QK_K * 4, "wrong q4_K block size/padding");

struct block_q8_Kx4 {
    float d[4];              // delta
    int8_t qs[QK_K * 4];     // quants
    int16_t bsums[QK_K / 4]; // sum of quants in groups of 16
};

static_assert(sizeof(block_q8_Kx4) == sizeof(float) * 4 + QK_K * 4 + (QK_K / 4) * sizeof(int16_t), "wrong q8_K block size/padding");

struct block_iq4_nlx4 {
    lm_ggml_half d[4];            // deltas for 4 iq4_nl blocks
    uint8_t   qs[QK4_NL * 2];  // nibbles / quants for 4 iq4_nl blocks
};

static_assert(sizeof(block_iq4_nlx4) == 4 * sizeof(lm_ggml_half) + QK4_NL * 2, "wrong iq4_nlx4 block size/padding");

#if defined(__cplusplus)
extern "C" {
#endif

void lm_ggml_quantize_mat_q8_0_4x4(const float * LM_GGML_RESTRICT x, void * LM_GGML_RESTRICT vy, int64_t k);
void lm_ggml_quantize_mat_q8_0_4x8(const float * LM_GGML_RESTRICT x, void * LM_GGML_RESTRICT vy, int64_t k);
void lm_ggml_quantize_mat_q8_K_4x8(const float * LM_GGML_RESTRICT x, void * LM_GGML_RESTRICT vy, int64_t k);
void lm_ggml_gemv_q4_0_4x4_q8_0(int n, float * LM_GGML_RESTRICT s, size_t bs, const void * LM_GGML_RESTRICT vx, const void * LM_GGML_RESTRICT vy, int nr, int nc);
void lm_ggml_gemv_q4_0_4x8_q8_0(int n, float * LM_GGML_RESTRICT s, size_t bs, const void * LM_GGML_RESTRICT vx, const void * LM_GGML_RESTRICT vy, int nr, int nc);
void lm_ggml_gemv_q4_0_8x8_q8_0(int n, float * LM_GGML_RESTRICT s, size_t bs, const void * LM_GGML_RESTRICT vx, const void * LM_GGML_RESTRICT vy, int nr, int nc);
void lm_ggml_gemv_q4_K_8x8_q8_K(int n, float * LM_GGML_RESTRICT s, size_t bs, const void * LM_GGML_RESTRICT vx, const void * LM_GGML_RESTRICT vy, int nr, int nc);
void lm_ggml_gemv_iq4_nl_4x4_q8_0(int n, float * LM_GGML_RESTRICT s, size_t bs, const void * LM_GGML_RESTRICT vx, const void * LM_GGML_RESTRICT vy, int nr, int nc);
void lm_ggml_gemm_q4_0_4x4_q8_0(int n, float * LM_GGML_RESTRICT s, size_t bs, const void * LM_GGML_RESTRICT vx, const void * LM_GGML_RESTRICT vy, int nr, int nc);
void lm_ggml_gemm_q4_0_4x8_q8_0(int n, float * LM_GGML_RESTRICT s, size_t bs, const void * LM_GGML_RESTRICT vx, const void * LM_GGML_RESTRICT vy, int nr, int nc);
void lm_ggml_gemm_q4_0_8x8_q8_0(int n, float * LM_GGML_RESTRICT s, size_t bs, const void * LM_GGML_RESTRICT vx, const void * LM_GGML_RESTRICT vy, int nr, int nc);
void lm_ggml_gemm_q4_K_8x8_q8_K(int n, float * LM_GGML_RESTRICT s, size_t bs, const void * LM_GGML_RESTRICT vx, const void * LM_GGML_RESTRICT vy, int nr, int nc);
void lm_ggml_gemm_iq4_nl_4x4_q8_0(int n, float * LM_GGML_RESTRICT s, size_t bs, const void * LM_GGML_RESTRICT vx, const void * LM_GGML_RESTRICT vy, int nr, int nc);

// Native implementations
void lm_ggml_quantize_mat_q8_0_4x4_generic(const float * LM_GGML_RESTRICT x, void * LM_GGML_RESTRICT vy, int64_t k);
void lm_ggml_quantize_mat_q8_0_4x8_generic(const float * LM_GGML_RESTRICT x, void * LM_GGML_RESTRICT vy, int64_t k);
void lm_ggml_quantize_mat_q8_K_4x8_generic(const float * LM_GGML_RESTRICT x, void * LM_GGML_RESTRICT vy, int64_t k);
void lm_ggml_gemv_q4_0_4x4_q8_0_generic(int n, float * LM_GGML_RESTRICT s, size_t bs, const void * LM_GGML_RESTRICT vx, const void * LM_GGML_RESTRICT vy, int nr, int nc);
void lm_ggml_gemv_q4_0_4x8_q8_0_generic(int n, float * LM_GGML_RESTRICT s, size_t bs, const void * LM_GGML_RESTRICT vx, const void * LM_GGML_RESTRICT vy, int nr, int nc);
void lm_ggml_gemv_q4_0_8x8_q8_0_generic(int n, float * LM_GGML_RESTRICT s, size_t bs, const void * LM_GGML_RESTRICT vx, const void * LM_GGML_RESTRICT vy, int nr, int nc);
void lm_ggml_gemv_q4_K_8x8_q8_K_generic(int n, float * LM_GGML_RESTRICT s, size_t bs, const void * LM_GGML_RESTRICT vx, const void * LM_GGML_RESTRICT vy, int nr, int nc);
void lm_ggml_gemv_iq4_nl_4x4_q8_0_generic(int n, float * LM_GGML_RESTRICT s, size_t bs, const void * LM_GGML_RESTRICT vx, const void * LM_GGML_RESTRICT vy, int nr, int nc);
void lm_ggml_gemm_q4_0_4x4_q8_0_generic(int n, float * LM_GGML_RESTRICT s, size_t bs, const void * LM_GGML_RESTRICT vx, const void * LM_GGML_RESTRICT vy, int nr, int nc);
void lm_ggml_gemm_q4_0_4x8_q8_0_generic(int n, float * LM_GGML_RESTRICT s, size_t bs, const void * LM_GGML_RESTRICT vx, const void * LM_GGML_RESTRICT vy, int nr, int nc);
void lm_ggml_gemm_q4_0_8x8_q8_0_generic(int n, float * LM_GGML_RESTRICT s, size_t bs, const void * LM_GGML_RESTRICT vx, const void * LM_GGML_RESTRICT vy, int nr, int nc);
void lm_ggml_gemm_q4_K_8x8_q8_K_generic(int n, float * LM_GGML_RESTRICT s, size_t bs, const void * LM_GGML_RESTRICT vx, const void * LM_GGML_RESTRICT vy, int nr, int nc);
void lm_ggml_gemm_iq4_nl_4x4_q8_0_generic(int n, float * LM_GGML_RESTRICT s, size_t bs, const void * LM_GGML_RESTRICT vx, const void * LM_GGML_RESTRICT vy, int nr, int nc);

#if defined(__cplusplus)
} // extern "C"
#endif
