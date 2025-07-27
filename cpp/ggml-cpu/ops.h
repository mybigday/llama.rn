#pragma once

#include "ggml.h"

//
// cache line
//

#if defined(__cpp_lib_hardware_interference_size)
#define CACHE_LINE_SIZE std::hardware_destructive_interference_size
#else
#if defined(__POWER9_VECTOR__)
#define CACHE_LINE_SIZE 128
#elif defined(__VXE__) || defined(__VXE2__)
#define CACHE_LINE_SIZE 256
#else
#define CACHE_LINE_SIZE 64
#endif
#endif

static const size_t CACHE_LINE_SIZE_F32 = CACHE_LINE_SIZE/sizeof(float);

// Work buffer size for im2col operations in CONV2D
#define LM_GGML_IM2COL_WORK_SIZE (16 * 1024 * 1024)

#ifdef __cplusplus
extern "C" {
#endif

void lm_ggml_compute_forward_dup(const struct lm_ggml_compute_params * params, struct lm_ggml_tensor * dst);
void lm_ggml_compute_forward_add(const struct lm_ggml_compute_params * params, struct lm_ggml_tensor * dst);
void lm_ggml_compute_forward_add1(const struct lm_ggml_compute_params * params, struct lm_ggml_tensor * dst);
void lm_ggml_compute_forward_acc(const struct lm_ggml_compute_params * params, struct lm_ggml_tensor * dst);
void lm_ggml_compute_forward_sum(const struct lm_ggml_compute_params * params, struct lm_ggml_tensor * dst);
void lm_ggml_compute_forward_sum_rows(const struct lm_ggml_compute_params * params, struct lm_ggml_tensor * dst);
void lm_ggml_compute_forward_mean(const struct lm_ggml_compute_params * params, struct lm_ggml_tensor * dst);
void lm_ggml_compute_forward_argmax(const struct lm_ggml_compute_params * params, struct lm_ggml_tensor * dst);
void lm_ggml_compute_forward_count_equal(const struct lm_ggml_compute_params * params, struct lm_ggml_tensor * dst);
void lm_ggml_compute_forward_repeat(const struct lm_ggml_compute_params * params, struct lm_ggml_tensor * dst);
void lm_ggml_compute_forward_repeat_back(const struct lm_ggml_compute_params * params, struct lm_ggml_tensor * dst);
void lm_ggml_compute_forward_concat(const struct lm_ggml_compute_params * params, struct lm_ggml_tensor * dst);
void lm_ggml_compute_forward_silu_back(const struct lm_ggml_compute_params * params, struct lm_ggml_tensor * dst);
void lm_ggml_compute_forward_norm(const struct lm_ggml_compute_params * params, struct lm_ggml_tensor * dst);
void lm_ggml_compute_forward_rms_norm(const struct lm_ggml_compute_params * params, struct lm_ggml_tensor * dst);
void lm_ggml_compute_forward_rms_norm_back(const struct lm_ggml_compute_params * params, struct lm_ggml_tensor * dst);
void lm_ggml_compute_forward_group_norm(const struct lm_ggml_compute_params * params, struct lm_ggml_tensor * dst);
void lm_ggml_compute_forward_l2_norm(const struct lm_ggml_compute_params * params, struct lm_ggml_tensor * dst);
void lm_ggml_compute_forward_out_prod(const struct lm_ggml_compute_params * params, struct lm_ggml_tensor * dst);
void lm_ggml_compute_forward_scale(const struct lm_ggml_compute_params * params, struct lm_ggml_tensor * dst);
void lm_ggml_compute_forward_set(const struct lm_ggml_compute_params * params, struct lm_ggml_tensor * dst);
void lm_ggml_compute_forward_cpy(const struct lm_ggml_compute_params * params, struct lm_ggml_tensor * dst);
void lm_ggml_compute_forward_cont(const struct lm_ggml_compute_params * params, struct lm_ggml_tensor * dst);
void lm_ggml_compute_forward_reshape(const struct lm_ggml_compute_params * params, struct lm_ggml_tensor * dst);
void lm_ggml_compute_forward_view(const struct lm_ggml_compute_params * params, struct lm_ggml_tensor * dst);
void lm_ggml_compute_forward_permute(const struct lm_ggml_compute_params * params, struct lm_ggml_tensor * dst);
void lm_ggml_compute_forward_transpose(const struct lm_ggml_compute_params * params, struct lm_ggml_tensor * dst);
void lm_ggml_compute_forward_get_rows(const struct lm_ggml_compute_params * params, struct lm_ggml_tensor * dst);
void lm_ggml_compute_forward_get_rows_back(const struct lm_ggml_compute_params * params, struct lm_ggml_tensor * dst);
void lm_ggml_compute_forward_set_rows(const struct lm_ggml_compute_params * params, struct lm_ggml_tensor * dst);
void lm_ggml_compute_forward_diag(const struct lm_ggml_compute_params * params, struct lm_ggml_tensor * dst);
void lm_ggml_compute_forward_diag_mask_inf(const struct lm_ggml_compute_params * params, struct lm_ggml_tensor * dst);
void lm_ggml_compute_forward_diag_mask_zero(const struct lm_ggml_compute_params * params, struct lm_ggml_tensor * dst);
void lm_ggml_compute_forward_soft_max(const struct lm_ggml_compute_params * params, struct lm_ggml_tensor * dst);
void lm_ggml_compute_forward_soft_max_ext_back(const struct lm_ggml_compute_params * params, struct lm_ggml_tensor * dst);
void lm_ggml_compute_forward_rope(const struct lm_ggml_compute_params * params, struct lm_ggml_tensor * dst);
void lm_ggml_compute_forward_rope_back(const struct lm_ggml_compute_params * params, struct lm_ggml_tensor * dst);
void lm_ggml_compute_forward_clamp(const struct lm_ggml_compute_params * params, struct lm_ggml_tensor * dst);
void lm_ggml_compute_forward_conv_transpose_1d(const struct lm_ggml_compute_params * params, struct lm_ggml_tensor * dst);
void lm_ggml_compute_forward_im2col(const struct lm_ggml_compute_params * params, struct lm_ggml_tensor * dst);
void lm_ggml_compute_forward_im2col_back_f32(const struct lm_ggml_compute_params * params, struct lm_ggml_tensor * dst);
void lm_ggml_compute_forward_conv_2d(const struct lm_ggml_compute_params * params, struct lm_ggml_tensor * dst);
void lm_ggml_compute_forward_conv_transpose_2d(const struct lm_ggml_compute_params * params, struct lm_ggml_tensor * dst);
void lm_ggml_compute_forward_conv_2d_dw(const struct lm_ggml_compute_params * params, struct lm_ggml_tensor * dst);
void lm_ggml_compute_forward_pool_1d(const struct lm_ggml_compute_params * params, struct lm_ggml_tensor * dst);
void lm_ggml_compute_forward_pool_2d(const struct lm_ggml_compute_params * params, struct lm_ggml_tensor * dst);
void lm_ggml_compute_forward_pool_2d_back(const struct lm_ggml_compute_params * params, struct lm_ggml_tensor * dst);
void lm_ggml_compute_forward_upscale(const struct lm_ggml_compute_params * params, struct lm_ggml_tensor * dst);
void lm_ggml_compute_forward_pad(const struct lm_ggml_compute_params * params, struct lm_ggml_tensor * dst);
void lm_ggml_compute_forward_pad_reflect_1d(const struct lm_ggml_compute_params * params, struct lm_ggml_tensor * dst);
void lm_ggml_compute_forward_roll(const struct lm_ggml_compute_params * params, struct lm_ggml_tensor * dst);
void lm_ggml_compute_forward_arange(const struct lm_ggml_compute_params * params, struct lm_ggml_tensor * dst);
void lm_ggml_compute_forward_timestep_embedding(const struct lm_ggml_compute_params * params, struct lm_ggml_tensor * dst);
void lm_ggml_compute_forward_argsort(const struct lm_ggml_compute_params * params, struct lm_ggml_tensor * dst);
void lm_ggml_compute_forward_leaky_relu(const struct lm_ggml_compute_params * params, struct lm_ggml_tensor * dst);
void lm_ggml_compute_forward_flash_attn_ext(
    const struct lm_ggml_compute_params * params,
    const struct lm_ggml_tensor * q,
    const struct lm_ggml_tensor * k,
    const struct lm_ggml_tensor * v,
    const struct lm_ggml_tensor * mask,
    struct lm_ggml_tensor * dst);
void lm_ggml_compute_forward_flash_attn_back(
        const struct lm_ggml_compute_params * params,
        const bool masked,
        struct lm_ggml_tensor * dst);
void lm_ggml_compute_forward_ssm_conv(const struct lm_ggml_compute_params * params, struct lm_ggml_tensor * dst);
void lm_ggml_compute_forward_ssm_scan(const struct lm_ggml_compute_params * params, struct lm_ggml_tensor * dst);
void lm_ggml_compute_forward_win_part(const struct lm_ggml_compute_params * params, struct lm_ggml_tensor * dst);
void lm_ggml_compute_forward_win_unpart(const struct lm_ggml_compute_params * params, struct lm_ggml_tensor * dst);
void lm_ggml_compute_forward_unary(const struct lm_ggml_compute_params * params, struct lm_ggml_tensor * dst);
void lm_ggml_compute_forward_glu(const struct lm_ggml_compute_params * params, struct lm_ggml_tensor * dst);
void lm_ggml_compute_forward_get_rel_pos(const struct lm_ggml_compute_params * params, struct lm_ggml_tensor * dst);
void lm_ggml_compute_forward_add_rel_pos(const struct lm_ggml_compute_params * params, struct lm_ggml_tensor * dst);
void lm_ggml_compute_forward_rwkv_wkv6(const struct lm_ggml_compute_params * params, struct lm_ggml_tensor * dst);
void lm_ggml_compute_forward_rwkv_wkv7(const struct lm_ggml_compute_params * params, struct lm_ggml_tensor * dst);
void lm_ggml_compute_forward_gla(const struct lm_ggml_compute_params * params, struct lm_ggml_tensor * dst);
void lm_ggml_compute_forward_map_custom1(const struct lm_ggml_compute_params * params, struct lm_ggml_tensor * dst);
void lm_ggml_compute_forward_map_custom2(const struct lm_ggml_compute_params * params, struct lm_ggml_tensor * dst);
void lm_ggml_compute_forward_map_custom3(const struct lm_ggml_compute_params * params, struct lm_ggml_tensor * dst);
void lm_ggml_compute_forward_custom(const struct lm_ggml_compute_params * params, struct lm_ggml_tensor * dst);
void lm_ggml_compute_forward_cross_entropy_loss(const struct lm_ggml_compute_params * params, struct lm_ggml_tensor * dst);
void lm_ggml_compute_forward_cross_entropy_loss_back(const struct lm_ggml_compute_params * params, struct lm_ggml_tensor * dst);
void lm_ggml_compute_forward_opt_step_adamw(const struct lm_ggml_compute_params * params, struct lm_ggml_tensor * dst);
void lm_ggml_compute_forward_mul_mat(const struct lm_ggml_compute_params * params, struct lm_ggml_tensor * dst);

#ifdef __cplusplus
}
#endif
