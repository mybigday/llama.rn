#pragma once

#include "ggml-metal-device.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct lm_ggml_metal_op * lm_ggml_metal_op_t;

lm_ggml_metal_op_t lm_ggml_metal_op_init(
        lm_ggml_metal_device_t dev,
        lm_ggml_metal_cmd_buf_t cmd_buf,
        struct lm_ggml_cgraph * gf,
        int  idx_start,
        int  idx_end,
        bool use_fusion,
        bool use_concurrency,
        bool use_capture,
        int  debug_graph,
        int  debug_fusion);

void lm_ggml_metal_op_free(lm_ggml_metal_op_t ctx);

int lm_ggml_metal_op_n_nodes(lm_ggml_metal_op_t ctx);

int lm_ggml_metal_op_encode(lm_ggml_metal_op_t ctx, int idx);

//
// available ops:
//

// tokens per expert
size_t lm_ggml_metal_op_mul_mat_id_extra_tpe(const struct lm_ggml_tensor * op);

// id map [n_tokens, n_expert]
size_t lm_ggml_metal_op_mul_mat_id_extra_ids(const struct lm_ggml_tensor * op);

// return true if we should use the FA vector kernel for this op
bool lm_ggml_metal_op_flash_attn_ext_use_vec(const struct lm_ggml_tensor * op);

size_t lm_ggml_metal_op_flash_attn_ext_extra_pad(const struct lm_ggml_tensor * op);
size_t lm_ggml_metal_op_flash_attn_ext_extra_blk(const struct lm_ggml_tensor * op);
size_t lm_ggml_metal_op_flash_attn_ext_extra_tmp(const struct lm_ggml_tensor * op);

int lm_ggml_metal_op_concat            (lm_ggml_metal_op_t ctx, int idx);
int lm_ggml_metal_op_repeat            (lm_ggml_metal_op_t ctx, int idx);
int lm_ggml_metal_op_acc               (lm_ggml_metal_op_t ctx, int idx);
int lm_ggml_metal_op_scale             (lm_ggml_metal_op_t ctx, int idx);
int lm_ggml_metal_op_fill              (lm_ggml_metal_op_t ctx, int idx);
int lm_ggml_metal_op_clamp             (lm_ggml_metal_op_t ctx, int idx);
int lm_ggml_metal_op_unary             (lm_ggml_metal_op_t ctx, int idx);
int lm_ggml_metal_op_glu               (lm_ggml_metal_op_t ctx, int idx);
int lm_ggml_metal_op_sum               (lm_ggml_metal_op_t ctx, int idx);
int lm_ggml_metal_op_sum_rows          (lm_ggml_metal_op_t ctx, int idx);
int lm_ggml_metal_op_cumsum            (lm_ggml_metal_op_t ctx, int idx);
int lm_ggml_metal_op_get_rows          (lm_ggml_metal_op_t ctx, int idx);
int lm_ggml_metal_op_set_rows          (lm_ggml_metal_op_t ctx, int idx);
int lm_ggml_metal_op_soft_max          (lm_ggml_metal_op_t ctx, int idx);
int lm_ggml_metal_op_ssm_conv          (lm_ggml_metal_op_t ctx, int idx);
int lm_ggml_metal_op_ssm_scan          (lm_ggml_metal_op_t ctx, int idx);
int lm_ggml_metal_op_rwkv              (lm_ggml_metal_op_t ctx, int idx);
int lm_ggml_metal_op_cpy               (lm_ggml_metal_op_t ctx, int idx);
int lm_ggml_metal_op_pool_2d           (lm_ggml_metal_op_t ctx, int idx);
int lm_ggml_metal_op_mul_mat           (lm_ggml_metal_op_t ctx, int idx);
int lm_ggml_metal_op_mul_mat_id        (lm_ggml_metal_op_t ctx, int idx);
int lm_ggml_metal_op_add_id            (lm_ggml_metal_op_t ctx, int idx);
int lm_ggml_metal_op_flash_attn_ext    (lm_ggml_metal_op_t ctx, int idx);
int lm_ggml_metal_op_bin               (lm_ggml_metal_op_t ctx, int idx);
int lm_ggml_metal_op_l2_norm           (lm_ggml_metal_op_t ctx, int idx);
int lm_ggml_metal_op_group_norm        (lm_ggml_metal_op_t ctx, int idx);
int lm_ggml_metal_op_norm              (lm_ggml_metal_op_t ctx, int idx);
int lm_ggml_metal_op_rope              (lm_ggml_metal_op_t ctx, int idx);
int lm_ggml_metal_op_im2col            (lm_ggml_metal_op_t ctx, int idx);
int lm_ggml_metal_op_conv_2d           (lm_ggml_metal_op_t ctx, int idx);
int lm_ggml_metal_op_conv_transpose_1d (lm_ggml_metal_op_t ctx, int idx);
int lm_ggml_metal_op_conv_transpose_2d (lm_ggml_metal_op_t ctx, int idx);
int lm_ggml_metal_op_upscale           (lm_ggml_metal_op_t ctx, int idx);
int lm_ggml_metal_op_pad               (lm_ggml_metal_op_t ctx, int idx);
int lm_ggml_metal_op_pad_reflect_1d    (lm_ggml_metal_op_t ctx, int idx);
int lm_ggml_metal_op_arange            (lm_ggml_metal_op_t ctx, int idx);
int lm_ggml_metal_op_timestep_embedding(lm_ggml_metal_op_t ctx, int idx);
int lm_ggml_metal_op_argmax            (lm_ggml_metal_op_t ctx, int idx);
int lm_ggml_metal_op_argsort           (lm_ggml_metal_op_t ctx, int idx);
int lm_ggml_metal_op_top_k             (lm_ggml_metal_op_t ctx, int idx);
int lm_ggml_metal_op_leaky_relu        (lm_ggml_metal_op_t ctx, int idx);
int lm_ggml_metal_op_tri               (lm_ggml_metal_op_t ctx, int idx);
int lm_ggml_metal_op_opt_step_adamw    (lm_ggml_metal_op_t ctx, int idx);
int lm_ggml_metal_op_opt_step_sgd      (lm_ggml_metal_op_t ctx, int idx);

#ifdef __cplusplus
}
#endif
