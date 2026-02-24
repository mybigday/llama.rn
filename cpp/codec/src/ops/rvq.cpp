#include "rvq.h"

static void codec_rvq_argmin_rows_map_custom1(
    lm_ggml_tensor * dst,
    const lm_ggml_tensor * src,
    int ith,
    int nth,
    void * userdata) {

    (void) userdata;

    if (dst == nullptr || src == nullptr || src->type != LM_GGML_TYPE_F32 || dst->type != LM_GGML_TYPE_F32) {
        return;
    }
    if (src->ne[0] <= 0 || src->ne[1] <= 0) {
        return;
    }

    const int64_t cols = src->ne[1] * src->ne[2] * src->ne[3];
    const int64_t start = (cols * ith) / nth;
    const int64_t end = (cols * (ith + 1)) / nth;

    for (int64_t col = start; col < end; ++col) {
        int64_t rem = col;
        const int64_t i1 = rem % src->ne[1];
        rem /= src->ne[1];
        const int64_t i2 = rem % src->ne[2];
        rem /= src->ne[2];
        const int64_t i3 = rem;

        const size_t col_offset = (size_t) i1 * src->nb[1] + (size_t) i2 * src->nb[2] + (size_t) i3 * src->nb[3];
        const float * src_col = reinterpret_cast<const float *>(reinterpret_cast<const char *>(src->data) + col_offset);
        float * dst_col = reinterpret_cast<float *>(reinterpret_cast<char *>(dst->data) + col_offset);

        float best = src_col[0];
        int32_t best_idx = 0;
        for (int32_t i = 1; i < (int32_t) src->ne[0]; ++i) {
            const float v = src_col[(size_t) i];
            if (v < best) {
                best = v;
                best_idx = i;
            }
        }

        dst_col[0] = (float) best_idx;
    }
}

lm_ggml_tensor * codec_rvq_argmin_map_custom1(
    lm_ggml_context * ctx_eval,
    lm_ggml_tensor * distances_ct) {

    if (ctx_eval == nullptr || distances_ct == nullptr || distances_ct->type != LM_GGML_TYPE_F32 || distances_ct->ne[0] <= 0 || distances_ct->ne[1] <= 0) {
        return nullptr;
    }

    lm_ggml_tensor * dist = lm_ggml_cont(ctx_eval, distances_ct);
    lm_ggml_tensor * argmin_full = lm_ggml_map_custom1(ctx_eval, dist, codec_rvq_argmin_rows_map_custom1, LM_GGML_N_TASKS_MAX, nullptr);
    if (argmin_full == nullptr) {
        return nullptr;
    }

    lm_ggml_tensor * argmin_row = lm_ggml_view_2d(ctx_eval, argmin_full, 1, dist->ne[1], argmin_full->nb[1], 0);
    argmin_row = lm_ggml_cont(ctx_eval, argmin_row);
    argmin_row = lm_ggml_reshape_1d(ctx_eval, argmin_row, dist->ne[1]);
    return lm_ggml_cast(ctx_eval, argmin_row, LM_GGML_TYPE_I32);
}

bool codec_rvq_build_layer_ggml(
    lm_ggml_context * ctx_eval,
    lm_ggml_tensor * residual_ct,
    lm_ggml_tensor * codebook_dc,
    codec_rvq_layer_result_ggml * out) {

    if (ctx_eval == nullptr || residual_ct == nullptr || codebook_dc == nullptr || out == nullptr) {
        return false;
    }
    if (residual_ct->type != LM_GGML_TYPE_F32 || codebook_dc->type != LM_GGML_TYPE_F32) {
        return false;
    }
    if (residual_ct->ne[0] <= 0 || residual_ct->ne[1] <= 0 || codebook_dc->ne[0] != residual_ct->ne[0] || codebook_dc->ne[1] <= 0) {
        return false;
    }

    // dist(c, t) = ||residual(:,t)||^2 + ||codebook(:,c)||^2 - 2 * dot(codebook(:,c), residual(:,t))
    lm_ggml_tensor * residual_sq = lm_ggml_sqr(ctx_eval, residual_ct);
    lm_ggml_tensor * codebook_sq = lm_ggml_sqr(ctx_eval, codebook_dc);
    lm_ggml_tensor * residual_norm = lm_ggml_sum_rows(ctx_eval, residual_sq); // [1, t]
    lm_ggml_tensor * codebook_norm = lm_ggml_cont(ctx_eval, lm_ggml_transpose(ctx_eval, lm_ggml_sum_rows(ctx_eval, codebook_sq))); // [cbs, 1]

    lm_ggml_tensor * dot = lm_ggml_mul_mat(ctx_eval, codebook_dc, residual_ct); // [cbs, t]
    lm_ggml_tensor * dist = lm_ggml_add(
        ctx_eval,
        lm_ggml_repeat(ctx_eval, codebook_norm, dot),
        lm_ggml_repeat(ctx_eval, residual_norm, dot));
    dist = lm_ggml_sub(ctx_eval, dist, lm_ggml_scale(ctx_eval, dot, 2.0f));

    lm_ggml_tensor * indices = codec_rvq_argmin_map_custom1(ctx_eval, dist);
    if (indices == nullptr) {
        return false;
    }

    lm_ggml_tensor * quantized = lm_ggml_get_rows(ctx_eval, codebook_dc, indices); // [d, t]
    if (quantized == nullptr) {
        return false;
    }

    out->indices = indices;
    out->residual = lm_ggml_sub(ctx_eval, residual_ct, quantized);
    return out->residual != nullptr;
}
