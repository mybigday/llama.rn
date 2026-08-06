#include "rvq.h"

#include <ggml.h>

// Native ggml implementation of the per-frame Euclidean-NN search and
// residual update used by the legacy RVQ codecs (DAC / Mimi / WavTokenizer).
//
// Identity used:
//   argmin_v ‖r_t − cb_v‖² = argmax_v (2·r_t · cb_v − ‖cb_v‖²)
//
// so the search is one mul_mat against the codebook plus a per-codebook
// bias (the squared-norms vector, computed in-graph from the codebook).
// `lm_ggml_argmax` reduces along ne[0]; with shape (V, t) that gives the
// per-frame index directly.

lm_ggml_tensor * codec_rvq_argmin_map_custom1(
    lm_ggml_context * ctx_eval,
    lm_ggml_tensor * distances_ct) {
    if (ctx_eval == nullptr || distances_ct == nullptr ||
        distances_ct->type != LM_GGML_TYPE_F32 ||
        distances_ct->ne[0] <= 0 || distances_ct->ne[1] <= 0) {
        return nullptr;
    }
    // argmin_v dist = argmax_v(-dist).
    lm_ggml_tensor * neg = lm_ggml_scale(ctx_eval, distances_ct, -1.0f);
    return lm_ggml_argmax(ctx_eval, neg);   // (t,) I32
}

lm_ggml_tensor * codec_rvq_select_indices_ggml(
    lm_ggml_context * ctx_eval,
    lm_ggml_tensor * residual_ct,
    lm_ggml_tensor * codebook_dc) {
    if (ctx_eval == nullptr || residual_ct == nullptr || codebook_dc == nullptr) {
        return nullptr;
    }
    if (residual_ct->type != LM_GGML_TYPE_F32 || codebook_dc->type != LM_GGML_TYPE_F32) {
        return nullptr;
    }
    if (residual_ct->ne[0] <= 0 || residual_ct->ne[1] <= 0 ||
        codebook_dc->ne[0] != residual_ct->ne[0] || codebook_dc->ne[1] <= 0) {
        return nullptr;
    }

    lm_ggml_tensor * residual = lm_ggml_cont(ctx_eval, residual_ct);
    lm_ggml_tensor * codebook = lm_ggml_cont(ctx_eval, codebook_dc);

    // dots[v, t] = sum_d cb[d, v] * r[d, t].  mul_mat contracts ne[0].
    lm_ggml_tensor * dots = lm_ggml_mul_mat(ctx_eval, codebook, residual);   // (V, t)
    if (dots == nullptr) return nullptr;
    lm_ggml_tensor * dots2 = lm_ggml_scale(ctx_eval, dots, 2.0f);

    // ‖cb_v‖² as a (V,) bias.  `lm_ggml_sum_rows` reduces ne[0] to 1, so the
    // result has shape (1, V); reshape to (V,) for broadcasting along t.
    lm_ggml_tensor * cb_sq = lm_ggml_mul(ctx_eval, codebook, codebook);
    lm_ggml_tensor * cb_sq_rows = lm_ggml_sum_rows(ctx_eval, cb_sq);          // (1, V)
    lm_ggml_tensor * cb_sq_v = lm_ggml_reshape_2d(ctx_eval, cb_sq_rows, codebook->ne[1], 1);  // (V, 1)
    lm_ggml_tensor * cb_sq_b = lm_ggml_repeat(ctx_eval, cb_sq_v, dots2);
    lm_ggml_tensor * scores = lm_ggml_sub(ctx_eval, dots2, cb_sq_b);          // (V, t)

    return lm_ggml_argmax(ctx_eval, scores);                                // (t,) I32
}

bool codec_rvq_build_layer_ggml(
    lm_ggml_context * ctx_eval,
    lm_ggml_tensor * residual_ct,
    lm_ggml_tensor * codebook_dc,
    codec_rvq_layer_result_ggml * out) {
    if (ctx_eval == nullptr || residual_ct == nullptr || codebook_dc == nullptr || out == nullptr) {
        return false;
    }

    lm_ggml_tensor * indices = codec_rvq_select_indices_ggml(ctx_eval, residual_ct, codebook_dc);
    if (indices == nullptr) return false;

    // Reconstructed code-vectors per frame: cb[idx[t]] for t in [0, T).
    // lm_ggml_get_rows on a 2D source (d, V) selects along ne[1], producing
    // (d, t) with each column = cb[:, idx[t]].
    lm_ggml_tensor * z_q = lm_ggml_get_rows(ctx_eval, codebook_dc, indices);   // (d, t)
    if (z_q == nullptr) return false;
    lm_ggml_tensor * residual = lm_ggml_sub(ctx_eval, residual_ct, z_q);
    if (residual == nullptr) return false;

    out->indices = indices;
    out->residual = residual;
    return true;
}
