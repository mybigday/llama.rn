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
// `ggml_argmax` reduces along ne[0]; with shape (V, t) that gives the
// per-frame index directly.

ggml_tensor * codec_rvq_argmin_map_custom1(
    ggml_context * ctx_eval,
    ggml_tensor * distances_ct) {
    if (ctx_eval == nullptr || distances_ct == nullptr ||
        distances_ct->type != GGML_TYPE_F32 ||
        distances_ct->ne[0] <= 0 || distances_ct->ne[1] <= 0) {
        return nullptr;
    }
    // argmin_v dist = argmax_v(-dist).
    ggml_tensor * neg = ggml_scale(ctx_eval, distances_ct, -1.0f);
    return ggml_argmax(ctx_eval, neg);   // (t,) I32
}

ggml_tensor * codec_rvq_select_indices_ggml(
    ggml_context * ctx_eval,
    ggml_tensor * residual_ct,
    ggml_tensor * codebook_dc) {
    if (ctx_eval == nullptr || residual_ct == nullptr || codebook_dc == nullptr) {
        return nullptr;
    }
    if (residual_ct->type != GGML_TYPE_F32 || codebook_dc->type != GGML_TYPE_F32) {
        return nullptr;
    }
    if (residual_ct->ne[0] <= 0 || residual_ct->ne[1] <= 0 ||
        codebook_dc->ne[0] != residual_ct->ne[0] || codebook_dc->ne[1] <= 0) {
        return nullptr;
    }

    ggml_tensor * residual = ggml_cont(ctx_eval, residual_ct);
    ggml_tensor * codebook = ggml_cont(ctx_eval, codebook_dc);

    // dots[v, t] = sum_d cb[d, v] * r[d, t].  mul_mat contracts ne[0].
    ggml_tensor * dots = ggml_mul_mat(ctx_eval, codebook, residual);   // (V, t)
    if (dots == nullptr) return nullptr;
    ggml_tensor * dots2 = ggml_scale(ctx_eval, dots, 2.0f);

    // ‖cb_v‖² as a (V,) bias.  `ggml_sum_rows` reduces ne[0] to 1, so the
    // result has shape (1, V); reshape to (V,) for broadcasting along t.
    ggml_tensor * cb_sq = ggml_mul(ctx_eval, codebook, codebook);
    ggml_tensor * cb_sq_rows = ggml_sum_rows(ctx_eval, cb_sq);          // (1, V)
    ggml_tensor * cb_sq_v = ggml_reshape_2d(ctx_eval, cb_sq_rows, codebook->ne[1], 1);  // (V, 1)
    ggml_tensor * cb_sq_b = ggml_repeat(ctx_eval, cb_sq_v, dots2);
    ggml_tensor * scores = ggml_sub(ctx_eval, dots2, cb_sq_b);          // (V, t)

    return ggml_argmax(ctx_eval, scores);                                // (t,) I32
}

bool codec_rvq_build_layer_ggml(
    ggml_context * ctx_eval,
    ggml_tensor * residual_ct,
    ggml_tensor * codebook_dc,
    codec_rvq_layer_result_ggml * out) {
    if (ctx_eval == nullptr || residual_ct == nullptr || codebook_dc == nullptr || out == nullptr) {
        return false;
    }

    ggml_tensor * indices = codec_rvq_select_indices_ggml(ctx_eval, residual_ct, codebook_dc);
    if (indices == nullptr) return false;

    // Reconstructed code-vectors per frame: cb[idx[t]] for t in [0, T).
    // ggml_get_rows on a 2D source (d, V) selects along ne[1], producing
    // (d, t) with each column = cb[:, idx[t]].
    ggml_tensor * z_q = ggml_get_rows(ctx_eval, codebook_dc, indices);   // (d, t)
    if (z_q == nullptr) return false;
    ggml_tensor * residual = ggml_sub(ctx_eval, residual_ct, z_q);
    if (residual == nullptr) return false;

    out->indices = indices;
    out->residual = residual;
    return true;
}
