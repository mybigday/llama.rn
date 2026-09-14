#include "lm_attn.h"
#include "ggml_ops.h"
#include "../runtime/tensor_utils.h"

#include <cmath>

ggml_tensor * codec_op_lm_attn_ctx_dth(
    ggml_context * ctx,
    ggml_tensor * q_dth,
    ggml_tensor * k_dth,
    ggml_tensor * v_dth,
    const codec_lm_attn_params * params) {

    if (ctx == nullptr || q_dth == nullptr || k_dth == nullptr || v_dth == nullptr) {
        return nullptr;
    }
    if (q_dth->ne[0] != k_dth->ne[0] || q_dth->ne[0] != v_dth->ne[0] ||
        q_dth->ne[1] != k_dth->ne[1] || q_dth->ne[1] != v_dth->ne[1] ||
        q_dth->ne[2] != k_dth->ne[2] || q_dth->ne[2] != v_dth->ne[2]) {
        return nullptr;
    }

    const int32_t head_dim = (int32_t) q_dth->ne[0];
    const float scale = (params != nullptr && params->scale > 0.0f)
        ? params->scale
        : (1.0f / std::sqrt((float) std::max(1, head_dim)));
    const bool causal = params != nullptr && params->causal;

    ggml_tensor * k_cont = ggml_cont(ctx, k_dth);
    ggml_tensor * attn_scores = ggml_mul_mat(ctx, k_cont, q_dth); // [t, t, h]
    if (attn_scores == nullptr) {
        return nullptr;
    }

    attn_scores = ggml_scale_inplace(ctx, attn_scores, scale);
    if (causal) {
        attn_scores = ggml_diag_mask_inf_inplace(ctx, attn_scores, 0);
    }
    if (params != nullptr && params->n_valid > 0 && params->n_valid < (int32_t) q_dth->ne[1]) {
        // Padding mask: set scores for keys at positions >= n_valid to -inf.
        // attn_scores shape is (k, q, h).  Mask the *last* (t - n_valid) rows
        // of ne[0] (the key dim).
        const int64_t t = q_dth->ne[1];
        const int64_t h = q_dth->ne[2];
        const int32_t n_valid = params->n_valid;
        // Build a 1D bias [t]: 0 for k < n_valid, -inf otherwise.  Use arange
        // + scale_bias + clamp + scale (mirrors the windowed-mask trick).
        ggml_tensor * arange_k = ggml_arange(ctx, 0.0f, (float) t, 1.0f);                  // [t]
        // diff = k - n_valid + 1; mask if diff > 0.  Equivalently mask if k > n_valid - 1.
        ggml_tensor * diff = ggml_scale_bias(ctx, arange_k, 1.0f, (float) -(n_valid - 1));
        // We want -inf where diff > 0, i.e. clip lo=0 then negate so clipped >= 0
        // becomes <= 0, then scale by -1e30.
        ggml_tensor * clipped = ggml_clamp(ctx, diff, 0.0f, 1e9f);                          // >= 0
        ggml_tensor * bias_1d = ggml_scale(ctx, clipped, -1e30f);                           // -inf where k >= n_valid
        // Broadcast to (t, t, h).
        ggml_tensor * bias_2d = ggml_reshape_2d(ctx, bias_1d, t, 1);
        ggml_tensor * bias_3d = ggml_reshape_3d(ctx, bias_2d, t, 1, 1);
        ggml_tensor * bias_dst = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, t, t, h);
        ggml_tensor * bias_rep = ggml_repeat(ctx, bias_3d, bias_dst);
        attn_scores = ggml_add(ctx, attn_scores, bias_rep);
    }
    if (causal && params != nullptr && params->window > 0 && params->window < (int32_t) q_dth->ne[1]) {
        // Sliding-window causal mask: in addition to the upper triangle
        // (handled by ggml_diag_mask_inf above), zero out keys older than
        // `window-1` steps.  attn_scores shape is (k, q, h).  We add a
        // -inf bias where `j < i - window + 1` (i query, j key).
        const int64_t t = q_dth->ne[1];
        const int64_t h = q_dth->ne[2];
        const int32_t window = params->window;
        // Build the mask in-graph from arange tensors.
        ggml_tensor * arange = ggml_arange(ctx, 0.0f, (float) t, 1.0f);                  // [t]
        ggml_tensor * row_q = ggml_reshape_2d(ctx, arange, 1, t);                         // [1, t]  (i along ne[1])
        ggml_tensor * col_k = ggml_reshape_2d(ctx, arange, t, 1);                         // [t, 1]  (j along ne[0])
        ggml_tensor * tmpl = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, t, t);
        ggml_tensor * row_q_rep = ggml_repeat(ctx, row_q, tmpl);
        ggml_tensor * col_k_rep = ggml_repeat(ctx, col_k, tmpl);
        // diff = j - (i - window + 1)  → if diff < 0 the key is too old; mask.
        ggml_tensor * diff = ggml_sub(ctx, col_k_rep, row_q_rep);
        diff = ggml_scale_bias(ctx, diff, 1.0f, (float) (window - 1));
        // Build a mask tensor: where diff < 0, output = -inf, else 0.  We
        // approximate this by clipping diff to (-large, 0) and scaling.
        // Concretely: bias = min(diff, 0) * INF_SCALE.
        ggml_tensor * clipped = ggml_clamp(ctx, diff, -1e9f, 0.0f);
        ggml_tensor * bias = ggml_scale(ctx, clipped, 1e30f);   // -inf where diff < 0
        ggml_tensor * bias_3d = ggml_reshape_3d(ctx, bias, t, t, 1);
        ggml_tensor * bias_dst = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, t, t, h);
        ggml_tensor * bias_rep = ggml_repeat(ctx, bias_3d, bias_dst);
        attn_scores = ggml_add(ctx, attn_scores, bias_rep);
    }

    ggml_tensor * attn_probs = ggml_soft_max(ctx, attn_scores);
    if (attn_probs == nullptr) {
        return nullptr;
    }

    ggml_tensor * v_tdh = ggml_cont(ctx, ggml_permute(ctx, v_dth, 1, 0, 2, 3));
    ggml_tensor * out_dth = ggml_mul_mat(ctx, v_tdh, attn_probs); // [d, t, h]
    if (params != nullptr && params->n_valid > 0 && params->n_valid < (int32_t) q_dth->ne[1]) {
        // Zero out output rows for queries at positions >= n_valid (mirrors
        // MOSS's `torch.where(valid_q, out, 0)`).  Build a per-time mask
        // valid_q[t] = 1 if t < n_valid else 0, then broadcast-multiply.
        const int64_t t = q_dth->ne[1];
        const int64_t d = q_dth->ne[0];
        const int64_t h = q_dth->ne[2];
        const int32_t n_valid = params->n_valid;
        ggml_tensor * arange_q = ggml_arange(ctx, 0.0f, (float) t, 1.0f);                  // [t]
        // diff = t_idx - (n_valid - 1); want mask=1 if diff <= 0 else 0.
        ggml_tensor * diff = ggml_scale_bias(ctx, arange_q, 1.0f, (float) -(n_valid - 1));
        // valid = clamp(diff, -1, 0) maps:
        //   t < n_valid: diff <= 0 → clamp gives diff (in [-(n_valid-1), 0])
        //   t >= n_valid: diff > 0 → clamp gives 0
        // Then valid = clamp + 1 ∈ [1 - (n_valid-1), 1] for valid, =1 for invalid.
        // That's not what we want.  Use a different construction:
        //   step_neg = clamp(diff, 0, 1) ∈ {0 if t<n_valid (diff<=0), 1 if t>=n_valid (diff>=1)}
        //   valid    = 1 - step_neg
        // For 0 < diff < 1 we'd be in trouble, but diff is integer-valued so OK.
        ggml_tensor * step_neg = ggml_clamp(ctx, diff, 0.0f, 1.0f);                         // 0 for valid, 1 for padded
        ggml_tensor * valid_q = ggml_scale_bias(ctx, step_neg, -1.0f, 1.0f);                // 1 for valid, 0 for padded
        // Broadcast to (d, t, h).
        ggml_tensor * valid_q_2d = ggml_reshape_2d(ctx, valid_q, 1, t);
        ggml_tensor * valid_q_3d = ggml_reshape_3d(ctx, valid_q_2d, 1, t, 1);
        ggml_tensor * valid_dst = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d, t, h);
        ggml_tensor * valid_rep = ggml_repeat(ctx, valid_q_3d, valid_dst);
        out_dth = ggml_mul(ctx, out_dth, valid_rep);
    }
    return out_dth;
}

ggml_tensor * codec_op_rel_shift_espnet(ggml_context * ctx, ggml_tensor * x) {
    if (ctx == nullptr || x == nullptr) return nullptr;
    const int64_t two_t1 = x->ne[0];
    const int64_t t = x->ne[1];
    const int64_t h = x->ne[2];
    if (two_t1 != 2 * t - 1) return nullptr;

    // 1. Pad zeros on the LEFT of ne[0] → (2t, t, h).
    ggml_tensor * zp = ggml_new_tensor_3d(ctx, x->type, 1, t, h);
    zp = ggml_scale(ctx, zp, 0.0f);
    ggml_tensor * x_pad = ggml_concat(ctx, zp, x, /*dim=*/0);
    if (x_pad == nullptr) return nullptr;

    // 2. Reshape (2t, t, h) → (t, 2t, h) (same memory, different view).
    ggml_tensor * x_pad_cont = ggml_cont(ctx, x_pad);
    ggml_tensor * x_view = ggml_reshape_3d(ctx, x_pad_cont, t, 2 * t, h);

    // 3. Drop first row of ne[1] → (t, 2t-1, h).
    ggml_tensor * x_drop = ggml_view_3d(
        ctx, x_view,
        /*ne0=*/t, /*ne1=*/2 * t - 1, /*ne2=*/h,
        /*nb1=*/x_view->nb[1],
        /*nb2=*/x_view->nb[2],
        /*offset=*/x_view->nb[1]);
    x_drop = ggml_cont(ctx, x_drop);

    // 4. Reshape back to (2t-1, t, h), then take first t entries of ne[0].
    ggml_tensor * x_back = ggml_reshape_3d(ctx, x_drop, 2 * t - 1, t, h);
    ggml_tensor * x_out = ggml_view_3d(
        ctx, x_back,
        /*ne0=*/t, /*ne1=*/t, /*ne2=*/h,
        /*nb1=*/x_back->nb[1],
        /*nb2=*/x_back->nb[2],
        /*offset=*/0);
    return ggml_cont(ctx, x_out);
}

ggml_tensor * codec_op_lm_attn_rel_pos_dth(
    ggml_context * ctx,
    ggml_tensor * q_dth,
    ggml_tensor * k_dth,
    ggml_tensor * v_dth,
    ggml_tensor * p_dth,
    ggml_tensor * pos_bias_u,
    ggml_tensor * pos_bias_v,
    const codec_lm_attn_params * params) {
    if (ctx == nullptr || q_dth == nullptr || k_dth == nullptr || v_dth == nullptr ||
        p_dth == nullptr || pos_bias_u == nullptr || pos_bias_v == nullptr) {
        return nullptr;
    }
    const int64_t head_dim = q_dth->ne[0];
    const int64_t t = q_dth->ne[1];
    const int64_t h = q_dth->ne[2];
    if (k_dth->ne[0] != head_dim || v_dth->ne[0] != head_dim || p_dth->ne[0] != head_dim ||
        k_dth->ne[1] != t || v_dth->ne[1] != t || p_dth->ne[1] != 2 * t - 1 ||
        k_dth->ne[2] != h || v_dth->ne[2] != h || p_dth->ne[2] != h) {
        return nullptr;
    }
    const float scale = (params != nullptr && params->scale > 0.0f)
        ? params->scale
        : (1.0f / std::sqrt((float) std::max<int64_t>(1, head_dim)));

    // Add per-head bias broadcast over t: q + bias_u, q + bias_v.
    auto add_bias = [&](ggml_tensor * q, ggml_tensor * bias_dh) -> ggml_tensor * {
        ggml_tensor * b3 = ggml_reshape_3d(ctx, bias_dh, head_dim, 1, h);
        return ggml_add(ctx, q, ggml_repeat(ctx, b3, q));
    };
    ggml_tensor * q_u = add_bias(q_dth, pos_bias_u);
    ggml_tensor * q_v = add_bias(q_dth, pos_bias_v);
    if (q_u == nullptr || q_v == nullptr) return nullptr;

    // matrix_ac = q_u · k.T  → (t, t, h).
    ggml_tensor * mat_ac = ggml_mul_mat(ctx, ggml_cont(ctx, k_dth), q_u);
    // matrix_bd = q_v · p.T  → (2t-1, t, h), then rel-shift to (t, t, h).
    ggml_tensor * mat_bd = ggml_mul_mat(ctx, ggml_cont(ctx, p_dth), q_v);
    if (mat_ac == nullptr || mat_bd == nullptr) return nullptr;
    mat_bd = codec_op_rel_shift_espnet(ctx, mat_bd);
    if (mat_bd == nullptr) return nullptr;

    ggml_tensor * scores = ggml_add(ctx, mat_ac, mat_bd);
    scores = ggml_scale(ctx, scores, scale);
    ggml_tensor * attn_w = ggml_soft_max(ctx, scores);

    ggml_tensor * v_tdh = ggml_cont(ctx, ggml_permute(ctx, v_dth, 1, 0, 2, 3));
    return ggml_mul_mat(ctx, v_tdh, attn_w);
}

ggml_tensor * codec_op_lm_attn_rel_key_dth(
    ggml_context * ctx,
    ggml_tensor * q_dth,
    ggml_tensor * k_dth,
    ggml_tensor * v_dth,
    ggml_tensor * dist_emb_dn,
    ggml_tensor * bucket_idx_1d,
    const codec_lm_attn_params * params) {
    if (ctx == nullptr || q_dth == nullptr || k_dth == nullptr || v_dth == nullptr ||
        dist_emb_dn == nullptr || bucket_idx_1d == nullptr) {
        return nullptr;
    }
    const int64_t head_dim = q_dth->ne[0];
    const int64_t t = q_dth->ne[1];
    const int64_t h = q_dth->ne[2];
    if (k_dth->ne[0] != head_dim || v_dth->ne[0] != head_dim ||
        k_dth->ne[1] != t || v_dth->ne[1] != t ||
        k_dth->ne[2] != h || v_dth->ne[2] != h ||
        dist_emb_dn->ne[0] != head_dim ||
        bucket_idx_1d->ne[0] != t * t) {
        return nullptr;
    }
    const float scale = (params != nullptr && params->scale > 0.0f)
        ? params->scale
        : (1.0f / std::sqrt((float) std::max<int64_t>(1, head_dim)));

    // Standard attention scores: [t_k, t_q, h].
    ggml_tensor * ac = ggml_mul_mat(ctx, ggml_cont(ctx, k_dth), q_dth);
    if (ac == nullptr) return nullptr;

    // Gather E[d, t_k * t_q] = D[d, bucket(t_k, t_q)]. The bucket index is laid
    // out row-major with t_k inner, t_q outer, so reshape to (t_k, t_q).
    ggml_tensor * E_flat = ggml_get_rows(ctx, dist_emb_dn, bucket_idx_1d);  // [d, t*t]
    if (E_flat == nullptr) return nullptr;
    ggml_tensor * E_3d = ggml_reshape_3d(ctx, E_flat, head_dim, t, t);      // [d, t_k, t_q]

    // Permute Q to per-t_q batch: [d, t_q, h] -> [d, h, t_q].
    ggml_tensor * q_dh_tq = ggml_cont(ctx, ggml_permute(ctx, q_dth, 0, 2, 1, 3));  // [d, h, t_q]

    // Per-t_q batched mul_mat: result ne=(t_k, h, t_q). mul_mat contracts d.
    ggml_tensor * rel = ggml_mul_mat(ctx, E_3d, q_dh_tq);                          // [t_k, h, t_q]
    if (rel == nullptr) return nullptr;

    // Permute to [t_k, t_q, h] to align with `ac`.
    rel = ggml_cont(ctx, ggml_permute(ctx, rel, 0, 2, 1, 3));

    ggml_tensor * scores = ggml_add(ctx, ac, rel);
    scores = ggml_scale(ctx, scores, scale);
    ggml_tensor * probs = ggml_soft_max(ctx, scores);

    ggml_tensor * v_tdh = ggml_cont(ctx, ggml_permute(ctx, v_dth, 1, 0, 2, 3));
    return ggml_mul_mat(ctx, v_tdh, probs);
}


// View the first `T` slices of a 3D `(in, out, N)` weight along ne[2].
static inline ggml_tensor * lm_per_pos_weight_slice(
        ggml_context * ctx, ggml_tensor * w_3d, int32_t T) {
    return ggml_view_3d(
        ctx, w_3d,
        w_3d->ne[0], w_3d->ne[1], (int64_t) T,
        w_3d->nb[1], w_3d->nb[2],
        /*offset=*/0);
}

ggml_tensor * codec_op_lm_per_pos_linear(
        ggml_context * ctx,
        ggml_tensor * w,
        ggml_tensor * x_2d,
        int32_t out_dim,
        int32_t T) {
    if (ctx == nullptr || w == nullptr || x_2d == nullptr) return nullptr;
    // 2D weights have ne[2] == 1 (ggml tensors are internally 4D with
    // missing dims set to 1).  3D per-pos weights have ne[2] = N >= T.
    if (w->ne[2] <= 1) {
        // Shared: plain matmul, gives (out_dim, T).  mul_mat handles
        // F16/BF16 src[0] natively — let codec_graph_mat_lhs pass those
        // through and cast only true quantized types.  Removing the
        // unconditional F32 cast that used to live here saves a full
        // weight dequant per graph execution, the dominant cost in the
        // residual_depth_ar prefix-recompute regime where this helper
        // runs depth_layers × 7 times per cb step.
        //
        // Force F32 accumulation explicitly: some backends default to
        // F16 accumulation for F16 src[0], which adds a few millibits
        // of drift versus the legacy cast-to-F32 path.  PREC_F32 keeps
        // the dot product accumulator in F32 without materialising a
        // dequanted weight tensor — best of both worlds.
        // Dequant F16/BF16 weights to F32 before mul_mat.  ggml_mul_mat with
        // an F16 src[0] converts src[1] (the activations) to F16 for the
        // F16 vec_dot path — which overflows to +/-inf when an activation
        // exceeds F16 max (65504).  The Qwen3-TTS depth FFN legitimately
        // produces SwiGLU activations ~1.4e5 (large-activation channels),
        // so the F16 activation cast turns them into inf -> NaN logits.
        // Casting the weight to F32 keeps both operands F32 (no activation
        // downcast); GGML_PREC_F32 alone does not prevent the src[1] cast.
        // codec_graph_cast_f32 is a no-op for weights already F32.
        ggml_tensor * w_lhs = codec_graph_cast_f32(ctx, w);
        ggml_tensor * y     = ggml_mul_mat(ctx, w_lhs, x_2d);
        ggml_mul_mat_set_prec(y, GGML_PREC_F32);
        return y;
    }
    // 3D per-pos branch: still cast to F32 because the broadcast
    // arrangement puts the weight slice as mul_mat's src[1] (which must
    // be F32) — see comment below.
    ggml_tensor * w_f32   = codec_graph_cast_f32(ctx, w);
    ggml_tensor * w_slice = lm_per_pos_weight_slice(ctx, w_f32, T);
    const int64_t in_dim  = x_2d->ne[0];
    ggml_tensor * x_3d    = ggml_reshape_3d(ctx, x_2d, in_dim, 1, (int64_t) T);
    // ggml's batch broadcast requires `b.ne[2] % a.ne[2] == 0` — only
    // `a` may broadcast.  Putting the input as `a` keeps the rule
    // satisfied for both balanced (a.ne[2] == b.ne[2] == T) and
    // broadcast-from-1 (when x is a single position repeated) cases.
    ggml_tensor * y_3d    = ggml_mul_mat(ctx, x_3d, w_slice);
    return ggml_reshape_2d(ctx, y_3d, (int64_t) out_dim, (int64_t) T);
}

ggml_tensor * codec_op_lm_llama_depth_block(
        ggml_context * ctx,
        ggml_tensor * x_ht,
        ggml_tensor * attn_norm_w,
        ggml_tensor * qw, ggml_tensor * kw, ggml_tensor * vw, ggml_tensor * ow,
        ggml_tensor * q_norm_w, ggml_tensor * k_norm_w,
        ggml_tensor * t_pos, ggml_tensor * freq_factors,
        ggml_tensor * ffn_norm_w,
        ggml_tensor * ffn_gate, ggml_tensor * ffn_up, ggml_tensor * ffn_down,
        int32_t head_dim,
        int32_t n_heads,
        int32_t n_kv_heads,
        float   rope_theta,
        float   rms_eps,
        int32_t rope_mode,
        bool    use_rope) {
    if (ctx == nullptr || x_ht == nullptr) return nullptr;

    const int64_t T      = x_ht->ne[1];
    const int32_t q_dim  = n_heads    * head_dim;
    const int32_t kv_dim = n_kv_heads * head_dim;

    // ── Attention ──────────────────────────────────────────────────
    ggml_tensor * h = codec_op_rms_norm_ct(ctx, x_ht, rms_eps, attn_norm_w);

    ggml_tensor * q = codec_op_lm_per_pos_linear(ctx, qw, h, q_dim,  T);
    ggml_tensor * k = codec_op_lm_per_pos_linear(ctx, kw, h, kv_dim, T);
    ggml_tensor * v = codec_op_lm_per_pos_linear(ctx, vw, h, kv_dim, T);

    q = ggml_reshape_3d(ctx, q, head_dim, n_heads,    T);
    k = ggml_reshape_3d(ctx, k, head_dim, n_kv_heads, T);
    v = ggml_reshape_3d(ctx, v, head_dim, n_kv_heads, T);

    if (q_norm_w != nullptr && k_norm_w != nullptr) {
        // Per-head RMSNorm on q/k (Qwen3 family, LFM2-Audio).
        q = codec_op_rms_norm_ct(ctx, q, rms_eps, q_norm_w);
        k = codec_op_rms_norm_ct(ctx, k, rms_eps, k_norm_w);
    }

    if (use_rope) {
        const int32_t rope_n_dims = head_dim;
        const int32_t n_ctx_orig  = 2048;
        const float   freq_scale  = 1.0f;
        const float   ext_factor  = 0.0f;
        const float   attn_factor = 1.0f;
        const float   beta_fast   = 32.0f;
        const float   beta_slow   = 1.0f;

        q = ggml_rope_ext(ctx, q, t_pos, freq_factors, rope_n_dims, rope_mode,
                          n_ctx_orig, rope_theta, freq_scale, ext_factor,
                          attn_factor, beta_fast, beta_slow);
        k = ggml_rope_ext(ctx, k, t_pos, freq_factors, rope_n_dims, rope_mode,
                          n_ctx_orig, rope_theta, freq_scale, ext_factor,
                          attn_factor, beta_fast, beta_slow);
    }

    // GQA via ggml_mul_mat's automatic n_kv_heads -> n_heads broadcast
    // on the batch axis (n_heads must be a multiple of n_kv_heads).
    ggml_tensor * q_p = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));
    ggml_tensor * k_p = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));

    ggml_tensor * scores = ggml_mul_mat(ctx, k_p, q_p);
    scores = ggml_scale(ctx, scores, 1.0f / std::sqrt((float) head_dim));
    scores = ggml_diag_mask_inf(ctx, scores, /*n_past=*/0);
    scores = ggml_soft_max(ctx, scores);

    ggml_tensor * v_p = ggml_cont(ctx, ggml_permute(ctx, v, 1, 2, 0, 3));
    ggml_tensor * attn = ggml_mul_mat(ctx, v_p, scores);
    attn = ggml_cont(ctx, ggml_permute(ctx, attn, 0, 2, 1, 3));
    attn = ggml_reshape_2d(ctx, attn, (int64_t) q_dim, T);

    const int32_t hidden = (int32_t) x_ht->ne[0];
    ggml_tensor * o = codec_op_lm_per_pos_linear(ctx, ow, attn, hidden, T);
    x_ht = ggml_add(ctx, x_ht, o);

    // ── FFN (SwiGLU) ───────────────────────────────────────────────
    h = codec_op_rms_norm_ct(ctx, x_ht, rms_eps, ffn_norm_w);

    // ffn_gate / ffn_up output dim = `ne[1]` of either weight (2D) or
    // `ne[1]` of the 3D weight slice (same number).
    const int32_t inter = (int32_t) ffn_gate->ne[1];
    ggml_tensor * gate = codec_op_lm_per_pos_linear(ctx, ffn_gate, h, inter, T);
    ggml_tensor * up   = codec_op_lm_per_pos_linear(ctx, ffn_up,   h, inter, T);
    ggml_tensor * mlp  = ggml_mul(ctx, ggml_silu(ctx, gate), up);
    ggml_tensor * down = codec_op_lm_per_pos_linear(ctx, ffn_down, mlp, hidden, T);
    x_ht = ggml_add(ctx, x_ht, down);

    return x_ht;
}

