#include "models.h"

lm_ggml_tensor * clip_graph_mimovl::build_mm(lm_ggml_tensor * w, lm_ggml_tensor * x) const {
    lm_ggml_tensor * cur = lm_ggml_mul_mat(ctx0, w, x);
    lm_ggml_mul_mat_set_prec(cur, LM_GGML_PREC_F32);
    return cur;
}

// MiMoVL vision tower for MiMo-V2.5 (non-Pro). Qwen2.5-VL-shaped ViT, except:
//   1. GQA in attention (32 Q / 8 KV heads, head_dim 64).
//   2. Per-head attention sinks on every windowed layer. The sinks adjust
//      the softmax denominator (equivalently, a virtual extra K column with V=0),
//      so they decay attention weight without contributing to the output.
//   3. Per-layer window-attention mode in hparams.wa_pattern_mode:
//        -1 -> full,  0 -> row-window+sinks,  1 -> col-window+sinks.
//      Col mode transposes the merge-unit grid on entry and restores
//      it on exit. Both patch and rotary orderings are pre-computed
//      host-side.
//   4. 1D banded sliding window (|q-k| > window_size -> -inf) as a
//      single 2D mask broadcast across heads.
//   5. Per-block MLP biases.
lm_ggml_cgraph * clip_graph_mimovl::build() {
    LM_GGML_ASSERT(model.patch_embeddings_0 != nullptr);
    LM_GGML_ASSERT(model.patch_embeddings_1 != nullptr);
    LM_GGML_ASSERT(model.class_embedding == nullptr);
    LM_GGML_ASSERT(hparams.n_head_kv > 0);
    LM_GGML_ASSERT(n_head % hparams.n_head_kv == 0);
    LM_GGML_ASSERT((int) hparams.wa_pattern_mode.size() == n_layer);

    const int batch_size = 1;
    const int n_pos      = n_patches;
    const int n_head_kv  = hparams.n_head_kv;
    const int merge      = hparams.n_merge > 0 ? hparams.n_merge : 2;
    const int merge_unit = merge * merge;
    const int n_units    = n_pos / merge_unit;
    LM_GGML_ASSERT(n_units * merge_unit == n_pos);

    // MiMoVL has head_dim=64 with n_embd=1280, so n_embd is NOT n_head*head_dim
    // (the base class's d_head = n_embd/n_head = 40 is wrong here). Derive
    // head_dim from the fused QKV projection: rows = (n_head + 2*n_head_kv)*head_dim.
    LM_GGML_ASSERT(model.layers[0].qkv_w != nullptr);
    const int qkv_rows     = model.layers[0].qkv_w->ne[1];
    const int head_dim     = qkv_rows / (n_head + 2 * n_head_kv);
    LM_GGML_ASSERT(head_dim * (n_head + 2 * n_head_kv) == qkv_rows);
    const float attn_scale = 1.0f / std::sqrt((float) head_dim);
    const int rope_n_dims = head_dim / 2;
    int mrope_sections[4] = {rope_n_dims/2, rope_n_dims/2, 0, 0};

    // Patch embed: Conv3D(kt=2) split into two Conv2D, then interleave-merge
    // along the height axis to match the merge-tile token order.
    lm_ggml_tensor * inp_raw = build_inp_raw();
    lm_ggml_tensor * inp = lm_ggml_conv_2d(ctx0, model.patch_embeddings_0, inp_raw,
                                     patch_size, patch_size, 0, 0, 1, 1);
    {
        lm_ggml_tensor * inp_1 = lm_ggml_conv_2d(ctx0, model.patch_embeddings_1, inp_raw,
                                           patch_size, patch_size, 0, 0, 1, 1);
        inp = lm_ggml_add(ctx0, inp, inp_1);

        LM_GGML_ASSERT(img.nx % (patch_size * 2) == 0);
        LM_GGML_ASSERT(img.ny % (patch_size * 2) == 0);

        inp = lm_ggml_permute(ctx0, inp, 1, 2, 0, 3);  // [w,h,c,b] -> [c,w,h,b]
        inp = lm_ggml_cont_4d(ctx0, inp, n_embd * 2, n_patches_x / 2, n_patches_y, batch_size);
        inp = lm_ggml_reshape_4d(ctx0, inp, n_embd * 2, n_patches_x / 2, 2, batch_size * (n_patches_y / 2));
        inp = lm_ggml_permute(ctx0, inp, 0, 2, 1, 3);
        inp = lm_ggml_cont_3d(ctx0, inp, n_embd, n_patches_x * n_patches_y, batch_size);
    }
    cb(inp, "patch_embed", -1);

    lm_ggml_tensor * positions_row = lm_ggml_new_tensor_1d(ctx0, LM_GGML_TYPE_I32, n_pos * 4);
    lm_ggml_set_name(positions_row, "mimovl_positions_row");
    lm_ggml_set_input(positions_row);

    lm_ggml_tensor * positions_col = lm_ggml_new_tensor_1d(ctx0, LM_GGML_TYPE_I32, n_pos * 4);
    lm_ggml_set_name(positions_col, "mimovl_positions_col");
    lm_ggml_set_input(positions_col);

    // idx_col is the col-major merge-unit permutation. Take it as F32 so we can
    // derive the inverse permutation in-graph via lm_ggml_argsort;
    // lm_ggml_get_rows requires its index tensor to be I32, so cast back as well.
    lm_ggml_tensor * idx_col_f = lm_ggml_new_tensor_1d(ctx0, LM_GGML_TYPE_F32, n_units);
    lm_ggml_set_name(idx_col_f, "mimovl_idx_col");
    lm_ggml_set_input(idx_col_f);
    lm_ggml_tensor * idx_col     = lm_ggml_cast(ctx0, idx_col_f, LM_GGML_TYPE_I32);
    lm_ggml_tensor * idx_col_inv = lm_ggml_argsort(ctx0, idx_col_f, LM_GGML_SORT_ORDER_ASC);

    lm_ggml_tensor * window_mask = lm_ggml_new_tensor_2d(ctx0, LM_GGML_TYPE_F32, n_pos, n_pos);
    lm_ggml_set_name(window_mask, "mimovl_window_mask");
    lm_ggml_set_input(window_mask);

    lm_ggml_tensor * window_mask_attn = (flash_attn_type == CLIP_FLASH_ATTN_TYPE_ENABLED)
        ? lm_ggml_cast(ctx0, window_mask, LM_GGML_TYPE_F16)
        : window_mask;

    // Reorder helper: permute patches at merge-unit granularity. The patch
    // sequence is laid out as n_units groups of merge_unit (=4) consecutive
    // patches; the row<->col transpose only permutes whole groups. We keep
    // the per-group (h,w) ordering intact by reshaping to
    // [n_embd*merge_unit, n_units] before lm_ggml_get_rows.
    auto reorder = [&](lm_ggml_tensor * x, lm_ggml_tensor * idx) {
        lm_ggml_tensor * y = lm_ggml_reshape_2d(ctx0, x, n_embd * merge_unit, n_units);
        y = lm_ggml_get_rows(ctx0, y, idx);
        return lm_ggml_reshape_3d(ctx0, y, n_embd, n_pos, batch_size);
    };

    lm_ggml_tensor * inpL = inp;
    int prev_mode = -1;

    for (int il = 0; il < n_layer; il++) {
        const auto & layer = model.layers[il];
        const int  mode    = hparams.wa_pattern_mode[il];
        const bool is_full = (mode == -1);
        const bool is_col  = (mode == 1);

        // Reorder transitions on entry/exit of a col-mode run.
        if (is_col && prev_mode != 1) {
            inpL = reorder(inpL, idx_col);
            cb(inpL, "reorder_to_col", il);
        } else if (!is_col && prev_mode == 1) {
            inpL = reorder(inpL, idx_col_inv);
            cb(inpL, "reorder_to_row", il);
        }

        lm_ggml_tensor * cur = inpL;

        // Pre-attention RMSNorm.
        cur = build_norm(cur, layer.ln_1_w, layer.ln_1_b, NORM_TYPE_RMS, eps, il);
        cb(cur, "ln1", il);

        // Fused QKV with GQA.
        lm_ggml_tensor * qkv = build_mm(layer.qkv_w, cur);
        qkv = lm_ggml_add(ctx0, qkv, layer.qkv_b);

        const size_t row    = lm_ggml_row_size(qkv->type, head_dim);
        const size_t off_k  = lm_ggml_row_size(qkv->type, n_head    * head_dim);
        const size_t off_v  = lm_ggml_row_size(qkv->type, (n_head + n_head_kv) * head_dim);

        lm_ggml_tensor * Qcur = lm_ggml_view_3d(ctx0, qkv, head_dim, n_head,    n_pos, row, qkv->nb[1], 0);
        lm_ggml_tensor * Kcur = lm_ggml_view_3d(ctx0, qkv, head_dim, n_head_kv, n_pos, row, qkv->nb[1], off_k);
        lm_ggml_tensor * Vcur = lm_ggml_view_3d(ctx0, qkv, head_dim, n_head_kv, n_pos, row, qkv->nb[1], off_v);

        cb(Qcur, "Qcur", il);
        cb(Kcur, "Kcur", il);
        cb(Vcur, "Vcur", il);

        // 2D RoPE
        lm_ggml_tensor * pos = is_col ? positions_col : positions_row;
        Qcur = lm_ggml_rope_multi(ctx0, Qcur, pos, nullptr, rope_n_dims, mrope_sections, LM_GGML_ROPE_TYPE_VISION, 32768, 10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
        Kcur = lm_ggml_rope_multi(ctx0, Kcur, pos, nullptr, rope_n_dims, mrope_sections, LM_GGML_ROPE_TYPE_VISION, 32768, 10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
        cb(Qcur, "Qcur_rope", il);
        cb(Kcur, "Kcur_rope", il);

        // Full layers: plain attention. Windowed layers: banded mask and per-head sinks.
        lm_ggml_tensor * mask  = is_full ? nullptr : window_mask_attn;
        lm_ggml_tensor * sinks = is_full ? nullptr : layer.attn_sinks;
        if (!is_full) {
            LM_GGML_ASSERT(layer.attn_sinks != nullptr);
        }
        lm_ggml_tensor * attn_out = build_attn(layer.o_w, layer.o_b, Qcur, Kcur, Vcur, mask, attn_scale, il, sinks);
        cb(attn_out, "attn_out", il);

        // Residual 1.
        cur = lm_ggml_add(ctx0, attn_out, inpL);
        inpL = cur;
        cb(cur, "ffn_inp", il);

        // Pre-FFN RMSNorm.
        cur = build_norm(cur, layer.ln_2_w, layer.ln_2_b, NORM_TYPE_RMS, eps, il);
        cb(cur, "ffn_inp_normed", il);

        // SwiGLU MLP with biases
        cur = build_ffn(cur,
            layer.ff_up_w,   layer.ff_up_b,
            layer.ff_gate_w, layer.ff_gate_b,
            layer.ff_down_w, layer.ff_down_b,
            hparams.ffn_op, il);
        cb(cur, "ffn_out", il);

        // Residual 2.
        cur = lm_ggml_add(ctx0, inpL, cur);
        cb(cur, "layer_out", il);

        inpL = cur;
        prev_mode = mode;
    }

    // If the last block was col-mode, undo the transpose so the merger sees patches in row order.
    if (prev_mode == 1) {
        inpL = reorder(inpL, idx_col_inv);
        cb(inpL, "reorder_to_row_final", -1);
    }

    // Merger: post-LayerNorm
    inpL = build_norm(inpL, model.post_ln_w, model.post_ln_b, NORM_TYPE_NORMAL, 1e-6f, n_layer);
    cb(inpL, "post_ln", -1);

    // Spatial merge: pack each merge_unit (=4) of patches into a single
    // (n_embd*merge_unit)-wide row, then run the 2-layer MLP.
    lm_ggml_tensor * embeddings = lm_ggml_reshape_3d(ctx0, inpL, n_embd * merge_unit, n_units, batch_size);
    embeddings = build_ffn(embeddings,
        model.mm_0_w, nullptr,
        nullptr,      nullptr,
        model.mm_1_w, nullptr,
        FFN_GELU, -1);
    cb(embeddings, "vit_out", -1);

    lm_ggml_build_forward_expand(gf, embeddings);
    return gf;
}
