#include "models.h"

lm_ggml_cgraph * clip_graph_qwen2vl::build() {
    LM_GGML_ASSERT(model.patch_bias == nullptr);
    LM_GGML_ASSERT(model.class_embedding == nullptr);

    const int batch_size       = 1;
    const bool use_window_attn = hparams.n_wa_pattern > 0;
    const int n_wa_pattern     = hparams.n_wa_pattern;
    const int n_pos            = n_patches;
    const int num_position_ids = n_pos * 4; // m-rope requires 4 dim per position

    norm_type norm_t = proj_type == PROJECTOR_TYPE_QWEN25VL
        ? NORM_TYPE_RMS // qwen 2.5 vl
        : NORM_TYPE_NORMAL; // qwen 2 vl

    int mrope_sections[4] = {d_head/4, d_head/4, d_head/4, d_head/4};

    lm_ggml_tensor * inp_raw = build_inp_raw();
    lm_ggml_tensor * inp = lm_ggml_conv_2d(ctx0, model.patch_embeddings_0, inp_raw, patch_size, patch_size, 0, 0, 1, 1);

    LM_GGML_ASSERT(img.nx % (patch_size * 2) == 0);
    LM_GGML_ASSERT(img.ny % (patch_size * 2) == 0);

    // second conv dimension
    {
        auto inp_1 = lm_ggml_conv_2d(ctx0, model.patch_embeddings_1, inp_raw, patch_size, patch_size, 0, 0, 1, 1);
        inp = lm_ggml_add(ctx0, inp, inp_1);

        inp = lm_ggml_permute(ctx0, inp, 1, 2, 0, 3);  // [w, h, c, b] -> [c, w, h, b]
        inp = lm_ggml_cont_4d(
            ctx0, inp,
            n_embd * 2, n_patches_x / 2, n_patches_y, batch_size);
        inp = lm_ggml_reshape_4d(
            ctx0, inp,
            n_embd * 2, n_patches_x / 2, 2, batch_size * (n_patches_y / 2));
        inp = lm_ggml_permute(ctx0, inp, 0, 2, 1, 3);
        inp = lm_ggml_cont_3d(
            ctx0, inp,
            n_embd, n_patches_x * n_patches_y, batch_size);
    }

    lm_ggml_tensor * inpL           = inp;
    lm_ggml_tensor * window_mask    = nullptr;
    lm_ggml_tensor * window_idx     = nullptr;
    lm_ggml_tensor * inv_window_idx = nullptr;

    lm_ggml_tensor * positions = lm_ggml_new_tensor_1d(ctx0, LM_GGML_TYPE_I32, num_position_ids);
    lm_ggml_set_name(positions, "positions");
    lm_ggml_set_input(positions);

    // pre-layernorm
    if (model.pre_ln_w) {
        inpL = build_norm(inpL, model.pre_ln_w, model.pre_ln_b, norm_t, eps, -1);
    }

    if (use_window_attn) {
        // handle window attention inputs
        inv_window_idx = lm_ggml_new_tensor_1d(ctx0, LM_GGML_TYPE_I32, n_pos / 4);
        lm_ggml_set_name(inv_window_idx, "inv_window_idx");
        lm_ggml_set_input(inv_window_idx);
        // mask for window attention
        window_mask = lm_ggml_new_tensor_2d(ctx0, LM_GGML_TYPE_F32, n_pos, n_pos);
        lm_ggml_set_name(window_mask, "window_mask");
        lm_ggml_set_input(window_mask);

        // if flash attn is used, we need to pad the mask and cast to f16
        if (flash_attn_type == CLIP_FLASH_ATTN_TYPE_ENABLED) {
            window_mask = lm_ggml_cast(ctx0, window_mask, LM_GGML_TYPE_F16);
        }

        // inpL shape: [n_embd, n_patches_x * n_patches_y, batch_size]
        LM_GGML_ASSERT(batch_size == 1);
        inpL = lm_ggml_reshape_2d(ctx0, inpL, n_embd * 4, n_patches_x * n_patches_y * batch_size / 4);
        inpL = lm_ggml_get_rows(ctx0, inpL, inv_window_idx);
        inpL = lm_ggml_reshape_3d(ctx0, inpL, n_embd, n_patches_x * n_patches_y, batch_size);
    }

    // loop over layers
    for (int il = 0; il < n_layer; il++) {
        const auto & layer = model.layers[il];
        const bool full_attn = use_window_attn ? (il + 1) % n_wa_pattern == 0 : true;

        lm_ggml_tensor * cur = inpL; // inpL = residual, cur = hidden_states

        // layernorm1
        cur = build_norm(cur, layer.ln_1_w, layer.ln_1_b, norm_t, eps, il);
        cb(cur, "ln1", il);

        // self-attention
        {
            lm_ggml_tensor * Qcur = lm_ggml_add(ctx0,
                lm_ggml_mul_mat(ctx0, layer.q_w, cur), layer.q_b);
            lm_ggml_tensor * Kcur = lm_ggml_add(ctx0,
                lm_ggml_mul_mat(ctx0, layer.k_w, cur), layer.k_b);
            lm_ggml_tensor * Vcur = lm_ggml_add(ctx0,
                lm_ggml_mul_mat(ctx0, layer.v_w, cur), layer.v_b);

            Qcur = lm_ggml_reshape_3d(ctx0, Qcur, d_head, n_head, n_patches);
            Kcur = lm_ggml_reshape_3d(ctx0, Kcur, d_head, n_head, n_patches);
            Vcur = lm_ggml_reshape_3d(ctx0, Vcur, d_head, n_head, n_patches);

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            // apply M-RoPE
            Qcur = lm_ggml_rope_multi(
                ctx0, Qcur, positions, nullptr,
                d_head/2, mrope_sections, LM_GGML_ROPE_TYPE_VISION, 32768, 10000, 1, 0, 1, 32, 1);
            Kcur = lm_ggml_rope_multi(
                ctx0, Kcur, positions, nullptr,
                d_head/2, mrope_sections, LM_GGML_ROPE_TYPE_VISION, 32768, 10000, 1, 0, 1, 32, 1);

            cb(Qcur, "Qcur_rope", il);
            cb(Kcur, "Kcur_rope", il);

            lm_ggml_tensor * attn_mask = full_attn ? nullptr : window_mask;

            cur = build_attn(layer.o_w, layer.o_b,
                Qcur, Kcur, Vcur, attn_mask, kq_scale, il);
            cb(cur, "attn_out", il);
        }

        // re-add the layer input, e.g., residual
        cur = lm_ggml_add(ctx0, cur, inpL);

        inpL = cur; // inpL = residual, cur = hidden_states

        cb(cur, "ffn_inp", il);

        // layernorm2
        cur = build_norm(cur, layer.ln_2_w, layer.ln_2_b, norm_t, eps, il);
        cb(cur, "ffn_inp_normed", il);

        // ffn
        cur = build_ffn(cur,
            layer.ff_up_w, layer.ff_up_b,
            layer.ff_gate_w, layer.ff_gate_b,
            layer.ff_down_w, layer.ff_down_b,
            hparams.ffn_op, il);

        cb(cur, "ffn_out", il);

        // residual 2
        cur = lm_ggml_add(ctx0, inpL, cur);
        cb(cur, "layer_out", il);

        inpL = cur;
    }

    // post-layernorm
    if (model.post_ln_w) {
        inpL = build_norm(inpL, model.post_ln_w, model.post_ln_b, norm_t, eps, n_layer);
    }

    // multimodal projection
    lm_ggml_tensor * embeddings = inpL;
    embeddings = lm_ggml_reshape_3d(ctx0, embeddings, n_embd * 4, n_pos / 4, batch_size);
    embeddings = build_ffn(embeddings,
                        model.mm_0_w, model.mm_0_b,
                        nullptr, nullptr,
                        model.mm_1_w, model.mm_1_b,
                        FFN_GELU,
                        -1);

    if (use_window_attn) {
        window_idx = lm_ggml_new_tensor_1d(ctx0, LM_GGML_TYPE_I32, n_pos / 4);
        lm_ggml_set_name(window_idx, "window_idx");
        lm_ggml_set_input(window_idx);

        // embeddings shape: [n_embd, n_patches_x * n_patches_y, batch_size]
        LM_GGML_ASSERT(batch_size == 1);
        embeddings = lm_ggml_reshape_2d(ctx0, embeddings, hparams.projection_dim, n_patches_x * n_patches_y / 4);
        embeddings = lm_ggml_get_rows(ctx0, embeddings, window_idx);
        embeddings = lm_ggml_reshape_3d(ctx0, embeddings, hparams.projection_dim, n_patches_x * n_patches_y / 4, batch_size);
    }

    // build the graph
    lm_ggml_build_forward_expand(gf, embeddings);

    return gf;
}
